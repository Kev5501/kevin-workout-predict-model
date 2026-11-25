from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from workout_ml.config import Promotion, TrainingSettings
from workout_ml.training import triggers as T
from workout_ml.training.gate import judge

NOW = datetime(2026, 12, 1, 9, tzinfo=UTC)


def run(days_ago, trigger='scheduled', mode='full', status='promoted', run_id=None):
    t = (NOW - timedelta(days=days_ago)).isoformat()
    return dict(run_id=run_id or f'r{days_ago}', trigger=trigger, mode=mode, status=status, started=t, finished=t,
                reason='why', version='v', gate={'candidate_mape': 2.7, 'champion_mape': 2.8, 'ewma_mape': 2.7,
                                                 'naive_mape': 3.0, 'reason': 'ok'})


def scored(n, ape, days_ago=3, version='champ'):
    day = (NOW - timedelta(days=days_ago)).date().isoformat()
    return [dict(model_version=version, actual_date=day, ape=ape)] * n


def decide(runs, docs=(), champion='champ', streak=None, **kw):
    s = TrainingSettings(**kw)
    return T.decide(s, champion, runs, T.live_performance(list(docs), champion, s, NOW), streak or {}, NOW)


# ------------------------------------------------------------------ scheduled
def test_first_model_then_monthly():
    assert decide([], champion=None).trigger == 'initial'
    assert not decide([run(29.9)]).train
    d = decide([run(30.1)])
    assert (d.trigger, d.mode) == ('scheduled', 'full')


def test_rejected_first_model_backs_off():
    assert not decide([run(1, 'initial', status='rejected')], champion=None).train
    assert decide([run(3.1, 'initial', status='rejected')], champion=None).trigger == 'initial'


def test_performance_retrains_do_not_postpone_the_monthly_sweep():
    d = decide([run(31), run(8, 'performance', 'tune')])
    assert d.trigger == 'scheduled'


def test_scheduled_beats_performance_and_can_be_disabled():
    bad = scored(50, 9.0)
    assert decide([run(31)], bad).trigger == 'scheduled'
    assert decide([run(31)], bad, scheduled={'enabled': False}).trigger == 'performance'


# ------------------------------------------------------------------ performance
def test_dip_needs_enough_sets_from_the_current_champion():
    assert not decide([run(10)], scored(39, 9.0)).train
    assert decide([run(10)], scored(40, 9.0)).trigger == 'performance'
    assert not decide([run(10)], scored(100, 9.0, version='old')).train       # an older model's errors
    assert not decide([run(10)], scored(100, 9.0, days_ago=15)).train         # outside the window


def test_threshold_and_mode():
    assert not decide([run(10)], scored(50, 3.75)).train                      # exactly at the limit is fine
    d = decide([run(10)], scored(50, 3.76))
    assert d.train and d.mode == 'tune'
    assert not decide([run(10)], scored(50, 9.0), performance={'enabled': False}).train
    assert not decide([run(10)], scored(50, 3.9), performance={'max_mape': 4.0}).train


def test_cooldown_counts_from_any_training_run():
    bad = scored(50, 9.0)
    assert 'cooldown' in decide([run(6.9)], bad).reason
    assert 'cooldown' in decide([run(20), run(5, 'manual', 'tune')], bad).reason
    assert decide([run(7.1)], bad).train
    assert decide([run(3)], bad, performance={'cooldown_days': 2}).train


# ------------------------------------------------------------------ failures
def test_failed_runs_back_off_and_running_blocks():
    failed = run(1, status='failed')
    assert not decide([run(40), failed]).train
    assert decide([run(40), run(3.1, status='failed')]).trigger == 'scheduled'
    assert not decide([failed], champion=None).train
    assert 'in progress' in decide([run(40), run(0.01, status='running')]).reason
    assert 'failed' in decide([run(40), run(1, status='running')]).reason      # crashed without finishing
    assert decide([run(40), run(5, status='running')]).train


# ------------------------------------------------------------------ back-to-back streak
def test_streak_counts_rejected_ignores_sweeps_and_failures():
    s = {}
    s = T.streak_after_run(s, run(30, 'performance', 'tune', 'promoted', 'a'))
    s = T.streak_after_run(s, run(22, 'performance', 'tune', 'rejected', 'b'))
    s = T.streak_after_run(s, run(15, 'scheduled', 'full', 'promoted', 'sweep'))
    s = T.streak_after_run(s, run(14, 'performance', 'tune', 'failed', 'x'))
    assert s == {'count': 2, 'run_ids': ['a', 'b']}


def test_streak_resets_only_on_proven_recovery():
    s = {'count': 2, 'run_ids': ['a', 'b']}
    st = TrainingSettings()

    def perf(docs):
        return T.live_performance(docs, 'champ', st, NOW)
    assert T.streak_after_check(s, perf(scored(50, 2.5)))['count'] == 0
    assert T.streak_after_check(s, perf(scored(10, 2.5))) == s                 # too little data to tell
    assert T.streak_after_check(s, perf(scored(50, 9.0))) == s


def test_alert_on_every_third():
    st = TrainingSettings()
    r = run(1, 'performance', 'tune')
    assert [T.should_alert({'count': n}, r, st) for n in range(1, 7)] == [False, False, True, False, False, True]
    assert not T.should_alert({'count': 3}, run(1), st)                        # a sweep mid-streak sends nothing
    assert T.should_alert({'count': 2}, r, TrainingSettings(performance={'alert_after_consecutive': 2}))


def test_alert_text_lists_each_run():
    runs = [run(i, 'performance', 'tune', run_id=f'p{i}') for i in (21, 14, 7)] + [run(3, run_id='other')]
    subject, body = T.alert_text({'count': 3, 'run_ids': ['p21', 'p14', 'p7']}, runs, TrainingSettings())
    assert '3 performance retrains in a row' in subject
    assert body.count('backtest MAPE') == 3 and '2.70%' in body


def test_pause_after_alert():
    bad = scored(50, 9.0)
    kw = dict(performance={'pause_after_alert': True})
    assert decide([run(10)], bad, streak={'count': 2}, **kw).train
    assert 'paused' in decide([run(10)], bad, streak={'count': 3}, **kw).reason
    assert decide([run(10)], bad, streak={'count': 3}).train


# ------------------------------------------------------------------ gate
OK = np.array([100.0, 200.0])


@pytest.mark.parametrize('cand, champ, naive, passed', [
    (2.7, 2.7, 3.0, True), (2.83, 2.7, 3.0, True), (2.84, 2.7, 3.0, False),    # 5% tolerance
    (3.0, 3.5, 3.0, False), (2.9, None, 3.0, True), (3.1, None, 3.0, False)])  # must beat naive
def test_gate(cand, champ, naive, passed):
    assert judge(cand, champ, naive, OK, OK, Promotion())[0] is passed


def test_gate_rejects_insane_predictions_and_honours_settings():
    assert not judge(1, 2, 3, [np.nan, 1], OK, Promotion())[0]
    assert not judge(1, 2, 3, [-5, 200], OK, Promotion())[0]
    assert not judge(1, 2, 3, [300, 200], OK, Promotion())[0]
    assert judge(3.2, 3.5, 3.0, OK, OK, Promotion(must_beat_naive=False))[0]
    assert judge(3.5, 2.7, 4.0, OK, OK, Promotion(max_relative_worse=0.5))[0]
