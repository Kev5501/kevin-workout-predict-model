from datetime import timedelta

import numpy as np
import pytest
from factories import LAST_DAY, make_history

from workout_ml.core.constants import FEATURE_SETTINGS
from workout_ml.core.data import make_dataset
from workout_ml.core.features import build_features
from workout_ml.errors import Busy
from workout_ml.training import run as R
from workout_ml.training import train as TR
from workout_ml.training.checks import check


def test_first_model_is_promoted_and_documented(trained):
    v = trained.registry.champion_version()
    doc = trained.docs.get('models', v)
    assert doc['status'] == 'champion' and doc['trigger'] == 'initial'
    b = trained.registry.load(v)
    assert {'gate', 'data_range', 'data_sha256', 'libraries', 'recipe', 'tuning', 'selection'} <= set(b.metadata)
    assert b.metadata['data_range'][1] == LAST_DAY.isoformat()
    assert set(b.recipe.features) <= set(b.candidates)
    run = trained.docs.list('training_runs')[0]
    assert run['status'] == 'promoted' and run['version'] == v


def test_tune_keeps_the_champions_features(trained, fast_search):
    before = trained.registry.load_champion()
    r = R.run_training(trained, 'tune', 'performance', 'test')
    assert r['features'] == before.recipe.features
    assert r['replaced'] == before.version
    assert r['gate']['champion_mape'] is not None and r['gate']['n_workouts'] == 10


def test_refit_keeps_the_whole_recipe(trained):
    before = trained.registry.load_champion()
    r = R.run_training(trained, 'refit', 'manual', 'test')
    assert r['status'] == 'promoted'
    after = trained.registry.load_champion()
    assert after.version != before.version and after.recipe == before.recipe
    assert r['gate']['candidate_mape'] == pytest.approx(r['gate']['champion_mape'])


def test_rejected_candidate_leaves_the_champion(trained):
    champion = trained.registry.champion_version()
    trained.settings.training.promotion.max_relative_worse = 0
    import workout_ml.training.train as train
    original = train.judge
    train.judge = lambda *a: (False, 'forced')
    try:
        r = R.run_training(trained, 'refit', 'manual', 'test')
    finally:
        train.judge = original
    assert r['status'] == 'rejected' and trained.registry.champion_version() == champion
    assert trained.docs.get('models', r['version'])['status'] == 'rejected'


def test_manual_approval(trained):
    champion = trained.registry.champion_version()
    trained.settings.training.promotion.auto = False
    r = R.run_training(trained, 'refit', 'manual', 'test')
    assert r['status'] == 'candidate' and trained.registry.champion_version() == champion
    trained.registry.promote(r['version'])
    assert trained.registry.champion_version() == r['version']


def test_failure_is_recorded_lock_released_champion_kept(trained, monkeypatch):
    champion = trained.registry.champion_version()
    monkeypatch.setattr(R, 'train', lambda *a, **k: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        R.run_training(trained, 'full', 'scheduled', 'test')
    failed = [r for r in trained.docs.list('training_runs') if r['status'] == 'failed']
    assert len(failed) == 1 and 'ZeroDivisionError' in failed[0]['error']
    assert trained.registry.champion_version() == champion
    assert trained.docs.get('locks', 'training') is None


def test_only_one_training_run_at_a_time(trained):
    trained.docs.acquire_lock('training', 'someone-else', timedelta(minutes=5))
    with pytest.raises(Busy):
        R.run_training(trained, 'refit', 'manual', 'test')


# ------------------------------------------------------------------ the backtest is honest
def test_search_never_sees_the_evaluation_workouts(settings, fast_search, monkeypatch):
    """Rewriting every evaluation workout must not change the chosen features or parameters."""
    history = make_history(30)
    cutoff = np.sort(history.date.unique())[-10]
    changed = history.copy()
    late = changed.date >= cutoff
    changed.loc[late, 'rpe_norm_weight'] *= 1.3
    changed.loc[late, 'weight'] *= 1.3
    a = TR.train(history, 'full', None, settings.training)
    b = TR.train(changed, 'full', None, settings.training)
    assert a.recipe == b.recipe
    assert a.gate.candidate_mape != b.gate.candidate_mape


def test_both_recipes_are_fit_on_the_same_earlier_data(settings, fast_search, monkeypatch, trained):
    seen = []
    original = TR.fit_recipe
    monkeypatch.setattr(TR, 'fit_recipe', lambda r, d, s: seen.append(d.dates.max()) or original(r, d, s))
    history = trained.sheets.history
    TR.train(history, 'refit', trained.registry.load_champion().recipe, settings.training)
    cutoff = np.sort(history.date.unique())[-10]
    assert seen[0] == seen[1] < cutoff and seen[2] == history.date.max()


def test_split_eval_needs_enough_workouts():
    d = make_dataset(build_features(make_history(3), **FEATURE_SETTINGS))
    with pytest.raises(ValueError, match='not enough workouts'):
        TR.split_eval(d.dates, 20)


def test_small_history_fails_cleanly(svc):
    svc.sheets.history = make_history(2)
    with pytest.raises(ValueError):
        R.run_training(svc, 'full', 'initial', 'test')
    assert svc.registry.champion_version() is None


# ------------------------------------------------------------------ streak and email
def test_third_back_to_back_retrain_sends_one_email(trained):
    for _ in range(3):
        trained.docs.set('locks', 'noop', {'owner': 'x', 'expires': '2000-01-01T00:00:00+00:00'})
        record = dict(run_id=f'p{_}', status='promoted' if _ else 'rejected', trigger='performance', mode='tune',
                      started=f'2026-12-0{_ + 1}T00:00:00+00:00', finished=f'2026-12-0{_ + 1}T01:00:00+00:00',
                      reason='live MAPE 4.1% is above 3.75%', version=f'v{_}', gate=None)
        trained.docs.set('training_runs', record['run_id'], record)
        R._update_streak(trained, record)
    assert len(trained.notifier.sent) == 1
    subject, body = trained.notifier.sent[0]
    assert '3 performance retrains' in subject and body.count('mode=tune') == 3


def test_mail_failure_does_not_fail_training(trained):
    def boom(*_):
        raise OSError('smtp down')
    trained.notifier.send = boom
    trained.docs.set('state', 'performance_streak', {'count': 2, 'run_ids': ['a', 'b']})
    R._update_streak(trained, dict(run_id='c', status='promoted', trigger='performance', mode='tune',
                                   finished='2026-12-01T00:00:00+00:00', reason='x', version='v'))
    assert trained.docs.get('state', 'performance_streak')['count'] == 3


def test_check_resets_streak_on_recovery(trained):
    trained.docs.set('state', 'performance_streak', {'count': 2, 'run_ids': ['a', 'b']})
    v = trained.registry.champion_version()
    from datetime import UTC, datetime
    today = datetime.now(UTC).date().isoformat()
    for i in range(50):
        trained.docs.set('scored_predictions', f's{i}', dict(model_version=v, actual_date=today, ape=2.0))
    d = check(trained)
    assert not d.train and trained.docs.get('state', 'performance_streak')['count'] == 0
