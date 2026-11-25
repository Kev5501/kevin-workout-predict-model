from datetime import date, timedelta

import numpy as np
import pytest
from factories import LAST_DAY, add_workout

from workout_ml.errors import Busy
from workout_ml.inference.batch import LOCK_TTL, run_batch, sync_sheet
from workout_ml.inference.history import UnsafeHistory
from workout_ml.inference.predict import UnsafePredictions

MON, TUE, WED, THU, FRI = (LAST_DAY + timedelta(days=i) for i in range(1, 6))


def go(svc, day, **kw):
    return run_batch(svc.settings, svc.docs, svc.registry, svc.sheets, day, **kw)


def ids(svc, col='predictions'):
    return sorted(d['id'] for d in svc.docs.list(col))


def test_first_batch(trained):
    r = go(trained, MON)
    assert [w['gym_day'] for w in r['workouts']] == ['Push', 'Pull', 'Legs']
    docs = trained.docs.list('predictions')
    assert len(docs) == r['n_predictions'] == 5 + 4 + 3
    assert len({(d['exercise'], d['set_number']) for d in docs}) == len(docs)        # non-overlapping
    d = docs[0]
    assert set(d['weights']) == {'5', '6', '7', '8', '9', '10'}
    assert all(w % 2.5 == 0 for w in d['weights'].values())
    assert list(d['weights'].values()) == sorted(d['weights'].values(), reverse=True)
    for d in docs:
        assert 0.85 < d['est_1rm'] / d['last_est_1rm'] < 1.15
        assert d['data_through'] == LAST_DAY.isoformat()


def test_sheet_matches_the_store(trained):
    go(trained, MON)
    assert trained.sheets.header[:6] == ['workout_date', 'weekday', 'gym_day', 'exercise', 'set_number', 'est_1rm']
    assert trained.sheets.header[6:12] == [f'weight_{r}_reps' for r in range(5, 11)]
    assert len(trained.sheets.rows) == 12
    assert [r[2] for r in trained.sheets.rows] == ['Push'] * 5 + ['Pull'] * 4 + ['Legs'] * 3


def test_rerun_overwrites_instead_of_duplicating(trained):
    go(trained, MON)
    first = ids(trained)
    go(trained, MON)
    go(trained, TUE)                                   # same batch, one day later
    assert ids(trained) == first
    assert trained.docs.list('scored_predictions') == []


def test_logged_workout_is_scored_frozen_and_the_batch_moves_on(trained):
    go(trained, MON)
    push = {d['id']: d for d in trained.docs.list('predictions') if d['gym_day'] == 'Push'}
    trained.sheets.history = add_workout(trained.sheets.history, TUE, 'Push', factor=1.02)
    r = go(trained, WED)
    assert [w['gym_day'] for w in r['workouts']] == ['Pull', 'Legs'] and r['n_scored'] == 5
    scored = trained.docs.list('scored_predictions')
    assert {s['id'] for s in scored} == set(push)
    for s in scored:
        assert s['est_1rm'] == push[s['id']]['est_1rm'] and s['actual_date'] == TUE.isoformat()
        assert s['ape'] == pytest.approx(abs(s['est_1rm'] - s['actual_est_1rm']) / s['actual_est_1rm'] * 100)
        assert 'collection' not in s
    go(trained, WED)
    assert trained.docs.list('scored_predictions') == scored         # frozen
    assert not any(d['gym_day'] == 'Push' for d in trained.docs.list('predictions'))


def test_todays_logged_workout_is_not_predicted_again(trained):
    go(trained, TUE)
    trained.sheets.history = add_workout(trained.sheets.history, TUE, 'Push')
    r = go(trained, TUE)                               # second run the same evening
    assert [w['gym_day'] for w in r['workouts']] == ['Pull', 'Legs'] and r['n_scored'] == 5


def test_late_workout_is_still_scored(trained):
    go(trained, TUE)
    go(trained, WED)                                   # Push was missed: its predictions wait
    assert len(trained.docs.list('pending_predictions')) == 5
    assert not any(d['gym_day'] == 'Push' for d in trained.docs.list('predictions'))
    trained.sheets.history = add_workout(trained.sheets.history, WED, 'Push')
    r = go(trained, THU)
    assert r['n_scored'] == 5
    assert not any(d['gym_day'] == 'Push' for d in trained.docs.list('pending_predictions'))


def test_missed_workout_expires(trained):
    go(trained, TUE)
    go(trained, WED)
    go(trained, THU)
    # Friday's batch predicts Bench Press again (Upper); Lateral Raise keeps waiting until Friday
    assert {d['exercise'] for d in trained.docs.list('pending_predictions')} <= {'Lateral Raise', 'Lat Pulldown',
                                                                                 'Reverse Fly'}
    for i in range(1, 6):
        go(trained, THU + timedelta(days=i))
    assert not any(d['gym_day'] == 'Push' and d['workout_date'] == TUE.isoformat()
                   for d in trained.docs.list('pending_predictions') + trained.docs.list('predictions'))
    assert trained.docs.list('scored_predictions') == []


def test_early_workout_is_scored_and_does_not_lose_history(trained):
    go(trained, MON)
    trained.sheets.history = add_workout(trained.sheets.history, MON, 'Push')       # a day early
    go(trained, TUE)                                   # Tuesday's Push is predicted again, same ids
    assert len(trained.docs.list('scored_predictions')) == 5
    assert {s['actual_date'] for s in trained.docs.list('scored_predictions')} == {MON.isoformat()}
    assert sum(d['gym_day'] == 'Push' for d in trained.docs.list('predictions')) == 5


def test_dry_run_writes_nothing(trained):
    before = trained.docs._data
    r = go(trained, MON, dry_run=True)
    assert len(r['predictions']) == 12
    assert trained.docs._data == before and trained.sheets.rows == []


def test_planned_rows_do_not_leak_between_workouts(trained):
    """Pull's predictions are the same whether or not Push is planned the day before."""
    with_push = {d['id']: d['est_1rm'] for d in go(trained, MON, dry_run=True)['predictions']}
    trained.settings.schedule = {**trained.settings.schedule, 'Tuesday': 'Rest'}
    without = {d['id']: d['est_1rm'] for d in go(trained, MON, dry_run=True)['predictions'] if d['gym_day'] == 'Pull'}
    assert without and all(with_push[k] == v for k, v in without.items())


# ------------------------------------------------------------------ safety
def snapshot(svc):
    return svc.docs.list('predictions')


def test_unsafe_history_leaves_predictions_alone(trained):
    go(trained, MON)
    before, good = snapshot(trained), trained.sheets.history
    for bad in (good.iloc[0:0], good.iloc[: len(good) // 2], good.drop(columns=['rpe_norm_weight'])):
        trained.sheets.history = bad
        with pytest.raises(UnsafeHistory):
            go(trained, TUE)
        assert snapshot(trained) == before
    trained.sheets.history = good
    go(trained, TUE)                                   # the lock was released after each failure


class Constant:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def test_unsafe_predictions_abort_or_warn(trained):
    go(trained, MON)
    before = snapshot(trained)
    bundle = trained.registry.load_champion()
    model, bundle.model = bundle.model, Constant(np.nan)
    with pytest.raises(UnsafePredictions):
        go(trained, TUE)
    bundle.model = Constant(5000.0)
    with pytest.raises(UnsafePredictions, match='too far'):
        go(trained, TUE)
    assert snapshot(trained) == before
    trained.settings.inference.on_jump = 'warn'
    assert len(go(trained, TUE)['flagged']) == 12
    bundle.model = model


def test_sheet_outage_keeps_predictions_and_can_be_retried(trained):
    trained.sheets.fail_writes = True
    assert go(trained, MON)['sheet_synced'] is False
    assert len(snapshot(trained)) == 12
    trained.sheets.fail_writes = False
    assert sync_sheet(trained.settings, trained.docs, trained.sheets) == 12 == len(trained.sheets.rows)


def test_concurrent_runs_are_refused(trained):
    trained.docs.acquire_lock('batch', 'someone-else', LOCK_TTL)
    with pytest.raises(Busy):
        go(trained, MON)


def test_blank_future_and_unfinished_rows_are_ignored(trained):
    import pandas as pd
    h = trained.sheets.history
    blank = pd.DataFrame([{c: '' for c in h.columns}])
    planned = h.iloc[[-1]].assign(date=pd.Timestamp(MON), rpe_norm_weight=np.nan, set_id='SET-90001')
    future = h.iloc[[-1]].assign(date=pd.Timestamp(date(2027, 1, 1)), set_id='SET-90002')
    clean = go(trained, MON, dry_run=True)
    trained.sheets.history = pd.concat([h.astype(object), planned.astype(object), future.astype(object), blank],
                                       ignore_index=True)
    dirty = go(trained, MON, dry_run=True)
    assert [d['est_1rm'] for d in dirty['predictions']] == [d['est_1rm'] for d in clean['predictions']]
    assert dirty['data_through'] == LAST_DAY.isoformat()


def test_typo_is_excluded(trained):
    h = trained.sheets.history.copy()
    i = h[(h.exercise == 'Squat') & (h.set_number == 1)].index[-1]
    h.loc[i, 'rpe_norm_weight'] = 3.7
    trained.sheets.history = h
    assert go(trained, MON, dry_run=True)['excluded_sets'] == 1
