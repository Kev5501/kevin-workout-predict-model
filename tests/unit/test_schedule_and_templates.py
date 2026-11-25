from datetime import date

import pandas as pd
import pytest
from factories import EXERCISES, LAST_DAY, SCHEDULE, add_workout, make_history

from workout_ml.inference.schedule import plan_batch
from workout_ml.inference.templates import build_template, future_rows


def names(run_date, logged=(), schedule=SCHEDULE, horizon=7):
    batch = plan_batch(run_date, schedule, lambda g: [e for e, _, _ in EXERCISES.get(g, [])],
                       lambda d, g: (d, g) in logged, horizon)
    return [w.gym_day for w in batch]


@pytest.mark.parametrize('day, expected', [
    (23, ['Push', 'Pull', 'Legs']),        # Monday (rest)
    (24, ['Push', 'Pull', 'Legs']),        # Tuesday
    (25, ['Pull', 'Legs']),                # Wednesday: Upper repeats Pull's lifts
    (26, ['Legs', 'Upper']),               # Thursday: Lower repeats Legs' lifts
    (27, ['Upper', 'Lower']),              # Friday (rest)
    (28, ['Upper', 'Lower']),              # Saturday
    (29, ['Lower', 'Push', 'Pull']),       # Sunday: Legs repeats Lower's lifts
])
def test_every_weekday(day, expected):
    assert names(date(2026, 11, day)) == expected


def test_today_is_skipped_once_logged():
    tue = date(2026, 11, 24)
    assert names(tue, logged={(tue, 'Push')}) == ['Pull', 'Legs']
    assert names(tue, logged={(tue, 'Upper')}) == ['Push', 'Pull', 'Legs']     # a different workout was logged


def test_dates_and_weekdays():
    batch = plan_batch(date(2026, 11, 23), SCHEDULE, lambda g: [g], lambda d, g: False, 4)
    assert [(w.date.day, w.weekday) for w in batch] == [(24, 'Tuesday'), (25, 'Wednesday'), (26, 'Thursday')]


def test_missing_days_rest_case_and_horizon():
    assert names(date(2026, 11, 23), schedule={'Wednesday': 'Pull'}) == ['Pull']
    assert names(date(2026, 11, 23), schedule={'Tuesday': 'rest', 'Wednesday': 'Pull'}) == ['Pull']
    assert names(date(2026, 11, 23), horizon=2) == ['Push']
    assert names(date(2026, 11, 23), schedule={}) == []


def test_gym_day_without_history_is_skipped():
    assert names(date(2026, 11, 23), schedule={'Tuesday': 'Arms', 'Wednesday': 'Pull'}) == ['Pull']


def test_template_has_every_combo_in_workout_order():
    t = build_template(make_history(8), 'Push', LAST_DAY, 30)
    assert t[['exercise', 'set_number']].values.tolist() == [
        ['Bench Press', 1], ['Bench Press', 2], ['Lateral Raise', 1], ['Lateral Raise', 2], ['Lateral Raise', 3]]


def test_template_lookback_excludes_old_and_includes_rare():
    h = make_history(12)
    old = h[(h.gym_day == 'Push') & (h.date < pd.Timestamp('2026-10-01'))].index[0]
    h.loc[old, 'exercise'] = 'Old Lift'
    rare = h[(h.gym_day == 'Push') & (h.date > pd.Timestamp('2026-11-10'))].index[0]
    h.loc[rare, 'exercise'] = 'Rare Lift'
    got = set(build_template(h, 'Push', LAST_DAY, 30).exercise)
    assert 'Rare Lift' in got and 'Old Lift' not in got
    assert 'Old Lift' in set(build_template(h, 'Push', LAST_DAY, 90).exercise)


def test_template_ignores_other_gym_days_and_the_future():
    h = add_workout(make_history(8), date(2026, 12, 5), 'Upper')
    h.loc[h.date == pd.Timestamp('2026-12-05'), 'exercise'] = 'Future Lift'
    assert 'Future Lift' not in set(build_template(h, 'Upper', LAST_DAY, 30).exercise)
    assert set(build_template(h, 'Legs', LAST_DAY, 30).exercise) == {'Quad Extension', 'Hip Adductor'}


def test_template_falls_back_to_latest_session():
    h = make_history(8, end=date(2026, 8, 30))
    t = build_template(h, 'Push', LAST_DAY, 30)
    assert len(t) == 5
    assert build_template(h, 'Never', LAST_DAY, 30).empty


def test_future_rows():
    t = build_template(make_history(8), 'Pull', LAST_DAY, 30)
    f = future_rows(t, 'Pull', date(2026, 11, 25))
    assert f.set_id.is_unique and (f.gym_day == 'Pull').all() and (f.date == pd.Timestamp('2026-11-25')).all()
