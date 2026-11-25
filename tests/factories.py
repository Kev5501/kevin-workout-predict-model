"""Synthetic workout data with the sheet's columns. No real data is used in the tests."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from workout_ml.config import WEEKDAYS
from workout_ml.core.constants import FEATURE_SETTINGS

SCHEDULE = {'Sunday': 'Lower', 'Monday': 'Rest', 'Tuesday': 'Push', 'Wednesday': 'Pull', 'Thursday': 'Legs',
            'Friday': 'Rest', 'Saturday': 'Upper'}
# Upper overlaps Push and Pull; Lower overlaps Legs (as in the real program).
EXERCISES = {
    'Push': [('Bench Press', 280, 2), ('Lateral Raise', 60, 3)],
    'Pull': [('Lat Pulldown', 250, 2), ('Reverse Fly', 120, 2)],
    'Legs': [('Quad Extension', 340, 2), ('Hip Adductor', 200, 1)],
    'Upper': [('Bench Press', 280, 2), ('Lat Pulldown', 250, 2)],
    'Lower': [('Squat', 380, 2), ('Quad Extension', 340, 2)],
}
LAST_DAY = date(2026, 11, 22)                 # a Sunday


def ratio(reps: int) -> float:
    return {5: .863, 6: .837, 7: .811, 8: .786, 9: .762, 10: .739}[reps]


def make_history(weeks: int = 30, end: date = LAST_DAY, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, sid = [], 1
    for offset in range(weeks * 7):
        day = end - timedelta(days=weeks * 7 - 1 - offset)
        gym_day = SCHEDULE[WEEKDAYS[day.weekday()]]
        if gym_day == 'Rest':
            continue
        flags = {c: int(rng.random() < 0.1) for c in FEATURE_SETTINGS['binary_cols']}
        for exercise, base, sets in EXERCISES[gym_day]:
            session = base * (1 + 0.0006 * offset) * (1 + rng.normal(0, 0.015))
            for s in range(1, sets + 1):
                e1rm = round(session * (1 - 0.02 * (s - 1)) * (1 + rng.normal(0, 0.008)), 1)
                reps = int(rng.integers(5, 11))
                rows.append(dict(date=pd.Timestamp(day), set_id=f'SET-{sid:05d}', gym_day=gym_day, exercise=exercise,
                                 type='Hypertrophy', set_number=s, weight=round(e1rm * ratio(reps), 2), reps=reps,
                                 rpe=10.0, rpe_norm_weight=e1rm, body_weight=175 + rng.normal(), comment='',
                                 workout_rating=4, pump_rating=3, **flags))
                sid += 1
    return pd.DataFrame(rows)


def add_workout(history: pd.DataFrame, day: date, gym_day: str, factor: float = 1.0) -> pd.DataFrame:
    """Log one more workout, its sets ``factor`` times the latest value of each lift."""
    rows, sid = [], len(history) + 1
    for exercise, _, sets in EXERCISES[gym_day]:
        for s in range(1, sets + 1):
            last = history[(history.exercise == exercise) & (history.set_number == s)].iloc[-1]
            e1rm = round(last['rpe_norm_weight'] * factor, 1)
            rows.append({**last.to_dict(), 'date': pd.Timestamp(day), 'set_id': f'SET-{sid:05d}', 'gym_day': gym_day,
                         'reps': 8, 'rpe_norm_weight': e1rm, 'weight': round(e1rm * ratio(8), 2)})
            sid += 1
    return pd.concat([history, pd.DataFrame(rows)], ignore_index=True)


class FakeSheets:
    """History is a plain attribute, so tests can log workouts between runs."""

    def __init__(self, history: pd.DataFrame):
        self.history = history
        self.header: list[str] = []
        self.rows: list[list] = []
        self.fail_writes = False

    def read_history(self) -> pd.DataFrame:
        return self.history.copy()

    def write_predictions(self, header, rows):
        if self.fail_writes:
            raise ConnectionError('sheets is down')
        self.header, self.rows = header, rows


class FakeNotifier:
    def __init__(self):
        self.sent: list[tuple[str, str]] = []

    def send(self, subject, body):
        self.sent.append((subject, body))


class FakeLauncher:
    def __init__(self):
        self.calls: list[tuple[str, str, str]] = []

    def launch(self, mode, trigger, reason):
        self.calls.append((mode, trigger, reason))
        return 'execution-1'
