import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TESTING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTING_DIR))

from modeling import features as F  # noqa: E402

PLAN = {
    'Push': [('Bench Press', 120), ('Cable Lateral Raises', 25)],
    'Pull': [('Lat Pulldown', 150), ('T-Bar Row', 180)],
    'Legs': [('Squat', 250), ('Hamstring Curl', 110)],
}


def make_raw(n_workouts: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic sheet data with the same columns and dtypes as Enriched_Data."""
    rng = np.random.default_rng(seed)
    rows, sid = [], 1
    dates = pd.date_range('2025-04-08', periods=n_workouts, freq='2D')
    for i, d in enumerate(dates):
        day = ['Push', 'Pull', 'Legs'][i % 3]
        flags = {c: (pd.NA if i < 20 else int(rng.random() < 0.1)) for c in F.BINARY_COLS}
        for ex, base in PLAN[day]:
            for s in (1, 2):
                e1rm = base * (1 + 0.0015 * i) - 3 * (s - 1) + rng.normal(0, base * 0.02)
                if flags['illness_ind'] is not pd.NA and flags['illness_ind'] == 1:
                    e1rm *= 0.95
                reps = int(rng.integers(5, 11))
                rows.append(dict(date=d.date(), set_id=f'SET-{sid:05d}', gym_day=day, exercise=ex,
                                 type='Hypertrophy', set_number=s, weight=round(e1rm * 0.8, 1), reps=reps,
                                 rpe=10.0, rpe_norm_weight=round(e1rm, 1), body_weight=175 + rng.normal(),
                                 comment='', workout_rating=int(rng.integers(3, 6)),
                                 pump_rating=int(rng.integers(1, 4)), **flags))
                sid += 1
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    for c in ['set_number', 'reps', 'workout_rating', 'pump_rating'] + F.BINARY_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
    return df


from modeling.config import FEATURE_SETTINGS  # noqa: E402

BUILD_KW = FEATURE_SETTINGS   # same settings as the real dataset build


@pytest.fixture(scope='session')
def raw():
    return make_raw()


@pytest.fixture(scope='session')
def feats(raw):
    return F.build_features(raw, **BUILD_KW)


@pytest.fixture(scope='session')
def dataset(feats):
    from modeling.data import make_dataset
    return make_dataset(feats)
