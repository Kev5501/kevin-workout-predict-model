"""Column types and the bad-data guard for the raw set log."""
from __future__ import annotations

import pandas as pd

from .constants import GUARD_MIN_SESSIONS, GUARD_RANGE, GUARD_WINDOW

DATE_COLS = ['date']
FLOAT_COLS = ['weight', 'rpe', 'rpe_norm_weight', 'body_weight']
STR_COLS = ['set_id', 'gym_day', 'exercise', 'type', 'comment']


def cast_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Types used everywhere downstream: datetime date, float measures, nullable Int64 for the rest.

    Blank cells ('' from the sheet) become NaN / <NA>.
    """
    df = df.copy()
    df.columns = df.columns.str.lower()
    int_cols = [c for c in df.columns if c not in FLOAT_COLS + STR_COLS + DATE_COLS]
    df['date'] = pd.to_datetime(df['date'])
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
    for c in FLOAT_COLS:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
    return df   # STR_COLS are left as the sheet returns them


def clean_raw(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off sets whose est. 1RM is impossible next to that exercise + set's own history.

    Each set is compared with the median of the previous ``GUARD_WINDOW`` sessions (past data
    only, so it behaves the same in training and at inference). Catches data-entry
    slips such as a missing digit. Returns ``(clean, flagged)``; a corrected set passes on the
    next build.
    """
    keys = ['exercise', 'set_number']
    value = pd.to_numeric(df['rpe_norm_weight'], errors='coerce').astype(float)
    daily = df.assign(_v=value).groupby(keys + ['date'], sort=True)['_v'].median().reset_index()
    daily['_ref'] = daily.groupby(keys)['_v'].transform(
        lambda s: s.shift(1).rolling(GUARD_WINDOW, min_periods=GUARD_MIN_SESSIONS).median())
    ref = df[keys + ['date']].merge(daily[keys + ['date', '_ref']], on=keys + ['date'], how='left')['_ref']
    ratio = value.to_numpy() / ref.to_numpy()
    bad = (ratio < GUARD_RANGE[0]) | (ratio > GUARD_RANGE[1])      # NaN compares False
    flagged = df[bad].copy()
    flagged['expected_est_1rm'] = ref.to_numpy()[bad]
    return df[~bad].reset_index(drop=True), flagged
