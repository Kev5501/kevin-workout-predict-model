"""Feature engineering for workout performance prediction.

Leakage rule: every history feature on a row dated D is built only from data
on dates strictly before D (time-based rolling with ``closed='left'`` and
``shift(1)`` for "most recent"). Same-session columns such as ``weight``,
``reps``, ``rpe``, ``rpe_norm_weight``/``est_1rm``, ``workout_rating``,
``pump_rating`` and ``form_change_ind`` must not be used raw as model inputs
for their own row; use the lagged versions this module builds instead.

Used with the settings in ``constants.FEATURE_SETTINGS``. Direct use::

    from workout_ml.core.features import build_features, get_feature_columns
    feats = build_features(df, time_lag_days=[90, 60, 30, 14, 7, 3, 1],
                           metrics=['avg', 'max', 'min', 'slope'],
                           binary_cols=BINARY_COLS)
    X = feats[get_feature_columns(feats)]
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_LAGS = [90, 60, 30, 14, 7, 3, 1]
DEFAULT_METRICS = ['avg', 'max', 'min', 'slope']
DEFAULT_EWMA_ALPHAS = [0.3, 0.5, 0.7]
DEFAULT_LAST_N = [2, 3]

BINARY_COLS = [
    'injury_ind', 'illness_ind', 'girlfriend_ind', 'fatigue_ind',
    'vacation_ind', 'form_change_ind', 'forgot_headphones_ind',
    'other_extenuating_circumstances_ind',
]

# Columns observed during/after the current set or workout. Not valid as raw
# inputs for their own row.
SAME_SESSION_COLS = [
    'weight', 'reps', 'rpe', 'rpe_norm_weight', 'est_1rm', 'comment',
    'workout_rating', 'pump_rating', 'form_change_ind',
]
ID_COLS = ['date', 'set_id']
CATEGORICAL_COLS = ['exercise', 'gym_day', 'type']

# metric name -> pandas Rolling method ('slope' is computed from rolling sums)
_ROLLING_METHODS = {
    'avg': 'mean', 'max': 'max', 'min': 'min',
    'std': 'std', 'count': 'count', 'sum': 'sum',
}
SUPPORTED_METRICS = set(_ROLLING_METHODS) | {'slope'}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _as_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def _validate(time_lag_days: Iterable[int], metrics: Iterable[str]) -> tuple[list[int], list[str]]:
    lags = sorted({int(lag) for lag in time_lag_days}, reverse=True)
    if any(lag < 1 for lag in lags):
        raise ValueError(f'time_lag_days must be positive integers, got {lags}')
    metrics = list(dict.fromkeys(metrics))  # dedupe, keep order
    unknown = set(metrics) - SUPPORTED_METRICS
    if unknown:
        raise ValueError(f'Unknown metric(s) {sorted(unknown)}; supported: {sorted(SUPPORTED_METRICS)}')
    return lags, metrics


def _realign(res: pd.DataFrame, n_group_levels: int) -> pd.DataFrame:
    """Drop groupby key levels so the result aligns with the input index."""
    return res.droplevel(list(range(n_group_levels))) if n_group_levels else res


def _merge_back(df: pd.DataFrame, feats: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    new_cols = [c for c in feats.columns if c not in keys]
    clash = [c for c in new_cols if c in df.columns]
    if clash:
        raise ValueError(f'Feature columns already exist in df: {clash}')
    out = df.merge(feats, on=keys, how='left', validate='many_to_one')
    assert len(out) == len(df), 'merge changed row count'
    out.index = df.index
    return out


# --------------------------------------------------------------------------- #
# 1. Preparation
# --------------------------------------------------------------------------- #
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Cast date, sort, add ``est_1rm`` and ``exercise_order``."""
    out = df.copy()
    out['date'] = pd.to_datetime(out['date'])
    out = out.sort_values(['date', 'set_id'], kind='stable').reset_index(drop=True)
    out['est_1rm'] = out['rpe_norm_weight'].astype(float)

    # Position of each exercise within that day's workout (1 = first exercise).
    pos = pd.Series(np.arange(len(out)), index=out.index)
    first_pos = pos.groupby([out['date'], out['exercise']]).transform('min')
    out['exercise_order'] = first_pos.groupby(out['date']).rank(method='dense').astype('Int64')
    return out


# --------------------------------------------------------------------------- #
# 2. Shared rolling engine
# --------------------------------------------------------------------------- #
def _rolling_engine(
    daily: pd.DataFrame,
    value_cols: Sequence[str],
    time_lag_days: Iterable[int],
    metrics: Iterable[str],
    group_cols: Sequence[str] = (),
    prefix: str | None = None,
    days_since_col: str | None = None,
) -> pd.DataFrame:
    """Rolling history features on a frame with one row per (group_cols, date).

    ``daily`` must be sorted by ``group_cols + ['date']``. Windows exclude the
    current date. Lag 1 produces ``{name}_most_recent`` (the previous session,
    however long ago) instead of ``*_1d`` columns.

    Returns ``daily[group_cols + ['date']]`` plus the feature columns.
    """
    group_cols = list(group_cols)
    value_cols = list(value_cols)
    lags, metrics = _validate(time_lag_days, metrics)
    daily = daily.reset_index(drop=True)

    def name(col: str) -> str:
        return f'{prefix}_{col}' if prefix else col

    work = daily[group_cols + ['date']].copy()
    for col in value_cols:
        work[col] = daily[col].astype(float)

    # Helper columns for slope: x = days since the first date, masked where y is NaN
    if 'slope' in metrics:
        x_all = (work['date'] - work['date'].min()).dt.days.astype(float)
        for col in value_cols:
            x = x_all.where(work[col].notna())
            work[f'{col}__x'] = x
            work[f'{col}__xy'] = x * work[col]
            work[f'{col}__xx'] = x * x

    grouped = work.groupby(group_cols, sort=False, dropna=False) if group_cols else None
    n_levels = len(group_cols)
    feats: dict[str, pd.Series] = {}

    # Lag 1 -> most recent previous session
    if 1 in lags:
        for col in value_cols:
            prev = grouped[col].shift(1) if grouped is not None else work[col].shift(1)
            feats[f'{name(col)}_most_recent'] = prev
        if days_since_col:
            prev_date = grouped['date'].shift(1) if grouped is not None else work['date'].shift(1)
            feats[days_since_col] = (work['date'] - prev_date).dt.days.astype('Int64')

    roll_cols = [c for c in work.columns if c not in group_cols + ['date']]
    for lag in (lag for lag in lags if lag != 1):
        window = f'{lag}D'
        if grouped is not None:
            roller = grouped[roll_cols + ['date']].rolling(window, on='date', closed='left')
        else:
            roller = work[roll_cols + ['date']].rolling(window, on='date', closed='left')

        cache: dict[str, pd.DataFrame] = {}

        def stat(method: str, cache=cache, roller=roller) -> pd.DataFrame:
            if method not in cache:
                cache[method] = _realign(getattr(roller, method)(), n_levels)
            return cache[method]

        for metric in metrics:
            for col in value_cols:
                out_name = f'{name(col)}_{metric}_{lag}d'
                if metric == 'slope':
                    s, n = stat('sum'), stat('count')[col]
                    sx, sy = s[f'{col}__x'], s[col]
                    sxy, sxx = s[f'{col}__xy'], s[f'{col}__xx']
                    denom = n * sxx - sx ** 2
                    slope = (n * sxy - sx * sy) / denom
                    feats[out_name] = slope.where((n >= 2) & (denom > 1e-9))
                elif metric in ('count', 'sum'):
                    # Empty window (no prior sessions) means 0, not missing
                    feats[out_name] = stat(_ROLLING_METHODS[metric])[col].fillna(0.0)
                else:
                    feats[out_name] = stat(_ROLLING_METHODS[metric])[col]

    feat_df = pd.DataFrame(feats, index=work.index)
    return pd.concat([daily[group_cols + ['date']], feat_df], axis=1)


# --------------------------------------------------------------------------- #
# 3. Exercise + set level rolling features
# --------------------------------------------------------------------------- #
def add_rolling_features(
    df: pd.DataFrame,
    time_lag_days: Iterable[int] = DEFAULT_LAGS,
    metrics: Iterable[str] = DEFAULT_METRICS,
    value_col: str | Sequence[str] = 'est_1rm',
    group_cols: Sequence[str] = ('exercise', 'set_number'),
    days_since_col: str | None = 'days_since_last_exercise_set',
    prefix: str | None = None,
) -> pd.DataFrame:
    """Every (lag, metric) combination of ``value_col`` history per exercise + set.

    Example: a row for Bench Press set 2 on 2025-06-10 gets ``est_1rm_avg_30d``
    = mean est_1rm of Bench Press set 2 over [2025-05-11, 2025-06-10), i.e. the
    30 calendar days before, current day excluded. Lag 1 becomes
    ``est_1rm_most_recent``.
    """
    value_cols = _as_list(value_col)
    group_cols = list(group_cols)
    keys = group_cols + ['date']
    daily = (
        df.groupby(keys, dropna=False, sort=True)[value_cols]
        .max()
        .astype(float)
        .reset_index()
    )
    feats = _rolling_engine(daily, value_cols, time_lag_days, metrics,
                            group_cols=group_cols, prefix=prefix,
                            days_since_col=days_since_col)
    return _merge_back(df, feats, keys)


# --------------------------------------------------------------------------- #
# 4. Workout-date level binary features
# --------------------------------------------------------------------------- #
def add_binary_features(
    df: pd.DataFrame,
    time_lag_days: Iterable[int] = DEFAULT_LAGS,
    binary_cols: Sequence[str] | None = None,
    metrics: Iterable[str] = ('avg', 'max'),
) -> pd.DataFrame:
    """Rolling history of binary flags, computed per workout date.

    Optional: ``binary_cols=None`` or ``[]`` returns ``df`` unchanged.
    ``avg`` = share of workout days flagged in the window, ``max`` = flagged at
    least once, ``most_recent`` = previous workout day's value. Days where a
    flag is missing are skipped.
    """
    binary_cols = _as_list(binary_cols)
    if not binary_cols:
        return df
    daily = df.groupby('date', sort=True)[binary_cols].max().astype(float).reset_index()
    feats = _rolling_engine(daily, binary_cols, time_lag_days, metrics)
    return _merge_back(df, feats, ['date'])


# --------------------------------------------------------------------------- #
# 5. Calendar features
# --------------------------------------------------------------------------- #
def add_time_features(
    df: pd.DataFrame,
    date_col: str = 'date',
    cyclical: bool = True,
    origin: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Week of year, month, weekday and friends.

    ``days_since_start`` is a trend term measured from ``origin`` (default: the
    earliest date in ``df``). Pass a fixed origin when featurizing new data so
    the scale matches training. ``cyclical`` adds sin/cos pairs, which help
    linear models and neural nets; tree models don't need them.
    """
    out = df.copy()
    d = pd.to_datetime(out[date_col])
    origin = pd.Timestamp(origin) if origin is not None else d.min()

    out['week_of_year'] = d.dt.isocalendar().week.astype('Int64').to_numpy()
    out['month'] = d.dt.month.astype('Int64')
    out['weekday'] = d.dt.day_of_week.astype('Int64')  # 0 = Monday
    out['day_of_year'] = d.dt.day_of_year.astype('Int64')
    out['quarter'] = d.dt.quarter.astype('Int64')
    out['is_weekend'] = (d.dt.day_of_week >= 5).astype('Int64')
    out['days_since_start'] = (d - origin).dt.days.astype('Int64')

    if cyclical:
        for col, period in [('week_of_year', 52.1775), ('month', 12), ('weekday', 7)]:
            vals = out[col].astype(float)
            if col == 'month':
                vals = vals - 1
            angle = 2 * np.pi * vals / period
            out[f'{col}_sin'] = np.sin(angle)
            out[f'{col}_cos'] = np.cos(angle)
    return out


# --------------------------------------------------------------------------- #
# 6. Extra features
# --------------------------------------------------------------------------- #
def add_pr_features(df: pd.DataFrame, group_cols: Sequence[str] = ('exercise', 'set_number'),
                    value_col: str = 'est_1rm') -> pd.DataFrame:
    """All-time best ``value_col`` before today, and the last session as % of it."""
    group_cols = list(group_cols)
    keys = group_cols + ['date']
    daily = df.groupby(keys, dropna=False, sort=True)[value_col].max().astype(float).reset_index()
    g = daily.groupby(group_cols, sort=False, dropna=False)[value_col]
    prev = g.shift(1)
    pr = prev.groupby([daily[c] for c in group_cols], sort=False, dropna=False).cummax()
    daily[f'{value_col}_pr_to_date'] = pr
    daily[f'{value_col}_pct_of_pr'] = prev / pr
    return _merge_back(df, daily.drop(columns=value_col), keys)


def add_smoothed_features(df: pd.DataFrame, ewma_alphas: Iterable[float] = DEFAULT_EWMA_ALPHAS,
                          last_n: Iterable[int] = DEFAULT_LAST_N,
                          group_cols: Sequence[str] = ('exercise', 'set_number'),
                          value_col: str = 'est_1rm') -> pd.DataFrame:
    """EWMA and last-n-session averages of ``value_col`` per exercise + set, before today.

    Counted in sessions, not calendar days. ``est_1rm_ewma_50`` is the EWMA with alpha 0.5.
    """
    group_cols = list(group_cols)
    keys = group_cols + ['date']
    daily = df.groupby(keys, dropna=False, sort=True)[value_col].max().astype(float).reset_index()
    g = daily.groupby(group_cols, sort=False, dropna=False)[value_col]
    feats = daily[keys].copy()
    for a in ewma_alphas:
        feats[f'{value_col}_ewma_{round(a * 100)}'] = g.transform(
            lambda s, a=a: s.ewm(alpha=a, adjust=False).mean().shift(1))
    for n in last_n:
        feats[f'{value_col}_mean_last{n}'] = g.transform(
            lambda s, n=n: s.rolling(n, min_periods=1).mean().shift(1))
    return _merge_back(df, feats, keys)


def add_relative_features(df: pd.DataFrame, value_col: str = 'est_1rm') -> pd.DataFrame:
    """``<feature>_vs_recent`` = feature / most recent session - 1.

    The raw levels are almost identical to ``est_1rm_most_recent`` (|rho| > 0.95), so a
    correlation filter drops them. How far each one sits from the last session is the signal.
    """
    recent = df[f'{value_col}_most_recent'].astype(float)
    prefixes = tuple(f'{value_col}_{m}_' for m in ('avg', 'max', 'min', 'ewma')) + (f'{value_col}_mean_last',)
    cols = [c for c in df.columns if c.startswith(prefixes)]
    rel = pd.DataFrame({f'{c}_vs_recent': df[c].astype(float) / recent - 1 for c in cols}, index=df.index)
    return pd.concat([df, rel], axis=1)


def add_volume_features(df: pd.DataFrame, windows: Iterable[int] = (7, 14)) -> pd.DataFrame:
    """Rolling volume load (sum of weight x reps) before today, per exercise and per gym_day."""
    vol = df['weight'].astype(float) * df['reps'].astype(float)
    out = df
    for level, col_name in [('exercise', 'volume_exercise'), ('gym_day', 'volume_gym_day')]:
        daily = (vol.groupby([df[level], df['date']], sort=True).sum()
                 .rename(col_name).reset_index())
        feats = _rolling_engine(daily, [col_name], windows, ['sum'], group_cols=[level])
        out = _merge_back(out, feats, [level, 'date'])
    return out


def add_recovery_features(df: pd.DataFrame) -> pd.DataFrame:
    """Days since the previous workout of any kind, and of the same gym_day."""
    dates = pd.Series(sorted(df['date'].unique()), name='date')
    any_ws = pd.DataFrame({'date': dates,
                           'days_since_last_workout': dates.diff().dt.days.astype('Int64')})
    out = _merge_back(df, any_ws, ['date'])

    gd = df[['gym_day', 'date']].drop_duplicates().sort_values(['gym_day', 'date'])
    gd['days_since_last_same_gym_day'] = (
        gd.groupby('gym_day')['date'].diff().dt.days.astype('Int64'))
    return _merge_back(out, gd, ['gym_day', 'date'])


def add_date_level_lags(
    df: pd.DataFrame,
    cols: Sequence[str],
    time_lag_days: Iterable[int],
    metrics: Iterable[str] = ('avg',),
) -> pd.DataFrame:
    """Rolling/lagged history of workout-level numeric columns (one value per date)."""
    cols = _as_list(cols)
    daily = df.groupby('date', sort=True)[cols].mean().astype(float).reset_index()
    feats = _rolling_engine(daily, cols, time_lag_days, metrics)
    return _merge_back(df, feats, ['date'])


def encode_categoricals(df: pd.DataFrame, cols: Sequence[str] = CATEGORICAL_COLS,
                        method: str = 'category') -> pd.DataFrame:
    """``'category'`` dtype (LightGBM/XGBoost/CatBoost) or ``'onehot'`` dummies."""
    cols = [c for c in cols if c in df.columns]
    if method == 'category':
        out = df.copy()
        for c in cols:
            out[c] = out[c].astype('category')
        return out
    if method == 'onehot':
        return pd.get_dummies(df, columns=cols, prefix=cols, dtype='int8')
    raise ValueError("method must be 'category' or 'onehot'")


# --------------------------------------------------------------------------- #
# 7. Pipeline
# --------------------------------------------------------------------------- #
def build_features(
    df: pd.DataFrame,
    time_lag_days: Iterable[int] = DEFAULT_LAGS,
    metrics: Iterable[str] = DEFAULT_METRICS,
    binary_cols: Sequence[str] | None = None,
    binary_metrics: Iterable[str] = ('avg', 'max'),
    include_extras: bool = True,
    cyclical: bool = True,
    ewma_alphas: Iterable[float] = DEFAULT_EWMA_ALPHAS,
    last_n: Iterable[int] = DEFAULT_LAST_N,
) -> pd.DataFrame:
    """Full feature table: one row per input set, same row count as ``df``.

    ``count`` is always added to the est_1rm metrics: ``est_1rm_count_30d`` is
    the number of sessions of that exercise + set in the last 30 days, which
    doubles as a training-frequency feature.
    """
    time_lag_days = list(time_lag_days)
    e1rm_metrics = list(dict.fromkeys(list(metrics) + ['count']))

    out = prepare(df)
    out = add_rolling_features(out, time_lag_days, e1rm_metrics, value_col='est_1rm')
    out = add_binary_features(out, time_lag_days, binary_cols, binary_metrics)
    out = add_time_features(out, cyclical=cyclical)

    if include_extras:
        # Previous session's raw weight / reps / rpe for this exercise + set
        out = add_rolling_features(out, [1], [], value_col=['weight', 'reps', 'rpe'],
                                   days_since_col=None)
        if 'days_since_last_exercise_set' in out:
            gap = out['days_since_last_exercise_set']
            out['is_return_from_break'] = gap.gt(14).astype('Int64').mask(gap.isna())
        out = add_pr_features(out)
        out = add_smoothed_features(out, ewma_alphas, last_n)
        if 'est_1rm_most_recent' in out:
            out = add_relative_features(out)
        out = add_volume_features(out)
        out = add_recovery_features(out)
        out = add_date_level_lags(out, ['body_weight'], [14, 30, 1], ['avg', 'slope'])
        out = add_date_level_lags(out, ['workout_rating', 'pump_rating'], [1])
    return out


def get_feature_columns(df: pd.DataFrame, allow_current_day_flags: Sequence[str] = ()) -> list[str]:
    """Columns safe to use as model inputs (drops ids, same-session values and raw flags).

    Raw current-day flags are excluded by default. Pass names in
    ``allow_current_day_flags`` (e.g. ``['injury_ind', 'illness_ind']``) if you
    know them before lifting.
    """
    allowed = set(allow_current_day_flags) - set(SAME_SESSION_COLS)
    drop = set(ID_COLS) | set(SAME_SESSION_COLS) | (set(BINARY_COLS) - allowed) | {'body_weight'}
    return [c for c in df.columns if c not in drop]
