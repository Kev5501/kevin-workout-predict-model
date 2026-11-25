"""Per-series SARIMA with the same fit/predict interface as the feature models.

One series per (exercise, set_number), indexed by session order (sessions are
irregular in calendar time), one value per date (daily max, matching
``features.py``).

One-step-ahead without leakage: at predict time the observed history of each
series is rebuilt from the row's own lag columns. ``est_1rm_most_recent`` is
the previous session's value and ``date - days_since_last_exercise_set`` is its
date. Fitted parameters are reused with a Kalman filter over the extended
series (the same as ``res.append(new_obs, refit=False)``), so predictions use
only sessions before each row's date.
"""
from __future__ import annotations

import warnings
from collections import Counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin
# Imported at module load on purpose: statsmodels installs an "always" filter for
# its ConvergenceWarning on first import, which would override the
# catch_warnings() blocks below if the import happened inside them.
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .config import BASE_COL, DAYS_SINCE_COL, SERIES_COLS


def _key(ex, sn) -> tuple:
    return (str(ex), float(sn))


def _sarimax(values, order, seasonal_order):
    # trend='c': a mean when d=0 and a drift when d=1 (SARIMAX applies trend after differencing)
    return SARIMAX(values, order=order, seasonal_order=seasonal_order, trend='c')


def _fit_series(values: np.ndarray, pq, d_opts, m: int) -> dict | None:
    """Pick (p,d,q)[+seasonal] by AIC. Comparing AIC across d is approximate."""
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        candidates = [((p, d, q), (0, 0, 0, 0)) for d in d_opts for p in pq for q in pq]
        for order, seas in candidates:
            try:
                res = _sarimax(values, order, seas).fit(disp=False)
            except Exception:
                continue
            if np.isfinite(res.aic) and (best is None or res.aic < best['aic']):
                best = dict(order=order, seasonal_order=seas, params=np.asarray(res.params), aic=float(res.aic))
        if best and m and len(values) >= 3 * m:
            for seas in [(1, 0, 0, m), (0, 0, 1, m)]:
                try:
                    res = _sarimax(values, best['order'], seas).fit(disp=False)
                except Exception:
                    continue
                if np.isfinite(res.aic) and res.aic < best['aic']:
                    best = dict(order=best['order'], seasonal_order=seas,
                                params=np.asarray(res.params), aic=float(res.aic))
    return best


class SarimaPerSeries(RegressorMixin, BaseEstimator):
    def __init__(self, pq=(0, 1, 2), d=(0, 1), seasonal_m=4, min_obs=20, n_jobs=1,
                 base_col=BASE_COL, days_col=DAYS_SINCE_COL):
        self.pq = pq
        self.d = d
        self.seasonal_m = seasonal_m
        self.min_obs = min_obs
        self.n_jobs = n_jobs
        self.base_col = base_col
        self.days_col = days_col

    def fit(self, X, y):
        df = X[SERIES_COLS + ['date']].copy()
        df['y'] = np.asarray(y, dtype=float)
        daily = (df.groupby(SERIES_COLS + ['date'], sort=True)['y'].max()
                 .dropna().reset_index())
        groups = [(_key(*k), g) for k, g in daily.groupby(SERIES_COLS, sort=True)]
        eligible = [(k, g) for k, g in groups if len(g) >= self.min_obs]
        fitted = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_series)(g['y'].to_numpy(), tuple(self.pq), tuple(self.d), self.seasonal_m)
            for _, g in eligible)
        self.series_ = {}
        for (k, g), res in zip(eligible, fitted):
            if res is not None:
                res['dates'] = g['date'].to_numpy(dtype='datetime64[ns]')
                res['values'] = g['y'].to_numpy(dtype=float)
                self.series_[k] = res
        self.n_series_ = len(groups)
        return self

    def coverage(self, X) -> float:
        """Share of rows whose series has a fitted SARIMA (the rest use naive_last)."""
        keys = [_key(e, s) for e, s in zip(X['exercise'], X['set_number'])]
        return float(np.mean([k in self.series_ for k in keys])) if keys else 0.0

    def orders_summary(self) -> pd.Series:
        return pd.Series(Counter(
            f"{s['order']}{'' if not s['seasonal_order'][3] else ' x ' + str(s['seasonal_order'])}"
            for s in self.series_.values())).sort_values(ascending=False)

    def predict(self, X):
        out = X[self.base_col].to_numpy(dtype=float).copy()   # naive fallback
        keys = pd.Series([_key(e, s) for e, s in zip(X['exercise'], X['set_number'])])
        dates = X['date'].to_numpy(dtype='datetime64[ns]')
        prev_dates = dates - pd.to_timedelta(X[self.days_col].to_numpy(dtype=float), unit='D').to_numpy()
        prev_vals = X[self.base_col].to_numpy(dtype=float)

        for key, idx in keys.groupby(keys).groups.items():
            s = self.series_.get(key)
            if s is None:
                continue
            idx = np.asarray(idx)
            last = s['dates'][-1]
            rev = pd.DataFrame({'d': prev_dates[idx], 'v': prev_vals[idx]})
            rev = rev[(rev['d'] > last) & rev['v'].notna()].groupby('d')['v'].max().sort_index()
            ext_dates = np.concatenate([s['dates'], rev.index.to_numpy(dtype='datetime64[ns]')])
            ext_vals = np.concatenate([s['values'], rev.to_numpy(dtype=float)])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    res = _sarimax(ext_vals, s['order'], s['seasonal_order']).filter(s['params'])
                    preds = np.asarray(res.predict(start=0, end=len(ext_vals)))
            except Exception:
                continue
            m = np.searchsorted(ext_dates, dates[idx], side='left')  # sessions strictly before each row
            p = preds[m]
            ok = np.isfinite(p)
            out[idx[ok]] = p[ok]
        return out
