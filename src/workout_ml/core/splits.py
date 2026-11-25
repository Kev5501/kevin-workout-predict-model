"""Date-grouped holdout and walk-forward splits. All sets from one day stay together."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def holdout_split(dates: pd.Series, frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Positional indices (dev, holdout); holdout = the last ``frac`` of workout dates."""
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    uniq = np.sort(dates.unique())
    n_hold = max(1, math.ceil(frac * len(uniq)))
    cutoff = uniq[-n_hold]
    is_hold = (dates >= cutoff).to_numpy()
    return np.flatnonzero(~is_hold), np.flatnonzero(is_hold)


class WalkForwardSplit:
    """Expanding-window folds over unique dates.

    The first ``min_train_frac`` of dates is always training. The rest is cut into
    ``n_splits`` contiguous validation blocks; each fold trains on every date
    before its block.
    """

    def __init__(self, n_splits: int = 5, min_train_frac: float = 0.6):
        self.n_splits = n_splits
        self.min_train_frac = min_train_frac

    def split(self, dates: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
        dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
        uniq = np.sort(dates.unique())
        n_train0 = int(len(uniq) * self.min_train_frac)
        blocks = np.array_split(uniq[n_train0:], self.n_splits)
        if n_train0 == 0 or any(len(b) == 0 for b in blocks):
            raise ValueError(f'Not enough dates ({len(uniq)}) for {self.n_splits} folds')
        d = dates.to_numpy()
        folds = []
        for block in blocks:
            start, end = block[0], block[-1]
            folds.append((np.flatnonzero(d < start), np.flatnonzero((d >= start) & (d <= end))))
        return folds

    @staticmethod
    def describe(dates: pd.Series, folds) -> pd.DataFrame:
        dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
        rows = []
        for i, (tr, va) in enumerate(folds, 1):
            rows.append({
                'fold': i,
                'train_start': dates.iloc[tr].min().date(), 'train_end': dates.iloc[tr].max().date(),
                'val_start': dates.iloc[va].min().date(), 'val_end': dates.iloc[va].max().date(),
                'train_rows': len(tr), 'val_rows': len(va),
                'val_workouts': dates.iloc[va].nunique(),
            })
        return pd.DataFrame(rows)
