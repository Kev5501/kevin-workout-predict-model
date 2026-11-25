"""Turn the feature table into model inputs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import features as F
from .constants import BASE_COL, DAYS_SINCE_COL, TARGET


@dataclass
class Dataset:
    X: pd.DataFrame            # candidate features + 'date' (date is for splitting/SARIMA only)
    y: pd.Series               # est_1rm
    features: list[str]        # candidate feature names (inference whitelist)
    categorical: list[str]
    n_rows_in: int
    n_dropped_no_history: int
    n_dropped_no_target: int

    @property
    def dates(self) -> pd.Series:
        return self.X['date']

    def subset(self, idx) -> Dataset:
        return Dataset(self.X.iloc[idx], self.y.iloc[idx], self.features, self.categorical,
                       self.n_rows_in, self.n_dropped_no_history, self.n_dropped_no_target)


def to_model_frame(feats: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Candidate features (numeric as float, categoricals as str) plus 'date'.

    Also used at inference on freshly built feature rows.
    """
    features = features if features is not None else F.get_feature_columns(feats)
    cats = [c for c in F.CATEGORICAL_COLS if c in features]
    cols = {}
    for c in features:
        if c in cats:
            cols[c] = feats[c].astype(object).where(feats[c].notna(), np.nan)
        else:
            cols[c] = pd.to_numeric(feats[c], errors='coerce').astype(float)
    cols['date'] = pd.to_datetime(feats['date'])
    return pd.DataFrame(cols, index=feats.index)


def make_dataset(feats: pd.DataFrame) -> Dataset:
    """Apply the inference whitelist and drop rows that can't be scored."""
    features = F.get_feature_columns(feats)
    missing = [c for c in (BASE_COL, DAYS_SINCE_COL, 'exercise', 'set_number') if c not in features]
    if missing:
        raise ValueError(f'Feature table is missing required columns: {missing}')

    n_in = len(feats)
    no_target = feats[TARGET].isna()
    no_history = feats[BASE_COL].isna() & ~no_target
    keep = ~(no_target | no_history)

    sort_cols = ['date', 'set_id'] if 'set_id' in feats else ['date']
    kept = feats[keep].sort_values(sort_cols, kind='stable').reset_index(drop=True)
    X = to_model_frame(kept, features)
    y = kept[TARGET].astype(float).rename(TARGET)
    cats = [c for c in F.CATEGORICAL_COLS if c in features]
    return Dataset(X, y, features, cats, n_in, int(no_history.sum()), int(no_target.sum()))
