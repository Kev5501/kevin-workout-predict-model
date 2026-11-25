"""Load the saved feature table. Model-input code is shared with production."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from workout_ml.core.data import Dataset, make_dataset, to_model_frame  # noqa: F401

from .config import DATA_PATH


def load_features(path: str | Path = DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f'{path} not found. Build it first: python -m modeling.build_features')
    df = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['date'])
    return df
