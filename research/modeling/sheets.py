"""Load the raw set log (Google Sheets or a saved file) and fix column types.

Type casting and the bad-data guard are shared with production (``workout_ml.core.cleaning``).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from workout_ml.core.cleaning import cast_raw, clean_raw  # noqa: F401

from .config import CREDENTIALS_PATH, SPREADSHEET, WORKSHEET


def load_sheet(credentials: str | Path = CREDENTIALS_PATH, spreadsheet: str = SPREADSHEET,
               worksheet: str = WORKSHEET) -> pd.DataFrame:
    """Read the worksheet as-is (all values as the sheet returns them)."""
    import gspread
    credentials = Path(credentials)
    if not credentials.exists():
        raise FileNotFoundError(f'Service-account key not found: {credentials}')
    gc = gspread.service_account(filename=str(credentials))
    ws = gc.open(spreadsheet).worksheet(worksheet)
    df = pd.DataFrame(ws.get_all_records())
    df.columns = df.columns.str.lower()
    return df


def read_raw_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    if path.suffix == '.csv':
        return pd.read_csv(path)
    raise ValueError(f'Unsupported raw file type: {path.suffix} (use .parquet or .csv)')


def load_raw_data(source: str | Path | None = None, credentials: str | Path = CREDENTIALS_PATH) -> pd.DataFrame:
    """Casted raw data from Google Sheets (``source=None``) or a saved .parquet/.csv file."""
    raw = load_sheet(credentials) if source is None else read_raw_file(source)
    return cast_raw(raw)
