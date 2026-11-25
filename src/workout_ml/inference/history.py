"""Load and validate the workout history before anything is written."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd

from ..core.cleaning import cast_raw, clean_raw
from ..core.constants import FEATURE_SETTINGS
from ..io.documents import DocumentStore
from ..io.sheets import SheetsGateway

log = logging.getLogger(__name__)
REQUIRED = ['date', 'set_id', 'gym_day', 'exercise', 'type', 'set_number', 'weight', 'reps', 'rpe',
            'rpe_norm_weight', 'body_weight', 'workout_rating', 'pump_rating', *FEATURE_SETTINGS['binary_cols']]


class UnsafeHistory(RuntimeError):
    """The source data looks wrong; nothing was written."""


@dataclass
class History:
    raw: pd.DataFrame        # cast, not cleaned
    clean: pd.DataFrame      # without sets flagged by the bad-data guard
    n_flagged: int

    @property
    def last_date(self) -> date:
        return self.clean['date'].max().date()


def load_history(sheets: SheetsGateway, docs: DocumentStore, max_row_drop: float, as_of: date) -> History:
    df = sheets.read_history()
    if df.empty:
        raise UnsafeHistory('the source sheet is empty')
    missing = [c for c in REQUIRED if c not in {str(c).lower() for c in df.columns}]
    if missing:
        raise UnsafeHistory(f'the source sheet is missing columns: {missing}')
    df = df.rename(columns=lambda c: str(c).lower())
    blank = df[['date', 'set_id', 'exercise']].astype(str).apply(lambda s: s.str.strip()).isin(['', 'nan', 'None'])
    raw = cast_raw(df[~blank.all(axis=1)].reset_index(drop=True))     # ignore empty sheet rows
    if raw['date'].isna().any() or raw['set_number'].isna().any():
        raise UnsafeHistory('the source sheet has rows without a date or set number')
    before = docs.get('state', 'source') or {}
    if before.get('rows') and len(raw) < before['rows'] * (1 - max_row_drop):
        raise UnsafeHistory(f'the source sheet shrank from {before["rows"]} to {len(raw)} rows')
    future = raw['date'] > pd.Timestamp(as_of)
    if future.any():
        log.warning('ignoring rows dated in the future', extra={'rows': int(future.sum())})
        raw = raw[~future].reset_index(drop=True)
    # Planned sets may be pre-filled without a result; they are not history.
    done = raw['rpe_norm_weight'].notna()
    clean, flagged = clean_raw(raw[done].reset_index(drop=True))
    if clean.empty:
        raise UnsafeHistory('no completed sets in the source sheet')
    if len(flagged):
        log.warning('sets excluded by the bad-data guard', extra={'set_ids': flagged['set_id'].tolist()})
    return History(raw, clean, len(flagged))


def remember_size(docs: DocumentStore, history: History) -> None:
    docs.set('state', 'source', {'rows': len(history.raw), 'last_date': history.last_date.isoformat()})
