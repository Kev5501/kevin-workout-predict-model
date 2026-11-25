"""Workout templates: the exercise + set combinations to predict for a gym day."""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd

log = logging.getLogger(__name__)


def build_template(history: pd.DataFrame, gym_day: str, as_of: date, lookback_days: int) -> pd.DataFrame:
    """Every exercise + set done on ``gym_day`` in the ``lookback_days`` up to ``as_of``, in workout order.

    Falls back to the most recent session if the window is empty. Columns: exercise, set_number, type.
    """
    cols = ['exercise', 'set_number', 'type']
    end = pd.Timestamp(as_of)
    rows = history[(history['gym_day'] == gym_day) & (history['date'] <= end)]
    if rows.empty:
        return pd.DataFrame(columns=cols)
    window = rows[rows['date'] > end - pd.Timedelta(days=lookback_days)]
    if window.empty:
        last = rows['date'].max()
        log.warning('gym day not trained in the lookback window; using its latest session',
                    extra={'gym_day': gym_day, 'latest': str(last.date())})
        window = rows[rows['date'] == last]

    window = window.sort_values(['date', 'set_id'], kind='stable')
    position = window.groupby(['date', 'exercise'], sort=False).ngroup()      # order within the log
    start = position.groupby(window['date']).transform('min')
    window = window.assign(order=position - start)
    order = window.groupby('exercise')['order'].last()                        # from its latest session
    latest = window.groupby(['exercise', 'set_number'], as_index=False).last()
    latest['order'] = latest['exercise'].map(order)
    return latest.sort_values(['order', 'exercise', 'set_number'], ignore_index=True)[cols]


def future_rows(template: pd.DataFrame, gym_day: str, day: date) -> pd.DataFrame:
    """Blank set rows for a planned workout, ready for feature building."""
    out = template.copy()
    out['date'] = pd.Timestamp(day)
    out['gym_day'] = gym_day
    out['set_id'] = [f'FUT-{i:05d}' for i in range(1, len(out) + 1)]
    return out
