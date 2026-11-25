"""Score current predictions once the real sets are logged."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from ..io.documents import Write
from .ids import prediction_id

CURRENT, PENDING, SCORED = 'predictions', 'pending_predictions', 'scored_predictions'


def reconcile(open_docs: list[dict], clean: pd.DataFrame, match_window_days: int) -> list[Write]:
    """Move predictions whose sets have been done to ``scored_predictions``.

    ``open_docs`` are the current predictions plus those pending from a recently missed workout.

    A batch holds each exercise + set once, so that pair identifies the prediction. Only sets logged
    *after* the data the prediction was made from count, within ``match_window_days`` of the planned
    date. Scored documents are never changed again.
    """
    if not open_docs:
        return []
    actual = (clean.groupby(['exercise', 'set_number', 'date'], as_index=False)['rpe_norm_weight'].max()
              .sort_values('date'))
    by_key = {k: g for k, g in actual.groupby(['exercise', 'set_number'])}
    writes = []
    for doc in open_docs:
        done = by_key.get((doc['exercise'], doc['set_number']))
        if done is None:
            continue
        planned = pd.Timestamp(doc['workout_date'])
        window = pd.Timedelta(days=match_window_days)
        hit = done[(done['date'] > pd.Timestamp(doc['data_through']))
                   & (done['date'] >= planned - window) & (done['date'] <= planned + window)]
        if hit.empty:
            continue
        row = hit.iloc[0]
        value = float(row['rpe_norm_weight'])
        day = row['date'].date().isoformat()
        scored = {k: v for k, v in doc.items() if k != 'collection'}
        scored |= {'actual_date': day, 'actual_est_1rm': value,
                   'ape': abs(doc['est_1rm'] - value) / value * 100}
        writes.append(Write('set', SCORED, prediction_id(day, doc['exercise'], doc['set_number']), scored))
        writes.append(Write('delete', doc['collection'], doc['id']))
    return writes


def retire(current: list[dict], pending: list[dict], new: list[dict], scored_ids: set[str], run_date: date,
           match_window_days: int) -> list[Write]:
    """What happens to open predictions that were not scored and are not in the new batch.

    A missed workout may still be done a day or two late, so its predictions wait in
    ``pending_predictions`` until the match window passes or the lift gets a newer prediction.
    """
    def key(d):
        return d['exercise'], d['set_number']

    new_ids, new_keys = {d['id'] for d in new}, {key(d) for d in new}

    def may_wait(d) -> bool:
        planned = date.fromisoformat(d['workout_date'])
        return planned < run_date <= planned + timedelta(days=match_window_days) and key(d) not in new_keys

    writes = []
    for d in current:
        if d['id'] in scored_ids or d['id'] in new_ids:
            continue
        writes.append(Write('delete', CURRENT, d['id']))
        if may_wait(d):
            writes.append(Write('set', PENDING, d['id'], d))
    for d in pending:
        if d['id'] not in scored_ids and not may_wait(d):
            writes.append(Write('delete', PENDING, d['id']))
    return writes
