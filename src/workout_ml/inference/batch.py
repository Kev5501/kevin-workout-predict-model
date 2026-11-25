"""The daily batch: score what happened, predict what's next, publish it."""
from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta

from ..config import Settings
from ..errors import Busy
from ..io.documents import DocumentStore, Write
from ..io.registry import ModelRegistry
from ..io.sheets import SheetsGateway
from .history import History, load_history, remember_size
from .ids import prediction_id
from .predict import check_predictions, predict_workout, rep_weights
from .reconcile import CURRENT, PENDING, SCORED, reconcile, retire
from .schedule import PlannedWorkout, plan_batch
from .templates import build_template

log = logging.getLogger(__name__)
LOCK_TTL = timedelta(minutes=10)


def plan(history: History, settings: Settings, run_date: date) -> tuple[list[PlannedWorkout], dict]:
    """The next batch and the template of each gym day in it."""
    inf = settings.inference
    templates: dict = {}

    def template(gym_day: str):
        if gym_day not in templates:
            templates[gym_day] = build_template(history.clean, gym_day, run_date, inf.lookback_days)
        return templates[gym_day]

    logged = {(d.date(), g) for d, g in history.clean[['date', 'gym_day']].drop_duplicates().itertuples(index=False)}
    workouts = plan_batch(run_date, settings.schedule, lambda g: template(g)['exercise'].tolist(),
                          lambda d, g: (d, g) in logged, inf.max_horizon_days)
    return workouts, templates


def run_batch(settings: Settings, docs: DocumentStore, registry: ModelRegistry, sheets: SheetsGateway,
              run_date: date, dry_run: bool = False) -> dict:
    owner = f'batch-{datetime.now(UTC).isoformat()}'
    if not dry_run and not docs.acquire_lock('batch', owner, LOCK_TTL):
        raise Busy('a prediction run is already in progress')
    try:
        return _run(settings, docs, registry, sheets, run_date, dry_run)
    finally:
        if not dry_run:
            docs.release_lock('batch', owner)


def _run(settings, docs, registry, sheets, run_date, dry_run) -> dict:
    inf = settings.inference
    history = load_history(sheets, docs, inf.max_row_drop, run_date)
    bundle = registry.load_champion()
    workouts, templates = plan(history, settings, run_date)
    now = datetime.now(UTC).isoformat()

    new, flagged = [], []
    for w in workouts:
        pred = predict_workout(history.clean, w, templates[w.gym_day], bundle)
        flagged += check_predictions(pred, inf)
        for i, r in enumerate(pred.itertuples(index=False), 1):
            day = w.date.isoformat()
            new.append({
                'id': prediction_id(day, r.exercise, r.set_number), 'workout_date': day, 'weekday': w.weekday,
                'gym_day': w.gym_day, 'exercise': r.exercise, 'set_number': int(r.set_number), 'position': i,
                'est_1rm': round(float(r.est_1rm), 2), 'last_est_1rm': round(float(r.last_est_1rm), 2),
                'weights': rep_weights(float(r.est_1rm), bundle, inf), 'rpe': inf.rpe,
                'model_version': bundle.version, 'generated_at': now,
                'data_through': history.last_date.isoformat()})

    current = docs.list(CURRENT)
    pending = docs.list(PENDING)
    open_docs = [{**d, 'collection': CURRENT} for d in current] + [{**d, 'collection': PENDING} for d in pending]
    scoring = reconcile(open_docs, history.clean, inf.match_window_days)
    scored_ids = {w.doc_id for w in scoring if w.op == 'delete'}
    listed = [{'date': w.date.isoformat(), 'weekday': w.weekday, 'gym_day': w.gym_day} for w in workouts]
    result = {'run_date': run_date.isoformat(), 'model_version': bundle.version, 'n_predictions': len(new),
              'workouts': listed,
              'n_scored': sum(w.collection == SCORED for w in scoring), 'flagged': flagged,
              'data_through': history.last_date.isoformat(),
              'excluded_sets': history.n_flagged, 'dry_run': dry_run}
    if dry_run:
        return {**result, 'predictions': new}

    retired = retire(current, pending, new, scored_ids, run_date, inf.match_window_days)
    docs.commit([*scoring, *retired, *[Write('set', CURRENT, d['id'], d) for d in new],
                 Write('set', 'prediction_batches', run_date.isoformat(), {**result, 'generated_at': now})])
    remember_size(docs, history)
    if flagged:
        log.warning('large changes from the last session', extra={'flagged': flagged})
    result['sheet_synced'] = _try_sync(settings, docs, sheets)
    log.info('prediction batch done', extra={k: v for k, v in result.items() if k != 'flagged'})
    return result


def current_predictions(docs: DocumentStore) -> list[dict]:
    return sorted(docs.list(CURRENT), key=lambda d: (d['workout_date'], d['position']))


def sync_sheet(settings: Settings, docs: DocumentStore, sheets: SheetsGateway) -> int:
    """Write the current batch from the document store to the output tab."""
    reps = [str(r) for r in settings.inference.reps]
    header = ['workout_date', 'weekday', 'gym_day', 'exercise', 'set_number', 'est_1rm',
              *[f'weight_{r}_reps' for r in reps], 'model_version', 'generated_at']
    rows = [[d['workout_date'], d['weekday'], d['gym_day'], d['exercise'], d['set_number'], round(d['est_1rm'], 1),
             *[d['weights'].get(r, '') for r in reps], d['model_version'], d['generated_at']]
            for d in current_predictions(docs)]
    sheets.write_predictions(header, rows)
    return len(rows)


def _try_sync(settings, docs, sheets) -> bool:
    """Predictions are already stored, so a Sheets outage is reported, not fatal (retry with /sync)."""
    try:
        sync_sheet(settings, docs, sheets)
        return True
    except Exception:
        log.exception('writing predictions to Google Sheets failed')
        return False
