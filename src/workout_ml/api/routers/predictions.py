from __future__ import annotations

from datetime import date

from fastapi import APIRouter
from pydantic import BaseModel

from ...inference.batch import current_predictions, plan, run_batch, sync_sheet
from ...inference.history import load_history
from ..deps import ServicesDep

router = APIRouter(prefix='/v1', tags=['predictions'])


class RunRequest(BaseModel):
    run_date: date | None = None      # default: today in the configured timezone
    dry_run: bool = False             # predict and return, write nothing


@router.post('/predictions/run')
def run(svc: ServicesDep, body: RunRequest | None = None) -> dict:
    """Score finished workouts, predict the next non-overlapping batch, publish it."""
    body = body or RunRequest()
    return run_batch(svc.settings, svc.docs, svc.registry, svc.sheets, body.run_date or svc.today(), body.dry_run)


@router.get('/predictions/current')
def current(svc: ServicesDep) -> list[dict]:
    return current_predictions(svc.docs)


@router.post('/predictions/sync')
def sync(svc: ServicesDep) -> dict:
    """Rewrite the output tab from the stored predictions."""
    return {'rows': sync_sheet(svc.settings, svc.docs, svc.sheets)}


@router.get('/schedule/preview')
def preview(svc: ServicesDep, run_date: date | None = None) -> dict:
    """Which workouts and sets a run would predict. Needs no model."""
    day = run_date or svc.today()
    history = load_history(svc.sheets, svc.docs, svc.settings.inference.max_row_drop, day)
    workouts, templates = plan(history, svc.settings, day)
    return {'run_date': day.isoformat(), 'workouts': [
        {'date': w.date.isoformat(), 'weekday': w.weekday, 'gym_day': w.gym_day,
         'sets': templates[w.gym_day][['exercise', 'set_number']].to_dict(orient='records')} for w in workouts]}
