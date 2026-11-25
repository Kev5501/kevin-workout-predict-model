from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from ...config import TrainMode
from ...training import triggers
from ...training.checks import check
from ..deps import ServicesDep

router = APIRouter(prefix='/v1/training', tags=['training'])


class TrainRequest(BaseModel):
    mode: TrainMode = 'full'
    reason: str = 'requested through the API'


@router.post('/check')
def run_check(svc: ServicesDep) -> dict:
    """Evaluate the triggers and start the trainer if one is due."""
    decision: triggers.Decision = check(svc)
    execution = None
    if decision.train:
        execution = svc.launcher.launch(decision.mode, decision.trigger, decision.reason)
    return {**decision.to_dict(), 'execution': execution}


@router.post('/run', status_code=202)
def run(svc: ServicesDep, body: TrainRequest) -> dict:
    """Start a training run now, whatever the triggers say."""
    return {'execution': svc.launcher.launch(body.mode, 'manual', body.reason), 'mode': body.mode}


@router.get('/runs')
def runs(svc: ServicesDep, limit: int = 20) -> list[dict]:
    return sorted(svc.docs.list('training_runs'), key=lambda r: r['started'], reverse=True)[:limit]
