from __future__ import annotations

from fastapi import APIRouter, Response

from ... import __version__
from ..deps import ServicesDep

router = APIRouter(tags=['health'])


@router.get('/health')
def health() -> dict:
    """Is the process up? Touches nothing else."""
    return {'status': 'ok', 'version': __version__}


@router.get('/ready')
def ready(svc: ServicesDep, response: Response) -> dict:
    """Is there a champion model to predict with?"""
    version = svc.registry.champion_version()
    if version is None:
        response.status_code = 503
    return {'ready': version is not None, 'champion': version}
