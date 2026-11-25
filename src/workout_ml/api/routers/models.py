from __future__ import annotations

from fastapi import APIRouter

from ..deps import ServicesDep

router = APIRouter(prefix='/v1/models', tags=['models'])


@router.get('')
def list_models(svc: ServicesDep) -> dict:
    return {'champion': svc.registry.champion_version(), 'models': svc.registry.list()}


@router.post('/{version}/promote')
def promote(version: str, svc: ServicesDep) -> dict:
    """Make a version the champion: approve a candidate, or roll back to an older model."""
    svc.registry.load(version)          # refuse to promote something that can't be loaded
    return svc.registry.promote(version)
