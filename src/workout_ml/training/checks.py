"""Trigger evaluation. Kept apart from ``run`` so the API image needs no training libraries."""
from __future__ import annotations

from datetime import UTC, datetime

from ..services import Services
from . import triggers


def check(svc: Services, now: datetime | None = None) -> triggers.Decision:
    """Evaluate the triggers and reset the streak if performance recovered. Doesn't train."""
    now = now or datetime.now(UTC)
    t = svc.settings.training
    champion = svc.registry.champion_version()
    perf = triggers.live_performance(svc.docs.list('scored_predictions'), champion, t, now)
    streak = triggers.streak_after_check(svc.docs.get('state', 'performance_streak') or {}, perf)
    svc.docs.set('state', 'performance_streak', streak)
    return triggers.decide(t, champion, svc.docs.list('training_runs'), perf, streak, now)
