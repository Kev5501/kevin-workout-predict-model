"""One training run, start to finish. Entry point of the trainer container.

    python -m workout_ml.training.run --mode full --trigger manual
"""
from __future__ import annotations

import argparse
import hashlib
import io
import logging
import os
import sys
from datetime import UTC, datetime, timedelta

from ..errors import Busy
from ..inference.history import load_history
from ..io.registry import ModelBundle
from ..services import Services, build_services
from . import triggers
from .train import train

log = logging.getLogger(__name__)
LOCK_TTL = timedelta(minutes=60)
RUNS, STATE = 'training_runs', 'state'


def run_training(svc: Services, mode: str, trigger: str, reason: str, now: datetime | None = None) -> dict:
    """Train, gate, register, promote, and record it. The champion only changes on a passed gate."""
    now = now or datetime.now(UTC)
    sha = os.environ.get('GIT_SHA', 'local')[:7]
    run_id = f'{now:%Y%m%d-%H%M%S}{now.microsecond // 1000:03d}-{sha}'       # sorts by time; also the version
    if not svc.docs.acquire_lock('training', run_id, LOCK_TTL):
        raise Busy('a training run is already in progress')
    record = dict(run_id=run_id, status='running', mode=mode, trigger=trigger, reason=reason,
                  started=now.isoformat(), git_sha=sha)
    svc.docs.set(RUNS, run_id, record)
    try:
        record |= _train_and_register(svc, run_id, mode, trigger, sha)
    except Exception as e:
        record |= dict(status='failed', error=f'{type(e).__name__}: {e}', finished=datetime.now(UTC).isoformat())
        svc.docs.set(RUNS, run_id, record)
        log.exception('training failed')
        raise
    finally:
        svc.docs.release_lock('training', run_id)
    record['finished'] = datetime.now(UTC).isoformat()
    svc.docs.set(RUNS, run_id, record)
    _update_streak(svc, record)
    log.info('training finished', extra={'run': record})
    return record


def _train_and_register(svc: Services, run_id: str, mode: str, trigger: str, sha: str) -> dict:
    s = svc.settings
    history = load_history(svc.sheets, svc.docs, s.inference.max_row_drop, svc.today())
    champion = svc.registry.champion_version()
    recipe = svc.registry.load(champion).recipe if champion else None
    out = train(history.clean, mode, recipe, s.training)

    buf = io.BytesIO()
    history.raw.to_parquet(buf, index=False)
    snapshot = buf.getvalue()
    gate = out.gate.to_dict()
    bundle = ModelBundle(
        version=run_id, model=out.model, recipe=out.recipe, candidates=out.candidates,
        feature_settings=out.feature_settings, ratio_table=out.ratio_table,
        metadata=dict(trigger=trigger, mode=mode, gate=gate, data_range=out.data_range, rows=out.rows, git_sha=sha,
                      data_sha256=hashlib.sha256(snapshot).hexdigest(), replaced=champion,
                      selection=out.selection, tuning=out.tuning, image=os.environ.get('IMAGE', '')))
    svc.registry.register(bundle, 'candidate' if out.gate.passed else 'rejected', snapshot)
    status = 'rejected'
    if out.gate.passed:
        status = 'candidate'
        if s.training.promotion.auto or champion is None:
            svc.registry.promote(run_id)
            status = 'promoted'
    return dict(status=status, version=run_id, gate=gate, features=out.recipe.features, replaced=champion)


def _update_streak(svc: Services, record: dict) -> None:
    streak = triggers.streak_after_run(svc.docs.get(STATE, 'performance_streak') or {}, record)
    svc.docs.set(STATE, 'performance_streak', streak)
    if triggers.should_alert(streak, record, svc.settings.training):
        subject, body = triggers.alert_text(streak, svc.docs.list(RUNS), svc.settings.training)
        try:
            svc.notifier.send(subject, body)
        except Exception:                      # a mail problem must not fail a finished training run
            log.exception('sending the alert email failed')


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description='Train, evaluate and (if it passes) promote a model')
    p.add_argument('--mode', choices=['full', 'tune', 'refit'], default='full')
    p.add_argument('--trigger', default='manual')
    p.add_argument('--reason', default='started by hand')
    args = p.parse_args(argv)
    svc = build_services()
    try:
        run_training(svc, args.mode, args.trigger, args.reason)
    except Busy as e:
        log.warning(str(e))
    return 0


if __name__ == '__main__':
    sys.exit(main())
