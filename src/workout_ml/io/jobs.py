"""Start the trainer: a Cloud Run job, or in-process for local runs."""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

log = logging.getLogger(__name__)


class TrainerLauncher(Protocol):
    def launch(self, mode: str, trigger: str, reason: str) -> str: ...


class CloudRunJobLauncher:
    def __init__(self, project: str, region: str, job: str, client=None):
        self._name = f'projects/{project}/locations/{region}/jobs/{job}'
        self._client = client

    def launch(self, mode, trigger, reason):
        from google.cloud import run_v2
        client = self._client or run_v2.JobsClient()
        override = run_v2.RunJobRequest.Overrides.ContainerOverride(
            args=['--mode', mode, '--trigger', trigger, '--reason', reason])
        op = client.run_job(request=run_v2.RunJobRequest(
            name=self._name, overrides=run_v2.RunJobRequest.Overrides(container_overrides=[override])))
        execution = op.metadata.name.rsplit('/', 1)[-1]
        log.info('started trainer job', extra={'execution': execution, 'mode': mode, 'trigger': trigger})
        return execution


class InProcessLauncher:
    """Runs training synchronously (local backend and tests)."""

    def __init__(self, run: Callable[[str, str, str], dict]):
        self._run = run

    def launch(self, mode, trigger, reason):
        return self._run(mode, trigger, reason)['run_id']
