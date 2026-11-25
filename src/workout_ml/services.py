"""Wire settings to concrete services. The only place that picks local or cloud implementations."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from zoneinfo import ZoneInfo

from .config import Settings, get_settings
from .io.documents import DocumentStore, FirestoreStore, LocalStore
from .io.jobs import CloudRunJobLauncher, InProcessLauncher, TrainerLauncher
from .io.notify import Notifier, build_notifier
from .io.registry import GcsArtifacts, LocalArtifacts, ModelRegistry
from .io.sheets import GoogleSheets, LocalSheets, SheetsGateway
from .logs import configure_logging


@dataclass
class Services:
    settings: Settings
    docs: DocumentStore
    registry: ModelRegistry
    sheets: SheetsGateway
    notifier: Notifier
    launcher: TrainerLauncher | None = None
    today: Callable[[], date] = field(default=date.today)


def build_services(settings: Settings | None = None) -> Services:
    s = settings or get_settings()
    configure_logging(s.app.log_level)
    tz = ZoneInfo(s.app.timezone)

    def today() -> date:
        return datetime.now(tz).date()

    if s.app.backend == 'gcp':
        docs: DocumentStore = FirestoreStore(s.gcp.project)
        artifacts = GcsArtifacts(s.gcp.bucket, s.gcp.project)
        sheets: SheetsGateway = GoogleSheets(s.sheets.spreadsheet_id, s.sheets.source_tab, s.sheets.output_tab)
        launcher: TrainerLauncher | None = CloudRunJobLauncher(s.gcp.project, s.gcp.region, s.gcp.trainer_job)
    else:
        root = s.app.local_dir
        docs = LocalStore(root / 'documents.json')
        artifacts = LocalArtifacts(root / 'artifacts')
        if s.sheets.local_history:
            sheets = LocalSheets(s.sheets.local_history, root)
        else:
            sheets = GoogleSheets(s.sheets.spreadsheet_id, s.sheets.source_tab, s.sheets.output_tab,
                                  s.sheets.credentials_path)
        launcher = None
    svc = Services(s, docs, ModelRegistry(artifacts, docs), sheets, build_notifier(s.alerts), launcher, today)
    if svc.launcher is None:
        def run_here(mode: str, trigger: str, reason: str) -> dict:
            from .training.run import run_training  # needs the training dependencies
            return run_training(svc, mode, trigger, reason)
        svc.launcher = InProcessLauncher(run_here)
    return svc
