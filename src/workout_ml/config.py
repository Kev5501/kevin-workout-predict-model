"""Settings: ``configs/pipeline.yaml`` with ``WML_*`` environment overrides."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource

WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
REST = 'Rest'
TrainMode = Literal['full', 'tune', 'refit']


def config_path() -> Path:
    default = Path(__file__).resolve().parents[2] / 'configs' / 'pipeline.yaml'
    path = Path(os.environ.get('WML_CONFIG_PATH', default))
    if not path.exists():                 # otherwise every day would silently be a rest day
        raise FileNotFoundError(f'config file not found: {path} (set WML_CONFIG_PATH)')
    return path


class AppSettings(BaseModel):
    timezone: str = 'America/New_York'
    log_level: str = 'INFO'
    backend: Literal['local', 'gcp'] = 'local'
    local_dir: Path = Path('.local')


class GcpSettings(BaseModel):
    project: str = ''
    region: str = 'us-east1'
    bucket: str = ''
    trainer_job: str = 'workout-trainer'


class SheetsSettings(BaseModel):
    spreadsheet_id: str = ''
    source_tab: str = 'Enriched_Data'
    output_tab: str = 'Predictions'
    credentials_path: str = ''            # local only; in the cloud the runtime identity is used
    local_history: str = ''               # backend=local: raw .parquet/.csv instead of Google Sheets

    @model_validator(mode='after')
    def _tabs_differ(self):
        if self.source_tab.strip().lower() == self.output_tab.strip().lower():
            raise ValueError('sheets.output_tab must differ from sheets.source_tab')
        return self


class InferenceSettings(BaseModel):
    lookback_days: int = Field(30, ge=1)
    max_horizon_days: int = Field(7, ge=1, le=28)
    reps: list[int] = [5, 6, 7, 8, 9, 10]
    rpe: float = Field(10, ge=1, le=10)
    round_to: float = Field(2.5, gt=0)
    match_window_days: int = Field(3, ge=0)
    max_row_drop: float = Field(0.05, ge=0, lt=1)
    max_jump: float = Field(0.30, gt=0)
    on_jump: Literal['abort', 'warn'] = 'abort'


class ScheduledTraining(BaseModel):
    enabled: bool = True
    every_days: int = Field(30, ge=1)
    mode: TrainMode = 'full'


class PerformanceTraining(BaseModel):
    enabled: bool = True
    mode: TrainMode = 'tune'
    window_days: int = Field(14, ge=1)
    max_mape: float = Field(3.75, gt=0)
    min_sets: int = Field(40, ge=1)
    cooldown_days: int = Field(7, ge=0)
    alert_after_consecutive: int = Field(3, ge=1)
    pause_after_alert: bool = False


class Promotion(BaseModel):
    auto: bool = True
    eval_workouts: int = Field(20, ge=1)
    max_relative_worse: float = Field(0.05, ge=0)
    must_beat_naive: bool = True


class Tuning(BaseModel):
    n_trials: int = Field(100, ge=1)
    timeout_minutes: float = Field(15, gt=0)
    trial_jobs: int = Field(4, ge=1)
    n_threads: int = Field(4, ge=1)


class TrainingSettings(BaseModel):
    model_name: str = 'lightgbm'
    seed: int = 42
    retry_failed_after_days: float = Field(3, ge=0)
    scheduled: ScheduledTraining = ScheduledTraining()
    performance: PerformanceTraining = PerformanceTraining()
    promotion: Promotion = Promotion()
    tuning: Tuning = Tuning()


class AlertSettings(BaseModel):
    enabled: bool = False
    smtp_host: str = 'smtp.gmail.com'
    smtp_port: int = 465
    smtp_user: str = ''
    smtp_password: SecretStr = SecretStr('')
    to: str = ''


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='WML_', env_nested_delimiter='__', extra='ignore')

    app: AppSettings = AppSettings()
    gcp: GcpSettings = GcpSettings()
    sheets: SheetsSettings = SheetsSettings()
    schedule: dict[str, str] = {}
    inference: InferenceSettings = InferenceSettings()
    training: TrainingSettings = TrainingSettings()
    alerts: AlertSettings = AlertSettings()

    @field_validator('schedule')
    @classmethod
    def _weekdays(cls, value: dict[str, str]) -> dict[str, str]:
        out = {}
        for day, workout in value.items():
            name = day.strip().capitalize()
            if name not in WEEKDAYS:
                raise ValueError(f'schedule: unknown weekday {day!r}')
            out[name] = workout.strip()
        return {d: out.get(d, REST) for d in WEEKDAYS}

    @model_validator(mode='after')
    def _cloud_values(self):
        if self.app.backend == 'gcp':
            missing = [n for n, v in {'WML_GCP__PROJECT': self.gcp.project, 'WML_GCP__BUCKET': self.gcp.bucket,
                                      'WML_SHEETS__SPREADSHEET_ID': self.sheets.spreadsheet_id}.items() if not v]
            if missing:
                raise ValueError(f'backend=gcp needs {", ".join(missing)}')
        if self.alerts.enabled and not (self.alerts.to and self.alerts.smtp_user):
            raise ValueError('alerts.enabled needs WML_ALERTS__TO and WML_ALERTS__SMTP_USER')
        return self

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings,
                                   file_secret_settings):
        yaml_source = YamlConfigSettingsSource(settings_cls, yaml_file=config_path())
        return init_settings, env_settings, yaml_source


@lru_cache
def get_settings() -> Settings:
    return Settings()
