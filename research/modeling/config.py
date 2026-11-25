"""Paths and run settings."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Shared settings live in the production package; re-exported here for the research code.
from workout_ml.core.constants import *  # noqa: F401,F403
from workout_ml.core.search import SearchConfig

TESTING_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TESTING_DIR.parent
DATA_DIR = TESTING_DIR / 'data'
RAW_PATH = DATA_DIR / 'raw_sets.parquet'              # snapshot of the sheet (history for predict_next)
DATA_PATH = DATA_DIR / 'features.parquet'             # pipeline input
FEATURES_CONFIG_PATH = DATA_DIR / 'features_config.json'
FLAGGED_PATH = DATA_DIR / 'flagged_sets.csv'          # sets excluded by the bad-data guard
ARTIFACTS_DIR = TESTING_DIR / 'artifacts'
REPORTS_DIR = TESTING_DIR / 'reports'



def _from_env_file(name: str) -> str:
    env = REPO_ROOT / '.env'
    lines = env.read_text(encoding='utf-8').splitlines() if env.exists() else []
    return next((v.split('=', 1)[1].strip() for v in lines if v.startswith(name + '=')), '')


# Google Sheets source
CREDENTIALS_PATH = REPO_ROOT / 'credentials.json'
SPREADSHEET = os.environ.get('RESEARCH_SPREADSHEET') or _from_env_file('RESEARCH_SPREADSHEET')   # .env, not committed
WORKSHEET = 'Enriched_Data'

# Model used for inference by default, chosen by hand (not by the leaderboard): it is level with
# the best models overall and does better after a break from a lift.
PRODUCTION_MODEL = 'lightgbm'

# Baselines need no training (ewma only picks its alpha on dev CV)
BASELINE_MODELS = ['naive_last', 'naive_drift', 'mean_last2', 'ewma']
SIMPLE_MODELS = ['linear_2feat']
TUNED_MODELS = ['linear_regression', 'elasticnet', 'random_forest', 'lightgbm', 'xgboost', 'catboost']
ALL_MODELS = BASELINE_MODELS + SIMPLE_MODELS + TUNED_MODELS + ['sarima']


@dataclass
class Config(SearchConfig):
    # Tuning
    models: list = field(default_factory=lambda: list(ALL_MODELS))
    trials: dict = field(default_factory=lambda: {
        'elasticnet': 50, 'random_forest': 50,
        'lightgbm': 100, 'xgboost': 100, 'catboost': 60,
    })
    device: str = 'auto'            # auto | cpu | gpu

    # SARIMA
    sarima_min_obs: int = 20
    sarima_pq: tuple = (0, 1, 2)
    sarima_d: tuple = (0, 1)
    sarima_seasonal_m: int = 4      # ~monthly cycle at ~1 session/week; 0 disables

    # Evaluation
    bootstrap_iters: int = 2000
    ensemble_size: int = 3

    quick: bool = False

    def apply_quick(self) -> 'Config':
        self.quick = True
        self.n_splits = 3
        self.k_grid = [0, 5, 10, 20, None]
        self.perm_repeats = 2
        self.trials = {k: 5 for k in self.trials}
        self.sarima_pq = (0, 1)
        self.sarima_seasonal_m = 0
        self.bootstrap_iters = 300
        return self

    def to_dict(self) -> dict:
        return asdict(self)
