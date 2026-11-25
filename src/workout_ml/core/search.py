"""Settings for splits, feature selection and tuning."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class SearchConfig:
    seed: int = 42

    # Splits
    holdout_frac: float = 0.15      # last 15% of workout dates, never used for selection/tuning
    n_splits: int = 5               # expanding-window walk-forward folds on the dev set
    min_train_frac: float = 0.6     # first 60% of dev dates are always training

    # Feature selection
    missing_thresh: float = 0.5
    constant_thresh: float = 0.99
    corr_thresh: float = 0.95
    n_probes: int = 3
    stability_top: int = 30
    perm_repeats: int = 5
    k_grid: list = field(default_factory=lambda: [0, 5, 10, 15, 20, 30, 40, 60, 80, None])  # None = all

    # Tuning
    trial_jobs: int = 4             # Optuna trials run in parallel on CPU
    n_threads: int = field(default_factory=lambda: os.cpu_count() or 4)

    @property
    def model_threads(self) -> int:
        """Threads per model fit when trials run in parallel on CPU."""
        return max(1, self.n_threads // max(1, self.trial_jobs))
