"""Walk-forward cross-validation and Optuna tuning."""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from .metrics import fold_metrics, mape
from .models import REGISTRY, BuildContext, build_model


def _quiet():
    warnings.filterwarnings('ignore')


def cv_predict(model, X: pd.DataFrame, y: pd.Series, folds) -> tuple[list[dict], np.ndarray]:
    """Fit on each fold's training dates and predict its validation block.

    Returns per-fold ``{mape, mae, rmse}`` and out-of-fold predictions (NaN outside validation blocks).
    """
    oof = np.full(len(X), np.nan)
    per_fold = []
    with warnings.catch_warnings():
        _quiet()
        for tr, va in folds:
            m = clone(model).fit(X.iloc[tr], y.iloc[tr])
            p = m.predict(X.iloc[va])
            oof[va] = p
            per_fold.append(fold_metrics(y.iloc[va], p))
    return per_fold, oof


def _storage(path: Path):
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
    path = str(path)
    return JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))


def tune_model(name: str, X: pd.DataFrame, y: pd.Series, folds, features: list[str],
               categorical: list[str], device: str, cfg, storage_path: Path | None, n_trials: int,
               log=print, timeout: float | None = None) -> dict:
    """Optuna TPE over the model's search space; objective = mean walk-forward MAPE.

    ``storage_path=None`` keeps trials in memory; ``timeout`` (seconds) stops starting new trials.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    spec = REGISTRY[name]
    gpu = device == 'gpu'
    if spec.grid:                       # exhaustive: one trial per combination, never pruned
        n_trials = int(np.prod([len(v) for v in spec.grid.values()]))
    trial_jobs = 1 if gpu else cfg.trial_jobs
    threads = cfg.n_threads if gpu else cfg.model_threads

    def make_ctx():
        return BuildContext(features, categorical, device=device, n_threads=threads, seed=cfg.seed, cfg=cfg)

    def objective(trial):
        if spec.grid:
            params = {k: trial.suggest_categorical(k, v) for k, v in spec.grid.items()}
        else:
            params = spec.search_space(trial)
        model = build_model(name, params, make_ctx())
        scores = []
        with warnings.catch_warnings():
            _quiet()
            for step, (tr, va) in enumerate(folds):
                m = clone(model).fit(X.iloc[tr], y.iloc[tr])
                p = m.predict(X.iloc[va])
                scores.append(mape(y.iloc[va], p))
                trial.report(float(np.mean(scores)), step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return float(np.mean(scores))

    sampler = (optuna.samplers.GridSampler(spec.grid, seed=cfg.seed) if spec.grid
               else optuna.samplers.TPESampler(seed=cfg.seed, n_startup_trials=min(10, max(2, n_trials // 3))))
    pruner = (optuna.pruners.NopPruner() if spec.grid
              else optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    study = optuna.create_study(study_name=name, direction='minimize', sampler=sampler, pruner=pruner,
                                storage=_storage(storage_path) if storage_path else None, load_if_exists=True)
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = max(0, n_trials - done)
    t0 = time.perf_counter()
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=trial_jobs, gc_after_trial=False,
                       timeout=timeout)
    elapsed = time.perf_counter() - t0

    best = study.best_trial
    params = (dict(best.params) if spec.grid
              else spec.search_space(optuna.trial.FixedTrial(best.params)))
    states = pd.Series([t.state.name for t in study.trials]).value_counts().to_dict()
    log(f'  {name}: best CV MAPE {best.value:.3f} after {len(study.trials)} trials '
        f'({states}) in {elapsed:.0f}s on {device}')
    return dict(params=params, best_value=float(best.value), n_trials=len(study.trials),
                trial_states=states, tune_seconds=round(elapsed, 1), device=device,
                trial_jobs=trial_jobs, threads_per_fit=threads)
