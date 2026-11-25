"""Train a candidate, backtest it against the champion's recipe, refit on all data."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..config import TrainingSettings
from ..core import features as F
from ..core.constants import BASE_COL, EWMA_COL, FEATURE_SETTINGS
from ..core.data import Dataset, make_dataset
from ..core.feature_selection import run_selection
from ..core.metrics import mape
from ..core.models import BuildContext, build_model
from ..core.reps import build_ratio_table
from ..core.search import SearchConfig
from ..core.splits import WalkForwardSplit
from ..core.tuning import tune_model
from ..io.registry import Recipe
from .gate import GateResult, judge

log = logging.getLogger(__name__)


@dataclass
class TrainingOutcome:
    model: Any
    recipe: Recipe
    gate: GateResult
    candidates: list[str]
    feature_settings: dict
    ratio_table: pd.DataFrame
    data_range: list[str]
    rows: int
    selection: dict | None            # feature-selection summary, only in full mode
    tuning: dict | None


def split_eval(dates: pd.Series, n_workouts: int) -> tuple[np.ndarray, np.ndarray]:
    """Positions (before, eval): eval = the last ``n_workouts`` workout dates."""
    uniq = np.sort(pd.to_datetime(dates).unique())
    if len(uniq) <= n_workouts * 2:
        raise ValueError(f'not enough workouts ({len(uniq)}) for a {n_workouts}-workout backtest')
    is_eval = (pd.to_datetime(dates) >= uniq[-n_workouts]).to_numpy()
    return np.flatnonzero(~is_eval), np.flatnonzero(is_eval)


def search_recipe(data: Dataset, mode: str, champion: Recipe | None, settings: TrainingSettings,
                  search: SearchConfig) -> tuple[Recipe, dict | None, dict | None]:
    """Choose features and parameters using only ``data``."""
    if mode == 'refit':
        if champion is None:
            raise ValueError('mode=refit needs a champion')
        return champion, None, None
    X, y = data.X.reset_index(drop=True), data.y.reset_index(drop=True)
    folds = WalkForwardSplit(search.n_splits, search.min_train_frac).split(X['date'])
    selection = None
    known = champion is not None and set(champion.features) <= set(data.features)
    if mode == 'full' or not known:
        if mode != 'full':
            log.warning('champion features are not all available; selecting features again')
        sel = run_selection(X, y, data.features, data.categorical, folds, search, log=log.info, families=('tree',))
        features = sel.selected['tree']
        selection = {'funnel': [list(f) for f in sel.funnel], 'features': features}
    else:
        features = list(champion.features)
    t = settings.tuning
    res = tune_model(settings.model_name, X, y, folds, features, data.categorical, 'cpu', search, None,
                     t.n_trials, log=log.info, timeout=t.timeout_minutes * 60)
    tuning = {k: res[k] for k in ('best_value', 'n_trials', 'trial_states', 'tune_seconds')}
    return Recipe(settings.model_name, features, res['params']), selection, tuning


def fit_recipe(recipe: Recipe, data: Dataset, search: SearchConfig):
    ctx = BuildContext(recipe.features, data.categorical, device='cpu', n_threads=search.n_threads, seed=search.seed)
    return build_model(recipe.model_name, recipe.params, ctx).fit(data.X, data.y)


def train(history: pd.DataFrame, mode: str, champion: Recipe | None, settings: TrainingSettings,
          feature_settings: dict | None = None, search: SearchConfig | None = None) -> TrainingOutcome:
    """``history`` is the cleaned set log."""
    feature_settings = feature_settings or FEATURE_SETTINGS
    search = search or SearchConfig(seed=settings.seed, trial_jobs=settings.tuning.trial_jobs,
                                    n_threads=settings.tuning.n_threads)
    feats = F.build_features(history, **feature_settings)
    data = make_dataset(feats)
    before, ev = split_eval(data.dates, settings.promotion.eval_workouts)
    past, recent = data.subset(before), data.subset(ev)

    recipe, selection, tuning = search_recipe(past, mode, champion, settings, search)

    # Both recipes are refit on the same earlier data, so neither has seen the evaluation workouts.
    p_new = fit_recipe(recipe, past, search).predict(recent.X)
    usable = champion is not None and set(champion.features) <= set(data.features)
    p_old = fit_recipe(champion, past, search).predict(recent.X) if usable else None
    last, ewma = recent.X[BASE_COL].to_numpy(), recent.X[EWMA_COL].to_numpy()
    m_new, m_naive = mape(recent.y, p_new), mape(recent.y, last)
    m_old = None if p_old is None else mape(recent.y, p_old)
    passed, reason = judge(m_new, m_old, m_naive, p_new, last, settings.promotion)
    gate = GateResult(passed, reason, m_new, m_old, m_naive, mape(recent.y, ewma), len(recent.y),
                      int(recent.dates.nunique()), str(recent.dates.min().date()), str(recent.dates.max().date()))
    log.info('gate result', extra={'gate': gate.to_dict()})

    final = fit_recipe(recipe, data, search)
    return TrainingOutcome(final, recipe, gate, data.features, dict(feature_settings), build_ratio_table(feats),
                           [str(data.dates.min().date()), str(data.dates.max().date())], len(data.y),
                           selection, tuning)
