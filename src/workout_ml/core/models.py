"""Model registry: every model exposes the same ``fit(X, y)`` / ``predict(X)``.

``X`` is always the full model frame from :func:`data.make_dataset` (all
candidate features plus 'date'); each model picks its own columns.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone

from .constants import BASE_CHOICES, BASE_COL, DAYS_SINCE_COL
from .preprocess import make_pipeline

warnings.filterwarnings('ignore', message='.*Falling back to prediction using DMatrix.*')
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')


# --------------------------------------------------------------------------- #
# Wrappers
# --------------------------------------------------------------------------- #
class TargetModeRegressor(RegressorMixin, BaseEstimator):
    """Fits on the change from ``base_col`` and always predicts the level.

    ``pct``: ``log(y / base)`` (a % change, matching the MAPE metric); ``delta``: ``y - base``;
    ``level``: ``y`` itself.
    """

    def __init__(self, estimator=None, target_mode='delta', base_col=BASE_COL):
        self.estimator = estimator
        self.target_mode = target_mode
        self.base_col = base_col

    def _base(self, X):
        return X[self.base_col].to_numpy(dtype=float)

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        if self.target_mode == 'pct':
            y = np.log(y / self._base(X))
        elif self.target_mode == 'delta':
            y = y - self._base(X)
        elif self.target_mode != 'level':
            raise ValueError(f'target_mode must be pct, delta or level, got {self.target_mode!r}')
        self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def predict(self, X):
        pred = np.asarray(self.estimator_.predict(X), dtype=float)
        if self.target_mode == 'pct':
            return self._base(X) * np.exp(pred)
        return pred + self._base(X) if self.target_mode == 'delta' else pred


class NaiveLast(RegressorMixin, BaseEstimator):
    """Predicts the previous session's est_1rm for this exercise + set."""

    def __init__(self, base_col=BASE_COL):
        self.base_col = base_col

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return X[self.base_col].to_numpy(dtype=float)


class NaiveDrift(RegressorMixin, BaseEstimator):
    """Previous session + 30-day slope x days since that session."""

    def __init__(self, base_col=BASE_COL, slope_col='est_1rm_slope_30d', days_col=DAYS_SINCE_COL):
        self.base_col = base_col
        self.slope_col = slope_col
        self.days_col = days_col

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        base = X[self.base_col].to_numpy(dtype=float)
        if self.slope_col not in X or self.days_col not in X:
            return base
        slope = X[self.slope_col].fillna(0).to_numpy(dtype=float)
        days = X[self.days_col].fillna(0).to_numpy(dtype=float)
        return base + slope * days


class ColumnBaseline(RegressorMixin, BaseEstimator):
    """Predicts one history column as-is (e.g. an EWMA); rows without it use the last session."""

    def __init__(self, column=BASE_COL, base_col=BASE_COL):
        self.column = column
        self.base_col = base_col

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        base = X[self.base_col].to_numpy(dtype=float)
        p = X[self.column].to_numpy(dtype=float)
        return np.where(np.isnan(p), base, p)


class LinearTwoFeature(RegressorMixin, BaseEstimator):
    """Plain OLS, no scaling: last session + 30-day average (missing average -> last session)."""

    def __init__(self, base_col=BASE_COL, avg_col='est_1rm_avg_30d'):
        self.base_col = base_col
        self.avg_col = avg_col

    def _design(self, X):
        base = X[self.base_col].to_numpy(dtype=float)
        avg = X[self.avg_col].to_numpy(dtype=float)
        return np.column_stack([base, np.where(np.isnan(avg), base, avg)])

    def fit(self, X, y):
        from sklearn.linear_model import LinearRegression
        self.model_ = LinearRegression().fit(self._design(X), np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        return self.model_.predict(self._design(X))


class CatBoostSK(RegressorMixin, BaseEstimator):
    """sklearn-clone-safe CatBoost: ``cat_features`` goes to ``fit`` instead of the constructor.

    (CatBoostRegressor rewrites ``cat_features`` in its constructor, which sklearn's
    ``clone`` rejects.)
    """

    def __init__(self, params=None, cat_features=None):
        self.params = params
        self.cat_features = cat_features

    def fit(self, X, y):
        from catboost import CatBoostRegressor
        self.model_ = CatBoostRegressor(**(self.params or {}))
        self.model_.fit(X, y, cat_features=list(self.cat_features) if self.cat_features else None)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class EnsembleModel(RegressorMixin, BaseEstimator):
    """Equal-weight average of already-chosen member models."""

    def __init__(self, members=None):
        self.members = members  # dict name -> unfitted estimator

    def fit(self, X, y):
        self.members_ = {k: clone(m).fit(X, y) for k, m in self.members.items()}
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.members_.values()], axis=0)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass
class ModelSpec:
    name: str
    kind: str                    # baseline | simple | feature | sarima
    family: str | None           # feature set used: tree | linear
    preprocess: str | None
    build: Callable              # (params, ctx) -> estimator
    search_space: Callable | None = None   # (trial) -> params
    gpu_lib: str | None = None   # library name for device handling
    grid: dict | None = None     # exhaustive search space (small models)


@dataclass
class BuildContext:
    features: list[str]
    categorical: list[str]
    device: str = 'cpu'          # cpu | gpu
    n_threads: int = 1
    seed: int = 42
    cfg: object = None           # Config, used by SARIMA


TARGET_MODES = ['pct', 'delta', 'level']


def _target(trial):
    return dict(target_mode=trial.suggest_categorical('target_mode', TARGET_MODES),
                base=trial.suggest_categorical('base', BASE_CHOICES))


def _split_params(params):
    """Separate the target settings (mode, base column) from the estimator's own parameters."""
    params = dict(params)
    return dict(target_mode=params.pop('target_mode', 'pct'), base_col=params.pop('base', BASE_COL)), params


def _wrap(spec_preprocess, est, target, ctx):
    pipe = make_pipeline(ctx.features, ctx.categorical, spec_preprocess, est)
    return TargetModeRegressor(pipe, **target)


# --- builders ---------------------------------------------------------------
def _build_linear(params, ctx):
    from sklearn.linear_model import LinearRegression
    mode, _ = _split_params(params)
    return _wrap('linear', LinearRegression(), mode, ctx)


def _build_elasticnet(params, ctx):
    from sklearn.linear_model import ElasticNet
    mode, p = _split_params(params)
    est = ElasticNet(alpha=p.get('alpha', 0.01), l1_ratio=p.get('l1_ratio', 0.5),
                     max_iter=20000, random_state=ctx.seed)
    return _wrap('linear', est, mode, ctx)


def _build_rf(params, ctx):
    from sklearn.ensemble import RandomForestRegressor
    mode, p = _split_params(params)
    est = RandomForestRegressor(
        n_estimators=p.get('n_estimators', 300), max_depth=p.get('max_depth'),
        min_samples_leaf=p.get('min_samples_leaf', 5), max_features=p.get('max_features', 0.5),
        n_jobs=ctx.n_threads, random_state=ctx.seed)
    return _wrap('ordinal', est, mode, ctx)


LGBM_DEFAULTS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8)


def _build_lgbm(params, ctx):
    import lightgbm as lgb
    mode, p = _split_params(params)
    kw = {**LGBM_DEFAULTS, **p}
    if ctx.device == 'gpu':
        kw['device_type'] = 'gpu'
    est = lgb.LGBMRegressor(**kw, n_jobs=ctx.n_threads, random_state=ctx.seed, verbose=-1)
    return _wrap('native', est, mode, ctx)


def _build_xgb(params, ctx):
    import xgboost as xgb
    mode, p = _split_params(params)
    kw = dict(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8)
    kw.update(p)
    est = xgb.XGBRegressor(**kw, tree_method='hist', enable_categorical=True,
                           device='cuda' if ctx.device == 'gpu' else 'cpu',
                           n_jobs=ctx.n_threads, random_state=ctx.seed)
    return _wrap('native', est, mode, ctx)


def _build_catboost(params, ctx):
    mode, p = _split_params(params)
    kw = dict(iterations=500, depth=6, learning_rate=0.05)
    kw.update(p)
    cats = [c for c in ctx.features if c in ctx.categorical]
    kw.update(dict(task_type='GPU', devices='0') if ctx.device == 'gpu' else dict(thread_count=ctx.n_threads))
    kw.update(random_seed=ctx.seed, verbose=0, allow_writing_files=False)
    return _wrap('native_string', CatBoostSK(params=kw, cat_features=cats or None), mode, ctx)


# --- search spaces ----------------------------------------------------------
def _space_elasticnet(trial):
    return dict(**_target(trial),
                alpha=trial.suggest_float('alpha', 1e-5, 10, log=True),   # pct targets are ~0.05, so allow tiny
                l1_ratio=trial.suggest_float('l1_ratio', 0.0, 1.0))


def _space_rf(trial):
    return dict(**_target(trial),
                n_estimators=trial.suggest_int('n_estimators', 100, 800, log=True),
                max_depth=trial.suggest_categorical('max_depth', [4, 6, 8, 12, 16, None]),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 40, log=True),
                max_features=trial.suggest_float('max_features', 0.2, 1.0))


def _space_lgbm(trial):
    return dict(**_target(trial),
                n_estimators=trial.suggest_int('n_estimators', 50, 2000, log=True),
                learning_rate=trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                num_leaves=trial.suggest_int('num_leaves', 4, 128, log=True),
                min_child_samples=trial.suggest_int('min_child_samples', 5, 100, log=True),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.3, 1.0),
                subsample=trial.suggest_float('subsample', 0.5, 1.0),
                subsample_freq=1,
                reg_alpha=trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
                reg_lambda=trial.suggest_float('reg_lambda', 1e-8, 10, log=True))


def _space_xgb(trial):
    return dict(**_target(trial),
                n_estimators=trial.suggest_int('n_estimators', 50, 2000, log=True),
                learning_rate=trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                max_depth=trial.suggest_int('max_depth', 2, 10),
                min_child_weight=trial.suggest_float('min_child_weight', 1, 30, log=True),
                subsample=trial.suggest_float('subsample', 0.5, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.3, 1.0),
                reg_alpha=trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
                reg_lambda=trial.suggest_float('reg_lambda', 1e-8, 10, log=True))


def _space_catboost(trial):
    return dict(**_target(trial),
                iterations=trial.suggest_int('iterations', 100, 2000, log=True),
                depth=trial.suggest_int('depth', 3, 8),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1, 30, log=True))


REGISTRY: dict[str, ModelSpec] = {
    'naive_last': ModelSpec('naive_last', 'baseline', None, None, lambda p, c: NaiveLast()),
    'naive_drift': ModelSpec('naive_drift', 'baseline', None, None, lambda p, c: NaiveDrift()),
    'mean_last2': ModelSpec('mean_last2', 'baseline', None, None,
                            lambda p, c: ColumnBaseline('est_1rm_mean_last2')),
    'ewma': ModelSpec('ewma', 'baseline', None, None,
                      lambda p, c: ColumnBaseline(f"est_1rm_ewma_{p.get('alpha_pct', 50)}"),
                      grid={'alpha_pct': [30, 50, 70]}),
    'linear_2feat': ModelSpec('linear_2feat', 'simple', None, None, lambda p, c: LinearTwoFeature()),
    'linear_regression': ModelSpec('linear_regression', 'feature', 'linear', 'linear', _build_linear,
                                   grid={'target_mode': TARGET_MODES, 'base': BASE_CHOICES}),
    'elasticnet': ModelSpec('elasticnet', 'feature', 'linear', 'linear', _build_elasticnet, _space_elasticnet),
    'random_forest': ModelSpec('random_forest', 'feature', 'tree', 'ordinal', _build_rf, _space_rf),
    'lightgbm': ModelSpec('lightgbm', 'feature', 'tree', 'native', _build_lgbm, _space_lgbm, gpu_lib='lightgbm'),
    'xgboost': ModelSpec('xgboost', 'feature', 'tree', 'native', _build_xgb, _space_xgb, gpu_lib='xgboost'),
    'catboost': ModelSpec('catboost', 'feature', 'tree', 'native_string', _build_catboost, _space_catboost,
                          gpu_lib='catboost'),
}


def build_model(name: str, params: dict, ctx: BuildContext):
    return REGISTRY[name].build(params or {}, ctx)
