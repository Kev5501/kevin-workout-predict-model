"""Load saved models and predict the next session.

    from modeling.inference import load_model, make_future_rows, predict_next, e1rm_to_weight

    future = make_future_rows('2026-11-24', 'Push', [
        {'exercise': 'Smith Incline Bench', 'sets': 2, 'type': 'Hypertrophy'},
        {'exercise': 'Machine Lateral Raises', 'sets': 3, 'type': 'Hypertrophy'},
    ])
    preds = predict_next(future)          # production model (LightGBM); history = data/raw_sets.parquet
    preds['weight_for_8_reps'] = e1rm_to_weight(preds['predicted_est_1rm'], reps=8, rpe=10)

Pass ``history=load_raw_data()`` to use the live sheet instead of the last snapshot.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from workout_ml.core.reps import build_ratio_table  # noqa: F401
from workout_ml.core.reps import e1rm_to_weight as _e1rm_to_weight

from . import features as F
from .config import ARTIFACTS_DIR, BASE_COL, FEATURE_SETTINGS, PRODUCTION_MODEL, RAW_PATH
from .data import to_model_frame
from .sheets import cast_raw, clean_raw


# --------------------------------------------------------------------------- #
# Runs and models
# --------------------------------------------------------------------------- #
def latest_run_dir(include_quick: bool = False) -> Path:
    runs = sorted(p for p in ARTIFACTS_DIR.glob('*') if (p / 'summary.json').exists())
    full = [p for p in runs if not p.name.endswith('_quick')]
    pick = full or (runs if include_quick or not full else [])
    if not pick:
        raise FileNotFoundError(f'No finished runs in {ARTIFACTS_DIR}. Run python -m modeling.run_pipeline first.')
    return pick[-1]


def _run_dir(run_dir) -> Path:
    return Path(run_dir) if run_dir else latest_run_dir(include_quick=True)


def list_models(run_dir=None) -> pd.DataFrame:
    rd = _run_dir(run_dir)
    rows = []
    for meta_path in sorted((rd / 'models').glob('*.json')):
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        rows.append({'model': meta['model'], **{k: meta['metrics'].get(k) for k in
                                                ('cv_mape', 'holdout_mape', 'cv_mae', 'holdout_mae')}})
    df = pd.DataFrame(rows)     # runs from before the switch to MAPE have no cv_mape
    return df.sort_values('cv_mape' if df['cv_mape'].notna().any() else 'cv_mae')


def load_model(name: str = PRODUCTION_MODEL, run_dir=None):
    """Return ``(model, metadata)``. Defaults to the production model (``config.PRODUCTION_MODEL``).

    ``'best'`` = lowest dev-CV MAPE overall (can be a baseline); ``'best_non_baseline'`` = the best
    model that isn't a baseline; or any model name.
    """
    rd = _run_dir(run_dir)
    if name in ('best', 'best_non_baseline'):
        summary = json.loads((rd / 'summary.json').read_text(encoding='utf-8'))
        name = summary['best_model' if name == 'best' else 'best_non_baseline']
    path = rd / 'models' / f'{name}.joblib'
    if not path.exists():
        raise FileNotFoundError(f'{name} was not trained in run {rd.name}')
    model = joblib.load(path)
    meta = json.loads((rd / 'models' / f'{name}.json').read_text(encoding='utf-8'))
    return model, meta


# --------------------------------------------------------------------------- #
# est_1rm -> weight
# --------------------------------------------------------------------------- #
def e1rm_to_weight(e1rm, reps, rpe=10.0, run_dir=None, table: pd.DataFrame | None = None,
                   round_to: float | None = None):
    """Weight that hits ``reps`` at ``rpe`` (ratio table from the run unless ``table`` is given)."""
    if table is None:
        table = pd.read_csv(_run_dir(run_dir) / 'ratio_table.csv')
    return _e1rm_to_weight(e1rm, reps, rpe, table=table, round_to=round_to)


# --------------------------------------------------------------------------- #
# Next-session prediction
# --------------------------------------------------------------------------- #
def make_future_rows(date, gym_day: str, plan: list[dict]) -> pd.DataFrame:
    """Placeholder rows for a planned workout.

    ``plan`` = ``[{'exercise': ..., 'sets': 2, 'type': 'Hypertrophy'}, ...]`` in workout order.
    """
    rows, i = [], 1
    for item in plan:
        for s in range(1, int(item.get('sets', 1)) + 1):
            rows.append(dict(date=pd.Timestamp(date), set_id=f'FUT-{i:05d}', gym_day=gym_day,
                             exercise=item['exercise'], type=item.get('type', 'Hypertrophy'), set_number=s))
            i += 1
    return pd.DataFrame(rows)


def predict_next(future: pd.DataFrame, history: pd.DataFrame | None = None,
                 model_name: str = PRODUCTION_MODEL, run_dir=None, build_kwargs: dict | None = None) -> pd.DataFrame:
    """Build features for ``future`` rows from ``history`` and predict their est_1rm.

    ``history`` defaults to the raw snapshot saved by ``python -m modeling.build_features``.
    Feature-building settings come from the run's ``features_config.json`` (the settings the
    model was trained with), falling back to ``config.FEATURE_SETTINGS``.
    """
    rd = _run_dir(run_dir)
    model, meta = load_model(model_name, rd)
    candidates = json.loads((rd / 'summary.json').read_text(encoding='utf-8'))['candidates']
    if build_kwargs is None:
        cfg_path = rd / 'features_config.json'
        build_kwargs = (json.loads(cfg_path.read_text(encoding='utf-8')) if cfg_path.exists()
                        else FEATURE_SETTINGS)
    if history is None:
        if not RAW_PATH.exists():
            raise FileNotFoundError(f'{RAW_PATH} not found: run python -m modeling.build_features, '
                                    'or pass history=load_raw_data()')
        history = cast_raw(pd.read_parquet(RAW_PATH))

    hist = history.copy()
    hist['date'] = pd.to_datetime(hist['date'])
    hist, _ = clean_raw(hist)     # same bad-data guard as build_features
    fut = future.copy()
    fut['date'] = pd.to_datetime(fut['date'])
    if fut['date'].min() <= hist['date'].max():
        raise ValueError('Future rows must be dated after the last history row')
    combined = pd.concat([hist, fut], ignore_index=True)
    feats = F.build_features(combined, **build_kwargs)
    feats = feats[feats['set_id'].isin(set(fut['set_id']))].reset_index(drop=True)

    X = to_model_frame(feats, candidates)
    has_history = X[BASE_COL].notna().to_numpy()
    pred = np.full(len(X), np.nan)
    if has_history.any():
        pred[has_history] = model.predict(X[has_history])
    out = feats[['date', 'gym_day', 'exercise', 'set_number']].copy()
    out[BASE_COL] = X[BASE_COL].to_numpy()
    out['predicted_est_1rm'] = pred
    out['model'] = meta['model']
    return out
