"""Predict planned workouts and turn est. 1RMs into working weights."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import InferenceSettings
from ..core import features as F
from ..core.constants import BASE_COL
from ..core.data import to_model_frame
from ..core.reps import e1rm_to_weight
from ..io.registry import ModelBundle
from .schedule import PlannedWorkout
from .templates import future_rows


class UnsafePredictions(RuntimeError):
    """Predictions failed a sanity check; nothing was written."""


def predict_workout(history: pd.DataFrame, workout: PlannedWorkout, template: pd.DataFrame,
                    bundle: ModelBundle) -> pd.DataFrame:
    """Predict one planned workout from the sessions logged before its date.

    Each workout is built on its own, so the blank rows of one planned workout never reach the
    features of another.
    """
    past = history[history['date'] < pd.Timestamp(workout.date)]
    future = future_rows(template, workout.gym_day, workout.date)
    combined = pd.concat([past, future], ignore_index=True)
    combined['set_number'] = combined['set_number'].astype('Int64')
    feats = F.build_features(combined, **bundle.feature_settings)
    feats = feats[feats['set_id'].isin(set(future['set_id']))].sort_values('set_id', ignore_index=True)
    X = to_model_frame(feats, bundle.candidates)
    out = feats[['exercise', 'set_number', 'type']].copy()
    out['last_est_1rm'] = X[BASE_COL].to_numpy()
    out['est_1rm'] = np.nan
    known = X[BASE_COL].notna()
    if known.any():
        out.loc[known, 'est_1rm'] = bundle.model.predict(X[known])
    return out


def check_predictions(pred: pd.DataFrame, settings: InferenceSettings) -> list[str]:
    """Problems worth flagging. Raises for non-finite values, or for big jumps when ``on_jump=abort``."""
    bad = pred[~np.isfinite(pred['est_1rm'].astype(float)) | (pred['est_1rm'] <= 0)]
    if len(bad):
        names = ', '.join(f'{r.exercise} set {r.set_number}' for r in bad.itertuples())
        raise UnsafePredictions(f'non-finite or non-positive predictions: {names}')
    jump = (pred['est_1rm'] / pred['last_est_1rm'] - 1).abs()
    flagged = [f'{r.exercise} set {r.set_number}: {r.last_est_1rm:.0f} -> {r.est_1rm:.0f}'
               for r in pred[jump > settings.max_jump].itertuples()]
    if flagged and settings.on_jump == 'abort':
        raise UnsafePredictions('predictions too far from the last session: ' + '; '.join(flagged))
    return flagged


def rep_weights(est_1rm: float, bundle: ModelBundle, settings: InferenceSettings) -> dict[str, float]:
    """Weight for each rep target, e.g. ``{'5': 250.0, ..., '10': 212.5}``."""
    w = e1rm_to_weight(np.full(len(settings.reps), est_1rm), settings.reps, settings.rpe,
                       table=bundle.ratio_table, round_to=settings.round_to)
    return {str(r): float(v) for r, v in zip(settings.reps, w, strict=True)}
