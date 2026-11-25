"""est_1rm <-> weight for given reps and RPE, from the athlete's own sets."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_ratio_table(feats: pd.DataFrame) -> pd.DataFrame:
    """weight / est_1rm as a function of effective reps (reps + (10 - rpe)), from your own sets.

    Derived from the data, so it matches the sheet's rpe_norm_weight formula.
    """
    df = feats[['weight', 'reps', 'rpe', 'rpe_norm_weight']].apply(pd.to_numeric, errors='coerce').dropna()
    df = df[(df['rpe_norm_weight'] > 0) & (df['weight'] > 0)]
    eff = (df['reps'] + (10 - df['rpe'])).round(1)
    ratio = (df['weight'] / df['rpe_norm_weight']).groupby(eff).median()
    return ratio.rename('ratio').rename_axis('effective_reps').reset_index()


def e1rm_to_weight(e1rm, reps, rpe=10.0, *, table: pd.DataFrame, round_to: float | None = None):
    """Weight that hits ``reps`` at ``rpe`` for a given est_1rm."""
    table = table.sort_values('effective_reps')
    eff = np.asarray(reps, dtype=float) + (10 - np.asarray(rpe, dtype=float))
    ratio = np.interp(eff, table['effective_reps'], table['ratio'])
    w = np.asarray(e1rm, dtype=float) * ratio
    if round_to:
        w = np.round(w / round_to) * round_to
    return pd.Series(w, index=e1rm.index) if isinstance(e1rm, pd.Series) else w
