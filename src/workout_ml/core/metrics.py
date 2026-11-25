"""Metrics and bootstrap confidence intervals (no plotting dependencies)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def metrics(y, p, y_naive=None) -> dict:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    err = y - p
    mae = float(np.mean(np.abs(err)))
    out = dict(mae=mae, rmse=float(np.sqrt(np.mean(err ** 2))),
               mape=float(np.mean(np.abs(err) / np.abs(y)) * 100),
               r2=float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
               bias=float(np.mean(p - y)))
    if y_naive is not None:
        naive = mape(y, y_naive)
        out['skill_vs_naive'] = 1 - out['mape'] / naive if naive > 0 else np.nan
    return out


def mape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(y - p) / np.abs(y)) * 100)


def fold_metrics(y, p) -> dict:
    m = metrics(y, p)
    return {k: m[k] for k in ('mape', 'mae', 'rmse')}


def block_bootstrap(y, preds: dict, dates, reference: str, n_iter: int, seed: int) -> pd.DataFrame:
    """Resample whole workout dates. CIs on each model's MAPE and MAE minus the reference's."""
    y = np.asarray(y, float)
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    codes, uniq = pd.factorize(dates)
    names = list(preds)
    counts = np.bincount(codes, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_iter, len(uniq)))
    denom = counts[idx].sum(axis=1)[:, None]
    ref = names.index(reference)
    abs_err = np.column_stack([np.abs(y - np.asarray(preds[n], float)) for n in names])
    out = pd.DataFrame({'model': names})
    for metric, err in [('mape', abs_err / np.abs(y)[:, None] * 100), ('mae', abs_err)]:
        sums = np.zeros((len(uniq), len(names)))
        np.add.at(sums, codes, err)
        diff = sums[idx].sum(axis=1) / denom                              # (n_iter, n_models)
        diff = diff - diff[:, [ref]]
        out[f'{metric}_diff'] = (err - err[:, [ref]]).mean(axis=0)      # exactly 0 for the reference
        out[f'{metric}_diff_low'], out[f'{metric}_diff_high'] = np.percentile(diff, [2.5, 97.5], axis=0)
        if metric == 'mape':
            out['p_better'] = (diff < 0).mean(axis=0)
    return out


def pick_winners(board: pd.DataFrame) -> dict:
    """Winners by dev-CV MAPE. The holdout never picks; baselines can win overall.

    Returns ``best_overall``, ``best_baseline`` and ``best_non_baseline`` (None if there isn't one).
    """
    b = board.sort_values('cv_mape', kind='stable')

    def first(rows):
        return rows.iloc[0]['model'] if len(rows) else None
    return dict(best_overall=first(b), best_baseline=first(b[b['group'] == 'baseline']),
                best_non_baseline=first(b[b['group'] != 'baseline']))


def head_to_head(y, p_a, p_b, dates, n_iter: int, seed: int) -> dict:
    """Paired difference A minus B (negative = A better), bootstrapped by workout date."""
    r = block_bootstrap(y, {'a': p_a, 'b': p_b}, dates, 'b', n_iter, seed).set_index('model').loc['a']
    return dict(r, mape_a=mape(y, p_a), mape_b=mape(y, p_b))


def per_exercise(y, preds: dict, exercise, metric: str = 'mape') -> pd.DataFrame:
    """Mean APE (%) or absolute error (lb) by exercise."""
    df = pd.DataFrame({'exercise': np.asarray(exercise), 'y': np.asarray(y, float)})
    out = df.groupby('exercise').size().rename('n').to_frame()
    for n, p in preds.items():
        e = (df['y'] - np.asarray(p, float)).abs()
        e = e / df['y'].abs() * 100 if metric == 'mape' else e
        out[n] = e.groupby(df['exercise']).mean()
    return out.sort_values('n', ascending=False)
