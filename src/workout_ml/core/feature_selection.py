"""Three-stage feature selection on the dev set: filter -> rank -> size.

Everything here runs only on dev data; the holdout is never seen.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .constants import BASE_COL, FORCE_KEEP
from .models import BuildContext, build_model
from .tuning import cv_predict

PROBE_PREFIX = '_probe_'


@dataclass
class SelectionResult:
    candidates: list[str]
    filtered: list[str]
    dropped: pd.DataFrame
    ranking: pd.DataFrame
    curves: pd.DataFrame
    selected: dict                      # family -> feature list
    chosen_k: dict                      # family -> k
    funnel: list = field(default_factory=list)

    def to_json(self) -> dict:
        return dict(selected=self.selected, chosen_k=self.chosen_k,
                    n_candidates=len(self.candidates), n_filtered=len(self.filtered))


# --------------------------------------------------------------------------- #
# Stage 1: filter
# --------------------------------------------------------------------------- #
def filter_features(X: pd.DataFrame, y_rel: pd.Series, candidates: list[str], categorical: list[str],
                    cfg) -> tuple[list[str], pd.DataFrame]:
    drops = []
    kept = []
    for c in candidates:
        if c in FORCE_KEEP:
            kept.append(c)
            continue
        miss = X[c].isna().mean()
        if miss > cfg.missing_thresh:
            drops.append((c, 'missing', f'{miss:.0%} missing'))
            continue
        top = X[c].value_counts(normalize=True, dropna=False).iloc[0]
        if top > cfg.constant_thresh:
            drops.append((c, 'near_constant', f'top value in {top:.1%} of rows'))
            continue
        kept.append(c)

    # Exact duplicates (numeric and categorical alike)
    seen, deduped = {}, []
    for c in kept:
        h = pd.util.hash_pandas_object(X[c].astype(str), index=False).sum()
        if c not in FORCE_KEEP and h in seen and X[c].astype(str).equals(X[seen[h]].astype(str)):
            drops.append((c, 'duplicate', f'identical to {seen[h]}'))
            continue
        seen.setdefault(h, c)
        deduped.append(c)

    # Correlation clusters: greedy keep by |rho with the % target|, then completeness
    nums = [c for c in deduped if c not in categorical]
    cats = [c for c in deduped if c in categorical]
    corr = X[nums].corr(method='spearman', min_periods=30).abs()
    target_rho = X[nums].corrwith(y_rel, method='spearman').abs().fillna(0)
    completeness = 1 - X[nums].isna().mean()
    order = sorted(nums, key=lambda c: (c not in FORCE_KEEP, -target_rho[c], -completeness[c]))
    kept_nums = []
    for c in order:
        hit = None if c in FORCE_KEEP else next((k for k in kept_nums if corr.loc[c, k] > cfg.corr_thresh), None)
        if hit is not None:
            drops.append((c, 'correlated', f'|rho|={corr.loc[c, hit]:.3f} with {hit}'))
        else:
            kept_nums.append(c)
    final = [c for c in candidates if c in set(kept_nums) | set(cats)]
    dropped = pd.DataFrame(drops, columns=['feature', 'reason', 'detail'])
    return final, dropped


# --------------------------------------------------------------------------- #
# Stage 2: rank
# --------------------------------------------------------------------------- #
def _add_probes(X: pd.DataFrame, n: int, seed: int, source_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    X = X.copy()
    names = []
    for i in range(n):
        name = f'{PROBE_PREFIX}{i}'
        if i % 3 == 0:
            X[name] = rng.normal(size=len(X))
        elif i % 3 == 1:
            X[name] = rng.uniform(size=len(X))
        else:  # shuffled real feature: realistic distribution, no signal
            src = source_cols[rng.integers(len(source_cols))]
            X[name] = rng.permutation(X[src].to_numpy())
        names.append(name)
    return X, names


def rank_features(X: pd.DataFrame, y: pd.Series, features: list[str], categorical: list[str], folds,
                  cfg, n_threads: int) -> pd.DataFrame:
    """Per fold: mean |TreeSHAP| and permutation importance on the validation block."""
    nums = [c for c in features if c not in categorical and c != BASE_COL]
    Xp, probes = _add_probes(X, cfg.n_probes, cfg.seed, nums or features)
    cols = features + probes
    ctx = BuildContext(cols, categorical, device='cpu', n_threads=n_threads, seed=cfg.seed)
    per_fold = []
    for i, (tr, va) in enumerate(folds):
        model = build_model('lightgbm', {'target_mode': 'pct'}, ctx)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(Xp.iloc[tr], y.iloc[tr])
            pipe = model.estimator_
            Xva = Xp.iloc[va]
            Xt = pipe.named_steps['prep'].transform(pipe.named_steps['select'].transform(Xva))
            contrib = pipe.named_steps['model'].predict(Xt, pred_contrib=True)
            shap = np.abs(np.asarray(contrib)[:, :-1]).mean(0)
            perm = permutation_importance(model, Xva[cols], y.iloc[va], scoring='neg_mean_absolute_percentage_error',
                                          n_repeats=cfg.perm_repeats, random_state=cfg.seed, n_jobs=1)
        f = pd.DataFrame({'feature': cols, 'shap': shap, 'perm': perm.importances_mean, 'fold': i})
        f['rank'] = (f['shap'].rank(ascending=False) + f['perm'].rank(ascending=False)) / 2
        f['fold_rank'] = f['rank'].rank(method='first')
        per_fold.append(f)

    allf = pd.concat(per_fold)
    agg = allf.groupby('feature').agg(shap_mean=('shap', 'mean'), perm_mean=('perm', 'mean'),
                                      mean_rank=('fold_rank', 'mean'))
    agg['stability'] = allf.assign(top=allf['fold_rank'] <= cfg.stability_top).groupby('feature')['top'].mean()
    agg['is_probe'] = agg.index.str.startswith(PROBE_PREFIX)
    best_probe = agg.loc[agg['is_probe'], 'mean_rank'].min() if agg['is_probe'].any() else np.inf
    agg['below_probe'] = (~agg['is_probe']) & (agg['mean_rank'] >= best_probe)
    agg = agg.sort_values('mean_rank').reset_index()
    agg.insert(0, 'position', np.arange(1, len(agg) + 1))
    return agg


# --------------------------------------------------------------------------- #
# Stage 3: size
# --------------------------------------------------------------------------- #
def size_search(X: pd.DataFrame, y: pd.Series, ordered: list[str], categorical: list[str], folds,
                cfg, n_threads: int, families=('tree', 'linear')) -> tuple[pd.DataFrame, dict]:
    """CV metrics for the top-k features (added to FORCE_KEEP), and per-workout-date APE per k."""
    rows, by_date = [], {}
    dates = X['date'].to_numpy()
    ks = sorted({len(ordered) if k is None else min(k, len(ordered)) for k in cfg.k_grid})
    for family, model_name, params in [('tree', 'lightgbm', {'target_mode': 'pct'}),
                                       ('linear', 'elasticnet', {'target_mode': 'pct',
                                                                 'alpha': 0.001, 'l1_ratio': 0.5})]:
        if family not in families:
            continue
        ape_by_k = {}
        for k in ks:
            feats = list(dict.fromkeys(FORCE_KEEP + ordered[:k]))
            ctx = BuildContext(feats, categorical, device='cpu', n_threads=n_threads, seed=cfg.seed)
            per_fold, oof = cv_predict(build_model(model_name, params, ctx), X, y, folds)
            ape = pd.Series(np.abs(oof - y.to_numpy()) / y.to_numpy() * 100)
            ape_by_k[k] = ape[~np.isnan(oof)].groupby(dates[~np.isnan(oof)]).mean()
            rows.append(dict(family=family, k=k, n_features=len(feats),
                             **{f'cv_{m}': np.mean([f[m] for f in per_fold]) for m in ('mape', 'mae', 'rmse')}))
        by_date[family] = pd.DataFrame(ape_by_k)
    return pd.DataFrame(rows), by_date


def choose_k(ape: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    """Smallest k not clearly worse than the best k.

    ``ape`` has one row per workout date and one column per k. Each k is compared with the best k
    on the **same dates**, so hard and easy periods cancel out (the standard error of the fold
    means does not do that). A k qualifies if its mean paired difference is within one standard
    error of that difference.
    """
    best = ape.mean().idxmin()
    diff = ape.sub(ape[best], axis=0)
    table = pd.DataFrame({'mean_diff': diff.mean(), 'se_diff': diff.std(ddof=1) / np.sqrt(len(diff))})
    table['ok'] = table['mean_diff'] <= table['se_diff'].fillna(0)
    return int(min(table.index[table['ok']])), table.rename_axis('k').reset_index()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_selection(X: pd.DataFrame, y: pd.Series, candidates: list[str], categorical: list[str], folds,
                  cfg, log=print, families=('tree', 'linear')) -> SelectionResult:
    y_rel = y / X[BASE_COL] - 1
    log(f'  stage 1: filtering {len(candidates)} candidates')
    filtered, dropped = filter_features(X, y_rel, candidates, categorical, cfg)
    log(f'    kept {len(filtered)}, dropped {len(dropped)} '
        f'({dropped["reason"].value_counts().to_dict() if len(dropped) else {}})')

    log('  stage 2: ranking (LightGBM TreeSHAP + permutation importance per fold)')
    ranking = rank_features(X, y, filtered, categorical, folds, cfg, cfg.n_threads)
    ordered = [f for f in ranking.loc[~ranking['is_probe'], 'feature'] if f not in FORCE_KEEP]

    log('  stage 3: CV MAPE vs number of features')
    curves, by_date = size_search(X, y, ordered, categorical, folds, cfg, cfg.n_threads, families)
    chosen_k, selected, paired_tables = {}, {}, []
    for family in families:
        k, paired = choose_k(by_date[family])
        chosen_k[family] = k
        selected[family] = list(dict.fromkeys(FORCE_KEEP + ordered[:k]))
        paired_tables.append(paired.assign(family=family))
        log(f'    {family}: k={k} -> {len(selected[family])} features')
    curves = curves.merge(pd.concat(paired_tables), on=['family', 'k'])

    funnel = [('candidates (inference whitelist)', len(candidates)),
              ('after filter', len(filtered)),
              ('ranked above best random probe',
               int((~ranking['is_probe'] & ~ranking['below_probe']).sum())),
              *[(f'selected: {f} models', len(selected[f])) for f in families]]
    return SelectionResult(candidates, filtered, dropped, ranking, curves, selected, chosen_k, funnel)
