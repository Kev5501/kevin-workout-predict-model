"""Markdown reports: feature_selection.md and model_selection.md."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def _table(df: pd.DataFrame, floatfmt: str = '.3f', index: bool = False) -> str:
    if df is None or len(df) == 0:
        return '_none_'
    return df.to_markdown(index=index, floatfmt=floatfmt)


def _details(summary: str, body: str) -> str:
    return f'<details><summary>{summary}</summary>\n\n{body}\n\n</details>'


def _data_section(ctx: dict) -> str:
    d = ctx['data_summary']
    lines = [
        '## Data and splits',
        '',
        f"- Feature table: **{d['n_rows_in']}** set rows. Dropped **{d['n_no_history']}** with no previous "
        f"session for that exercise + set (nothing to predict from) and **{d['n_no_target']}** with no target.",
        f"- **Dev:** {d['dev_rows']} rows, {d['dev_workouts']} workouts ({d['dev_start']} to {d['dev_end']}).",
        f"- **Holdout:** {d['hold_rows']} rows, {d['hold_workouts']} workouts ({d['hold_start']} to "
        f"{d['hold_end']}), the last {ctx['cfg']['holdout_frac']:.0%} of workout dates. Never used for "
        'feature selection or tuning.',
        f"- **Walk-forward CV on dev:** {len(ctx['folds_table'])} expanding-window folds, split by workout date.",
        '',
        _table(ctx['folds_table']),
    ]
    return '\n'.join(lines)


def _header(title: str, ctx: dict) -> str:
    quick = ' (quick smoke run: few trials, reduced grids. Not for decisions.)' if ctx['cfg']['quick'] else ''
    return (f'# {title}\n\n'
            f"Run `{ctx['run_id']}`, generated {datetime.now():%Y-%m-%d %H:%M}{quick}. "
            'Target: **est_1rm** of the next session of each exercise + set. Primary metric: **MAPE** (%); '
            'MAE and RMSE are in lb of e1RM.\n')


# --------------------------------------------------------------------------- #
def write_feature_selection(path: Path, ctx: dict, sel, figs: dict) -> Path:
    cfg = ctx['cfg']
    r = sel.ranking
    probes = r[r['is_probe']]
    best_probe_pos = int(probes['position'].min()) if len(probes) else None
    below = r[r['below_probe']]['feature'].tolist()
    curves = sel.curves.copy()

    drop_counts = (sel.dropped['reason'].value_counts().rename_axis('reason').reset_index(name='dropped')
                   if len(sel.dropped) else pd.DataFrame())
    top = r.head(40)[['position', 'feature', 'shap_mean', 'perm_mean', 'mean_rank', 'stability',
                      'is_probe', 'below_probe']]

    parts = [
        _header('Feature selection', ctx),
        '## Outcome',
        '',
        _table(pd.DataFrame(sel.funnel, columns=['stage', 'features']), floatfmt='.0f'),
        '',
        f"- **Tree models** (LightGBM, XGBoost, CatBoost, RandomForest) use the top **{sel.chosen_k['tree']}** "
        f"ranked features plus the two force-kept ones (`est_1rm_most_recent`, `est_1rm_ewma_50`): "
        f"**{len(sel.selected['tree'])}** in total.",
        f"- **Linear models** (OLS, ElasticNet) use the top **{sel.chosen_k['linear']}** "
        f"plus the same two: **{len(sel.selected['linear'])}** in total.",
        f'- The best random probe ranked **#{best_probe_pos}**, so {len(below)} real features scored no better '
        'than random noise.' if best_probe_pos else '',
        '',
        _data_section(ctx),
        '',
        '## Method',
        '',
        '1. **Filter** the inference whitelist (`features.get_feature_columns()`): drop features that are more '
        f"than {cfg['missing_thresh']:.0%} missing, near-constant (one value in more than "
        f"{cfg['constant_thresh']:.0%} of rows), exact duplicates, or in a Spearman "
        f"|ρ| > {cfg['corr_thresh']} cluster. Per cluster, keep the feature most correlated with the "
        '% change from the last session. The `_vs_recent` features exist because the raw history levels '
        'all correlate above this threshold with `est_1rm_most_recent` and would be dropped.',
        '2. **Rank** on each dev fold with a default LightGBM (% target): mean |TreeSHAP| and permutation '
        'importance (MAPE increase), both measured on the fold\'s **validation** block. Rank = average of the two '
        f"ranks, averaged over folds. {cfg['n_probes']} random probe columns show where noise lands. Stability = "
        f"share of folds where the feature is in the top {cfg['stability_top']}.",
        '3. **Size:** CV MAPE for the top-k features with default LightGBM (tree) and default ElasticNet (linear). '
        'Each k is compared with the best k on the same workout dates. The chosen k is the smallest whose mean '
        'paired difference (`mean_diff`) is within one standard error of it (`se_diff`).',
        '',
        '## Stage 1: filter',
        '',
        _table(drop_counts, floatfmt='.0f'),
        '',
        _details(f'All {len(sel.dropped)} dropped features', _table(sel.dropped)),
        '',
        '## Stage 2: ranking',
        '',
        f"![importance]({figs['importance']})",
        '',
        _table(top),
        '',
        _details(f'Full ranking ({len(r)} rows incl. probes)', _table(
            r[['position', 'feature', 'shap_mean', 'perm_mean', 'mean_rank', 'stability', 'is_probe', 'below_probe']])),
        '',
        _details(f'{len(below)} features ranked at or below the best random probe',
                 ', '.join(f'`{f}`' for f in below) or '_none_'),
        '',
        '## Stage 3: how many features',
        '',
        f"![k curve]({figs['k_curve']})",
        '',
        _table(curves),
        '',
        '## Selected features',
        '',
        f"**Tree models ({len(sel.selected['tree'])}):** " + ', '.join(f'`{f}`' for f in sel.selected['tree']),
        '',
        f"**Linear models ({len(sel.selected['linear'])}):** " + ', '.join(f'`{f}`' for f in sel.selected['linear']),
        '',
        '## Caveats',
        '',
        '- Selection and tuning share the dev folds, so dev-CV scores are slightly optimistic. The holdout '
        'in `model_selection.md` is the unbiased comparison.',
        '- Correlation clustering keeps one representative per cluster. A dropped feature can be as useful as '
        'the one kept; the drop table shows which feature replaced it.',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(p for p in parts if p is not None), encoding='utf-8')
    return path


# --------------------------------------------------------------------------- #
def _outcome(ctx: dict) -> list[str]:
    board = ctx['leaderboard'].set_index('model')
    w = ctx['winners']
    best, nb, bl = w['best_overall'], w['best_non_baseline'], w['best_baseline']
    roles: dict = {}
    prod = ctx.get('production')
    for role, name in [('Production', prod), ('Best overall', best), ('Best non-baseline', nb), ('Best baseline', bl),
                       ('Repeat last session', 'naive_last')]:
        if name:
            roles.setdefault(name, []).append(role if name not in roles else role.lower())
    rows = []
    for name, names in roles.items():
        b = board.loc[name]
        rows.append({'role': ' + '.join(names), 'model': f'`{name}`', 'CV MAPE %': b['cv_mape'],
                     'CV MAE': b['cv_mae'], 'CV RMSE': b['cv_rmse'], 'holdout MAPE %': b['holdout_mape'],
                     'holdout MAE': b['holdout_mae'], 'holdout RMSE': b['holdout_rmse']})
    lines = [(f'**Production model: `{prod}`** (default for inference; set by `PRODUCTION_MODEL` in `config.py`, '
              'not by the leaderboard).  ') if prod else '',
             f'**Best overall by CV: `{best}`** ({"a baseline" if best == bl else "not a baseline"}).  ',
             f'**Best non-baseline by CV: `{nb}`**.' if nb else '', '',
             _table(pd.DataFrame(rows), floatfmt='.2f'), '']
    duel = ctx.get('duel')
    if duel is not None and len(duel):
        lines += [f'**Compared with the best baseline `{bl}`** (negative = better than `{bl}`; 95% CI from '
                  'resampling whole workout dates):', '']
        for _, r in duel.iterrows():
            if r['mape_diff_high'] < 0:
                verdict = f"**`{r['model']}` is clearly better**"
            elif r['mape_diff_low'] > 0:
                verdict = f'**`{bl}` is clearly better**'
            else:
                verdict = '**within noise** (the interval includes zero)'
            lines.append(f"- `{r['model']}`, {r['data']}: MAPE {r['mape_diff']:+.3f} points "
                         f"[{r['mape_diff_low']:+.3f}, {r['mape_diff_high']:+.3f}], MAE {r['mae_diff']:+.3f} lb "
                         f"[{r['mae_diff_low']:+.3f}, {r['mae_diff_high']:+.3f}]: {verdict}.")
    lines += ['', 'Winners are picked by dev-CV MAPE only. The holdout is a check, and it has already been viewed '
              'in earlier runs, so treat it as a sanity check more than a fresh test.']
    return lines


def _charts_section(ctx: dict) -> list[str]:
    ch = ctx.get('charts')
    if not ch:
        return ['## Forecast charts', '', '_not drawn_']
    root = 'figures/charts'
    ex = ', '.join(f'[{n}]({root}/by_exercise/{f}.png)' for n, f in ch['by_exercise'])
    return [
        '## Forecast charts', '',
        'est_1rm averaged over the sets in a session. Black = actual (dotted in the holdout). Every model point is '
        'a one-step-ahead prediction on data that model was not trained on: CV folds first, then the holdout. '
        'These are not multi-week forecasts.', '',
        f"Interactive versions (hover, click a model to hide it): `research/artifacts/{ctx['run_id']}/charts/`.", '',
        f'**By exercise:** {ex}', '',
        (f"No out-of-sample predictions, so no chart: {', '.join(ch['skipped'])}." if ch['skipped'] else ''),
    ]


def write_model_selection(path: Path, ctx: dict, figs: dict) -> Path:
    board: pd.DataFrame = ctx['leaderboard']
    w = ctx['winners']
    cols = ['model', 'group', 'cv_mape', 'cv_mape_std', 'cv_mae', 'cv_rmse', 'holdout_mape', 'holdout_mae',
            'holdout_rmse', 'holdout_r2', 'holdout_bias', 'skill_vs_naive', 'target', 'n_features', 'device']
    lb = board[[c for c in cols if c in board]].sort_values('cv_mape').copy()
    lb['target'] = lb['target'].fillna('—')
    lb['n_features'] = [('—' if pd.isna(v) else str(int(v))) for v in lb['n_features']]
    vs_naive = board[['model', 'mape_diff', 'mape_diff_low', 'mape_diff_high', 'mae_diff', 'mae_diff_low',
                      'mae_diff_high', 'p_better']].sort_values('mape_diff')

    params_md = []
    for name, info in ctx['tuned'].items():
        if info.get('params'):
            params_md.append(f'**{name}**\n\n```json\n{json.dumps(info["params"], indent=2, default=str)}\n```\n')

    tuning_rows = pd.DataFrame([
        dict(model=n, trials=i.get('n_trials', 0), states=i.get('trial_states', ''),
             tune_seconds=i.get('tune_seconds', 0), device=i.get('device', 'cpu'),
             parallel_trials=i.get('trial_jobs', ''), threads_per_fit=i.get('threads_per_fit', ''))
        for n, i in ctx['tuned'].items() if i.get('n_trials')])

    sar = ctx.get('sarima_summary') or {}
    sar_md = '_SARIMA not run._'
    if sar:
        sar_md = '\n'.join([
            f"- {sar['n_fitted']} of {sar['n_series']} exercise + set series had at least "
            f"{ctx['cfg']['sarima_min_obs']} sessions and got a SARIMA. Rows from the other series fall back "
            "to `naive_last`.",
            f"- Holdout rows covered by a fitted SARIMA: **{sar['holdout_coverage']:.0%}**.",
            '- Orders chosen by AIC (final model):', '',
            _table(sar['orders'], floatfmt='.0f'),
        ])

    ens = ctx.get('ensemble_members')
    parts = [
        _header('Model selection', ctx),
        '## Outcome', '', *_outcome(ctx), '',
        f"![leaderboard]({figs['leaderboard']})", '',
        '## Leaderboard', '',
        'Sorted by dev-CV MAPE. MAPE and skill are primary; MAE and RMSE are in lb. `skill_vs_naive` is the holdout '
        'MAPE improvement over `naive_last`. `holdout_bias` is the mean over-prediction in lb. `target` is what '
        'the model was trained to predict (% or lb change from the last session or from the EWMA).', '',
        _table(lb), '',
        f"`ensemble_top3` = equal-weight mean of {', '.join(f'`{m}`' for m in ens)}, the best non-baseline models "
        'by dev CV.' if ens else '', '',
        '### Holdout difference from `naive_last`', '',
        'Negative = better. 95% CIs; `p_better` = share of bootstrap samples with lower MAPE.', '',
        _table(vs_naive), '',
        *_charts_section(ctx), '',
        _data_section(ctx), '',
        '## Per-fold CV MAPE (%)', '', _table(ctx['fold_table']), '',
        '## Holdout MAPE (%) by exercise', '', _table(ctx['per_exercise']['mape'], index=True, floatfmt='.2f'), '',
        '## Holdout MAE (lb) by exercise', '', _table(ctx['per_exercise']['mae'], index=True, floatfmt='.2f'), '',
        f"![pred vs actual]({figs['pred_vs_actual']})", '',
        f"![error over time]({figs['error_over_time']})", '',
        '## SARIMA', '', sar_md, '',
        '## GPU vs CPU', '',
        _table(ctx['device_table']) if ctx.get('device_table') is not None else '_not run_', '',
        'In `auto` mode each library uses whichever gives more fits per second. CPU runs '
        f"{ctx['cfg']['trial_jobs']} Optuna trials in parallel; GPU trials run one at a time.", '',
        '## Tuning', '', _table(tuning_rows), '',
        '## Best hyperparameters', '', *params_md, '',
        '## Artifacts', '',
        f"Saved in `research/artifacts/{ctx['run_id']}/models/`: one `<model>.joblib` per model type, refit on "
        'dev + holdout (all data), plus a `<model>.json` with parameters, features and metrics.', '',
        '```python',
        'from modeling.inference import load_model, predict_next, e1rm_to_weight',
        f"model, meta = load_model()                      # production: {ctx.get('production')}",
        f"model, meta = load_model('best')                # best CV MAPE: {w['best_overall']}",
        "model, meta = load_model('ewma')                # any model by name",
        '```', '',
        '## Caveats', '',
        '- All scores are one step ahead: each prediction uses every session before that workout.',
        '- Feature selection, tuning, the EWMA alpha and the ensemble members all used the same dev folds, so CV '
        'scores are slightly optimistic, more so for models with more tuning.',
        '- Many models are compared, so small gaps between them are expected by chance.',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(p for p in parts if p is not None), encoding='utf-8')
    return path
