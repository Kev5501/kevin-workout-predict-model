"""End-to-end pipeline: select -> tune -> evaluate (charts included).

From the ``research`` folder, using the project venv::

    ..\\.venv\\Scripts\\python -m modeling.run_pipeline --quick
    ..\\.venv\\Scripts\\python -m modeling.run_pipeline --device auto
    ..\\.venv\\Scripts\\python -m modeling.run_pipeline --run-id 20261122_101500 --stages evaluate
    ..\\.venv\\Scripts\\python -m modeling.run_pipeline --run-id 20261122_101500 --stages charts   # redraw only
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from . import charts as C
from . import evaluate as E
from .config import ALL_MODELS, ARTIFACTS_DIR, DATA_PATH, PRODUCTION_MODEL, REPORTS_DIR, Config
from .data import load_features, make_dataset
from .device import resolve_devices
from .feature_selection import run_selection
from .inference import build_ratio_table
from .models import REGISTRY, BuildContext, EnsembleModel, build_model
from .reporting import write_feature_selection, write_model_selection
from .splits import WalkForwardSplit, holdout_split
from .tuning import cv_predict, tune_model

STAGES = ['select', 'tune', 'evaluate']


class Logger:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, msg: str = ''):
        print(msg, flush=True)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(f'{datetime.now():%H:%M:%S} {msg}\n')


def _versions() -> dict:
    out = {}
    for pkg in ['pandas', 'numpy', 'scikit-learn', 'lightgbm', 'xgboost', 'catboost', 'statsmodels', 'optuna']:
        try:
            out[pkg] = version(pkg)
        except Exception:  # noqa: BLE001
            out[pkg] = None
    return out


def _dump_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o)),
                    encoding='utf-8')


def _cv_summary(per_fold: list[dict]) -> dict:
    df = pd.DataFrame(per_fold)
    return dict(cv_folds=per_fold, cv_mape=float(df['mape'].mean()), cv_mape_std=float(df['mape'].std(ddof=1)),
                cv_mae=float(df['mae'].mean()), cv_rmse=float(df['rmse'].mean()))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--features', default=str(DATA_PATH), help='feature table parquet')
    p.add_argument('--stages', default='all', help='comma list of select,tune,evaluate,charts or "all" (evaluate draws charts too)')
    p.add_argument('--device', choices=['auto', 'cpu', 'gpu'], default='auto')
    p.add_argument('--trials', type=int, help='override trial count for every tuned model')
    p.add_argument('--trial-jobs', type=int, help='parallel Optuna trials on CPU')
    p.add_argument('--models', help=f'comma list, default all: {",".join(ALL_MODELS)}')
    p.add_argument('--run-id', help='reuse/resume an existing run folder')
    p.add_argument('--quick', action='store_true', help='few trials and small grids, for smoke testing')
    p.add_argument('--no-charts', action='store_true', help='skip the forecast charts')
    p.add_argument('--no-report-copy', action='store_true', help="don't overwrite research/reports/")
    return p.parse_args(argv)


def build_config(args) -> Config:
    cfg = Config()
    if args.quick:
        cfg.apply_quick()
    cfg.device = args.device
    if args.trials is not None:
        cfg.trials = {k: args.trials for k in cfg.trials}
    if args.trial_jobs:
        cfg.trial_jobs = args.trial_jobs
    if args.models:
        cfg.models = [m.strip() for m in args.models.split(',')]
        unknown = set(cfg.models) - set(ALL_MODELS)
        if unknown:
            raise SystemExit(f'Unknown models: {sorted(unknown)}')
    if 'naive_last' not in cfg.models:
        cfg.models.insert(0, 'naive_last')   # reference for skill scores and bootstrap
    return cfg


def main(argv=None) -> Path:
    args = parse_args(argv)
    cfg = build_config(args)
    stages = STAGES if args.stages == 'all' else [s.strip() for s in args.stages.split(',')]
    run_id = args.run_id or datetime.now().strftime('%Y%m%d_%H%M%S') + ('_quick' if cfg.quick else '')
    run_dir = ARTIFACTS_DIR / run_id
    for sub in ['models', 'predictions', 'plots']:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    log = Logger(run_dir / 'run.log')
    log(f'Run {run_id} | stages {stages} | device {cfg.device} | quick={cfg.quick}')
    warnings.filterwarnings('ignore')

    # ---------------------------------------------------------------- data
    feats = load_features(args.features)
    data = make_dataset(feats)
    dev_idx, hold_idx = holdout_split(data.dates, cfg.holdout_frac)
    dev, hold = data.subset(dev_idx), data.subset(hold_idx)
    assert dev.dates.max() < hold.dates.min(), 'holdout must be strictly after dev'
    folds = WalkForwardSplit(cfg.n_splits, cfg.min_train_frac).split(dev.dates)
    folds_table = WalkForwardSplit.describe(dev.dates, folds)
    d = lambda s: pd.to_datetime(s)  # noqa: E731
    ctx = dict(
        run_id=run_id, cfg=cfg.to_dict(), folds_table=folds_table,
        data_summary=dict(
            n_rows_in=data.n_rows_in, n_no_history=data.n_dropped_no_history, n_no_target=data.n_dropped_no_target,
            dev_rows=len(dev.y), dev_workouts=dev.dates.nunique(),
            dev_start=d(dev.dates.min()).date(), dev_end=d(dev.dates.max()).date(),
            hold_rows=len(hold.y), hold_workouts=hold.dates.nunique(),
            hold_start=d(hold.dates.min()).date(), hold_end=d(hold.dates.max()).date()))
    log(f'{len(data.y)} rows ({len(data.features)} candidate features) | dev {len(dev.y)} rows | '
        f'holdout {len(hold.y)} rows from {ctx["data_summary"]["hold_start"]}')
    _dump_json(dict(cfg=cfg.to_dict(), args=vars(args), versions=_versions()), run_dir / 'run_config.json')
    fc = Path(args.features).with_name('features_config.json')
    if fc.exists():
        shutil.copy(fc, run_dir / 'features_config.json')

    report_dirs = [run_dir / 'reports'] + ([] if args.no_report_copy else [REPORTS_DIR])

    def draw_charts():
        """Forecast charts; a failure is logged but never stops the run."""
        if args.no_charts:
            return None
        try:
            log('  drawing charts')
            result = C.draw_run_charts(run_dir, log=log)
            for rd in report_dirs:
                C.copy_pngs(run_dir / 'charts', rd / 'figures' / 'charts')
            return result
        except Exception as e:  # noqa: BLE001
            log(f'  WARNING: charts failed: {type(e).__name__}: {e}')
            return None

    if stages == ['charts']:
        draw_charts()
        log(f'\nDone. Run folder: {run_dir}')
        return run_dir

    # ---------------------------------------------------------------- select
    sel_path = run_dir / 'selected_features.json'
    if 'select' in stages:
        log('\n[1/3] Feature selection')
        sel = run_selection(dev.X, dev.y, data.features, data.categorical, folds, cfg, log=log)
        _dump_json({**sel.to_json(), 'candidates': data.features, 'categorical': data.categorical}, sel_path)
        sel.ranking.to_csv(run_dir / 'feature_ranking.csv', index=False)
        sel.dropped.to_csv(run_dir / 'feature_drops.csv', index=False)
        sel.curves.to_csv(run_dir / 'feature_k_curves.csv', index=False)
        for rd in report_dirs:
            figs = {'importance': 'figures/fs_importance.png', 'k_curve': 'figures/fs_k_curve.png'}
            E.plot_importance(sel.ranking, rd / figs['importance'])
            E.plot_k_curves(sel.curves, sel.chosen_k, rd / figs['k_curve'])
            write_feature_selection(rd / 'feature_selection.md', ctx, sel, figs)
        log(f'  wrote feature_selection.md')
    if not sel_path.exists():
        raise SystemExit(f'{sel_path} missing: run the select stage first')
    selection = json.loads(sel_path.read_text(encoding='utf-8'))
    selected = selection['selected']

    def features_for(name):
        fam = REGISTRY[name].family
        return selected[fam] if fam else data.features

    # ---------------------------------------------------------------- tune
    tuned_path = run_dir / 'tuned.json'
    if 'tune' in stages:
        log('\n[2/3] Tuning (walk-forward CV on dev)')
        gpu_models = [m for m in cfg.models if REGISTRY[m].gpu_lib]
        devices, device_table = ({}, None)
        if gpu_models:
            devices, device_table = resolve_devices(cfg, dev.X, dev.y, selected['tree'], data.categorical)
            device_table.to_csv(run_dir / 'device_benchmark.csv', index=False)
            log('  devices: ' + ', '.join(f'{k}={v}' for k, v in devices.items()))
        tuned, oof = {}, pd.DataFrame({'date': dev.dates.to_numpy(), 'exercise': dev.X['exercise'].to_numpy(),
                                       'set_number': dev.X['set_number'].to_numpy(), 'y': dev.y.to_numpy()})
        oof['fold'] = np.nan
        for i, (_, va) in enumerate(folds, 1):
            oof.loc[va, 'fold'] = i
        for name in cfg.models:
            spec = REGISTRY[name]
            device = devices.get(spec.gpu_lib, 'cpu')
            feats_ = features_for(name)
            info = dict(kind=spec.kind, group='trained' if spec.kind == 'feature' else spec.kind,
                        family=spec.family, device=device,
                        n_features=len(feats_) if spec.kind == 'feature' else None)
            if spec.search_space or spec.grid:
                res = tune_model(name, dev.X, dev.y, folds, feats_, data.categorical, device, cfg,
                                 run_dir / 'optuna_journal.log', cfg.trials.get(name, 20), log=log)
                info.update(res)
            else:
                info['params'] = {}
            ctx_b = BuildContext(feats_, data.categorical, device=device, n_threads=cfg.n_threads,
                                 seed=cfg.seed, cfg=cfg)
            per_fold, pred = cv_predict(build_model(name, info['params'], ctx_b), dev.X, dev.y, folds)
            mode, base = info['params'].get('target_mode'), info['params'].get('base')
            info.update(_cv_summary(per_fold), features=feats_ if spec.kind == 'feature' else None,
                        target=None if mode is None else mode if mode == 'level'
                        else f"{mode} vs {'ewma' if base and 'ewma' in base else 'last'}")
            oof[name] = pred
            tuned[name] = info
            log(f'  {name:18s} CV MAPE {info["cv_mape"]:.3f}%  MAE {info["cv_mae"]:.3f}')

        # ensemble of the best non-baseline models by dev CV
        cands = sorted((n for n, i in tuned.items() if i['kind'] != 'baseline'), key=lambda n: tuned[n]['cv_mape'])
        members = cands[:cfg.ensemble_size]
        if len(members) >= 2:
            pred = oof[members].mean(axis=1).to_numpy()
            per_fold = [E.fold_metrics(dev.y.to_numpy()[va], pred[va]) for _, va in folds]
            oof['ensemble_top3'] = pred
            tuned['ensemble_top3'] = dict(kind='ensemble', group='ensemble', family=None, members=members,
                                          params={}, device='cpu', **_cv_summary(per_fold))
            log(f'  {"ensemble_top3":18s} CV MAPE {tuned["ensemble_top3"]["cv_mape"]:.3f}% (members {members})')
        oof.to_parquet(run_dir / 'predictions' / 'cv_oof.parquet', index=False)
        _dump_json(dict(tuned=tuned, device_table=None if device_table is None
                        else device_table.to_dict(orient='records')), tuned_path)
    if not tuned_path.exists():
        raise SystemExit(f'{tuned_path} missing: run the tune stage first')
    tuned_blob = json.loads(tuned_path.read_text(encoding='utf-8'))
    tuned = tuned_blob['tuned']

    # ---------------------------------------------------------------- evaluate
    if 'evaluate' in stages:
        log('\n[3/3] Holdout evaluation and final refit')
        hold_preds, finals, sarima_summary = {}, {}, None
        for name, info in tuned.items():
            if info['kind'] == 'ensemble':
                continue
            feats_ = features_for(name)
            ctx_b = BuildContext(feats_, data.categorical, device=info.get('device', 'cpu'),
                                 n_threads=cfg.n_threads, seed=cfg.seed, cfg=cfg)
            model = build_model(name, info['params'], ctx_b)
            m_dev = clone(model).fit(dev.X, dev.y)
            hold_preds[name] = m_dev.predict(hold.X)
            final = clone(model).fit(data.X, data.y)
            finals[name] = final
            if name == 'sarima':
                sarima_summary = dict(n_series=final.n_series_, n_fitted=len(final.series_),
                                      holdout_coverage=m_dev.coverage(hold.X),
                                      orders=final.orders_summary().rename_axis('order')
                                      .reset_index(name='series'))
            log(f'  {name:18s} holdout MAPE {E.mape(hold.y, hold_preds[name]):.3f}%')

        if 'ensemble_top3' in tuned:
            members = tuned['ensemble_top3']['members']
            hold_preds['ensemble_top3'] = np.mean([hold_preds[m] for m in members], axis=0)
            ens = EnsembleModel(members={m: build_model(m, tuned[m]['params'], BuildContext(
                features_for(m), data.categorical, device=tuned[m].get('device', 'cpu'),
                n_threads=cfg.n_threads, seed=cfg.seed, cfg=cfg)) for m in members})
            ens.members_ = {m: finals[m] for m in members}
            finals['ensemble_top3'] = ens

        # metrics
        y_h = hold.y.to_numpy()
        naive = hold_preds['naive_last']
        boot = E.block_bootstrap(y_h, hold_preds, hold.dates, 'naive_last', cfg.bootstrap_iters, cfg.seed)
        rows = []
        for name, p in hold_preds.items():
            info = tuned[name]
            m = E.metrics(y_h, p, naive)
            rows.append(dict(model=name, group=info['group'],
                             **{k: info[k] for k in ('cv_mape', 'cv_mape_std', 'cv_mae', 'cv_rmse')},
                             **{f'holdout_{k}': v for k, v in m.items() if k != 'skill_vs_naive'},
                             skill_vs_naive=m['skill_vs_naive'], target=info.get('target'),
                             n_features=info.get('n_features'), device=info.get('device')))
        board = pd.DataFrame(rows).merge(boot, on='model').sort_values('cv_mape').reset_index(drop=True)
        win = E.pick_winners(board)
        best, best_nb, best_bl = win['best_overall'], win['best_non_baseline'], win['best_baseline']

        prod = PRODUCTION_MODEL if PRODUCTION_MODEL in hold_preds else None

        # production model and best non-baseline vs best baseline, on dev CV and on the holdout
        oof = pd.read_parquet(run_dir / 'predictions' / 'cv_oof.parquet')
        oof = oof[oof['fold'].notna()]
        args_ = (cfg.bootstrap_iters, cfg.seed)
        duel = pd.DataFrame([
            dict(model=m, data=label, **E.head_to_head(y_, a, b_, dates_, *args_))
            for m in dict.fromkeys([prod, best_nb]) if m and m != best_bl
            for label, y_, a, b_, dates_ in [
                ('dev CV', oof['y'], oof[m], oof[best_bl], oof['date']),
                ('holdout', y_h, hold_preds[m], hold_preds[best_bl], hold.dates)]])
        fold_table = pd.DataFrame({n: [f['mape'] for f in i['cv_folds']] for n, i in tuned.items()},
                                  index=pd.Index(range(1, len(folds) + 1), name='fold')).T
        fold_table['mean'] = fold_table.mean(axis=1)
        fold_table = fold_table.sort_values('mean').reset_index(names='model')
        show = [m for m in dict.fromkeys([prod, best_bl, best_nb, 'naive_last'] + board['model'].tolist()) if m]
        per_ex = {m: E.per_exercise(y_h, {k: hold_preds[k] for k in show}, hold.X['exercise'], m)
                  for m in ('mape', 'mae')}

        # artifacts
        pd.DataFrame({'date': hold.dates.to_numpy(), 'exercise': hold.X['exercise'].to_numpy(),
                      'set_number': hold.X['set_number'].to_numpy(), 'y': y_h, **hold_preds}
                     ).to_parquet(run_dir / 'predictions' / 'holdout.parquet', index=False)
        board.to_csv(run_dir / 'leaderboard.csv', index=False)
        ratio = build_ratio_table(feats)
        ratio.to_csv(run_dir / 'ratio_table.csv', index=False)
        versions = _versions()
        for name, final in finals.items():
            info = tuned[name]
            joblib.dump(final, run_dir / 'models' / f'{name}.joblib', compress=3)
            row = board.set_index('model').loc[name].to_dict()
            _dump_json(dict(model=name, run_id=run_id, kind=info['kind'], params=info.get('params'),
                            target=info.get('target'), features=info.get('features'), group=info['group'],
                            members=info.get('members'), cv_folds=info['cv_folds'], metrics=row,
                            device=info.get('device'), dev_range=[str(ctx['data_summary']['dev_start']),
                                                                  str(ctx['data_summary']['dev_end'])],
                            final_train_range=[str(d(data.dates.min()).date()), str(d(data.dates.max()).date())],
                            final_train_rows=len(data.y), versions=versions, created=datetime.now().isoformat()),
                       run_dir / 'models' / f'{name}.json')
        _dump_json(dict(production_model=prod, best_model=best, best_non_baseline=best_nb, best_baseline=best_bl, metric='cv_mape',
                        run_id=run_id, candidates=data.features), run_dir / 'summary.json')

        ctx.update(leaderboard=board, winners=win, production=prod, duel=duel, charts=draw_charts(), tuned=tuned,
                   fold_table=fold_table, per_exercise=per_ex, sarima_summary=sarima_summary,
                   ensemble_members=tuned.get('ensemble_top3', {}).get('members'),
                   device_table=pd.DataFrame(tuned_blob['device_table']) if tuned_blob.get('device_table') else None)
        for rd in report_dirs:
            figs = {'leaderboard': 'figures/ms_leaderboard.png', 'pred_vs_actual': 'figures/ms_pred_vs_actual.png',
                    'error_over_time': 'figures/ms_error_over_time.png'}
            E.plot_leaderboard(board, rd / figs['leaderboard'])
            shown = prod or best_nb or best
            E.plot_pred_vs_actual(y_h, hold_preds[shown], shown, rd / figs['pred_vs_actual'])
            E.plot_error_over_time(hold.dates, y_h, {k: hold_preds[k] for k in dict.fromkeys(
                ['naive_last', best_bl, prod, best_nb]) if k in hold_preds}, rd / figs['error_over_time'])
            write_model_selection(rd / 'model_selection.md', ctx, figs)
        log(f'  production model: {prod} | best overall by dev CV MAPE: {best} | best non-baseline: {best_nb}')
        log(f'  wrote model_selection.md and {len(finals)} model artifacts to {run_dir / "models"}')

    log(f'\nDone. Run folder: {run_dir}')
    return run_dir


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
