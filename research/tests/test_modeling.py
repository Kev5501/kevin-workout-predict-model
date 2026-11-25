import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from modeling import features as F
from conftest import BUILD_KW, make_raw
from modeling import charts as C
from modeling import evaluate as E
from modeling.config import BASE_COL, EWMA_COL, FORCE_KEEP, PRODUCTION_MODEL, Config
from modeling.feature_selection import choose_k
from modeling.sheets import clean_raw
from modeling.models import REGISTRY, BuildContext, TargetModeRegressor, build_model
from modeling.splits import WalkForwardSplit, holdout_split


# --------------------------------------------------------------------------- splits
def test_holdout_and_folds_respect_time(dataset):
    dates = dataset.dates
    dev, hold = holdout_split(dates, 0.15)
    assert dates.iloc[dev].max() < dates.iloc[hold].min()
    assert len(set(dev) & set(hold)) == 0 and len(dev) + len(hold) == len(dates)

    dev_dates = dates.iloc[dev].reset_index(drop=True)
    folds = WalkForwardSplit(5, 0.6).split(dev_dates)
    prev_val_end = None
    for tr, va in folds:
        assert dev_dates.iloc[tr].max() < dev_dates.iloc[va].min()          # train strictly before val
        tr_days, va_days = set(dev_dates.iloc[tr]), set(dev_dates.iloc[va])
        assert not tr_days & va_days                                       # no day split across sets
        if prev_val_end is not None:
            assert dev_dates.iloc[va].min() > prev_val_end                 # blocks move forward
        prev_val_end = dev_dates.iloc[va].max()


# --------------------------------------------------------------------------- data loading / build script
def test_cast_raw_handles_sheet_strings():
    from modeling.sheets import cast_raw
    sheet = pd.DataFrame({'Date': ['4/8/2025', '4/10/2025'], 'Set_ID': ['SET-00001', 'SET-00002'],
                          'Weight': ['22.5', ''], 'Reps': [6, ''], 'RPE': [10, 9.5], 'Injury_Ind': ['', 1],
                          'Exercise': ['Bench Press', 'Squat'], 'Comment': ['', 'felt good']})
    df = cast_raw(sheet)
    assert list(df.columns) == [c.lower() for c in sheet.columns]
    assert pd.api.types.is_datetime64_any_dtype(df['date'])
    assert df['weight'].dtype == float and np.isnan(df['weight'].iloc[1])
    assert str(df['reps'].dtype) == 'Int64' and df['reps'].isna().iloc[1]
    assert str(df['injury_ind'].dtype) == 'Int64' and df['injury_ind'].iloc[1] == 1


def test_build_features_script_offline(raw, tmp_path):
    from modeling.build_features import main
    raw_file = tmp_path / 'raw.parquet'
    raw.to_parquet(raw_file, index=False)
    out = tmp_path / 'data' / 'features.parquet'
    paths = main(['--input', str(raw_file), '--output', str(out)])
    feats = pd.read_parquet(paths['features'])
    assert len(feats) == len(raw)
    assert json.loads(paths['config'].read_text()) == json.loads(json.dumps(BUILD_KW))
    expected = F.build_features(raw, **BUILD_KW)
    assert list(feats.columns) == list(expected.columns)
    np.testing.assert_allclose(feats['est_1rm_avg_30d'].to_numpy(dtype=float),
                               expected['est_1rm_avg_30d'].to_numpy(dtype=float))
    assert 'raw' not in paths   # offline rebuild never overwrites the snapshot


# --------------------------------------------------------------------------- target mode
def test_delta_reconstruction_is_exact(dataset):
    X, y = dataset.X, dataset.y
    m = TargetModeRegressor(DummyRegressor(strategy='constant', constant=2.5), target_mode='delta').fit(X, y)
    np.testing.assert_allclose(m.predict(X), X[BASE_COL].to_numpy() + 2.5)
    m = TargetModeRegressor(DummyRegressor(strategy='constant', constant=7.0), target_mode='level').fit(X, y)
    np.testing.assert_allclose(m.predict(X), 7.0)


def test_pct_reconstruction_is_exact(dataset):
    X, y = dataset.X, dataset.y
    for base in (BASE_COL, EWMA_COL):
        m = TargetModeRegressor(DummyRegressor(strategy='constant', constant=np.log(1.03)), target_mode='pct',
                                base_col=base).fit(X, y)
        np.testing.assert_allclose(m.predict(X), X[base].to_numpy() * 1.03)


# --------------------------------------------------------------------------- metric / winners / k
def test_mape_by_hand():
    assert E.mape([100, 200], [110, 190]) == pytest.approx((10 + 5) / 2)
    m = E.metrics([100, 200], [110, 190], y_naive=[120, 200])
    assert m['mae'] == 10 and m['rmse'] == 10 and m['skill_vs_naive'] == pytest.approx(1 - 7.5 / 10)


def test_bootstrap_difference_signs():
    y = np.tile([100.0, 200.0], 30)
    dates = np.repeat(pd.date_range('2026-01-01', periods=30), 2)
    r = E.block_bootstrap(y, {'ref': y * 1.1, 'good': y * 1.02}, dates, 'ref', 200, 0).set_index('model')
    assert r.loc['good', 'mape_diff'] == pytest.approx(-8) and r.loc['good', 'mape_diff_high'] < 0
    assert r.loc['ref', 'mape_diff'] == 0 and r.loc['good', 'p_better'] == 1


def test_pick_winners_lets_baseline_win_and_names_non_baseline():
    b = pd.DataFrame({'model': ['lightgbm', 'ewma', 'linear_2feat', 'naive_last'],
                      'group': ['trained', 'baseline', 'simple', 'baseline'],
                      'cv_mape': [2.9, 2.8, 2.95, 3.0], 'holdout_mape': [1.0, 9, 9, 9]})
    assert E.pick_winners(b) == dict(best_overall='ewma', best_baseline='ewma', best_non_baseline='lightgbm')
    b.loc[0, 'cv_mape'] = 2.0
    assert E.pick_winners(b)['best_overall'] == 'lightgbm' and E.pick_winners(b)['best_baseline'] == 'ewma'


def test_choose_k_ignores_difficulty_but_respects_real_gaps():
    rng = np.random.default_rng(0)
    hard = rng.uniform(1, 20, 100)                       # dates differ a lot in difficulty
    ape = pd.DataFrame({5: hard + 1.0 + rng.normal(0, .1, 100), 20: hard + rng.normal(0, .1, 100),
                        40: hard + rng.normal(0, .1, 100) - .005})
    k, _ = choose_k(ape)
    assert k == 20                                       # 5 is clearly worse on every date; 40 isn't clearly better


# --------------------------------------------------------------------------- guard and smoothed features
def test_guard_catches_typo_and_nothing_else(raw):
    clean, flagged = clean_raw(raw)
    assert len(flagged) == 0 and len(clean) == len(raw)
    typo = raw.copy()
    i = typo.index[(typo.exercise == 'Squat') & (typo.set_number == 1)][20]
    typo.loc[i, ['weight', 'rpe_norm_weight']] = typo.loc[i, ['weight', 'rpe_norm_weight']] / 100
    clean, flagged = clean_raw(typo)
    assert flagged['set_id'].tolist() == [typo.loc[i, 'set_id']] and len(clean) == len(raw) - 1

    # a real 35% drop (e.g. a machine or form change) is not an outlier and must be kept
    drop = raw.copy()
    late = (drop.exercise == 'Squat') & (drop['date'] > drop['date'].median())
    drop.loc[late, ['weight', 'rpe_norm_weight']] *= 0.65
    assert len(clean_raw(drop)[1]) == 0


def test_smoothed_and_relative_features_by_hand(feats):
    s = feats[(feats.exercise == 'Squat') & (feats.set_number == 1)].sort_values('date').reset_index(drop=True)
    i = 12
    past = s.loc[:i - 1, 'est_1rm'].to_numpy()
    e = past[0]
    for v in past[1:]:
        e = 0.5 * v + 0.5 * e
    row = s.loc[i]
    assert row['est_1rm_ewma_50'] == pytest.approx(e)
    assert row['est_1rm_mean_last2'] == pytest.approx(past[-2:].mean())
    assert row['est_1rm_mean_last3'] == pytest.approx(past[-3:].mean())
    assert row['est_1rm_ewma_50_vs_recent'] == pytest.approx(e / past[-1] - 1)
    assert np.isnan(s.loc[0, 'est_1rm_ewma_50'])


# --------------------------------------------------------------------------- charts
def test_session_means_and_dotted_join():
    df = pd.DataFrame({'date': pd.to_datetime(['2026-01-01'] * 2 + ['2026-01-05'] * 2 + ['2026-01-09']),
                       'y': [100, 110, 120, 130, 140.0], 'm': [np.nan, np.nan, 100, 110, 150],
                       'is_holdout': [False, False, False, False, True]})
    s = C.session_means(df, ['m'])
    assert s['y'].tolist() == [105, 125, 140] and s['n_sets'].tolist() == [2, 2, 1]
    assert np.isnan(s['m'][0]) and s['m'].tolist()[1:] == [105, 150]
    dev, hold = C.split_actual(s)
    assert dev['date'].max() == hold['date'].min() == pd.Timestamp('2026-01-05')     # joined at the boundary
    assert hold['y'].tolist() == [125, 140]
    x, y = C.break_gaps(pd.DataFrame({'date': pd.to_datetime(['2026-01-01', '2026-06-01']), 'y': [1.0, 2.0]}), 'y')
    assert len(x) == 3 and np.isnan(y.iloc[1])


# --------------------------------------------------------------------------- leakage
def test_features_ignore_future_targets(raw):
    cutoff = raw['date'].sort_values().iloc[len(raw) // 2]
    changed = raw.copy()
    later = changed['date'] > cutoff
    changed.loc[later, 'rpe_norm_weight'] = changed.loc[later, 'rpe_norm_weight'] * 3
    changed.loc[later, 'weight'] = changed.loc[later, 'weight'] * 3
    a = F.build_features(raw, **BUILD_KW)
    b = F.build_features(changed, **BUILD_KW)
    cols = F.get_feature_columns(a)
    early = a['date'] <= cutoff
    pd.testing.assert_frame_equal(a.loc[early, cols].reset_index(drop=True),
                                  b.loc[early, cols].reset_index(drop=True))


def test_sarima_uses_only_past_sessions(dataset):
    cfg = Config().apply_quick()
    X, y = dataset.X, dataset.y
    dev, hold = holdout_split(X['date'], 0.3)
    ctx = BuildContext(dataset.features, dataset.categorical, n_threads=2, cfg=cfg)
    model = build_model('sarima', {}, ctx).fit(X.iloc[dev], y.iloc[dev])
    assert len(model.series_) > 0, 'expected SARIMA fits on synthetic series'

    Xh = X.iloc[hold].reset_index(drop=True)
    base_pred = model.predict(Xh)
    mid = Xh['date'].sort_values().iloc[len(Xh) // 2]
    tampered = Xh.copy()
    later = tampered['date'] > mid
    tampered.loc[later, BASE_COL] = tampered.loc[later, BASE_COL] * 5   # rewrite "future" sessions
    new_pred = model.predict(tampered)
    np.testing.assert_allclose(base_pred[~later.to_numpy()], new_pred[~later.to_numpy()])
    assert not np.allclose(base_pred[later.to_numpy()], new_pred[later.to_numpy()])
    assert model.coverage(Xh) > 0


# --------------------------------------------------------------------------- registry
@pytest.mark.parametrize('name', list(REGISTRY))
def test_every_model_fits_predicts_and_persists(name, dataset, tmp_path):
    cfg = Config().apply_quick()
    X, y = dataset.X, dataset.y
    spec = REGISTRY[name]
    feats = dataset.features if spec.family is None else [BASE_COL, 'exercise', 'gym_day',
                                                           'est_1rm_avg_30d', 'est_1rm_slope_30d',
                                                           'days_since_last_exercise_set', 'week_of_year']
    params = {'target_mode': 'delta'} if spec.kind == 'feature' else {}
    if name in ('lightgbm', 'xgboost', 'random_forest'):
        params['n_estimators'] = 30
    if name == 'catboost':
        params['iterations'] = 30
    ctx = BuildContext(feats, dataset.categorical, device='cpu', n_threads=2, cfg=cfg)
    n = int(len(X) * 0.8)
    model = build_model(name, params, ctx).fit(X.iloc[:n], y.iloc[:n])
    pred = model.predict(X.iloc[n:])
    assert pred.shape == (len(X) - n,) and np.isfinite(pred).all()

    path = tmp_path / f'{name}.joblib'
    joblib.dump(model, path)
    np.testing.assert_allclose(joblib.load(path).predict(X.iloc[n:]), pred)


def test_unseen_category_does_not_crash(dataset):
    X, y = dataset.X, dataset.y
    feats = [BASE_COL, 'exercise', 'est_1rm_avg_30d']
    for name in ('lightgbm', 'xgboost', 'catboost', 'random_forest', 'elasticnet'):
        params = {'target_mode': 'delta'}
        params.update({'iterations': 20} if name == 'catboost' else
                      {'n_estimators': 20} if name != 'elasticnet' else {})
        m = build_model(name, params, BuildContext(feats, dataset.categorical, n_threads=1)).fit(X, y)
        Xn = X.head(5).copy()
        Xn['exercise'] = 'Brand New Exercise'
        assert np.isfinite(m.predict(Xn)).all(), name


# --------------------------------------------------------------------------- end to end
@pytest.fixture(scope='module')
def quick_run(feats, tmp_path_factory, monkeypatch_module):
    import modeling.inference as inf
    import modeling.run_pipeline as rp
    root = tmp_path_factory.mktemp('run')
    path = root / 'features.parquet'
    feats.to_parquet(path, index=False)
    (root / 'features_config.json').write_text(json.dumps(BUILD_KW))
    monkeypatch_module.setattr(rp, 'ARTIFACTS_DIR', root / 'artifacts')
    monkeypatch_module.setattr(inf, 'ARTIFACTS_DIR', root / 'artifacts')
    run_dir = rp.main(['--features', str(path), '--quick', '--device', 'cpu', '--trials', '2',
                       '--run-id', 'pytest_quick', '--no-report-copy', '--trial-jobs', '2'])
    return run_dir


@pytest.fixture(scope='module')
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_pipeline_writes_reports_and_artifacts(quick_run, dataset):
    rd = quick_run
    for f in ['reports/feature_selection.md', 'reports/model_selection.md', 'selected_features.json',
              'tuned.json', 'summary.json', 'leaderboard.csv', 'predictions/holdout.parquet',
              'predictions/cv_oof.parquet', 'ratio_table.csv', 'features_config.json']:
        assert (rd / f).exists(), f
    board = pd.read_csv(rd / 'leaderboard.csv')
    expected = set(REGISTRY) | {'ensemble_top3'}
    assert set(board['model']) == expected
    for name in expected:
        model = joblib.load(rd / 'models' / f'{name}.joblib')
        meta = json.loads((rd / 'models' / f'{name}.json').read_text())
        assert np.isfinite(model.predict(dataset.X)).all(), name
        assert meta['final_train_rows'] == len(dataset.y)
    sel = json.loads((rd / 'selected_features.json').read_text())
    for fam in ('tree', 'linear'):
        assert set(FORCE_KEEP) <= set(sel['selected'][fam])
    summary = json.loads((rd / 'summary.json').read_text())
    assert summary['best_model'] == board.sort_values('cv_mape')['model'].iloc[0]
    assert summary['best_non_baseline'] == board[board['group'] != 'baseline'].sort_values('cv_mape')['model'].iloc[0]
    assert {'cv_mape', 'cv_mae', 'cv_rmse', 'holdout_mape', 'holdout_mae', 'holdout_rmse'} <= set(board.columns)
    ms = (rd / 'reports' / 'model_selection.md').read_text(encoding='utf-8')
    assert 'Production model: `lightgbm`' in ms and 'Best overall' in ms and 'Best non-baseline' in ms
    assert summary['production_model'] == PRODUCTION_MODEL


def test_ensemble_reload_is_identical(quick_run, dataset):
    """The saved ensemble equals the mean of its separately saved members."""
    meta = json.loads((quick_run / 'models' / 'ensemble_top3.json').read_text())
    ens = joblib.load(quick_run / 'models' / 'ensemble_top3.joblib').predict(dataset.X)
    parts = [joblib.load(quick_run / 'models' / f'{m}.joblib').predict(dataset.X) for m in meta['members']]
    np.testing.assert_allclose(ens, np.mean(parts, axis=0))


def test_charts_written(quick_run):
    ch = quick_run / 'charts'
    assert sorted(p.name for p in ch.iterdir()) == ['by_exercise', 'plotly.min.js']   # no overall or gym-day charts
    for f in ['by_exercise/Squat', 'by_exercise/T_Bar_Row']:
        assert (ch / f'{f}.png').exists() and (ch / f'{f}.html').exists(), f
    assert (ch / 'plotly.min.js').exists()
    assert (quick_run / 'reports' / 'figures' / 'charts' / 'by_exercise' / 'Squat.png').exists()
    ms = (quick_run / 'reports' / 'model_selection.md').read_text(encoding='utf-8')
    assert 'figures/charts/by_exercise/Squat.png' in ms
    assert 'overall.png' not in ms and 'by_gym_day' not in ms


def test_selection_and_tuning_never_see_the_holdout(feats, dataset, tmp_path, monkeypatch):
    """Rewriting every holdout row must leave selected features, tuned parameters and CV scores identical."""
    import modeling.run_pipeline as rp
    _, hold = holdout_split(dataset.dates, Config().holdout_frac)
    start = dataset.dates.iloc[hold].min()
    tampered = feats.copy()
    num = [c for c in tampered.columns if pd.api.types.is_float_dtype(tampered[c])]
    late = tampered['date'] >= start
    assert late.any()
    tampered.loc[late, num] = tampered.loc[late, num] * 3 + 7
    results = []
    for name, f in [('a', feats), ('b', tampered)]:
        d = tmp_path / name
        d.mkdir()
        f.to_parquet(d / 'features.parquet', index=False)
        monkeypatch.setattr(rp, 'ARTIFACTS_DIR', d / 'artifacts')
        run = rp.main(['--features', str(d / 'features.parquet'), '--quick', '--device', 'cpu', '--trials', '3',
                       '--trial-jobs', '1', '--run-id', 'iso', '--no-report-copy', '--stages', 'select,tune',
                       '--models', 'naive_last,ewma,linear_regression,elasticnet,lightgbm'])
        sel = json.loads((run / 'selected_features.json').read_text())
        tuned = json.loads((run / 'tuned.json').read_text())['tuned']
        results.append((sel, {k: (v['params'], v['cv_mape']) for k, v in tuned.items()}))
    assert results[0] == results[1]


def test_inference_helpers(quick_run, raw):
    from modeling.inference import e1rm_to_weight, load_model, make_future_rows, predict_next
    assert load_model(run_dir=quick_run)[1]['model'] == PRODUCTION_MODEL == 'lightgbm'   # inference default
    with pytest.raises(FileNotFoundError):
        load_model('not_a_model', quick_run)
    model, meta = load_model('best', quick_run)
    assert meta['model'] in REGISTRY or meta['model'] == 'ensemble_top3'

    # synthetic data has weight = 0.8 * e1RM at every rep count
    w = e1rm_to_weight(pd.Series([100.0, 200.0]), reps=8, rpe=10, run_dir=quick_run)
    np.testing.assert_allclose(w, [80.0, 160.0], rtol=0.01)

    nxt = raw['date'].max() + pd.Timedelta(days=2)
    future = make_future_rows(nxt, 'Push', [{'exercise': 'Bench Press', 'sets': 2},
                                            {'exercise': 'Never Done Before', 'sets': 1}])
    out = predict_next(future, history=raw, run_dir=quick_run)
    bench = out[out['exercise'] == 'Bench Press']
    assert (out['model'] == PRODUCTION_MODEL).all()
    assert len(out) == 3 and np.isfinite(bench['predicted_est_1rm']).all()
    assert out.loc[out['exercise'] == 'Never Done Before', 'predicted_est_1rm'].isna().all()
    # prediction should be near the last bench e1RM
    last = raw[raw['exercise'] == 'Bench Press'].groupby('set_number')['rpe_norm_weight'].last()
    assert (abs(bench['predicted_est_1rm'].to_numpy() - last.to_numpy(dtype=float)) < 15).all()
