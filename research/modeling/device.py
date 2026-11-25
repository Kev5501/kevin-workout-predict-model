"""GPU probing and CPU vs GPU benchmarking for the gradient-boosting libraries.

Findings on this machine (GTX 1080 Ti, compute capability 6.1):
- CatBoost GPU works.
- XGBoost 3.4 accepts ``device='cuda'`` but silently trains on CPU. The probe
  catches this by reading the device recorded in the trained booster's config.
- The PyPI LightGBM wheel has no GPU support.
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd

from .models import BuildContext, build_model

GPU_LIBS = ['lightgbm', 'xgboost', 'catboost']
_MODEL_FOR_LIB = {'lightgbm': 'lightgbm', 'xgboost': 'xgboost', 'catboost': 'catboost'}


def probe(lib: str) -> tuple[bool, str]:
    rng = np.random.default_rng(0)
    X = rng.random((300, 5))
    y = X.sum(1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if lib == 'lightgbm':
                import lightgbm as lgb
                lgb.LGBMRegressor(n_estimators=5, device_type='gpu', verbose=-1).fit(X, y)
                return True, 'OpenCL GPU build'
            if lib == 'xgboost':
                import xgboost as xgb
                m = xgb.XGBRegressor(n_estimators=5, device='cuda').fit(X, y)
                dev = json.loads(m.get_booster().save_config())['learner']['generic_param'].get('device', '')
                if str(dev).startswith('cuda'):
                    return True, f'trained on {dev}'
                return False, f'silently fell back to {dev or "cpu"} (cause not determined)'
            if lib == 'catboost':
                from catboost import CatBoostRegressor
                CatBoostRegressor(iterations=5, task_type='GPU', verbose=0, allow_writing_files=False).fit(X, y)
                return True, 'CUDA'
    except Exception as e:  # noqa: BLE001 - any failure means "no GPU"
        return False, f'{type(e).__name__}: {str(e).splitlines()[0][:120]}'
    return False, 'unknown library'


def _time_fit(name, device, X, y, features, categorical, threads, seed) -> float:
    ctx = BuildContext(features, categorical, device=device, n_threads=threads, seed=seed)
    model = build_model(name, {}, ctx)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t = time.perf_counter()
        model.fit(X, y)
        return time.perf_counter() - t


def resolve_devices(cfg, X: pd.DataFrame, y: pd.Series, features: list[str],
                    categorical: list[str]) -> tuple[dict, pd.DataFrame]:
    """Return {lib: 'cpu'|'gpu'} plus a table explaining the choice.

    In ``auto`` mode the decision compares throughput, not single-fit time:
    on CPU, ``trial_jobs`` Optuna trials run at once (each with
    ``model_threads`` threads), while GPU trials run one at a time.
    """
    rows, devices = [], {}
    for lib in GPU_LIBS:
        if cfg.device == 'cpu':
            devices[lib] = 'cpu'
            rows.append(dict(library=lib, gpu_available='not checked', note='--device cpu',
                             cpu_fit_s=None, gpu_fit_s=None, chosen='cpu'))
            continue
        ok, note = probe(lib)
        row = dict(library=lib, gpu_available=ok, note=note, cpu_fit_s=None, gpu_fit_s=None)
        if not ok:
            devices[lib] = 'cpu'
        elif cfg.device == 'gpu':
            devices[lib] = 'gpu'
        else:  # auto: benchmark one default fit on the dev data
            name = _MODEL_FOR_LIB[lib]
            cpu_s = _time_fit(name, 'cpu', X, y, features, categorical, cfg.model_threads, cfg.seed)
            gpu_s = _time_fit(name, 'gpu', X, y, features, categorical, 1, cfg.seed)
            row.update(cpu_fit_s=round(cpu_s, 3), gpu_fit_s=round(gpu_s, 3))
            cpu_throughput = cfg.trial_jobs / cpu_s
            gpu_throughput = 1.0 / gpu_s
            devices[lib] = 'gpu' if gpu_throughput > cpu_throughput else 'cpu'
            row['note'] = (f'{note}; fits/s CPU x{cfg.trial_jobs} parallel = {cpu_throughput:.2f}, '
                           f'GPU = {gpu_throughput:.2f}')
        row['chosen'] = devices[lib]
        rows.append(row)
    return devices, pd.DataFrame(rows)
