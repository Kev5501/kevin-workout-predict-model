"""Build the feature dataset used by the modeling pipeline.

Google Sheets -> cast -> ``features.build_features`` -> ``research/data/``:

- ``features.parquet``      pipeline input (one row per set, all features)
- ``features_config.json``  settings used, so inference rebuilds features identically
- ``raw_sets.parquet``      snapshot of the casted sheet (history for ``predict_next``)
- ``flagged_sets.csv``      sets excluded by the bad-data guard (fix them in the sheet)

From the ``research`` folder::

    ..\\.venv\\Scripts\\python -m modeling.build_features
    ..\\.venv\\Scripts\\python -m modeling.build_features --input data/raw_sets.parquet   # offline rebuild
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import features as F
from .config import CREDENTIALS_PATH, DATA_PATH, FEATURE_SETTINGS, RAW_PATH
from .sheets import clean_raw, load_raw_data


def build_dataset(raw: pd.DataFrame, settings: dict = FEATURE_SETTINGS) -> pd.DataFrame:
    feats = F.build_features(raw, **settings)
    if len(feats) != len(raw):
        raise RuntimeError(f'Feature build changed the row count: {len(raw)} -> {len(feats)}')
    return feats


def save_dataset(feats: pd.DataFrame, settings: dict, output: Path, raw: pd.DataFrame | None = None,
                 raw_output: Path | None = None) -> dict:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(output, index=False)
    config_path = output.with_name('features_config.json')
    config_path.write_text(json.dumps(settings, indent=2), encoding='utf-8')
    paths = {'features': output, 'config': config_path}
    if raw is not None and raw_output is not None:
        Path(raw_output).parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(raw_output, index=False)
        paths['raw'] = Path(raw_output)
    return paths


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', help='raw .parquet/.csv to use instead of Google Sheets')
    p.add_argument('--credentials', default=str(CREDENTIALS_PATH), help='service-account key for Google Sheets')
    p.add_argument('--output', default=str(DATA_PATH), help='feature parquet to write')
    p.add_argument('--raw-output', default=str(RAW_PATH),
                   help='where to save the casted raw snapshot ("" to skip)')
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    source = 'Google Sheets' if args.input is None else args.input
    print(f'Loading raw sets from {source}')
    raw = load_raw_data(args.input, credentials=args.credentials)
    print(f'  {len(raw)} sets, {raw["date"].nunique()} workout dates '
          f'({raw["date"].min().date()} to {raw["date"].max().date()})')

    clean, flagged = clean_raw(raw)
    flagged_path = Path(args.output).with_name('flagged_sets.csv')
    flagged_path.parent.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(flagged_path, index=False)
    if len(flagged):
        print(f'  EXCLUDED {len(flagged)} set(s) with an impossible est. 1RM. Check these in the sheet:')
        for _, r in flagged.iterrows():
            print(f'    {r["set_id"]} {r["date"].date()} {r["exercise"]} set {r["set_number"]}: weight '
                  f'{r["weight"]}, est. 1RM {r["rpe_norm_weight"]} (recent sessions ~{r["expected_est_1rm"]:.0f})')

    feats = build_dataset(clean, FEATURE_SETTINGS)
    n_feature_cols = len(F.get_feature_columns(feats))
    write_raw = args.raw_output and args.input is None   # don't overwrite the snapshot with itself
    paths = save_dataset(feats, FEATURE_SETTINGS, Path(args.output),
                         raw if write_raw else None, Path(args.raw_output) if write_raw else None)
    paths['flagged'] = flagged_path
    print(f'  built {feats.shape[1]} columns ({n_feature_cols} usable model features)')
    for k, v in paths.items():
        print(f'  wrote {k}: {v}')
    return paths


if __name__ == '__main__':
    main()
    sys.exit(0)
