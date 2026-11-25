"""Report plots. Metrics live in ``workout_ml.core.metrics`` and are re-exported here."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from workout_ml.core.metrics import *  # noqa: E402,F401,F403


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_k_curves(curves: pd.DataFrame, chosen_k: dict, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ks = sorted(curves['k'].unique())
    pos = {k: i for i, k in enumerate(ks)}          # evenly spaced: the grid is irregular
    for (family, g), color in zip(curves.groupby('family'), ['#2a78d6', '#eb6834']):
        x = [pos[k] for k in g['k']]
        ax.plot(x, g['cv_mape'], marker='o', lw=2, color=color, label=family)
        sel = g[g['k'] == chosen_k[family]]
        ax.scatter([pos[k] for k in sel['k']], sel['cv_mape'], s=180, facecolors='none',
                   edgecolors=color, linewidths=2)
    ax.set_xticks(range(len(ks)), [str(k) for k in ks])
    ax.set_xlabel('number of ranked features added to the two force-kept ones')
    ax.set_ylabel('walk-forward CV MAPE (%)')
    ax.set_title('CV MAPE vs number of features (circled = chosen)')
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_importance(ranking: pd.DataFrame, path: Path, top: int = 30):
    r = ranking.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(r))))
    colors = ['#bbbbbb' if p else ('#d98c3a' if b else '#3a78b5')
              for p, b in zip(r['is_probe'], r['below_probe'])]
    ax.barh(r['feature'], -r['mean_rank'] + ranking['mean_rank'].max() + 1, color=colors)
    ax.set_xlabel('importance (higher = better mean rank across folds)')
    ax.set_title(f'Top {len(r)} by combined SHAP + permutation rank (grey = random probe)')
    ax.set_xticks([])
    return _save(fig, path)


def plot_leaderboard(board: pd.DataFrame, path: Path):
    b = board.sort_values('cv_mape')
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(b) + 1.5))
    y = np.arange(len(b))
    ax.barh(y, b['cv_mape'], color=['#9a9994' if g == 'baseline' else '#2a78d6' for g in b['group']], height=0.6)
    ax.scatter(b['holdout_mape'], y, color='#0b0b0b', zorder=3)
    ax.set_yticks(y, b['model'])
    ax.invert_yaxis()
    ax.set_xlabel('MAPE (%)')
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(color='#9a9994', label='dev CV, baseline'), Patch(color='#2a78d6', label='dev CV, non-baseline'),
               Line2D([], [], marker='o', ls='', color='#0b0b0b', label='holdout')]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    ax.grid(axis='x', alpha=0.3)
    return _save(fig, path)


def plot_pred_vs_actual(y, p, name: str, path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y, p, s=8, alpha=0.4)
    lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
    ax.plot([lo, hi], [lo, hi], color='grey', lw=1)
    ax.set_xlabel('actual est_1rm')
    ax.set_ylabel('predicted est_1rm')
    ax.set_title(f'Holdout: {name}')
    return _save(fig, path)


def plot_error_over_time(dates, y, preds: dict, path: Path):
    df = pd.DataFrame({'date': pd.to_datetime(dates), 'y': np.asarray(y, float)})
    fig, ax = plt.subplots(figsize=(9, 4))
    for n, p in preds.items():
        e = ((df['y'] - np.asarray(p, float)).abs() / df['y'].abs() * 100).groupby(df['date']).mean()
        ax.plot(e.index, e.rolling(5, min_periods=1).mean(), label=n)
    ax.set_ylabel('MAPE per workout, % (5-workout rolling)')
    ax.set_title('Holdout error over time')
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, path)
