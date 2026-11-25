"""Forecast charts by exercise: actual est_1rm by session with one line per model.

``artifacts/<run_id>/charts/by_exercise/*``, each as a static PNG and an interactive HTML. Model lines are out-of-sample only (CV folds, then holdout).
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .evaluate import mape  # noqa: E402

# Two panels so no panel needs more than the palette's 8 hues. Colors belong to the model and are
# the same on every chart (fixed palette order; ensemble keeps its color in both panels).
PANELS = [
    ('Baselines and simple models', ['naive_last', 'naive_drift', 'mean_last2', 'ewma', 'linear_2feat',
                                     'linear_regression', 'sarima', 'ensemble_top3']),
    ('Machine-learning models', ['elasticnet', 'random_forest', 'lightgbm', 'xgboost', 'catboost',
                                 'ensemble_top3']),
]
HUES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
COLORS = {m: HUES[i] for _, models in PANELS for i, m in enumerate(models) if m != 'ensemble_top3'}
COLORS['ensemble_top3'], COLORS['sarima'] = HUES[6], HUES[7]   # violet stays clear of the ML panel's orange
INK, INK_2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e1', '#fcfcfb'
MAX_GAP_DAYS = 45
PNG_LEAD_DAYS = 60     # PNGs start this long before the first prediction; HTML shows all history
META_COLS = ['date', 'exercise', 'set_number', 'y', 'fold', 'is_holdout']
SUBTITLE = ('Average over the sets in a session. Solid = actual, dotted = actual in the holdout. Model lines: one step '
            'ahead, CV folds then holdout. Legend: MAPE per set.')


def slug(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '_', str(name)).strip('_')


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_predictions(run_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Dev rows (predictions only in CV folds) + holdout rows."""
    dev = pd.read_parquet(Path(run_dir) / 'predictions' / 'cv_oof.parquet').assign(is_holdout=False)
    hold = pd.read_parquet(Path(run_dir) / 'predictions' / 'holdout.parquet').assign(is_holdout=True)
    models = [c for c in hold.columns if c not in META_COLS]
    df = pd.concat([dev, hold], ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    return df, models


def session_means(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """One row per date: actual and each model averaged over the same sets."""
    g = df.groupby('date', sort=True)
    out = g[['y', *models]].mean()
    out['n_sets'] = g.size()
    out['is_holdout'] = g['is_holdout'].first()
    return out.reset_index()


def break_gaps(s: pd.DataFrame, col: str, max_gap: int = MAX_GAP_DAYS) -> tuple[pd.Series, pd.Series]:
    """Dates and values with a NaN inserted wherever two points are more than ``max_gap`` days apart."""
    s = s[['date', col]].dropna().sort_values('date')
    gap = s['date'].diff().dt.days > max_gap
    breaks = pd.DataFrame({'date': s.loc[gap, 'date'] - pd.Timedelta(days=1), col: np.nan})
    s = pd.concat([s, breaks]).sort_values('date')
    return s['date'], s[col]


def split_actual(s: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dev part and holdout part; the holdout part starts at the last dev point so the lines join."""
    dev, hold = s[~s['is_holdout']], s[s['is_holdout']]
    if len(dev) and len(hold):
        hold = pd.concat([dev.tail(1), hold])
    return dev, hold


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def _panels(models):
    return [(title, [m for m in names if m in models]) for title, names in PANELS
            if any(m in models for m in names)]


def _labels(rows: pd.DataFrame, models) -> dict:
    """Set-level MAPE on this chart's out-of-sample rows, so it matches the report."""
    out = {}
    for m in models:
        r = rows[rows[m].notna()]
        out[m] = f'{m} ({mape(r["y"], r[m]):.2f}%)' if len(r) else m
    return out


def draw_png(path: Path, title: str, rows: pd.DataFrame, s: pd.DataFrame, models, holdout_start):
    first = s.loc[s[models].notna().any(axis=1), 'date'].min()
    s = s[s['date'] >= first - pd.Timedelta(days=PNG_LEAD_DAYS)]
    panels = _panels(models)
    labels = _labels(rows, models)
    dev, hold = split_actual(s)
    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 4.6 * len(panels) + 0.8), sharex=True, sharey=True,
                             squeeze=False, facecolor=SURFACE)
    for ax, (name, members) in zip(axes[:, 0], panels):
        ax.set_facecolor(SURFACE)
        for m in members:
            x, y = break_gaps(s, m)
            ax.plot(x, y, color=COLORS.get(m, INK_2), lw=1.4, alpha=0.9, solid_capstyle='round', label=labels[m],
                    marker='o' if len(x) < 4 else None, ms=5)
        ax.plot(*break_gaps(dev, 'y'), color=INK, lw=2.4, label='actual', zorder=5)
        ax.plot(*break_gaps(hold, 'y'), color=INK, lw=2.4, ls=(0, (1, 1.2)), label='actual (holdout)', zorder=5)
        ax.axvline(holdout_start, color=INK_2, lw=1)
        ax.annotate('holdout starts', (holdout_start, 1), xycoords=('data', 'axes fraction'), xytext=(4, -4),
                    textcoords='offset points', va='top', fontsize=9, color=INK_2)
        ax.set_title(name, loc='left', fontsize=11, color=INK_2)
        ax.set_ylabel('est_1rm (lb)', color=INK_2)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_2, length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=9)
    locator = mdates.AutoDateLocator()
    axes[-1, 0].xaxis.set_major_locator(locator)
    axes[-1, 0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.suptitle(title, x=0.01, ha='left', fontsize=15, color=INK, y=0.985)
    fig.text(0.01, 0.985 - 0.36 / fig.get_figheight() - 0.01, SUBTITLE, fontsize=9, color=INK_2, va='top')
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.65 / fig.get_figheight()))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def draw_html(path: Path, title: str, rows: pd.DataFrame, s: pd.DataFrame, models, holdout_start, plotly_js: str):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    panels = _panels(models)
    labels = _labels(rows, models)
    dev, hold = split_actual(s)
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, shared_yaxes='all', vertical_spacing=0.08,
                        subplot_titles=[p[0] for p in panels])
    step = 1 / len(panels)
    for i, (_, members) in enumerate(panels, 1):
        legend = 'legend' if i == 1 else f'legend{i}'
        kw = dict(row=i, col=1)
        for m in members:
            x, y = break_gaps(s, m)
            fig.add_trace(go.Scatter(x=x, y=y, name=labels[m], mode='lines+markers', legend=legend,
                                     line=dict(color=COLORS.get(m, INK_2), width=2), marker=dict(size=4),
                                     connectgaps=False, hovertemplate='%{y:.1f}'), **kw)
        for part, name, dash in [(dev, 'actual', 'solid'), (hold, 'actual (holdout)', 'dot')]:
            x, y = break_gaps(part, 'y')
            fig.add_trace(go.Scatter(x=x, y=y, name=name, mode='lines', legend=legend, connectgaps=False,
                                     line=dict(color=INK, width=3, dash=dash), hovertemplate='%{y:.1f}'), **kw)
        fig.add_vline(x=holdout_start, line=dict(color=INK_2, width=1), **kw)
        fig.update_layout({legend: dict(y=1 - (i - 1) * step, yanchor='top', x=1.01, bgcolor='rgba(0,0,0,0)')})
    fig.update_xaxes(gridcolor=GRID, showticklabels=True)
    fig.update_yaxes(gridcolor=GRID, title_text='est_1rm (lb)')
    fig.update_layout(title=dict(text=f'{title}<br><sup>{SUBTITLE} Click a legend entry to hide a model.</sup>'),
                      hovermode='x unified', height=430 * len(panels) + 120, paper_bgcolor=SURFACE,
                      plot_bgcolor=SURFACE, font=dict(family='system-ui, sans-serif', color=INK_2),
                      margin=dict(t=110, r=40, l=60, b=40))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs=plotly_js, full_html=True)


def draw_run_charts(run_dir: Path, log=print) -> dict:
    """Draw one chart per exercise for a run. Returns the names and file stems for the report."""
    from plotly.offline import get_plotlyjs
    run_dir = Path(run_dir)
    out = run_dir / 'charts'
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / 'plotly.min.js').write_text(get_plotlyjs(), encoding='utf-8')
    df, models = load_predictions(run_dir)
    holdout_start = df.loc[df['is_holdout'], 'date'].min()
    result = dict(by_exercise=[], skipped=[])
    for name, rows in df.groupby('exercise', sort=True):
        if not rows[models].notna().any(axis=None):
            result['skipped'].append(str(name))
            continue
        s = session_means(rows, models)
        stem, title = out / 'by_exercise' / slug(name), f'{name}: average est_1rm per session'
        draw_png(stem.with_suffix('.png'), title, rows, s, models, holdout_start)
        draw_html(stem.with_suffix('.html'), title, rows, s, models, holdout_start, '../plotly.min.js')
        result['by_exercise'].append((str(name), slug(name)))
    log(f"  drew charts for {len(result['by_exercise'])} exercises -> {out}")
    return result


def copy_pngs(src: Path, dst: Path):
    """Copy the PNGs next to a report (the folder is cleared first)."""
    if dst.exists():
        shutil.rmtree(dst)
    for png in Path(src).rglob('*.png'):
        target = dst / png.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(png, target)
