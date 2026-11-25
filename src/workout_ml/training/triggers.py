"""Decide whether to retrain. Pure functions over stored documents, so every rule is unit tested."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

from ..config import TrainingSettings

FINISHED = ('promoted', 'rejected', 'candidate')       # runs that produced a model
STALE_AFTER = timedelta(hours=2)                      # a 'running' run older than this has crashed


@dataclass(frozen=True)
class Performance:
    n_sets: int
    mape: float | None         # None when there are too few scored sets to judge
    threshold: float

    @property
    def degraded(self) -> bool:
        return self.mape is not None and self.mape > self.threshold

    @property
    def healthy(self) -> bool:
        return self.mape is not None and not self.degraded


@dataclass(frozen=True)
class Decision:
    train: bool
    mode: str | None
    trigger: str | None        # initial | scheduled | performance
    reason: str
    performance: Performance | None

    def to_dict(self) -> dict:
        return asdict(self)


def live_performance(scored: list[dict], champion: str | None, settings: TrainingSettings,
                     now: datetime) -> Performance:
    """Moving MAPE of the current champion's scored predictions (errors of older models don't count)."""
    p = settings.performance
    since = (now - timedelta(days=p.window_days)).date().isoformat()
    ape = [d['ape'] for d in scored if d['model_version'] == champion and d['actual_date'] > since]
    mape = sum(ape) / len(ape) if len(ape) >= p.min_sets else None
    return Performance(len(ape), mape, p.max_mape)


def _finished(runs: list[dict], **match) -> list[dict]:
    out = [r for r in runs if r['status'] in FINISHED and all(r.get(k) == v for k, v in match.items())]
    return sorted(out, key=lambda r: r['finished'])


def _age_days(run: dict, now: datetime) -> float:
    return (now - datetime.fromisoformat(run['finished'])) / timedelta(days=1)


def _blocked_by_last_run(settings: TrainingSettings, runs: list[dict], now: datetime) -> str | None:
    """A run in progress, or one that failed (or died without finishing) too recently."""
    if not runs:
        return None
    last = max(runs, key=lambda r: r['started'])
    age = now - datetime.fromisoformat(last['started'])
    if last['status'] == 'running' and age < STALE_AFTER:
        return 'a training run is in progress'
    if last['status'] in ('running', 'failed') and age < timedelta(days=settings.retry_failed_after_days):
        return f'the last run failed; retrying after {settings.retry_failed_after_days:g} days'
    return None


def decide(settings: TrainingSettings, champion: str | None, runs: list[dict], perf: Performance,
           streak: dict, now: datetime) -> Decision:
    """Priority: no model yet, then the scheduled sweep, then a performance dip."""
    blocked = _blocked_by_last_run(settings, runs, now)
    if blocked:
        return Decision(False, None, None, blocked, perf)
    if champion is None:
        tried = _finished(runs)          # a rejected first model must not retrain every day
        if tried and _age_days(tried[-1], now) < settings.retry_failed_after_days:
            return Decision(False, None, None, 'no champion yet; the last attempt was not promoted, retrying after '
                            f'{settings.retry_failed_after_days:g} days', perf)
        return Decision(True, 'full', 'initial', 'no champion model yet', perf)

    s = settings.scheduled
    if s.enabled:
        sweeps = [r for r in _finished(runs) if r['trigger'] in ('initial', 'scheduled') or r['mode'] == s.mode]
        if not sweeps:
            return Decision(True, s.mode, 'scheduled', 'no sweep on record', perf)
        age = _age_days(sweeps[-1], now)
        if age >= s.every_days:
            return Decision(True, s.mode, 'scheduled', f'last sweep was {age:.1f} days ago', perf)

    p = settings.performance
    if not p.enabled:
        return Decision(False, None, None, 'nothing due (performance trigger disabled)', perf)
    if perf.mape is None:
        return Decision(False, None, None, f'only {perf.n_sets} scored sets, need {p.min_sets}', perf)
    if not perf.degraded:
        return Decision(False, None, None, f'live MAPE {perf.mape:.2f}% is within {p.max_mape}%', perf)
    if p.pause_after_alert and streak.get('count', 0) >= p.alert_after_consecutive:
        return Decision(False, None, None, 'performance retraining paused after repeated attempts', perf)
    finished = _finished(runs)
    if finished and _age_days(finished[-1], now) < p.cooldown_days:
        return Decision(False, None, None,
                        f'live MAPE {perf.mape:.2f}% is above {p.max_mape}%, but within the '
                        f'{p.cooldown_days}-day cooldown', perf)
    return Decision(True, p.mode, 'performance', f'live MAPE {perf.mape:.2f}% is above {p.max_mape}%', perf)


# --------------------------------------------------------------------------- #
# Back-to-back performance retrains
# --------------------------------------------------------------------------- #
def streak_after_check(streak: dict, perf: Performance) -> dict:
    """Performance recovered -> start over. Too little data, or still degraded -> unchanged."""
    return {'count': 0, 'run_ids': []} if perf.healthy else streak or {'count': 0, 'run_ids': []}


def streak_after_run(streak: dict, run: dict) -> dict:
    """Only finished performance retrains count (promoted or not); scheduled sweeps leave it alone."""
    streak = streak or {'count': 0, 'run_ids': []}
    if run['trigger'] != 'performance' or run['status'] not in FINISHED:
        return streak
    return {'count': streak['count'] + 1, 'run_ids': [*streak['run_ids'], run['run_id']]}


def should_alert(streak: dict, run: dict, settings: TrainingSettings) -> bool:
    n = settings.performance.alert_after_consecutive
    return run['trigger'] == 'performance' and run['status'] in FINISHED \
        and streak['count'] > 0 and streak['count'] % n == 0


def alert_text(streak: dict, runs: list[dict], settings: TrainingSettings) -> tuple[str, str]:
    p = settings.performance
    subject = f'[workout-ml] {streak["count"]} performance retrains in a row'
    lines = [f'The live error has triggered {streak["count"]} retrains in a row without recovering to',
             f'{p.max_mape}% MAPE or lower over {p.window_days} days.', '']
    wanted = set(streak['run_ids'])
    for r in sorted((r for r in runs if r['run_id'] in wanted), key=lambda r: r['finished']):
        g = r.get('gate') or {}
        lines += [f'{r["finished"][:16]}  {r["status"].upper()}  mode={r["mode"]}  version={r.get("version")}',
                  f'  why: {r["reason"]}',
                  f'  backtest MAPE: candidate {_pct(g.get("candidate_mape"))}, previous model '
                  f'{_pct(g.get("champion_mape"))}, EWMA {_pct(g.get("ewma_mape"))}, last session '
                  f'{_pct(g.get("naive_mape"))}',
                  f'  gate: {g.get("reason", "n/a")}', '']
    lines += ['Retraining is now paused until performance recovers or a scheduled sweep runs.' if p.pause_after_alert
              else f'Retraining continues, at most once every {p.cooldown_days} days.',
              'Check for data-entry problems, a training break, or a change of program.']
    return subject, '\n'.join(lines)


def _pct(v) -> str:
    return 'n/a' if v is None else f'{v:.2f}%'
