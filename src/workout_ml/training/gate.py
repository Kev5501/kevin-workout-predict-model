"""Promotion gate: is the candidate good enough to replace the champion?"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..config import Promotion


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reason: str
    candidate_mape: float
    champion_mape: float | None      # champion recipe refit on the same data; None for the first model
    naive_mape: float
    ewma_mape: float                 # reported for reference, not a criterion
    n_sets: int
    n_workouts: int
    eval_start: str
    eval_end: str

    def to_dict(self) -> dict:
        return asdict(self)


def judge(candidate: float, champion: float | None, naive: float, predictions, last_session,
          settings: Promotion) -> tuple[bool, str]:
    p, last = np.asarray(predictions, float), np.asarray(last_session, float)
    if not np.isfinite(p).all() or (p <= 0).any():
        return False, 'candidate produced non-finite or non-positive predictions'
    if (np.abs(p / last - 1) > 0.6).any():
        return False, 'candidate produced a prediction more than 60% away from the last session'
    if settings.must_beat_naive and candidate >= naive:
        return False, f'candidate {candidate:.2f}% does not beat repeating the last session ({naive:.2f}%)'
    if champion is not None and candidate > champion * (1 + settings.max_relative_worse):
        return False, (f'candidate {candidate:.2f}% is more than {settings.max_relative_worse:.0%} worse than the '
                       f'champion {champion:.2f}%')
    if champion is None:
        return True, 'first model'
    return True, f'candidate {candidate:.2f}% vs champion {champion:.2f}%'
