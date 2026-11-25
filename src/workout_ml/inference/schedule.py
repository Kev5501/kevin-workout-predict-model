"""Which workouts belong in the next batch."""
from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from datetime import date, timedelta

from ..config import REST, WEEKDAYS

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlannedWorkout:
    date: date
    gym_day: str

    @property
    def weekday(self) -> str:
        return WEEKDAYS[self.date.weekday()]


def plan_batch(run_date: date, schedule: Mapping[str, str], exercises_for: Callable[[str], Collection[str]],
               is_logged: Callable[[date, str], bool], max_horizon_days: int = 7) -> list[PlannedWorkout]:
    """Upcoming workouts until one would repeat an exercise already in the batch.

    A prediction uses everything logged so far. Once a lift appears a second time, its second
    prediction would be missing the first session's result, so the batch stops there. Today counts
    unless its workout is already logged.
    """
    batch: list[PlannedWorkout] = []
    seen: set[str] = set()
    for offset in range(max_horizon_days):
        day = run_date + timedelta(days=offset)
        gym_day = schedule.get(WEEKDAYS[day.weekday()], REST)
        if gym_day.lower() == REST.lower():
            continue
        if offset == 0 and is_logged(day, gym_day):
            continue
        exercises = set(exercises_for(gym_day))
        if not exercises:
            log.warning('no history for scheduled gym day', extra={'gym_day': gym_day, 'date': day.isoformat()})
            continue
        if seen & exercises:
            break
        seen |= exercises
        batch.append(PlannedWorkout(day, gym_day))
    return batch
