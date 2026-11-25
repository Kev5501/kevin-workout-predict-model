"""Stable document ids, so a rerun overwrites instead of duplicating."""
from __future__ import annotations

import re


def slug(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')


def prediction_id(day: str, exercise: str, set_number: int) -> str:
    return f'{day}_{slug(exercise)}_s{int(set_number)}'
