"""Shared dependencies."""
from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from ..services import Services, build_services


@lru_cache
def get_services() -> Services:
    return build_services()


ServicesDep = Annotated[Services, Depends(get_services)]
