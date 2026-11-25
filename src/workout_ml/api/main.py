"""FastAPI application. In the cloud it is private: Cloud Run IAM checks every caller."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .. import __version__
from ..errors import Busy
from ..inference.history import UnsafeHistory
from ..inference.predict import UnsafePredictions
from ..io.registry import ModelNotFound
from .routers import health, models, predictions, training

ERROR_STATUS = {UnsafeHistory: 422, UnsafePredictions: 422, ModelNotFound: 404, Busy: 409}


def create_app() -> FastAPI:
    app = FastAPI(title='Workout ML', version=__version__,
                  description='Batch strength predictions and continuous training')
    for router in (health.router, predictions.router, training.router, models.router):
        app.include_router(router)

    async def known_error(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({'error': type(exc).__name__, 'detail': str(exc)}, ERROR_STATUS[type(exc)])

    for exc_type in ERROR_STATUS:
        app.add_exception_handler(exc_type, known_error)
    return app


app = create_app()
