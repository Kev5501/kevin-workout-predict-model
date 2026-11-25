"""Re-exports the shared models (saved ``.joblib`` files refer to ``modeling.models``) and
registers SARIMA, which is research-only."""
from workout_ml.core.models import *  # noqa: F401,F403
from workout_ml.core.models import REGISTRY, ModelSpec


def _build_sarima(params, ctx):
    from .sarima import SarimaPerSeries
    cfg = ctx.cfg
    return SarimaPerSeries(
        pq=tuple(cfg.sarima_pq) if cfg else (0, 1, 2),
        d=tuple(cfg.sarima_d) if cfg else (0, 1),
        seasonal_m=cfg.sarima_seasonal_m if cfg else 4,
        min_obs=cfg.sarima_min_obs if cfg else 20,
        n_jobs=ctx.n_threads)


REGISTRY.setdefault('sarima', ModelSpec('sarima', 'sarima', None, None, _build_sarima))
