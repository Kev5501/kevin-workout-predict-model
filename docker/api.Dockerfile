# API image: batch inference, model registry, training triggers. No training libraries.
FROM python:3.13-slim AS build
WORKDIR /build
RUN python -m venv /venv
ENV PATH=/venv/bin:$PATH PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements/api.txt .
RUN pip install -r api.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

FROM python:3.13-slim
# libgomp1: OpenMP runtime used by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app
COPY --from=build /venv /venv
COPY configs /app/configs
WORKDIR /app
ARG GIT_SHA=local
ENV PATH=/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    WML_CONFIG_PATH=/app/configs/pipeline.yaml GIT_SHA=$GIT_SHA PORT=8080
USER app
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn workout_ml.api.main:app --host 0.0.0.0 --port ${PORT} --no-access-log"]
