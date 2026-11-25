"""Model versions: immutable files in an artifact store, status and champion pointer in documents.

``models/<version>/`` holds ``model.joblib``, ``metadata.json`` and the training data snapshot.
"""
from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

import joblib
import pandas as pd

from .documents import DocumentStore, Write

log = logging.getLogger(__name__)
TRACKED_LIBRARIES = ['lightgbm', 'scikit-learn', 'pandas', 'numpy', 'joblib']
STATUSES = ('candidate', 'champion', 'archived', 'rejected')


class ModelNotFound(LookupError):
    pass


class ArtifactStore(Protocol):
    def write(self, path: str, data: bytes) -> None: ...   # must refuse to overwrite
    def read(self, path: str) -> bytes: ...


class LocalArtifacts:
    def __init__(self, root: Path):
        self._root = Path(root)

    def write(self, path, data):
        target = self._root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, 'xb') as f:            # x: fail if it exists
            f.write(data)

    def read(self, path):
        return (self._root / path).read_bytes()


class GcsArtifacts:
    def __init__(self, bucket: str, project: str, client=None):
        if client is None:
            from google.cloud import storage
            client = storage.Client(project=project)
        self._bucket = client.bucket(bucket)

    def write(self, path, data):
        self._bucket.blob(path).upload_from_string(data, if_generation_match=0)   # fail if it exists

    def read(self, path):
        return self._bucket.blob(path).download_as_bytes()


def library_versions() -> dict[str, str]:
    out = {}
    for name in TRACKED_LIBRARIES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = 'missing'
    return out


@dataclass(frozen=True)
class Recipe:
    """Everything needed to retrain the same model on different data."""

    model_name: str
    features: list[str]
    params: dict[str, Any]

    def to_dict(self) -> dict:
        return dict(model_name=self.model_name, features=list(self.features), params=dict(self.params))


@dataclass
class ModelBundle:
    version: str
    model: Any
    recipe: Recipe
    candidates: list[str]                 # columns of the model frame the model was trained on
    feature_settings: dict
    ratio_table: pd.DataFrame
    metadata: dict = field(default_factory=dict)


class ModelRegistry:
    def __init__(self, artifacts: ArtifactStore, documents: DocumentStore):
        self._artifacts = artifacts
        self._docs = documents
        self._cache: ModelBundle | None = None

    # ---------------------------------------------------------------- write
    def register(self, bundle: ModelBundle, status: str, snapshot: bytes | None = None) -> None:
        assert status in ('candidate', 'rejected'), status      # only promote() makes a champion
        buf = io.BytesIO()
        joblib.dump(bundle.model, buf, compress=3)
        meta = {**bundle.metadata, 'version': bundle.version, 'recipe': bundle.recipe.to_dict(),
                'candidates': bundle.candidates, 'feature_settings': bundle.feature_settings,
                'ratio_table': bundle.ratio_table.to_dict(orient='list'), 'libraries': library_versions()}
        base = f'models/{bundle.version}'
        self._artifacts.write(f'{base}/model.joblib', buf.getvalue())
        if snapshot is not None:
            self._artifacts.write(f'{base}/training_data.parquet', snapshot)
        # metadata last: a version without it was never fully written
        self._artifacts.write(f'{base}/metadata.json', json.dumps(meta, indent=1, default=str).encode())
        summary = {k: bundle.metadata.get(k) for k in ('trigger', 'mode', 'gate', 'data_range', 'rows', 'git_sha')}
        self._docs.set('models', bundle.version, {
            'version': bundle.version, 'status': status, 'created': datetime.now(UTC).isoformat(),
            'model_name': bundle.recipe.model_name, 'features': list(bundle.recipe.features), **summary})

    def promote(self, version_: str) -> dict:
        """Make ``version_`` the champion (also used to roll back). Archives the previous champion."""
        doc = self._docs.get('models', version_)
        if doc is None:
            raise ModelNotFound(version_)
        now = datetime.now(UTC).isoformat()
        writes = []
        previous = self.champion_version()
        if previous and previous != version_:
            old = self._docs.get('models', previous)
            if old:
                writes.append(Write('set', 'models', previous, {**old, 'status': 'archived'}))
        doc = {**doc, 'status': 'champion', 'promoted': now}
        writes += [Write('set', 'models', version_, doc),
                   Write('set', 'state', 'champion', {'version': version_, 'since': now, 'previous': previous})]
        self._docs.commit(writes)
        log.info('promoted model', extra={'version': version_, 'previous': previous})
        return doc

    # ---------------------------------------------------------------- read
    def champion_version(self) -> str | None:
        state = self._docs.get('state', 'champion')
        return state['version'] if state else None

    def list(self) -> list[dict]:
        return sorted(self._docs.list('models'), key=lambda d: d['version'], reverse=True)

    def load(self, version_: str) -> ModelBundle:
        if self._cache and self._cache.version == version_:
            return self._cache
        if self._docs.get('models', version_) is None:
            raise ModelNotFound(version_)
        base = f'models/{version_}'
        meta = json.loads(self._artifacts.read(f'{base}/metadata.json'))
        here = library_versions()
        diff = {k: (v, here.get(k)) for k, v in meta['libraries'].items() if here.get(k) != v}
        if diff:
            log.warning('library versions differ from training', extra={'model_version': version_, 'diff': diff})
        model = joblib.load(io.BytesIO(self._artifacts.read(f'{base}/model.joblib')))
        self._cache = ModelBundle(
            version=version_, model=model, recipe=Recipe(**meta['recipe']), candidates=meta['candidates'],
            feature_settings=meta['feature_settings'], ratio_table=pd.DataFrame(meta['ratio_table']), metadata=meta)
        return self._cache

    def load_champion(self) -> ModelBundle:
        v = self.champion_version()
        if v is None:
            raise ModelNotFound('no champion model: run the trainer with mode=full first')
        return self.load(v)
