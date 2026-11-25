"""Document storage: one small interface with a Firestore and a local-file implementation.

All reads and writes of predictions, models, runs and state go through :class:`DocumentStore`, so
every business rule is tested against the local version and runs unchanged on Firestore.
"""
from __future__ import annotations

import copy
import json
import os
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol

Doc = dict[str, Any]


@dataclass(frozen=True)
class Write:
    """One operation of an atomic commit."""

    op: Literal['set', 'delete']
    collection: str
    doc_id: str
    data: Doc | None = None


class DocumentStore(Protocol):
    def get(self, collection: str, doc_id: str) -> Doc | None: ...
    def list(self, collection: str) -> list[Doc]: ...
    def set(self, collection: str, doc_id: str, data: Doc) -> None: ...
    def commit(self, writes: Iterable[Write]) -> None: ...
    def acquire_lock(self, name: str, owner: str, ttl: timedelta, now: datetime | None = None) -> bool: ...
    def release_lock(self, name: str, owner: str) -> None: ...


class LocalStore:
    """In memory, optionally saved to a JSON file (``path``). For tests and local demos."""

    def __init__(self, path: Path | None = None):
        self._path = Path(path) if path else None
        self._mutex = threading.RLock()
        self._data: dict[str, dict[str, Doc]] = {}
        if self._path and self._path.exists():
            self._data = json.loads(self._path.read_text(encoding='utf-8'))

    def _save(self) -> None:
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix('.tmp')
            tmp.write_text(json.dumps(self._data, indent=1), encoding='utf-8')
            os.replace(tmp, self._path)

    def get(self, collection, doc_id):
        with self._mutex:
            doc = self._data.get(collection, {}).get(doc_id)
            return copy.deepcopy(doc)

    def list(self, collection):
        with self._mutex:
            return [copy.deepcopy(d) for _, d in sorted(self._data.get(collection, {}).items())]

    def set(self, collection, doc_id, data):
        self.commit([Write('set', collection, doc_id, data)])

    def commit(self, writes):
        with self._mutex:
            new = copy.deepcopy(self._data)
            for w in writes:
                docs = new.setdefault(w.collection, {})
                if w.op == 'set':
                    docs[w.doc_id] = json.loads(json.dumps(w.data))     # same types Firestore accepts
                else:
                    docs.pop(w.doc_id, None)
            self._data = new
            self._save()

    def acquire_lock(self, name, owner, ttl, now=None):
        now = now or datetime.now(UTC)
        with self._mutex:
            held = self.get('locks', name)
            if held and datetime.fromisoformat(held['expires']) > now and held['owner'] != owner:
                return False
            self.set('locks', name, {'owner': owner, 'expires': (now + ttl).isoformat()})
            return True

    def release_lock(self, name, owner):
        with self._mutex:
            held = self.get('locks', name)
            if held and held['owner'] == owner:
                self.commit([Write('delete', 'locks', name)])


class FirestoreStore:
    """Cloud Firestore (Native mode, default database)."""

    def __init__(self, project: str, client=None):
        if client is None:
            from google.cloud import firestore
            client = firestore.Client(project=project)
        self._db = client

    def get(self, collection, doc_id):
        snap = self._db.collection(collection).document(doc_id).get()
        return snap.to_dict() if snap.exists else None

    def list(self, collection):
        return [s.to_dict() for s in self._db.collection(collection).stream()]

    def set(self, collection, doc_id, data):
        self._db.collection(collection).document(doc_id).set(data)

    def commit(self, writes):
        batch = self._db.batch()              # atomic: all writes apply or none do
        for w in writes:
            ref = self._db.collection(w.collection).document(w.doc_id)
            if w.op == 'set':
                batch.set(ref, w.data)
            else:
                batch.delete(ref)
        batch.commit()

    def acquire_lock(self, name, owner, ttl, now=None):
        from google.cloud import firestore
        now = now or datetime.now(UTC)
        ref = self._db.collection('locks').document(name)

        @firestore.transactional
        def attempt(tx) -> bool:
            snap = ref.get(transaction=tx)
            held = snap.to_dict() if snap.exists else None
            if held and datetime.fromisoformat(held['expires']) > now and held['owner'] != owner:
                return False
            tx.set(ref, {'owner': owner, 'expires': (now + ttl).isoformat()})
            return True
        return attempt(self._db.transaction())

    def release_lock(self, name, owner):
        held = self.get('locks', name)
        if held and held['owner'] == owner:
            self._db.collection('locks').document(name).delete()
