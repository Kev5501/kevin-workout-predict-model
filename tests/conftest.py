import copy
import os
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
# The repository's config, however the package is installed
os.environ['WML_CONFIG_PATH'] = str(Path(__file__).resolve().parents[1] / 'configs' / 'pipeline.yaml')

from factories import LAST_DAY, FakeLauncher, FakeNotifier, FakeSheets, make_history  # noqa: E402

from workout_ml.config import Settings  # noqa: E402
from workout_ml.core.search import SearchConfig  # noqa: E402
from workout_ml.io.documents import LocalStore  # noqa: E402
from workout_ml.io.registry import LocalArtifacts, ModelRegistry  # noqa: E402
from workout_ml.services import Services  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in [k for k in os.environ if k.startswith('WML_') and k != 'WML_CONFIG_PATH']:
        monkeypatch.delenv(k)


@pytest.fixture
def settings():
    s = Settings()
    s.training.tuning.n_trials = 2
    s.training.tuning.trial_jobs = 1
    s.training.tuning.n_threads = 2
    s.training.promotion.eval_workouts = 10
    return s


@pytest.fixture
def fast_search(monkeypatch):
    """Small grids so a full sweep takes seconds."""
    import workout_ml.training.train as train

    def small(**kw):
        return SearchConfig(**kw, n_splits=2, perm_repeats=1, k_grid=[0, 5, None])
    monkeypatch.setattr(train, 'SearchConfig', small)


def build(settings, root, history=None, today=LAST_DAY) -> Services:
    docs = LocalStore()
    svc = Services(settings, docs, ModelRegistry(LocalArtifacts(root / 'artifacts'), docs),
                   FakeSheets(make_history() if history is None else history), FakeNotifier(), FakeLauncher())
    svc.today = lambda: today
    return svc


@pytest.fixture
def svc(settings, tmp_path):
    return build(settings, tmp_path)


@pytest.fixture(scope='session')
def _trained_state(tmp_path_factory):
    """One trained champion, reused by every test that needs a model."""
    from workout_ml.training.run import run_training

    s = Settings()
    s.training.tuning.n_trials, s.training.tuning.trial_jobs, s.training.tuning.n_threads = 2, 1, 2
    s.training.promotion.eval_workouts = 10
    root = tmp_path_factory.mktemp('champion')
    svc = build(s, root)
    import workout_ml.training.train as train
    original = train.SearchConfig
    train.SearchConfig = lambda **kw: original(**kw, n_splits=2, perm_repeats=1, k_grid=[0, 5, None])
    try:
        record = run_training(svc, 'full', 'initial', 'test')
    finally:
        train.SearchConfig = original
    assert record['status'] == 'promoted', record
    return svc, root


@pytest.fixture
def trained(_trained_state, settings):
    """Fresh documents per test (same artifacts), so tests can't affect each other."""
    base, root = _trained_state
    svc = build(settings, root)
    svc.docs._data = copy.deepcopy(base.docs._data)
    svc.registry = ModelRegistry(LocalArtifacts(root / 'artifacts'), svc.docs)
    return svc


@pytest.fixture
def monday():
    return date(2026, 11, 23)
