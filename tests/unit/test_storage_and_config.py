from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from factories import make_history
from pydantic import ValidationError
from sklearn.dummy import DummyRegressor

from workout_ml.config import AlertSettings, AppSettings, Settings, SheetsSettings
from workout_ml.core.reps import build_ratio_table, e1rm_to_weight
from workout_ml.io.documents import LocalStore, Write
from workout_ml.io.notify import LogNotifier, build_notifier
from workout_ml.io.registry import LocalArtifacts, ModelBundle, ModelNotFound, ModelRegistry, Recipe
from workout_ml.io.sheets import GoogleSheets


# ------------------------------------------------------------------ config
def test_defaults_and_env_overrides(monkeypatch):
    s = Settings()
    assert s.schedule['Tuesday'] == 'Push' and s.schedule['Friday'] == 'Rest'
    monkeypatch.setenv('WML_TRAINING__PERFORMANCE__MAX_MAPE', '4.5')
    monkeypatch.setenv('WML_INFERENCE__LOOKBACK_DAYS', '45')
    monkeypatch.setenv('WML_SCHEDULE', '{"Friday": "Arms"}')
    s = Settings()
    assert s.training.performance.max_mape == 4.5 and s.training.performance.cooldown_days == 7
    assert s.inference.lookback_days == 45
    assert s.schedule['Friday'] == 'Arms' and s.schedule['Tuesday'] == 'Push'


def test_invalid_config_is_rejected(monkeypatch):
    with pytest.raises(ValidationError):
        Settings(schedule={'Funday': 'Push'})
    with pytest.raises(ValidationError):
        SheetsSettings(source_tab='Data', output_tab='data ')
    with pytest.raises(ValidationError):
        Settings(app=AppSettings(backend='gcp'))
    with pytest.raises(ValidationError):
        Settings(alerts=AlertSettings(enabled=True))
    monkeypatch.setenv('WML_CONFIG_PATH', 'does/not/exist.yaml')
    with pytest.raises(FileNotFoundError):
        Settings()


def test_alerts_off_by_default():
    assert isinstance(build_notifier(Settings().alerts), LogNotifier)


# ------------------------------------------------------------------ documents
def test_commit_is_atomic_and_persists(tmp_path):
    path = tmp_path / 'docs.json'
    s = LocalStore(path)
    s.set('c', 'a', {'x': 1})
    with pytest.raises(TypeError):
        s.commit([Write('delete', 'c', 'a'), Write('set', 'c', 'b', {'bad': object()})])
    assert s.get('c', 'a') == {'x': 1} and s.get('c', 'b') is None
    assert LocalStore(path).get('c', 'a') == {'x': 1}


def test_lock_excludes_and_expires():
    s, now = LocalStore(), datetime(2026, 1, 1, tzinfo=UTC)
    ttl = timedelta(minutes=10)
    assert s.acquire_lock('job', 'one', ttl, now)
    assert not s.acquire_lock('job', 'two', ttl, now + timedelta(minutes=9))
    assert s.acquire_lock('job', 'two', ttl, now + timedelta(minutes=11))
    s.release_lock('job', 'one')                       # not the owner any more: ignored
    assert s.get('locks', 'job')['owner'] == 'two'


# ------------------------------------------------------------------ registry
def bundle(v, value=1.0):
    return ModelBundle(v, DummyRegressor(strategy='constant', constant=value).fit([[0]], [value]),
                       Recipe('lightgbm', ['f'], {'p': 1}), ['f'], {'metrics': ['avg']},
                       pd.DataFrame({'effective_reps': [5, 10], 'ratio': [.86, .74]}), {'trigger': 't'})


def test_register_promote_rollback(tmp_path):
    docs = LocalStore()
    reg = ModelRegistry(LocalArtifacts(tmp_path), docs)
    with pytest.raises(ModelNotFound):
        reg.load_champion()
    for v in ('v1', 'v2'):
        reg.register(bundle(v, float(v[1])), 'candidate')
        assert reg.champion_version() != v              # registering never makes a champion
        reg.promote(v)
    assert reg.load_champion().version == 'v2'
    assert {m['version']: m['status'] for m in reg.list()} == {'v1': 'archived', 'v2': 'champion'}
    reg.promote('v1')                                   # rollback
    loaded = reg.load_champion()
    assert loaded.version == 'v1' and loaded.model.predict([[0]])[0] == 1.0
    assert loaded.recipe == Recipe('lightgbm', ['f'], {'p': 1})
    assert {m['version']: m['status'] for m in reg.list()} == {'v1': 'champion', 'v2': 'archived'}
    with pytest.raises(ModelNotFound):
        reg.promote('v9')


def test_versions_are_never_overwritten(tmp_path):
    reg = ModelRegistry(LocalArtifacts(tmp_path), LocalStore())
    reg.register(bundle('v1'), 'rejected')
    with pytest.raises(FileExistsError):
        reg.register(bundle('v1'), 'candidate')


# ------------------------------------------------------------------ weights
def test_weights_from_ratio_table():
    table = build_ratio_table(make_history(20))
    w = e1rm_to_weight(300.0, [5, 8, 10], 10, table=table, round_to=2.5)
    assert w.tolist() == [260.0, 235.0, 222.5]          # 258.9, 235.8, 221.7 to the nearest 2.5
    assert (pd.Series(e1rm_to_weight(300, [5, 6, 7, 8, 9, 10], 10, table=table)).diff().dropna() < 0).all()


# ------------------------------------------------------------------ google sheets (fake client)
class FakeWorksheet:
    def __init__(self, rows, cols):
        self.row_count, self.col_count, self.values = rows, cols, None

    def resize(self, rows, cols):
        self.row_count, self.col_count = rows, cols

    def update(self, values, cell, value_input_option):
        assert cell == 'A1' and value_input_option == 'RAW'
        self.values = values


class FakeBook:
    def __init__(self, ws=None):
        self.tabs = {'Predictions': ws} if ws else {}

    def worksheet(self, name):
        import gspread
        if name not in self.tabs:
            raise gspread.WorksheetNotFound(name)
        return self.tabs[name]

    def add_worksheet(self, name, rows, cols):
        self.tabs[name] = FakeWorksheet(rows, cols)
        return self.tabs[name]


def gateway(book):
    client = type('C', (), {'open_by_key': lambda self, _: book})()
    return GoogleSheets('id', 'Enriched_Data', 'Predictions', client=client)


def test_sheet_write_blanks_leftover_rows():
    ws = FakeWorksheet(5, 2)
    gateway(FakeBook(ws)).write_predictions(['a', 'b'], [[1, 2]])
    assert ws.values == [['a', 'b'], [1, 2], ['', ''], ['', ''], ['', '']]


def test_sheet_is_created_and_grown():
    book = FakeBook()
    gateway(book).write_predictions(['a'], [[i] for i in range(150)])
    assert len(book.tabs['Predictions'].values) == 151


def test_sheet_never_targets_the_source_tab():
    with pytest.raises(ValueError):
        GoogleSheets('id', 'Enriched_Data', 'enriched_data')
