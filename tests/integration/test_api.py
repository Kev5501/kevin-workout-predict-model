import json
import logging

import pytest
from fastapi.testclient import TestClient

from workout_ml.api.deps import get_services
from workout_ml.api.main import create_app
from workout_ml.logs import JsonFormatter


def client_for(svc):
    app = create_app()
    app.dependency_overrides[get_services] = lambda: svc
    return TestClient(app)


@pytest.fixture
def client(trained):
    return client_for(trained)


def test_health_needs_nothing():
    assert TestClient(create_app()).get('/health').json()['status'] == 'ok'


def test_ready(client, svc):
    assert client.get('/ready').status_code == 200
    r = client_for(svc).get('/ready')
    assert r.status_code == 503 and r.json() == {'ready': False, 'champion': None}


def test_run_current_and_sync(client, trained):
    r = client.post('/v1/predictions/run', json={'run_date': '2026-11-23'})
    assert r.status_code == 200 and r.json()['n_predictions'] == 12
    current = client.get('/v1/predictions/current').json()
    assert [d['gym_day'] for d in current[:5]] == ['Push'] * 5
    trained.sheets.rows = []
    assert client.post('/v1/predictions/sync').json() == {'rows': 12} and len(trained.sheets.rows) == 12


def test_run_without_body_uses_today(client):
    assert client.post('/v1/predictions/run').json()['run_date'] == '2026-11-22'     # the fixture's "today"


def test_dry_run(client, trained):
    r = client.post('/v1/predictions/run', json={'run_date': '2026-11-27', 'dry_run': True}).json()
    assert [w['gym_day'] for w in r['workouts']] == ['Upper', 'Lower'] and len(r['predictions']) == 8
    assert client.get('/v1/predictions/current').json() == []


def test_preview_needs_no_model(svc):
    r = client_for(svc).get('/v1/schedule/preview', params={'run_date': '2026-11-25'}).json()
    assert [w['gym_day'] for w in r['workouts']] == ['Pull', 'Legs']
    assert {'exercise': 'Lat Pulldown', 'set_number': 1} in r['workouts'][0]['sets']


def test_known_errors_have_clear_statuses(client, svc, trained):
    assert client_for(svc).post('/v1/predictions/run').status_code == 404            # no champion
    trained.sheets.history = trained.sheets.history.iloc[0:0]
    r = client.post('/v1/predictions/run')
    assert r.status_code == 422 and 'empty' in r.json()['detail']
    assert client.post('/v1/models/nope/promote').status_code == 404
    assert client.post('/v1/predictions/run', json={'run_date': 'soon'}).status_code == 422


def test_busy_is_409(client, trained):
    from workout_ml.inference.batch import LOCK_TTL
    trained.docs.acquire_lock('batch', 'other', LOCK_TTL)
    assert client.post('/v1/predictions/run').status_code == 409


def test_check_launches_only_when_due(client, svc, trained):
    r = client.post('/v1/training/check').json()
    assert r['train'] is False and r['execution'] is None and trained.launcher.calls == []
    r = client_for(svc).post('/v1/training/check').json()
    assert r['trigger'] == 'initial' and r['execution'] == 'execution-1'
    assert svc.launcher.calls == [('full', 'initial', 'no champion model yet')]


def test_manual_training(client, trained):
    r = client.post('/v1/training/run', json={'mode': 'tune', 'reason': 'demo'})
    assert r.status_code == 202 and trained.launcher.calls == [('tune', 'manual', 'demo')]
    assert client.post('/v1/training/run', json={'mode': 'everything'}).status_code == 422
    assert client.get('/v1/training/runs').json()[0]['status'] == 'promoted'


def test_models_and_rollback(client, trained):
    from workout_ml.training.run import run_training
    first = trained.registry.champion_version()
    run_training(trained, 'refit', 'manual', 'test')
    listed = client.get('/v1/models').json()
    assert listed['champion'] != first and len(listed['models']) == 2
    assert client.post(f'/v1/models/{first}/promote').json()['status'] == 'champion'
    assert client.get('/v1/models').json()['champion'] == first


def test_logs_are_json():
    record = logging.LogRecord('x', logging.WARNING, '', 0, 'hello %s', ('you',), None)
    record.rows = 3
    out = json.loads(JsonFormatter().format(record))
    assert out['severity'] == 'WARNING' and out['message'] == 'hello you' and out['rows'] == 3
