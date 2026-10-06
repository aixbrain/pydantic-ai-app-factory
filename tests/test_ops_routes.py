import asyncio
import logging
from collections.abc import Mapping
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.models.test import TestModel

from pydantic_ai_app_factory import Feature
from pydantic_ai_app_factory.app import create_agent, create_agent_app
from tests._support import Storage, make_client, strict_auth


class _Healthy(Feature):
    name = 'healthy'

    async def health(self) -> Mapping[str, Any] | None:
        return {'backend': 'ok'}


class _Unhealthy(Feature):
    name = 'unhealthy'

    async def health(self) -> Mapping[str, Any] | None:
        raise RuntimeError('db down')


class _Configured(Feature):
    name = 'configured'

    def config(self) -> Mapping[str, Any] | None:
        return {'create_route': False}


class _AlsoConfigured(Feature):
    name = 'also-configured'

    def config(self) -> Mapping[str, Any] | None:
        return {'history_source': 'client'}


class _Mutable(Feature):
    name = 'mutable'

    def __init__(self) -> None:
        self.value = 1

    def config(self) -> Mapping[str, Any] | None:
        return {'value': self.value}


def test_healthz_lists_features() -> None:
    body = make_client(features=[Storage()]).get('/healthz').json()
    assert body['status'] == 'ok'
    assert 'storage' in body['features']


def test_config_exposes_the_composed_extras_schema() -> None:
    # Published under the wire name a browser client sends, not the feature's Python field name.
    body = make_client(features=[Storage()]).get('/config').json()
    assert 'storage' in body['features']
    assert 'caseId' in body['extras_schema']['properties']


def test_config_lists_the_media_types_a_file_may_have() -> None:
    # The same rule the endpoints refuse a file by, so a frontend can gate its picker on it.
    accepts = make_client().get('/config').json()['accepts']
    assert accepts == sorted(accepts)
    assert {'image/png', 'audio/mpeg', 'video/mp4', 'application/pdf'} <= set(accepts)
    assert 'application/sql' not in accepts


def test_feature_config_reports_each_features_config_under_its_name() -> None:
    body = make_client(features=[_Configured(), _AlsoConfigured()]).get('/config/features').json()
    assert body['feature_config'] == {
        'configured': {'create_route': False},
        'also-configured': {'history_source': 'client'},
    }


def test_a_feature_reporting_no_config_contributes_no_key() -> None:
    # `Storage` does not override `config()`; absent, not empty -- the same rule `health()` follows.
    body = make_client(features=[Storage(), _Configured()]).get('/config/features').json()
    assert body['feature_config'] == {'configured': {'create_route': False}}


def test_feature_config_sits_behind_the_apps_auth() -> None:
    app = create_agent_app(create_agent(TestModel()), auth=strict_auth, features=[_Configured()])
    client = TestClient(app)
    assert client.get('/config/features').status_code == 401
    assert client.get('/config/features', headers={'x-api-key': 'k'}).status_code == 200


def test_config_stays_public_and_carries_no_feature_supplied_content() -> None:
    # The invariant: `/config` is derived by the factory, so it cannot leak what a feature hands it.
    app = create_agent_app(create_agent(TestModel()), auth=strict_auth, features=[_Configured()])
    body = TestClient(app).get('/config').json()  # no credential
    assert body['features'] == ['configured']
    assert 'feature_config' not in body


def test_feature_config_is_not_cacheable() -> None:
    # It sits under the public `/config` prefix; a cache must never store this authenticated body.
    resp = make_client(features=[_Configured()]).get('/config/features')
    assert resp.headers['cache-control'] == 'no-store'


def test_config_is_read_per_request_so_it_can_change_at_runtime() -> None:
    feature = _Mutable()
    client = make_client(features=[feature])
    assert client.get('/config/features').json()['feature_config']['mutable'] == {'value': 1}
    feature.value = 2
    assert client.get('/config/features').json()['feature_config']['mutable'] == {'value': 2}


def test_readyz_is_ok_when_no_feature_reports() -> None:
    resp = make_client(features=[Storage()]).get('/readyz')
    assert resp.status_code == 200
    assert resp.json() == {'status': 'ok', 'reports': {}}


def test_readyz_includes_a_healthy_feature_report() -> None:
    resp = make_client(features=[_Healthy()]).get('/readyz')
    assert resp.status_code == 200
    assert resp.json()['reports'] == {'healthy': {'backend': 'ok'}}


def test_readyz_is_503_when_a_feature_is_unhealthy() -> None:
    resp = make_client(features=[_Healthy(), _Unhealthy()]).get('/readyz')
    assert resp.status_code == 503
    body = resp.json()
    assert body['failures'] == {'unhealthy': 'RuntimeError'}  # type only, no leaked message
    assert body['reports'] == {'healthy': {'backend': 'ok'}}


def test_readyz_times_out_a_hung_health_check_instead_of_hanging_the_worker() -> None:
    class _Hung(Feature):
        name = 'hung'

        async def health(self) -> Mapping[str, Any] | None:
            await asyncio.sleep(10)  # never resolves within the probe's budget

    resp = make_client(features=[_Healthy(), _Hung()], health_timeout=0.05).get('/readyz')
    assert resp.status_code == 503
    assert resp.json()['failures'] == {'hung': 'TimeoutError'}  # the hang is bounded and reported unhealthy
    assert resp.json()['reports'] == {'healthy': {'backend': 'ok'}}  # the sound check still reports


def test_readyz_logs_the_failure_it_reduces_to_a_type_name(caplog: pytest.LogCaptureFixture) -> None:
    client = make_client(features=[_Unhealthy()])
    with caplog.at_level(logging.ERROR):
        client.get('/readyz')
    assert 'db down' in caplog.text
