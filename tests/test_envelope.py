import json
import logging
from collections.abc import AsyncIterator

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_ai_app_factory import AgentError, AppDeps, AuthProvider, Feature
from pydantic_ai_app_factory.app import create_agent, create_agent_app
from tests._support import auth, make_client, submit


async def _boom_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    raise ModelHTTPError(status_code=503, model_name='secret-model')
    yield ''  # unreachable; only marks this as an async generator


async def _boom_mid_tool_call(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
    yield {0: DeltaToolCall(name='search', json_args='{"q": ')}
    raise ModelHTTPError(status_code=503, model_name='secret-model')


def _boom_http(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise ModelHTTPError(status_code=503, model_name='secret-model')


def _reject_key(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise ModelHTTPError(status_code=401, model_name='secret-model', body={'error': 'invalid_api_key'})


async def _reject_key_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    raise ModelHTTPError(status_code=401, model_name='secret-model', body={'error': 'invalid_api_key'})
    yield ''  # unreachable; only marks this as an async generator


def _boom_value(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise ValueError('secret dsn postgres://u:p@h/db')


async def _boom_value_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    raise ValueError('secret dsn postgres://u:p@h/db')
    yield ''  # unreachable; only marks this as an async generator


# --- error rendering ---


def test_run_maps_a_transient_error_to_503() -> None:
    resp = make_client(agent=Agent(FunctionModel(_boom_http), deps_type=AppDeps)).post('/run', json={'prompt': 'hi'})
    assert resp.status_code == 503
    assert resp.json()['code'] == 'model-unavailable'
    assert 'secret-model' not in resp.text


def test_run_maps_an_unknown_error_to_500_without_leaking() -> None:
    resp = make_client(agent=Agent(FunctionModel(_boom_value), deps_type=AppDeps)).post('/run', json={'prompt': 'hi'})
    assert resp.status_code == 500
    assert resp.json()['code'] == 'internal'
    assert 'postgres' not in resp.text


def test_chat_renders_a_safe_error_chunk_without_leaking() -> None:
    agent = Agent(FunctionModel(stream_function=_boom_stream), deps_type=AppDeps)
    resp = make_client(agent=agent).post('/chat', json=submit('hi'))
    assert resp.status_code == 200
    assert '"code":"model-unavailable"' in resp.text
    assert 'secret-model' not in resp.text


def test_chat_data_error_chunk_carries_the_safe_message() -> None:
    # The machine-readable chunk must stand alone: a client that keys on `data-error` should not have
    # to reassemble the text from the separate `ErrorChunk` to render (or re-render) the failure.
    agent = Agent(FunctionModel(stream_function=_boom_stream), deps_type=AppDeps)
    resp = make_client(agent=agent).post('/chat', json=submit('hi'))
    chunk = next(line for line in resp.text.splitlines() if '"type":"data-error"' in line)
    assert json.loads(chunk.removeprefix('data: '))['data'] == {
        'code': 'model-unavailable',
        'message': 'The model is temporarily unavailable.',
        'retryable': True,
    }


def test_chat_error_stream_still_reports_an_error_finish_reason() -> None:
    # The base `on_error` sets the error finish reason (and flushes half-streamed tool inputs);
    # replacing it wholesale would end the stream looking like a clean completion to the client.
    agent = Agent(FunctionModel(stream_function=_boom_stream), deps_type=AppDeps)
    resp = make_client(agent=agent).post('/chat', json=submit('hi'))
    assert '"finishReason":"error"' in resp.text


def test_chat_error_still_announces_a_half_streamed_tool_call() -> None:
    # The base `on_error` flushes tool calls whose input was streamed but never announced; without
    # it a frontend is left waiting in `input-streaming` forever.
    agent = Agent(FunctionModel(stream_function=_boom_mid_tool_call), deps_type=AppDeps)
    resp = make_client(agent=agent).post('/chat', json=submit('hi'))
    assert 'tool-input-available' in resp.text
    assert '"code":"model-unavailable"' in resp.text
    assert 'secret-model' not in resp.text


def test_run_reports_a_rejected_provider_key_as_model_unauthorized() -> None:
    response = make_client(agent=create_agent(FunctionModel(_reject_key))).post('/run', json={'prompt': 'hi'})
    assert response.status_code == 502
    assert response.json() == {
        'code': 'model-unauthorized',
        'message': 'The model provider rejected the credentials.',
        'retryable': False,
    }
    assert 'secret-model' not in response.text


def test_chat_reports_a_rejected_provider_key_as_model_unauthorized() -> None:
    agent = create_agent(FunctionModel(stream_function=_reject_key_stream))
    response = make_client(agent=agent).post('/chat', json=submit('hi'))
    assert response.status_code == 200
    assert '"code":"model-unauthorized"' in response.text
    assert 'secret-model' not in response.text


def test_a_fallback_chain_of_rejected_keys_is_model_unauthorized_not_retryable() -> None:
    model = FallbackModel(FunctionModel(_reject_key), FunctionModel(_reject_key))
    response = make_client(agent=create_agent(model)).post('/run', json={'prompt': 'hi'})
    assert response.status_code == 502
    assert response.json()['code'] == 'model-unauthorized'
    assert response.json()['retryable'] is False


def test_a_feature_route_error_uses_the_status_the_raiser_chose() -> None:
    def _missing() -> None:
        raise AgentError('storage/not-found', 'No such conversation.', http_status=404)

    class _Missing(Feature):
        name = 'missing'

        def routers(self, *, auth: AuthProvider) -> list[APIRouter]:
            router = APIRouter()
            router.add_api_route('/gone', _missing, methods=['GET'])
            return [router]

    app = create_agent_app(create_agent(TestModel()), auth=auth, features=[_Missing()])
    resp = TestClient(app, raise_server_exceptions=False).get('/gone')

    assert resp.status_code == 404
    assert resp.json()['code'] == 'storage/not-found'


def test_a_feature_route_raising_an_agent_error_gets_the_wire_contract() -> None:
    def _explode() -> None:
        raise AgentError('storage/not-found', 'No such conversation.')

    class _Exploding(Feature):
        name = 'exploding'

        def routers(self, *, auth: AuthProvider) -> list[APIRouter]:
            router = APIRouter()
            router.add_api_route('/boom', _explode, methods=['GET'])
            return [router]

    app = create_agent_app(create_agent(TestModel()), auth=auth, features=[_Exploding()])
    resp = TestClient(app, raise_server_exceptions=False).get('/boom')

    assert resp.status_code == 500
    assert resp.json() == {'code': 'storage/not-found', 'message': 'No such conversation.', 'retryable': False}


def test_retryable_rides_the_json_body_when_the_status_no_longer_implies_it() -> None:
    def _throttled() -> None:
        raise AgentError('quota/rate-limited', 'Too many requests.', retryable=True, http_status=429)

    class _Throttling(Feature):
        name = 'throttling'

        def routers(self, *, auth: AuthProvider) -> list[APIRouter]:
            router = APIRouter()
            router.add_api_route('/throttled', _throttled, methods=['GET'])
            return [router]

    app = create_agent_app(create_agent(TestModel()), auth=auth, features=[_Throttling()])
    resp = TestClient(app, raise_server_exceptions=False).get('/throttled')

    assert resp.status_code == 429
    assert resp.json()['retryable'] is True


def test_raising_a_fixed_handler_does_not_attach_the_request_to_the_shared_instance() -> None:
    # The handler instance outlives the request; the raise must not hang the hidden cause on it.
    fixed = AgentError('input/bad', 'Bad input.', http_status=422)
    client = make_client(
        agent=Agent(FunctionModel(_boom_value), deps_type=AppDeps), error_handlers=[(ValueError, fixed)]
    )

    assert client.post('/run', json={'prompt': 'hi'}).status_code == 422
    assert fixed.__cause__ is None and fixed.__traceback__ is None


# --- logging ---


def test_run_logs_the_original_error_it_hides_from_the_caller(caplog: pytest.LogCaptureFixture) -> None:
    client = make_client(agent=Agent(FunctionModel(_boom_value), deps_type=AppDeps))
    with caplog.at_level(logging.ERROR):
        client.post('/run', json={'prompt': 'hi'})
    assert 'postgres' in caplog.text  # the detail withheld from the client must reach the log


def test_run_logs_the_cause_of_a_transient_failure_it_hides_from_the_caller(caplog: pytest.LogCaptureFixture) -> None:
    # The retryable path is WARNING, not ERROR -- but it still carries the cause, so the model detail
    # withheld from the caller reaches the log (the README's "full detail goes to the log" promise).
    client = make_client(agent=Agent(FunctionModel(_boom_http), deps_type=AppDeps))
    with caplog.at_level(logging.WARNING):
        resp = client.post('/run', json={'prompt': 'hi'})
    assert resp.status_code == 503
    assert 'secret-model' not in resp.text
    assert 'secret-model' in caplog.text


def test_chat_logs_an_unexpected_fault_at_error_with_a_traceback(caplog: pytest.LogCaptureFixture) -> None:
    agent = Agent(FunctionModel(stream_function=_boom_value_stream), deps_type=AppDeps)
    with caplog.at_level(logging.INFO):
        resp = make_client(agent=agent).post('/chat', json=submit('hi'))
    assert '"code":"internal"' in resp.text
    assert 'postgres' not in resp.text  # never leaks to the client
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert error_records and 'postgres' in caplog.text  # a real 500 stays a loud, traced ERROR


def test_chat_logs_an_expected_error_at_info_without_a_traceback(caplog: pytest.LogCaptureFixture) -> None:
    # A stream error the product classifies as a 4xx is an expected outcome: recorded at INFO, no
    # traceback, and the withheld detail never reaches the log either.
    agent = Agent(FunctionModel(stream_function=_boom_value_stream), deps_type=AppDeps)
    handlers = [(ValueError, AgentError('input/bad', 'Bad input.', http_status=422))]
    with caplog.at_level(logging.INFO):
        resp = make_client(agent=agent, error_handlers=handlers).post('/chat', json=submit('hi'))
    assert '"code":"input/bad"' in resp.text
    assert [r.levelno for r in caplog.records if 'input/bad' in r.getMessage()] == [logging.INFO]
    assert 'postgres' not in caplog.text  # no exc_info, so the raw exception detail is not captured
