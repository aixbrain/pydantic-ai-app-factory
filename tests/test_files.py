"""The files a client sends: which media types the app admits, and how a run refusing one is reported."""

import base64
import logging
from typing import Any

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from pydantic import TypeAdapter
from pydantic_ai import Agent
from pydantic_ai.messages import (
    BinaryContent,
    DocumentUrl,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserContent,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_ai_app_factory import AppDeps, UserContext
from pydantic_ai_app_factory.app import create_agent, create_agent_app
from tests._support import auth, make_client


def _content(*items: Any) -> list[Any]:
    """The prompt's content as a JSON client puts it on the wire."""
    return TypeAdapter(list[UserContent]).dump_python(list(items), mode='json')


def _wire(*messages: ModelMessage) -> list[dict[str, Any]]:
    """The history as a JSON client puts it on the wire."""
    return ModelMessagesTypeAdapter.dump_python(list(messages), mode='json')


# --- /run ---


def test_run_rejects_a_file_of_a_media_type_pydantic_ai_does_not_classify_with_415() -> None:
    prompt = _content('what is this?', BinaryContent(data=b'select 1', media_type='application/sql'))

    resp = make_client().post('/run', json={'prompt': prompt})

    assert resp.status_code == 415
    assert resp.json() == {
        'code': 'file-unsupported',
        'message': 'Files of type application/sql are not accepted.',
        'retryable': False,
    }


def test_run_rejects_a_file_url_of_a_media_type_pydantic_ai_does_not_classify_with_415() -> None:
    # The type is inferred from the extension, as the earlier 422 check requires; the class it lands
    # in is the same question for a URL as for inline data.
    prompt = _content('what is this?', DocumentUrl(url='https://example.com/schema.sql'))

    resp = make_client().post('/run', json={'prompt': prompt})

    assert resp.status_code == 415
    assert resp.json()['message'] == 'Files of type application/sql are not accepted.'


def test_run_names_each_unsupported_media_type_once() -> None:
    prompt = _content(
        BinaryContent(data=b'select 1', media_type='application/sql'),
        BinaryContent(data=b'print()', media_type='text/x-python'),
        BinaryContent(data=b'select 2', media_type='application/sql'),
    )

    resp = make_client().post('/run', json={'prompt': prompt})

    assert resp.json()['message'] == 'Files of type application/sql or text/x-python are not accepted.'


def test_run_rejects_an_unsupported_file_in_the_history_with_415() -> None:
    history = _wire(
        ModelRequest(parts=[UserPromptPart(content=['look', BinaryContent(data=b'x', media_type='application/sql')])]),
        ModelResponse(parts=[TextPart(content='ok')]),
    )

    resp = make_client().post('/run', json={'prompt': 'again', 'message_history': history})

    assert resp.status_code == 415
    assert resp.json()['code'] == 'file-unsupported'


def test_run_checks_the_prompt_after_sanitizing_it() -> None:
    # A disallowed scheme is stripped before the check, so a file that never reaches the run is not
    # what the request is refused for.
    prompt = _content('what is this?', DocumentUrl(url='s3://bucket/schema.sql'))

    with pytest.warns(UserWarning, match='file URLs with scheme'):
        resp = make_client().post('/run', json={'prompt': prompt})

    assert resp.status_code == 200


def test_run_refuses_an_unsupported_file_before_the_deps_builder_runs(caplog: pytest.LogCaptureFixture) -> None:
    built: list[UserContext] = []

    def builder(request: Request, user: UserContext) -> AppDeps:
        built.append(user)
        return AppDeps(user=user)

    app = create_agent_app(create_agent(TestModel()), auth=auth, deps=builder)
    prompt = _content(BinaryContent(data=b'x', media_type='application/sql'))
    with caplog.at_level(logging.INFO):
        resp = TestClient(app).post('/run', json={'prompt': prompt})

    assert resp.status_code == 415
    assert built == []
    # A refused request is the expected 4xx outcome, not a fault of the service.
    assert [r.levelno for r in caplog.records if 'file-unsupported' in r.getMessage()] == [logging.INFO]


# --- /chat ---


def _file(media_type: str, data: bytes = b'x') -> dict[str, Any]:
    """A Vercel file part carrying inline data, as a browser upload arrives."""
    encoded = base64.b64encode(data).decode()
    return {'type': 'file', 'mediaType': media_type, 'url': f'data:{media_type};base64,{encoded}'}


def _user(*parts: dict[str, Any], id: str = 'm1') -> dict[str, Any]:
    return {'id': id, 'role': 'user', 'parts': list(parts)}


def _chat(*messages: dict[str, Any]) -> dict[str, Any]:
    return {'trigger': 'submit-message', 'id': 'c1', 'messages': list(messages)}


def test_chat_rejects_an_unsupported_file_as_json_before_the_stream_starts() -> None:
    body = _chat(_user({'type': 'text', 'text': 'what is this?'}, _file('application/sql')))

    resp = make_client().post('/chat', json=body)

    assert resp.status_code == 415
    assert resp.json() == {
        'code': 'file-unsupported',
        'message': 'Files of type application/sql are not accepted.',
        'retryable': False,
    }


def test_chat_rejects_an_unsupported_file_in_an_earlier_message() -> None:
    # The protocol resends the whole conversation; a file anywhere in it reaches the model again.
    body = _chat(
        _user({'type': 'text', 'text': 'look'}, _file('application/sql'), id='m1'),
        {'id': 'm2', 'role': 'assistant', 'parts': [{'type': 'text', 'text': 'ok'}]},
        _user({'type': 'text', 'text': 'again'}, id='m3'),
    )

    resp = make_client().post('/chat', json=body)

    assert resp.status_code == 415


def test_chat_refuses_an_unsupported_file_before_the_deps_builder_runs() -> None:
    built: list[UserContext] = []

    def builder(request: Request, user: UserContext) -> AppDeps:
        built.append(user)
        return AppDeps(user=user)

    app = create_agent_app(create_agent(TestModel()), auth=auth, deps=builder)
    resp = TestClient(app).post('/chat', json=_chat(_user(_file('application/sql'))))

    assert resp.status_code == 415
    assert built == []


# --- a model class that does not support the file ---


def _unsupporting_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    # What a model class raises while building its request for a content class it has no mapping
    # for; the same type it raises for missing streaming, token counting or compaction.
    raise NotImplementedError('AudioUrl is not supported in Anthropic user prompts')


def test_a_model_class_that_does_not_support_an_accepted_file_stays_internal() -> None:
    # pydantic-ai gives this case no type of its own, so the factory does not infer one from the
    # presence of a file. A product that needs it named maps NotImplementedError in error_handlers.
    prompt = _content('transcribe', BinaryContent(data=b'\xff\xfb', media_type='audio/mpeg'))
    agent: Agent[AppDeps, str] = Agent(FunctionModel(_unsupporting_model), deps_type=AppDeps)

    resp = make_client(agent=agent).post('/run', json={'prompt': prompt})

    assert resp.status_code == 500
    assert resp.json()['code'] == 'internal'
