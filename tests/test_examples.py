"""The shipped examples build and serve, so a change to the package cannot break them silently."""

import json
from typing import Any

from fastapi.testclient import TestClient
from pydantic_ai import Agent

CHAT_BODY = {
    'id': 'c1',
    'trigger': 'submit-message',
    'messages': [{'id': 'm1', 'role': 'user', 'parts': [{'type': 'text', 'text': 'Hello'}]}],
}


def streamed_text(sse: str) -> str:
    """Join the `text-delta` chunks of a Vercel data stream into the text the client renders."""
    chunks: list[dict[str, Any]] = [json.loads(line[6:]) for line in sse.splitlines() if line.startswith('data: {')]
    return ''.join(chunk['delta'] for chunk in chunks if chunk['type'] == 'text-delta')


def test_quickstart_serves_the_factory_under_api() -> None:
    from pydantic_ai_app_factory_examples import quickstart

    try:
        with TestClient(quickstart.root) as client:
            assert client.get('/api/healthz').status_code == 200
            response = client.post('/api/chat', json=CHAT_BODY)
            assert response.status_code == 200
            assert streamed_text(response.text) == 'success (no tool calls)'
            assert client.post('/chat', json=CHAT_BODY).status_code == 404
            # The root app's own `/docs` list the prefixed routes.
            assert client.get('/docs').status_code == 200
            assert '/api/chat' in client.get('/openapi.json').json()['paths']
    finally:
        # The example instruments every agent; leave the other tests as they were.
        Agent.instrument_all(False)


def test_notes_feature_serves_its_routes() -> None:
    from pydantic_ai_app_factory_examples import notes_feature

    with TestClient(notes_feature.app) as client:
        assert client.get('/healthz').status_code == 200
        assert client.get('/readyz').status_code == 200
        assert client.get('/notes/about').status_code == 200
        assert client.get('/notes').json() == []
        assert client.get('/notes/notebooks').status_code == 200
