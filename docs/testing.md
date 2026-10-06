# Testing

This guide shows you how to test an app built with this package: driving it
end to end, scripting the model's side, asserting on the error contract, and
testing streaming runs.

The composed app is a FastAPI app, so `TestClient` drives it through auth,
extras validation, your capabilities and the error boundary. Point the agent
at a fake model so no request leaves the process.

## A feature end to end

`TestModel` returns a canned response without calling a provider. That is
enough whenever the assertion is about the app rather than the model's words:

```python
from fastapi.testclient import TestClient
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_app_factory import AppDeps, UserContext, create_agent_app


def _auth() -> UserContext:
    return UserContext(user_id='test')


def test_documents_feature_serves_its_route() -> None:
    agent = Agent(TestModel(), deps_type=AppDeps)
    app = create_agent_app(agent, auth=_auth, features=[Documents(store)])

    with TestClient(app) as client:  # `with`, so feature lifespans run
        assert client.get('/documents').json() == ['Quarterly report']
```

Use the context manager form. A bare `TestClient(app)` skips startup and
shutdown, so any feature that opens its resources in `lifespan()` will be
holding `None` when the route runs -- and so will a `deps` builder that reads
them from `request.app.state`.

## Scripting the model

When the assertion is about what the agent *did* -- which tool it called, what
it did with the result -- `FunctionModel` lets you script the model's side.
That is Pydantic AI's own tooling, covered in
[its testing guide](https://pydantic.dev/docs/ai/guides/testing/), including
`agent.override(model=...)` for swapping the model into an existing agent.

## Streaming runs

A plain `function` covers `POST /run` -- until something streams. Two things
do: `/chat` always streams, and overriding `wrap_run_event_stream` in any
capability switches `/run` into streaming execution too. In both cases a
`FunctionModel` built only from a `function` raises, and on `/chat` that
failure surfaces to the client as an opaque `internal` -- the real error is in
the server log.

Give the model a `stream_function` instead: an async generator over the
messages and agent info, yielding text deltas:

```python
from collections.abc import AsyncIterator

from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel


async def stream_reply(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    yield 'partial '
    yield 'answer'


agent = Agent(FunctionModel(stream_function=stream_reply), deps_type=AppDeps)
```

A `FunctionModel` may carry both: `function` serves plain runs,
`stream_function` serves streamed ones.

## Asserting on errors

The error contract is part of your API. Test the code, not the message: the
message is free to change, the code is not:

```python
def test_a_missing_document_is_a_404_with_a_stable_code() -> None:
    response = client.post('/run', json={'prompt': 'load d9'})

    assert response.status_code == 404
    assert response.json()['code'] == 'documents/not-found'
```

On `/chat` a failure inside the run leaves the response a 200, because the
run has already begun streaming when it occurs. Assert on the body instead:
the `data-error` chunk carrying `{code, message, retryable}`. Only a failure
before the stream starts -- auth, validation, the `deps` builder -- answers
with a status, as on `/run`.

## Composition failures

Duplicate feature names, extras collisions, route clashes, an agent whose
`deps_type` the factory cannot serve, and a handler annotation FastAPI could
not resolve are all raised by `create_agent_app` itself as a `ValueError`, so
they are assertions about the constructor:

```python
import pytest


def test_two_features_cannot_claim_the_same_extras_field() -> None:
    with pytest.raises(ValueError):
        create_agent_app(agent, auth=_auth, features=[Translation(), Summaries()])
```

Tool-name collisions are the exception: Pydantic AI raises them when a run
assembles its toolsets, so that test needs an actual run.

Going deeper: [Error contract](error-contract.md) for the codes you assert on,
[Features](features.md) for what a lifespan opens and closes,
[Reference](api-reference.md) for wire shapes.
