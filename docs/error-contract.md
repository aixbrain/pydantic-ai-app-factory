# Error Contract

This guide shows you how to give every failure a stable, client-usable shape:
raising your own errors, mapping your infrastructure's exceptions, choosing
HTTP statuses, and keeping the same contract inside feature routes.

Prerequisites: [Building an app](building-an-app.md); the route section
additionally assumes [Features](features.md).

Every failure in a run leaves the app as an `AgentError` -- there is no other
exit. The reserved codes the package emits itself are listed in the
[reference](api-reference.md#agenterror).

## Raising your own errors

Raise `AgentError(code, message, retryable=..., http_status=...)` to surface a
failure with a stable `code` the client can act on, instead of it being masked
as a generic `internal` error. Namespace codes by feature; the un-namespaced
codes `internal`, `model-unavailable`, `model-unauthorized` and
`file-unsupported` are reserved for this package. The `message` must be
user-safe -- it is shown to the caller -- while the runner keeps an
unrecognized error's detail in the log rather than leaking `str(error)`.

```python
from pydantic_ai_app_factory import AgentError


@agent.tool
async def load_document(ctx: RunContext[AppDeps], document_id: str) -> str:
    document = await document_store.get(document_id, owner=ctx.deps.user.user_id)
    if document is None:
        raise AgentError('documents/not-found', f'No document {document_id}.', http_status=404)
    return document.text
```

## The package's own codes

Four codes are emitted without you raising anything. `model-unavailable`
(503, retryable) is a transient provider failure: a 429, a 5xx, a connection
error, or every model of a `FallbackModel` failing -- whatever the individual
failures were, unless every one of them was a credential rejection.
`model-unauthorized` (502, not retryable) is the provider rejecting the
deployment's credentials -- a 401 or 403, or every model of a `FallbackModel`
reporting one -- which is a configuration error, not a bug in your code, and
retrying will not help. `file-unsupported` (415, not retryable) is a file
whose media type is outside what pydantic-ai classifies as an image, audio,
video or document, refused before the run; `GET /config` lists the accepted
types as `accepts`. Everything else is `internal` (500), including a model
class raising for a class it does not support, see
[Agent endpoints](agent-endpoints.md#the-new-turn).

The split is an approximation over HTTP status: Gemini reports a bad key as
400 and OpenAI reports an exhausted balance as 429, neither distinguishable
from a genuine bad request or rate limit without reading the response body,
which the runner deliberately does not do. Translate those in
`error_handlers` if you need them.

## Choosing a status

`code` and `http_status` are independent choices, and an error carries both.
The code names what happened and is yours to invent; the status tells
HTTP-aware machinery -- proxies, retry middleware, a frontend's fetch wrapper
-- how to treat the response. One 404 can carry `documents/not-found` or
`conversations/not-found`, and a client that needs to tell them apart reads
the code, never the status.

Set `http_status` at the raise site: only the raise site knows that a missing
document is a 404 or a spent quota a 429. Unset, it falls back to 503 when
`retryable`, else 500. Run failures are logged by that status: an expected 4xx
at INFO, a retryable 5xx at WARNING, and only a non-retryable fault at ERROR
with the traceback -- so a deliberate `not-found` does not page anyone.

## What the client receives

On `/run` and on feature routes, the error is a JSON body under its status:

```json
{"code": "documents/not-found", "message": "No document d9.", "retryable": false}
```

`retryable` rides along even where the status implies it, so a client reads
one field whatever the status was. Mid-stream on `/chat` there is no status
left to send, so the error arrives inside the stream instead: a
machine-readable `data-error` chunk carrying the same `{code, message,
retryable}` body, followed by an `ErrorChunk` with the safe message for SDK
consumers that only understand that one. A failure *before* the stream
starts -- a refused file, or your `deps` builder raising -- is answered as
JSON with its status on `/chat` as well.

Invalid request bodies never reach the contract at all: a malformed body or
invalid extras is FastAPI's standard 422, in the same shape on both endpoints.

## Translating your exceptions

An `error_handlers` sequence -- typed `ErrorHandlers`, importable if a
function of yours builds one -- translates your infrastructure's exceptions at
the boundary of every run, first matching type wins:

```python
app = create_agent_app(
    agent,
    auth=auth,
    error_handlers=[
        (StorageTimeout, AgentError('storage/timeout', 'Storage is slow, retry shortly.', retryable=True)),
        (StorageError, lambda exc: AgentError('storage/failed', 'Storage is unavailable.', retryable=exc.is_transient)),
    ],
)
```

A callable handler should derive the wire error's *shape* from the exception
-- its code, `retryable`, `http_status` -- and never put `str(exc)` in the
`message`, which can carry a DSN or other secret. A handler that raises is
caught and falls back to `internal`, so a bug in a handler cannot break the
no-leak contract. The same handlers apply to an exception raised by a `deps`
builder, which runs right before the run.

## Errors in feature routes

An `AgentError` raised in a feature route is rendered by the same app-level
handler -- but `error_handlers` wrap agent runs only, so any *other* exception
in a route surfaces as FastAPI's plain 500. Catch it yourself and pass it,
with the handlers, to `to_agent_error(exc, error_handlers)`. It returns the
resulting `AgentError` -- it does not raise anything -- so you raise what it
gives back.

To use the same handler list in both places, hand it to the feature's
constructor:

```python
from fastapi import APIRouter

from pydantic_ai_app_factory import AgentError, AuthProvider, ErrorHandlers, Feature, create_agent_app, to_agent_error


class Imports(Feature):
    name = 'imports'

    def __init__(self, handlers: ErrorHandlers, store: DocumentStore):
        self._handlers = handlers
        self._store = store

    def routers(self, *, auth: AuthProvider) -> list[APIRouter]:
        router = APIRouter(prefix='/imports')

        @router.post('')
        async def import_documents() -> dict[str, int]:
            try:
                return {'imported': await self._store.import_batch()}
            except Exception as exc:
                raise to_agent_error(exc, self._handlers) from exc

        return [router]


handlers = [(StorageTimeout, AgentError('storage/timeout', 'Retry shortly.', retryable=True))]
app = create_agent_app(agent, auth=auth, features=[Imports(handlers, store)], error_handlers=handlers)
```

And to change how an `AgentError` renders as JSON, register your own handler
with `app.add_exception_handler(AgentError, mine)` -- before the first
request: Starlette freezes exception handlers when it builds the middleware
stack, so registering later is a silent no-op.

## Usage limits

Pass `usage_limits=` to `create_agent_app` to bound every run -- Pydantic AI
accepts limits only on the run call, so the factory forwards them to each run
for you. A fixed `UsageLimits(...)` applies to every caller; a
callable over the authenticated `UserContext` gives per-tier budgets:

```python
from pydantic_ai.usage import UsageLimits

app = create_agent_app(agent, auth=auth, usage_limits=UsageLimits(request_limit=10))
```

`UsageLimitsSource` names that argument's type -- a `UsageLimits`, a callable
over the `UserContext`, or `None` -- for products that annotate their own
limits configuration.

Map the resulting `UsageLimitExceeded` to a client-facing error yourself -- it
is deliberately not mapped by default, because the exception covers request,
tool-call, and token limits, which deserve different codes:

```python
app = create_agent_app(
    agent,
    auth=auth,
    usage_limits=lambda user: TIER_LIMITS[user.tenant_id],
    error_handlers=[(UsageLimitExceeded, AgentError('limits/spent', 'Budget spent.', http_status=429))],
)
```

Inside a capability, `ctx.usage_limits` is the live object the run enforces
against: mutating a field changes what is enforced, while *replacing* the
object (`ctx.usage_limits = ...`) is a silent no-op. With the factory knob you
should not need either.
