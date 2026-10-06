# Features

This guide shows you how to build a `Feature`: one unit of functionality that
travels as a whole -- its capabilities, its endpoints, its startup, its
readiness check -- behind a single name. Only `name` is required; every other
member has a no-op default -- implement only what your feature offers.

Prerequisites: an app assembled as in [Building an app](building-an-app.md). A
runnable feature that implements every member ships as
`pydantic_ai_app_factory_examples/notes_feature.py`; start it with
`uv run -m pydantic_ai_app_factory_examples.notes_feature`.

Features are passed to `create_agent_app`, which folds them into one request
model and one capability list for every run:

```python
app = create_agent_app(agent, auth=auth, features=[Storage(pool), Documents(store)])
```

## Why features compose

**Features never add fields to the contract type.** They extend two mappings
on `AppDeps` instead: `extras` for validated per-request parameters, and
`scratch` for per-run state. Nothing a feature does changes the type its
neighbours see, so there is no shared type to negotiate and no ordering to get
right. That holds when a product runs the agent with its own subclass of
`AppDeps` ([Bringing your own deps](building-an-app.md#bringing-your-own-deps)):
a feature keeps seeing `user`, `extras` and `scratch` and needs no knowledge
of the subclass.

The two mappings differ in who writes them, and in naming:

- `extras` is **client input**: declared as a pydantic model, validated at the
  boundary, read-only in spirit. Its keys are the declared field names, flat
  and unprefixed -- all features read the same mapping, so collisions are
  caught at build time ([Request extras](request-extras.md)).
- `scratch` is **server state for one run**: a plain mutable mapping, empty at
  the start of every run. Its keys are yours to choose -- prefix them with the
  feature's `name`, as error codes are (`ctx.deps.scratch['storage/merged']`
  for a feature named `storage`), so two features cannot trample each other.

Prefer neither when the message history already knows the answer. Deriving
per-run state from `ctx.messages` -- scanning for your own tool calls and
their returns -- survives more edge cases than a scratch key, because the
history is what the model actually saw. Keep `scratch` for genuine scratch the
history cannot carry, such as a "have I already merged the stored
conversation" guard.

## Capabilities

A feature contributes capabilities through `capabilities()`: instances of
Pydantic AI's `AbstractCapability`, which the factory passes to every run,
where they see the composed `AppDeps` as `ctx.deps`. Writing one --
subclassing, the run hooks such as `after_run` and `before_model_request` --
is [Pydantic AI's contract](https://pydantic.dev/docs/ai/capabilities/overview/).

```python
from pydantic_ai.capabilities import AbstractCapability

from pydantic_ai_app_factory import AppDeps, Feature


class Documents(Feature):
    name = 'documents'
    extras_type = DocumentExtras

    def __init__(self, store: DocumentStore):
        self._store = store

    def capabilities(self) -> list[AbstractCapability[AppDeps]]:
        return [DocumentSearch(self._store)]
```

A capability instance is a singleton shared by every concurrent run, so
per-run state lives in `ctx.deps.scratch`, never on `self` -- and in stream
wrappers never in generator locals, because `wrap_run_event_stream` fires once
per node, not once per run. Tool results arrive there as
`FunctionToolResultEvent` with the tool name on `event.part.tool_name`.

A capability capping how often one run may call each tool -- `usage_limits`
bounds tool calls in total, not per tool -- keeps its tally there:

```python
@dataclass
class ToolBudget(AbstractCapability[AppDeps]):
    limit: int = 5

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AppDeps],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        # The hook fires per tool call, so the tally has to survive between them. A
        # field on `self` would be one tally shared by every caller at once, since
        # this instance serves every concurrent run.
        key = f'budget/{call.tool_name}'
        used = ctx.deps.scratch.get(key, 0) + 1
        ctx.deps.scratch[key] = used
        if used > self.limit:
            raise AgentError(
                'budget/tool-exhausted',
                f'The assistant called {call.tool_name} more than {self.limit} times.',
                http_status=429,
            )
        return await handler(args)
```

A wrapper may insert events, not only filter them. An inserted event reaches
the client and never enters the run's history. A tool has a typed route on
pydantic-ai 2.38 and later, `ctx.emit(CustomEvent(...))`, which arrives as a
`data-*` chunk; a capability hook has none, because `emit` restricts it to
`CapabilityEvent` and UI streams do not forward those.

Unlike routes and extras fields, tool names are not checked at build time --
that is deliberately left to Pydantic AI, which rejects a clash when the run
assembles its toolsets: a `UserError` naming the conflicting tool. That
exception is no `AgentError`, so the caller sees only a generic `internal`
error; the real message is in the server log. When two features could
plausibly ship a tool of the same name, wrap the capability in
`capability.prefix_tools('<feature-name>')` and the clash cannot happen.

One switch to know about: overriding `wrap_run_event_stream` in *any*
capability switches `agent.run()` -- and therefore `POST /run` -- into
streaming execution. The wire response is unchanged; what
changes is the model side, so tests need a `FunctionModel` with a
`stream_function` ([Testing](testing.md#streaming-runs)).

## Feature routes

`routers()` are mounted behind the app's auth automatically;
`public_routers()` is the explicit opt-out for callers that cannot
authenticate (webhooks, callbacks). Give routers an
`APIRouter(prefix='/<feature-name>')` so the feature name namespaces the URL
space.

Both methods receive the app's `auth` provider. The automatic guard only
*enforces* auth; a handler that needs to *read* the user declares a plain
`Depends(auth)` on that same object (not `Security(...)` with scopes, not
`use_cache=False`). FastAPI caches a dependency per request by identity, so
auth runs once and the handler sees the user the guard admitted -- a feature
that built its own equivalent provider would run it twice and could see a
different one:

```python
from typing import Annotated

from fastapi import APIRouter, Depends

from pydantic_ai_app_factory import AuthProvider, Feature, UserContext


class Documents(Feature):
    name = 'documents'

    def __init__(self, store: DocumentStore):
        self._store = store

    def routers(self, *, auth: AuthProvider) -> list[APIRouter]:
        router = APIRouter(prefix='/documents')

        @router.get('')
        async def index(user: Annotated[UserContext, Depends(auth)]) -> list[str]:
            return await self._store.list_titles(owner=user.user_id)

        return [router]
```

`public_routers()` gets `auth` too. Nothing is enforced on those routes, but a
public route that wants identity when a credential happens to be present can
wrap the provider in a dependency of its own that tolerates a missing
credential -- awaiting the result if the provider is async. That only works
for a provider the wrapper can call itself, typically one taking a `Request`;
a provider declared with FastAPI-injected parameters has to be the tolerant
one.

One pitfall: do not use `from __future__ import annotations` in a module that
defines handlers like this. With it, every annotation is a string, and
`Annotated[..., Depends(auth)]` names the closure-scoped `auth` that FastAPI
cannot resolve from the module's globals -- it never sees the dependency,
files the parameter as a plain path or query parameter, and the route would
fail at request time with nothing pointing at the cause. The factory refuses
such a router when the app is built, naming the feature, the route and the
name it could not resolve, so the mistake cannot reach a running server.

Exceptions in a feature route are yours to catch: `error_handlers` wrap agent
runs only, not routes. [Error contract](error-contract.md#errors-in-feature-routes)
shows the pattern.

## Health checks

`health()` is the feature's part of the readiness probe. Return a mapping and
it appears under the feature's name in the `/readyz` response; return `None`
when there is nothing to report; raise -- or exceed the time budget -- and
`/readyz` answers 503:

```python
from collections.abc import Mapping
from typing import Any


class Storage(Feature):
    name = 'storage'

    async def health(self) -> Mapping[str, Any] | None:
        await self._pool.fetchval('SELECT 1')
        return {'pool': 'ok'}
```

Checks run concurrently, each bounded by `create_agent_app`'s `health_timeout`
(default 5 seconds), so one slow dependency cannot hang the probe. A failing
check reports only the exception's *type name* on the wire; the full detail
goes to the log. Keep the report itself operational and free of secrets too --
`/readyz` is unauthenticated, so everything it returns is public.

## Feature config

`config()` is how a feature tells a client what it is doing. Return a mapping
and it appears under the feature's name on `GET /config/features`; return
`None` -- the default -- and the feature contributes no key at all:

```python
from collections.abc import Mapping
from typing import Any


class Conversations(Feature):
    name = 'conversations'

    def config(self) -> Mapping[str, Any] | None:
        return {'create_route': self._enable_create_route, 'history_source': self._history_source}
```

It is sync and called per request, so it must not do I/O -- read attributes
you already hold. Being per-request is deliberate: it costs nothing when the
method is sync, and it leaves the door open for a feature whose configuration
changes at runtime. A `config()` that raises fails the whole response, for
every feature.

Return the switches a client needs to behave correctly -- whether an optional
route is mounted, which side owns the conversation history, what a limit is
set to. Never return connection strings, paths, or credentials.
`/config/features` sits behind the app's auth, which bounds the blast radius
but does not make the payload safe: in a multi-tenant deployment every
authenticated caller sees it. The discipline is the one the error contract
uses for `str(error)` -- decide deliberately what crosses the boundary.

Route *presence* alone needs no `config()`: FastAPI already serves
`/openapi.json` publicly, and a frontend can read the mounted paths from it. `config()`
earns its place for semantics a route listing cannot express, like which side
of the wire owns the history.

## Feature infrastructure

`lifespan()` is [FastAPI's lifespan](https://fastapi.tiangolo.com/advanced/events/)
scoped to one feature: a sync method returning an async context manager, or
`None` when the feature owns no infrastructure. The factory enters it at
startup and exits it at shutdown -- it does not `await` the method itself; the
async work lives inside the context manager, around its `yield`:

```python
from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI


class Storage(Feature):
    name = 'storage'

    def lifespan(self, app: FastAPI) -> AbstractAsyncContextManager[None] | None:
        return self._pool_lifespan(app)

    @asynccontextmanager
    async def _pool_lifespan(self, app: FastAPI) -> AsyncGenerator[None]:
        self._pool = await open_pool(...)  # startup
        app.state.pool = self._pool
        try:
            yield
        finally:
            await self._pool.close()  # shutdown
```

Stash shared resources on `app.state` (the pool above) so routes outside the
feature -- and a product's `deps` builder -- can reach them. The factory
chains every feature's lifespan on an `AsyncExitStack`: they open in order and
are exited in reverse -- including when a later feature's startup fails. That
failure arrives at your `yield` as an exception, which is why the close
belongs in `finally`: a bare line after the `yield` would be skipped and the
pool would leak.

## Naming and composition rules

The feature `name` -- a lowercase slug, `[a-z][a-z0-9_-]*` in full, enforced
with a `TypeError` at class definition -- shows up on every surface the
feature touches:

| Surface | Shape |
|---|---|
| URLs | `APIRouter(prefix='/storage')` |
| Error codes | `storage/not-found` |
| `/config` | `storage` listed in the `features` array |
| `/config/features` | report keyed `storage` |
| `/readyz` | report keyed `storage` |
| `scratch` keys | `storage/merged`, by convention |

A base class that lists `ABC` in its own bases stays abstract and needs no
`name`. Un-namespaced error codes (`internal`, `model-unavailable`,
`model-unauthorized`) are reserved for the package.

Composition fails loud, at build time, with a `ValueError`: an agent whose
`deps_type` the factory cannot serve, duplicate feature names, extras fields
two features declare, and routes that FastAPI could not resolve, collide with
a built-in endpoint or with another feature. The checks run in that order and
stop at the first failing category; within extras, every colliding field is
reported at once. Two features may split one path across different methods --
`GET /export` in one, `POST /export` in another -- because routing is per
method-and-path. Tool names are the one deliberate exception, checked by
Pydantic AI at run time as described under [Capabilities](#capabilities).

What you end up with: one unit under one name -- routes behind the app's
`auth`, capabilities in every run, a report keyed on `/readyz` and
`/config/features` -- that any app composes in with a single `features=`
entry.
