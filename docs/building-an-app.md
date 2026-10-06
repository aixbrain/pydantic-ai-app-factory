# Building an App

This package does two things. First, it turns any Pydantic AI agent into a FastAPI application. Second, it lets you extend that application with **Features**: reusable units that pair Pydantic AI capabilities with their own endpoints, startup and health checks.

> [!NOTE]
> Throughout the documentation, every name imports from the package root unless shown otherwise.

## From Agent to App

`create_agent_app` combines a Pydantic AI agent with an authentication mechanism into a FastAPI application:

```python
def create_agent_app(
    agent: Agent[AppDeps, Any],  # your agent
    *,
    auth: AuthProvider,  # maps incoming requests to a UserContext
    deps: DepsBuilder[AppDeps] | None = None,  # builds your own AppDeps subclass per request
    features: Sequence[Feature] = (),  # Feature functionality
    shared_extras_fields: frozenset[str] = frozenset(),  # Feature functionality
    health_timeout: float = 5.0,  # Feature functionality
    vercel_sdk_version: VercelSdkVersion = 5,  # Vercel AI SDK major `/chat` targets
    error_handlers: ErrorHandlers | None = None,  # maps exceptions to AgentErrors
    usage_limits: UsageLimitsSource = None,  # Pydantic AI's, forwarded to every run
) -> FastAPI: ...
```

The three parameters related to Features are covered in the next section. The other parameters are:

- **`agent`** is any [Pydantic AI agent](https://pydantic.dev/docs/ai/core-concepts/agent/),
  with one condition on its `deps_type`: `AppDeps`, or a subclass of it
  together with a `deps` builder ([below](#bringing-your-own-deps)). That is
  the object the factory builds per request, so a tool reads the caller at
  `ctx.deps.user`. An agent that never declared a `deps_type` runs with a
  plain `AppDeps`; any other `deps_type` is refused when the app is built.
  `create_agent(model, instructions=...)` is the one-liner for a plain
  string-output agent.
- **`auth`** is any FastAPI dependency returning a `UserContext`.
  [Sub-dependencies](https://fastapi.tiangolo.com/tutorial/dependencies/sub-dependencies/)
  resolve as usual. It guards `/chat`, `/run`, `/config/features` and every
  feature route by default.
- **`deps`** builds the agent's deps for each request when the agent runs
  with your own subclass of `AppDeps`. Leave it `None` for `AppDeps` itself.
- **`error_handlers`** maps your own exception types to an `AgentError`. An
  `AgentError` is the exception class every failure in a run is turned into: a
  stable `code` for the client, plus a message safe to show the user. An exception that is not mapped turns into a generic `AgentError('internal', 'An internal error occurred.')`.
- **`usage_limits`** is Pydantic AI's
  [`UsageLimits`](https://pydantic.dev/docs/ai/core-concepts/agent/#usage-limits),
  fixed or resolved per caller.
- **`vercel_sdk_version`** is the Vercel AI SDK major `/chat` targets. It
  stays at pydantic-ai's default of 5; raise it to match a frontend on a
  later SDK.

### Example App (without Features)

```python
from typing import Annotated

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits

from pydantic_ai_app_factory import AgentError, AppDeps, UserContext, create_agent_app

bearer = OAuth2PasswordBearer(tokenUrl='token')


async def auth(token: Annotated[str, Depends(bearer)]) -> UserContext:
    claims = decode_and_verify(token)  # your JWT validation
    return UserContext(user_id=claims['sub'], tenant_id=claims.get('org'))


agent = Agent('openai:gpt-5', deps_type=AppDeps, instructions='Help the user to find the right documents.')


@agent.tool
async def list_documents(ctx: RunContext[AppDeps]) -> list[str]:
    """Titles of the caller's documents."""
    return await document_store.list_titles(owner=ctx.deps.user.user_id)


app = create_agent_app(
    agent,
    auth=auth,
    error_handlers=[(StorageTimeout, AgentError('storage/timeout', 'Retry shortly.', retryable=True))],
    usage_limits=UsageLimits(request_limit=10),
)
```

The resulting app is a plain `FastAPI` instance serving six endpoints:

- Requiring user authentication:
  - `/chat` streams the Vercel AI chat protocol, targeting SDK major 5 unless
    `vercel_sdk_version` says otherwise
  - `/run` works the same as `/chat` but sends plain JSON instead; what a
    client sends to either is covered in [Agent Endpoints](agent-endpoints.md)
  - `/config/features` reports how each feature is configured
- Public endpoints:
  - `/config` for self-discovery: the features, their extras schema and the
    file media types the app accepts (pydantic-ai's classes, not what the
    model supports)
  - `/healthz` for liveness probes
  - `/readyz` for feature health probes

What the two endpoints carry differs. `/chat` takes the Vercel protocol's
message list, and a client may put its whole local history there or only the
new turn -- the factory forwards what arrives, unchanged. `/run` takes the same
choice as an optional `message_history` beside the prompt. Neither shape
decides where the model's context comes from, though; the factory only
guarantees that both endpoints carry a conversation id and that the run reads
it as `ctx.conversation_id`. Owning the history is a feature's job, composed in
through `features=` like any other.

Remember that the app is extendible with anything FastAPI offers: `include_router`, middleware like `CORSMiddleware`, and the rest of the framework are all available.

### Bringing your own deps

An agent you already have usually carries its own deps class: a pool, a
client, config. Subclass `AppDeps` with it and give `create_agent_app` a
builder; your tools keep reading `ctx.deps.pool` as before. This example
continues the one above, with the same `auth` (`Pool` is your own client's
type):

```python
from dataclasses import dataclass

from fastapi import Request
from pydantic_ai import Agent, RunContext

from pydantic_ai_app_factory import AppDeps, UserContext, create_agent_app


@dataclass(kw_only=True)
class MyDeps(AppDeps):
    pool: Pool


agent = Agent('openai:gpt-5', deps_type=MyDeps)


@agent.tool
async def search(ctx: RunContext[MyDeps], q: str) -> list[str]:
    return await ctx.deps.pool.fetch(q, owner=ctx.deps.user.user_id)


def build_deps(request: Request, user: UserContext) -> MyDeps:
    return MyDeps(user=user, pool=request.app.state.pool)  # opened in a feature's lifespan


app = create_agent_app(agent, auth=auth, deps=build_deps)
```

The builder runs once per request and may be async. It must return a new
instance per request: the factory sets fields on the object you return, so a
builder that hands back a shared or cached instance leaks one request's `user`
into another's. Read objects your lifespan opened from `request.app.state`; a
pool does not exist when the app is built. Whatever the builder sets on
`user`, `extras` and `scratch` is overwritten by the factory afterwards, so a
subclass cannot hand a run the wrong user. `AppDeps` is keyword-only so your
subclass can add required fields after the defaulted `extras`/`scratch`;
declaring your own subclass `kw_only=True` keeps its construction keyword-only
too, but is not required. `create_agent_app` types the `deps` argument as
`DepsBuilder[MyDeps]`, exported alongside `AppDeps` for products that annotate
their builder.

An agent whose `deps_type` the factory cannot serve is refused when the app is
built, not on the first tool call: a class that does not subclass `AppDeps`,
or a subclass without a builder. A builder that returns an instance of the
wrong class is caught per request, before any tool runs, and answered as a
generic `internal` error.

A builder that raises is mapped through the same
[error contract](error-contract.md) as a run failure: raise `AgentError` for a
failure the client should see by name, and anything else becomes a generic
`internal` error. Because the builder runs before the stream starts, that
error is a JSON response with its status on `/chat` as well.

Feature capabilities keep their `AbstractCapability[AppDeps]` signature. They
see the product's subclass at run time and can rely on `user`, `extras` and
`scratch` being there.

We recommend continuing with the section on Features. More reading on the above concepts can be found in:

1. [Authentication and Authorization](auth.md) for building `auth` and
   subclassing `UserContext`
2. [Error Contract](error-contract.md) for status semantics and
   code namespacing
3. [Testing](testing.md) for how to cover apps by tests
4. [API Reference](api-reference.md)

## Extending with features

Pydantic AI's own
[capabilities](https://pydantic.dev/docs/ai/capabilities/overview/) are
_composable units of agent behavior_: tools, model hooks, the things that
happen during a run. A capability alone stops there -- it has no endpoint, no
startup, no health check. `Feature` is where that stops being true: it pairs
one or more capabilities with the FastAPI side they need. Only `name` is
required; every other member has a no-op default.

```python
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic_ai.capabilities import AbstractCapability

from pydantic_ai_app_factory import AppDeps, AuthProvider, Feature, UserContext


class Documents(Feature):
    name = 'documents'

    def __init__(self, store: DocumentStore):
        self._store = store

    def capabilities(self) -> list[AbstractCapability[AppDeps]]:
        return [DocumentSearch(self._store)]

    def routers(self, *, auth: AuthProvider) -> list[APIRouter]:
        router = APIRouter(prefix='/documents')

        @router.get('')
        async def index(user: Annotated[UserContext, Depends(auth)]) -> list[str]:
            return await self._store.list_titles(owner=user.user_id)

        return [router]


app = create_agent_app(agent, auth=auth, features=[Documents(document_store)])
```

That alone changes what the app does: `DocumentSearch` now runs on every
request the app serves, `GET /config` reports `"documents"` among the composed
features, and the app now also serves `GET /documents` -- already behind your
`auth`, with no guard to wire up yourself. The `auth` the handler reads the
user from is the app's own provider, handed to `routers()` by the factory.

Going deeper: [Features](features.md) covers the rest of what a `Feature` can
carry -- `public_routers()` for endpoints with no auth, `lifespan()` for
startup and shutdown, `health()` and `config()` for `/readyz` and
`/config/features`, `extras_type` for per-request parameters -- and how the
name namespaces all of it. [Request extras](request-extras.md) covers
`shared_extras_fields` for a parameter two features both declare.
