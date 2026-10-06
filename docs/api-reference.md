# Reference

Signatures, defaults, and wire shapes. For what these are *for*, see
[Building an app](building-an-app.md).

## Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/chat` | POST | Vercel AI chat protocol, streaming |
| `/run` | POST | Plain JSON `{prompt, conversation_id?, message_history?, extras?}` -> `{output, conversation_id, messages}` |
| `/config` | GET | Feature list, composed extras JSON schema and accepted file media types |
| `/config/features` | GET | How each feature is configured, from `Feature.config()` |
| `/healthz` | GET | Liveness: the process is up, no external dependency checked |
| `/readyz` | GET | Readiness: every feature's `health()`, 503 when any fails |

`/chat`, `/run`, `/config/features` and every feature route sit behind the
app's `auth`. The three discovery-and-probe endpoints (`/config`, `/healthz`,
`/readyz`) are **unauthenticated by design**, which also means `/config`
exposes the feature list and extras schema publicly. The split follows one
rule: what `/config` returns is derived by the factory from feature names,
pydantic models and pydantic-ai's media type lists, so it cannot carry a
secret, while `/config/features` relays whatever a feature's `config()` hands
it. `/config/features` answers with `Cache-Control: no-store`, since it sits
under the otherwise public `/config` prefix.

Being a FastAPI app, it also serves FastAPI's own defaults publicly: `/docs`,
`/redoc`, and `/openapi.json` -- the last enumerating every route and schema,
feature routes and extras models included. There is currently no factory
switch to turn them off.

`/chat` builds its [Vercel AI adapter](https://pydantic.dev/docs/ai/integrations/ui/vercel-ai/)
internally, on pydantic-ai's defaults. The SDK major the wire targets is
`vercel_sdk_version`, which keeps pydantic-ai's default of 5. Raise it to match
a frontend on a later SDK: 6 adds tool-approval streaming and provider metadata
on tool chunks, and 7 emits the same wire as 6.

`/run` accepts a `conversation_id` and echoes the one the run used -- minted
when the call omitted it -- so a multi-turn JSON client continues by passing
it back; over `/chat`, the Vercel protocol's chat `id` serves this role.
Either way the run reads it as `ctx.conversation_id`, which is what a
capability keys its own storage on.
`prompt` takes a string or a list of content parts, and `message_history` the
prior messages in pydantic-ai's wire form; both are sanitized as the `/chat`
adapter sanitizes a frontend's messages. The response's `messages` are the
run's own messages in that same form -- what a client appends to its history
-- with any file the model produced inline. For what a client sends to either
endpoint and what the app strips from it, see
[Agent endpoints](agent-endpoints.md).

Extras fields bind by their snake_case name and by their camelCase alias; the
schema on `/config` and in OpenAPI lists the camelCase form
([Request extras](request-extras.md#what-clients-send)).

### Request shapes

```
POST /chat            {"id": "c1", "trigger": "submit-message", "messages": [...], ...extras}
POST /run             {"prompt": "..." | [...], "conversation_id": "...", "message_history": [...], "extras": {...}}
```

On `/run` only `prompt` is required; a list holds content parts in pydantic-ai's
`UserContent` wire form, `message_history` prior messages in its `ModelMessage`
wire form. OpenAPI lists `message_history` and the response's `messages` as
arrays of plain objects, because pydantic-ai's message schema fails OpenAPI
validation ([pydantic/pydantic-ai#8679](https://github.com/pydantic/pydantic-ai/issues/8679)).

### Response shapes

```
POST /run             {"output": "...", "conversation_id": "...", "messages": [...]}
GET /config           {"features": ["storage"], "extras_schema": {...}, "accepts": ["application/pdf", ...]}
GET /config/features  {"feature_config": {"storage": {"history_source": "server"}}}
GET /healthz          {"status": "ok", "features": ["storage"]}
GET /readyz           {"status": "ok", "reports": {"storage": {"pool": "ok"}}}
```

A failing `/readyz` answers 503 with `{"status": "unhealthy", "reports": {...}, "failures": {...}}`,
where `failures` maps each failing feature to the exception's *type name*
only; the detail goes to the log.

`accepts` is the set of media types the app accepts -- pydantic-ai's image,
audio, video and document classes, identical for every model -- and says
nothing about which of them the deployed model supports; see
[Agent endpoints](agent-endpoints.md#the-new-turn).

### Error shapes

A failed run or feature route answers with the `AgentError`'s `http_status`
and this body:

```json
{"code": "documents/not-found", "message": "No document d9.", "retryable": false}
```

Mid-stream on `/chat`, the HTTP status is a 200 whatever happens -- the run
has begun streaming before it can fail. The failure travels in the body: a
`data-error` chunk carrying the same `{code, message, retryable}` object,
followed by an `ErrorChunk` with the safe message. A failure before the stream
starts -- auth, validation, the `deps` builder -- is answered with a status
like on `/run`. A malformed body or invalid extras is FastAPI's standard 422
on both endpoints.

## `create_agent`

```python
def create_agent(model: Model | str, *, instructions: str | None = None) -> Agent[AppDeps, str]: ...
```

A plain string-output agent already wired to `deps_type=AppDeps`. Covers the
common case; build the `Agent` yourself for anything else -- a different
output type, model settings, toolsets of your own.

## `create_agent_app`

```python
def create_agent_app(
    agent: Agent[AppDeps, Any],
    *,
    auth: AuthProvider,
    deps: DepsBuilder[AppDeps] | None = None,
    features: Sequence[Feature] = (),
    shared_extras_fields: frozenset[str] = frozenset(),
    health_timeout: float = 5.0,
    error_handlers: ErrorHandlers | None = None,
    usage_limits: UsageLimitsSource = None,
    vercel_sdk_version: Literal[5, 6, 7] = 5,
) -> FastAPI: ...
```

| Argument | Default | Notes |
|---|---|---|
| `agent` | required | `deps_type` is `AppDeps`, a subclass of it (then `deps` is required), or undeclared |
| `auth` | required | Any FastAPI dependency returning a `UserContext` |
| `deps` | `None` | Builds the agent's `AppDeps` subclass per request; sync or async |
| `features` | `()` | Names must be unique |
| `shared_extras_fields` | `frozenset()` | Field names two features may both declare |
| `health_timeout` | `5.0` | Seconds, per feature check, on `/readyz` |
| `error_handlers` | `None` | `(type, AgentError \| callable)` pairs, first match wins |
| `usage_limits` | `None` | A `UsageLimits`, or a callable over `UserContext` |
| `vercel_sdk_version` | `5` | Vercel AI SDK major the `/chat` wire targets |

An overload pairs `agent: Agent[DepsT, Any]` with `deps: DepsBuilder[DepsT]`,
so a product's subclass type-checks end to end.

## `Feature`

| Member | Kind | Default |
|---|---|---|
| `name` | class attribute | required, lowercase slug |
| `extras_type` | class attribute | none |
| `capabilities()` | sync | `[]` |
| `routers(*, auth)` | sync | `[]` |
| `public_routers(*, auth)` | sync | `[]` |
| `lifespan(app)` | sync, returns a context manager | `None` |
| `health()` | async | `None` |
| `config()` | sync | `None` |

`name` must match `[a-z][a-z0-9_-]*` in full; a non-conforming name raises
`TypeError` at class-definition time. Abstract bases (those listing `ABC`) are
exempt -- see [Naming and composition rules](features.md#naming-and-composition-rules).
`auth` is the app's own provider, for handlers that declare `Depends(auth)`
to read the user.

## `UserContext`

Identity and permissions, as resolved by `auth`. A frozen dataclass; a
provider may return a subclass carrying more.

| Field | Type | Default |
|---|---|---|
| `user_id` | `str` | required |
| `tenant_id` | `str \| None` | `None` |
| `scopes` | `frozenset[str]` | `frozenset()` |

## `AppDeps`

The agent's `deps_type`, reachable in every tool and capability as `ctx.deps`.
A keyword-only dataclass, so a subclass can add required fields after the
defaulted ones ([Bringing your own deps](building-an-app.md#bringing-your-own-deps)).

| Field | Type | Notes |
|---|---|---|
| `user` | `UserContext` | The caller |
| `extras` | `Mapping[str, Any]` | Validated per-request parameters, flat across all features |
| `scratch` | `dict[str, Any]` | Per-run feature state, empty at the start of every run |

## Type aliases

```python
AuthProvider = Callable[..., UserContext | Awaitable[UserContext]]
DepsBuilder = Callable[[Request, UserContext], DepsT | Awaitable[DepsT]]
ErrorHandlers = Sequence[tuple[type[Exception], AgentError | Callable[[Exception], AgentError]]]
UsageLimitsSource = UsageLimits | Callable[[UserContext], UsageLimits] | None
VercelSdkVersion = Literal[5, 6, 7]
```

`DepsT` is the type variable bound to `AppDeps` that `DepsBuilder` is generic
over; it is exported for code that is generic over the deps type itself.

## `AgentError`

```
AgentError(code: str, message: str, *, retryable: bool = False, http_status: int | None = None)
```

`code` is a string in the response body; `http_status` is the status the
response is sent under, defaulting to 503 when `retryable` and 500 otherwise.
`message` is sent to the client, so it must be user-safe.

Reserved, un-namespaced codes emitted by the package:

| Code | Status | When | Retryable |
|---|---|---|---|
| `model-unavailable` | 503 | Model backend returned 429 or 5xx, was unreachable, or every model of a `FallbackModel` failed | yes |
| `model-unauthorized` | 502 | Model backend returned 401 or 403 -- on every model of a `FallbackModel`, if one is used | no |
| `file-unsupported` | 415 | A file of a media type outside `accepts` on `GET /config`, refused before the run | no |
| `internal` | 500 | Any unrecognized exception, or a handler that itself raised | no |

A client-side model error (a 4xx other than 429, 401 and 403) is a bug, not
an outage, and maps to `internal`. Namespace your own codes as
`<feature>/<code>`.

## Package exports

```python
from pydantic_ai_app_factory import (
    AgentError,  # the wire error type
    AppDeps,  # the agent's deps_type
    AuthProvider,  # type alias for the auth dependency
    DepsBuilder,  # type alias for the deps builder
    DepsT,  # type variable bound to AppDeps
    ErrorHandlers,  # type alias for the error_handlers sequence
    Feature,  # the extension base class
    UsageLimitsSource,  # type alias for the usage_limits argument
    UserContext,  # identity and permissions
    VercelSdkVersion,  # type alias for the vercel_sdk_version argument
    create_agent,  # a string-output agent wired to AppDeps
    create_agent_app,
    to_agent_error,  # apply the handler chain to one exception; returns, never raises
)
```
