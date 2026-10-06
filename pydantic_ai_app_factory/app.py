"""Compose an agent and its features into a FastAPI application."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from types import NoneType
from typing import Any, get_origin, overload

from fastapi import APIRouter, Depends, FastAPI
from pydantic_ai import Agent
from pydantic_ai.models import Model

from pydantic_ai_app_factory._agent_routes import agent_router
from pydantic_ai_app_factory._envelope import handle_agent_error
from pydantic_ai_app_factory._extras import compose_extras_model
from pydantic_ai_app_factory._ops_routes import ops_router
from pydantic_ai_app_factory._routing import iter_routes, reserved_path, route_endpoints, unresolvable_name
from pydantic_ai_app_factory.deps import (
    AppDeps,
    AuthProvider,
    DepsBuilder,
    DepsT,
    UsageLimitsSource,
    VercelSdkVersion,
)
from pydantic_ai_app_factory.errors import AgentError, ErrorHandlers
from pydantic_ai_app_factory.feature import Feature


def create_agent(model: Model | str, *, instructions: str | None = None) -> Agent[AppDeps, str]:
    """A plain string-output agent wired to `AppDeps`."""
    return Agent(model, instructions=instructions, deps_type=AppDeps)


def _check_deps_type(agent: Agent[Any, Any], deps: DepsBuilder[AppDeps] | None) -> type[AppDeps]:
    # An agent that never declared `deps_type` has `object` there (or `NoneType`, the explicit
    # deps-free spelling): it has no opinion, and the AppDeps the factory passes at run time still
    # reaches every capability. Anything else must be AppDeps or a subclass -- pyright misses the
    # inline `Agent(...)` form, so say it here. The returned type is what a builder's result is
    # checked against at run time (see `resolve_deps`).
    declared = agent.deps_type
    origin = get_origin(declared) or declared
    if origin in (object, NoneType):
        return AppDeps
    if not (isinstance(origin, type) and issubclass(origin, AppDeps)):
        raise ValueError(f'agent deps_type {declared!r} must subclass AppDeps')
    if origin is not AppDeps and deps is None:
        # The factory would build a plain AppDeps and the product's tools would read fields that
        # are not there: the same late failure this check exists to prevent.
        raise ValueError(f'agent deps_type {declared!r} needs a `deps` builder that constructs it')
    return origin


@overload
def create_agent_app(
    agent: Agent[AppDeps, Any],
    *,
    auth: AuthProvider,
    deps: None = None,
    features: Sequence[Feature] = (),
    shared_extras_fields: frozenset[str] = frozenset(),
    health_timeout: float = 5.0,
    vercel_sdk_version: VercelSdkVersion = 5,
    error_handlers: ErrorHandlers | None = None,
    usage_limits: UsageLimitsSource = None,
) -> FastAPI: ...


@overload
def create_agent_app(
    agent: Agent[DepsT, Any],
    *,
    auth: AuthProvider,
    deps: DepsBuilder[DepsT],
    features: Sequence[Feature] = (),
    shared_extras_fields: frozenset[str] = frozenset(),
    health_timeout: float = 5.0,
    vercel_sdk_version: VercelSdkVersion = 5,
    error_handlers: ErrorHandlers | None = None,
    usage_limits: UsageLimitsSource = None,
) -> FastAPI: ...


def create_agent_app(
    agent: Agent[Any, Any],
    *,
    auth: AuthProvider,
    deps: DepsBuilder[Any] | None = None,
    features: Sequence[Feature] = (),
    shared_extras_fields: frozenset[str] = frozenset(),
    health_timeout: float = 5.0,
    vercel_sdk_version: VercelSdkVersion = 5,
    error_handlers: ErrorHandlers | None = None,
    usage_limits: UsageLimitsSource = None,
) -> FastAPI:
    """Compose an agent and its features into a FastAPI app serving the Vercel AI chat protocol.

    `deps` builds the agent's deps for each request when the agent runs with a subclass of `AppDeps`;
    the factory sets `user`, `extras` and `scratch` on the result. Leave it `None` for `AppDeps` itself.

    `features` order is significant. Capabilities are middleware, outermost first, and a feature's
    capabilities are passed per run -- so they nest *inside* the ones the agent was constructed with,
    in the order the features are listed here. Route mounting follows the same order, so the first
    feature to claim a `(method, path)` is the one that gets it and any later claim is rejected.

    `health_timeout` bounds each feature's `/readyz` check (seconds); a slower one is reported unhealthy.

    `vercel_sdk_version` is the Vercel AI SDK major the `/chat` wire targets. It stays at pydantic-ai's
    own default of 5; raise it to match a frontend on a later SDK, which adds tool-approval streaming
    and provider metadata to the chunks.
    """
    expected_deps = _check_deps_type(agent, deps)
    # Fail loud before anything is built: a doubled feature would run its capabilities twice.
    names = [feature.name for feature in features]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f'features must have unique names; duplicated: {", ".join(duplicates)}')

    # Fold the features' contributions into one request model and one capability list for every run.
    extras_model = compose_extras_model(features, shared_extras_fields=shared_extras_fields)
    capabilities = [capability for feature in features for capability in feature.capabilities()]

    # Chain the feature lifespans on one stack: they open in order and close in reverse on shutdown.
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        async with AsyncExitStack() as stack:
            for feature in features:
                context = feature.lifespan(app)
                if context is not None:
                    await stack.enter_async_context(context)
            yield

    app = FastAPI(lifespan=lifespan)
    app.add_exception_handler(AgentError, handle_agent_error)

    # The package's own routers mount first, so no feature can shadow them.
    builtin_routers = (
        agent_router(
            agent,
            auth,
            extras_model,
            capabilities,
            error_handlers,
            usage_limits,
            deps,
            expected_deps,
            vercel_sdk_version,
        ),
        ops_router(features, auth, extras_model, health_timeout),
    )
    for router in builtin_routers:
        app.include_router(router)
    # Reserve from the assembled app rather than from those routers: FastAPI mounts `/docs`, `/redoc`
    # and `/openapi.json` itself, and a feature route there would lose the first-match race and be
    # silently unreachable -- the exact failure this check exists to turn into a loud one.
    reserved_paths = {reserved_path(path) for _, path in route_endpoints(app.routes)}

    claimed_by: dict[tuple[str, str], str] = {}  # (method, path) -> the feature that mounted it

    def include_feature_router(feature: Feature, router: APIRouter, *, public: bool) -> None:
        endpoints = route_endpoints(router.routes)
        for path, route in iter_routes(router.routes):
            if (name := unresolvable_name(route)) is not None:
                raise ValueError(
                    f"feature '{feature.name}' route '{path}': a handler annotation references '{name}', "
                    "which is not defined in the handler's module -- with `from __future__ import "
                    'annotations` a closure-scoped name such as the `auth` argument of routers() is '
                    'invisible to FastAPI; drop that import from the module or make the name a module global'
                )
        # A builtin owns its whole path (slash sibling included); two features only collide on a shared
        # `(method, path)` -- `GET /items` and `POST /items` from different features coexist.
        reserved = sorted({path for _, path in endpoints if reserved_path(path) in reserved_paths})
        if reserved:
            raise ValueError(f"feature '{feature.name}' claims reserved route(s): {', '.join(reserved)}")
        # Starlette resolves a duplicated `(method, path)` by first match, leaving the later route
        # unreachable -- and serving it without auth whenever the winner came from `public_routers`.
        taken = sorted(endpoints.intersection(claimed_by))
        if taken:
            # `.lstrip()` drops the leading space a methodless (websocket) key leaves in `'{method} {path}'`.
            mounted = ', '.join(f"{m} {p} (feature '{claimed_by[m, p]}')".lstrip() for m, p in taken)
            raise ValueError(f"feature '{feature.name}' claims route(s) already mounted: {mounted}")
        # Feature routes sit behind the app's auth by default; `public_routers` is the explicit opt-out.
        app.include_router(router, dependencies=None if public else [Depends(auth)])
        claimed_by.update(dict.fromkeys(endpoints, feature.name))

    for feature in features:
        for feature_router in feature.routers(auth=auth):
            include_feature_router(feature, feature_router, public=False)
        for feature_router in feature.public_routers(auth=auth):
            include_feature_router(feature, feature_router, public=True)

    return app
