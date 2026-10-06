"""The request-scoped state a composed agent runs with."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, TypeVar

from fastapi import Request
from pydantic_ai.usage import UsageLimits


@dataclass(frozen=True)
class UserContext:
    """Who is making the request, as resolved by the auth provider.

    A provider may return a subclass carrying richer data; the app factory passes that instance through
    unchanged. Because deps are logged and serialized, declare any per-user credential on that
    subclass as `pydantic.SecretStr` to keep it out of reprs and payloads.
    """

    user_id: str
    """Stable identity of the caller."""
    tenant_id: str | None = None
    """Tenant the caller acts in; `None` in single-tenant deployments."""
    scopes: frozenset[str] = frozenset()
    """Permissions granted to this caller."""


@dataclass(kw_only=True)
class AppDeps:
    """The `deps_type` every agent this package composes runs with.

    Features extend `extras` and `scratch` under their own namespace, never the type. A product may
    subclass it to carry its own objects (a pool, a client, config) and hand `create_agent_app` a
    `deps` builder; the factory sets `user`, `extras` and `scratch` after the builder returns, so a
    subclass cannot get them wrong. Keyword-only so a subclass can add required fields after these
    defaulted ones.
    """

    user: UserContext
    """The caller."""
    extras: Mapping[str, Any] = field(default_factory=dict)
    """The caller's per-request feature parameters, validated at the boundary and opaque to this package."""
    scratch: dict[str, Any] = field(default_factory=dict)
    """Per-run feature state, opaque to this package.

    Capabilities are agent singletons shared by concurrent runs, so run state cannot live on `self`.
    Namespace keys under the feature name.
    """


DepsT = TypeVar('DepsT', bound=AppDeps)
"""The deps type an agent runs with: `AppDeps` or a product's subclass of it."""

AuthProvider: TypeAlias = Callable[..., UserContext | Awaitable[UserContext]]
"""A FastAPI dependency that resolves the request to a `UserContext`."""

DepsBuilder: TypeAlias = Callable[[Request, UserContext], DepsT | Awaitable[DepsT]]
"""Builds a product's `AppDeps` subclass for one request.

Called once per run with the request and the resolved user. Read lifespan-opened objects from
`request.app.state`. Whatever it sets on `user`, `extras` and `scratch` is overwritten by the factory.
Return a new instance per request: the factory sets fields on the object you return, so a builder that
hands back a shared or cached instance leaks one request's `user` into another's.
"""

UsageLimitsSource: TypeAlias = UsageLimits | Callable[[UserContext], UsageLimits] | None
"""Fixed limits for every run, a per-caller callable, or `None` for pydantic-ai's defaults."""

VercelSdkVersion: TypeAlias = Literal[5, 6, 7]
"""Vercel AI SDK major the `/chat` wire targets; 5 is pydantic-ai's own default."""
