"""The composition unit: what a feature offers the app factory."""

from __future__ import annotations

import re
from abc import ABC
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from typing import Any, ClassVar

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel
from pydantic_ai.capabilities import AbstractCapability

from pydantic_ai_app_factory.deps import AppDeps, AuthProvider

# The name flows into wire contracts: JSON keys on /config and /readyz, the `<feature>/<code>`
# error-code namespace, and `data-<feature>-<kind>` chunk types -- hence the slug restriction.
_NAME_PATTERN = re.compile(r'[a-z][a-z0-9_-]*')


class Feature(ABC):
    """A unit of functionality the app factory composes into the app."""

    name: ClassVar[str]
    """The feature's identifier and namespace; a lowercase slug."""
    extras_type: type[BaseModel] | None = None
    """Schema for this feature's slice of the request extras, or `None` if it declares none."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if ABC in cls.__bases__:
            # An explicitly abstract base contributes no feature of its own and needs no name.
            return
        name = getattr(cls, 'name', None)
        if not name:
            raise TypeError(f'{cls.__name__} must define a class-level `name` (or subclass ABC to stay abstract).')
        if not _NAME_PATTERN.fullmatch(name):
            raise TypeError(f'{cls.__name__}.name {name!r} must be a slug: {_NAME_PATTERN.pattern!r}.')

    def capabilities(self) -> list[AbstractCapability[AppDeps]]:
        """Capabilities this feature adds to the agent, added per run.

        They nest inside the agent's own constructor capabilities, and inside those of any feature
        listed before this one -- capabilities are middleware and the first is outermost.
        """
        return []

    def routers(self, *, auth: AuthProvider) -> list[APIRouter]:
        """API routes this feature mounts; the app mounts them behind its auth.

        `auth` is the app's own provider. A handler that needs the resolved user declares a plain
        `Depends(auth)` on exactly this object (not `Security(...)` with scopes, not `use_cache=False`):
        the app already guards the router with it, and FastAPI caches a dependency per request by
        identity, so auth runs once and the handler sees the same user the guard admitted. Two features
        claiming the same path is a composition error. Prefer `APIRouter(prefix='/<name>')` so the
        feature name namespaces the URL space as it does every other wire contract.
        """
        return []

    def public_routers(self, *, auth: AuthProvider) -> list[APIRouter]:
        """API routes mounted without auth, for callers that cannot authenticate (webhooks, callbacks).

        `auth` is passed for routes that want identity when a credential happens to be present;
        nothing is enforced here.
        """
        return []

    def lifespan(self, app: FastAPI) -> AbstractAsyncContextManager[None] | None:
        """Async context manager for this feature's infrastructure (pools, migrations), or `None`."""
        return None

    async def health(self) -> Mapping[str, Any] | None:
        """Readiness detail for this feature, or `None` when it has nothing to report.

        Returning a mapping means healthy (its content is opaque to the app factory); raising means unhealthy.
        """
        return None

    def config(self) -> Mapping[str, Any] | None:
        """How this feature is configured, reported on `/config/features`, or `None` to report nothing.

        Called per request, so it must not do I/O, and raising fails the response for every feature.
        The result reaches every authenticated caller -- return the switches a client needs, never
        connection strings, paths, or credentials.
        """
        return None
