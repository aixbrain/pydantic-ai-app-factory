"""Inspect a router the way Starlette will route it: each leaf, its mounted path, its annotations."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, ForwardRef, get_type_hints


def iter_routes(routes: Iterable[Any], prefix: str = '') -> Iterator[tuple[str, Any]]:
    """Every leaf route under `routes`, paired with the path it is mounted at.

    Recurses so a mount or a nested `include_router` is inspected too: FastAPI 0.137+ keeps a nested
    include as an opaque object with no `.path`, and keying that as `''` would blind every check built
    on this walk. A route that cannot be resolved to a concrete path raises rather than slip through --
    a silently blind collision check is worse than none.
    """
    for route in routes:
        path = getattr(route, 'path', None)
        nested = getattr(route, 'routes', None)
        if nested:  # a Mount / sub-application carrying its own routes
            yield from iter_routes(nested, prefix + (path or ''))
        elif path is not None:  # a concrete Route / WebSocketRoute
            yield prefix + path, route
        elif hasattr(route, 'original_router'):  # FastAPI's opaque nested-include wrapper
            nested_prefix = prefix + getattr(route.include_context, 'prefix', '')
            yield from iter_routes(route.original_router.routes, nested_prefix)
        else:
            raise ValueError(f'cannot inspect route {route!r} for collisions; flatten the feature router')


def route_endpoints(routes: Iterable[Any]) -> set[tuple[str, str]]:
    """The `(method, path)` operations served: the unit Starlette routes on, so the unit a collision is measured in.

    A methodless websocket route keys on `''`.
    """
    return {(method, path) for path, route in iter_routes(routes) for method in getattr(route, 'methods', None) or {''}}


def reserved_path(path: str) -> str:
    """The path a builtin owns: the whole path including its trailing-slash sibling.

    `/chat` and `/chat/` are distinct routes, and a public feature route at `/chat/` would answer anonymously.
    """
    return path.rstrip('/') or '/'


def unresolvable_name(route: Any) -> str | None:
    """The name a handler annotation references that is not defined in the handler's module, or `None`.

    With `from __future__ import annotations` in the handler's module every annotation is a string,
    and one that names a closure-scoped object -- typically `Depends(auth)` with `auth` being the
    `routers()` argument -- cannot be resolved from the module's globals: it stays a `ForwardRef` and
    FastAPI never sees the `Depends` inside it. But a surviving `ForwardRef` alone is not proof of that
    trap -- a module-level router whose handler annotates a model defined further down the same module
    is analysed by FastAPI at import time, before that name exists, and is left with exactly the same
    `ForwardRef`; by the time the app is built the module has finished importing and the name is a
    module global. So re-evaluate the annotation now, against the endpoint's own module: `NameError`
    means the name really is invisible to FastAPI (the closure-scoped case) and the router is refused;
    anything else -- resolves fine, or fails for some unrelated reason -- means the route is not this
    trap, and is let through.
    """
    dependant = getattr(route, 'dependant', None)  # a mounted plain Starlette route has none
    if dependant is None:
        return None
    # FastAPI files every unresolved-ForwardRef parameter under `query_params` or `path_params`; the
    # header/cookie/body lists are scanned too, defensively, in case a future FastAPI files one there.
    param_lists = (
        dependant.path_params,
        dependant.query_params,
        dependant.header_params,
        dependant.cookie_params,
        dependant.body_params,
    )
    has_forward_ref = any(
        isinstance(param.field_info.annotation, ForwardRef) for params in param_lists for param in params
    )
    if not has_forward_ref:
        return None
    try:
        get_type_hints(route.endpoint, include_extras=True)
    except NameError as exc:
        return exc.name
    except Exception:  # best effort: cannot judge, never refuse a build over it
        return None
    return None
