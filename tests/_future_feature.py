"""A feature module with the `__future__` import: closure-scoped `Depends(auth)` stays a string here."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from pydantic_ai_app_factory import AuthProvider, Feature, UserContext


def module_dep(request: Request) -> str:
    """A module-level dependency: resolvable from a stringized annotation, so not the trap."""
    return 'ok'


class Broken(Feature):
    """One handler references the closure-scoped `auth`; FastAPI cannot resolve it from the string."""

    name = 'broken'

    def routers(self, *, auth: AuthProvider) -> list[APIRouter]:
        router = APIRouter()

        @router.get('/fine')
        def fine(q: str) -> dict[str, str]:
            return {'q': q}

        @router.get('/who')
        def who(user: Annotated[UserContext, Depends(auth)]) -> dict[str, str]:
            return {'user': user.user_id}

        return [router]


class BrokenPath(Feature):
    """The unresolved handler parameter shares its name with a path template segment.

    FastAPI classifies a parameter as a path variable purely from its name matching the route's path
    template -- it never looks at the (unresolved) annotation -- so this lands in `dependant.path_params`
    rather than `query_params` and would slip past a check that only scans query parameters.
    """

    name = 'broken-path'

    def routers(self, *, auth: AuthProvider) -> list[APIRouter]:
        router = APIRouter()

        @router.get('/items/{user}')
        def item(user: Annotated[UserContext, Depends(auth)]) -> dict[str, str]:
            return {'user': user.user_id}

        return [router]


class Benign(Feature):
    """Handlers under the same import that resolve fine: a plain query, no parameters, a module-level dependency."""

    name = 'benign'

    def routers(self, *, auth: AuthProvider) -> list[APIRouter]:
        router = APIRouter()

        @router.get('/plain')
        def plain(q: str) -> dict[str, str]:
            return {'q': q}

        @router.get('/none')
        def none() -> dict[str, bool]:
            return {'ok': True}

        @router.get('/module')
        def module(x: Annotated[str, Depends(module_dep)]) -> dict[str, str]:
            return {'x': x}

        return [router]
