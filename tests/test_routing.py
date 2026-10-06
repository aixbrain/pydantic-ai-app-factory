from typing import Annotated, Any, ForwardRef

import pytest
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.testclient import TestClient
from pydantic_ai.models.test import TestModel

from pydantic_ai_app_factory import AuthProvider, Feature, UserContext
from pydantic_ai_app_factory.app import create_agent, create_agent_app
from tests._support import auth, make_client


def test_the_endpoint_walker_recurses_into_a_mount() -> None:
    from starlette.routing import Mount, Route

    from pydantic_ai_app_factory._routing import route_endpoints

    def _hit() -> None: ...

    mount = Mount('/sub', routes=[Route('/x', _hit, methods=['GET'])])
    assert {path for _, path in route_endpoints([mount])} == {'/sub/x'}


def test_the_endpoint_walker_refuses_a_route_it_cannot_inspect() -> None:
    # The safety net: a route we can't resolve to a concrete path is a loud error, not a silent skip,
    # so the collision check can never go quietly blind on a future routing-internals change.
    from pydantic_ai_app_factory._routing import iter_routes

    with pytest.raises(ValueError, match='cannot inspect'):
        list(iter_routes([object()]))


def test_routers_receive_the_apps_auth_and_it_runs_once_per_request() -> None:
    # The factory guards the router with `Depends(auth)`; a handler that declares `Depends(auth)` on
    # the same object gets the resolved user from FastAPI's per-request cache. A feature building an
    # equivalent-but-different callable would run auth twice and could see two different users.
    calls: list[str] = []

    def counting_auth() -> UserContext:
        calls.append('auth')
        return UserContext(user_id='u1')

    class _WhoAmI(Feature):
        name = 'whoami'

        def routers(self, *, auth: AuthProvider) -> list[APIRouter]:
            router = APIRouter()

            @router.get('/whoami')
            def whoami(user: Annotated[UserContext, Depends(auth)]) -> dict[str, str]:
                return {'user': user.user_id}

            return [router]

    app = create_agent_app(create_agent(TestModel()), auth=counting_auth, features=[_WhoAmI()])

    assert TestClient(app).get('/whoami').json() == {'user': 'u1'}
    assert calls == ['auth']


def test_public_routers_receive_auth_so_a_public_route_can_resolve_identity_when_present() -> None:
    # A public route is not guarded, but it may still want to personalise when a credential is
    # there. It gets the same provider and wraps it in a dependency of its own that tolerates a
    # missing credential; without the argument this is impossible.
    def header_auth(request: Request) -> UserContext:
        if request.headers.get('x-api-key') != 'k':
            raise HTTPException(status_code=401)
        return UserContext(user_id='u1')

    class _Landing(Feature):
        name = 'landing'

        def public_routers(self, *, auth: AuthProvider) -> list[APIRouter]:
            router = APIRouter()

            def optional_user(request: Request) -> UserContext | None:
                try:
                    user = auth(request)
                except HTTPException:
                    return None
                assert isinstance(user, UserContext)  # this test's provider is synchronous
                return user

            @router.get('/landing')
            def landing(user: Annotated[UserContext | None, Depends(optional_user)]) -> dict[str, str]:
                return {'greeting': f'hi {user.user_id}' if user else 'hi stranger'}

            return [router]

    client = TestClient(create_agent_app(create_agent(TestModel()), auth=header_auth, features=[_Landing()]))

    assert client.get('/landing').json() == {'greeting': 'hi stranger'}
    assert client.get('/landing', headers={'x-api-key': 'k'}).json() == {'greeting': 'hi u1'}


def test_a_router_whose_handler_annotation_could_not_be_resolved_is_refused_at_build_time() -> None:
    # `from __future__ import annotations` in the feature's module turns `Depends(auth)` into a
    # string FastAPI cannot resolve; the parameter silently becomes a query parameter and the route
    # answers 422 with no hint. The mount-time check names the handler and the cause instead.
    from tests._future_feature import Broken

    with pytest.raises(ValueError, match=r"feature 'broken' route '/who': a handler annotation references 'auth'"):
        create_agent_app(create_agent(TestModel()), auth=auth, features=[Broken()])


def test_an_unresolved_annotation_on_a_path_parameter_is_refused_too() -> None:
    # A parameter named like a path segment is filed under path parameters, not query parameters;
    # the check has to look at every parameter kind or this shape slips through.
    from tests._future_feature import BrokenPath

    match = r"feature 'broken-path' route '/items/\{user\}': a handler annotation references 'auth'"
    with pytest.raises(ValueError, match=match):
        create_agent_app(create_agent(TestModel()), auth=auth, features=[BrokenPath()])


def test_resolvable_annotations_under_the_future_import_are_not_refused() -> None:
    # The check must not fire on handlers the import does not break: a plain query parameter, no
    # parameters, and a module-level dependency all resolve from the string.
    from tests._future_feature import Benign

    client = make_client(features=[Benign()])
    assert client.get('/plain', params={'q': 'x'}).json() == {'q': 'x'}
    assert client.get('/module').json() == {'x': 'ok'}


def test_a_forward_reference_that_resolves_once_the_module_is_imported_is_not_refused() -> None:
    # A module-level router is analysed at import time, before a model defined further down exists;
    # the annotation survives as a forward reference that resolves fine once the module has loaded.
    # Refusing it would reject a working app and tell the author to drop an import the module needs.
    from tests._future_late_model import LateModel

    client = make_client(features=[LateModel()])
    assert client.post('/late', json={'x': 'hi'}).json() == {'x': 'hi'}


def test_an_annotation_the_helper_cannot_judge_is_not_refused() -> None:
    # A `ForwardRef` survives, but re-evaluating it fails for a reason other than a missing name -- here,
    # the endpoint is not something `get_type_hints` can introspect at all. The helper cannot tell
    # whether this is the trap or not, so best effort means it must not refuse the build over something
    # it does not understand. Exercised directly: no real FastAPI route reaches this state, because
    # FastAPI's own registration-time evaluation only ever leaves a `ForwardRef` behind on a `NameError`,
    # so by construction a later `NameError` or success are the only outcomes a real route can produce.
    from types import SimpleNamespace

    from pydantic_ai_app_factory._routing import unresolvable_name

    param = SimpleNamespace(name='body', field_info=SimpleNamespace(annotation=ForwardRef('Whatever')))
    no_params: list[Any] = []
    dependant = SimpleNamespace(
        path_params=no_params,
        query_params=no_params,
        header_params=no_params,
        cookie_params=no_params,
        body_params=[param],
    )
    route = SimpleNamespace(dependant=dependant, endpoint=42)  # not a module, class, method, or function
    assert unresolvable_name(route) is None


def test_a_mounted_sub_application_inside_a_feature_router_is_not_inspected_for_annotations() -> None:
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route

    async def hello(request: Request) -> PlainTextResponse:
        return PlainTextResponse('hi')

    class _Sub(Feature):
        name = 'sub'

        def routers(self, *, auth: AuthProvider) -> list[APIRouter]:
            router = APIRouter()
            router.mount('/sub', Starlette(routes=[Route('/hello', hello)]))
            return [router]

    assert make_client(features=[_Sub()]).get('/sub/hello').text == 'hi'
