from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from types import NoneType

import pytest
from fastapi import APIRouter, FastAPI, Request, WebSocket
from fastapi.testclient import TestClient
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_app_factory import AppDeps, AuthProvider, Feature, UserContext
from pydantic_ai_app_factory.app import create_agent, create_agent_app
from tests._support import Storage, auth, make_client, ping, strict_auth, submit


class _Routed(Feature):
    name = 'routed'

    def routers(self, *, auth: AuthProvider) -> list[APIRouter]:
        router = APIRouter()
        router.add_api_route('/ping', ping, methods=['GET'])
        return [router]


class _Hooked(Feature):
    name = 'hooked'

    def routers(self, *, auth: AuthProvider) -> list[APIRouter]:
        router = APIRouter()
        router.add_api_route('/manage', ping, methods=['GET'])
        return [router]

    def public_routers(self, *, auth: AuthProvider) -> list[APIRouter]:
        router = APIRouter()
        router.add_api_route('/webhook', ping, methods=['GET'])
        return [router]


# --- composition ---


def test_create_agent_returns_an_agent() -> None:
    assert isinstance(create_agent(TestModel()), Agent)


def test_duplicate_feature_names_fail_loudly() -> None:
    with pytest.raises(ValueError, match='unique names'):
        create_agent_app(create_agent(TestModel()), auth=auth, features=[Storage(), Storage()])


def test_a_feature_router_cannot_shadow_a_core_endpoint() -> None:
    class _Shadow(Feature):
        name = 'shadow'

        def routers(self, *, auth: AuthProvider) -> list[APIRouter]:
            router = APIRouter()
            router.add_api_route('/healthz', ping, methods=['GET'])
            return [router]

    with pytest.raises(ValueError, match='reserved'):
        create_agent_app(create_agent(TestModel()), auth=auth, features=[_Shadow()])


def test_a_feature_router_cannot_shadow_a_fastapi_doc_route() -> None:
    # `/docs`, `/redoc` and `/openapi.json` are mounted by FastAPI itself, not by our builtin routers.
    # Starlette resolves the collision by first match, so an unreserved feature route there is dead.
    class _Docs(Feature):
        name = 'docs'

        def routers(self, *, auth: AuthProvider) -> list[APIRouter]:
            router = APIRouter()
            router.add_api_route('/openapi.json', ping, methods=['GET'])
            return [router]

    with pytest.raises(ValueError, match='reserved'):
        create_agent_app(create_agent(TestModel()), auth=auth, features=[_Docs()])


def test_feature_routers_are_mounted() -> None:
    assert make_client(features=[_Routed()]).get('/ping').json() == {'pong': True}


def test_two_features_cannot_claim_the_same_endpoint() -> None:
    # Starlette resolves a collision by first match, so the loser would be silently unreachable.
    class _AlsoRouted(Feature):
        name = 'also-routed'

        def routers(self, *, auth: AuthProvider) -> list[APIRouter]:
            router = APIRouter()
            router.add_api_route('/ping', ping, methods=['GET'])
            return [router]

    match = r"'also-routed' claims route\(s\) already mounted: GET /ping \(feature 'routed'\)"
    with pytest.raises(ValueError, match=match):
        create_agent_app(create_agent(TestModel()), auth=auth, features=[_Routed(), _AlsoRouted()])


def test_two_features_may_split_the_methods_of_one_path() -> None:
    # A collision is a shared (method, path), not a shared path: GET and POST on /items from different
    # features route unambiguously and must both stand -- rejecting them would police more than the merge
    # breaks.
    def _list_items() -> dict[str, bool]:
        return {'listed': True}

    def _add_item() -> dict[str, bool]:
        return {'added': True}

    class _Reader(Feature):
        name = 'reader'

        def routers(self, *, auth: AuthProvider) -> list[APIRouter]:
            router = APIRouter()
            router.add_api_route('/items', _list_items, methods=['GET'])
            return [router]

    class _Writer(Feature):
        name = 'writer'

        def routers(self, *, auth: AuthProvider) -> list[APIRouter]:
            router = APIRouter()
            router.add_api_route('/items', _add_item, methods=['POST'])
            return [router]

    client = make_client(features=[_Reader(), _Writer()])
    assert client.get('/items').json() == {'listed': True}
    assert client.post('/items').json() == {'added': True}


def test_a_public_router_cannot_claim_a_path_another_feature_mounted_behind_auth() -> None:
    # The dangerous ordering: the public route wins first match, and the authed route's path answers
    # anonymously from then on.
    class _PublicPing(Feature):
        name = 'public-ping'

        def public_routers(self, *, auth: AuthProvider) -> list[APIRouter]:
            router = APIRouter()
            router.add_api_route('/ping', ping, methods=['GET'])
            return [router]

    with pytest.raises(ValueError, match='already mounted'):
        create_agent_app(create_agent(TestModel()), auth=auth, features=[_PublicPing(), _Routed()])


def test_two_features_cannot_claim_the_same_websocket_path() -> None:
    # A websocket route carries no HTTP method; the collision is still real (first match wins), and the
    # message names the path without a method.
    async def _socket(sock: WebSocket) -> None: ...  # pragma: no cover -- never connected

    def _ws_feature(feature_name: str) -> Feature:
        class _Streamed(Feature):
            name = feature_name

            def routers(self, *, auth: AuthProvider) -> list[APIRouter]:
                router = APIRouter()
                router.add_api_websocket_route('/stream', _socket)
                return [router]

        return _Streamed()

    features = [_ws_feature('stream-a'), _ws_feature('stream-b')]
    with pytest.raises(ValueError, match=r"already mounted: /stream \(feature 'stream-a'\)"):
        create_agent_app(create_agent(TestModel()), auth=auth, features=features)


def test_a_nested_feature_router_cannot_shadow_a_builtin() -> None:
    # FastAPI 0.137+ keeps a nested include_router as an opaque object with no `.path`; the detector
    # must recurse into it rather than go blind and let a nested `/healthz` compose.
    class _Nested(Feature):
        name = 'nested'

        def routers(self, *, auth: AuthProvider) -> list[APIRouter]:
            inner = APIRouter()
            inner.add_api_route('/healthz', ping, methods=['GET'])
            outer = APIRouter()
            outer.include_router(inner)
            return [outer]

    with pytest.raises(ValueError, match='reserved'):
        create_agent_app(create_agent(TestModel()), auth=auth, features=[_Nested()])


def test_a_feature_cannot_claim_a_trailing_slash_sibling_of_a_builtin() -> None:
    # `/chat/` is a distinct route from `/chat`; a public one there answers anonymously and, once it
    # exists, Starlette stops redirecting `/chat/` callers to the agent. Reserve the slash variants too.
    class _ChatSlash(Feature):
        name = 'chat'

        def public_routers(self, *, auth: AuthProvider) -> list[APIRouter]:
            router = APIRouter(prefix='/chat')
            router.add_api_route('/', ping, methods=['POST'])
            return [router]

    with pytest.raises(ValueError, match='reserved'):
        create_agent_app(create_agent(TestModel()), auth=auth, features=[_ChatSlash()])


def test_one_feature_cannot_mount_a_path_publicly_and_behind_auth() -> None:
    # Same path in both lists silently resolved to whichever list was included first.
    class _Both(Feature):
        name = 'both'

        def routers(self, *, auth: AuthProvider) -> list[APIRouter]:
            router = APIRouter()
            router.add_api_route('/ping', ping, methods=['GET'])
            return [router]

        def public_routers(self, *, auth: AuthProvider) -> list[APIRouter]:
            router = APIRouter()
            router.add_api_route('/ping', ping, methods=['GET'])
            return [router]

    match = r"'both' claims route\(s\) already mounted: GET /ping \(feature 'both'\)"
    with pytest.raises(ValueError, match=match):
        create_agent_app(create_agent(TestModel()), auth=auth, features=[_Both()])


def test_feature_routes_sit_behind_the_apps_auth_by_default() -> None:
    app = create_agent_app(create_agent(TestModel()), auth=strict_auth, features=[_Hooked()])
    client = TestClient(app)
    assert client.get('/manage').status_code == 401
    assert client.get('/manage', headers={'x-api-key': 'k'}).status_code == 200


def test_public_routers_opt_out_of_auth() -> None:
    app = create_agent_app(create_agent(TestModel()), auth=strict_auth, features=[_Hooked()])
    assert TestClient(app).get('/webhook').status_code == 200


def test_chat_and_run_sit_behind_the_apps_auth() -> None:
    # The agent endpoints resolve `auth` as a dependency; without a valid credential neither runs.
    client = TestClient(create_agent_app(create_agent(TestModel()), auth=strict_auth))
    assert client.post('/chat', json=submit('hi')).status_code == 401
    assert client.post('/run', json={'prompt': 'hi'}).status_code == 401
    assert client.post('/chat', json=submit('hi'), headers={'x-api-key': 'k'}).status_code == 200
    assert client.post('/run', json={'prompt': 'hi'}, headers={'x-api-key': 'k'}).status_code == 200


def test_a_public_router_cannot_shadow_a_builtin_endpoint_either() -> None:
    class _Shadow(Feature):
        name = 'shadow'

        def public_routers(self, *, auth: AuthProvider) -> list[APIRouter]:
            router = APIRouter()
            router.add_api_route('/healthz', ping, methods=['GET'])
            return [router]

    with pytest.raises(ValueError, match='reserved'):
        create_agent_app(create_agent(TestModel()), auth=auth, features=[_Shadow()])


def test_feature_lifespans_run_on_startup_and_shutdown() -> None:
    events: list[str] = []

    @asynccontextmanager
    async def _pool() -> AsyncGenerator[None]:
        events.append('start')
        yield
        events.append('stop')

    class _Pooled(Feature):
        name = 'pooled'

        def lifespan(self, app: FastAPI) -> AbstractAsyncContextManager[None] | None:
            return _pool()

    with make_client(features=[_Pooled(), Storage()]):
        assert events == ['start']  # startup entered the feature lifespan
    assert events == ['start', 'stop']  # shutdown exited it


# --- deps_type guard ---


def test_an_agent_with_a_foreign_deps_type_is_rejected_at_build_time() -> None:
    # Without the check this builds and dies on the first tool call with a generic `internal` 500;
    # pyright misses it for an inline `Agent(...)`, so the factory has to say it.
    @dataclass
    class Foreign:
        db: str = 'pool'

    agent = Agent(TestModel(), deps_type=Foreign)

    with pytest.raises(ValueError, match=r'deps_type .*Foreign.* must subclass AppDeps'):
        create_agent_app(agent, auth=auth)  # pyright: ignore[reportArgumentType]


def test_an_agent_without_a_declared_deps_type_is_accepted() -> None:
    # `Agent(model)` leaves deps_type at `object`: the agent has no opinion, and the AppDeps the
    # factory passes at run time still reaches every feature capability.
    agent = Agent(TestModel())

    assert make_client(agent=agent).post('/run', json={'prompt': 'hi'}).status_code == 200


def test_an_agent_declaring_nonetype_deps_is_accepted_like_object() -> None:
    # `Agent(model, deps_type=NoneType)` is the explicit spelling of "no deps opinion"; it must be
    # accepted exactly like the implicit `object` default.
    agent = Agent(TestModel(), deps_type=NoneType)

    assert make_client(agent=agent).post('/run', json={'prompt': 'hi'}).status_code == 200  # pyright: ignore[reportArgumentType]


def test_a_subclass_deps_type_without_a_builder_is_rejected() -> None:
    # Without a builder the factory would construct a plain AppDeps and the product's tools would
    # read fields that are not there -- the same late failure, just one level down.
    @dataclass(kw_only=True)
    class ProductDeps(AppDeps):
        db: str

    agent = Agent(TestModel(), deps_type=ProductDeps)

    with pytest.raises(ValueError, match=r'deps_type .*ProductDeps.* needs a `deps` builder'):
        create_agent_app(agent, auth=auth)  # pyright: ignore[reportArgumentType]


def test_a_parameterised_deps_type_is_inspected_by_its_origin() -> None:
    # `get_origin` unwraps a generic alias so the subclass check sees the real class.
    agent = Agent(TestModel(), deps_type=list[int])

    with pytest.raises(ValueError, match=r'deps_type .*list.* must subclass AppDeps'):
        create_agent_app(agent, auth=auth)  # pyright: ignore[reportArgumentType]


def test_a_subclass_deps_type_with_a_builder_is_accepted() -> None:
    # The path the `deps` parameter exists to serve: a strict AppDeps subclass with a builder
    # supplied must build cleanly, not just fail loudly when the builder is missing.
    @dataclass(kw_only=True)
    class ProductDeps(AppDeps):
        db: str

    def build(request: Request, user: UserContext) -> ProductDeps:
        return ProductDeps(user=user, db='pool')

    agent = Agent(TestModel(), deps_type=ProductDeps)

    create_agent_app(agent, auth=auth, deps=build)  # must not raise
