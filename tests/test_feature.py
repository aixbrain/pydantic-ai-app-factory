import asyncio
from abc import ABC
from collections.abc import AsyncGenerator, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

import pytest
from fastapi import APIRouter, FastAPI
from pydantic import BaseModel

from pydantic_ai_app_factory import AuthProvider, Feature, UserContext


def _auth() -> UserContext:
    return UserContext(user_id='u1')


def test_a_feature_must_declare_a_name() -> None:
    with pytest.raises(TypeError, match='name'):
        # Defining the subclass is the tested side effect; pyright cannot see that.
        class Nameless(Feature):  # pyright: ignore[reportUnusedClass]
            pass


def test_a_name_must_be_a_slug() -> None:
    with pytest.raises(TypeError, match='slug'):

        class Slashed(Feature):  # pyright: ignore[reportUnusedClass]
            name = 'my/feature'


def test_an_abstract_intermediate_base_needs_no_name() -> None:
    class Database(Feature, ABC):
        pass

    class Concrete(Database):
        name = 'concrete'

    assert Concrete().name == 'concrete'


def test_a_concrete_subclass_of_an_abstract_base_still_needs_a_name() -> None:
    class Database(Feature, ABC):
        pass

    with pytest.raises(TypeError, match='name'):

        class Nameless(Database):  # pyright: ignore[reportUnusedClass]
            pass


def test_a_minimal_feature_contributes_nothing_by_default() -> None:
    class Minimal(Feature):
        name = 'minimal'

    feature = Minimal()
    assert feature.name == 'minimal'
    assert feature.extras_type is None
    assert feature.capabilities() == []
    assert feature.routers(auth=_auth) == []
    assert feature.lifespan(FastAPI()) is None


def test_default_health_reports_nothing() -> None:
    class Minimal(Feature):
        name = 'minimal'

    assert asyncio.run(Minimal().health()) is None


def test_a_feature_contributes_its_routers_lifespan_extras_and_health() -> None:
    router = APIRouter()

    @asynccontextmanager
    async def _pool() -> AsyncGenerator[None]:
        yield

    class Extras(BaseModel):
        case_id: str

    class Full(Feature):
        name = 'full'
        extras_type = Extras

        def routers(self, *, auth: AuthProvider) -> list[APIRouter]:
            return [router]

        def lifespan(self, app: FastAPI) -> AbstractAsyncContextManager[None] | None:
            return _pool()

        async def health(self) -> Mapping[str, Any] | None:
            return {'backend': 'ok'}

    feature = Full()
    assert feature.extras_type is Extras
    assert feature.routers(auth=_auth) == [router]
    assert feature.lifespan(FastAPI()) is not None
    assert asyncio.run(feature.health()) == {'backend': 'ok'}


def test_health_that_raises_signals_unhealthy() -> None:
    class Sick(Feature):
        name = 'sick'

        async def health(self) -> Mapping[str, Any] | None:
            raise RuntimeError('db down')

    with pytest.raises(RuntimeError, match='db down'):
        asyncio.run(Sick().health())


def test_default_config_reports_nothing() -> None:
    class Minimal(Feature):
        name = 'minimal'

    assert Minimal().config() is None
