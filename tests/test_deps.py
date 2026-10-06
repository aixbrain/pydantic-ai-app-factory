from dataclasses import FrozenInstanceError, dataclass, field
from typing import Annotated

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from pydantic_ai_app_factory import AppDeps, AuthProvider, UserContext


def test_user_context_needs_only_an_id() -> None:
    user = UserContext(user_id='u1')
    assert (user.tenant_id, user.scopes) == (None, frozenset())


def test_user_context_is_frozen() -> None:
    # The auth provider's verdict is settled once the request enters the run; a feature
    # that could edit it could edit its own permissions. pyright rejects the assignment
    # at author time, which is the real guarantee -- this pins that runtime agrees.
    user = UserContext(user_id='u1')
    with pytest.raises(FrozenInstanceError):
        user.user_id = 'u2'  # pyright: ignore[reportAttributeAccessIssue]


def test_core_deps_start_without_feature_state() -> None:
    deps = AppDeps(user=UserContext(user_id='u1'))
    assert deps.extras == {}
    assert deps.scratch == {}


def test_core_deps_hold_the_providers_own_user_type() -> None:
    """The pass-through guarantee: the provider's instance arrives, not a contract-shaped copy."""

    @dataclass(frozen=True)
    class RichUser(UserContext):
        memberships: tuple[str, ...] = ()

    provider_user = RichUser(user_id='u1', memberships=('acme',))
    deps = AppDeps(user=provider_user)

    assert deps.user is provider_user


def test_a_credential_on_a_provider_subclass_does_not_reach_a_repr() -> None:
    """The documented way to carry a per-user credential: `SecretStr` on the provider's own type."""

    @dataclass(frozen=True)
    class TokenUser(UserContext):
        access_token: SecretStr = field(default_factory=lambda: SecretStr(''))

    deps = AppDeps(user=TokenUser(user_id='u1', access_token=SecretStr('super-secret')))

    assert 'super-secret' not in repr(deps)
    assert isinstance(deps.user, TokenUser)
    assert deps.user.access_token.get_secret_value() == 'super-secret'


def test_scratch_is_not_shared_between_requests() -> None:
    # Capabilities are singletons across concurrent runs, so a shared mutable default here
    # would leak one caller's run state into another's.
    first = AppDeps(user=UserContext(user_id='u1'))
    second = AppDeps(user=UserContext(user_id='u2'))

    first.scratch['storage'] = 'loaded'

    assert second.scratch == {}


@dataclass(frozen=True)
class _TokenUser(UserContext):
    access_token: SecretStr = field(default_factory=lambda: SecretStr(''))


def _resolve_user() -> UserContext:
    # Placeholder dependency; each test overrides it with the AuthProvider under test.
    raise NotImplementedError


def read_whoami(user: Annotated[UserContext, Depends(_resolve_user)]) -> dict[str, str]:
    # The provider's own subclass reaches the route, and its credential is usable here.
    assert isinstance(user, _TokenUser)
    return {'user_id': user.user_id, 'token_len': str(len(user.access_token.get_secret_value()))}


def _client_for(provider: AuthProvider) -> TestClient:
    app = FastAPI()
    app.add_api_route('/whoami', read_whoami, methods=['GET'])
    app.dependency_overrides[_resolve_user] = provider
    return TestClient(app)


def test_async_auth_provider_delivers_its_user_subclass() -> None:
    async def provider() -> UserContext:
        return _TokenUser(user_id='u1', access_token=SecretStr('downstream-token'))

    body = _client_for(provider).get('/whoami').json()
    assert body == {'user_id': 'u1', 'token_len': str(len('downstream-token'))}


def test_sync_auth_provider_is_accepted_too() -> None:
    def provider() -> UserContext:
        return _TokenUser(user_id='u2', access_token=SecretStr('x'))

    assert _client_for(provider).get('/whoami').json()['user_id'] == 'u2'


def test_app_deps_is_keyword_only() -> None:
    # A subclass must be able to add a required field after `extras`/`scratch`, which have defaults;
    # a dataclass only allows that when every field is keyword-only.
    with pytest.raises(TypeError, match='positional'):
        AppDeps(UserContext(user_id='u1'))  # type: ignore[misc]


def test_a_product_can_subclass_app_deps_with_a_required_field() -> None:
    # Plain `@dataclass`, not `kw_only=True`: this only passes because the BASE is keyword-only,
    # which is what lets a required field follow the defaulted `extras`/`scratch`.
    @dataclass
    class ProductDeps(AppDeps):
        db: str

    deps = ProductDeps(user=UserContext(user_id='u1'), db='pool')

    assert (deps.db, deps.extras, deps.scratch) == ('pool', {}, {})
