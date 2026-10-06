# Requests and Identity

This guide shows you how to authenticate callers: writing the auth provider,
deciding what belongs on the user object, carrying richer per-user data, and
authenticating surfaces that have no user accounts at all.

Prerequisites: the app from [Building an app](building-an-app.md). The
factory takes your provider as `auth=` and declares it as a dependency on
every guarded route; all actual verification logic is yours.

## Writing an auth provider

An `AuthProvider` is any FastAPI dependency, sync or async, returning a
`UserContext` -- and a real, importable type
(`from pydantic_ai_app_factory import AuthProvider`), so your own code can
name it too, the `auth` argument of a feature's `routers()` included.
[Sub-dependencies](https://fastapi.tiangolo.com/tutorial/dependencies/sub-dependencies/)
resolve as usual, so header parsing and OAuth2 scopes compose normally:

```python
from typing import Annotated

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer

from pydantic_ai_app_factory import UserContext, create_agent_app

bearer = OAuth2PasswordBearer(tokenUrl='token')


async def auth(token: Annotated[str, Depends(bearer)]) -> UserContext:
    claims = decode_and_verify(token)  # your JWT validation
    return UserContext(
        user_id=claims['sub'],
        tenant_id=claims.get('org'),
        scopes=frozenset(claims.get('scopes', ())),
    )


app = create_agent_app(agent, auth=auth)
```

What you end up with: every run sees the returned object as `ctx.deps.user`,
and every guarded route rejects requests your provider rejects. Feature
routes get the same provider handed to their `routers()`, so a handler that
reads the user declares `Depends(auth)` on the very object the guard already
ran ([Feature routes](features.md#feature-routes)).

## What belongs on `UserContext`

`user_id` is required on every request, even for apps without real user
accounts. On a public surface, mint a session-scoped id in the auth provider
rather than leaving it empty: downstream features key on it for storage scope,
per-user budgets, and tracing. Keep `UserContext` itself to identity and
permissions (`user_id`, `tenant_id`, `scopes`) -- it is the input to security
decisions and nothing else. Preferences that describe the request rather than
the caller -- a language tag, say -- belong in the
[request extras](request-extras.md).

## Carrying more than identity

A provider may return a subclass carrying richer data. Declare per-user
credentials on it as `pydantic.SecretStr`: deps are logged and serialized, so
a plain `str` would leak through reprs and payloads. A tool that needs the
richer type narrows it back
(`assert isinstance(ctx.deps.user, PortalUser)`):

```python
from dataclasses import dataclass

from pydantic import SecretStr


@dataclass(frozen=True, kw_only=True)
class PortalUser(UserContext):
    portal_token: SecretStr
```

## Public surfaces without accounts

A public chat that authenticates the service rather than the user -- an API
key, identity self-asserted by the client -- is just a provider that verifies
the key header and mints the session `UserContext` itself. Such a provider may
read the request body: Starlette caches the parsed body, so the endpoint's own
parse still works.

```python
import uuid

from fastapi import Header, HTTPException, Request


async def auth(request: Request, x_api_key: Annotated[str, Header()]) -> UserContext:
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail='invalid API key')
    body = await request.json()
    chat_id = body.get('id')  # the Vercel protocol's chat id; absent on /run
    return UserContext(user_id=f'session:{chat_id or uuid.uuid4()}')
```

This provider assumes a JSON body, which `/chat` and `/run` always have;
guarding a feature's body-less GET route would need the parse wrapped in a
fallback. Note what deriving the id from the chat means: everything scoped by
`user_id` -- stored conversations, per-user budgets, traces -- then separates
per conversation rather than per caller.

## Where your provider sits

For debugging it helps to know the order of a request:

1. Your provider resolves the caller. On `/run`, FastAPI decodes the JSON
   body first, so unparseable JSON is a 422 before your provider runs; on
   `/chat` the provider runs first and the body is parsed after it.
2. Either way the provider runs before the body is validated against any
   schema. A 401 can therefore mask a malformed request: schema errors
   surface only once auth passes.
3. Body and extras are validated, rejected with 422 before the run starts.
4. The factory builds `AppDeps` ([fields](api-reference.md#appdeps)) -- or
   calls your `deps` builder -- and the run executes.
5. The [error boundary](error-contract.md) normalizes whatever failed, in the
   builder or in the run.

So by the time a tool reads `ctx.deps.user`, the caller is authenticated and
the request fully validated -- and an auth failure means steps 3 to 5 never
happened.
