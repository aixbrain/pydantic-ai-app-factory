"""An example feature with every `Feature` member filled in, and the app that composes it.

The feature is named `notes` and keeps notes for the caller. The agent gets two tools, `remember`
to write a note and `recall` to read back the existing ones, and two endpoints list the notes and
the notebooks, so the same data is reachable without asking the model. The notebook a call works
in arrives with the request as an extras field. Who the notes belong to is decided once, by the
composition at the bottom of this file, as either the caller's whole tenant or the caller alone.

`NoteStore` is a dict. It lives as long as the process and every note is gone on restart. A real
feature keeps a pool, a client or a bucket there instead, opened in `lifespan()`, where this one
sets up its dict.

Run it with `uv run -m pydantic_ai_app_factory_examples.notes_feature`.
"""

from collections.abc import AsyncGenerator, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypeAlias

from fastapi import APIRouter, Depends, FastAPI
from pydantic import BaseModel
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AgentToolset, FunctionToolset

from pydantic_ai_app_factory import (
    AgentError,
    AppDeps,
    AuthProvider,
    Feature,
    UserContext,
    create_agent,
    create_agent_app,
)

MAX_NOTES_PER_TURN = 5
"""How many notes one run may write before the feature refuses the next one."""

NoteOwner: TypeAlias = Literal['tenant', 'user']
"""Whether a whole tenant shares its notes, or every caller keeps their own."""


def owner_of(user: UserContext, owned_by: NoteOwner) -> str:
    """Return the store key for `owned_by`, the caller's tenant or the caller alone."""
    if owned_by == 'user':
        return user.user_id
    if user.tenant_id is None:
        # Falling back to the user here would look like it works and quietly change the data model.
        raise AgentError('notes/no-tenant', 'This deployment shares notes per tenant.', http_status=403)
    return user.tenant_id


class Note(BaseModel):
    """One stored note; colleagues share a notebook, so it records who wrote it."""

    author: str
    text: str


class NoteStore:
    """In-memory note storage, standing in for the pool or client a real feature would open."""

    def __init__(self) -> None:
        self._notes: dict[tuple[str, str], list[Note]] = {}
        self._open = False

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False
        self._notes.clear()

    def check(self) -> None:
        """Raise unless the store is usable, as a real one would by pinging its connection."""
        if not self._open:
            raise RuntimeError('The note store is closed.')

    def add(self, owner: str, notebook: str, author: str, text: str) -> None:
        self._notes.setdefault((owner, notebook), []).append(Note(author=author, text=text))

    def notes(self, owner: str, notebook: str) -> list[Note]:
        return list(self._notes.get((owner, notebook), ()))

    def notebooks(self, owner: str) -> list[str]:
        return sorted(notebook for stored_owner, notebook in self._notes if stored_owner == owner)


class NotesExtras(BaseModel):
    """The feature's per-request extras, naming the notebook this call works in."""

    notebook: str = 'default'


@dataclass
class NotesCapability(AbstractCapability[AppDeps]):
    """The feature's tools over one owner's notebook. `remember` writes a note and `recall` reads them."""

    store: NoteStore
    owned_by: NoteOwner

    def get_toolset(self) -> AgentToolset[AppDeps]:
        store, owned_by = self.store, self.owned_by

        def notebook(ctx: RunContext[AppDeps]) -> str:
            return str(ctx.deps.extras.get('notebook', 'default'))

        # Each tool's docstring is the description the model reads when it decides to call it.
        async def remember(ctx: RunContext[AppDeps], text: str) -> str:
            """Store a note in the caller's notebook."""
            # One capability instance serves every run, so a per-run tally belongs in `scratch`.
            written = ctx.deps.scratch.get('notes/written', 0) + 1
            if written > MAX_NOTES_PER_TURN:
                raise AgentError('notes/turn-limit', f'At most {MAX_NOTES_PER_TURN} notes per turn.', http_status=429)
            ctx.deps.scratch['notes/written'] = written
            store.add(owner_of(ctx.deps.user, owned_by), notebook(ctx), ctx.deps.user.user_id, text)
            return f'Stored. {written} note(s) this turn.'

        async def recall(ctx: RunContext[AppDeps]) -> list[str]:
            """List every note in this notebook, with who wrote it."""
            notes = store.notes(owner_of(ctx.deps.user, owned_by), notebook(ctx))
            return [f'{note.author}: {note.text}' for note in notes]

        return FunctionToolset[AppDeps]([remember, recall])


class Notes(Feature):
    """The `notes` feature with its capability, routes, startup, health check and config."""

    name = 'notes'
    extras_type = NotesExtras

    def __init__(self, store: NoteStore, *, owned_by: NoteOwner = 'tenant') -> None:
        self._store = store
        # Annotated, or pyright widens the literal to `str` and every use of it stops checking.
        self._owned_by: NoteOwner = owned_by

    def capabilities(self) -> list[AbstractCapability[AppDeps]]:
        return [NotesCapability(self._store, self._owned_by)]

    def routers(self, *, auth: AuthProvider) -> list[APIRouter]:
        router = APIRouter(prefix='/notes')

        # The factory mounts these behind the app's auth; `Depends(auth)` reads the admitted user.
        @router.get('')
        async def index(user: Annotated[UserContext, Depends(auth)], notebook: str = 'default') -> list[Note]:
            return self._store.notes(owner_of(user, self._owned_by), notebook)

        @router.get('/notebooks')
        async def notebooks(user: Annotated[UserContext, Depends(auth)]) -> list[str]:
            return self._store.notebooks(owner_of(user, self._owned_by))

        return [router]

    def public_routers(self, *, auth: AuthProvider) -> list[APIRouter]:
        router = APIRouter(prefix='/notes')

        @router.get('/about')
        async def about() -> dict[str, Any]:
            return {'feature': self.name, 'max_notes_per_turn': MAX_NOTES_PER_TURN}

        return [router]

    def lifespan(self, app: FastAPI) -> AbstractAsyncContextManager[None]:
        return self._store_lifespan(app)

    @asynccontextmanager
    async def _store_lifespan(self, app: FastAPI) -> AsyncGenerator[None]:
        self._store.open()
        app.state.notes = self._store
        try:
            yield
        finally:
            # In `finally`, because a later feature failing to start arrives here as an exception.
            self._store.close()

    async def health(self) -> Mapping[str, Any]:
        """Fail the readiness probe when the store is unusable. Report no data, since `/readyz` is public."""
        # A mapping means healthy and raising means unhealthy. The body carries no verdict.
        self._store.check()
        return {'store': 'ready'}

    def config(self) -> Mapping[str, Any]:
        # `owned_by` belongs here so a frontend can label the notes as the caller's own or the team's.
        return {
            'owned_by': self._owned_by,
            'max_notes_per_turn': MAX_NOTES_PER_TURN,
            'default_notebook': 'default',
        }


def auth() -> UserContext:
    """Stands in for real authentication. Every request is the same demo user in the same tenant."""
    return UserContext(user_id='demo', tenant_id='demo-tenant')


agent = create_agent('test', instructions='Keep notes for the user.')
# Everyone in the tenant shares its notes; `owned_by='user'` gives each caller their own instead.
app = create_agent_app(agent, auth=auth, features=[Notes(NoteStore(), owned_by='tenant')])


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='127.0.0.1', port=8000)
