"""Keep a `/chat` run going after its client disconnects."""

from __future__ import annotations

import asyncio
import logging
import weakref
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class _End:
    error: Exception | None = None


@dataclass
class _Relay(Generic[T]):
    """What a run's consumer task hands to the response, and whether the response still reads it."""

    queue: asyncio.Queue[T | _End] = field(default_factory=lambda: asyncio.Queue())
    reading: bool = True


async def _consume(stream: AsyncIterator[T], relay: _Relay[T]) -> None:
    error = None
    try:
        async for item in stream:
            if relay.reading:
                relay.queue.put_nowait(item)
    except Exception as exc:
        error = exc
        if not relay.reading:
            logger.error('agent run failed after its client disconnected', exc_info=exc)
    finally:
        relay.queue.put_nowait(_End(error))


async def _read(relay: _Relay[T]) -> AsyncGenerator[T]:
    try:
        while not isinstance(item := await relay.queue.get(), _End):
            yield item
    finally:
        relay.reading = False
    if item.error is not None:
        raise item.error


@dataclass
class DetachedRuns:
    """The `/chat` runs that outlive their request.

    A task stays here until it ends, because the event loop keeps only a weak reference to it.
    """

    _tasks: set[asyncio.Task[None]] = field(default_factory=set[asyncio.Task[None]])

    def start(self, stream: AsyncIterator[T]) -> AsyncGenerator[T]:
        """Consume `stream` on a task started here and return the response's reader of it.

        Closing the reader, as Starlette does on a disconnect, ends the reading only; the task is
        started before the response so a client that leaves before the first chunk does not stop it.
        """
        relay = _Relay[T]()
        task = asyncio.create_task(_consume(stream, relay))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        reader = _read(relay)
        # A reader Starlette never starts never reaches its `finally`, so dropping it ends the reading too.
        weakref.finalize(reader, setattr, relay, 'reading', False)
        return reader

    async def drain(self) -> None:
        """Wait for the runs still in flight, as the server waits for connected ones.

        Only a cancelled shutdown cancels them, and it waits for them to unwind before the features close.
        """
        pending = list(self._tasks)
        if not pending:
            return
        try:
            await asyncio.wait(pending)
        except asyncio.CancelledError:
            for task in pending:
                task.cancel()
            await asyncio.wait(pending)
            raise
