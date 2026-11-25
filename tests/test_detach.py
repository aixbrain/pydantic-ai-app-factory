import asyncio
import logging
from collections.abc import AsyncIterator

import pytest

from pydantic_ai_app_factory._detach import DetachedRuns


def _others() -> list[asyncio.Task[object]]:
    return [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]


def test_an_error_in_the_stream_reaches_the_reader() -> None:
    async def failing() -> AsyncIterator[int]:
        yield 1
        raise ValueError('boom')

    async def scenario() -> list[int]:
        received: list[int] = []
        with pytest.raises(ValueError, match='boom'):
            async for item in DetachedRuns().start(failing()):
                received.append(item)
        return received

    assert asyncio.run(scenario()) == [1]


@pytest.mark.parametrize('started', [True, False], ids=['closed-reader', 'never-started-reader'])
def test_an_error_after_the_reader_left_is_logged(started: bool, caplog: pytest.LogCaptureFixture) -> None:
    # Nobody is left to raise it to. A reader Starlette dropped unstarted must end the reading as a closed one does.
    async def scenario() -> None:
        gate = asyncio.Event()

        async def failing() -> AsyncIterator[int]:
            yield 1
            await gate.wait()
            raise ValueError('boom')

        reader = DetachedRuns().start(failing())
        if started:
            assert await anext(reader) == 1
            await reader.aclose()
        del reader
        gate.set()
        await asyncio.gather(*_others())

    with caplog.at_level(logging.ERROR):
        asyncio.run(scenario())
    assert 'agent run failed after its client disconnected' in caplog.text


def test_drain_without_runs_returns_at_once() -> None:
    asyncio.run(DetachedRuns().drain())


def test_a_cancelled_drain_cancels_the_runs_and_waits_for_them() -> None:
    async def scenario() -> list[str]:
        seen: list[str] = []
        release = asyncio.Event()

        async def waiting_forever() -> AsyncIterator[int]:
            try:
                await asyncio.Event().wait()
                yield 1
            except asyncio.CancelledError:
                await release.wait()  # unwinding that has to finish before the drain may return
                seen.append('cancelled')
                raise

        runs = DetachedRuns()
        reader = runs.start(waiting_forever())
        drain = asyncio.create_task(runs.drain())
        await asyncio.sleep(0)
        drain.cancel()
        for _ in range(3):
            await asyncio.sleep(0)
        assert not drain.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await drain
        del reader
        return seen

    assert asyncio.run(scenario()) == ['cancelled']
