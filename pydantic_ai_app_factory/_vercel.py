"""Render errors safely onto the Vercel AI event stream."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from pydantic_ai.ui.vercel_ai import VercelAIAdapter, VercelAIEventStream
from pydantic_ai.ui.vercel_ai.response_types import BaseChunk, DataChunk, ErrorChunk
from starlette.requests import Request

from pydantic_ai_app_factory._envelope import error_body, log_agent_error
from pydantic_ai_app_factory._files import message_media_types
from pydantic_ai_app_factory.deps import AppDeps
from pydantic_ai_app_factory.errors import ErrorHandlers, to_agent_error


@dataclass
class VercelEventStream(VercelAIEventStream[AppDeps, Any]):
    """Event stream whose errors reach the client as a safe `AgentError`."""

    error_handlers: ErrorHandlers | None = None

    async def on_error(self, error: Exception) -> AsyncIterator[BaseChunk]:
        agent_error = to_agent_error(error, self.error_handlers)
        log_agent_error(agent_error, error, 'agent run')
        # Delegate, so the base's own error handling still runs -- it flushes tool calls whose input
        # was streamed but never announced, and sets the error finish reason. Only the chunk that
        # would leak `str(error)` is swapped.
        async for chunk in super().on_error(error):
            if isinstance(chunk, ErrorChunk):
                # The message is user-safe by contract, so the machine-readable chunk carries it too:
                # a client keying on `data-error` renders the failure without stitching two chunks
                # back together. The `ErrorChunk` stays for SDK consumers that only understand it.
                yield DataChunk(type='data-error', data=error_body(agent_error))
                yield ErrorChunk(error_text=agent_error.message)
            else:
                yield chunk


@dataclass
class VercelAdapter(VercelAIAdapter[AppDeps, Any]):
    """Vercel AI adapter that builds a `VercelEventStream`."""

    error_handlers: ErrorHandlers | None = None

    @classmethod
    async def from_request(cls, request: Request, **kwargs: Any) -> VercelAdapter:
        # Narrow the upstream return type, which erases `Self` to the base adapter.
        adapter = await super().from_request(request, **kwargs)
        assert isinstance(adapter, cls)
        return adapter

    @cached_property
    def file_media_types(self) -> list[str]:
        """The media type of every file the run will send, taken from the sanitized messages.

        Sanitizing first, as `run_stream` does, so a file it strips is not what a request is refused for.
        """
        messages = self.sanitize_messages(self.messages, deferred_tool_results=self.deferred_tool_results)
        return message_media_types(messages)

    def build_event_stream(self) -> VercelAIEventStream[AppDeps, Any]:
        return VercelEventStream(
            self.run_input,
            accept=self.accept,
            sdk_version=self.sdk_version,
            server_message_id=self.server_message_id,
            error_handlers=self.error_handlers,
        )
