"""How a normalized `AgentError` reaches the caller and the log, on either transport."""

from __future__ import annotations

import logging
from typing import Any, cast

from fastapi import Request
from fastapi.responses import JSONResponse

from pydantic_ai_app_factory.errors import AgentError

logger = logging.getLogger(__name__)


def error_body(error: AgentError) -> dict[str, Any]:
    """The wire shape of an error: the JSON body on `/run` and feature routes, the `data-error` chunk on `/chat`.

    `retryable` rides along even where the status implies it: mid-stream there is no status left to send.
    """
    return {'code': error.code, 'message': error.message, 'retryable': error.retryable}


def log_agent_error(error: AgentError, cause: Exception, what: str) -> None:
    """Log a failure at the level its status implies.

    The level tracks whether *this* service failed: an expected 4xx outcome is INFO without a
    traceback; a transient backend hiccup (retryable 5xx) is WARNING with the cause, a rate to watch
    that still carries the detail the envelope hides; only a non-retryable 5xx is ERROR with a
    traceback. `what` names the step that failed (`agent run`, `deps resolution`).
    """
    if error.http_status < 500:
        logger.info('%s rejected: %s', what, error.code)
    elif error.retryable:
        logger.warning('%s failed transiently: %s', what, error.code, exc_info=cause)
    else:
        logger.error('%s failed', what, exc_info=cause)


async def handle_agent_error(request: Request, exc: Exception) -> JSONResponse:
    """Render an `AgentError` raised anywhere in the app, feature routes included."""
    error = cast(AgentError, exc)
    return JSONResponse(status_code=error.http_status, content=error_body(error))
