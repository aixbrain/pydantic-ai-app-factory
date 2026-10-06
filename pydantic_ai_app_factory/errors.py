"""Errors that a run may surface to its caller."""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

from pydantic_ai.exceptions import FallbackExceptionGroup, ModelAPIError, ModelHTTPError

logger = logging.getLogger(__name__)


def _rebuild(cls: type[AgentError], args: tuple[Any, ...]) -> AgentError:
    """Recreate an error without re-running `__init__`, whose signature a subclass may have changed."""
    error = Exception.__new__(cls)
    Exception.__init__(error, *args)
    return error


class AgentError(Exception):
    """An error whose `code` and `message` are safe to surface to the caller."""

    code: str
    """Stable identifier the caller keys on.

    Un-namespaced codes (`internal`, `model-unavailable`, `model-unauthorized`, `file-unsupported`) are
    reserved for this package; features and products namespace theirs as `<feature>/<code>`.
    """
    message: str
    """Human-readable text safe to show the caller."""
    retryable: bool
    """Whether the caller may retry the same request unchanged."""
    http_status: int
    """Status for the JSON endpoints; defaults to 503 when retryable and 500 otherwise."""

    def __init__(self, code: str, message: str, *, retryable: bool = False, http_status: int | None = None) -> None:
        self.code = code
        self.message = message
        self.retryable = retryable
        self.http_status = http_status if http_status is not None else (503 if retryable else 500)
        super().__init__(message)

    def __reduce__(self) -> tuple[Callable[..., AgentError], tuple[Any, ...], dict[str, Any]]:
        # Pickle restores the attributes from the state dict: the base Exception would otherwise
        # pickle only `message` (its sole `super().__init__` argument) and drop `code`/`retryable`.
        return (_rebuild, (self.__class__, self.args), self.__dict__)

    def __str__(self) -> str:
        return f'{self.code}: {self.message}'


ErrorHandlers: TypeAlias = Sequence[tuple[type[Exception], AgentError | Callable[[Exception], AgentError]]]
"""Ordered `(exception type, handler)` pairs; the first matching type wins.

A handler is a fixed `AgentError` or a callable that derives one from the exception.
"""


def to_agent_error(exc: Exception, error_handlers: ErrorHandlers | None = None) -> AgentError:
    """Normalize an exception into a client-safe `AgentError`.

    The first `error_handlers` entry whose type matches translates it, to a fixed `AgentError` or a
    callable that derives one from the exception. An `AgentError` otherwise passes through; a model
    provider rejecting the credentials maps to `model-unauthorized`; a transient model-backend failure
    maps to `model-unavailable`; anything else becomes a generic `internal` error so no unrecognized
    exception leaks its message to the caller.

    This is total: a callable handler that itself raises falls back to `internal` rather than escape as
    an unhandled exception, so a buggy handler cannot break the no-leak contract.
    """
    for exc_type, handler in error_handlers or ():
        if isinstance(exc, exc_type):
            if isinstance(handler, AgentError):
                # The fixed handler is one instance for the life of the app; raising it would hang this
                # request's `__cause__` and `__traceback__` on it for every later request to retain and
                # log. A copy carries the wire fields (`__reduce__` above) and nothing of the request.
                return copy.copy(handler)
            try:
                return handler(exc)
            except Exception:
                logger.exception('error handler for %s raised; falling back to internal', exc_type.__name__)
                return AgentError('internal', 'An internal error occurred.')
    if isinstance(exc, AgentError):
        return exc
    if _provider_rejected_credentials(exc):
        # Not retryable and not our fault: 502 says "the upstream refused us", where 500 would say
        # "we broke" and 503 would invite a retry against a key that will never work. `/run` logs a
        # non-retryable 5xx at ERROR with a traceback on every such request; that is deliberate here
        # too -- the deployment is misconfigured and needs attention, even though the fault is upstream.
        return AgentError('model-unauthorized', 'The model provider rejected the credentials.', http_status=502)
    if _model_backend_unavailable(exc):
        return AgentError('model-unavailable', 'The model is temporarily unavailable.', retryable=True)
    return AgentError('internal', 'An internal error occurred.')


def _provider_rejected_credentials(exc: Exception) -> bool:
    """Whether the model provider refused the deployment's credentials, on every model that was tried.

    An approximation over HTTP status, because providers pass their raw status through: most report a
    bad key as 401 (OpenAI, Anthropic, Azure, Groq, Mistral), some as 403 (Bedrock, Vertex on a missing
    IAM permission). Two cases are out of reach without reading the response body, which this package
    deliberately does not do -- Gemini reports a bad key as 400, and OpenAI reports an exhausted balance
    as 429; map those in `error_handlers` if a product needs them. With `FallbackModel` the group has
    to be credential failures throughout: one model down and another rejecting the key is still an
    outage worth retrying.
    """
    failures = exc.exceptions if isinstance(exc, FallbackExceptionGroup) else (exc,)
    return all(isinstance(e, ModelHTTPError) and e.status_code in (401, 403) for e in failures)


def _model_backend_unavailable(exc: Exception) -> bool:
    """Whether the model provider failed transiently (a client-side 4xx is a bug, not this)."""
    if isinstance(exc, ModelHTTPError):
        return exc.status_code == 429 or exc.status_code >= 500
    return isinstance(exc, ModelAPIError | FallbackExceptionGroup)
