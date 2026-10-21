"""Errors that a run may surface to its caller."""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

from pydantic_ai.exceptions import (
    ConcurrencyLimitExceeded,
    ContentFilterError,
    FallbackExceptionGroup,
    ModelAPIError,
    ModelContextWindowExceededError,
    ModelHTTPError,
    ModelQuotaExceededError,
    RunCancelled,
    RunTimedOut,
)

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

    Un-namespaced codes (`internal`, `model-unavailable`, `model-unauthorized`, `model-quota-exceeded`,
    `model-context-exceeded`, `model-refused`, `run-timeout`, `run-cancelled`, `file-unsupported`) are
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
    provider rejecting the credentials maps to `model-unauthorized`, an exhausted quota to
    `model-quota-exceeded`, a request too long for the model's context window to `model-context-exceeded`
    and the provider's content filter refusing to `model-refused`; a run past its `timeout` maps to
    `run-timeout` and any other cancellation to `run-cancelled`; a transient model-backend failure or a
    full concurrency queue maps to `model-unavailable`; anything else becomes a generic `internal` error
    so no unrecognized exception leaks its message to the caller.

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
    if _every_failure_is(exc, ModelQuotaExceededError):
        # 502 and not retryable like `model-unauthorized`: the account stays empty until someone tops it up,
        # and the deployment needs attention. A 402 would tell the end user to pay.
        return AgentError('model-quota-exceeded', 'The model provider account is out of quota.', http_status=502)
    if _every_failure_is(exc, ModelContextWindowExceededError):
        # A 4xx so it logs at INFO, because the remedy is a shorter history or a new conversation, and not
        # retryable, because the same request overflows again. Avoiding the overflow is the agent's job.
        return AgentError('model-context-exceeded', 'The conversation is too long for the model.', http_status=413)
    if isinstance(exc, ContentFilterError):
        # Not retryable, because a retry of the same request is unlikely to help. A 4xx because the remedy is
        # usually a different request, and so the refusal logs at INFO, since nothing in the deployment needs attention.
        return AgentError('model-refused', 'The model provider declined to answer this request.', http_status=400)
    if isinstance(exc, RunTimedOut):
        # 504 because the run exceeded the deployment's time limit, retryable because the usual cause, a
        # hung provider or tool, is transient.
        return AgentError('run-timeout', 'The agent run timed out.', retryable=True, http_status=504)
    if isinstance(exc, RunCancelled):
        # The app's own code stopped the run on purpose. A 4xx so it logs at INFO instead of paging anyone,
        # 409 because no status fits a deliberate stop better, and not retryable because whatever stopped
        # the run will likely stop it again.
        return AgentError('run-cancelled', 'The agent run was cancelled.', http_status=409)
    if _model_backend_unavailable(exc):
        return AgentError('model-unavailable', 'The model is temporarily unavailable.', retryable=True)
    return AgentError('internal', 'An internal error occurred.')


def _failures(exc: Exception) -> Sequence[Exception]:
    """The failure, or with `FallbackModel` the failure of every model that was tried."""
    return exc.exceptions if isinstance(exc, FallbackExceptionGroup) else (exc,)


def _provider_rejected_credentials(exc: Exception) -> bool:
    """Whether the model provider refused the deployment's credentials, on every model that was tried.

    An approximation over HTTP status, because providers pass their raw status through and pydantic-ai
    has no error category for it: most report a bad key as 401 (OpenAI, Anthropic, Azure, Groq, Mistral),
    some as 403 (Bedrock, Vertex on a missing IAM permission). Gemini reports a bad key as 400, out of
    reach without reading the response body, which this package deliberately does not do; map it in
    `error_handlers` if a product needs it. With `FallbackModel` the group has to be credential failures
    throughout: one model down and another rejecting the key is still an outage worth retrying.
    """
    return all(isinstance(e, ModelHTTPError) and e.status_code in (401, 403) for e in _failures(exc))


def _every_failure_is(exc: Exception, category: type[ModelAPIError]) -> bool:
    """Whether pydantic-ai put the failure in `category`, on every model that was tried, as for credentials.

    pydantic-ai sets a category only when the provider's error names the cause, so a plain
    `ModelHTTPError` falls through to the status rules.
    """
    return all(isinstance(e, category) for e in _failures(exc))


def _model_backend_unavailable(exc: Exception) -> bool:
    """Whether the run cannot reach the model at the moment (a client-side 4xx is a bug, not this).

    That is the provider failing transiently, or a concurrency limit's `max_queued` turning the run away,
    whether the limit sits on the model or on the agent's runs. The caller answers both by retrying later.
    """
    if isinstance(exc, ModelHTTPError):
        return exc.status_code == 429 or exc.status_code >= 500
    return isinstance(exc, ModelAPIError | FallbackExceptionGroup | ConcurrencyLimitExceeded)
