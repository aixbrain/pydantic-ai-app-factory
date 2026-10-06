import pickle

import httpx
import pytest
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelAPIError, ModelHTTPError

from pydantic_ai_app_factory import AgentError
from pydantic_ai_app_factory.errors import to_agent_error


def test_agent_error_carries_code_message_and_defaults_to_not_retryable() -> None:
    err = AgentError('conversation-storage/not-found', 'No such conversation.')
    assert (err.code, err.message, err.retryable) == ('conversation-storage/not-found', 'No such conversation.', False)


def test_retryable_is_keyword_only_and_settable() -> None:
    err = AgentError('payment/limit-reached', 'The model is busy, try again.', retryable=True)
    assert err.retryable is True


def test_http_status_defaults_from_retryable() -> None:
    assert AgentError('x/y', 'boom').http_status == 500
    assert AgentError('x/y', 'boom', retryable=True).http_status == 503


def test_http_status_can_be_set_by_the_raiser() -> None:
    # The feature raising the error is the only place that knows a missing resource means 404.
    assert AgentError('storage/not-found', 'Gone.', http_status=404).http_status == 404


def test_http_status_survives_a_pickle_roundtrip() -> None:
    restored = pickle.loads(pickle.dumps(AgentError('quota/exceeded', 'Slow down.', http_status=429)))
    assert restored.http_status == 429


def test_str_shows_code_and_message() -> None:
    assert str(AgentError('x/y', 'boom')) == 'x/y: boom'


def test_is_raisable_and_catchable() -> None:
    with pytest.raises(AgentError) as exc_info:
        raise AgentError('x/y', 'boom')
    assert exc_info.value.code == 'x/y'


def test_survives_a_pickle_roundtrip_with_all_attributes() -> None:
    # Errors may cross a serialization boundary (durable execution); the base Exception would
    # pickle only `message` and drop `code`/`retryable` without the custom `__reduce__`.
    restored = pickle.loads(pickle.dumps(AgentError('payment/limit-reached', 'busy', retryable=True)))
    assert (restored.code, restored.message, restored.retryable) == ('payment/limit-reached', 'busy', True)


class _NotFound(AgentError):
    """A feature's own error type -- the documented way to namespace a code."""

    def __init__(self, what: str) -> None:
        super().__init__('storage/not-found', f'No such {what}.')


def test_a_subclass_with_its_own_signature_survives_pickling() -> None:
    # Subclassing is the documented pattern, so reconstruction must not re-call `__init__`
    # with the base class's `(code, message)` arguments.
    restored = pickle.loads(pickle.dumps(_NotFound('conversation')))
    assert isinstance(restored, _NotFound)
    assert (restored.code, restored.retryable) == ('storage/not-found', False)
    assert restored.message == 'No such conversation.'


def test_an_agent_error_passes_through_unchanged() -> None:
    err = AgentError('storage/not-found', 'No such conversation.')
    assert to_agent_error(err) is err


def test_error_handlers_translates_a_foreign_exception() -> None:
    error = to_agent_error(ValueError('boom'), error_handlers=[(ValueError, AgentError('bad-input', 'Bad input.'))])
    assert (error.code, error.retryable) == ('bad-input', False)


def test_error_handlers_callable_derives_the_wire_error_from_the_exception() -> None:
    def derive(exc: Exception) -> AgentError:
        return AgentError('bad-input', str(exc))

    assert to_agent_error(ValueError('too long'), error_handlers=[(ValueError, derive)]).message == 'too long'


def test_a_raising_error_handler_falls_back_to_internal_not_a_broken_envelope() -> None:
    # A buggy product handler must not break the no-leak contract: the failure becomes `internal`,
    # never an unhandled exception that escapes as a bare 500.
    def buggy(exc: Exception) -> AgentError:
        raise RuntimeError('handler bug')

    error = to_agent_error(ValueError('boom'), error_handlers=[(ValueError, buggy)])
    assert (error.code, error.http_status) == ('internal', 500)


def test_error_handlers_win_over_the_default_mapping() -> None:
    error_handlers = [(ModelHTTPError, AgentError('model/custom', 'Custom.'))]
    error = to_agent_error(ModelHTTPError(status_code=503, model_name='m'), error_handlers=error_handlers)
    assert error.code == 'model/custom'


def test_error_handlers_leaves_unlisted_exceptions_to_the_default() -> None:
    error = to_agent_error(ValueError('x'), error_handlers=[(KeyError, AgentError('kv/missing', 'Missing.'))])
    assert error.code == 'internal'


def test_a_transient_model_http_error_is_retryable_and_unavailable() -> None:
    error = to_agent_error(ModelHTTPError(status_code=503, model_name='m'))
    assert (error.code, error.retryable) == ('model-unavailable', True)


def test_a_client_side_model_http_error_is_internal_not_unavailable() -> None:
    error = to_agent_error(ModelHTTPError(status_code=400, model_name='m'))
    assert (error.code, error.retryable) == ('internal', False)


def test_a_provider_connection_failure_is_model_unavailable() -> None:
    error = to_agent_error(ModelAPIError(model_name='m', message='connection reset'))
    assert (error.code, error.retryable) == ('model-unavailable', True)


def test_all_fallback_models_failing_is_model_unavailable() -> None:
    group = FallbackExceptionGroup('all failed', [ModelAPIError(model_name='m', message='down')])
    error = to_agent_error(group)
    assert (error.code, error.retryable) == ('model-unavailable', True)


def test_an_unrecognized_exception_becomes_internal_without_leaking() -> None:
    error = to_agent_error(ValueError('secret dsn postgres://user:pw@host/db'))
    assert (error.code, error.retryable) == ('internal', False)
    assert 'postgres' not in error.message and 'secret' not in error.message


def test_a_provider_rejecting_the_key_is_model_unauthorized_not_internal() -> None:
    # The most common deployment mistake -- a wrong or expired API key -- must not look like a bug
    # in this package to the caller. 502: the upstream refused us, and retrying will not help.
    error = to_agent_error(ModelHTTPError(status_code=401, model_name='m', body={'error': 'invalid_api_key'}))
    assert (error.code, error.retryable, error.http_status) == ('model-unauthorized', False, 502)


def test_a_provider_forbidding_the_account_is_model_unauthorized_too() -> None:
    # Bedrock and Vertex report a missing permission as 403 rather than 401.
    error = to_agent_error(ModelHTTPError(status_code=403, model_name='m'))
    assert (error.code, error.retryable, error.http_status) == ('model-unauthorized', False, 502)


def test_every_fallback_model_rejecting_the_key_is_model_unauthorized() -> None:
    # FallbackModel wraps the per-model failures in a group; without looking inside, a bad key
    # would be classified as a transient outage and retried forever.
    group = FallbackExceptionGroup(
        'all failed',
        [ModelHTTPError(status_code=401, model_name='a'), ModelHTTPError(status_code=403, model_name='b')],
    )
    error = to_agent_error(group)
    assert (error.code, error.retryable) == ('model-unauthorized', False)


def test_a_mixed_fallback_group_stays_a_retryable_outage() -> None:
    # One model rejecting the key while another is down is still worth a retry: the second model
    # may come back.
    group = FallbackExceptionGroup(
        'all failed',
        [ModelHTTPError(status_code=401, model_name='a'), ModelHTTPError(status_code=503, model_name='b')],
    )
    error = to_agent_error(group)
    assert (error.code, error.retryable) == ('model-unavailable', True)


def test_a_401_from_a_tools_own_http_client_stays_internal() -> None:
    # The rule keys on the exception type pydantic-ai raises for the model provider, never on a
    # bare status: a tool talking to some other service is that tool's business.
    request = httpx.Request('GET', 'https://db.internal/rows')
    exc = httpx.HTTPStatusError('unauthorized', request=request, response=httpx.Response(401, request=request))
    assert to_agent_error(exc).code == 'internal'


def test_a_rate_limit_stays_a_retryable_outage_not_a_credential_rejection() -> None:
    # OpenAI reports an exhausted balance as 429 too; without the body it is indistinguishable from
    # a rate limit, so it stays retryable and is left to `error_handlers` -- as the docs promise.
    error = to_agent_error(ModelHTTPError(status_code=429, model_name='m'))
    assert (error.code, error.retryable) == ('model-unavailable', True)


def test_error_handlers_win_over_the_credential_rejection_rule() -> None:
    # A product may want its own code and message for a rejected key; its handler runs first.
    handlers = [(ModelHTTPError, AgentError('model/custom-unauthorized', 'Custom.'))]
    error = to_agent_error(ModelHTTPError(status_code=401, model_name='m'), error_handlers=handlers)
    assert error.code == 'model/custom-unauthorized'


def test_a_fixed_handler_is_copied_so_raising_it_cannot_mutate_the_shared_instance() -> None:
    # A fixed `AgentError` in `error_handlers` is one instance for the life of the app. Raising it
    # would attach the request's `__cause__` and `__traceback__` to it -- retaining the detail the
    # envelope hides and, under concurrency, showing one request's frames in another's log.
    fixed = _NotFound('conversation')
    first = to_agent_error(ValueError('a'), error_handlers=[(ValueError, fixed)])
    second = to_agent_error(ValueError('b'), error_handlers=[(ValueError, fixed)])

    assert first is not fixed and second is not first
    assert isinstance(first, _NotFound)
    assert (first.code, first.message, first.retryable, first.http_status) == (fixed.code, fixed.message, False, 500)
