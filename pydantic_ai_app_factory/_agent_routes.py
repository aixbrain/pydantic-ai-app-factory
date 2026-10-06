"""The endpoints that run the agent."""

# No `from __future__ import annotations` here: FastAPI resolves `Depends(...)` from real annotation
# objects, and stringized annotations cannot see a closure-scoped dependency like `auth`.

import inspect
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Generic, TypeVar

from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import RequestValidationError
from pydantic import (
    BaseModel,
    Field,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    FileUrl,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    UserContent,
    UserPromptPart,
    sanitize_messages,
)
from pydantic_ai.usage import UsageLimits
from pydantic_core import CoreSchema, core_schema
from starlette.responses import Response

from pydantic_ai_app_factory._envelope import log_agent_error
from pydantic_ai_app_factory._files import content_media_types, message_media_types, require_accepted
from pydantic_ai_app_factory._vercel import VercelAdapter
from pydantic_ai_app_factory.deps import (
    AppDeps,
    AuthProvider,
    DepsBuilder,
    UsageLimitsSource,
    UserContext,
    VercelSdkVersion,
)
from pydantic_ai_app_factory.errors import AgentError, ErrorHandlers, to_agent_error

ExtrasT = TypeVar('ExtrasT', bound=BaseModel)


def require_media_types(content: Sequence[UserContent]) -> None:
    """Reject a file URL whose media type is neither given nor inferable from its extension.

    pydantic-ai raises for such a URL only when a provider reads `media_type`, halfway through
    the run, where the error would surface as a 500. The `/chat` adapter never hits this because
    the Vercel file part carries the type explicitly; raising here makes it the 422 a malformed
    body gets.
    """
    for item in content:
        if isinstance(item, FileUrl):
            _ = item.media_type


class _OpaqueMessagesSchema:
    """Validates and serializes `list[ModelMessage]` without exposing its schema to FastAPI.

    Since pydantic-ai 2.42 the validation-mode schema of `ModelMessage` has a discriminator mapping
    that OpenAPI rejects, and `/openapi.json` fails (pydantic/pydantic-ai#8679). FastAPI emits every
    definition in a field's core schema, so the message types are kept out of it and the schema is
    a plain array of objects. Remove once the upstream fix is released.
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(
            ModelMessagesTypeAdapter.validate_python,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda messages, info: ModelMessagesTypeAdapter.dump_python(
                    messages, mode='json' if info.mode_is_json() else 'python'
                ),
                info_arg=True,
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(cls, schema: CoreSchema, handler: GetJsonSchemaHandler) -> dict[str, Any]:
        return {'type': 'array', 'items': {'type': 'object'}}


WireMessages = Annotated[list[ModelMessage], _OpaqueMessagesSchema]


class RunRequest(BaseModel, Generic[ExtrasT]):
    """Body of a `/run` call, parameterized with the composed extras model."""

    prompt: str | list[UserContent] = Field(
        description='The user message for this run; a list to send files or images alongside the text.'
    )
    conversation_id: str | None = Field(default=None, description='Continues this conversation; omit to run stateless.')
    message_history: WireMessages | None = Field(
        default=None,
        description='Prior messages to run on top of, in pydantic-ai wire form. Sanitized before the run.',
    )
    extras: ExtrasT = Field(
        description='Per-request feature parameters; schema composed from the deployed features (see GET /config).'
    )

    @field_validator('prompt')
    @classmethod
    def _prompt_media_types(cls, prompt: str | list[UserContent]) -> str | list[UserContent]:
        if not isinstance(prompt, str):
            require_media_types(prompt)
        return prompt

    @field_validator('message_history')
    @classmethod
    def _history_media_types(cls, history: list[ModelMessage] | None) -> list[ModelMessage] | None:
        for message in history or ():
            for part in message.parts:
                if isinstance(part, UserPromptPart) and not isinstance(part.content, str):
                    require_media_types(part.content)
        return history

    @model_validator(mode='before')
    @classmethod
    def _default_extras(cls, data: Any) -> Any:
        # Omitting `extras` validates the composed model against `{}`, so a feature's required field
        # is reported by its own name rather than as a missing `extras` object.
        if isinstance(data, Mapping) and 'extras' not in data:
            return {**data, 'extras': {}}
        return data


class RunResponse(BaseModel):
    """Result of a `/run` call."""

    output: Any
    conversation_id: str
    """The conversation this run belongs to; pass it back to continue. Minted when the call omitted one."""
    messages: WireMessages
    """The messages this run produced, in the wire form `message_history` takes.

    The run's own messages only, never the sent history: a client holding the history appends
    these and sends the whole list on the next call. A file the model produced is in here as
    inline data; `output` stays the agent's output type.
    """


def run_metadata(user: UserContext) -> dict[str, Any]:
    """Attributes identifying the caller on the agent run span."""
    return {'user_id': user.user_id, 'tenant_id': user.tenant_id}


def sanitize_prompt(prompt: str | list[UserContent]) -> str | Sequence[UserContent]:
    """The new turn's content, under the rules the sent history goes through.

    `sanitize_messages` works on messages rather than on content, so the turn is wrapped in a
    throwaway `ModelRequest` and unwrapped again -- a plain string has nothing to strip and skips
    the round trip. A prompt stripped down to nothing stays an empty turn instead of
    becoming an error, which is what `/chat` does with a message whose every part was disallowed.
    """
    if isinstance(prompt, str):
        return prompt
    [request] = sanitize_messages([ModelRequest(parts=[UserPromptPart(content=prompt)])])
    [user_part] = [part for part in request.parts if isinstance(part, UserPromptPart)]
    return user_part.content


def resolve_usage_limits(source: UsageLimitsSource, user: UserContext) -> UsageLimits | None:
    """The limits for this caller's run."""
    return source(user) if callable(source) else source


async def resolve_deps(
    deps_builder: DepsBuilder[AppDeps] | None,
    request: Request,
    user: UserContext,
    extras: Mapping[str, Any],
    expected_deps: type[AppDeps],
) -> AppDeps:
    """The deps for one run: the product's subclass via its builder, or a plain `AppDeps`.

    The managed fields are set here, after the builder returns, so a builder cannot hand the run a
    different user or stale scratch -- what it puts there is overwritten. `expected_deps` is the type
    `_check_deps_type` computed from the agent's declaration; a builder that returns something else
    (a stale return, a copy-paste from a different agent) is caught here rather than on the first
    tool call that reads a field the returned instance does not have.
    """
    if deps_builder is None:
        return AppDeps(user=user, extras=extras)
    built = deps_builder(request, user)
    if inspect.isawaitable(built):
        built = await built
    if not isinstance(built, expected_deps):
        raise TypeError(f'deps builder returned {type(built).__name__}, expected {expected_deps.__name__}')
    built.user = user
    built.extras = extras
    built.scratch = {}
    return built


def agent_router(
    agent: Agent[AppDeps, Any],
    auth: AuthProvider,
    extras_model: type[BaseModel],
    capabilities: Sequence[AbstractCapability[AppDeps]],
    error_handlers: ErrorHandlers | None,
    usage_limits: UsageLimitsSource,
    deps_builder: DepsBuilder[AppDeps] | None,
    expected_deps: type[AppDeps],
    vercel_sdk_version: VercelSdkVersion,
) -> APIRouter:
    """The `/chat` and `/run` endpoints over the composed agent."""
    router = APIRouter()
    request_model = RunRequest[extras_model]  # pyright: ignore[reportInvalidTypeArguments]

    @router.post('/chat')
    async def chat(request: Request, user: Annotated[UserContext, Depends(auth)]) -> Response:
        """Run the agent over the Vercel AI chat protocol and stream the response."""
        try:
            adapter = await VercelAdapter.from_request(
                request, agent=agent, error_handlers=error_handlers, sdk_version=vercel_sdk_version
            )
            extras = extras_model.model_validate(adapter.run_input.model_extra or {})
        except ValidationError as exc:
            # Raise rather than answer, so the body matches FastAPI's own 422 for every other route.
            raise RequestValidationError([{**e, 'loc': ('body', *e['loc'])} for e in exc.errors()]) from exc
        try:
            require_accepted(adapter.file_media_types)
        except AgentError as exc:
            # Refused before the stream starts, so it is answered as JSON with its status like a
            # failure in the deps builder; logged as the 4xx it is, with the error as its own cause.
            log_agent_error(exc, exc, 'agent run')
            raise
        try:
            deps = await resolve_deps(deps_builder, request, user, extras.model_dump(), expected_deps)
        except Exception as exc:
            agent_error = to_agent_error(exc, error_handlers)
            log_agent_error(agent_error, exc, 'deps resolution')
            # Raising (rather than answering) lets the app-level handler render the envelope, the
            # same one a failure inside the run itself gets from the stream's `on_error`.
            raise agent_error from exc
        stream = adapter.run_stream(
            deps=deps,
            capabilities=capabilities,
            metadata=run_metadata(user),
            usage_limits=resolve_usage_limits(usage_limits, user),
        )
        return adapter.streaming_response(stream)

    # The body model is parameterized at composition time with the features' extras, which pyright
    # cannot express as a static type -- hence the ignore on the `body` parameter.
    @router.post('/run')
    async def run(
        request: Request,
        body: request_model,  # pyright: ignore[reportInvalidTypeForm, reportUnknownParameterType]
        user: Annotated[UserContext, Depends(auth)],
    ) -> RunResponse:
        """Run the agent once and return its output as JSON.

        A `message_history` on the body is sanitized as the `/chat` adapter sanitizes the messages
        a frontend sends -- client-sent system prompts, non-http file URLs and dangling tool calls
        are dropped -- so both transports apply the same trust to the caller. The body's
        `conversation_id` wins over one the history carries.
        """
        prompt = sanitize_prompt(body.prompt)
        history = sanitize_messages(body.message_history) if body.message_history else None
        try:
            require_accepted([*message_media_types(history or ()), *content_media_types(prompt)])
            deps = await resolve_deps(deps_builder, request, user, body.extras.model_dump(), expected_deps)
            result = await agent.run(
                prompt,
                deps=deps,
                message_history=history,
                conversation_id=body.conversation_id,
                capabilities=capabilities,
                metadata=run_metadata(user),
                usage_limits=resolve_usage_limits(usage_limits, user),
            )
        except Exception as exc:
            agent_error = to_agent_error(exc, error_handlers)
            log_agent_error(agent_error, exc, 'agent run')
            raise agent_error from exc
        return RunResponse(output=result.output, conversation_id=result.conversation_id, messages=result.new_messages())

    return router
