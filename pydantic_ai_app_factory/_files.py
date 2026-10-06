"""The files a client sends: which media types the app admits, and how a run refusing one is reported."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import get_args

from pydantic_ai.messages import (
    AudioMediaType,
    BinaryContent,
    DocumentMediaType,
    FileUrl,
    ImageMediaType,
    ModelMessage,
    ModelRequest,
    UserContent,
    UserPromptPart,
    VideoMediaType,
)

from pydantic_ai_app_factory.errors import AgentError

ACCEPTED_MEDIA_TYPES: frozenset[str] = frozenset(
    get_args(ImageMediaType) + get_args(AudioMediaType) + get_args(VideoMediaType) + get_args(DocumentMediaType)
)
"""The media types pydantic-ai classifies as an image, audio, video or document.

Read from its public type aliases, so the set follows the library. It is the same for every model:
what pydantic-ai can send, not what the deployed model supports. Every model class raises for a file
outside it, halfway through the run, where the error would surface as a 500.
"""


def content_media_types(content: str | Sequence[UserContent]) -> list[str]:
    """The media type of every file in one turn's content, in order."""
    if isinstance(content, str):
        return []
    return [item.media_type for item in content if isinstance(item, BinaryContent | FileUrl)]


def message_media_types(messages: Iterable[ModelMessage]) -> list[str]:
    """The media type of every file a client put in `messages`, in order."""
    return [
        media_type
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
        for media_type in content_media_types(part.content)
    ]


def require_accepted(media_types: Sequence[str]) -> None:
    """Refuse a request carrying a file outside `ACCEPTED_MEDIA_TYPES` before anything runs.

    The message names each unsupported type once, in order.
    """
    unsupported = [media_type for media_type in media_types if media_type not in ACCEPTED_MEDIA_TYPES]
    if unsupported:
        named = ' or '.join(dict.fromkeys(unsupported))
        raise AgentError('file-unsupported', f'Files of type {named} are not accepted.', http_status=415)
