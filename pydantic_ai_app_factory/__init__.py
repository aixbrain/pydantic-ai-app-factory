"""Compose a Pydantic AI agent and its features into a FastAPI application.

The contract names (features and products import) and the build surface (`create_agent`,
`create_agent_app`, which products call) are re-exported here from their public submodules.
"""

from pydantic_ai_app_factory.app import create_agent, create_agent_app
from pydantic_ai_app_factory.deps import (
    AppDeps,
    AuthProvider,
    DepsBuilder,
    DepsT,
    UsageLimitsSource,
    UserContext,
    VercelSdkVersion,
)
from pydantic_ai_app_factory.errors import AgentError, ErrorHandlers, to_agent_error
from pydantic_ai_app_factory.feature import Feature

__all__ = [
    'AgentError',
    'AppDeps',
    'AuthProvider',
    'DepsBuilder',
    'DepsT',
    'ErrorHandlers',
    'Feature',
    'UsageLimitsSource',
    'UserContext',
    'VercelSdkVersion',
    'create_agent',
    'create_agent_app',
    'to_agent_error',
]
