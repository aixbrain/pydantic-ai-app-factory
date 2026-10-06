"""Helpers shared by the app-level test modules: a permissive auth, a client factory, fixture features."""

from collections.abc import Sequence
from typing import Annotated, Any

from fastapi import Header, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_app_factory import AppDeps, Feature, UserContext
from pydantic_ai_app_factory.app import create_agent, create_agent_app


def auth() -> UserContext:
    return UserContext(user_id='u1')


def make_client(
    *, agent: Agent[AppDeps, Any] | None = None, features: Sequence[Feature] = (), **kwargs: Any
) -> TestClient:
    app = create_agent_app(agent or create_agent(TestModel()), auth=auth, features=list(features), **kwargs)
    return TestClient(app)


def submit(text: str, **extras: Any) -> dict[str, Any]:
    message = {'id': 'm1', 'role': 'user', 'parts': [{'type': 'text', 'text': text}]}
    return {'trigger': 'submit-message', 'id': 'c1', 'messages': [message], **extras}


class StorageExtras(BaseModel):
    case_id: str


class Storage(Feature):
    name = 'storage'
    extras_type = StorageExtras


def ping() -> dict[str, bool]:
    return {'pong': True}


def strict_auth(x_api_key: Annotated[str | None, Header()] = None) -> UserContext:
    if x_api_key != 'k':
        raise HTTPException(status_code=401)
    return UserContext(user_id='u1')
