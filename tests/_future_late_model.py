"""A module-level router whose handler annotates a model defined below it: valid under the future import."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from pydantic_ai_app_factory import AuthProvider, Feature

router = APIRouter()


@router.post('/late')
def late(body: Later) -> dict[str, str]:
    return {'x': body.x}


class Later(BaseModel):
    x: str


class LateModel(Feature):
    name = 'late-model'

    def routers(self, *, auth: AuthProvider) -> list[APIRouter]:
        return [router]
