"""The endpoints that describe and probe the composed app."""

import asyncio
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pydantic_ai_app_factory._files import ACCEPTED_MEDIA_TYPES
from pydantic_ai_app_factory.deps import AuthProvider
from pydantic_ai_app_factory.feature import Feature

logger = logging.getLogger(__name__)


def ops_router(
    features: Sequence[Feature], auth: AuthProvider, extras_model: type[BaseModel], health_timeout: float
) -> APIRouter:
    """The `/config`, `/config/features`, `/healthz`, and `/readyz` endpoints."""
    router = APIRouter()

    @router.get('/config')
    def config() -> dict[str, Any]:
        """The composed feature set, the extras schema it accepts, and the media types a file may have."""
        return {
            'features': [f.name for f in features],
            'extras_schema': extras_model.model_json_schema(),
            'accepts': sorted(ACCEPTED_MEDIA_TYPES),
        }

    @router.get('/config/features', dependencies=[Depends(auth)])
    def feature_config(response: Response) -> dict[str, Any]:
        """How each feature is configured; a feature reporting nothing contributes no key."""
        # This is the one authenticated GET, and it sits under the public `/config` prefix -- a cache
        # rule written for that prefix would otherwise store an authenticated body.
        response.headers['Cache-Control'] = 'no-store'
        return {'feature_config': {f.name: c for f in features if (c := f.config()) is not None}}

    @router.get('/healthz')
    def healthz() -> dict[str, Any]:
        """Liveness: the process is up; no external dependency is checked."""
        return {'status': 'ok', 'features': [f.name for f in features]}

    async def _check(feature: Feature) -> Mapping[str, Any] | None:
        # Bound each check: a hung health() would otherwise hang the probe -- and the worker -- forever.
        # A timeout raises, so it lands in the unhealthy branch below like any other failure.
        async with asyncio.timeout(health_timeout):
            return await feature.health()

    @router.get('/readyz')
    async def readyz() -> Any:
        """Readiness: every feature's health check; 503 when any fails or times out."""
        # Run the checks concurrently (total time = the slowest, not the sum); return_exceptions so one
        # failing (or timed-out) check becomes a value here rather than aborting the whole probe.
        results = await asyncio.gather(*(_check(feature) for feature in features), return_exceptions=True)
        reports: dict[str, Any] = {}
        failures: dict[str, str] = {}
        for feature, result in zip(features, results, strict=True):
            if isinstance(result, BaseException):
                # Type only, never str(exc) -- the full detail belongs in the log, not a reachable probe.
                logger.error("feature '%s' is unhealthy", feature.name, exc_info=result)
                failures[feature.name] = type(result).__name__
            elif result is not None:
                reports[feature.name] = result
        if failures:
            content = {'status': 'unhealthy', 'reports': reports, 'failures': failures}
            return JSONResponse(status_code=503, content=content)
        return {'status': 'ok', 'reports': reports}

    return router
