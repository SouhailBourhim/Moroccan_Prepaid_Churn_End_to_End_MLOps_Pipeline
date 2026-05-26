"""API key authentication dependency for the churn prediction API.

Enable by setting CHURN_API_KEYS to a comma-separated list of valid keys:

    export CHURN_API_KEYS="secret-key-1,secret-key-2"

When the variable is absent or empty the API runs in open dev mode.
The /health endpoint is always exempt from auth (liveness probes must work).
"""
from __future__ import annotations

import os
from typing import Annotated

from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _valid_keys() -> frozenset[str]:
    raw = os.environ.get("CHURN_API_KEYS", "")
    return frozenset(k.strip() for k in raw.split(",") if k.strip())


async def require_api_key(
    api_key: Annotated[str | None, Security(_HEADER)],
) -> str:
    """FastAPI dependency — 401 when CHURN_API_KEYS is set and header is absent or wrong.

    Returns the validated key string, or 'dev' when no keys are configured.
    """
    keys = _valid_keys()
    if not keys:
        return "dev"
    if api_key is None or api_key not in keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return api_key
