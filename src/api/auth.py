"""API key authentication dependency for the churn prediction API.

Authentication fails closed. Configure valid keys as a comma-separated list:

    export CHURN_API_KEYS="secret-key-1,secret-key-2"

For local development only, auth can be switched off explicitly:

    export CHURN_AUTH_DISABLED=1

With neither variable set, protected endpoints answer 503 ("auth not
configured") instead of silently running open. The /health endpoint is always
exempt so liveness probes keep working.
"""
from __future__ import annotations

import hmac
import os
from typing import Annotated

from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _valid_keys() -> frozenset[str]:
    raw = os.environ.get("CHURN_API_KEYS", "")
    return frozenset(k.strip() for k in raw.split(",") if k.strip())


def auth_disabled() -> bool:
    return os.environ.get("CHURN_AUTH_DISABLED", "").strip().lower() in _TRUTHY


def _matches(candidate: str, keys: frozenset[str]) -> bool:
    # Compare against every key in constant time so timing reveals nothing.
    supplied = candidate.encode()
    matched = False
    for key in keys:
        matched |= hmac.compare_digest(supplied, key.encode())
    return matched


async def require_api_key(
    api_key: Annotated[str | None, Security(_HEADER)],
) -> str:
    """FastAPI dependency enforcing the X-API-Key header.

    Returns the validated key, or 'dev' when CHURN_AUTH_DISABLED is set.
    401 for a missing or wrong key; 503 when no keys are configured.
    """
    if auth_disabled():
        return "dev"
    keys = _valid_keys()
    if not keys:
        raise HTTPException(
            status_code=503,
            detail=(
                "API authentication is not configured: set CHURN_API_KEYS "
                "(or CHURN_AUTH_DISABLED=1 for local development)."
            ),
        )
    if api_key is None or not _matches(api_key, keys):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return api_key
