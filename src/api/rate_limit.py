"""In-memory sliding-window rate limiter for the churn prediction API.

Requests are keyed by X-Forwarded-For (first hop) when present, falling back
to the TCP client host. Thread-safe via threading.Lock; compatible with
uvicorn's default multi-threaded worker model.

Env vars (optional overrides):
    CHURN_RATE_LIMIT_GENERAL  – max req / 60 s for general endpoints (default 200)
    CHURN_RATE_LIMIT_PREDICT  – max req / 60 s for /predict (default 30)
"""
from __future__ import annotations

import os
import time
from collections import deque
from threading import Lock

from fastapi import HTTPException, Request

_WINDOW_SECONDS = 60


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class SlidingWindowLimiter:
    """Thread-safe per-client sliding-window request counter."""

    def __init__(self, max_requests: int) -> None:
        self.max_requests = max_requests
        self._buckets: dict[str, deque[float]] = {}
        self._lock = Lock()

    def _client_key(self, request: Request) -> str:
        fwd = request.headers.get("X-Forwarded-For")
        if fwd:
            return fwd.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def check(self, request: Request) -> None:
        """Raise HTTP 429 if the client has exceeded the configured rate limit."""
        key = self._client_key(request)
        now = time.monotonic()
        cutoff = now - _WINDOW_SECONDS
        with self._lock:
            bucket = self._buckets.setdefault(key, deque())
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Rate limit exceeded: {self.max_requests} requests"
                        f" per {_WINDOW_SECONDS}s."
                    ),
                    headers={"Retry-After": str(_WINDOW_SECONDS)},
                )
            bucket.append(now)


general_limiter = SlidingWindowLimiter(_env_int("CHURN_RATE_LIMIT_GENERAL", 200))
predict_limiter = SlidingWindowLimiter(_env_int("CHURN_RATE_LIMIT_PREDICT", 30))
