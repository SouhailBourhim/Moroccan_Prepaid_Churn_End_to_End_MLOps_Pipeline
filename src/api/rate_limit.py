"""In-memory sliding-window rate limiter for the churn prediction API.

Client identity
---------------
Requests are keyed by the TCP peer address. X-Forwarded-For is honoured only
when the peer is a trusted reverse proxy listed in CHURN_TRUSTED_PROXIES; the
client is then the right-most address in the header that is not itself a
trusted proxy. (The left-most entry is whatever the client chose to send, so
trusting it would let anyone bypass the limit by rotating the header.)

Multiple workers
----------------
State is per process. With API_WORKERS=N uvicorn processes, each limiter is
given ceil(limit / N) so the total across workers stays close to the configured
limit. Connections are spread across workers by the kernel, so this is
approximate; use a shared store (Redis) or the gateway for exact limits.

Memory
------
Idle client buckets are swept once the table exceeds max_clients, and if every
bucket is still active the least recently seen clients are evicted, so spoofed
or one-off clients cannot grow memory without bound.

Env vars (optional overrides):
    CHURN_RATE_LIMIT_GENERAL  – max req / 60 s for general endpoints (default 200)
    CHURN_RATE_LIMIT_PREDICT  – max req / 60 s for /predict (default 30)
    CHURN_TRUSTED_PROXIES     – comma-separated proxy IPs allowed to set X-Forwarded-For
    API_WORKERS               – number of uvicorn worker processes (default 1)
"""
from __future__ import annotations

import math
import os
import time
from collections import deque
from threading import Lock

from fastapi import HTTPException, Request

_WINDOW_SECONDS = 60
_DEFAULT_MAX_CLIENTS = 10_000


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _trusted_proxies() -> frozenset[str]:
    raw = os.environ.get("CHURN_TRUSTED_PROXIES", "")
    return frozenset(p.strip() for p in raw.split(",") if p.strip())


def per_worker_limit(total: int, workers: int) -> int:
    """Share a global per-client limit across `workers` independent processes."""
    return max(1, math.ceil(total / max(1, workers)))


class SlidingWindowLimiter:
    """Thread-safe per-client sliding-window request counter."""

    def __init__(
        self,
        max_requests: int,
        trusted_proxies: frozenset[str] = frozenset(),
        max_clients: int = _DEFAULT_MAX_CLIENTS,
    ) -> None:
        self.max_requests = max_requests
        self.trusted_proxies = trusted_proxies
        self.max_clients = max_clients
        self._buckets: dict[str, deque[float]] = {}
        self._lock = Lock()

    def client_key(self, request: Request) -> str:
        peer = request.client.host if request.client else "unknown"
        if peer not in self.trusted_proxies:
            return peer
        fwd = request.headers.get("X-Forwarded-For", "")
        hops = [h.strip() for h in fwd.split(",") if h.strip()]
        for hop in reversed(hops):
            if hop not in self.trusted_proxies:
                return hop
        return peer

    def _evict(self, cutoff: float) -> None:
        """Make room for one new client: drop idle buckets, then least recently seen."""
        for key in [k for k, b in self._buckets.items() if not b or b[-1] < cutoff]:
            del self._buckets[key]
        overflow = len(self._buckets) - (self.max_clients - 1)
        if overflow > 0:
            oldest = sorted(self._buckets, key=lambda k: self._buckets[k][-1])[:overflow]
            for key in oldest:
                del self._buckets[key]

    def check(self, request: Request) -> None:
        """Raise HTTP 429 if the client has exceeded the configured rate limit."""
        key = self.client_key(request)
        now = time.monotonic()
        cutoff = now - _WINDOW_SECONDS
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                if len(self._buckets) >= self.max_clients:
                    self._evict(cutoff)
                bucket = self._buckets[key] = deque()
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


_WORKERS = _env_int("API_WORKERS", 1)
general_limiter = SlidingWindowLimiter(
    per_worker_limit(_env_int("CHURN_RATE_LIMIT_GENERAL", 200), _WORKERS),
    trusted_proxies=_trusted_proxies(),
)
predict_limiter = SlidingWindowLimiter(
    per_worker_limit(_env_int("CHURN_RATE_LIMIT_PREDICT", 30), _WORKERS),
    trusted_proxies=_trusted_proxies(),
)
