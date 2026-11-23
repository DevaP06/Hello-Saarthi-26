"""In-memory rate limiting (production: use Redis or similar)."""

import hmac
import os
import time
from typing import Tuple

# Production: Caddy, the only real proxy, sends this secret with every request
# (deploy/Caddyfile). Other callers on the internal network, such as code running in
# the sandbox, cannot forge it, so their X-Forwarded-For is ignored and they cannot
# dodge the per-IP limits (e.g. on login). Unset (local dev): the header is trusted.
_PROXY_SECRET = os.environ.get("PROXY_SECRET", "")

# Key -> (count, window_start_ts)
_buckets: dict[str, Tuple[int, float]] = {}
# Limits: (requests per window, window seconds)
RATE_LIMIT_GENERAL = (300, 60)   # 300/min per IP
RATE_LIMIT_AUTH = (20, 60)       # 20/min per IP for login/signup
RATE_LIMIT_UPLOAD = (20, 60)    # 20/min per IP for upload
RATE_LIMIT_ANON_CHAT = (10, 60)  # 10/min per IP for the signed-out chat widget (each call costs model time)
RATE_LIMIT_MATERIAL_FILE = (120, 60)  # 120/min per IP for material file


def _window_key(identifier: str, prefix: str) -> str:
    return f"{prefix}:{identifier}"


def _current_window(window_sec: int) -> int:
    return int(time.time() / window_sec)


def _bucket_key(identifier: str, prefix: str, window_sec: int) -> str:
    return f"{_window_key(identifier, prefix)}:{_current_window(window_sec)}"


def check_rate_limit(identifier: str, prefix: str, limit: int, window_sec: int) -> Tuple[bool, int]:
    """
    Return (allowed, retry_after_seconds).
    If allowed is False, retry_after_seconds is the suggested Retry-After value.
    """
    key = _bucket_key(identifier, prefix, window_sec)
    now = time.time()
    if key not in _buckets:
        _buckets[key] = (1, now)
        return True, 0
    count, start = _buckets[key]
    if count >= limit:
        # Next window starts at (current_window + 1) * window_sec
        next_window = (_current_window(window_sec) + 1) * window_sec
        retry_after = max(1, int(next_window - now))
        return False, retry_after
    _buckets[key] = (count + 1, start)
    return True, 0


def get_identifier_from_request(request) -> str:
    """Client IP: X-Forwarded-For when it comes from our own proxy, else the connecting peer."""
    forwarded = request.headers.get("x-forwarded-for")
    from_proxy = not _PROXY_SECRET or hmac.compare_digest(
        request.headers.get("x-saarthi-proxy", "").encode(), _PROXY_SECRET.encode()
    )
    if forwarded and from_proxy:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def cleanup_old_buckets(max_age_sec: int = 300) -> None:
    """Drop buckets older than max_age_sec to avoid unbounded memory growth."""
    now = time.time()
    to_drop = [k for k, (_, start) in _buckets.items() if now - start > max_age_sec]
    for k in to_drop:
        del _buckets[k]
