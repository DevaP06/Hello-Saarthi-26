#!/bin/sh
# Apply DB migrations, then start the API. One worker: the rate limiter, semantic
# cache and token budget keep state in process memory.
set -e

cd /app
echo "[entrypoint] applying migrations..."
python saarthi_backend/migrations/run_migrations.py

echo "[entrypoint] starting API on :8000"
# No --forwarded-allow-ips='*': uvicorn must not take X-Forwarded-For from just anyone
# (code in the sandbox can reach this port). The rate limiter trusts that header only
# from Caddy, via PROXY_SECRET (saarthi_backend/utils/rate_limit.py).
exec uvicorn saarthi_backend.main:app \
    --host 0.0.0.0 --port 8000 \
    --workers 1
