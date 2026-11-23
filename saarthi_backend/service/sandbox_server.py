"""Code runner for the coding lab, in its own container (docker-compose `sandbox` service).

The backend never executes student code itself; it POSTs it here (SANDBOX_URL). This
container holds no secrets, has no internet (internal network only), a read-only
filesystem, memory/process caps, and runs as an unprivileged user. So code that reads
environment variables, files or the network finds nothing worth taking.

Run from /app with PYTHONPATH=/app:
    python saarthi_backend/service/sandbox_server.py      # listens on :8100
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:  # Run as a script in the container: load sandbox.py alone, without the backend's services.
    from sandbox import TIMEOUT, local_runtimes, run_here
except ImportError:  # Imported as a package module (tests).
    from saarthi_backend.service.sandbox import TIMEOUT, local_runtimes, run_here

MAX_BODY = 256 * 1024  # code + stdin


class Handler(BaseHTTPRequestHandler):
    def _reply(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._reply(200, {"status": "ok"})
        elif self.path == "/runtimes":
            self._reply(200, {"items": local_runtimes()})
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/run":
            self._reply(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_BODY:
            self._reply(413, {"error": "request too large"})
            return
        try:
            req = json.loads(self.rfile.read(length))
            timeout = max(1, min(int(req.get("timeout") or TIMEOUT), TIMEOUT))
            result = run_here(str(req["language"]), str(req["code"]), str(req.get("stdin") or ""), timeout)
        except (KeyError, ValueError, TypeError) as e:
            self._reply(400, {"error": f"bad request: {e}"})
            return
        self._reply(200, result)

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - keep student code out of the logs
        pass


if __name__ == "__main__":
    port = int(os.environ.get("SANDBOX_PORT", "8100"))
    print(f"Code sandbox listening on :{port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
