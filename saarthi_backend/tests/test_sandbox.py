"""With SANDBOX_URL set, code runs in the sandbox service and never in the backend process."""
from __future__ import annotations

import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

from saarthi_backend.service import sandbox
from saarthi_backend.service.sandbox_server import Handler


def test_matlab_runs_a_script_through_the_octave_cli(monkeypatch):
    code = "disp(2 + 3);"

    def fake_process(command, stdin, cwd, timeout):
        assert command[:-1] == ["octave-cli", "--quiet", "--no-init-file", "--no-history"]
        assert Path(command[-1]).name == "main.m"
        assert Path(command[-1]).read_text(encoding="utf-8") == code
        assert stdin == "" and timeout == 5
        return True, "5\n", ""

    monkeypatch.setattr(sandbox, "SANDBOX_URL", "")
    monkeypatch.setenv("CODE_SANDBOX", "local")
    monkeypatch.setattr(sandbox, "runtime_missing", lambda language: None)
    monkeypatch.setattr(sandbox, "_run_shell", fake_process)
    result = sandbox.run_code("matlab", code, timeout=5)
    assert result["exit_code"] == 0 and result["stdout"] == "5\n"
    assert result["runtime"] == "octave-cli"


def test_remote_runner_and_no_local_fallback(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(sandbox, "SANDBOX_URL", f"http://127.0.0.1:{server.server_address[1]}")
        assert sandbox.mode() == "remote"
        assert any(row["language"] == "matlab" and row["runtime"] == "octave-cli" for row in sandbox.runtimes())
        r = sandbox.run_code("python", "print(input() * 2)", "ab")
        assert r["sandbox"] == "remote", r
        assert r["stdout"].strip() == "abab", r

        # Runner down: the student gets an error; the code is NOT run here instead.
        monkeypatch.setattr(sandbox, "SANDBOX_URL", "http://127.0.0.1:9")
        r = sandbox.run_code("python", "print('ran in the backend')")
        assert r["exit_code"] == 1
        assert "ran in the backend" not in r["stdout"]
        assert "not available" in r["stderr"]
    finally:
        server.shutdown()
