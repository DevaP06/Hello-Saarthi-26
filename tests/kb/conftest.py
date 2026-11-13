from __future__ import annotations

import pytest

import scripts.env_loader as env_loader
from src.knowledge_base.config import KBConfig


@pytest.fixture(autouse=True)
def isolate_from_real_env_kb(tmp_path, monkeypatch):
    """load_config() calls env_loader.load_env_file(), which reads this
    machine's real .env.kb (repo root) and mutates the real os.environ.
    Without this, a real .env.kb on disk leaks machine-specific values
    (SAARTHI_KB_DATA_ROOT, the OCR flags, ...) into every test in this
    process, since os.environ persists across tests — monkeypatch's
    setenv/delenv only undo what the TEST itself set, not what code under
    test (load_config) sets as a side effect. Pointing the loader at a
    nonexistent path makes load_env_file() a no-op during tests."""
    monkeypatch.setattr(env_loader, "DEFAULT_ENV_KB_PATH", tmp_path / "nonexistent.env.kb")


@pytest.fixture
def kb_config(tmp_path):
    config = KBConfig(data_root=tmp_path / "SaarthiDataTest")
    config.ensure_directories()
    return config
