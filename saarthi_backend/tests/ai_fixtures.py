"""Shared fixtures for the AI regression tests (test_ai_*.py).

`sandbox` gives a test its own knowledge_base, semantic cache, traces and usage file in a temp
folder, with 32-dimension fake embeddings, so nothing touches the real course indexes, needs
Ollama, or leaves files behind. Import the fixtures into a test module with:

    from saarthi_backend.tests.ai_fixtures import sandbox  # noqa: F401
"""
import pytest


def make_sandbox(tmp_path, mp):
    """Point every on-disk thing the AI code touches at tmp_path (mp is a pytest MonkeyPatch)."""
    pytest.importorskip("faiss")
    from langchain_community.embeddings import DeterministicFakeEmbedding

    import src.cache.semantic_cache as sc
    import src.telemetry.traces as traces
    import src.utils.budget as budget
    import src.utils.llm as llm
    from src.retrieval import store
    from saarthi_backend.service import indexing_service as ix

    mp.setattr(llm, "embeddings", lambda: DeterministicFakeEmbedding(size=32))
    mp.setattr(llm, "embedding_model_name", lambda: "fake-32")
    mp.chdir(tmp_path)  # retrieval reads the relative knowledge_base/ folder
    kb = tmp_path / "knowledge_base"
    mp.setattr(ix, "_KB_ROOT", kb / "courses")
    mp.setattr(ix, "_VIDEO_KB_ROOT", kb / "videos")
    mp.setattr(traces, "TRACE_DIR", tmp_path / "traces")
    mp.setattr(sc, "DB_PATH", tmp_path / "cache" / "semantic_cache.sqlite")
    mp.setattr(sc, "_singleton", None)
    mp.setattr(budget, "_USAGE_FILE", tmp_path / "cache" / "llm_usage.json")
    store._cache.clear()


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    import src.tools.retrieve as retrieve_mod
    from src.retrieval import store

    make_sandbox(tmp_path, monkeypatch)
    token = retrieve_mod.allowed_courses.set(None)
    yield tmp_path
    retrieve_mod.allowed_courses.reset(token)
    store._cache.clear()


def index_text(course_id: int, title: str, text: str, material_id: int = 1) -> None:
    """Index a text as one course material (same path the upload worker uses)."""
    from saarthi_backend.service import indexing_service as ix

    path = ix._KB_ROOT.parent / "tmp_upload" / f"{title}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    assert ix._index_document_sync(path, title, course_id, material_id)
