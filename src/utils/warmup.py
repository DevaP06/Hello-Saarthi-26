"""Load everything a first question needs before the first student asks it.

Without this the first few questions after a restart pay for Python imports, the embedding
client, the classifier weights, every course index (FAISS + BM25), the optional reranker and
loading the chat model into GPU memory: about 4 s each, measured, when five students arrived
together. Runs once in a background thread at start-up; AI_WARMUP=0 switches it off.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_done = threading.Event()


def _step(name: str, fn, timings: dict) -> None:
    t = time.perf_counter()
    try:
        fn()
        timings[name] = round((time.perf_counter() - t) * 1000)
    except Exception as e:  # warm-up must never stop the server
        timings[name] = f"skipped: {e}"


def warm_up(kb_root: str | Path = "knowledge_base") -> dict:
    """Blocking. Returns per-step milliseconds (or the reason a step was skipped)."""
    timings: dict = {}

    def _embed():
        from src.utils.llm import embeddings
        embeddings().embed_query("warm up")

    def _classifier():
        from src.orchestrator import query_classifier
        query_classifier._load_model()

    def _indexes():
        from src.retrieval import store
        root = Path(kb_root)
        for vs in sorted(root.glob("courses/*/vector_store")) + sorted(root.glob("videos/*/vector_store")):
            store.load(vs)

    def _reranker():
        from src.retrieval import rerankers
        rerankers.cross_encoder_order("warm up", ["a", "b"])

    def _chat_model():
        # A 1-token request loads the Tier 1 model into memory (the model server keeps it loaded).
        from src.utils.llm import chat_llm
        chat_llm(max_tokens=1, tier=1).invoke("hi")

    _step("embedding", _embed, timings)
    _step("classifier", _classifier, timings)
    _step("course_indexes", _indexes, timings)
    _step("reranker", _reranker, timings)
    _step("chat_model", _chat_model, timings)
    _done.set()
    logger.info("AI warm-up finished: %s", timings)
    return timings


def start_background_warmup() -> None:
    if os.environ.get("AI_WARMUP", "1").strip().lower() in ("0", "false", "no", "off"):
        return
    threading.Thread(target=warm_up, name="ai-warmup", daemon=True).start()
