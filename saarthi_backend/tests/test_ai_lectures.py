"""Found on real lecture transcripts (2026-11-17): the model's own "not covered" sentence, and course
chat not seeing video transcripts.

    pytest saarthi_backend/tests/test_ai_lectures.py -v
"""
import asyncio
import json

from saarthi_backend.tests.ai_fixtures import sandbox  # noqa: F401
from src.prompts import tutor

CHUNKS = [{"title": "Lecture 27 June", "page": 1, "text": "The ROC consists of strips parallel to the j omega axis."}]


def _run_reply(monkeypatch, model_text: str):
    import saarthi_backend.ai.adapter as adapter

    async def fake_tokens(messages, tier=None, max_tokens=3000):
        for i in range(0, len(model_text), 5):
            yield adapter._sse_data(model_text[i:i + 5])

    monkeypatch.setattr(adapter, "_stream_llm_tokens", fake_tokens)

    async def run():
        return [f async for f in adapter._emit_reply("grounded", "What shape is the ROC?", CHUNKS, [], meta={})]

    frames = asyncio.run(run())
    shown = "".join(f[6:].rstrip("\n") for f in frames if f.startswith("data: ") and not f.startswith("data: [DONE]"))
    meta = json.loads(next(f for f in frames if f.startswith("event: meta")).split("data: ", 1)[1])
    return shown, meta


def test_an_answer_followed_by_the_refusal_sentence_keeps_only_the_answer(monkeypatch):
    shown, meta = _run_reply(monkeypatch, "The ROC is a strip parallel to the j omega axis [Lecture 27 June, p. 1]. "
                                          + tutor.not_covered("Signals and Systems"))
    assert "strip parallel" in shown
    assert "couldn't find" not in shown.lower() and "couldn’t find" not in shown.lower()
    assert meta["reply_mode"] == "grounded" and meta["confidence"] == "low"   # goes to the review queue


def test_a_reply_that_is_only_the_refusal_becomes_a_not_covered_turn(monkeypatch):
    shown, meta = _run_reply(monkeypatch, tutor.not_covered("Signals and Systems"))
    assert shown.strip().startswith("I couldn't find this")
    assert "Sources" not in shown                       # no sources under a refusal
    assert meta["reply_mode"] == "not_covered" and meta["citations"] == []


def test_a_normal_answer_is_untouched(monkeypatch):
    shown, meta = _run_reply(monkeypatch, "The ROC is a strip parallel to the j omega axis [Lecture 27 June, p. 1].")
    assert "strip parallel" in shown and meta["reply_mode"] == "grounded"


def test_course_chat_finds_a_course_video_and_forgets_it_when_deleted(sandbox):  # noqa: F811
    from saarthi_backend.service import indexing_service as ix
    from src.tools.citations import source_label
    from src.tools.retrieve import retrieve

    segs = [{"text": "Today the region of convergence of the Laplace transform, strips parallel to the j omega axis.",
             "start_sec": 158, "duration": 50},
            {"text": "A right sided signal has an ROC to the right of the rightmost pole.", "start_sec": 210, "duration": 50}]
    assert ix._index_video_segments_sync(7, "Lecture 27 June", segs, course_id=3)
    hits = retrieve("region of convergence Laplace", course_id=3, k=3)
    assert hits and all(h.video_id == 7 for h in hits)
    assert source_label(hits[0]).startswith("[Video: Lecture 27 June, ")

    assert ix._index_video_segments_sync(7, "Lecture 27 June", segs[:1], course_id=3)   # re-index replaces
    assert len(retrieve("region of convergence Laplace", course_id=3, k=10)) == 1

    ix.delete_video_transcript_index(7, course_id=3)
    assert retrieve("region of convergence Laplace", course_id=3, k=10) == []
