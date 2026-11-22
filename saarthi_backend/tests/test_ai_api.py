"""AI features through the real HTTP routers: instructor review queue, /api/v1/saarthi contract,
per-course retrieval scope, PII scrubbing of study-tool prompts. In-process SQLite, fake embeddings,
no LLM server needed (the LLM client is replaced where a prompt is inspected).

    pip install -r saarthi_backend/requirements-ai-extras.txt     # aiosqlite
    pytest saarthi_backend/tests/test_ai_api.py -v
"""
import asyncio
import json
import sys
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("aiosqlite")
pytest.importorskip("faiss")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: E402

from saarthi_backend.model import Base, User  # noqa: E402
from saarthi_backend.routers import api_router  # noqa: E402
from saarthi_backend.main import saarthi_backend_exception_handler  # noqa: E402
from saarthi_backend.service import chat_service  # noqa: E402
from saarthi_backend.tests.ai_fixtures import make_sandbox  # noqa: E402
from saarthi_backend.utils.exceptions import SaarthiBackendError  # noqa: E402
from saarthi_backend.utils.jwt_utils import create_access_token  # noqa: E402

OFF_TOPIC = "Who won the football world cup in 2018?"
PII = "Priya Sharma (priya.s@iiit.ac.in, S20990000001, +91 98765 43210)"


def _notes(marker):
    return (f"Sampling and Nyquist notes {marker}. The Nyquist rate is twice the highest frequency present "
            f"in a band-limited signal. Sampling below the Nyquist rate causes aliasing. ").encode() * 12


class Env:
    """Two courses (A: teacher T, B: teacher T2), student S1 enrolled in A only."""

    def __init__(self, client, users):
        self.client = client
        self.ids = users

    def h(self, email):
        return {"Authorization": f"Bearer {create_access_token(str(self.ids[email]))}"}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ai_api")
    mp = pytest.MonkeyPatch()
    make_sandbox(tmp, mp)
    uploads = tmp / "uploads"
    uploads.mkdir()
    # `saarthi_backend.routers.course_router` is the APIRouter in the package namespace; patch the modules
    for name in ("course_router", "teacher_router"):
        mp.setattr(sys.modules[f"saarthi_backend.routers.{name}"], "_UPLOAD_DIR", uploads)

    # a pooled file database: one shared in-memory connection cannot serve concurrent requests
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp / 'api.db').as_posix()}")
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False, autoflush=False)

    async def setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.execute(text(
                "CREATE TABLE IF NOT EXISTS saarthi_conversation_context (conversation_id VARCHAR(36) PRIMARY KEY, "
                "summary TEXT, metadata TEXT, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"))
        async with Session() as s:
            users = [
                User(email="t@x.io", password_hash="x", full_name="Teacher T", role="teacher"),
                User(email="t2@x.io", password_hash="x", full_name="Teacher Two", role="teacher"),
                User(email="s1@x.io", password_hash="x", full_name="Priya Sharma", role="student"),
                User(email="s2@x.io", password_hash="x", full_name="Student Two", role="student"),
                User(email="out@x.io", password_hash="x", full_name="Outsider", role="student"),
                User(email="adm@x.io", password_hash="x", full_name="Admin A", role="admin"),
            ]
            s.add_all(users)
            await s.commit()
            return {u.email: u.id for u in users}

    ids = asyncio.run(setup())
    app = FastAPI()
    app.state.db_session_factory = Session
    app.include_router(api_router, prefix="/api")
    app.add_exception_handler(SaarthiBackendError, saarthi_backend_exception_handler)
    e = Env(TestClient(app, raise_server_exceptions=False), ids)

    def make_course(owner, title, marker):
        c = e.client
        cid = c.post("/api/courses", json={"title": title, "code": title[:5], "instructor": "x"}, headers=e.h(owner)).json()["id"]
        up = c.post("/api/courses/upload", files={"file": ("n.txt", _notes(marker))}, headers=e.h(owner)).json()
        mid = c.post(f"/api/courses/{cid}/materials", json={"title": "Notes " + marker, "type": "doc",
                     "url": "http://localhost:8000" + up["url"]}, headers=e.h(owner)).json()["id"]
        for _ in range(150):
            if c.get(f"/api/courses/{cid}/materials/{mid}/index-status", headers=e.h(owner)).json()["status"] == "done":
                break
            time.sleep(0.2)
        return cid

    e.A = make_course("t@x.io", "Signals A", "MARKER_ALPHA")
    e.B = make_course("t2@x.io", "Signals B", "MARKER_BRAVO")
    for who in ("s1@x.io", "s2@x.io"):
        code = e.client.post(f"/api/courses/{e.A}/invite", json={"email": "*"}, headers=e.h("t@x.io")).json()["inviteCode"]
        e.client.post("/api/courses/join", json={"inviteCode": code}, headers=e.h(who))
    yield e
    mp.undo()


def stream(env, who, message, **extra):
    return env.client.post("/api/chat/stream", json={"message": message, "conversationHistory": [], **extra},
                           headers=env.h(who))


def reviews(env, who="t@x.io", **params):
    return env.client.get("/api/reviews", params={"courseId": env.A, **params}, headers=env.h(who))


# ── instructor review queue ─────────────────────────────────────────────────

def test_uncovered_question_is_flagged_for_the_instructor_and_deduplicated(env):
    r = stream(env, "s1@x.io", OFF_TOPIC, courseId=int(env.A))
    assert r.status_code == 200 and "not_covered" in r.text
    items = reviews(env).json()["items"]
    assert len(items) == 1
    assert items[0]["query"] == OFF_TOPIC and items[0]["source"] == "low_confidence"
    assert items[0]["studentName"] == "[student]" or items[0]["studentName"] == "Priya Sharma"
    assert items[0]["answer"]

    stream(env, "s1@x.io", OFF_TOPIC, courseId=int(env.A))
    items = reviews(env).json()["items"]
    assert len(items) == 1 and items[0]["occurrences"] == 2          # same student, same question: one row

    stream(env, "s2@x.io", OFF_TOPIC, courseId=int(env.A))
    assert len(reviews(env).json()["items"]) == 2                    # another student adds a row


def test_only_enrolled_students_can_add_to_the_queue(env):
    before = env.client.get("/api/reviews/stats", params={"courseId": env.A}, headers=env.h("t@x.io")).json()["total"]
    stream(env, "out@x.io", "outsider question about cooking pasta", courseId=int(env.A))
    env.client.post("/api/chat/feedback", json={"query": "outsider fb", "rating": "down", "courseId": int(env.A),
                                                "answer": "x"}, headers=env.h("out@x.io"))
    after = env.client.get("/api/reviews/stats", params={"courseId": env.A}, headers=env.h("t@x.io")).json()["total"]
    assert after == before


def test_review_endpoints_are_staff_only(env):
    c = env.client
    rid = reviews(env).json()["items"][0]["id"]
    assert reviews(env, "s1@x.io").status_code == 403
    assert reviews(env, "t2@x.io").status_code == 403                # a teacher of another course
    body = {"correction": "my own answer is better"}
    assert c.post(f"/api/reviews/{rid}/correct", json=body, headers=env.h("s1@x.io")).status_code == 403
    assert c.post(f"/api/reviews/{rid}/correct", json=body, headers=env.h("t2@x.io")).status_code == 403
    assert c.post("/api/reviews/99999/dismiss", headers=env.h("t@x.io")).status_code == 404
    assert c.post(f"/api/reviews/{rid}/correct", json={"correction": "no"}, headers=env.h("t@x.io")).status_code == 422


def test_correction_is_saved_indexed_and_used_by_retrieval(env):
    from src.tools.grounding import ground
    from src.tools.retrieve import retrieve

    rid = next(i["id"] for i in reviews(env).json()["items"] if i["query"] == OFF_TOPIC)
    fix = "France won the 2018 football world cup, beating Croatia 4-2 in the final."
    r = env.client.post(f"/api/reviews/{rid}/correct", json={"correction": fix}, headers=env.h("t@x.io"))
    assert r.status_code == 200 and r.json()["indexed"] is True and r.json()["review"]["status"] == "corrected"
    assert all(i["id"] != rid for i in reviews(env).json()["items"])  # leaves the open list
    assert any("France won" in c.text for c in retrieve(OFF_TOPIC, course_id=int(env.A), k=3))
    assert ground(OFF_TOPIC, course_id=int(env.A), n=3).covered


def test_thumbs_down_creates_a_review_and_dismiss_closes_it(env):
    r = env.client.post("/api/chat/feedback", json={"query": "What is aliasing?", "rating": "down",
                        "courseId": int(env.A), "answer": "Aliasing is when things fade."}, headers=env.h("s1@x.io"))
    assert r.status_code == 200 and r.json()["recorded"]
    td = [i for i in reviews(env).json()["items"] if i["source"] == "thumbs_down"]
    assert len(td) == 1 and "fade" in td[0]["answer"]
    d = env.client.post(f"/api/reviews/{td[0]['id']}/dismiss", headers=env.h("t@x.io"))
    assert d.status_code == 200 and d.json()["status"] == "dismissed"
    assert reviews(env, status="all").json()["total"] >= 3


# ── /api/v1/saarthi contract ────────────────────────────────────────────────

BASE = "/api/v1/saarthi"


def search(env, who, **body):
    return env.client.post(f"{BASE}/retrieval/search", json=body, headers=env.h(who) if who else {})


def test_search_returns_contract_shaped_chunks_for_a_member(env):
    assert search(env, None, query="nyquist").status_code == 401
    j = search(env, "s1@x.io", query="what is the Nyquist rate", course_id=int(env.A), top_k=3).json()
    assert j["query"] == "what is the Nyquist rate" and 1 <= j["total_returned"] <= 3
    c = j["results"][0]
    assert c["content_type"] == "notes" and c["source_id"] and "Nyquist" in c["text"]
    assert isinstance(c["score"], float) and c["metadata"]["title"].startswith("Notes ")
    assert all(x["score"] is None for x in search(env, "s1@x.io", query="Nyquist", course_id=int(env.A),
                                                  include_scores=False).json()["results"])
    assert len(search(env, "s1@x.io", query="Nyquist", course_id=int(env.A), top_k=500).json()["results"]) <= 20


def test_search_access_rules_and_input_checks(env):
    assert search(env, "t@x.io", query="Nyquist", course_id=int(env.A)).status_code == 200          # owner
    assert search(env, "out@x.io", query="Nyquist", course_id=int(env.A)).status_code == 403        # not a member
    assert search(env, "s1@x.io", query="x", course_id=999).status_code == 404
    assert search(env, "s1@x.io", query="x", content_types=["bogus"]).status_code == 400
    assert search(env, "s1@x.io", query="").status_code == 422
    empty = search(env, "s1@x.io", query="Nyquist", content_types=["code", "exercises"])
    assert empty.status_code == 200 and empty.json()["total_returned"] == 0     # not ingested yet
    assert search(env, "out@x.io", query="Nyquist").json()["total_returned"] == 0  # no private course material


def test_context_roundtrip_and_ownership(env):
    c = env.client
    s1, s2 = env.h("s1@x.io"), env.h("s2@x.io")
    conv = c.post("/api/chat/conversations", json={"title": "t"}, headers=s1).json()["id"]
    assert c.get(f"{BASE}/context/{conv}", headers=s1).status_code == 404            # nothing stored yet
    r = c.post(f"{BASE}/context", json={"conversation_id": conv, "summary": "Talked about aliasing.",
                                        "metadata": {"last_intent": "THEORY"}}, headers=s1)
    assert r.status_code == 200
    got = c.get(f"{BASE}/context/{conv}", headers=s1).json()
    assert got["summary"] == "Talked about aliasing." and got["metadata"] == {"last_intent": "THEORY"} and got["updated_at"]

    async def seed():  # the chat's own rolling-summary bookmark lives in the same row
        from sqlalchemy.ext.asyncio import AsyncSession
        factory = c.app.state.db_session_factory
        async with factory() as s:
            await s.execute(text("UPDATE saarthi_conversation_context SET metadata = :m WHERE conversation_id = :c"),
                            {"m": '{"up_to_message_id": 42}', "c": str(conv)})
            await s.commit()

    asyncio.run(seed())
    c.post(f"{BASE}/context", json={"conversation_id": conv, "summary": "Third.", "metadata": {"last_topic": "aliasing"}}, headers=s1)
    got = c.get(f"{BASE}/context/{conv}", headers=s1).json()
    assert got["metadata"] == {"last_topic": "aliasing", "up_to_message_id": 42}

    assert c.get(f"{BASE}/context/{conv}", headers=s2).status_code == 404            # not their conversation
    assert c.post(f"{BASE}/context", json={"conversation_id": conv, "summary": "hijack"}, headers=s2).status_code == 404
    assert c.get(f"{BASE}/context/not-a-number", headers=s1).status_code == 404
    assert c.post(f"{BASE}/context", json={"conversation_id": conv, "metadata": {"k": "x" * 9000}}, headers=s1).status_code == 400
    assert c.post(f"{BASE}/context", json={"conversation_id": conv, "summary": "x" * 5000}, headers=s1).status_code == 422
    assert c.get(f"{BASE}/context/{conv}").status_code == 401


# ── retrieval scope through the API ─────────────────────────────────────────

def _spy_retrieval(monkeypatch, store_as):
    """Replace the chat entry points with ones that run an unscoped retrieve() inside the request."""
    from fastapi.concurrency import run_in_threadpool
    from src.tools.retrieve import retrieve

    seen = {}

    async def fake_stateless(message, history, course_id=None, **kw):
        chunks = await run_in_threadpool(lambda: retrieve("Nyquist sampling notes", course_id=course_id, k=10))
        seen["text"] = " ".join(c.text for c in chunks)
        return "ok"

    monkeypatch.setattr(chat_service, "stateless_message", fake_stateless)
    return seen


def test_unscoped_chat_only_reaches_the_callers_own_courses(env, monkeypatch):
    seen = _spy_retrieval(monkeypatch, "text")
    post = lambda who, **b: env.client.post("/api/chat/message", json={"message": "hi", **b}, headers=env.h(who) if who else {})

    assert post("s1@x.io").status_code == 200
    assert "MARKER_ALPHA" in seen["text"] and "MARKER_BRAVO" not in seen["text"]
    assert post(None).status_code == 200 and seen["text"] == ""                  # signed-out visitor: no course material
    post("t2@x.io")
    assert "MARKER_BRAVO" in seen["text"] and "MARKER_ALPHA" not in seen["text"]
    post("adm@x.io")
    assert "MARKER_ALPHA" in seen["text"] and "MARKER_BRAVO" in seen["text"]     # admin is unrestricted
    post("s1@x.io", courseId=int(env.B))
    assert seen["text"] == ""                                                    # explicit foreign course id
    post("s1@x.io", courseId=int(env.A))
    assert "MARKER_ALPHA" in seen["text"]


def test_streamed_unscoped_chat_inherits_the_scope(env, monkeypatch):
    from fastapi.concurrency import run_in_threadpool
    import saarthi_backend.ai.adapter as adapter
    from src.tools.retrieve import retrieve

    seen = {}

    async def fake_stream(question, history, **kw):
        chunks = await run_in_threadpool(lambda: retrieve("Nyquist sampling notes", k=10))
        seen["text"] = " ".join(c.text for c in chunks)
        yield adapter._sse_data("done")
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(adapter, "run_chat_stream", fake_stream)
    assert stream(env, "s1@x.io", "hello there").status_code == 200
    assert "MARKER_ALPHA" in seen["text"] and "MARKER_BRAVO" not in seen["text"]


# ── PII never reaches the model from study tools ────────────────────────────

@pytest.fixture
def prompts(monkeypatch):
    import saarthi_backend.service.code_service as code_service
    import src.utils.llm as llm

    sent = []

    class Completions:
        @staticmethod
        async def create(**kw):
            sent.append(" ".join(m["content"] for m in kw["messages"]))
            body = json.dumps({"weekPlan": [], "summary": "s", "priorityTopics": [], "cards": [], "challenge": "c",
                               "hints": [], "predictions": [], "nodes": [], "explanation": "e", "suggestions": []})
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=body))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions))
    for mod in (chat_service, code_service):
        monkeypatch.setattr(mod, "async_openai", lambda: client)
    monkeypatch.setattr(llm, "async_openai", lambda: client)
    return sent


def _no_pii(prompt):
    return all(x not in prompt for x in ("Priya", "Sharma", "priya.s@", "S20990000001", "98765"))


def test_study_tools_scrub_pii_and_stay_inside_the_callers_courses(env, prompts):
    c, s1 = env.client, env.h("s1@x.io")

    c.post("/api/chat/study-plan", json={"hoursPerDay": 2, "focusArea": f"help {PII} with Nyquist", "courses": [],
                                         "deadlines": [], "recentQuizScores": []}, headers=s1)
    assert _no_pii(prompts[-1]) and "[email]" in prompts[-1] and "MARKER_BRAVO" not in prompts[-1]

    c.post("/api/chat/socratic", json={"claim": f"I am {PII}, Nyquist is half the rate", "topic": "Nyquist",
                                       "difficulty": "moderate"}, headers=s1)
    assert _no_pii(prompts[-1]) and "MARKER_ALPHA" in prompts[-1] and "MARKER_BRAVO" not in prompts[-1]

    c.post("/api/chat/flashcards/generate", json={"topic": "Nyquist", "sourceText": f"Notes by {PII}. Nyquist rate = 2 fmax.",
                                                  "maxCards": 3}, headers=s1)
    assert _no_pii(prompts[-1]) and "Nyquist rate = 2 fmax" in prompts[-1]

    c.post("/api/chat/exam-prediction", json={"courseTitle": "Signals A", "hoursUntilExam": 10, "recentQuizScores": [],
                                              "notesTopics": ["mail me priya.s@iiit.ac.in", "Nyquist"]}, headers=s1)
    assert "priya.s@" not in prompts[-1] and "Nyquist" in prompts[-1]

    c.post("/api/chat/concept-graph", json={"courseTitle": "Signals A", "courses": [], "deadlines": [],
                                            "recentQuizScores": []}, headers=s1)
    assert "MARKER_ALPHA" in prompts[-1] and "MARKER_BRAVO" not in prompts[-1]

    r = c.post("/api/quizzes/generate", json={"topic": "Nyquist sampling", "numQuestions": 3, "pastAttempts": []}, headers=s1)
    assert r.status_code == 200 and "MARKER_ALPHA" in prompts[-1] and "MARKER_BRAVO" not in prompts[-1]
    assert "add no outside facts" in prompts[-1]


def test_code_explain_keeps_code_but_drops_emails_and_roll_numbers(env, prompts):
    body = {"code": "x = 1234567890123\nprint('mail priya.s@iiit.ac.in S20990000001')", "language": "python",
            "stdout": "", "stderr": "Error for priya.s@iiit.ac.in", "exitCode": 1}
    r = env.client.post("/api/code/explain", json=body, headers=env.h("s1@x.io"))
    assert r.status_code == 200
    assert "priya.s@" not in prompts[-1] and "S20990000001" not in prompts[-1] and "1234567890123" in prompts[-1]


# ── Ask the Notes page: AI core by default, Gemini only when switched on ───

@pytest.fixture
def kb_model(monkeypatch):
    import saarthi_backend.routers.kb_router as kb

    sent = []

    def fake(system, user):
        sent.append(system + "\n" + user)
        return "The Nyquist rate is twice the highest frequency [Notes MARKER_ALPHA]."

    monkeypatch.setenv("CHAT_SOURCE", "core")  # our model deployed
    monkeypatch.setattr(kb, "_llm_answer", fake)
    return sent


def ask_notes(env, who, question):
    return env.client.post("/api/kb/ask", json={"question": question, "topK": 5}, headers=env.h(who))


def test_ask_notes_uses_the_ai_core_scoped_and_scrubbed(env, kb_model):
    assert env.client.get("/api/kb/status", headers=env.h("s1@x.io")).json()["source"] == "core"
    r = ask_notes(env, "s1@x.io", f"I am {PII}. What is the Nyquist rate?")
    assert r.status_code == 200, r.text
    body = r.json()
    texts = " ".join(s["text"] for s in body["sources"])
    assert body["sources"] and "MARKER_ALPHA" in texts and "MARKER_BRAVO" not in texts   # caller's course only
    assert "Nyquist" in body["answer"]
    assert len(kb_model) == 1 and _no_pii(kb_model[0])                                  # nothing personal sent


def test_ask_notes_refuses_uncovered_questions_without_a_model_call(env, kb_model):
    r = ask_notes(env, "s1@x.io", "What is the capital city of Peru?")
    assert r.status_code == 200 and r.json()["sources"] == []
    assert "couldn't find" in r.json()["answer"].lower() and kb_model == []
    assert ask_notes(env, "out@x.io", "What is the Nyquist rate?").json()["sources"] == []  # not enrolled anywhere


def test_ask_notes_falls_back_to_the_ai_core_without_a_gemini_index(env, kb_model, monkeypatch):
    from src.knowledge_base import gemini_rag

    monkeypatch.delenv("CHAT_SOURCE", raising=False)
    monkeypatch.setattr(gemini_rag, "index_ready", lambda: False)
    assert env.client.get("/api/kb/status", headers=env.h("s1@x.io")).json()["source"] == "core"
    assert ask_notes(env, "s1@x.io", "What is the Nyquist rate?").json()["sources"]


def test_ask_notes_uses_gemini_by_default_and_scrubs_the_question(env, monkeypatch):
    from src.knowledge_base import gemini_rag

    seen = {}

    def fake_answer(question, k, history=None):
        seen["q"] = question
        return {"answer": "ok", "sources": []}

    monkeypatch.delenv("CHAT_SOURCE", raising=False)  # default
    monkeypatch.setattr(gemini_rag, "index_ready", lambda: True)
    monkeypatch.setattr(gemini_rag, "answer", fake_answer)
    r = ask_notes(env, "s1@x.io", f"I am {PII}. What is the Nyquist rate?")
    assert r.status_code == 200 and r.json()["answer"] == "ok"
    assert _no_pii(seen["q"]) and "Nyquist" in seen["q"]


def test_chat_uses_gemini_by_default_scrubbed_and_the_ai_core_with_chat_source_core(env, monkeypatch):
    from src.knowledge_base import gemini_rag

    seen = {}

    def fake_answer(question, k, history=None):
        seen["q"] = question
        return {"answer": "from gemini", "sources": []}

    async def fake_run_chat(prompt, history, mind_mode=False):
        seen["core"] = prompt
        return "from the AI core"

    monkeypatch.setattr(gemini_rag, "index_ready", lambda: True)
    monkeypatch.setattr(gemini_rag, "answer", fake_answer)
    monkeypatch.setattr(chat_service, "run_chat", fake_run_chat)
    h = env.h("s1@x.io")
    conv = env.client.post("/api/chat/conversations", json={"title": "t"}, headers=h).json()["id"]
    send = lambda: env.client.post(f"/api/chat/conversations/{conv}/messages",
                                   json={"message": f"I am {PII}. What is aliasing?"}, headers=h)

    monkeypatch.delenv("CHAT_SOURCE", raising=False)
    r = send()
    assert r.status_code in (200, 201), r.text
    assert "from gemini" in r.text and _no_pii(seen["q"])

    monkeypatch.setenv("CHAT_SOURCE", "core")  # our model deployed
    r = send()
    assert "from the AI core" in r.text and _no_pii(seen["core"])
