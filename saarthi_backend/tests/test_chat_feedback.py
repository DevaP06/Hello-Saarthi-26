"""Thumbs up/down is saved on the answer, and only on the student's own answers."""
from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from saarthi_backend.deps import get_current_user, get_db
from saarthi_backend.main import app
from saarthi_backend.model import User
from saarthi_backend.model.chat_model import ChatMessage, Conversation


def _user(user_id: int) -> User:
    u = User(email=f"s{user_id}@test.local", password_hash="x", full_name=f"Student {user_id}", role="student")
    u.id = user_id
    return u


@pytest_asyncio.fixture
async def client_as(db_session):
    db_session.add_all([_user(1), _user(2)])
    await db_session.flush()
    db_session.add(Conversation(id=1, user_id=1, title="FFT"))
    await db_session.flush()
    db_session.add_all([
        ChatMessage(id=10, conversation_id=1, role="user", content="What is FFT?"),
        ChatMessage(id=11, conversation_id=1, role="assistant", content="A fast DFT algorithm."),
    ])
    await db_session.commit()

    async def _db():
        yield db_session

    def _make(user_id: int):
        app.dependency_overrides[get_db] = _db
        app.dependency_overrides[get_current_user] = lambda: _user(user_id)
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    yield _make, db_session
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_rating_saved_on_own_answer_only(client_as):
    make, db = client_as
    body = {"query": "What is FFT?", "rating": "up", "messageId": 11}

    async with make(2) as c:  # another student
        assert (await c.post("/api/chat/feedback", json=body)).status_code == 404
    async with make(1) as c:
        assert (await c.post("/api/chat/feedback", json={**body, "messageId": 10})).status_code == 404  # not an answer
        r = await c.post("/api/chat/feedback", json=body)
        assert r.status_code == 200, r.text
        detail = (await c.get("/api/chat/conversations/1")).json()
    assert (await db.get(ChatMessage, 11)).rating == "up"
    assert [m.get("rating") for m in detail["messages"]] == [None, "up"]
