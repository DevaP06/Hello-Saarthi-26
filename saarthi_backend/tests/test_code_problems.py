"""Code Lab problems by course lesson: who sees, adds, edits and deletes them.

Real FastAPI app on SQLite in-memory, with get_db/get_current_user overridden
(same pattern as test_qa_review.py).
"""
from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from saarthi_backend.deps import get_current_user, get_db
from saarthi_backend.main import app
from saarthi_backend.model import CodeProblem, Course, Enrollment, User


def _user(user_id: int, role: str) -> User:
    u = User(email=f"u{user_id}@test.local", password_hash="x", full_name=f"User {user_id}", role=role)
    u.id = user_id
    return u


@pytest_asyncio.fixture
async def client_as(db_session):
    """Teacher 1 owns course 10 (student 3 enrolled), teacher 2 owns course 20,
    student 4 has no course; one practice problem (no course) exists."""
    db_session.add_all([_user(1, "teacher"), _user(2, "teacher"), _user(3, "student"), _user(4, "student")])
    db_session.add_all([
        Course(id=10, title="DSP", code="DSP", instructor="T1", owner_id=1),
        Course(id=20, title="ML", code="ML", instructor="T2", owner_id=2),
    ])
    await db_session.flush()
    db_session.add(Enrollment(user_id=3, course_id=10))
    db_session.add(CodeProblem(
        id=1, title="Practice", description="p",
        test_cases_json='[{"input": "", "expectedOutput": "1", "hidden": false}]',
    ))
    await db_session.commit()

    async def _db():
        yield db_session

    def _make(role: str, user_id: int):
        app.dependency_overrides[get_db] = _db
        app.dependency_overrides[get_current_user] = lambda: _user(user_id, role)
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    yield _make
    app.dependency_overrides.clear()


async def _titles(c: AsyncClient) -> list[str]:
    return [p["title"] for p in (await c.get("/api/code/problems")).json()]


@pytest.mark.asyncio
async def test_lesson_problems_access_and_editing(client_as):
    new = {
        "title": "FFT size", "description": "Print 8", "courseId": 10, "topic": "Fourier",
        "testCases": [{"input": "", "expectedOutput": "8", "hidden": True}],
    }

    # Teachers add problems only to their own courses; course-less practice problems are admin-only.
    async with client_as("teacher", 2) as c:
        assert (await c.post("/api/code/problems", json=new)).status_code == 403
        assert (await c.post("/api/code/problems", json={**new, "courseId": None})).status_code == 400
    async with client_as("teacher", 1) as c:
        r = await c.post("/api/code/problems", json=new)
        assert r.status_code == 201, r.text
        pid = r.json()["id"]
        assert r.json()["canEdit"] is True
        assert r.json()["testCases"][0]["expectedOutput"] == "8"  # the editor sees hidden cases in full

    # The enrolled student sees practice + course problem, with the hidden case masked.
    async with client_as("student", 3) as c:
        got = {p["title"]: p for p in (await c.get("/api/code/problems")).json()}
        assert list(got) == ["Practice", "FFT size"]  # practice first
        fft = got["FFT size"]
        assert (fft["courseTitle"], fft["topic"], fft["canEdit"]) == ("DSP", "Fourier", False)
        assert fft["testCases"] == [{"input": None, "expectedOutput": None, "hidden": True}]
        assert (await c.patch(f"/api/code/problems/{pid}", json={"title": "x"})).status_code == 403

    # Outsiders cannot see or change it; practice problems stay admin-only.
    async with client_as("student", 4) as c:
        assert await _titles(c) == ["Practice"]
        assert (await c.get(f"/api/code/problems/{pid}")).status_code == 404
    async with client_as("teacher", 2) as c:
        assert (await c.patch(f"/api/code/problems/{pid}", json={"title": "x"})).status_code == 404
        assert (await c.patch("/api/code/problems/1", json={"title": "x"})).status_code == 403

    # The owner edits it; renaming the lesson carries the problem along; deleting hides it.
    async with client_as("teacher", 1) as c:
        r = await c.patch(f"/api/code/problems/{pid}", json={"title": "FFT length"})
        assert r.json()["title"] == "FFT length"
        fourier = next(m for m in (await c.get("/api/courses/10/modules")).json() if m["title"] == "Fourier")
        assert (await c.patch(f"/api/courses/10/modules/{fourier['id']}", json={"title": "DFT & FFT"})).status_code == 200
        assert (await c.get(f"/api/code/problems/{pid}")).json()["topic"] == "DFT & FFT"
        assert (await c.delete(f"/api/code/problems/{pid}")).status_code == 204
    async with client_as("student", 3) as c:
        assert await _titles(c) == ["Practice"]
