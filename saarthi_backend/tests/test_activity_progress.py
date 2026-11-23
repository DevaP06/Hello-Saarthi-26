"""Activity permissions, retry safety, and the student and teacher dashboard paths."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from saarthi_backend.deps import get_current_user, get_db
from saarthi_backend.main import app
from saarthi_backend.model import ActivityEvent, CodeProblem, CodeSubmission, Course, CourseModule, Enrollment, Material, User, Video


@pytest_asyncio.fixture
async def activity_client(db_session):
    people = {
        user_id: User(id=user_id, email=f"person{user_id}@example.test", full_name=f"Person {user_id}",
                      role=role, password_hash="unused")
        for user_id, role in ((1, "student"), (2, "student"), (3, "student"), (4, "teacher"), (5, "teacher"))
    }
    db_session.add_all(people.values())
    await db_session.flush()
    db_session.add_all([
        Course(id=11, title="DSP", code="DSP", owner_id=4, instructor="Teacher"),
        Course(id=22, title="Other", code="OTHER", owner_id=5, instructor="Other teacher"),
    ])
    await db_session.flush()
    db_session.add_all([
        Enrollment(user_id=1, course_id=11), Enrollment(user_id=2, course_id=11),
        CourseModule(course_id=11, title="Sampling", sort_order=0),
        CourseModule(course_id=11, title="Transforms", sort_order=1),
        Material(id=51, course_id=11, title="Notes", type="pdf", topic="sampling", url="/notes.pdf"),
        Video(id=61, course_id=11, title="Lecture", topic="Transforms", duration_seconds=120, url="/video.mp4"),
        CodeProblem(id=71, course_id=11, title="Sampling Exercise", topic="Sampling"),
    ])
    await db_session.flush()
    db_session.add_all([
        CodeSubmission(user_id=1, problem_id=71, language="python", code="pass", passed=1, total=2),
        ActivityEvent(user_id=2, event_id=str(uuid4()), course_id=11, material_id=51,
                      created_at=datetime.now(timezone.utc) - timedelta(days=9)),
    ])
    await db_session.commit()

    async def get_test_db():
        yield db_session

    def as_user(user_id):
        app.dependency_overrides[get_db] = get_test_db
        app.dependency_overrides[get_current_user] = lambda: people[user_id]
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    yield as_user
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_resource_activity_is_scoped_and_updates_readership_and_progress(activity_client):
    opened = {"eventId": str(uuid4()), "resourceType": "material", "resourceId": 51, "activeSeconds": 0}
    heartbeat = {**opened, "eventId": str(uuid4()), "activeSeconds": 20}

    async with activity_client(3) as outsider:
        assert (await outsider.post("/api/progress/activity", json=opened)).status_code == 404
    async with activity_client(5) as teacher:
        assert (await teacher.get("/api/progress/class?courseId=11")).status_code == 404
        assert (await teacher.post("/api/progress/activity", json=opened)).status_code == 403

    async with activity_client(1) as student:
        assert (await student.get("/api/progress/class")).status_code == 403
        assert (await student.post("/api/progress/activity", json={**heartbeat, "activeSeconds": 31})).status_code == 422
        assert (await student.post("/api/progress/activity", json=opened)).status_code == 204
        for _ in range(2):
            assert (await student.post("/api/progress/activity", json=heartbeat)).status_code == 204
        response = await student.get("/api/progress/me")
        assert response.status_code == 200, response.text
        progress = response.json()
        assert progress["monthly"]["materialOpens"] == 1
        assert progress["monthly"]["studySeconds"] == 20
        assert progress["summary"]["streakDays"] == 1
        assert progress["topics"][0]["accuracy"] == 50
        course = progress["courses"][0]
        assert course["modules"][0] == {
            "title": "Sampling", "covered": True, "completed": 2, "total": 2, "studySeconds": 20,
        }
        assert course["estimatedCompletionDate"]

    async with activity_client(4) as teacher:
        response = await teacher.get("/api/progress/class?courseId=11")
        assert response.status_code == 200, response.text
        readership = response.json()["performance"]["readership"]
        assert readership["activeLearners7Days"] == 1
        assert readership["topics"][0] == {"topic": "Sampling", "learners": 1, "opens": 1, "averageSeconds": 10}
        assert readership["dropoffs"] == [{"fromTopic": "Sampling", "toTopic": "Transforms", "learners": 1}]

    async with activity_client(1) as student:
        video_open = {**opened, "eventId": str(uuid4()), "resourceType": "video", "resourceId": 61}
        assert (await student.post("/api/progress/activity", json=video_open)).status_code == 204
        response = await student.put("/api/videos/61/progress", json={"positionSeconds": 120, "completed": True})
        assert response.status_code == 200, response.text
        progress = (await student.get("/api/progress/me")).json()
        assert progress["monthly"]["videoOpens"] == 1
        assert progress["courses"][0]["coverage"] == 100
        assert progress["courses"][0]["estimatedCompletionDate"] is None
