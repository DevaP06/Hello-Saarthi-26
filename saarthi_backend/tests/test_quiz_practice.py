"""Review exposes feedback only for the learner's own previously graded questions."""

from datetime import datetime, timedelta, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from saarthi_backend.deps import get_current_user, get_db
from saarthi_backend.main import app
from saarthi_backend.model import Course, Enrollment, Quiz, QuizPractice, QuizQuestion, User


@pytest.mark.asyncio
async def test_spaced_review_preserves_assessment_boundaries_and_retries(db_session):
    users = {key: User(id=key, email=f"review{key}@example.test", full_name=f"User {key}", role=role, password_hash="x")
             for key, role in ((1, "student"), (2, "student"), (3, "teacher"))}
    db_session.add_all(users.values())
    await db_session.flush()
    db_session.add(Course(id=1, owner_id=3, title="Digital Signal Processing", code="DSP", instructor="Instructor"))
    await db_session.flush()
    db_session.add_all([Enrollment(user_id=1, course_id=1), Enrollment(user_id=2, course_id=1),
                        Quiz(id=11, course_id=1, title="Sampling quiz", topic="Sampling")])
    await db_session.flush()
    db_session.add(QuizQuestion(id=21, quiz_id=11, question_text="What causes aliasing?", correct_index=0,
                               options_json='["Undersampling", "Oversampling"]',
                               explanation="Copies of the spectrum overlap. Source: Lecture 2, p. 4"))
    await db_session.commit()

    async def get_test_db():
        yield db_session

    def as_user(user_id):
        app.dependency_overrides[get_db] = get_test_db
        app.dependency_overrides[get_current_user] = lambda: users[user_id]
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    queue_url = "/api/quizzes/review/queue"
    answer_url = "/api/quizzes/review/21/answer"
    try:
        async with as_user(1) as student:
            before = await student.get("/api/quizzes/11")
            assert before.json()["questions"][0]["correctIndex"] is None
            assert before.json()["questions"][0]["explanation"] is None
            started = await student.post("/api/quizzes/11/attempts")
            attempt_id = started.json()["id"]
            answer = {"attemptId": int(attempt_id), "revision": 0, "selectedIndex": 1}
            assert (await student.get(queue_url)).json()["items"] == []
            assert (await student.post(answer_url, json=answer)).status_code == 404
            submitted = await student.post(f"/api/quizzes/11/attempts/{attempt_id}/submit", json={
                "answers": [{"questionId": "21", "selectedIndex": 1}],
            })
            assert submitted.status_code == 200 and submitted.json()["score"] == 0
        async with as_user(2) as other_student:
            assert (await other_student.get(queue_url)).json()["items"] == []
            assert (await other_student.post(answer_url, json=answer)).status_code == 404
        async with as_user(1) as student:
            queue = (await student.get(queue_url, params={"courseId": 1, "topic": "sampling"})).json()
            assert queue["dueCount"] == 1 and queue["items"][0]["weakTopic"]
            assert "correctIndex" not in queue["items"][0]
            checked = await student.post(answer_url, json=answer)
            assert checked.status_code == 200, checked.text
            feedback = checked.json()
            assert not feedback["isCorrect"] and feedback["correctIndex"] == 0
            assert feedback["explanation"] == "Copies of the spectrum overlap."
            assert feedback["source"] == "Lecture 2, p. 4"
            assert (await student.post(answer_url, json=answer)).json() == feedback
            assert (await student.get(queue_url)).json()["dueCount"] == 0
            assert (await student.post(answer_url, json={**answer, "selectedIndex": 0})).status_code == 400

            schedule = await db_session.get(QuizPractice, (1, 21))
            schedule.next_review_at = datetime.now(timezone.utc) - timedelta(seconds=1)
            await db_session.commit()
            correct = await student.post(answer_url, json={**answer, "revision": 1, "selectedIndex": 0})
            assert correct.status_code == 200 and correct.json()["isCorrect"]
            assert datetime.fromisoformat(correct.json()["nextReviewAt"]) > datetime.now(timezone.utc) + timedelta(hours=23)
        async with as_user(3) as instructor:
            updated = await instructor.patch("/api/quizzes/11/questions/21", json={"correctIndex": 1})
            assert updated.status_code == 200, updated.text
        async with as_user(1) as student:
            assert (await student.post(answer_url, json={**answer, "revision": 2})).status_code == 404
    finally:
        app.dependency_overrides.clear()
