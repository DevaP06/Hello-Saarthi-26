"""Spaced review of questions a student has already seen graded."""

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from saarthi_backend.model import Enrollment, Quiz, QuizAttempt, QuizPractice, QuizQuestion, User
from saarthi_backend.schema.quiz_schemas import QuizPracticeAnswer, QuizPracticeFeedback, QuizPracticeQuestion, QuizPracticeQueue
from saarthi_backend.utils.exceptions import ForbiddenError, NotFoundError, ValidationError

_CORRECT_INTERVALS = (1, 3, 7, 14, 30)
_SOURCE_LABEL = re.compile(r"(?:^|\s)Source:\s*(.+)$", re.IGNORECASE | re.DOTALL)


def _utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)


@dataclass
class PracticeItem:
    question: QuizQuestion
    quiz: Quiz
    attempt_id: int
    options: list[str]
    schedule: QuizPractice | None
    due_at: datetime
    last_correct: bool

    @property
    def current_schedule(self) -> QuizPractice | None:
        return self.schedule if self.schedule and self.schedule.attempt_id == self.attempt_id else None

    @property
    def topic(self) -> str:
        return self.question.topic or self.quiz.topic or "General"


async def _practice_items(
    db: AsyncSession, user: User, *, course_id: int | None = None, question_id: int | None = None,
) -> list[PracticeItem]:
    if user.role != "student":
        raise ForbiddenError("Practice is available to students.", details=None)

    enrolled = select(Enrollment.course_id).where(Enrollment.user_id == user.id)
    latest = (
        select(QuizAttempt.id, QuizAttempt.quiz_id, QuizAttempt.answers_json, QuizAttempt.submitted_at,
               func.row_number().over(partition_by=QuizAttempt.quiz_id,
                                      order_by=(QuizAttempt.submitted_at.desc(), QuizAttempt.id.desc())).label("rank"))
        .where(QuizAttempt.user_id == user.id, QuizAttempt.submitted_at.is_not(None), QuizAttempt.score.is_not(None))
        .subquery()
    )
    statement = (
        select(QuizQuestion, Quiz, latest.c.id, latest.c.answers_json, latest.c.submitted_at, QuizPractice)
        .join(Quiz, Quiz.id == QuizQuestion.quiz_id)
        .join(latest, and_(latest.c.quiz_id == Quiz.id, latest.c.rank == 1))
        .outerjoin(QuizPractice, and_(QuizPractice.question_id == QuizQuestion.id, QuizPractice.user_id == user.id))
        .where(or_(Quiz.course_id.is_(None), Quiz.course_id.in_(enrolled)),
               func.coalesce(QuizQuestion.revised_at, QuizQuestion.created_at) <= latest.c.submitted_at)
    )
    if course_id is not None:
        statement = statement.where(Quiz.course_id == course_id)
    if question_id is not None:
        statement = statement.where(QuizQuestion.id == question_id)

    parsed_attempts: dict[int, dict[str, int | None]] = {}
    items = []
    for question, quiz, attempt_id, answers_json, submitted_at, schedule in await db.execute(statement):
        if attempt_id not in parsed_attempts:
            try:
                rows = json.loads(answers_json or "[]")
            except ValueError:
                rows = []
            parsed_attempts[attempt_id] = {
                str(row.get("questionId")): row.get("selectedIndex")
                for row in rows if isinstance(row, dict) and "questionId" in row
            } if isinstance(rows, list) else {}
        answers = parsed_attempts[attempt_id]
        if str(question.id) not in answers:
            continue
        try:
            options = json.loads(question.options_json)
        except ValueError:
            continue
        if not isinstance(options, list) or not all(isinstance(value, str) for value in options):
            continue
        if not 0 <= question.correct_index < len(options):
            continue

        if schedule and schedule.attempt_id == attempt_id:
            due_at = _utc(schedule.next_review_at)
            correct = schedule.correct_streak > 0
        else:
            correct = answers[str(question.id)] == question.correct_index
            due_at = _utc(submitted_at) + timedelta(days=3 if correct else 0)
        items.append(PracticeItem(question, quiz, attempt_id, options, schedule, due_at, correct))
    return items


async def review_queue(
    db: AsyncSession, user: User, *, limit: int, course_id: int | None, topic: str,
) -> QuizPracticeQueue:
    now = datetime.now(timezone.utc)
    items = await _practice_items(db, user, course_id=course_id)
    topic_scores: dict[tuple[int | None, str], list[bool]] = defaultdict(list)

    def topic_key(item: PracticeItem) -> tuple[int | None, str]:
        return item.quiz.course_id, item.topic.strip().casefold()

    for item in items:
        topic_scores[topic_key(item)].append(item.last_correct)

    def weak(item: PracticeItem) -> bool:
        scores = topic_scores[topic_key(item)]
        return sum(scores) * 100 < len(scores) * 70

    if topic:
        items = [item for item in items if topic.casefold() in item.topic.casefold()]
    due = [item for item in items if item.due_at <= now]
    due.sort(key=lambda item: (not weak(item), item.last_correct, item.due_at, item.question.id))
    upcoming = min((item.due_at for item in items if item.due_at > now), default=None)
    return QuizPracticeQueue(
        items=[QuizPracticeQuestion(
            id=str(item.question.id), attemptId=item.attempt_id,
            revision=item.current_schedule.revision if item.current_schedule else 0,
            quizTitle=item.quiz.title, courseId=item.quiz.course_id, topic=item.topic,
            difficulty=item.question.difficulty, questionText=item.question.question_text, options=item.options,
            dueAt=item.due_at.isoformat(), weakTopic=weak(item),
        ) for item in due[:limit]],
        dueCount=len(due), nextReviewAt=upcoming.isoformat() if upcoming else None,
    )


def _feedback(item: PracticeItem, review: QuizPractice) -> QuizPracticeFeedback:
    explanation = item.question.explanation or ""
    source = _SOURCE_LABEL.search(explanation)
    return QuizPracticeFeedback(
        questionId=str(item.question.id), selectedIndex=review.selected_index, correctIndex=item.question.correct_index,
        isCorrect=review.selected_index == item.question.correct_index,
        explanation=(explanation[:source.start()].strip() if source else explanation.strip()) or None,
        source=source.group(1).strip() if source else item.quiz.title,
        nextReviewAt=_utc(review.next_review_at).isoformat(),
    )


async def answer_review(db: AsyncSession, user: User, question_id: int, body: QuizPracticeAnswer) -> QuizPracticeFeedback:
    # A retry or a second tab cannot advance the same review twice.
    await db.execute(select(User.id).where(User.id == user.id).with_for_update())
    await db.execute(select(QuizQuestion.id).where(QuizQuestion.id == question_id).with_for_update())
    items = await _practice_items(db, user, question_id=question_id)
    if not items:
        raise NotFoundError("Practice question not found.", details=None)
    item = items[0]
    current = item.current_schedule
    if body.attemptId != item.attempt_id or body.selectedIndex >= len(item.options):
        raise ValidationError("This practice question has changed. Reload the review queue.", details=None)

    if current and current.revision == body.revision + 1 and current.selected_index == body.selectedIndex:
        return _feedback(item, current)
    now = datetime.now(timezone.utc)
    revision = current.revision if current else 0
    if body.revision != revision or item.due_at > now:
        raise ValidationError("This question is no longer due. Reload the review queue.", details=None)

    correct = body.selectedIndex == item.question.correct_index
    streak = ((current.correct_streak if current else 0) + 1) if correct else 0
    interval = timedelta(days=_CORRECT_INTERVALS[min(streak, len(_CORRECT_INTERVALS)) - 1]) if correct else timedelta(minutes=10)
    schedule = item.schedule or QuizPractice(user_id=user.id, question_id=question_id)
    schedule.attempt_id = item.attempt_id
    schedule.revision = revision + 1
    schedule.correct_streak = streak
    schedule.selected_index = body.selectedIndex
    schedule.next_review_at = now + interval
    db.add(schedule)
    await db.commit()
    return _feedback(item, schedule)
