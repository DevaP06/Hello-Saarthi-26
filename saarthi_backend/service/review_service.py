"""Instructor review queue: flag weak AI answers, let instructors correct them (AI WBS 3.3.4 / 3.6.3).

Flow:
  1. A course-scoped answer ends "low" confidence (not covered / uncited / critic FAIL) or the
     student rates it thumbs-down  ->  record_review() (one open row per student+question).
  2. The instructor lists the open rows and writes the correct answer  ->  correct_review().
  3. The correction is indexed into the course (indexing_service.index_instructor_correction),
     the cached answers for the course are dropped, and the next student gets it, cited as
     "[Instructor correction]".
"""
from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta, timezone

from fastapi.concurrency import run_in_threadpool
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from saarthi_backend.dao.review_dao import AnswerReviewDAO
from saarthi_backend.model import ChatMessage, Conversation, Course, User
from saarthi_backend.model.review_model import AnswerReview
from saarthi_backend.utils.exceptions import ForbiddenError, NotFoundError, ValidationError
from saarthi_backend.utils.logging import get_logger

logger = get_logger(__name__)

_MAX_ANSWER_CHARS = 6000


async def record_review(
    db: AsyncSession,
    *,
    course_id: int,
    user_id: int | None,
    query: str,
    answer: str,
    reply_mode: str | None,
    confidence: str | None,
    reasons: list[str] | None,
    source: str,
) -> AnswerReview | None:
    """Add (or bump) an open review row. Never raises: flagging must not break a chat turn."""
    query = (query or "").strip()
    if not query or not course_id:
        return None
    try:
        async with db.begin_nested():
            existing = await AnswerReviewDAO.find_open(db, course_id, user_id, query)
            if existing is not None:
                existing.occurrences = (existing.occurrences or 1) + 1
                if answer:
                    existing.answer = answer[:_MAX_ANSWER_CHARS]
                if source == "thumbs_down":
                    existing.source = source
                return existing
            return await AnswerReviewDAO.create(
                db,
                course_id=course_id,
                user_id=user_id,
                query=query[:2000],
                answer=(answer or "")[:_MAX_ANSWER_CHARS],
                reply_mode=reply_mode,
                confidence=confidence,
                reasons=json.dumps(reasons or []),
                source=source,
                status="open",
                occurrences=1,
            )
    except Exception as e:
        logger.warning("review not recorded: %s", e)
        return None


async def require_course_staff(db: AsyncSession, user: User, course_id: int) -> Course:
    """Admin, or the teacher who owns the course."""
    if user.role not in ("admin", "teacher"):
        raise ForbiddenError("Teacher or admin role required.", details=None)
    course = await db.get(Course, course_id)
    if course is None:
        raise NotFoundError("Course not found.", details=None)
    if user.role != "admin" and course.owner_id != user.id:
        raise ForbiddenError("Only the course owner can review its AI answers.", details=None)
    return course


async def get_review_for_staff(db: AsyncSession, user: User, review_id: int) -> AnswerReview:
    review = await AnswerReviewDAO.get(db, review_id)
    if review is None:
        raise NotFoundError("Review not found.", details=None)
    await require_course_staff(db, user, review.course_id)
    return review


async def correct_review(db: AsyncSession, review: AnswerReview, correction: str, instructor_id: int) -> AnswerReview:
    """Save the instructor's answer, then make it retrievable. Caller commits first."""
    from datetime import datetime, timezone

    correction = (correction or "").strip()
    if len(correction) < 5:
        raise ValidationError("Write the correct answer (at least a sentence).", details=None)
    review.correction = correction[:4000]
    review.status = "corrected"
    review.corrected_by = instructor_id
    review.corrected_at = datetime.now(timezone.utc)
    await db.flush()
    return review


async def index_correction(review: AnswerReview) -> bool:
    from saarthi_backend.service.indexing_service import index_instructor_correction

    return await run_in_threadpool(
        index_instructor_correction, review.course_id, review.id, review.query, review.correction or ""
    )


async def dismiss_review(db: AsyncSession, review: AnswerReview) -> AnswerReview:
    review.status = "dismissed"
    await db.flush()
    return review


def _course_questions(course_id: int):
    return (
        select(ChatMessage).join(Conversation, Conversation.id == ChatMessage.conversation_id)
        .join(User, User.id == Conversation.user_id)
        .where(ChatMessage.course_id == course_id, ChatMessage.role == "user",
               ChatMessage.shared_query.is_not(None), ChatMessage.shared_query != "", User.role == "student")
    )


async def course_query_log(
    db: AsyncSession, course_id: int, limit: int, offset: int, topic: str, on_date: date | None,
) -> tuple[list[dict], int, list[dict]]:
    from src.safety.guard import scrub_pii

    questions = _course_questions(course_id)
    if topic:
        questions = questions.where(func.lower(ChatMessage.topic).contains(topic.lower(), autoescape=True))
    if on_date:
        start = datetime.combine(on_date, time.min, tzinfo=timezone.utc)
        questions = questions.where(ChatMessage.created_at >= start)
        if on_date != date.max:
            questions = questions.where(ChatMessage.created_at < start + timedelta(days=1))

    total = await db.scalar(select(func.count()).select_from(questions.subquery()))
    answer = aliased(ChatMessage)
    rows = await db.execute(
        questions.add_columns(answer, User.full_name)
        .outerjoin(answer, answer.reply_to_id == ChatMessage.id)
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc()).limit(limit).offset(offset)
    )
    items = []
    for question, reply, name in rows:
        metadata = reply.answer_metadata or {} if reply else {}
        items.append({
            "id": str(question.id), "conversationId": str(question.conversation_id), "topic": question.topic or "General",
            "query": question.shared_query,
            "answer": scrub_pii(reply.content, [name] if name else [])[0] if reply else None,
            "confidence": metadata.get("confidence"), "replyMode": metadata.get("reply_mode"),
            "rating": reply.rating if reply else None, "instructorNote": question.instructor_note,
            "createdAt": question.created_at.isoformat(),
        })

    filtered = questions.subquery()
    group_topic = func.coalesce(filtered.c.topic, "General")
    occurrences = func.count()
    repeated = (
        select(group_topic.label("topic"), func.min(filtered.c.shared_query).label("query"), occurrences.label("count"))
        .group_by(group_topic, func.lower(filtered.c.shared_query)).having(occurrences > 1).subquery()
    )
    ranked = select(repeated, func.row_number().over(
        partition_by=repeated.c.topic, order_by=(repeated.c.count.desc(), repeated.c.query),
    ).label("rank")).subquery()
    frequent = (await db.execute(
        select(ranked.c.topic, ranked.c.query, ranked.c.count)
        .where(ranked.c.rank <= 5).order_by(ranked.c.topic, ranked.c.rank)
    )).mappings().all()
    return items, total or 0, [dict(row) for row in frequent]


async def annotate_course_query(db: AsyncSession, course_id: int, query_id: int, note: str) -> str | None:
    question = await db.scalar(_course_questions(course_id).where(ChatMessage.id == query_id))
    if question is None:
        raise NotFoundError("Course query not found.", details=None)
    question.instructor_note = note.strip() or None
    await db.commit()
    return question.instructor_note


async def correct_course_query(
    db: AsyncSession, course_id: int, query_id: int, correction: str, instructor_id: int,
) -> AnswerReview:
    from src.safety.guard import scrub_pii

    row = (await db.execute(
        _course_questions(course_id).add_columns(User.id, User.full_name)
        .where(ChatMessage.id == query_id).with_for_update(of=ChatMessage)
    )).one_or_none()
    if row is None:
        raise NotFoundError("Course query not found.", details=None)
    question, user_id, student_name = row
    query = question.shared_query
    review = await db.scalar(select(AnswerReview)
                             .where(AnswerReview.course_id == course_id, AnswerReview.user_id == user_id,
                                    AnswerReview.query == query, AnswerReview.status.in_(("open", "corrected")))
                             .order_by((AnswerReview.status == "open").desc(), AnswerReview.id.desc()).limit(1))
    if review is None:
        reply = await db.scalar(select(ChatMessage).where(ChatMessage.reply_to_id == query_id).limit(1))
        metadata = reply.answer_metadata or {} if reply else {}
        original = scrub_pii(reply.content, [student_name] if student_name else [])[0] if reply else ""
        review = await AnswerReviewDAO.create(
            db, course_id=course_id, user_id=user_id, query=query, answer=original[:_MAX_ANSWER_CHARS],
            reply_mode=metadata.get("reply_mode"), confidence=metadata.get("confidence"),
            reasons=json.dumps(metadata.get("confidence_reasons") or []), source="instructor", status="open", occurrences=1,
        )
    return await correct_review(db, review, correction, instructor_id)


def review_to_dict(r: AnswerReview, student_name: str | None = None) -> dict:
    try:
        reasons = json.loads(r.reasons or "[]")
    except ValueError:
        reasons = []
    return {
        "id": str(r.id),
        "courseId": str(r.course_id),
        "studentName": student_name,
        "query": r.query,
        "answer": r.answer,
        "replyMode": r.reply_mode,
        "confidence": r.confidence,
        "reasons": reasons,
        "source": r.source,
        "status": r.status,
        "occurrences": r.occurrences,
        "correction": r.correction,
        "createdAt": r.created_at.isoformat() if r.created_at else "",
        "correctedAt": r.corrected_at.isoformat() if r.corrected_at else None,
    }
