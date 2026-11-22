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

from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from saarthi_backend.dao.review_dao import AnswerReviewDAO
from saarthi_backend.model import Course, User
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
