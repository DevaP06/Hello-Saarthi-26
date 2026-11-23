"""Instructor review queue for AI answers (under /api/reviews)."""

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from saarthi_backend.dao.review_dao import AnswerReviewDAO
from saarthi_backend.deps import get_current_user, get_db, get_pagination
from saarthi_backend.model import User
from saarthi_backend.schema.common_schemas import PaginatedResponse, PaginationParams
from saarthi_backend.service import review_service
from saarthi_backend.utils.logging import get_logger

router = APIRouter(prefix="/reviews", tags=["ai-review"])
logger = get_logger(__name__)


class ReviewResponse(BaseModel):
    id: str
    courseId: str
    studentName: str | None = None
    query: str
    answer: str
    replyMode: str | None = None
    confidence: str | None = None
    reasons: list[str] = []
    source: str
    status: str
    occurrences: int
    correction: str | None = None
    createdAt: str
    correctedAt: str | None = None


class CorrectionRequest(BaseModel):
    correction: str = Field(..., min_length=5, max_length=4000)


class CorrectionResponse(BaseModel):
    review: ReviewResponse
    indexed: bool


class ReviewStatsResponse(BaseModel):
    open: int = 0
    corrected: int = 0
    dismissed: int = 0
    total: int = 0


class QueryLogItem(BaseModel):
    id: str
    conversationId: str
    topic: str
    query: str
    answer: str | None = None
    confidence: str | None = None
    replyMode: str | None = None
    rating: str | None = None
    instructorNote: str | None = None
    createdAt: str


class FrequentQuery(BaseModel):
    topic: str
    query: str
    count: int


class QueryLogResponse(PaginatedResponse[QueryLogItem]):
    frequent: list[FrequentQuery]


class QueryNoteRequest(BaseModel):
    note: str = Field(..., max_length=4000)


@router.get("/queries", response_model=QueryLogResponse)
async def list_course_queries(
    courseId: int,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    pagination: Annotated[PaginationParams, Depends(get_pagination)],
    topic: Annotated[str | None, Query(max_length=128)] = None,
    onDate: date | None = None,
):
    await review_service.require_course_staff(db, user, courseId)
    items, total, frequent = await review_service.course_query_log(
        db, courseId, pagination.limit, pagination.offset, (topic or "").strip(), onDate,
    )
    return QueryLogResponse(items=items, total=total, limit=pagination.limit, offset=pagination.offset, frequent=frequent)


@router.post("/queries/{query_id}/correct", response_model=CorrectionResponse)
async def correct_course_query(
    courseId: int,
    query_id: int,
    body: CorrectionRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    await review_service.require_course_staff(db, user, courseId)
    review = await review_service.correct_course_query(db, courseId, query_id, body.correction, user.id)
    await db.commit()
    return CorrectionResponse(review=ReviewResponse(**review_service.review_to_dict(review)),
                              indexed=await review_service.index_correction(review))


@router.patch("/queries/{query_id}/note")
async def annotate_course_query(
    courseId: int,
    query_id: int,
    body: QueryNoteRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    await review_service.require_course_staff(db, user, courseId)
    return {"instructorNote": await review_service.annotate_course_query(db, courseId, query_id, body.note)}


@router.get("", response_model=PaginatedResponse[ReviewResponse])
async def list_reviews(
    courseId: int,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    pagination: Annotated[PaginationParams, Depends(get_pagination)],
    status: str | None = "open",
):
    """AI answers of this course that need an instructor (open / corrected / dismissed)."""
    await review_service.require_course_staff(db, user, courseId)
    status = None if status in (None, "", "all") else status
    rows = await AnswerReviewDAO.list_by_course(db, courseId, status, pagination.limit, pagination.offset)
    total = await AnswerReviewDAO.count_filtered(db, courseId, status)
    names: dict[int, str] = {}
    uids = {r.user_id for r in rows if r.user_id}
    if uids:
        for u in (await db.execute(select(User).where(User.id.in_(uids)))).scalars().all():
            names[u.id] = u.full_name or "Student"
    return PaginatedResponse(
        items=[ReviewResponse(**review_service.review_to_dict(r, names.get(r.user_id))) for r in rows],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get("/stats", response_model=ReviewStatsResponse)
async def review_stats(
    courseId: int,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    await review_service.require_course_staff(db, user, courseId)
    counts = await AnswerReviewDAO.count_by_course(db, courseId)
    return ReviewStatsResponse(
        open=counts.get("open", 0), corrected=counts.get("corrected", 0),
        dismissed=counts.get("dismissed", 0), total=sum(counts.values()),
    )


@router.post("/{review_id}/correct", response_model=CorrectionResponse)
async def correct_review(
    review_id: int,
    body: CorrectionRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Save the instructor's answer; it is indexed into the course so the AI uses it next time."""
    review = await review_service.get_review_for_staff(db, user, review_id)
    await review_service.correct_review(db, review, body.correction, user.id)
    await db.commit()
    indexed = await review_service.index_correction(review)
    return CorrectionResponse(review=ReviewResponse(**review_service.review_to_dict(review)), indexed=indexed)


@router.post("/{review_id}/dismiss", response_model=ReviewResponse)
async def dismiss_review(
    review_id: int,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    review = await review_service.get_review_for_staff(db, user, review_id)
    await review_service.dismiss_review(db, review)
    await db.commit()
    return ReviewResponse(**review_service.review_to_dict(review))
