"""DAO for the instructor review queue."""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from saarthi_backend.model.review_model import AnswerReview


class AnswerReviewDAO:
    @staticmethod
    async def get(db: AsyncSession, review_id: int) -> AnswerReview | None:
        return (await db.execute(select(AnswerReview).where(AnswerReview.id == review_id))).scalar_one_or_none()

    @staticmethod
    async def find_open(db: AsyncSession, course_id: int, user_id: int | None, query: str) -> AnswerReview | None:
        """The same student asking the same thing again should bump one row, not add another."""
        q = select(AnswerReview).where(
            AnswerReview.course_id == course_id,
            AnswerReview.status == "open",
            func.lower(AnswerReview.query) == query.strip().lower(),
        )
        q = q.where(AnswerReview.user_id == user_id) if user_id is not None else q.where(AnswerReview.user_id.is_(None))
        return (await db.execute(q.limit(1))).scalar_one_or_none()

    @staticmethod
    async def create(db: AsyncSession, **fields) -> AnswerReview:
        row = AnswerReview(**fields)
        db.add(row)
        await db.flush()
        return row

    @staticmethod
    async def list_by_course(
        db: AsyncSession, course_id: int, status: str | None = None, limit: int = 20, offset: int = 0
    ) -> list[AnswerReview]:
        q = select(AnswerReview).where(AnswerReview.course_id == course_id)
        if status:
            q = q.where(AnswerReview.status == status)
        q = q.order_by(AnswerReview.occurrences.desc(), AnswerReview.created_at.desc()).limit(limit).offset(offset)
        return list((await db.execute(q)).scalars().all())

    @staticmethod
    async def count_by_course(db: AsyncSession, course_id: int | None = None) -> dict[str, int]:
        q = select(AnswerReview.status, func.count()).group_by(AnswerReview.status)
        if course_id is not None:
            q = q.where(AnswerReview.course_id == course_id)
        return {status: n for status, n in (await db.execute(q)).all()}

    @staticmethod
    async def count_filtered(db: AsyncSession, course_id: int, status: str | None) -> int:
        q = select(func.count()).select_from(AnswerReview).where(AnswerReview.course_id == course_id)
        if status:
            q = q.where(AnswerReview.status == status)
        return (await db.execute(q)).scalar() or 0
