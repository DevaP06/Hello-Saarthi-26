"""Which courses' materials may this request read? (retrieval access scope)

The course indexes are private to the course's members. An unscoped question ("global" chat, study
tools) used to search every course on disk; now the API binds the caller's courses to
`src.tools.retrieve.allowed_courses` for the request and retrieval only looks there.
"""
from typing import Annotated

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from saarthi_backend.deps import get_current_user, get_current_user_optional, get_db
from saarthi_backend.model import Course, Enrollment, User


async def allowed_course_ids(db: AsyncSession, user: User) -> set[int] | None:
    """Courses the user is enrolled in or owns. Admins are unrestricted (None)."""
    if user.role == "admin":
        return None
    enrolled = (await db.execute(select(Enrollment.course_id).where(Enrollment.user_id == user.id))).scalars().all()
    owned = (await db.execute(select(Course.id).where(Course.owner_id == user.id))).scalars().all()
    return {int(i) for i in (*enrolled, *owned)}


async def retrieval_scope(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> set[int] | None:
    """Route dependency: bind the caller's courses for this request's retrieval calls."""
    from src.tools.retrieve import allowed_courses

    ids = await allowed_course_ids(db, user)
    allowed_courses.set(ids)
    return ids


async def retrieval_scope_optional(
    user: Annotated[User | None, Depends(get_current_user_optional)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> set[int] | None:
    """For endpoints open to signed-out visitors (the floating chatbot): a visitor reaches no
    course material, a signed-in user reaches theirs."""
    from src.tools.retrieve import allowed_courses

    ids: set[int] | None = set() if user is None else await allowed_course_ids(db, user)
    allowed_courses.set(ids)
    return ids
