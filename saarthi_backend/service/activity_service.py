"""Record student activity and summarize it for learning dashboards."""

from collections import defaultdict
from datetime import datetime, timedelta
from math import ceil
from uuid import uuid4

from sqlalchemy import case, func, select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from saarthi_backend.model import (
    ActivityEvent,
    Assignment,
    AssignmentSubmission,
    CodeProblem,
    CodeSubmission,
    Enrollment,
    Material,
    Quiz,
    QuizAttempt,
    User,
    Video,
    VideoProgress,
)
from saarthi_backend.schema.activity_schemas import ResourceActivityRequest
from saarthi_backend.utils.exceptions import ForbiddenError, NotFoundError


def record_signin(db: AsyncSession, user: User) -> None:
    if user.role == "student":
        db.add(ActivityEvent(user_id=user.id, event_id=str(uuid4())))


async def record_resource_activity(db: AsyncSession, user: User, body: ResourceActivityRequest) -> None:
    if user.role != "student":
        raise ForbiddenError("Only student activity is recorded.", details=None)

    is_material = body.resourceType == "material"
    resource = await db.get(Material if is_material else Video, body.resourceId)
    if resource is None:
        raise NotFoundError("Resource not found.", details=None)

    course_id = resource.course_id
    if course_id is not None:
        enrolled = await db.scalar(
            select(Enrollment.id).where(Enrollment.course_id == course_id, Enrollment.user_id == user.id).limit(1)
        )
        if enrolled is None:
            raise NotFoundError("Resource not found.", details=None)

    # Retries use the same event ID, including an overlapping request after a network timeout.
    insert = sqlite_insert if db.get_bind().dialect.name == "sqlite" else postgres_insert
    await db.execute(insert(ActivityEvent).values(
        user_id=user.id,
        event_id=str(body.eventId),
        course_id=course_id,
        material_id=resource.id if is_material else None,
        video_id=None if is_material else resource.id,
        active_seconds=body.activeSeconds,
    ).on_conflict_do_nothing(index_elements=[ActivityEvent.user_id, ActivityEvent.event_id]))
    await db.commit()


def _resource_events():
    return ActivityEvent.material_id.is_not(None) | ActivityEvent.video_id.is_not(None)


def topic_key(name: str | None) -> str:
    return (name or "").strip().casefold() or "general"


async def student_topic_activity(db: AsyncSession, user_id: int, course_ids: list[int]) -> dict[tuple[int, str], int]:
    seconds_by_topic: dict[tuple[int, str], int] = defaultdict(int)
    if course_ids:
        topic = func.coalesce(Material.topic, Video.topic)
        rows = await db.execute(
            select(ActivityEvent.course_id, topic, func.sum(ActivityEvent.active_seconds))
            .outerjoin(Material, Material.id == ActivityEvent.material_id)
            .outerjoin(Video, Video.id == ActivityEvent.video_id)
            .where(ActivityEvent.user_id == user_id, ActivityEvent.course_id.in_(course_ids), _resource_events())
            .group_by(ActivityEvent.course_id, topic)
        )
        for course_id, name, seconds in rows:
            seconds_by_topic[(course_id, topic_key(name))] += seconds
    return seconds_by_topic


async def syllabus_items(db: AsyncSession, user_id: int, course_ids: list[int]) -> dict[tuple[int, str], list[datetime | None]]:
    """First completion (or None) for each current syllabus item."""
    by_topic: dict[tuple[int, str], list[datetime | None]] = defaultdict(list)
    if not course_ids:
        return by_topic

    for resource, resource_id, learner_id, completed_at, filters in (
        (Material, ActivityEvent.material_id, ActivityEvent.user_id, ActivityEvent.created_at, ()),
        (Video, VideoProgress.video_id, VideoProgress.user_id, VideoProgress.updated_at, (VideoProgress.completed.is_(True),)),
        (Quiz, QuizAttempt.quiz_id, QuizAttempt.user_id, QuizAttempt.submitted_at, ()),
        (CodeProblem, CodeSubmission.problem_id, CodeSubmission.user_id, CodeSubmission.created_at, ()),
        (Assignment, AssignmentSubmission.assignment_id, AssignmentSubmission.user_id, AssignmentSubmission.submitted_at, ()),
    ):
        first_completion = (
            select(resource_id.label("resource_id"), func.min(completed_at).label("at"))
            .where(learner_id == user_id, *filters).group_by(resource_id).subquery()
        )
        statement = (
            select(resource.course_id, resource.topic, first_completion.c.at)
            .outerjoin(first_completion, first_completion.c.resource_id == resource.id)
            .where(resource.course_id.in_(course_ids))
        )
        if resource is CodeProblem:
            statement = statement.where(CodeProblem.is_active.is_(True))
        for course_id, topic, completion in await db.execute(statement):
            by_topic[(course_id, topic_key(topic))].append(completion)
    return by_topic


def estimated_finish(dates: list[datetime | None], now: datetime) -> str | None:
    """Project the current 30-day pace of completed modules onto remaining modules."""
    remaining = dates.count(None)
    from_date = (now - timedelta(days=30)).date()
    finished_recently = sum(value is not None and value.date() >= from_date for value in dates)
    if not remaining or not finished_recently:
        return None
    return (now.date() + timedelta(days=ceil(remaining * 30 / finished_recently))).isoformat()


async def total_study_seconds(db: AsyncSession, user_id: int) -> int:
    return await db.scalar(select(func.coalesce(func.sum(ActivityEvent.active_seconds), 0))
                           .where(ActivityEvent.user_id == user_id))


async def monthly_activity(db: AsyncSession, user_id: int, month_start: datetime) -> dict:
    opens = case((ActivityEvent.active_seconds == 0, 1), else_=0)
    result = await db.execute(
        select(
            func.coalesce(func.sum(ActivityEvent.active_seconds), 0),
            func.coalesce(func.sum(case((ActivityEvent.material_id.is_not(None), opens), else_=0)), 0),
            func.coalesce(func.sum(case((ActivityEvent.video_id.is_not(None), opens), else_=0)), 0),
        ).where(ActivityEvent.user_id == user_id, ActivityEvent.created_at >= month_start, _resource_events())
    )
    seconds, material_opens, video_opens = result.one()
    return {"studySeconds": seconds, "materialOpens": material_opens, "videoOpens": video_opens}


async def class_readership(
    db: AsyncSession, course_id: int, student_ids: list[int], topics: list[str], now: datetime,
) -> dict:
    """Summarize the last seven days. Drop-off compares consecutive topics in course order."""
    since = now - timedelta(days=7)
    topic = func.coalesce(Material.topic, Video.topic, "General")
    rows = await db.execute(
        select(
            topic, ActivityEvent.user_id, func.sum(ActivityEvent.active_seconds),
            func.sum(case((ActivityEvent.active_seconds == 0, 1), else_=0)),
        )
        .outerjoin(Material, Material.id == ActivityEvent.material_id)
        .outerjoin(Video, Video.id == ActivityEvent.video_id)
        .where(ActivityEvent.course_id == course_id, ActivityEvent.user_id.in_(student_ids),
               ActivityEvent.created_at >= since, _resource_events())
        .group_by(topic, ActivityEvent.user_id)
    ) if student_ids else []

    names = {topic_key(name): name for name in topics}
    topic_keys = list(names)
    visitors: dict[str, set[int]] = {}
    active_students: set[int] = set()
    seconds_by_topic: dict[str, int] = {}
    opens_by_topic: dict[str, int] = {}
    for name, user_id, seconds, opens in rows:
        key = topic_key(name)
        names.setdefault(key, name.strip() or "General")
        visitors.setdefault(key, set()).add(user_id)
        active_students.add(user_id)
        seconds_by_topic[key] = seconds_by_topic.get(key, 0) + seconds
        opens_by_topic[key] = opens_by_topic.get(key, 0) + opens

    if student_ids:
        enrolled = set(student_ids)
        for work, attempt, foreign_key, completed_at in (
            (Quiz, QuizAttempt, QuizAttempt.quiz_id, QuizAttempt.submitted_at),
            (Assignment, AssignmentSubmission, AssignmentSubmission.assignment_id, AssignmentSubmission.submitted_at),
            (CodeProblem, CodeSubmission, CodeSubmission.problem_id, CodeSubmission.created_at),
        ):
            participants = await db.scalars(
                select(attempt.user_id).join(work, work.id == foreign_key)
                .where(work.course_id == course_id, completed_at >= since).distinct()
            )
            active_students.update(enrolled.intersection(participants))
    dropoffs = [
        {"fromTopic": names[first], "toTopic": names[second],
         "learners": len(visitors.get(first, set()) - visitors.get(second, set()))}
        for first, second in zip(topic_keys, topic_keys[1:])
        if visitors.get(first, set()) - visitors.get(second, set())
    ]
    return {
        "activeLearners7Days": len(active_students),
        "topics": [
            {"topic": name, "learners": len(visitors.get(key, set())), "opens": opens_by_topic.get(key, 0),
             "averageSeconds": round(seconds_by_topic.get(key, 0) / len(student_ids)) if student_ids else 0}
            for key, name in names.items()
        ],
        "dropoffs": dropoffs,
    }
