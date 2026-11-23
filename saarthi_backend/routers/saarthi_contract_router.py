"""Orchestrator-facing HTTP contract (backend framework docs 03/04/11), served under /api/v1/saarthi.

  POST /retrieval/search           raw retrieval (no LLM)
  GET  /context/{conversation_id}  stored conversation summary + metadata
  POST /context                    upsert it

The expert endpoints (/experts/*) are not served: the experts run in-process (src/experts).
Every call needs a signed-in user. Course content is private: searching it needs enrolment in
that course (or being its owner / an admin); without a course_id only the shared knowledge base
(notes, books, video) is searched. Context is readable and writable by the conversation's owner.
"""
import json
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from saarthi_backend.dao import ConversationDAO
from saarthi_backend.deps import get_current_user, get_db
from saarthi_backend.model import Course, Material, User
from saarthi_backend.service import course_service
from saarthi_backend.service.access_scope import allowed_course_ids
from saarthi_backend.utils.exceptions import ForbiddenError, NotFoundError, ValidationError

router = APIRouter(prefix="/v1/saarthi", tags=["saarthi-contract"])

_MAX_TOP_K = 20
_MAX_SUMMARY_CHARS = 4000
_MAX_METADATA_BYTES = 8000
_SHARED_TYPES = {"notes": "notes", "books": "books", "video": "video"}
_COURSE_TYPES = {"code", "exercises", "exercise"}


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    content_types: list[str] | None = None
    top_k: int = Field(5, ge=1)
    include_scores: bool = True
    course_id: int | None = Field(None, description="Search this course's materials (enrolled students, owner, admin).")
    video_id: int | None = Field(None, description="Search one video's transcript.")
    include_courses: bool = Field(False, description="Include all course indexes the caller can access.")


class SearchChunk(BaseModel):
    content_type: str
    source_id: str
    text: str
    score: float | None = None
    metadata: dict[str, Any] = {}


class SearchResponse(BaseModel):
    results: list[SearchChunk]
    query: str
    total_returned: int


class ContextUpsert(BaseModel):
    conversation_id: str = Field(..., min_length=1, max_length=36)
    summary: str | None = Field(None, max_length=_MAX_SUMMARY_CHARS)
    metadata: dict[str, Any] = {}


class ContextResponse(BaseModel):
    conversation_id: str
    summary: str | None = None
    metadata: dict[str, Any] = {}
    updated_at: str | None = None


def _contract_type(internal: str) -> str:
    return {"course_material": "notes", "books": "notes", "notes": "notes", "video": "video"}.get(internal, internal)


async def _require_course_access(db: AsyncSession, user: User, course_id: int) -> None:
    course = await db.get(Course, course_id)
    if course is None:
        raise NotFoundError("Course not found.", details=None)
    if user.role == "admin" or course.owner_id == user.id:
        return
    if not await course_service.get_enrollment(db, user.id, course_id):
        raise ForbiddenError("Join this course to search its materials.", details=None)


@router.post("/retrieval/search", response_model=SearchResponse)
async def retrieval_search(
    body: SearchRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Ranked chunks for a query, same hybrid retrieval the chat uses."""
    from src.tools.retrieve import allowed_courses, retrieve

    requested = [t.lower() for t in (body.content_types or [])]
    unknown = [t for t in requested if t not in _SHARED_TYPES and t not in _COURSE_TYPES]
    if unknown:
        raise ValidationError(f"Unknown content type(s): {', '.join(unknown)}. Use notes, video, code or exercises.", details=None)

    if body.course_id is not None:
        await _require_course_access(db, user, body.course_id)
    if body.video_id is not None:
        video = await _get_video(db, body.video_id)
        if video is None or (body.course_id is not None and body.course_id != video.course_id):
            raise NotFoundError("Video not found.", details=None)
        if video.course_id is not None:
            await _require_course_access(db, user, video.course_id)

    search_course_indexes = body.include_courses or body.course_id is not None or body.video_id is not None
    shared = [_SHARED_TYPES[t] for t in requested if t in _SHARED_TYPES]
    if requested and not shared and not search_course_indexes:
        return SearchResponse(results=[], query=body.query, total_returned=0)  # only code/exercises asked for
    top_k = min(body.top_k, _MAX_TOP_K)
    scope = await allowed_course_ids(db, user) if search_course_indexes else set()
    token = allowed_courses.set(scope)
    try:
        chunks = await run_in_threadpool(
            lambda: retrieve(
                body.query,
                content_types=None if search_course_indexes else (shared or ["notes", "books", "video"]),
                course_id=body.course_id,
                video_id=body.video_id,
                k=top_k * 4 if search_course_indexes and requested else top_k,
            )
        )
    finally:
        allowed_courses.reset(token)

    ids = {c.material_id for c in chunks if c.material_id}
    materials = {m.id: m for m in (await db.scalars(select(Material).where(Material.id.in_(ids))))} if ids else {}
    tags = {_contract_type(t) for t in requested if t != "exercises"}
    if "exercises" in requested:
        tags.add("exercise")
    results = []
    for c in chunks:
        if scope is not None and c.course_id is not None and c.course_id not in scope:
            continue
        material = materials.get(c.material_id)
        content_type = _contract_type(c.content_type)
        if material and material.course_id == c.course_id and material.type in ("exercise", "code"):
            content_type = material.type
        if c.video_id is not None or c.start_sec is not None:
            content_type = "video"
        if tags and search_course_indexes and content_type not in tags:
            continue
        results.append(SearchChunk(
            content_type=content_type,
            source_id=str(c.id or ""),
            text=c.text,
            score=round(float(c.score), 4) if (body.include_scores and c.score is not None) else None,
            metadata={k: v for k, v in {
                "title": c.title, "source_file": c.source_file, "page": c.page,
                "start_sec": c.start_sec, "end_sec": c.end_sec,
                "course_id": c.course_id, "material_id": c.material_id, "video_id": c.video_id,
            }.items() if v is not None},
        ))
        if len(results) >= top_k:
            break
    return SearchResponse(results=results, query=body.query, total_returned=len(results))


async def _get_video(db: AsyncSession, video_id: int):
    from saarthi_backend.model import Video

    return await db.get(Video, video_id)


async def _owned_conversation_id(db: AsyncSession, user: User, conversation_id: str) -> int:
    """Conversation ids are the chat conversations' integer ids; only their owner may touch the context."""
    try:
        cid = int(conversation_id)
    except (TypeError, ValueError):
        raise NotFoundError("Conversation not found.", details=None)
    if await ConversationDAO.get_by_id(db, cid, user.id) is None:
        raise NotFoundError("Conversation not found.", details=None)
    return cid


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, (str, bytes)):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}
    return {}


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


@router.get("/context/{conversation_id}", response_model=ContextResponse)
async def get_context(
    conversation_id: str,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Stored context of one of the caller's conversations; 404 when there is none."""
    cid = await _owned_conversation_id(db, user, conversation_id)
    row = (await db.execute(
        text("SELECT summary, metadata, updated_at FROM saarthi_conversation_context WHERE conversation_id = :cid"),
        {"cid": str(cid)},
    )).fetchone()
    if row is None:
        raise NotFoundError("No stored context for this conversation.", details=None)
    return ContextResponse(conversation_id=str(cid), summary=row[0], metadata=_as_dict(row[1]), updated_at=_iso(row[2]))


@router.post("/context", response_model=ContextResponse)
async def upsert_context(
    body: ContextUpsert,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Upsert summary + metadata for the caller's conversation. Replaces both (not a merge); the
    chat's own rolling summary lives in the same row (metadata.up_to_message_id), so that key is kept."""
    cid = await _owned_conversation_id(db, user, body.conversation_id)
    meta_json = json.dumps(body.metadata or {})
    if len(meta_json.encode("utf-8")) > _MAX_METADATA_BYTES:
        raise ValidationError(f"metadata is limited to {_MAX_METADATA_BYTES} bytes.", details=None)

    existing = (await db.execute(
        text("SELECT metadata FROM saarthi_conversation_context WHERE conversation_id = :cid"), {"cid": str(cid)},
    )).fetchone()
    meta = dict(body.metadata or {})
    kept = _as_dict(existing[0]).get("up_to_message_id") if existing else None
    if kept is not None and "up_to_message_id" not in meta:
        meta["up_to_message_id"] = kept
    is_pg = db.get_bind().dialect.name == "postgresql"
    cast = "CAST(:meta AS jsonb)" if is_pg else ":meta"
    now_sql = "now()" if is_pg else "CURRENT_TIMESTAMP"
    await db.execute(
        text(f"""
            INSERT INTO saarthi_conversation_context (conversation_id, summary, metadata, updated_at)
            VALUES (:cid, :summary, {cast}, {now_sql})
            ON CONFLICT (conversation_id) DO UPDATE
              SET summary = EXCLUDED.summary, metadata = EXCLUDED.metadata, updated_at = {now_sql}
        """),
        {"cid": str(cid), "summary": body.summary, "meta": json.dumps(meta)},
    )
    await db.commit()
    return ContextResponse(
        conversation_id=str(cid), summary=body.summary, metadata=meta,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
