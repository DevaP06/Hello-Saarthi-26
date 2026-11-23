"""Chat routes: stateless message, conversation CRUD, SSE stream, study plan,
socratic, concept graph, exam prediction, flashcards."""
import logging
from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from saarthi_backend.deps import get_current_user, get_db, get_pagination
from saarthi_backend.dao import ConversationDAO
from saarthi_backend.model import Course, CourseModule, Material, User, Video
from saarthi_backend.schema.chat_schemas import (
    AnswerFeedbackRequest,
    AnswerFeedbackResponse,
    ChatMessageRequest,
    ChatMessageResponse,
    ChatMessageResponseItem,
    ConceptGraphResponse,
    ConversationDetailResponse,
    ConversationResponse,
    CreateConversationRequest,
    ExamPredictionRequest,
    ExamPredictionResponse,
    FlashcardDeckRequest,
    FlashcardDeckResponse,
    FlashcardListResponse,
    FlashcardReviewRequest,
    FlashcardReviewResponse,
    SendMessageRequest,
    SendMessageResponse,
    SocraticChallengeRequest,
    SocraticChallengeResponse,
    StreamMessageRequest,
    StudyPlanRequest,
    StudyPlanResponse,
    UpdateConversationRequest,
)
from saarthi_backend.schema.common_schemas import PaginatedResponse, PaginationParams
from saarthi_backend.service import chat_service
from saarthi_backend.service.access_scope import retrieval_scope, retrieval_scope_optional
from saarthi_backend.utils.exceptions import NotFoundError, ValidationError

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)

# Exact sentinel frames emitted by adapter.run_chat_stream — used to tell a
# real streamed token apart from a control frame. Previously this was done
# with `chunk.startswith("data: [")`, which silently dropped any legitimate
# answer whose *first token* happened to render as e.g. "[1]" (a citation
# marker the tutor is explicitly instructed to use).
_SSE_DONE = "data: [DONE]\n\n"
_SSE_ERROR_PREFIX = "data: [ERROR]"


def _names(user: User) -> list[str]:
    return [user.full_name] if user.full_name else []


def _is_content_frame(chunk: str) -> bool:
    """True if this SSE frame carries an actual answer token."""
    if chunk.startswith("event: delta\ndata: "):
        return True
    if not chunk.startswith("data: "):
        return False
    if chunk == _SSE_DONE:
        return False
    if chunk.startswith(_SSE_ERROR_PREFIX):
        return False
    return True


def _conversation_to_response(c) -> ConversationResponse:
    return ConversationResponse(
        id=str(c.id),
        title=c.title,
        createdAt=c.created_at.isoformat(),
        updatedAt=c.updated_at.isoformat(),
        courseId=c.course_id,
    )


def _message_to_item(m) -> ChatMessageResponseItem:
    return ChatMessageResponseItem(
        id=str(m.id),
        role=m.role,
        content=m.content,
        createdAt=m.created_at.isoformat(),
        rating=m.rating if m.rating in ("up", "down") else None,
        courseId=m.course_id,
        topic=m.topic,
        replyToId=m.reply_to_id,
        answerMetadata=m.answer_metadata,
    )


# ─── Stateless (AIChatbot widget) ────────────────────────────────────────────
@router.post("/message", response_model=ChatMessageResponse)
async def chat_message(
    body: ChatMessageRequest,
    _scope: Annotated[set[int] | None, Depends(retrieval_scope_optional)],
):
    """Stateless chat (AIChatbot, material viewer). No persistence."""
    history = [{"role": m.role, "content": m.content} for m in body.conversationHistory]
    answer = await chat_service.stateless_message(
        body.message, history,
        context_material_title=body.contextMaterialTitle,
        course_id=body.courseId,
        context_video_id=body.contextVideoId,
        context_video_title=body.contextVideoTitle,
    )
    return ChatMessageResponse(response=answer)


# ─── SSE Streaming endpoint ───────────────────────────────────────────────────
def _retrieval_query(question: str, history: list[dict], policy) -> str:
    """What to search for this turn. Socratic continuations search for the thread's problem;
    other follow-ups prepend the previous student question; everything else uses the question."""
    if policy.socratic.stage in ("hint", "reveal") and policy.socratic.thread_query:
        if policy.socratic.thread_query.strip() != question.strip():
            return f"{policy.socratic.thread_query} {question}"
        return question
    if policy.qclass.kind == "followup":
        prev = next((m.get("content", "") for m in reversed(history) if m.get("role") == "user"), "")
        return f"{prev} {question}".strip()
    return question


@router.post("/stream")
async def stream_message(
    request: Request,
    body: StreamMessageRequest,
    scope: Annotated[set[int] | None, Depends(retrieval_scope)],
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Stream an AI response token-by-token via Server-Sent Events.
    Also persists the full exchange to the conversation when complete.

    Per turn: scrub PII → query classifier (Tier 0 = stored repeated query served from the
    semantic cache with no LLM call; else Tier 1 / Tier 2) → Socratic stage → answer ONLY
    from the course materials (grounding rule) with inline citations → trace.
    """
    import time as _time

    from saarthi_backend.ai.adapter import _sse_data, run_chat_stream, sse_delta_text, stream_scoped_answer, structured_sse
    from saarthi_backend.service.memory_service import build_context, maybe_update_summary_bg
    from saarthi_backend.service.weak_topics_service import flag_weak_topic, get_weak_topics
    from saarthi_backend.dao import ChatMessageDAO
    from saarthi_backend.service import course_service
    from src.cache.semantic_cache import get_cache
    from src.orchestrator.policy import plan_turn
    from src.orchestrator.topics import topic_of
    from src.safety.guard import check_injection, scrub_pii
    from src.telemetry.traces import hash_user, summarize_usage, write_trace
    from src.tools.grounding import ground, ground_multi
    from src.tools.retrieve import allowed_courses
    from src.utils import budget
    from src.utils.llm import chat_model_name, reset_tier, set_tier

    json_delivery = request.headers.get("accept") == "application/json"
    if json_delivery:
        if not body.conversationId:
            raise ValidationError("Start a conversation before sending a chat turn.", details=None)
        body.structured = True
    saved_turn: SendMessageResponse | None = None

    # No student PII reaches the LLM, the cache or the trace (spec 2.3, R7).
    question, pii_scrubbed = scrub_pii(body.message, [user.full_name] if user.full_name else [])
    injection = check_injection(question)
    user_key = hash_user(user.id)

    conv_id_int = int(body.conversationId) if body.conversationId else None
    if conv_id_int is not None and conv_id_int > 2_147_483_647:
        raise NotFoundError("Conversation not found.", details=None)
    conv = await ConversationDAO.get_by_id(db, conv_id_int, user.id) if conv_id_int else None
    if conv_id_int and not conv:
        raise NotFoundError("Conversation not found.", details=None)

    if conv and conv.course_id is not None:
        if body.courseId is not None and body.courseId != conv.course_id:
            raise ValidationError("Start a new chat to ask about another course.", details=None)
        body.courseId = conv.course_id

    if body.contextVideoId is not None:
        video = await db.get(Video, body.contextVideoId)
        if video is None or (body.courseId is not None and video.course_id != body.courseId):
            raise NotFoundError("Video not found.", details=None)
        body.courseId = video.course_id or body.courseId

    material = None
    if body.contextMaterialId is not None:
        material = await db.get(Material, body.contextMaterialId)
        if material is None or (body.courseId is not None and material.course_id != body.courseId):
            raise NotFoundError("Material not found.", details=None)
        body.courseId, body.contextMaterialTitle = material.course_id, material.title

    course = None
    if body.courseId is not None:
        if scope is not None and body.courseId not in scope:
            raise NotFoundError("Course not found.", details=None)
        course = await db.get(Course, body.courseId)
        if course is None:
            raise NotFoundError("Course not found.", details=None)
    topic = topic_of(question)[:128]
    if body.moduleId is not None:
        module = await db.get(CourseModule, body.moduleId)
        if module is None or module.course_id != body.courseId:
            raise NotFoundError("Module not found.", details=None)
        topic = module.title
    elif body.courseId is not None:
        topic = (material.topic if material else None) or "General"

    if conv:
        saved_messages = await ChatMessageDAO.list_by_conversation(db, conv.id)
        raw_history = [{"role": m.role, "content": m.content} for m in saved_messages if m.course_id == body.courseId]
    else:
        saved_messages = []
        raw_history = [{"role": m.role, "content": m.content} for m in body.conversationHistory]
    names = _names(user)
    raw_history = [dict(m, content=scrub_pii(m["content"], names)[0]) for m in raw_history]
    history = (await build_context(db, conv.id, raw_history)
               if conv and conv.course_id == body.courseId else raw_history[-10:])
    history = [dict(m, content=scrub_pii(m["content"], names)[0]) for m in history]

    course_name = course.title if course else None
    scoped = bool(body.courseId or body.contextMaterialTitle or body.contextVideoId)
    clarification = chat_service.portion_clarification(question) if not scoped else None

    policy = await run_in_threadpool(
        lambda: plan_turn(
            question,
            raw_history,
            course_id=body.courseId,
            material_title=body.contextMaterialTitle,
            video_id=body.contextVideoId,
            mode="planning" if body.planning else "chat",
            answer_mode=body.answerMode,
            user_key=user_key,
            role=user.role,
        )
    )
    weak_topics = await get_weak_topics(db, user.id, body.courseId)
    flagged_topic = None
    if policy.socratic.escalated and not clarification:
        # Spec 4.4: after 3 unsuccessful attempts the topic is flagged as a weakness.
        flagged_topic = topic_of(policy.socratic.thread_query or question)
        if await flag_weak_topic(db, user.id, flagged_topic):
            await db.commit()
    system_extra = policy.socratic.system_instructions()
    meta_extra = {
        "tier": policy.tier,
        "requested_tier": policy.decision.tier,  # 1 simple / 2 complex, before budget or a missing Tier 2
        "socratic": policy.socratic.stage,
        "cached": False,
        "classifier": policy.decision.source,
    }
    if policy.cap.get("over") or policy.cap.get("near"):
        meta_extra["cap"] = policy.cap

    async def event_generator():
        nonlocal saved_turn
        import asyncio
        import json as _json
        full_response = []
        sink: dict = {}
        cache_hit = policy.cache_hit if not clarification else None
        stream = None
        completed = False
        failed = False
        started = _time.perf_counter()
        usage_token = budget.request_usage.set([])
        tier_token = set_tier(policy.tier or 1)
        scope_token = allowed_courses.set(scope)  # the generator runs outside the dependency's context
        wire_token = structured_sse.set(body.structured)
        from saarthi_backend.ai.adapter import _sse_status

        try:
            if await request.is_disconnected():
                return

            if clarification:
                yield _sse_data(clarification)
                full_response.append(clarification)
                completed = True
                if not body.structured:
                    yield _SSE_DONE
            elif cache_hit is not None:
                # Tier 0: the classifier matched a stored repeated query — no LLM call.
                yield _sse_status("cache", "Answered from a previous identical question")
                meta = {**meta_extra, "cached": True, "tier": 0, "citations": cache_hit.citations,
                        "cache_entry_id": cache_hit.entry_id, "confidence": "ok", "reply_mode": "grounded"}
                yield f"event: meta\ndata: {_json.dumps(meta)}\n\n"
                answer = cache_hit.answer
                for i in range(0, len(answer), 160):
                    yield _sse_data(answer[i:i + 160])
                full_response.append(answer)
                sink.update({"citations": cache_hit.citations, "chunk_ids": cache_hit.chunk_ids,
                             "reply_mode": "grounded", "confidence": "ok", "covered": True})
                completed = True
                if not body.structured:
                    yield _SSE_DONE
            elif scoped:
                yield _sse_status("retrieving", "Searching your course materials")
                # Follow-ups ("I'm stuck", "explain more") carry no course terms of their own:
                # retrieve for the problem / previous question they refer to.
                retrieval_query = _retrieval_query(question, raw_history, policy)
                ground_fn = ground_multi if body.planning else ground
                result = await run_in_threadpool(
                    lambda: ground_fn(
                        retrieval_query,
                        course_id=body.courseId,
                        material_title=body.contextMaterialTitle,
                        video_id=body.contextVideoId,
                        n=policy.top_n,
                        boost_terms=weak_topics,
                    )
                )
                grounding, plan_steps = result if body.planning else (result, None)
                stream = stream_scoped_answer(
                    question, grounding, history, kind=policy.qclass.kind, tier=policy.tier,
                    system_extra=system_extra, course_name=course_name, meta_extra=meta_extra, sink=sink,
                    steps=plan_steps, concise=policy.decision.tier == 1,
                )
            else:
                stream = run_chat_stream(
                    question, history, planning=body.planning, tier=policy.tier, system_extra=system_extra,
                    boost_terms=weak_topics, meta_extra=meta_extra, sink=sink,
                    kind=policy.qclass.kind, n=policy.top_n, course_name=course_name,
                )
            if stream is not None:
                async for chunk in stream:
                    if await request.is_disconnected():
                        return
                    if chunk == _SSE_DONE:
                        completed = True
                        if body.structured:
                            continue
                    if chunk.startswith(_SSE_ERROR_PREFIX):
                        failed = True
                        if body.structured:
                            continue
                    yield chunk
                    if _is_content_frame(chunk):
                        full_response.append(sse_delta_text(chunk))
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Chat answer stream failed")
            failed = True
            if not body.structured:
                yield "data: [ERROR] Could not complete the answer.\n\n"
        finally:
            usage = budget.request_usage.get() or []
            for var_reset in (lambda: reset_tier(tier_token), lambda: budget.request_usage.reset(usage_token),
                              lambda: allowed_courses.reset(scope_token), lambda: structured_sse.reset(wire_token)):
                try:
                    var_reset()
                except ValueError:  # generator closed from another context (client disconnect)
                    pass

            assistant_content = "".join(full_response)
            tokens = summarize_usage(usage)
            budget.record_user(user_key, tokens["prompt"] + tokens["completion"])
            try:
                # Tier 0 write-back: only verified, cited, grounded answers are reused.
                if (completed and not failed and cache_hit is None and policy.cache_scope and assistant_content
                        and sink.get("reply_mode") == "grounded" and sink.get("confidence") == "ok"
                        and not injection):
                    await run_in_threadpool(
                        lambda: get_cache().store(
                            question,
                            policy.cache_scope,
                            assistant_content,
                            citations=sink.get("citations") or [],
                            chunk_ids=[c for c in (sink.get("chunk_ids") or []) if c],
                            answer_model=chat_model_name(tier=policy.tier),
                            tier=policy.tier,
                            embedding=policy.embedding,
                        )
                    )
                write_trace({
                    "path": "stream",
                    "user": user_key,
                    "role": user.role,
                    "conversation": conv_id_int,
                    "scope": {"course_id": body.courseId, "material": body.contextMaterialTitle, "video_id": body.contextVideoId},
                    "query": question,
                    **policy.trace(),
                    "tier": 0 if cache_hit else policy.tier,
                    "model": None if cache_hit else chat_model_name(tier=policy.tier),
                    "cache": {"hit": cache_hit is not None, "similarity": getattr(cache_hit, "similarity", None),
                              "entry_id": getattr(cache_hit, "entry_id", None)},
                    "reply_mode": sink.get("reply_mode"),
                    "covered": sink.get("covered"),
                    "cited": sink.get("cited"),
                    "confidence": sink.get("confidence"),
                    "confidence_reasons": sink.get("confidence_reasons"),
                    "expert": sink.get("expert"),
                    "plan": sink.get("plan"),
                    "critic_ok": sink.get("critic_ok"),
                    "weak_topics": weak_topics,
                    "flagged_weak_topic": flagged_topic,
                    "safety": {"pii_scrubbed": pii_scrubbed, "injection_suspected": injection, "leak": sink.get("leaked")},
                    "retrieval": sink.get("retrieval"),
                    "cite_ids": [c for c in (sink.get("chunk_ids") or []) if c],
                    "n_citations": len(sink.get("citations") or []),
                    "tokens": tokens,
                    "latency_ms": round((_time.perf_counter() - started) * 1000),
                    "answer_chars": len(assistant_content),
                })
            except Exception:
                logger.debug("Cache or trace write skipped", exc_info=True)

        if failed or not completed or not assistant_content.strip():
            if not body.structured:
                return
            assistant_content = "Saarthi could not complete this answer. Please try again."
            sink.update({"reply_mode": "error", "confidence": "low", "citations": [],
                         "confidence_reasons": ["The answer did not finish"]})

        if sink.get("leaked"):
            from src.safety.guard import SAFE_REFUSAL
            assistant_content = SAFE_REFUSAL
            sink["citations"] = []

        turn_topic = "General" if clarification else topic
        citations = sink.get("citations") or []
        metadata = _json.loads(_json.dumps({
            "socratic": None if clarification else policy.socratic.stage,
            "confidence": sink.get("confidence"),
            "confidence_reasons": sink.get("confidence_reasons") or [],
            "reply_mode": sink.get("reply_mode"),
            "citations": citations,
            "cache_entry_id": getattr(cache_hit, "entry_id", None),
            "material_title": body.contextMaterialTitle,
            "video_id": body.contextVideoId,
            "cap": meta_extra.get("cap"),
        }, default=str))

        if conv_id_int:
            try:
                if body.courseId is not None and body.moduleId is None:
                    for citation in citations:
                        material_id, video_id = citation.get("material_id"), citation.get("video_id")
                        if not (material_id or video_id):
                            continue
                        resource = await db.get(Material if material_id else Video, material_id or video_id)
                        if resource and resource.course_id == body.courseId and resource.topic:
                            turn_topic = resource.topic
                            break
                student_message = await ChatMessageDAO.create(
                    db, conv_id_int, "user", body.message, course_id=body.courseId, topic=turn_topic,
                    shared_query=" ".join(question.split())[:2000] if body.courseId and not injection else None,
                )
                answer = await ChatMessageDAO.create(
                    db, conv_id_int, "assistant", assistant_content, course_id=body.courseId, topic=turn_topic,
                    reply_to_id=student_message.id, answer_metadata=metadata,
                )
                if not saved_messages and conv.title == "New Chat":
                    await ConversationDAO.update_title(db, conv_id_int, user.id, body.message.strip()[:200] or "New Chat")
                await ConversationDAO.touch(db, conv_id_int)
                await db.commit()
                if body.structured:
                    saved_turn = SendMessageResponse(userMessage=_message_to_item(student_message), assistantMessage=_message_to_item(answer))
                    if not json_delivery:
                        yield f"event: saved\ndata: {saved_turn.model_dump_json()}\n\n"
            except Exception:
                logger.exception("Could not persist streamed answer")
                await db.rollback()
                if body.structured:
                    yield "event: error\ndata: Could not save the chat. Please try again.\n\n"
                return

        # Instructor review queue: a course answer that ended "low" confidence (not covered,
        # uncited, critic FAIL) is flagged so the instructor can write the right answer.
        try:
            if (body.courseId and sink.get("confidence") == "low" and cache_hit is None
                    and policy.qclass.kind != "smalltalk" and not injection and assistant_content
                    and await course_service.get_enrollment(db, user.id, body.courseId)):
                from saarthi_backend.service.review_service import record_review
                await record_review(
                    db, course_id=body.courseId, user_id=user.id, query=" ".join(question.split())[:2000], answer=assistant_content,
                    reply_mode=sink.get("reply_mode"), confidence="low",
                    reasons=sink.get("confidence_reasons"), source="low_confidence",
                )
                await db.commit()
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("review flag skipped: %s", e)
            try:
                await db.rollback()
            except Exception:
                pass

        if conv_id_int and conv.course_id == body.courseId:
            try:
                all_msgs = await ChatMessageDAO.list_by_conversation(db, conv_id_int)
                session_factory = request.app.state.db_session_factory
                scoped_messages = [m for m in all_msgs if m.course_id == body.courseId]
                asyncio.ensure_future(maybe_update_summary_bg(session_factory, conv_id_int, scoped_messages, student_names=names))
            except Exception:
                logger.debug("Memory update skipped", exc_info=True)

        if body.structured:
            yield _SSE_DONE

    if json_delivery:
        async for _ in event_generator():
            pass
        if saved_turn is None:
            raise HTTPException(status_code=503, detail="Could not save the chat. Please try again.")
        return saved_turn

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─── Study Plan Generator ─────────────────────────────────────────────────────
@router.post("/study-plan", response_model=StudyPlanResponse)
async def generate_study_plan(
    body: StudyPlanRequest,
    _scope: Annotated[set[int] | None, Depends(retrieval_scope)],
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Generate a personalized weekly study plan based on enrollments, deadlines, and quiz scores."""
    plan = await chat_service.generate_study_plan(db, user.id, body, student_names=_names(user))
    return plan


# ─── Conversations ────────────────────────────────────────────────────────────
@router.post("/conversations", response_model=ConversationResponse, status_code=201)
async def create_conversation(
    body: CreateConversationRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    scope: Annotated[set[int] | None, Depends(retrieval_scope)],
):
    if body.courseId is not None and ((scope is not None and body.courseId not in scope) or not await db.get(Course, body.courseId)):
        raise NotFoundError("Course not found.", details=None)
    c = await chat_service.create_conversation(db, user.id, title=body.title, course_id=body.courseId)
    await db.commit()
    return _conversation_to_response(c)


@router.get("/conversations", response_model=PaginatedResponse[ConversationResponse])
async def list_conversations(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    pagination: Annotated[PaginationParams, Depends(get_pagination)],
    topic: Annotated[str | None, Query(max_length=128)] = None,
    onDate: date | None = None,
):
    items, total = await chat_service.list_conversations(
        db, user.id, limit=pagination.limit, offset=pagination.offset, topic=(topic or "").strip(), on_date=onDate,
    )
    return PaginatedResponse(
        items=[_conversation_to_response(c) for c in items],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: int,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    c, messages = await chat_service.get_conversation(db, conversation_id, user.id)
    if not c:
        raise NotFoundError("Conversation not found.", details=None)
    return ConversationDetailResponse(
        **_conversation_to_response(c).model_dump(),
        messages=[_message_to_item(m) for m in messages],
    )


@router.patch("/conversations/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: int,
    body: UpdateConversationRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    c = await chat_service.update_conversation_title(db, conversation_id, user.id, body.title)
    if not c:
        raise NotFoundError("Conversation not found.", details=None)
    await db.commit()
    return _conversation_to_response(c)


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: int,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    deleted = await chat_service.delete_conversation(db, conversation_id, user.id)
    if not deleted:
        raise NotFoundError("Conversation not found.", details=None)
    await db.commit()


@router.post("/conversations/{conversation_id}/messages", response_model=SendMessageResponse, status_code=201)
async def send_message(
    conversation_id: int,
    body: SendMessageRequest,
    scope: Annotated[set[int] | None, Depends(retrieval_scope)],
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Non-streaming: append user message, call AI, append assistant reply."""
    conversation = await ConversationDAO.get_by_id(db, conversation_id, user.id)
    if conversation is None:
        raise NotFoundError("Conversation not found.", details=None)
    if conversation.course_id is not None and body.courseId not in (None, conversation.course_id):
        raise ValidationError("Start a new chat to ask about another course.", details=None)
    course_id = conversation.course_id or body.courseId
    if course_id is not None and ((scope is not None and course_id not in scope) or not await db.get(Course, course_id)):
        raise NotFoundError("Course not found.", details=None)
    result = await chat_service.send_message(
        db, conversation_id, user.id, body.message,
        context_material_title=body.contextMaterialTitle,
        course_id=course_id,
    )
    if not result:
        raise NotFoundError("Conversation not found.", details=None)
    user_msg, assistant_msg = result
    await db.commit()
    return SendMessageResponse(
        userMessage=_message_to_item(user_msg),
        assistantMessage=_message_to_item(assistant_msg),
    )


# ─── Socratic Challenge Mode ───────────────────────────────────────────────────
@router.post("/socratic", response_model=SocraticChallengeResponse)
async def socratic_challenge(
    body: SocraticChallengeRequest,
    _scope: Annotated[set[int] | None, Depends(retrieval_scope)],
    user: Annotated[User, Depends(get_current_user)],
):
    """AI challenges the student's claim using the Socratic method to force active recall."""
    return await chat_service.socratic_challenge(body, student_names=_names(user))


# ─── Concept Graph ─────────────────────────────────────────────────────────────
class ConceptGraphRequest(StudyPlanRequest):
    courseTitle: str = ""
    model_config = {"extra": "ignore"}


@router.post("/concept-graph", response_model=ConceptGraphResponse)
async def concept_graph(
    body: ConceptGraphRequest,
    _scope: Annotated[set[int] | None, Depends(retrieval_scope)],
    user: Annotated[User, Depends(get_current_user)],
):
    """Generate a concept dependency graph with mastery levels for a course."""
    course_title = body.courseTitle or (body.courses[0].title if body.courses else "Engineering Course")
    return await chat_service.generate_concept_graph(course_title, body.recentQuizScores)


# ─── Exam Prediction ───────────────────────────────────────────────────────────
@router.post("/exam-prediction", response_model=ExamPredictionResponse)
async def exam_prediction(
    body: ExamPredictionRequest,
    _scope: Annotated[set[int] | None, Depends(retrieval_scope)],
    user: Annotated[User, Depends(get_current_user)],
):
    """Predict likely exam topics and student readiness for each."""
    return await chat_service.predict_exam_topics(body)


# ─── Flashcards (Spaced Repetition) ───────────────────────────────────────────
@router.get("/flashcards", response_model=FlashcardListResponse)
async def list_flashcards(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    topic: str | None = None,
):
    """Saved library for this user. New generates APPEND; they never wipe older topics."""
    return await chat_service.list_flashcards(db, user.id, topic=topic)


@router.post("/flashcards/generate", response_model=FlashcardDeckResponse)
async def generate_flashcards(
    body: FlashcardDeckRequest,
    _scope: Annotated[set[int] | None, Depends(retrieval_scope)],
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Extract Anki-style flashcards and append them to the user's library."""
    return await chat_service.generate_flashcards(body, db=db, user_id=user.id, student_names=_names(user))


@router.post("/flashcards/review", response_model=FlashcardReviewResponse)
async def review_flashcard(
    body: FlashcardReviewRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """SM-2 next interval, written back onto the saved card."""
    return await chat_service.review_flashcard(body, db=db, user_id=user.id)


# ─── Answer feedback (thumbs up / down) ──────────────────────────────────────
@router.post("/feedback", response_model=AnswerFeedbackResponse)
async def answer_feedback(
    body: AnswerFeedbackRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Record a rating for the researcher dashboard (traces) and, on thumbs-down,
    evict the answer from the semantic cache so it is never served again (WBS 3.4.4).
    With messageId, the rating is also saved on that answer (only the student's own)."""
    from sqlalchemy import select

    from saarthi_backend.model.chat_model import ChatMessage, Conversation
    from src.cache.semantic_cache import build_scope, get_cache
    from src.safety.guard import scrub_pii
    from src.telemetry.traces import hash_user, write_trace

    course_id, material_title, video_id = body.courseId, body.contextMaterialTitle, body.contextVideoId
    conversation_id, cache_id = body.conversationId, body.cacheEntryId
    query_text, answer_text = body.query, body.answer or ""
    if body.messageId is not None:
        msg = (await db.execute(
            select(ChatMessage).join(Conversation, Conversation.id == ChatMessage.conversation_id)
            .where(ChatMessage.id == body.messageId, ChatMessage.role == "assistant", Conversation.user_id == user.id)
        )).scalar_one_or_none()
        if msg is None:
            raise NotFoundError("Answer not found.", details=None)
        metadata = msg.answer_metadata or {}
        course_id, conversation_id = msg.course_id, str(msg.conversation_id)
        material_title, video_id = metadata.get("material_title"), metadata.get("video_id")
        cache_id = metadata.get("cache_entry_id")
        answer_text = msg.content
        if msg.reply_to_id is not None:
            question = await db.get(ChatMessage, msg.reply_to_id)
            if question and question.conversation_id == msg.conversation_id:
                query_text = question.content
        if msg.rating == body.rating:
            return AnswerFeedbackResponse(recorded=True, cacheEvicted=0)
        msg.rating = body.rating

    names = _names(user)
    query, _ = scrub_pii(query_text, names)
    comment = scrub_pii(body.comment, names)[0] if body.comment else None
    evicted = 0
    if body.rating == "down":
        scope = build_scope(course_id=course_id, material_title=material_title, video_id=video_id, mode="chat")
        evicted = await run_in_threadpool(lambda: get_cache().forget(scope, query, cache_id))
        from saarthi_backend.service import course_service
        if course_id and await course_service.get_enrollment(db, user.id, course_id):
            from saarthi_backend.service.review_service import record_review
            await record_review(
                db, course_id=course_id, user_id=user.id, query=query, answer=scrub_pii(answer_text, names)[0],
                reply_mode=None, confidence=None, reasons=[comment] if comment else ["student rated this answer down"],
                source="thumbs_down",
            )
    await db.commit()
    write_trace({
        "type": "feedback",
        "user": hash_user(user.id),
        "role": user.role,
        "rating": body.rating,
        "comment": comment,
        "query": query,
        "scope": {"course_id": course_id, "material": material_title, "video_id": video_id},
        "conversation": conversation_id,
        "cache_evicted": evicted,
    })
    return AnswerFeedbackResponse(recorded=True, cacheEvicted=evicted)
