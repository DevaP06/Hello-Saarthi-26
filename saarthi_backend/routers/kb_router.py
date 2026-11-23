"""Ask the course notes (under /api/kb): the "Ask the Notes" page.

Default for now: the Gemini knowledge-base index (src/knowledge_base/gemini_rag.py), with the
question PII-scrubbed before it leaves our server. Once our own model is deployed, CHAT_SOURCE=core
switches this page (and chat) to the Saarthi AI core: retrieval limited to the caller's courses,
the coverage gate refuses questions the notes don't cover (no model call), and the configured model
(e.g. the fine-tuned qwen3-saarthi-14b) answers only from the excerpts, with citations. If the
Gemini index is not built, the AI core is used. Reasons: saarthi_backend/service/chat_service.py
(_kb_chat_reply).
"""

import os
import re
from typing import Annotated, Optional

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from saarthi_backend.deps import get_current_user
from saarthi_backend.model import User
from saarthi_backend.service.access_scope import retrieval_scope
from saarthi_backend.utils.exceptions import AIServiceError, RetrievalError
from saarthi_backend.utils.logging import get_logger

router = APIRouter(prefix="/kb", tags=["knowledge-base"])
logger = get_logger(__name__)


class KBAskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)
    topK: int = Field(5, ge=1, le=10)


class KBSource(BaseModel):
    n: int
    sourceFile: Optional[str] = None
    page: Optional[int | str] = None
    topic: Optional[str] = None
    sourceType: Optional[str] = None
    score: Optional[float] = None  # similarity (Gemini path); None on the AI core path (rank order instead)
    courseId: Optional[int] = None
    materialId: Optional[int] = None
    videoId: Optional[int] = None
    startSeconds: Optional[int] = None
    text: str


class KBAskResponse(BaseModel):
    answer: str
    sources: list[KBSource]


def _names(user: User) -> list[str]:
    return [n for n in (getattr(user, "full_name", None),) if n]


def _use_gemini() -> bool:
    if os.environ.get("CHAT_SOURCE", "gemini").strip().lower() in ("core", "legacy"):
        return False
    from src.knowledge_base import gemini_rag

    return gemini_rag.index_ready()


# ── AI core path (default) ──────────────────────────────────────────────────

def _llm_answer(system: str, user: str) -> str:
    """One grounded model call (separate so tests can replace it)."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from src.utils.llm import chat_llm

    return chat_llm(temperature=0.2, max_tokens=900).invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    ).content or ""


def _core_answer(question: str, k: int, names: list[str] | None = None) -> dict:
    from src.prompts.tutor import grounded_system, grounded_user, not_covered
    from src.safety.guard import leaks_prompt, scrub_pii
    from src.tools.citations import _title, _ts, excerpts_block, is_cited, sources_suffix
    from src.tools.grounding import ground

    q, _ = scrub_pii(question, names)  # also the signed-in student's own name
    # The scrubber leaves placeholders ([email], [roll-no], ...); they are not course words, so drop them
    # before search and the coverage gate, or a question with an email in it would be refused.
    q = re.sub(r"\[[a-z-]+\]", " ", q)
    q = re.sub(r"\(\s*[,\s]*\)|\s{2,}", " ", q).strip()
    g = ground(q, n=k)  # retrieval limited to the caller's courses (retrieval_scope dependency)
    if not g.chunks or not g.covered:
        return {"answer": not_covered(None), "sources": []}
    answer = _llm_answer(grounded_system(None), grounded_user(q, excerpts_block(g.chunks))).strip()
    if leaks_prompt(answer):
        return {"answer": not_covered(None), "sources": []}
    if not is_cited(answer, g.chunks):
        answer += sources_suffix(answer, g.chunks)

    def field(c, key):
        return c.get(key) if isinstance(c, dict) else getattr(c, key, None)

    sources = []
    for n, c in enumerate(g.chunks, start=1):
        start = field(c, "start_sec")
        page = field(c, "page") if field(c, "page") is not None else field(c, "page_number")
        sources.append({
            "n": n,
            "sourceFile": _title(c),
            "page": f"{_ts(start)} (video)" if start is not None else page,
            "sourceType": "video" if start is not None else "notes",
            "score": None,
            "courseId": field(c, "course_id"),
            "materialId": field(c, "material_id"),
            "videoId": field(c, "video_id"),
            "startSeconds": start,
            "text": field(c, "text") or "",
        })
    return {"answer": answer, "sources": sources}


def _core_status() -> dict:
    return {"ready": True, "chunks": 0, "source": "core"}


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/status")
async def kb_status(user: Annotated[User, Depends(get_current_user)]):
    """Whether the page can answer: Gemini (default) when its index is built, else the AI core,
    which searches the caller's indexed courses."""
    if not _use_gemini():
        return _core_status()
    from src.knowledge_base import gemini_rag

    return {**(await run_in_threadpool(gemini_rag.status)), "source": "gemini"}


@router.post("/ask", response_model=KBAskResponse)
async def kb_ask(
    body: KBAskRequest,
    user: Annotated[User, Depends(get_current_user)],
    _scope: Annotated[set[int] | None, Depends(retrieval_scope)],
):
    """Answer from the course notes with citations (Gemini by default, AI core with CHAT_SOURCE=core)."""
    if not _use_gemini():
        try:
            return await run_in_threadpool(lambda: _core_answer(body.question.strip(), body.topK, _names(user)))
        except Exception as e:  # noqa: BLE001 - model server down, bad model name
            logger.warning("KB ask (AI core) failed: %s", e)
            raise AIServiceError("KB_ASK_FAILED", "Could not get an answer from the course model. Try again shortly.",
                                 status_code=502) from e

    from src.knowledge_base import gemini_rag
    from src.safety.guard import scrub_pii

    question, _ = scrub_pii(body.question.strip(), _names(user))  # never send names, emails or roll numbers to an outside API
    try:
        return await run_in_threadpool(lambda: gemini_rag.answer(question, body.topK))
    except FileNotFoundError as e:
        raise RetrievalError(str(e)) from e
    except Exception as e:  # noqa: BLE001 - Gemini errors (quota, overload, bad key)
        logger.warning("KB ask failed: %s", e)
        msg = str(e)
        if "429" in msg or "quota" in msg.lower():
            text = "The Gemini free-tier limit was reached. Wait a minute and try again."
        elif "503" in msg or "overloaded" in msg.lower() or "high demand" in msg.lower():
            text = "Gemini is busy right now. Try again in a few seconds."
        else:
            text = "Could not get an answer from Gemini. Check the API key and model settings."
        raise AIServiceError("KB_ASK_FAILED", text, status_code=502) from e
