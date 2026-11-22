"""Grounded MCQ generation (WBS 3.5.1, spec §7.3, decision D4 = LLM-generated + instructor review).

Questions are written ONLY from retrieved course excerpts; each question carries the
citation label of the excerpt it comes from (spec acceptance: "All MCQs linked to at
least one source chunk"). Invalid items (not 4 distinct options, bad answer index,
no explanation, source not among the excerpts) are dropped, never repaired by guessing.
"""
from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

LEVELS = {
    "L1": "recall — check that the student remembers a definition, formula or fact stated in the excerpts",
    "L2": "application — the student must apply a formula or property from the excerpts to a small new case",
    "L3": "analysis — the student must reason about why/when something in the excerpts holds or compare two ideas",
}

_SYSTEM = (
    "You write multiple-choice questions for engineering students. Use ONLY the course excerpts provided. "
    "Return JSON only: {\"questions\": [{\"question\": str, \"options\": [4 strings], \"answer_index\": 0-3, "
    "\"explanation\": str, \"source\": the bracketed label of the excerpt used}]}. "
    "Distractors must be plausible but clearly wrong according to the excerpts. "
    "Math in LaTeX inside $...$."
)


_JSON_BACKSLASH = re.compile(r"\\\\|\\(?=[A-Za-z]{2,})|\\(?![\\\"/bfnrtu])")


def _latex_safe(text: str) -> str:
    """Models write LaTeX inside JSON strings (\\int, \\frac, \\( ... \\)), which is invalid JSON.
    Double a backslash that starts a LaTeX command or an escape JSON does not know; keep
    existing double backslashes, \\" and \\uXXXX. qwen2.5:7b lost 18 of 30 MCQs to this."""
    return _JSON_BACKSLASH.sub(lambda m: m.group(0) if m.group(0) == "\\\\" else "\\\\", text)


def _loads(text: str):
    for candidate in (text, _latex_safe(text)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _parse(raw: str) -> list[dict]:
    raw = raw or ""
    if "```" in raw:
        raw = raw.split("```", 2)[1]
        raw = raw.split("\n", 1)[1] if raw.lstrip().lower().startswith("json") else raw
    data = _loads(raw)
    if data is None:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return []
        data = _loads(m.group(0))
        if data is None:
            return []
    items = data.get("questions") if isinstance(data, dict) else data
    return items if isinstance(items, list) else []


def _valid(q: dict, labels: list[str]) -> dict | None:
    if not isinstance(q, dict):
        return None
    opts = [str(o).strip() for o in (q.get("options") or [])]
    try:
        ans = int(q.get("answer_index"))
    except (TypeError, ValueError):
        return None
    question = str(q.get("question") or "").strip()
    explanation = str(q.get("explanation") or "").strip()
    if len(opts) != 4 or len({o.lower() for o in opts}) != 4 or not 0 <= ans <= 3 or len(question) < 10 or not explanation:
        return None
    src = str(q.get("source") or "").strip()
    match = next((lab for lab in labels if src and (src.lower() in lab.lower() or lab.lower().strip("[]") in src.lower())), None)
    if match is None:
        return None
    return {"question": question, "options": opts, "answer_index": ans, "explanation": explanation, "source": match}


def _shuffle_options(q: dict, rng) -> dict:
    """Shuffle the four options and move answer_index with the correct one. Measured 2026-11-19: every
    model put the correct answer on option A far too often (qwen3:14b 57 %, the fine-tune 62 %), which
    lets students guess. Only the order changes; question, options, explanation and source stay the same."""
    order = list(range(len(q["options"])))
    rng.shuffle(order)
    return {**q, "options": [q["options"][i] for i in order], "answer_index": order.index(q["answer_index"])}


def generate_mcqs(
    topic: str,
    *,
    course_id: int,
    n: int = 5,
    level: str = "L1",
    material_title: str | None = None,
) -> dict:
    """Blocking (retrieval + one LLM call)."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from src.tools.citations import excerpts_block, labels
    from src.tools.grounding import coverage
    from src.tools.retrieve import retrieve
    from src.utils.llm import chat_llm

    level = level if level in LEVELS else "L1"
    n = max(1, min(int(n), 10))
    chunks = retrieve(topic, course_id=course_id, material_title=material_title, k=max(3, min(n, 8)))[: max(3, min(n, 8))]
    covered, _ = coverage(topic, chunks)
    if not chunks or not covered:
        return {"topic": topic, "level": level, "questions": [], "not_covered": True, "sources": []}

    labs = labels(chunks)
    # MCQ writing is an instructor-side, non-interactive task: it may use a stronger model than
    # the chat's Tier 1 (MCQ_MODEL, served by the Tier 1 endpoint), e.g. qwen2.5:3b on Ollama.
    # No provider JSON mode: small local models loop on it; the parser below is tolerant instead.
    extra = {"model": os.environ["MCQ_MODEL"]} if os.environ.get("MCQ_MODEL") else {}
    llm = chat_llm(temperature=0.3, max_tokens=1600, tier=2 if level == "L3" else 1, **extra)
    prompt = (
        f"Course excerpts:\n{excerpts_block(chunks)}\n\n"
        f"Write {n} questions about \"{topic}\" at level {level}: {LEVELS[level]}.\n"
        f"Valid source labels: {', '.join(labs)}"
    )
    try:
        raw = llm.invoke([SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)]).content
    except Exception as e:
        logger.warning("MCQ generation failed: %s", e)
        return {"topic": topic, "level": level, "questions": [], "error": str(e), "sources": labs}
    items = _parse(raw)
    valid = [v for v in (_valid(q, labs) for q in items) if v]
    if os.environ.get("MCQ_SHUFFLE", "1").strip().lower() not in ("0", "false", "no", "off"):
        import random

        rng = random.Random()
        valid = [_shuffle_options(q, rng) for q in valid]
    return {
        "topic": topic,
        "level": level,
        "questions": valid[:n],
        "dropped": len(items) - len(valid),
        "sources": labs,
        "reviewed": False,  # spec 7.3: human-reviewed before deployment
    }
