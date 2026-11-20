"""Grounding rule (DEC-1): retrieve the course excerpts for a turn and decide whether the
course materials cover the question at all; confidence signal for the UI (spec §7.1).

Coverage gate: at least one retrieved chunk must share a content term with the question
(BM25 tokens, framing words ignored). Dense search always returns *something*; this
stops "who won the world cup" from being answered out of a Fourier lecture.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.retrieval.bm25 import tokenize

_FRAMING = frozenset(
    """explain define definition meaning mean means describe tell give example examples simple simply words
    detail details please help understand know about question answer lecture course notes
    cause causes caused causing prevent prevents preventing avoid avoids avoiding work works working use used
    uses using happen happens happening differ differs difference differences important importance
    reason reasons way ways thing things kind type types find show shows see look""".split()
)


def _stem(t: str) -> str:
    """Light suffix stripping so aliasing/aliases/alias and causes/cause compare equal."""
    for suf in ("ings", "ing", "ed", "es", "s"):
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            t = t[: -len(suf)]
            break
    # convolve/convolved/convolution, solve/solution, resolve/resolution share one stem. Before this,
    # "Convolve x[n] = {1, 2, 3} with ..." was refused when the excerpts only said "convolution".
    if t.endswith("olution") and len(t) > 7:
        return t[:-7] + "olv"
    if t.endswith("olve"):
        return t[:-1]
    return t


_FRAMING_STEMS = frozenset(_stem(w) for w in _FRAMING) | _FRAMING


@dataclass
class Grounding:
    chunks: list = field(default_factory=list)
    covered: bool = False
    overlap_terms: list = field(default_factory=list)


def content_terms(text: str) -> set[str]:
    return {t for t in tokenize(text) if len(t) > 2 and t not in _FRAMING_STEMS and _stem(t) not in _FRAMING_STEMS}


def coverage(query: str, chunks, alternatives: dict[str, str] | None = None) -> tuple[bool, list[str]]:
    """Covered when the excerpts contain at least half of the question's content terms,
    or at least three of them. One shared generic word is not enough: "orange zebra
    theorem" must not count as covered because a lecture mentions "theorem".
    `alternatives` maps an abbreviation to the full form that replaced it in `query`
    ("DFT" -> "discrete fourier transform"). A word of the full form also counts as found
    when the excerpts only use the abbreviation: "What is the DFT frequency resolution ..."
    was refused on a lecture that never spells out "discrete Fourier transform"."""
    q = content_terms(query)
    if not q:
        return bool(chunks), []
    chunk_stems: set[str] = set()
    for c in chunks:
        text = c.get("text") if isinstance(c, dict) else getattr(c, "text", "")
        chunk_stems |= {_stem(t) for t in tokenize(text or "")}
    via_abbrev: dict[str, set[str]] = {}
    for abbrev, full in (alternatives or {}).items():
        for w in content_terms(full):
            via_abbrev.setdefault(w, set()).add(abbrev.lower())

    def found(t: str) -> bool:
        return _stem(t) in chunk_stems or any(a in chunk_stems for a in via_abbrev.get(t, ()))

    hits = {t for t in q if found(t)}
    return (len(hits) >= 3 or len(hits) / len(q) >= 0.5), sorted(hits)


def ground(
    query: str,
    *,
    course_id: int | None = None,
    material_title: str | None = None,
    video_id: int | None = None,
    n: int = 3,
    boost_terms: list[str] | None = None,
) -> Grounding:
    from src.tools.retrieve import retrieve

    from src.tools.retrieve import prepare_query

    chunks = retrieve(
        query, course_id=course_id, material_title=material_title, video_id=video_id,
        k=n, boost_terms=boost_terms,
    )[:n]
    prepared = prepare_query(query, None, course_id, video_id)
    covered, terms = coverage(prepared.coverage_query, chunks, prepared.expanded)
    return Grounding(chunks=chunks, covered=covered, overlap_terms=terms)


def _snippet(c, n: int = 140) -> str:
    from src.tools.citations import source_label

    text = c.get("text") if isinstance(c, dict) else getattr(c, "text", "")
    return f"{source_label(c)} {' '.join((text or '').split())[:n]}"


def ground_multi(
    query: str,
    *,
    course_id: int | None = None,
    material_title: str | None = None,
    video_id: int | None = None,
    n: int = 3,
    boost_terms: list[str] | None = None,
) -> tuple[Grounding, list[dict]]:
    """Planning mode for course chat: several visible retrieval steps, still grounded.

    Step 1 searches with the question, step 2 searches again with only its key terms
    (catches chunks that the full sentence ranks low), step 3 checks whether the
    passages cover the question. The returned steps use the same shape the chat UI
    already renders for agent reasoning. This is deterministic multi-query retrieval,
    not an LLM-driven ReAct loop: small local models are unreliable at tool use.
    """
    from src.tools.retrieve import retrieve

    kw = dict(course_id=course_id, material_title=material_title, video_id=video_id, k=n, boost_terms=boost_terms)
    steps: list[dict] = []

    first = retrieve(query, **kw)[:n]
    steps.append({
        "agent": "course_materials", "step": 1,
        "thought": "Search the course materials for the question as asked.",
        "tools_called": [{"name": "search_course_materials", "args": {"query": query}}],
        "observations": [_snippet(c) for c in first] or ["No passages found."],
    })

    terms = [t for t in tokenize(query) if len(t) > 2 and t not in _FRAMING]
    key_query = " ".join(dict.fromkeys(terms))
    merged = list(first)
    if key_query and key_query.lower() != query.lower().strip():
        seen = {c.id for c in merged}
        second = retrieve(key_query, **kw)[:n]
        fresh = [c for c in second if c.id not in seen]
        merged.extend(fresh)
        steps.append({
            "agent": "course_materials", "step": 2,
            "thought": f"Search again with only the key terms: {key_query}.",
            "tools_called": [{"name": "search_course_materials", "args": {"query": key_query}}],
            "observations": [_snippet(c) for c in fresh] or ["No new passages."],
        })

    merged = merged[:n]
    from src.tools.retrieve import prepare_query

    prepared = prepare_query(query, None, course_id, video_id)
    covered, matched = coverage(prepared.coverage_query, merged, prepared.expanded)
    steps.append({
        "agent": "course_materials", "step": len(steps) + 1,
        "thought": "Check whether these passages cover the question." if covered
        else "These passages do not cover the question, so I will not answer from outside the course.",
        "tools_called": [],
        "observations": [f"Matched terms: {', '.join(matched) or 'none'}", f"Covered: {'yes' if covered else 'no'}"],
    })
    return Grounding(chunks=merged, covered=covered, overlap_terms=matched), steps


def confidence(*, covered: bool, cited: bool, critic_ok: bool | None, n_chunks: int) -> tuple[str, list[str]]:
    reasons = []
    if not covered or n_chunks == 0:
        reasons.append("not covered by course materials")
    if critic_ok is False:
        reasons.append("critic could not verify the answer against the excerpts")
    if covered and not cited:
        reasons.append("answer did not cite a source")
    return ("low" if reasons else "ok"), reasons
