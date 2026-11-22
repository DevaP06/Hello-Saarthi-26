"""Sprint 3: per-dimension judges for the deterministic answer evaluation engine.

Every judge is a pure function of the dataset item and signals already extracted from the
normalised response. Judges never call each other, a model, the network or application code.

What is (and is not) decided here, per the approved Sprint 3 design:
  behaviour     evaluated for answer and abstain items; for clarify items via the clarification
                heuristic (decision D2), reported as a heuristic.
  completeness  regex-checkable required key points only; reports lower and upper bounds (D6).
  correctness   always not_evaluated for answer items (D1: forbidden claims are free text);
                not_applicable for abstain and clarify items.
  groundedness  not_evaluated (no retrieved chunks / no grounding method in Sprint 3); never
                inferred from citations. not_applicable for a clean abstention.
  language      not_evaluated for every item (no validated detector); never script detection.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

from eval.answer_eval.abstention_detector import AbstentionResult
from eval.answer_eval.citation_parser import CitationParseResult
from eval.answer_eval.evaluation_result import (
    FAIL, NOT_APPLICABLE, NOT_EVALUATED, PARTIAL, PASS, DimensionResult,
)
from eval.answer_eval.schema import MATERIALS, Item

__all__ = [
    "DETECTED_BEHAVIOURS", "match_key_points", "detect_clarification", "classify_behaviour",
    "judge_behaviour", "judge_completeness", "judge_correctness", "judge_groundedness",
    "judge_language", "citation_info", "resolve_material",
]

# Detected behaviours (what the response does, independent of what was expected).
ANSWER = "answer"
ABSTENTION = "abstention"
REFUSAL = "refusal"
MIXED = "mixed"
EMPTY = "empty"
CLARIFICATION = "clarification"
ASSUMPTION = "assumption"
DETECTED_BEHAVIOURS = (ANSWER, ABSTENTION, REFUSAL, MIXED, EMPTY, CLARIFICATION, ASSUMPTION)

# Abstention categories that a clarifying question can explain ("I need more detail: which system?").
# not_in_materials and scope_refusal are real abstentions even when followed by a question.
_CLARIFIABLE_ABSTENTIONS = ("insufficient_information", "cannot_answer", "does_not_know")


# ── Key points (completeness signals) ─────────────────────────────────────────────────────────

def match_key_points(item: Item, text: str) -> dict:
    """Match each key point's `match` regex against the normalised response.

    Uses re.search with re.IGNORECASE, exactly as the Sprint 1 validator checks a pattern against
    the item's own reference answer. A hit is recorded as evidence "pattern": it shows the pattern's
    number or formula appears, not that the point is stated correctly.
    """
    out = {"matched": [], "missing": [], "unverifiable": [], "optional": []}
    for kp in item.key_points:
        hit = re.search(kp.match, text, re.IGNORECASE) if kp.match else None
        if not kp.required:
            out["optional"].append({
                "id": kp.id, "text": kp.text,
                "matched": None if not kp.match else bool(hit),
                "span": hit.group(0) if hit else None,
            })
        elif not kp.match:
            out["unverifiable"].append({"id": kp.id, "text": kp.text})
        elif hit:
            out["matched"].append({"id": kp.id, "text": kp.text, "span": hit.group(0),
                                   "start": hit.start(), "end": hit.end(), "evidence": "pattern"})
        else:
            out["missing"].append({"id": kp.id, "text": kp.text, "pattern": kp.match})
    return out


# ── Clarification heuristic (decision D2) ────────────────────────────────────────────────────

_YOU = r"(?:you|u)"
# Each pattern describes a question or request addressed to the student about what they mean.
_ASK_PATTERNS = [
    # "Which signal do you mean?", "which system are you referring to?"
    re.compile(r"\bwhich\b[^?.!\n]{0,60}?\b(?:do|did|are|were|would)\s+" + _YOU +
               r"\s+(?:mean|have\s+in\s+mind|refer(?:ring)?\s+to|talking\s+about|asking\s+about)\b", re.I),
    # "Do you mean ...?", "What do you mean by ...?"
    re.compile(r"\b(?:do|did)\s+" + _YOU + r"\s+mean\b", re.I),
    # "Are you asking about ...?", "Are you referring to ...?"
    re.compile(r"\bare\s+" + _YOU + r"\s+(?:asking|referring|talking)\s+(?:about|to)\b", re.I),
    # "Could you clarify which ...?", "Can you specify what ...?"
    re.compile(r"\b(?:could|can|would)\s+" + _YOU + r"\s+(?:please\s+)?(?:clarify|specify|tell\s+me|confirm|"
               r"let\s+me\s+know)\b[^?.!\n]{0,80}?\b(?:which|what|whether)\b", re.I),
    # "Please clarify which system you mean." / "Let me know which filter you have in mind."
    re.compile(r"\b(?:please\s+)?(?:clarify|specify|let\s+me\s+know|tell\s+me)\b[^?.!\n]{0,40}?\b(?:which|what)\b"
               r"[^?.!\n]{0,60}?\b" + _YOU + r"\s+(?:mean|have\s+in\s+mind|are\s+(?:asking|referring|talking))\b", re.I),
    # Hinglish: "Aap kaunsa signal ke baare mein pooch rahe ho?", "kis system ki baat kar rahe hain?"
    re.compile(r"\b(?:kaun\s*s[aeio]|kis)\b[^?.!\n]{0,40}?\b(?:ke\s+baare\s+mein\s+(?:pooch|puch)|ki\s+baat\s+kar|"
               r"(?:se\s+)?matlab\s+hai)", re.I),
]
# A request for missing information ("Could you share h[n]?") counts only when it names something
# the item's acceptable interpretations ask about (the "target", e.g. signal / system / filter).
_INFO_REQUEST = re.compile(r"\b(?:could|can|would)\s+" + _YOU + r"\s+(?:please\s+)?(?:share|give|provide|"
                           r"tell\s+me|specify|send)\b", re.I)
_ASSUMPTION = re.compile(
    r"\b(?:assuming|i\s*['’]?ll\s+assume|i\s+will\s+assume|i\s*['’]?m\s+assuming|i\s+am\s+assuming|"
    r"let\s*['’]?s\s+assume|i\s+assume|if\s+" + _YOU + r"\s+(?:mean|are\s+(?:asking|referring)|['’]re\s+"
    r"(?:asking|referring))|maan\s+(?:ke|lete|lijiye|lo))\b", re.I)
_TARGET = re.compile(r"\bwhich\s+([a-z]+)", re.I)
_TARGET_STOP = {"one", "they", "of", "is", "the", "a", "an"}
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]*")
_FENCED = re.compile(r"```.*?```|~~~.*?~~~", re.S)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_QUOTED = re.compile(r"\"[^\"\n]*\"|“[^”\n]*”|‘[^’\n]*’|(?<![A-Za-z])'[^'\n]{2,}?'(?![A-Za-z])")


def _blank(text: str, patterns: Iterable[re.Pattern]) -> str:
    chars = list(text)
    for pat in patterns:
        for m in pat.finditer("".join(chars)):
            for i in range(m.start(), m.end()):
                if chars[i] != "\n":
                    chars[i] = " "
    return "".join(chars)


def clarification_targets(item: Item) -> list[str]:
    """Nouns the item's acceptable interpretations ask about ("which signal" -> "signal")."""
    found: list[str] = []
    for interp in item.acceptable_interpretations:
        for m in _TARGET.finditer(interp):
            word = m.group(1).lower()
            if word not in _TARGET_STOP and word not in found:
                found.append(word)
    return found


_ASSUMPTION_INTERPRETATION = re.compile(r"^\s*assumes\b", re.I)


def _accepts_assumption(item: Item) -> bool:
    """True only when an acceptable interpretation is itself a stated assumption ("Assumes ...").

    Matches the word "Assumes" at the start of the interpretation (case-insensitive, leading
    whitespace allowed). A mention of assumptions elsewhere, e.g. "Does not accept assumptions;
    ask the student instead", does not permit one.
    """
    return any(_ASSUMPTION_INTERPRETATION.match(i) for i in item.acceptable_interpretations)


def _accepts_general_answer(item: Item) -> bool:
    """True when an acceptable interpretation is a general answer that neither asks nor assumes."""
    for i in item.acceptable_interpretations:
        if re.match(r"^\s*(?:gives|states|provides)\b", i, re.I) and not re.search(r"\b(?:ask|assum)", i, re.I):
            return True
    return False


def detect_clarification(item: Item, text: str) -> dict:
    """Deterministic clarification heuristic (decision D2).

    asks                 a student-directed clarifying question or request was found
    question             the sentence that matched, or None
    pattern              "clarifying_question" or "information_request"
    states_assumption    the response explicitly states an assumption ("Assuming you mean ...")
    assumption           the assumption sentence, or None
    targets              nouns from the item's acceptable interpretations (e.g. ["signal"])
    target_mentioned     the matched question names one of the targets
    method               always "heuristic"

    Quoted text and code are ignored. A question is not enough on its own: rhetorical questions
    ("What is the period? It is 20 ms."), offers ("Would you like an example?") and questions to
    the reader ("Do you know why?") do not match.
    """
    targets = clarification_targets(item)
    scan = _blank(text or "", (_FENCED, _INLINE_CODE, _QUOTED))
    asks, question, pattern, target_mentioned = False, None, None, False
    assumption = None
    for m in _SENTENCE.finditer(scan):
        sentence = m.group(0)
        original = (text or "")[m.start():m.end()].strip()
        if not asks:
            if any(p.search(sentence) for p in _ASK_PATTERNS):
                asks, question, pattern = True, original, "clarifying_question"
            elif "?" in sentence and _INFO_REQUEST.search(sentence) and targets and \
                    any(re.search(r"\b" + re.escape(t), sentence, re.I) for t in targets):
                asks, question, pattern = True, original, "information_request"
            if asks:
                target_mentioned = any(re.search(r"\b" + re.escape(t), sentence, re.I) for t in targets)
        if assumption is None and _ASSUMPTION.search(sentence):
            assumption = original
    return {
        "asks": asks,
        "question": question,
        "pattern": pattern,
        "states_assumption": assumption is not None,
        "assumption": assumption,
        "targets": targets,
        "target_mentioned": target_mentioned,
        "method": "heuristic",
    }


# ── Behaviour ────────────────────────────────────────────────────────────────────────────────

def classify_behaviour(abstention: AbstentionResult, clarification: dict | None,
                       expected_behavior: str) -> str:
    """Map detector outputs to one detected behaviour.

    Order: empty; real abstentions; clarifying question; (clarify items only) stated assumption;
    otherwise answer. An insufficient-information style abstention followed by a clarifying question
    ("I need more detail: which system do you mean?") is a clarification. "Assuming ..." is only
    meaningful for clarify items, because ordinary answers state assumptions all the time.
    """
    asks = bool(clarification and clarification.get("asks"))
    if abstention.empty:
        return EMPTY
    if abstention.abstained:
        explainable = all(m.category in _CLARIFIABLE_ABSTENTIONS for m in abstention.matches)
        if not (asks and explainable):
            if abstention.mixed:
                return MIXED
            return REFUSAL if abstention.is_refusal else ABSTENTION
    if asks:
        return CLARIFICATION
    if expected_behavior == "clarify" and clarification and clarification.get("states_assumption"):
        return ASSUMPTION
    return ANSWER


def judge_behaviour(item: Item, detected: str) -> tuple[DimensionResult, list[str], list[str]]:
    """Return (result, behavioural_violations, warnings) per the approved compliance matrix."""
    exp = item.expected_behavior
    violations: list[str] = []
    warnings: list[str] = []

    def r(status, reason, **details):
        return DimensionResult(status, reason, {"expected": exp, "detected": detected, **details})

    if detected == EMPTY:
        violations.append("empty_response")
        return r(FAIL, "empty response"), violations, warnings

    if exp == "answer":
        if detected == ANSWER:
            return r(PASS, "answered"), violations, warnings
        if detected in (ABSTENTION, REFUSAL):
            violations.append("false_refusal")
            return r(FAIL, "abstained on an answerable question"), violations, warnings
        if detected == MIXED:
            violations.append("answer_with_abstention")
            return r(PARTIAL, "answered but also said it could not"), violations, warnings
        if detected == CLARIFICATION:
            violations.append("unnecessary_clarification")
            warnings.append("clarification_detected_by_heuristic")
            return r(PARTIAL, "asked for clarification on an unambiguous question"), violations, warnings

    if exp == "abstain":
        if detected == ABSTENTION:
            return r(PASS, "abstained"), violations, warnings
        if detected == REFUSAL:
            warnings.append("scope_refusal_not_inability")
            return r(PASS, "refused on scope (accepted, D5)"), violations, warnings
        if detected == MIXED:
            warnings.append("content_beyond_abstention")
            violations.append("content_beyond_abstention")
            return r(PARTIAL, "abstained but added other content that cannot be checked (D4)"), violations, warnings
        if detected == ANSWER:
            violations.append("answered_uncovered_question")
            return r(FAIL, "answered a question the materials do not cover"), violations, warnings
        if detected == CLARIFICATION:
            violations.append("no_explicit_abstention")
            warnings.append("clarification_detected_by_heuristic")
            return r(FAIL, "asked for clarification instead of saying the materials do not cover it"), \
                violations, warnings

    if exp == "clarify":
        warnings.append("clarification_detected_by_heuristic")
        if detected == CLARIFICATION:
            return r(PASS, "asked the student to clarify (heuristic)"), violations, warnings
        if detected == ASSUMPTION:
            if _accepts_assumption(item):
                warnings.append("assumption_content_not_verified")
                return r(PASS, "stated an assumption, an accepted interpretation (heuristic)"), violations, warnings
            violations.append("assumed_without_asking")
            return r(PARTIAL, "stated an assumption, but no interpretation accepts one"), violations, warnings
        if detected == ANSWER:
            if _accepts_general_answer(item):
                return r(NOT_EVALUATED, "no clarifying question or stated assumption found; an accepted "
                                        "interpretation is a general answer, which cannot be verified "
                                        "deterministically"), violations, warnings
            violations.append("answered_without_clarifying")
            return r(FAIL, "answered an ambiguous question without asking or stating an assumption"), \
                violations, warnings
        if detected in (ABSTENTION, REFUSAL):
            violations.append("false_refusal")
            return r(FAIL, "abstained on a question that has course evidence"), violations, warnings
        if detected == MIXED:
            violations.append("answer_with_abstention")
            return r(PARTIAL, "said it could not answer and also gave other content"), violations, warnings

    raise ValueError(f"unhandled behaviour combination: expected {exp!r}, detected {detected!r}")


# ── Completeness ─────────────────────────────────────────────────────────────────────────────

def judge_completeness(item: Item, key_points: dict) -> tuple[DimensionResult, list[str]]:
    """Required key-point coverage over regex-checkable points, with lower and upper bounds (D6).

    lower = matched / total x 100 (the requested formula; unverifiable points count as not matched)
    upper = (matched + unverifiable) / total x 100 (assumes every unverifiable point is present)
    """
    warnings: list[str] = []
    matched, missing, unverifiable = (len(key_points[k]) for k in ("matched", "missing", "unverifiable"))
    total = matched + missing + unverifiable
    checkable = matched + missing
    details = {"matched": matched, "missing": missing, "unverifiable": unverifiable, "total": total,
               "checkable": checkable, "lower": None, "upper": None}
    if matched:
        warnings.append("pattern_match_is_not_semantic_confirmation")
    if total == 0:
        return DimensionResult(NOT_APPLICABLE, "item has no required key points", details), warnings
    if checkable == 0:
        return DimensionResult(NOT_EVALUATED, "no checkable key point (no required key point has a regex)",
                               details), warnings
    details["lower"] = round(matched / total * 100, 1)
    details["upper"] = round((matched + unverifiable) / total * 100, 1)
    if matched == 0:
        return DimensionResult(FAIL, "no checkable required key point found", details), warnings
    if missing:
        return DimensionResult(PARTIAL, f"{missing} of {checkable} checkable required key points missing",
                               details), warnings
    if unverifiable:
        warnings.append("unverified_key_points")
        return DimensionResult(PARTIAL, f"all checkable points found; {unverifiable} cannot be checked",
                               details), warnings
    return DimensionResult(PASS, "all required key points found (pattern evidence)", details), warnings


# ── Not-evaluated dimensions ─────────────────────────────────────────────────────────────────

def judge_correctness(item: Item) -> DimensionResult:
    """D1: forbidden claims are free text, so correctness is never decided in Sprint 3."""
    if item.expected_behavior != "answer":
        return DimensionResult(NOT_APPLICABLE, f"not scored for {item.expected_behavior} items")
    return DimensionResult(NOT_EVALUATED, "forbidden claims are free text; no deterministic detection",
                           {"claims_to_check": len(item.forbidden_claims)})


def judge_groundedness(item: Item, detected: str, retrieved_chunks: Sequence | None) -> DimensionResult:
    """Never inferred from citations. A clean abstention has nothing to ground."""
    details = {"retrieved_chunks_supplied": None if retrieved_chunks is None else len(retrieved_chunks)}
    if item.expected_behavior == "abstain" and detected in (ABSTENTION, REFUSAL):
        return DimensionResult(NOT_APPLICABLE, "clean abstention; no claims to ground", details)
    if retrieved_chunks is None:
        return DimensionResult(NOT_EVALUATED, "retrieved chunks not supplied", details)
    return DimensionResult(NOT_EVALUATED, "retrieved chunks recorded, but Sprint 3 has no grounding method",
                           details)


def judge_language(item: Item) -> DimensionResult:
    """No validated detector; English and Hinglish share the Latin script."""
    return DimensionResult(NOT_EVALUATED, "no validated language detector",
                           {"expected": item.expected_answer_language})


# ── Citation information (reported, never scored) ───────────────────────────────────────────

def _squash_title(t: str) -> str:
    t = re.sub(r"\.pdf$", "", t.strip(), flags=re.I)
    return re.sub(r"[\s_]+", " ", t).strip().lower()


_TITLE_INDEX = {}
for _title, _file in MATERIALS.items():
    _TITLE_INDEX[_squash_title(_title)] = _title
    _TITLE_INDEX[_squash_title(_file)] = _title


def resolve_material(title: str | None) -> str | None:
    """Map a cited title or file name to a MATERIALS title, or None if unknown."""
    if not title:
        return None
    return _TITLE_INDEX.get(_squash_title(title))


def _page(p: str | None) -> int | None:
    try:
        return int(p) if p is not None else None
    except ValueError:
        return None


def citation_info(item: Item, parsed: CitationParseResult) -> dict:
    """Describe citations and compare page-bearing labels with expected_sources.

    expected_source_cited is True when a page-bearing label cites an expected (material, page),
    False when page-bearing labels exist but none is expected, and None (unknown) otherwise: numeric
    [n] citations cannot be resolved without the endpoint's sources list.
    """
    expected = {(s["material"], s["page"]) for s in item.expected_sources}
    cited = []
    for c in parsed.citations:
        if c.kind not in ("page_label", "source_label", "title_label"):
            continue
        material = resolve_material(c.title)
        page = _page(c.page)
        cited.append({
            "text": c.text, "kind": c.kind, "material": material, "page": page,
            "in_sources_block": c.in_sources_block,
            "expected": (material, page) in expected if (material and page is not None) else None,
        })
    with_page = [c for c in cited if c["expected"] is not None]
    if with_page:
        expected_cited = any(c["expected"] for c in with_page)
    else:
        expected_cited = None
    return {
        "inline_count": len(parsed.inline),
        "total_count": len(parsed.citations),
        "sources_block": parsed.has_sources_block,
        "counts_by_kind": {k: sum(1 for c in parsed.citations if c.kind == k)
                           for k in dict.fromkeys(c.kind for c in parsed.citations)},
        "counts_by_key": dict(parsed.counts()),
        "unique_keys": list(parsed.unique_keys()),
        "malformed": [{"text": m.text, "reason": m.reason} for m in parsed.malformed],
        "cited_sources": cited,
        "expected_sources": [dict(s) for s in item.expected_sources],
        "expected_source_cited": expected_cited,
        "note": "citation information only; never used as evidence of groundedness",
    }
