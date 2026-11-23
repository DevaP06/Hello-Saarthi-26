"""Sprint 3: deterministic answer evaluation engine.

    evaluate_response(item, response, *, known_titles, retrieved_chunks, metadata) -> EvaluationResult
    evaluate_batch(items, responses, *, reviewed_only, metadata, ...)          -> BatchReport

Pipeline for one (item, response) pair:
    normalise (response_normalizer) -> extract signals (citation_parser, abstention_detector,
    key-point regexes, clarification heuristic) -> judge each dimension independently ->
    overall status by rule (no weights, decision D3).

The engine evaluates text the caller supplies. It never calls Saarthi, a model or the network.
Correctness, groundedness and language compliance are reported as not_evaluated with a reason;
they are never inferred from citation presence, regex hits or script detection.
Every result for a draft-labelled item is provisional.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping, Sequence

from eval.answer_eval.abstention_detector import detect_abstention
from eval.answer_eval.citation_parser import parse_citations
from eval.answer_eval.evaluation_dimensions import (
    citation_info, classify_behaviour, detect_clarification, judge_behaviour, judge_completeness,
    judge_correctness, judge_groundedness, judge_language, match_key_points,
)
from eval.answer_eval.evaluation_result import (
    DIMENSION_STATUSES, DIMENSIONS, NOT_APPLICABLE, NOT_EVALUATED, OVERALL_STATUSES,
    BatchReport, DimensionResult, EvaluationResult, overall_status,
)
from eval.answer_eval.response_normalizer import normalize_response
from eval.answer_eval.schema import MATERIALS, Item, check_item

__all__ = ["evaluate_response", "evaluate_batch", "APPLICABILITY", "InvalidDatasetError"]

# Which dimensions can apply to which expected behaviour (design section 5.1). A dimension listed
# here may still end up not_applicable for a particular response (e.g. groundedness of a clean
# abstention); one not listed is always not_applicable.
APPLICABILITY = {
    "answer": ("behaviour", "completeness", "correctness", "groundedness", "language"),
    "abstain": ("behaviour", "groundedness", "language"),
    "clarify": ("behaviour", "completeness", "groundedness", "language"),
}

_LIMITATION_TEXT = {
    "correctness": "correctness not evaluated: forbidden claims are free text (D1)",
    "groundedness": "groundedness not evaluated: no retrieved chunks or grounding method in Sprint 3",
    "language": "language compliance not evaluated: no validated detector",
    "completeness": "completeness not evaluated: no required key point has a regex",
    "behaviour": "behaviour not evaluated: response could not be classified deterministically",
}


class InvalidDatasetError(ValueError):
    """Raised by evaluate_batch when items fail schema checks."""


def _no_response_result(item: Item, metadata: dict | None) -> EvaluationResult:
    reason = "no response supplied"
    dims = {name: DimensionResult(NOT_EVALUATED, reason) for name in DIMENSIONS}
    return EvaluationResult(
        item_id=item.id, label_status=item.label_status, provisional=item.label_status != "reviewed",
        category=item.category, response=None, normalized_response=None,
        expected_behavior=item.expected_behavior, detected_behavior=None, abstention=None,
        clarification=None, key_points={"matched": [], "missing": [], "unverifiable": [], "optional": []},
        forbidden_claims=_claims_block(item), behavioural_violations=[],
        must_not_to_check=list(item.must_not), citations=None, dimensions=dims,
        overall_status=NOT_EVALUATED,
        warnings=["draft_label"] if item.label_status == "draft" else [],
        limitations=["no_response"], expected_answer_language=item.expected_answer_language,
        metadata=dict(metadata or {}),
    )


def _claims_block(item: Item) -> dict:
    return {
        "detection": NOT_EVALUATED,
        "detected": None,  # None = not checked; never read as "no forbidden claim asserted"
        "claims_to_check": [{"text": c.text, "reason": c.reason} for c in item.forbidden_claims],
    }


def evaluate_response(
    item: Item,
    response: str | None,
    *,
    known_titles: Iterable[str] | None = None,
    retrieved_chunks: Sequence[dict] | None = None,
    metadata: dict | None = None,
) -> EvaluationResult:
    """Evaluate one response against one dataset item.

    response          the assistant's text. None means "no response supplied" (overall not_evaluated);
                      "" or whitespace is an empty response (behaviour fail).
    known_titles      titles a bare "[Title]" may cite; defaults to the four course materials.
    retrieved_chunks  recorded (count only) for a later grounding check; not scored in Sprint 3.
    metadata          passed through unchanged (engine, endpoint, model, latency, ...).
    """
    if response is not None and not isinstance(response, str):
        raise TypeError(f"response must be str or None, not {type(response).__name__}")
    if response is None:
        return _no_response_result(item, metadata)

    titles = list(MATERIALS) if known_titles is None else list(known_titles)
    norm = normalize_response(response)
    text = norm.normalized

    abstention = detect_abstention(text)
    clarification = detect_clarification(item, text)
    detected = classify_behaviour(abstention, clarification, item.expected_behavior)
    parsed = parse_citations(text, known_titles=titles)
    key_points = match_key_points(item, text)

    applicable = APPLICABILITY[item.expected_behavior]
    warnings: list[str] = []

    behaviour, violations, w = judge_behaviour(item, detected)
    warnings += w

    if "completeness" in applicable:
        completeness, w = judge_completeness(item, key_points)
        warnings += w
    else:
        completeness = DimensionResult(NOT_APPLICABLE, f"not scored for {item.expected_behavior} items")

    dims = {
        "behaviour": behaviour,
        "completeness": completeness,
        "correctness": judge_correctness(item),
        "groundedness": judge_groundedness(item, detected, retrieved_chunks),
        "language": judge_language(item),
    }
    for name in DIMENSIONS:
        if name not in applicable and dims[name].status != NOT_APPLICABLE:
            raise AssertionError(f"{name} must be not_applicable for {item.expected_behavior} items")

    citations = citation_info(item, parsed)
    if citations["malformed"]:
        warnings.append("malformed_citations")
    if item.label_status == "draft":
        warnings.append("draft_label")

    limitations = [_LIMITATION_TEXT[name] for name in DIMENSIONS if dims[name].status == NOT_EVALUATED]
    if detected in ("clarification", "assumption") or item.expected_behavior == "clarify":
        limitations.append("clarification judged by a deterministic heuristic (D2)")

    return EvaluationResult(
        item_id=item.id,
        label_status=item.label_status,
        provisional=item.label_status != "reviewed",
        category=item.category,
        response=norm.original,
        normalized_response=text,
        expected_behavior=item.expected_behavior,
        detected_behavior=detected,
        abstention={
            "abstained": abstention.abstained, "category": abstention.category,
            "matched_text": abstention.matched_text, "mixed": abstention.mixed,
            "is_refusal": abstention.is_refusal, "empty": abstention.empty, "reason": abstention.reason,
            "matches": [{"category": m.category, "text": m.text, "sentence": m.sentence}
                        for m in abstention.matches],
        },
        clarification=clarification,
        key_points=key_points,
        forbidden_claims=_claims_block(item),
        behavioural_violations=violations,
        must_not_to_check=list(item.must_not),
        citations=citations,
        dimensions=dims,
        overall_status=overall_status(dims),
        warnings=list(dict.fromkeys(warnings)),
        limitations=limitations,
        expected_answer_language=item.expected_answer_language,
        metadata=dict(metadata or {}),
    )


def _count(results: list[EvaluationResult]) -> dict:
    overall = Counter(r.overall_status for r in results)
    dims = {name: Counter(r.dimensions[name].status for r in results) for name in DIMENSIONS}
    return {
        "items": len(results),
        "overall_status_counts": {s: overall.get(s, 0) for s in OVERALL_STATUSES},
        "dimension_status_counts": {name: {s: dims[name].get(s, 0) for s in DIMENSION_STATUSES}
                                    for name in DIMENSIONS},
    }


def evaluate_batch(
    items: Sequence[Item],
    responses: Mapping[str, str | None] | Iterable[tuple[str, str | None]],
    *,
    reviewed_only: bool = False,
    metadata: Mapping[str, dict] | None = None,
    retrieved_chunks: Mapping[str, Sequence[dict]] | None = None,
    known_titles: Iterable[str] | None = None,
    dataset_path: str | None = None,
) -> BatchReport:
    """Evaluate many responses. Adds no logic beyond evaluate_response and counting.

    items      dataset items; each must pass schema.check_item and ids must be unique
               (InvalidDatasetError otherwise). Full dataset + corpus validation is run by the CLI.
    responses  {item_id: text} or an iterable of (item_id, text) pairs. Pairs allow duplicate ids
               to be reported as input errors instead of silently overwritten.
    Items with no response get overall not_evaluated and are listed in not_responded; they are
    never counted as failures. Results follow dataset order.
    """
    errors = []
    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            errors.append(f"{item.id}: duplicate item id")
        seen.add(item.id)
        errors += check_item(item)
    if errors:
        raise InvalidDatasetError("; ".join(errors))

    pairs = list(responses.items()) if isinstance(responses, Mapping) else list(responses)
    by_id = {item.id: item for item in items}
    input_errors: list[dict] = []
    given: dict[str, str | None] = {}
    duplicated: set[str] = set()
    for pair in pairs:
        rid, text = pair
        if rid not in by_id:
            input_errors.append({"id": rid, "error": "unknown item id"})
        elif rid in given or rid in duplicated:
            input_errors.append({"id": rid, "error": "duplicate response for item id; none evaluated"})
            duplicated.add(rid)
            given.pop(rid, None)
        elif text is not None and not isinstance(text, str):
            input_errors.append({"id": rid, "error": f"response must be a string, not {type(text).__name__}"})
            duplicated.add(rid)  # an invalid response is not evaluated
        else:
            given[rid] = text

    selected = [i for i in items if i.label_status == "reviewed"] if reviewed_only else list(items)
    selected_ids = {i.id for i in selected}
    excluded = [i.id for i in items if i.id not in selected_ids]

    results: list[EvaluationResult] = []
    not_responded: list[str] = []
    for item in selected:
        meta = dict((metadata or {}).get(item.id, {}))
        if item.id not in given or given[item.id] is None:
            not_responded.append(item.id)
            results.append(evaluate_response(item, None, metadata=meta))
            continue
        results.append(evaluate_response(
            item, given[item.id], known_titles=known_titles,
            retrieved_chunks=(retrieved_chunks or {}).get(item.id), metadata=meta,
        ))

    totals = _count(results)
    by_behavior = {b: _count([r for r in results if r.expected_behavior == b])
                   for b in ("answer", "abstain", "clarify")}
    categories = list(dict.fromkeys(i.category for i in selected))
    by_category = {c: _count([r for r in results if r.category == c]) for c in categories}
    labels = Counter(i.label_status for i in selected)

    return BatchReport(
        provisional=any(r.provisional for r in results) or not results,
        dataset={"path": dataset_path, "items": len(selected), "reviewed": labels.get("reviewed", 0),
                 "draft": labels.get("draft", 0), "reviewed_only": reviewed_only},
        input_errors=input_errors,
        not_responded=not_responded,
        excluded_by_filter=excluded,
        overall_status_counts=totals["overall_status_counts"],
        dimension_status_counts=totals["dimension_status_counts"],
        by_expected_behavior=by_behavior,
        by_category=by_category,
        results=results,
    )
