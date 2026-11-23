"""Sprint 3: result types for the deterministic answer evaluation engine.

Statuses are a closed set. A dimension that cannot be checked with the data available is
`not_evaluated` (always with a reason), never silently counted as pass or fail. A dimension that
makes no sense for an item's expected behaviour is `not_applicable`.

Overall status (approved decision D3) is a rule, not a weighted score:
    fail          any evaluated, applicable dimension failed
    partial       otherwise, any evaluated, applicable dimension is partial
    pass          every applicable dimension was evaluated and passed
    inconclusive  everything that could be checked passed, but an applicable dimension is not_evaluated
    not_evaluated no response was supplied for the item
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "PASS", "PARTIAL", "FAIL", "NOT_EVALUATED", "NOT_APPLICABLE", "INCONCLUSIVE",
    "DIMENSION_STATUSES", "OVERALL_STATUSES", "DIMENSIONS",
    "DimensionResult", "EvaluationResult", "BatchReport", "overall_status",
]

PASS = "pass"
PARTIAL = "partial"
FAIL = "fail"
NOT_EVALUATED = "not_evaluated"
NOT_APPLICABLE = "not_applicable"
INCONCLUSIVE = "inconclusive"

DIMENSION_STATUSES = (PASS, PARTIAL, FAIL, NOT_EVALUATED, NOT_APPLICABLE)
OVERALL_STATUSES = (PASS, PARTIAL, FAIL, INCONCLUSIVE, NOT_EVALUATED)

# Report order. Names match the JSON keys under "dimensions".
DIMENSIONS = ("behaviour", "completeness", "correctness", "groundedness", "language")


@dataclass(frozen=True)
class DimensionResult:
    status: str
    reason: str
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in DIMENSION_STATUSES:
            raise ValueError(f"unknown dimension status {self.status!r}")
        if self.status == NOT_EVALUATED and not self.reason:
            raise ValueError("not_evaluated needs a reason")

    def to_dict(self) -> dict:
        return {"status": self.status, "reason": self.reason, "details": _json_safe(self.details)}


def overall_status(dimensions: dict[str, DimensionResult]) -> str:
    """Combine per-dimension statuses without weights (decision D3).

    `not_applicable` dimensions are ignored. With no applicable dimension at all the result is
    `inconclusive` (nothing was shown to pass).
    """
    applicable = [d.status for d in dimensions.values() if d.status != NOT_APPLICABLE]
    if FAIL in applicable:
        return FAIL
    if PARTIAL in applicable:
        return PARTIAL
    if applicable and all(s == PASS for s in applicable):
        return PASS
    return INCONCLUSIVE


@dataclass
class EvaluationResult:
    item_id: str
    label_status: str
    provisional: bool
    category: str
    response: str | None
    normalized_response: str | None
    expected_behavior: str
    detected_behavior: str | None
    abstention: dict | None
    clarification: dict | None
    key_points: dict
    forbidden_claims: dict
    behavioural_violations: list[str]
    must_not_to_check: list[str]
    citations: dict | None
    dimensions: dict[str, DimensionResult]
    overall_status: str
    warnings: list[str]
    limitations: list[str]
    expected_answer_language: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "category": self.category,
            "label_status": self.label_status,
            "provisional": self.provisional,
            "response": self.response,
            "normalized_response": self.normalized_response,
            "expected_behavior": self.expected_behavior,
            "detected_behavior": self.detected_behavior,
            "expected_answer_language": self.expected_answer_language,
            "abstention": _json_safe(self.abstention),
            "clarification": _json_safe(self.clarification),
            "key_points": _json_safe(self.key_points),
            "forbidden_claims": _json_safe(self.forbidden_claims),
            "behavioural_violations": list(self.behavioural_violations),
            "must_not_to_check": list(self.must_not_to_check),
            "citations": _json_safe(self.citations),
            "dimensions": {name: self.dimensions[name].to_dict() for name in DIMENSIONS},
            "overall_status": self.overall_status,
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
            "metadata": _json_safe(self.metadata),
        }


@dataclass
class BatchReport:
    provisional: bool
    dataset: dict
    input_errors: list[dict]
    not_responded: list[str]
    excluded_by_filter: list[str]
    overall_status_counts: dict[str, int]
    dimension_status_counts: dict[str, dict[str, int]]
    by_expected_behavior: dict[str, dict]
    by_category: dict[str, dict]
    results: list[EvaluationResult]

    def to_dict(self) -> dict:
        return {
            "provisional": self.provisional,
            "dataset": _json_safe(self.dataset),
            "input_errors": _json_safe(self.input_errors),
            "not_responded": list(self.not_responded),
            "excluded_by_filter": list(self.excluded_by_filter),
            "overall_status_counts": dict(self.overall_status_counts),
            "dimension_status_counts": _json_safe(self.dimension_status_counts),
            "by_expected_behavior": _json_safe(self.by_expected_behavior),
            "by_category": _json_safe(self.by_category),
            "results": [r.to_dict() for r in self.results],
        }


def _json_safe(value: Any) -> Any:
    """Convert tuples, sets and dataclass-like values into plain JSON types (recursively)."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_safe(v) for v in value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)
