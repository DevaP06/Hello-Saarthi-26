"""Deterministic ranking metrics for retrieval evaluation (Sprint 5).

Every metric is computed for ONE ranked list against ONE set of relevance judgements, at a cutoff K,
and returns a MetricValue with a status:

  exact           the value is fully determined by the judgements
  bounded         unjudged items in the top K make the value uncertain; `lower` and `upper` give the
                  tightest range consistent with the judgements, `value` is None
  inconclusive    the value cannot be bounded honestly (NDCG with unjudged items in the top K)
  not_applicable  no labelled relevant item exists, so the metric is undefined (0/0)

Judgements
  grades       {item_id: grade}; grade is an integer >= 0, 0 meaning "judged not relevant".
  exhaustive   True: every item NOT listed is judged not relevant (grade 0).
               False (partial): an item not listed is UNJUDGED. It is never scored as irrelevant.
An item whose id is unknown (None, e.g. a hit the adapter could not resolve to a document) is always
unjudged, because it might be any item, including a relevant one.

Ranking preparation (prepare_ranking)
  duplicates "collapse" (default): only the first occurrence of an id is kept, order is preserved and
             K counts positions in the collapsed list. A repeated id can never earn credit twice.
  duplicates "penalize": a repeated id keeps its slot but has grade 0 (a redundant result), so it
             uses up a position in the top K without adding relevance.
  None ids are never treated as duplicates of each other.
  The recorded order IS the ranking. Scores never reorder results, so ties in score are resolved by
  the recorded order; the result is deterministic for a given input.

Definitions (rel(i) = grade(i) >= threshold; R = number of labelled items with rel; positions 1-based)
  Precision@K  = |{i <= K : rel(i)}| / K. The denominator is always K: empty slots (fewer than K
                 results) count as not relevant. Bounds with u unjudged items in the top K:
                 [hits / K, (hits + u) / K].
  Recall@K     = |{i <= K : rel(i)}| / R. With partial judgements this is recall of the LABELLED
                 relevant items (basis "labelled_relevant_only"); the true recall is unknown.
  RR@K         = 1 / (position of the first relevant item within the top K), 0 if none. MRR is the
                 mean of RR@K over cases. If an unjudged item precedes the first relevant one, the
                 value is bounded: [1 / first_relevant (or 0), 1 / first_unjudged].
  NDCG@K       = DCG@K / IDCG@K, DCG@K = sum_{i<=K} gain(grade_i) / log2(i + 1),
                 IDCG@K = the same sum over the labelled grades sorted in descending order.
                 gain "linear": g (the scikit-learn convention); "exponential": 2^g - 1.
                 NDCG uses the grades themselves, not the binary threshold; it is not_applicable if
                 no labelled grade is positive. Any unjudged item in the top K makes it inconclusive:
                 an unjudged item could change both DCG and IDCG, so no honest bound exists.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

EXACT = "exact"
BOUNDED = "bounded"
INCONCLUSIVE = "inconclusive"
NOT_APPLICABLE = "not_applicable"
NOT_EVALUATED = "not_evaluated"   # used by the benchmark when no labels or no ranking exist
METRIC_STATUSES = (EXACT, BOUNDED, INCONCLUSIVE, NOT_APPLICABLE, NOT_EVALUATED)

METRICS = ("precision", "recall", "mrr", "ndcg")
DUPLICATE_POLICIES = ("collapse", "penalize")
GAINS = ("linear", "exponential")


class _Duplicate:
    """Marker for a repeated id kept in place under the 'penalize' policy."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "DUPLICATE"


DUPLICATE = _Duplicate()


@dataclass(frozen=True)
class Judgements:
    grades: Mapping[str, int]
    exhaustive: bool

    def __post_init__(self) -> None:
        for item_id, g in self.grades.items():
            if not isinstance(item_id, str) or not item_id:
                raise ValueError(f"judged item id must be a non-empty string, got {item_id!r}")
            if isinstance(g, bool) or not isinstance(g, int) or g < 0:
                raise ValueError(f"grade for {item_id!r} must be an integer >= 0, got {g!r}")
        object.__setattr__(self, "grades", MappingProxyType(dict(self.grades)))

    def grade(self, item) -> int | None:
        """Grade of a ranked item, or None if it is unjudged."""
        if item is DUPLICATE:
            return 0
        if item is None:
            return None
        if item in self.grades:
            return self.grades[item]
        return 0 if self.exhaustive else None

    def relevant_count(self, threshold: int) -> int:
        return sum(1 for g in self.grades.values() if g >= threshold)


@dataclass(frozen=True)
class Ranking:
    """A prepared ranking: items are ids (str), None (unknown item) or DUPLICATE."""

    items: tuple
    duplicate_positions: tuple[int, ...] = ()   # 1-based positions in the ORIGINAL list
    original_length: int = 0


@dataclass(frozen=True)
class MetricValue:
    status: str
    value: float | None = None
    lower: float | None = None
    upper: float | None = None
    reason: str = ""
    details: Mapping = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"status": self.status, "value": self.value, "lower": self.lower, "upper": self.upper,
                "reason": self.reason, "details": dict(self.details)}


def _check_k(k) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"K must be an integer >= 1, got {k!r}")


def _check_threshold(threshold) -> None:
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 1:
        raise ValueError(f"relevance threshold must be an integer >= 1, got {threshold!r}")


def prepare_ranking(retrieved: Sequence[str | None], duplicates: str = "collapse") -> Ranking:
    """Apply the duplicate policy to a ranked list of ids (None = unknown item)."""
    if duplicates not in DUPLICATE_POLICIES:
        raise ValueError(f"duplicates must be one of {DUPLICATE_POLICIES}, got {duplicates!r}")
    if isinstance(retrieved, (str, bytes)):
        raise ValueError("retrieved must be a sequence of ids, not a string")
    seen: set[str] = set()
    items: list = []
    dup_positions: list[int] = []
    for pos, item in enumerate(retrieved, 1):
        if item is not None and (not isinstance(item, str) or not item):
            raise ValueError(f"retrieved id at position {pos} must be a non-empty string or None, got {item!r}")
        if item is not None and item in seen:
            dup_positions.append(pos)
            if duplicates == "penalize":
                items.append(DUPLICATE)
            continue
        if item is not None:
            seen.add(item)
        items.append(item)
    return Ranking(tuple(items), tuple(dup_positions), len(retrieved))


def _top(ranking: Ranking, judgements: Judgements, k: int) -> list[int | None]:
    return [judgements.grade(item) for item in ranking.items[:k]]


def _no_relevant(what: str) -> MetricValue:
    return MetricValue(NOT_APPLICABLE, reason=f"no labelled relevant item, so {what} is undefined")


def precision_at_k(ranking: Ranking, judgements: Judgements, k: int, *, threshold: int = 1) -> MetricValue:
    _check_k(k)
    _check_threshold(threshold)
    if judgements.relevant_count(threshold) == 0:
        return _no_relevant("precision")
    grades = _top(ranking, judgements, k)
    hits = sum(1 for g in grades if g is not None and g >= threshold)
    unjudged = sum(1 for g in grades if g is None)
    details = {"relevant_in_top_k": hits, "unjudged_in_top_k": unjudged, "retrieved_in_top_k": len(grades)}
    if unjudged == 0:
        return MetricValue(EXACT, hits / k, hits / k, hits / k, "", details)
    return MetricValue(BOUNDED, None, hits / k, (hits + unjudged) / k,
                       f"{unjudged} unjudged item(s) in the top {k}", details)


def recall_at_k(ranking: Ranking, judgements: Judgements, k: int, *, threshold: int = 1) -> MetricValue:
    _check_k(k)
    _check_threshold(threshold)
    total = judgements.relevant_count(threshold)
    if total == 0:
        return _no_relevant("recall")
    grades = _top(ranking, judgements, k)
    hits = sum(1 for g in grades if g is not None and g >= threshold)
    basis = "exhaustive_labels" if judgements.exhaustive else "labelled_relevant_only"
    reason = "" if judgements.exhaustive else "partial labels: recall of the labelled relevant items only"
    return MetricValue(EXACT, hits / total, hits / total, hits / total, reason,
                       {"relevant_in_top_k": hits, "labelled_relevant": total, "basis": basis})


def reciprocal_rank(ranking: Ranking, judgements: Judgements, k: int, *, threshold: int = 1) -> MetricValue:
    _check_k(k)
    _check_threshold(threshold)
    if judgements.relevant_count(threshold) == 0:
        return _no_relevant("reciprocal rank")
    first_rel = first_unj = None
    for pos, g in enumerate(_top(ranking, judgements, k), 1):
        if g is None and first_unj is None:
            first_unj = pos
        if g is not None and g >= threshold:
            first_rel = pos
            break
    details = {"first_relevant_rank": first_rel, "first_unjudged_rank": first_unj}
    lower = 1 / first_rel if first_rel else 0.0
    if first_unj is None:
        return MetricValue(EXACT, lower, lower, lower, "", details)
    return MetricValue(BOUNDED, None, lower, 1 / first_unj,
                       f"an unjudged item at rank {first_unj} precedes the first relevant item", details)


def _gain(grade: int, gain: str) -> float:
    return float(grade) if gain == "linear" else float(2 ** grade - 1)


def ndcg_at_k(ranking: Ranking, judgements: Judgements, k: int, *, gain: str = "linear") -> MetricValue:
    _check_k(k)
    if gain not in GAINS:
        raise ValueError(f"gain must be one of {GAINS}, got {gain!r}")
    if not any(g > 0 for g in judgements.grades.values()):
        return _no_relevant("NDCG")
    grades = _top(ranking, judgements, k)
    unjudged = sum(1 for g in grades if g is None)
    if unjudged:
        return MetricValue(INCONCLUSIVE, reason=f"{unjudged} unjudged item(s) in the top {k}; an unjudged "
                           f"item could change both DCG and IDCG", details={"unjudged_in_top_k": unjudged})
    dcg = math.fsum(_gain(g, gain) / math.log2(i + 1) for i, g in enumerate(grades, 1))
    ideal = sorted(judgements.grades.values(), reverse=True)[:k]
    idcg = math.fsum(_gain(g, gain) / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    value = dcg / idcg
    return MetricValue(EXACT, value, value, value, "", {"dcg": dcg, "idcg": idcg})


def compute_all(ranking: Ranking, judgements: Judgements, k: int, *, threshold: int = 1,
                gain: str = "linear") -> dict[str, MetricValue]:
    return {
        "precision": precision_at_k(ranking, judgements, k, threshold=threshold),
        "recall": recall_at_k(ranking, judgements, k, threshold=threshold),
        "mrr": reciprocal_rank(ranking, judgements, k, threshold=threshold),
        "ndcg": ndcg_at_k(ranking, judgements, k, gain=gain),
    }
