"""Sprint 5 metric tests. Expected values are computed by hand from the definitions in metrics.py."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.retrieval_eval import metrics as M  # noqa: E402
from eval.retrieval_eval.metrics import Judgements, prepare_ranking  # noqa: E402

L2 = math.log2


def J(grades, exhaustive=True):
    return Judgements(grades, exhaustive)


def R(*ids, duplicates="collapse"):
    return prepare_ranking(list(ids), duplicates)


def approx(x):
    return pytest.approx(x, abs=1e-12)


# ── Precision@K ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("k,expected", [(1, 1.0), (2, 0.5), (3, 2 / 3), (4, 0.5), (5, 0.4), (10, 0.2)])
def test_precision_at_multiple_k(k, expected):
    mv = M.precision_at_k(R("a", "x", "b", "y"), J({"a": 1, "b": 1}), k)
    assert mv.status == M.EXACT and mv.value == approx(expected)


def test_precision_denominator_is_k_even_with_fewer_results():
    mv = M.precision_at_k(R("a"), J({"a": 1}), 5)
    assert mv.value == approx(0.2) and mv.details["retrieved_in_top_k"] == 1


def test_precision_empty_retrieval_is_zero_not_undefined():
    mv = M.precision_at_k(R(), J({"a": 1}), 3)
    assert mv.status == M.EXACT and mv.value == 0.0


def test_precision_empty_relevant_set_is_not_applicable():
    for judg in (J({}), J({"a": 0, "b": 0}), J({}, exhaustive=False)):
        mv = M.precision_at_k(R("a", "b"), judg, 2)
        assert mv.status == M.NOT_APPLICABLE and mv.value is None


def test_precision_partial_labels_are_bounded_not_scored_as_irrelevant():
    mv = M.precision_at_k(R("a", "u1", "u2"), J({"a": 1}, exhaustive=False), 3)
    assert mv.status == M.BOUNDED and mv.value is None
    assert (mv.lower, mv.upper) == (approx(1 / 3), approx(1.0))


def test_precision_partial_labels_with_only_judged_items_is_exact():
    mv = M.precision_at_k(R("a", "b"), J({"a": 1, "b": 0}, exhaustive=False), 2)
    assert mv.status == M.EXACT and mv.value == approx(0.5)


def test_unknown_item_is_unjudged_even_with_exhaustive_labels():
    mv = M.precision_at_k(R("a", None), J({"a": 1}), 2)
    assert mv.status == M.BOUNDED and (mv.lower, mv.upper) == (approx(0.5), approx(1.0))


def test_relevance_threshold_for_binary_metrics():
    judg = J({"a": 1, "b": 2})
    assert M.precision_at_k(R("a", "b"), judg, 2, threshold=2).value == approx(0.5)
    assert M.recall_at_k(R("a"), judg, 1, threshold=2).value == 0.0
    assert M.precision_at_k(R("a"), J({"a": 1}), 1, threshold=2).status == M.NOT_APPLICABLE


# ── Recall@K ────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("k,expected", [(1, 0.0), (2, 1 / 3), (3, 1 / 3), (4, 2 / 3), (10, 2 / 3)])
def test_recall_at_multiple_k(k, expected):
    mv = M.recall_at_k(R("x", "a", "y", "b"), J({"a": 1, "b": 1, "c": 1}), k)
    assert mv.status == M.EXACT and mv.value == approx(expected)


def test_recall_partial_labels_states_its_basis():
    mv = M.recall_at_k(R("a", "u"), J({"a": 1, "b": 1}, exhaustive=False), 2)
    assert mv.status == M.EXACT and mv.value == approx(0.5)
    assert mv.details["basis"] == "labelled_relevant_only" and "labelled" in mv.reason
    assert M.recall_at_k(R("a"), J({"a": 1}), 1).details["basis"] == "exhaustive_labels"


def test_recall_empty_cases():
    assert M.recall_at_k(R(), J({"a": 1}), 5).value == 0.0
    assert M.recall_at_k(R("a"), J({}), 5).status == M.NOT_APPLICABLE


# ── Reciprocal rank / MRR ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("k,expected", [(1, 0.0), (2, 0.0), (3, 1 / 3), (5, 1 / 3)])
def test_reciprocal_rank_respects_the_cutoff(k, expected):
    mv = M.reciprocal_rank(R("x", "y", "a", "b"), J({"a": 1, "b": 1}), k)
    assert mv.status == M.EXACT and mv.value == approx(expected)


def test_reciprocal_rank_uses_first_relevant_only():
    assert M.reciprocal_rank(R("a", "b"), J({"a": 1, "b": 3}), 5).value == 1.0


def test_reciprocal_rank_bounded_when_unjudged_precedes_first_relevant():
    mv = M.reciprocal_rank(R("x", "u", "a"), J({"a": 1, "x": 0}, exhaustive=False), 5)
    assert mv.status == M.BOUNDED and (mv.lower, mv.upper) == (approx(1 / 3), approx(0.5))
    assert mv.details == {"first_relevant_rank": 3, "first_unjudged_rank": 2}


def test_reciprocal_rank_bounded_when_no_relevant_but_unjudged():
    mv = M.reciprocal_rank(R("u", "v"), J({"a": 1}, exhaustive=False), 5)
    assert mv.status == M.BOUNDED and (mv.lower, mv.upper) == (0.0, 1.0)


def test_reciprocal_rank_unjudged_after_first_relevant_does_not_matter():
    mv = M.reciprocal_rank(R("a", "u"), J({"a": 1}, exhaustive=False), 5)
    assert mv.status == M.EXACT and mv.value == 1.0


def test_reciprocal_rank_empty_cases():
    assert M.reciprocal_rank(R(), J({"a": 1}), 3).value == 0.0
    assert M.reciprocal_rank(R("a"), J({}), 3).status == M.NOT_APPLICABLE


# ── NDCG@K ──────────────────────────────────────────────────────────────────────────────────

def test_ndcg_binary_hand_computed():
    # ranking: x(0) a(1) b(1), labels a, b, c relevant; K = 3
    dcg = 1 / L2(3) + 1 / L2(4)
    idcg = 1 + 1 / L2(3) + 1 / L2(4)
    mv = M.ndcg_at_k(R("x", "a", "b"), J({"a": 1, "b": 1, "c": 1}), 3)
    assert mv.status == M.EXACT and mv.value == approx(dcg / idcg)


def test_ndcg_graded_linear_and_exponential_gain():
    judg = J({"a": 3, "b": 2, "c": 1})
    ranking = R("c", "a", "b")
    lin = (1 + 3 / L2(3) + 2 / L2(4)) / (3 + 2 / L2(3) + 1 / L2(4))
    exp = (1 + 7 / L2(3) + 3 / L2(4)) / (7 + 3 / L2(3) + 1 / L2(4))
    assert M.ndcg_at_k(ranking, judg, 3).value == approx(lin)
    assert M.ndcg_at_k(ranking, judg, 3, gain="exponential").value == approx(exp)


@pytest.mark.parametrize("k", [1, 2, 3, 10])
def test_ndcg_perfect_ranking_is_one(k):
    assert M.ndcg_at_k(R("a", "b", "c"), J({"a": 3, "b": 2, "c": 1}), k).value == approx(1.0)


def test_ndcg_ideal_is_truncated_at_k():
    # K = 1: ideal is just the top grade (3); ranking puts grade 2 first
    assert M.ndcg_at_k(R("b", "a"), J({"a": 3, "b": 2}), 1).value == approx(2 / 3)


def test_ndcg_ranking_order_matters():
    judg = J({"a": 1})
    assert M.ndcg_at_k(R("a", "x"), judg, 2).value > M.ndcg_at_k(R("x", "a"), judg, 2).value


def test_ndcg_inconclusive_with_unjudged_in_top_k_but_exact_beyond_it():
    judg = J({"a": 1}, exhaustive=False)
    assert M.ndcg_at_k(R("a", "u"), judg, 2).status == M.INCONCLUSIVE
    assert M.ndcg_at_k(R("a", "u"), judg, 1).status == M.EXACT


def test_ndcg_empty_cases():
    assert M.ndcg_at_k(R(), J({"a": 1}), 3).value == 0.0
    assert M.ndcg_at_k(R("a"), J({"a": 0}), 3).status == M.NOT_APPLICABLE


def test_ndcg_uses_grades_not_the_binary_threshold():
    mv = M.ndcg_at_k(R("a"), J({"a": 1}), 1)
    assert mv.status == M.EXACT and mv.value == 1.0


# ── Duplicates ──────────────────────────────────────────────────────────────────────────────

def test_collapse_keeps_first_occurrence_and_reports_positions():
    r = R("a", "b", "a", None, None, "b")
    assert r.items == ("a", "b", None, None) and r.duplicate_positions == (3, 6) and r.original_length == 6


def test_collapsed_duplicates_never_earn_credit_twice():
    judg = J({"a": 1, "b": 1})
    assert M.precision_at_k(R("a", "a", "a"), judg, 3).value == approx(1 / 3)
    assert M.recall_at_k(R("a", "a"), judg, 2).value == approx(0.5)
    assert M.precision_at_k(R("a", "a", "b"), judg, 2).value == approx(1.0)   # b moves up into the top 2


def test_penalize_keeps_duplicate_slots_with_grade_zero():
    judg = J({"a": 1, "b": 1}, exhaustive=False)
    r = R("a", "a", "b", duplicates="penalize")
    assert r.items == ("a", M.DUPLICATE, "b")
    mv = M.precision_at_k(r, judg, 2)
    assert mv.status == M.EXACT and mv.value == approx(0.5)          # the duplicate is judged, not unjudged
    assert M.ndcg_at_k(r, J({"a": 1, "b": 1}), 3).value == approx((1 + 1 / L2(4)) / (1 + 1 / L2(3)))


def test_ties_in_score_are_not_reordered():
    # metrics see only the recorded order
    assert M.reciprocal_rank(R("x", "a"), J({"a": 1}), 2).value == 0.5


# ── Invalid input ───────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("k", [0, -1, 1.5, True, "3", None])
def test_invalid_k_is_rejected(k):
    for fn in (M.precision_at_k, M.recall_at_k, M.reciprocal_rank, M.ndcg_at_k):
        with pytest.raises(ValueError):
            fn(R("a"), J({"a": 1}), k)


@pytest.mark.parametrize("grades", [{"a": -1}, {"a": 1.5}, {"a": True}, {"": 1}, {"a": "1"}])
def test_invalid_grades_are_rejected(grades):
    with pytest.raises(ValueError):
        J(grades)


@pytest.mark.parametrize("ids", [["a", ""], ["a", 3], "abc"])
def test_invalid_ranked_ids_are_rejected(ids):
    with pytest.raises(ValueError):
        prepare_ranking(ids)


def test_invalid_policy_gain_and_threshold_are_rejected():
    with pytest.raises(ValueError):
        prepare_ranking(["a"], "drop")
    with pytest.raises(ValueError):
        M.ndcg_at_k(R("a"), J({"a": 1}), 1, gain="log")
    with pytest.raises(ValueError):
        M.precision_at_k(R("a"), J({"a": 1}), 1, threshold=0)


def test_judgements_are_read_only():
    judg = J({"a": 1})
    with pytest.raises(TypeError):
        judg.grades["b"] = 1


def test_metrics_are_deterministic():
    args = (R("c", "a", "u", "b"), J({"a": 2, "b": 1, "c": 0}, exhaustive=False), 3)
    first = {k: v.to_dict() for k, v in M.compute_all(*args).items()}
    for _ in range(5):
        assert {k: v.to_dict() for k, v in M.compute_all(*args).items()} == first
