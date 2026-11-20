"""Query classifier (rules + learned tier model), Socratic policy, semantic cache.

    pytest saarthi_backend/tests/test_ai_orchestrator.py -v
"""
import json

import numpy as np
import pytest

from saarthi_backend.tests.ai_fixtures import sandbox  # noqa: F401
from src.orchestrator.classifier import classify
from src.orchestrator.socratic import plan_socratic

PROBLEM = "How do I solve for the convolution of u(t) and e^{-2t}u(t)?"


def U(c):
    return {"role": "user", "content": c}


def A(c):
    return {"role": "assistant", "content": c}


# ── rule-based classifier ───────────────────────────────────────────────────

@pytest.mark.parametrize("query,kind", [
    ("hi", "smalltalk"),
    ("What is the Nyquist rate?", "definition"),
    ("define ROC", "definition"),
    ("How do I solve for the convolution of x(t) = u(t) and h(t) = e^-2t u(t)?", "problem"),
    ("Find the Laplace transform of e^{-3t}u(t)", "problem"),
    ("Compare the Fourier and Laplace transforms and explain why the ROC matters", "conceptual"),
    ("explain fourier transform", "conceptual"),
])
def test_rule_classifier_kinds(query, kind):
    assert classify(query).kind == kind


def test_followup_is_detected_from_history():
    c = classify("explain that more simply", [U("what is ROC"), A("...")])
    assert c.kind == "followup" and not c.cacheable


# ── learned Tier 1 / Tier 2 model (weights file replaced by a tiny deterministic one) ─────

def _write_model(path, emb_name="fake-8", dim=8):
    from src.orchestrator.tier_features import CUE_NAMES

    coef = np.zeros(dim + len(CUE_NAMES))
    coef[dim + CUE_NAMES.index("complex_cue")] = 6.0      # "compare / why / prove" -> Tier 2
    coef[dim + CUE_NAMES.index("definition_start")] = -6.0  # "what is ..." -> Tier 1
    path.write_text(json.dumps({"version": 1, "embedding_model": emb_name, "dim": dim,
                                "coef": coef.tolist(), "intercept": 0.0, "threshold": 0.5}), encoding="utf-8")


@pytest.fixture
def tiny_model(tmp_path, monkeypatch):
    import src.orchestrator.query_classifier as qc
    import src.utils.llm as llm

    path = tmp_path / "tier.json"
    _write_model(path)
    monkeypatch.setattr(qc, "MODEL_PATH", path)
    monkeypatch.setattr(qc, "_model", None)
    monkeypatch.setattr(llm, "embedding_model_name", lambda: "fake-8")
    return qc


def test_learned_tier_follows_the_weights(tiny_model):
    emb = [1.0] + [0.0] * 7
    tier, p = tiny_model.learned_tier("Compare the Fourier and Laplace transforms and explain why the ROC matters", emb)
    assert tier == 2 and p > 0.9
    tier, p = tiny_model.learned_tier("What is the Nyquist rate?", emb)
    assert tier == 1 and p < 0.1


def test_learned_tier_ignores_a_model_trained_on_another_embedding(tiny_model, monkeypatch):
    import src.utils.llm as llm

    monkeypatch.setattr(llm, "embedding_model_name", lambda: "some-other-model")
    assert tiny_model.learned_tier("What is the Nyquist rate?", [1.0] + [0.0] * 7) is None
    assert tiny_model.learned_tier("What is the Nyquist rate?", [1.0] * 5) is None  # wrong size


def test_missing_model_file_falls_back_to_rules(tmp_path, monkeypatch):
    import src.orchestrator.query_classifier as qc

    monkeypatch.setattr(qc, "MODEL_PATH", tmp_path / "nope.json")
    monkeypatch.setattr(qc, "_model", None)
    assert qc.learned_tier("anything", [0.1] * 8) is None


def test_shipped_model_matches_its_feature_layout():
    from src.orchestrator.query_classifier import MODEL_PATH
    from src.orchestrator.tier_features import CUE_NAMES

    m = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    assert len(m["coef"]) == m["dim"] + len(CUE_NAMES)


# ── Socratic policy ─────────────────────────────────────────────────────────

def test_new_problem_starts_with_a_first_hint():
    p = plan_socratic(PROBLEM, [])
    assert (p.stage, p.level) == ("hint", 1)


def test_hints_escalate_then_reveal_after_three():
    h = [U(PROBLEM), A("hint1")]
    assert plan_socratic("I'm stuck, I don't get it", h).level == 2
    h += [U("is it the integral of e^-2tau?"), A("hint2")]
    assert plan_socratic("so do I integrate from 0 to t?", h).level == 3
    h += [U("so do I integrate from 0 to t?"), A("hint3")]
    assert plan_socratic("still don't know", h).stage == "reveal"


def test_asking_for_the_answer_early_reveals():
    assert plan_socratic("just tell me the answer", [U(PROBLEM), A("hint1")]).stage == "reveal"


def test_definitions_and_direct_mode_are_not_hinted():
    assert plan_socratic("What is the Nyquist rate?", []).stage == "off"
    assert plan_socratic(PROBLEM, [], answer_mode="direct").stage == "off"
    assert plan_socratic("explain fourier transform", [], answer_mode="socratic").stage == "hint"


def test_a_new_explicit_problem_restarts_the_thread():
    h = [U(PROBLEM), A("h1"), U("ok"), A("h2")]
    p = plan_socratic("Now find the Laplace transform of t e^{-t}", h)
    assert (p.stage, p.level) == ("hint", 1)


# ── semantic cache ──────────────────────────────────────────────────────────

ANSWER = "The Nyquist rate is twice the highest frequency present in the signal; sampling below it causes aliasing."


def test_cache_hits_same_question_only_in_same_scope(sandbox):
    from src.cache.semantic_cache import SemanticCache, build_scope

    c = SemanticCache(sandbox / "c.sqlite")
    a, b = build_scope(course_id=1, mode="chat"), build_scope(course_id=2, mode="chat")
    c.store("What is the Nyquist rate?", a, ANSWER, citations=[{"source_file": "lec1"}], chunk_ids=["x1"])
    assert c.lookup("What is the Nyquist rate?", a)[0] is not None
    assert c.lookup("What is the Nyquist rate?", b)[0] is None, "another course must never see it"
    assert c.lookup("What is the Laplace transform?", a)[0] is None


def test_cache_refuses_error_text_and_supports_invalidation(sandbox):
    from src.cache.semantic_cache import SemanticCache, build_scope

    c = SemanticCache(sandbox / "c.sqlite")
    a = build_scope(course_id=1, mode="chat")
    assert not c.store("q?", a, "Something went wrong while generating a response. Please try again.")
    c.store("What is the Nyquist rate?", a, ANSWER)
    c.invalidate_course(1)
    assert c.lookup("What is the Nyquist rate?", a)[0] is None


@pytest.mark.parametrize("a,b,same", [
    ("Is the system x(t)=2x(t-1) causal?", "Is the system x(t)=2x(t+1) causal?", False),
    ("What is the ROC of a right-sided signal?", "What is the ROC of a left-sided signal?", False),
    ("What is the Nyquist rate?", "Define the Nyquist rate.", True),
    ("What is the Nyquist rate?", "what is the nyquist rate", True),
    ("Explain the sampling theorem", "Explain the sampling theorem in simple words", False),
    ("Find the Laplace transform of e^{-3t}u(t)", "Find the Laplace transform of e^{-2t}u(t)", False),
])
def test_cache_guard_separates_questions_that_differ_in_the_math(a, b, same):
    from src.cache.semantic_cache import lexically_compatible

    assert lexically_compatible(a, b) is same
