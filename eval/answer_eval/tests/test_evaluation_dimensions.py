"""Sprint 3: unit tests for the per-dimension judges and the clarification heuristic.

Run: python -m pytest eval/answer_eval/tests -q -p no:cacheprovider
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.answer_eval.abstention_detector import detect_abstention  # noqa: E402
from eval.answer_eval.citation_parser import parse_citations  # noqa: E402
from eval.answer_eval.evaluation_dimensions import (  # noqa: E402
    DETECTED_BEHAVIOURS, _accepts_assumption, citation_info, clarification_targets, classify_behaviour, detect_clarification,
    judge_behaviour, judge_completeness, judge_correctness, judge_groundedness, judge_language,
    match_key_points, resolve_material,
)
from eval.answer_eval.evaluation_result import (  # noqa: E402
    FAIL, INCONCLUSIVE, NOT_APPLICABLE, NOT_EVALUATED, PARTIAL, PASS, DimensionResult, overall_status,
)
from eval.answer_eval.schema import MATERIALS, KeyPoint, load_dataset  # noqa: E402

DATASET = Path(__file__).resolve().parents[1] / "dataset.jsonl"
ITEMS, _ERRORS = load_dataset(DATASET)
BY_ID = {i.id: i for i in ITEMS}
M1 = "EE210 Module 1 Signals and Sampling"
M2 = "EE210 Module 2 LTI Systems and Convolution"


def item(item_id):
    return BY_ID[item_id]


def completeness(item_id, text):
    it = item(item_id)
    return judge_completeness(it, match_key_points(it, text))


def detected(item_id, text):
    it = item(item_id)
    return classify_behaviour(detect_abstention(text), detect_clarification(it, text), it.expected_behavior)


def test_dataset_loads_cleanly():
    assert _ERRORS == [] and len(ITEMS) == 37


# ── Key-point matching ──────────────────────────────────────────────────────────────────────

def test_match_records_pattern_evidence_and_span():
    kp = match_key_points(item("mh-01"), "No: fs must exceed 18 kHz, so the tone appears at 3 kHz.")
    assert [m["id"] for m in kp["matched"]] == ["k1", "k2"]
    assert kp["matched"][0]["span"] == "18 kHz" and kp["matched"][0]["evidence"] == "pattern"
    assert kp["missing"] == [] and kp["unverifiable"] == []


def test_match_is_case_insensitive_like_the_validator():
    kp = match_key_points(item("mh-01"), "fs must exceed 18 KHZ; tone at 3 KHZ")
    assert len(kp["matched"]) == 2


def test_optional_key_points_reported_separately():
    kp = match_key_points(item("df-04"), "You get 7 samples, because L + M - 1.")
    assert [m["id"] for m in kp["matched"]] == ["k1"]
    assert kp["optional"] == [{"id": "k2", "text": item("df-04").key_points[1].text, "matched": True,
                               "span": "L + M - 1"}]


def test_required_point_without_regex_is_unverifiable():
    kp = match_key_points(item("df-08"), "It has len(x) + len(h) - 1 samples.")
    assert [u["id"] for u in kp["unverifiable"]] == ["k1"]
    assert [m["id"] for m in kp["matched"]] == ["k2"]


# ── Completeness ───────────────────────────────────────────────────────────────────────────

def test_completeness_pass_when_every_required_point_matches():
    r, w = completeness("mh-01", "Not satisfied: needs 18 kHz. It appears at 3 kHz.")
    assert r.status == PASS
    assert r.details["lower"] == r.details["upper"] == 100.0
    assert "pattern_match_is_not_semantic_confirmation" in w


def test_completeness_partial_with_counts():
    r, _ = completeness("mh-01", "It needs 18 kHz, so it is not satisfied.")
    assert r.status == PARTIAL
    assert (r.details["matched"], r.details["missing"], r.details["total"]) == (1, 1, 2)
    assert r.details["lower"] == 50.0 and r.details["upper"] == 50.0


def test_completeness_fail_when_nothing_matches():
    r, w = completeness("mh-01", "It is fine, the tone stays at 9 kHz.")
    assert r.status == FAIL and r.details["lower"] == 0.0
    assert "pattern_match_is_not_semantic_confirmation" not in w


def test_unverifiable_points_never_allow_pass():
    r, w = completeness("df-08", "np.convolve returns len(x) + len(h) - 1 samples of the linear convolution.")
    assert r.status == PARTIAL
    assert r.details["lower"] == 50.0 and r.details["upper"] == 100.0
    assert "unverified_key_points" in w


def test_completeness_not_evaluated_without_any_regex():
    r, _ = completeness("co-01", "Aliased components are mixed with real ones and cannot be separated.")
    assert r.status == NOT_EVALUATED and "no checkable key point" in r.reason
    assert r.details["lower"] is None and r.details["upper"] is None


def test_completeness_not_applicable_for_abstain_items():
    r, _ = completeness("ic-01", "The course materials do not cover this.")
    assert r.status == NOT_APPLICABLE


def test_optional_points_never_lower_completeness():
    r, _ = completeness("df-04", "The output has 7 samples.")
    assert r.status == PASS


def test_completeness_runs_on_normalised_text():
    from eval.answer_eval.response_normalizer import normalize_text
    raw = "Needs 18 kHz.\r\nAppears at 3 kHz."
    r, _ = completeness("mh-01", normalize_text(raw))
    assert r.status == PASS


def test_regex_false_positive_is_pinned_and_flagged():
    # Known limitation (design 5.2): "8 samples, not 7" still contains 7. The engine must not hide it.
    r, w = completeness("df-04", "You get 8 samples, not 7.")
    assert r.status == PASS
    assert "pattern_match_is_not_semantic_confirmation" in w


def test_unicode_minus_is_preserved_so_ascii_pattern_does_not_match():
    # The normaliser keeps U+2212; df-04's optional L + M - 1 pattern uses ASCII "-". Pinned behaviour.
    kp = match_key_points(item("df-04"), "7 samples, from L + M − 1.")
    assert kp["optional"][0]["matched"] is False


# ── Clarification heuristic (D2) ────────────────────────────────────────────────────────────

def test_targets_come_from_acceptable_interpretations():
    assert clarification_targets(item("am-01")) == ["signal"]
    assert clarification_targets(item("am-02")) == ["system"]
    assert clarification_targets(item("am-03")) == ["filter"]
    assert clarification_targets(item("df-01")) == []


@pytest.mark.parametrize("text", [
    "Which signal do you mean?",
    "Which signal are you referring to: cos(2 pi 50 t) or cos(pi n / 4)?",
    "Do you mean cos(2 pi 50 t) or cos(pi n / 4)?",
    "Are you asking about the continuous-time or the discrete-time signal?",
    "Could you clarify which signal you mean?",
    "Can you specify what signal you have in mind?",
    "Please clarify which signal you mean.",
    "Let me know which one you have in mind.",
    "Aap kaunsa signal ke baare mein pooch rahe ho?",
    "Kis signal ki baat kar rahe ho?",
    "What do you mean by the period here?",
])
def test_clarifying_questions_are_detected(text):
    c = detect_clarification(item("am-01"), text)
    assert c["asks"] and c["pattern"] == "clarifying_question" and c["method"] == "heuristic"


def test_information_request_needs_an_interpretation_target():
    asks = detect_clarification(item("am-04"), "Could you share the system's impulse response h[n]?")
    assert asks["asks"] and asks["pattern"] == "information_request" and asks["target_mentioned"]
    no_target = detect_clarification(item("am-04"), "Could you share your notes?")
    assert not no_target["asks"]


@pytest.mark.parametrize("text", [
    "What is the period? It is 20 ms for cos(2 pi 50 t).",                       # rhetorical
    "Why does aliasing happen? Because components above fs/2 fold back.",        # rhetorical
    "Would you like me to also explain the DFT?",                                 # offer
    "Do you want a worked example?",                                              # offer
    "Do you know why the Nyquist rate is 2B? It prevents overlap.",              # question to reader
    "Which signal has the larger period? The first one, 20 ms.",                 # rhetorical "which"
    "A student might ask \"which signal do you mean?\" here.",                    # quoted
    "Use `which signal do you mean?` as the prompt string.",                      # inline code
    "The period of cos(2 pi 50 t) is 20 ms.",                                     # plain answer
    "If you mean the DFT, see Module 3.",                                         # conditional, not a question
    "",
])
def test_non_clarifications_are_not_detected(text):
    assert not detect_clarification(item("am-01"), text)["asks"]


def test_assumption_detected_but_not_a_question():
    c = detect_clarification(item("am-01"), "Assuming you mean cos(2 pi 50 t), the period is 20 ms.")
    assert c["states_assumption"] and not c["asks"]
    assert c["assumption"].startswith("Assuming")


# ── Behaviour classification ────────────────────────────────────────────────────────────────

def test_detected_behaviour_vocabulary():
    assert set(DETECTED_BEHAVIOURS) == {"answer", "abstention", "refusal", "mixed", "empty",
                                        "clarification", "assumption"}


@pytest.mark.parametrize("item_id,text,expected", [
    ("df-01", "", "empty"),
    ("df-01", "   \n ", "empty"),
    ("df-01", "The Nyquist rate is 2B.", "answer"),
    ("df-01", "I couldn't find this in your course materials, so I won't guess.", "abstention"),
    ("ic-05", "I can only help with questions about your course.", "refusal"),
    ("df-01", "I couldn't find this in your course materials. However, the Nyquist rate is 2B, twice "
              "the highest frequency in the signal.", "mixed"),
    ("am-01", "Which signal do you mean?", "clarification"),
    ("am-01", "Assuming you mean cos(2 pi 50 t), the period is 20 ms.", "assumption"),
    ("df-01", "Assuming a band-limited signal, the Nyquist rate is 2B.", "answer"),  # not clarify item
    ("am-01", "I need more information: which signal do you mean?", "clarification"),
    ("am-01", "I can't tell which signal you mean. Could you clarify which one?", "clarification"),
    ("ic-02", "The Hamming window isn't covered in your course materials. Did you mean the Hann window?",
     "abstention"),
])
def test_classify_behaviour(item_id, text, expected):
    assert detected(item_id, text) == expected


# ── Behaviour compliance matrix ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("item_id,det,status,violation", [
    ("df-01", "answer", PASS, None),
    ("df-01", "abstention", FAIL, "false_refusal"),
    ("df-01", "refusal", FAIL, "false_refusal"),
    ("df-01", "mixed", PARTIAL, "answer_with_abstention"),
    ("df-01", "empty", FAIL, "empty_response"),
    ("df-01", "clarification", PARTIAL, "unnecessary_clarification"),
    ("ic-01", "answer", FAIL, "answered_uncovered_question"),
    ("ic-01", "abstention", PASS, None),
    ("ic-01", "refusal", PASS, None),
    ("ic-01", "mixed", PARTIAL, "content_beyond_abstention"),
    ("ic-01", "empty", FAIL, "empty_response"),
    ("ic-01", "clarification", FAIL, "no_explicit_abstention"),
    ("am-01", "clarification", PASS, None),
    ("am-01", "assumption", PASS, None),
    ("am-01", "answer", FAIL, "answered_without_clarifying"),
    ("am-04", "answer", FAIL, "answered_without_clarifying"),
    ("am-02", "answer", NOT_EVALUATED, None),
    ("am-01", "abstention", FAIL, "false_refusal"),
    ("am-01", "refusal", FAIL, "false_refusal"),
    ("am-01", "mixed", PARTIAL, "answer_with_abstention"),
    ("am-01", "empty", FAIL, "empty_response"),
])
def test_behaviour_matrix(item_id, det, status, violation):
    r, violations, _ = judge_behaviour(item(item_id), det)
    assert r.status == status
    assert r.details == {"expected": item(item_id).expected_behavior, "detected": det}
    assert violations == ([violation] if violation else [])


def test_scope_refusal_passes_with_warning_on_abstain_items():
    _, _, w = judge_behaviour(item("ic-06"), "refusal")
    assert "scope_refusal_not_inability" in w


def test_mixed_abstention_on_abstain_item_is_partial_with_warning():
    r, _, w = judge_behaviour(item("ic-02"), "mixed")
    assert r.status == PARTIAL and "content_beyond_abstention" in w


def test_assumption_without_accepting_interpretation_is_partial():
    # am-04 has no "Assumes ..." interpretation.
    r, v, _ = judge_behaviour(item("am-04"), "assumption")
    assert r.status == PARTIAL and v == ["assumed_without_asking"]


@pytest.mark.parametrize("item_id,accepted", [
    ("am-01", True), ("am-02", True), ("am-03", True), ("am-04", False),
])
def test_assumption_acceptance_per_clarify_item(item_id, accepted):
    assert _accepts_assumption(item(item_id)) is accepted
    r, v, w = judge_behaviour(item(item_id), "assumption")
    if accepted:
        assert r.status == PASS and v == [] and "assumption_content_not_verified" in w
    else:
        assert r.status == PARTIAL and v == ["assumed_without_asking"]
        assert "assumption_content_not_verified" not in w


@pytest.mark.parametrize("interpretation", [
    "Does not accept assumptions; ask the student instead",
    "Asks which signal is meant; making an assumption is not acceptable",
    "Never assumes a signal",
    "Asks first, then assumes nothing",
    "Reassumes the previous context",
])
def test_mentioning_assumptions_does_not_permit_one(interpretation):
    it = replace(item("am-01"), acceptable_interpretations=["Asks the student which signal they mean",
                                                            interpretation])
    assert _accepts_assumption(it) is False
    r, v, _ = judge_behaviour(it, "assumption")
    assert r.status == PARTIAL and v == ["assumed_without_asking"]


@pytest.mark.parametrize("interpretation", [
    "Assumes cos(2 pi 50 t) and answers T0 = 20 ms, saying so",
    "assumes cos(2 pi 50 t), saying so",
    "ASSUMES the RC filter, saying so",
    "   Assumes the moving average, saying so",
    "\tAssumes h[n] = (0.5)^n u[n], saying so",
])
def test_assumption_interpretation_is_case_insensitive_and_tolerates_leading_whitespace(interpretation):
    it = replace(item("am-04"), acceptable_interpretations=["Asks the student which system they mean",
                                                            interpretation])
    assert _accepts_assumption(it) is True
    assert judge_behaviour(it, "assumption")[0].status == PASS


def test_assumption_word_must_be_whole_word_at_start():
    it = replace(item("am-01"), acceptable_interpretations=["Assumesque wording", "Assumed signal"])
    assert _accepts_assumption(it) is False


def test_clarify_items_always_flag_the_heuristic():
    for det in ("clarification", "answer", "assumption"):
        _, _, w = judge_behaviour(item("am-01"), det)
        assert "clarification_detected_by_heuristic" in w


def test_assumption_content_is_flagged_unverified():
    _, _, w = judge_behaviour(item("am-01"), "assumption")
    assert "assumption_content_not_verified" in w


def test_unknown_detected_behaviour_raises():
    with pytest.raises(ValueError):
        judge_behaviour(item("df-01"), "assumption")


# ── Not-evaluated dimensions ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("item_id", [i.id for i in ITEMS])
def test_correctness_is_never_decided(item_id):
    r = judge_correctness(item(item_id))
    if item(item_id).expected_behavior == "answer":
        assert r.status == NOT_EVALUATED and r.reason
    else:
        assert r.status == NOT_APPLICABLE


@pytest.mark.parametrize("item_id", [i.id for i in ITEMS])
def test_language_is_never_decided(item_id):
    r = judge_language(item(item_id))
    assert r.status == NOT_EVALUATED
    assert r.details["expected"] == item(item_id).expected_answer_language


def test_groundedness_not_evaluated_with_or_without_chunks():
    assert judge_groundedness(item("df-01"), "answer", None).status == NOT_EVALUATED
    r = judge_groundedness(item("df-01"), "answer", [{"text": "x", "material": M1, "page": 3}])
    assert r.status == NOT_EVALUATED and r.details["retrieved_chunks_supplied"] == 1


def test_groundedness_not_applicable_only_for_clean_abstention():
    assert judge_groundedness(item("ic-01"), "abstention", None).status == NOT_APPLICABLE
    assert judge_groundedness(item("ic-01"), "refusal", None).status == NOT_APPLICABLE
    assert judge_groundedness(item("ic-01"), "mixed", None).status == NOT_EVALUATED
    assert judge_groundedness(item("ic-01"), "answer", None).status == NOT_EVALUATED


# ── Citation information ────────────────────────────────────────────────────────────────────

def test_resolve_material_accepts_titles_and_file_names():
    for title, file in MATERIALS.items():
        assert resolve_material(title) == title
        assert resolve_material(file) == title
        assert resolve_material(title.lower()) == title
    assert resolve_material("Some other book") is None and resolve_material(None) is None


def cinfo(item_id, text):
    return citation_info(item(item_id), parse_citations(text, known_titles=list(MATERIALS)))


def test_expected_page_label_is_recognised():
    c = cinfo("df-01", f"The Nyquist rate is 2B [{M1}, p. 3].")
    assert c["expected_source_cited"] is True and c["inline_count"] == 1
    assert c["cited_sources"][0]["material"] == M1 and c["cited_sources"][0]["page"] == 3


def test_wrong_page_is_reported_false():
    assert cinfo("df-01", f"The Nyquist rate is 2B [{M1}, p. 2].")["expected_source_cited"] is False


def test_source_label_with_file_name_resolves():
    c = cinfo("df-01", "2B [1]\n\nSources:\n[1] [Source: EE210_Module_1_Signals_and_Sampling.pdf, p.3]")
    assert c["expected_source_cited"] is True and c["sources_block"] is True


def test_numeric_only_is_unknown():
    c = cinfo("df-01", "The Nyquist rate is 2B [1].")
    assert c["expected_source_cited"] is None and c["counts_by_kind"] == {"numeric": 1}


def test_no_citations():
    c = cinfo("df-01", "The Nyquist rate is 2B.")
    assert c["total_count"] == 0 and c["expected_source_cited"] is None and c["malformed"] == []


def test_title_only_label_is_unknown():
    assert cinfo("df-01", f"2B [{M1}]")["expected_source_cited"] is None


def test_citation_on_abstain_item_is_unexpected():
    assert cinfo("ic-01", f"Not covered [{M2}, p. 1].")["expected_source_cited"] is False


# ── Overall status rule (D3) ────────────────────────────────────────────────────────────────

def dims(**statuses):
    return {k: DimensionResult(v, "r") for k, v in statuses.items()}


@pytest.mark.parametrize("statuses,expected", [
    ({"a": PASS, "b": PASS}, PASS),
    ({"a": PASS, "b": NOT_APPLICABLE}, PASS),
    ({"a": PASS, "b": NOT_EVALUATED}, INCONCLUSIVE),
    ({"a": PASS, "b": PARTIAL, "c": NOT_EVALUATED}, PARTIAL),
    ({"a": FAIL, "b": PARTIAL, "c": PASS}, FAIL),
    ({"a": FAIL, "b": NOT_EVALUATED}, FAIL),
    ({"a": NOT_APPLICABLE}, INCONCLUSIVE),
    ({"a": NOT_EVALUATED}, INCONCLUSIVE),
])
def test_overall_status(statuses, expected):
    assert overall_status(dims(**statuses)) == expected


def test_dimension_result_validation():
    with pytest.raises(ValueError):
        DimensionResult("maybe", "x")
    with pytest.raises(ValueError):
        DimensionResult(NOT_EVALUATED, "")


def test_judges_do_not_mutate_items():
    it = item("mh-01")
    before = (list(it.key_points), list(it.forbidden_claims), list(it.must_not))
    judge_completeness(it, match_key_points(it, "18 kHz, 3 kHz"))
    judge_behaviour(it, "answer")
    detect_clarification(it, "Which tone do you mean?")
    assert (list(it.key_points), list(it.forbidden_claims), list(it.must_not)) == before


def test_completeness_with_constructed_item_without_required_points():
    it = replace(item("df-04"), key_points=[KeyPoint("k9", "optional only", False, (), r"7")])
    r, _ = judge_completeness(it, match_key_points(it, "7"))
    assert r.status == NOT_APPLICABLE
