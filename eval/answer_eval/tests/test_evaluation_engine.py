"""Sprint 3: integration, batch, CLI, edge-case and regression tests for the evaluation engine.

Run: python -m pytest eval/answer_eval/tests -q -p no:cacheprovider
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.answer_eval import run_evaluation  # noqa: E402
from eval.answer_eval.evaluation_engine import (  # noqa: E402
    APPLICABILITY, InvalidDatasetError, evaluate_batch, evaluate_response,
)
from eval.answer_eval.evaluation_result import (  # noqa: E402
    DIMENSION_STATUSES, DIMENSIONS, OVERALL_STATUSES,
)
from eval.answer_eval.schema import load_dataset  # noqa: E402

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
DATASET = PKG / "dataset.jsonl"
FIXTURE = HERE / "fixtures" / "responses_sample.jsonl"
ITEMS, _ = load_dataset(DATASET)
BY_ID = {i.id: i for i in ITEMS}
M1 = "EE210 Module 1 Signals and Sampling"
SAARTHI_REFUSAL = ("I couldn't find this in your course materials, so I won't guess. Try rephrasing with "
                   "terms from your lectures, pick the right course, or ask your instructor to upload "
                   "material on it.")

RESULT_KEYS = {
    "item_id", "category", "label_status", "provisional", "response", "normalized_response",
    "expected_behavior", "detected_behavior", "expected_answer_language", "abstention", "clarification",
    "key_points", "forbidden_claims", "behavioural_violations", "must_not_to_check", "citations",
    "dimensions", "overall_status", "warnings", "limitations", "metadata",
}


def ev(item_id, text, **kw):
    return evaluate_response(BY_ID[item_id], text, **kw)


# ── End to end, one response ────────────────────────────────────────────────────────────────

def test_good_answer_is_inconclusive_not_pass():
    r = ev("df-01", f"The Nyquist rate is 2B, twice the highest frequency [{M1}, p. 3].")
    s = {k: v.status for k, v in r.dimensions.items()}
    assert s == {"behaviour": "pass", "completeness": "pass", "correctness": "not_evaluated",
                 "groundedness": "not_evaluated", "language": "not_evaluated"}
    assert r.overall_status == "inconclusive"
    assert r.citations["expected_source_cited"] is True


def test_citations_never_change_groundedness():
    with_cite = ev("df-01", f"The Nyquist rate is 2B [{M1}, p. 3].")
    without = ev("df-01", "The Nyquist rate is 2B.")
    assert with_cite.dimensions["groundedness"] == without.dimensions["groundedness"]
    assert with_cite.overall_status == without.overall_status


def test_regex_hits_never_change_correctness():
    hit = ev("mh-01", "Needs 18 kHz; appears at 3 kHz.")
    miss = ev("mh-01", "It is fine.")
    assert hit.dimensions["correctness"] == miss.dimensions["correctness"]
    assert hit.dimensions["correctness"].status == "not_evaluated"


def test_hinglish_answer_language_is_not_evaluated():
    r = ev("ml-02", "Nahi, 18 kHz chahiye. Tone 3 kHz pe dikhega.")
    assert r.dimensions["language"].status == "not_evaluated"
    assert r.expected_answer_language == "hinglish"
    assert r.dimensions["completeness"].status == "pass"


def test_partial_answer():
    r = ev("mh-01", "Not satisfied, it needs 18 kHz.")
    assert r.dimensions["completeness"].status == "partial" and r.overall_status == "partial"
    assert [m["id"] for m in r.key_points["missing"]] == ["k2"]


def test_false_refusal_on_answer_item_fails():
    r = ev("df-01", SAARTHI_REFUSAL)
    assert r.detected_behavior == "abstention" and r.overall_status == "fail"
    assert r.behavioural_violations == ["false_refusal"]


def test_saarthi_refusal_passes_behaviour_on_abstain_item():
    r = ev("ic-02", SAARTHI_REFUSAL)
    assert r.dimensions["behaviour"].status == "pass"
    assert r.dimensions["groundedness"].status == "not_applicable"
    assert r.overall_status == "inconclusive"  # language still not evaluated (D3)


def test_saarthi_not_covered_from_app_source():
    from src.prompts.tutor import not_covered  # read-only import, as in Sprint 2
    r = ev("ic-01", not_covered())
    assert r.detected_behavior == "abstention" and r.dimensions["behaviour"].status == "pass"


def test_mixed_abstention_on_abstain_item():
    r = ev("ic-03", "I couldn't find this in your course materials. In general the ROC of e^{-2t} u(t) "
                    "is Re(s) > -2 because the integral converges there.")
    assert r.detected_behavior == "mixed" and r.dimensions["behaviour"].status == "partial"
    assert "content_beyond_abstention" in r.warnings
    assert r.dimensions["groundedness"].status == "not_evaluated"


def test_scope_refusal_on_abstain_item():
    r = ev("ic-05", "I can only help with questions about your course.")
    assert r.detected_behavior == "refusal" and r.dimensions["behaviour"].status == "pass"
    assert "scope_refusal_not_inability" in r.warnings


def test_answering_uncovered_question_fails():
    r = ev("ic-05", "France won the 2018 football world cup.")
    assert r.overall_status == "fail" and r.behavioural_violations == ["answered_uncovered_question"]


def test_clarify_item_clarification():
    r = ev("am-01", "Which signal do you mean: cos(2 pi 50 t) or cos(pi n / 4)?")
    assert r.detected_behavior == "clarification" and r.dimensions["behaviour"].status == "pass"
    assert r.clarification["method"] == "heuristic" and r.clarification["target_mentioned"]
    assert r.overall_status == "inconclusive"
    assert any("heuristic" in lim for lim in r.limitations)


def test_clarify_item_stated_assumption():
    r = ev("am-01", "Assuming you mean cos(2 pi 50 t), the period is T0 = 20 ms.")
    assert r.detected_behavior == "assumption" and r.dimensions["behaviour"].status == "pass"
    assert "assumption_content_not_verified" in r.warnings


def test_clarify_item_plain_answer_fails():
    r = ev("am-04", "Yes, it is stable.")
    assert r.overall_status == "fail" and r.behavioural_violations == ["answered_without_clarifying"]


def test_clarify_item_general_answer_not_evaluated():
    r = ev("am-02", "The step response s[n] is the running sum of h[k] for k <= n.")
    assert r.dimensions["behaviour"].status == "not_evaluated"
    assert r.overall_status == "inconclusive"


def test_unnecessary_clarification_on_answer_item():
    r = ev("df-01", "Do you mean the Nyquist rate or the Nyquist frequency?")
    assert r.dimensions["behaviour"].status == "partial"
    assert r.behavioural_violations == ["unnecessary_clarification"]


def test_applicability_matrix_is_respected_for_every_item():
    for it in ITEMS:
        r = evaluate_response(it, it.reference_answer)
        for name in DIMENSIONS:
            if name not in APPLICABILITY[it.expected_behavior]:
                assert r.dimensions[name].status == "not_applicable", (it.id, name)


# ── Output structure ────────────────────────────────────────────────────────────────────────

def test_result_dict_has_every_requested_field_and_is_json_safe():
    r = ev("df-01", "The Nyquist rate is 2B.", metadata={"engine": "A", "latency_ms": 12}).to_dict()
    assert set(r) == RESULT_KEYS
    assert set(r["dimensions"]) == set(DIMENSIONS)
    for d in r["dimensions"].values():
        assert d["status"] in DIMENSION_STATUSES and isinstance(d["reason"], str)
    assert r["overall_status"] in OVERALL_STATUSES
    assert r["metadata"] == {"engine": "A", "latency_ms": 12}
    json.dumps(r)


def test_forbidden_claims_are_listed_but_never_marked_detected():
    r = ev("mh-01", "It appears at 9 kHz.").to_dict()
    assert r["forbidden_claims"]["detection"] == "not_evaluated"
    assert r["forbidden_claims"]["detected"] is None
    assert r["forbidden_claims"]["claims_to_check"] == [
        {"text": c.text, "reason": c.reason} for c in BY_ID["mh-01"].forbidden_claims]


def test_must_not_is_echoed_for_review():
    assert ev("ic-02", SAARTHI_REFUSAL).must_not_to_check == BY_ID["ic-02"].must_not


def test_original_and_normalised_response_both_kept():
    r = ev("df-01", "The Nyquist rate is 2B.\r\n")
    assert r.response == "The Nyquist rate is 2B.\r\n"
    assert r.normalized_response == "The Nyquist rate is 2B."


@pytest.mark.parametrize("item_id", [i.id for i in ITEMS])
def test_every_draft_item_is_provisional(item_id):
    r = ev(item_id, "anything")
    assert r.label_status == "draft" and r.provisional is True and "draft_label" in r.warnings


def test_reviewed_item_is_not_provisional():
    reviewed = replace(BY_ID["df-01"], label_status="reviewed", reviewer="someone")
    r = evaluate_response(reviewed, "The Nyquist rate is 2B.")
    assert r.provisional is False and "draft_label" not in r.warnings


def test_limitations_name_every_not_evaluated_dimension():
    r = ev("co-01", "Aliasing mixes components.")
    joined = " ".join(r.limitations)
    for name in ("correctness", "groundedness", "language", "completeness"):
        assert name in joined


def test_deterministic_output():
    a = json.dumps(ev("mh-01", "18 kHz [1]").to_dict(), sort_keys=True)
    b = json.dumps(ev("mh-01", "18 kHz [1]").to_dict(), sort_keys=True)
    assert a == b


# ── Edge inputs ─────────────────────────────────────────────────────────────────────────────

def test_none_response_is_not_evaluated_not_failed():
    r = ev("df-01", None)
    assert r.overall_status == "not_evaluated" and r.detected_behavior is None
    assert all(d.status == "not_evaluated" for d in r.dimensions.values())
    assert r.limitations == ["no_response"]


@pytest.mark.parametrize("text", ["", "   ", "\n\t\n", "​"])
def test_empty_or_whitespace_response_fails(text):
    r = ev("df-01", text)
    assert r.detected_behavior == "empty" and r.overall_status == "fail"
    assert r.behavioural_violations == ["empty_response"]


@pytest.mark.parametrize("bad", [1, 2.5, b"bytes", ["list"]])
def test_non_string_response_raises(bad):
    with pytest.raises(TypeError):
        ev("df-01", bad)


def test_very_long_response():
    text = ("Filler sentence about signals. " * 5000) + "It needs 18 kHz and appears at 3 kHz."
    r = ev("mh-01", text)
    assert r.dimensions["completeness"].status == "pass"


def test_code_fence_preserved_and_matched():
    r = ev("df-08", "```python\ny = np.convolve(x, h)\nn = len(x) + len(h) - 1\n```")
    assert "```python" in r.normalized_response
    assert [m["id"] for m in r.key_points["matched"]] == ["k2"]


def test_retrieved_chunks_do_not_change_results():
    chunks = [{"material": M1, "page": 3, "text": "The Nyquist rate is 2B."}]
    a = ev("df-01", "The Nyquist rate is 2B.").to_dict()
    b = ev("df-01", "The Nyquist rate is 2B.", retrieved_chunks=chunks).to_dict()
    assert a["overall_status"] == b["overall_status"]
    assert b["dimensions"]["groundedness"]["status"] == "not_evaluated"
    assert b["dimensions"]["groundedness"]["details"]["retrieved_chunks_supplied"] == 1


def test_malformed_citation_warning():
    r = ev("df-01", "The Nyquist rate is 2B [Source: ].")
    assert len(r.citations["malformed"]) == 1 and "malformed_citations" in r.warnings
    assert "malformed_citations" not in ev("df-01", "The Nyquist rate is 2B.").warnings


# ── Batch ───────────────────────────────────────────────────────────────────────────────────

def test_batch_counts_add_up_and_missing_are_not_failures():
    rep = evaluate_batch(ITEMS, {"df-01": "The Nyquist rate is 2B.", "ic-01": SAARTHI_REFUSAL})
    d = rep.to_dict()
    assert sum(d["overall_status_counts"].values()) == 37
    assert d["overall_status_counts"]["not_evaluated"] == 35
    assert d["overall_status_counts"]["fail"] == 0
    assert len(d["not_responded"]) == 35
    for counts in d["dimension_status_counts"].values():
        assert sum(counts.values()) == 37
    assert [r["item_id"] for r in d["results"]] == [i.id for i in ITEMS]
    json.dumps(d)


def test_batch_groups():
    d = evaluate_batch(ITEMS, {}).to_dict()
    assert {b: g["items"] for b, g in d["by_expected_behavior"].items()} == \
        {"answer": 27, "abstain": 6, "clarify": 4}
    assert sum(g["items"] for g in d["by_category"].values()) == 37
    assert d["provisional"] is True and d["dataset"]["draft"] == 37


def test_batch_unknown_and_duplicate_ids():
    pairs = [("df-01", "2B"), ("df-01", "twice B"), ("xx-99", "?"), ("df-02", 5), ("df-03", None)]
    d = evaluate_batch(ITEMS, pairs).to_dict()
    errs = {(e["id"], e["error"].split(";")[0]) for e in d["input_errors"]}
    assert ("df-01", "duplicate response for item id") in errs
    assert ("xx-99", "unknown item id") in errs
    assert any(e["id"] == "df-02" for e in d["input_errors"])
    assert {"df-01", "df-02", "df-03"} <= set(d["not_responded"])


def test_batch_reviewed_only_filters_everything_today():
    d = evaluate_batch(ITEMS, {"df-01": "2B"}, reviewed_only=True).to_dict()
    assert d["results"] == [] and len(d["excluded_by_filter"]) == 37
    assert d["dataset"]["items"] == 0 and d["provisional"] is True


def test_batch_reviewed_only_keeps_reviewed_items():
    items = [replace(BY_ID["df-01"], label_status="reviewed", reviewer="r")] + \
        [i for i in ITEMS if i.id != "df-01"]
    d = evaluate_batch(items, {"df-01": "2B"}, reviewed_only=True).to_dict()
    assert [r["item_id"] for r in d["results"]] == ["df-01"] and d["provisional"] is False


def test_batch_rejects_invalid_items():
    bad = replace(BY_ID["df-01"], expected_behavior="guess")
    with pytest.raises(InvalidDatasetError):
        evaluate_batch([bad], {})
    with pytest.raises(InvalidDatasetError):
        evaluate_batch([BY_ID["df-01"], BY_ID["df-01"]], {})


def test_batch_metadata_and_chunks_pass_through():
    d = evaluate_batch(ITEMS, {"df-01": "2B"}, metadata={"df-01": {"engine": "A"}},
                       retrieved_chunks={"df-01": [{"text": "x"}]}).to_dict()
    r = d["results"][0]
    assert r["metadata"] == {"engine": "A"}
    assert r["dimensions"]["groundedness"]["details"]["retrieved_chunks_supplied"] == 1


# ── Regression ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("it", [i for i in ITEMS if i.expected_behavior == "answer"], ids=lambda i: i.id)
def test_reference_answers_never_fail_completeness(it):
    r = evaluate_response(it, it.reference_answer)
    assert r.dimensions["completeness"].status in ("pass", "partial", "not_evaluated")
    assert r.dimensions["behaviour"].status == "pass"
    assert r.key_points["missing"] == []


def test_reference_answer_completeness_counts_match_dataset_profile():
    statuses = [evaluate_response(i, i.reference_answer).dimensions["completeness"].status
                for i in ITEMS if i.expected_behavior == "answer"]
    assert statuses.count("pass") == 10 and statuses.count("not_evaluated") == 9
    assert statuses.count("partial") == 8


def test_no_item_can_reach_pass_in_sprint_3():
    for it in ITEMS:
        assert evaluate_response(it, it.reference_answer).overall_status != "pass"


SPRINT_1_2_FILES = {
    "__init__.py": "234c81b9f9d350e9f21e6f0ec117198620209a4f",
    "abstention_detector.py": "6fd79f9a3d8d053107ce10de5e45da8de83e04e6",
    "archive/README.md": "5ab2cc0d947652f8ca94cfca55507364acd7a500",
    "archive/removed_hi_mr_2026-11-23.jsonl": "876b5f6cdd82e3d6bd4a99e53439127d311b05da",
    "citation_parser.py": "d3cf2583c32e22d5e1ca8f04215107fe6a674bbd",
    "dataset.jsonl": "0414b04d7f97d264cf688cf88b81da0eb35d9580",
    "response_normalizer.py": "ed8a71ca7f44a9ce42e3382fec40094c896f2f1e",
    "schema.py": "cf53b206016a62deb8e5bd0a961c37f0f119d987",
    "validate_dataset.py": "371179eb9afd1747c72f7b6ebe10fa6a35404945",
}


@pytest.mark.parametrize("rel,blob", sorted(SPRINT_1_2_FILES.items()))
def test_sprint_1_and_2_files_unchanged(rel, blob):
    """Git blob ids of the files committed in 39328a1 (line endings normalised to LF)."""
    data = (PKG / rel).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == blob


# ── CLI ─────────────────────────────────────────────────────────────────────────────────────

def test_cli_writes_report(tmp_path, capsys):
    out = tmp_path / "report.json"
    assert run_evaluation.main(["--responses", str(FIXTURE), "--out", str(out)]) == 0
    d = json.loads(out.read_text(encoding="utf-8"))
    by = {r["item_id"]: r for r in d["results"]}
    assert by["df-01"]["overall_status"] == "inconclusive"
    assert by["cm-03"]["overall_status"] == "fail"
    assert by["am-04"]["overall_status"] == "fail"
    assert by["ic-03"]["dimensions"]["behaviour"]["status"] == "partial"
    assert by["ml-02"]["dimensions"]["groundedness"]["details"]["retrieved_chunks_supplied"] == 1
    assert by["df-01"]["metadata"] == {"engine": "synthetic-fixture"}
    assert d["provisional"] is True and d["input_errors"] == []
    assert "PROVISIONAL" in capsys.readouterr().out


def test_cli_reports_bad_lines_and_unknown_ids(tmp_path):
    resp = tmp_path / "r.jsonl"
    resp.write_text('{"id": "df-01", "response": "2B"}\nnot json\n{"id": "zz-01", "response": "x"}\n'
                    '{"response": "no id"}\n', encoding="utf-8")
    out = tmp_path / "o.json"
    assert run_evaluation.main(["--responses", str(resp), "--out", str(out)]) == 0
    errs = json.loads(out.read_text(encoding="utf-8"))["input_errors"]
    assert {e.get("line") for e in errs if "line" in e} == {2, 4}
    assert any(e.get("id") == "zz-01" for e in errs)


def test_cli_missing_responses_file(tmp_path):
    assert run_evaluation.main(["--responses", str(tmp_path / "missing.jsonl")]) == 1


def test_cli_refuses_invalid_dataset(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(DATASET.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    assert run_evaluation.main(["--dataset", str(bad), "--responses", str(FIXTURE)]) == 2
