"""Sprint 5 tests: benchmark case schema, Sprint 1 derivation, recorded-result format."""
from __future__ import annotations

import copy
import datetime as dt
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.answer_eval.schema import load_dataset  # noqa: E402
from eval.retrieval_eval import cases as C  # noqa: E402
from eval.retrieval_eval import records as RC  # noqa: E402

DATASET = ROOT / "eval" / "answer_eval" / "dataset.jsonl"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ITEMS, _ = load_dataset(DATASET)
TODAY = dt.date.today()


def lab(grades=None, status="fixture", completeness="exhaustive", **extra):
    out = {"status": status, "completeness": completeness, "grades": {"d1": 1} if grades is None else grades,
           "provenance": "test"}
    out.update(extra)
    return out


def case(**over):
    raw = {"id": "fx-01", "question": "q?", "language": "en", "category": "c", "source": "fixture",
           "labels": {"document": lab(), "page": None}}
    raw.update(over)
    return raw


def parse(raw):
    return C.parse_case(raw, "t", today=TODAY)


# ── Case schema ─────────────────────────────────────────────────────────────────────────────

def test_valid_case_parses_and_missing_levels_have_no_labels():
    c, errs = parse(case())
    assert errs == [] and set(c.labels) == {"document"}
    assert c.to_dict()["labels"]["chunk"] is None and c.to_dict()["labels"]["page"] is None


@pytest.mark.parametrize("over,needle", [
    ({"id": "bad id"}, "id must match"),
    ({"language": "hi"}, "language must be one of"),
    ({"language": "mr"}, "language must be one of"),
    ({"source": "web"}, "source must be one of"),
    ({"question": " "}, "question must be"),
    ({"labels": {"sentence": lab()}}, "unknown label level"),
    ({"labels": []}, "labels must be an object"),
    ({"metadata": []}, "metadata must be an object"),
    ({"extra": 1}, "unknown field"),
])
def test_invalid_case_fields(over, needle):
    c, errs = parse(case(**over))
    assert c is None and any(needle in e for e in errs), errs


def test_missing_case_field():
    raw = case()
    del raw["question"]
    assert any("missing field" in e for e in parse(raw)[1])


@pytest.mark.parametrize("labels,needle", [
    (lab(grades={"d1": 4}), "grade for 'd1' must be an integer 0..3"),
    (lab(grades={"d1": -1}), "must be an integer 0..3"),
    (lab(grades={"d1": True}), "must be an integer 0..3"),
    (lab(grades={"d1": 1.0}), "must be an integer 0..3"),
    (lab(grades=[]), "grades must be an object"),
    (lab(status="approved"), "status must be one of"),
    (lab(completeness="full"), "completeness must be one of"),
    ({"status": "fixture", "grades": {}}, "missing field"),
    (lab(colour="red"), "unknown field"),
])
def test_invalid_labels(labels, needle):
    c, errs = parse(case(labels={"document": labels}))
    assert c is None and any(needle in e for e in errs), errs


def test_page_ids_must_name_a_page():
    errs = parse(case(labels={"page": lab(grades={"doc-1 page 3": 1})}))[1]
    assert any("must look like '<document id>#p<page>'" in e for e in errs)
    assert parse(case(labels={"page": lab(grades={"doc-1#p3": 1})}))[1] == []


def test_fixture_labels_cannot_mix_with_ground_truth():
    assert any("fixture labels may only appear" in e
               for e in parse(case(source="manual", labels={"document": lab()}))[1])
    errs = parse(case(labels={"document": lab(status="unverified", completeness="partial")}))[1]
    assert any("fixture case may only carry fixture labels" in e for e in errs)


def test_unverified_labels_cannot_claim_exhaustive_completeness():
    errs = parse(case(source="manual", labels={"document": lab(status="unverified")}))[1]
    assert any("cannot be exhaustive" in e for e in errs)
    assert parse(case(source="manual", labels={"document": lab(status="unverified", completeness="partial")}))[1] == []


def test_verified_labels_need_a_reviewer_and_a_past_date():
    base = dict(status="verified", completeness="exhaustive")
    assert any("need a reviewer" in e for e in parse(case(source="manual", labels={"document": lab(**base, reviewed_on=TODAY.isoformat())}))[1])
    assert any("ISO date" in e for e in parse(case(source="manual", labels={"document": lab(**base, reviewer="kartik", reviewed_on="25-11-2026")}))[1])
    future = (TODAY + dt.timedelta(days=2)).isoformat()
    assert any("in the future" in e for e in parse(case(source="manual", labels={"document": lab(**base, reviewer="kartik", reviewed_on=future)}))[1])
    ok = lab(**base, reviewer="kartik", reviewed_on=(TODAY - dt.timedelta(days=1)).isoformat())
    assert parse(case(source="manual", labels={"document": ok}))[1] == []
    assert any("only allowed on verified" in e for e in parse(case(labels={"document": lab(reviewer="kartik")}))[1])


def test_duplicate_case_ids_and_invalid_json(tmp_path):
    p = tmp_path / "cases.jsonl"
    p.write_text(json.dumps(case()) + "\n\n{not json\n" + json.dumps(case()) + "\n", encoding="utf-8")
    cases, errs = C.load_cases(p, today=TODAY)
    assert len(cases) == 1
    assert any("invalid JSON" in e for e in errs) and any("duplicate case id 'fx-01'" in e for e in errs)


def test_case_round_trip_and_read_only():
    c, _ = parse(case(metadata={"k": 1}))
    again, errs = parse(json.loads(json.dumps(c.to_dict())))
    assert errs == [] and again == c
    with pytest.raises(TypeError):
        c.labels["document"].grades["x"] = 1


def test_illustrative_fixture_cases_are_valid_and_marked_as_fixtures():
    cases, errs = C.load_cases(FIXTURES / "illustrative_cases.jsonl", today=TODAY)
    assert errs == [] and len(cases) == 4
    assert all(c.source == "fixture" and all(l.status == "fixture" for l in c.labels.values()) for c in cases)
    assert {c.language for c in cases} == {"en", "hinglish"}


# ── Sprint 1 derivation ─────────────────────────────────────────────────────────────────────

DERIVED = C.cases_from_answer_items(ITEMS)
BY_ID = {c.id: c for c in DERIVED}


def test_derivation_covers_every_item_and_keeps_language_scope():
    assert [c.id for c in DERIVED] == [i.id for i in ITEMS] and len(DERIVED) == 37
    assert {c.language for c in DERIVED} <= {"en", "hinglish"}
    assert all(c.source == "sprint1_dataset" for c in DERIVED)


def test_derived_labels_are_unverified_partial_and_match_the_evidence_exactly():
    for item in ITEMS:
        c = BY_ID[item.id]
        assert set(c.labels) == {"document", "page"}            # no chunk labels are invented
        for lab_ in c.labels.values():
            assert (lab_.status, lab_.completeness) == ("unverified", "partial")
        assert dict(c.labels["page"].grades) == {f"{e.material}#p{e.page}": 1 for e in item.evidence}
        assert dict(c.labels["document"].grades) == {e.material: 1 for e in item.evidence}
        assert c.question == item.query and c.category == item.category


def test_derived_page_labels_equal_the_validated_expected_sources():
    for item in ITEMS:
        expected = {f"{s['material']}#p{s['page']}" for s in item.expected_sources}
        assert set(BY_ID[item.id].labels["page"].grades) == expected


def test_abstain_items_have_no_relevant_labels():
    abstain = [i for i in ITEMS if i.expected_behavior == "abstain"]
    assert len(abstain) == 6
    for item in abstain:
        assert all(len(l.grades) == 0 for l in BY_ID[item.id].labels.values())


def test_derived_cases_pass_the_case_validator():
    rows = [(c.id, json.loads(json.dumps(c.to_dict()))) for c in DERIVED]
    parsed, errs = C.parse_cases(rows, today=TODAY)
    assert errs == [] and parsed == DERIVED


def test_derivation_does_not_modify_items_or_leak_answers():
    before = copy.deepcopy(ITEMS)
    cases = C.cases_from_answer_items(ITEMS)
    assert ITEMS == before
    for c, item in zip(cases, ITEMS):
        blob = json.dumps(c.to_dict()["metadata"], ensure_ascii=False)
        assert item.reference_answer not in blob
        assert all(e.quote not in blob for e in item.evidence)


def test_export_round_trip(tmp_path):
    p = tmp_path / "derived.jsonl"
    C.write_cases(DERIVED, p)
    loaded, errs = C.load_cases(p, today=TODAY)
    assert errs == [] and loaded == DERIVED


# ── Recorded results ────────────────────────────────────────────────────────────────────────

RUN = {"record_type": "run", "run_id": "r1", "provenance": "recorded"}


def records(*rows):
    return RC.parse_records([(f"l{i}", r) for i, r in enumerate(rows, 1)])


def res(**over):
    out = {"record_type": "result", "run_id": "r1", "case_id": "fx-01",
           "hits": [{"document_id": "d1", "page": 2, "chunk_id": "c1", "score": 0.5}]}
    out.update(over)
    return out


def test_valid_records_and_level_rankings():
    rs = records(RUN, res(hits=[{"document_id": "d1", "page": 2, "chunk_id": "c1"},
                                {"document_id": None, "page": 1, "chunk_id": "c2"},
                                {"document_id": "d1", "chunk_id": "c3"}]))
    assert rs.errors == []
    r = rs.results[("r1", "fx-01")]
    assert r.ranking("document") == ("d1", None, "d1")
    assert r.ranking("page") == ("d1#p2", None, None)
    assert r.ranking("chunk") == ("c1", "c2", "c3")


def test_levels_without_information_are_unavailable_not_empty():
    r = records(RUN, res(hits=[{"document_id": "d1"}])).results[("r1", "fx-01")]
    assert r.ranking("page") is None and r.ranking("chunk") is None
    empty = records(RUN, res(hits=[])).results[("r1", "fx-01")]
    assert empty.ranking("document") == () and empty.ranking("page") == () and empty.ranking("chunk") == ()


def test_independent_id_lists():
    r = records(RUN, res(hits=None, **{"document_ids": ["d1", "d2"], "chunk_ids": ["c9"]}))
    assert r.errors[0].endswith("give either hits or document_ids/chunk_ids, not both")  # a 'hits' key counts
    rs = records(RUN, {"record_type": "result", "run_id": "r1", "case_id": "fx-01",
                       "document_ids": ["d1", "d2"]})
    got = rs.results[("r1", "fx-01")]
    assert got.ranking("document") == ("d1", "d2") and got.ranking("chunk") is None and got.ranking("page") is None


@pytest.mark.parametrize("row,needle", [
    ({"record_type": "result", "run_id": "r1", "case_id": "fx-01"}, "needs hits, document_ids or chunk_ids"),
    (res(hits=[{"page": 1}]), "needs document_id"),
    (res(hits=[{"document_id": ""}]), "document_id must be a non-empty string"),
    (res(hits=[{"document_id": "d", "page": 0}]), "page must be an integer >= 1"),
    (res(hits=[{"document_id": "d", "page": "3"}]), "page must be an integer >= 1"),
    (res(hits=[{"document_id": "d", "score": "high"}]), "score must be a finite number"),
    (res(hits=[{"document_id": "d", "score": float("nan")}]), "score must be a finite number"),
    (res(hits=[{"document_id": "d", "rank": 1}]), "unknown hit field"),
    (res(hits=["d1"]), "a hit must be an object"),
    (res(document_ids=["d1"]), "either hits or document_ids/chunk_ids"),
    (res(error="boom"), "must not also carry retrieved items"),
    (res(run_id="r2"), "has no run header"),
    (res(colour="x"), "unknown field"),
    ({"record_type": "score"}, "record_type must be"),
    ({"record_type": "run", "run_id": "r9", "provenance": "live"}, "provenance must be one of"),
    ({"record_type": "run", "run_id": "bad id", "provenance": "fixture"}, "run_id must match"),
])
def test_malformed_records_are_reported_and_skipped(row, needle):
    rs = records(RUN, row)
    assert any(needle in e for e in rs.errors), rs.errors
    assert rs.results == {}            # the malformed record is skipped, never half-used


def test_duplicate_results_and_headers():
    rs = records(RUN, res(), res(hits=[]), RUN)
    assert any("duplicate result" in e and "kept the first" in e for e in rs.errors)
    assert any("duplicate run header" in e for e in rs.errors)
    assert len(rs.results[("r1", "fx-01")].hits) == 1


def test_error_results_and_header_after_results():
    rs = records({"record_type": "result", "run_id": "r1", "case_id": "fx-01", "error": "TimeoutError: slow"}, RUN)
    assert rs.errors == [] and rs.results[("r1", "fx-01")].error == "TimeoutError: slow"


def test_load_records_reports_line_numbers(tmp_path):
    p = tmp_path / "runs.jsonl"
    p.write_text(json.dumps(RUN) + "\n{oops\n" + json.dumps(res(hits=[{"page": 1}])) + "\n", encoding="utf-8")
    rs = RC.load_records([p])
    assert any(e.startswith("runs.jsonl:2: invalid JSON") for e in rs.errors)
    assert any(e.startswith("runs.jsonl:3") for e in rs.errors)
    with pytest.raises(OSError):
        RC.load_records([tmp_path / "missing.jsonl"])


def test_illustrative_runs_fixture_is_valid_and_marked():
    rs = RC.load_records([FIXTURES / "illustrative_runs.jsonl"])
    assert rs.errors == [] and set(rs.runs) == {"fixture-bm25", "fixture-hybrid"}
    assert all(r.provenance == "fixture" and "ILLUSTRATIVE" in r.description for r in rs.runs.values())
