"""Sprint 5 tests: benchmark engine, diagnostics, comparison, CLI and the live-retriever adapter."""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.answer_eval.schema import MATERIALS, load_dataset  # noqa: E402
from eval.retrieval_eval import adapters as A  # noqa: E402
from eval.retrieval_eval import run_benchmark as CLI  # noqa: E402
from eval.retrieval_eval.benchmark import BenchmarkConfig, run_benchmark  # noqa: E402
from eval.retrieval_eval.cases import cases_from_answer_items, load_cases, parse_cases  # noqa: E402
from eval.retrieval_eval.records import load_records, parse_records  # noqa: E402

FIX = Path(__file__).resolve().parents[1] / "fixtures"
FX_CASES = FIX / "illustrative_cases.jsonl"
FX_RUNS = FIX / "illustrative_runs.jsonl"
CASES, _ = load_cases(FX_CASES)
ITEMS, _ = load_dataset(ROOT / "eval" / "answer_eval" / "dataset.jsonl")
L2 = math.log2


def approx(x):
    return pytest.approx(x, abs=1e-12)


def lab(grades, completeness="exhaustive"):
    return {"status": "fixture", "completeness": completeness, "grades": grades, "provenance": "test"}


def mk_cases(*specs):
    rows = []
    for cid, labels in specs:
        rows.append((cid, {"id": cid, "question": f"question {cid}?", "language": "en", "category": "t",
                           "source": "fixture", "labels": labels}))
    cases, errs = parse_cases(rows)
    assert errs == []
    return cases


def mk_records(*rows):
    rs = parse_records([(f"l{i}", r) for i, r in enumerate(rows, 1)])
    return rs


def run_hdr(rid, provenance="fixture"):
    return {"record_type": "run", "run_id": rid, "provenance": provenance}


def result(rid, cid, docs):
    return {"record_type": "result", "run_id": rid, "case_id": cid,
            "hits": [{"document_id": d} for d in docs]}


def level(report, run_id, case_id, lv="document"):
    run = next(r for r in report["runs"] if r["run"]["run_id"] == run_id)
    return next(c for c in run["cases"] if c["case_id"] == case_id)["levels"][lv]


def agg(report, run_id, lv, k, metric):
    run = next(r for r in report["runs"] if r["run"]["run_id"] == run_id)
    return run["aggregates"][lv][str(k)][metric]


# ── Per-case scoring ────────────────────────────────────────────────────────────────────────

TWO = mk_cases(("a", {"document": lab({"d1": 1, "d2": 1})}), ("b", {"document": lab({"d3": 1})}))


def test_per_case_metrics_and_aggregate_means():
    rs = mk_records(run_hdr("r"), result("r", "a", ["d1", "x", "d2"]), result("r", "b", ["y", "z", "d3"]))
    rep = run_benchmark(TWO, rs, BenchmarkConfig(ks=(1, 3), levels=("document",)))
    a, b = level(rep, "r", "a"), level(rep, "r", "b")
    assert a["metrics"]["3"]["precision"]["value"] == approx(2 / 3)
    assert b["metrics"]["3"]["mrr"]["value"] == approx(1 / 3)
    assert agg(rep, "r", "document", 1, "precision")["mean"] == approx(0.5)            # (1 + 0) / 2
    assert agg(rep, "r", "document", 3, "recall")["mean"] == approx(1.0)
    assert agg(rep, "r", "document", 3, "mrr")["mean"] == approx((1 + 1 / 3) / 2)
    ndcg_a = (1 + 1 / L2(4)) / (1 + 1 / L2(3))
    ndcg_b = (1 / L2(4)) / 1
    assert agg(rep, "r", "document", 3, "ndcg")["mean"] == approx((ndcg_a + ndcg_b) / 2)


def test_missing_results_are_reported_not_scored_as_zero():
    rs = mk_records(run_hdr("r"), result("r", "a", ["d1"]))
    rep = run_benchmark(TWO, rs, BenchmarkConfig(ks=(1,), levels=("document",)))
    run = rep["runs"][0]
    assert run["cases_with_result"] == 1 and run["diagnostics"]["input_problems"]["missing_results"] == ["b"]
    assert agg(rep, "r", "document", 1, "precision")["mean"] == 1.0            # 'b' is not averaged in as 0
    assert [c["case_id"] for c in run["cases"]] == ["a"]


def test_empty_retrieval_is_a_real_zero_and_flagged():
    rs = mk_records(run_hdr("r"), result("r", "a", []), result("r", "b", ["d3"]))
    rep = run_benchmark(TWO, rs, BenchmarkConfig(ks=(3,), levels=("document",)))
    assert level(rep, "r", "a")["metrics"]["3"]["recall"] == {**level(rep, "r", "a")["metrics"]["3"]["recall"],
                                                              "status": "exact", "value": 0.0}
    rf = rep["runs"][0]["diagnostics"]["retrieval_failures"]
    assert rf["empty_results"] == [{"case_id": "a"}]
    assert rf["zero_relevant_retrieved"][0]["case_id"] == "a"


def test_unknown_case_ids_are_reported_and_ignored():
    rs = mk_records(run_hdr("r"), result("r", "a", ["d1"]), result("r", "zz-99", ["d1"]))
    rep = run_benchmark(TWO, rs, BenchmarkConfig(ks=(1,), levels=("document",)))
    assert rep["unmatched_results"] == [{"run_id": "r", "case_id": "zz-99"}]
    assert all(c["case_id"] != "zz-99" for c in rep["runs"][0]["cases"])


def test_missing_labels_are_not_evaluated_and_diagnosed_as_ground_truth_gaps():
    cases = mk_cases(("a", {"document": lab({"d1": 1}), "chunk": None}))
    rs = mk_records(run_hdr("r"), {"record_type": "result", "run_id": "r", "case_id": "a",
                                   "hits": [{"document_id": "x", "chunk_id": "c1"}]})
    rep = run_benchmark(cases, rs, BenchmarkConfig(ks=(1,)))
    chunk = level(rep, "r", "a", "chunk")
    assert chunk["status"] == "not_evaluated" and "no chunk-level labels" in chunk["reason"]
    d = rep["runs"][0]["diagnostics"]
    assert {"case_id": "a", "level": "chunk"} in d["ground_truth_unavailable"]["missing_labels"]
    assert agg(rep, "r", "chunk", 1, "precision")["counts"]["not_evaluated"] == 1
    assert agg(rep, "r", "chunk", 1, "precision")["mean"] is None
    # the document miss IS a retrieval failure
    assert d["retrieval_failures"]["zero_relevant_retrieved"][0]["level"] == "document"


def test_partial_labels_give_bounds_and_unjudged_diagnostics_not_failures():
    cases = mk_cases(("a", {"document": lab({"d1": 1}, "partial")}))
    rs = mk_records(run_hdr("r"), result("r", "a", ["u1", "d1"]))
    rep = run_benchmark(cases, rs, BenchmarkConfig(ks=(1, 2), levels=("document",)))
    p1 = level(rep, "r", "a")["metrics"]["1"]["precision"]
    assert p1["status"] == "bounded" and (p1["lower"], p1["upper"]) == (0.0, 1.0)
    a = agg(rep, "r", "document", 1, "precision")
    assert a["mean"] is None and a["mean_exact_only"] is None and (a["lower_mean"], a["upper_mean"]) == (0.0, 1.0)
    assert agg(rep, "r", "document", 2, "ndcg")["lower_mean"] is None            # inconclusive -> no bounds
    d = rep["runs"][0]["diagnostics"]
    assert d["ground_truth_unavailable"]["unjudged_in_top_k"][0]["unjudged_ids"] == ["u1"]
    assert d["retrieval_failures"]["zero_relevant_retrieved"] == []


def test_no_labelled_relevant_items_is_not_applicable_and_not_a_failure():
    cases = mk_cases(("a", {"document": lab({}, "partial")}))
    rs = mk_records(run_hdr("r"), result("r", "a", ["d1", "d2"]))
    rep = run_benchmark(cases, rs, BenchmarkConfig(ks=(2,), levels=("document",)))
    assert {m["status"] for m in level(rep, "r", "a")["metrics"]["2"].values()} == {"not_applicable"}
    d = rep["runs"][0]["diagnostics"]
    assert d["ground_truth_unavailable"]["no_labelled_relevant"] == [{"case_id": "a", "level": "document", "retrieved": 2}]
    assert not any(d["retrieval_failures"].values())


def test_poor_recall_and_relevant_ranked_low():
    cases = mk_cases(("a", {"document": lab({"d1": 1, "d2": 1, "d3": 1})}))
    rs = mk_records(run_hdr("r"), result("r", "a", ["x", "y", "z", "w", "d1", "q", "d2"]))
    rep = run_benchmark(cases, rs, BenchmarkConfig(ks=(5,), levels=("document",), low_rank_cutoff=3))
    rf = rep["runs"][0]["diagnostics"]["retrieval_failures"]
    assert rf["poor_recall"] == [{"case_id": "a", "level": "document", "k": 5, "recall": approx(1 / 3)}]
    assert rf["relevant_ranked_low"] == [{"case_id": "a", "level": "document", "first_relevant_rank": 5, "cutoff": 3}]


def test_relevant_beyond_kmax_is_reported_with_its_rank():
    cases = mk_cases(("a", {"document": lab({"d1": 1})}))
    rs = mk_records(run_hdr("r"), result("r", "a", ["x", "y", "d1"]))
    rep = run_benchmark(cases, rs, BenchmarkConfig(ks=(1, 2), levels=("document",)))
    z = rep["runs"][0]["diagnostics"]["retrieval_failures"]["zero_relevant_retrieved"]
    assert z == [{"case_id": "a", "level": "document", "k": 2, "found_at_ranks": [3]}]


def test_duplicate_chunks_are_flagged_and_collapsed():
    cases = mk_cases(("a", {"chunk": lab({"c1": 1, "c2": 1})}))
    rs = mk_records(run_hdr("r"), {"record_type": "result", "run_id": "r", "case_id": "a",
                                   "hits": [{"document_id": "d", "chunk_id": c} for c in ("c1", "c1", "c2")]})
    rep = run_benchmark(cases, rs, BenchmarkConfig(ks=(2,), levels=("chunk",)))
    assert level(rep, "r", "a", "chunk")["metrics"]["2"]["precision"]["value"] == 1.0
    assert rep["runs"][0]["diagnostics"]["retrieval_failures"]["duplicate_chunks"] == [{"case_id": "a", "positions": [2]}]
    pen = run_benchmark(cases, rs, BenchmarkConfig(ks=(2,), levels=("chunk",), duplicates="penalize"))
    assert level(pen, "r", "a", "chunk")["metrics"]["2"]["precision"]["value"] == 0.5


def test_retriever_errors_are_not_scored():
    rs = mk_records(run_hdr("r"), {"record_type": "result", "run_id": "r", "case_id": "a", "error": "Timeout"},
                    result("r", "b", ["d3"]))
    rep = run_benchmark(TWO, rs, BenchmarkConfig(ks=(1,), levels=("document",)))
    assert level(rep, "r", "a")["status"] == "not_evaluated"
    assert rep["runs"][0]["diagnostics"]["retrieval_failures"]["retriever_errors"] == [{"case_id": "a", "error": "Timeout"}]
    assert agg(rep, "r", "document", 1, "precision")["mean"] == 1.0


def test_score_order_violations_are_reported_but_never_reorder():
    cases = mk_cases(("a", {"document": lab({"d1": 1})}))
    rs = mk_records(run_hdr("r"), {"record_type": "result", "run_id": "r", "case_id": "a",
                                   "hits": [{"document_id": "x", "score": 0.2}, {"document_id": "d1", "score": 0.9}]})
    rep = run_benchmark(cases, rs, BenchmarkConfig(ks=(1,), levels=("document",)))
    assert level(rep, "r", "a")["metrics"]["1"]["precision"]["value"] == 0.0
    assert rep["runs"][0]["diagnostics"]["input_problems"]["score_order_violations"] == [{"case_id": "a", "positions": [2]}]


def test_unresolved_hits_are_unjudged_not_irrelevant():
    cases = mk_cases(("a", {"document": lab({"d1": 1})}))
    rs = mk_records(run_hdr("r"), {"record_type": "result", "run_id": "r", "case_id": "a",
                                   "hits": [{"document_id": None}, {"document_id": "d1"}]})
    rep = run_benchmark(cases, rs, BenchmarkConfig(ks=(2,), levels=("document",)))
    assert level(rep, "r", "a")["metrics"]["2"]["precision"]["status"] == "bounded"
    assert rep["runs"][0]["diagnostics"]["ground_truth_unavailable"]["unresolved_hits"] == [{"case_id": "a", "count": 1}]


# ── Comparison ──────────────────────────────────────────────────────────────────────────────

def test_comparison_is_paired_with_wins_losses_ties():
    rs = mk_records(run_hdr("base"), run_hdr("new"),
                    result("base", "a", ["x", "d1"]), result("base", "b", ["d3"]),
                    result("new", "a", ["d1", "x"]), result("new", "b", ["d3"]))
    rep = run_benchmark(TWO, rs, BenchmarkConfig(ks=(1,), levels=("document",)))
    c = rep["comparison"]
    assert c["baseline"] == "base" and set(c["runs"]) == {"new"}
    mrr = c["runs"]["new"]["document"]["1"]["mrr"]
    assert mrr == {"paired_cases": 2, "baseline_mean": 0.5, "run_mean": 1.0, "delta": 0.5,
                   "wins": 1, "losses": 0, "ties": 1, "excluded_cases": 0}


def test_comparison_excludes_unpaired_cases_and_honours_baseline():
    rs = mk_records(run_hdr("base"), run_hdr("new"), result("base", "a", ["d1"]), result("base", "b", ["d3"]),
                    result("new", "a", ["x"]))
    rep = run_benchmark(TWO, rs, BenchmarkConfig(ks=(1,), levels=("document",), baseline="new"))
    p = rep["comparison"]["runs"]["base"]["document"]["1"]["precision"]
    assert rep["comparison"]["baseline"] == "new" and p["paired_cases"] == 1 and p["excluded_cases"] == 1
    assert p["delta"] == 1.0
    with pytest.raises(ValueError):
        run_benchmark(TWO, rs, BenchmarkConfig(ks=(1,), baseline="ghost"))


def test_single_run_has_no_comparison_and_mixed_provenance_is_flagged():
    rs = mk_records(run_hdr("only"), result("only", "a", ["d1"]))
    assert run_benchmark(TWO, rs, BenchmarkConfig(ks=(1,)))["comparison"] is None
    rs2 = mk_records(run_hdr("f"), run_hdr("r", "recorded"), result("f", "a", ["d1"]), result("r", "a", ["d1"]))
    assert run_benchmark(TWO, rs2, BenchmarkConfig(ks=(1,)))["comparison"]["mixed_provenance"] is True


# ── Determinism and honesty of the report ───────────────────────────────────────────────────

def test_report_is_deterministic_and_independent_of_input_order():
    rows = [(f"l{i}", json.loads(line)) for i, line in enumerate(FX_RUNS.read_text().splitlines(), 1) if line.strip()]
    first = json.dumps(run_benchmark(CASES, parse_records(rows)), sort_keys=True)
    for seed in range(3):
        shuffled = rows[:]
        random.Random(seed).shuffle(shuffled)
        cases = CASES[:]
        random.Random(seed).shuffle(cases)
        again = run_benchmark(cases, parse_records([(f"l{i}", r) for i, (_, r) in enumerate(shuffled, 1)]))
        assert json.dumps(again, sort_keys=True) == first


def test_report_says_it_is_offline_and_marks_fixtures():
    rep = run_benchmark(CASES, load_records([FX_RUNS]))
    assert rep["benchmark"]["evaluates_live_retrieval"] is False
    assert any("No retriever was executed" in n for n in rep["benchmark"]["notes"])
    assert any("illustrative example data" in n for n in rep["benchmark"]["notes"])
    assert all(r["illustrative"] for r in rep["runs"])


def test_illustrative_fixture_report_values():
    rep = run_benchmark(CASES, load_records([FX_RUNS]), BenchmarkConfig(ks=(1, 3, 5)))
    # hand-checked: fixture-bm25 chunk level, cases fx-01 (graded) and fx-03 (empty retrieval)
    ndcg3_fx01 = (1 + 3 / L2(3)) / (3 + 1 / L2(3) + 1 / L2(4))
    assert agg(rep, "fixture-bm25", "chunk", 3, "ndcg")["mean"] == approx(ndcg3_fx01 / 2)
    assert agg(rep, "fixture-bm25", "chunk", 1, "precision")["mean"] == approx(0.5)
    assert agg(rep, "fixture-hybrid", "document", 1, "precision")["mean"] == approx(2 / 3)
    assert {"case_id": "fx-01", "positions": [3]} in \
        rep["runs"][0]["diagnostics"]["retrieval_failures"]["duplicate_chunks"]


def test_derived_sprint1_cases_never_score_chunks_or_unjudged_pages_as_irrelevant():
    cases = cases_from_answer_items(ITEMS)
    m1 = "EE210 Module 1 Signals and Sampling"
    rs = mk_records(run_hdr("r"), {"record_type": "result", "run_id": "r", "case_id": "df-01",
                                   "hits": [{"document_id": m1, "page": 2, "chunk_id": "x"},
                                            {"document_id": m1, "page": 3, "chunk_id": "y"}]})
    rep = run_benchmark(cases, rs, BenchmarkConfig(ks=(1, 3)))
    assert level(rep, "r", "df-01", "chunk")["status"] == "not_evaluated"
    p1 = level(rep, "r", "df-01", "page")["metrics"]["1"]["precision"]
    assert p1["status"] == "bounded" and (p1["lower"], p1["upper"]) == (0.0, 1.0)
    assert level(rep, "r", "df-01", "page")["metrics"]["3"]["recall"]["value"] == 1.0
    assert any("unverified" in n for n in rep["benchmark"]["notes"])
    assert rep["benchmark"]["labels"]["chunk"] == {"missing": 37}


def test_invalid_config_is_rejected():
    for kw in ({"ks": ()}, {"ks": (0,)}, {"ks": (True,)}, {"duplicates": "x"}, {"gain": "x"},
               {"relevance_threshold": 0}, {"poor_recall_threshold": 0}, {"low_rank_cutoff": 0},
               {"levels": ("sentence",)}):
        with pytest.raises(ValueError):
            BenchmarkConfig(**kw)
    assert BenchmarkConfig(ks=(10, 1, 3, 3)).ks == (1, 3, 10)


# ── CLI ─────────────────────────────────────────────────────────────────────────────────────

def test_cli_fixture_run_writes_report(tmp_path, capsys):
    out = tmp_path / "report.json"
    code = CLI.main(["--cases", str(FX_CASES), "--results", str(FX_RUNS), "--k", "1", "3", "--out", str(out)])
    assert code == 0
    report = json.loads(out.read_text())
    assert [r["run"]["run_id"] for r in report["runs"]] == ["fixture-bm25", "fixture-hybrid"]
    text = capsys.readouterr().out
    assert "ILLUSTRATIVE FIXTURE" in text and "offline" in text


def test_cli_exit_codes(tmp_path):
    bad_cases = tmp_path / "bad.jsonl"
    bad_cases.write_text('{"id": "x"}\n', encoding="utf-8")
    assert CLI.main(["--cases", str(bad_cases), "--results", str(FX_RUNS)]) == 2
    assert CLI.main(["--cases", str(FX_CASES), "--results", str(tmp_path / "missing.jsonl")]) == 1
    assert CLI.main(["--cases", str(FX_CASES)]) == 2                                  # nothing to score
    assert CLI.main(["--cases", str(FX_CASES), "--results", str(FX_RUNS), "--k", "0"]) == 2
    assert CLI.main(["--cases", str(FX_CASES), "--results", str(FX_RUNS), "--baseline", "ghost"]) == 2
    partial = tmp_path / "partial.jsonl"
    partial.write_text("\n".join(FX_RUNS.read_text().splitlines()[:2]) + "\n{oops\n", encoding="utf-8")
    out = tmp_path / "r.json"
    assert CLI.main(["--cases", str(FX_CASES), "--results", str(partial), "--out", str(out)]) == 0
    assert CLI.main(["--cases", str(FX_CASES), "--results", str(partial), "--out", str(out), "--strict"]) == 1
    assert CLI.main(["--cases", str(FX_CASES), "--results", str(FX_RUNS), "--out", str(out), "--strict"]) == 0


def test_cli_default_cases_come_from_the_validated_sprint1_dataset(tmp_path):
    exported = tmp_path / "derived.jsonl"
    assert CLI.main(["--export-cases", str(exported)]) == 0
    cases, errs = load_cases(exported)
    assert errs == [] and len(cases) == 37 and all(c.source == "sprint1_dataset" for c in cases)


# ── Adapter (future live integration) ───────────────────────────────────────────────────────

def chunk(id_, title, page, score, source_file=None):
    return SimpleNamespace(id=id_, title=title, source_file=source_file, page=page, score=score,
                           content_type="course_material", via="dense+bm25", material_id=7)


def test_resolver_is_exact_after_normalisation_and_never_guesses():
    for title, filename in MATERIALS.items():
        assert A.resolve_ee210_document_id(title) == title
        assert A.resolve_ee210_document_id(filename) == title
        assert A.resolve_ee210_document_id(filename.lower().replace(".pdf", "")) == title
    assert A.resolve_ee210_document_id("EE210 Module 1") is None
    assert A.resolve_ee210_document_id(None, "", "notes.pdf") is None
    assert A.resolve_ee210_document_id(None, "EE210_Module_2_LTI_Systems_and_Convolution.pdf") == \
        "EE210 Module 2 LTI Systems and Convolution"


def test_hit_from_chunk_maps_saarthi_chunks():
    h = A.hit_from_chunk(chunk("uuid-1", "EE210 Module 1 Signals and Sampling", 3, 0.7))
    assert h["document_id"] == "EE210 Module 1 Signals and Sampling" and h["page"] == 3
    assert h["chunk_id"] == "uuid-1" and h["score"] == 0.7 and h["metadata"]["via"] == "dense+bm25"
    unknown = A.hit_from_chunk(chunk("uuid-2", "Some upload", None, None))
    assert unknown["document_id"] is None and unknown["page"] is None and unknown["score"] is None
    assert A.hit_from_chunk({"id": "c", "title": "x", "page": True})["page"] is None


def test_record_run_end_to_end_without_leaking_labels(tmp_path):
    cases = cases_from_answer_items(ITEMS)[:3]
    seen = []

    def fake_retriever(request):
        seen.append(request)
        if request.case_id == cases[2].id:
            raise RuntimeError("index offline")
        return [chunk("c1", "EE210 Module 1 Signals and Sampling", 3, 0.9)]

    records = A.record_run(cases, fake_retriever, run_id="fake-run", description="test double",
                           retriever_config={"k": 1})
    assert records[0]["provenance"] == "recorded" and records[0]["metadata"]["include_scope"] is False
    for req in seen:
        assert set(vars(req)) == {"case_id", "question", "language", "scope"} and req.scope is None
    assert records[-1]["error"] == "RuntimeError: index offline"
    path = tmp_path / "fake.jsonl"
    A.write_records(records, path)
    rs = load_records([path])
    assert rs.errors == []
    rep = run_benchmark(cases, rs, BenchmarkConfig(ks=(1,)))
    assert rep["runs"][0]["illustrative"] is False
    assert rep["runs"][0]["diagnostics"]["retrieval_failures"]["retriever_errors"][0]["case_id"] == cases[2].id


def test_scope_is_only_passed_when_asked_for():
    case = next(c for c in cases_from_answer_items(ITEMS) if c.metadata["scope"])
    assert A.request_for(case).scope is None
    assert dict(A.request_for(case, include_scope=True).scope) == dict(case.metadata["scope"])


def test_adapter_does_not_import_the_application():
    import ast
    src = (Path(A.__file__)).read_text(encoding="utf-8")
    imported = {n.module for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom) and n.module} | \
               {a.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m.startswith(("src", "saarthi_backend", "llm-backend")) for m in imported)
