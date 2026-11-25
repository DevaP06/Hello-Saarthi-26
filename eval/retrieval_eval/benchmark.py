"""Offline retrieval benchmark: score recorded results against benchmark cases (Sprint 5).

This evaluates RECORDED retrieval output only. It never runs a retriever, a model or a network call,
and a report never claims to measure live retrieval: it measures the runs in the results file(s),
with each run's provenance (recorded | fixture) carried into the report.

run_benchmark(cases, records, config) -> report dict (JSON-safe, deterministic):
  benchmark   configuration, case counts, label status/completeness counts per level, notes
  runs        per run (sorted by run_id): per-case metrics for every level and K, aggregates and
              diagnostics
  comparison  when two or more runs exist: each run against the baseline, on PAIRED cases only
  input_errors / unmatched_results   records that could not be used, and why

Per-case level status
  scored         labels and a ranking exist; metrics were computed
  not_evaluated  no labels for the level, or the result carries no ranking for it, or the
                 retriever reported an error. Never folded into averages.

Aggregates (per run, level, K, metric), over the cases with a result:
  counts of each metric status
  mean             mean of exact values, reported ONLY when every scored case is exact
  mean_exact_only  mean over the exact cases (always reported when there are any)
  lower_mean / upper_mean  mean of the bounds over exact + bounded cases (exact counts as both),
                   reported when no case is inconclusive
  not_applicable / not_evaluated cases are excluded from every mean.

Diagnostics are split so a missing label is never reported as a retrieval failure:
  retrieval_failures        zero_relevant_retrieved, poor_recall, relevant_ranked_low, empty_results,
                            duplicate_chunks, retriever_errors
  ground_truth_unavailable  missing_labels, no_labelled_relevant, unjudged_in_top_k, unresolved_hits
  input_problems            missing_results, score_order_violations
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from eval.retrieval_eval.cases import LEVELS, BenchmarkCase
from eval.retrieval_eval.metrics import (BOUNDED, DUPLICATE_POLICIES, EXACT, GAINS, INCONCLUSIVE, METRICS,
                                          NOT_APPLICABLE, NOT_EVALUATED, compute_all, prepare_ranking)
from eval.retrieval_eval.records import RecordedResult, RecordSet

SCORED = "scored"
OFFLINE_NOTE = ("Offline evaluation of recorded retrieval results. No retriever was executed by this "
                "benchmark; the numbers describe the recorded runs only.")
FIXTURE_NOTE = ("One or more runs have provenance 'fixture': they are illustrative example data, not "
                "Saarthi benchmark results.")
UNVERIFIED_NOTE = ("Some labels are unverified (derived or drafted, not judged by a person). Treat their "
                   "metrics as provisional.")
_TIE = 1e-12


@dataclass(frozen=True)
class BenchmarkConfig:
    ks: tuple[int, ...] = (1, 3, 5, 10)
    duplicates: str = "collapse"
    relevance_threshold: int = 1
    gain: str = "linear"
    poor_recall_threshold: float = 0.5
    low_rank_cutoff: int = 3
    baseline: str | None = None
    levels: tuple[str, ...] = field(default=LEVELS)

    def __post_init__(self) -> None:
        ks = tuple(self.ks)
        if not ks or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in ks):
            raise ValueError(f"ks must be integers >= 1, got {self.ks!r}")
        object.__setattr__(self, "ks", tuple(sorted(set(ks))))
        if self.duplicates not in DUPLICATE_POLICIES:
            raise ValueError(f"duplicates must be one of {DUPLICATE_POLICIES}")
        if self.gain not in GAINS:
            raise ValueError(f"gain must be one of {GAINS}")
        t = self.relevance_threshold
        if isinstance(t, bool) or not isinstance(t, int) or t < 1:
            raise ValueError("relevance_threshold must be an integer >= 1")
        p = self.poor_recall_threshold
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 < p <= 1:
            raise ValueError("poor_recall_threshold must be in (0, 1]")
        c = self.low_rank_cutoff
        if isinstance(c, bool) or not isinstance(c, int) or c < 1:
            raise ValueError("low_rank_cutoff must be an integer >= 1")
        levels = tuple(self.levels)
        if not levels or set(levels) - set(LEVELS):
            raise ValueError(f"levels must be drawn from {LEVELS}")
        object.__setattr__(self, "levels", tuple(lv for lv in LEVELS if lv in levels))

    def to_dict(self) -> dict:
        return {"ks": list(self.ks), "duplicates": self.duplicates,
                "relevance_threshold": self.relevance_threshold, "gain": self.gain,
                "poor_recall_threshold": self.poor_recall_threshold, "low_rank_cutoff": self.low_rank_cutoff,
                "baseline": self.baseline, "levels": list(self.levels)}


def _score_order_violations(result: RecordedResult) -> list[int]:
    """1-based positions whose score is higher than the score of the result ranked just above."""
    scores = result.scores()
    out = []
    for i in range(1, len(scores)):
        a, b = scores[i - 1], scores[i]
        if a is not None and b is not None and b > a + _TIE:
            out.append(i + 1)
    return out


def evaluate_level(case: BenchmarkCase, result: RecordedResult | None, level: str,
                   config: BenchmarkConfig) -> dict:
    """Metrics and per-level diagnostics for one case, one level."""
    labels = case.labels.get(level)
    if labels is None:
        return {"status": NOT_EVALUATED, "reason": f"no {level}-level labels for this case",
                "labels": None, "metrics": {}}
    label_info = {"status": labels.status, "completeness": labels.completeness,
                  "labelled_relevant": sum(1 for g in labels.grades.values() if g >= config.relevance_threshold)}
    if result is None:
        return {"status": NOT_EVALUATED, "reason": "no recorded result for this case",
                "labels": label_info, "metrics": {}}
    if result.error is not None:
        return {"status": NOT_EVALUATED, "reason": f"retriever reported an error: {result.error}",
                "labels": label_info, "metrics": {}}
    ids = result.ranking(level)
    if ids is None:
        return {"status": NOT_EVALUATED, "reason": f"the result carries no {level} ids",
                "labels": label_info, "metrics": {}}
    ranking = prepare_ranking(ids, config.duplicates)
    judgements = labels.judgements()
    metrics = {str(k): {name: mv.to_dict() for name, mv in
                        compute_all(ranking, judgements, k, threshold=config.relevance_threshold,
                                    gain=config.gain).items()}
               for k in config.ks}
    grades = [judgements.grade(item) for item in ranking.items]
    relevant_ranks = [i for i, g in enumerate(grades, 1) if g is not None and g >= config.relevance_threshold]
    kmax = config.ks[-1]
    info = {
        "retrieved": ranking.original_length,
        "distinct_retrieved": sum(1 for it in ranking.items if isinstance(it, str)),
        "unknown_items": sum(1 for it in ids if it is None),
        "duplicate_positions": list(ranking.duplicate_positions),
        "relevant_ranks": relevant_ranks,
        "unjudged_in_top_kmax": [it for it, g in zip(ranking.items[:kmax], grades[:kmax])
                                 if g is None and isinstance(it, str)],
        "unknown_in_top_kmax": sum(1 for it in ranking.items[:kmax] if it is None),
    }
    return {"status": SCORED, "reason": "", "labels": label_info, "metrics": metrics, "ranking": info}


def evaluate_case(case: BenchmarkCase, result: RecordedResult | None, config: BenchmarkConfig) -> dict:
    out = {"case_id": case.id, "language": case.language, "category": case.category,
           "source": case.source, "has_result": result is not None,
           "retriever_error": None if result is None else result.error,
           "levels": {lv: evaluate_level(case, result, lv, config) for lv in config.levels}}
    if result is not None and result.error is None:
        out["score_order_violations"] = _score_order_violations(result)
    return out


def _aggregate(case_results: list[dict], level: str, k: int, metric: str) -> dict:
    counts = Counter()
    exact, lows, ups = [], [], []
    for cr in case_results:
        lv = cr["levels"][level]
        if lv["status"] != SCORED:
            counts[NOT_EVALUATED] += 1
            continue
        mv = lv["metrics"][str(k)][metric]
        counts[mv["status"]] += 1
        if mv["status"] == EXACT:
            exact.append(mv["value"])
            lows.append(mv["value"])
            ups.append(mv["value"])
        elif mv["status"] == BOUNDED:
            lows.append(mv["lower"])
            ups.append(mv["upper"])
    scored = counts[EXACT] + counts[BOUNDED] + counts[INCONCLUSIVE]
    mean_exact = math.fsum(exact) / len(exact) if exact else None
    return {
        "counts": {s: counts[s] for s in (EXACT, BOUNDED, INCONCLUSIVE, NOT_APPLICABLE, NOT_EVALUATED)},
        "mean": mean_exact if exact and counts[EXACT] == scored else None,
        "mean_exact_only": mean_exact,
        "lower_mean": math.fsum(lows) / len(lows) if lows and counts[INCONCLUSIVE] == 0 else None,
        "upper_mean": math.fsum(ups) / len(ups) if ups and counts[INCONCLUSIVE] == 0 else None,
    }


def _diagnostics(case_results: list[dict], cases_without_result: list[str], config: BenchmarkConfig) -> dict:
    kmax = config.ks[-1]
    rf = {"zero_relevant_retrieved": [], "poor_recall": [], "relevant_ranked_low": [], "empty_results": [],
          "duplicate_chunks": [], "retriever_errors": []}
    gt = {"missing_labels": [], "no_labelled_relevant": [], "unjudged_in_top_k": [], "unresolved_hits": []}
    ip = {"missing_results": sorted(cases_without_result), "score_order_violations": []}
    for cr in case_results:
        cid = cr["case_id"]
        if cr["retriever_error"] is not None:
            rf["retriever_errors"].append({"case_id": cid, "error": cr["retriever_error"]})
            continue
        if cr.get("score_order_violations"):
            ip["score_order_violations"].append({"case_id": cid, "positions": cr["score_order_violations"]})
        emptied = False
        for level, lv in cr["levels"].items():
            if lv["labels"] is None:
                gt["missing_labels"].append({"case_id": cid, "level": level})
                continue
            if lv["status"] != SCORED:
                continue
            info = lv["ranking"]
            if info["retrieved"] == 0 and not emptied:
                rf["empty_results"].append({"case_id": cid})
                emptied = True
            if level == "chunk" and info["duplicate_positions"]:
                rf["duplicate_chunks"].append({"case_id": cid, "positions": info["duplicate_positions"]})
            if level == "document" and info["unknown_items"]:
                gt["unresolved_hits"].append({"case_id": cid, "count": info["unknown_items"]})
            if lv["labels"]["labelled_relevant"] == 0:
                gt["no_labelled_relevant"].append({"case_id": cid, "level": level, "retrieved": info["retrieved"]})
                continue
            if info["unjudged_in_top_kmax"] or info["unknown_in_top_kmax"]:
                gt["unjudged_in_top_k"].append({"case_id": cid, "level": level, "k": kmax,
                                                "unjudged_ids": info["unjudged_in_top_kmax"],
                                                "unknown_items": info["unknown_in_top_kmax"]})
            recall = lv["metrics"][str(kmax)]["recall"]
            if recall["status"] == EXACT and recall["value"] == 0:
                rf["zero_relevant_retrieved"].append({"case_id": cid, "level": level, "k": kmax,
                                                      "found_at_ranks": info["relevant_ranks"]})
            elif recall["status"] == EXACT and recall["value"] < config.poor_recall_threshold:
                rf["poor_recall"].append({"case_id": cid, "level": level, "k": kmax, "recall": recall["value"]})
            ranks = info["relevant_ranks"]
            if ranks and ranks[0] > config.low_rank_cutoff:
                rf["relevant_ranked_low"].append({"case_id": cid, "level": level, "first_relevant_rank": ranks[0],
                                                  "cutoff": config.low_rank_cutoff})
    return {"retrieval_failures": rf, "ground_truth_unavailable": gt, "input_problems": ip}


def evaluate_run(cases: Sequence[BenchmarkCase], records: RecordSet, run_id: str, config: BenchmarkConfig) -> dict:
    results = records.results_for(run_id)
    ordered = sorted(cases, key=lambda c: c.id)
    case_results = [evaluate_case(c, results[c.id], config) for c in ordered if c.id in results]
    without = [c.id for c in ordered if c.id not in results]
    aggregates = {lv: {str(k): {m: _aggregate(case_results, lv, k, m) for m in METRICS} for k in config.ks}
                  for lv in config.levels}
    info = records.runs[run_id]
    return {"run": info.to_dict(), "illustrative": info.provenance == "fixture",
            "cases_with_result": len(case_results), "cases_without_result": len(without),
            "aggregates": aggregates, "diagnostics": _diagnostics(case_results, without, config),
            "cases": case_results}


def _paired(base: dict, other: dict, level: str, k: int, metric: str) -> dict:
    def values(run):
        out = {}
        for cr in run["cases"]:
            lv = cr["levels"][level]
            if lv["status"] == SCORED and lv["metrics"][str(k)][metric]["status"] == EXACT:
                out[cr["case_id"]] = lv["metrics"][str(k)][metric]["value"]
        return out
    a, b = values(base), values(other)
    common = sorted(set(a) & set(b))
    wins = sum(1 for c in common if b[c] > a[c] + _TIE)
    losses = sum(1 for c in common if b[c] < a[c] - _TIE)
    base_mean = math.fsum(a[c] for c in common) / len(common) if common else None
    other_mean = math.fsum(b[c] for c in common) / len(common) if common else None
    return {"paired_cases": len(common), "baseline_mean": base_mean, "run_mean": other_mean,
            "delta": None if not common else other_mean - base_mean,
            "wins": wins, "losses": losses, "ties": len(common) - wins - losses,
            "excluded_cases": len((set(a) | set(b)) - set(common))}


def compare_runs(run_reports: list[dict], config: BenchmarkConfig) -> dict | None:
    if len(run_reports) < 2:
        return None
    ids = [r["run"]["run_id"] for r in run_reports]
    baseline = config.baseline or ids[0]
    if baseline not in ids:
        raise ValueError(f"baseline run {baseline!r} is not in the results (runs: {ids})")
    base = run_reports[ids.index(baseline)]
    provenances = sorted({r["run"]["provenance"] for r in run_reports})
    out = {"baseline": baseline, "paired_on": "cases where both runs have an exact value",
           "mixed_provenance": len(provenances) > 1, "runs": {}}
    for rr in run_reports:
        rid = rr["run"]["run_id"]
        if rid == baseline:
            continue
        out["runs"][rid] = {lv: {str(k): {m: _paired(base, rr, lv, k, m) for m in METRICS} for k in config.ks}
                            for lv in config.levels}
    return out


def _label_summary(cases: Sequence[BenchmarkCase]) -> dict:
    out = {}
    for lv in LEVELS:
        c = Counter()
        for case in cases:
            lab = case.labels.get(lv)
            c["missing" if lab is None else f"{lab.status}/{lab.completeness}"] += 1
        out[lv] = dict(sorted(c.items()))
    return out


def run_benchmark(cases: Sequence[BenchmarkCase], records: RecordSet,
                  config: BenchmarkConfig | None = None) -> dict:
    config = config or BenchmarkConfig()
    case_ids = {c.id for c in cases}
    unmatched = sorted({(rid, cid) for (rid, cid) in records.results if cid not in case_ids})
    run_reports = [evaluate_run(cases, records, rid, config) for rid in sorted(records.runs)]
    notes = [OFFLINE_NOTE]
    if any(r["illustrative"] for r in run_reports):
        notes.append(FIXTURE_NOTE)
    if any(lab.status == "unverified" for c in cases for lab in c.labels.values()):
        notes.append(UNVERIFIED_NOTE)
    runs_without_results = sorted(rid for rid in records.runs if not records.results_for(rid))
    return {
        "benchmark": {"config": config.to_dict(), "cases": len(cases),
                      "case_sources": dict(sorted(Counter(c.source for c in cases).items())),
                      "languages": dict(sorted(Counter(c.language for c in cases).items())),
                      "labels": _label_summary(cases), "notes": notes, "evaluates_live_retrieval": False},
        "runs": run_reports,
        "comparison": compare_runs(run_reports, config),
        "input_errors": list(records.errors),
        "unmatched_results": [{"run_id": r, "case_id": c} for r, c in unmatched],
        "runs_without_results": runs_without_results,
    }
