"""Score recorded retrieval results offline (Sprint 5 retrieval benchmark CLI).

Run from the repository root:
    python -m eval.retrieval_eval.run_benchmark --results runs.jsonl --out report.json
    python -m eval.retrieval_eval.run_benchmark --cases eval/retrieval_eval/fixtures/illustrative_cases.jsonl \
        --results eval/retrieval_eval/fixtures/illustrative_runs.jsonl --k 1 3 5
    python -m eval.retrieval_eval.run_benchmark --export-cases cases.jsonl   # write the derived cases

Cases: --cases FILE, or by default the cases derived from the Sprint 1 answer-evaluation dataset
(validated first, including its evidence quotes against the EE210 PDFs; see cases.py for what the
derived labels do and do not claim). Results: one or more --results files (see records.py).

This scores recorded results only; it never runs a retriever, a model or a network call.
Exit codes: 0 report written; 1 a results file is unreadable, or --strict and there were input
errors, unmatched results or missing results; 2 invalid cases or dataset, or a bad option.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.retrieval_eval.benchmark import BenchmarkConfig, run_benchmark  # noqa: E402
from eval.retrieval_eval.cases import cases_from_answer_items, load_cases, write_cases  # noqa: E402
from eval.retrieval_eval.records import load_records  # noqa: E402


def _fmt(agg: dict) -> str:
    if agg["mean"] is not None:
        return f"{agg['mean']:.3f}"
    if agg["lower_mean"] is not None:
        return f"[{agg['lower_mean']:.3f}, {agg['upper_mean']:.3f}]"
    if agg["mean_exact_only"] is not None:
        return f"{agg['mean_exact_only']:.3f}* (exact cases only)"
    return "n/a"


def summary_lines(report: dict) -> list[str]:
    b = report["benchmark"]
    lines = [f"Cases: {b['cases']} {b['case_sources']}  (offline: recorded results only)"]
    for note in b["notes"][1:]:
        lines.append(f"NOTE: {note}")
    for run in report["runs"]:
        rid = run["run"]["run_id"]
        tag = "  ILLUSTRATIVE FIXTURE" if run["illustrative"] else ""
        lines.append(f"Run {rid}{tag}: {run['cases_with_result']} case(s) with results, "
                     f"{run['cases_without_result']} without")
        for level, by_k in run["aggregates"].items():
            for k, by_metric in by_k.items():
                rec = by_metric["recall"]["counts"]
                if rec["exact"] + rec["bounded"] + rec["inconclusive"] == 0:
                    continue
                lines.append(f"  {level} @{k}: " + ", ".join(f"{m} {_fmt(a)}" for m, a in by_metric.items()))
    if report["input_errors"]:
        lines.append(f"Input errors: {len(report['input_errors'])}")
    if report["unmatched_results"]:
        lines.append(f"Results for unknown case ids: {len(report['unmatched_results'])}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", type=Path, help="benchmark cases JSONL (default: derived from Sprint 1)")
    ap.add_argument("--results", type=Path, action="append", default=[], help="recorded results JSONL")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--duplicates", choices=["collapse", "penalize"], default="collapse")
    ap.add_argument("--relevance-threshold", type=int, default=1)
    ap.add_argument("--gain", choices=["linear", "exponential"], default="linear")
    ap.add_argument("--poor-recall", type=float, default=0.5, help="flag recall@Kmax below this")
    ap.add_argument("--low-rank-cutoff", type=int, default=3, help="flag a first relevant rank above this")
    ap.add_argument("--levels", nargs="+", choices=["document", "page", "chunk"],
                    default=["document", "page", "chunk"])
    ap.add_argument("--baseline", help="run_id to compare the other runs against (default: first by id)")
    ap.add_argument("--out", type=Path, help="write the JSON report here (default: stdout)")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any record was invalid, unmatched or missing")
    ap.add_argument("--export-cases", type=Path, help="write the benchmark cases as JSONL and exit")
    args = ap.parse_args(argv)

    try:
        config = BenchmarkConfig(ks=tuple(args.k), duplicates=args.duplicates,
                                 relevance_threshold=args.relevance_threshold, gain=args.gain,
                                 poor_recall_threshold=args.poor_recall, low_rank_cutoff=args.low_rank_cutoff,
                                 baseline=args.baseline, levels=tuple(args.levels))
    except ValueError as e:
        print(f"Invalid option: {e}", file=sys.stderr)
        return 2

    if args.cases:
        try:
            cases, errors = load_cases(args.cases)
        except OSError as e:
            print(f"Cannot read cases file: {e}", file=sys.stderr)
            return 2
    else:
        from eval.answer_eval.validate_dataset import validate
        v = validate()
        errors = [f"Sprint 1 dataset: {e}" for e in v["errors"]]
        cases = cases_from_answer_items(v["items"]) if not errors else []
    if errors:
        print(f"Benchmark cases invalid ({len(errors)} error(s)); not evaluating:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 2

    if args.export_cases:
        write_cases(cases, args.export_cases)
        print(f"{len(cases)} case(s) written to {args.export_cases}")
        return 0
    if not args.results:
        print("No --results given: nothing to score (this benchmark only scores recorded results).",
              file=sys.stderr)
        return 2
    try:
        records = load_records(args.results)
    except OSError as e:
        print(f"Cannot read results file: {e}", file=sys.stderr)
        return 1
    try:
        report = run_benchmark(cases, records, config)
    except ValueError as e:
        print(f"Invalid option: {e}", file=sys.stderr)
        return 2

    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        for line in summary_lines(report):
            print(line)
        print(f"Report written to {args.out}")
    else:
        print(text)
    problems = (report["input_errors"] or report["unmatched_results"] or report["runs_without_results"]
                or any(r["cases_without_result"] for r in report["runs"]))
    return 1 if args.strict and problems else 0


if __name__ == "__main__":
    sys.exit(main())
