"""Evaluate collected responses against the answer-evaluation dataset (Sprint 3 CLI).

Run from the repository root:
    python -m eval.answer_eval.run_evaluation --responses responses.jsonl --out report.json

Responses file: JSON Lines, one object per line:
    {"id": "df-01", "response": "...", "metadata": {...}, "retrieved_chunks": [...]}
"metadata" and "retrieved_chunks" are optional. A null "response" means no response was collected.

The dataset (and its evidence quotes against the synthetic corpus) is validated first; an invalid
dataset stops the run with exit code 2. Exit codes: 0 report written, 1 unreadable responses file,
2 invalid dataset. Makes no network or model calls.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.answer_eval.evaluation_engine import evaluate_batch  # noqa: E402
from eval.answer_eval.validate_dataset import CORPUS_DIR, DATASET_PATH, validate  # noqa: E402


def read_responses(path: Path) -> tuple[list[tuple[str, object]], dict, dict, list[dict]]:
    """Return (id/response pairs, metadata by id, retrieved chunks by id, input errors)."""
    pairs, metadata, chunks, errors = [], {}, {}, []
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append({"line": n, "error": f"invalid JSON: {e.msg}"})
                continue
            if not isinstance(obj, dict) or not isinstance(obj.get("id"), str) or "response" not in obj:
                errors.append({"line": n, "error": "each line needs a string 'id' and a 'response' key"})
                continue
            pairs.append((obj["id"], obj["response"]))
            if isinstance(obj.get("metadata"), dict):
                metadata[obj["id"]] = obj["metadata"]
            if isinstance(obj.get("retrieved_chunks"), list):
                chunks[obj["id"]] = obj["retrieved_chunks"]
    return pairs, metadata, chunks, errors


def summary_lines(report: dict) -> list[str]:
    lines = [f"Items: {report['dataset']['items']} (draft {report['dataset']['draft']}, "
             f"reviewed {report['dataset']['reviewed']})" + ("  PROVISIONAL" if report["provisional"] else "")]
    lines.append("Overall: " + ", ".join(f"{k} {v}" for k, v in report["overall_status_counts"].items()))
    for dim, counts in report["dimension_status_counts"].items():
        lines.append(f"  {dim}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    if report["not_responded"]:
        lines.append(f"No response: {len(report['not_responded'])}")
    if report["input_errors"]:
        lines.append(f"Input errors: {len(report['input_errors'])}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, default=DATASET_PATH)
    ap.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    ap.add_argument("--responses", type=Path, required=True, help="JSONL file of responses")
    ap.add_argument("--out", type=Path, help="write the JSON report here (default: stdout)")
    ap.add_argument("--reviewed-only", action="store_true", help="evaluate reviewed items only")
    args = ap.parse_args(argv)

    v = validate(args.dataset, args.corpus_dir)
    if v["errors"]:
        print(f"Dataset invalid ({len(v['errors'])} error(s)); not evaluating:", file=sys.stderr)
        for e in v["errors"]:
            print(f"  - {e}", file=sys.stderr)
        return 2

    try:
        pairs, metadata, chunks, line_errors = read_responses(args.responses)
    except OSError as e:
        print(f"Cannot read responses file: {e}", file=sys.stderr)
        return 1

    report = evaluate_batch(v["items"], pairs, reviewed_only=args.reviewed_only, metadata=metadata,
                            retrieved_chunks=chunks, dataset_path=str(args.dataset)).to_dict()
    report["input_errors"] = line_errors + report["input_errors"]

    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        for line in summary_lines(report):
            print(line)
        print(f"Report written to {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
