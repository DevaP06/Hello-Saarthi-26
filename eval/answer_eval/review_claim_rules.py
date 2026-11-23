"""Inspect the forbidden-claim rule registry (claim_rules.jsonl) against the dataset. Read-only.

    python eval/answer_eval/review_claim_rules.py                 # from the repo root
    python eval/answer_eval/review_claim_rules.py --item ic-02    # one item only
    python eval/answer_eval/review_claim_rules.py --status draft  # rules with one status
    python eval/answer_eval/review_claim_rules.py --brief         # counts only, no per-claim lists

Reports: total forbidden claims, claims with no registry record, claims grouped by feasibility,
rules grouped by status, must_not coverage records and uncovered must_not entries, and every
registry validation error. Exit code 1 if the dataset or registry has errors, else 0.

This tool never approves, rejects or edits anything; reviewers change claim_rules.jsonl by hand
and the change goes through a normal pull request. No network, no model, no application code.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.answer_eval.claim_rules import (  # noqa: E402
    FEASIBILITY, REGISTRY_PATH, RULE_STATUSES, dataset_claims, dataset_must_not, load_registry,
)
from eval.answer_eval.schema import load_dataset  # noqa: E402

DATASET_PATH = Path(__file__).resolve().parent / "dataset.jsonl"
UNASSESSED = "unassessed"
CONFLICTING = "conflicting"


def claim_feasibility(rules) -> str:
    """One claim's feasibility from its rule records: the shared value, conflicting, or unassessed."""
    values = {r.feasibility for r in rules}
    if not values:
        return UNASSESSED
    return values.pop() if len(values) == 1 else CONFLICTING


def report(dataset: Path, registry_path: Path, *, item: str | None = None, status: str | None = None,
           brief: bool = False) -> tuple[list[str], int]:
    """Build the report lines and the exit code. Pure apart from reading the two files."""
    items, ds_errors = load_dataset(dataset)
    out: list[str] = []
    if ds_errors:
        out.append(f"Dataset {dataset} has {len(ds_errors)} error(s); fix it first "
                   f"(python eval/answer_eval/validate_dataset.py):")
        out += [f"  - {e}" for e in ds_errors]
        return out, 1

    reg = load_registry(registry_path, items)
    claims = dataset_claims(items)
    must_not = dataset_must_not(items)
    if item:
        if item not in {i.id for i in items}:
            return [f"Unknown item id '{item}'."], 1
        claims = {k: v for k, v in claims.items() if v[0].id == item}
        must_not = {k: v for k, v in must_not.items() if v[0].id == item}
    rules = [r for r in reg.rules if not item or r.item_id == item]

    reasons = Counter(v[3] for v in claims.values())
    out.append(f"Dataset: {dataset}")
    out.append(f"Registry: {registry_path}" + (f"  (item {item})" if item else ""))
    out.append(f"Forbidden claims: {len(claims)} ("
               + ", ".join(f"{r} {reasons[r]}" for r in sorted(reasons)) + ")")
    out.append(f"must_not entries: {len(must_not)}")
    out.append(f"Registry records: {len(reg.rules)} rule(s), {len(reg.coverage)} coverage record(s)"
               + ("" if reg.valid else "  [only valid records are counted]"))

    by_status = Counter(r.status for r in rules)
    out.append("\nRules by status: " + ", ".join(f"{s} {by_status.get(s, 0)}" for s in RULE_STATUSES))
    if status:
        chosen = [r for r in rules if r.status == status]
        out.append(f"  {status}: " + (", ".join(r.rule_id for r in chosen) if chosen else "none"))
    elif not brief:
        for r in rules:
            out.append(f"  {r.rule_id}  {r.status:<8} {r.feasibility:<14} "
                       f"reviewer={r.reviewer or '-'}  {r.claim_text}")

    groups: dict[str, list[str]] = {}
    for cid in claims:
        groups.setdefault(claim_feasibility(reg.rules_for_claim(cid)), []).append(cid)
    order = list(FEASIBILITY) + [CONFLICTING, UNASSESSED]
    out.append("\nClaims by feasibility: " + ", ".join(f"{g} {len(groups.get(g, []))}" for g in order))
    if not brief:
        for g in order[:-1]:
            if groups.get(g):
                out.append(f"  {g}: {', '.join(groups[g])}")

    unrecorded = [cid for cid in claims if not reg.rules_for_claim(cid)]
    out.append(f"\nClaims with no registry record: {len(unrecorded)} of {len(claims)}")
    if not brief:
        for cid in unrecorded:
            _, _, text, reason = claims[cid]
            out.append(f"  {cid:<10} [{reason}] {text}")

    covered = {c.coverage_id: c for c in reg.coverage}
    out.append(f"\nmust_not coverage (traceability only; never scored): "
               f"{sum(1 for k in must_not if k in covered)} recorded, "
               f"{sum(1 for k in must_not if k not in covered)} uncovered")
    if not brief:
        for k, (_, _, text) in must_not.items():
            c = covered.get(k)
            what = (f"covered_by={c.covered_by}" + (f" {','.join(c.claim_ids)}" if c.claim_ids else "")
                    if c else "UNCOVERED")
            out.append(f"  {k:<10} {what:<36} {text}")

    if reg.errors:
        out.append(f"\nRegistry INVALID: {len(reg.errors)} error(s). Approved rules are ignored until fixed.")
        out += [f"  - {e}" for e in reg.errors]
        return out, 1
    out.append("\nRegistry OK.")
    return out, 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="Read-only: reviewers edit claim_rules.jsonl by hand. Exit 1 on dataset or registry errors.")
    ap.add_argument("--dataset", type=Path, default=DATASET_PATH, help="dataset JSONL (default: %(default)s)")
    ap.add_argument("--registry", type=Path, default=REGISTRY_PATH, help="registry JSONL (default: %(default)s)")
    ap.add_argument("--item", help="show one item only, e.g. ic-02")
    ap.add_argument("--status", choices=RULE_STATUSES, help="list the rules that have this status")
    ap.add_argument("--brief", action="store_true", help="counts only, no per-claim or per-rule lists")
    args = ap.parse_args(argv)
    lines, code = report(args.dataset, args.registry, item=args.item, status=args.status, brief=args.brief)
    print("\n".join(lines))
    return code


if __name__ == "__main__":
    sys.exit(main())
