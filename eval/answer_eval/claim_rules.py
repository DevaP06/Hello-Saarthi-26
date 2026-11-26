"""Sprint 4 Phase 1: registry of forbidden-claim detection rules and must_not coverage records.

The registry (claim_rules.jsonl, JSON Lines) is kept separate from dataset.jsonl so the dataset
schema stays unchanged. Phase 1 is infrastructure only: nothing here is used by the evaluation
engine, and no record changes any evaluation status.

ID conventions (derived from dataset position, one-based, in the dataset's existing order):
    claim_id      "{item_id}#c{n}"     the n-th entry of the item's forbidden_claims   e.g. df-04#c2
    rule_id       "{claim_id}/r{k}"    the k-th rule written for that claim            e.g. df-04#c2/r1
    coverage_id   "{item_id}#m{n}"     the n-th entry of the item's must_not          e.g. ic-02#m1

Because IDs come from positions, every record also stores an exact copy of the claim (text and
reason) or must_not text. Validation compares that copy with the current dataset, so an edited,
re-ordered or removed dataset entry is reported instead of being silently linked to an old record.

Record types:
    forbidden_claim_rule   a (possibly future) detection rule for one forbidden claim
    must_not_coverage      traceability only: records what already covers a must_not entry
                           (Sprint 3 behaviour, a forbidden claim, or nothing). It is never scored.

Only rules with status "approved" that pass every check are returned by approved_rules(); draft,
rejected and retired rules never are.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from eval.answer_eval.schema import CLAIM_REASONS, LANGUAGES, Item

__all__ = [
    "REGISTRY_PATH", "RECORD_TYPES", "FEASIBILITY", "RULE_STATUSES", "COVERED_BY",
    "RULE_FIELDS", "COVERAGE_FIELDS", "NEGATION_CUE",
    "ClaimRule", "MustNotCoverage", "Registry",
    "claim_id", "coverage_id", "rule_id", "parse_claim_id", "parse_coverage_id",
    "dataset_claims", "dataset_must_not", "parse_registry", "load_registry",
]

REGISTRY_PATH = Path(__file__).resolve().parent / "claim_rules.jsonl"

RULE_RECORD = "forbidden_claim_rule"
COVERAGE_RECORD = "must_not_coverage"
RECORD_TYPES = (RULE_RECORD, COVERAGE_RECORD)
FEASIBILITY = ("deterministic", "careful", "semantic", "underspecified")
RULE_STATUSES = ("draft", "approved", "rejected", "retired")
COVERED_BY = ("behaviour", "forbidden_claim", "none")

# Feasibility values for which a regex rule may be approved. Semantic and underspecified claims
# need a method other than a pattern, so an approved pattern for them is a contradiction.
APPROVABLE_FEASIBILITY = ("deterministic", "careful")
MIN_POSITIVE_EXAMPLES = 2
MIN_NEGATIVE_EXAMPLES = 3

# An approved rule needs at least one negative example that is a negation or false-positive case.
# This is a check on the examples a reviewer supplied; it is not a detection method.
NEGATION_CUE = re.compile(
    r"\b(?:not|no|never|none|neither|nor|without|incorrect|wrong|false|mistake|misconception|myth|"
    r"unlike|nahi|nahin|mat)\b|n['’]t\b", re.I)

RULE_FIELDS: dict[str, type | tuple[type, ...]] = {
    "record_type": str,
    "rule_id": str,
    "claim_id": str,
    "item_id": str,
    "claim_text": str,
    "reason": str,
    "feasibility": str,
    "pattern": (str, type(None)),
    "anchors": list,
    "exclusions": list,
    "languages": list,
    "status": str,
    "author": str,
    "reviewer": (str, type(None)),
    "reviewed_on": (str, type(None)),
    "rationale": str,
    "limitations": str,
    "positive_examples": list,
    "negative_examples": list,
}
COVERAGE_FIELDS: dict[str, type | tuple[type, ...]] = {
    "record_type": str,
    "coverage_id": str,
    "item_id": str,
    "must_not_text": str,
    "covered_by": str,
    "claim_ids": list,
    "rationale": str,
    "author": str,
}

_ITEM = r"[a-z]{2}-\d{2}"
_CLAIM_ID = re.compile(rf"^({_ITEM})#c([1-9]\d*)$")
_RULE_ID = re.compile(rf"^({_ITEM}#c[1-9]\d*)/r([1-9]\d*)$")
_COVERAGE_ID = re.compile(rf"^({_ITEM})#m([1-9]\d*)$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ── IDs ──────────────────────────────────────────────────────────────────────────────────────

def claim_id(item_id: str, n: int) -> str:
    """ID of the n-th (one-based) forbidden claim of an item."""
    return f"{item_id}#c{n}"


def rule_id(claim: str, k: int) -> str:
    return f"{claim}/r{k}"


def coverage_id(item_id: str, n: int) -> str:
    """ID of the n-th (one-based) must_not entry of an item."""
    return f"{item_id}#m{n}"


def parse_claim_id(cid: str) -> tuple[str, int] | None:
    m = _CLAIM_ID.match(cid) if isinstance(cid, str) else None
    return (m.group(1), int(m.group(2))) if m else None


def parse_coverage_id(cov: str) -> tuple[str, int] | None:
    m = _COVERAGE_ID.match(cov) if isinstance(cov, str) else None
    return (m.group(1), int(m.group(2))) if m else None


def dataset_claims(items: Sequence[Item]) -> dict[str, tuple[Item, int, str, str]]:
    """{claim_id: (item, position, text, reason)} for every forbidden claim, in dataset order."""
    return {claim_id(it.id, n): (it, n, c.text, c.reason)
            for it in items for n, c in enumerate(it.forbidden_claims, 1)}


def dataset_must_not(items: Sequence[Item]) -> dict[str, tuple[Item, int, str]]:
    """{coverage_id: (item, position, text)} for every must_not entry, in dataset order."""
    return {coverage_id(it.id, n): (it, n, text) for it in items for n, text in enumerate(it.must_not, 1)}


# ── Records ──────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ClaimRule:
    rule_id: str
    claim_id: str
    item_id: str
    claim_text: str
    reason: str
    feasibility: str
    pattern: str | None
    anchors: tuple[str, ...]
    exclusions: tuple[str, ...]
    languages: tuple[str, ...]
    status: str
    author: str
    reviewer: str | None
    reviewed_on: str | None
    rationale: str
    limitations: str
    positive_examples: tuple[str, ...]
    negative_examples: tuple[str, ...]
    line: int = 0

    @property
    def is_approved(self) -> bool:
        return self.status == "approved"


@dataclass(frozen=True)
class MustNotCoverage:
    """Traceability only. Has no status, score or effect on any evaluation dimension."""
    coverage_id: str
    item_id: str
    must_not_text: str
    covered_by: str
    claim_ids: tuple[str, ...]
    rationale: str
    author: str
    line: int = 0


@dataclass
class Registry:
    rules: list[ClaimRule] = field(default_factory=list)
    coverage: list[MustNotCoverage] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    path: str | None = None

    @property
    def valid(self) -> bool:
        return not self.errors

    def approved_rules(self) -> list[ClaimRule]:
        """Approved rules of a fully valid registry; [] if the registry has any error.

        Draft, rejected and retired rules are never returned. Phase 1 has no consumer of this list.
        """
        if self.errors:
            return []
        return [r for r in self.rules if r.status == "approved"]

    def rules_for_claim(self, cid: str) -> list[ClaimRule]:
        return [r for r in self.rules if r.claim_id == cid]

    def coverage_for(self, cov_id: str) -> MustNotCoverage | None:
        return next((c for c in self.coverage if c.coverage_id == cov_id), None)


# ── Validation ───────────────────────────────────────────────────────────────────────────────

def _type_name(t) -> str:
    if isinstance(t, tuple):
        return " or ".join("null" if x is type(None) else x.__name__ for x in t)
    return t.__name__


def _check_fields(raw: dict, spec: dict, where: str) -> list[str]:
    errors = []
    for name, typ in spec.items():
        if name not in raw:
            errors.append(f"{where}: missing field '{name}'")
        elif not isinstance(raw[name], typ) or isinstance(raw[name], bool):
            errors.append(f"{where}: '{name}' must be {_type_name(typ)}")
    for name in sorted(set(raw) - set(spec)):
        errors.append(f"{where}: unknown field '{name}'")
    return errors


def _string_list(raw: dict, name: str, where: str) -> list[str]:
    values = raw.get(name)
    if isinstance(values, list) and not all(isinstance(v, str) and v.strip() for v in values):
        return [f"{where}: '{name}' must contain non-empty strings only"]
    return []


def _compiles(pattern: str, what: str, where: str) -> list[str]:
    try:
        re.compile(pattern)
    except re.error as e:
        return [f"{where}: {what} is not a valid regex ({e})"]
    return []


def _claim_drift(cid: str, text: str, reason: str, item: Item | None, n: int,
                 claims: dict, where: str) -> list[str]:
    """Compare a record's copy of a claim with the dataset; explain any drift precisely."""
    if item is None:
        return []
    if cid not in claims:
        return [f"{where}: unknown claim id '{cid}' ({item.id} has {len(item.forbidden_claims)} "
                f"forbidden claim(s))"]
    _, _, cur_text, cur_reason = claims[cid]
    if cur_text == text:
        if cur_reason != reason:
            return [f"{where}: claim reason drift for {cid}: registry says '{reason}', "
                    f"dataset says '{cur_reason}'"]
        return []
    moved = [m for m, c in enumerate(item.forbidden_claims, 1) if c.text == text]
    if moved:
        return [f"{where}: claim position drift for {cid}: this text is now claim "
                f"{', '.join(claim_id(item.id, m) for m in moved)}; position {n} holds {cur_text!r}"]
    return [f"{where}: claim text drift for {cid}: registry has {text!r}, dataset has {cur_text!r}"]


def _check_rule(raw: dict, where: str, by_id: dict[str, Item], claims: dict,
                today: _dt.date) -> list[str]:
    errors = _check_fields(raw, RULE_FIELDS, where)
    if errors:
        return errors
    for name in ("anchors", "exclusions", "languages", "positive_examples", "negative_examples"):
        errors += _string_list(raw, name, where)
    for name in ("rule_id", "claim_id", "item_id", "claim_text", "author"):
        if not raw[name].strip():
            errors.append(f"{where}: '{name}' must not be empty")

    if raw["feasibility"] not in FEASIBILITY:
        errors.append(f"{where}: 'feasibility' must be one of {', '.join(FEASIBILITY)}")
    if raw["status"] not in RULE_STATUSES:
        errors.append(f"{where}: 'status' must be one of {', '.join(RULE_STATUSES)}")
    if raw["reason"] not in CLAIM_REASONS:
        errors.append(f"{where}: 'reason' must be one of {', '.join(CLAIM_REASONS)}")
    bad_langs = [x for x in raw["languages"] if x not in LANGUAGES]
    if bad_langs or not raw["languages"]:
        errors.append(f"{where}: 'languages' must be a non-empty subset of {', '.join(LANGUAGES)}")

    rid = _RULE_ID.match(raw["rule_id"])
    parsed = parse_claim_id(raw["claim_id"])
    if not parsed:
        errors.append(f"{where}: 'claim_id' must look like 'df-04#c2' (item id, '#c', one-based position)")
    if not rid:
        errors.append(f"{where}: 'rule_id' must look like '<claim_id>/r1'")
    elif rid.group(1) != raw["claim_id"]:
        errors.append(f"{where}: 'rule_id' must start with its claim_id '{raw['claim_id']}'")

    item = by_id.get(raw["item_id"])
    if item is None:
        errors.append(f"{where}: unknown item id '{raw['item_id']}'")
    if parsed:
        if parsed[0] != raw["item_id"]:
            errors.append(f"{where}: claim_id '{raw['claim_id']}' does not belong to item '{raw['item_id']}'")
        elif item is not None:
            errors += _claim_drift(raw["claim_id"], raw["claim_text"], raw["reason"], item, parsed[1],
                                   claims, where)
    if item is not None and raw["languages"] and item.expected_answer_language not in raw["languages"]:
        errors.append(f"{where}: 'languages' must include the item's answer language "
                      f"'{item.expected_answer_language}'")

    if raw["pattern"] is not None:
        if not raw["pattern"].strip():
            errors.append(f"{where}: 'pattern' must be null or a non-empty regex")
        else:
            errors += _compiles(raw["pattern"], "'pattern'", where)
    for name in ("anchors", "exclusions"):
        for i, p in enumerate(raw[name]):
            if isinstance(p, str) and p.strip():
                errors += _compiles(p, f"'{name}[{i}]'", where)

    if raw["reviewed_on"] is not None:
        errors += _check_date(raw["reviewed_on"], where, today)

    if raw["status"] == "approved":
        errors += _check_approved(raw, where)
    return errors


def _check_date(value: str, where: str, today: _dt.date) -> list[str]:
    if not _DATE.match(value):
        return [f"{where}: 'reviewed_on' must be an ISO date YYYY-MM-DD"]
    try:
        d = _dt.date.fromisoformat(value)
    except ValueError:
        return [f"{where}: 'reviewed_on' is not a real date ({value})"]
    if d > today:
        return [f"{where}: 'reviewed_on' is in the future ({value})"]
    return []


def _check_approved(raw: dict, where: str) -> list[str]:
    errors = []
    if raw["pattern"] is None or not raw["pattern"].strip():
        errors.append(f"{where}: an approved rule needs a 'pattern'")
    if raw["feasibility"] not in APPROVABLE_FEASIBILITY:
        errors.append(f"{where}: a '{raw['feasibility']}' claim cannot have an approved pattern rule "
                      f"(only {', '.join(APPROVABLE_FEASIBILITY)})")
    reviewer = (raw["reviewer"] or "").strip()
    if not reviewer:
        errors.append(f"{where}: an approved rule needs a 'reviewer'")
    elif reviewer.casefold() == raw["author"].strip().casefold():
        errors.append(f"{where}: an approved rule needs a reviewer other than its author")
    if raw["reviewed_on"] is None:
        errors.append(f"{where}: an approved rule needs 'reviewed_on'")
    if len(raw["positive_examples"]) < MIN_POSITIVE_EXAMPLES:
        errors.append(f"{where}: an approved rule needs at least {MIN_POSITIVE_EXAMPLES} positive examples")
    if len(raw["negative_examples"]) < MIN_NEGATIVE_EXAMPLES:
        errors.append(f"{where}: an approved rule needs at least {MIN_NEGATIVE_EXAMPLES} negative examples")
    if not any(isinstance(x, str) and NEGATION_CUE.search(x) for x in raw["negative_examples"]):
        errors.append(f"{where}: an approved rule needs a negative example that negates or corrects the "
                      f"claim (e.g. 'not ...', 'it is wrong to say ...')")
    if not raw["rationale"].strip():
        errors.append(f"{where}: an approved rule needs a 'rationale'")
    return errors


def _check_coverage(raw: dict, where: str, by_id: dict[str, Item], must_not: dict,
                    claims: dict) -> list[str]:
    errors = _check_fields(raw, COVERAGE_FIELDS, where)
    if errors:
        return errors
    errors += _string_list(raw, "claim_ids", where)
    for name in ("coverage_id", "item_id", "must_not_text", "author"):
        if not raw[name].strip():
            errors.append(f"{where}: '{name}' must not be empty")
    if raw["covered_by"] not in COVERED_BY:
        errors.append(f"{where}: 'covered_by' must be one of {', '.join(COVERED_BY)}")

    parsed = parse_coverage_id(raw["coverage_id"])
    item = by_id.get(raw["item_id"])
    if not parsed:
        errors.append(f"{where}: 'coverage_id' must look like 'ic-02#m1' (item id, '#m', one-based position)")
    if item is None:
        errors.append(f"{where}: unknown item id '{raw['item_id']}'")
    if parsed and parsed[0] != raw["item_id"]:
        errors.append(f"{where}: coverage_id '{raw['coverage_id']}' does not belong to item '{raw['item_id']}'")
    elif parsed and item is not None:
        cov = raw["coverage_id"]
        if cov not in must_not:
            errors.append(f"{where}: unknown coverage id '{cov}' ({item.id} has {len(item.must_not)} "
                          f"must_not entr{'y' if len(item.must_not) == 1 else 'ies'})")
        elif must_not[cov][2] != raw["must_not_text"]:
            moved = [m for m, t in enumerate(item.must_not, 1) if t == raw["must_not_text"]]
            if moved:
                errors.append(f"{where}: must_not position drift for {cov}: this text is now "
                              f"{', '.join(coverage_id(item.id, m) for m in moved)}")
            else:
                errors.append(f"{where}: must_not text drift for {cov}: registry has {raw['must_not_text']!r}, "
                              f"dataset has {must_not[cov][2]!r}")

    if raw["covered_by"] == "forbidden_claim":
        if not raw["claim_ids"]:
            errors.append(f"{where}: covered_by 'forbidden_claim' needs at least one claim id")
        for cid in raw["claim_ids"]:
            p = parse_claim_id(cid) if isinstance(cid, str) else None
            if not p:
                errors.append(f"{where}: claim id '{cid}' is malformed")
            elif cid not in claims:
                errors.append(f"{where}: unknown claim id '{cid}'")
            elif p[0] != raw["item_id"]:
                errors.append(f"{where}: claim id '{cid}' belongs to another item")
    elif raw["claim_ids"]:
        errors.append(f"{where}: 'claim_ids' must be empty unless covered_by is 'forbidden_claim'")
    return errors


def parse_registry(lines: Sequence[str], items: Sequence[Item], *, source: str = "registry",
                   today: _dt.date | None = None) -> Registry:
    """Parse and validate registry lines against the current dataset items.

    Blank lines are allowed (an empty registry is valid). Every error names the line and, when
    known, the rule or coverage id. A registry with errors returns no approved rules.
    """
    today = today or _dt.date.today()
    by_id = {it.id: it for it in items}
    claims = dataset_claims(items)
    must_not = dataset_must_not(items)
    reg = Registry(path=source)
    seen_rules: dict[str, int] = {}
    seen_cov: dict[str, int] = {}

    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as e:
            reg.errors.append(f"{source}:{n}: invalid JSON ({e.msg})")
            continue
        if not isinstance(raw, dict):
            reg.errors.append(f"{source}:{n}: each line must be a JSON object")
            continue
        rtype = raw.get("record_type")
        if rtype not in RECORD_TYPES:
            reg.errors.append(f"{source}:{n}: 'record_type' must be one of {', '.join(RECORD_TYPES)}, "
                              f"got {rtype!r}")
            continue

        if rtype == RULE_RECORD:
            rid = raw.get("rule_id") if isinstance(raw.get("rule_id"), str) else "?"
            where = f"{source}:{n} [{rid}]"
            errs = _check_rule(raw, where, by_id, claims, today)
            if rid in seen_rules:
                errs.append(f"{where}: duplicate rule_id (first on line {seen_rules[rid]})")
            seen_rules.setdefault(rid, n)
            reg.errors += errs
            if not errs:
                reg.rules.append(ClaimRule(
                    rule_id=raw["rule_id"], claim_id=raw["claim_id"], item_id=raw["item_id"],
                    claim_text=raw["claim_text"], reason=raw["reason"], feasibility=raw["feasibility"],
                    pattern=raw["pattern"], anchors=tuple(raw["anchors"]), exclusions=tuple(raw["exclusions"]),
                    languages=tuple(raw["languages"]), status=raw["status"], author=raw["author"],
                    reviewer=raw["reviewer"], reviewed_on=raw["reviewed_on"], rationale=raw["rationale"],
                    limitations=raw["limitations"], positive_examples=tuple(raw["positive_examples"]),
                    negative_examples=tuple(raw["negative_examples"]), line=n,
                ))
        else:
            cov = raw.get("coverage_id") if isinstance(raw.get("coverage_id"), str) else "?"
            where = f"{source}:{n} [{cov}]"
            errs = _check_coverage(raw, where, by_id, must_not, claims)
            if cov in seen_cov:
                errs.append(f"{where}: duplicate coverage_id (first on line {seen_cov[cov]})")
            seen_cov.setdefault(cov, n)
            reg.errors += errs
            if not errs:
                reg.coverage.append(MustNotCoverage(
                    coverage_id=raw["coverage_id"], item_id=raw["item_id"],
                    must_not_text=raw["must_not_text"], covered_by=raw["covered_by"],
                    claim_ids=tuple(raw["claim_ids"]), rationale=raw["rationale"], author=raw["author"],
                    line=n,
                ))
    return reg


def load_registry(path: Path | str, items: Sequence[Item], *, today: _dt.date | None = None) -> Registry:
    """Read and validate a registry file. A missing file is an error, not an empty registry."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        return Registry(errors=[f"{path.name}: cannot read registry ({e.strerror or e})"], path=str(path))
    return parse_registry(text.splitlines(), items, source=path.name, today=today)
