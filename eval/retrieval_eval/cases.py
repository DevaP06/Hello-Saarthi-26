"""Retrieval benchmark cases: schema, loader, validator and the Sprint 1 derivation.

A case file is JSON Lines, one case per line (blank lines allowed):

  {"id": "df-01", "question": "...", "language": "en", "category": "direct_factual",
   "source": "sprint1_dataset",
   "labels": {
     "document": {"status": "unverified", "completeness": "partial",
                  "grades": {"EE210 Module 1 Signals and Sampling": 1},
                  "provenance": "...", "reviewer": null, "reviewed_on": null},
     "page":     {... "grades": {"EE210 Module 1 Signals and Sampling#p3": 1} ...},
     "chunk":    null
   },
   "metadata": {...}}

Levels   document | page | chunk. A level that is absent or null has NO labels: its metrics are
         not_evaluated (never scored as if nothing were relevant).
Page ids are "<document id>#p<page>" (1-based page).
status   verified    judged by a person; needs "reviewer" and a non-future ISO "reviewed_on"
         unverified  derived or drafted, not yet judged by a person
         fixture     example data for tests and documentation only; never ground truth
completeness  exhaustive  every unlisted item is judged not relevant (verified or fixture labels only)
              partial     unlisted items are unjudged
grades   {item id: integer 0..3}; 0 = judged not relevant.
source   sprint1_dataset | manual | fixture. A fixture case must have fixture labels only, and
         fixture labels may only appear in a fixture case, so example data cannot be mixed into
         ground truth.

The Sprint 1 derivation (cases_from_answer_items) reads the answer-evaluation items without
changing them:
  - answer/clarify items: every evidence (material, page) becomes a relevant page (grade 1) and its
    material a relevant document (grade 1). Sprint 1's validator checks that each quote occurs on its
    stated page, but nobody has judged which OTHER pages are relevant, and the dataset labels are
    draft, so both levels are status "unverified" with completeness "partial".
  - abstain items: no labelled relevant item (completeness "partial", empty grades), so every metric
    is not_applicable; the case still reports how much the retriever returned.
  - chunk level: no labels. Sprint 1 records materials and pages only, and chunk ids are specific to
    one index build.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from eval.retrieval_eval.metrics import Judgements

LEVELS = ("document", "page", "chunk")
LABEL_STATUSES = ("verified", "unverified", "fixture")
COMPLETENESS = ("exhaustive", "partial")
CASE_SOURCES = ("sprint1_dataset", "manual", "fixture")
LANGUAGES = ("en", "hinglish")
MAX_GRADE = 3

ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}$")
PAGE_ID_RE = re.compile(r"^.+#p[1-9][0-9]*$")
_LABEL_KEYS = {"status", "completeness", "grades", "provenance"}
_LABEL_OPTIONAL = {"reviewer", "reviewed_on"}
_CASE_KEYS = {"id", "question", "language", "category", "source", "labels"}
_CASE_OPTIONAL = {"metadata"}

SPRINT1_PROVENANCE = (
    "derived from Sprint 1 answer-evaluation evidence: each listed page holds an evidence quote that "
    "validate_dataset verifies on that page; other pages are unjudged; dataset labels are draft and "
    "retrieval relevance has not been judged by a person")
SPRINT1_ABSTAIN_PROVENANCE = (
    "derived from a Sprint 1 abstain item: the dataset says the corpus cannot answer it (absent_terms "
    "verified absent); no item is labelled relevant and retrieval relevance has not been judged by a person")


def page_id(document_id: str, page: int) -> str:
    return f"{document_id}#p{page}"


@dataclass(frozen=True)
class LevelLabels:
    level: str
    status: str
    completeness: str
    grades: Mapping[str, int]
    provenance: str
    reviewer: str | None = None
    reviewed_on: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "grades", MappingProxyType(dict(self.grades)))

    def judgements(self) -> Judgements:
        return Judgements(self.grades, self.completeness == "exhaustive")

    def to_dict(self) -> dict:
        return {"status": self.status, "completeness": self.completeness,
                "grades": dict(sorted(self.grades.items())), "provenance": self.provenance,
                "reviewer": self.reviewer, "reviewed_on": self.reviewed_on}


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    question: str
    language: str
    category: str
    source: str
    labels: Mapping[str, LevelLabels]          # only levels that HAVE labels
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", MappingProxyType(dict(self.labels)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict:
        return {"id": self.id, "question": self.question, "language": self.language,
                "category": self.category, "source": self.source,
                "labels": {lv: (self.labels[lv].to_dict() if lv in self.labels else None) for lv in LEVELS},
                "metadata": dict(self.metadata)}


def _is_iso_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return _dt.date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def _parse_labels(level: str, raw: Any, where: str, today: _dt.date) -> tuple[LevelLabels | None, list[str]]:
    w = f"{where} labels.{level}"
    if raw is None:
        return None, []
    if not isinstance(raw, dict):
        return None, [f"{w}: must be an object or null"]
    errors: list[str] = []
    missing = _LABEL_KEYS - set(raw)
    extra = set(raw) - _LABEL_KEYS - _LABEL_OPTIONAL
    if missing:
        errors.append(f"{w}: missing field(s) {sorted(missing)}")
    if extra:
        errors.append(f"{w}: unknown field(s) {sorted(extra)}")
    if errors:
        return None, errors
    status, completeness, grades = raw["status"], raw["completeness"], raw["grades"]
    if status not in LABEL_STATUSES:
        errors.append(f"{w}: status must be one of {LABEL_STATUSES}")
    if completeness not in COMPLETENESS:
        errors.append(f"{w}: completeness must be one of {COMPLETENESS}")
    if not isinstance(raw["provenance"], str) or not raw["provenance"].strip():
        errors.append(f"{w}: provenance must be a non-empty string")
    reviewer, reviewed_on = raw.get("reviewer"), raw.get("reviewed_on")
    if status == "verified":
        if not isinstance(reviewer, str) or not reviewer.strip():
            errors.append(f"{w}: verified labels need a reviewer")
        if not _is_iso_date(reviewed_on):
            errors.append(f"{w}: verified labels need reviewed_on as an ISO date (YYYY-MM-DD)")
        elif _dt.date.fromisoformat(reviewed_on) > today:
            errors.append(f"{w}: reviewed_on {reviewed_on} is in the future")
    else:
        if reviewer is not None or reviewed_on is not None:
            errors.append(f"{w}: reviewer/reviewed_on are only allowed on verified labels")
    if not isinstance(grades, dict):
        errors.append(f"{w}: grades must be an object of item id -> grade")
        grades = {}
    for item_id, g in grades.items():
        if not item_id or not item_id.strip():
            errors.append(f"{w}: empty item id in grades")
        elif level == "page" and not PAGE_ID_RE.match(item_id):
            errors.append(f"{w}: page id {item_id!r} must look like '<document id>#p<page>'")
        if isinstance(g, bool) or not isinstance(g, int) or not 0 <= g <= MAX_GRADE:
            errors.append(f"{w}: grade for {item_id!r} must be an integer 0..{MAX_GRADE}")
    if completeness == "exhaustive" and status == "unverified":
        errors.append(f"{w}: unverified labels cannot be exhaustive: that would score every unlisted item "
                      f"as not relevant without anyone having judged it; use completeness 'partial'")
    if errors:
        return None, errors
    return LevelLabels(level, status, completeness, grades, raw["provenance"], reviewer, reviewed_on), []


def parse_case(raw: Any, where: str = "case", *, today: _dt.date | None = None) -> tuple[BenchmarkCase | None, list[str]]:
    today = today or _dt.date.today()
    if not isinstance(raw, dict):
        return None, [f"{where}: must be a JSON object"]
    errors: list[str] = []
    missing = _CASE_KEYS - set(raw)
    extra = set(raw) - _CASE_KEYS - _CASE_OPTIONAL
    if missing:
        errors.append(f"{where}: missing field(s) {sorted(missing)}")
    if extra:
        errors.append(f"{where}: unknown field(s) {sorted(extra)}")
    if errors:
        return None, errors
    cid = raw["id"]
    if not isinstance(cid, str) or not ID_RE.match(cid):
        errors.append(f"{where}: id must match {ID_RE.pattern}")
    w = f"{where} [{cid}]" if isinstance(cid, str) else where
    for key in ("question", "category"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            errors.append(f"{w}: {key} must be a non-empty string")
    if raw["language"] not in LANGUAGES:
        errors.append(f"{w}: language must be one of {LANGUAGES} (Sprint 5 keeps the English/Hinglish scope)")
    if raw["source"] not in CASE_SOURCES:
        errors.append(f"{w}: source must be one of {CASE_SOURCES}")
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        errors.append(f"{w}: metadata must be an object")
    labels_raw = raw["labels"]
    labels: dict[str, LevelLabels] = {}
    if not isinstance(labels_raw, dict):
        errors.append(f"{w}: labels must be an object")
    else:
        unknown = set(labels_raw) - set(LEVELS)
        if unknown:
            errors.append(f"{w}: unknown label level(s) {sorted(unknown)}")
        for level in LEVELS:
            parsed, errs = _parse_labels(level, labels_raw.get(level), w, today)
            errors += errs
            if parsed is not None:
                labels[level] = parsed
    if not errors:
        statuses = {lab.status for lab in labels.values()}
        if raw["source"] == "fixture" and statuses - {"fixture"}:
            errors.append(f"{w}: a fixture case may only carry fixture labels")
        if raw["source"] != "fixture" and "fixture" in statuses:
            errors.append(f"{w}: fixture labels may only appear in a fixture case")
    if errors:
        return None, errors
    return BenchmarkCase(cid, raw["question"], raw["language"], raw["category"], raw["source"], labels,
                         metadata), []


def parse_cases(rows: Iterable[tuple[str, Any]], *, today: _dt.date | None = None) -> tuple[list[BenchmarkCase], list[str]]:
    """rows: (where, raw object). Returns (cases, errors); duplicate ids are errors."""
    cases: list[BenchmarkCase] = []
    errors: list[str] = []
    seen: dict[str, str] = {}
    for where, raw in rows:
        case, errs = parse_case(raw, where, today=today)
        errors += errs
        if case is None:
            continue
        if case.id in seen:
            errors.append(f"{where}: duplicate case id {case.id!r} (first at {seen[case.id]})")
            continue
        seen[case.id] = where
        cases.append(case)
    return cases, errors


def load_cases(path: Path | str, *, today: _dt.date | None = None) -> tuple[list[BenchmarkCase], list[str]]:
    rows: list[tuple[str, Any]] = []
    errors: list[str] = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append((f"line {n}", json.loads(line)))
        except json.JSONDecodeError as e:
            errors.append(f"line {n}: invalid JSON ({e})")
    cases, errs = parse_cases(rows, today=today)
    return cases, errors + errs


def cases_from_answer_items(items: Sequence) -> list[BenchmarkCase]:
    """Derive retrieval cases from Sprint 1 answer-evaluation items (read-only; see module docstring)."""
    cases = []
    for item in items:
        meta = {"expected_behavior": item.expected_behavior, "difficulty": item.difficulty,
                "answer_label_status": item.label_status, "scope": dict(item.scope),
                "derived_from": "eval/answer_eval/dataset.jsonl"}
        if item.evidence:
            docs = {ev.material: 1 for ev in item.evidence}
            pages = {page_id(ev.material, ev.page): 1 for ev in item.evidence}
            labels = {
                "document": LevelLabels("document", "unverified", "partial", docs, SPRINT1_PROVENANCE),
                "page": LevelLabels("page", "unverified", "partial", pages, SPRINT1_PROVENANCE),
            }
        else:
            labels = {lv: LevelLabels(lv, "unverified", "partial", {}, SPRINT1_ABSTAIN_PROVENANCE)
                      for lv in ("document", "page")}
        cases.append(BenchmarkCase(item.id, item.query, item.language, item.category, "sprint1_dataset",
                                   labels, meta))
    return cases


def write_cases(cases: Sequence[BenchmarkCase], path: Path | str) -> None:
    text = "".join(json.dumps(c.to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for c in cases)
    Path(path).write_text(text, encoding="utf-8")
