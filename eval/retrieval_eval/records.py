"""Recorded retrieval results: format, loader and validation.

A results file is JSON Lines (blank lines allowed) holding two record types.

Run header (exactly one per run_id, before or after its results):
  {"record_type": "run", "run_id": "bm25-k10", "provenance": "recorded",
   "description": "...", "retriever": {"name": "...", "config": {...}}, "metadata": {...}}
  provenance  recorded  captured from a real retriever (e.g. through adapters.record_run)
              fixture   example data for tests and documentation; never a real benchmark result

Result (at most one per run_id and case_id):
  {"record_type": "result", "run_id": "bm25-k10", "case_id": "df-01",
   "hits": [{"document_id": "...", "page": 3, "chunk_id": "...", "score": 0.82, "metadata": {...}}, ...],
   "metadata": {...}}
  hits       ordered best first. document_id may be null when the retriever's output could not be
             resolved to a document (such a hit is unjudged at document and page level); page,
             chunk_id, score and metadata are optional.
  Instead of "hits", a result may give independent ordered lists "document_ids" and/or "chunk_ids"
  (for retrievers that do not report which chunk came from which document).
  "error": "..." records that the retriever failed for this case; the case is then not scored.

Level rankings derived from a result
  document  document_id of each hit (repeats are expected: several chunks of one document)
  page      "<document_id>#p<page>" of each hit; a hit without document or page is an unknown item.
            The page level is unavailable if no hit carries a page.
  chunk     chunk_id of each hit; unavailable if no hit carries a chunk_id.

Invalid records are skipped and reported with their line number; they never abort the load.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from eval.retrieval_eval.cases import ID_RE, page_id

PROVENANCES = ("recorded", "fixture")
_RUN_KEYS = {"record_type", "run_id", "provenance"}
_RUN_OPTIONAL = {"description", "retriever", "metadata"}
_RESULT_KEYS = {"record_type", "run_id", "case_id"}
_RESULT_OPTIONAL = {"hits", "document_ids", "chunk_ids", "metadata", "error"}
_HIT_KEYS = {"document_id", "page", "chunk_id", "score", "metadata"}


@dataclass(frozen=True)
class Hit:
    document_id: str | None
    page: int | None = None
    chunk_id: str | None = None
    score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    provenance: str
    description: str = ""
    retriever: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"run_id": self.run_id, "provenance": self.provenance, "description": self.description,
                "retriever": dict(self.retriever), "metadata": dict(self.metadata)}


@dataclass(frozen=True)
class RecordedResult:
    run_id: str
    case_id: str
    where: str
    hits: tuple[Hit, ...] | None = None          # None when the independent lists are used
    document_ids: tuple[str | None, ...] | None = None
    chunk_ids: tuple[str | None, ...] | None = None
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def ranking(self, level: str) -> tuple[str | None, ...] | None:
        """The ranked ids for a level, or None if the result carries no information for it."""
        if self.hits is not None:
            if level == "document":
                return tuple(h.document_id for h in self.hits)
            if level == "page":
                if not any(h.page is not None for h in self.hits):
                    return None if self.hits else ()
                return tuple(page_id(h.document_id, h.page) if h.document_id and h.page is not None else None
                             for h in self.hits)
            if level == "chunk":
                if not any(h.chunk_id is not None for h in self.hits):
                    return None if self.hits else ()
                return tuple(h.chunk_id for h in self.hits)
            raise ValueError(f"unknown level {level!r}")
        if level == "document":
            return self.document_ids
        if level == "chunk":
            return self.chunk_ids
        if level == "page":
            return None
        raise ValueError(f"unknown level {level!r}")

    def scores(self) -> tuple[float | None, ...]:
        return tuple(h.score for h in self.hits) if self.hits is not None else ()


@dataclass
class RecordSet:
    runs: dict[str, RunInfo] = field(default_factory=dict)
    results: dict[tuple[str, str], RecordedResult] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def results_for(self, run_id: str) -> dict[str, RecordedResult]:
        return {cid: r for (rid, cid), r in self.results.items() if rid == run_id}


def _is_str_or_none(v: Any) -> bool:
    return v is None or (isinstance(v, str) and bool(v.strip()))


def _parse_hit(raw: Any, w: str) -> tuple[Hit | None, list[str]]:
    if not isinstance(raw, dict):
        return None, [f"{w}: a hit must be an object"]
    errors = []
    extra = set(raw) - _HIT_KEYS
    if extra:
        errors.append(f"{w}: unknown hit field(s) {sorted(extra)}")
    if "document_id" not in raw:
        errors.append(f"{w}: a hit needs document_id (null if it could not be resolved)")
    doc, page, chunk, score = raw.get("document_id"), raw.get("page"), raw.get("chunk_id"), raw.get("score")
    if not _is_str_or_none(doc):
        errors.append(f"{w}: document_id must be a non-empty string or null")
    if page is not None and (isinstance(page, bool) or not isinstance(page, int) or page < 1):
        errors.append(f"{w}: page must be an integer >= 1 or null")
    if not _is_str_or_none(chunk):
        errors.append(f"{w}: chunk_id must be a non-empty string or null")
    if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float))
                              or not math.isfinite(score)):
        errors.append(f"{w}: score must be a finite number or null")
    md = raw.get("metadata", {})
    if not isinstance(md, dict):
        errors.append(f"{w}: hit metadata must be an object")
    if errors:
        return None, errors
    return Hit(doc, page, chunk, None if score is None else float(score), MappingProxyType(dict(md))), []


def _parse_id_list(raw: Any, w: str, name: str) -> tuple[tuple | None, list[str]]:
    if not isinstance(raw, list):
        return None, [f"{w}: {name} must be a list"]
    bad = [i for i, v in enumerate(raw, 1) if not _is_str_or_none(v)]
    if bad:
        return None, [f"{w}: {name} entries must be non-empty strings or null (position(s) {bad})"]
    return tuple(raw), []


def parse_records(rows: Iterable[tuple[str, Any]]) -> RecordSet:
    rs = RecordSet()
    pending: list[tuple[str, dict]] = []
    for where, raw in rows:
        if not isinstance(raw, dict):
            rs.errors.append(f"{where}: must be a JSON object")
            continue
        rtype = raw.get("record_type")
        if rtype == "run":
            missing, extra = _RUN_KEYS - set(raw), set(raw) - _RUN_KEYS - _RUN_OPTIONAL
            errs = ([f"{where}: run header missing field(s) {sorted(missing)}"] if missing else []) + \
                   ([f"{where}: run header has unknown field(s) {sorted(extra)}"] if extra else [])
            if not errs:
                if not isinstance(raw["run_id"], str) or not ID_RE.match(raw["run_id"]):
                    errs.append(f"{where}: run_id must match {ID_RE.pattern}")
                if raw["provenance"] not in PROVENANCES:
                    errs.append(f"{where}: provenance must be one of {PROVENANCES}")
                if not isinstance(raw.get("description", ""), str):
                    errs.append(f"{where}: description must be a string")
                for key in ("retriever", "metadata"):
                    if not isinstance(raw.get(key, {}), dict):
                        errs.append(f"{where}: {key} must be an object")
            if not errs and raw["run_id"] in rs.runs:
                errs.append(f"{where}: duplicate run header for {raw['run_id']!r}")
            if errs:
                rs.errors += errs
                continue
            rs.runs[raw["run_id"]] = RunInfo(raw["run_id"], raw["provenance"], raw.get("description", ""),
                                             MappingProxyType(dict(raw.get("retriever", {}))),
                                             MappingProxyType(dict(raw.get("metadata", {}))))
        elif rtype == "result":
            pending.append((where, raw))
        else:
            rs.errors.append(f"{where}: record_type must be 'run' or 'result'")
    for where, raw in pending:
        result, errs = _parse_result(raw, where)
        if not errs and result.run_id not in rs.runs:
            errs = [f"{where}: result for run {result.run_id!r}, which has no run header"]
        if not errs and (result.run_id, result.case_id) in rs.results:
            first = rs.results[(result.run_id, result.case_id)].where
            errs = [f"{where}: duplicate result for run {result.run_id!r} case {result.case_id!r} "
                    f"(first at {first}; kept the first)"]
        if errs:
            rs.errors += errs
            continue
        rs.results[(result.run_id, result.case_id)] = result
    return rs


def _parse_result(raw: dict, where: str) -> tuple[RecordedResult | None, list[str]]:
    missing, extra = _RESULT_KEYS - set(raw), set(raw) - _RESULT_KEYS - _RESULT_OPTIONAL
    errs = ([f"{where}: result missing field(s) {sorted(missing)}"] if missing else []) + \
           ([f"{where}: result has unknown field(s) {sorted(extra)}"] if extra else [])
    if errs:
        return None, errs
    run_id, case_id = raw["run_id"], raw["case_id"]
    if not isinstance(run_id, str) or not ID_RE.match(run_id):
        errs.append(f"{where}: run_id must match {ID_RE.pattern}")
    if not isinstance(case_id, str) or not case_id:
        errs.append(f"{where}: case_id must be a non-empty string")
    w = f"{where} [{run_id}/{case_id}]"
    md = raw.get("metadata", {})
    if not isinstance(md, dict):
        errs.append(f"{w}: metadata must be an object")
    error = raw.get("error")
    has_hits = "hits" in raw
    has_lists = "document_ids" in raw or "chunk_ids" in raw
    if error is not None:
        if not isinstance(error, str) or not error.strip():
            errs.append(f"{w}: error must be a non-empty string")
        if has_hits or has_lists:
            errs.append(f"{w}: a result with an error must not also carry retrieved items")
        if errs:
            return None, errs
        return RecordedResult(run_id, case_id, where, error=error, metadata=MappingProxyType(dict(md))), []
    if has_hits and has_lists:
        errs.append(f"{w}: give either hits or document_ids/chunk_ids, not both")
    if not has_hits and not has_lists:
        errs.append(f"{w}: a result needs hits, document_ids or chunk_ids (an empty list means nothing "
                    f"was retrieved)")
    if errs:
        return None, errs
    hits = doc_ids = chunk_ids = None
    if has_hits:
        if not isinstance(raw["hits"], list):
            return None, [f"{w}: hits must be a list"]
        parsed = []
        for i, h in enumerate(raw["hits"], 1):
            hit, e = _parse_hit(h, f"{w} hit {i}")
            errs += e
            parsed.append(hit)
        hits = tuple(parsed)
    else:
        if "document_ids" in raw:
            doc_ids, e = _parse_id_list(raw["document_ids"], w, "document_ids")
            errs += e
        if "chunk_ids" in raw:
            chunk_ids, e = _parse_id_list(raw["chunk_ids"], w, "chunk_ids")
            errs += e
    if errs:
        return None, errs
    return RecordedResult(run_id, case_id, where, hits, doc_ids, chunk_ids, None, MappingProxyType(dict(md))), []


def load_records(paths: Iterable[Path | str]) -> RecordSet:
    """Load one or more results files into one RecordSet. Raises OSError if a file cannot be read."""
    rows: list[tuple[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        p = Path(path)
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            where = f"{p.name}:{n}"
            try:
                rows.append((where, json.loads(line)))
            except json.JSONDecodeError as e:
                errors.append(f"{where}: invalid JSON ({e})")
    rs = parse_records(rows)
    rs.errors = errors + rs.errors
    return rs
