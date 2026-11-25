"""Adapter for recording a live retriever's output for the offline benchmark (Sprint 5).

This module never imports the Saarthi application. A caller supplies the retriever as a function, so
the benchmark has no dependency on the live app, its index, its models or its API keys:

    from src.tools.retrieve import retrieve                     # in the CALLER's script, not here
    from eval.retrieval_eval import adapters

    def saarthi(request):                                        # RetrievalRequest -> list of chunks
        return retrieve(request.question, course_id=1, k=10)

    records = adapters.record_run(cases, saarthi, run_id="saarthi-hybrid-k10",
                                  description="hybrid dense+bm25, k=10, course 1",
                                  retriever_config={"k": 10, "legs": "dense,bm25"},
                                  to_hits=adapters.hits_from_saarthi_chunks)
    adapters.write_records(records, "runs/saarthi-hybrid-k10.jsonl")

Then score the file offline with run_benchmark.py. The recorded file is the evidence; the benchmark
does not re-run anything.

Leakage guard: the retriever receives a RetrievalRequest holding the case id, question and language
only. Relevance labels are never passed. The Sprint 1 scope (e.g. the material being viewed) is
passed only when include_scope=True, because scoping a search to the material that holds the answer
makes document-level metrics trivially high; a scoped run is marked in its run header.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

from eval.answer_eval.schema import MATERIALS
from eval.retrieval_eval.cases import BenchmarkCase


@dataclass(frozen=True)
class RetrievalRequest:
    case_id: str
    question: str
    language: str
    scope: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.scope is not None:
            object.__setattr__(self, "scope", MappingProxyType(dict(self.scope)))


Retriever = Callable[[RetrievalRequest], Sequence[Any]]


def request_for(case: BenchmarkCase, *, include_scope: bool = False) -> RetrievalRequest:
    scope = case.metadata.get("scope") if include_scope else None
    return RetrievalRequest(case.id, case.question, case.language, dict(scope) if scope else None)


def _norm(name: str) -> str:
    name = re.sub(r"\.pdf$", "", name.strip().lower())
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


_EE210_LOOKUP = {}
for _title, _file in MATERIALS.items():
    _EE210_LOOKUP[_norm(_title)] = _title
    _EE210_LOOKUP[_norm(_file)] = _title


def resolve_ee210_document_id(*names: Any) -> str | None:
    """Map a hit's title / source file tag to the canonical EE210 material title, or None.

    Matching is exact after normalisation (lower case, '.pdf' removed, non-alphanumerics -> '_').
    No fuzzy matching: an unresolved hit stays unresolved and is reported as unjudged, never guessed."""
    for name in names:
        if isinstance(name, str) and name.strip():
            hit = _EE210_LOOKUP.get(_norm(name))
            if hit:
                return hit
    return None


def hit_from_chunk(chunk: Any, resolve_document_id: Callable[..., str | None] = resolve_ee210_document_id) -> dict:
    """Convert one retrieved chunk (duck-typed like src.schemas.models.Chunk) into a recorded hit."""
    get = (chunk.get if isinstance(chunk, Mapping) else lambda k, d=None: getattr(chunk, k, d))
    page = get("page")
    score = get("score")
    chunk_id = get("id")
    return {
        "document_id": resolve_document_id(get("title"), get("source_file")),
        "page": page if isinstance(page, int) and not isinstance(page, bool) and page >= 1 else None,
        "chunk_id": str(chunk_id) if chunk_id not in (None, "") else None,
        "score": float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else None,
        "metadata": {k: get(k) for k in ("title", "source_file", "content_type", "via", "material_id")
                     if get(k) is not None},
    }


def hits_from_saarthi_chunks(chunks: Iterable[Any]) -> list[dict]:
    return [hit_from_chunk(c) for c in chunks]


def record_run(cases: Sequence[BenchmarkCase], retriever: Retriever, *, run_id: str, description: str = "",
               retriever_config: Mapping[str, Any] | None = None,
               to_hits: Callable[[Sequence[Any]], list[dict]] = hits_from_saarthi_chunks,
               include_scope: bool = False, metadata: Mapping[str, Any] | None = None) -> list[dict]:
    """Call the retriever once per case and return JSONL records (provenance 'recorded').

    A retriever exception is recorded as an 'error' result for that case (not scored), so one failing
    question never aborts the run and is never mistaken for 'nothing relevant was retrieved'."""
    header = {"record_type": "run", "run_id": run_id, "provenance": "recorded", "description": description,
              "retriever": dict(retriever_config or {}),
              "metadata": {**dict(metadata or {}), "include_scope": include_scope}}
    records: list[dict] = [header]
    for case in sorted(cases, key=lambda c: c.id):
        try:
            output = retriever(request_for(case, include_scope=include_scope))
            records.append({"record_type": "result", "run_id": run_id, "case_id": case.id,
                            "hits": to_hits(output)})
        except Exception as e:  # recorded, not raised: see docstring
            records.append({"record_type": "result", "run_id": run_id, "case_id": case.id,
                            "error": f"{type(e).__name__}: {e}"})
    return records


def write_records(records: Iterable[Mapping[str, Any]], path: Path | str) -> None:
    text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
    Path(path).write_text(text, encoding="utf-8")
