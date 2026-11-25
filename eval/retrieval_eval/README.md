# Sprint 5: Retrieval evaluation and benchmarking

An offline, deterministic benchmark for Saarthi's retrieval step. It scores **recorded** retrieval
results (which documents, pages and chunks came back, in which order) against relevance labels. It
does not run a retriever, a model or a network call. It needs no API key and adds no dependency.

> **Status.** The repository holds no recorded results from the real Saarthi retriever yet. So there
> are **no Saarthi retrieval benchmark numbers**. Every result in this document and in `fixtures/` is
> an **illustrative fixture** and is marked that way.

## Architecture

```
benchmark cases ──┐
 (cases.py)       │      ┌───────────────┐      ┌──────────────────────────────┐
                  ├────▶ │ benchmark.py  │ ───▶ │ JSON report: per case,       │
recorded results ─┘      │ metrics.py    │      │ aggregates, diagnostics,     │
 (records.py)            └───────────────┘      │ run comparison               │
       ▲                                         └──────────────────────────────┘
       │ adapters.record_run(cases, retriever, ...)   (run by YOU, outside the benchmark)
 live retriever (e.g. src.tools.retrieve.retrieve), never imported by this package
```

| File | Role |
|---|---|
| `metrics.py` | Precision@K, Recall@K, reciprocal rank (MRR), NDCG@K, the duplicate policy and the edge cases |
| `cases.py` | Benchmark case schema and loader. Read-only derivation of cases from the Sprint 1 dataset |
| `records.py` | Recorded-results format and loader. Malformed records are reported, not fatal |
| `benchmark.py` | Scores runs, builds aggregates and diagnostics, compares runs |
| `adapters.py` | Records a live retriever's output for later scoring. Never imports the app |
| `run_benchmark.py` | Command-line interface |

Sprint 1–4 code, the Sprint 1 dataset and the application are unchanged. The Sprint 4 claim matcher
is not used here.

## Benchmark cases

These are JSON Lines, one case per line. `cases.py` holds the full rules.

```json
{"id": "df-01", "question": "What is the Nyquist rate ...?", "language": "en", "category": "direct_factual",
 "source": "sprint1_dataset",
 "labels": {"document": {"status": "unverified", "completeness": "partial",
                         "grades": {"EE210 Module 1 Signals and Sampling": 1}, "provenance": "..."},
            "page":     {"status": "unverified", "completeness": "partial",
                         "grades": {"EE210 Module 1 Signals and Sampling#p3": 1}, "provenance": "..."},
            "chunk": null},
 "metadata": {"expected_behavior": "answer", "scope": {...}}}
```

- **Levels.** There are three: `document`, `page` (the id is `<document id>#p<page>`) and `chunk`.
  A missing level is `null` and its metrics are `not_evaluated`.
- **Grades.** Grades are integers from 0 to 3, so relevance can be graded. Grade 0 means "judged not relevant".
- **Label status.** Labels are `verified` (by a named reviewer, with a date that is not in the
  future), `unverified` (derived or drafted, not yet judged by a person), or `fixture` (example data
  only). Fixture labels can appear only in fixture cases.
- **Completeness.** Labels are `exhaustive` (every unlisted item is judged not relevant) or
  `partial` (unlisted items are **unjudged**). Unverified labels cannot be exhaustive.
- **Language.** Questions are English or Hinglish only (`en`, `hinglish`).

### Cases derived from the Sprint 1 dataset (the default)

The derivation reads `eval/answer_eval/dataset.jsonl` without changing it. The dataset is validated
first, including checking each evidence quote against the EE210 PDFs.

| Sprint 1 item | Derived labels |
|---|---|
| answer / clarify (31) | Each evidence page gets grade 1, and its material is relevant at document level. Labels are `unverified` and `partial`. |
| abstain (6) | No relevant item is labelled, so every metric is `not_applicable`. The report still shows how much was retrieved. |
| every item | No chunk labels. Sprint 1 records materials and pages, and chunk ids are specific to one index build. |

What these labels claim: each listed page holds an evidence quote that the validator finds on that
page. What they do not claim: that no other page is relevant. Nobody has judged that, and the
dataset's own labels are still `draft`. Unlisted pages and documents are therefore unjudged, never
irrelevant. To export the cases for human review, run
`python -m eval.retrieval_eval.run_benchmark --export-cases cases.jsonl`. Once reviewed, mark them
`verified` with a reviewer and date, and pass them back with `--cases`.

## Recorded results

These are JSON Lines with one run header per run and one result per (run, case). `records.py` holds
the full rules.

```json
{"record_type": "run", "run_id": "saarthi-hybrid-k10", "provenance": "recorded",
 "description": "dense+bm25, k=10", "retriever": {"k": 10, "legs": "dense,bm25"}}
{"record_type": "result", "run_id": "saarthi-hybrid-k10", "case_id": "df-01",
 "hits": [{"document_id": "EE210 Module 1 Signals and Sampling", "page": 3, "chunk_id": "4f1c...", "score": 0.031},
          {"document_id": null, "page": null, "chunk_id": "9a2e...", "score": 0.029}]}
{"record_type": "result", "run_id": "saarthi-hybrid-k10", "case_id": "df-02", "error": "TimeoutError: ..."}
```

- **Hits.** Hits are ordered best first. A hit with `document_id: null` could not be resolved to a
  document and is unjudged.
- **ID lists.** A retriever that cannot say which chunk came from which document can give
  independent `document_ids` and `chunk_ids` lists instead of `hits`.
- **Provenance.** `recorded` means captured from a real retriever. `fixture` means illustrative.
  Reports carry it through.

## Metric definitions and edge cases

Notation: rel(i) means grade ≥ threshold (default 1). R is the number of labelled relevant items.
Positions start at 1.

| Metric | Definition |
|---|---|
| Precision@K | relevant results in the top K, divided by **K**. Empty slots count as not relevant. |
| Recall@K | relevant results in the top K, divided by R. With partial labels this is recall of the *labelled* relevant items. |
| RR@K / MRR | RR@K is 1 / the rank of the first relevant result within the top K, or 0 if there is none. MRR is the mean of RR@K over cases. |
| NDCG@K | DCG / IDCG, where DCG = Σ gain(gradeᵢ) / log₂(i + 1). Gain is linear (g, the default) or exponential (2^g − 1). IDCG uses the labelled grades sorted in descending order. |

Every value carries a status:

| Status | Meaning |
|---|---|
| `exact` | Fully determined by the labels. |
| `bounded` | Unjudged items in the top K make the value uncertain, so `lower` and `upper` are given. Precision: [hits/K, (hits + unjudged)/K]. RR: an unjudged item before the first relevant one gives [1/first_relevant, 1/first_unjudged]. |
| `inconclusive` | NDCG with an unjudged item in the top K. No honest bound exists, because that item could change both DCG and IDCG. |
| `not_applicable` | No labelled relevant item exists, so the metric is 0/0. Precision is included: a question the corpus cannot answer is not a precision failure. |
| `not_evaluated` | No labels for the level, no result for the case, the result carries no ids for the level, or the retriever reported an error. |

- **Duplicates.**
  - `collapse` (the default): only the first occurrence is kept, and K counts collapsed positions.
  - `penalize`: a repeat keeps its slot with grade 0.
  - Either way, a repeated id never earns credit twice, and repeated chunks are reported.
- **Ties.** The recorded order is the ranking. Scores never reorder results. A score that rises down
  the list is reported as a `score_order_violation`.
- **Empty results.** An empty result is a real zero (recall 0, precision 0) and is flagged as
  `empty_results`. A missing result is reported as missing and never averaged in as zero.

## Running the benchmark

From the repository root:

```bash
# your recorded runs, scored against the cases derived from Sprint 1
python -m eval.retrieval_eval.run_benchmark --results runs.jsonl --k 1 3 5 10 --out report.json
# several runs, compared against a chosen baseline
python -m eval.retrieval_eval.run_benchmark --results bm25.jsonl --results hybrid.jsonl --baseline bm25-k10 --out report.json
# the illustrative fixtures
python -m eval.retrieval_eval.run_benchmark --cases eval/retrieval_eval/fixtures/illustrative_cases.jsonl \
    --results eval/retrieval_eval/fixtures/illustrative_runs.jsonl --k 1 3 5 --out report.json
# tests
python -m pytest -q eval/retrieval_eval/tests
```

Options:

| Option | Effect |
|---|---|
| `--duplicates collapse\|penalize` | Duplicate policy (see above). |
| `--relevance-threshold N` | Minimum grade that counts as relevant for the binary metrics. |
| `--gain linear\|exponential` | NDCG gain. |
| `--levels` | Which of document, page and chunk to score. |
| `--poor-recall X` | Flag recall@Kmax below X. |
| `--low-rank-cutoff N` | Flag a first relevant rank above N. |
| `--strict` | Exit 1 if any record was invalid, unmatched or missing. |

Exit codes: 0 = report written; 1 = unreadable results file, or `--strict` with input problems;
2 = invalid cases, invalid dataset or a bad option.

Example output (an excerpt) from the **illustrative fixtures**. These are invented data, not
Saarthi results:

```
Cases: 4 {'fixture': 4}  (offline: recorded results only)
NOTE: One or more runs have provenance 'fixture': they are illustrative example data, not Saarthi benchmark results.
Run fixture-bm25  ILLUSTRATIVE FIXTURE: 4 case(s) with results, 0 without
  document @1: precision [0.333, 0.667], recall 0.167, mrr [0.333, 0.667], ndcg 0.250* (exact cases only)
  chunk @3: precision 0.333, recall 0.333, mrr 0.500, ndcg 0.350
Run fixture-hybrid  ILLUSTRATIVE FIXTURE: 4 case(s) with results, 0 without
  document @1: precision 0.667, recall 0.500, mrr 0.667, ndcg 0.667
  chunk @3: precision 0.667, recall 1.000, mrr 0.750, ndcg 0.815
```

## Interpreting results

- **Reading an aggregate.**
  - A plain number is a mean over cases that are all `exact`.
  - `[a, b]` is the range of the mean when some cases are bounded.
  - `x* (exact cases only)` means some cases were inconclusive; the mean covers the exact ones only.
  - Every aggregate also lists how many cases had each status.
- **Diagnostics come in three groups, so that missing labels are never reported as retrieval failures.**
  - `retrieval_failures`: `zero_relevant_retrieved` (with the ranks where relevant items did turn
    up, if beyond K), `poor_recall`, `relevant_ranked_low`, `empty_results`, `duplicate_chunks`,
    `retriever_errors`.
  - `ground_truth_unavailable`: `missing_labels`, `no_labelled_relevant`, `unjudged_in_top_k`
    (which items need judging), `unresolved_hits`.
  - `input_problems`: `missing_results`, `score_order_violations`. Invalid records and results for
    unknown case ids are listed at the top level (`input_errors`, `unmatched_results`).
- **Comparisons are paired.** Each run is compared with the baseline only on cases where both have an
  exact value, and the report gives the mean of each run, the delta, and wins/losses/ties per case.
  `mixed_provenance` warns when a fixture run is compared with a recorded one.
- **Provisional labels.** Metrics on `unverified` labels are provisional, and the report says so.

## Connecting the live retriever later

Write a small script outside this package that calls the retriever and records its output:

```python
from src.tools.retrieve import retrieve
from eval.retrieval_eval import adapters
from eval.retrieval_eval.cases import load_cases

cases, errors = load_cases("cases.jsonl")          # or cases_from_answer_items(...)
records = adapters.record_run(
    cases, lambda req: retrieve(req.question, course_id=1, k=10),
    run_id="saarthi-hybrid-k10", description="dense+bm25 RRF, k=10, course 1",
    retriever_config={"k": 10, "RETRIEVAL_LEGS": "dense,bm25"})
adapters.write_records(records, "runs/saarthi-hybrid-k10.jsonl")
```

- **What the retriever receives.** It gets a `RetrievalRequest` with the case id, question and
  language. It never receives labels. The Sprint 1 `scope` is passed only with `include_scope=True`
  and is recorded in the run header: scoping the search to the material that holds the answer would
  inflate document-level metrics.
- **Errors.** A retriever exception is recorded as an `error` result, not as "nothing retrieved".
- **Mapping hits to documents.** `hits_from_saarthi_chunks` maps Saarthi `Chunk` objects
  (`id`, `title`, `source_file`, `page`, `score`) to hits. `resolve_ee210_document_id` matches
  titles and file names exactly after normalisation. It never matches fuzzily, so an unmatched
  chunk stays unresolved and unjudged. For another corpus, pass your own resolver to `hit_from_chunk`.
- **Before trusting live numbers:**
  - confirm that the indexed material titles or file tags resolve (check `unresolved_hits`);
  - record the index build and the retrieval settings in the run header;
  - judge the unjudged pages listed by `unjudged_in_top_k`.

## Known limitations

- There are no Saarthi benchmark results yet, only the fixtures. Nothing here measures live
  retrieval until someone records a run.
- The Sprint 1-derived labels are unverified and partial, and there are none at chunk level.
  Document- and page-level precision is mostly `bounded`, and NDCG is often `inconclusive`, until a
  person judges the unlisted pages.
- Only EE210 titles have a built-in resolver. The corpus is 4 documents × 4 pages, so document-level
  precision at large K is capped by the small corpus.
- `relevant_ranked_low` and `poor_recall` use fixed, configurable thresholds. They are triage aids,
  not quality gates.
- The older `eval/run_eval.py` retrieval check, which runs the live retriever with heuristic
  material + keyword relevance, is separate and unchanged.
