"""Offline retrieval evaluation for Saarthi (AI Architects, Sprint 5).

metrics.py        Precision@K, Recall@K, reciprocal rank (MRR), NDCG@K with explicit edge cases
cases.py          benchmark case schema and loader; read-only derivation from the Sprint 1 dataset
records.py        recorded retrieval results format and loader
benchmark.py      per-case metrics, aggregates, diagnostics and run comparison
adapters.py       how a live retriever's output is recorded for the benchmark (no app import)
run_benchmark.py  CLI

Nothing here imports or changes application code, the Sprint 1 dataset or the Sprint 3/4 modules.
See README.md.
"""
