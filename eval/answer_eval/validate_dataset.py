"""Validate eval/answer_eval/dataset.jsonl: schema rules (schema.py) plus evidence checks against
the text of the EE210 PDFs.

    python eval/answer_eval/validate_dataset.py            # from the repo root; exit 1 on any error
    python eval/answer_eval/validate_dataset.py --verbose  # also list every verified quote

Evidence checks (need pypdf, already in the project's requirements):
  * every evidence quote occurs on its stated page. Matching ignores whitespace, because pypdf splits
    some kerned words in these PDFs ("T aking", "T s", "DFT s");
  * no quote is taken from the page footer ("EE210 Module ... - page N") or an "EE210 tag: SYN_..." line;
  * every absent_terms entry occurs nowhere in the 16 pages (case- and whitespace-insensitive), which
    proves an abstain item really cannot be answered from the corpus.

Only the active dataset is read. Items archived under archive/ (removed Hindi and Marathi items) are
never validated or counted here.

No network, no model, no application code is used.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.answer_eval.schema import (  # noqa: E402
    MATERIALS,
    PAGES_PER_MATERIAL,
    Item,
    load_dataset,
    validate_dataset_items,
)

DATASET_PATH = Path(__file__).resolve().parent / "dataset.jsonl"
CORPUS_DIR = ROOT / "eval" / "synthetic" / "course"

_WS = re.compile(r"\s+")
# Lines that are page furniture, not course content.
_FOOTER = re.compile(r"EE210 Module .+? - page \d+")
_TAG_LINE = re.compile(r"EE210 tag:\s*\S+")


def squash(text: str) -> str:
    """Remove all whitespace (pypdf splits kerned words: 'T s' for 'Ts')."""
    return _WS.sub("", text)


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> dict[str, list[str]]:
    """{material title: [page 1 text, page 2 text, ...]} extracted with pypdf, as indexing_service does."""
    from pypdf import PdfReader

    logging.getLogger("pypdf").setLevel(logging.ERROR)  # two PDFs log a recoverable startxref warning
    corpus: dict[str, list[str]] = {}
    for title, filename in MATERIALS.items():
        path = Path(corpus_dir) / filename
        if not path.exists():
            raise FileNotFoundError(f"corpus PDF missing: {path}")
        reader = PdfReader(str(path))
        corpus[title] = [(page.extract_text() or "") for page in reader.pages]
        if len(corpus[title]) != PAGES_PER_MATERIAL:
            raise ValueError(f"{filename}: expected {PAGES_PER_MATERIAL} pages, found {len(corpus[title])}")
    return corpus


def is_page_furniture(quote: str) -> bool:
    """True if the quote overlaps the footer or a SYN_ tag line."""
    q = squash(quote)
    return "SYN_" in q or "EE210tag:" in q or bool(re.search(r"EE210Module.*-page\d", q))


def _content_only(page_text: str) -> str:
    """Page text with the footer and tag lines removed, so a quote cannot match inside them."""
    return _TAG_LINE.sub(" ", _FOOTER.sub(" ", page_text))


def check_evidence(items: list[Item], corpus: dict[str, list[str]]) -> tuple[list[str], int]:
    """Returns (errors, number of verified quotes)."""
    errors: list[str] = []
    verified = 0
    all_text = squash(" ".join(t for pages in corpus.values() for t in pages)).lower()
    for item in items:
        for ev in item.evidence:
            where = f"item {item.id} evidence {ev.id}"
            if is_page_furniture(ev.quote):
                errors.append(f"{where}: quote uses footer or SYN_ tag text, which is not course content")
                continue
            pages = corpus.get(ev.material)
            if pages is None or not 1 <= ev.page <= len(pages):
                errors.append(f"{where}: no page {ev.page} in '{ev.material}'")
                continue
            if squash(ev.quote) in squash(_content_only(pages[ev.page - 1])):
                verified += 1
            else:
                elsewhere = [f"{m} p.{i}" for m, ps in corpus.items() for i, t in enumerate(ps, 1)
                             if squash(ev.quote) in squash(_content_only(t))]
                hint = f" (found on {', '.join(elsewhere)})" if elsewhere else ""
                errors.append(f"{where}: quote not found on {ev.material} p.{ev.page}{hint}: {ev.quote!r}")
        for term in item.absent_terms:
            if squash(term).lower() in all_text:
                errors.append(f"item {item.id}: absent term {term!r} occurs in the corpus")
    return errors, verified


def validate(dataset_path: Path = DATASET_PATH, corpus_dir: Path = CORPUS_DIR) -> dict:
    items, errors = load_dataset(dataset_path)
    errors += validate_dataset_items(items)
    corpus = load_corpus(corpus_dir)
    ev_errors, verified = check_evidence(items, corpus)
    errors += ev_errors
    return {
        "items": items,
        "errors": errors,
        "verified_quotes": verified,
        "total_quotes": sum(len(i.evidence) for i in items),
        "absent_terms": sum(len(i.absent_terms) for i in items),
        "categories": Counter(i.category for i in items),
        "behaviors": Counter(i.expected_behavior for i in items),
        "languages": Counter(i.language for i in items),
        "paraphrased": sorted(i.id for i in items if "paraphrased" in i.tags),
        "claims": Counter(c.reason for i in items for c in i.forbidden_claims),
        "must_not": sum(len(i.must_not) for i in items),
        "label_status": Counter(i.label_status for i in items),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, default=DATASET_PATH)
    ap.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    r = validate(args.dataset, args.corpus_dir)
    items = r["items"]
    print(f"Dataset: {args.dataset}")
    print(f"Items parsed: {len(items)}")
    print("Categories: " + ", ".join(f"{k} {v}" for k, v in sorted(r["categories"].items())))
    print("Expected behaviour: " + ", ".join(f"{k} {v}" for k, v in sorted(r["behaviors"].items())))
    print("Languages: " + ", ".join(f"{k} {v}" for k, v in sorted(r["languages"].items())))
    print(f"Paraphrased ({len(r['paraphrased'])}): {', '.join(r['paraphrased'])}")
    print("Forbidden claims by reason: " + ", ".join(f"{k} {v}" for k, v in sorted(r["claims"].items())))
    print(f"Behavioural rubric entries (must_not): {r['must_not']}")
    print("Label status: " + ", ".join(f"{k} {v}" for k, v in sorted(r["label_status"].items())))
    print(f"Evidence quotes verified: {r['verified_quotes']}/{r['total_quotes']}")
    print(f"Absent terms checked: {r['absent_terms']}")
    if args.verbose:
        for it in items:
            for ev in it.evidence:
                print(f"  {it.id} {ev.id}  {ev.material} p.{ev.page}: {ev.quote}")
    if r["errors"]:
        print(f"\nFAILED with {len(r['errors'])} error(s):")
        for e in r["errors"]:
            print(f"  - {e}")
        return 1
    print("\nOK: dataset is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
