"""Sprint 2: citation parser. Run: python -m pytest eval/answer_eval/tests -q -p no:cacheprovider"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.answer_eval.citation_parser import CITATION_KINDS, parse_citations  # noqa: E402

M1 = "EE210 Module 1 Signals and Sampling"
M3 = "EE210 Module 3 Fourier Analysis DFT and Filters"


def texts(answer, **kw):
    return [c.text for c in parse_citations(answer, **kw).citations]


def kinds(answer, **kw):
    return [c.kind for c in parse_citations(answer, **kw).citations]


# ── Empty and no citations ──────────────────────────────────────────────────

@pytest.mark.parametrize("answer", [None, "", "   ", "The Nyquist rate is 2B, twice the highest frequency."])
def test_no_citations_gives_an_empty_result(answer):
    r = parse_citations(answer)
    assert r.citations == () and r.malformed == () and r.has_sources_block is False
    assert r.unique_keys() == [] and r.counts() == {}


# ── Supported formats ───────────────────────────────────────────────────────

def test_one_numeric_citation_with_position():
    answer = "The Nyquist rate is 2B [1]."
    (c,) = parse_citations(answer).citations
    assert (c.kind, c.text, c.numbers, c.key) == ("numeric", "[1]", (1,), "n:1")
    assert answer[c.start:c.end] == "[1]"


def test_numeric_groups_and_adjacent_numeric_citations():
    r = parse_citations("Aliasing folds tones [2, 3] and is irreversible [1][4].")
    assert [(c.text, c.numbers) for c in r.citations] == [("[2, 3]", (2, 3)), ("[1]", (1,)), ("[4]", (4,))]


def test_page_label_as_written_by_source_label():
    answer = f"The Nyquist rate is 2B [{M1}, p. 3]."
    (c,) = parse_citations(answer).citations
    assert c.kind == "page_label" and c.title == M1 and c.page == "3"
    assert c.text == f"[{M1}, p. 3]" and answer[c.start:c.end] == c.text


@pytest.mark.parametrize("label", [f"[{M1}, p.3]", f"[{M1}, page 3]", f"[{M1},  p.  3]"])
def test_page_label_spelling_variants(label):
    (c,) = parse_citations(f"Rate is 2B {label}.").citations
    assert c.kind == "page_label" and c.page == "3"


@pytest.mark.parametrize("label, ts", [("[Video: DSP Week 3, 14:22]", "14:22"), ("[Video: Long lecture, 1:02:03]", "1:02:03")])
def test_video_labels(label, ts):
    (c,) = parse_citations(f"See {label}.").citations
    assert c.kind == "video_label" and c.timestamp == ts


def test_gemini_sources_block_with_numbered_source_labels():
    answer = ("The Nyquist rate is 2B [1], see also [2].\n\n**Sources**\n"
              "[1] [Source: notes.pdf, p.4]\n[2] [Source: notes2.pdf]")
    r = parse_citations(answer)
    assert r.has_sources_block
    assert [(c.kind, c.text, c.in_sources_block) for c in r.citations] == [
        ("numeric", "[1]", False), ("numeric", "[2]", False),
        ("numeric", "[1]", True), ("source_label", "[Source: notes.pdf, p.4]", True),
        ("numeric", "[2]", True), ("source_label", "[Source: notes2.pdf]", True),
    ]
    assert [c.text for c in r.inline] == ["[1]", "[2]"]
    src = [c for c in r.citations if c.kind == "source_label"]
    assert (src[0].title, src[0].page) == ("notes.pdf", "4") and (src[1].title, src[1].page) == ("notes2.pdf", None)


def test_sources_suffix_appended_by_the_app_is_marked_as_not_inline():
    # Shape of src/tools/citations.sources_suffix: "\n\n**Sources:** " + "; ".join(labels)
    answer = f"The Nyquist rate is 2B.\n\n**Sources:** [{M1}, p. 3]; [Lecture 01 Fourier]"
    r = parse_citations(answer)
    assert r.has_sources_block and r.inline == ()
    assert [(c.kind, c.title) for c in r.citations] == [("page_label", M1), ("title_label", "Lecture 01 Fourier")]


def test_bare_title_label_needs_known_titles_outside_a_sources_block():
    answer = f"Convolution is commutative [{M3}]."
    assert texts(answer) == []
    assert kinds(answer, known_titles=[M3]) == ["title_label"]
    assert kinds(answer, known_titles=[M3.upper()]) == ["title_label"]  # case-insensitive


def test_several_labels_in_one_bracket_are_split():
    answer = f"Both facts [{M1}, p. 3; {M3}, p. 2] hold."
    r = parse_citations(answer)
    assert [(c.title, c.page) for c in r.citations] == [(M1, "3"), (M3, "2")]
    for c in r.citations:
        assert answer[c.start:c.end] == c.text


def test_every_supported_kind_is_reachable():
    answer = (f"a [1] b [{M1}, p. 3] c [Video: W3, 1:05] d [Source: n.pdf] e [{M3}]")
    assert set(kinds(answer, known_titles=[M3])) == set(CITATION_KINDS)


# ── Repeats ─────────────────────────────────────────────────────────────────

def test_repeated_citations_are_all_returned_and_counted():
    answer = f"A [1]. B [1]. C [{M1}, p. 3]. D [{M1.lower()},  p. 3]. E [2]."
    r = parse_citations(answer)
    assert len(r.citations) == 5
    assert r.counts() == {"n:1": 2, f"page:{M1.lower()}|3": 2, "n:2": 1}
    assert r.unique_keys() == ["n:1", f"page:{M1.lower()}|3", "n:2"]


# ── Malformed markers ───────────────────────────────────────────────────────

@pytest.mark.parametrize("answer, reason", [
    ("See [Source: ].", "source label"),
    ("See [Video: Lecture 3].", "video label"),
    (f"See [{M1}, p. ].", "page label without a page number"),
    ("See [Source: notes.pdf, p.4", "not closed"),
])
def test_malformed_markers_are_reported_not_raised(answer, reason):
    r = parse_citations(answer)
    assert r.citations == ()
    assert len(r.malformed) == 1 and reason in r.malformed[0].reason
    m = r.malformed[0]
    assert answer[m.start:m.end] == m.text


def test_malformed_part_of_a_multi_label_bracket():
    r = parse_citations(f"x [{M1}, p. 3; Video: broken] y")
    assert [c.page for c in r.citations] == ["3"]
    assert len(r.malformed) == 1 and r.malformed[0].text == "Video: broken"


@pytest.mark.parametrize("answer", ["[", "]", "[[", "[1", "1]", "[]", "[ ]", "[x]", "[;]", "[Source:", "]]][[["])
def test_garbage_never_raises(answer):
    parse_citations(answer)


# ── Things that are not citations ───────────────────────────────────────────

@pytest.mark.parametrize("answer", [
    "The Nyquist rate is 2B, so 3000 samples per second; the 1500 Hz term aliases to 500 Hz.",
    "x[n] = {1, 2, 2} has energy 9 and h[n-k] is the shifted response.",
    "Index a[1], the next a[1][2], and f(x)[3].",
    "The array x = [1, 2, 3] and the call f([1]).",
    "On the interval [0, 1] the gain is 0.5; also [0.5], [2018], [1-3] and [100].",
    "Read [the notes](http://example.com) or [1](http://example.com).",
    "- [ ] todo\n- [x] done",
    "The even part is [x(t) + x(-t)]/2 and cos(wt) = [e^{jwt} + e^{-jwt}] / 2.",
    "Inline math $x[1] + [2]$, display $$y = [3]$$, \\(z[4]\\), \\[ [5] \\], code `a[6]` and\n```\nb = [7]\n```",
    "Sources of error include rounding [noise].",
])
def test_numbers_maths_and_other_brackets_are_not_citations(answer):
    r = parse_citations(answer)
    assert r.citations == (), [c.text for c in r.citations]


def test_a_plain_sentence_starting_with_sources_is_not_a_sources_block():
    r = parse_citations("Sources of error include quantisation [1].")
    assert r.has_sources_block is False and r.citations[0].in_sources_block is False


def test_parser_makes_no_validity_judgement():
    # A citation to something that does not exist is still just a citation reference.
    (c,) = parse_citations("Claim [Nonexistent Lecture 99, p. 400].").citations
    assert c.kind == "page_label" and c.page == "400"
    assert not hasattr(c, "valid") and not hasattr(c, "supported")


# ── Against the real dataset and corpus (regression for false positives) ───

def test_no_reference_answer_in_the_dataset_contains_a_citation():
    lines = (ROOT / "eval" / "answer_eval" / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    for raw in (json.loads(x) for x in lines if x.strip()):
        assert parse_citations(raw["reference_answer"]).citations == (), raw["id"]


def test_no_citation_is_found_in_the_ee210_page_text():
    pytest.importorskip("pypdf")
    from eval.answer_eval.validate_dataset import load_corpus

    for material, pages in load_corpus().items():
        for n, page in enumerate(pages, 1):
            assert parse_citations(page).citations == (), (material, n)
