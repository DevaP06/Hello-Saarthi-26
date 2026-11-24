"""Sprint 4 Phase 2 (revised): the 11 draft forbidden-claim rules in claim_rules.jsonl.

Run: python -m pytest eval/answer_eval/tests -q -p no:cacheprovider

PATTERN-LEVEL TESTS ONLY. There is no production matcher (Phase 3) and the evaluation engine never
reads the registry. `assess()` below is the single, explicit definition of how a draft rule is read.
A later phase must implement these semantics deliberately; passing these tests says nothing about
accuracy on real Saarthi responses, and no rule is approved.

Draft-rule semantics (what assess() does)
1. Normalisation: Sprint 2 normalize_text (CRLF, NFC, NBSP, BOM/zero-width, spacing).
2. Quoted text: spans inside "..." or “...” are blanked before matching. A quoted phrase is a
   mention, not an assertion. Apostrophes and code are NOT blanked.
3. Sentences: split after . ! ? followed by whitespace (or end), and at newlines. A full stop after
   a known abbreviation (approx, e.g, i.e, vs, etc, fig, eq, cf, resp, no, ca, al) does not end a
   sentence. Decimal points never do (no whitespace follows them).
4. Clauses: a sentence is split into clauses only at strong boundaries: ";"; a spaced dash
   (" — ", " – ", or " - " before a letter); before a parenthetical that opens with a word
   (" (commonly"); before a comma + connective (but, though, although, however, whereas, yet, so,
   hence, thus, which, since, because, while, not); before a spaced but/though/although/whereas/
   however. Commas inside code or maths such as alias_frequency(9000, 12000) are not boundaries.
5. Matching: every match of the rule's pattern in a sentence is assessed, case-insensitively
   (re.IGNORECASE is applied by the reader; patterns carry no flags).
6. Anchors: ALL anchors must match somewhere in the same sentence (case-insensitive).
7. Exclusions: each exclusion starts with a regex comment naming it and, optionally, its scope:
   "(?#name)" applies to the claim clause (the whole clause containing the match, matched text
   included, so a negation captured inside a match such as "France never won" still counts);
   "(?#name;next-clause)" applies only to the clause immediately after it (a trailing rejection
   such as ", which is wrong"). Exclusions never look at other clauses or sentences.
8. Outcome per match: "no_anchor" if an anchor fails; else "excluded" if any exclusion fires; else
   "candidate" (an affirmative assertion of the claim). A text is a candidate if any match is.
9. Ambiguity: when the claim clause carries negation, correction, a question, a reported claim
   (false claims only) or an explicit hypothetical (false claims only), the match is NOT flagged.
   This is deliberate: a missed detection is preferred to accusing a correct answer. Unsupported
   claims (abstain items) do not exclude reported or hypothetical wording, because relaying or
   hypothesising uncovered content still leaks it.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.answer_eval import review_claim_rules  # noqa: E402
from eval.answer_eval.claim_rules import (  # noqa: E402
    NEGATION_CUE, REGISTRY_PATH, dataset_claims, load_registry, parse_registry,
)
from eval.answer_eval.response_normalizer import normalize_text  # noqa: E402
from eval.answer_eval.schema import load_dataset  # noqa: E402

PKG = Path(__file__).resolve().parents[1]
DATASET = PKG / "dataset.jsonl"
FIXTURE = PKG / "tests" / "fixtures" / "responses_sample.jsonl"
ITEMS, _ = load_dataset(DATASET)
BY_ID = {i.id: i for i in ITEMS}
CLAIMS = dataset_claims(ITEMS)
RAW = [json.loads(line) for line in REGISTRY_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
RULES = {r["claim_id"]: r for r in RAW if r["record_type"] == "forbidden_claim_rule"}

# Bucket 1 of the Sprint 4 feasibility analysis (C04, C05, C07, C08, C12, C13, C33, C42, C43, C45,
# C47). mh-05#c1 (C32) was bucket 2 and is deliberately absent.
EXPECTED = {
    "df-02#c2": "false", "df-03#c1": "false", "df-04#c1": "false", "df-04#c2": "false",
    "df-07#c1": "false", "df-07#c2": "false", "mh-05#c2": "false",
    "ic-01#c1": "unsupported", "ic-02#c1": "unsupported", "ic-03#c1": "unsupported",
    "ic-05#c1": "unsupported",
}
SHARED_FALSE = ("negation", "correction", "reported", "question", "hypothetical", "rejected-after")
SHARED_UNSUPPORTED = ("negation", "correction", "question", "rejected-after")


# ── The reader (test-local; see module docstring) ───────────────────────────────────────────

ABBREVIATIONS = ("approx", "e.g", "i.e", "vs", "etc", "fig", "eq", "cf", "resp", "no", "ca", "al")
_END = re.compile(r"[.!?]+(?=\s|$)|\n+")
_QUOTED = re.compile(r"\"[^\"\n]*\"|“[^”\n]*”")
_CLAUSE = re.compile(
    r"\s*;\s*"
    r"|\s+[—–]\s+|\s+-{1,2}\s+(?=[A-Za-z])"
    r"|\s+(?=\([A-Za-z]{2,})"
    r"|,\s*(?=(?:but|though|although|however|whereas|yet|so|hence|thus|which|since|because|while|not)\b)"
    r"|\s+(?=(?:but|though|although|whereas|however)\s)", re.I)
_TAG = re.compile(r"^\(\?#([a-z0-9-]+)(?:;(next-clause))?\)")


def sentences(text: str) -> list[str]:
    t = _QUOTED.sub(lambda m: " " * len(m.group(0)), normalize_text(text))
    out, start = [], 0
    for m in _END.finditer(t):
        if m.group(0).startswith("."):
            word = re.search(r"([A-Za-z.]+)$", t[start:m.start()])
            if word and word.group(1).lower().rstrip(".") in ABBREVIATIONS:
                continue
        if t[start:m.end()].strip():
            out.append(t[start:m.end()].strip())
        start = m.end()
    if t[start:].strip():
        out.append(t[start:].strip())
    return out


def clauses(sentence: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in _CLAUSE.finditer(sentence):
        if m.start() > start:
            spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(sentence)))
    return [(a, b) for a, b in spans if sentence[a:b].strip()]


def exclusion_tag(exclusion: str) -> tuple[str | None, str]:
    m = _TAG.match(exclusion)
    return (m.group(1), m.group(2) or "clause") if m else (None, "clause")


@dataclass
class Decision:
    outcome: str                 # candidate | excluded | no_anchor
    sentence: str
    match: str
    clause: str
    failed_anchors: list[int] = field(default_factory=list)
    fired: list[str] = field(default_factory=list)


def assess(rule: dict, text: str) -> list[Decision]:
    out = []
    for s in sentences(text):
        spans = clauses(s)
        for m in re.finditer(rule["pattern"], s, re.I):
            failed = [i for i, a in enumerate(rule["anchors"]) if not re.search(a, s, re.I)]
            idx = next((i for i, (a, b) in enumerate(spans) if a <= m.start() < b), len(spans) - 1)
            a, b = spans[idx]
            claim_clause = s[a:b]
            following = s[spans[idx + 1][0]:spans[idx + 1][1]] if idx + 1 < len(spans) else ""
            fired = []
            for ex in rule["exclusions"]:
                tag, scope = exclusion_tag(ex)
                target = following if scope == "next-clause" else claim_clause
                if target and re.search(ex, target, re.I):
                    fired.append(tag)
            outcome = "no_anchor" if failed else ("excluded" if fired else "candidate")
            out.append(Decision(outcome, s, m.group(0), s[a:b], failed, fired))
    return out


def is_candidate(rule: dict, text: str) -> bool:
    return any(d.outcome == "candidate" for d in assess(rule, text))


def without_exclusion(rule: dict, tag: str) -> dict:
    return {**rule, "exclusions": [e for e in rule["exclusions"] if exclusion_tag(e)[0] != tag]}


def without_anchor(rule: dict, index: int) -> dict:
    return {**rule, "anchors": [a for i, a in enumerate(rule["anchors"]) if i != index]}


# ── Reader semantics ────────────────────────────────────────────────────────────────────────

def test_sentence_split_protects_abbreviations_and_decimals():
    assert sentences("The SQNR is approx. 16 dB. Next, e.g. this one.") == \
        ["The SQNR is approx. 16 dB.", "Next, e.g. this one."]
    assert sentences("w[n] = 0.54 - 0.46 cos(x). Done") == ["w[n] = 0.54 - 0.46 cos(x).", "Done"]
    assert sentences("Line one\nLine two") == ["Line one", "Line two"]


@pytest.mark.parametrize("sentence,expected", [
    ("I couldn't find this in your notes, but France won.", ["I couldn't find this in your notes", "but France won."]),
    ("Your notes don't cover it (commonly 0.54 - 0.46 cos(x)).", ["Your notes don't cover it", "(commonly 0.54 - 0.46 cos(x))."]),
    ("The period is 50 ms, so it does not depend on amplitude.", ["The period is 50 ms", "so it does not depend on amplitude."]),
    ("It is 20 ms, not 50 ms.", ["It is 20 ms", "not 50 ms."]),
    ("It is wrong; the output has 8 samples.", ["It is wrong", "the output has 8 samples."]),
    ("Does alias_frequency(9000, 12000) return 21000?", ["Does alias_frequency(9000, 12000) return 21000?"]),
    ("w[n] = 0.54 - 0.46 cos(2 pi n / (N - 1)) here.", ["w[n] = 0.54 - 0.46 cos(2 pi n / (N - 1)) here."]),
])
def test_clause_boundaries(sentence, expected):
    assert [sentence[a:b] for a, b in clauses(sentence)] == expected


def test_quoted_mention_is_ignored_but_code_quotes_do_not_suppress():
    r = RULES["df-04#c1"]
    assert not is_candidate(r, 'Students sometimes write "the output has 8 samples", but it has 7.')
    assert is_candidate(RULES["mh-05#c2"], 'alias_frequency(9000, 12000) returns 21000, as print("x") shows.')


def test_matching_is_case_insensitive():
    for cid, r in RULES.items():
        for p in r["positive_examples"]:
            assert is_candidate(r, p.upper()), (cid, p)
            assert is_candidate(r, p.lower()), (cid, p)
            assert is_candidate(r, p.swapcase()), (cid, p)


def test_exclusion_scope_is_the_claim_clause_only():
    r = RULES["df-03#c1"]
    d = assess(r, "The period is 50 ms, so it does not depend on the amplitude.")[0]
    assert d.outcome == "candidate" and d.fired == []
    d = assess(r, "The period is not 50 ms, so the amplitude is irrelevant.")[0]
    assert d.outcome == "excluded" and d.fired == ["negation"]


def test_next_clause_scope_only_sees_the_following_clause():
    r = RULES["df-04#c1"]
    assert assess(r, "The output has 8 samples, which is wrong.")[0].fired == ["rejected-after"]
    assert is_candidate(r, "Which is wrong is unclear to students; the output has 8 samples.")


# ── Inventory and metadata ──────────────────────────────────────────────────────────────────

def test_exactly_the_eleven_expected_draft_rules():
    assert sorted(RULES) == sorted(EXPECTED) and len(RAW) == 11


def test_count_by_reason():
    reasons = [r["reason"] for r in RULES.values()]
    assert reasons.count("false") == 7 and reasons.count("unsupported") == 4


def test_bucket_two_claim_mh05_c1_is_not_drafted():
    assert "mh-05#c1" not in RULES
    assert CLAIMS["mh-05#c1"][2] == "alias_frequency(9000, 12000) returns 9000"


def test_no_must_not_coverage_records_yet():
    assert all(r["record_type"] == "forbidden_claim_rule" for r in RAW)


@pytest.mark.parametrize("cid", sorted(EXPECTED))
def test_rule_references_the_dataset_claim(cid):
    r = RULES[cid]
    item, pos, text, reason = CLAIMS[cid]
    assert r["rule_id"] == f"{cid}/r1"
    assert r["item_id"] == item.id == cid.split("#")[0]
    assert int(cid.split("#c")[1]) == pos
    assert r["claim_text"] == text and r["reason"] == reason == EXPECTED[cid]


@pytest.mark.parametrize("cid", sorted(EXPECTED))
def test_rule_metadata(cid):
    r = RULES[cid]
    assert r["feasibility"] == "deterministic" and r["status"] == "draft"
    assert r["reviewer"] is None and r["reviewed_on"] is None and r["author"] == "claude"
    assert r["languages"] == ["en"] == [BY_ID[r["item_id"]].expected_answer_language]
    assert r["rationale"].strip() and r["limitations"].strip()


@pytest.mark.parametrize("cid", sorted(EXPECTED))
def test_regexes_compile_and_every_exclusion_is_named(cid):
    r = RULES[cid]
    for p in [r["pattern"], *r["anchors"], *r["exclusions"]]:
        re.compile(p)
    assert r["anchors"]
    tags = [exclusion_tag(e)[0] for e in r["exclusions"]]
    assert None not in tags and len(tags) == len(set(tags))
    shared = SHARED_FALSE if r["reason"] == "false" else SHARED_UNSUPPORTED
    assert set(shared) <= set(tags)
    if r["reason"] == "unsupported":
        assert not {"reported", "hypothetical"} & set(tags), "relaying uncovered content still leaks it"


@pytest.mark.parametrize("cid", sorted(EXPECTED))
def test_example_counts_and_kinds(cid):
    r = RULES[cid]
    pos, neg = r["positive_examples"], r["negative_examples"]
    assert len(pos) >= 2 and len(neg) >= 3
    assert len(set(pos)) == len(pos) and len(set(neg)) == len(neg)
    assert any(NEGATION_CUE.search(n) for n in neg)
    assert any(n.rstrip().endswith("?") for n in neg)
    if r["reason"] == "false":
        assert BY_ID[r["item_id"]].reference_answer in neg
    else:
        assert any("course" in n.lower() or "notes" in n.lower() for n in neg)


# ── Stored examples, dataset texts and fixtures ─────────────────────────────────────────────

def _examples():
    for cid, r in RULES.items():
        for p in r["positive_examples"]:
            yield pytest.param(cid, p, True, id=f"{cid}+{p[:30]}")
        for n in r["negative_examples"]:
            yield pytest.param(cid, n, False, id=f"{cid}-{n[:30]}")


@pytest.mark.parametrize("cid,text,expected", list(_examples()))
def test_stored_examples(cid, text, expected):
    assert is_candidate(RULES[cid], text) is expected, assess(RULES[cid], text)


@pytest.mark.parametrize("cid", sorted(EXPECTED))
def test_no_rule_fires_on_any_dataset_text(cid):
    for it in ITEMS:
        for text in (it.query, it.reference_answer, *it.acceptable_interpretations,
                     *(e.quote for e in it.evidence)):
            assert not is_candidate(RULES[cid], text), (it.id, text)


def test_rules_do_not_fire_on_each_others_positive_examples():
    for cid, r in RULES.items():
        for p in r["positive_examples"]:
            assert [o for o, ro in RULES.items() if o != cid and is_candidate(ro, p)] == [], (cid, p)


def test_sprint3_fixture_responses():
    """The ic-03 mixed abstention states the unsupported ROC in its own sentence: a true positive."""
    hits = []
    for line in FIXTURE.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        hits += [(row["id"], cid) for cid, r in RULES.items() if is_candidate(r, row["response"])]
    assert hits == [("ic-03", "ic-03#c1")]


# ── Decisive exclusions: each case is rejected by exactly one exclusion ─────────────────────

DECISIVE_EXCLUSIONS = {
    ("df-02#c2", "negation"): "An N-bit quantizer does not have N^2 levels.",
    ("df-02#c2", "correction"): "Saying an N-bit quantizer has N^2 levels is a common mistake.",
    ("df-02#c2", "reported"): "Some textbooks say an N-bit quantizer has N^2 levels.",
    ("df-02#c2", "question"): "So an N-bit quantizer has N^2 levels?",
    ("df-02#c2", "hypothetical"): "If you swapped base and exponent, an N-bit quantizer would have N^2 levels.",
    ("df-02#c2", "rejected-after"): "An N-bit quantizer has N^2 levels, which is wrong.",
    ("df-02#c2", "two-to-the-n"): "For an N-bit quantizer, the 2^N levels exceed N^2 levels once N is above 4.",
    ("df-03#c1", "negation"): "The period is never 50 ms for this signal.",
    ("df-03#c1", "correction"): "Calling the period 50 ms is a common mistake.",
    ("df-03#c1", "reported"): "Some textbooks say the period is 50 ms.",
    ("df-03#c1", "question"): "So the period is 50 ms?",
    ("df-03#c1", "hypothetical"): "If you took the 50 as a time, the period would be 50 ms.",
    ("df-03#c1", "rejected-after"): "The period is 50 ms, which is incorrect.",
    ("df-03#c1", "other-frequency"): "A 20 Hz tone has a period of 50 ms.",
    ("df-03#c1", "observation-window"): "The period appears 2.5 times in a 50 ms window.",
    ("df-04#c1", "negation"): "The convolution output never has 8 samples.",
    ("df-04#c1", "correction"): "Claiming the output has 8 samples is a common mistake.",
    ("df-04#c1", "reported"): "Students often say the output has 8 samples.",
    ("df-04#c1", "question"): "So the output has 8 samples?",
    ("df-04#c1", "hypothetical"): "If the lengths simply added, the output would have 8 samples.",
    ("df-04#c1", "rejected-after"): "The output has 8 samples, which is a common mistake.",
    ("df-04#c1", "intermediate"): "The output length before subtracting the overlap is 8 samples.",
    ("df-04#c1", "fft-padding"): "The convolution result is computed on 8 points after zero-padding.",
    ("df-04#c2", "negation"): "The convolution output never has 15 samples.",
    ("df-04#c2", "correction"): "Claiming the output has 15 samples is a common mistake.",
    ("df-04#c2", "reported"): "Students often say the output has 15 samples.",
    ("df-04#c2", "question"): "So the output has 15 samples?",
    ("df-04#c2", "hypothetical"): "If the lengths multiplied, the output would have 15 samples.",
    ("df-04#c2", "rejected-after"): "The output has 15 samples, which is wrong.",
    ("df-04#c2", "intermediate"): "The uncorrected output count is 15 samples.",
    ("df-04#c2", "fft-padding"): "The convolution result uses a 15-point DFT.",
    ("df-07#c1", "negation"): "The SQNR is never 16 dB for a 16-bit converter.",
    ("df-07#c1", "correction"): "An SQNR of 16 dB is far too low for 16 bits.",
    ("df-07#c1", "reported"): "Some sources say the SQNR is 16 dB.",
    ("df-07#c1", "question"): "So the SQNR is 16 dB?",
    ("df-07#c1", "hypothetical"): "That would make the SQNR 16 dB.",
    ("df-07#c1", "rejected-after"): "The SQNR is 16 dB, which is incorrect.",
    ("df-07#c1", "gain-context"): "The SQNR test adds 16 dB of gain first.",
    ("df-07#c1", "correct-formula"): "Using 6.02 N, the SQNR of 16 dB is off by far.",
    ("df-07#c2", "negation"): "The SQNR never reaches 160 dB.",
    ("df-07#c2", "correction"): "An SQNR of 160 dB is unrealistic.",
    ("df-07#c2", "reported"): "Many people think the SQNR is 160 dB.",
    ("df-07#c2", "question"): "So the SQNR is 160 dB?",
    ("df-07#c2", "hypothetical"): "At 10 dB per bit, that would give an SQNR of 160 dB.",
    ("df-07#c2", "rejected-after"): "The SQNR is 160 dB, which is a misconception.",
    ("df-07#c2", "gain-context"): "The SQNR bench uses a 160 dB attenuator.",
    ("df-07#c2", "correct-formula"): "With the 1.76 dB term, the SQNR is 160 dB.",
    ("mh-05#c2", "negation"): "alias_frequency(9000, 12000) never returns 21000.",
    ("mh-05#c2", "correction"): "Saying alias_frequency(9000, 12000) returns 21000 is a common mistake.",
    ("mh-05#c2", "reported"): "Students often say alias_frequency(9000, 12000) returns 21000.",
    ("mh-05#c2", "question"): "So alias_frequency(9000, 12000) returns 21000?",
    ("mh-05#c2", "hypothetical"): "A naive alias_frequency would return 21000.",
    ("mh-05#c2", "rejected-after"): "alias_frequency(9000, 12000) returns 21000, which is wrong.",
    ("mh-05#c2", "spectral-image"): "alias_frequency's plot also yields 21000 as an image.",
    ("ic-01#c1", "negation"): "The triangular pulse does not transform to sinc^2 in your notes.",
    ("ic-01#c1", "correction"): "Calling the triangular pulse's transform sinc^2 would be a mistake here.",
    ("ic-01#c1", "question"): "So a triangular pulse transforms to sinc^2?",
    ("ic-01#c1", "rejected-after"): "A triangular pulse transforms to sinc^2, which is incorrect.",
    ("ic-01#c1", "rectangular-only"): "Your notes mention a triangular pulse; the rectangular pulse's energy spectrum is T^2 sinc^2(fT).",
    ("ic-02#c1", "negation"): "The Hamming coefficients are not 0.54 and 0.46 in your notes.",
    ("ic-02#c1", "correction"): "Quoting the Hamming coefficients as 0.54 and 0.46 is wrong for this course.",
    ("ic-02#c1", "question"): "Are the Hamming coefficients 0.54 and 0.46?",
    ("ic-02#c1", "rejected-after"): "The Hamming coefficients are 0.54 and 0.46, which is not right here.",
    ("ic-03#c1", "negation"): "The ROC is not Re(s) > -2 according to your notes.",
    ("ic-03#c1", "correction"): "Stating the ROC as Re(s) > -2 would be a mistake here.",
    ("ic-03#c1", "question"): "So the ROC is Re(s) > -2?",
    ("ic-03#c1", "rejected-after"): "The ROC is Re(s) > -2, but the correct answer depends on your course.",
    ("ic-05#c1", "negation"): "France never won the 2018 World Cup.",
    ("ic-05#c1", "correction"): "Saying France won the 2018 World Cup is wrong here.",
    ("ic-05#c1", "question"): "So France won the 2018 World Cup?",
    ("ic-05#c1", "rejected-after"): "France won the 2018 World Cup, which is a common misconception.",
    ("ic-05#c1", "missing-2018"): "France won the World Cup.",
}


def test_every_exclusion_of_every_rule_has_a_decisive_case():
    needed = {(cid, exclusion_tag(e)[0]) for cid, r in RULES.items() for e in r["exclusions"]}
    assert needed == set(DECISIVE_EXCLUSIONS)


@pytest.mark.parametrize("cid,tag", sorted(DECISIVE_EXCLUSIONS))
def test_exclusion_is_decisive(cid, tag):
    text = DECISIVE_EXCLUSIONS[(cid, tag)]
    rule = RULES[cid]
    decisions = assess(rule, text)
    assert decisions, "the pattern must match so that the exclusion is what decides"
    assert all(d.outcome == "excluded" and d.fired == [tag] and not d.failed_anchors for d in decisions), decisions
    assert is_candidate(without_exclusion(rule, tag), text), "removing this exclusion must flip the decision"


# ── Decisive anchors: the pattern matches, no exclusion fires, only this anchor fails ────────

DECISIVE_ANCHORS = {
    ("df-02#c2", 0): "An N-bit quantizer maps each sample to one of N^2 codes.",
    ("df-02#c2", 1): "A ladder with N^2 levels is tiring to climb.",
    ("df-03#c1", 0): "The waveform repeats every 50 ms.",
    ("df-04#c1", 0): "Each frame holds 8 samples.",
    ("df-04#c2", 0): "The moving average uses 15 samples.",
    ("df-07#c1", 0): "The microphone SNR is about 16 dB.",
    ("df-07#c2", 0): "The theoretical SNR here is 160 dB.",
    ("mh-05#c2", 0): "The helper returns 21000 for these inputs.",
    ("ic-01#c1", 0): "Squaring gives sinc^2 in the spectrum.",
    ("ic-02#c1", 0): "The window weights are 0.54 and 0.46.",
    ("ic-03#c1", 0): "The inequality Re(s) > -2 holds for this example.",
    ("ic-05#c1", 0): "France won the 2018 final.",
}


def test_every_anchor_has_a_decisive_case():
    assert {(cid, i) for cid, r in RULES.items() for i in range(len(r["anchors"]))} == set(DECISIVE_ANCHORS)


@pytest.mark.parametrize("cid,index", sorted(DECISIVE_ANCHORS))
def test_anchor_is_decisive(cid, index):
    text = DECISIVE_ANCHORS[(cid, index)]
    decisions = assess(RULES[cid], text)
    assert decisions and all(d.outcome == "no_anchor" and d.failed_anchors == [index] and d.fired == []
                             for d in decisions), decisions
    assert is_candidate(without_anchor(RULES[cid], index), text), "removing this anchor must flip the decision"


# ── Bare-value patterns would be wrong: the real pattern rejects what the bare value accepts ─

BARE_VALUE = {
    "df-02#c2": (r"\b(?:9|25|36|64|100|256)\s+levels\b", "A 6-bit quantizer has 64 levels."),
    "df-03#c1": (r"\b50\b", "The period of the 50 Hz tone is short."),
    "df-04#c1": (r"\b8\b", "The convolution output peaks at sample 8."),
    "df-04#c2": (r"\b15\b", "The convolution output peaks at sample 15."),
    "df-07#c1": (r"16\s*dB", "The SQNR changed by 0.16 dB after calibration."),
    "df-07#c2": (r"160\s*dB", "The SQNR changed by 0.160 dB after calibration."),
    "mh-05#c2": (r"21,?000", "alias_frequency(9000, 12000) is tested in cell 21000 of the notebook."),
    "ic-01#c1": (r"sinc", "For a triangular pulse, the notes plot sinc functions for comparison."),
    "ic-02#c1": (r"0?\.54", "The Hamming coefficient table lists 0.54 for one entry."),
    "ic-03#c1": (r"-\s*2", "The Laplace transform of e^{-2t} u(t) has a pole at s = -2."),
    "ic-05#c1": (r"France", "France reached the 2018 World Cup final."),
}


def test_every_rule_has_a_bare_value_case():
    assert set(BARE_VALUE) == set(RULES)


@pytest.mark.parametrize("cid", sorted(BARE_VALUE))
def test_real_pattern_is_stricter_than_the_bare_value(cid):
    bare, text = BARE_VALUE[cid]
    assert not is_candidate(RULES[cid], text), assess(RULES[cid], text)
    assert is_candidate({**RULES[cid], "pattern": bare}, text), "the case must catch a bare-value pattern"


# ── Mixed abstention, hypotheticals and corrections ──────────────────────────────────────────

@pytest.mark.parametrize("cid,text", [
    ("ic-01#c1", "I couldn't find this in your notes, but generally the triangle pulse transforms to sinc^2."),
    ("ic-02#c1", "Your notes don't cover the Hamming window (commonly 0.54 - 0.46 cos(2 pi n/(N-1)))."),
    ("ic-05#c1", "This isn't in your course, but France won the 2018 World Cup."),
    ("ic-03#c1", "Not covered in your notes, though the ROC would be Re(s) > -2."),
    ("ic-02#c1", "I couldn't find the Hamming window in your course materials. In general, it is defined as w[n] = 0.54 - 0.46 cos(2πn/(N-1))."),
    ("ic-03#c1", "Your notes don't discuss the Laplace transform; its ROC would be Re(s) > -2."),
])
def test_mixed_abstention_leak_is_detected(cid, text):
    assert is_candidate(RULES[cid], text), assess(RULES[cid], text)


@pytest.mark.parametrize("cid,text", [
    ("ic-01#c1", "I couldn't find the triangular pulse in your notes, so I won't give its transform."),
    ("ic-02#c1", "I can't give the 0.54 and 0.46 coefficients because they are not in your notes."),
    ("ic-05#c1", "I can't tell you whether France won the 2018 World Cup from these notes."),
])
def test_clean_or_self_negated_abstentions_are_not_flagged(cid, text):
    assert not is_candidate(RULES[cid], text)


@pytest.mark.parametrize("cid,text,expected", [
    # affirmative (flagged)
    ("df-04#c1", "Convolving them, you would get 8 samples.", True),
    ("df-07#c1", "Each bit adds about 1 dB, so the SQNR of a 16-bit converter is 16 dB.", True),
    ("df-03#c1", "The period is 50 ms, not 20 ms.", True),
    ("df-04#c1", "The output has 8 samples, one more than the 7 you might expect.", True),
    # explicit hypothetical calculation (not flagged)
    ("df-04#c1", "Step 1: Naively adding the input lengths, the output length would be 8 samples.", False),
    ("df-07#c1", "That would give 16 dB SQNR for 16 bits.", False),
    ("mh-05#c2", "If alias_frequency added fs, it would return 21000.", False),
    # description of a known mistake (not flagged)
    ("mh-05#c2", "A buggy version of alias_frequency that adds fs gives 21000 for these inputs.", False),
    ("df-04#c2", "Forgetting the overlap, the output appears to have 15 samples.", False),
    # explicit correction (not flagged)
    ("df-03#c1", "It is 20 ms, not 50 ms; the period of cos(2 pi 50 t) is 20 ms.", False),
    ("df-04#c1", "The output has 8 samples — a common misconception; it has 7.", False),
    # unsupported claims: hypothetical wording still leaks
    ("ic-03#c1", "If your course covered it, the ROC would be Re(s) > -2.", True),
    ("ic-02#c1", "Some references give 0.54 and 0.46 as the Hamming coefficients.", True),
])
def test_affirmative_hypothetical_mistake_and_correction(cid, text, expected):
    assert is_candidate(RULES[cid], text) is expected, assess(RULES[cid], text)


@pytest.mark.parametrize("cid,text", [
    ("df-02#c2", "An N-bit quantizer has N^2 levels, so the quantization error shrinks quickly."),
    ("df-07#c1", "With 16 bits, the quantization error gives an SQNR of about 16 dB."),
    ("df-03#c1", "The period is 50 ms, so it does not depend on amplitude."),
    ("df-04#c1", "Index n runs from 0 to 7, so the output has 8 samples."),
    ("df-04#c2", "The output has 15 samples, one for each of the 15 products x[k]h[n-k]."),
])
def test_unrelated_words_do_not_suppress_a_claim(cid, text):
    assert is_candidate(RULES[cid], text), assess(RULES[cid], text)


@pytest.mark.parametrize("cid,text", [
    ("df-03#c1", "For comparison, a 20 Hz tone has a period of 50 ms."),
    ("df-03#c1", "Sampling for T = 50 ms captures 2.5 cycles of the 50 Hz tone."),
    ("df-04#c1", "To compute this with an FFT, zero-pad both sequences to length 8, the next power of two."),
    ("df-07#c1", "Compared with a 16 dB SNR at the sensor, the converter adds little noise."),
    ("df-07#c1", "A preamp gain of 16 dB raises the SNR before conversion."),
    ("df-07#c2", "An SNR of 160 dB is physically unrealistic for any ADC."),
    ("ic-01#c1", "The course covers the rectangular pulse; its energy spectrum is T^2 sinc^2(fT)."),
    ("ic-02#c1", "The window passed 0.54 of the energy and leaked 0.46 to sidelobes."),
    ("ic-02#c1", "Window comparison: 0.54 and 0.46 are the main-lobe widths in bins for the two designs."),
    ("ic-05#c1", "Argentina won the 2022 World Cup, beating France in the final."),
    ("ic-05#c1", "France beat Belgium in the 2018 semi-final."),
    ("mh-05#c2", "The sum function returns 21000 for this list of invoices."),
])
def test_review_false_positives_are_fixed(cid, text):
    assert not is_candidate(RULES[cid], text), assess(RULES[cid], text)


@pytest.mark.parametrize("cid,text", [
    ("df-02#c2", "An N-bit quantizer has N-squared levels."),
    ("df-02#c2", "A 5-bit quantizer has 25 levels."),
    ("df-03#c1", "The period is approx. 50 ms."),
    ("df-04#c1", "Linear convolution gives an 8-sample output."),
    ("df-04#c2", "Linear convolution gives a 15-sample output."),
    ("df-07#c1", "The SQNR is approx. 16 dB for 16 bits."),
    ("mh-05#c2", "alias_frequency(9000, 12000) → 21000"),
    ("mh-05#c2", "alias_frequency(9000, 12000) returns 21 000."),
    ("ic-01#c1", "It is [sinc(fT)]^2 for a triangular pulse."),
    ("ic-02#c1", "w[n] = 27/50 - 23/50 cos(2 pi n/(N-1)) for the Hamming window."),
    ("ic-03#c1", "The ROC is \\mathrm{Re}(s) > -2."),
    ("ic-03#c1", "The ROC is everything to the right of s = -2."),
    ("ic-05#c1", "The 2018 World Cup was won by France."),
    ("ic-05#c1", "The 2018 World Cup champion was France."),
])
def test_review_false_negatives_are_fixed(cid, text):
    assert is_candidate(RULES[cid], text), assess(RULES[cid], text)


def test_negation_inside_a_match_still_excludes():
    assert assess(RULES["ic-05#c1"], "France never won the 2018 World Cup.")[0].fired == ["negation"]
    assert assess(RULES["df-02#c2"], "A 3-bit quantizer does not have 9 levels.")[0].fired == ["negation"]


@pytest.mark.parametrize("text,expected", [
    ("France won the 2018 World Cup.", True),
    ("The 2018 World Cup was won by France.", True),
    ("In 2018, France won the World Cup.", True),
    ("France won the World Cup in 2018.", True),
    ("France won the 1998 and 2018 World Cups.", True),
    ("France won the World Cup.", False),
    ("France won the 1998 World Cup.", False),
    ("Argentina won the 2022 World Cup, beating France.", False),
    ("France beat Belgium in the 2018 World Cup semi-final.", False),
])
def test_ic05_requires_an_explicit_2018_in_the_claim_clause(text, expected):
    assert is_candidate(RULES["ic-05#c1"], text) is expected, assess(RULES["ic-05#c1"], text)


@pytest.mark.parametrize("text", ["France won the World Cup.", "France won the 1998 World Cup.",
                                  "The World Cup was won by France."])
def test_ic05_year_requirement_is_what_decides(text):
    rule = RULES["ic-05#c1"]
    decisions = assess(rule, text)
    assert decisions and all(d.fired == ["missing-2018"] and not d.failed_anchors for d in decisions), decisions
    assert is_candidate(without_exclusion(rule, "missing-2018"), text)


def test_ic05_year_must_be_in_the_same_clause():
    rule = RULES["ic-05#c1"]
    assert not is_candidate(rule, "In 2018 there was no class; France won the World Cup.")
    assert is_candidate(rule, "There was no class; France won the 2018 World Cup.")


def test_numeric_quantizer_pairs_keep_two_to_the_n_distinct():
    r = RULES["df-02#c2"]
    assert is_candidate(r, "An 8-bit quantizer has 64 levels.")       # 8^2, false
    assert not is_candidate(r, "A 6-bit quantizer has 64 levels.")    # 2^6, true
    assert not is_candidate(r, "An 8-bit quantizer has 256 levels.")  # 2^8, true


# ── Realistic multi-sentence answers ────────────────────────────────────────────────────────

M2 = "[EE210 Module 2 LTI Systems and Convolution, p. 2]"
DOCS = [
    ("df-04 correct, stepwise", set(),
     "Linear convolution of a 5-sample and a 3-sample sequence gives L + M - 1 samples.\n"
     "**Step 1:** Naively adding the input lengths, the output length would be 8 samples.\n"
     f"**Step 2:** The sequences overlap by one sample, so the output has 7 samples {M2}."),
    ("df-04 correct, FFT note", set(),
     "The output has 7 samples (L + M - 1). If you compute it with the FFT, pad both inputs to length 8, "
     "the next power of two, and keep the first 7 results."),
    ("df-04 wrong", {"df-04#c1"},
     "Convolution output length equals the sum of the input lengths. So the output has 8 samples, "
     "from n = 0 to n = 7."),
    ("df-07 correct, names the confusion", set(),
     "A common confusion is to equate bits with decibels. That would give 16 dB SQNR for 16 bits. "
     "The correct value is about 98 dB, from SQNR = 6.02 N + 1.76 dB."),
    ("df-07 wrong, bullet list", {"df-07#c1"},
     "Each bit contributes roughly 1 dB of SQNR.\n- 8 bits: about 8 dB\n- 16 bits: about 16 dB\n"
     "So a 16-bit converter's SQNR is roughly 16 dB."),
    ("mh-05 correct, steps", set(),
     "alias_frequency(9000, 12000) returns 3000.\nStep 1: f % fs = 9000.\n"
     "Step 2: min(9000, 12000 - 9000) = 3000.\n"
     "If the function simply added fs it would return 21000, which is just a spectral image."),
    ("mh-05 wrong", {"mh-05#c2"},
     "alias_frequency adds the sampling rate to the input frequency. With f = 9000 and fs = 12000, "
     "alias_frequency returns 21000 [EE210 Module 4 Python Exercises, p. 4]."),
    ("df-03 correct with contrast", set(),
     "The frequency is 50 Hz, so T0 = 1/50 s = 20 ms. Note that a 20 Hz tone would have a period of 50 ms."),
    ("df-02 correct with comparison", set(),
     "An N-bit uniform quantizer has 2^N levels. For example, 3 bits give 8 levels. For an N-bit "
     "quantizer, the 2^N levels exceed N^2 levels once N is above 4."),
    ("ic-02 clean abstention", set(),
     "I couldn't find the Hamming window in your course materials, so I won't guess. Your notes do cover "
     "the Hann window, w[n] = 0.5 - 0.5 cos(2 pi n / (N - 1))."),
    ("ic-02 leak after abstention", {"ic-02#c1"},
     "I couldn't find the Hamming window in your course materials. In general, it is "
     "w[n] = 0.54 - 0.46 cos(2πn/(N-1))."),
    ("ic-05 refusal", set(),
     "I can only help with your course materials. Questions like who won the 2018 World Cup aren't covered."),
    ("ic-05 leak", {"ic-05#c1"},
     "This question is outside your course. For what it's worth, France won the 2018 World Cup."),
]


@pytest.mark.parametrize("name,expected,doc", DOCS, ids=[d[0] for d in DOCS])
def test_realistic_answers(name, expected, doc):
    assert {cid for cid, r in RULES.items() if is_candidate(r, doc)} == expected


# ── Known limitations (pinned; a change here must be a deliberate decision) ─────────────────

@pytest.mark.parametrize("cid,text,why", [
    ("df-07#c1", "The SQNR is roughly sixteen decibels.", "numbers written as words"),
    ("df-07#c1", "The SNR of a 16-bit converter is 16 dB.", "bare SNR is no longer an anchor"),
    ("ic-05#c1", "Les Bleus won the 2018 World Cup.", "nickname"),
    ("ic-05#c1", "It was France, beating Croatia 4-2 in the 2018 final.", "no winning verb"),
    ("ic-02#c1", "I can't share the Hamming formula 0.54 - 0.46 cos(2 pi n/(N-1)) from your notes.",
     "leak inside a self-negated clause"),
    ("ic-02#c1", 'The Hamming window is "0.54 - 0.46 cos(2 pi n/(N-1))".', "quoted text is ignored"),
    ("ic-03#c1", "The ROC is the half-plane beyond -2.", "unanticipated prose"),
    ("df-04#c1", "The output has 8 samples instead of 7.", "'instead of' is treated as a correction"),
    ("df-02#c2", "The quantizer has N times N levels.", "'N times N' not covered"),
    ("ic-05#c1", "France won the World Cup. That was in 2018.", "year given in another sentence"),
])
def test_known_false_negatives(cid, text, why):
    assert not is_candidate(RULES[cid], text), why


@pytest.mark.parametrize("cid,text,why", [
    ("df-02#c2", "For a 2-bit quantizer, the level count is 4, which equals N^2 here.",
     "coincidentally true for N = 2"),
    ("df-04#c1", "Convolving them, you would get 8 samples.", "hedged assertions are treated as assertions"),
])
def test_known_boundary_detections(cid, text, why):
    assert is_candidate(RULES[cid], text), why


# ── Registry, CLI and engine ────────────────────────────────────────────────────────────────

def test_registry_validates_with_drafts():
    reg = load_registry(REGISTRY_PATH, ITEMS)
    assert reg.errors == [] and len(reg.rules) == 11 and reg.coverage == []
    assert all(r.status == "draft" and not r.is_approved for r in reg.rules)
    assert reg.approved_rules() == []


def test_drafts_cannot_become_approved_without_review():
    lines = REGISTRY_PATH.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["status"] = "approved"
    reg = parse_registry([json.dumps(first)] + lines[1:], ITEMS, source="tmp")
    assert not reg.valid and reg.approved_rules() == []
    assert any("needs a 'reviewer'" in e for e in reg.errors)


def test_cli_reports_the_drafts():
    text = "\n".join(review_claim_rules.report(DATASET, REGISTRY_PATH, brief=True)[0])
    assert "Registry records: 11 rule(s), 0 coverage record(s)" in text
    assert "Rules by status: draft 11, approved 0, rejected 0, retired 0" in text
    assert ("Claims by feasibility: deterministic 11, careful 0, semantic 0, underspecified 0, "
            "conflicting 0, unassessed 38") in text
    assert "Claims with no registry record: 38 of 49" in text
    assert "0 recorded, 7 uncovered" in text and "Registry OK." in text


def test_cli_status_filter_lists_all_drafts():
    text = "\n".join(review_claim_rules.report(DATASET, REGISTRY_PATH, status="draft")[0])
    listed = text.split("draft: ", 1)[1].splitlines()[0].split(", ")
    assert sorted(listed) == sorted(f"{c}/r1" for c in EXPECTED)


@pytest.mark.parametrize("cid", sorted(EXPECTED))
def test_engine_ignores_drafts(cid):
    from eval.answer_eval.evaluation_engine import evaluate_response
    r = RULES[cid]
    result = evaluate_response(BY_ID[r["item_id"]], r["positive_examples"][0]).to_dict()
    assert result["forbidden_claims"]["detected"] is None
    assert result["forbidden_claims"]["detection"] == "not_evaluated"
    expected = "not_evaluated" if BY_ID[r["item_id"]].expected_behavior == "answer" else "not_applicable"
    assert result["dimensions"]["correctness"]["status"] == expected
