"""Sprint 4 Phase 3: the production forbidden-claim matcher (claim_matcher.py).

Run: python -m pytest eval/answer_eval/tests -q -p no:cacheprovider

Approved-rule behaviour is tested only with SYNTHETIC approved rules (rule ids ending in /r9) in
temporary registry text. The real registry is never modified and all 11 real rules stay drafts.
"""
from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_claim_rule_drafts as P2  # noqa: E402  (shared case tables; no matching logic)
from eval.answer_eval import claim_matcher as M  # noqa: E402
from eval.answer_eval.claim_rules import REGISTRY_PATH  # noqa: E402
from eval.answer_eval.response_normalizer import normalize_text  # noqa: E402
from eval.answer_eval.schema import Claim  # noqa: E402

ITEMS, BY_ID = P2.ITEMS, P2.BY_ID

# ── Clock ────────────────────────────────────────────────────────────────────────────────────
# Every fixture date is derived from the matcher's own current date, so the suite does not depend on
# the calendar or the host time zone. When CLAIM_MATCHER_TEST_TODAY=YYYY-MM-DD is set (only by
# test_suite_passes_under_a_controlled_earlier_or_later_clock, in a subprocess), the matcher's clock
# is replaced by a fixed date so the whole suite can be rerun as if on another day. The matcher's
# real-date upper bound (Q5) is unchanged: it simply reads the controlled date.
CLOCK_ENV = "CLAIM_MATCHER_TEST_TODAY"
_REAL_DATE = dt.date


class _ControlledDateMeta(type):
    def __instancecheck__(cls, obj):  # real date objects stay instances of the shimmed date type
        return isinstance(obj, _REAL_DATE)


class _ControlledDate(_REAL_DATE, metaclass=_ControlledDateMeta):
    @classmethod
    def today(cls):
        return _REAL_DATE.fromisoformat(os.environ[CLOCK_ENV])


if os.environ.get(CLOCK_ENV):
    _shim = types.SimpleNamespace(**{k: getattr(dt, k) for k in dir(dt) if not k.startswith("__")})
    _shim.date = _ControlledDate
    M._dt = _shim  # only in the controlled-clock subprocess


def current_date() -> dt.date:
    """The date the matcher treats as today (real, or controlled in the subprocess run)."""
    return M._dt.date.today()


REAL_TODAY = current_date()
TODAY = REAL_TODAY - dt.timedelta(days=30)        # fixture compilation date, always in the past
REVIEWED_ON = TODAY - dt.timedelta(days=4)        # synthetic approvals are reviewed before TODAY
REGISTRY_TEXT = REGISTRY_PATH.read_text(encoding="utf-8")
REAL = M.compile_registry_text(REGISTRY_TEXT, ITEMS, source="claim_rules.jsonl", today=TODAY)
RAW = {json.loads(line)["claim_id"]: json.loads(line) for line in REGISTRY_TEXT.splitlines() if line.strip()}


def synthetic(cid: str, k: int = 9, **over) -> dict:
    """A synthetic APPROVED copy of a real draft (rule id .../r9) for temporary registries only."""
    r = copy.deepcopy(RAW[cid])
    r.update(rule_id=f"{cid}/r{k}", status="approved", reviewer="kartik",
             reviewed_on=REVIEWED_ON.isoformat())
    r.update(over)
    return r


def registry(*records: dict, crlf: bool = False) -> M.CompiledRegistry:
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    if crlf:
        text = text.replace("\n", "\r\n")
    return M.compile_registry_text(text, ITEMS, source="tmp.jsonl", today=TODAY)


def spans(text_spans):
    return [(s.start, s.end, s.text) for s in text_spans]


# ── Reader: fixed outputs and offsets ────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("One. Two.", [(0, 4, "One."), (5, 9, "Two.")]),
    ("The SQNR is approx. 16 dB. Next, e.g. this.", [(0, 26, "The SQNR is approx. 16 dB."), (27, 43, "Next, e.g. this.")]),
    ("w = 0.54 - 0.46 cos(x). Done", [(0, 23, "w = 0.54 - 0.46 cos(x)."), (24, 28, "Done")]),
    ("Line one\nLine two", [(0, 8, "Line one"), (9, 17, "Line two")]),
    ("Why? Because.", [(0, 4, "Why?"), (5, 13, "Because.")]),
    ("The answer is no. The output has 8 samples.", [(0, 17, "The answer is no."), (18, 43, "The output has 8 samples.")]),
    ("See No. 5 for details. Next.", [(0, 22, "See No. 5 for details."), (23, 28, "Next.")]),
    ("Refer to no. 12 here. End.", [(0, 21, "Refer to no. 12 here."), (22, 26, "End.")]),
    ("Fig. 3 shows it. vs. the other.", [(0, 16, "Fig. 3 shows it."), (17, 31, "vs. the other.")]),
    ("Done!!  Next", [(0, 6, "Done!!"), (8, 12, "Next")]),
    ("- 8 bits: 8 dB\n- 16 bits: 16 dB", [(0, 14, "- 8 bits: 8 dB"), (15, 31, "- 16 bits: 16 dB")]),
    ("   ", []),
])
def test_sentence_spans(text, expected):
    assert spans(M.split_sentences(text)) == expected


@pytest.mark.parametrize("sentence,expected", [
    ("I couldn't find this, but France won.", ["I couldn't find this", "but France won."]),
    ("Your notes don't cover it (commonly 0.54 - 0.46 cos(x)).", ["Your notes don't cover it", "(commonly 0.54 - 0.46 cos(x))."]),
    ("The period is 50 ms, so it does not depend on amplitude.", ["The period is 50 ms", "so it does not depend on amplitude."]),
    ("It is 20 ms, not 50 ms.", ["It is 20 ms", "not 50 ms."]),
    ("It is wrong; the output has 8 samples.", ["It is wrong", "the output has 8 samples."]),
    ("The output has 8 samples — a common misconception.", ["The output has 8 samples", "a common misconception."]),
    ("The output has 8 samples - wrong.", ["The output has 8 samples", "wrong."]),
    ("The output has 8 samples -- wrong.", ["The output has 8 samples", "wrong."]),
    ("I can't say though it is listed.", ["I can't say", "though it is listed."]),
    ("Does alias_frequency(9000, 12000) return 21000?", ["Does alias_frequency(9000, 12000) return 21000?"]),
    ("w[n] = 0.54 - 0.46 cos(2 pi n / (N - 1)) here.", ["w[n] = 0.54 - 0.46 cos(2 pi n / (N - 1)) here."]),
    ("For comparison, a 20 Hz tone has 50 ms.", ["For comparison, a 20 Hz tone has 50 ms."]),
    ("The output, which is long, has 8 samples.", ["The output", "which is long, has 8 samples."]),
])
def test_clause_texts(sentence, expected):
    assert [c.text for c in M.split_clauses(sentence)] == expected


def test_clause_offsets_are_absolute_and_exact():
    text = "First sentence. I couldn't find this, but France won the 2018 World Cup."
    sent = M.split_sentences(text)[1]
    for c in M.split_clauses(sent):
        assert text[c.start:c.end] == c.text
    assert [c.start for c in M.split_clauses(sent)] == [16, 38]


def test_prepare_text_blanks_quotes_without_moving_offsets():
    raw = 'He said "the output has 8 samples" today.'
    prepared = M.prepare_text(raw)
    assert len(prepared) == len(normalize_text(raw)) and "8 samples" not in prepared
    assert M.prepare_text("It’s 8 samples") == "It’s 8 samples"


def test_assessment_spans_index_the_normalised_text():
    corpus = [P2.DECISIVE_EXCLUSIONS[k] for k in sorted(P2.DECISIVE_EXCLUSIONS)] + [d for _, _, d in P2.DOCS]
    corpus += ["The period is 50 ms.\r\nNext line.", "Line\r\none: the output has 8 samples."]
    checked = 0
    for text in corpus:
        norm = normalize_text(text)
        for r in RAW.values():
            for a in M.assess_rule(r, text):
                for sp in [a.match, a.sentence, a.clause, *a.touched_clauses, *(h.span for h in a.exclusions)]:
                    assert norm[sp.start:sp.end] == sp.text
                    checked += 1
                if a.next_clause:
                    assert norm[a.next_clause.start:a.next_clause.end] == a.next_clause.text
                assert a.sentence.start <= a.match.start and a.match.end <= a.sentence.end
    assert checked > 300


def test_quoted_match_offsets_show_original_text():
    rule = RAW["mh-05#c2"]
    a = M.assess_rule(rule, 'alias_frequency(9000, 12000) returns 21000, as print("x") shows.')[0]
    assert a.is_affirmative and a.sentence.text.endswith('print("x") shows.')


# ── D5 and D4 (intentional changes from the Phase 2 test reader) ─────────────────────────────

def test_d5_ordinary_no_full_stop_ends_a_sentence():
    a = M.assess_rule(RAW["df-04#c1"], "That is not right, the answer is no. The output has 8 samples.")
    assert [x.outcome for x in a] == ["affirmative"]
    assert a[0].sentence.text == "The output has 8 samples."


def test_d5_numbered_no_is_still_an_abbreviation():
    assert len(M.split_sentences("No. 5 of the notes says the output has 8 samples.")) == 1
    assert len(M.split_sentences("See no. 12. The end.")) == 2


def test_d4_exclusions_cover_every_clause_the_match_touches():
    a = M.assess_rule(RAW["df-04#c1"], "The output length, which is not 7, is 8.")[0]
    assert a.match.text == "length, which is not 7, is 8"
    assert len(a.touched_clauses) == 2
    assert a.outcome == "excluded" and a.exclusion_names == ("negation",)
    assert a.clause.text == "The output length, which is not 7, is 8."


def test_d4_next_clause_is_after_the_last_touched_clause():
    a = M.assess_rule(RAW["df-04#c1"], "The output length, which varies, is 8, which is wrong.")[0]
    assert a.next_clause.text == "which is wrong." and a.exclusion_names == ("rejected-after",)


def test_d4_single_clause_matches_are_unchanged():
    a = M.assess_rule(RAW["df-03#c1"], "The period is 50 ms, so it does not depend on the amplitude.")[0]
    assert a.outcome == "affirmative" and len(a.touched_clauses) == 1


# ── Exclusion names (D2) ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ex,expected", [
    (r"(?#negation)\bnot\b", ("negation", "clause")),
    (r"(?#rejected-after;next-clause)^\W*wrong", ("rejected-after", "next-clause")),
    (r"(?#missing-2018)^(?!.*\b2018\b)", ("missing-2018", "clause")),
])
def test_exclusion_names_parse(ex, expected):
    assert M.parse_exclusion_name(ex) == expected


@pytest.mark.parametrize("ex", [
    r"\bnot\b", r"(?#)x", r"(?#Negation)x", r"(?#neg ation)x", r"(?#neg_ation)x",
    r"(?#x;nextclause)y", r"(?#x;clause)y", r"(?#x;next-clause;more)y", r"(?#-x)y",
])
def test_bad_exclusion_names_are_rejected(ex):
    with pytest.raises(ValueError):
        M.parse_exclusion_name(ex)


@pytest.mark.parametrize("bad", [r"\bnot\b", r"(?#x;nextclause)y", r"(?#Bad)y"])
def test_bad_exclusion_name_makes_the_whole_registry_invalid(bad):
    r = copy.deepcopy(RAW["df-04#c1"])
    r["exclusions"] = r["exclusions"] + [bad]
    reg = registry(r, synthetic("df-04#c2"))
    assert not reg.valid and reg.rules == () and reg.errors
    rep = M.match_response(BY_ID["df-04"], "The output has 15 samples.", reg, include_drafts=True)
    assert rep.status == "not_evaluated" and rep.confirmed == () and rep.shadow == () and rep.assessments == ()


def test_duplicate_exclusion_names_rejected():
    r = copy.deepcopy(RAW["df-04#c1"])
    r["exclusions"] = r["exclusions"] + [r"(?#negation)\bnever\b"]
    assert not registry(r).valid


# ── All 11 real rules: drafts only, never confirmed ──────────────────────────────────────────

def test_real_registry_compiles_with_eleven_drafts_and_no_eligible_rule():
    assert REAL.valid and REAL.errors == ()
    assert len(REAL.rules) == 11 and {r.status for r in REAL.rules} == {"draft"}
    assert REAL.approved_rule_ids == frozenset()
    assert not any(r.eligible for r in REAL.rules)


@pytest.mark.parametrize("cid", sorted(RAW))
def test_real_rules_never_confirm(cid):
    r = RAW[cid]
    item = BY_ID[r["item_id"]]
    for text in r["positive_examples"]:
        rep = M.match_response(item, text, REAL)
        assert rep.status == "not_evaluated" and rep.confirmed == () and rep.shadow == ()
        assert rep.status_reason == "no production-eligible rule for this item"
        shadow = M.match_response(item, text, REAL, include_drafts=True)
        assert shadow.status == "not_evaluated" and shadow.confirmed == ()
        assert [d.claim_id for d in shadow.shadow] == [cid]
        assert all(not a.production_eligible for a in shadow.assessments)
        assert all(not d.production_eligible for d in shadow.shadow)


# ── Shadow pipeline reproduces every Phase 2 decision ────────────────────────────────────────

@pytest.mark.parametrize("cid,tag", sorted(P2.DECISIVE_EXCLUSIONS))
def test_decisive_exclusions_through_match_response(cid, tag):
    text = P2.DECISIVE_EXCLUSIONS[(cid, tag)]
    rep = M.match_response(BY_ID[cid.split("#")[0]], text, REAL, include_drafts=True)
    mine = [a for a in rep.assessments if a.claim_id == cid]
    assert mine and all(a.outcome == "excluded" and a.exclusion_names == (tag,) for a in mine)
    assert cid not in {d.claim_id for d in rep.shadow} and rep.confirmed == ()


@pytest.mark.parametrize("cid,index", sorted(P2.DECISIVE_ANCHORS))
def test_decisive_anchors_through_match_response(cid, index):
    rep = M.match_response(BY_ID[cid.split("#")[0]], P2.DECISIVE_ANCHORS[(cid, index)], REAL, include_drafts=True)
    mine = [a for a in rep.assessments if a.claim_id == cid]
    assert mine and all(a.outcome == "no_anchor" and a.failed_anchors == (index,) for a in mine)


@pytest.mark.parametrize("cid", sorted(P2.BARE_VALUE))
def test_bare_values_through_match_response(cid):
    bare, text = P2.BARE_VALUE[cid]
    rep = M.match_response(BY_ID[cid.split("#")[0]], text, REAL, include_drafts=True)
    assert cid not in {d.claim_id for d in rep.shadow}
    assert any(a.is_affirmative for a in M.assess_rule({**RAW[cid], "pattern": bare}, text))


@pytest.mark.parametrize("name,expected,doc", P2.DOCS, ids=[d[0] for d in P2.DOCS])
def test_realistic_answers_through_match_response(name, expected, doc):
    found = set()
    for item_id in {cid.split("#")[0] for cid in RAW}:
        rep = M.match_response(BY_ID[item_id], doc, REAL, include_drafts=True)
        found |= {d.claim_id for d in rep.shadow}
        assert rep.confirmed == ()
    assert found == expected


@pytest.mark.parametrize("cid,text", [
    ("ic-01#c1", "I couldn't find this in your notes, but generally the triangle pulse transforms to sinc^2."),
    ("ic-02#c1", "Your notes don't cover the Hamming window (commonly 0.54 - 0.46 cos(2 pi n/(N-1)))."),
    ("ic-05#c1", "This isn't in your course, but France won the 2018 World Cup."),
    ("ic-03#c1", "Not covered in your notes, though the ROC would be Re(s) > -2."),
])
def test_mixed_abstention_leak_in_shadow(cid, text):
    rep = M.match_response(BY_ID[cid.split("#")[0]], text, REAL, include_drafts=True)
    assert [d.claim_id for d in rep.shadow] == [cid] and rep.shadow[0].reason == "unsupported"
    occ = rep.shadow[0].occurrences[0]
    assert "not" not in occ.clause.text.lower() and "n't" not in occ.clause.text


@pytest.mark.parametrize("cid,text,names", [
    ("df-04#c1", "The output does not have 8 samples; it has 7.", ("negation",)),
    ("df-04#c1", "Step 1: Naively adding the input lengths, the output length would be 8 samples.", ("hypothetical",)),
    ("mh-05#c2", "If alias_frequency added fs, it would return 21000.", ("hypothetical",)),
    ("df-07#c1", "Some sources say the SQNR is 16 dB.", ("reported",)),
    ("df-04#c1", "The output has 8 samples, which is wrong.", ("rejected-after",)),
])
def test_negation_hypothetical_reported_rejection(cid, text, names):
    rep = M.match_response(BY_ID[cid.split("#")[0]], text, REAL, include_drafts=True)
    mine = [a for a in rep.assessments if a.claim_id == cid]
    assert mine and all(a.exclusion_names == names for a in mine) and rep.shadow == ()


# ── Synthetic approved rules ─────────────────────────────────────────────────────────────────

def test_synthetic_approved_rule_confirms():
    reg = registry(synthetic("df-04#c2"))
    assert reg.valid and reg.rules[0].eligible and reg.rules[0].ineligibility == ()
    rep = M.match_response(BY_ID["df-04"], "Convolving gives a 15-sample output.", reg)
    assert rep.status == "affirmative_match" and rep.production_rule_ids == ("df-04#c2/r9",)
    d = rep.confirmed[0]
    assert (d.claim_id, d.reason, d.rule_ids, d.rule_statuses, d.production_eligible) == \
        ("df-04#c2", "false", ("df-04#c2/r9",), ("approved",), True)
    assert d.claim_text == BY_ID["df-04"].forbidden_claims[1].text
    assert d.occurrences[0].match.text == "15-sample"


def test_synthetic_approved_rule_no_match_is_not_a_verdict():
    reg = registry(synthetic("df-04#c2"))
    rep = M.match_response(BY_ID["df-04"], BY_ID["df-04"].reference_answer, reg)
    assert rep.status == "no_affirmative_match" and rep.confirmed == ()
    words = " ".join([rep.status, rep.status_reason, rep.to_dict()["note"]]).lower()
    assert not re.search(r"\b(?:pass(?:ed|es)?|correct(?:ly|ness)?|valid answer)\b", words)


def test_unsupported_and_false_are_reported_separately():
    reg = registry(synthetic("ic-02#c1"), synthetic("df-04#c2"))
    rep = M.match_response(BY_ID["ic-02"], "In general, w[n] = 0.54 - 0.46 cos(2 pi n/(N-1)).", reg)
    out = rep.to_dict()
    assert out["confirmed_by_reason"] == {"false": [], "unsupported": ["ic-02#c1"]}
    rep2 = M.match_response(BY_ID["df-04"], "The output has 15 samples.", reg)
    assert rep2.to_dict()["confirmed_by_reason"] == {"false": ["df-04#c2"], "unsupported": []}


def test_rules_only_apply_to_their_own_item():
    reg = registry(synthetic("df-04#c2"))
    rep = M.match_response(BY_ID["df-03"], "The convolution output has 15 samples.", reg)
    assert rep.status == "not_evaluated" and rep.assessments == ()


def test_drafts_and_approved_together():
    reg = registry(synthetic("df-04#c2"), RAW["df-04#c1"])
    text = "The output has 8 samples. Or rather, the output has 15 samples."
    rep = M.match_response(BY_ID["df-04"], text, reg)
    assert [d.claim_id for d in rep.confirmed] == ["df-04#c2"] and rep.shadow == ()
    rep = M.match_response(BY_ID["df-04"], text, reg, include_drafts=True)
    assert [d.claim_id for d in rep.confirmed] == ["df-04#c2"]
    assert [d.claim_id for d in rep.shadow] == ["df-04#c1"] and rep.status == "affirmative_match"
    only_draft = M.match_response(BY_ID["df-04"], "The output has 8 samples.", reg, include_drafts=True)
    assert only_draft.status == "no_affirmative_match" and only_draft.confirmed == ()
    assert [d.claim_id for d in only_draft.shadow] == ["df-04#c1"]


# ── Aggregation ──────────────────────────────────────────────────────────────────────────────

def test_duplicate_sentences_aggregate_into_one_detection():
    reg = registry(synthetic("df-04#c2"))
    rep = M.match_response(BY_ID["df-04"], "The output has 15 samples. Again: the output has 15 samples.", reg)
    assert len(rep.confirmed) == 1 and len(rep.confirmed[0].occurrences) == 2
    starts = [o.match.start for o in rep.confirmed[0].occurrences]
    assert starts == sorted(starts)


def test_multiple_rules_for_one_claim_merge():
    reg = registry(synthetic("df-04#c2", 8), synthetic("df-04#c2", 9))
    rep = M.match_response(BY_ID["df-04"], "The output has 15 samples.", reg)
    assert len(rep.confirmed) == 1
    assert rep.confirmed[0].rule_ids == ("df-04#c2/r8", "df-04#c2/r9")
    assert len(rep.confirmed[0].occurrences) == 2


def test_multiple_claims_stay_separate_and_ordered():
    reg = registry(synthetic("df-04#c2"), synthetic("df-04#c1"))
    rep = M.match_response(BY_ID["df-04"], "It has 15 output samples. No wait, the output has 8 samples.", reg)
    assert [d.claim_id for d in rep.confirmed] == ["df-04#c1", "df-04#c2"]


# ── Rejected, retired and invalid registries ─────────────────────────────────────────────────

@pytest.mark.parametrize("status", ["rejected", "retired"])
def test_rejected_and_retired_rules_are_never_evaluated(status):
    r = synthetic("df-04#c2", status=status)
    reg = registry(r)
    assert reg.valid and reg.rules == ()
    rep = M.match_response(BY_ID["df-04"], "The output has 15 samples.", reg, include_drafts=True)
    assert rep.status == "not_evaluated" and rep.assessments == () and rep.shadow == ()


def test_invalid_registry_evaluates_nothing():
    good = json.dumps(synthetic("df-04#c2"))
    text = good + "\n{not json\n"
    reg = M.compile_registry_text(text, ITEMS, source="tmp", today=TODAY)
    assert not reg.valid and reg.rules == () and any("invalid JSON" in e for e in reg.errors)
    rep = M.match_response(BY_ID["df-04"], "The output has 15 samples.", reg, include_drafts=True)
    assert rep.status == "not_evaluated" and "registry invalid" in rep.status_reason
    assert rep.confirmed == () and rep.shadow == () and rep.assessments == ()


def test_registry_with_a_drifted_record_is_invalid_as_a_whole():
    drift = copy.deepcopy(RAW["df-03#c1"])
    drift["claim_text"] = "The period is 51 ms"
    reg = registry(synthetic("df-04#c2"), drift)
    assert not reg.valid and reg.rules == ()


def test_missing_registry_file(tmp_path):
    reg = M.load_compiled_registry(ITEMS, tmp_path / "nope.jsonl")
    assert not reg.valid and reg.rules == ()


def test_load_compiled_registry_matches_text_compile():
    loaded = M.load_compiled_registry(ITEMS, REGISTRY_PATH, today=TODAY)
    assert loaded.digest == REAL.digest and loaded.valid and len(loaded.rules) == 11


# ── Example validation (D3) and eligibility recheck ─────────────────────────────────────────

def test_failing_positive_example_makes_rule_ineligible():
    r = synthetic("df-04#c2", positive_examples=["The output has 15 samples.", "Nothing relevant here."])
    rule = registry(r).rules[0]
    assert not rule.eligible and "positive example 2 is not an affirmative match" in rule.ineligibility


def test_affirmative_negative_example_makes_rule_ineligible():
    negs = list(RAW["df-04#c2"]["negative_examples"]) + ["The convolution output has 15 samples."]
    rule = registry(synthetic("df-04#c2", negative_examples=negs)).rules[0]
    assert not rule.eligible and any("negative example" in r for r in rule.ineligibility)


def test_rule_matching_its_reference_answer_is_ineligible():
    r = synthetic("df-04#c2", pattern=r"\bsamples\b", exclusions=[r"(?#question)\?"],
                  positive_examples=["The output samples come from convolution.", "Its output has samples."],
                  negative_examples=["Is it about output samples?", "Not relevant.", "It is not samples."])
    rule = registry(r).rules[0]
    assert "matches its own item's reference answer" in rule.ineligibility and not rule.eligible


def test_rule_matching_its_question_is_ineligible():
    r = synthetic("df-04#c2", pattern=r"\bsequence\b", anchors=[r"convol"], exclusions=[r"(?#negation)\bnot\b"],
                  positive_examples=["The sequence convolution result.", "Convolve the sequence."],
                  negative_examples=["It is not a sequence convolution.", "Nothing here.", "No match."])
    rule = registry(r).rules[0]
    assert rule.ineligibility == ("matches its own item's question",)


def test_ineligible_approved_rule_never_runs():
    r = synthetic("df-04#c2", positive_examples=["The output has 15 samples.", "Nothing relevant here."])
    reg = registry(r)
    rep = M.match_response(BY_ID["df-04"], "The output has 15 samples.", reg, include_drafts=True)
    assert rep.status == "not_evaluated" and rep.assessments == () and rep.shadow == ()
    assert rep.ineligible_rules[0][0] == "df-04#c2/r9"


def test_match_response_rechecks_eligibility():
    reg = registry(synthetic("df-04#c2"))
    rule = reg.rules[0]
    forged_draft = dataclasses.replace(reg, rules=(dataclasses.replace(rule, status="draft"),))
    assert M.match_response(BY_ID["df-04"], "The output has 15 samples.", forged_draft).confirmed == ()
    not_listed = dataclasses.replace(reg, approved_rule_ids=frozenset())
    assert M.match_response(BY_ID["df-04"], "The output has 15 samples.", not_listed).confirmed == ()
    invalid = dataclasses.replace(reg, valid=False)
    assert M.match_response(BY_ID["df-04"], "The output has 15 samples.", invalid).status == "not_evaluated"
    forged_eligible_draft = dataclasses.replace(reg, rules=(dataclasses.replace(rule, status="draft", eligible=True),))
    assert forged_eligible_draft.approved_rule_ids == frozenset({"df-04#c2/r9"})
    rep = M.match_response(BY_ID["df-04"], "The output has 15 samples.", forged_eligible_draft)
    assert rep.confirmed == () and rep.status == "not_evaluated"
    marked_ineligible = dataclasses.replace(reg, rules=(dataclasses.replace(rule, eligible=False),))
    assert M.match_response(BY_ID["df-04"], "The output has 15 samples.", marked_ineligible).confirmed == ()


def test_match_response_rechecks_the_claim_against_the_item():
    reg = registry(synthetic("df-04#c2"))
    item = BY_ID["df-04"]
    changed = dataclasses.replace(item, forbidden_claims=[item.forbidden_claims[0], Claim("The output has 16 samples", "false")])
    rep = M.match_response(changed, "The output has 15 samples.", reg)
    assert rep.confirmed == () and rep.status == "not_evaluated"
    # Since Q4 the altered item is refused before any rule runs (stricter than the claim recheck) ...
    assert "differs from the compiled dataset snapshot" in rep.status_reason and rep.assessments == ()
    # ... and the claim recheck itself still detects the change.
    assert M._claim_mismatch(reg.rules[0], changed) == "claim text or reason differs from the item"
    assert M._claim_mismatch(reg.rules[0], item) is None


def test_no_rule_is_auto_approved():
    for r in REAL.rules:
        assert r.status == "draft" and not r.eligible
    assert all(json.loads(line)["status"] == "draft" for line in REGISTRY_TEXT.splitlines() if line.strip())


# ── Digest, serialisation, determinism ───────────────────────────────────────────────────────

def test_digest_is_stable_across_crlf_and_lf():
    a, b = registry(synthetic("df-04#c2")), registry(synthetic("df-04#c2"), crlf=True)
    assert a.digest == b.digest and len(a.digest) == 64
    assert registry(synthetic("df-04#c2", rationale="changed")).digest != a.digest
    assert REAL.digest == M.registry_digest(REGISTRY_TEXT.replace("\r\n", "\n"))


def _json_types_only(value):
    if isinstance(value, dict):
        return all(isinstance(k, str) and _json_types_only(v) for k, v in value.items())
    if isinstance(value, list):
        return all(_json_types_only(v) for v in value)
    return value is None or isinstance(value, (str, int, float, bool))


def test_report_is_json_safe_and_deterministic():
    reg = registry(synthetic("df-04#c2"), RAW["df-04#c1"])
    text = "The output has 15 samples. The output has 8 samples, which is wrong. The output has 15 samples."
    first = M.match_response(BY_ID["df-04"], text, reg, include_drafts=True).to_dict()
    again = M.match_response(BY_ID["df-04"], text, registry(synthetic("df-04#c2"), RAW["df-04#c1"]),
                             include_drafts=True).to_dict()
    assert _json_types_only(first)
    assert json.dumps(first, sort_keys=True) == json.dumps(again, sort_keys=True)
    assert first["registry"]["digest"] == reg.digest
    assert first["status"] in M.STATUSES


def test_report_carries_full_traceability():
    reg = registry(synthetic("df-04#c2"))
    out = M.match_response(BY_ID["df-04"], "Note. The output has 15 samples.", reg).to_dict()
    occ = out["confirmed"][0]["occurrences"][0]
    assert occ["rule_id"] == "df-04#c2/r9" and occ["claim_id"] == "df-04#c2" and occ["item_id"] == "df-04"
    assert occ["rule_status"] == "approved" and occ["production_eligible"] is True
    assert out["normalized_text"][occ["match"]["start"]:occ["match"]["end"]] == occ["match"]["text"]
    assert occ["clause"]["text"] == "The output has 15 samples." and occ["failed_anchors"] == []
    assert out["confirmed"][0]["feasibility"] == ["deterministic"]


def test_report_records_exclusion_names_and_spans():
    reg = registry(synthetic("df-04#c2"))
    out = M.match_response(BY_ID["df-04"], "The output never has 15 samples.", reg).to_dict()
    hit = out["assessments"][0]["exclusions"][0]
    assert hit["name"] == "negation" and hit["scope"] == "clause" and hit["span"]["text"] == "never"
    assert out["status"] == "no_affirmative_match"


def test_results_are_immutable():
    rep = M.match_response(BY_ID["df-04"], "x", REAL)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rep.status = "affirmative_match"


def test_statuses_never_use_verdict_words():
    assert set(M.STATUSES) == {"not_evaluated", "no_affirmative_match", "affirmative_match"}
    src = (ROOT / "eval/answer_eval/claim_matcher.py").read_text(encoding="utf-8")
    assert not re.search(r"\b(?:pass(?:es|ed|ing)?|correct(?:ly|ness)?)\b", src, re.I)


# ── Edge inputs ──────────────────────────────────────────────────────────────────────────────

def test_none_response_is_not_evaluated():
    rep = M.match_response(BY_ID["df-04"], None, registry(synthetic("df-04#c2")))
    assert rep.status == "not_evaluated" and rep.status_reason == "no response supplied"
    assert rep.normalized_text is None


@pytest.mark.parametrize("bad", [5, b"bytes", ["x"]])
def test_non_string_input_raises(bad):
    with pytest.raises(TypeError):
        M.match_response(BY_ID["df-04"], bad, REAL)
    with pytest.raises(TypeError):
        M.assess_rule(RAW["df-04#c2"], bad)


def test_wrong_registry_type_raises():
    with pytest.raises(TypeError):
        M.match_response(BY_ID["df-04"], "x", {"rules": []})


@pytest.mark.parametrize("text", ["", "   ", "\n\n", "​"])
def test_empty_responses(text):
    rep = M.match_response(BY_ID["df-04"], text, registry(synthetic("df-04#c2")))
    assert rep.status == "no_affirmative_match" and rep.assessments == ()


def test_long_response_and_markdown():
    filler = "Filler sentence about signals. " * 3000
    text = filler + "\n\n**Answer:** the output has 15 samples.\n\n```python\nprint(len(y))\n```"
    rep = M.match_response(BY_ID["df-04"], text, registry(synthetic("df-04#c2")))
    assert rep.status == "affirmative_match" and len(rep.confirmed[0].occurrences) == 1


def test_crlf_nbsp_and_unicode_minus_inputs():
    reg = registry(synthetic("ic-03#c1"))
    rep = M.match_response(BY_ID["ic-03"], "Line one.\r\nThe ROC is Re(s) > −2.", reg)
    assert rep.status == "affirmative_match"


# ── Isolation from the evaluation engine ─────────────────────────────────────────────────────

@pytest.mark.parametrize("module", ["evaluation_engine.py", "evaluation_dimensions.py", "evaluation_result.py",
                                    "run_evaluation.py", "claim_rules.py", "review_claim_rules.py"])
def test_matcher_is_not_wired_into_other_modules(module):
    assert "claim_matcher" not in (ROOT / "eval/answer_eval" / module).read_text(encoding="utf-8")


def test_matcher_does_not_import_the_engine():
    src = (ROOT / "eval/answer_eval/claim_matcher.py").read_text(encoding="utf-8")
    assert "evaluation_engine" not in src and "evaluation_dimensions" not in src


# ══ Phase 3 review fixes ═════════════════════════════════════════════════════════════════════

# ── F1: eligibility is bound to the authoritative compilation of the registry source ─────────

X15 = "The output has 15 samples."


def test_f1_a1_hand_built_registry_cannot_confirm():
    forged_rule = dataclasses.replace(M.compile_rule(RAW["df-04#c2"]), status="approved", eligible=True,
                                      ineligibility=())
    forged = M.CompiledRegistry("forged", "0" * 64, True, (), (forged_rule,), frozenset({"df-04#c2/r1"}))
    rep = M.match_response(BY_ID["df-04"], X15, forged)
    assert rep.status == "not_evaluated" and rep.confirmed == () and rep.assessments == ()
    assert "does not match its source" in rep.status_reason


def test_f1_a1_hand_built_registry_with_genuine_source_text_cannot_confirm():
    forged_rule = dataclasses.replace(M.compile_rule(RAW["df-04#c2"]), status="approved", eligible=True)
    forged = dataclasses.replace(REAL, rules=(forged_rule,), approved_rule_ids=frozenset({"df-04#c2/r1"}))
    rep = M.match_response(BY_ID["df-04"], X15, forged, include_drafts=True)
    assert rep.status == "not_evaluated" and rep.confirmed == () and rep.shadow == ()
    assert "differs from the authoritative compilation" in rep.status_reason


def test_f1_a2_replace_eligible_and_ineligibility_cannot_confirm():
    genuine = registry(synthetic("df-04#c2", positive_examples=["The output has 15 samples.", "Nothing relevant here."]))
    assert not genuine.rules[0].eligible
    forged = dataclasses.replace(genuine, rules=(dataclasses.replace(genuine.rules[0], eligible=True, ineligibility=()),))
    rep = M.match_response(BY_ID["df-04"], X15, forged)
    assert rep.status == "not_evaluated" and rep.confirmed == ()
    assert M.match_response(BY_ID["df-04"], X15, genuine).confirmed == ()


def test_f1_a3_draft_relabelled_as_approved_cannot_confirm():
    genuine = registry(RAW["df-04#c2"])
    forged = dataclasses.replace(genuine, rules=(dataclasses.replace(genuine.rules[0], status="approved", eligible=True),),
                                 approved_rule_ids=frozenset({"df-04#c2/r1"}))
    rep = M.match_response(BY_ID["df-04"], X15, forged)
    assert rep.status == "not_evaluated" and rep.confirmed == ()


def test_f1_tampered_source_text_fails_the_digest_check():
    approved_text = json.dumps(synthetic("df-04#c2")) + "\n"
    rep = M.match_response(BY_ID["df-04"], X15, dataclasses.replace(REAL, source_text=approved_text))
    assert rep.status == "not_evaluated" and "does not match its source" in rep.status_reason


def test_f1_swapped_source_with_matching_digest_still_must_equal_its_compilation():
    approved_text = json.dumps(synthetic("df-04#c2")) + "\n"
    swapped = dataclasses.replace(REAL, source_text=approved_text, digest=M.registry_digest(approved_text))
    rep = M.match_response(BY_ID["df-04"], X15, swapped)
    assert rep.status == "not_evaluated" and rep.confirmed == ()


def test_f1_tampered_dataset_snapshot_fails_the_fingerprint_check():
    reg = registry(synthetic("df-04#c2"))
    item = reg.items[0]
    changed = dataclasses.replace(item, reference_answer=item.reference_answer + " x")
    forged = dataclasses.replace(reg, items=(changed,) + reg.items[1:])
    assert M._authoritative(forged) is None  # the fingerprint check alone rejects it
    assert M.match_response(BY_ID["df-04"], X15, forged).status == "not_evaluated"


def test_f1_mutating_items_after_compile_does_not_affect_the_registry():
    items = copy.deepcopy(list(ITEMS))
    reg = M.compile_registry_text(json.dumps(synthetic("df-04#c2")) + "\n", items, source="t", today=TODAY)
    next(i for i in items if i.id == "df-04").forbidden_claims.clear()
    assert M.match_response(BY_ID["df-04"], X15, reg).status == "affirmative_match"


def test_f1_genuine_compilation_still_confirms_and_is_its_own_authority():
    reg = registry(synthetic("df-04#c2"))
    assert reg.source_text and reg.items_fingerprint and reg.today == TODAY
    assert M._authoritative(reg) == reg
    assert M.match_response(BY_ID["df-04"], X15, reg).status == "affirmative_match"
    again = registry(synthetic("df-04#c2"))
    assert M.match_response(BY_ID["df-04"], X15, again).status == "affirmative_match"


def test_f1_review_date_is_part_of_the_binding():
    reg = registry(synthetic("df-04#c2"))
    moved = dataclasses.replace(reg, today=REVIEWED_ON - dt.timedelta(days=1))   # before reviewed_on
    assert M.compile_registry_text(reg.source_text, ITEMS, source=reg.source, today=moved.today).valid is False
    rep = M.match_response(BY_ID["df-04"], X15, moved)
    assert rep.status == "not_evaluated" and rep.confirmed == ()
    assert "differs from the authoritative compilation" in rep.status_reason


# ── F2: offsets refer to normalized_text ─────────────────────────────────────────────────────

def test_f2_report_declares_the_offset_contract():
    out = M.match_response(BY_ID["df-04"], X15, registry(synthetic("df-04#c2"))).to_dict()
    assert out["offsets_refer_to"] == "normalized_text" == M.OFFSETS_REFER_TO
    assert M.match_response(BY_ID["df-04"], None, REAL).to_dict()["offsets_refer_to"] == "normalized_text"


def test_f2_offsets_slice_normalized_text_with_unicode_spacing_and_crlf():
    raw = ("Intro line,​  with   spaces.\r\nThe  output has 15 samples—really.\r\n"
           "Again:\tthe output has 15 samples, café.")
    rep = M.match_response(BY_ID["df-04"], raw, registry(synthetic("df-04#c2")), include_drafts=True)
    out = rep.to_dict()
    norm = out["normalized_text"]
    assert norm == normalize_text(raw) and norm != raw
    occs = out["confirmed"][0]["occurrences"]
    assert len(occs) == 2
    for o in occs:
        for key in ("match", "sentence", "clause"):
            assert norm[o[key]["start"]:o[key]["end"]] == o[key]["text"]
        assert o["match"]["text"] == "15 samples"
    assert raw[occs[0]["match"]["start"]:occs[0]["match"]["end"]] != "15 samples"  # not original offsets


# ── F3: pattern-less drafts ──────────────────────────────────────────────────────────────────

def _no_pattern_draft():
    r = copy.deepcopy(RAW["df-03#c1"])
    r.update(rule_id="df-03#c1/r2", pattern=None)
    return r


def test_f3_pattern_less_draft_is_valid_but_not_runnable():
    reg = registry(synthetic("df-04#c2"), _no_pattern_draft())
    assert reg.valid and reg.not_runnable == (("df-03#c1/r2", "draft has no pattern"),)
    assert "df-03#c1/r2" not in {r.rule_id for r in reg.rules}
    assert M.match_response(BY_ID["df-04"], X15, reg).status == "affirmative_match"
    rep = M.match_response(BY_ID["df-03"], "The period is 50 ms.", reg, include_drafts=True)
    assert rep.status == "not_evaluated" and rep.assessments == ()
    assert rep.to_dict()["not_runnable_rules"] == [{"rule_id": "df-03#c1/r2", "reason": "draft has no pattern"}]


def test_f3_approved_rule_without_pattern_is_an_error():
    reg = registry(synthetic("df-04#c2", pattern=None))
    assert not reg.valid and any("needs a 'pattern'" in e for e in reg.errors)


@pytest.mark.parametrize("status", ["draft", "approved"])
def test_f3_malformed_pattern_is_rejected(status):
    r = synthetic("df-04#c2", pattern="(unclosed") if status == "approved" else {**RAW["df-04#c2"], "pattern": "(unclosed"}
    reg = registry(r)
    assert not reg.valid and reg.rules == ()


# ── F4: newline sentence splitting (known limitation; documents current behaviour) ───────────

@pytest.mark.parametrize("text", [
    "The output has 15\nsamples.",                                      # single newline
    "The output has 15\r\nsamples.",                                    # CRLF
    "The convolution of the two sequences gives an output that has 15\nsamples in total.",  # Markdown wrap
])
def test_f4_known_limitation_claim_split_by_newline_is_not_detected(text):
    """KNOWN LIMITATION: a single newline always ends a sentence. Not detecting this claim is a
    detection limitation, not evidence that the claim is absent."""
    rep = M.match_response(BY_ID["df-04"], text, registry(synthetic("df-04#c2")))
    assert rep.status == "no_affirmative_match" and rep.confirmed == ()
    assert M.match_response(BY_ID["df-04"], text.replace("\r\n", " ").replace("\n", " "),
                            registry(synthetic("df-04#c2"))).status == "affirmative_match"


# ── F5: stray quotation marks (known limitation) ────────────────────────────────────────────

def test_f5_known_limitation_stray_inch_quote_hides_a_claim():
    """KNOWN LIMITATION: 12" pairs with the next quote and blanks the claim between them."""
    text = 'The screen is 12" wide; the output has 15 samples, as the "result" panel shows.'
    prepared = M.prepare_text(text)
    assert "15 samples" not in prepared
    assert M.match_response(BY_ID["df-04"], text, registry(synthetic("df-04#c2"))).status == "no_affirmative_match"


def test_f5_control_same_claim_without_the_stray_quote_is_detected():
    text = "The screen is 12 inches wide; the output has 15 samples, as the result panel shows."
    assert M.match_response(BY_ID["df-04"], text, registry(synthetic("df-04#c2"))).status == "affirmative_match"


# ── F6: next-sentence corrections and attributions (known false-positive risks) ─────────────

@pytest.mark.parametrize("cid,text", [
    ("df-04#c1", "The output has 8 samples. That is wrong: it has 7."),
    ("df-07#c1", "Many students answer that the SQNR is 16 dB. This is incorrect."),
    ("df-03#c1", "The period is 50 ms. Actually, no: it is 20 ms."),
])
def test_f6_known_false_positive_risk_next_sentence_correction(cid, text):
    """KNOWN FALSE-POSITIVE RISK: a correction or attribution in the next sentence is not seen, so
    an explained misconception is flagged. Must be considered before approving the affected rule."""
    rep = M.match_response(BY_ID[cid.split("#")[0]], text, REAL, include_drafts=True)
    assert [d.claim_id for d in rep.shadow] == [cid] and rep.confirmed == ()


# ── F7: exclusion declarations are validated on every record ────────────────────────────────

@pytest.mark.parametrize("status", ["rejected", "retired"])
@pytest.mark.parametrize("bad", [r"\bnot\b", r"(?#x;nextclause)y", r"(?#Bad)y"])
def test_f7_malformed_exclusion_on_rejected_or_retired_rule_invalidates_registry(status, bad):
    r = synthetic("df-04#c2", 8, status=status, rationale="kept for history")
    r["exclusions"] = r["exclusions"] + [bad]
    reg = registry(synthetic("df-04#c2"), r)
    assert not reg.valid and reg.rules == () and any("df-04#c2/r8" in e for e in reg.errors)


def test_f7_duplicate_exclusion_names_on_retired_rule_invalidate_registry():
    r = synthetic("df-04#c2", 8, status="retired", rationale="old")
    r["exclusions"] = r["exclusions"] + [r"(?#negation)\bnever\b"]
    assert not registry(r).valid


# ── F8: exclusions that match the empty string ──────────────────────────────────────────────

@pytest.mark.parametrize("ex", ["(?#always)", r"(?#x)a*", r"(?#anything)x*", r"(?#blank)\b|^"])
def test_f8_always_firing_exclusion_is_rejected_explicitly(ex):
    r = copy.deepcopy(RAW["df-04#c2"])
    r["exclusions"] = r["exclusions"] + [ex]
    reg = registry(r)
    assert not reg.valid and any("matches the empty string" in e and "silently disable" in e for e in reg.errors)


def test_f8_empty_matching_exclusion_without_positive_examples_is_rejected():
    r = copy.deepcopy(RAW["df-04#c2"])
    r.update(exclusions=[r"(?#missing-year)^(?!.*\b2018\b)"], positive_examples=[])
    reg = registry(r)
    assert not reg.valid and any("no positive examples" in e for e in reg.errors)


def test_f8_zero_width_requirement_that_never_fires_on_positives_is_accepted():
    assert REAL.valid  # ic-05#c1/r1 keeps its (?#missing-2018) requirement
    ex = next(e for e in RAW["ic-05#c1"]["exclusions"] if e.startswith("(?#missing-2018)"))
    assert re.compile(ex).search("") is not None
    assert M._exclusion_problems(RAW["ic-05#c1"]) == []


# ── F9: direct tests for individual safeguards ──────────────────────────────────────────────

def _eligible_registry():
    reg = registry(synthetic("df-04#c2"))
    assert M._production_eligible(reg.rules[0], reg, BY_ID["df-04"])
    return reg


def test_f9_production_eligible_requires_a_valid_registry():
    reg = _eligible_registry()
    assert not M._production_eligible(reg.rules[0], dataclasses.replace(reg, valid=False), BY_ID["df-04"])


def test_f9_production_eligible_requires_no_ineligibility_reasons():
    reg = _eligible_registry()
    rule = dataclasses.replace(reg.rules[0], ineligibility=("positive example 1 is not an affirmative match",))
    assert rule.eligible and not M._production_eligible(rule, reg, BY_ID["df-04"])


def test_f9_production_eligible_requires_the_approved_status_and_listing():
    reg = _eligible_registry()
    assert not M._production_eligible(dataclasses.replace(reg.rules[0], status="draft"), reg, BY_ID["df-04"])
    assert not M._production_eligible(reg.rules[0], dataclasses.replace(reg, approved_rule_ids=frozenset()),
                                      BY_ID["df-04"])


def test_f9_rules_for_item_filters_by_item():
    reg = registry(synthetic("df-04#c2"), synthetic("df-03#c1"))
    assert [r.rule_id for r in M._rules_for_item(reg, BY_ID["df-04"])] == ["df-04#c2/r9"]
    assert [r.rule_id for r in M._rules_for_item(reg, BY_ID["df-03"])] == ["df-03#c1/r9"]
    assert M._rules_for_item(reg, BY_ID["mh-01"]) == []


def test_f9_repeated_matches_of_one_rule_list_its_id_once():
    reg = registry(synthetic("df-04#c2"))
    rep = M.match_response(BY_ID["df-04"], "The output has 15 samples. The output has 15 samples. "
                                           "Yes, the output has 15 samples.", reg)
    assert rep.confirmed[0].rule_ids == ("df-04#c2/r9",) and len(rep.confirmed[0].occurrences) == 3
    assert rep.to_dict()["confirmed"][0]["rule_ids"] == ["df-04#c2/r9"]


def test_f9_match_text_spanning_a_quote_blanked_span_shows_the_original_text():
    text = 'The output length "as computed" is 15.'
    rep = M.match_response(BY_ID["df-04"], text, registry(synthetic("df-04#c2")))
    occ = rep.confirmed[0].occurrences[0]
    assert occ.match.text == 'length "as computed" is 15'
    assert rep.normalized_text[occ.match.start:occ.match.end] == occ.match.text


# ── F8 / Option A conditions ────────────────────────────────────────────────────────────────

MISSING_2018 = next(e for e in RAW["ic-05#c1"]["exclusions"] if e.startswith("(?#missing-2018)"))


def test_f8_condition1_exception_lost_if_it_fires_on_any_positive_example():
    """Rejected when it fires on ANY positive example, even if other positives are fine."""
    r = copy.deepcopy(RAW["ic-05#c1"])
    r["exclusions"] = r["exclusions"] + [r"(?#missing-croatia)^(?!.*\bcroatia\b)"]
    r["positive_examples"] = ["France won the 2018 World Cup, beating Croatia in the final.",
                              "France won the 2018 FIFA World Cup."]
    assert any(re.search("croatia", p, re.I) for p in r["positive_examples"])  # one positive is fine
    reg = registry(r)
    assert not reg.valid
    assert any("'missing-croatia' matches the empty string and fires on a positive example" in e for e in reg.errors)


def test_f8_condition2_no_positive_examples_means_no_exception():
    r = copy.deepcopy(RAW["ic-05#c1"])
    r["positive_examples"] = []
    reg = registry(r)
    assert not reg.valid
    assert any("'missing-2018' matches the empty string and the rule has no positive examples" in e
               for e in reg.errors)


def test_f8_condition3_exception_does_not_bypass_approval_requirements():
    # Phase 1 approval: reviewer must differ from author.
    same_reviewer = synthetic("ic-05#c1", reviewer="claude")
    reg = registry(same_reviewer)
    assert not reg.valid and any("reviewer other than its author" in e for e in reg.errors)
    # Phase 1 approval: a review date is required.
    assert not registry(synthetic("ic-05#c1", reviewed_on=None)).valid


def test_f8_condition3_exception_does_not_bypass_example_validation():
    negs = list(RAW["ic-05#c1"]["negative_examples"]) + ["France won the 2018 World Cup."]
    rule = registry(synthetic("ic-05#c1", negative_examples=negs)).rules[0]
    assert not rule.eligible and any("negative example" in r for r in rule.ineligibility)
    rep = M.match_response(BY_ID["ic-05"], "France won the 2018 World Cup.", registry(synthetic("ic-05#c1",
                           negative_examples=negs)))
    assert rep.status == "not_evaluated" and rep.confirmed == ()


def test_f8_condition3_exception_does_not_bypass_name_validation():
    r = copy.deepcopy(RAW["ic-05#c1"])
    r["exclusions"] = r["exclusions"] + [r"(?#Missing-Year)^(?!.*\b2018\b)"]
    assert not registry(r).valid


def test_f8_condition3_genuine_synthetic_approval_with_the_exception_still_confirms():
    reg = registry(synthetic("ic-05#c1"))
    assert reg.valid and reg.rules[0].eligible
    rep = M.match_response(BY_ID["ic-05"], "This isn't in your course, but France won the 2018 World Cup.", reg)
    assert rep.status == "affirmative_match" and rep.confirmed[0].reason == "unsupported"


@pytest.mark.parametrize("clause,fires", [
    ("France won the World Cup", True),
    ("France won the World Cup in 1998", True),
    ("France won the World Cup in 20180", True),
    ("", True),
    ("France won the 2018 World Cup", False),
    ("In 2018, France won the World Cup", False),
    ("the 2018 final", False),
])
def test_f8_condition5_missing_2018_on_clauses(clause, fires):
    assert (re.compile(MISSING_2018, re.I).search(clause) is not None) is fires


@pytest.mark.parametrize("text,outcome,names", [
    ("France won the World Cup.", "excluded", ("missing-2018",)),
    ("In 2018 there was no class; France won the World Cup.", "excluded", ("missing-2018",)),
    ("There was no class in 1998; France won the 2018 World Cup.", "affirmative", ()),
    ("France won the 2018 World Cup.", "affirmative", ()),
    ("France won the 1998 and 2018 World Cups.", "affirmative", ()),
])
def test_f8_condition5_missing_2018_is_scoped_to_the_claim_clause(text, outcome, names):
    [a] = M.assess_rule(RAW["ic-05#c1"], text)
    assert (a.outcome, a.exclusion_names) == (outcome, names)


# ══ Final security-review fixes (Q4, Q5, Q6, E8) ═════════════════════════════════════════════

# ── Q4: the evaluated item must equal the compiled dataset item ─────────────────────────────

@pytest.mark.parametrize("field,value", [
    ("query", "What is the length of the output?"),
    ("reference_answer", "The output has 7 samples."),
])
def test_q4_modified_question_or_reference_is_rejected(field, value):
    reg = registry(synthetic("df-04#c2"))
    changed = dataclasses.replace(BY_ID["df-04"], **{field: value})
    rep = M.match_response(changed, X15, reg)
    assert rep.status == "not_evaluated" and rep.assessments == () and rep.confirmed == ()
    assert "differs from the compiled dataset snapshot" in rep.status_reason


def test_q4_genuine_item_succeeds():
    assert M.match_response(BY_ID["df-04"], X15, registry(synthetic("df-04#c2"))).status == "affirmative_match"


def test_q4_modified_forbidden_claims_are_rejected():
    item = BY_ID["df-04"]
    changed = dataclasses.replace(item, forbidden_claims=list(reversed(item.forbidden_claims)))
    rep = M.match_response(changed, X15, registry(synthetic("df-04#c2")))
    assert rep.status == "not_evaluated" and "differs from the compiled dataset snapshot" in rep.status_reason


def test_q4_modified_item_id_is_rejected():
    rep = M.match_response(dataclasses.replace(BY_ID["df-04"], id="zz-99"), X15, registry(synthetic("df-04#c2")))
    assert rep.status == "not_evaluated" and "not in the compiled dataset snapshot" in rep.status_reason


def test_q4_item_missing_from_the_compiled_dataset_is_rejected():
    subset = [i for i in ITEMS if i.id != "df-03"]
    reg = M.compile_registry_text(json.dumps(synthetic("df-04#c2")) + "\n", subset, source="t", today=TODAY)
    rep = M.match_response(BY_ID["df-03"], "The period is 50 ms.", reg)
    assert rep.status == "not_evaluated" and "not in the compiled dataset snapshot" in rep.status_reason


def test_q4_a_different_item_reusing_an_id_cannot_bypass_equality():
    impostor = dataclasses.replace(BY_ID["df-03"], id="df-04")
    rep = M.match_response(impostor, X15, registry(synthetic("df-04#c2")))
    assert rep.status == "not_evaluated" and "differs from the compiled dataset snapshot" in rep.status_reason


def test_q4_altered_reference_answer_cannot_sidestep_the_example_checks():
    """Review reproduction: a rule that matches df-04's real reference answer is ineligible on the real
    dataset, eligible when compiled against an altered copy, and must not confirm on the real item."""
    rule = synthetic("df-04#c2", pattern=r"\bL\s*\+\s*M\b", anchors=[r"output"], exclusions=[r"(?#negation)\bnot\b"],
                     positive_examples=["The output length is L + M.", "Output: L + M."],
                     negative_examples=["It is not L + M output.", "none", "nothing"])
    text = json.dumps(rule) + "\n"
    assert not M.compile_registry_text(text, ITEMS, source="t", today=TODAY).rules[0].eligible
    altered = copy.deepcopy(list(ITEMS))
    next(i for i in altered if i.id == "df-04").reference_answer = "The output has 7 samples."
    alt_reg = M.compile_registry_text(text, altered, source="t", today=TODAY)
    assert alt_reg.rules[0].eligible
    rep = M.match_response(BY_ID["df-04"], "The output length is L + M.", alt_reg)
    assert rep.status == "not_evaluated" and rep.confirmed == ()


# ── Q5: future compilation dates ────────────────────────────────────────────────────────────

def test_q5_current_date_compilation_behaves_normally():
    before = current_date()
    reg = M.compile_registry_text(json.dumps(synthetic("df-04#c2")) + "\n", ITEMS, source="t")
    assert reg.valid and before <= reg.today <= current_date()   # tolerant of a midnight rollover
    assert M.match_response(BY_ID["df-04"], X15, reg).status == "affirmative_match"


def test_q5_future_compilation_date_is_rejected():
    future = current_date() + dt.timedelta(days=30)
    reg = M.compile_registry_text(json.dumps(synthetic("df-04#c2")) + "\n", ITEMS, source="t", today=future)
    assert not reg.valid and reg.rules == () and any("later than the current date" in e for e in reg.errors)
    rep = M.match_response(BY_ID["df-04"], X15, reg)
    assert rep.status == "not_evaluated" and rep.assessments == ()


def test_q5_future_review_date_stays_invalid_even_with_a_later_supplied_date():
    reviewed = (current_date() + dt.timedelta(days=10)).isoformat()
    text = json.dumps(synthetic("df-04#c2", reviewed_on=reviewed)) + "\n"
    assert not M.compile_registry_text(text, ITEMS, source="t").valid
    later = M.compile_registry_text(text, ITEMS, source="t", today=current_date() + dt.timedelta(days=40))
    assert not later.valid
    assert M.match_response(BY_ID["df-04"], X15, later).status == "not_evaluated"


def test_q5_future_date_on_a_genuine_object_is_never_authoritative():
    reg = registry(synthetic("df-04#c2"))
    moved = dataclasses.replace(reg, today=current_date() + dt.timedelta(days=2))  # +2: rollover-safe
    assert M._authoritative(moved) is None
    assert "does not match its source text or dataset snapshot" in \
        M.match_response(BY_ID["df-04"], X15, moved).status_reason
    assert M.match_response(BY_ID["df-04"], X15, moved).status == "not_evaluated"


def test_q5_historical_dates_still_work():
    assert REVIEWED_ON < TODAY < REAL_TODAY <= current_date()
    past = M.compile_registry_text(json.dumps(synthetic("df-04#c2")) + "\n", ITEMS, source="t",
                                   today=REVIEWED_ON + dt.timedelta(days=1))
    assert past.valid and M.match_response(BY_ID["df-04"], X15, past).status == "affirmative_match"


# ── Q6: authoritative cache identity ────────────────────────────────────────────────────────

def test_q6_identical_inputs_reuse_the_authoritative_compilation():
    reg = registry(synthetic("df-04#c2"))
    again = registry(synthetic("df-04#c2"))     # a new compile refreshes the same single cache entry
    assert M._cache_key(reg) == M._cache_key(again)
    canonical = M._AUTHORITATIVE[M._cache_key(reg)]
    assert M._authoritative(reg) is canonical and M._authoritative(again) is canonical  # one cached entry
    assert reg == canonical and reg is not canonical and again is not reg              # callers get copies


def test_q6_lf_and_crlf_do_not_overwrite_each_other():
    lf, crlf = registry(synthetic("df-04#c2")), registry(synthetic("df-04#c2"), crlf=True)
    assert lf.digest == crlf.digest and lf.source_text != crlf.source_text
    assert M._cache_key(lf) != M._cache_key(crlf)
    assert M._AUTHORITATIVE[M._cache_key(lf)] is not M._AUTHORITATIVE[M._cache_key(crlf)]
    for reg in (lf, crlf, lf):
        assert M._authoritative(reg) == reg and M._authoritative(reg).source_text == reg.source_text
        assert M.match_response(BY_ID["df-04"], X15, reg).status == "affirmative_match"


def test_q6_source_labels_do_not_overwrite_each_other():
    text = json.dumps(synthetic("df-04#c2")) + "\n"
    one = M.compile_registry_text(text, ITEMS, source="one", today=TODAY)
    two = M.compile_registry_text(text, ITEMS, source="two", today=TODAY)
    assert M._cache_key(one) != M._cache_key(two)
    for reg in (one, two, one):
        assert M.match_response(BY_ID["df-04"], X15, reg).status == "affirmative_match"


def test_q6_different_datasets_are_isolated():
    text = json.dumps(synthetic("df-04#c2")) + "\n"
    altered = copy.deepcopy(list(ITEMS))
    alt_item = next(i for i in altered if i.id == "df-04")
    alt_item.query = alt_item.query + " (variant)"
    real_reg = M.compile_registry_text(text, ITEMS, source="t", today=TODAY)
    alt_reg = M.compile_registry_text(text, altered, source="t", today=TODAY)
    assert M._cache_key(real_reg) != M._cache_key(alt_reg)
    assert M.match_response(BY_ID["df-04"], X15, real_reg).status == "affirmative_match"
    assert M.match_response(alt_item, X15, alt_reg).status == "affirmative_match"
    assert M.match_response(BY_ID["df-04"], X15, alt_reg).status == "not_evaluated"


def test_q6_different_dates_are_isolated():
    text = json.dumps(synthetic("df-04#c2")) + "\n"
    a = M.compile_registry_text(text, ITEMS, source="t", today=TODAY)
    b = M.compile_registry_text(text, ITEMS, source="t", today=TODAY - dt.timedelta(days=1))
    assert M._cache_key(a) != M._cache_key(b)
    for reg in (a, b, a):
        assert M.match_response(BY_ID["df-04"], X15, reg).status == "affirmative_match"


def test_q6_forged_objects_still_fail_with_any_label_or_line_ending():
    for reg in (registry(synthetic("df-04#c2", positive_examples=[X15, "Nothing relevant here."])),
                registry(synthetic("df-04#c2", positive_examples=[X15, "Nothing relevant here."]), crlf=True)):
        forged = dataclasses.replace(reg, rules=(dataclasses.replace(reg.rules[0], eligible=True, ineligibility=()),))
        assert M.match_response(BY_ID["df-04"], X15, forged).status == "not_evaluated"
        relabelled = dataclasses.replace(forged, source="other-label")
        assert M.match_response(BY_ID["df-04"], X15, relabelled).status == "not_evaluated"


# ── E8: the empty-match exception uses clause-level matching ────────────────────────────────

def test_e8_missing_2018_positive_examples_remain_valid():
    assert M._exclusion_problems(RAW["ic-05#c1"]) == [] and REAL.valid


def test_e8_positive_with_2018_only_in_another_clause_is_rejected():
    r = copy.deepcopy(RAW["ic-05#c1"])
    r["positive_examples"] = r["positive_examples"] + ["In 2018 there was no class; France won the World Cup."]
    problems = M._exclusion_problems(r)
    assert problems == ["exclusion 'missing-2018' matches the empty string and fires on a positive example's "
                        "claim match; it would silently disable the rule"]
    assert not registry(r).valid


def test_e8_positive_affected_by_a_next_clause_exclusion_is_rejected():
    r = copy.deepcopy(RAW["df-04#c2"])
    r["exclusions"] = r["exclusions"] + [r"(?#needs-ok;next-clause)^(?!.*\bok\b)"]
    r["positive_examples"] = ["The output has 15 samples, which is ok.", "The output has 15 samples, which is clear."]
    assert any("'needs-ok'" in p and "fires on a positive example's claim match" in p for p in M._exclusion_problems(r))
    r["positive_examples"] = ["The output has 15 samples, which is ok.", "Convolution gives a 15-sample output."]
    assert M._exclusion_problems(r) == []  # no next clause, or one containing "ok": it never fires


def test_e8_pattern_less_rule_cannot_use_the_exception():
    r = copy.deepcopy(RAW["ic-05#c1"])
    r.update(pattern=None, rule_id="ic-05#c1/r2")
    assert any("no usable pattern" in p for p in M._exclusion_problems(r))


def test_e8_positives_without_any_claim_match_cannot_use_the_exception():
    r = copy.deepcopy(RAW["ic-05#c1"])
    r["positive_examples"] = ["Nothing about football here.", "Another unrelated sentence."]
    assert any("no positive example produces a claim match" in p for p in M._exclusion_problems(r))


def test_e8_exception_is_not_shared_between_rules():
    good = RAW["ic-05#c1"]
    bad = copy.deepcopy(RAW["ic-05#c1"])
    bad.update(rule_id="ic-05#c1/r2", positive_examples=["France won the World Cup.", "France won the 2018 World Cup."])
    assert M._exclusion_problems(good) == [] and M._exclusion_problems(bad) != []
    assert not registry(good, bad).valid and not registry(bad, good).valid
    assert M._exclusion_problems(good) == [] and registry(good).valid


def test_f1_digest_mismatch_is_rejected_before_any_compilation(monkeypatch):
    """The explicit digest check must refuse tampered source text without compiling it at all."""
    approved_text = json.dumps(synthetic("df-04#c2")) + "\n"
    tampered = dataclasses.replace(REAL, source_text=approved_text)

    def no_compile(*args, **kwargs):
        raise AssertionError("a tampered registry must not be compiled")

    monkeypatch.setattr(M, "_compile", no_compile)
    assert M._authoritative(tampered) is None
    assert M.match_response(BY_ID["df-04"], X15, tampered).status == "not_evaluated"


# ══ Low-finding fixes: canonical cache isolation (L2) and clock-independent fixtures (L1) ═══════

SNAPSHOT_REASON = "does not match its source text or dataset snapshot"
CANONICAL_REASON = "differs from the authoritative compilation"


def _df04_index(reg: M.CompiledRegistry) -> int:
    return next(i for i, it in enumerate(reg.items) if it.id == "df-04")


def _assert_snapshot_edit_rejected(reg: M.CompiledRegistry) -> None:
    """The edit leaves the eligibility fingerprint intact, so only the full snapshot check can catch it."""
    assert M._items_fingerprint(reg.items) == reg.items_fingerprint
    assert M._snapshot_digest(reg.items) != reg.snapshot_digest
    assert M._authoritative(reg) is None
    rep = M.match_response(BY_ID["df-04"], X15, reg)
    assert rep.status == "not_evaluated" and rep.confirmed == () and rep.assessments == ()
    assert SNAPSHOT_REASON in rep.status_reason


def test_l2_genuine_returned_compilation_is_authoritative_before_any_edit():
    reg = registry(synthetic("df-04#c2"))
    assert M._snapshot_digest(reg.items) == reg.snapshot_digest
    assert M.match_response(BY_ID["df-04"], X15, reg).status == "affirmative_match"


def test_l2_mutating_notes_on_the_returned_snapshot_is_not_authoritative():
    reg = registry(synthetic("df-04#c2"))
    reg.items[_df04_index(reg)].notes = "tampered"
    _assert_snapshot_edit_rejected(reg)


def test_l2_mutating_category_on_the_returned_snapshot_is_not_authoritative():
    reg = registry(synthetic("df-04#c2"))
    reg.items[_df04_index(reg)].category = "multi_hop"
    _assert_snapshot_edit_rejected(reg)


@pytest.mark.parametrize("edit", [
    lambda it: it.tags.append("tampered"),                       # nested list
    lambda it: it.scope.update({"tampered": True}),               # nested dict
    lambda it: it.raw.update({"notes": "tampered"}),              # nested dict excluded from Item ==
    lambda it: it.evidence.append(it.evidence[0]),                # list of frozen records
], ids=["tags", "scope", "raw", "evidence"])
def test_l2_mutating_a_nested_mutable_field_is_not_authoritative(edit):
    reg = registry(synthetic("df-04#c2"))
    edit(reg.items[_df04_index(reg)])
    _assert_snapshot_edit_rejected(reg)


def test_l2_raw_is_excluded_from_item_equality_so_only_the_digest_catches_it():
    reg = registry(synthetic("df-04#c2"))
    before = copy.deepcopy(reg.items)
    reg.items[_df04_index(reg)].raw["notes"] = "tampered"
    assert reg.items == before                       # == alone would not notice
    assert M._authoritative(reg) is None


def test_l2_edited_snapshot_stays_rejected_after_its_cache_entry_is_evicted():
    reg = registry(synthetic("df-04#c2"))
    reg.items[_df04_index(reg)].notes = "tampered"
    for i in range(M._AUTHORITATIVE_MAX + 8):
        registry(synthetic("df-04#c2", rationale=f"eviction filler {i}"))
    assert M._cache_key(reg) not in M._AUTHORITATIVE
    _assert_snapshot_edit_rejected(reg)
    assert M._cache_key(reg) not in M._AUTHORITATIVE   # nothing was recompiled from the edited snapshot


def test_l2_other_returned_fields_are_compared_with_the_canonical_compilation():
    reg = registry(synthetic("df-04#c2"), _no_pattern_draft())
    assert reg.not_runnable
    object.__setattr__(reg, "not_runnable", ())
    assert M._authoritative(reg) is not None         # source, dataset and date checks still hold
    rep = M.match_response(BY_ID["df-04"], X15, reg)
    assert rep.status == "not_evaluated" and CANONICAL_REASON in rep.status_reason


def test_l2_returned_objects_share_no_mutable_state_with_the_cache_or_each_other():
    a = registry(synthetic("df-04#c2"))
    b = registry(synthetic("df-04#c2"))
    canonical = M._AUTHORITATIVE[M._cache_key(a)]
    assert a == b == canonical
    for x, y in ((a, canonical), (b, canonical), (a, b)):
        assert x is not y and x.items is not y.items
        assert all(p is not q for p, q in zip(x.items, y.items))
        assert all(p.tags is not q.tags and p.raw is not q.raw for p, q in zip(x.items, y.items))


def test_l2_mutating_a_returned_compilation_cannot_alter_the_canonical_one():
    reg = registry(synthetic("df-04#c2"))
    canonical = M._AUTHORITATIVE[M._cache_key(reg)]
    pristine = copy.deepcopy(canonical)
    i = _df04_index(reg)
    reg.items[i].notes = "tampered"
    reg.items[i].tags.append("tampered")
    reg.items[i].raw["x"] = 1
    object.__setattr__(reg, "rules", ())
    object.__setattr__(reg, "errors", ("tampered",))
    assert M._AUTHORITATIVE[M._cache_key(pristine)] is canonical
    assert canonical == pristine
    assert M._snapshot_digest(canonical.items) == canonical.snapshot_digest == pristine.snapshot_digest
    assert canonical.items[i].raw == pristine.items[i].raw


def test_l2_fresh_compilation_stays_authoritative_after_another_copy_is_mutated():
    victim = registry(synthetic("df-04#c2"))
    victim.items[_df04_index(victim)].notes = "tampered"
    fresh = registry(synthetic("df-04#c2"))
    assert fresh.items[_df04_index(fresh)].notes == BY_ID["df-04"].notes
    assert M._authoritative(fresh) == fresh
    assert M.match_response(BY_ID["df-04"], X15, fresh).status == "affirmative_match"
    assert M.match_response(BY_ID["df-04"], X15, victim).status == "not_evaluated"


def test_l2_mutating_the_internal_comparison_result_cannot_reach_later_callers():
    reg = registry(synthetic("df-04#c2"))
    # match_response uses the canonical object internally; its report carries only immutable values
    rep = M.match_response(BY_ID["df-04"], X15, reg)
    for d in rep.confirmed:
        assert isinstance(d.rule_ids, tuple) and isinstance(d.occurrences, tuple)
    assert M.match_response(BY_ID["df-04"], X15, registry(synthetic("df-04#c2"))).status == "affirmative_match"


def test_l2_datasets_differing_only_outside_the_fingerprint_do_not_share_a_cache_entry():
    text = json.dumps(synthetic("df-04#c2")) + "\n"
    noted = copy.deepcopy(list(ITEMS))
    next(i for i in noted if i.id == "df-04").notes = "a different note"
    real_reg = M.compile_registry_text(text, ITEMS, source="t", today=TODAY)
    noted_reg = M.compile_registry_text(text, noted, source="t", today=TODAY)
    assert real_reg.items_fingerprint == noted_reg.items_fingerprint
    assert M._cache_key(real_reg) != M._cache_key(noted_reg)
    assert M.match_response(BY_ID["df-04"], X15, real_reg).status == "affirmative_match"   # not overwritten


def test_l2_q4_still_rejects_an_evaluated_item_edited_outside_the_fingerprint():
    reg = registry(synthetic("df-04#c2"))
    changed = dataclasses.replace(BY_ID["df-04"], notes="tampered")
    rep = M.match_response(changed, X15, reg)
    assert rep.status == "not_evaluated" and "differs from the compiled dataset snapshot" in rep.status_reason
    assert M.match_response(BY_ID["df-04"], X15, reg).status == "affirmative_match"


def test_l2_snapshot_digest_is_stable_and_order_independent_for_dicts():
    a, b = copy.deepcopy(list(ITEMS)), copy.deepcopy(list(ITEMS))
    item = next(i for i in b if i.id == "df-04")
    item.raw = dict(reversed(list(item.raw.items())))
    assert M._snapshot_digest(a) == M._snapshot_digest(ITEMS) == M._snapshot_digest(b)


# ── L1: fixture dates follow the clock ───────────────────────────────────────────────────────

def test_l1_fixture_dates_are_relative_to_the_current_date():
    assert REVIEWED_ON < TODAY < REAL_TODAY <= current_date()
    assert (REAL_TODAY - TODAY).days == 30 and (TODAY - REVIEWED_ON).days == 4
    assert REAL.valid and REAL.today == TODAY


def test_l1_controlled_clock_is_in_effect_when_requested():
    """Runs meaningfully only inside the controlled-clock subprocess (asserted there via -v output)."""
    if not os.environ.get(CLOCK_ENV):
        pytest.skip("controlled clock not requested")
    fixed = dt.date.fromisoformat(os.environ[CLOCK_ENV])
    assert current_date() == fixed == REAL_TODAY
    assert M._dt.date.today() == fixed
    reg = M.compile_registry_text(json.dumps(synthetic("df-04#c2")) + "\n", ITEMS, source="t",
                                  today=fixed + dt.timedelta(days=1))
    assert not reg.valid and any("later than the current date" in e for e in reg.errors)   # Q5 holds


@pytest.mark.skipif(bool(os.environ.get(CLOCK_ENV)), reason="already inside a controlled-clock run")
@pytest.mark.parametrize("fixed", ["2026-11-23", "2021-03-01", "2031-07-15"])
def test_suite_passes_under_a_controlled_earlier_or_later_clock(fixed):
    """Rerun this whole file with the matcher's clock fixed to another date (before 2026-11-24,
    long before, and later), proving no test depends on the real calendar date."""
    env = {**os.environ, CLOCK_ENV: fixed, "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run([sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider",
                           str(Path(__file__).resolve())],
                          cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out[-4000:]
    assert re.search(r"test_l1_controlled_clock_is_in_effect_when_requested PASSED", out), out[-2000:]
    summary = out.strip().splitlines()[-1]
    assert re.search(r"\b\d{3,} passed\b", summary) and "failed" not in summary and "error" not in summary, summary
