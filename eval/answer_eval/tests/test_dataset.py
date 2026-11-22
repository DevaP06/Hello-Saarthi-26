"""Sprint 1 tests (after the correction pass of 2026-11-23): dataset schema, counts, references,
claim reasons, behavioural rubrics, key-point patterns and evidence.

Run from the repo root with explicit paths (a bare `pytest` would also collect other checkouts
inside the folder, such as `_upstream/`):

    python -m pytest eval/answer_eval/tests -q -p no:cacheprovider

Needs only pytest and pypdf (both in the project's requirements). No network, no model, no backend.
"""
from __future__ import annotations

import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.answer_eval import schema  # noqa: E402
from eval.answer_eval.schema import (  # noqa: E402
    CLAIM_REASONS,
    EXPECTED_DISTRIBUTION,
    MATERIALS,
    check_item,
    load_dataset,
    parse_item,
    validate_dataset_items,
)

DATASET = ROOT / "eval" / "answer_eval" / "dataset.jsonl"
ARCHIVE = ROOT / "eval" / "answer_eval" / "archive" / "removed_hi_mr_2026-11-23.jsonl"


# ── Fixtures and helpers ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_items() -> list[dict]:
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def items():
    parsed, errors = load_dataset(DATASET)
    assert errors == [], errors
    return parsed


@pytest.fixture(scope="module")
def by_id(items):
    return {i.id: i for i in items}


@pytest.fixture(scope="module")
def corpus():
    pytest.importorskip("pypdf")
    from eval.answer_eval.validate_dataset import load_corpus

    return load_corpus()


def _raw(raw_items, iid):
    return copy.deepcopy(next(r for r in raw_items if r["id"] == iid))


def _errors_for(raw: dict) -> list[str]:
    item, errors = parse_item(raw)
    return errors if item is None else check_item(item)


def _pattern(by_id, iid, kid):
    return next(k for k in by_id[iid].key_points if k.id == kid).match


# ── Counts and status ───────────────────────────────────────────────────────

def test_dataset_parses_and_passes_every_schema_rule(items):
    assert validate_dataset_items(items) == []


def test_active_dataset_has_exactly_37_entries(items):
    assert len(items) == 37


def test_expected_behaviour_counts(items):
    assert Counter(i.expected_behavior for i in items) == {"answer": 27, "abstain": 6, "clarify": 4}


def test_category_distribution_is_recalculated_from_the_remaining_entries(items):
    assert dict(Counter(i.category for i in items)) == EXPECTED_DISTRIBUTION
    assert EXPECTED_DISTRIBUTION["multilingual"] == 1


def test_ids_are_unique(items):
    ids = [i.id for i in items]
    assert len(ids) == len(set(ids))


def test_behaviour_follows_the_category(items):
    for i in items:
        assert i.expected_behavior in schema.CATEGORY_BEHAVIOR[i.category], i.id


def test_every_active_item_is_a_draft_without_a_reviewer(items):
    assert all(i.label_status == "draft" and i.reviewer is None for i in items)


def test_six_to_eight_paraphrased_items_keep_their_audit_wording(items):
    para = [i for i in items if "paraphrased" in i.tags]
    assert 6 <= len(para) <= 8
    assert all(i.original_query and i.original_query != i.query for i in para)


# ── Languages ───────────────────────────────────────────────────────────────

def test_only_english_and_hinglish_are_in_the_active_dataset(items):
    assert Counter(i.language for i in items) == {"en": 36, "hinglish": 1}
    assert set(schema.LANGUAGES) == {"en", "hinglish"}


def test_expected_answer_language_matches_the_question_language(items):
    for i in items:
        assert i.expected_answer_language == i.language, i.id


def test_only_multilingual_items_use_another_language(items):
    for i in items:
        assert (i.language != "en") == (i.category == "multilingual"), i.id


@pytest.mark.parametrize("field, value, expected", [
    ("language", "hi", "not in the active set"),
    ("language", "mr", "not in the active set"),
    ("expected_answer_language", "en", "must match the question language"),
    ("expected_answer_language", "hi", "not in the active set"),
])
def test_other_languages_and_mismatched_answer_language_are_rejected(raw_items, field, value, expected):
    raw = _raw(raw_items, "ml-02")
    raw[field] = value
    assert any(expected in e for e in _errors_for(raw))


def test_removed_hindi_and_marathi_items_are_archived_not_active(items):
    active = {i.id for i in items}
    archived = [json.loads(line) for line in ARCHIVE.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [a["id"] for a in archived] == ["ml-01", "ml-03", "ml-04"]
    assert {a["language"] for a in archived} == {"hi", "mr"}
    assert active.isdisjoint(a["id"] for a in archived)
    # Kept verbatim in the Sprint 1 shape, so the current validator rejects them: they cannot slip back in.
    _, errors = load_dataset(ARCHIVE)
    assert errors


# ── ml-02 and its English twin ──────────────────────────────────────────────

def test_ml02_covers_both_parts_of_mh01(by_id):
    ml, mh = by_id["ml-02"], by_id["mh-01"]
    assert ml.paired_with == "mh-01" and ml.language == "hinglish"
    required = lambda it: sorted(k.match for k in it.key_points if k.required)  # noqa: E731
    assert required(ml) == required(mh) and len(required(ml)) == 2
    assert "theorem" in ml.query and "frequency" in ml.query
    for p in required(ml):
        assert re.search(p, ml.reference_answer, re.I)
    assert "18 kHz" in ml.reference_answer and "3 kHz" in ml.reference_answer


def test_a_pair_with_different_required_points_is_rejected(items):
    broken = copy.deepcopy(items)
    ml = next(i for i in broken if i.id == "ml-02")
    ml.key_points = [k for k in ml.key_points if k.id != "k1"]
    assert any("differ from its twin" in e for e in validate_dataset_items(broken))


def test_multilingual_answer_items_are_checkable_in_any_language(items):
    for i in items:
        if i.language != "en" and i.expected_behavior == "answer":
            assert all(k.match for k in i.key_points if k.required), i.id


# ── Claims: false vs unsupported ────────────────────────────────────────────

def test_every_claim_has_a_valid_reason(items):
    for i in items:
        for c in i.forbidden_claims:
            assert c.reason in CLAIM_REASONS and c.text.strip(), i.id


def test_abstention_claims_are_classified_by_meaning(by_id):
    expected = {
        "ic-01": {("The Fourier transform of a triangular pulse is a sinc-squared function", "unsupported")},
        "ic-02": {("The Hamming window is 0.54 - 0.46 cos(2 pi n / (N - 1))", "unsupported"),
                  ("The Hamming window is 0.5 - 0.5 cos(2 pi n / (N - 1))", "false")},
        "ic-03": {("The region of convergence is Re(s) > -2", "unsupported")},
        "ic-04": {("The cutoff frequency is pre-warped with tan() before the bilinear transform is applied", "unsupported")},
        "ic-05": {("France won the 2018 World Cup", "unsupported")},
        "ic-06": set(),
    }
    for iid, claims in expected.items():
        assert {(c.text, c.reason) for c in by_id[iid].forbidden_claims} == claims, iid


def test_answer_and_clarify_items_only_forbid_false_claims(items):
    for i in items:
        if i.expected_behavior != "abstain":
            assert all(c.reason == "false" for c in i.forbidden_claims), i.id


@pytest.mark.parametrize("claims, expected", [
    (["A plain string"], "exactly 'text' and 'reason'"),
    ([{"text": "x", "reason": "wrong"}], "'reason' must be one of"),
    ([{"text": " ", "reason": "false"}], "'text' must be a non-empty string"),
    ([{"text": "x"}], "exactly 'text' and 'reason'"),
])
def test_malformed_claims_are_rejected(raw_items, claims, expected):
    raw = _raw(raw_items, "df-01")
    raw["forbidden_claims"] = claims
    assert any(expected in e for e in _errors_for(raw))


# ── Behavioural rubrics are not claims ──────────────────────────────────────

def test_behavioural_rubrics_are_stored_in_must_not(by_id):
    for iid in ("am-01", "am-02", "am-03", "am-04", "ic-06"):
        item = by_id[iid]
        assert item.must_not, iid
        assert item.forbidden_claims == [], iid
    assert by_id["ic-06"].must_not == ["Give a recipe or cooking instructions"]


def test_no_forbidden_claim_reads_as_a_behaviour(items):
    for i in items:
        for c in i.forbidden_claims:
            assert not schema._BEHAVIOUR_OPENERS.match(c.text), (i.id, c.text)


def test_a_behaviour_written_as_a_claim_is_rejected(raw_items):
    raw = _raw(raw_items, "am-01")
    raw["forbidden_claims"] = [{"text": "States a single period without saying which signal it belongs to", "reason": "false"}]
    assert any("put it in must_not" in e for e in _errors_for(raw))


def test_must_not_entries_must_be_non_empty_strings(raw_items):
    raw = _raw(raw_items, "am-01")
    raw["must_not"] = [""]
    assert any("'must_not' must contain non-empty strings only" in e for e in _errors_for(raw))


# ── Individual corrections ──────────────────────────────────────────────────

def test_ic02_is_a_hamming_only_abstention(by_id):
    it = by_id["ic-02"]
    assert it.query == "What are the coefficients of the Hamming window?"
    assert "hann" not in it.query.lower()
    assert it.expected_behavior == "abstain" and it.absent_terms == ["hamming"]
    assert "not covered" in it.reference_answer
    assert any("Hann" in m for m in it.must_not)  # substituting the course's Hann window is a failure


def test_co04_asks_about_true_resolution_without_a_false_premise_tag(by_id):
    co, df = by_id["co-04"], by_id["df-06"]
    assert "false_premise" not in co.tags
    assert co.query == "Does zero-padding improve the true frequency resolution of a DFT?"
    assert co.expected_behavior == "answer"
    texts = " ".join(k.text for k in co.key_points)
    assert "finer-spaced bins" in texts and "true resolution does not improve" in texts
    assert all(k.required for k in co.key_points)
    # df-06 keeps the bin-spacing meaning and no longer carries an irrelevant zero-padding claim.
    assert [c.text for c in df.forbidden_claims] == ["The resolution is N / fs"]


def test_small_corrections(by_id):
    assert "linearly convolve" in by_id["df-04"].query
    assert {by_id[i].difficulty for i in ("df-03", "df-07", "co-05")} == {"recall"}
    assert "initially at rest" in by_id["mh-03"].query
    assert "sampled" in by_id["co-01"].query and "digitised" not in by_id["co-01"].query
    assert [k.required for k in by_id["co-01"].key_points] == [True, False]
    assert [k.id for k in by_id["cm-02"].key_points if k.required] == ["k1", "k3"]
    mh06 = by_id["mh-06"]
    assert all(k.required for k in mh06.key_points)
    assert "how many times fewer" in mh06.query and "Python" in mh06.query


# ── Key-point patterns accept and reject the review's examples ──────────────

REGEX_CASES = [
    ("df-01", "k1", "The Nyquist rate is 2B.", True),
    ("df-01", "k1", "You must sample above 2 f_max.", True),
    ("df-01", "k1", "It is 2f_{max}.", True),
    ("df-01", "k1", "It is twice the maximum frequency.", True),
    ("df-01", "k1", "It is double the maximum frequency.", True),
    ("df-01", "k1", "The Nyquist rate is fs/2.", False),
    ("df-01", "k1", "It equals the bandwidth B.", False),
    ("cm-04", "k2", "h[n] = h1[n] + h2[n]", True),
    ("cm-04", "k2", "h_1[n] + h_2[n]", True),
    ("cm-04", "k2", "h_{1}[n] + h_{2}[n]", True),
    ("cm-04", "k2", "h1(t) + h2(t)", True),
    ("cm-04", "k2", "h1 + h2", True),
    ("cm-04", "k2", "h1[n] * h2[n]", False),
    ("cm-04", "k2", "h1 * h2", False),
    ("mh-06", "k3", "about 204.8 times fewer", True),
    ("mh-06", "k3", "204.80 times", True),
    ("mh-06", "k3", "about 205 times fewer", True),
    ("mh-06", "k3", "2048 operations", False),
    ("mh-06", "k3", "1205 times", False),
    ("mh-06", "k3", "20.48 times", False),
    ("mh-06", "k3", "204 times", False),
    ("mh-02", "k1", "The Nyquist rate is 3000 samples per second.", True),
    ("mh-02", "k1", "The Nyquist rate is 3,000 samples/s.", True),
    ("mh-02", "k1", "The Nyquist rate is 13000 Hz.", False),
    ("mh-02", "k1", "The Nyquist rate is 13,000 Hz.", False),
    ("mh-02", "k1", "The Nyquist rate is 30000 Hz.", False),
    ("ml-02", "k1", "Nahi, 18 kHz chahiye.", True),
    ("ml-02", "k2", "Tone 3 kHz pe dikhega.", True),
    ("ml-02", "k2", "Tone 9 kHz pe dikhega.", False),
]


@pytest.mark.parametrize("iid, kid, text, accepted", REGEX_CASES)
def test_corrected_patterns_accept_and_reject_the_review_examples(by_id, iid, kid, text, accepted):
    pattern = _pattern(by_id, iid, kid)
    assert bool(re.search(pattern, text, re.I)) is accepted


def test_mh02_k3_has_no_pattern_and_relies_on_semantic_scoring(by_id):
    assert _pattern(by_id, "mh-02", "k3") is None


def test_regex_key_points_match_their_reference_answers(items):
    for i in items:
        for k in i.key_points:
            if k.match:
                assert re.search(k.match, i.reference_answer, re.I), (i.id, k.id)


# ── Sources and evidence ────────────────────────────────────────────────────

def test_materials_are_full_module_titles(items):
    for i in items:
        for e in i.evidence:
            assert e.material in MATERIALS, (i.id, e.material)


def test_expected_sources_are_derived_from_evidence(items):
    for i in items:
        if i.expected_behavior != "abstain":
            assert {(s["material"], s["page"]) for s in i.expected_sources} == {(e.material, e.page) for e in i.evidence}, i.id


def test_corpus_has_four_materials_of_four_pages(corpus):
    assert set(corpus) == set(MATERIALS)
    assert all(len(pages) == 4 for pages in corpus.values())


def test_every_quote_is_found_on_its_stated_page(items, corpus):
    from eval.answer_eval.validate_dataset import check_evidence

    errors, verified = check_evidence(items, corpus)
    assert errors == []
    assert verified == sum(len(i.evidence) for i in items) == 61


def test_absent_terms_do_not_occur_in_the_corpus(items, corpus):
    from eval.answer_eval.validate_dataset import squash

    text = squash(" ".join(t for pages in corpus.values() for t in pages)).lower()
    abstain = [i for i in items if i.expected_behavior == "abstain"]
    assert len(abstain) == 6
    for i in abstain:
        assert i.absent_terms
        for term in i.absent_terms:
            assert squash(term).lower() not in text, (i.id, term)


def test_whitespace_insensitive_match_handles_pdf_kerning(corpus):
    from eval.answer_eval.validate_dataset import squash

    page = corpus["EE210 Module 1 Signals and Sampling"][2]
    assert "T aking" in page  # pypdf splits the kerned "Ta"
    assert squash("Taking x[n] = x(n Ts)") in squash(page)


def test_validator_cli_passes(capsys):
    pytest.importorskip("pypdf")
    from eval.answer_eval.validate_dataset import main

    assert main([]) == 0
    out = capsys.readouterr().out
    assert "Items parsed: 37" in out and "Evidence quotes verified: 61/61" in out


# ── Invalid entries are rejected ────────────────────────────────────────────

@pytest.mark.parametrize("mutate, expected", [
    (lambda r: r.pop("query"), "missing field 'query'"),
    (lambda r: r.pop("must_not"), "missing field 'must_not'"),
    (lambda r: r.pop("expected_answer_language"), "missing field 'expected_answer_language'"),
    (lambda r: r.update(category="trivia"), "unknown category"),
    (lambda r: r.update(expected_behavior="abstain"), "cannot have expected_behavior"),
    (lambda r: r.update(difficulty="hard"), "unknown difficulty"),
    (lambda r: r.update(tags=["paraphrased"]), "original_query"),
    (lambda r: r.update(tags=["shiny"]), "unknown tag"),
    (lambda r: r.update(label_status="reviewed"), "needs a reviewer"),
    (lambda r: r.update(reviewer="someone"), "must not name a reviewer"),
    (lambda r: r.update(unexpected=1), "unknown field"),
    (lambda r: r["evidence"][0].update(material="Module 1"), "use the full module title"),
    (lambda r: r["evidence"][0].update(page=7), "out of range"),
    (lambda r: r["evidence"][0].update(quote="2B"), "shorter than"),
    (lambda r: r["key_points"][0].update(evidence_ids=["e9"]), "unknown evidence id"),
    (lambda r: r["key_points"][0].update(match="("), "not a valid regex"),
    (lambda r: r["key_points"][0].update(match="banana"), "does not match the item's own reference_answer"),
    (lambda r: r.update(key_points=[]), "at least one required key point"),
    (lambda r: r.update(expected_sources=[]), "expected_sources must list exactly"),
    (lambda r: r.update(absent_terms=["fourier"]), "must not have absent_terms"),
    (lambda r: r.update(scope={"material_title": "EE210 Module 3 Fourier Analysis DFT and Filters"}), "scope material_title differs"),
    (lambda r: r.update(language="hinglish", expected_answer_language="hinglish"), "only multilingual items"),
    (lambda r: r.update(id="DF1"), "id must look like"),
])
def test_invalid_answer_entries_are_rejected(raw_items, mutate, expected):
    raw = _raw(raw_items, "df-01")
    mutate(raw)
    errors = _errors_for(raw)
    assert any(expected in e for e in errors), errors


@pytest.mark.parametrize("iid, mutate, expected", [
    ("ic-01", lambda r: r.update(absent_terms=[]), "needs absent_terms"),
    ("ic-01", lambda r: r.update(evidence=[{"id": "e1", "material": "EE210 Module 1 Signals and Sampling", "page": 3,
                                            "quote": "The Nyquist rate is 2B; the Nyquist frequency is fs/2."}]),
     "has no evidence"),
    ("am-01", lambda r: r.update(acceptable_interpretations=["only one"]), "at least two acceptable_interpretations"),
    ("mh-01", lambda r: r.update(evidence=r["evidence"][:1], key_points=[{**r["key_points"][0], "evidence_ids": ["e1"]}]),
     "at least two evidence passages"),
    ("ml-02", lambda r: r.update(paired_with=None), "needs paired_with"),
    ("ml-02", lambda r: r.update(language="en", expected_answer_language="en"), "must not be in English"),
    ("ml-02", lambda r: r["key_points"][0].pop("match"), "need a 'match' regex"),
    ("mh-03", lambda r: r.update(tags=[]), "cross_page"),
    ("cm-02", lambda r: r.update(tags=[]), "cross_module"),
])
def test_invalid_entries_of_other_kinds_are_rejected(raw_items, iid, mutate, expected):
    raw = _raw(raw_items, iid)
    mutate(raw)
    errors = _errors_for(raw)
    assert any(expected in e for e in errors), errors


def test_dataset_level_rules_catch_duplicates_dangling_pairs_and_counts(items):
    dup = copy.deepcopy(items[0])
    errs = validate_dataset_items(items + [dup])
    assert any("is used 2 times" in e for e in errs)
    assert any("38 items; expected 37" in e for e in errs)
    assert any("category distribution" in e for e in errs)
    assert any("expected_behavior counts" in e for e in errs)

    dangling = copy.deepcopy(items)
    next(i for i in dangling if i.id == "ml-02").paired_with = "zz-99"
    assert any("paired_with 'zz-99' does not exist" in e for e in validate_dataset_items(dangling))

    wrong_twin = copy.deepcopy(items)
    next(i for i in wrong_twin if i.id == "ml-02").paired_with = "ic-05"  # answer item paired with an abstain item
    assert any("differs from its twin" in e for e in validate_dataset_items(wrong_twin))


def test_quotes_from_page_furniture_or_wrong_pages_are_rejected(items, corpus):
    from eval.answer_eval.schema import Evidence
    from eval.answer_eval.validate_dataset import check_evidence

    base = copy.deepcopy(next(i for i in items if i.id == "df-01"))
    cases = {
        "footer": Evidence("e1", "EE210 Module 1 Signals and Sampling", 3, "EE210 Module 1 Signals and Sampling - page 3"),
        "tag": Evidence("e1", "EE210 Module 1 Signals and Sampling", 3, "EE210 tag: SYN_M1_SAMPLING_P3"),
        "wrong page": Evidence("e1", "EE210 Module 1 Signals and Sampling", 2, "The Nyquist rate is 2B; the Nyquist frequency is fs/2."),
        "invented": Evidence("e1", "EE210 Module 1 Signals and Sampling", 3, "The Nyquist rate is three times the bandwidth."),
    }
    for name, evidence in cases.items():
        base.evidence = [evidence]
        errors, verified = check_evidence([base], corpus)
        assert errors and verified == 0, name
    base.evidence = [cases["wrong page"]]
    assert "found on EE210 Module 1 Signals and Sampling p.3" in check_evidence([base], corpus)[0][0]


def test_an_absent_term_that_occurs_is_rejected(items, corpus):
    from eval.answer_eval.validate_dataset import check_evidence

    item = copy.deepcopy(next(i for i in items if i.id == "ic-02"))
    item.absent_terms = ["Hann window"]  # occurs on Module 3 p.3
    errors, _ = check_evidence([item], corpus)
    assert any("absent term 'Hann window' occurs" in e for e in errors)
