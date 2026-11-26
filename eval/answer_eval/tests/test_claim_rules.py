"""Sprint 4 Phase 1: claim rule registry (claim_rules.py) and review CLI (review_claim_rules.py).

Run: python -m pytest eval/answer_eval/tests -q -p no:cacheprovider

All rule records here are synthetic. Patterns are deliberately meaningless placeholders
(SYNTHETIC_TEST_PATTERN): Phase 1 creates no real detection rules.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.answer_eval import review_claim_rules  # noqa: E402
from eval.answer_eval.claim_rules import (  # noqa: E402
    COVERAGE_FIELDS, REGISTRY_PATH, RULE_FIELDS, MustNotCoverage, claim_id, coverage_id, dataset_claims,
    dataset_must_not, load_registry, parse_claim_id, parse_coverage_id, parse_registry, rule_id,
)
from eval.answer_eval.schema import Claim, load_dataset  # noqa: E402

PKG = Path(__file__).resolve().parents[1]
DATASET = PKG / "dataset.jsonl"
ITEMS, _ERRS = load_dataset(DATASET)
BY_ID = {i.id: i for i in ITEMS}
TODAY = dt.date(2026, 11, 23)
PATTERN = "SYNTHETIC_TEST_PATTERN"


def rule(**over) -> dict:
    """A valid draft rule for df-04's second claim ("The output has 15 samples")."""
    r = {
        "record_type": "forbidden_claim_rule",
        "rule_id": "df-04#c2/r1",
        "claim_id": "df-04#c2",
        "item_id": "df-04",
        "claim_text": "The output has 15 samples",
        "reason": "false",
        "feasibility": "deterministic",
        "pattern": None,
        "anchors": [],
        "exclusions": [],
        "languages": ["en"],
        "status": "draft",
        "author": "claude",
        "reviewer": None,
        "reviewed_on": None,
        "rationale": "",
        "limitations": "",
        "positive_examples": [],
        "negative_examples": [],
    }
    r.update(over)
    return r


def approved(**over) -> dict:
    base = dict(
        status="approved", pattern=PATTERN, reviewer="kartik", reviewed_on="2026-11-23",
        rationale="synthetic test rule",
        positive_examples=["synthetic positive one", "synthetic positive two"],
        negative_examples=["synthetic negative one", "synthetic negative two",
                           "it is not the synthetic claim"],
    )
    base.update(over)
    return rule(**base)


def coverage(**over) -> dict:
    c = {
        "record_type": "must_not_coverage",
        "coverage_id": "ic-02#m2",
        "item_id": "ic-02",
        "must_not_text": "Give Hamming window coefficients from general knowledge",
        "covered_by": "forbidden_claim",
        "claim_ids": ["ic-02#c1"],
        "rationale": "synthetic test record",
        "author": "claude",
    }
    c.update(over)
    return c


def parse(*records, items=ITEMS, raw_lines=None):
    lines = raw_lines if raw_lines is not None else [json.dumps(r) for r in records]
    return parse_registry(lines, items, source="test.jsonl", today=TODAY)


def errors_of(*records, **kw) -> str:
    return "\n".join(parse(*records, **kw).errors)


def test_dataset_loads():
    assert _ERRS == [] and len(ITEMS) == 37


# ── IDs ─────────────────────────────────────────────────────────────────────────────────────

def test_id_helpers():
    assert claim_id("df-04", 2) == "df-04#c2"
    assert rule_id("df-04#c2", 1) == "df-04#c2/r1"
    assert coverage_id("ic-02", 1) == "ic-02#m1"
    assert parse_claim_id("df-04#c2") == ("df-04", 2)
    assert parse_coverage_id("ic-02#m1") == ("ic-02", 1)
    for bad in ("df-04#c0", "df-04#2", "DF-04#c1", "df-04#c01", "df-04#m1", "", None):
        assert parse_claim_id(bad) is None


def test_derived_ids_follow_dataset_positions():
    claims = dataset_claims(ITEMS)
    assert len(claims) == 49
    assert list(claims)[:2] == ["df-01#c1", "df-01#c2"]
    assert claims["df-04#c2"][2:] == ("The output has 15 samples", "false")
    assert claims["ic-02#c1"][3] == "unsupported" and claims["ic-02#c2"][3] == "false"
    assert "ic-06#c1" not in claims
    must_not = dataset_must_not(ITEMS)
    assert len(must_not) == 7 and list(must_not)[0] == "am-01#m1"
    assert set(must_not) >= {"ic-02#m1", "ic-02#m2", "ic-06#m1"}


# ── Loading ─────────────────────────────────────────────────────────────────────────────────

def test_production_registry_is_valid_and_has_no_approved_rules():
    reg = load_registry(REGISTRY_PATH, ITEMS)
    assert reg.errors == []
    assert reg.approved_rules() == []
    assert not any(r.status == "approved" for r in reg.rules)


def test_empty_registry_loads(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    reg = load_registry(p, ITEMS)
    assert reg.valid and reg.rules == [] and reg.coverage == [] and reg.approved_rules() == []


def test_blank_lines_are_allowed():
    assert parse(raw_lines=["", "   ", json.dumps(rule()), ""]).valid


def test_missing_registry_file_is_an_error(tmp_path):
    reg = load_registry(tmp_path / "nope.jsonl", ITEMS)
    assert not reg.valid and "cannot read registry" in reg.errors[0]


def test_valid_draft_rule_loads():
    reg = parse(rule())
    assert reg.valid and len(reg.rules) == 1
    r = reg.rules[0]
    assert (r.rule_id, r.claim_id, r.status, r.line) == ("df-04#c2/r1", "df-04#c2", "draft", 1)
    assert not r.is_approved


def test_valid_draft_with_pattern_and_examples_loads():
    assert parse(rule(pattern=PATTERN, anchors=["SYNTH_A"], exclusions=["SYNTH_X"],
                      positive_examples=["p"], negative_examples=["n"])).valid


@pytest.mark.parametrize("status", ["rejected", "retired"])
def test_rejected_and_retired_rules_load(status):
    reg = parse(rule(status=status, rationale="no longer used"))
    assert reg.valid and reg.approved_rules() == []


def test_valid_coverage_loads():
    reg = parse(coverage())
    assert reg.valid and reg.coverage[0].covered_by == "forbidden_claim"
    assert reg.coverage[0].claim_ids == ("ic-02#c1",)


@pytest.mark.parametrize("over", [
    {"covered_by": "behaviour", "claim_ids": [], "coverage_id": "am-01#m1", "item_id": "am-01",
     "must_not_text": "States a single period without saying which signal it belongs to"},
    {"covered_by": "none", "claim_ids": [], "coverage_id": "ic-06#m1", "item_id": "ic-06",
     "must_not_text": "Give a recipe or cooking instructions"},
])
def test_other_coverage_kinds_load(over):
    assert parse(coverage(**over)).valid


def test_errors_carry_line_numbers_and_ids():
    reg = parse(raw_lines=["", json.dumps(rule(feasibility="maybe"))])
    assert reg.errors[0].startswith("test.jsonl:2 [df-04#c2/r1]:")


# ── Structure ───────────────────────────────────────────────────────────────────────────────

def test_invalid_json():
    assert "invalid JSON" in errors_of(raw_lines=["{not json"])


def test_non_object_line():
    assert "must be a JSON object" in errors_of(raw_lines=["[1, 2]"])


@pytest.mark.parametrize("rtype", ["rule", "forbidden_claim", None, 3])
def test_invalid_record_type_rejected(rtype):
    assert "'record_type' must be one of" in errors_of(rule(record_type=rtype))


def test_missing_record_type_rejected():
    r = rule()
    del r["record_type"]
    assert "'record_type' must be one of" in errors_of(r)


@pytest.mark.parametrize("name", sorted(RULE_FIELDS))
def test_rule_missing_field_rejected(name):
    if name == "record_type":
        return
    r = rule()
    del r[name]
    assert f"missing field '{name}'" in errors_of(r)


@pytest.mark.parametrize("name", sorted(COVERAGE_FIELDS))
def test_coverage_missing_field_rejected(name):
    if name == "record_type":
        return
    c = coverage()
    del c[name]
    assert f"missing field '{name}'" in errors_of(c)


def test_unknown_field_rejected():
    assert "unknown field 'match'" in errors_of(rule(match="x"))
    assert "unknown field 'status'" in errors_of(coverage(status="approved"))


@pytest.mark.parametrize("name,value", [("anchors", "x"), ("languages", "en"), ("pattern", 5),
                                        ("reviewer", 7), ("positive_examples", None), ("author", None)])
def test_wrong_field_types_rejected(name, value):
    assert f"'{name}' must be" in errors_of(rule(**{name: value}))


def test_list_entries_must_be_non_empty_strings():
    assert "non-empty strings only" in errors_of(rule(anchors=["ok", ""]))
    assert "non-empty strings only" in errors_of(rule(negative_examples=[1]))


def test_duplicate_rule_ids_rejected():
    e = errors_of(rule(), rule())
    assert "duplicate rule_id (first on line 1)" in e and "test.jsonl:2" in e


def test_two_rules_for_one_claim_with_distinct_ids_load():
    assert parse(rule(), rule(rule_id="df-04#c2/r2")).valid


def test_duplicate_coverage_ids_rejected():
    assert "duplicate coverage_id" in errors_of(coverage(), coverage())


@pytest.mark.parametrize("value", ["exact", "", "Deterministic"])
def test_invalid_feasibility_rejected(value):
    assert "'feasibility' must be one of" in errors_of(rule(feasibility=value))


@pytest.mark.parametrize("value", ["proposed", "", "APPROVED"])
def test_invalid_status_rejected(value):
    assert "'status' must be one of" in errors_of(rule(status=value))


def test_invalid_reason_rejected():
    assert "'reason' must be one of" in errors_of(rule(reason="wrong"))


@pytest.mark.parametrize("langs", [[], ["hi"], ["en", "mr"]])
def test_languages_limited_to_active_scope(langs):
    assert "'languages' must be a non-empty subset" in errors_of(rule(languages=langs))


def test_languages_must_include_item_answer_language():
    e = errors_of(rule(rule_id="ml-02#c2/r1", claim_id="ml-02#c2", item_id="ml-02",
                       claim_text="The tone appears at 9 kHz", languages=["en"]))
    assert "must include the item's answer language 'hinglish'" in e
    assert parse(rule(rule_id="ml-02#c2/r1", claim_id="ml-02#c2", item_id="ml-02",
                      claim_text="The tone appears at 9 kHz", languages=["hinglish"])).valid


@pytest.mark.parametrize("rid", ["df-04#c2", "df-04#c2/1", "df-04#c2/r0", "df-04#c1/r1"])
def test_rule_id_convention(rid):
    assert "'rule_id'" in errors_of(rule(rule_id=rid))


# ── References and drift ────────────────────────────────────────────────────────────────────

def test_unknown_item_rejected():
    e = errors_of(rule(rule_id="zz-99#c1/r1", claim_id="zz-99#c1", item_id="zz-99"))
    assert "unknown item id 'zz-99'" in e


def test_unknown_claim_id_rejected():
    e = errors_of(rule(rule_id="df-04#c3/r1", claim_id="df-04#c3"))
    assert "unknown claim id 'df-04#c3' (df-04 has 2 forbidden claim(s))" in e


def test_claim_on_item_without_claims_rejected():
    e = errors_of(rule(rule_id="ic-06#c1/r1", claim_id="ic-06#c1", item_id="ic-06"))
    assert "unknown claim id 'ic-06#c1'" in e


@pytest.mark.parametrize("cid", ["df-04-c2", "df-04#claim2", "df-4#c2"])
def test_malformed_claim_id_rejected(cid):
    assert "'claim_id' must look like" in errors_of(rule(claim_id=cid))


def test_claim_id_must_belong_to_item_id():
    e = errors_of(rule(item_id="df-03"))
    assert "does not belong to item 'df-03'" in e


def test_claim_text_drift_detected():
    e = errors_of(rule(claim_text="The output has 16 samples"))
    assert "claim text drift for df-04#c2" in e


def test_whitespace_only_change_is_still_drift():
    assert "claim text drift" in errors_of(rule(claim_text="The output has 15 samples "))


def test_claim_reason_drift_detected():
    e = errors_of(rule(reason="unsupported"))
    assert "claim reason drift for df-04#c2: registry says 'unsupported', dataset says 'false'" in e


def test_positional_drift_in_record_detected():
    # Record says position 1 but copies the text that is at position 2.
    e = errors_of(rule(rule_id="df-04#c1/r1", claim_id="df-04#c1"))
    assert "claim position drift for df-04#c1: this text is now claim df-04#c2" in e


def test_positional_drift_after_dataset_reorder_detected():
    df04 = BY_ID["df-04"]
    swapped = replace(df04, forbidden_claims=list(reversed(df04.forbidden_claims)))
    items = [swapped if i.id == "df-04" else i for i in ITEMS]
    e = errors_of(rule(), items=items)
    assert "claim position drift for df-04#c2: this text is now claim df-04#c1" in e


def test_dataset_reason_change_detected():
    df04 = BY_ID["df-04"]
    changed = replace(df04, forbidden_claims=[df04.forbidden_claims[0],
                                              Claim("The output has 15 samples", "unsupported")])
    items = [changed if i.id == "df-04" else i for i in ITEMS]
    assert "claim reason drift" in errors_of(rule(), items=items)


def test_dataset_claim_removal_detected():
    df04 = BY_ID["df-04"]
    items = [replace(df04, forbidden_claims=df04.forbidden_claims[:1]) if i.id == "df-04" else i
             for i in ITEMS]
    assert "unknown claim id 'df-04#c2'" in errors_of(rule(), items=items)


def test_coverage_text_drift_and_position_drift():
    assert "must_not text drift for ic-02#m2" in errors_of(coverage(must_not_text="Give coefficients"))
    e = errors_of(coverage(coverage_id="ic-02#m1"))
    assert "must_not position drift for ic-02#m1: this text is now ic-02#m2" in e


def test_coverage_unknown_ids():
    assert "unknown coverage id 'ic-02#m3'" in errors_of(coverage(coverage_id="ic-02#m3"))
    assert "unknown item id 'zz-01'" in errors_of(coverage(coverage_id="zz-01#m1", item_id="zz-01"))
    assert "'coverage_id' must look like" in errors_of(coverage(coverage_id="ic-02#c1"))


def test_coverage_claim_links_validated():
    assert "needs at least one claim id" in errors_of(coverage(claim_ids=[]))
    assert "unknown claim id 'ic-02#c9'" in errors_of(coverage(claim_ids=["ic-02#c9"]))
    assert "belongs to another item" in errors_of(coverage(claim_ids=["df-04#c1"]))
    assert "is malformed" in errors_of(coverage(claim_ids=["ic-02"]))
    assert "must be empty unless covered_by" in errors_of(coverage(covered_by="behaviour"))
    assert "'covered_by' must be one of" in errors_of(coverage(covered_by="claim"))


# ── Regexes ─────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("over,what", [
    ({"pattern": "(unclosed"}, "'pattern' is not a valid regex"),
    ({"anchors": ["[a-"]}, "'anchors[0]' is not a valid regex"),
    ({"exclusions": ["ok", "*bad"]}, "'exclusions[1]' is not a valid regex"),
    ({"pattern": "   "}, "'pattern' must be null or a non-empty regex"),
])
def test_invalid_regexes_rejected(over, what):
    assert what in errors_of(rule(**over))


# ── Approval requirements ───────────────────────────────────────────────────────────────────

def test_synthetic_approved_rule_is_valid_and_returned():
    reg = parse(approved())
    assert reg.valid and [r.rule_id for r in reg.approved_rules()] == ["df-04#c2/r1"]


@pytest.mark.parametrize("reviewer", [None, "", "claude", " Claude "])
def test_approved_needs_independent_reviewer(reviewer):
    e = errors_of(approved(reviewer=reviewer))
    assert "needs a 'reviewer'" in e or "reviewer other than its author" in e


def test_approved_needs_review_date():
    assert "needs 'reviewed_on'" in errors_of(approved(reviewed_on=None))


@pytest.mark.parametrize("date,msg", [
    ("23-11-2026", "ISO date"), ("2026-02-30", "not a real date"), ("2026-11-24", "in the future"),
    ("2026/11/23", "ISO date"),
])
def test_review_date_must_be_valid(date, msg):
    assert msg in errors_of(approved(reviewed_on=date))


def test_review_date_checked_on_drafts_too():
    assert "ISO date" in errors_of(rule(reviewed_on="yesterday"))


def test_approved_needs_pattern():
    assert "needs a 'pattern'" in errors_of(approved(pattern=None))


@pytest.mark.parametrize("n", [0, 1])
def test_approved_needs_two_positive_examples(n):
    assert "at least 2 positive examples" in errors_of(approved(positive_examples=["p"] * n))


@pytest.mark.parametrize("n", [0, 2])
def test_approved_needs_three_negative_examples(n):
    negs = ["it is not that"] + ["x"] * (n - 1) if n else []
    assert "at least 3 negative examples" in errors_of(approved(negative_examples=negs))


def test_approved_needs_a_negation_negative_example():
    e = errors_of(approved(negative_examples=["alpha", "beta", "gamma"]))
    assert "negates or corrects the claim" in e


@pytest.mark.parametrize("neg", ["It is not 15 samples.", "It's wrong to say 15.", "15 isn't right",
                                 "A common misconception is 15 samples.", "15 samples nahi hote"])
def test_negation_examples_recognised(neg):
    assert parse(approved(negative_examples=["a", "b", neg])).valid


def test_approved_needs_rationale():
    assert "needs a 'rationale'" in errors_of(approved(rationale=" "))


@pytest.mark.parametrize("feasibility", ["semantic", "underspecified"])
def test_semantic_or_underspecified_cannot_be_approved(feasibility):
    assert "cannot have an approved pattern rule" in errors_of(approved(feasibility=feasibility))


def test_careful_rule_can_be_approved():
    assert parse(approved(feasibility="careful")).valid


# ── Draft never approved ────────────────────────────────────────────────────────────────────

def test_draft_rules_are_never_approved():
    complete_draft = approved(status="draft")
    reg = parse(complete_draft)
    assert reg.valid and reg.rules[0].status == "draft"
    assert not reg.rules[0].is_approved and reg.approved_rules() == []


def test_invalid_approved_rule_is_not_loaded():
    reg = parse(approved(reviewer="claude"))
    assert reg.rules == [] and reg.approved_rules() == []


def test_any_registry_error_disables_all_approved_rules():
    reg = parse(approved(), rule(rule_id="df-04#c1/r1", claim_id="df-04#c1",
                                  claim_text="The output has 8 samples", feasibility="bogus"))
    assert len(reg.rules) == 1 and reg.rules[0].is_approved
    assert reg.approved_rules() == []


def test_mixed_statuses_only_approved_returned():
    reg = parse(approved(), rule(rule_id="df-04#c2/r2"), rule(rule_id="df-04#c2/r3", status="rejected"),
                rule(rule_id="df-04#c2/r4", status="retired"))
    assert [r.rule_id for r in reg.approved_rules()] == ["df-04#c2/r1"]


# ── must_not coverage is traceability only ───────────────────────────────────────────────────

def test_coverage_records_carry_no_status_or_score():
    assert {"status", "score", "result", "severity"}.isdisjoint(COVERAGE_FIELDS)
    fields = set(MustNotCoverage.__dataclass_fields__)
    assert {"status", "score", "result", "severity"}.isdisjoint(fields)


def test_coverage_never_produces_approved_rules():
    reg = parse(coverage(), coverage(coverage_id="ic-02#m1",
                                     must_not_text="Answer with the Hann window formula from the course "
                                                   "as if it were the Hamming window",
                                     claim_ids=["ic-02#c2"]))
    assert reg.valid and len(reg.coverage) == 2 and reg.rules == [] and reg.approved_rules() == []


@pytest.mark.parametrize("module", ["evaluation_engine.py", "evaluation_dimensions.py",
                                    "evaluation_result.py", "run_evaluation.py"])
def test_evaluation_engine_does_not_use_the_registry(module):
    src = (PKG / module).read_text(encoding="utf-8")
    assert "claim_rules" not in src


def test_engine_output_independent_of_registry_contents(tmp_path):
    from eval.answer_eval.evaluation_engine import evaluate_response
    before = evaluate_response(BY_ID["ic-02"], "The Hamming window is 0.54 - 0.46 cos(2 pi n / (N - 1)).").to_dict()
    parse(approved(), coverage())  # loading a registry has no side effects on the engine
    after = evaluate_response(BY_ID["ic-02"], "The Hamming window is 0.54 - 0.46 cos(2 pi n / (N - 1)).").to_dict()
    assert before == after
    assert after["forbidden_claims"]["detected"] is None
    assert after["dimensions"]["correctness"]["status"] == "not_applicable"


# ── Dataset untouched ────────────────────────────────────────────────────────────────────────

def test_dataset_unchanged():
    data = DATASET.read_bytes().replace(b"\r\n", b"\n")
    blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    assert blob == "0414b04d7f97d264cf688cf88b81da0eb35d9580"


def test_dataset_claim_and_must_not_profile():
    claims = [c for i in ITEMS for c in i.forbidden_claims]
    assert len(claims) == 49
    assert sum(c.reason == "false" for c in claims) == 44
    assert sum(c.reason == "unsupported" for c in claims) == 5
    assert sum(len(i.must_not) for i in ITEMS) == 7


def test_parsing_does_not_mutate_items():
    snapshot = copy.deepcopy([(i.id, list(i.forbidden_claims), list(i.must_not)) for i in ITEMS])
    parse(rule(), approved(rule_id="df-04#c2/r2"), coverage(), rule(claim_text="drift"))
    assert snapshot == [(i.id, list(i.forbidden_claims), list(i.must_not)) for i in ITEMS]


# ── Review CLI ──────────────────────────────────────────────────────────────────────────────

def write(tmp_path, *records) -> Path:
    p = tmp_path / "reg.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return p


def test_cli_on_production_registry():
    lines, code = review_claim_rules.report(DATASET, REGISTRY_PATH, brief=True)
    text = "\n".join(lines)
    assert code == 0
    assert "Forbidden claims: 49 (false 44, unsupported 5)" in text
    assert "must_not entries: 7" in text
    assert "approved 0" in text and "Registry OK." in text


def test_cli_lists_unrecorded_claims_and_uncovered_must_not(tmp_path):
    lines, code = review_claim_rules.report(DATASET, write(tmp_path, rule(), coverage()), item="ic-02")
    text = "\n".join(lines)
    assert code == 0
    assert "Claims with no registry record: 2 of 2" in text
    assert "ic-02#m1   UNCOVERED" in text
    assert "ic-02#m2   covered_by=forbidden_claim ic-02#c1" in text


def test_cli_groups_by_feasibility_and_status(tmp_path):
    reg = write(tmp_path, rule(), rule(rule_id="df-04#c1/r1", claim_id="df-04#c1",
                                       claim_text="The output has 8 samples", feasibility="careful"),
                rule(rule_id="df-04#c1/r2", claim_id="df-04#c1", claim_text="The output has 8 samples",
                     feasibility="semantic", status="rejected"))
    text = "\n".join(review_claim_rules.report(DATASET, reg, item="df-04")[0])
    assert "Rules by status: draft 2, approved 0, rejected 1, retired 0" in text
    assert "deterministic 1" in text and "conflicting 1" in text and "unassessed 0" in text
    assert "Claims with no registry record: 0 of 2" in text


def test_cli_reports_errors_and_exit_code(tmp_path, capsys):
    reg = write(tmp_path, rule(claim_text="drifted"))
    assert review_claim_rules.main(["--registry", str(reg), "--brief"]) == 1
    out = capsys.readouterr().out
    assert "Registry INVALID: 1 error(s)" in out and "claim text drift for df-04#c2" in out


def test_cli_status_filter_and_unknown_item(tmp_path):
    reg = write(tmp_path, rule(), rule(rule_id="df-04#c2/r2", status="retired", rationale="old"))
    text = "\n".join(review_claim_rules.report(DATASET, reg, status="retired")[0])
    assert "retired: df-04#c2/r2" in text
    lines, code = review_claim_rules.report(DATASET, reg, item="zz-99")
    assert code == 1 and "Unknown item id" in lines[0]


def test_cli_is_read_only(tmp_path):
    reg = write(tmp_path, approved(status="draft"))
    before = reg.read_bytes()
    review_claim_rules.main(["--registry", str(reg)])
    assert reg.read_bytes() == before


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as e:
        review_claim_rules.main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--dataset", "--registry", "--item", "--status", "--brief", "Read-only"):
        assert flag in out
