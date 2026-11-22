"""Sprint 2: abstention detector. Run: python -m pytest eval/answer_eval/tests -q -p no:cacheprovider"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.answer_eval.abstention_detector import CATEGORIES, detect_abstention  # noqa: E402


# ── Clear abstentions (true positives) ──────────────────────────────────────

@pytest.mark.parametrize("answer, category", [
    ("I don't know.", "does_not_know"),
    ("I do not know the answer to that.", "does_not_know"),
    ("I cannot determine this from the available information.", "cannot_answer"),
    ("I can't answer that from these excerpts.", "cannot_answer"),
    ("I'm unable to answer that from the notes.", "cannot_answer"),
    ("I am not able to tell from the material provided.", "cannot_answer"),
    ("The provided context does not contain enough information.", "insufficient_information"),
    ("I don't have sufficient information to answer.", "insufficient_information"),
    ("There is not enough information to determine the period.", "insufficient_information"),
    ("The course materials do not cover the Hamming window.", "not_in_materials"),
    ("The notes don't mention the Laplace transform.", "not_in_materials"),
    ("The Hamming window is not covered in the course materials.", "not_in_materials"),
    ("The bilinear transform isn't covered in your notes.", "not_in_materials"),
    ("There is no information about Butterworth filters in the provided excerpts.", "not_in_materials"),
    ("I couldn't find anything about this in the lecture notes.", "not_in_materials"),
    ("I can only help with questions about your course.", "scope_refusal"),
    ("That question is outside the scope of this course.", "scope_refusal"),
])
def test_clear_english_abstentions(answer, category):
    r = detect_abstention(answer)
    assert r.abstained and r.category == category, r
    assert r.matched_text and r.matched_text in answer


def test_saarthis_own_not_covered_reply_is_a_full_abstention():
    from src.prompts.tutor import not_covered  # read-only use of the app's fixed reply text

    for course in (None, "EE210"):
        r = detect_abstention(not_covered(course))
        assert r.abstained and r.category == "not_in_materials"
        assert r.mixed is False, r.reason  # its advice sentence is not answer content


@pytest.mark.parametrize("answer", ["I DON'T KNOW.", "i don't know", "  I   cannot   determine  this.  ",
                                    "THE NOTES DO NOT COVER THIS.", "I don’t know.", "I couldn’t find this in your course materials."])
def test_capitalisation_whitespace_and_curly_apostrophes(answer):
    assert detect_abstention(answer).abstained


@pytest.mark.parametrize("answer, category", [
    ("Mujhe nahi pata.", "does_not_know"),
    ("Sorry, mujhe iska nahi pata.", "does_not_know"),
    ("Notes mein iska jawab nahi hai.", "not_in_materials"),
    ("Ye topic course materials mein cover nahi kiya gaya hai.", "not_in_materials"),
    ("Main is sawal ka jawab nahi de sakta.", "cannot_answer"),
])
def test_hinglish_abstentions(answer, category):
    r = detect_abstention(answer)
    assert r.abstained and r.category == category


# ── Things that are not abstentions (false-positive guards) ─────────────────

@pytest.mark.parametrize("answer", [
    "Aliasing cannot be removed after sampling, because the folded components are already mixed.",
    "The original frequency cannot be determined from the aliased samples alone.",
    "An unknown input x[n] gives an output y[n] = x[n] + 2, which is not linear.",
    "The DFT does not have enough data points to resolve the two tones.",
    "The samples carry insufficient information about frequencies above fs/2.",
    "You cannot recover the signal if fs < 2B.",
    "IIR filters can be unstable and do not have linear phase.",
    "Zero-padding does not improve the true resolution.",
    "Nahi, sampling theorem satisfy nahi hota: fs > 18 kHz chahiye, isliye tone 3 kHz pe dikhega.",
    "Sirf samples se original frequency ka pata nahi chalta.",
])
def test_ordinary_answers_with_negative_words_are_not_abstentions(answer):
    r = detect_abstention(answer)
    assert not r.abstained, r.reason


@pytest.mark.parametrize("answer", [
    'A tutor should never reply "I don\'t know" without checking the notes.',
    "The phrase “the notes do not cover this” is what Saarthi says when retrieval fails.",
    "Use `I cannot determine this` as the fallback string.",
    "If the notes do not cover a topic, I will say so. The Nyquist rate is 2B.",
    "It is not true that the context does not contain enough information: the rate is 2B.",
    "Students often say I don't know when they are unsure; the answer here is 2B.",
])
def test_quoted_conditional_negated_and_reported_phrases_are_ignored(answer):
    r = detect_abstention(answer)
    assert not r.abstained, r.reason


def test_temporal_when_does_not_hide_a_real_abstention():
    assert detect_abstention("When I searched, I couldn't find this in your course materials.").abstained


# ── Mixed answers ───────────────────────────────────────────────────────────

def test_mixed_answer_is_an_abstention_flagged_as_mixed():
    answer = ("The notes do not cover the derivation. However, the result is that the Nyquist rate is 2B, "
              "twice the highest frequency in the signal.")
    r = detect_abstention(answer)
    assert r.abstained and r.mixed and r.category == "not_in_materials"


def test_short_preamble_does_not_make_an_abstention_mixed():
    r = detect_abstention("Sorry! I don't know.")
    assert r.abstained and not r.mixed


def test_several_matches_are_all_reported_in_order():
    r = detect_abstention("I don't know. The notes do not cover this topic.")
    assert [m.category for m in r.matches] == ["does_not_know", "not_in_materials"]
    assert r.matches[0].start < r.matches[1].start


def test_refusal_is_distinguished_from_inability():
    assert detect_abstention("I can only help with questions about your course.").is_refusal is True
    assert detect_abstention("I don't know.").is_refusal is False
    mixed_kinds = detect_abstention("I can only help with course questions, and the notes do not cover this.")
    assert mixed_kinds.abstained and mixed_kinds.is_refusal is False


# ── Empty input and result shape ────────────────────────────────────────────

@pytest.mark.parametrize("answer", [None, "", "   \n\t "])
def test_empty_responses_are_safe_and_not_abstentions(answer):
    r = detect_abstention(answer)
    assert r.abstained is False and r.empty is True and r.matches == ()


def test_categories_are_documented():
    assert set(CATEGORIES) == {"not_in_materials", "insufficient_information", "does_not_know",
                               "cannot_answer", "scope_refusal"}


# ── Regression against the dataset ──────────────────────────────────────────

def test_no_answer_or_clarify_reference_answer_is_detected_as_an_abstention():
    lines = (ROOT / "eval" / "answer_eval" / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    for raw in (json.loads(x) for x in lines if x.strip()):
        if raw["expected_behavior"] in ("answer", "clarify"):
            r = detect_abstention(raw["reference_answer"])
            assert not r.abstained, (raw["id"], r.reason)
