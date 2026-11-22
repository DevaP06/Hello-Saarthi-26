"""Sprint 2: response normaliser. Run: python -m pytest eval/answer_eval/tests -q -p no:cacheprovider"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.answer_eval.response_normalizer import normalize_response, normalize_text  # noqa: E402


# ── Whitespace and line endings ─────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("  The rate is 2B.  ", "The rate is 2B."),
    ("\n\n The rate is 2B.\n\n", "The rate is 2B."),
    ("The  rate   is    2B.", "The rate is 2B."),
    ("The\trate\t\tis 2B.", "The rate is 2B."),
    ("line one   \nline two\t\n", "line one\nline two"),
    ("a\r\nb\r\nc", "a\nb\nc"),
    ("a\rb\rc", "a\nb\nc"),
    ("a\r\n\r\n\r\n\r\nb", "a\n\nb"),
    ("para 1\n\n\n\n\npara 2", "para 1\n\npara 2"),
    ("para 1\n\npara 2", "para 1\n\npara 2"),
])
def test_whitespace_and_line_endings(raw, expected):
    assert normalize_text(raw) == expected


def test_windows_and_unix_line_endings_give_the_same_result():
    unix = "## Answer\n\nThe rate is 2B [1].\n- point one\n- point two\n"
    assert normalize_text(unix.replace("\n", "\r\n")) == normalize_text(unix)


def test_leading_indentation_is_kept_for_markdown_structure():
    raw = "- item\n    - nested   item\n  continuation"
    assert normalize_text(raw) == "- item\n    - nested item\n  continuation"


def test_fenced_code_keeps_its_layout():
    raw = "Code:\r\n```python\r\ndef f(x):\r\n    return  x   *  2   \r\n\r\n\r\n\r\n```\r\nDone   now"
    out = normalize_text(raw)
    assert "def f(x):\n    return  x   *  2   \n\n\n\n```" in out
    assert out.endswith("Done now")


def test_no_break_and_thin_spaces_become_ordinary_spaces():
    assert normalize_text("98 dB and 3 kHz and 5 ms") == "98 dB and 3 kHz and 5 ms"


def test_zero_width_space_and_bom_are_removed_but_joiners_kept():
    assert normalize_text("﻿Nyquist​ rate") == "Nyquist rate"
    assert normalize_text("क्‍ष") == "क्‍ष"


# ── Unicode ─────────────────────────────────────────────────────────────────

def test_unicode_is_normalised_to_nfc():
    decomposed = "Fourier transform of the résumé"
    out = normalize_text(decomposed)
    assert out == unicodedata.normalize("NFC", decomposed) and "́" not in out


def test_compatibility_characters_are_not_folded():
    # NFKC would change these; NFC must not.
    assert normalize_text("x² + y³, ﬁlter, ½") == "x² + y³, ﬁlter, ½"


def test_unicode_minus_signs_are_preserved():
    raw = "len(x) + len(h) − 1 and L + M − 1 and a – b and −3 dB"
    assert normalize_text(raw) == raw
    assert "−" in normalize_text("−3 dB")


# ── Empty and None ──────────────────────────────────────────────────────────

def test_none_and_empty_inputs():
    assert normalize_text(None) == ""
    assert normalize_text("") == ""
    assert normalize_text(" \n\t\r\n ") == ""
    r = normalize_response(None)
    assert r.original is None and r.normalized == "" and r.changed is False


def test_non_string_input_is_rejected_clearly():
    with pytest.raises(TypeError):
        normalize_text(42)  # type: ignore[arg-type]


def test_original_is_kept_alongside_the_normalised_text():
    raw = "  The rate is 2B.  "
    r = normalize_response(raw)
    assert r.original == raw and r.normalized == "The rate is 2B." and r.changed is True
    same = normalize_response("Already clean.")
    assert same.changed is False and same.original == same.normalized


def test_normalisation_is_idempotent():
    raw = " a  b\r\n\r\n\r\n c d ​"
    once = normalize_text(raw)
    assert normalize_text(once) == once


# ── Meaning is preserved ────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "The Nyquist rate is 2B; it is NOT fs/2.",
    "No, the sampling theorem is not satisfied: 12 kHz < 18 kHz, so the tone appears at 3 kHz.",
    "fs = 8000 Hz and N = 256 give 31.25 Hz per bin; bin 1000 / 31.25 = 32.",
    "SQNR = 6.02 N + 1.76 dB, about 98.08 dB for N = 16.",
    "$$X(f) = \\int x(t) e^{-j 2 \\pi f t} \\, dt$$ and $h[n] = (0.5)^n u[n]$.",
    "y[n] = x[n] + 2 isn't linear!? Really: a zero input gives 2.",
    "The rate is 2B [1], see [EE210 Module 1 Signals and Sampling, p. 3] and [Video: W3, 14:22].",
    "**Sources:** [EE210 Module 1 Signals and Sampling, p. 3]; [Lecture 01 Fourier]",
    "Nahi, sampling theorem satisfy nahi hota.",
    "5 ms, 0.02 s, 1 kOhm, 1 uF, 159 Hz, -3 dB, 1,048,576 and 204.8",
])
def test_text_without_extra_whitespace_is_unchanged(text):
    assert normalize_text(text) == text


def test_normalising_keeps_every_reference_answer_and_its_key_point_patterns():
    lines = (ROOT / "eval" / "answer_eval" / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    for raw in (json.loads(x) for x in lines if x.strip()):
        ref = raw["reference_answer"]
        assert normalize_text(ref) == ref, raw["id"]
        noisy = "  " + ref.replace(" ", "   ").replace("\n", "\r\n") + "\r\n\r\n\r\n"
        cleaned = normalize_text(noisy)
        for kp in raw["key_points"]:
            if kp.get("match"):
                assert re.search(kp["match"], cleaned, re.I), (raw["id"], kp["id"])
