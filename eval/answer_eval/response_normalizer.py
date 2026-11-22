"""Conservative response normaliser (Sprint 2, offline, standard library only).

Makes generated responses consistent for later comparison without changing what they say.

    from eval.answer_eval.response_normalizer import normalize_response
    r = normalize_response(raw)
    r.original     # exactly what was passed in (None stays None)
    r.normalized   # the cleaned text ("" for None)
    r.changed      # whether anything was altered

Rules, applied in this order:
  1. None -> "" (r.original keeps None).
  2. Line endings: CRLF and lone CR -> LF.
  3. Unicode NFC (composes characters; never NFKC, which would turn "x²" into "x2" or "ﬁ" into "fi").
  4. A leading byte-order mark and zero-width spaces (U+200B, U+FEFF) are removed. Zero-width
     joiners/non-joiners (U+200C/U+200D) are kept: they matter in Devanagari and emoji.
  5. Outside fenced code blocks only:
       - no-break, thin and other fixed-width spaces (U+00A0, U+2007, U+2009, U+202F ...) -> " ";
       - tabs and runs of spaces inside a line -> one space (leading indentation is kept, so markdown
         lists and quoted blocks keep their structure);
       - trailing spaces on each line are removed;
       - three or more consecutive newlines -> one blank line.
     Inside ``` fences only line endings, NFC and rule 4 apply; code layout is preserved.
  6. Leading and trailing whitespace of the whole response is removed.

Never changed: words, letter case, punctuation, numbers, units, mathematical expressions and LaTeX,
citations, negations, and Unicode minus signs (U+2212 etc.; their handling is deferred to Sprint 3).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

__all__ = ["NormalizedResponse", "normalize_response", "normalize_text"]

_FENCE = re.compile(r"^[ \t]*(```|~~~)")
_ZERO_WIDTH = dict.fromkeys(map(ord, "​﻿"), None)
# Fixed-width and no-break spaces that render as an ordinary space. U+3000 (ideographic) is left alone.
_SPACES = re.compile("[   -   ]")
_INNER_RUN = re.compile(r"(?<=\S)[ \t]{2,}|(?<=\S)\t")


@dataclass(frozen=True)
class NormalizedResponse:
    original: str | None
    normalized: str
    changed: bool


def _normalize_line(line: str) -> str:
    line = _SPACES.sub(" ", line)
    indent_len = len(line) - len(line.lstrip(" \t"))
    indent, body = line[:indent_len], line[indent_len:]
    body = _INNER_RUN.sub(" ", body)
    return (indent + body).rstrip(" \t")


def normalize_text(text: str | None) -> str:
    """Return the normalised text ("" for None). See the module docstring for the rules."""
    if text is None:
        return ""
    if not isinstance(text, str):
        raise TypeError(f"expected str or None, got {type(text).__name__}")
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = unicodedata.normalize("NFC", t)
    t = t.translate(_ZERO_WIDTH)

    out: list[str] = []
    in_fence = False
    fence_marker = ""
    blank_run = 0
    for line in t.split("\n"):
        m = _FENCE.match(line)
        if in_fence:
            out.append(line)
            if m and m.group(1) == fence_marker:
                in_fence = False
            continue
        if m:
            in_fence, fence_marker = True, m.group(1)
            blank_run = 0
            out.append(line.rstrip(" \t"))
            continue
        line = _normalize_line(line)
        if line == "":
            blank_run += 1
            if blank_run > 1:
                continue
        else:
            blank_run = 0
        out.append(line)
    return "\n".join(out).strip()


def normalize_response(text: str | None) -> NormalizedResponse:
    """Normalise `text` and keep the original alongside it.

    Returns NormalizedResponse(original=text, normalized=..., changed=...). `changed` is False when the
    normalised text equals the input (None counts as changed only if you compare to ""; it is reported
    as changed=False because there was nothing to alter).
    """
    normalized = normalize_text(text)
    return NormalizedResponse(original=text, normalized=normalized,
                              changed=(text is not None and normalized != text))
