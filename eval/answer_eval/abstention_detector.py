"""Deterministic abstention detector (Sprint 2, offline, standard library only).

Decides whether an assistant response explicitly says it cannot answer. This is a phrase-pattern
baseline, NOT a semantic model: it recognises the explicit wordings listed below and will miss
paraphrases it has no pattern for (see "Known limitations").

    from eval.answer_eval.abstention_detector import detect_abstention
    r = detect_abstention(response_text)
    r.abstained, r.category, r.matched_text, r.mixed

Categories (what kind of "cannot answer" was said):

  not_in_materials          the course materials / notes / excerpts / context do not cover it; includes
                            Saarthi's own fixed reply "I couldn't find this in your course materials ..."
                            (src/prompts/tutor.py NOT_COVERED) and the Gemini prompt's "the notes do not
                            cover it" wording.
  insufficient_information  not enough / insufficient information or context to answer.
  does_not_know             "I don't know" and Hinglish "mujhe nahi pata".
  cannot_answer             first-person inability: "I cannot answer / determine / tell ...".
  scope_refusal             a refusal on scope or policy ("I can only help with course topics",
                            "outside the scope of this course"). Reported with is_refusal=True, so a
                            later stage can treat inability and refusal differently.

A match is ignored when it is:
  * inside double quotes (straight or curly), inline code or a fenced code block, i.e. quoted or
    mentioned rather than said ("Never reply \\"I don't know\\"");
  * reported speech: directly after "say", "said", "reply", "write", "answer with" ... in the same sentence;
  * conditional: after "if", "unless", "in case", "suppose", "agar" in the same sentence
    ("If the notes do not cover a topic, I will say so");
  * explicitly negated: after "not true that", "isn't true that", "not the case that", "wrong to say".

Mixed answers: when the response also contains substantial other content (at least MIXED_MIN_CHARS
letters/digits outside the matched sentences), `mixed` is True. `abstained` stays True; a later stage
decides how to score a partial answer. Saarthi's adapter uses the same 40-character idea
(saarthi_backend/ai/adapter._substantive).

Known limitations: hedges ("I'm not sure", "probably") are not abstentions here; indirect refusals with
no pattern ("Let's focus on the syllabus instead") are missed; Hinglish coverage is a small phrase list;
clarifying questions are not detected (that is a separate behaviour).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["AbstentionMatch", "AbstentionResult", "detect_abstention", "CATEGORIES", "MIXED_MIN_CHARS"]

CATEGORIES = ("not_in_materials", "insufficient_information", "does_not_know", "cannot_answer", "scope_refusal")
MIXED_MIN_CHARS = 40

_NEG = r"(?:not|n't|n’t)"
_SRC = (r"(?:course\s+materials?|materials?|lecture\s+notes|notes|excerpts?|context|documents?|sources?|"
        r"passages?|lectures?|transcripts?|study\s+materials?|slides)")
_DET = r"(?:the|your|these|this|my|provided|given|available|supplied|retrieved)"
_APOS = r"['’]"

_RULES: list[tuple[str, re.Pattern]] = [
    # ── insufficient_information (checked first: "the context does not contain enough information") ──
    # Subject must be the assistant or the source ("I", "we", "the provided context"): "the DFT does not
    # have enough data points" and "the samples carry insufficient information about ..." are DSP, not abstention.
    ("insufficient_information", re.compile(
        rf"\b(?:I|we|{_SRC})\s+(?:do|does|did)\s*{_NEG}\s+(?:have|contain|provide|include|give)\s+(?:enough|sufficient)\s+"
        r"(?:information|context|details?|evidence)\b", re.I)),
    ("insufficient_information", re.compile(
        rf"\b(?:I|we|{_SRC})\s+(?:don{_APOS}t|doesn{_APOS}t|didn{_APOS}t)\s+(?:have|contain|provide|include|give)\s+"
        r"(?:enough|sufficient)\s+(?:information|context|details?|evidence)\b", re.I)),
    ("insufficient_information", re.compile(
        rf"\bthere\s+(?:is|isn{_APOS}t|is\s+not)\s+(?:not\s+)?enough\s+(?:information|context|details)\s+"
        r"(?:to|for)\s+(?:answer|determine|say|tell|respond|give|work\s+out)", re.I)),
    ("insufficient_information", re.compile(
        r"\b(?:insufficient|not\s+enough)\s+(?:information|context|details|evidence)\s+(?:to|for)\s+"
        r"(?:answer|determine|say|tell|respond|give|work\s+out)\b", re.I)),
    # ── not_in_materials ──
    ("not_in_materials", re.compile(
        rf"\bI\s+(?:could\s*{_NEG}|couldn{_APOS}t|cannot|can{_APOS}t|can\s+not|was\s+unable\s+to|am\s+unable\s+to|"
        rf"did\s*{_NEG}|didn{_APOS}t)\s+find\s+(?:this|that|it|anything|any\s+information|information|an?\s+answer|the\s+answer)"
        rf"\b[^.\n]{{0,80}}?\b{_SRC}", re.I)),
    ("not_in_materials", re.compile(
        rf"\b{_DET}(?:\s+{_DET})?\s+(?:course\s+|provided\s+)?{_SRC}\s+(?:do|does|did)\s*{_NEG}\s+"
        r"(?:cover|contain|include|mention|discuss|provide|address|explain|describe|give|say|answer)\b", re.I)),
    ("not_in_materials", re.compile(
        rf"\b{_DET}(?:\s+{_DET})?\s+(?:course\s+|provided\s+)?{_SRC}\s+(?:don{_APOS}t|doesn{_APOS}t|didn{_APOS}t)\s+"
        r"(?:cover|contain|include|mention|discuss|provide|address|explain|describe|give|say|answer)\b", re.I)),
    ("not_in_materials", re.compile(
        rf"\b(?:is|are|was|were)\s*{_NEG}\s+(?:covered|mentioned|discussed|included|described|explained|addressed)\s+"
        rf"(?:in|by)\s+{_DET}(?:\s+{_DET})?\s+(?:course\s+|provided\s+)?{_SRC}", re.I)),
    ("not_in_materials", re.compile(
        rf"\b(?:isn{_APOS}t|aren{_APOS}t|wasn{_APOS}t)\s+(?:covered|mentioned|discussed|included|described|explained)\s+"
        rf"(?:in|by)\s+{_DET}(?:\s+{_DET})?\s+(?:course\s+|provided\s+)?{_SRC}", re.I)),
    ("not_in_materials", re.compile(
        rf"\b(?:there\s+is\s+)?(?:no|nothing)\s+(?:information|mention|details?|coverage)\s+(?:about|on|of|regarding)\b"
        rf"[^.\n]{{0,80}}?\b(?:in|within)\s+{_DET}(?:\s+{_DET})?\s+(?:course\s+|provided\s+)?{_SRC}", re.I)),
    # ── insufficient_information ──
    # ── does_not_know ──
    ("does_not_know", re.compile(rf"\bI\s+(?:do\s*{_NEG}|don{_APOS}t|dont)\s+know\b", re.I)),
    ("does_not_know", re.compile(r"\b(?:mujhe|mujhko|muje)\s+(?:ye\s+|yeh\s+|iska\s+|is\s+ka\s+|is\s+baare\s+mein\s+)?"
                                 r"(?:nahi|nahin|nhi)\s+pata\b", re.I)),
    # "pata nahi chalta" means "cannot be told" in an explanation ("sirf samples se pata nahi chalta"), not "I don't know"
    ("does_not_know", re.compile(r"\bpata\s+(?:nahi|nahin|nhi)\b(?!\s+(?:chalta|chalti|chalega|lagta|lagti|lag))", re.I)),
    # ── cannot_answer (first person) ──
    ("cannot_answer", re.compile(
        rf"\bI(?:\s+(?:cannot|can{_APOS}t|can\s+not|am\s+unable\s+to|am\s+not\s+able\s+to)|{_APOS}m\s+(?:unable|not\s+able)\s+to)"
        r"\s+(?:answer|determine|tell|say|find|provide|give|confirm|verify|work\s+out)\b", re.I)),
    ("cannot_answer", re.compile(
        r"\b(?:jawab|answer|uttar)\s+(?:nahi|nahin|nhi)\s+de\s+(?:sakta|sakti|sakte|paunga|paungi|payenge)\b", re.I)),
    ("cannot_answer", re.compile(
        r"\b(?:main|mai|mein)\s+(?:ye\s+|yeh\s+|iska\s+|is\s+ka\s+)?(?:nahi|nahin|nhi)\s+(?:bata|keh)\s+(?:sakta|sakti)\b", re.I)),
    ("not_in_materials", re.compile(
        r"\b(?:notes|materials?|course\s+materials?|context|excerpts?|lectures?)\s+(?:mein|me|mai|mei)\s+"
        r"(?:(?:ye|yeh|iska|is\s+ka|iske\s+baare\s+mein|is\s+baare\s+mein|is\s+topic\s+ka|koi)\s+)*"
        r"(?:(?:jawab|answer|jaankari|information|info|mention|zikr|detail)\s+)?(?:nahi|nahin|nhi)\s+(?:hai|he|diya|mila|milta|hota)\b",
        re.I)),
    ("not_in_materials", re.compile(
        r"\b(?:cover|mention)\s+(?:nahi|nahin|nhi)\s+(?:kiya|kiye|hai|hota|karte|karta|karti)\b", re.I)),
    # ── scope_refusal ──
    ("scope_refusal", re.compile(rf"\bI\s+(?:can\s+only|only)\s+(?:help|answer|assist)\s+(?:with|you\s+with)\b", re.I)),
    ("scope_refusal", re.compile(
        r"\b(?:outside|beyond)\s+(?:the\s+)?scope\s+of\s+(?:this|the|your|these)\s+(?:course|class|subject|materials?|notes)\b", re.I)),
    ("scope_refusal", re.compile(rf"\bI\s+(?:can{_APOS}t|cannot|can\s+not|won{_APOS}t|will\s+not)\s+help\s+with\b", re.I)),
]

# Sentences that only accompany an abstention (Saarthi's NOT_COVERED reply continues with advice such
# as "Try rephrasing ... or ask your instructor to upload material on it."). They are not counted as
# answer content when deciding `mixed`.
_SUPPORT = re.compile(
    r"\b(?:try\s+(?:rephrasing|asking|again|searching)|rephrase\s+(?:your|the)\s+question|"
    r"ask\s+your\s+(?:instructor|teacher|professor|ta)|upload\s+(?:the\s+)?material|pick\s+the\s+right\s+course|"
    r"check\s+(?:your|the)\s+(?:course\s+)?(?:materials?|notes|syllabus)|consult\s+(?:your|the)\s+(?:textbook|instructor)|"
    r"so\s+I\s+won['’]t\s+guess)", re.I)

_QUOTED = [
    re.compile(r"```.*?(?:```|\Z)", re.S),
    re.compile(r"`[^`\n]*`"),
    re.compile(r"\"[^\"\n]{1,300}\""),
    re.compile(r"“[^”\n]{1,300}”"),
]
_SENTENCE_END = re.compile(r"[.!?\n](?=\s|$)|\n")
_REPORTED = re.compile(r"\b(?:say|says|said|saying|reply|replies|replied|write|writes|wrote|respond|responds|"
                       r"answer\s+with|answers\s+with)\s*[:,]?\s*$", re.I)
# Only clearly conditional openers: "when"/"jab" are usually temporal ("When I searched, I couldn't find ...").
_CONDITIONAL = re.compile(r"\b(?:if|unless|suppose|supposing|in\s+case|agar)\b", re.I)
_NEGATED = re.compile(r"\b(?:not\s+true\s+that|isn['’]t\s+true\s+that|not\s+the\s+case\s+that|wrong\s+to\s+say(?:\s+that)?|"
                      r"never\s+said)\b", re.I)


@dataclass(frozen=True)
class AbstentionMatch:
    category: str
    text: str        # the matched phrase, as written in the response
    start: int       # offsets in the response that was passed in
    end: int
    sentence: str    # the whole sentence containing the match


@dataclass(frozen=True)
class AbstentionResult:
    """Result of detect_abstention().

    abstained      True if at least one explicit abstention statement was found.
    category       category of the first match, or None.
    matched_text   the first matched phrase, or None.
    matches        every match, in order of position.
    is_refusal     True if every match is a scope_refusal (refusal rather than inability).
    mixed          True if the response also has substantial other content (a partial answer).
    empty          True for None, "" or whitespace-only input (not counted as an abstention).
    reason         short human-readable explanation of the decision.
    """
    abstained: bool
    category: str | None = None
    matched_text: str | None = None
    matches: tuple[AbstentionMatch, ...] = ()
    is_refusal: bool = False
    mixed: bool = False
    empty: bool = False
    reason: str = ""


def _blank_quoted(text: str) -> str:
    chars = list(text)
    for pat in _QUOTED:
        for m in pat.finditer("".join(chars)):
            for i in range(m.start(), m.end()):
                if chars[i] != "\n":
                    chars[i] = " "
    return "".join(chars)


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    s = 0
    for m in _SENTENCE_END.finditer(text, 0, start):
        s = m.end()
    m = _SENTENCE_END.search(text, end)
    e = m.end() if m else len(text)
    return s, e


def detect_abstention(text: str | None, *, mixed_min_chars: int = MIXED_MIN_CHARS) -> AbstentionResult:
    """Detect an explicit abstention in `text`.

    text             the assistant response (raw or normalised). None, "" and whitespace return
                     AbstentionResult(abstained=False, empty=True).
    mixed_min_chars  letters/digits that must remain outside the matched sentences for `mixed`.
    """
    if text is None or not text.strip():
        return AbstentionResult(False, empty=True, reason="empty response")

    scan = _blank_quoted(text)
    found: list[AbstentionMatch] = []
    taken: list[tuple[int, int]] = []
    for category, pattern in _RULES:
        for m in pattern.finditer(scan):
            s, e = m.start(), m.end()
            if any(s < te and e > ts for ts, te in taken):
                continue
            sent_s, sent_e = _sentence_bounds(scan, s, e)
            prefix = scan[sent_s:s]
            if _REPORTED.search(prefix) or _CONDITIONAL.search(prefix) or _NEGATED.search(prefix):
                continue
            taken.append((s, e))
            found.append(AbstentionMatch(category, text[s:e], s, e, text[sent_s:sent_e].strip()))

    if not found:
        return AbstentionResult(False, reason="no explicit abstention phrase")

    found.sort(key=lambda a: a.start)
    covered = [False] * len(text)
    for a in found:
        s, e = _sentence_bounds(scan, a.start, a.end)
        for i in range(s, e):
            covered[i] = True
    # Advice sentences that only accompany the abstention do not count as answer content.
    pos = 0
    for m in list(_SENTENCE_END.finditer(scan)) + [None]:
        end = m.end() if m else len(scan)
        if _SUPPORT.search(scan[pos:end]):
            for i in range(pos, end):
                covered[i] = True
        pos = end
    remainder = sum(1 for i, ch in enumerate(text) if not covered[i] and ch.isalnum())
    mixed = remainder >= mixed_min_chars
    first = found[0]
    is_refusal = all(a.category == "scope_refusal" for a in found)
    reason = f"{first.category}: {first.text!r}" + (f"; plus {remainder} characters of other content" if mixed else "")
    return AbstentionResult(True, first.category, first.text, tuple(found), is_refusal, mixed, False, reason)
