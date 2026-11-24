"""Sprint 4 Phase 3: deterministic forbidden-claim matcher.

This module is the single source of the matching semantics used by the forbidden-claim rules in
claim_rules.jsonl. It is not used by the evaluation engine (no integration in Phase 3) and never
approves, scores or changes a rule.

Reading a response (deterministic, no model, no network)
1. Normalise with response_normalizer.normalize_text (line endings, NFC, NBSP, zero-width, spacing).
2. Blank quoted spans "..." and “...” with spaces of the same length: a quoted phrase is a mention,
   not an assertion. Apostrophes and code are not blanked. Offsets are unchanged by blanking, so
   every span reported below indexes the normalised text.
3. Sentences end after . ! ? followed by whitespace or the end of the text, and at newlines. A full
   stop after approx, e.g, i.e, vs, etc, fig, eq, cf, resp, ca or al never ends a sentence; "No." /
   "no." does not end one only when a number follows ("No. 5"). Decimal points never do.
4. Clauses split a sentence only at strong boundaries: ";"; a spaced dash (" — ", " – ", or
   " - "/" -- " before a letter); before a parenthetical that opens with a word (" (commonly");
   before a comma + connective (but, though, although, however, whereas, yet, so, hence, thus,
   which, since, because, while, not); before a spaced but/though/although/whereas/however.
5. A rule's pattern is searched in each sentence case-insensitively (re.IGNORECASE is applied here;
   stored patterns carry no flags). Every non-overlapping match is assessed.
6. Anchors: every anchor must match somewhere in the same sentence.
7. Exclusions are named by a leading regex comment: "(?#name)" (claim scope) or
   "(?#name;next-clause)". Claim scope is the contiguous text of every clause the match touches
   (from the first touched clause to the last), matched text included. Next-clause scope is the
   clause right after the last touched clause. Exclusions never look further.
8. Outcome per match: "no_anchor" if an anchor fails; else "excluded" if any exclusion fires; else
   "affirmative". An excluded match is never a detection: ambiguity is resolved conservatively.

Eligibility (who may produce confirmed detections)
A rule is production-eligible only if: the registry is valid as a whole; the rule's status is
"approved" and the registry's approved_rules() returns it; its exclusion names and scopes are valid;
each positive example is affirmative and no negative example is; and it is not affirmative on its
own item's question or reference answer.

Eligibility is bound to the registry SOURCE TEXT, not to the compiled objects. A CompiledRegistry
keeps its source text, a snapshot of the dataset items it was compiled against, their fingerprint
and the review date used. At match time match_response verifies the digest and fingerprint,
obtains the authoritative compilation of that source (cached by digest, dataset fingerprint and
date) and evaluates nothing unless the supplied object is identical to it. Hand-built objects or
edits such as dataclasses.replace(rule, eligible=True) therefore cannot confirm anything: only the
source text decides, and the source must satisfy the registry's own approval rules. The full
eligibility conditions, including the claim recheck against the item being evaluated, are then
checked again on the authoritative rules.

The authoritative cache is keyed by the normalised digest, a hash of the exact source text, the
source label, the dataset fingerprint, a digest of the full dataset snapshot and the review date, so
different line endings, labels or datasets never replace each other's entries. The cache holds
private canonical compilations: compile_registry_text returns an independent deep copy, so no
reference to a cached object or its dataset snapshot reaches the caller. A returned compilation
whose snapshot items are edited in place (any field, including nested values and fields excluded
from ==) no longer matches its stored snapshot digest and evaluates nothing; any other edit makes it
differ from the canonical compilation. The review date may be injected (for
deterministic tests) but never later than the real current date: a future date would let a
future-dated approval count as valid, so it makes the registry invalid. At match time the item being
evaluated must be identical to the item with the same id in the compiled dataset snapshot; otherwise
nothing is evaluated (an altered question or reference answer cannot sidestep the example checks).

Draft rules run only in shadow mode when asked for, and never affect status. A draft without a
pattern is valid but not runnable and is reported as such. Exclusion declarations are validated on
every record, including rejected and retired rules, which are never compiled for matching. An
exclusion that matches the empty string is an error unless it is a zero-width requirement proven
not to fire on any of the rule's positive examples (see _exclusion_problems). An invalid registry
evaluates nothing. The empty-string exception is decided with the matcher's own clause-level
semantics: the exclusion must not fire (claim or next-clause scope) on any match the rule finds in
any of its positive examples.

Offsets
Every start/end offset refers to the NORMALISED response (MatchReport.normalized_text, serialised
as "normalized_text" with "offsets_refer_to": "normalized_text"), not to the original input.

Known limitations (documented by tests, not fixed here)
- A single newline (LF or CRLF) always ends a sentence, so a claim wrapped across lines is not
  detected. A missed detection is not evidence that the claim is absent.
- A stray double quote (e.g. 12") pairs with a later quote and blanks the text between them, which
  can hide a real claim.
- Corrections or attributions in the NEXT sentence ("... 8 samples. That is wrong.") are not seen,
  so an explained misconception can be flagged. This must be considered before approving a rule.

Statuses
  not_evaluated         no response, invalid registry, or no production-eligible rule for the item
  no_affirmative_match  eligible rules ran and none matched affirmatively. This is a statement about
                        rule matches only, never a verdict on the answer.
  affirmative_match     at least one confirmed detection from an eligible rule
"""
from __future__ import annotations

import copy
import dataclasses
import datetime as _dt
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from eval.answer_eval.claim_rules import REGISTRY_PATH, ClaimRule, parse_registry
from eval.answer_eval.response_normalizer import normalize_text
from eval.answer_eval.schema import Item

__all__ = [
    "NOT_EVALUATED", "NO_AFFIRMATIVE_MATCH", "AFFIRMATIVE_MATCH", "STATUSES",
    "AFFIRMATIVE", "EXCLUDED", "NO_ANCHOR", "OUTCOMES", "EXCLUSION_SCOPES", "ABBREVIATIONS",
    "Span", "ExclusionHit", "Assessment", "ClaimDetection", "CompiledExclusion", "CompiledRule",
    "CompiledRegistry", "MatchReport", "OFFSETS_REFER_TO",
    "prepare_text", "split_sentences", "split_clauses", "parse_exclusion_name", "compile_rule",
    "assess_rule", "registry_digest", "compile_registry_text", "load_compiled_registry",
    "match_response",
]

NOT_EVALUATED = "not_evaluated"
NO_AFFIRMATIVE_MATCH = "no_affirmative_match"
AFFIRMATIVE_MATCH = "affirmative_match"
STATUSES = (NOT_EVALUATED, NO_AFFIRMATIVE_MATCH, AFFIRMATIVE_MATCH)

AFFIRMATIVE = "affirmative"
EXCLUDED = "excluded"
NO_ANCHOR = "no_anchor"
OUTCOMES = (AFFIRMATIVE, EXCLUDED, NO_ANCHOR)

CLAUSE_SCOPE = "clause"
NEXT_CLAUSE_SCOPE = "next-clause"
EXCLUSION_SCOPES = (CLAUSE_SCOPE, NEXT_CLAUSE_SCOPE)

ABBREVIATIONS = ("approx", "e.g", "i.e", "vs", "etc", "fig", "eq", "cf", "resp", "ca", "al")
NUMBERED_ABBREVIATIONS = ("no",)  # abbreviation only when a number follows (D5)

OFFSETS_REFER_TO = "normalized_text"

NO_MATCH_NOTE = ("no_affirmative_match means only that no production-eligible rule matched "
                 "affirmatively; it is never a verdict on the answer.")

_END = re.compile(r"[.!?]+(?=\s|$)|\n+")
_WORD_BEFORE_DOT = re.compile(r"([A-Za-z.]+)$")
_NUMBER_AFTER = re.compile(r"\s*\d")
_QUOTED = re.compile(r"\"[^\"\n]*\"|“[^”\n]*”")
_CLAUSE = re.compile(
    r"\s*;\s*"
    r"|\s+[—–]\s+|\s+-{1,2}\s+(?=[A-Za-z])"
    r"|\s+(?=\([A-Za-z]{2,})"
    r"|,\s*(?=(?:but|though|although|however|whereas|yet|so|hence|thus|which|since|because|while|not)\b)"
    r"|\s+(?=(?:but|though|although|whereas|however)\s)", re.I)
_NAME_COMMENT = re.compile(r"^\(\?#([^)]*)\)")
_NAME = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")


# ── Result structures (immutable, JSON-safe via to_dict) ─────────────────────────────────────

@dataclass(frozen=True)
class Span:
    """A half-open [start, end) range of the normalised response text, with that text."""
    start: int
    end: int
    text: str

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "text": self.text}


@dataclass(frozen=True)
class ExclusionHit:
    name: str
    scope: str
    span: Span

    def to_dict(self) -> dict:
        return {"name": self.name, "scope": self.scope, "span": self.span.to_dict()}


@dataclass(frozen=True)
class Assessment:
    """One pattern match of one rule, and why it was or was not affirmative."""
    rule_id: str
    claim_id: str
    item_id: str
    reason: str
    rule_status: str
    production_eligible: bool
    outcome: str
    match: Span
    sentence: Span
    clause: Span
    touched_clauses: tuple[Span, ...]
    next_clause: Span | None
    failed_anchors: tuple[int, ...]
    exclusions: tuple[ExclusionHit, ...]

    @property
    def is_affirmative(self) -> bool:
        return self.outcome == AFFIRMATIVE

    @property
    def exclusion_names(self) -> tuple[str, ...]:
        return tuple(h.name for h in self.exclusions)

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id, "claim_id": self.claim_id, "item_id": self.item_id,
            "reason": self.reason, "rule_status": self.rule_status,
            "production_eligible": self.production_eligible, "outcome": self.outcome,
            "match": self.match.to_dict(), "sentence": self.sentence.to_dict(),
            "clause": self.clause.to_dict(),
            "touched_clauses": [c.to_dict() for c in self.touched_clauses],
            "next_clause": self.next_clause.to_dict() if self.next_clause else None,
            "failed_anchors": list(self.failed_anchors),
            "exclusions": [h.to_dict() for h in self.exclusions],
        }


@dataclass(frozen=True)
class ClaimDetection:
    """All affirmative matches for one dataset claim, from one or more rules."""
    claim_id: str
    item_id: str
    claim_text: str
    reason: str
    rule_ids: tuple[str, ...]
    rule_statuses: tuple[str, ...]
    feasibility: tuple[str, ...]
    production_eligible: bool
    occurrences: tuple[Assessment, ...]

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id, "item_id": self.item_id, "claim_text": self.claim_text,
            "reason": self.reason, "rule_ids": list(self.rule_ids),
            "rule_statuses": list(self.rule_statuses), "feasibility": list(self.feasibility),
            "production_eligible": self.production_eligible,
            "occurrence_count": len(self.occurrences),
            "occurrences": [o.to_dict() for o in self.occurrences],
        }


@dataclass(frozen=True)
class CompiledExclusion:
    name: str
    scope: str
    source: str
    regex: re.Pattern


@dataclass(frozen=True)
class CompiledRule:
    rule_id: str
    claim_id: str
    item_id: str
    claim_text: str
    reason: str
    status: str
    feasibility: str
    pattern: re.Pattern
    anchors: tuple[re.Pattern, ...]
    exclusions: tuple[CompiledExclusion, ...]
    eligible: bool = False
    ineligibility: tuple[str, ...] = ()
    order: int = 0


@dataclass(frozen=True)
class CompiledRegistry:
    source: str
    digest: str
    valid: bool
    errors: tuple[str, ...]
    rules: tuple[CompiledRule, ...]          # runnable draft and approved rules; empty if invalid
    approved_rule_ids: frozenset[str]
    source_text: str = ""                    # the registry text this was compiled from
    items: tuple[Item, ...] = ()             # snapshot of the dataset items used
    items_fingerprint: str = ""
    today: _dt.date | None = None            # review date used for the approval checks
    not_runnable: tuple[tuple[str, str], ...] = ()   # (rule_id, reason), e.g. drafts without a pattern
    snapshot_digest: str = ""                # sha256 over the full content of every snapshot item


@dataclass(frozen=True)
class MatchReport:
    item_id: str
    status: str
    status_reason: str
    registry_source: str
    registry_digest: str
    registry_valid: bool
    normalized_text: str | None
    production_rule_ids: tuple[str, ...]
    shadow_rule_ids: tuple[str, ...]
    ineligible_rules: tuple[tuple[str, tuple[str, ...]], ...]
    confirmed: tuple[ClaimDetection, ...]
    shadow: tuple[ClaimDetection, ...]
    assessments: tuple[Assessment, ...]
    not_runnable_rules: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict:
        def split(dets):
            return {
                "false": [d.claim_id for d in dets if d.reason == "false"],
                "unsupported": [d.claim_id for d in dets if d.reason == "unsupported"],
            }
        return {
            "item_id": self.item_id,
            "status": self.status,
            "status_reason": self.status_reason,
            "note": NO_MATCH_NOTE,
            "registry": {"source": self.registry_source, "digest": self.registry_digest,
                         "valid": self.registry_valid},
            "normalized_text": self.normalized_text,
            "offsets_refer_to": OFFSETS_REFER_TO,
            "production_rule_ids": list(self.production_rule_ids),
            "shadow_rule_ids": list(self.shadow_rule_ids),
            "ineligible_rules": [{"rule_id": rid, "reasons": list(rs)} for rid, rs in self.ineligible_rules],
            "not_runnable_rules": [{"rule_id": rid, "reason": why} for rid, why in self.not_runnable_rules],
            "confirmed": [d.to_dict() for d in self.confirmed],
            "confirmed_by_reason": split(self.confirmed),
            "shadow": [d.to_dict() for d in self.shadow],
            "shadow_by_reason": split(self.shadow),
            "assessments": [a.to_dict() for a in self.assessments],
        }


# ── Reader ───────────────────────────────────────────────────────────────────────────────────

def prepare_text(text: str) -> str:
    """Normalise, then blank double-quoted spans with spaces (same length)."""
    return _QUOTED.sub(lambda m: " " * len(m.group(0)), normalize_text(text))


def _stripped(text: str, start: int, end: int) -> Span | None:
    seg = text[start:end]
    if not seg.strip():
        return None
    lead = len(seg) - len(seg.lstrip())
    trail = len(seg) - len(seg.rstrip())
    return Span(start + lead, end - trail, text[start + lead:end - trail])


def split_sentences(text: str) -> list[Span]:
    """Sentence spans of an already prepared text (see prepare_text)."""
    spans: list[Span] = []
    start = 0
    for m in _END.finditer(text):
        if m.group(0).startswith("."):
            word = _WORD_BEFORE_DOT.search(text[start:m.start()])
            if word:
                w = word.group(1).lower().rstrip(".")
                if w in ABBREVIATIONS:
                    continue
                if w in NUMBERED_ABBREVIATIONS and _NUMBER_AFTER.match(text[m.end():]):
                    continue
        span = _stripped(text, start, m.end())
        if span:
            spans.append(span)
        start = m.end()
    span = _stripped(text, start, len(text))
    if span:
        spans.append(span)
    return spans


def split_clauses(sentence: Span | str) -> list[Span]:
    """Clause spans of a sentence. A str is treated as a sentence starting at offset 0."""
    if isinstance(sentence, str):
        sentence = Span(0, len(sentence), sentence)
    s = sentence.text
    out: list[Span] = []
    start = 0
    for m in _CLAUSE.finditer(s):
        if m.start() > start and s[start:m.start()].strip():
            out.append(Span(sentence.start + start, sentence.start + m.start(), s[start:m.start()]))
        start = m.end()
    if s[start:].strip():
        out.append(Span(sentence.start + start, sentence.end, s[start:]))
    return out


def parse_exclusion_name(exclusion: str) -> tuple[str, str]:
    """Return (name, scope) of an exclusion, or raise ValueError if it is not properly named."""
    m = _NAME_COMMENT.match(exclusion) if isinstance(exclusion, str) else None
    if not m:
        raise ValueError("exclusion must start with a name comment such as (?#negation)")
    parts = m.group(1).split(";")
    if len(parts) > 2:
        raise ValueError(f"exclusion name comment {m.group(0)!r} has more than one ';'")
    name = parts[0]
    scope = parts[1] if len(parts) == 2 else CLAUSE_SCOPE
    if not _NAME.match(name):
        raise ValueError(f"exclusion name {name!r} must be lowercase words joined by '-'")
    if scope not in EXCLUSION_SCOPES or (len(parts) == 2 and scope == CLAUSE_SCOPE):
        raise ValueError(f"exclusion scope {scope!r} is not allowed (only ';next-clause' may be given)")
    return name, scope


def _getter(rule: ClaimRule | Mapping[str, Any]):
    return (lambda k, d=None: getattr(rule, k, d)) if isinstance(rule, ClaimRule) else rule.get


def _exclusion_problems(rule: ClaimRule | Mapping[str, Any]) -> list[str]:
    """Problems with a rule's exclusion declarations (names, scopes, duplicates, empty matches).

    An exclusion that matches the empty string fires on any text that lacks whatever it is looking
    for, so it can silently disable a rule. It is rejected unless the rule has positive examples
    and, using the matcher's own clause-level semantics (claim scope and next-clause scope), it
    fires on none of the matches the rule finds in them. A zero-width requirement such as
    "(?#missing-2018)^(?!.*\\b2018\\b)" fires only when the claim clause lacks the year.
    Only this rule's own examples are used; nothing is cached or shared between rules.
    """
    get = _getter(rule)
    problems, names = [], set()
    positives = list(get("positive_examples", ()) or ())
    assessments: list[Assessment] | None = None  # this rule's matches on its own positives

    def positive_assessments() -> list[Assessment] | None:
        try:
            compiled = compile_rule(rule)  # low-level compile only: no registry, no eligibility
        except (ValueError, re.error):
            return None
        return [a for p in positives for a in _assess(compiled, normalize_text(p), prepare_text(p), False)]

    for ex in get("exclusions", ()) or ():
        try:
            name, _scope = parse_exclusion_name(ex)
            regex = re.compile(ex, re.I)
        except (ValueError, re.error) as e:
            problems.append(str(e))
            continue
        if name in names:
            problems.append(f"duplicate exclusion name {name!r}")
        names.add(name)
        if regex.search("") is not None:
            if not positives:
                problems.append(f"exclusion {name!r} matches the empty string and the rule has no positive "
                                f"examples to show it does not fire on every claim")
                continue
            if assessments is None:
                assessments = positive_assessments()
            if assessments is None:
                problems.append(f"exclusion {name!r} matches the empty string and cannot be checked because "
                                f"the rule has no usable pattern")
            elif not assessments:
                problems.append(f"exclusion {name!r} matches the empty string and no positive example "
                                f"produces a claim match to check it against")
            elif any(name in a.exclusion_names for a in assessments):
                problems.append(f"exclusion {name!r} matches the empty string and fires on a positive "
                                f"example's claim match; it would silently disable the rule")
    return problems


def compile_rule(rule: ClaimRule | Mapping[str, Any], *, order: int = 0) -> CompiledRule:
    """Compile one rule's regexes; raise ValueError on any naming or regex problem."""
    get = _getter(rule)
    exclusions, names = [], set()
    for ex in get("exclusions", ()) or ():
        name, scope = parse_exclusion_name(ex)
        if name in names:
            raise ValueError(f"duplicate exclusion name {name!r}")
        names.add(name)
        exclusions.append(CompiledExclusion(name, scope, ex, re.compile(ex, re.I)))
    pattern = get("pattern")
    if not isinstance(pattern, str) or not pattern.strip():
        raise ValueError("rule has no pattern")
    return CompiledRule(
        rule_id=get("rule_id", ""), claim_id=get("claim_id", ""), item_id=get("item_id", ""),
        claim_text=get("claim_text", ""), reason=get("reason", ""), status=get("status", ""),
        feasibility=get("feasibility", ""), pattern=re.compile(pattern, re.I),
        anchors=tuple(re.compile(a, re.I) for a in (get("anchors", ()) or ())),
        exclusions=tuple(exclusions), order=order,
    )


def _assess(rule: CompiledRule, normalized: str, prepared: str, eligible: bool) -> list[Assessment]:
    out: list[Assessment] = []
    for sent in split_sentences(prepared):
        clauses = split_clauses(sent)
        s = sent.text
        for m in rule.pattern.finditer(s):
            m_start, m_end = sent.start + m.start(), sent.start + m.end()
            failed = tuple(i for i, a in enumerate(rule.anchors) if not a.search(s))
            touched = [i for i, c in enumerate(clauses) if c.start < max(m_end, m_start + 1) and c.end > m_start]
            if not touched:  # match lies only in boundary characters: use the clause before it
                before = [i for i, c in enumerate(clauses) if c.start <= m_start]
                touched = [before[-1] if before else 0]
            first, last = clauses[touched[0]], clauses[touched[-1]]
            claim = Span(first.start, last.end, normalized[first.start:last.end])
            claim_prepared = prepared[first.start:last.end]
            nxt = clauses[touched[-1] + 1] if touched[-1] + 1 < len(clauses) else None
            hits = []
            for ex in rule.exclusions:
                if ex.scope == NEXT_CLAUSE_SCOPE:
                    if nxt is None:
                        continue
                    target, base = prepared[nxt.start:nxt.end], nxt.start
                else:
                    target, base = claim_prepared, claim.start
                h = ex.regex.search(target)
                if h:
                    hits.append(ExclusionHit(ex.name, ex.scope,
                                             Span(base + h.start(), base + h.end(),
                                                  normalized[base + h.start():base + h.end()])))
            outcome = NO_ANCHOR if failed else (EXCLUDED if hits else AFFIRMATIVE)
            out.append(Assessment(
                rule_id=rule.rule_id, claim_id=rule.claim_id, item_id=rule.item_id, reason=rule.reason,
                rule_status=rule.status, production_eligible=eligible, outcome=outcome,
                match=Span(m_start, m_end, normalized[m_start:m_end]),
                sentence=Span(sent.start, sent.end, normalized[sent.start:sent.end]),
                clause=claim,
                touched_clauses=tuple(Span(clauses[i].start, clauses[i].end,
                                           normalized[clauses[i].start:clauses[i].end]) for i in touched),
                next_clause=Span(nxt.start, nxt.end, normalized[nxt.start:nxt.end]) if nxt else None,
                failed_anchors=failed, exclusions=tuple(hits),
            ))
    return out


def assess_rule(rule: CompiledRule | ClaimRule | Mapping[str, Any], text: str) -> list[Assessment]:
    """Assess every match of one rule in a text. Diagnostic: never marks anything production-eligible."""
    if not isinstance(text, str):
        raise TypeError(f"text must be str, not {type(text).__name__}")
    compiled = rule if isinstance(rule, CompiledRule) else compile_rule(rule)
    return _assess(compiled, normalize_text(text), prepare_text(text), eligible=False)


# ── Registry compilation and eligibility ─────────────────────────────────────────────────────

def registry_digest(text: str) -> str:
    """sha256 of the registry text with line endings normalised to LF (CRLF and LF agree)."""
    return hashlib.sha256(text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")).hexdigest()


def _example_problems(rule: CompiledRule, positives: Sequence[str], negatives: Sequence[str],
                      item: Item | None) -> list[str]:
    problems = []

    def affirmative(text):
        return any(a.is_affirmative for a in _assess(rule, normalize_text(text), prepare_text(text), False))

    for i, p in enumerate(positives):
        if not affirmative(p):
            problems.append(f"positive example {i + 1} is not an affirmative match")
    for i, n in enumerate(negatives):
        if affirmative(n):
            problems.append(f"negative example {i + 1} is an affirmative match")
    if item is None:
        problems.append("item not found in dataset")
    else:
        if affirmative(item.query):
            problems.append("matches its own item's question")
        if affirmative(item.reference_answer):
            problems.append("matches its own item's reference answer")
    return problems


def _items_fingerprint(items: Sequence[Item]) -> str:
    """sha256 over every item field that compilation and eligibility depend on."""
    parts = [(it.id, it.expected_answer_language, it.query, it.reference_answer,
              tuple((c.text, c.reason) for c in it.forbidden_claims), tuple(it.must_not)) for it in items]
    return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()


def _plain(value: Any) -> Any:
    """A type-tagged, order-stable plain form of a snapshot value, for _snapshot_digest."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        # every field, including those excluded from ==/repr (e.g. Item.raw)
        return ["dataclass", type(value).__name__,
                [[f.name, _plain(getattr(value, f.name))] for f in dataclasses.fields(value)]]
    if isinstance(value, dict):
        pairs = [[_plain(k), _plain(v)] for k, v in value.items()]
        return ["dict", sorted(pairs, key=lambda p: json.dumps(p[0], ensure_ascii=False))]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [_plain(v) for v in value]]
    if isinstance(value, (set, frozenset)):
        return [type(value).__name__, sorted((_plain(v) for v in value),
                                             key=lambda p: json.dumps(p, ensure_ascii=False))]
    if value is None or isinstance(value, (str, int, float, bool)):
        return [type(value).__name__, value]
    return [type(value).__name__, repr(value)]


def _snapshot_digest(items: Sequence[Item]) -> str:
    """sha256 over the FULL content of every snapshot item (all fields, nested values included).

    Unlike _items_fingerprint (the fields eligibility depends on), any change to any item field
    changes this digest, so an in-place edit of a returned snapshot is detected."""
    text = json.dumps([_plain(it) for it in items], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


# Private canonical compilations. Callers only ever receive deep copies (see _compile), so no
# reference to a cached object, its rules or its dataset snapshot escapes through the API.
_AUTHORITATIVE_MAX = 32
_AUTHORITATIVE: "dict[tuple, CompiledRegistry]" = {}


def _raw_hash(text: str) -> str:
    """sha256 of the exact source text (line endings included), unlike registry_digest."""
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _cache_key(compiled: CompiledRegistry) -> tuple:
    return (compiled.digest, _raw_hash(compiled.source_text), compiled.source, compiled.items_fingerprint,
            compiled.snapshot_digest, compiled.today)


def _remember(compiled: CompiledRegistry) -> None:
    key = _cache_key(compiled)
    _AUTHORITATIVE.pop(key, None)
    _AUTHORITATIVE[key] = compiled
    while len(_AUTHORITATIVE) > _AUTHORITATIVE_MAX:
        _AUTHORITATIVE.pop(next(iter(_AUTHORITATIVE)))


def _authoritative(compiled: CompiledRegistry) -> CompiledRegistry | None:
    """The private canonical compilation of compiled.source_text, or None if digest/dataset don't match.

    The result is for internal comparison and matching only; it must never be returned to a caller."""
    if not isinstance(compiled.source_text, str) or registry_digest(compiled.source_text) != compiled.digest:
        return None
    items = tuple(compiled.items)
    if _items_fingerprint(items) != compiled.items_fingerprint or not isinstance(compiled.today, _dt.date):
        return None
    if _snapshot_digest(items) != compiled.snapshot_digest:  # any in-place edit of the snapshot
        return None
    if compiled.today > _dt.date.today():  # Q5: a future compilation date is never authoritative
        return None
    key = _cache_key(compiled)
    if key not in _AUTHORITATIVE:
        _compile(compiled.source_text, items, source=compiled.source, today=compiled.today)
    return _AUTHORITATIVE.get(key)


def compile_registry_text(text: str, items: Sequence[Item], *, source: str = "registry",
                          today: _dt.date | None = None) -> CompiledRegistry:
    """Parse, validate and compile a registry. Any error makes the whole registry invalid."""
    return _compile(text, items, source=source, today=today or _dt.date.today())


def _compile(text: str, items: Sequence[Item], *, source: str, today: _dt.date) -> CompiledRegistry:
    digest = registry_digest(text)
    snapshot = tuple(copy.deepcopy(list(items)))
    fingerprint = _items_fingerprint(snapshot)
    snap_digest = _snapshot_digest(snapshot)

    def done(registry: CompiledRegistry) -> CompiledRegistry:
        # Cache the canonical object privately; the caller gets an independent deep copy.
        _remember(registry)
        return copy.deepcopy(registry)

    def invalid(errors) -> CompiledRegistry:
        return done(CompiledRegistry(source, digest, False, tuple(errors), (), frozenset(), text, snapshot,
                                     fingerprint, today, (), snap_digest))

    if today > _dt.date.today():
        return invalid([f"{source}: compilation date {today.isoformat()} is later than the current date "
                        f"{_dt.date.today().isoformat()}; a future date cannot validate review dates"])
    reg = parse_registry(text.splitlines(), snapshot, source=source, today=today)
    errors = list(reg.errors)
    for rule in reg.rules:  # every record, including rejected and retired ones (F7)
        errors += [f"{source}:{rule.line} [{rule.rule_id}]: {p}" for p in _exclusion_problems(rule)]
    compiled: list[CompiledRule] = []
    not_runnable: list[tuple[str, str]] = []
    by_id = {it.id: it for it in snapshot}
    raw_rules = []
    for rule in reg.rules:
        if rule.status not in ("draft", "approved"):
            continue
        if rule.status == "draft" and rule.pattern is None:
            not_runnable.append((rule.rule_id, "draft has no pattern"))
            continue
        raw_rules.append(rule)
    for order, rule in enumerate(raw_rules):
        try:
            compiled.append(compile_rule(rule, order=order))
        except (ValueError, re.error) as e:
            errors.append(f"{source}:{rule.line} [{rule.rule_id}]: {e}")
    if errors:
        return invalid(errors)

    approved_ids = frozenset(r.rule_id for r in reg.approved_rules())
    final = []
    for c, rule in zip(compiled, raw_rules):
        if c.status != "approved":
            final.append(c)
            continue
        reasons = []
        if c.rule_id not in approved_ids:
            reasons.append("not returned by the registry as approved")
        reasons += _example_problems(c, rule.positive_examples, rule.negative_examples, by_id.get(c.item_id))
        final.append(CompiledRule(**{**c.__dict__, "eligible": not reasons, "ineligibility": tuple(reasons)}))
    return done(CompiledRegistry(source, digest, True, (), tuple(final), approved_ids, text, snapshot,
                                 fingerprint, today, tuple(not_runnable), snap_digest))


def load_compiled_registry(items: Sequence[Item], path: Path | str = REGISTRY_PATH, *,
                           today: _dt.date | None = None) -> CompiledRegistry:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        return CompiledRegistry(path.name, "", False, (f"{path.name}: cannot read registry ({e})",), (),
                                frozenset(), "", (), "", today, ())
    return compile_registry_text(text, items, source=path.name, today=today)


# ── Matching ─────────────────────────────────────────────────────────────────────────────────

def _claim_mismatch(rule: CompiledRule, item: Item) -> str | None:
    """Recheck, against the item being evaluated, that the rule still names the same claim."""
    if rule.item_id != item.id:
        return "rule belongs to another item"
    try:
        n = int(rule.claim_id.split("#c", 1)[1])
    except (IndexError, ValueError):
        return "malformed claim id"
    if not 1 <= n <= len(item.forbidden_claims):
        return "claim position no longer exists on the item"
    claim = item.forbidden_claims[n - 1]
    if claim.text != rule.claim_text or claim.reason != rule.reason:
        return "claim text or reason differs from the item"
    return None


def _rules_for_item(compiled: CompiledRegistry, item: Item) -> list[CompiledRule]:
    """Runnable rules that belong to this item (rules never run on another item's response)."""
    return [r for r in compiled.rules if r.item_id == item.id and r.status in ("approved", "draft")]


def _production_eligible(rule: CompiledRule, compiled: CompiledRegistry, item: Item) -> bool:
    return (compiled.valid and rule.status == "approved" and rule.eligible and not rule.ineligibility
            and rule.rule_id in compiled.approved_rule_ids and _claim_mismatch(rule, item) is None)


def _aggregate(assessments: Iterable[Assessment], rules: Mapping[str, CompiledRule],
               eligible: bool) -> tuple[ClaimDetection, ...]:
    groups: dict[str, list[Assessment]] = {}
    for a in assessments:
        if a.is_affirmative:
            groups.setdefault(a.claim_id, []).append(a)
    out = []
    for claim_id, occ in groups.items():
        occ = sorted(occ, key=lambda a: (a.match.start, rules[a.rule_id].order, a.match.end))
        rule_ids = tuple(dict.fromkeys(a.rule_id for a in sorted(occ, key=lambda a: rules[a.rule_id].order)))
        first = rules[rule_ids[0]]
        out.append(ClaimDetection(
            claim_id=claim_id, item_id=first.item_id, claim_text=first.claim_text, reason=first.reason,
            rule_ids=rule_ids, rule_statuses=tuple(rules[r].status for r in rule_ids),
            feasibility=tuple(rules[r].feasibility for r in rule_ids), production_eligible=eligible,
            occurrences=tuple(occ),
        ))
    return tuple(sorted(out, key=lambda d: int(d.claim_id.split("#c", 1)[1])))


def match_response(item: Item, response: str | None, compiled: CompiledRegistry, *,
                   include_drafts: bool = False) -> MatchReport:
    """Match one response against the item's rules. Only eligible approved rules can confirm."""
    if not isinstance(compiled, CompiledRegistry):
        raise TypeError("compiled must be a CompiledRegistry (use compile_registry_text)")
    if response is not None and not isinstance(response, str):
        raise TypeError(f"response must be str or None, not {type(response).__name__}")

    def report(status, reason, *, valid=None, text=None, prod=(), shadow_ids=(), inelig=(), confirmed=(),
               shadow=(), assessments=(), not_runnable=()):
        return MatchReport(item.id, status, reason, compiled.source, compiled.digest,
                           compiled.valid if valid is None else valid, text, tuple(prod), tuple(shadow_ids),
                           tuple(inelig), tuple(confirmed), tuple(shadow), tuple(assessments),
                           tuple(not_runnable))

    if response is None:
        return report(NOT_EVALUATED, "no response supplied")
    # Bind to the authoritative compilation of the registry source (F1). Nothing supplied by the
    # caller (rules, flags, approved ids) is trusted unless it is identical to that compilation.
    authoritative = _authoritative(compiled)
    if authoritative is None:
        return report(NOT_EVALUATED, "compiled registry does not match its source text or dataset snapshot; "
                                     "nothing evaluated", valid=False)
    if compiled != authoritative:
        return report(NOT_EVALUATED, "compiled registry differs from the authoritative compilation of its "
                                     "source; nothing evaluated", valid=False)
    compiled = authoritative
    if not compiled.valid:
        return report(NOT_EVALUATED, f"registry invalid ({len(compiled.errors)} error(s)); nothing evaluated")
    # Q4: the item being evaluated must be exactly the item the registry was compiled against.
    snapshot_item = next((it for it in compiled.items if it.id == item.id), None)
    if snapshot_item is None:
        return report(NOT_EVALUATED, f"item {item.id!r} is not in the compiled dataset snapshot; nothing evaluated")
    if snapshot_item != item:
        return report(NOT_EVALUATED, f"item {item.id!r} differs from the compiled dataset snapshot; "
                                     f"nothing evaluated")

    item_rules = _rules_for_item(compiled, item)
    item_not_runnable = [(rid, why) for rid, why in compiled.not_runnable if rid.split("#", 1)[0] == item.id]
    production, shadow_rules, ineligible = [], [], []
    for r in item_rules:
        # The eligibility check is the only gate to production, whatever the rule claims to be.
        if _production_eligible(r, compiled, item):
            production.append(r)
        elif r.status == "approved":
            mismatch = _claim_mismatch(r, item)
            ineligible.append((r.rule_id, tuple(r.ineligibility) + ((mismatch,) if mismatch else ())
                               or ("not production-eligible",)))
        elif r.status == "draft" and include_drafts and _claim_mismatch(r, item) is None:
            shadow_rules.append(r)

    normalized, prepared = normalize_text(response), prepare_text(response)
    by_id = {r.rule_id: r for r in compiled.rules}
    prod_assess = [a for r in production for a in _assess(r, normalized, prepared, eligible=True)]
    shadow_assess = [a for r in shadow_rules for a in _assess(r, normalized, prepared, eligible=False)]
    confirmed = _aggregate(prod_assess, by_id, eligible=True)
    shadow = _aggregate(shadow_assess, by_id, eligible=False)

    if not production:
        status, reason = NOT_EVALUATED, "no production-eligible rule for this item"
    elif confirmed:
        status, reason = AFFIRMATIVE_MATCH, f"{len(confirmed)} claim(s) matched affirmatively by eligible rules"
    else:
        status, reason = NO_AFFIRMATIVE_MATCH, (f"{len(production)} eligible rule(s) ran; none matched "
                                                f"affirmatively (not a verdict on the answer)")
    return report(status, reason, text=normalized, prod=[r.rule_id for r in production],
                  shadow_ids=[r.rule_id for r in shadow_rules], inelig=ineligible, confirmed=confirmed,
                  shadow=shadow, assessments=prod_assess + shadow_assess, not_runnable=item_not_runnable)
