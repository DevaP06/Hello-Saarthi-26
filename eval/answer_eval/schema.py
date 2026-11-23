"""Dataset schema for the answer-evaluation set (eval/answer_eval/dataset.jsonl).

Pure standard library, so it can be imported and tested without the backend installed.

    from eval.answer_eval.schema import load_dataset, validate_dataset_items
    items, errors = load_dataset(DATASET_PATH)
    errors += validate_dataset_items(items)

Checks done here (structure only). The evidence checks against the PDF text live in
validate_dataset.py, because they need pypdf.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Enums ────────────────────────────────────────────────────────────────────

CATEGORIES = (
    "direct_factual",
    "conceptual",
    "multi_hop",
    "comparison",
    "ambiguous",
    "insufficient_context",
    "multilingual",
)

# Composition of the active set after the Sprint 1 correction pass (2026-11-23): the Hindi and
# Marathi items were archived (archive/), leaving one Hinglish item in "multilingual".
EXPECTED_DISTRIBUTION = {
    "direct_factual": 8,
    "conceptual": 7,
    "multi_hop": 6,
    "comparison": 5,
    "ambiguous": 4,
    "insufficient_context": 6,
    "multilingual": 1,
}
EXPECTED_BEHAVIOR_COUNTS = {"answer": 27, "abstain": 6, "clarify": 4}
EXPECTED_TOTAL = 37

BEHAVIORS = ("answer", "abstain", "clarify")

# The behaviour each category must have; multilingual items may answer or abstain.
CATEGORY_BEHAVIOR = {
    "direct_factual": {"answer"},
    "conceptual": {"answer"},
    "multi_hop": {"answer"},
    "comparison": {"answer"},
    "ambiguous": {"clarify"},
    "insufficient_context": {"abstain"},
    "multilingual": {"answer", "abstain"},
}

# Active-set languages. Hindi and Marathi are out of scope for now (see archive/README.md).
LANGUAGES = ("en", "hinglish")

# Why a statement must not be asserted:
#   false        factually wrong
#   unsupported  possibly true, but not supported by the course materials (grounding failure, not a factual error)
CLAIM_REASONS = ("false", "unsupported")

# A forbidden claim is a statement about the subject. Text that starts like an instruction about the
# assistant's behaviour belongs in `must_not` instead, so a factual-claim checker never sees it.
_BEHAVIOUR_OPENERS = re.compile(
    r"^(answers?|asks?|declares?|gives?|provides?|refuses?|says?|states?|lists?|uses?|substitutes?)\b", re.I
)
DIFFICULTIES = ("recall", "application", "analysis")
LABEL_STATUSES = ("draft", "reviewed")
TAGS = (
    "paraphrased",   # wording shares few content words with the source page
    "numeric",       # the answer is a number that a regex can check in any language
    "formula",       # the answer is a formula
    "cross_page",    # evidence on two or more pages of one module
    "cross_module",  # evidence in two or more modules
    "near_miss",     # in-domain but absent; the lexical coverage gate is expected to pass it
    "off_topic",     # nothing to do with the course
    "code",          # evidence is Python code from Module 4
    "false_premise",
    "followup",
)

CORPUS = "ee210"

# Material id = the full module title (what citation labels print); value = PDF file name
# in eval/synthetic/course/ (ee210_course.write_pdfs: title.replace(" ", "_") + ".pdf").
MATERIALS = {
    "EE210 Module 1 Signals and Sampling": "EE210_Module_1_Signals_and_Sampling.pdf",
    "EE210 Module 2 LTI Systems and Convolution": "EE210_Module_2_LTI_Systems_and_Convolution.pdf",
    "EE210 Module 3 Fourier Analysis DFT and Filters": "EE210_Module_3_Fourier_Analysis_DFT_and_Filters.pdf",
    "EE210 Module 4 DSP in Python Solved Exercises": "EE210_Module_4_DSP_in_Python_Solved_Exercises.pdf",
}
PAGES_PER_MATERIAL = 4
MIN_QUOTE_CHARS = 15  # after removing whitespace; stops trivially short "evidence"

_ID_RE = re.compile(r"^[a-z]{2}-\d{2}$")

REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "id": str,
    "category": str,
    "tags": list,
    "language": str,
    "expected_answer_language": str,  # recorded for future language-compliance scoring; not scored yet
    "paired_with": (str, type(None)),
    "difficulty": str,
    "query": str,
    "history": list,
    "corpus": str,
    "scope": dict,
    "expected_behavior": str,
    "reference_answer": str,
    "key_points": list,
    "forbidden_claims": list,   # [{"text": str, "reason": "false" | "unsupported"}]
    "must_not": list,           # behavioural rubric: things the reply must not do (not factual claims)
    "evidence": list,
    "expected_sources": list,
    "absent_terms": list,
    "acceptable_interpretations": list,
    "label_status": str,
    "author": str,
    "reviewer": (str, type(None)),
    "provenance": str,
    "notes": str,
}
OPTIONAL_FIELDS = {"original_query": str}


# ── Records ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Evidence:
    id: str
    material: str
    page: int
    quote: str


@dataclass(frozen=True)
class Claim:
    text: str
    reason: str  # one of CLAIM_REASONS


@dataclass(frozen=True)
class KeyPoint:
    id: str
    text: str
    required: bool
    evidence_ids: tuple[str, ...]
    match: str | None = None  # optional regex for a deterministic check


@dataclass
class Item:
    id: str
    category: str
    tags: list[str]
    language: str
    expected_answer_language: str
    paired_with: str | None
    difficulty: str
    query: str
    history: list[dict]
    corpus: str
    scope: dict
    expected_behavior: str
    reference_answer: str
    key_points: list[KeyPoint]
    forbidden_claims: list[Claim]
    must_not: list[str]
    evidence: list[Evidence]
    expected_sources: list[dict]
    absent_terms: list[str]
    acceptable_interpretations: list[str]
    label_status: str
    author: str
    reviewer: str | None
    provenance: str
    notes: str
    original_query: str | None = None
    raw: dict = field(default_factory=dict, repr=False, compare=False)


# ── Parsing (one item) ───────────────────────────────────────────────────────

def _type_name(t) -> str:
    if isinstance(t, tuple):
        return " or ".join("null" if x is type(None) else x.__name__ for x in t)
    return t.__name__


def parse_item(raw: Any, where: str = "item") -> tuple[Item | None, list[str]]:
    """Check one JSON object's fields and types and build an Item. Returns (item or None, errors)."""
    errors: list[str] = []
    if not isinstance(raw, dict):
        return None, [f"{where}: not a JSON object"]
    where = f"{where} ({raw.get('id', '?')})"

    for name, typ in REQUIRED_FIELDS.items():
        if name not in raw:
            errors.append(f"{where}: missing field '{name}'")
        elif not isinstance(raw[name], typ) or (typ is str and isinstance(raw[name], bool)):
            errors.append(f"{where}: field '{name}' must be {_type_name(typ)}")
    for name in raw:
        if name not in REQUIRED_FIELDS and name not in OPTIONAL_FIELDS:
            errors.append(f"{where}: unknown field '{name}'")
    if "original_query" in raw and not isinstance(raw["original_query"], str):
        errors.append(f"{where}: field 'original_query' must be str")
    if errors:
        return None, errors

    key_points: list[KeyPoint] = []
    for i, kp in enumerate(raw["key_points"]):
        w = f"{where} key_points[{i}]"
        if not isinstance(kp, dict):
            errors.append(f"{w}: not an object")
            continue
        missing = [k for k in ("id", "text", "required", "evidence_ids") if k not in kp]
        if missing:
            errors.append(f"{w}: missing {', '.join(missing)}")
            continue
        extra = set(kp) - {"id", "text", "required", "evidence_ids", "match"}
        if extra:
            errors.append(f"{w}: unknown field(s) {', '.join(sorted(extra))}")
        if not isinstance(kp["id"], str) or not isinstance(kp["text"], str) or not kp["text"].strip():
            errors.append(f"{w}: 'id' and 'text' must be non-empty strings")
        if not isinstance(kp["required"], bool):
            errors.append(f"{w}: 'required' must be true or false")
        if not isinstance(kp["evidence_ids"], list) or not all(isinstance(e, str) for e in kp["evidence_ids"]):
            errors.append(f"{w}: 'evidence_ids' must be a list of strings")
        match = kp.get("match")
        if match is not None:
            if not isinstance(match, str) or not match:
                errors.append(f"{w}: 'match' must be a non-empty regex string")
                match = None
            else:
                try:
                    re.compile(match)
                except re.error as e:
                    errors.append(f"{w}: 'match' is not a valid regex ({e})")
                    match = None
        key_points.append(KeyPoint(str(kp["id"]), str(kp["text"]), bool(kp["required"]),
                                   tuple(kp["evidence_ids"]) if isinstance(kp["evidence_ids"], list) else (),
                                   match))

    evidence: list[Evidence] = []
    for i, ev in enumerate(raw["evidence"]):
        w = f"{where} evidence[{i}]"
        if not isinstance(ev, dict) or set(ev) != {"id", "material", "page", "quote"}:
            errors.append(f"{w}: must have exactly id, material, page, quote")
            continue
        if not isinstance(ev["page"], int) or isinstance(ev["page"], bool):
            errors.append(f"{w}: 'page' must be an integer")
            continue
        if not all(isinstance(ev[k], str) for k in ("id", "material", "quote")):
            errors.append(f"{w}: 'id', 'material' and 'quote' must be strings")
            continue
        evidence.append(Evidence(ev["id"], ev["material"], ev["page"], ev["quote"]))

    claims: list[Claim] = []
    for i, c in enumerate(raw["forbidden_claims"]):
        w = f"{where} forbidden_claims[{i}]"
        if not isinstance(c, dict) or set(c) != {"text", "reason"}:
            errors.append(f"{w}: must be an object with exactly 'text' and 'reason'")
            continue
        if not isinstance(c["text"], str) or not c["text"].strip():
            errors.append(f"{w}: 'text' must be a non-empty string")
            continue
        if c["reason"] not in CLAIM_REASONS:
            errors.append(f"{w}: 'reason' must be one of {', '.join(CLAIM_REASONS)}")
            continue
        claims.append(Claim(c["text"], c["reason"]))

    for i, src in enumerate(raw["expected_sources"]):
        if not isinstance(src, dict) or set(src) != {"material", "page"}:
            errors.append(f"{where} expected_sources[{i}]: must have exactly material, page")
    for name in ("must_not", "absent_terms", "acceptable_interpretations", "tags"):
        if not all(isinstance(x, str) and x.strip() for x in raw[name]):
            errors.append(f"{where}: '{name}' must contain non-empty strings only")
    for i, turn in enumerate(raw["history"]):
        if not isinstance(turn, dict) or turn.get("role") not in ("user", "assistant") or not isinstance(turn.get("content"), str):
            errors.append(f"{where} history[{i}]: needs role user|assistant and string content")
    if set(raw["scope"]) - {"material_title"}:
        errors.append(f"{where}: scope may only contain 'material_title'")

    if errors:
        return None, errors
    item = Item(
        id=raw["id"], category=raw["category"], tags=list(raw["tags"]), language=raw["language"],
        expected_answer_language=raw["expected_answer_language"], paired_with=raw["paired_with"], difficulty=raw["difficulty"], query=raw["query"],
        history=list(raw["history"]), corpus=raw["corpus"], scope=dict(raw["scope"]),
        expected_behavior=raw["expected_behavior"], reference_answer=raw["reference_answer"],
        key_points=key_points, forbidden_claims=claims, must_not=list(raw["must_not"]), evidence=evidence,
        expected_sources=list(raw["expected_sources"]), absent_terms=list(raw["absent_terms"]),
        acceptable_interpretations=list(raw["acceptable_interpretations"]),
        label_status=raw["label_status"], author=raw["author"], reviewer=raw["reviewer"],
        provenance=raw["provenance"], notes=raw["notes"], original_query=raw.get("original_query"), raw=raw,
    )
    return item, []


# ── Rules for one item ───────────────────────────────────────────────────────

def _sources_from_evidence(evidence: list[Evidence]) -> list[dict]:
    seen = sorted({(e.material, e.page) for e in evidence}, key=lambda mp: (list(MATERIALS).index(mp[0]) if mp[0] in MATERIALS else 99, mp[1]))
    return [{"material": m, "page": p} for m, p in seen]


def check_item(item: Item) -> list[str]:
    """Enums and the rules that depend on category / expected behaviour."""
    w = f"item {item.id}"
    errors: list[str] = []

    if not _ID_RE.match(item.id):
        errors.append(f"{w}: id must look like 'df-01'")
    if item.category not in CATEGORIES:
        errors.append(f"{w}: unknown category '{item.category}'")
    if item.expected_behavior not in BEHAVIORS:
        errors.append(f"{w}: unknown expected_behavior '{item.expected_behavior}'")
    elif item.category in CATEGORY_BEHAVIOR and item.expected_behavior not in CATEGORY_BEHAVIOR[item.category]:
        errors.append(f"{w}: category '{item.category}' cannot have expected_behavior '{item.expected_behavior}'")
    if item.language not in LANGUAGES:
        errors.append(f"{w}: language '{item.language}' is not in the active set ({', '.join(LANGUAGES)})")
    if item.expected_answer_language not in LANGUAGES:
        errors.append(f"{w}: expected_answer_language '{item.expected_answer_language}' is not in the active set")
    elif item.expected_answer_language != item.language:
        errors.append(f"{w}: expected_answer_language must match the question language '{item.language}'")
    for c in item.forbidden_claims:
        if _BEHAVIOUR_OPENERS.match(c.text.strip()):
            errors.append(f"{w}: forbidden claim {c.text!r} reads as a behaviour; put it in must_not")
    if item.difficulty not in DIFFICULTIES:
        errors.append(f"{w}: unknown difficulty '{item.difficulty}'")
    if item.label_status not in LABEL_STATUSES:
        errors.append(f"{w}: unknown label_status '{item.label_status}'")
    if item.label_status == "reviewed" and not item.reviewer:
        errors.append(f"{w}: a reviewed item needs a reviewer")
    if item.label_status == "draft" and item.reviewer:
        errors.append(f"{w}: a draft item must not name a reviewer yet")
    if item.corpus != CORPUS:
        errors.append(f"{w}: corpus must be '{CORPUS}'")
    for t in item.tags:
        if t not in TAGS:
            errors.append(f"{w}: unknown tag '{t}'")
    if len(set(item.tags)) != len(item.tags):
        errors.append(f"{w}: duplicate tags")
    for name in ("query", "author", "provenance"):
        if not getattr(item, name).strip():
            errors.append(f"{w}: '{name}' must not be empty")
    if "paraphrased" in item.tags and not item.original_query:
        errors.append(f"{w}: a paraphrased item records the audit wording in 'original_query'")

    scope_title = item.scope.get("material_title")
    if scope_title is not None and scope_title not in MATERIALS:
        errors.append(f"{w}: scope material_title '{scope_title}' is not a known material")

    # Language rules
    if item.category == "multilingual":
        if item.language == "en":
            errors.append(f"{w}: a multilingual item must not be in English")
        if not item.paired_with:
            errors.append(f"{w}: a multilingual item needs paired_with (its English twin)")
    else:
        if item.language != "en":
            errors.append(f"{w}: only multilingual items may use language '{item.language}'")
        if item.paired_with:
            errors.append(f"{w}: only multilingual items have paired_with")

    # Evidence and key points
    ev_ids = [e.id for e in item.evidence]
    if len(set(ev_ids)) != len(ev_ids):
        errors.append(f"{w}: duplicate evidence ids")
    kp_ids = [k.id for k in item.key_points]
    if len(set(kp_ids)) != len(kp_ids):
        errors.append(f"{w}: duplicate key point ids")
    for e in item.evidence:
        if e.material not in MATERIALS:
            errors.append(f"{w} evidence {e.id}: unknown material '{e.material}' (use the full module title)")
        if not 1 <= e.page <= PAGES_PER_MATERIAL:
            errors.append(f"{w} evidence {e.id}: page {e.page} out of range 1..{PAGES_PER_MATERIAL}")
        if len(re.sub(r"\s+", "", e.quote)) < MIN_QUOTE_CHARS:
            errors.append(f"{w} evidence {e.id}: quote shorter than {MIN_QUOTE_CHARS} characters")
    for k in item.key_points:
        for ref in k.evidence_ids:
            if ref not in ev_ids:
                errors.append(f"{w} key point {k.id}: unknown evidence id '{ref}'")
        if k.match and item.reference_answer and not re.search(k.match, item.reference_answer, re.I):
            errors.append(f"{w} key point {k.id}: 'match' regex does not match the item's own reference_answer")

    b = item.expected_behavior
    if b == "answer":
        if not item.reference_answer.strip():
            errors.append(f"{w}: an answer item needs a reference_answer")
        if not item.key_points or not any(k.required for k in item.key_points):
            errors.append(f"{w}: an answer item needs at least one required key point")
        if not item.evidence:
            errors.append(f"{w}: an answer item needs evidence")
        if item.absent_terms:
            errors.append(f"{w}: an answer item must not have absent_terms")
        if item.acceptable_interpretations:
            errors.append(f"{w}: only clarify items have acceptable_interpretations")
        for k in item.key_points:
            if not k.evidence_ids:
                errors.append(f"{w} key point {k.id}: answer key points must cite evidence")
        cited = {ref for k in item.key_points for ref in k.evidence_ids}
        for e in item.evidence:
            if e.id not in cited:
                errors.append(f"{w} evidence {e.id}: not used by any key point")
        if scope_title and any(e.material != scope_title for e in item.evidence):
            errors.append(f"{w}: scope material_title differs from an evidence material")
        if item.category == "multi_hop" and len(item.evidence) < 2:
            errors.append(f"{w}: a multi_hop item needs at least two evidence passages")
        if item.language != "en":
            # Multilingual constraint: required key points are checkable in any reply language.
            for k in item.key_points:
                if k.required and not k.match:
                    errors.append(f"{w} key point {k.id}: non-English answer items need a 'match' regex on every required key point")
    elif b == "abstain":
        if not item.absent_terms:
            errors.append(f"{w}: an abstain item needs absent_terms (proof the corpus cannot answer it)")
        if item.evidence or item.key_points or item.expected_sources:
            errors.append(f"{w}: an abstain item has no evidence, key points or expected_sources")
        if item.acceptable_interpretations:
            errors.append(f"{w}: only clarify items have acceptable_interpretations")
    elif b == "clarify":
        if len(item.acceptable_interpretations) < 2:
            errors.append(f"{w}: a clarify item needs at least two acceptable_interpretations")
        if not item.evidence:
            errors.append(f"{w}: a clarify item needs evidence for its interpretations")
        if item.absent_terms:
            errors.append(f"{w}: a clarify item must not have absent_terms")

    if b in ("answer", "clarify") and item.expected_sources != _sources_from_evidence(item.evidence):
        errors.append(f"{w}: expected_sources must list exactly the (material, page) pairs of its evidence, in module/page order")

    # Tags that can be derived must agree with the evidence.
    pages = {(e.material, e.page) for e in item.evidence}
    mats = {e.material for e in item.evidence}
    if ("cross_module" in item.tags) != (len(mats) > 1):
        errors.append(f"{w}: tag 'cross_module' must be present exactly when evidence spans modules")
    if ("cross_page" in item.tags) != (len(mats) == 1 and len(pages) > 1):
        errors.append(f"{w}: tag 'cross_page' must be present exactly when evidence spans pages of one module")
    return errors


# ── Rules across the dataset ─────────────────────────────────────────────────

def validate_dataset_items(items: list[Item], *, expected_distribution: dict[str, int] | None = EXPECTED_DISTRIBUTION,
                           expected_behaviors: dict[str, int] | None = EXPECTED_BEHAVIOR_COUNTS,
                           expected_total: int | None = EXPECTED_TOTAL,
                           paraphrased_range: tuple[int, int] | None = (6, 8)) -> list[str]:
    errors: list[str] = []
    for item in items:
        errors += check_item(item)

    counts = Counter(i.id for i in items)
    for iid, n in counts.items():
        if n > 1:
            errors.append(f"dataset: id '{iid}' is used {n} times")

    by_id = {i.id: i for i in items}
    for item in items:
        if item.paired_with:
            twin = by_id.get(item.paired_with)
            if twin is None:
                errors.append(f"item {item.id}: paired_with '{item.paired_with}' does not exist")
            elif twin.language != "en":
                errors.append(f"item {item.id}: paired_with '{twin.id}' must be an English item")
            elif twin.expected_behavior != item.expected_behavior:
                errors.append(f"item {item.id}: expected_behavior differs from its twin '{twin.id}'")
            else:
                # A twin pair measures the language gap only if both ask for the same required points.
                mine = sorted(k.match or k.text for k in item.key_points if k.required)
                theirs = sorted(k.match or k.text for k in twin.key_points if k.required)
                if mine != theirs:
                    errors.append(f"item {item.id}: its required key points differ from its twin '{twin.id}' "
                                  f"(a pair must ask for the same points)")

    if expected_total is not None and len(items) != expected_total:
        errors.append(f"dataset: {len(items)} items; expected {expected_total}")
    if expected_behaviors is not None:
        got_b = Counter(i.expected_behavior for i in items)
        if dict(got_b) != expected_behaviors:
            errors.append(f"dataset: expected_behavior counts {dict(sorted(got_b.items()))} != expected {dict(sorted(expected_behaviors.items()))}")
    if expected_distribution is not None:
        got = Counter(i.category for i in items)
        if dict(got) != expected_distribution:
            errors.append(f"dataset: category distribution {dict(sorted(got.items()))} != expected {dict(sorted(expected_distribution.items()))}")
    if paraphrased_range is not None:
        n = sum("paraphrased" in i.tags for i in items)
        lo, hi = paraphrased_range
        if not lo <= n <= hi:
            errors.append(f"dataset: {n} items tagged 'paraphrased'; expected {lo}-{hi}")
    return errors


def load_dataset(path: Path) -> tuple[list[Item], list[str]]:
    """Parse a JSONL file. Returns (items that parsed, errors). Blank lines are allowed."""
    items: list[Item] = []
    errors: list[str] = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {n}: invalid JSON ({e})")
            continue
        item, errs = parse_item(raw, where=f"line {n}")
        errors += errs
        if item is not None:
            items.append(item)
    return items, errors
