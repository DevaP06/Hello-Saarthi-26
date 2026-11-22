"""Find citation references in an assistant response (Sprint 2, offline, standard library only).

The parser only *identifies* citation markers. It never decides whether a citation is valid, whether
its source exists, or whether it supports the answer; that is a later evaluation stage.

    from eval.answer_eval.citation_parser import parse_citations
    result = parse_citations(response_text)
    for c in result.citations:
        print(c.kind, c.text, c.start, c.end)

Formats recognised (taken from what the Saarthi engines actually emit):

  numeric       [1]  [2, 3]  [1][2]          Gemini knowledge-base answers ("Cite ... like [1] or [2]",
                                             src/knowledge_base/gemini_rag.py) and mind-mode references.
  page_label    [EE210 Module 1 Signals and Sampling, p. 3]   also "p.3", "page 3"
                                             src/tools/citations.source_label for course material.
  video_label   [Video: DSP Week 3, 14:22]   source_label for video chunks (m:ss or h:mm:ss).
  source_label  [Source: notes.pdf, p.4]  [Source: notes.pdf]
                                             the "**Sources**" list appended by chat_service._kb_chat_reply.
  title_label   [EE210 Module 2 LTI Systems and Convolution]
                                             a bare title label; only recognised when the caller passes
                                             `known_titles`, because "[anything]" is otherwise ambiguous.

Several labels in one bracket separated by ";" are split ("[A, p. 1; B, p. 2]"). Citations inside a
"Sources" block (the "**Sources:** a; b" suffix of src/tools/citations.sources_suffix, or the
"**Sources**" list) are marked `in_sources_block=True`, so a later stage can tell an inline citation
written by the model from one appended by the application.

Never treated as citations: text inside inline code, fenced code blocks and LaTeX math ($...$, $$...$$,
\\(...\\), \\[...\\]); indexing such as x[n], h[n-k], a[1]; arrays and intervals such as = [1, 2, 3],
[0, 1], [0.5]; markdown links [text](url); task boxes [ ] / [x]; numbers outside 1-99 such as [2018].

Not supported (documented limitations): numeric ranges such as [1-3], footnote markers [^1], superscript
citations, and author-year citations.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["Citation", "MalformedCitation", "CitationParseResult", "parse_citations", "CITATION_KINDS"]

CITATION_KINDS = ("numeric", "page_label", "video_label", "source_label", "title_label")


@dataclass(frozen=True)
class Citation:
    """One citation reference found in the response.

    text        the exact substring of the original response, e.g. "[1]" or "[Lecture 4, p. 12]".
                For a split multi-label bracket it is that label's part, without brackets.
    kind        one of CITATION_KINDS.
    start, end  character offsets in the original response (text == response[start:end] for whole
                brackets; for a part of a ";"-separated bracket, the offsets of that part).
    numbers     numeric citations: the referenced numbers, e.g. (2, 3); otherwise ().
    title       label citations: the source title or file name as written; otherwise None.
    page        page_label / source_label: the page as written ("3"); otherwise None.
    timestamp   video_label: the timestamp as written ("14:22"); otherwise None.
    in_sources_block  True when the citation sits in a trailing "Sources" block.
    key         a normalised identity for counting repeats: "n:2", "page:<title>|3", "video:<title>|14:22",
                "source:<title>|4", "title:<title>" (title lower-cased, whitespace collapsed).
    """
    text: str
    kind: str
    start: int
    end: int
    numbers: tuple[int, ...] = ()
    title: str | None = None
    page: str | None = None
    timestamp: str | None = None
    in_sources_block: bool = False
    key: str = ""


@dataclass(frozen=True)
class MalformedCitation:
    """A marker that clearly tries to be a citation but cannot be parsed (reported, never raised)."""
    text: str
    start: int
    end: int
    reason: str


@dataclass(frozen=True)
class CitationParseResult:
    citations: tuple[Citation, ...] = ()
    malformed: tuple[MalformedCitation, ...] = ()
    has_sources_block: bool = False

    @property
    def inline(self) -> tuple[Citation, ...]:
        """Citations written in the body of the answer (not in a trailing Sources block)."""
        return tuple(c for c in self.citations if not c.in_sources_block)

    def unique_keys(self) -> list[str]:
        """Distinct citation keys in order of first appearance (repeats collapsed)."""
        return list(dict.fromkeys(c.key for c in self.citations))

    def counts(self) -> dict[str, int]:
        """How many times each citation key occurs."""
        out: dict[str, int] = {}
        for c in self.citations:
            out[c.key] = out.get(c.key, 0) + 1
        return out


# ── Masking: regions where brackets are never citations ─────────────────────

_MASK_PATTERNS = [
    re.compile(r"```.*?(?:```|\Z)", re.S),          # fenced code (unclosed fence runs to the end)
    re.compile(r"`[^`\n]*`"),                        # inline code
    re.compile(r"\$\$.*?\$\$", re.S),               # display math
    re.compile(r"\\\[.*?\\\]", re.S),               # \[ ... \]
    re.compile(r"\\\(.*?\\\)", re.S),               # \( ... \)
    re.compile(r"(?<![\\$])\$(?!\s)[^$\n]+?(?<!\s)\$(?!\d)"),  # inline $...$ (not "$5 and $6")
]


def _masked(text: str) -> str:
    """Same length as `text`, with code and math regions replaced by spaces (offsets stay valid)."""
    chars = list(text)
    for pat in _MASK_PATTERNS:
        for m in pat.finditer("".join(chars)):
            for i in range(m.start(), m.end()):
                if chars[i] != "\n":
                    chars[i] = " "
    return "".join(chars)


# ── Patterns ────────────────────────────────────────────────────────────────

_BRACKET = re.compile(r"\[([^\[\]\n]{1,300})\]")
_NUMERIC_BODY = re.compile(r"\s*([1-9]\d?)\s*(?:,\s*([1-9]\d?)\s*)*")
_PAGE = re.compile(r"^(?P<title>.+?),\s*(?:p\.\s*|pp\.\s*|page\s+)(?P<page>\d{1,4})\s*$", re.I)
_PAGE_BROKEN = re.compile(r"^(?P<title>.+?),\s*(?:p\.|pp\.|page)\s*$", re.I)
_VIDEO = re.compile(r"^Video:\s*(?P<title>.+?),\s*(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\s*$", re.I)
_SOURCE = re.compile(r"^Source:\s*(?P<title>[^,]+?)(?:,\s*(?:p\.\s*|page\s+)(?P<page>\d{1,4}))?\s*$", re.I)
# "**Sources:**", "**Sources**", "__Sources__:", "Sources:", "## Sources" at the start of a line. A plain
# "Sources of error ..." sentence is not a heading (no colon, no bold, no #).
_SOURCES_HEADING = re.compile(
    r"(?im)^[ \t]*(?:(?:\*\*|__)\s*(?:sources|references)\s*:?\s*(?:\*\*|__)\s*:?"
    r"|#{1,6}[ \t]*(?:sources|references)[ \t]*:?[ \t]*$"
    r"|(?:sources|references)[ \t]*:)"
)
_UNCLOSED_PREFIX = re.compile(r"\[(?:Source|Video):[^\]\n]*$", re.I | re.M)


def _norm_title(t: str) -> str:
    return " ".join(t.split()).lower()


def _numeric_context_ok(masked: str, start: int, end: int, prev_citation_end: int = -1) -> bool:
    """A numeric bracket is a citation only in running text, not as indexing, an array or a link.
    "[1][2]" is allowed when the bracket right before it was itself accepted as a citation."""
    before = masked[:start].rstrip(" \t")
    prev = masked[start - 1] if start > 0 else ""
    if prev == "]" and start == prev_citation_end:
        prev = " "
    if prev and (prev.isalnum() or prev in "_])}'\"^\\"):
        return False                      # x[1], a_[1], f(x)[1], ^[1]
    if before.endswith(("=", "(", "{", "[", ":=", "<-", "∈")):
        return False                      # = [1, 2], f([1]), {[1]}, [[1]]
    nxt = masked[end] if end < len(masked) else ""
    if nxt == "(":
        return False                      # markdown link [1](url)
    if nxt == ":" and (start == 0 or masked[start - 1] == "\n"):
        return False                      # reference-link definition "[1]: http://..."
    return True


def _sources_block_start(masked: str) -> int | None:
    """Offset where a trailing Sources block begins (the last "Sources" heading), or None."""
    last = None
    for m in _SOURCES_HEADING.finditer(masked):
        last = m.start()
    return last


def _label_citation(part: str, start: int, end: int, known: set[str],
                    in_block: bool = False) -> Citation | MalformedCitation | None:
    s = part.strip()
    if not s:
        return None
    m = _VIDEO.match(s)
    if m:
        t = m.group("title").strip()
        return Citation(part, "video_label", start, end, title=t, timestamp=m.group("ts"),
                        key=f"video:{_norm_title(t)}|{m.group('ts')}")
    if re.match(r"^Video:", s, re.I):
        return MalformedCitation(part, start, end, "video label without a title and m:ss timestamp")
    m = _SOURCE.match(s)
    if m and m.group("title").strip():
        t, p = m.group("title").strip(), m.group("page")
        return Citation(part, "source_label", start, end, title=t, page=p,
                        key=f"source:{_norm_title(t)}|{p or ''}")
    if re.match(r"^Source:", s, re.I):
        return MalformedCitation(part, start, end, "source label without a file name, or with an unreadable page")
    m = _PAGE.match(s)
    if m:
        t = m.group("title").strip()
        return Citation(part, "page_label", start, end, title=t, page=m.group("page"),
                        key=f"page:{_norm_title(t)}|{m.group('page')}")
    if _PAGE_BROKEN.match(s):
        return MalformedCitation(part, start, end, "page label without a page number")
    if _norm_title(s) in known or (in_block and re.search(r"[A-Za-z]", s)):
        # A bare "[Title]" is a citation if the title is known, or if it sits in a Sources block
        # (source_label() prints "[Title]" for material without page numbers).
        return Citation(part, "title_label", start, end, title=s, key=f"title:{_norm_title(s)}")
    return None


def parse_citations(text: str | None, known_titles: list[str] | tuple[str, ...] | set[str] | None = None) -> CitationParseResult:
    """Return every citation reference in `text`, in order of position.

    text          the assistant response (raw or normalised; offsets refer to the string you pass).
                  None or "" gives an empty result.
    known_titles  optional source titles (e.g. the course's material titles). Only with these can a
                  bare "[Title]" label be recognised; without them it is ignored.

    Never raises on malformed input: unparseable citation-like markers are listed in `malformed`.
    """
    if not text:
        return CitationParseResult()
    known = {_norm_title(t) for t in (known_titles or ()) if t and t.strip()}
    masked = _masked(text)
    src_start = _sources_block_start(masked)

    found: list[Citation] = []
    broken: list[MalformedCitation] = []

    for m in _BRACKET.finditer(masked):
        start, end = m.start(), m.end()
        inner = text[start + 1:end - 1]
        if not inner.strip():
            continue                                              # "[ ]" task box or empty
        in_block = src_start is not None and start >= src_start

        num = _NUMERIC_BODY.fullmatch(inner)
        if num:
            if not _numeric_context_ok(masked, start, end, found[-1].end if found else -1):
                continue
            nums = tuple(int(x) for x in re.findall(r"\d+", inner))
            found.append(Citation(text[start:end], "numeric", start, end, numbers=nums,
                                  in_sources_block=in_block, key="n:" + ",".join(map(str, nums))))
            continue
        if re.fullmatch(r"[\s\d.,;:+\-*/^=<>()xX]+", inner):
            continue                                              # [0, 1], [0.5], [2018], [1-3], [x]
        if text[end:end + 1] == "(":
            continue                                              # markdown link [text](url)

        # Label citations, possibly several separated by ";"
        parts: list[tuple[str, int, int]] = []
        pos = start + 1
        for piece in inner.split(";"):
            parts.append((piece, pos, pos + len(piece)))
            pos += len(piece) + 1
        results = [_label_citation(p, s, e, known, in_block) for p, s, e in parts]
        if len(parts) == 1:
            r = results[0]
            if isinstance(r, Citation):
                found.append(Citation(text[start:end], r.kind, start, end, title=r.title, page=r.page,
                                      timestamp=r.timestamp, in_sources_block=in_block, key=r.key))
            elif isinstance(r, MalformedCitation):
                broken.append(MalformedCitation(text[start:end], start, end, r.reason))
            continue
        if not any(isinstance(r, (Citation, MalformedCitation)) for r in results):
            continue                                              # ordinary text with ";" in brackets
        for (piece, s, e), r in zip(parts, results):
            lead = len(piece) - len(piece.lstrip())
            trail = len(piece) - len(piece.rstrip())
            s2, e2 = s + lead, e - trail
            if isinstance(r, Citation):
                found.append(Citation(text[s2:e2], r.kind, s2, e2, title=r.title, page=r.page,
                                      timestamp=r.timestamp, in_sources_block=in_block, key=r.key))
            elif isinstance(r, MalformedCitation):
                broken.append(MalformedCitation(text[s2:e2], s2, e2, r.reason))
            elif piece.strip():
                broken.append(MalformedCitation(text[s2:e2], s2, e2, "unrecognised part of a multi-label citation"))

    # Unclosed "[Source: ..." / "[Video: ..." at the end of a line
    for m in _UNCLOSED_PREFIX.finditer(masked):
        broken.append(MalformedCitation(text[m.start():m.end()], m.start(), m.end(), "citation bracket is not closed"))

    # Unlabelled "Sources:" suffix ("**Sources:** Title, p. 3; Title 2, p. 1"), as written by sources_suffix
    if src_start is not None:
        heading = _SOURCES_HEADING.match(masked, src_start)
        line_end = masked.find("\n", heading.end())
        line_end = len(masked) if line_end == -1 else line_end
        tail = masked[heading.end():line_end]
        if tail.strip() and "[" not in tail:
            pos = heading.end()
            for piece in tail.split(";"):
                s = pos + len(piece) - len(piece.lstrip())
                e = pos + len(piece.rstrip())
                r = _label_citation(text[s:e], s, e, known, True)
                if isinstance(r, Citation):
                    found.append(Citation(text[s:e], r.kind, s, e, title=r.title, page=r.page,
                                          timestamp=r.timestamp, in_sources_block=True, key=r.key))
                pos += len(piece) + 1

    found.sort(key=lambda c: c.start)
    broken.sort(key=lambda b: b.start)
    return CitationParseResult(tuple(found), tuple(broken), src_start is not None)
