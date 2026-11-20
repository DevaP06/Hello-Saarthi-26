"""Query preprocessor (repo doc 07 §3.1): runs before every search.

  1. Unicode math → words:  ω → omega, ∫ → integral, π → pi ...
  2. Abbreviations:         "ROC" → also searches "region of convergence"
  3. Typo correction:       "nyquist rte", "aliassing" → corrected against the words that actually
                            occur in the index being searched (never invents words)

`search_query`   what the dense + BM25 legs receive (original text + appended expansions)
`coverage_query` what the grounding gate compares against the chunks (abbreviations replaced
                 by their full form, typos corrected), so "FT" is not "uncovered" just because
                 the lecture spells out "Fourier transform". The gate also receives `expanded`,
                 so a lecture that only writes "DFT" still covers "discrete fourier transform".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

ABBREVIATIONS = {
    "ft": "fourier transform",
    "ctft": "continuous-time fourier transform",
    "dtft": "discrete-time fourier transform",
    "dft": "discrete fourier transform",
    "idft": "inverse discrete fourier transform",
    "fft": "fast fourier transform",
    "ifft": "inverse fast fourier transform",
    "stft": "short-time fourier transform",
    "fs": "sampling frequency",
    "roc": "region of convergence",
    "lti": "linear time-invariant",
    "ltv": "linear time-varying",
    "bibo": "bounded-input bounded-output",
    "fir": "finite impulse response",
    "iir": "infinite impulse response",
    "dsp": "digital signal processing",
    "adc": "analog-to-digital converter",
    "dac": "digital-to-analog converter",
    "psd": "power spectral density",
    "esd": "energy spectral density",
    "snr": "signal-to-noise ratio",
    "ct": "continuous-time",
    "dt": "discrete-time",
    "zoh": "zero-order hold",
}

_UNICODE = {
    "ω": "omega", "Ω": "omega", "π": "pi", "∫": "integral", "∑": "sum", "Σ": "sum", "δ": "delta", "Δ": "delta",
    "τ": "tau", "θ": "theta", "φ": "phi", "ϕ": "phi", "α": "alpha", "β": "beta", "σ": "sigma", "μ": "mu",
    "λ": "lambda", "∞": "infinity", "√": "sqrt", "−": "-", "–": "-", "—": "-", "’": "'", "“": '"', "”": '"',
    "²": "^2", "³": "^3", "×": "*", "·": "*", "≤": "<=", "≥": ">=",
}

_WORD = re.compile(r"[A-Za-z]+")


@dataclass
class Prepared:
    original: str
    search_query: str
    coverage_query: str
    expanded: dict = field(default_factory=dict)    # abbreviation -> full form
    corrected: dict = field(default_factory=dict)   # typo -> correction

    @property
    def changed(self) -> bool:
        return bool(self.expanded or self.corrected) or self.search_query != self.original


# Ambiguous in prose / integrals ("dt"): expand only when written in capitals.
_CAPS_ONLY = frozenset({"dt", "ct", "fs"})

# Ordinary English words that are one edit away from technical words ("there" / "three"):
# these are never "corrected".
_COMMON = frozenset("""
about above after again against also always another around because before being below between both
could does doing done down during each either else enough even ever every first from further given
going have here however into just later least less like made make many might more most much must
never next none often only other others over same several should since some still such than that
their them then there these they this those though through thus under until upon using very want
were what when where whether which while whose will with within without would write your
""".split())


def normalize_unicode(text: str) -> str:
    out = []
    for c in text:
        r = _UNICODE.get(c)
        if r is None:
            out.append(c)
        elif r.isalpha():
            out.append(f" {r} ")      # ω -> " omega ": keep it a separate word
        else:
            out.append(r)
    return "".join(out)


def _edit_distance(a: str, b: str, limit: int) -> int:
    """Damerau-Levenshtein (adjacent transposition counts 1), early exit above `limit`."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if prev2 is not None and i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                v = min(v, prev2[j - 2] + 1)
            cur.append(v)
        prev2, prev = prev, cur
        if min(prev) > limit:
            return limit + 1
    return prev[-1]


def correct_token(token: str, vocab) -> str | None:
    """A unique close match from `vocab`, or None. Only for words that are NOT in the vocabulary."""
    t = token.lower()
    if len(t) < 6 or t in vocab or t in _COMMON:
        return None
    limit = 1 if len(t) <= 7 else 2
    best, best_d, tie = None, limit + 1, False
    for w in vocab:
        if len(w) < 4 or w[0] != t[0] or abs(len(w) - len(t)) > limit:
            continue
        d = _edit_distance(t, w, limit)
        if d < best_d:
            best, best_d, tie = w, d, False
        elif d == best_d and w != best:
            tie = True
    return None if tie or best is None or best_d > limit else best


def prepare(query: str, vocab=None) -> Prepared:
    """vocab: words occurring in the searched index (e.g. BM25 idf keys); typo correction is
    skipped when None/empty. Never raises."""
    original = query or ""
    try:
        text = " ".join(normalize_unicode(original).split())
        expanded: dict[str, str] = {}
        corrected: dict[str, str] = {}
        cov_parts: list[str] = []
        extra: list[str] = []
        pos = 0
        out_words: list[str] = []
        for m in _WORD.finditer(text):
            out_words.append(text[pos:m.start()])
            pos = m.end()
            w, low = m.group(0), m.group(0).lower()
            if low in ABBREVIATIONS and (w.isupper() or low not in _CAPS_ONLY):
                expanded[w] = ABBREVIATIONS[low]
                extra.append(ABBREVIATIONS[low])
                out_words.append(w)
                cov_parts.append(ABBREVIATIONS[low])
                continue
            fix = correct_token(w, vocab) if vocab else None
            if fix:
                corrected[w] = fix
                out_words.append(fix)
                cov_parts.append(fix)
            else:
                out_words.append(w)
                cov_parts.append(w)
        out_words.append(text[pos:])
        search = "".join(out_words).strip()
        if extra:
            search = f"{search} {' '.join(dict.fromkeys(extra))}"
        return Prepared(original, search, " ".join(cov_parts) if cov_parts else search, expanded, corrected)
    except Exception:  # preprocessing must never break a search
        return Prepared(original, original, original)
