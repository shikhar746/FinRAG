"""Metrics: retrieval (Recall@k, Hit@k, MRR), numeric answer matching,
citation accuracy, and paired bootstrap confidence intervals.

Items being ranked are hashable page keys, e.g. ("3M_2018_10K", 59).
"""
import math
import re

import numpy as np

# ---------------------------------------------------------------- retrieval


def recall_at_k(ranked: list, gold: set, k: int) -> float:
    """Fraction of the gold pages that appear in the top k."""
    if not gold:
        raise ValueError("question has no gold pages")
    return len(set(ranked[:k]) & gold) / len(gold)


def hit_at_k(ranked: list, gold: set, k: int) -> float:
    """1.0 if at least one gold page is in the top k, else 0.0."""
    return float(bool(set(ranked[:k]) & gold))


def reciprocal_rank(ranked: list, gold: set) -> float:
    """1 / rank of the first gold page (rank starts at 1); 0 if none is ranked."""
    for i, item in enumerate(ranked):
        if item in gold:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------- comparisons


def paired_bootstrap(a, b, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05) -> dict:
    """95% CI for mean(b - a) when a and b are per-question scores of two systems
    on the SAME questions.

    "Paired" means each resample draws whole questions, keeping system A's and
    system B's score on that question together. This removes the noise from
    question difficulty, so it is much tighter than comparing two separate CIs.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired scores must have the same length")
    diff = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean_diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0), "n": len(diff)}


# ---------------------------------------------------------------- numeric answers

_SCALES = {"thousand": 1e3, "k": 1e3, "million": 1e6, "millions": 1e6, "mn": 1e6, "m": 1e6,
           "mm": 1e6, "billion": 1e9, "billions": 1e9, "bn": 1e9, "b": 1e9, "trillion": 1e12}

# One number expression: optional minus / "(", optional currency, digits with
# thousands commas, optional decimals, optional ")" / "%", optional scale word.
# The lookbehind stops "FY2018" from being read as the number 2018.
_NUM = (r"(?<![\w.])(?P<neg>-|\()?\s*(?:\$|usd\s*)?\s*"
        r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
        r"\s*\)?\s*(?P<pct>%)?\s*(?:(?P<scale>" + "|".join(sorted(_SCALES, key=len, reverse=True)) + r")\b)?")
_NUM_FULL = re.compile(r"^\s*" + _NUM + r"\s*\.?\s*$", re.IGNORECASE)
_NUM_ANY = re.compile(_NUM, re.IGNORECASE)


def _value(m: re.Match) -> float:
    v = float(m.group("num").replace(",", ""))
    if m.group("scale"):
        v *= _SCALES[m.group("scale").lower()]
    return -v if m.group("neg") else v


def parse_number(text: str) -> float | None:
    """Parse a string that is ONLY a number ("$1,577.00", "(305)", "10.3%", "$2.1 billion").

    Returns None for anything with extra words, so sentence answers are left
    for human labelling. "%" is dropped without dividing by 100: "10.3%" -> 10.3.
    """
    m = _NUM_FULL.match(text)
    return _value(m) if m else None


def rounding_tol(text: str) -> float:
    """Half a unit of the last digit shown in a pure-number string.

    Gold "1.9%" may be any value in [1.85, 1.95), so a prediction of 1.86 is
    consistent with it: rounding_tol("1.9%") = 0.05. Scale words count:
    rounding_tol("$2,018mn") = 0.5e6.
    """
    m = _NUM_FULL.match(text)
    if not m:
        raise ValueError(f"not a pure number: {text!r}")
    decimals = len(m.group("num").partition(".")[2])
    scale = _SCALES[m.group("scale").lower()] if m.group("scale") else 1.0
    return 0.5 * 10 ** -decimals * scale


def first_number(text: str) -> float | None:
    """The first number expression anywhere in text (used on the model's final-answer line)."""
    m = _NUM_ANY.search(text)
    return _value(m) if m else None


def numbers_match(pred: float | None, gold: float, rel_tol: float, abs_tol: float = 0.0,
                  allow_scale: bool = False) -> bool:
    """True if pred is within tolerance of gold.

    allow_scale=True also accepts pred that equals gold times 1e3/1e6/1e9 (or
    divided by it), for answers like "1.577 billion" when gold is "1577" in millions.
    """
    if pred is None:
        return False
    candidates = [pred]
    if allow_scale:
        candidates += [pred * f for f in (1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9)]
    return any(math.isclose(c, gold, rel_tol=rel_tol, abs_tol=abs_tol) for c in candidates)


# ---------------------------------------------------------------- citations


def citation_scores(cited: set[int], gold: set[int]) -> dict:
    """cited_gold: at least one cited page is a gold page.
    citation_precision: share of cited pages that are gold (0 if nothing cited)."""
    hit = cited & gold
    return {"cited_gold": float(bool(hit)),
            "citation_precision": len(hit) / len(cited) if cited else 0.0}
