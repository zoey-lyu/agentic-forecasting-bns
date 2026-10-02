"""A context-aware word-count measure of how a statement leans.

Version 1 (:mod:`lexicon`) counted fixed words. It works — its tilt separates
hikes from cuts cleanly — but it is blunt in three specific ways, and all three
can be fixed without looking at a single outcome:

**Ambiguous direction words.** "lower" is dovish in "inflation is lower" and
hawkish in "lower unemployment". A bare word list cannot tell them apart. Here
a direction word only counts when a subject appears near it, and its polarity
is the product of the two: *down* × *unemployment* reads hawkish, *down* ×
*inflation* reads dovish.

**Negation.** "not expected to tighten" counted as hawkish in v1. A negation
within three words now flips the term's sign.

**Coverage.** The v1 lists hold 13 and 14 terms. Central-bank communication
vocabulary is larger than that, and a term the list is missing is a
measurement lost.

Everything here is derived from domain vocabulary, **not** from the data: no
term was chosen by checking whether it improved a result. The development split
(statements before 2023) exists so the *mechanics* can be compared against v1
without touching the 2023-26 window the market test uses.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


#: Unambiguous tightening vocabulary — hawkish wherever it appears.
HAWKISH_TERMS = (
    "inflationary",
    "overheating",
    "tighten",
    "tightening",
    "restrictive",
    "excess demand",
    "upside risk",
    "upside risks",
    "price pressures",
    "broadening",
    "entrenched",
    "de-anchor",
    "de-anchored",
    "unanchored",
    "second-round",
    "wage pressures",
    "overheated",
    "persistent",
    "stronger than expected",
    "above target",
    "elevated",
)

#: Unambiguous easing vocabulary — dovish wherever it appears.
DOVISH_TERMS = (
    "slack",
    "accommodative",
    "stimulus",
    "disinflation",
    "disinflationary",
    "excess supply",
    "downside risk",
    "downside risks",
    "recession",
    "contraction",
    "subdued",
    "soft",
    "softening",
    "sluggish",
    "muted",
    "below target",
    "weaker than expected",
    "spare capacity",
    "deflation",
    "material slowdown",
)

#: Deliberate non-commitment.
HEDGING_TERMS = (
    "gradual",
    "gradually",
    "carefully",
    "patient",
    "patiently",
    "monitor",
    "monitoring",
    "assess",
    "assessing",
    "data-dependent",
    "wait",
    "watch",
    "watching",
    "proceeding carefully",
    "take time",
    "evaluate",
)

#: Direction words and their sign: +1 is "up", -1 is "down".
_DIRECTION_TERMS = {
    "higher": 1,
    "rise": 1,
    "rising": 1,
    "rose": 1,
    "increase": 1,
    "increased": 1,
    "increasing": 1,
    "strengthen": 1,
    "strengthening": 1,
    "stronger": 1,
    "picked up": 1,
    "accelerate": 1,
    "accelerated": 1,
    "above": 1,
    "lower": -1,
    "fall": -1,
    "fell": -1,
    "falling": -1,
    "decline": -1,
    "declined": -1,
    "declining": -1,
    "weaken": -1,
    "weakening": -1,
    "weaker": -1,
    "slowdown": -1,
    "slowed": -1,
    "ease": -1,
    "eased": -1,
    "easing": -1,
    "reduce": -1,
    "reduced": -1,
    "moderate": -1,
    "moderated": -1,
    "below": -1,
}

#: Subjects and their sign: +1 means "more of this argues for tightening".
_SUBJECT_TERMS = {
    "inflation": 1,
    "prices": 1,
    "price": 1,
    "cpi": 1,
    "demand": 1,
    "wages": 1,
    "wage": 1,
    "growth": 1,
    "gdp": 1,
    "activity": 1,
    "output": 1,
    "employment": 1,
    "spending": 1,
    "consumption": 1,
    "exports": 1,
    "housing": 1,
    "unemployment": -1,
    "slack": -1,
    "unemployed": -1,
    "layoffs": -1,
    "spare capacity": -1,
    "excess supply": -1,
}

#: Words that flip the term they precede.
_NEGATIONS = ("not", "no", "never", "without", "unlikely", "neither", "nor", "little")

#: How many words after a direction word to look for its subject.
_SUBJECT_WINDOW = 4

#: How many words before a term to look for a negation.
_NEGATION_WINDOW = 3


def _tokens(text: str) -> list[str]:
    """Lower-cased word tokens."""
    return re.findall(r"\b[\w-]+\b", text.lower())


def _negated(tokens: list[str], position: int) -> bool:
    """Report whether a negation precedes the token at ``position``."""
    start = max(0, position - _NEGATION_WINDOW)
    return any(token in _NEGATIONS for token in tokens[start:position])


def _phrase_hits(text: str, terms: tuple[str, ...]) -> int:
    """Count occurrences of fixed terms and phrases, negation-aware.

    Multi-word terms are matched on the raw text; single words go through the
    token path so a preceding negation can flip them out.
    """
    lowered = text.lower()
    tokens = _tokens(text)
    hits = 0
    for term in terms:
        if " " in term:
            hits += len(re.findall(rf"\b{re.escape(term)}\b", lowered))
            continue
        for position, token in enumerate(tokens):
            if token == term and not _negated(tokens, position):
                hits += 1
    return hits


def _directional_hits(text: str) -> tuple[int, int]:
    """Count context-resolved directional statements as ``(hawkish, dovish)``.

    A direction word contributes only when a subject appears within the next
    few words. Polarity is ``direction × subject``, flipped again by a
    preceding negation — so "unemployment has not declined" reads dovish.
    """
    tokens = _tokens(text)
    hawkish = dovish = 0
    for position, token in enumerate(tokens):
        direction = _DIRECTION_TERMS.get(token)
        if direction is None:
            continue
        window = tokens[position + 1 : position + 1 + _SUBJECT_WINDOW]
        subject = next((_SUBJECT_TERMS[word] for word in window if word in _SUBJECT_TERMS), None)
        if subject is None:
            continue
        polarity = direction * subject
        if _negated(tokens, position):
            polarity = -polarity
        if polarity > 0:
            hawkish += 1
        else:
            dovish += 1
    return hawkish, dovish


def lexicon_scores_v2(text: str) -> dict[str, float]:
    """Score one statement, per hundred words.

    Returns
    -------
    dict
        ``lex2_tilt`` — hawkish minus dovish; ``lex2_intensity`` — the two
        summed; ``lex2_hedging`` — non-commitment vocabulary; and the two
        component counts, so a surprising tilt can be traced.
    """
    words = max(len(_tokens(text)), 1)
    fixed_hawkish = _phrase_hits(text, HAWKISH_TERMS)
    fixed_dovish = _phrase_hits(text, DOVISH_TERMS)
    context_hawkish, context_dovish = _directional_hits(text)
    hawkish = fixed_hawkish + context_hawkish
    dovish = fixed_dovish + context_dovish
    return {
        "lex2_tilt": (hawkish - dovish) / words * 100.0,
        "lex2_intensity": (hawkish + dovish) / words * 100.0,
        "lex2_hedging": _phrase_hits(text, HEDGING_TERMS) / words * 100.0,
        "lex2_hawkish": float(hawkish),
        "lex2_dovish": float(dovish),
    }


def score_statements_v2(paths: dict[pd.Timestamp, Path]) -> pd.DataFrame:
    """Score every cached statement, indexed by statement date."""
    rows = {date: lexicon_scores_v2(path.read_text()) for date, path in paths.items()}
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


__all__ = [
    "DOVISH_TERMS",
    "HAWKISH_TERMS",
    "HEDGING_TERMS",
    "lexicon_scores_v2",
    "score_statements_v2",
]
