"""A contamination-free measure of how strongly a statement leans.

The stance scores in A2 come from a model that has read 2009-2026. Using them
to predict what the Bank does next is therefore suspect: a tone score shaded by
knowing the answer would produce exactly the relationship we want to find.

This module measures the same thing mechanically. A fixed word list, counted.
No model, no memory, no way for the future to leak in. If the relationship
survives here, it is real; if it does not, the stance result was an artifact.

The lists are deliberately small, standard in the central-bank communication
literature (Apel & Blix Grimaldi and descendants), and **fixed before the
measure was computed even once** — a list tuned until the answer came out right
would be worth nothing.

``intensity`` is the share of directional words of either kind. It is the
mechanical analogue of ``|hawk_dove|``: not which way the Bank leans, but how
hard. That is the variable the stance table flagged, so it is the one to check.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


#: Words signalling a tightening lean.
HAWKISH_TERMS = (
    "inflationary",
    "overheating",
    "tighten",
    "tightening",
    "restrictive",
    "raise",
    "raising",
    "increase",
    "higher",
    "persistent",
    "elevated",
    "upside",
    "excess demand",
)

#: Words signalling an easing lean.
DOVISH_TERMS = (
    "slack",
    "weaken",
    "weakening",
    "weaker",
    "slowdown",
    "downside",
    "ease",
    "easing",
    "accommodative",
    "lower",
    "lowering",
    "reduce",
    "stimulus",
    "excess supply",
)

#: Words signalling deliberate inaction — the Bank explicitly not committing.
HEDGING_TERMS = (
    "gradual",
    "gradually",
    "carefully",
    "patient",
    "monitor",
    "monitoring",
    "assess",
    "assessing",
    "data-dependent",
    "wait",
    "watch",
)


def _count(text: str, terms: tuple[str, ...]) -> int:
    """Count whole-word occurrences of any term, case-insensitively."""
    lowered = text.lower()
    return sum(len(re.findall(rf"\b{re.escape(term)}\b", lowered)) for term in terms)


def lexicon_scores(text: str) -> dict[str, float]:
    """Score one statement on the fixed word lists.

    Returns
    -------
    dict
        ``lex_intensity`` — directional words as a share of all words, the
        mechanical stand-in for ``|hawk_dove|``; ``lex_tilt`` — hawkish minus
        dovish share, the stand-in for ``hawk_dove``; ``lex_hedging`` — hedging
        words as a share, which should move the opposite way if the measure
        means anything.
    """
    words = max(len(re.findall(r"\b\w+\b", text)), 1)
    hawkish = _count(text, HAWKISH_TERMS)
    dovish = _count(text, DOVISH_TERMS)
    return {
        "lex_intensity": (hawkish + dovish) / words * 100.0,
        "lex_tilt": (hawkish - dovish) / words * 100.0,
        "lex_hedging": _count(text, HEDGING_TERMS) / words * 100.0,
    }


def score_statements(paths: dict[pd.Timestamp, Path]) -> pd.DataFrame:
    """Score every cached statement, indexed by statement date."""
    rows = {date: lexicon_scores(path.read_text()) for date, path in paths.items()}
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


__all__ = ["DOVISH_TERMS", "HAWKISH_TERMS", "HEDGING_TERMS", "lexicon_scores", "score_statements"]
