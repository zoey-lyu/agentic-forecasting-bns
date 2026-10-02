"""Test whether the Bank's language tracks its own activity level.

The calibration test showed the market's real advantage: it re-estimates how
active the Bank currently is, continuously, while anything fitted to history
cannot. That is a base-rate gap, and the only plausible way to close it without
market prices is to read it out of what the Bank says.

So, before writing any agent, the cheap question: **is the regime visible in
the statements at all?** A2's stance table already covers all 141 statements
from 2009 to 2026, so this costs nothing.

The test that matters is not whether language correlates with activity — a
statement issued during an easing cycle obviously sounds like one. It is
whether language says anything about the *coming* months **that the recent move
rate does not already say**. The trailing move rate is free and needs no text;
if language only repeats it, an agent reading the same statements adds nothing.

Hence every model here is fitted twice: with the trailing move rate alone, and
with the trailing move rate plus a language variable. The gap between them is
the answer.

Overlapping forward windows make neighbouring observations share meetings, so
the full-sample correlations are optimistic. A non-overlapping subsample is
reported alongside as the conservative reading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


#: Meetings of history used to measure "how active has the Bank been lately".
TRAILING_MEETINGS = 8

#: Meetings ahead used to measure "how active is it about to be".
FORWARD_MEETINGS = 4


def build_regime_frame(
    stance: pd.DataFrame,
    directions: pd.DataFrame,
    *,
    trailing: int = TRAILING_MEETINGS,
    forward: int = FORWARD_MEETINGS,
) -> pd.DataFrame:
    """Join stance readings to trailing and forward activity, one row per meeting.

    Parameters
    ----------
    stance : pd.DataFrame
        A2's committed stance table (``statement_date``, ``hawk_dove``,
        ``guidance``, ``guidance_score``).
    directions : pd.DataFrame
        Canonical direction series (``timestamp``, ``value``).
    trailing, forward : int
        Window lengths in meetings.

    Returns
    -------
    pd.DataFrame
        Indexed by meeting date, with ``moved``, ``trailing_rate`` (excluding
        the meeting itself), ``forward_rate``, and the language variables.
        Rows without a full forward window are dropped.
    """
    moves = directions.assign(timestamp=pd.to_datetime(directions["timestamp"])).sort_values("timestamp")
    moves["moved"] = (moves["value"].astype(float) != 0.0).astype(float)
    frame = moves.set_index("timestamp")[["moved"]]

    # Trailing rate excludes the current meeting: it is what a forecaster knows
    # before the decision. Forward rate starts at the next meeting.
    frame["trailing_rate"] = frame["moved"].shift(1).rolling(trailing, min_periods=trailing).mean()
    frame["forward_rate"] = frame["moved"].shift(-1).rolling(forward, min_periods=forward).mean().shift(-(forward - 1))

    text = stance.assign(statement_date=pd.to_datetime(stance["statement_date"])).set_index("statement_date")
    frame["hawk_dove"] = text["hawk_dove"].astype(float)
    frame["stance_intensity"] = frame["hawk_dove"].abs()
    frame["guidance_score"] = text["guidance_score"].astype(float)
    frame["guidance_intensity"] = frame["guidance_score"].abs()
    frame["signals_anything"] = (text["guidance"] != "none").astype(float)

    return frame.dropna(subset=["trailing_rate", "forward_rate", "hawk_dove"])


def incremental_r2(frame: pd.DataFrame, language_column: str) -> dict[str, object]:
    """Measure what a language variable adds over the trailing move rate.

    Fits two ordinary least-squares models of ``forward_rate`` — one on the
    trailing rate alone, one with the language variable added — and reports
    both R² values and the gain. The gain is the number that matters: a large
    raw correlation with a zero gain means the text is only restating history.
    """
    y = frame["forward_rate"].to_numpy(dtype=float)

    def r_squared(columns: list[str]) -> float:
        x = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy(dtype=float) for c in columns])
        coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
        residuals = y - x @ coefficients
        total = float(((y - y.mean()) ** 2).sum())
        return 1.0 - float((residuals**2).sum()) / total if total > 0 else float("nan")

    base = r_squared(["trailing_rate"])
    with_text = r_squared(["trailing_rate", language_column])
    return {
        "language": language_column,
        "n": int(len(frame)),
        "r2_trailing_only": round(base, 4),
        "r2_with_language": round(with_text, 4),
        "gain": round(with_text - base, 4),
        "raw_corr_with_forward": round(float(frame[language_column].corr(frame["forward_rate"])), 4),
        "corr_with_trailing": round(float(frame[language_column].corr(frame["trailing_rate"])), 4),
    }


__all__ = ["FORWARD_MEETINGS", "TRAILING_MEETINGS", "build_regime_frame", "incremental_r2"]
