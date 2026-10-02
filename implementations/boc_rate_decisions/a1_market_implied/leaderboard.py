"""Score predictors against *two* references: climatology and the market.

Skill is ``1 - mean_score / mean_score_reference``, so the reference defines
what "good" means. The existing leaderboard scores skill against climatology
— a predictor that reports the historical hold frequency at every meeting and
knows nothing else. Clearing that bar says little. The CORRA futures curve is
the real opponent: it already contains the press releases, the data flow and
the positioning of everyone trading Canadian rates.

These helpers work from per-meeting score *series* rather than harness result
objects, so freshly computed scores and scores stored by earlier runs can sit
in one table, and every comparison is restricted to the meetings all the
methods actually cover. Nothing here modifies
:mod:`boc_rate_decisions.analysis`; ``score_leaderboard`` there remains the
single-reference view over live results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from aieng.forecasting.evaluation.backtest import compute_rps


#: Category order for the BoC direction task, lowest ordinal first.
CATEGORY_ORDER = ["cut", "hold", "hike"]

#: Series value for each category, as declared by the specs.
CATEGORY_VALUES = {"cut": -1.0, "hold": 0.0, "hike": 1.0}


def outcome_index(value: float) -> int:
    """Map an observed direction value (-1/0/+1) to its ordered category index."""
    for index, label in enumerate(CATEGORY_ORDER):
        if np.isclose(value, CATEGORY_VALUES[label]):
            return index
    raise ValueError(f"Observed direction {value} is not a declared category value.")


def per_meeting_rps(probabilities: pd.DataFrame, outcomes: pd.Series) -> pd.Series:
    """Score a table of forecasts meeting by meeting.

    Parameters
    ----------
    probabilities : pd.DataFrame
        Indexed by meeting date, with one column per label in
        :data:`CATEGORY_ORDER`.
    outcomes : pd.Series
        Realised direction values (-1/0/+1) indexed by meeting date.

    Returns
    -------
    pd.Series
        RPS per meeting (unnormalised Epstein/Murphy convention, matching the
        harness), indexed by meeting date. Meetings absent from ``outcomes``
        are dropped.
    """
    rows: dict[pd.Timestamp, float] = {}
    for meeting, row in probabilities.iterrows():
        if meeting not in outcomes.index:
            continue
        distribution = [float(row[label]) for label in CATEGORY_ORDER]
        rows[meeting] = compute_rps([distribution], [outcome_index(float(outcomes.loc[meeting]))])
    return pd.Series(rows, name="rps").sort_index()


def climatology_probabilities(directions: pd.DataFrame, origins: pd.Series) -> pd.DataFrame:
    """Reproduce the climatology baseline offline, one row per origin.

    Mirrors ``CategoricalFrequencyPredictor``: at each origin, the empirical
    category frequencies over every meeting whose outcome was public by then.

    Parameters
    ----------
    directions : pd.DataFrame
        Canonical direction series (``timestamp``, ``value``).
    origins : pd.Series
        Origin dates indexed by the meeting each one forecasts.

    Returns
    -------
    pd.DataFrame
        Indexed by meeting date, with ``cut``/``hold``/``hike`` columns.
    """
    history = directions.assign(timestamp=pd.to_datetime(directions["timestamp"])).sort_values("timestamp")
    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for meeting, origin in origins.items():
        visible = history[history["timestamp"] <= pd.Timestamp(origin)]
        if visible.empty:
            raise ValueError(f"No meeting history visible at origin {origin}.")
        counts = dict.fromkeys(CATEGORY_ORDER, 0)
        for value in visible["value"].astype(float):
            counts[CATEGORY_ORDER[outcome_index(value)]] += 1
        n = len(visible)
        rows[pd.Timestamp(meeting)] = {label: counts[label] / n for label in CATEGORY_ORDER}
    return pd.DataFrame.from_dict(rows, orient="index").loc[:, CATEGORY_ORDER].sort_index()


def skill_table(
    scores: dict[str, pd.Series],
    references: dict[str, str],
    *,
    restrict_to_common: bool = True,
) -> pd.DataFrame:
    """Build a leaderboard with one skill column per reference.

    Parameters
    ----------
    scores : dict[str, pd.Series]
        Mapping ``predictor_id -> per-meeting scores`` (lower is better),
        each indexed by meeting date.
    references : dict[str, str]
        Mapping ``column suffix -> predictor_id``, e.g.
        ``{"climatology": "categorical_frequency", "market": "market_implied"}``
        produces ``skill_vs_climatology`` and ``skill_vs_market``.
    restrict_to_common : bool
        When ``True`` (default) every mean is taken over the meetings covered
        by *all* supplied predictors, so the skill columns compare like with
        like. ``n_meetings`` then equals that common count and
        ``n_meetings_available`` records each predictor's own coverage.

    Returns
    -------
    pd.DataFrame
        Sorted by ``mean_rps`` ascending.
    """
    missing = {name: pid for name, pid in references.items() if pid not in scores}
    if missing:
        raise ValueError(f"Reference predictors absent from scores: {missing}")

    common = set.intersection(*(set(series.index) for series in scores.values())) if scores else set()
    index = sorted(common)

    rows: list[dict[str, object]] = []
    for predictor_id, series in scores.items():
        used = series.loc[index] if restrict_to_common else series
        rows.append(
            {
                "predictor_id": predictor_id,
                "metric": "rps",
                "mean_rps": float(used.mean()),
                "n_meetings": int(len(used)),
                "n_meetings_available": int(len(series)),
            }
        )
    board = pd.DataFrame(rows)

    for name, reference_id in references.items():
        reference_scores = scores[reference_id].loc[index] if restrict_to_common else scores[reference_id]
        reference_mean = float(reference_scores.mean())
        board[f"skill_vs_{name}"] = (
            (1.0 - board["mean_rps"] / reference_mean).round(4) if reference_mean > 0 else np.nan
        )

    return board.sort_values("mean_rps").reset_index(drop=True)


def paired_deltas(
    scores: dict[str, pd.Series],
    reference_id: str,
    *,
    seed: int = 42,
    restrict_to_common: bool = True,
) -> pd.DataFrame:
    """Bootstrap each predictor's mean-RPS gap against one reference.

    A leaderboard of means alone invites over-reading a 0.02 gap on twelve
    meetings. This pairs the scores meeting by meeting and bootstraps
    ``mean(predictor - reference)``, reusing :func:`boc_rate_decisions.gate.bootstrap_ci`
    so the interval convention (95%, 10k draws, seeded) matches the gate.

    Parameters
    ----------
    scores : dict[str, pd.Series]
        Per-meeting scores, as passed to :func:`skill_table`.
    reference_id : str
        Predictor to difference against.
    seed : int
        Bootstrap seed; fixed so a verdict is reproducible.
    restrict_to_common : bool
        When ``True`` (default) every pair is differenced over the meetings
        covered by *all* predictors in ``scores`` — the same window
        :func:`skill_table` reports means on, so a leaderboard never mixes a
        12-meeting skill column with a 14-meeting interval on the same row.

    Returns
    -------
    pd.DataFrame
        Columns ``predictor_id``, ``delta_vs_reference`` (positive = worse than
        the reference), ``ci_lo``, ``ci_hi``, ``significant``.
    """
    from ..gate import bootstrap_ci  # noqa: PLC0415 - avoids importing sklearn at module import

    reference = scores[reference_id]
    common = set.intersection(*(set(series.index) for series in scores.values())) if scores else set()
    rows: list[dict[str, object]] = []
    for predictor_id, series in scores.items():
        shared = series.index.intersection(reference.index)
        if restrict_to_common:
            shared = shared.intersection(pd.Index(sorted(common)))
        delta, lo, hi = bootstrap_ci(series.loc[shared].to_numpy(), reference.loc[shared].to_numpy(), seed=seed)
        rows.append(
            {
                "predictor_id": predictor_id,
                "delta_vs_reference": round(delta, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "significant": "yes" if (lo > 0 or hi < 0) else "no",
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "CATEGORY_ORDER",
    "CATEGORY_VALUES",
    "climatology_probabilities",
    "outcome_index",
    "paired_deltas",
    "per_meeting_rps",
    "skill_table",
]
