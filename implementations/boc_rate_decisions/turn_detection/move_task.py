"""Build the binary turn question — whether the Bank moves at a meeting.

Step 2 of the turn-detection plan. Steps 0 and 1 established that the entire
difference between the market and a cheap statistical model lives at turning
points, and that a router which can tell a live meeting from a quiet one would
be worth 0.045 RPS. This module builds the instrument that measures exactly
that ability, stripped of everything else:

    P(the Bank moves) — a cut or a hike, either direction — scored with Brier.

Why a separate task rather than reading it off the three-way forecast: on the
direction task, ten of every fourteen meetings are holds that every method gets
nearly right, so the numbers are dominated by meetings that carry no
information. Collapsing to move/hold puts all of the statistical weight on the
eleven events that matter.

Every arm here is a projection of something that already exists — no new
forecasting method is introduced:

- the market's ``p_cut + p_hike``,
- the logistic's ``1 - p_hold``,
- the historical share of meetings that moved.

The Murphy decomposition is included because a turn detector's *calibration*
is the point of interest, not just its score: A3 suggested the market prices
standing insurance against a move, which would show up as a reliability term
and as a mean forecast above the realised move rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


MOVE_EVENT_SERIES_ID = "boc_rate_move_event"
"""Derived per-meeting 0/1 series: 1.0 if the Bank changed the target rate."""


def derive_move_events(directions: pd.DataFrame) -> pd.DataFrame:
    """Collapse a -1/0/+1 direction series into a 0/1 move indicator.

    Parameters
    ----------
    directions : pd.DataFrame
        Canonical direction series (``timestamp``, ``value``, ``released_at``).

    Returns
    -------
    pd.DataFrame
        The same rows with ``value`` set to 1.0 for cuts and hikes, 0.0 for
        holds.
    """
    events = directions.copy()
    events["value"] = (events["value"].astype(float) != 0.0).astype(float)
    return events


def move_probability(probabilities: pd.DataFrame) -> pd.Series:
    """Project a cut/hold/hike table onto ``P(move)``.

    Uses ``1 - p_hold`` rather than ``p_cut + p_hike`` so the result is exactly
    a probability even when the three columns carry rounding error.
    """
    return (1.0 - probabilities["hold"].astype(float)).clip(0.0, 1.0)


def historical_move_frequency(directions: pd.DataFrame, origins: pd.Series) -> pd.Series:
    """Return the share of meetings resolved before each origin that moved.

    Parameters
    ----------
    directions : pd.DataFrame
        Canonical direction series; rows dated after an origin are ignored for
        that origin.
    origins : pd.Series
        Origin dates indexed by the meeting each one forecasts.

    Returns
    -------
    pd.Series
        ``P(move)`` indexed by meeting date.
    """
    history = directions.assign(timestamp=pd.to_datetime(directions["timestamp"])).sort_values("timestamp")
    rows: dict[pd.Timestamp, float] = {}
    for meeting, origin in origins.items():
        visible = history[history["timestamp"] <= pd.Timestamp(origin)]
        if visible.empty:
            raise ValueError(f"No meeting history visible at origin {origin}.")
        rows[pd.Timestamp(meeting)] = float((visible["value"].astype(float) != 0.0).mean())
    return pd.Series(rows).sort_index()


def brier_scores(probabilities: pd.Series, outcomes: pd.Series) -> pd.Series:
    """Per-meeting Brier score ``(p - y)**2`` over the meetings both cover."""
    index = probabilities.index.intersection(outcomes.index)
    return pd.Series(
        (probabilities.loc[index].to_numpy() - outcomes.loc[index].to_numpy()) ** 2,
        index=index,
    ).sort_index()


def murphy_decomposition(probabilities: pd.Series, outcomes: pd.Series, *, n_bins: int = 4) -> dict[str, float]:
    """Split the Brier score into reliability, resolution and uncertainty.

    Reliability is the penalty for forecasts that do not happen as often as
    they claim (lower is better); resolution rewards separating high-risk
    meetings from low-risk ones (higher is better); uncertainty is the base
    rate's own variance and belongs to the window, not the forecaster.

    ``Brier = reliability - resolution + uncertainty`` holds exactly only when
    every distinct forecast value forms its own bin. Continuous forecasts
    binned into a handful of buckets leave a within-bin variance term, which is
    reported as ``residual`` so the four numbers always add back to ``brier``.
    Binning is kept deliberately: with distinct-value bins each bucket would
    hold one meeting, reliability would be identically zero, and the diagnostic
    would say nothing.

    With 28 meetings the bins are coarse either way, so read the terms as
    direction rather than as measurements.
    """
    index = probabilities.index.intersection(outcomes.index)
    p = probabilities.loc[index].to_numpy(dtype=float)
    y = outcomes.loc[index].to_numpy(dtype=float)
    n = len(p)
    base_rate = float(y.mean())

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    reliability = 0.0
    resolution = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (p >= lower) & (p <= upper if upper >= 1.0 else p < upper)
        count = int(in_bin.sum())
        if count == 0:
            continue
        mean_forecast = float(p[in_bin].mean())
        observed = float(y[in_bin].mean())
        reliability += count * (mean_forecast - observed) ** 2
        resolution += count * (observed - base_rate) ** 2

    brier = float(np.mean((p - y) ** 2))
    reliability /= n
    resolution /= n
    uncertainty = base_rate * (1.0 - base_rate)
    return {
        "brier": round(brier, 4),
        "reliability": round(reliability, 4),
        "resolution": round(resolution, 4),
        "uncertainty": round(uncertainty, 4),
        "residual": round(brier - (reliability - resolution + uncertainty), 4),
        "mean_forecast": round(float(p.mean()), 4),
        "observed_rate": round(base_rate, 4),
        "bias": round(float(p.mean()) - base_rate, 4),
    }


def platt_scale(
    train_probabilities: pd.Series,
    train_outcomes: pd.Series,
    apply_to: pd.Series,
) -> pd.Series:
    """Fit a two-parameter recalibration on one window and apply it to another.

    Logistic regression of the outcome on the forecast's log-odds. Fitting and
    scoring must use different windows — the pre-registration forbids reporting
    an in-sample number, because a recalibration always improves the data it
    was fitted on.
    """
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    def log_odds(series: pd.Series) -> np.ndarray:
        clipped = series.astype(float).clip(1e-6, 1 - 1e-6).to_numpy()
        transformed: np.ndarray = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        return transformed

    index = train_probabilities.index.intersection(train_outcomes.index)
    classes = set(train_outcomes.loc[index].astype(int))
    if len(classes) < 2:  # noqa: PLR2004 - a fit needs both outcomes
        raise ValueError(
            f"Platt scaling needs both outcomes in the training window; it contains only {sorted(classes)}."
        )
    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(log_odds(train_probabilities.loc[index]), train_outcomes.loc[index].astype(int).to_numpy())
    return pd.Series(model.predict_proba(log_odds(apply_to))[:, 1], index=apply_to.index)


__all__ = [
    "MOVE_EVENT_SERIES_ID",
    "brier_scores",
    "derive_move_events",
    "historical_move_frequency",
    "move_probability",
    "murphy_decomposition",
    "platt_scale",
]
