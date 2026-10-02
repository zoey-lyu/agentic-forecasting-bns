"""The t+2 target series: what the Bank does at the meeting *after* the next one.

A1 showed the market prices the next meeting well enough that no method here
beats it. The premise of A3 is that this is a statement about *horizon*, not
about forecasting: four weeks out, the next decision is largely settled, but
the one after it is ten weeks out and the curve is thinner there. This module
builds the target that lets the claim be tested.

The harness trick
-----------------
A task carries one fixed horizon, but the gap from origin to the t+2 meeting
varies (63-77 days here) because the Bank's calendar is irregular. Rather than
bend the harness, the *series* carries the shift:

- ``timestamp`` — the **t+1** meeting date, so ``origin + 28 days`` lands on it
  exactly as in the canonical task;
- ``value`` — the decision at the **next** meeting after that date;
- ``released_at`` — that later meeting's date, so cutoff enforcement keeps the
  outcome invisible until it is actually announced.

A predictor therefore sees the ordinary 28-day task shape while being scored on
a decision roughly ten weeks ahead. Every existing predictor — climatology, the
macro logistic, A2's text arm — works against it unchanged, because they all
read ``task.target_series_id`` generically.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters.base import BaseAdapter

from ..data import DIRECTION_SERIES_ID, build_boc_service


DIRECTION_T2_SERIES_ID = "boc_rate_decision_direction_t2"
"""Derived per-meeting series: the direction of the decision one meeting later."""


class _FrameAdapter(BaseAdapter):
    """Serve an already-computed canonical frame.

    The t+2 series is derived from a series the service already holds, so there
    is nothing to fetch; this adapter exists only to satisfy the registration
    contract.
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def fetch(self) -> pd.DataFrame:
        """Return the precomputed canonical frame."""
        return self._frame.copy()


def derive_t2_directions(directions: pd.DataFrame) -> pd.DataFrame:
    """Shift a per-meeting direction series one meeting into the future.

    Parameters
    ----------
    directions : pd.DataFrame
        Canonical direction series (``timestamp``, ``value``, ``released_at``),
        one row per resolved meeting, sorted ascending.

    Returns
    -------
    pd.DataFrame
        Canonical series with ``timestamp`` = each meeting date, ``value`` =
        the *following* meeting's direction, ``released_at`` = the following
        meeting's date. The last meeting is dropped: its successor has not
        happened yet.
    """
    frame = directions.sort_values("timestamp").reset_index(drop=True)
    if len(frame) < 2:  # noqa: PLR2004 - a shift needs a successor
        raise ValueError("Need at least two resolved meetings to derive the t+2 series.")

    timestamps = pd.to_datetime(frame["timestamp"])
    return pd.DataFrame(
        {
            "timestamp": timestamps.iloc[:-1].to_numpy(),
            "value": frame["value"].astype(float).iloc[1:].to_numpy(),
            "released_at": timestamps.iloc[1:].to_numpy(),
        }
    )


def build_boc_t2_service(
    statcan_cache_dir: Path | None = None,
    fred_cache_dir: Path | None = None,
    *,
    as_of: pd.Timestamp | None = None,
) -> DataService:
    """Return the standard BoC service with the t+2 direction series added.

    Parameters
    ----------
    statcan_cache_dir, fred_cache_dir : Path or None
        Cache directories, passed through to
        :func:`boc_rate_decisions.data.build_boc_service`.
    as_of : pd.Timestamp or None
        Cutoff used when reading the source direction series to derive the
        shifted one. Defaults to today. The derived series carries its own
        ``released_at`` stamps, so predictors remain cutoff-safe regardless.

    Returns
    -------
    DataService
        Everything :func:`build_boc_service` registers, plus
        :data:`DIRECTION_T2_SERIES_ID`.
    """
    service = build_boc_service(statcan_cache_dir=statcan_cache_dir, fred_cache_dir=fred_cache_dir)
    cutoff = pd.Timestamp.today().normalize() if as_of is None else pd.Timestamp(as_of)
    directions = service.get_series(DIRECTION_SERIES_ID, as_of=cutoff)

    service.register(
        DIRECTION_T2_SERIES_ID,
        _FrameAdapter(derive_t2_directions(directions)),
        SeriesMetadata(
            series_id=DIRECTION_T2_SERIES_ID,
            description=(
                "Direction of the BoC decision at the meeting AFTER the one on this date: "
                "-1 cut, 0 hold, +1 hike. Timestamped at the intervening meeting so a "
                "28-day task resolves onto the later decision; released_at is the later "
                "meeting's own announcement date."
            ),
            source=f"Derived (shift of {DIRECTION_SERIES_ID})",
            units="-1/0/+1 direction indicator",
            frequency="irregular (8 fixed announcement dates per year)",
        ),
    )
    return service


__all__ = ["DIRECTION_T2_SERIES_ID", "build_boc_t2_service", "derive_t2_directions"]
