"""Read-only access to precomputed AutoARIMA anchor lookup tables.

The tables are written by ``build_anchor_table.py`` (seed=42, num_samples=500)
so that every anchored-agent variant reads byte-identical anchors instead of
re-fitting AutoARIMA per run. See
``planning-docs/anchor-externalization-interview-notes.md`` for why this
matters: unseeded AutoARIMA moves the anchor median by ~0.48 USD between runs,
the same order of magnitude as the drift being measured.

This module has no local imports so both ``agent.py`` (prompt injection) and
``anchored_predictor.py`` (reconstruction) can depend on it without a cycle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


_ANCHORS_DIR = Path(__file__).parent.parent / "anchors"


class AnchorEntry(BaseModel):
    """One (as_of, horizon) anchor: a statistical point forecast and quantile grid.

    Attributes
    ----------
    point_forecast : float
        AutoARIMA median forecast.
    quantiles : dict[float, float]
        Mapping from each :data:`~aieng.forecasting.evaluation.prediction.STANDARD_QUANTILES`
        level to its forecast value.
    """

    point_forecast: float
    quantiles: dict[float, float]

    @property
    def half_width(self) -> float:
        """Half of the anchor's 90% interval: ``(q0.95 - q0.05) / 2``."""
        return (self.quantiles[0.95] - self.quantiles[0.05]) / 2


class AnchorSource:
    """Read-only view over one task's precomputed anchor table.

    Parameters
    ----------
    table : dict
        Parsed ``anchor_<spec_id>.json`` contents.
    task_id : str
        Task within the table to serve anchors for.
    """

    def __init__(self, table: dict[str, Any], *, task_id: str) -> None:
        self._spec_id = table["spec_id"]
        self._task_id = task_id
        try:
            self._origins: dict[str, dict[str, dict]] = table["tasks"][task_id]
        except KeyError as exc:
            raise KeyError(
                f"Anchor table {self._spec_id!r} has no task {task_id!r}. Available tasks: {sorted(table['tasks'])}"
            ) from exc

    @classmethod
    def from_spec_id(cls, spec_id: str, *, task_id: str = "wti_oil_price_forecast") -> "AnchorSource":
        """Load ``anchors/anchor_<spec_id>.json`` (path relative to this package).

        Parameters
        ----------
        spec_id : str
            Backtest spec id, e.g. ``"energy_oil_eval"`` or ``"energy_oil_eval_smoke"``.
        task_id : str
            Task within the spec to serve anchors for.

        Returns
        -------
        AnchorSource
        """
        path = _ANCHORS_DIR / f"anchor_{spec_id}.json"
        table = json.loads(path.read_text())
        return cls(table, task_id=task_id)

    def get(self, *, as_of: Any, horizon: int) -> AnchorEntry:
        """Look up the anchor for one ``(as_of, horizon)`` pair.

        Parameters
        ----------
        as_of : Any
            Forecast origin date; stringified and truncated to ``YYYY-MM-DD``,
            matching the convention used throughout ``agent.py``.
        horizon : int
            Horizon step, matching a key in the precomputed table.

        Returns
        -------
        AnchorEntry

        Raises
        ------
        KeyError
            If the origin or horizon was not precomputed — the anchor table is
            an external data boundary, so failing loudly here (rather than
            falling back to a live AutoARIMA fit) is deliberate: a silent
            fallback would defeat the reason the table exists.
        """
        origin_key = str(as_of)[:10]
        origin = self._origins.get(origin_key)
        if origin is None:
            raise KeyError(
                f"No precomputed anchor for spec={self._spec_id!r} task={self._task_id!r} "
                f"as_of={origin_key!r}. Available origins: {sorted(self._origins)}"
            )
        entry = origin.get(str(horizon))
        if entry is None:
            raise KeyError(
                f"No precomputed anchor for spec={self._spec_id!r} task={self._task_id!r} "
                f"as_of={origin_key!r} horizon={horizon}. Available horizons: {sorted(origin)}"
            )
        return AnchorEntry(
            point_forecast=entry["point_forecast"],
            quantiles={float(q): v for q, v in entry["quantiles"].items()},
        )

    def available_horizons(self, *, as_of: Any) -> list[int]:
        """Return the horizons with a precomputed anchor for this origin.

        A horizon can be legitimately absent even for a valid origin: the
        table is built by scoring an AutoARIMA backtest against realized
        closes (see ``build_anchor_table.py``), and a horizon whose
        ``forecast_date`` lands on a market holiday has no realized close to
        score against, so it's silently dropped when the table is built.
        Callers that need to forecast several horizons per origin (e.g.
        ``AnchoredWtiPromptBuilder``) should request signals only for the
        horizons this returns, rather than assuming every task horizon has
        an anchor.

        Parameters
        ----------
        as_of : Any
            Forecast origin date; stringified and truncated to ``YYYY-MM-DD``.

        Returns
        -------
        list[int]
            Sorted horizons available for this origin.

        Raises
        ------
        KeyError
            If the origin itself was not precomputed at all — unlike a
            single missing horizon, this is not the market-holiday case and
            should still fail loudly.
        """
        origin_key = str(as_of)[:10]
        origin = self._origins.get(origin_key)
        if origin is None:
            raise KeyError(
                f"No precomputed anchor for spec={self._spec_id!r} task={self._task_id!r} "
                f"as_of={origin_key!r}. Available origins: {sorted(self._origins)}"
            )
        return sorted(int(h) for h in origin)
