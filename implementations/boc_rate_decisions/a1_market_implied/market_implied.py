"""``MarketImpliedPredictor`` — the reference forecast every method must beat.

Reads the committed ``origin_date -> (p_cut, p_hold, p_hike)`` table built by
:mod:`build_table` from One-Month CORRA futures settlements and replays it as a
:class:`~aieng.forecasting.evaluation.predictor.Predictor`. No fitting, no
network, no LLM: the table *is* the model.

Why this matters more than the climatology baseline: climatology knows nothing
and predicts ``p_hold ~= 0.80`` at every meeting, so beating it proves almost
nothing. The futures curve aggregates everything public — the press release,
the data flow, the dealers' positions — priced by people with money at stake.
Skill measured against it answers the question the project actually cares
about: does reading text add anything the market has not already priced?

Usage::

    from boc_rate_decisions.a1_market_implied import MarketImpliedPredictor

    predictor = MarketImpliedPredictor()
    result = evaluate(predictor=predictor, eval_spec=spec, data_service=svc)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import CategoricalForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask


DEFAULT_TABLE_PATH = Path(__file__).resolve().parent / "data" / "market_implied_boc.csv"

_REQUIRED_COLUMNS = {"origin_date", "meeting_date", "p_cut", "p_hold", "p_hike", "method"}


def load_market_table(path: Path | None = None) -> pd.DataFrame:
    """Load the implied-probability table, indexed by normalised origin date.

    Raises
    ------
    FileNotFoundError
        If the table has not been built yet (run :mod:`build_table`).
    ValueError
        If required columns are missing or a row's probabilities do not sum to 1.
    """
    table_path = path if path is not None else DEFAULT_TABLE_PATH
    if not table_path.exists():
        raise FileNotFoundError(
            f"Market-implied table not found at {table_path}. Build it with "
            "`python -m boc_rate_decisions.a1_market_implied.build_table`."
        )
    table = pd.read_csv(table_path)
    missing = _REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"{table_path} is missing columns: {sorted(missing)}")

    totals = table[["p_cut", "p_hold", "p_hike"]].sum(axis=1)
    off = table.loc[(totals - 1.0).abs() > 1e-6, "origin_date"].tolist()
    if off:
        raise ValueError(f"Probabilities do not sum to 1 for origins: {off}")

    table["origin_date"] = pd.to_datetime(table["origin_date"]).dt.normalize()
    table["meeting_date"] = pd.to_datetime(table["meeting_date"]).dt.normalize()
    return table.set_index("origin_date").sort_index()


class MarketImpliedPredictor(Predictor):
    """Replay market-implied decision probabilities from a committed table.

    Parameters
    ----------
    table_path : Path or None
        Override the table location (used in tests). Defaults to the committed
        ``data/market_implied_boc.csv``.
    strict : bool
        When ``True`` (default) an origin absent from the table raises, so a
        missing quote can never be silently scored as a guess. Set ``False`` to
        skip such origins instead, returning no prediction for them.
    """

    def __init__(self, table_path: Path | None = None, *, strict: bool = True) -> None:
        self._table = load_market_table(table_path)
        self._strict = strict

    @property
    def predictor_id(self) -> str:
        """Stable identifier for this predictor."""
        return "market_implied_corra_futures"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Emit the table's distribution for this origin.

        Raises
        ------
        ValueError
            If the task is not a single-horizon categorical task, if its
            categories are not cut/hold/hike, or (in strict mode) if the origin
            has no committed quote.
        """
        if task.payload_type != "categorical":
            raise ValueError(
                f"{type(self).__name__} requires a categorical task; task '{task.task_id}' "
                f"declares payload_type='{task.payload_type}'."
            )
        if len(task.horizons) != 1:
            raise ValueError(f"{type(self).__name__} requires exactly one horizon; got {task.horizons}.")
        labels = [category.label for category in task.categories or []]
        if set(labels) != {"cut", "hold", "hike"}:
            raise ValueError(f"{type(self).__name__} expects cut/hold/hike categories; got {labels}.")

        origin = pd.Timestamp(context.as_of).normalize()
        if origin not in self._table.index:
            if self._strict:
                raise ValueError(
                    f"No market-implied quote for origin {origin:%Y-%m-%d}. Rebuild the table for this spec "
                    "or construct the predictor with strict=False."
                )
            return []

        row = self._table.loc[origin]
        probabilities = {label: float(row[f"p_{label}"]) for label in labels}
        offset = pd.tseries.frequencies.to_offset(task.frequency)
        forecast_date = (origin + offset * task.horizons[0]).to_pydatetime()

        return [
            Prediction(
                predictor_id=self.predictor_id,
                task_id=task.task_id,
                issued_at=datetime.now(tz=timezone.utc).replace(tzinfo=None),
                as_of=context.as_of,
                forecast_date=forecast_date,
                payload=CategoricalForecast(probabilities=probabilities),
                metadata={
                    "method": str(row["method"]),
                    "expected_move_pp": float(row["expected_move_pp"]),
                    "corra_now": float(row["corra_now"]),
                    "source": str(row["source"]),
                },
            )
        ]


__all__ = ["DEFAULT_TABLE_PATH", "MarketImpliedPredictor", "load_market_table"]
