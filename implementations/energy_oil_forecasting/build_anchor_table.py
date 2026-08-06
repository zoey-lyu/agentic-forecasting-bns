"""Build the deterministic AutoARIMA anchor lookup table.

The anchor is the statistical baseline every anchored-agent variant is measured
against.  It has to be *identical* across those variants, which rules out
re-running AutoARIMA inside each predictor: the sampler is unseeded by default,
so two runs on the same origin differ by roughly half a dollar on the median —
the same order of magnitude as the agent drift we are trying to measure.

So the anchor is computed once here with a fixed ``random_state``, flattened to
a lookup table keyed by origin and horizon, and persisted.  Predictors read the
table; nobody re-fits.

Run from this directory (the yfinance cache and prediction store are resolved
relative to the working directory, matching how the notebooks run)::

    uv run python build_anchor_table.py

Writes ``anchors/anchor_<spec_id>.json`` per spec.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import energy_oil_forecasting
import pandas as pd
import yaml
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation import MultiTargetBacktestSpec, cached_multi_backtest
from aieng.forecasting.evaluation.prediction import Prediction
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.numerical.darts_arima import DartsAutoARIMAPredictor
from energy_oil_forecasting.data import build_wti_service


SEED = 42
NUM_SAMPLES = 500
SPEC_FILES = ["energy_oil_backtest.yaml", "energy_oil_eval.yaml"]
OUT_DIR = Path(__file__).parent / "anchors"


class SeededAutoARIMA(DartsAutoARIMAPredictor):
    """Seeded AutoARIMA that reports per-origin progress.

    The seed and sample count go into ``predictor_id`` because the artifact
    cache key is derived from it: two runs with different seeds are different
    anchors and must not silently collide in the store.
    """

    def __init__(self) -> None:
        super().__init__(num_samples=NUM_SAMPLES, random_state=SEED)
        self._n = 0

    @property
    def predictor_id(self) -> str:
        """Return the seed-tagged identifier used as the cache key."""
        return f"darts_autoarima_seed{SEED}_n{NUM_SAMPLES}"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Forecast one origin, printing how long it took."""
        self._n += 1
        started = time.perf_counter()
        predictions = super().predict(task, context)
        elapsed = time.perf_counter() - started
        print(f"  [{self._n:>3}] {str(context.as_of)[:10]}  {elapsed:5.1f}s", flush=True)
        return predictions


def _flatten(predictions: list[Prediction], horizons: list[int], frequency: str) -> dict:
    """Group predictions into ``{as_of: {horizon: payload}}``.

    An origin does not always emit one prediction per horizon — the harness
    drops a step whose ``forecast_date`` has no observed actual to score
    against — so the horizon is recovered by stepping the origin forward with
    the task's own frequency offset rather than by pairing sorted lists.
    """
    offset = pd.tseries.frequencies.to_offset(frequency)

    table: dict[str, dict[str, dict]] = {}
    for prediction in predictions:
        as_of = pd.Timestamp(prediction.as_of)
        expected = {str((as_of + offset * h).date()): h for h in horizons}
        forecast_date = str(prediction.forecast_date)[:10]
        horizon = expected.get(forecast_date)
        if horizon is None:
            raise ValueError(f"{as_of.date()}: forecast_date {forecast_date} matches no horizon in {horizons}")
        table.setdefault(str(as_of.date()), {})[str(horizon)] = {
            "forecast_date": forecast_date,
            "point_forecast": prediction.payload.point_forecast,
            "quantiles": {str(q): v for q, v in prediction.payload.quantiles.items()},
        }
    return table


def main() -> None:
    """Run both specs and write one anchor table per spec."""
    OUT_DIR.mkdir(exist_ok=True)
    service = build_wti_service()
    spec_dir = Path(energy_oil_forecasting.__file__).parent / "specs"

    for spec_file in SPEC_FILES:
        with open(spec_dir / spec_file) as f:
            spec = MultiTargetBacktestSpec.model_validate(yaml.safe_load(f))

        tasks_by_id = {t.task_id: t for t in spec.tasks}
        print(f"\n=== {spec.spec_id} — {len(tasks_by_id)} task(s) ===", flush=True)

        results = cached_multi_backtest(predictor=SeededAutoARIMA(), spec=spec, data_service=service)

        table = {
            "spec_id": spec.spec_id,
            "seed": SEED,
            "num_samples": NUM_SAMPLES,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "tasks": {
                task_id: _flatten(
                    result.predictions,
                    list(tasks_by_id[task_id].horizons),
                    tasks_by_id[task_id].frequency,
                )
                for task_id, result in results.items()
            },
        }

        out_path = OUT_DIR / f"anchor_{spec.spec_id}.json"
        out_path.write_text(json.dumps(table, indent=2))
        n_origins = sum(len(t) for t in table["tasks"].values())
        print(f"wrote {out_path}  ({n_origins} origins)", flush=True)


if __name__ == "__main__":
    main()
