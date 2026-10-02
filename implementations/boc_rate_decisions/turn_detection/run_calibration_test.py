"""Test whether fixing the logistic's calibration alone can beat the market.

No LLM is involved. Step 2 left a specific arithmetic opening. At the t+2 horizon the four-feature
logistic already separates live meetings from quiet ones *better* than the
market does (resolution 0.054 against 0.046); it loses only because its stated
probabilities are dishonest (reliability 0.036 against 0.009). Give it the
market's honesty while keeping its own discrimination and it would score 0.132
against the market's 0.155.

This script tests whether that is reachable, by fitting a two-parameter Platt
correction on one window's forecasts and scoring it on the other — both
directions, never in sample. The rule (``PREREGISTRATION.md``, 2026-09-21
addendum) is that the calibrated logistic counts as beating the market only if
it wins in **both** directions.

The market gets the same treatment as a **negative control**: Step 2 found it
already calibrated, so a correction should barely move its score. If instead
the procedure "improves" the market a lot, it is fitting noise and the
logistic result has to be thrown out with it.

Both horizons are reported, because the opening is a claim about horizon: at
t+1 the market's calibration *and* resolution beat the logistic, so nothing
should work there.

Usage
-----
``python -m boc_rate_decisions.turn_detection.run_calibration_test``
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation import EvalSpec
from aieng.forecasting.evaluation.prediction import CategoricalForecast

from ..a1_market_implied.build_table import origins_from_spec
from ..a1_market_implied.leaderboard import CATEGORY_ORDER
from ..a2_text_features.text_logistic import BoCTextLogisticPredictor
from ..a3_t2_horizon.build_table_t2 import origin_triples
from ..a3_t2_horizon.data_t2 import DIRECTION_T2_SERIES_ID, build_boc_t2_service
from .move_task import brier_scores, move_probability, murphy_decomposition, platt_scale
from .run_steps import SPEC_2023_2024, SPEC_2025_2026, USE_CASE_DIR, _CachedContext, _market_probabilities


DATA_DIR = Path(__file__).resolve().parent / "data"
SPECS_DIR = Path(__file__).resolve().parent / "specs"

T2_SPECS = {
    "2023-2024": (SPECS_DIR / "boc_rate_direction_t2_2023_2024.yaml", DATA_DIR / "market_implied_t2_2023_2024.csv"),
    "2025-2026": (
        USE_CASE_DIR / "a3_t2_horizon" / "specs" / "boc_rate_direction_t2_eval.yaml",
        USE_CASE_DIR / "a3_t2_horizon" / "data" / "market_implied_t2.csv",
    ),
}
T1_SPECS = {
    "2023-2024": (SPEC_2023_2024, DATA_DIR / "market_implied_2023_2024.csv"),
    "2025-2026": (SPEC_2025_2026, DATA_DIR / "market_implied_2025_2026.csv"),
}


def _logistic_move_probabilities(spec_path: Path, service: Any, *, t2: bool) -> pd.Series:
    """Run the macro logistic at every origin and project onto P(move)."""
    spec = EvalSpec.model_validate(yaml.safe_load(spec_path.read_text()))
    predictor = BoCTextLogisticPredictor(use_text=False)
    pairs = (
        [(origin, target) for origin, _, target in origin_triples(spec_path)] if t2 else origins_from_spec(spec_path)
    )

    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for origin, scored_on in pairs:
        context = _CachedContext(service.context(as_of=origin.to_pydatetime()))
        (prediction,) = predictor.predict(spec.task, cast(ForecastContext, context))
        payload = prediction.payload
        if not isinstance(payload, CategoricalForecast):
            raise TypeError(f"Expected a categorical forecast for {scored_on:%Y-%m-%d}.")
        rows[scored_on] = {label: float(payload.probabilities[label]) for label in CATEGORY_ORDER}
    return move_probability(pd.DataFrame.from_dict(rows, orient="index").loc[:, CATEGORY_ORDER].sort_index())


def _market_move_probabilities(cache: Path, *, t2: bool) -> pd.Series:
    """Load a committed market table and project onto P(move)."""
    if t2:
        table = pd.read_csv(cache, parse_dates=["target_meeting"]).set_index("target_meeting")
        frame = table.rename(columns={f"p_{label}": label for label in CATEGORY_ORDER}).loc[:, CATEGORY_ORDER]
        return move_probability(frame.sort_index())
    return move_probability(_market_probabilities(Path(str(cache).replace("data/", "data/")), cache))


def _collect(horizon: str, specs: dict[str, tuple[Path, Path]], service: Any) -> dict[str, dict[str, pd.Series]]:
    """Gather market and logistic P(move) per window for one horizon."""
    t2 = horizon == "t+2"
    out: dict[str, dict[str, pd.Series]] = {}
    for window, (spec_path, cache) in specs.items():
        market = (
            _market_move_probabilities(cache, t2=True)
            if t2
            else move_probability(_market_probabilities(spec_path, cache))
        )
        out[window] = {
            "market_implied": market,
            "boc_logistic_macro": _logistic_move_probabilities(spec_path, service, t2=t2),
        }
    return out


def _calibration_rows(
    horizon: str, arms: dict[str, dict[str, pd.Series]], outcomes: pd.Series
) -> list[dict[str, object]]:
    """Fit on one window, score on the other, both directions, for every arm."""
    rows: list[dict[str, object]] = []
    for train, test in (("2023-2024", "2025-2026"), ("2025-2026", "2023-2024")):
        for arm in ("boc_logistic_macro", "market_implied"):
            raw_test = arms[test][arm]
            corrected = platt_scale(arms[train][arm], outcomes, raw_test)
            market_raw = float(brier_scores(arms[test]["market_implied"], outcomes).mean())
            rows.append(
                {
                    "horizon": horizon,
                    "arm": arm,
                    "fitted_on": train,
                    "scored_on": test,
                    "raw_brier": round(float(brier_scores(raw_test, outcomes).mean()), 4),
                    "calibrated_brier": round(float(brier_scores(corrected, outcomes).mean()), 4),
                    "market_brier": round(market_raw, 4),
                    "beats_market": bool(float(brier_scores(corrected, outcomes).mean()) < market_raw),
                }
            )
    return rows


def main() -> None:
    """Run the out-of-sample calibration test at both horizons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default="2026-09-21")
    parser.add_argument("--statcan-cache-dir", type=Path, default=USE_CASE_DIR / "data" / "statcan")
    parser.add_argument("--fred-cache-dir", type=Path, default=USE_CASE_DIR / "data" / "fred")
    args = parser.parse_args()

    as_of = pd.Timestamp(args.as_of)
    service = build_boc_t2_service(args.statcan_cache_dir, args.fred_cache_dir, as_of=as_of)

    shifted = service.get_series(DIRECTION_T2_SERIES_ID, as_of=as_of)
    outcomes_t2 = pd.Series(
        (shifted["value"].astype(float) != 0.0).astype(float).to_numpy(),
        index=pd.to_datetime(shifted["released_at"]).to_numpy(),
    )
    directions = service.get_series("boc_rate_decision_direction", as_of=as_of)
    outcomes_t1 = pd.Series(
        (directions["value"].astype(float) != 0.0).astype(float).to_numpy(),
        index=pd.to_datetime(directions["timestamp"]).to_numpy(),
    )

    rows: list[dict[str, object]] = []
    baseline: list[dict[str, object]] = []
    for horizon, specs, outcomes in (("t+2", T2_SPECS, outcomes_t2), ("t+1", T1_SPECS, outcomes_t1)):
        arms = _collect(horizon, specs, service)
        rows.extend(_calibration_rows(horizon, arms, outcomes))
        for window, per_arm in arms.items():
            for arm, series in per_arm.items():
                baseline.append(
                    {"horizon": horizon, "window": window, "arm": arm, **murphy_decomposition(series, outcomes)}
                )

    results = pd.DataFrame(rows)
    results.to_csv(DATA_DIR / "calibration_test.csv", index=False)
    pd.DataFrame(baseline).to_csv(DATA_DIR / "calibration_baseline.csv", index=False)

    print("=== raw scores by horizon and window (P(move), Brier) ===")
    print(pd.DataFrame(baseline).to_string(index=False))
    print("\n=== out-of-sample calibration (fit one window, score the other) ===")
    print(results.to_string(index=False))

    for horizon in ("t+2", "t+1"):
        logistic = results[(results["horizon"] == horizon) & (results["arm"] == "boc_logistic_macro")]
        market = results[(results["horizon"] == horizon) & (results["arm"] == "market_implied")]
        passed = bool(logistic["beats_market"].all())
        control_shift = float((market["calibrated_brier"] - market["raw_brier"]).abs().max())
        print(f"\n{horizon}: calibrated logistic beats the market in both directions? {'YES' if passed else 'NO'}")
        print(f"  negative control - recalibrating the market moves its Brier by at most {control_shift:.4f}")

    print(f"\nwrote calibration_test.csv and calibration_baseline.csv in {DATA_DIR}")


if __name__ == "__main__":
    main()
