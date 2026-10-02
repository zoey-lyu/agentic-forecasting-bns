"""Run Step 2: score P(move) and test whether the market's move odds are biased.

Builds the binary turn instrument over the same 28 meetings Steps 0-1 used, on
the same cutoff-safe arms, and then asks the calibration question A3 raised:
does the market systematically over-price the chance of a move? If it does, in
both windows, a recalibration fitted on one window and scored on the other is
the cheapest thing that could beat the market — and the pre-registration only
allows that comparison out of sample, in both directions.

Usage
-----
``python -m boc_rate_decisions.turn_detection.run_step2``
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
from ..a3_t2_horizon.build_table_t2 import DEFAULT_SPEC as SPEC_T2
from ..a3_t2_horizon.build_table_t2 import origin_triples as t2_origin_triples
from ..a3_t2_horizon.data_t2 import DIRECTION_T2_SERIES_ID, build_boc_t2_service
from ..data import DIRECTION_SERIES_ID, build_boc_service
from .move_task import (
    brier_scores,
    derive_move_events,
    historical_move_frequency,
    move_probability,
    murphy_decomposition,
    platt_scale,
)
from .run_steps import SPEC_2023_2024, SPEC_2025_2026, USE_CASE_DIR, _CachedContext, _market_probabilities


DATA_DIR = Path(__file__).resolve().parent / "data"


def _window_move_probabilities(
    spec_path: Path, cache: Path, service: Any, directions: pd.DataFrame
) -> dict[str, pd.Series]:
    """Project every cutoff-safe arm onto P(move) for one window."""
    spec = EvalSpec.model_validate(yaml.safe_load(spec_path.read_text()))
    pairs = origins_from_spec(spec_path)
    origins = pd.Series({meeting: origin for origin, meeting in pairs}, name="origin")

    predictor = BoCTextLogisticPredictor(use_text=False)
    logistic_rows: dict[pd.Timestamp, dict[str, float]] = {}
    for origin, meeting in pairs:
        context = _CachedContext(service.context(as_of=origin.to_pydatetime()))
        (prediction,) = predictor.predict(spec.task, cast(ForecastContext, context))
        payload = prediction.payload
        if not isinstance(payload, CategoricalForecast):
            raise TypeError(f"Expected a categorical forecast for {meeting:%Y-%m-%d}.")
        logistic_rows[meeting] = {label: float(payload.probabilities[label]) for label in CATEGORY_ORDER}
    logistic = pd.DataFrame.from_dict(logistic_rows, orient="index").loc[:, CATEGORY_ORDER].sort_index()

    return {
        "market_implied": move_probability(_market_probabilities(spec_path, cache)),
        "boc_logistic_macro": move_probability(logistic),
        "historical_frequency": historical_move_frequency(directions, origins),
    }


def _t2_panel(args: argparse.Namespace, as_of: pd.Timestamp) -> pd.DataFrame:
    """Secondary panel: the same question one meeting further out.

    Only the 2025-26 origins are available at this horizon, so it is 13
    meetings rather than 28 and carries no pre-registered claim. It is here
    because A3 found the market's turn edge decays with horizon, which makes
    t+2 the place where an agent has the most room.
    """
    service = build_boc_t2_service(args.statcan_cache_dir, args.fred_cache_dir, as_of=as_of)
    shifted = service.get_series(DIRECTION_T2_SERIES_ID, as_of=as_of)
    outcomes = pd.Series(
        (shifted["value"].astype(float) != 0.0).astype(float).to_numpy(),
        index=pd.to_datetime(shifted["released_at"]).to_numpy(),
    )

    spec = EvalSpec.model_validate(yaml.safe_load(SPEC_T2.read_text()))
    market_table = pd.read_csv(
        USE_CASE_DIR / "a3_t2_horizon" / "data" / "market_implied_t2.csv", parse_dates=["target_meeting"]
    ).set_index("target_meeting")
    market = move_probability(
        market_table.rename(columns={f"p_{label}": label for label in CATEGORY_ORDER}).loc[:, CATEGORY_ORDER]
    )

    predictor = BoCTextLogisticPredictor(use_text=False)
    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for origin, _, target in t2_origin_triples(SPEC_T2):
        context = _CachedContext(service.context(as_of=origin.to_pydatetime()))
        (prediction,) = predictor.predict(spec.task, cast(ForecastContext, context))
        payload = prediction.payload
        if not isinstance(payload, CategoricalForecast):
            raise TypeError(f"Expected a categorical forecast for {target:%Y-%m-%d}.")
        rows[target] = {label: float(payload.probabilities[label]) for label in CATEGORY_ORDER}
    logistic = move_probability(pd.DataFrame.from_dict(rows, orient="index").loc[:, CATEGORY_ORDER].sort_index())

    return pd.DataFrame(
        [
            {"horizon": "t+2", "predictor_id": arm, **murphy_decomposition(series, outcomes)}
            for arm, series in (("market_implied", market), ("boc_logistic_macro", logistic))
        ]
    )


def _report_step2(
    board: pd.DataFrame,
    calibration: pd.DataFrame,
    market_bias: dict[str, float],
    bias_supported: bool,
    recalibration: pd.DataFrame | None,
) -> None:
    """Print the leaderboard, the calibration table and both pre-registered verdicts."""
    print("=== Step 2: P(move), Brier (lower is better) ===")
    print(board.to_string(index=False))
    print("\n=== calibration (Murphy decomposition) ===")
    print(calibration.to_string(index=False))

    print("\n=== is the market's P(move) biased high? (pre-registered: must hold in BOTH windows) ===")
    for window, value in market_bias.items():
        print(f"  {window}: mean forecast - observed rate = {value:+.4f}")
    print(f"  VERDICT: {'SUPPORTED' if bias_supported else 'NOT SUPPORTED'}")

    if recalibration is not None:
        print("\n=== out-of-sample recalibration (must improve in BOTH directions) ===")
        print(recalibration.to_string(index=False))
        verdict = "BEATS THE MARKET" if bool(recalibration["improved"].all()) else "DOES NOT BEAT THE MARKET"
        print(f"  VERDICT: {verdict}")
    else:
        print("\nrecalibration test skipped: the bias claim was not supported in both windows.")


def main() -> None:
    """Score P(move), report calibration, and run the out-of-sample recalibration test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default="2026-09-21", help="Cutoff for reading realised outcomes.")
    parser.add_argument("--statcan-cache-dir", type=Path, default=USE_CASE_DIR / "data" / "statcan")
    parser.add_argument("--fred-cache-dir", type=Path, default=USE_CASE_DIR / "data" / "fred")
    args = parser.parse_args()

    as_of = pd.Timestamp(args.as_of)
    service = build_boc_service(statcan_cache_dir=args.statcan_cache_dir, fred_cache_dir=args.fred_cache_dir)
    directions = service.get_series(DIRECTION_SERIES_ID, as_of=as_of)
    moves = derive_move_events(directions)
    outcomes = pd.Series(moves["value"].astype(float).to_numpy(), index=pd.to_datetime(moves["timestamp"]).to_numpy())

    windows = {
        "2023-2024": (SPEC_2023_2024, DATA_DIR / "market_implied_2023_2024.csv"),
        "2025-2026": (SPEC_2025_2026, DATA_DIR / "market_implied_2025_2026.csv"),
    }
    probabilities = {
        window: _window_move_probabilities(spec, cache, service, directions)
        for window, (spec, cache) in windows.items()
    }
    probabilities["pooled"] = {
        arm: pd.concat([probabilities[w][arm] for w in windows]).sort_index() for arm in probabilities["2023-2024"]
    }

    # ---- leaderboard -------------------------------------------------------
    rows: list[dict[str, object]] = []
    for window, arms in probabilities.items():
        market_brier = float(brier_scores(arms["market_implied"], outcomes).mean())
        for arm, series in arms.items():
            brier = float(brier_scores(series, outcomes).mean())
            rows.append(
                {
                    "window": window,
                    "predictor_id": arm,
                    "n_meetings": int(len(series)),
                    "brier": round(brier, 4),
                    "skill_vs_market": round(1.0 - brier / market_brier, 4) if market_brier > 0 else None,
                }
            )
    board = pd.DataFrame(rows)
    board.to_csv(DATA_DIR / "step2_move_leaderboard.csv", index=False)

    # ---- calibration -------------------------------------------------------
    calibration = pd.DataFrame(
        [
            {"window": window, "predictor_id": arm, **murphy_decomposition(series, outcomes)}
            for window, arms in probabilities.items()
            for arm, series in arms.items()
        ]
    )
    calibration.to_csv(DATA_DIR / "step2_calibration.csv", index=False)

    market_bias = {
        window: float(
            calibration[(calibration["window"] == window) & (calibration["predictor_id"] == "market_implied")][
                "bias"
            ].iloc[0]
        )
        for window in windows
    }
    bias_supported = all(value > 0 for value in market_bias.values())

    # ---- out-of-sample recalibration --------------------------------------
    recalibration = None
    if bias_supported:
        pairs = [("2023-2024", "2025-2026"), ("2025-2026", "2023-2024")]
        recal_rows = []
        for train, test in pairs:
            corrected = platt_scale(
                probabilities[train]["market_implied"], outcomes, probabilities[test]["market_implied"]
            )
            raw_brier = float(brier_scores(probabilities[test]["market_implied"], outcomes).mean())
            new_brier = float(brier_scores(corrected, outcomes).mean())
            recal_rows.append(
                {
                    "fitted_on": train,
                    "scored_on": test,
                    "market_brier": round(raw_brier, 4),
                    "recalibrated_brier": round(new_brier, 4),
                    "improved": bool(new_brier < raw_brier),
                }
            )
        recalibration = pd.DataFrame(recal_rows)
        recalibration.to_csv(DATA_DIR / "step2_recalibration.csv", index=False)

    _report_step2(board, calibration, market_bias, bias_supported, recalibration)

    t2 = _t2_panel(args, as_of)
    t2.to_csv(DATA_DIR / "step2_t2_panel.csv", index=False)
    print("\n=== secondary: the same question at t+2 (13 meetings, 2025-26 only, no claim attached) ===")
    print(t2.to_string(index=False))

    print(f"\nwrote step2_move_leaderboard.csv, step2_calibration.csv, step2_t2_panel.csv in {DATA_DIR}")


if __name__ == "__main__":
    main()
