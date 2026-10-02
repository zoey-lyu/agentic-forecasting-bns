"""Run Steps 0 and 1 of the turn-detection plan and apply the pre-registered rules.

Step 0 asks whether a router is worth building: how much could a perfect one
win over the market (the oracle), and how much of that does a fixed 0.5 blend
already take without any agent at all?

Step 1 asks whether the finding that motivates the router replicates. A3 found,
on 2025-26, that the market is the *worst* of the three cutoff-safe arms on
meetings where the Bank held and the *best* on meetings where it moved. This
re-runs that split on 2023-24 — 14 meetings nobody had scored, in a different
regime, containing the only two hikes available anywhere in this project.

Only cutoff-safe arms appear here. An LLM has read 2023-24, so no text arm is
valid on that window and none is run.

Thresholds and verdict wording come from ``PREREGISTRATION.md``, written before
any number below existed. Nothing in this script may be tuned.

Usage
-----
``python -m boc_rate_decisions.turn_detection.run_steps``
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
from aieng.forecasting.evaluation.predictor import Predictor

from ..a1_market_implied.build_table import build as build_market_table
from ..a1_market_implied.build_table import origins_from_spec
from ..a1_market_implied.leaderboard import CATEGORY_ORDER, climatology_probabilities, per_meeting_rps
from ..a2_text_features.text_logistic import BoCTextLogisticPredictor
from ..data import DIRECTION_SERIES_ID, build_boc_service
from .ceiling import ceiling_summary, linear_pool, router_requirement, weight_curve


USE_CASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"
SPEC_2023_2024 = Path(__file__).resolve().parent / "specs" / "boc_rate_direction_2023_2024.yaml"
SPEC_2025_2026 = USE_CASE_DIR / "specs" / "boc_rate_direction_eval.yaml"

#: Pre-registered thresholds. See PREREGISTRATION.md; do not edit.
CAPTURED_SHARE_KILL = 0.70
HEADROOM_KILL = 0.02
MAX_DROPPED_ORIGINS = 3


class _CachedContext:
    """Memoise ``get_series`` per origin so repeated arms re-read nothing."""

    def __init__(self, context: Any) -> None:
        self._context = context
        self._cache: dict[str, pd.DataFrame] = {}

    @property
    def as_of(self) -> Any:
        """The wrapped context's information cutoff."""
        return self._context.as_of

    def get_series(self, series_id: str) -> pd.DataFrame:
        """Return the cutoff-filtered series, reading it at most once."""
        if series_id not in self._cache:
            self._cache[series_id] = self._context.get_series(series_id)
        return self._cache[series_id]


def _predictor_probabilities(
    predictor: Predictor, spec: EvalSpec, contexts: dict[pd.Timestamp, _CachedContext]
) -> pd.DataFrame:
    """Run one predictor at every origin, indexed by the meeting it forecasts."""
    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for meeting, context in contexts.items():
        (prediction,) = predictor.predict(spec.task, cast(ForecastContext, context))
        payload = prediction.payload
        if not isinstance(payload, CategoricalForecast):
            raise TypeError(f"Expected a categorical forecast for {meeting:%Y-%m-%d}.")
        rows[meeting] = {label: float(payload.probabilities[label]) for label in CATEGORY_ORDER}
    return pd.DataFrame.from_dict(rows, orient="index").loc[:, CATEGORY_ORDER].sort_index()


def _market_probabilities(spec_path: Path, cache_path: Path) -> pd.DataFrame:
    """Load or build the market-implied table for a window."""
    if not cache_path.exists():
        build_market_table(spec_path).to_csv(cache_path, index=False)
    table = pd.read_csv(cache_path, parse_dates=["meeting_date"]).set_index("meeting_date")
    return table.rename(columns={f"p_{label}": label for label in CATEGORY_ORDER}).loc[:, CATEGORY_ORDER].sort_index()


def _window_scores(
    spec_path: Path,
    market_cache: Path,
    service: Any,
    outcomes: pd.Series,
    directions: pd.DataFrame,
) -> tuple[dict[str, pd.Series], pd.DataFrame, pd.DataFrame]:
    """Score every cutoff-safe arm on one window.

    Returns ``(per-meeting scores, market probabilities, logistic probabilities)``.
    """
    spec = EvalSpec.model_validate(yaml.safe_load(spec_path.read_text()))
    pairs = origins_from_spec(spec_path)
    contexts = {meeting: _CachedContext(service.context(as_of=origin.to_pydatetime())) for origin, meeting in pairs}
    origins = pd.Series({meeting: origin for origin, meeting in pairs}, name="origin")

    market = _market_probabilities(spec_path, market_cache)
    logistic = _predictor_probabilities(BoCTextLogisticPredictor(use_text=False), spec, contexts)
    climatology = climatology_probabilities(directions, origins)

    scores = {
        "market_implied": per_meeting_rps(market, outcomes),
        "boc_logistic_macro": per_meeting_rps(logistic, outcomes),
        "categorical_frequency": per_meeting_rps(climatology, outcomes),
        "blend_0.5": per_meeting_rps(linear_pool(market, logistic, 0.5), outcomes),
    }
    return scores, market, logistic


def _subset_table(scores: dict[str, pd.Series], outcomes: pd.Series, window: str) -> pd.DataFrame:
    """Mean RPS per arm, split into meetings where the Bank moved and held."""
    index = pd.DataFrame(scores).index
    realised = outcomes.reindex(index)
    rows: list[dict[str, object]] = []
    for subset, subset_index in (("moves", index[realised != 0.0]), ("holds", index[realised == 0.0])):
        for predictor_id, series in scores.items():
            rows.append(
                {
                    "window": window,
                    "subset": subset,
                    "predictor_id": predictor_id,
                    "n_meetings": len(subset_index),
                    "mean_rps": round(float(series.loc[subset_index].mean()), 4) if len(subset_index) else None,
                }
            )
    return pd.DataFrame(rows)


def _step1_verdict(subsets: pd.DataFrame) -> tuple[str, list[str]]:
    """Apply the pre-registered replication rule to the 2023-24 window."""
    arms = ["market_implied", "boc_logistic_macro", "categorical_frequency"]
    notes: list[str] = []

    holds = subsets[(subsets["subset"] == "holds") & (subsets["predictor_id"].isin(arms))]
    moves = subsets[(subsets["subset"] == "moves") & (subsets["predictor_id"].isin(arms))]
    if holds["mean_rps"].isna().any() or moves["mean_rps"].isna().any():
        return "INCONCLUSIVE (a subset is empty)", notes

    worst_on_holds = holds.loc[holds["mean_rps"].idxmax(), "predictor_id"]
    best_on_moves = moves.loc[moves["mean_rps"].idxmin(), "predictor_id"]
    condition_a = worst_on_holds == "market_implied"
    condition_b = best_on_moves == "market_implied"

    notes.append(f"(a) worst on holds = {worst_on_holds} -> {'PASS' if condition_a else 'FAIL'}")
    notes.append(f"(b) best on moves  = {best_on_moves} -> {'PASS' if condition_b else 'FAIL'}")
    return ("REPLICATED" if condition_a and condition_b else "FALSIFIED"), notes


def _report(ceiling: pd.DataFrame, step0: str, subsets: pd.DataFrame, verdict: str, notes: list[str]) -> None:
    """Print both steps' tables and their pre-registered verdicts."""
    print("\n=== Step 0: how much can a router win? ===")
    print(ceiling.to_string(index=False))
    print(f"\nVERDICT (pre-registered, pooled window): {step0}")

    print("\n=== Step 1: does the structural finding replicate on 2023-2024? ===")
    print(subsets[subsets["window"] == "2023-2024"].to_string(index=False))
    for note in notes:
        print(f"  {note}")
    print(f"\nVERDICT (pre-registered, 2023-2024 alone): {verdict}")

    print("\n(reference) same split on the other windows:")
    print(subsets[subsets["window"] != "2023-2024"].to_string(index=False))


def _step0_verdict(ceiling: pd.DataFrame) -> str:
    """Apply the pre-registered kill rules to the pooled window."""
    pooled = ceiling[ceiling["window"] == "pooled"].iloc[0]
    headroom = float(pooled["headroom"])
    captured = pooled["captured_share"]
    if headroom < HEADROOM_KILL:
        return f"DROP router direction - headroom {headroom:.4f} < {HEADROOM_KILL} (too little to win)"
    if captured is not None and float(captured) >= CAPTURED_SHARE_KILL:
        return f"DROP router direction - a fixed blend already captures {float(captured):.0%} of the ceiling"
    share = "n/a" if captured is None else f"{float(captured):.0%}"
    return f"PROCEED to Step 2 - headroom {headroom:.4f}, fixed blend captures {share}"


def _score_all_windows(
    service: Any, outcomes: pd.Series, directions: pd.DataFrame
) -> tuple[dict[str, dict[str, pd.Series]], dict[str, tuple[pd.DataFrame, pd.DataFrame]]]:
    """Score every window, report dropped origins, and add the pooled window."""
    windows = {
        "2023-2024": (SPEC_2023_2024, DATA_DIR / "market_implied_2023_2024.csv"),
        "2025-2026": (SPEC_2025_2026, DATA_DIR / "market_implied_2025_2026.csv"),
    }
    all_scores: dict[str, dict[str, pd.Series]] = {}
    probabilities: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for window, (spec_path, cache) in windows.items():
        scores, market, logistic = _window_scores(spec_path, cache, service, outcomes, directions)
        all_scores[window] = scores
        probabilities[window] = (market, logistic)
        declared = len(yaml.safe_load(spec_path.read_text())["origin_dates"])
        dropped = declared - len(scores["market_implied"])
        if dropped:
            print(f"{window}: {dropped} of {declared} origins dropped (no futures quote or no resolved outcome)")
            if window == "2023-2024" and dropped > MAX_DROPPED_ORIGINS:
                print(f"  more than {MAX_DROPPED_ORIGINS} dropped - Step 1 is INCONCLUSIVE by pre-registration")

    all_scores["pooled"] = {
        name: pd.concat([all_scores[w][name] for w in windows]).sort_index() for name in all_scores["2023-2024"]
    }
    return all_scores, probabilities


def main() -> None:
    """Run both steps and print the verdicts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default="2026-09-20", help="Cutoff for reading realised outcomes.")
    parser.add_argument("--statcan-cache-dir", type=Path, default=USE_CASE_DIR / "data" / "statcan")
    parser.add_argument("--fred-cache-dir", type=Path, default=USE_CASE_DIR / "data" / "fred")
    args = parser.parse_args()

    as_of = pd.Timestamp(args.as_of)
    service = build_boc_service(statcan_cache_dir=args.statcan_cache_dir, fred_cache_dir=args.fred_cache_dir)
    directions = service.get_series(DIRECTION_SERIES_ID, as_of=as_of)
    outcomes = pd.Series(
        directions["value"].astype(float).to_numpy(),
        index=pd.to_datetime(directions["timestamp"]).to_numpy(),
    )

    all_scores, probabilities = _score_all_windows(service, outcomes, directions)

    per_meeting = pd.DataFrame(all_scores["pooled"])
    per_meeting.index.name = "meeting_date"
    per_meeting.to_csv(DATA_DIR / "per_meeting_rps_pooled.csv")

    ceiling = pd.DataFrame(
        [
            ceiling_summary(scores["market_implied"], scores["boc_logistic_macro"], scores["blend_0.5"], window=window)
            for window, scores in all_scores.items()
        ]
    )
    ceiling.to_csv(DATA_DIR / "step0_ceiling.csv", index=False)

    curves = []
    for window, (market, logistic) in probabilities.items():
        curve = weight_curve(market, logistic, outcomes)
        curve.insert(0, "window", window)
        curves.append(curve)
    pd.concat(curves).to_csv(DATA_DIR / "weight_curve.csv", index=False)

    subsets = pd.concat(
        [_subset_table(scores, outcomes, window) for window, scores in all_scores.items()], ignore_index=True
    )
    subsets.to_csv(DATA_DIR / "step1_replication.csv", index=False)
    verdict, notes = _step1_verdict(subsets[subsets["window"] == "2023-2024"])

    grid, headline = router_requirement(subsets)
    grid.to_csv(DATA_DIR / "router_requirement.csv")

    _report(ceiling, _step0_verdict(ceiling), subsets, verdict, notes)
    print("\nwhat a router would have to achieve (pooled; descriptive, not pre-registered):")
    print(f"  market alone {headline['market_alone']}, perfect router {headline['perfect_router']}")
    print(
        f"  missing a move costs {headline['miss_cost']:+.4f} RPS, a false alarm {headline['false_alarm_cost']:+.4f}"
        f" - misses are {headline['miss_to_false_alarm_ratio']}x more expensive"
    )
    print(grid.to_string())
    print(
        f"\nwrote step0_ceiling.csv, step1_replication.csv, weight_curve.csv and per_meeting_rps_pooled.csv in {DATA_DIR}"
    )


if __name__ == "__main__":
    main()
