#!/usr/bin/env python3
"""Inspect BoC research evidence, prompts, forecasts, and smoke comparisons."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from aieng.forecasting.data import DataService, ForecastContext
from aieng.forecasting.evaluation import BacktestSpec, CategoricalForecast, Prediction, Predictor, backtest
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods import CategoricalFrequencyPredictor
from boc_rate_decisions.analysis import score_leaderboard
from boc_rate_decisions.analyst_agent import (
    BoCDecisionPromptBuilder,
    build_boc_agent_predictor,
    build_boc_basic_config,
    build_boc_research_predictor,
)
from boc_rate_decisions.data import BOND_YIELD_2YR_SERIES_ID, TARGET_RATE_SERIES_ID, build_boc_service
from boc_rate_decisions.predictors.logistic_baseline import BoCLogisticPredictor
from boc_rate_decisions.research import DEFAULT_RESEARCH_SOURCES, format_research_evidence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC_PATH = REPO_ROOT / "implementations/boc_rate_decisions/specs/boc_rate_direction_smoke.yaml"
DEFAULT_REPORTS_DIR = REPO_ROOT / "data/reports/boc_press_releases"
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
ADVANCED_MODEL = "gemini-3.5-flash"


class TwoYearYieldMarketProxy(Predictor):
    """Diagnostic yield-spread proxy; not a meeting-specific CORRA/OIS forecast."""

    @property
    def predictor_id(self) -> str:
        """Return a stable method identifier."""
        return "boc_two_year_yield_market_proxy"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Map the cutoff-visible two-year yield spread to cut/hold/hike probabilities."""
        if task.categories is None:
            raise ValueError("TwoYearYieldMarketProxy requires task categories.")
        rate = float(context.get_series(TARGET_RATE_SERIES_ID)["value"].iloc[-1])
        yield_2y = float(context.get_series(BOND_YIELD_2YR_SERIES_ID)["value"].iloc[-1])
        spread = yield_2y - rate
        scaled = spread / 0.25
        scores = np.array([-scaled, 1.5 - abs(scaled), scaled], dtype=float)
        weights = np.exp(scores - scores.max())
        values = weights / weights.sum()
        labels = [category.label for category in task.categories]
        probabilities = {label: float(value) for label, value in zip(labels, values, strict=True)}
        offset = pd.tseries.frequencies.to_offset(task.frequency)
        forecast_date = pd.Timestamp(context.as_of) + offset * task.horizons[0]
        return [
            Prediction(
                predictor_id=self.predictor_id,
                task_id=task.task_id,
                issued_at=datetime.now(tz=timezone.utc).replace(tzinfo=None),
                as_of=context.as_of,
                forecast_date=forecast_date.to_pydatetime(),
                payload=CategoricalForecast(probabilities=probabilities),
                metadata={"yield_2y": yield_2y, "policy_rate": rate, "yield_spread": spread},
            )
        ]


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Display BoC evidence, prompt, and optional forecasts.")
    parser.add_argument("--origin", default="2024-05-08", help="Forecast cutoff (YYYY-MM-DD).")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH, help="Backtest YAML spec.")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--max-documents", type=int, default=3)
    parser.add_argument("--max-chars-per-document", type=int, default=6_000)
    parser.add_argument("--live", action="store_true", help="Make one live research-grounded forecast.")
    parser.add_argument(
        "--model",
        choices=(DEFAULT_MODEL, ADVANCED_MODEL),
        default=DEFAULT_MODEL,
        help="Model for every agent invocation (default: %(default)s).",
    )
    parser.add_argument("--compare", action="store_true", help="Run the three-origin smoke comparison.")
    parser.add_argument(
        "--include-agents",
        action="store_true",
        help="Add both agents to --compare; this makes six model calls.",
    )
    return parser.parse_args()


def load_spec(spec_path: Path) -> BacktestSpec:
    """Load a backtest YAML spec."""
    with spec_path.open(encoding="utf-8") as file:
        return BacktestSpec.model_validate(yaml.safe_load(file))


def print_section(title: str, body: str) -> None:
    """Print a readable terminal section."""
    print(f"\n{title}\n{'=' * len(title)}\n{body}")


def run_comparison(args: argparse.Namespace, spec: BacktestSpec, service: DataService) -> None:
    """Run deterministic and optional agent methods across the smoke origins."""
    methods: dict[str, Predictor] = {
        "historical_frequency": CategoricalFrequencyPredictor(),
        "logistic_regression": BoCLogisticPredictor(),
        "two_year_market_proxy": TwoYearYieldMarketProxy(),
    }
    if args.include_agents:
        methods["quantitative_only_agent"] = build_boc_agent_predictor(
            build_boc_basic_config(model=args.model)
        )
        methods["research_grounded_agent"] = build_boc_research_predictor(
            model=args.model,
            max_documents=args.max_documents,
            max_chars_per_document=args.max_chars_per_document,
        )

    results = {}
    for name, method in methods.items():
        print(f"\nRunning {name}...")
        results[name] = backtest(method, spec, service)

    board = score_leaderboard(results, reference_id="historical_frequency")
    print_section("RPS leaderboard", board.to_string(index=False))

    resolved = service.get_series(spec.task.target_series_id, as_of=datetime.now())
    value_to_label = {category.value: category.label for category in spec.task.categories or []}
    outcome_by_date = {
        pd.Timestamp(ts).date(): value_to_label[float(value)]
        for ts, value in zip(resolved["timestamp"], resolved["value"], strict=True)
    }
    rows: list[dict[str, object]] = []
    for method_name, result in results.items():
        for prediction, rps in zip(result.predictions, result.scores, strict=True):
            meeting = pd.Timestamp(prediction.forecast_date).date()
            probabilities = prediction.payload.probabilities
            predicted = max(probabilities, key=lambda label: probabilities[label])
            actual = outcome_by_date.get(meeting)
            rows.append(
                {
                    "method": method_name,
                    "meeting": meeting,
                    "predicted": predicted,
                    "actual": actual,
                    "correct": predicted == actual,
                    "confidence": round(probabilities[predicted], 4),
                    "p_cut": round(probabilities["cut"], 4),
                    "p_hold": round(probabilities["hold"], 4),
                    "p_hike": round(probabilities["hike"], 4),
                    "rps": round(rps, 4),
                }
            )
    comparison = pd.DataFrame(rows).sort_values(["meeting", "method"])
    print_section("Predicted versus actual outcomes", comparison.to_string(index=False))


def main() -> None:
    """Display one prompt, optionally predict, and optionally compare methods."""
    args = parse_args()
    if args.include_agents and not args.compare:
        raise ValueError("--include-agents requires --compare")
    if not args.reports_dir.is_dir():
        raise FileNotFoundError(
            f"BoC release cache not found at {args.reports_dir.resolve()}. "
            "Run: uv run python scripts/fetch_boc_press_releases.py --year 2024"
        )

    spec = load_spec(args.spec)
    task = spec.task
    origin = datetime.fromisoformat(args.origin)
    service = build_boc_service(reports_dir=args.reports_dir)
    context = service.context(origin)
    print(f"Using agent model: {args.model}")

    evidence = format_research_evidence(
        context,
        sources=DEFAULT_RESEARCH_SOURCES,
        max_documents=args.max_documents,
        max_chars_per_document=args.max_chars_per_document,
    )
    print_section("Cutoff-visible evidence", json.dumps(evidence, indent=2, ensure_ascii=False))

    builder = BoCDecisionPromptBuilder(
        document_sources=DEFAULT_RESEARCH_SOURCES,
        max_documents=args.max_documents,
        max_chars_per_document=args.max_chars_per_document,
    )
    print_section("Generated prompt", builder(task=task, context=context))

    if args.live:
        predictor = build_boc_research_predictor(
            model=args.model,
            max_documents=args.max_documents,
            max_chars_per_document=args.max_chars_per_document,
        )
        predictions = predictor.predict(task, context)
        if not predictions:
            raise RuntimeError("The research agent returned no prediction.")
        prediction = predictions[0]
        probabilities = prediction.payload.probabilities
        result = {
            "predictor_id": prediction.predictor_id,
            "as_of": str(prediction.as_of),
            "forecast_date": str(prediction.forecast_date),
            "probabilities": probabilities,
            "predicted_outcome": max(probabilities, key=lambda label: probabilities[label]),
            "metadata": prediction.metadata,
        }
        print_section("Live model prediction", json.dumps(result, indent=2, default=str))
    else:
        print("\nLive prediction skipped. Add --live to call the selected model.")

    if args.compare:
        run_comparison(args, spec, service)


if __name__ == "__main__":
    main()
