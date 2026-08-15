#!/usr/bin/env python3
"""Inspect BoC research evidence and prompts, with an optional live forecast.

Run from the repository root with ``uv run python`` or execute this file after
the workspace has been installed. The model call is disabled unless ``--live``
is supplied.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import yaml
from aieng.forecasting.evaluation import BacktestSpec
from aieng.forecasting.evaluation.task import ForecastingTask
from boc_rate_decisions.analyst_agent import BoCDecisionPromptBuilder, build_boc_research_predictor
from boc_rate_decisions.data import build_boc_service
from boc_rate_decisions.research import DEFAULT_RESEARCH_SOURCES, format_research_evidence


DEFAULT_SPEC_PATH = Path(__file__).parent / "specs" / "boc_rate_direction_smoke.yaml"
DEFAULT_REPORTS_DIR = Path("data/reports/boc_press_releases")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Display cutoff-visible BoC evidence and the generated agent prompt."
    )
    parser.add_argument(
        "--origin",
        default="2024-05-08",
        help="Forecast cutoff in YYYY-MM-DD format (default: %(default)s).",
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH, help="Backtest YAML containing the task.")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Directory containing cached BoC press-release artifacts.",
    )
    parser.add_argument("--max-documents", type=int, default=3, help="Maximum number of recent documents.")
    parser.add_argument(
        "--max-chars-per-document",
        type=int,
        default=6_000,
        help="Per-document prompt character budget.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Make a live model prediction after displaying evidence and prompt.",
    )
    return parser.parse_args()


def load_task(spec_path: Path) -> ForecastingTask:
    """Load the forecasting task from a backtest YAML spec."""
    with spec_path.open(encoding="utf-8") as file:
        return BacktestSpec.model_validate(yaml.safe_load(file)).task


def print_section(title: str, body: str) -> None:
    """Print a readable terminal section."""
    rule = "=" * len(title)
    print(f"\n{title}\n{rule}\n{body}")


def main() -> None:
    """Build one cutoff-scoped context and optionally invoke the model."""
    args = parse_args()
    origin = datetime.fromisoformat(args.origin)
    task = load_task(args.spec)

    service = build_boc_service(reports_dir=args.reports_dir)
    context = service.context(origin)

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
    prompt = builder(task=task, context=context)
    print_section("Generated prompt", prompt)

    if not args.live:
        print("\nLive prediction skipped. Re-run with --live to call the configured model.")
        return

    predictor = build_boc_research_predictor(
        max_documents=args.max_documents,
        max_chars_per_document=args.max_chars_per_document,
    )
    predictions = predictor.predict(task, context)
    if not predictions:
        raise RuntimeError("The research agent returned no prediction.")

    prediction = predictions[0]
    result = {
        "predictor_id": prediction.predictor_id,
        "as_of": str(prediction.as_of),
        "forecast_date": str(prediction.forecast_date),
        "probabilities": prediction.payload.model_dump(),
        "metadata": prediction.metadata,
    }
    print_section("Live model prediction", json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
