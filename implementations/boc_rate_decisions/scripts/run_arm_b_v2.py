"""Regenerate the anonymised (arm B) forecasts against the v2 payload.

Arm B v1 lives in ``data/recall_probe/`` and was produced before the payload
was tightened; it is left untouched. The v2 predictor id is identical to v1's
(the agent name and model did not change), so this writes to its own store
directory — reusing v1's would silently overwrite it.

Arm B is no longer half of a matched contrast. It is a standalone testbed: a
tool-free LLM that cannot recall the period, scored against the deterministic
logistic baseline over 120 origins instead of the honest window's 12.

    uv run python scripts/run_arm_b_v2.py [--model ...] [--out DIR]
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv


warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / ".env", override=False)

import boc_rate_decisions  # noqa: E402
from aieng.forecasting.evaluation import BacktestSpec, cached_backtest  # noqa: E402
from aieng.forecasting.methods.agentic import AgentPredictor  # noqa: E402
from aieng.forecasting.methods.agentic.outputs import CategoricalAgentForecastOutput  # noqa: E402
from aieng.forecasting.models import LITE_MODEL  # noqa: E402
from boc_rate_decisions.analyst_agent import (  # noqa: E402
    AnonymizedPromptBuilder,
    build_boc_anonymized_config,
)
from boc_rate_decisions.data import build_boc_service  # noqa: E402
from boc_rate_decisions.gate import rps  # noqa: E402
from boc_rate_decisions.predictors import BoCLogisticPredictor  # noqa: E402


PKG = Path(boc_rate_decisions.__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="boc_rate_direction_backtest.yaml")
    parser.add_argument("--spec-id", default="boc_arm_b_v2_backtest")
    parser.add_argument("--model", default=LITE_MODEL)
    parser.add_argument("--out", default=str(PKG / "data" / "arm_b_v2"))
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = BacktestSpec.model_validate(yaml.safe_load((PKG / "specs" / args.spec).read_text()))
    service = build_boc_service()
    labels = [c.label for c in spec.task.categories]
    values = [c.value for c in spec.task.categories]

    truth = service.get_series(spec.task.target_series_id, pd.Timestamp.now().to_pydatetime())
    by_date = {pd.Timestamp(t).normalize(): v for t, v in zip(truth["timestamp"], truth["value"])}
    value_to_idx = {v: i for i, v in enumerate(values)}

    predictors = {
        "arm_b_v2": AgentPredictor(
            agent_config=build_boc_anonymized_config(model=args.model),
            prompt_builder=AnonymizedPromptBuilder(),
            output_schema=CategoricalAgentForecastOutput,
        ),
        "logistic": BoCLogisticPredictor(),
    }

    frames = []
    for name, predictor in predictors.items():
        print(f"running {name} ...", flush=True)
        result = cached_backtest(
            predictor,
            spec,
            f"{args.spec_id}__{name}",
            service,
            store_dir=out_dir / "predictions",
            force_refresh=args.force_refresh,
        )
        rows = []
        for p in result.predictions:
            as_of = pd.Timestamp(p.as_of).normalize()
            forecast_date = pd.Timestamp(p.forecast_date).normalize()
            probs = [float(p.payload.probabilities[c]) for c in labels]
            rows.append(
                {
                    "as_of": as_of,
                    "model": name,
                    **{f"p_{c}": v for c, v in zip(labels, probs)},
                    "outcome_idx": value_to_idx[int(by_date[forecast_date])],
                }
            )
        frame = pd.DataFrame(rows)
        cells = rps(frame[[f"p_{c}" for c in labels]].to_numpy(), frame["outcome_idx"].to_numpy())
        print(f"  n={len(frame)}  mean RPS={cells.mean():.4f}", flush=True)
        frames.append(frame)

    cells = pd.concat(frames, ignore_index=True)
    cells["era"] = np.where(cells["as_of"].dt.year >= 2022, "2022+", "pre2022")  # noqa: PLR2004
    cells.to_csv(out_dir / "cells.csv", index=False)
    print(f"\nwrote {out_dir / 'cells.csv'}  ({len(cells)} rows)")


if __name__ == "__main__":
    main()
