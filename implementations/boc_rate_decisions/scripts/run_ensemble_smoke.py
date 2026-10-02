"""Step 0 of the ensemble plan: does the same agent give different answers when re-run?

The ensemble work rests on one assumption — that repeated runs of the *same*
agent on the *same* origin differ. If they do not, averaging M runs measures
nothing and the whole plan collapses. This script checks that assumption before
any of it is built.

Nothing is changed between runs except ``AgentConfig.seed``: same model, same
prompt builder, same payload, same temperature (whatever the provider default
is — deliberately not set, so each run is a draw from the same distribution the
existing single-run backtests drew from once).

    uv run python scripts/run_ensemble_smoke.py [--runs 3] [--model ...]

Writes nothing to disk. Two origins, three runs each, probabilities printed.
"""

from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv


warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / ".env", override=False)

import boc_rate_decisions  # noqa: E402
from aieng.forecasting.evaluation import BacktestSpec  # noqa: E402
from aieng.forecasting.evaluation.backtest import run_eval_loop  # noqa: E402
from aieng.forecasting.methods.agentic import AgentPredictor  # noqa: E402
from aieng.forecasting.methods.agentic.outputs import CategoricalAgentForecastOutput  # noqa: E402
from aieng.forecasting.models import LITE_MODEL  # noqa: E402
from boc_rate_decisions.analyst_agent import (  # noqa: E402
    BoCDecisionPromptBuilder,
    build_boc_basic_config,
)
from boc_rate_decisions.data import build_boc_service  # noqa: E402


PKG = Path(boc_rate_decisions.__file__).parent

# One genuinely uncertain origin (2015-01-21 was a surprise cut) and one calm
# hold, so the check does not rest on a single easy case.
DEFAULT_ORIGINS = ["2014-12-24", "2013-06-19"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--model", default=LITE_MODEL)
    parser.add_argument("--origins", nargs="*", default=DEFAULT_ORIGINS)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Leave unset to use the provider default (what the existing backtests used). "
        "Set explicitly to test whether the default is greedy decoding.",
    )
    args = parser.parse_args()

    with (PKG / "specs" / "boc_rate_direction_backtest.yaml").open() as f:
        spec = BacktestSpec.model_validate(yaml.safe_load(f))
    service = build_boc_service()
    labels = [c.label for c in spec.task.categories]
    origins = [datetime.fromisoformat(o) for o in args.origins]

    print(f"model   = {args.model}")
    print(f"origins = {[o.date().isoformat() for o in origins]}")
    print(f"runs    = {args.runs}  (seed varies 0..{args.runs - 1}, nothing else)")
    print(f"temp    = {args.temperature if args.temperature is not None else 'provider default'}\n")

    # run index -> origin -> probability vector
    table: dict[int, dict[str, np.ndarray]] = {}
    for r in range(args.runs):
        predictor = AgentPredictor(
            # model_copy rather than a new factory arg: step 0 must not change
            # the production config path. seed is the only field that moves.
            agent_config=build_boc_basic_config(model=args.model).model_copy(update={"seed": r, "temperature": args.temperature}),
            prompt_builder=BoCDecisionPromptBuilder(),
            output_schema=CategoricalAgentForecastOutput,
        )
        preds, _, skipped = run_eval_loop(
            predictor=predictor,
            task=spec.task,
            origins=origins,
            warmup=spec.warmup,
            data_service=service,
            max_retries=2,
            retry_delay=2.0,
        )
        table[r] = {
            p.as_of.date().isoformat(): np.array([float(p.payload.probabilities[c]) for c in labels])
            for p in preds
        }
        if skipped:
            print(f"  run {r}: skipped {skipped}")
        print(f"  run {r} done", flush=True)

    print("\n=== probabilities ===")
    header = "  ".join(f"{c:>7s}" for c in labels)
    for o in [d.date().isoformat() for d in origins]:
        print(f"\norigin {o}")
        print(f"  run  {header}")
        stack = []
        for r in sorted(table):
            v = table[r].get(o)
            if v is None:
                print(f"  {r:>3d}  (missing)")
                continue
            stack.append(v)
            print(f"  {r:>3d}  " + "  ".join(f"{x:7.4f}" for x in v))
        if len(stack) > 1:
            arr = np.vstack(stack)
            spread = arr.max(axis=0) - arr.min(axis=0)
            print(f"  {'range':>3s}  " + "  ".join(f"{x:7.4f}" for x in spread))
            identical = bool(np.allclose(arr, arr[0], atol=1e-9))
            print(f"  -> {'IDENTICAL — ensemble would measure nothing' if identical else 'runs differ'}")

    print(
        "\nVerdict: if every origin says IDENTICAL, stop and fix temperature/seed\n"
        "before building the EnsemblePredictor."
    )


if __name__ == "__main__":
    main()
