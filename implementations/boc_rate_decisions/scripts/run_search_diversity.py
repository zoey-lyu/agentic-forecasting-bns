"""Does giving the agent web search make repeated runs differ?

Step 0 established that the tool-free BoC agent is deterministic: three runs of
the same origin returned probabilities identical to four decimal places, since
``temperature`` is left unset and the provider decodes greedily. That kills
ensembling and the supervisor, both of which need several genuinely different
forecasts to work on.

The AIA Forecaster's diversity does not come from decoding randomness either.
It comes from each agent running its **own** search and therefore reading
different evidence. This script tests whether that mechanism is available to
us: same prompt, same model, ``temperature`` still unset, search switched on.
Any spread that appears can only have come from the search path.

Pre-registered bar, fixed before the first run: the widest per-category range
across runs must exceed **0.05** at a majority of origins. Reference points
from step 0 — tool-free at default settings gave 0.00, and forcing
``temperature=1.0`` gave only 0.02-0.03.

Origins come from the 2025-2026 window because the model has no memory of those
decisions, so its uncertainty is real. **Only dispersion is measured — nothing
is scored against outcomes** — so the protected eval window keeps its value.

    uv run python scripts/run_search_diversity.py [--runs 5] [--origins ...]
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv


warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / ".env", override=False)

import boc_rate_decisions  # noqa: E402
from aieng.forecasting.evaluation import BacktestSpec  # noqa: E402
from aieng.forecasting.evaluation.backtest import run_eval_loop  # noqa: E402
from aieng.forecasting.models import LITE_MODEL  # noqa: E402
from boc_rate_decisions.analyst_agent import build_boc_agent_predictor, build_boc_news_config  # noqa: E402
from boc_rate_decisions.data import build_boc_service  # noqa: E402


PKG = Path(boc_rate_decisions.__file__).parent

DISPERSION_BAR = 0.05

# Five origins spanning the honest window: two cuts, a pause, and two 2026
# meetings, so the check does not rest on one kind of decision.
DEFAULT_ORIGINS = ["2025-01-01", "2025-05-07", "2025-08-20", "2026-02-18", "2026-05-13"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--model", default=LITE_MODEL)
    parser.add_argument("--origins", nargs="*", default=DEFAULT_ORIGINS)
    parser.add_argument("--out", default=str(PKG / "data" / "search_diversity"))
    args = parser.parse_args()

    spec = BacktestSpec.model_validate(
        yaml.safe_load((PKG / "specs" / "boc_rate_direction_eval.yaml").read_text())
    )
    service = build_boc_service()
    labels = [c.label for c in spec.task.categories]
    origins = [datetime.fromisoformat(o) for o in args.origins]

    print(f"model   = {args.model}   (search ON, temperature unset — greedy)")
    print(f"origins = {args.origins}")
    print(f"runs    = {args.runs}   bar: widest range > {DISPERSION_BAR} at most origins\n")

    records = []
    for run in range(args.runs):
        predictor = build_boc_agent_predictor(build_boc_news_config(model=args.model))
        preds, _, skipped = run_eval_loop(
            predictor=predictor,
            task=spec.task,
            origins=origins,
            warmup=spec.warmup,
            data_service=service,
            max_retries=2,
            retry_delay=2.0,
        )
        for p in preds:
            records.append(
                {
                    "run": run,
                    "as_of": pd.Timestamp(p.as_of).date().isoformat(),
                    **{c: float(p.payload.probabilities[c]) for c in labels},
                    "reasoning": (p.metadata or {}).get("rationale", "")[:400],
                }
            )
        if skipped:
            print(f"  run {run}: skipped {skipped}")
        print(f"  run {run} done ({len(preds)} origins)", flush=True)

    df = pd.DataFrame(records)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "runs.csv", index=False)

    print("\n=== dispersion per origin ===")
    summary = []
    for origin, group in df.groupby("as_of"):
        arr = group[labels].to_numpy()
        rng = arr.max(axis=0) - arr.min(axis=0)
        widest = float(rng.max())
        summary.append({"as_of": origin, "n_runs": len(group), "widest_range": widest})
        print(f"\norigin {origin}   ({len(group)} runs)")
        print("  run  " + "  ".join(f"{c:>7s}" for c in labels))
        for _, row in group.iterrows():
            print(f"  {int(row['run']):>3d}  " + "  ".join(f"{row[c]:7.4f}" for c in labels))
        print("  rng  " + "  ".join(f"{x:7.4f}" for x in rng) + f"   widest={widest:.4f}")

    s = pd.DataFrame(summary)
    passing = int((s["widest_range"] > DISPERSION_BAR).sum())
    verdict = passing > len(s) / 2
    print("\n=== verdict ===")
    print(f"  origins over the {DISPERSION_BAR} bar: {passing} / {len(s)}")
    print(f"  median widest range: {s['widest_range'].median():.4f}")
    print(f"  step 0 reference: tool-free = 0.0000, temperature=1.0 = 0.02-0.03")
    print(
        f"\n  {'PASS — search creates usable diversity; ensemble/supervisor are back on' if verdict else 'FAIL — search does not diversify; close the ensemble/supervisor line'}"
    )

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "runs": args.runs,
                "origins": args.origins,
                "bar": DISPERSION_BAR,
                "per_origin": summary,
                "origins_over_bar": passing,
                "passed": bool(verdict),
            },
            indent=2,
        )
    )
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
