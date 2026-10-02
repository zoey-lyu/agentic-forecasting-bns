"""Measure how much of the BoC agent's backtest skill is recall, not reasoning.

The 120-origin backtest spans 2010-2024, entirely inside any current model's
training data, and BoC decisions are discrete widely-reported events. The
analyst instruction asks the model not to use remembered outcomes; this script
does not rely on that request being honoured. It removes the information
instead and measures the difference.

    arm B  anonymised payload  - no dates, no institution   -> recall impossible
    arm C  identified payload  - real dates, "Bank of Canada" -> recall possible

Both arms are news-free, so neither is exposed to the second contamination
route (a 2026 web search for a 2022 decision returns the outcome). Everything
else is held fixed: same origins, same model, same output schema.

``C - B`` is the recall estimate. If it is ~0 the backtest window is usable
for LLM arms, which is a result worth having on its own; if it is clearly
negative, every LLM number on this window has to be discounted.

    uv run python scripts/run_recall_probe.py [--spec ...] [--model ...] [--out DIR]

Writes to its own directory and never touches the shared prediction store.
"""

from __future__ import annotations

import argparse
import json
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
from boc_rate_decisions.analyst_agent import (  # noqa: E402
    AnonymizedPromptBuilder,
    BoCDecisionPromptBuilder,
    build_boc_anonymized_config,
    build_boc_basic_config,
)
from boc_rate_decisions.data import build_boc_service  # noqa: E402
from boc_rate_decisions.gate import bootstrap_ci, rps  # noqa: E402
from boc_rate_decisions.predictors import BoCLogisticPredictor  # noqa: E402


PKG = Path(boc_rate_decisions.__file__).parent


def build_arms(model: str) -> dict[str, AgentPredictor]:
    """The two LLM arms, differing only in payload and instruction."""
    return {
        "B_anonymized": AgentPredictor(
            agent_config=build_boc_anonymized_config(model=model),
            prompt_builder=AnonymizedPromptBuilder(),
            output_schema=CategoricalAgentForecastOutput,
        ),
        "C_identified": AgentPredictor(
            agent_config=build_boc_basic_config(model=model),
            prompt_builder=BoCDecisionPromptBuilder(),
            output_schema=CategoricalAgentForecastOutput,
        ),
    }


def main() -> None:
    """Run both arms plus the logistic floor and report the recall estimate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="boc_rate_direction_backtest.yaml")
    parser.add_argument("--spec-id", default="boc_recall_probe_backtest")
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--out", default=str(PKG / "data" / "recall_probe"))
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    store_dir = out_dir / "predictions"

    with (PKG / "specs" / args.spec).open() as f:
        spec = BacktestSpec.model_validate(yaml.safe_load(f))
    service = build_boc_service()

    labels = [c.label for c in spec.task.categories]
    values = [c.value for c in spec.task.categories]
    truth = service.get_series(spec.task.target_series_id, pd.Timestamp.now().to_pydatetime())
    by_date = {pd.Timestamp(t).normalize(): v for t, v in zip(truth["timestamp"], truth["value"])}
    value_to_idx = {v: i for i, v in enumerate(values)}

    def per_cell(result) -> tuple[np.ndarray, list[pd.Timestamp]]:
        dates = [pd.Timestamp(p.forecast_date).normalize() for p in result.predictions]
        probs = np.array([[float(p.payload.probabilities[label]) for label in labels] for p in result.predictions])
        outcome = np.array([value_to_idx[int(by_date[d])] for d in dates])
        return rps(probs, outcome), dates

    arms = {"A_logistic": BoCLogisticPredictor(), **build_arms(args.model)}
    scores: dict[str, pd.Series] = {}
    for name, predictor in arms.items():
        print(f"running {name} ({predictor.predictor_id}) ...", flush=True)
        result = cached_backtest(
            predictor, spec, f"{args.spec_id}__{name}", service,
            store_dir=store_dir, force_refresh=args.force_refresh,
        )
        cells, dates = per_cell(result)
        scores[name] = pd.Series(cells, index=pd.DatetimeIndex(dates))
        print(f"  n={len(cells)}  mean RPS={cells.mean():.4f}", flush=True)

    # Score only on origins every arm resolved, so no comparison is
    # confounded by one arm having skipped a harder subset.
    common = None
    for s in scores.values():
        common = s.index if common is None else common.intersection(s.index)
    aligned = {k: v.loc[common].to_numpy() for k, v in scores.items()}
    print(f"\ncommon origins across all arms: n={len(common)}")
    for name, arr in aligned.items():
        print(f"  {name:14s} mean RPS = {arr.mean():.4f}")

    print("\n=== the measurement ===")
    comparisons = [
        ("C - B  (value of knowing the date and the institution = RECALL)", "C_identified", "B_anonymized"),
        ("B - A  (anonymised agent vs logistic floor)", "B_anonymized", "A_logistic"),
        ("C - A  (identified agent vs logistic floor)", "C_identified", "A_logistic"),
    ]
    report = []
    for title, a, b in comparisons:
        d, lo, hi = bootstrap_ci(aligned[a], aligned[b])
        verdict = "significant" if (hi < 0 or lo > 0) else "not significant"
        print(f"  {title}\n    {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  {verdict}")
        report.append({"comparison": title, "delta": d, "lo": lo, "hi": hi, "significant": verdict})

    pd.DataFrame(aligned, index=common).to_csv(out_dir / "per_origin_rps.csv")
    (out_dir / "summary.json").write_text(
        json.dumps({"model": args.model, "spec": args.spec, "n": len(common), "comparisons": report}, indent=2)
    )
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
