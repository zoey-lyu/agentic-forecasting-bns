"""Search vs no search on the honest window, and the first ensemble we can build.

This spends the protected eval window deliberately: 13 resolved meetings,
Jan 2025 - Jul 2026, the only origins where the model has no memory of the
outcome. It cannot deliver significance — thirteen origins resolve differences
of about 0.10 RPS and every effect in the AIA report is an order of magnitude
smaller than that — so it is a case study with numbers attached, not a verdict.

Three rows:

- **A, no search** — the tool-free analyst. Deterministic, so one run is the
  whole story.
- **B, search, single run** — mean and spread over five independent runs. The
  spread is the point: it is what a single-run comparison would have hidden.
- **B, search, mean of 5** — the ensemble AIA prescribes, which only became
  available once search supplied the diversity that greedy decoding could not.

The planned third arm (payload plus a market-implied probability) is absent on
purpose: per-meeting consensus probabilities come from CORRA futures, which are
paid, and the closest free proxy — a yield spread — is already in the payload,
so the arm would not have differed from A. The consensus channel is measured
instead by counting how often B's rationale cites someone else's forecast.

    uv run python scripts/run_search_scored.py [--runs 5]
"""

from __future__ import annotations

import argparse
import json
import re
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
from aieng.forecasting.evaluation import BacktestSpec  # noqa: E402
from aieng.forecasting.evaluation.backtest import run_eval_loop  # noqa: E402
from aieng.forecasting.models import LITE_MODEL  # noqa: E402
from boc_rate_decisions.analyst_agent import (  # noqa: E402
    build_boc_agent_predictor,
    build_boc_basic_config,
    build_boc_news_config,
)
from boc_rate_decisions.data import build_boc_service  # noqa: E402
from boc_rate_decisions.gate import bootstrap_ci, rps  # noqa: E402


PKG = Path(boc_rate_decisions.__file__).parent

# Phrases that mark a retrieved third-party view of the decision itself, as
# opposed to retrieved data. This is the consensus channel the verifier audit
# turned up: several searches came back carrying an outside forecast that
# happened to be right.
CONSENSUS = re.compile(
    r"consensus|economists (expect|anticipat|forecast)|market(s)? (expect|anticipat|pric)"
    r"|widely (expect|anticipat)|survey|swaps? (imply|pric)|C\.?D\.? Howe",
    re.IGNORECASE,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--model", default=LITE_MODEL)
    parser.add_argument("--out", default=str(PKG / "data" / "search_scored"))
    args = parser.parse_args()

    spec = BacktestSpec.model_validate(
        yaml.safe_load((PKG / "specs" / "boc_rate_direction_eval.yaml").read_text())
    )
    service = build_boc_service()
    labels = [c.label for c in spec.task.categories]
    values = [c.value for c in spec.task.categories]
    origins = spec.origins()

    truth = service.get_series(spec.task.target_series_id, pd.Timestamp.now().to_pydatetime())
    by_date = {pd.Timestamp(t).normalize(): v for t, v in zip(truth["timestamp"], truth["value"])}
    to_idx = {v: i for i, v in enumerate(values)}

    def run(config, tag: str) -> pd.DataFrame:
        preds, _, skipped = run_eval_loop(
            predictor=build_boc_agent_predictor(config),
            task=spec.task,
            origins=origins,
            warmup=spec.warmup,
            data_service=service,
            max_retries=2,
            retry_delay=2.0,
        )
        if skipped:
            print(f"  {tag}: skipped {skipped}", flush=True)
        rows = []
        for p in preds:
            rationale = (p.metadata or {}).get("rationale", "") or ""
            rows.append(
                {
                    "as_of": pd.Timestamp(p.as_of).normalize(),
                    **{c: float(p.payload.probabilities[c]) for c in labels},
                    "outcome_idx": to_idx[int(by_date[pd.Timestamp(p.forecast_date).normalize()])],
                    "cites_consensus": bool(CONSENSUS.search(rationale)),
                }
            )
        print(f"  {tag} done ({len(rows)} origins)", flush=True)
        return pd.DataFrame(rows)

    print(f"origins = {len(origins)}   model = {args.model}\n")

    print("A: no search (deterministic)")
    frame_a = run(build_boc_basic_config(model=args.model), "A")

    print(f"\nB: search, {args.runs} independent runs")
    frames_b = [run(build_boc_news_config(model=args.model), f"B run {i}") for i in range(args.runs)]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_a.assign(arm="A_no_search", run=0).to_csv(out_dir / "arm_a.csv", index=False)
    pd.concat([f.assign(arm="B_search", run=i) for i, f in enumerate(frames_b)]).to_csv(
        out_dir / "arm_b.csv", index=False
    )

    # Align every arm on the origins all of them resolved.
    common = set(frame_a["as_of"])
    for f in frames_b:
        common &= set(f["as_of"])
    common = sorted(common)

    def cells(frame: pd.DataFrame) -> np.ndarray:
        f = frame.set_index("as_of").loc[common]
        return rps(f[labels].to_numpy(), f["outcome_idx"].to_numpy())

    cells_a = cells(frame_a)
    cells_b_each = np.vstack([cells(f) for f in frames_b])

    stacked = np.stack([f.set_index("as_of").loc[common][labels].to_numpy() for f in frames_b])
    ens = stacked.mean(axis=0)
    outcomes = frame_a.set_index("as_of").loc[common]["outcome_idx"].to_numpy()
    cells_ens = rps(ens, outcomes)

    print(f"\n=== scored on {len(common)} origins ===")
    print(f"  A  no search                 RPS {cells_a.mean():.4f}")
    print(
        f"  B  search, single run        RPS {cells_b_each.mean(axis=1).mean():.4f}   "
        f"(across runs: {cells_b_each.mean(axis=1).min():.4f} - {cells_b_each.mean(axis=1).max():.4f})"
    )
    print(f"  B  search, mean of {args.runs}       RPS {cells_ens.mean():.4f}")

    print("\n=== differences (bootstrap over origins) ===")
    comparisons = [
        ("search single-run vs no search", cells_b_each.mean(axis=0), cells_a),
        ("search ensemble  vs no search", cells_ens, cells_a),
        ("search ensemble  vs single run", cells_ens, cells_b_each.mean(axis=0)),
    ]
    report = []
    for name, after, before in comparisons:
        delta, lo, hi = bootstrap_ci(after, before)
        verdict = "better" if hi < 0 else ("worse" if lo > 0 else "not significant")
        print(f"  {name:32s} {delta:+.4f} [{lo:+.4f}, {hi:+.4f}]  {verdict}")
        report.append({"comparison": name, "delta": delta, "lo": lo, "hi": hi, "verdict": verdict})

    rate_b = float(np.mean([f["cites_consensus"].mean() for f in frames_b]))
    rate_a = float(frame_a["cites_consensus"].mean())
    print("\n=== consensus channel ===")
    print(f"  rationales citing an outside forecast:  A {rate_a:.0%}   B {rate_b:.0%}")

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "n_origins": len(common),
                "runs": args.runs,
                "rps": {
                    "A_no_search": float(cells_a.mean()),
                    "B_single_run_mean": float(cells_b_each.mean()),
                    "B_ensemble": float(cells_ens.mean()),
                },
                "comparisons": report,
                "consensus_citation_rate": {"A": rate_a, "B": rate_b},
            },
            indent=2,
        )
    )
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
