"""Does Platt scaling (extremization) improve the anonymised agent's forecasts?

LLMs hedge toward the middle of the simplex; Platt scaling pushes probabilities
back toward their bounds. For an ordered categorical forecast the transform is
``p ** d`` renormalised, which for K=2 is exactly the log-odds extremization of
Baron et al. (2014) and which ``gate.Adjustment(op="power")`` already
implements. ``d`` is fixed at sqrt(3) after Neyman and Roughgarden (2022) — the
value the AIA Forecaster adopts precisely so the coefficient is not fitted to
the benchmark it is scored on. Fitting ``d`` here would be worse than useless:
120 origins cannot support it, and a free parameter turns the gate into a
fishing licence.

Two things are reported, following the AIA report's own handling of residual
leakage — bound the effect rather than chase it to zero:

1. the effect on all 120 origins;
2. the same on the 97 origins before 2022, where the re-identification probe
   found no residual clock. If the two agree, the leak does not matter here.

    uv run python scripts/run_platt_test.py [--cells PATH]

Offline: reads the stored cells, makes no LLM calls.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]

import boc_rate_decisions  # noqa: E402
from boc_rate_decisions.gate import (  # noqa: E402
    Adjustment,
    Condition,
    Correction,
    bootstrap_ci,
    rps,
    run_gate,
)


PKG = Path(boc_rate_decisions.__file__).parent
LABELS = ("cut", "hold", "hike")
PROB_COLS = [f"p_{c}" for c in LABELS]
D = float(np.sqrt(3.0))


def _score(frame: pd.DataFrame, adjustment: Adjustment | None = None) -> np.ndarray:
    probs = frame[PROB_COLS].to_numpy()
    if adjustment is not None:
        probs = adjustment.apply(probs, labels=LABELS)
    return rps(probs, frame["outcome_idx"].to_numpy())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", default=str(PKG / "data" / "arm_b_v2" / "cells.csv"))
    parser.add_argument("--model", default="arm_b_v2")
    args = parser.parse_args()

    cells = pd.read_csv(args.cells, parse_dates=["as_of"])
    platt = Adjustment(channel="sharpness", op="power", value=D)

    print(f"Platt / extremization with d = sqrt(3) = {D:.4f}\n")

    subsets = {
        "all 120 origins": lambda f: f,
        "pre-2022 only (no residual clock)": lambda f: f[f["as_of"].dt.year < 2022],  # noqa: PLR2004
    }

    for model in (args.model, "logistic"):
        frame_all = cells[cells["model"] == model]
        if frame_all.empty:
            continue
        print(f"=== {model} ===")
        for name, select in subsets.items():
            frame = select(frame_all)
            before = _score(frame)
            after = _score(frame, platt)
            delta, lo, hi = bootstrap_ci(after, before)
            verdict = "helps" if hi < 0 else ("hurts" if lo > 0 else "no effect")
            print(
                f"  {name:36s} n={len(frame):3d}  "
                f"RPS {before.mean():.4f} -> {after.mean():.4f}   "
                f"delta {delta:+.4f} [{lo:+.4f}, {hi:+.4f}]  {verdict}"
            )
        print()

    # The four-bar gate, with the two eras as the cross-regime windows bar 4
    # needs. This is a harder test than the raw comparison above: bar 4 asks
    # the correction to survive a regime it was not measured on.
    frame = cells[cells["model"] == args.model].copy()
    windows = {
        "2009-2017": frame[frame["as_of"].dt.year < 2018],  # noqa: PLR2004
        "2018-2024": frame[frame["as_of"].dt.year >= 2018],  # noqa: PLR2004
    }
    verdict = run_gate(
        Correction(condition=Condition(), adjustment=platt),
        windows,
        prob_cols=PROB_COLS,
        labels=LABELS,
    )
    print("=== four-bar gate ===")
    print(f"  correction: {verdict.correction if hasattr(verdict, 'correction') else platt}")
    for bar in verdict.bars:
        mark = {True: "PASS", False: "FAIL", None: "n/a "}[bar.passed]
        print(f"  [{mark}] {bar.name}: {bar.detail}")
    print(f"\n  ACCEPTED: {verdict.accepted}")


if __name__ == "__main__":
    main()
