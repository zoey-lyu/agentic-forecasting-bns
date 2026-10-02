"""Run the four-bar validation gate on BoC logistic-baseline corrections.

Reads nothing from and writes nothing to the shared prediction store: the
backtests are recomputed in-process (the logistic baseline is deterministic —
verified at 0.00e+00 max probability difference across two runs — so this is
free of the seed problem that forced the WTI anchor to be externalised) and
the per-cell frames plus verdicts are written to a dedicated directory.

    uv run python scripts/run_gate.py [--out DIR]

The three cases run here are deliberate:

1. a **no-op** correction, which must produce exactly zero difference —
   this tests the machinery, not the science;
2. a **known-bad** correction of the shape ``hyp-001`` has (condition on a
   regime where the same effect is present everywhere), which bar 3 must
   reject;
3. **genuinely open** candidates — is the logistic baseline over- or
   under-confident, and does that depend on the rate cycle?
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
from aieng.forecasting.evaluation import BacktestSpec, backtest  # noqa: E402
from boc_rate_decisions.data import build_boc_service  # noqa: E402
from boc_rate_decisions.gate import (  # noqa: E402
    Adjustment,
    Condition,
    Correction,
    rps,
    run_gate,
)
from boc_rate_decisions.predictors import BoCLogisticPredictor  # noqa: E402


PKG = Path(boc_rate_decisions.__file__).parent
SPECS = {
    "backtest 2010-2024": "boc_rate_direction_backtest.yaml",
    "eval 2025-2026": "boc_rate_direction_eval.yaml",
}


def build_cells(spec_file: str, service) -> tuple[pd.DataFrame, list[str]]:
    """Recompute one window and flatten it into a per-cell frame.

    Conditioning columns come from the predictor's own logged metadata, so no
    feature is recomputed here and none can drift from what the model saw.
    """
    with (PKG / "specs" / spec_file).open() as f:
        spec = BacktestSpec.model_validate(yaml.safe_load(f))
    result = backtest(BoCLogisticPredictor(), spec, service)

    labels = [c.label for c in spec.task.categories]
    values = [c.value for c in spec.task.categories]

    rows = []
    for prediction in result.predictions:
        probs = prediction.payload.probabilities
        features = prediction.metadata.get("features") or {}
        momentum = features.get("rate_momentum")
        spread = features.get("yield_spread")
        rows.append(
            {
                "as_of": pd.Timestamp(prediction.as_of),
                **{f"p_{label}": float(probs[label]) for label in labels},
                "n_train": int(prediction.metadata.get("n_train", 0)),
                "model": prediction.metadata.get("model", ""),
                "rate_momentum_raw": momentum,
                "yield_spread_raw": spread,
                "rate_momentum": _cycle_bucket(momentum),
                "yield_spread": _spread_bucket(spread),
            }
        )
    cells = pd.DataFrame(rows)
    cells["era"] = np.where(cells["as_of"] < pd.Timestamp("2018-01-01"), "pre2018", "post2018")
    prob_cols = [f"p_{label}" for label in labels]

    # Realised outcomes come from the target series itself, keyed by the
    # forecast date. An earlier version recovered them by finding which class
    # reproduced the recorded RPS; that is circular and it silently absorbed a
    # normalization mismatch, so it was replaced with a direct lookup plus the
    # assertion below.
    # Read the outcome series as of "now" — this is the resolution step, not a
    # forecast input, so it deliberately sees past every origin's as_of.
    truth = service.get_series(spec.task.target_series_id, pd.Timestamp.now().to_pydatetime())
    by_date = {pd.Timestamp(t).normalize(): v for t, v in zip(truth["timestamp"], truth["value"])}
    value_to_idx = {v: i for i, v in enumerate(values)}
    forecast_dates = [pd.Timestamp(p.forecast_date).normalize() for p in result.predictions]
    missing = [d for d in forecast_dates if d not in by_date]
    if missing:
        raise KeyError(f"{len(missing)} forecast dates absent from {spec.task.target_series_id}: {missing[:3]}")
    cells["outcome_idx"] = [value_to_idx[int(by_date[d])] for d in forecast_dates]

    # The previous meeting's decision, which is visible at the origin: the
    # spec guarantees at least 35 days between scheduled meetings against a
    # 28-day lead, so the prior outcome has always been announced. Taken from
    # the outcome series rather than from this window's own rows so the first
    # origin of a window still gets a real value instead of NaN.
    labels_by_idx = dict(enumerate(labels))
    prior_dates = sorted(by_date)
    cells["last_decision"] = [
        labels_by_idx[value_to_idx[int(by_date[prev])]]
        if (prev := max((t for t in prior_dates if t < d), default=None)) is not None
        else "unknown"
        for d in forecast_dates
    ]

    # Machinery check: this module's RPS, on independently-looked-up outcomes,
    # must reproduce the score the backtest already computed. If it does not,
    # either the outcome lookup or the scoring convention is wrong, and every
    # gate verdict downstream would be meaningless.
    ours = rps(cells[prob_cols].to_numpy(dtype=float), cells["outcome_idx"].to_numpy())
    gap = float(np.max(np.abs(ours - np.asarray(result.scores, dtype=float))))
    if gap > 1e-9:  # noqa: PLR2004
        raise AssertionError(f"RPS mismatch vs backtest scores: max abs diff {gap:.3e} (expected < 1e-9)")
    print(f"  [check] RPS reproduces backtest scores exactly (max abs diff {gap:.1e})")
    return cells, prob_cols


def _cycle_bucket(momentum: float | None) -> str:
    """Rate-cycle bucket from the trailing 90-day target-rate change."""
    if momentum is None:
        return "unknown"
    if momentum < -1e-9:
        return "easing"
    if momentum > 1e-9:
        return "tightening"
    return "flat"


def _spread_bucket(spread: float | None) -> str:
    """2-year yield minus policy rate, bucketed."""
    if spread is None:
        return "unknown"
    if spread < -0.5:  # noqa: PLR2004
        return "cuts_priced"
    if spread > 0.5:  # noqa: PLR2004
        return "hikes_priced"
    return "neutral"


def main() -> None:
    """Build both windows, run the gate on each candidate, persist results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(PKG / "data" / "gate_runs"),
        help="directory for cell frames and verdicts (created if absent; never the prediction store)",
    )
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    service = build_boc_service()
    windows: dict[str, pd.DataFrame] = {}
    prob_cols: list[str] = []
    for name, spec_file in SPECS.items():
        cells, prob_cols = build_cells(spec_file, service)
        windows[name] = cells
        cells.to_csv(out_dir / f"cells_{name.split()[0]}.csv", index=False)
        print(f"{name}: n={len(cells)}  baseline RPS={rps(cells[prob_cols].to_numpy(), cells['outcome_idx']).mean():.4f}")

    print("\nbucket counts (primary window):")
    for col in ("rate_momentum", "yield_spread", "era", "model", "last_decision"):
        print(f"  {col}: {windows['backtest 2010-2024'][col].value_counts().to_dict()}")

    # Machinery check, asserted rather than gated: a no-op must move nothing.
    # Running it through the gate would only show bar 1 refusing to accept a
    # zero improvement, which is correct but tests the threshold, not the code.
    primary = windows["backtest 2010-2024"]
    p_raw = primary[prob_cols].to_numpy(dtype=float)
    noop = Correction(Condition(), Adjustment("sharpness", "power", 1.0)).apply(primary, p_raw)
    noop_gap = float(np.max(np.abs(noop - p_raw)))
    if noop_gap > 1e-12:  # noqa: PLR2004
        raise AssertionError(f"no-op adjustment changed probabilities by {noop_gap:.3e}")
    print(f"\n[check] no-op adjustment is exactly identity (max abs diff {noop_gap:.1e})")

    # Pre-registered candidate list. Fixed before the run and deliberately
    # small: 3 sharpness + 8 mass-shift = 11 comparisons on one window, which
    # is already enough multiplicity that a single nominal 95% pass would be
    # unsurprising by chance. Any acceptance below should be read with that in
    # mind and re-tested, not banked.
    #
    # Every mass-shift candidate has a stated domain reason. The last pair is
    # the most interesting: the BoC analyst prompt *asserts* to the LLM that
    # "direct cut-to-hike reversals between adjacent meetings essentially
    # never happen". If that is true and the logistic model has not already
    # absorbed it, it is exactly the kind of rule that should survive.
    candidates: list[tuple[str, Correction]] = [
        ("flatten everywhere", Correction(Condition(), Adjustment("sharpness", "power", 0.8))),
        ("sharpen everywhere", Correction(Condition(), Adjustment("sharpness", "power", 1.25))),
        ("blend 10% toward uniform", Correction(Condition(), Adjustment("mass", "mix_uniform", 0.10))),
        (
            "flatten ONLY in easing cycles (hyp-001 shape: bar 3 should reject)",
            Correction(Condition("rate_momentum", "easing"), Adjustment("sharpness", "power", 0.8)),
        ),
        (
            "flatten ONLY when the curve prices cuts",
            Correction(Condition("yield_spread", "cuts_priced"), Adjustment("sharpness", "power", 0.8)),
        ),
        (
            "flatten ONLY pre-2018 (thin training set)",
            Correction(Condition("era", "pre2018"), Adjustment("sharpness", "power", 0.8)),
        ),
    ]
    for lam in (0.10, 0.20):
        candidates += [
            (
                f"easing cycle -> shift {lam:.0%} toward cut",
                Correction(Condition("rate_momentum", "easing"), Adjustment("mass", "shift_toward", lam, "cut")),
            ),
            (
                f"tightening cycle -> shift {lam:.0%} toward hike",
                Correction(
                    Condition("rate_momentum", "tightening"), Adjustment("mass", "shift_toward", lam, "hike")
                ),
            ),
            (
                f"curve prices cuts -> shift {lam:.0%} toward cut",
                Correction(Condition("yield_spread", "cuts_priced"), Adjustment("mass", "shift_toward", lam, "cut")),
            ),
            (
                f"last decision was a cut -> take {lam:.0%} off hike (no direct reversals)",
                Correction(Condition("last_decision", "cut"), Adjustment("mass", "shift_away", lam, "hike")),
            ),
        ]

    verdicts = []
    for title, correction in candidates:
        print(f"\n--- {title} ---")
        verdict = run_gate(correction, windows, prob_cols=prob_cols)
        print(verdict)
        verdicts.append(
            {
                "title": title,
                "correction": str(correction),
                "accepted": verdict.accepted,
                "bars": [{"name": b.name, "passed": b.passed, "detail": b.detail} for b in verdict.bars],
            }
        )

    (out_dir / "verdicts.json").write_text(json.dumps(verdicts, indent=2))
    print(f"\nwrote cell frames and verdicts to {out_dir}")


if __name__ == "__main__":
    main()
