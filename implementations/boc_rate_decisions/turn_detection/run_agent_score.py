"""Score the frozen turn agent against the market on the post-cutoff window.

Reads the assessments produced by :mod:`run_agent_labelfree` — the prompt was
frozen before this ran, temperature is 0, so re-running would reproduce them —
joins the realised outcomes, and scores every arm on Brier against the market's
own P(move).

The controls are scored alongside the agent on purpose. An agent that beats the
market while its ``no_statement`` twin does just as well has not read anything;
it has found a good prior. That comparison is the one the pre-registration
makes decisive.

Usage
-----
``python -m boc_rate_decisions.turn_detection.run_agent_score``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..a3_t2_horizon.data_t2 import DIRECTION_T2_SERIES_ID, build_boc_t2_service
from ..data import DIRECTION_SERIES_ID, build_boc_service
from ..gate import bootstrap_ci
from .move_task import brier_scores, move_probability, murphy_decomposition
from .run_steps import SPEC_2025_2026, USE_CASE_DIR, _market_probabilities


DATA_DIR = Path(__file__).resolve().parent / "data"


def _parse_args() -> argparse.Namespace:
    """Build the CLI for scoring the frozen agent."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default="2026-09-21")
    parser.add_argument("--horizon", choices=("t+1", "t+2"), default="t+1")
    parser.add_argument("--statcan-cache-dir", type=Path, default=USE_CASE_DIR / "data" / "statcan")
    parser.add_argument("--fred-cache-dir", type=Path, default=USE_CASE_DIR / "data" / "fred")
    return parser.parse_args()


def main() -> None:
    """Score the agent and its controls against the market reference."""
    args = _parse_args()

    assessments = DATA_DIR / f"agent_labelfree_2025-2026_{args.horizon.replace('+', '')}.csv"
    if not assessments.exists():
        raise SystemExit(f"No assessments at {assessments}. Run run_agent_labelfree first.")

    as_of = pd.Timestamp(args.as_of)
    if args.horizon == "t+2":
        service = build_boc_t2_service(args.statcan_cache_dir, args.fred_cache_dir, as_of=as_of)
        shifted = service.get_series(DIRECTION_T2_SERIES_ID, as_of=as_of)
        outcomes = pd.Series(
            (shifted["value"].astype(float) != 0.0).astype(float).to_numpy(),
            index=pd.to_datetime(shifted["released_at"]).to_numpy(),
        )
    else:
        service = build_boc_service(statcan_cache_dir=args.statcan_cache_dir, fred_cache_dir=args.fred_cache_dir)
        directions = service.get_series(DIRECTION_SERIES_ID, as_of=as_of)
        outcomes = pd.Series(
            (directions["value"].astype(float) != 0.0).astype(float).to_numpy(),
            index=pd.to_datetime(directions["timestamp"]).to_numpy(),
        )

    frame = pd.read_csv(assessments, parse_dates=["meeting"])
    arms = {f"agent_{name}": group.set_index("meeting")["p_move"] for name, group in frame.groupby("arm")}
    if args.horizon == "t+2":
        table = pd.read_csv(
            USE_CASE_DIR / "a3_t2_horizon" / "data" / "market_implied_t2.csv", parse_dates=["target_meeting"]
        ).set_index("target_meeting")
        arms["market_implied"] = move_probability(
            table.rename(columns={"p_cut": "cut", "p_hold": "hold", "p_hike": "hike"}).loc[:, ["cut", "hold", "hike"]]
        )
    else:
        arms["market_implied"] = move_probability(
            _market_probabilities(SPEC_2025_2026, DATA_DIR / "market_implied_2025_2026.csv")
        )

    market_brier = float(brier_scores(arms["market_implied"], outcomes).mean())
    rows = []
    for name, series in arms.items():
        brier = float(brier_scores(series, outcomes).mean())
        rows.append(
            {
                "predictor_id": name,
                "n_meetings": int(len(series.index.intersection(outcomes.index))),
                "brier": round(brier, 4),
                "skill_vs_market": round(1.0 - brier / market_brier, 4),
                **{k: v for k, v in murphy_decomposition(series, outcomes).items() if k != "brier"},
            }
        )
    # Paired bootstrap against the market, same convention as the gate.
    market_scores = brier_scores(arms["market_implied"], outcomes)
    for entry in rows:
        series = arms[str(entry["predictor_id"])]
        scores = brier_scores(series, outcomes)
        shared = scores.index.intersection(market_scores.index)
        delta, low, high = bootstrap_ci(scores.loc[shared].to_numpy(), market_scores.loc[shared].to_numpy())
        entry.update(
            {
                "delta_vs_market": round(delta, 4),
                "ci_lo": round(low, 4),
                "ci_hi": round(high, 4),
                "significant": "yes" if (low > 0 or high < 0) else "no",
            }
        )

    board = pd.DataFrame(rows).sort_values("brier").reset_index(drop=True)
    board.to_csv(DATA_DIR / f"agent_score_{args.horizon.replace('+', '')}.csv", index=False)

    agent = float(board.loc[board["predictor_id"] == "agent_baseline", "brier"].iloc[0])
    no_statement = float(board.loc[board["predictor_id"] == "agent_no_statement", "brier"].iloc[0])
    verbatim = frame[frame["arm"] == "baseline"]
    grounded = int(verbatim["quotes"].sum()) == int(verbatim["quotes_verbatim"].sum())

    print(f"=== P(move) at {args.horizon}, post-cutoff meetings, Brier (lower is better) ===")
    print(board.to_string(index=False))
    print("\n=== pre-registered go/no-go ===")
    print(
        f"  (1) agent beats the market?              {'YES' if agent < market_brier else 'NO'}"
        f"  ({agent:.4f} vs {market_brier:.4f})"
    )
    print(
        f"  (2) the statement is doing work?         {'YES' if no_statement > agent else 'NO'}"
        f"  (no_statement {no_statement:.4f} vs agent {agent:.4f})"
    )
    print(f"  (3) evidence still verbatim?             {'YES' if grounded else 'NO'}")
    interval = board[board["predictor_id"] == "agent_baseline"].iloc[0]
    print(
        f"\n  paired bootstrap, agent - market: {interval['delta_vs_market']:+.4f} "
        f"(95% CI {interval['ci_lo']:+.4f} to {interval['ci_hi']:+.4f}, significant: {interval['significant']})"
    )
    print("  No significance is available at n=14 with 4 moves; this is a go/no-go, not a result.")
    print(f"\nwrote agent_score_{args.horizon.replace('+', '')}.csv in {DATA_DIR}")


if __name__ == "__main__":
    main()
