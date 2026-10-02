"""Run the long-sample text test and apply its pre-registered rules.

See :mod:`long_sample`. Four arms, ~130 meetings, no LLM anywhere: the frozen
lexicon is word counting, so unlike the agent it can be scored across the whole
history without a contamination caveat.

Usage
-----
``python -m boc_rate_decisions.turn_detection.run_long_sample``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..a2_text_features.extract_stance_table import DEFAULT_CACHE_DIR, statement_paths
from ..data import (
    BOND_YIELD_2YR_SERIES_ID,
    CPI_SERIES_ID,
    DIRECTION_SERIES_ID,
    TARGET_RATE_SERIES_ID,
    UNEMPLOYMENT_SERIES_ID,
    build_boc_service,
)
from ..gate import bootstrap_ci
from ..predictors.logistic_baseline import FEATURE_NAMES
from .lexicon_v2 import score_statements_v2
from .long_sample import (
    TEXT_FEATURES,
    build_panel,
    era_of,
    fit_at_origin_probabilities,
    shuffled_panel,
)
from .move_task import brier_scores, murphy_decomposition
from .run_steps import USE_CASE_DIR


DATA_DIR = Path(__file__).resolve().parent / "data"


def main() -> None:
    """Score the four arms, apply both rules, and report the era split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default="2026-09-21")
    parser.add_argument("--statcan-cache-dir", type=Path, default=USE_CASE_DIR / "data" / "statcan")
    parser.add_argument("--fred-cache-dir", type=Path, default=USE_CASE_DIR / "data" / "fred")
    args = parser.parse_args()

    as_of = pd.Timestamp(args.as_of)
    service = build_boc_service(statcan_cache_dir=args.statcan_cache_dir, fred_cache_dir=args.fred_cache_dir)
    series = {
        "rate": service.get_series(TARGET_RATE_SERIES_ID, as_of=as_of),
        "yield": service.get_series(BOND_YIELD_2YR_SERIES_ID, as_of=as_of),
        "cpi": service.get_series(CPI_SERIES_ID, as_of=as_of),
        "unemployment": service.get_series(UNEMPLOYMENT_SERIES_ID, as_of=as_of),
    }
    meetings = service.get_series(DIRECTION_SERIES_ID, as_of=as_of)
    lexicon = score_statements_v2(statement_paths(DEFAULT_CACHE_DIR))

    panel = build_panel(meetings, lexicon, series)
    panel.to_csv(DATA_DIR / "long_sample_panel.csv")
    outcomes = panel["moved"]

    arms = {
        "macro": fit_at_origin_probabilities(panel, list(FEATURE_NAMES)),
        "macro_plus_text": fit_at_origin_probabilities(panel, [*FEATURE_NAMES, *TEXT_FEATURES]),
        "macro_plus_shuffled": fit_at_origin_probabilities(shuffled_panel(panel), [*FEATURE_NAMES, *TEXT_FEATURES]),
        "historical_frequency": outcomes.shift(1).expanding().mean().bfill(),
    }

    rows = []
    macro_scores = brier_scores(arms["macro"], outcomes)
    for name, series_of_probabilities in arms.items():
        scores = brier_scores(series_of_probabilities, outcomes)
        delta, low, high = bootstrap_ci(scores.to_numpy(), macro_scores.to_numpy())
        rows.append(
            {
                "arm": name,
                "n": int(len(scores)),
                "brier": round(float(scores.mean()), 4),
                **{k: v for k, v in murphy_decomposition(series_of_probabilities, outcomes).items() if k != "brier"},
                "delta_vs_macro": round(delta, 4),
                "ci_lo": round(low, 4),
                "ci_hi": round(high, 4),
                "significant": "yes" if (low > 0 or high < 0) else "no",
            }
        )
    board = pd.DataFrame(rows)
    board.to_csv(DATA_DIR / "long_sample_leaderboard.csv", index=False)

    eras = pd.DataFrame(
        [
            {
                "era": era,
                "n": int(mask.sum()),
                "move_rate": round(float(outcomes[mask].mean()), 3),
                "macro": round(float(brier_scores(arms["macro"][mask], outcomes[mask]).mean()), 4),
                "macro_plus_text": round(float(brier_scores(arms["macro_plus_text"][mask], outcomes[mask]).mean()), 4),
            }
            for era in ["2009-2014", "2015-2019", "2020-2022", "2023-2026"]
            if (mask := pd.Series(panel.index.map(era_of) == era, index=panel.index)).any()
        ]
    )
    eras["delta"] = (eras["macro_plus_text"] - eras["macro"]).round(4)
    eras.to_csv(DATA_DIR / "long_sample_eras.csv", index=False)

    text_row = board[board["arm"] == "macro_plus_text"].iloc[0]
    control_row = board[board["arm"] == "macro_plus_shuffled"].iloc[0]
    condition_1 = bool(text_row["delta_vs_macro"] < 0 and text_row["significant"] == "yes")
    condition_2 = not (control_row["delta_vs_macro"] < 0 and control_row["significant"] == "yes")

    print(f"=== long sample: {len(panel)} meetings, {outcomes.mean():.1%} moved ===")
    print(board.to_string(index=False))
    print("\n=== by era (reported, not decided on) ===")
    print(eras.to_string(index=False))
    print("\n=== pre-registered verdict ===")
    print(
        f"  (1) text beats macro, significantly?   {'YES' if condition_1 else 'NO'}"
        f"  (delta {text_row['delta_vs_macro']:+.4f}, CI {text_row['ci_lo']:+.4f} to {text_row['ci_hi']:+.4f})"
    )
    print(
        f"  (2) shuffled control stays quiet?      {'YES' if condition_2 else 'NO'}"
        f"  (delta {control_row['delta_vs_macro']:+.4f}, significant: {control_row['significant']})"
    )
    print(f"\n  VERDICT: {'TEXT ADDS INFORMATION' if condition_1 and condition_2 else 'NOT ESTABLISHED'}")
    print("  (claim is against a market-priced macro baseline, not against the futures curve)")
    print(f"\nwrote long_sample_leaderboard.csv, long_sample_eras.csv, long_sample_panel.csv in {DATA_DIR}")


if __name__ == "__main__":
    main()
