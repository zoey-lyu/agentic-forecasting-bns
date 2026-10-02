"""Compare lexicon v1 and v2 on the development split only.

The protocol, fixed when the market-error test came back inconclusive: a
refined lexicon may be developed **only on statements published before 2023**,
because the market data used by the test begins in 2023 and a measure tuned
while watching that window would be fitting the answer.

So this script splits at 2023-01-01, evaluates both lexicon versions on the
development half against the same forward-activity target the regime check
used, and **refuses to compute anything on the held-out half**. Whatever comes
out here is frozen; only then does the market test run again.

Usage
-----
``python -m boc_rate_decisions.turn_detection.run_lexicon_dev``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..a2_text_features.extract_stance_table import DEFAULT_CACHE_DIR, statement_paths
from ..a2_text_features.features import DEFAULT_STANCE_PATH
from ..data import DIRECTION_SERIES_ID, build_boc_service
from .lexicon import score_statements
from .lexicon_v2 import score_statements_v2
from .regime_language import FORWARD_MEETINGS, build_regime_frame, incremental_r2
from .run_steps import USE_CASE_DIR


DATA_DIR = Path(__file__).resolve().parent / "data"

#: Everything from here on is held out for the market test. Never developed against.
HELD_OUT_FROM = pd.Timestamp("2023-01-01")

V1_COLUMNS = ["lex_tilt", "lex_intensity", "lex_hedging"]
V2_COLUMNS = ["lex2_tilt", "lex2_intensity", "lex2_hedging"]


def main() -> None:
    """Score both lexicon versions on the development split and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default="2026-09-21")
    parser.add_argument("--statcan-cache-dir", type=Path, default=USE_CASE_DIR / "data" / "statcan")
    parser.add_argument("--fred-cache-dir", type=Path, default=USE_CASE_DIR / "data" / "fred")
    args = parser.parse_args()

    service = build_boc_service(statcan_cache_dir=args.statcan_cache_dir, fred_cache_dir=args.fred_cache_dir)
    directions = service.get_series(DIRECTION_SERIES_ID, as_of=pd.Timestamp(args.as_of))
    paths = statement_paths(DEFAULT_CACHE_DIR)

    frame = build_regime_frame(pd.read_csv(DEFAULT_STANCE_PATH), directions)
    frame = frame.join(score_statements(paths), how="left").join(score_statements_v2(paths), how="left")
    development = frame[frame.index < HELD_OUT_FROM].dropna(subset=V1_COLUMNS + V2_COLUMNS)

    columns = V1_COLUMNS + V2_COLUMNS
    full = pd.DataFrame([incremental_r2(development, column) for column in columns])
    sparse_frame = development.iloc[::FORWARD_MEETINGS]
    sparse = pd.DataFrame([incremental_r2(sparse_frame, column) for column in columns])
    full.insert(0, "sample", "dev, overlapping")
    sparse.insert(0, "sample", "dev, non-overlapping")
    results = pd.concat([full, sparse], ignore_index=True)
    results.to_csv(DATA_DIR / "lexicon_dev_comparison.csv", index=False)

    held_out = int((frame.index >= HELD_OUT_FROM).sum())
    print(f"development split: {len(development)} meetings before {HELD_OUT_FROM:%Y-%m-%d}")
    print(f"held out (NOT evaluated here): {held_out} meetings from {HELD_OUT_FROM:%Y-%m-%d} onward\n")
    print(results.to_string(index=False))

    for sample in ("dev, overlapping", "dev, non-overlapping"):
        subset = results[results["sample"] == sample]
        v1 = subset[subset["language"] == "lex_tilt"]["gain"].iloc[0]
        v2 = subset[subset["language"] == "lex2_tilt"]["gain"].iloc[0]
        verdict = "v2 improves the measure" if v2 > v1 else "v2 does not improve the measure"
        print(f"\n{sample}: tilt gain v1 {v1:+.4f} -> v2 {v2:+.4f}   ({verdict})")

    print(f"\nwrote lexicon_dev_comparison.csv in {DATA_DIR}")


if __name__ == "__main__":
    main()
