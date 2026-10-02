"""Ask whether the Bank's language carries regime information the data does not.

See :mod:`regime_language`. Free: reads A2's committed stance table and the
cached rate history, calls nothing.

Usage
-----
``python -m boc_rate_decisions.turn_detection.run_regime_check``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..a2_text_features.extract_stance_table import DEFAULT_CACHE_DIR, statement_paths
from ..a2_text_features.features import DEFAULT_STANCE_PATH
from ..data import DIRECTION_SERIES_ID, build_boc_service
from .lexicon import score_statements
from .regime_language import FORWARD_MEETINGS, build_regime_frame, incremental_r2
from .run_steps import USE_CASE_DIR


DATA_DIR = Path(__file__).resolve().parent / "data"

LANGUAGE_COLUMNS = ["stance_intensity", "guidance_intensity", "signals_anything", "hawk_dove", "guidance_score"]

#: The contamination-free counterparts, from a fixed word list.
LEXICON_COLUMNS = ["lex_intensity", "lex_tilt", "lex_hedging"]


def main() -> None:
    """Report whether language adds anything over the trailing move rate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default="2026-09-21")
    parser.add_argument("--statcan-cache-dir", type=Path, default=USE_CASE_DIR / "data" / "statcan")
    parser.add_argument("--fred-cache-dir", type=Path, default=USE_CASE_DIR / "data" / "fred")
    args = parser.parse_args()

    service = build_boc_service(statcan_cache_dir=args.statcan_cache_dir, fred_cache_dir=args.fred_cache_dir)
    directions = service.get_series(DIRECTION_SERIES_ID, as_of=pd.Timestamp(args.as_of))
    stance = pd.read_csv(DEFAULT_STANCE_PATH)

    frame = build_regime_frame(stance, directions)
    lexicon = score_statements(statement_paths(DEFAULT_CACHE_DIR))
    frame = frame.join(lexicon, how="left").dropna(subset=LEXICON_COLUMNS)
    frame.to_csv(DATA_DIR / "regime_language_frame.csv")

    columns = LANGUAGE_COLUMNS + LEXICON_COLUMNS
    full = pd.DataFrame([incremental_r2(frame, column) for column in columns])
    # Non-overlapping forward windows: every FORWARD_MEETINGS-th meeting.
    sparse_frame = frame.iloc[::FORWARD_MEETINGS]
    sparse = pd.DataFrame([incremental_r2(sparse_frame, column) for column in columns])
    full.insert(0, "sample", "all meetings (overlapping windows)")
    sparse.insert(0, "sample", "non-overlapping")
    results = pd.concat([full, sparse], ignore_index=True)
    results.to_csv(DATA_DIR / "regime_language_results.csv", index=False)

    print(f"meetings usable: {len(frame)} (2009-2026), non-overlapping subsample: {len(sparse_frame)}")
    print(f"forward window: next {FORWARD_MEETINGS} meetings; trailing window: previous 8\n")
    print("does language predict the NEXT few meetings' activity, beyond the recent move rate?\n")
    print(results.to_string(index=False))

    conservative = results[results["sample"] == "non-overlapping"]
    for label, subset in (
        ("LLM-extracted (contaminated: the model has read 2009-2026)", LANGUAGE_COLUMNS),
        ("fixed word list (contamination-free)", LEXICON_COLUMNS),
    ):
        best = conservative[conservative["language"].isin(subset)].sort_values("gain", ascending=False).iloc[0]
        print(
            f"\nbest {label}:\n  {best['language']} -> R² {best['r2_trailing_only']} to "
            f"{best['r2_with_language']} (gain {best['gain']}, n={best['n']})"
        )
    print(f"\nwrote regime_language_results.csv and regime_language_frame.csv in {DATA_DIR}")


if __name__ == "__main__":
    main()
