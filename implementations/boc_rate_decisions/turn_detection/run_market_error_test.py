"""Test whether statement language predicts the market's own errors.

The regime check established that statements say something about the Bank's
coming activity that its move history does not. That is not yet an edge: the
market reads the same statements, and if it has already priced them there is
nothing left for an agent to find.

This asks the question directly. For each meeting, take the market's P(move),
the realised outcome, and the language of the most recent statement published
before the forecast origin. If language predicts the market's **signed error**
— ``outcome - market_P(move)`` — then the market has not priced it.

Both pre-registered conditions must hold (``PREREGISTRATION.md``, 2026-09-21):
the correlation must carry the same sign in both windows, and a correction
fitted on one window must beat the raw market on the other, both directions. A
shuffled-language arm runs alongside as the negative control.

The sample is 28 meetings in two windows of fourteen. That is enough to detect
a large effect and nothing else, so a null here means "not detectable", not
"absent".

Usage
-----
``python -m boc_rate_decisions.turn_detection.run_market_error_test``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..a1_market_implied.build_table import origins_from_spec
from ..a2_text_features.extract_stance_table import DEFAULT_CACHE_DIR, statement_paths
from ..a3_t2_horizon.build_table_t2 import origin_triples
from ..a3_t2_horizon.data_t2 import DIRECTION_T2_SERIES_ID, build_boc_t2_service
from ..data import DIRECTION_SERIES_ID
from .lexicon import score_statements
from .lexicon_v2 import score_statements_v2
from .move_task import brier_scores, move_probability
from .run_calibration_test import T1_SPECS, T2_SPECS, _market_move_probabilities
from .run_steps import USE_CASE_DIR, _market_probabilities


DATA_DIR = Path(__file__).resolve().parent / "data"

#: Pre-registered primary variable. v1 for the first look, ``lex2_tilt`` for the
#: second after the refined lexicon was frozen on the pre-2023 split.
PRIMARY_LANGUAGE = "lex2_tilt"

#: Seed for the shuffled-language negative control.
SHUFFLE_SEED = 42


def _language_at_origins(pairs: list[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.DataFrame:
    """Language of the most recent statement published on or before each origin."""
    paths = statement_paths(DEFAULT_CACHE_DIR)
    lexicon = score_statements(paths).join(score_statements_v2(paths))
    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for origin, scored_on in pairs:
        visible = lexicon[lexicon.index <= origin]
        if visible.empty:
            continue
        rows[scored_on] = visible.iloc[-1].to_dict()
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def _window_frame(spec_path: Path, cache: Path, outcomes: pd.Series, *, t2: bool) -> pd.DataFrame:
    """Market P(move), outcome, signed error and language, one row per meeting."""
    pairs = (
        [(origin, target) for origin, _, target in origin_triples(spec_path)] if t2 else origins_from_spec(spec_path)
    )
    market = (
        _market_move_probabilities(cache, t2=True) if t2 else move_probability(_market_probabilities(spec_path, cache))
    )
    language = _language_at_origins(pairs)

    frame = pd.DataFrame({"market_p_move": market}).join(language, how="inner")
    frame["outcome"] = outcomes.reindex(frame.index)
    frame = frame.dropna(subset=["outcome"])
    frame["signed_error"] = frame["outcome"] - frame["market_p_move"]
    return frame


def _fit_correction(train: pd.DataFrame, column: str) -> tuple[float, float]:
    """Least-squares fit of signed error on one language variable."""
    x = np.column_stack([np.ones(len(train)), train[column].to_numpy(dtype=float)])
    coefficients, *_ = np.linalg.lstsq(x, train["signed_error"].to_numpy(dtype=float), rcond=None)
    return float(coefficients[0]), float(coefficients[1])


def _apply_correction(test: pd.DataFrame, intercept: float, slope: float, column: str) -> pd.Series:
    """Add the predicted error back to the market forecast, clipped to [0, 1]."""
    adjustment = intercept + slope * test[column].astype(float)
    return (test["market_p_move"] + adjustment).clip(0.0, 1.0)


def _run_horizon(
    horizon: str, frames: dict[str, pd.DataFrame], columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Correlations per window and out-of-sample corrections, for every variable."""
    correlation_rows = [
        {
            "horizon": horizon,
            "window": window,
            "language": column,
            "n": int(len(frame)),
            "corr_with_signed_error": round(float(frame[column].corr(frame["signed_error"])), 4),
        }
        for window, frame in frames.items()
        for column in columns
    ]

    correction_rows = []
    for train_name, test_name in (("2023-2024", "2025-2026"), ("2025-2026", "2023-2024")):
        train, test = frames[train_name], frames[test_name]
        raw = float(brier_scores(test["market_p_move"], test["outcome"]).mean())
        for column in columns:
            intercept, slope = _fit_correction(train, column)
            corrected = _apply_correction(test, intercept, slope, column)
            corrected_brier = float(brier_scores(corrected, test["outcome"]).mean())
            correction_rows.append(
                {
                    "horizon": horizon,
                    "language": column,
                    "fitted_on": train_name,
                    "scored_on": test_name,
                    "market_brier": round(raw, 4),
                    "corrected_brier": round(corrected_brier, 4),
                    "improved": bool(corrected_brier < raw),
                }
            )
    return pd.DataFrame(correlation_rows), pd.DataFrame(correction_rows)


def main() -> None:
    """Run the market-error test at both horizons and print the verdicts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default="2026-09-21")
    parser.add_argument("--statcan-cache-dir", type=Path, default=USE_CASE_DIR / "data" / "statcan")
    parser.add_argument("--fred-cache-dir", type=Path, default=USE_CASE_DIR / "data" / "fred")
    args = parser.parse_args()

    as_of = pd.Timestamp(args.as_of)
    service = build_boc_t2_service(args.statcan_cache_dir, args.fred_cache_dir, as_of=as_of)
    directions = service.get_series(DIRECTION_SERIES_ID, as_of=as_of)
    outcomes_t1 = pd.Series(
        (directions["value"].astype(float) != 0.0).astype(float).to_numpy(),
        index=pd.to_datetime(directions["timestamp"]).to_numpy(),
    )
    shifted = service.get_series(DIRECTION_T2_SERIES_ID, as_of=as_of)
    outcomes_t2 = pd.Series(
        (shifted["value"].astype(float) != 0.0).astype(float).to_numpy(),
        index=pd.to_datetime(shifted["released_at"]).to_numpy(),
    )

    rng = np.random.default_rng(SHUFFLE_SEED)
    all_correlations, all_corrections = [], []
    for horizon, specs, outcomes, t2 in (
        ("t+1", T1_SPECS, outcomes_t1, False),
        ("t+2", T2_SPECS, outcomes_t2, True),
    ):
        frames = {}
        for window, (spec_path, cache) in specs.items():
            frame = _window_frame(spec_path, cache, outcomes, t2=t2)
            frame[f"{PRIMARY_LANGUAGE}_shuffled"] = rng.permutation(frame[PRIMARY_LANGUAGE].to_numpy())
            frames[window] = frame
        columns = [
            "lex2_tilt",
            "lex2_intensity",
            "lex2_hedging",
            "lex_tilt",
            f"{PRIMARY_LANGUAGE}_shuffled",
        ]
        correlations, corrections = _run_horizon(horizon, frames, columns)
        all_correlations.append(correlations)
        all_corrections.append(corrections)

    correlations = pd.concat(all_correlations, ignore_index=True)
    corrections = pd.concat(all_corrections, ignore_index=True)
    correlations.to_csv(DATA_DIR / "market_error_correlations.csv", index=False)
    corrections.to_csv(DATA_DIR / "market_error_corrections.csv", index=False)

    print("=== does language correlate with the market's signed error? ===")
    print(correlations.to_string(index=False))
    print("\n=== out-of-sample correction (fit one window, score the other) ===")
    print(corrections.to_string(index=False))

    print("\n=== pre-registered verdicts (primary variable: " + PRIMARY_LANGUAGE + ") ===")
    for horizon in ("t+1", "t+2"):
        signs = correlations[(correlations["horizon"] == horizon) & (correlations["language"] == PRIMARY_LANGUAGE)][
            "corr_with_signed_error"
        ]
        consistent = bool((signs > 0).all() or (signs < 0).all())
        improved = corrections[(corrections["horizon"] == horizon) & (corrections["language"] == PRIMARY_LANGUAGE)][
            "improved"
        ]
        control = corrections[
            (corrections["horizon"] == horizon) & (corrections["language"] == f"{PRIMARY_LANGUAGE}_shuffled")
        ]["improved"]
        print(f"\n{horizon}:")
        print(f"  (1) same sign in both windows? {'YES' if consistent else 'NO'}  ({list(signs.round(3))})")
        print(f"  (2) beats the market both directions? {'YES' if bool(improved.all()) else 'NO'}")
        print(f"  negative control (shuffled) improved in {int(control.sum())}/2 directions")
        verdict = "EDGE DETECTED" if consistent and bool(improved.all()) else "NOT DETECTABLE AT THIS SAMPLE SIZE"
        print(f"  VERDICT: {verdict}")

    print(f"\nwrote market_error_correlations.csv and market_error_corrections.csv in {DATA_DIR}")


if __name__ == "__main__":
    main()
