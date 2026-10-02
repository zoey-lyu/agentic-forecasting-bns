"""Rebuild the BoC eval leaderboard with skill against climatology *and* the market.

Assembles per-meeting RPS for every method on the 2025-2026 protected window
and reports both skill columns. Three sources of scores, all free of LLM calls:

- **market_implied** — scored here from the committed futures table.
- **categorical_frequency** (climatology) and **boc_logistic_macro** — both
  recomputed here from cached series. The logistic run is re-scored rather
  than read from disk so the offline pipeline can be checked against the
  stored per-origin RPS of the earlier harness run; the check is printed.
- **agent (identified / anonymised)** — read from
  ``data/recall_probe_eval/per_origin_rps.csv``, the stored output of the
  de-identification probe. Re-running them would cost LLM calls and burn the
  spec's ``max_runs`` budget for no new information.

Outputs ``data/per_meeting_rps_a1.csv`` and ``data/leaderboard_a1.csv``.

Usage
-----
``python -m boc_rate_decisions.a1_market_implied.run_leaderboard``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from aieng.forecasting.evaluation import EvalSpec
from aieng.forecasting.evaluation.prediction import CategoricalForecast

from ..data import DIRECTION_SERIES_ID, build_boc_service
from ..predictors import BoCLogisticPredictor
from .build_table import DEFAULT_SPEC, origins_from_spec
from .leaderboard import CATEGORY_ORDER, climatology_probabilities, paired_deltas, per_meeting_rps, skill_table
from .market_implied import load_market_table


USE_CASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"

#: Stored per-meeting RPS from the de-identification probe (agent arms). The
#: agent is **not** reproducible run to run — two runs minutes apart move 5 of
#: 14 cells and shift the mean by 0.02 — so every available run is read and the
#: per-meeting scores are averaged across them. Averaging scores estimates the
#: expected score of a stochastic method, which is the quantity a leaderboard
#: row should carry; a single run is one draw presented as a measurement.
STORED_AGENT_RUNS = sorted((USE_CASE_DIR / "data").glob("recall_probe_eval_run*/per_origin_rps.csv"))

#: Display names for the stored probe columns.
STORED_AGENT_COLUMNS = {
    "C_identified": "agent_identified",
    "B_anonymized": "agent_anonymised",
    "A_logistic": "boc_logistic_macro_stored",
}


def _load_outcomes(as_of: pd.Timestamp, statcan_dir: Path, fred_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Return the full direction history and its outcome series, both dated."""
    service = build_boc_service(statcan_cache_dir=statcan_dir, fred_cache_dir=fred_dir)
    directions = service.get_series(DIRECTION_SERIES_ID, as_of=as_of)
    outcomes = pd.Series(
        directions["value"].astype(float).to_numpy(),
        index=pd.to_datetime(directions["timestamp"]).to_numpy(),
    )
    return directions, outcomes


def _logistic_probabilities(spec_path: Path, statcan_dir: Path, fred_dir: Path) -> pd.DataFrame:
    """Run the logistic baseline at every origin in the spec (no LLM, no budget)."""
    spec = EvalSpec.model_validate(yaml.safe_load(spec_path.read_text()))
    service = build_boc_service(statcan_cache_dir=statcan_dir, fred_cache_dir=fred_dir)
    predictor = BoCLogisticPredictor()

    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for origin, meeting in origins_from_spec(spec_path):
        context = service.context(as_of=origin.to_pydatetime())
        (prediction,) = predictor.predict(spec.task, context)
        payload = prediction.payload
        if not isinstance(payload, CategoricalForecast):
            raise TypeError(f"Expected a categorical forecast at {origin:%Y-%m-%d}; got {type(payload).__name__}.")
        rows[meeting] = {label: float(payload.probabilities[label]) for label in CATEGORY_ORDER}
    return pd.DataFrame.from_dict(rows, orient="index").loc[:, CATEGORY_ORDER].sort_index()


def main() -> None:
    """Score every method on the protected window and print the leaderboard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--as-of", default="2026-09-19", help="Cutoff for reading realised outcomes.")
    parser.add_argument("--statcan-cache-dir", type=Path, default=USE_CASE_DIR / "data" / "statcan")
    parser.add_argument("--fred-cache-dir", type=Path, default=USE_CASE_DIR / "data" / "fred")
    args = parser.parse_args()

    as_of = pd.Timestamp(args.as_of)
    directions, outcomes = _load_outcomes(as_of, args.statcan_cache_dir, args.fred_cache_dir)
    origins = pd.Series(
        {meeting: origin for origin, meeting in origins_from_spec(args.spec)},
        name="origin",
    )

    market = load_market_table().reset_index().set_index("meeting_date")
    market_probabilities = market.rename(columns={f"p_{label}": label for label in CATEGORY_ORDER}).loc[
        :, CATEGORY_ORDER
    ]

    scores: dict[str, pd.Series] = {
        "market_implied": per_meeting_rps(market_probabilities, outcomes),
        "categorical_frequency": per_meeting_rps(climatology_probabilities(directions, origins), outcomes),
        "boc_logistic_macro": per_meeting_rps(
            _logistic_probabilities(args.spec, args.statcan_cache_dir, args.fred_cache_dir), outcomes
        ),
    }

    if STORED_AGENT_RUNS:
        runs = [pd.read_csv(path, index_col=0, parse_dates=True) for path in STORED_AGENT_RUNS]
        print(f"agent arms averaged over {len(runs)} stored runs: {[p.parent.name for p in STORED_AGENT_RUNS]}")
        for column, name in STORED_AGENT_COLUMNS.items():
            present = [run[column].sort_index() for run in runs if column in run.columns]
            if not present:
                continue
            stacked = pd.concat(present, axis=1)
            scores[name] = stacked.mean(axis=1)
            if len(present) > 1:
                spread = (stacked.max(axis=1) - stacked.min(axis=1)).mean()
                print(
                    f"  {name}: mean RPS per run {[round(float(s.mean()), 4) for s in present]}, "
                    f"mean per-meeting spread {spread:.4f}"
                )

    # Cross-check: the freshly run logistic must reproduce the stored harness run.
    if "boc_logistic_macro_stored" in scores:
        stored_logistic = scores.pop("boc_logistic_macro_stored")
        shared = scores["boc_logistic_macro"].index.intersection(stored_logistic.index)
        gap = (scores["boc_logistic_macro"].loc[shared] - stored_logistic.loc[shared]).abs().max()
        print(f"logistic reproduction check over {len(shared)} meetings: max |ΔRPS| = {gap:.6f}\n")

    per_meeting = pd.DataFrame(scores)
    per_meeting.index.name = "meeting_date"
    per_meeting.to_csv(DATA_DIR / "per_meeting_rps_a1.csv")

    board = skill_table(
        scores,
        references={"climatology": "categorical_frequency", "market": "market_implied"},
    )
    board = board.merge(paired_deltas(scores, "market_implied"), on="predictor_id", how="left")
    board.to_csv(DATA_DIR / "leaderboard_a1.csv", index=False)

    print(per_meeting.round(4).to_string())
    print()
    print(board.round(4).to_string(index=False))
    print(f"\nwrote {DATA_DIR / 'leaderboard_a1.csv'} and {DATA_DIR / 'per_meeting_rps_a1.csv'}")


if __name__ == "__main__":
    main()
