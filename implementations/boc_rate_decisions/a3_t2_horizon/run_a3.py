"""Score the t+2 horizon, and compare the market's edge at t+1 against t+2.

The A3 question is not "which predictor wins" but "does the market's advantage
decay with horizon". A1 established that nothing here beats the futures curve
four weeks before a decision. If the curve is genuinely thinner ten weeks out,
the same methods should close some of that gap; if the gap is unchanged, then
switching target is not the way forward and the remaining A3 options can be
skipped.

Arms (all on the shifted t+2 target series, all free of LLM calls at predict
time):

``market_implied_t2``
    The two-leg futures decomposition from :mod:`implied_t2`.
``categorical_frequency``
    The library climatology baseline, run through a real ``ForecastContext`` so
    cutoff enforcement uses ``released_at`` — on this series the outcome is
    published two meetings after the row's own timestamp, so filtering by
    timestamp (as A1's offline helper does for the t+1 series) would leak.
``boc_logistic_macro`` / ``boc_logistic_macro_text``
    The A1 baseline and the A2 text arm, unchanged; both read
    ``task.target_series_id`` generically, so they retrain on t+2 labels
    without modification.

Outputs ``data/leaderboard_a3.csv``, ``data/per_meeting_rps_a3.csv`` and
``data/horizon_comparison_a3.csv``.

Usage
-----
``python -m boc_rate_decisions.a3_t2_horizon.run_a3``
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation import EvalSpec
from aieng.forecasting.evaluation.prediction import CategoricalForecast
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.methods import CategoricalFrequencyPredictor

from ..a1_market_implied.leaderboard import CATEGORY_ORDER, paired_deltas, per_meeting_rps, skill_table
from ..a2_text_features.features import StanceTable
from ..a2_text_features.text_logistic import BoCTextLogisticPredictor
from .build_table_t2 import DEFAULT_SPEC, origin_triples
from .data_t2 import DIRECTION_T2_SERIES_ID, build_boc_t2_service


USE_CASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"

#: A1/A2's per-meeting t+1 scores, reused for the horizon comparison.
A1_PER_MEETING = USE_CASE_DIR / "a1_market_implied" / "data" / "per_meeting_rps_a1.csv"
A2_PER_MEETING = USE_CASE_DIR / "a2_text_features" / "data" / "per_meeting_rps_a2.csv"


class _CachedContext:
    """Memoise ``get_series`` per origin so repeated arms re-read nothing."""

    def __init__(self, context: Any) -> None:
        self._context = context
        self._cache: dict[str, pd.DataFrame] = {}

    @property
    def as_of(self) -> Any:
        """The wrapped context's information cutoff."""
        return self._context.as_of

    def get_series(self, series_id: str) -> pd.DataFrame:
        """Return the cutoff-filtered series, reading it at most once."""
        if series_id not in self._cache:
            self._cache[series_id] = self._context.get_series(series_id)
        return self._cache[series_id]


def _arm_probabilities(
    predictor: Predictor,
    spec: EvalSpec,
    contexts: dict[pd.Timestamp, _CachedContext],
) -> pd.DataFrame:
    """Run one arm at every origin, indexed by the meeting it is scored on."""
    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for target, context in contexts.items():
        (prediction,) = predictor.predict(spec.task, cast(ForecastContext, context))
        payload = prediction.payload
        if not isinstance(payload, CategoricalForecast):
            raise TypeError(f"Expected a categorical forecast for {target:%Y-%m-%d}.")
        rows[target] = {label: float(payload.probabilities[label]) for label in CATEGORY_ORDER}
    return pd.DataFrame.from_dict(rows, orient="index").loc[:, CATEGORY_ORDER].sort_index()


def _horizon_comparison(t2_scores: dict[str, pd.Series]) -> pd.DataFrame | None:
    """Compare each method at both horizons on the **same target meetings**.

    This is the controlled version of the A3 question. Scoring t+1 over its own
    13 meetings and t+2 over its own 13 is not a horizon comparison: the two
    windows are shifted by one meeting and contain a different number of cuts,
    so a baseline can look better at t+2 purely because that window is easier.
    Restricting both horizons to the meetings they share removes that: the
    outcomes are then identical and only the lead time differs.

    The t+1 scores come from the committed A1/A2 per-meeting tables, which are
    indexed by the meeting they forecast — the same key A3 scores on.
    """
    if not A1_PER_MEETING.exists() or not A2_PER_MEETING.exists():
        return None

    a1 = pd.read_csv(A1_PER_MEETING, index_col=0, parse_dates=True)
    a2 = pd.read_csv(A2_PER_MEETING, index_col=0, parse_dates=True)
    t1_scores = {
        "market_implied": a1["market_implied"],
        "categorical_frequency": a1["categorical_frequency"],
        "boc_logistic_macro": a1["boc_logistic_macro"],
        "boc_logistic_macro_text": a2["boc_logistic_macro_text"],
    }

    common = t2_scores["market_implied_t2"].index
    for series in t1_scores.values():
        common = common.intersection(series.index)
    for series in t2_scores.values():
        common = common.intersection(series.index)
    common = pd.DatetimeIndex(sorted(common))

    market_t1 = float(t1_scores["market_implied"].loc[common].mean())
    market_t2 = float(t2_scores["market_implied_t2"].loc[common].mean())

    rows: list[dict[str, object]] = []
    for name, series in t1_scores.items():
        key = "market_implied_t2" if name == "market_implied" else name
        if key not in t2_scores:
            continue
        mean_t1 = float(series.loc[common].mean())
        mean_t2 = float(t2_scores[key].loc[common].mean())
        rows.append(
            {
                "predictor": name,
                "n_meetings": len(common),
                "mean_rps_t1": round(mean_t1, 4),
                "mean_rps_t2": round(mean_t2, 4),
                "skill_vs_market_t1": round(1.0 - mean_t1 / market_t1, 4),
                "skill_vs_market_t2": round(1.0 - mean_t2 / market_t2, 4),
            }
        )
    return pd.DataFrame(rows)


def _outcome_decomposition(scores: dict[str, pd.Series], outcomes: pd.Series) -> pd.DataFrame:
    """Split each arm's mean RPS by what actually happened at the meeting.

    Conditioning on the outcome is **descriptive, not a skill claim**: nobody
    can select the quiet meetings in advance, so "skill on holds" is not a
    number any method can bank. It answers a different and more useful
    question — *where does the difference between two methods live?* — and on
    this window the answer turns out to be the whole story.

    Parameters
    ----------
    scores : dict[str, pd.Series]
        Per-meeting RPS by predictor.
    outcomes : pd.Series
        Realised direction values indexed by meeting date.

    Returns
    -------
    pd.DataFrame
        One row per (subset, predictor): ``n_meetings``, ``mean_rps``, and
        skill against the market reference computed inside that subset.
    """
    index = pd.DataFrame(scores).index
    realised = outcomes.reindex(index)
    subsets = {
        "all": index,
        "moves (cut or hike)": index[realised != 0.0],
        "holds": index[realised == 0.0],
    }

    rows: list[dict[str, object]] = []
    for subset_name, subset_index in subsets.items():
        if len(subset_index) == 0:
            continue
        market_mean = float(scores["market_implied_t2"].loc[subset_index].mean())
        for predictor_id, series in scores.items():
            mean = float(series.loc[subset_index].mean())
            rows.append(
                {
                    "subset": subset_name,
                    "predictor_id": predictor_id,
                    "n_meetings": len(subset_index),
                    "mean_rps": round(mean, 4),
                    "skill_vs_market": round(1.0 - mean / market_mean, 4) if market_mean > 0 else None,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Score the t+2 arms and write the leaderboard plus the horizon comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--as-of", default="2026-09-19", help="Cutoff for reading realised outcomes.")
    parser.add_argument("--statcan-cache-dir", type=Path, default=USE_CASE_DIR / "data" / "statcan")
    parser.add_argument("--fred-cache-dir", type=Path, default=USE_CASE_DIR / "data" / "fred")
    args = parser.parse_args()

    spec = EvalSpec.model_validate(yaml.safe_load(args.spec.read_text()))
    triples = origin_triples(args.spec)
    as_of = pd.Timestamp(args.as_of)
    service = build_boc_t2_service(args.statcan_cache_dir, args.fred_cache_dir, as_of=as_of)

    shifted = service.get_series(DIRECTION_T2_SERIES_ID, as_of=as_of)
    # Score on the target meeting, not the row's own timestamp.
    outcomes = pd.Series(
        shifted["value"].astype(float).to_numpy(),
        index=pd.to_datetime(shifted["released_at"]).to_numpy(),
    )

    contexts = {target: _CachedContext(service.context(as_of=origin.to_pydatetime())) for origin, _, target in triples}
    arms: dict[str, Predictor] = {
        "categorical_frequency": CategoricalFrequencyPredictor(),
        "boc_logistic_macro": BoCTextLogisticPredictor(use_text=False),
        "boc_logistic_macro_text": BoCTextLogisticPredictor(StanceTable.from_csv()),
    }
    scores = {
        name: per_meeting_rps(_arm_probabilities(predictor, spec, contexts), outcomes)
        for name, predictor in arms.items()
    }

    market = pd.read_csv(DATA_DIR / "market_implied_t2.csv", parse_dates=["target_meeting"]).set_index("target_meeting")
    scores["market_implied_t2"] = per_meeting_rps(
        market.rename(columns={f"p_{label}": label for label in CATEGORY_ORDER}).loc[:, CATEGORY_ORDER], outcomes
    )

    per_meeting = pd.DataFrame(scores)
    per_meeting.index.name = "target_meeting"
    per_meeting.to_csv(DATA_DIR / "per_meeting_rps_a3.csv")

    board = skill_table(
        scores, references={"climatology": "categorical_frequency", "market": "market_implied_t2"}
    ).merge(paired_deltas(scores, "market_implied_t2"), on="predictor_id")
    board.to_csv(DATA_DIR / "leaderboard_a3.csv", index=False)

    decomposition = _outcome_decomposition(scores, outcomes)
    decomposition.to_csv(DATA_DIR / "outcome_decomposition_a3.csv", index=False)

    comparison = _horizon_comparison(scores)
    if comparison is not None:
        comparison.to_csv(DATA_DIR / "horizon_comparison_a3.csv", index=False)

    print(f"outcomes in window: {outcomes.reindex(per_meeting.index).value_counts().to_dict()}\n")
    print(per_meeting.round(4).to_string())
    print()
    print(board.round(4).to_string(index=False))
    if comparison is not None:
        print("\nhorizon comparison on identical target meetings (t+1 scores from A1/A2):")
        print(comparison.to_string(index=False))
    print("\nwhere the difference lives (descriptive; conditions on the outcome):")
    print(decomposition.to_string(index=False))
    print(f"\nwrote leaderboard, per-meeting, horizon-comparison and outcome-decomposition CSVs in {DATA_DIR}")


if __name__ == "__main__":
    main()
