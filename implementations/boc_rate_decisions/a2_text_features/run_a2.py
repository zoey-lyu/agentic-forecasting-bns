"""Score the text-feature arm against its ablation, its control and the market.

Runs four arms over the protected eval window and reports both skill columns
from A1, so the A2 question — *does reading the statement add anything?* — is
answered against the bar that matters (the futures curve), not against
climatology.

Arms
----
``boc_logistic_macro``
    Ablation: the existing four macro features. Checked against A1's stored
    per-meeting RPS; a mismatch aborts, because the comparison is only
    meaningful if this arm is byte-for-byte the old baseline.
``boc_logistic_macro_text``
    Plus the three statement-stance features.
``boc_logistic_macro_text_shuffled``
    Same features, same parameter count, stances permuted across meetings. One
    permutation is one draw, so the script also runs a permutation null of
    ``--n-shuffles`` further seeds and reports where the real arm falls in it.
``market_implied`` / ``categorical_frequency``
    The two references, from A1.

Everything here is free: the stances were extracted once into a committed CSV,
and no arm calls an LLM at predict time.

Usage
-----
``python -m boc_rate_decisions.a2_text_features.run_a2``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation import EvalSpec
from aieng.forecasting.evaluation.prediction import CategoricalForecast

from ..a1_market_implied.build_table import DEFAULT_SPEC, origins_from_spec
from ..a1_market_implied.leaderboard import (
    CATEGORY_ORDER,
    climatology_probabilities,
    paired_deltas,
    per_meeting_rps,
    skill_table,
)
from ..a1_market_implied.market_implied import load_market_table
from ..data import DIRECTION_SERIES_ID, build_boc_service
from .features import StanceTable
from .text_logistic import BoCTextLogisticPredictor


USE_CASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"

#: A1's per-meeting scores, used to verify the ablation arm reproduces the baseline.
A1_PER_MEETING = USE_CASE_DIR / "a1_market_implied" / "data" / "per_meeting_rps_a1.csv"

#: Seed for the shuffled-stance control. Fixed so the control is reproducible.
SHUFFLE_SEED = 42


class _CachedContext:
    """A ``ForecastContext`` wrapper that memoises ``get_series`` per origin.

    Twenty-odd arms are fitted at the same origins here, and each one asks the
    context for the same five cutoff-filtered series. The cutoff is fixed by
    the origin, so the answer cannot differ between arms; caching it turns the
    permutation null from minutes per arm into seconds. Only this script uses
    it — the predictors still take a plain context, exactly as the harness
    hands them one.
    """

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
    predictor: BoCTextLogisticPredictor,
    spec: EvalSpec,
    contexts: dict[pd.Timestamp, _CachedContext],
) -> pd.DataFrame:
    """Run one arm at every origin, returning probabilities indexed by meeting."""
    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for meeting, context in contexts.items():
        (prediction,) = predictor.predict(spec.task, cast(ForecastContext, context))
        payload = prediction.payload
        if not isinstance(payload, CategoricalForecast):
            raise TypeError(f"Expected a categorical forecast for {meeting:%Y-%m-%d}.")
        rows[meeting] = {label: float(payload.probabilities[label]) for label in CATEGORY_ORDER}
    return pd.DataFrame.from_dict(rows, orient="index").loc[:, CATEGORY_ORDER].sort_index()


def _permutation_null(
    n_shuffles: int,
    *,
    spec: EvalSpec,
    contexts: dict[pd.Timestamp, _CachedContext],
    outcomes: pd.Series,
    real_mean: float,
    macro_mean: float,
) -> dict[str, object] | None:
    """Refit the text arm on ``n_shuffles`` permuted stance tables.

    A single shuffled control is a single draw: it can land anywhere by luck.
    Refitting on many permutations gives the distribution of "what a text
    block with no information would have scored", and the real arm's rank in
    it is the honest read on whether the statements carried anything. Returns
    ``None`` when the null is skipped.
    """
    if n_shuffles <= 0:
        return None

    means: list[float] = []
    for seed in range(n_shuffles):
        predictor = BoCTextLogisticPredictor(StanceTable.from_csv(shuffle_seed=seed), predictor_suffix=f"s{seed}")
        means.append(float(per_meeting_rps(_arm_probabilities(predictor, spec, contexts), outcomes).mean()))

    better = sum(1 for mean in means if mean <= real_mean)
    p_value = (1 + better) / (n_shuffles + 1)
    summary: dict[str, object] = {
        "n_shuffles": n_shuffles,
        "real_text_mean_rps": round(real_mean, 4),
        "macro_only_mean_rps": round(macro_mean, 4),
        "null_mean": round(float(np.mean(means)), 4),
        "null_min": round(float(np.min(means)), 4),
        "null_max": round(float(np.max(means)), 4),
        "n_null_at_least_as_good": better,
        "p_value": round(p_value, 4),
    }
    print(
        f"\npermutation null ({n_shuffles} shuffles): real text arm {real_mean:.4f}, "
        f"null mean {summary['null_mean']:.4f} (range {summary['null_min']:.4f}-{summary['null_max']:.4f}); "
        f"{better}/{n_shuffles} shuffles matched or beat it -> p = {p_value:.3f}"
    )
    return summary


def main() -> None:
    """Score the arms, check the ablation, and write the A2 leaderboard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--as-of", default="2026-09-19", help="Cutoff for reading realised outcomes.")
    parser.add_argument("--statcan-cache-dir", type=Path, default=USE_CASE_DIR / "data" / "statcan")
    parser.add_argument("--fred-cache-dir", type=Path, default=USE_CASE_DIR / "data" / "fred")
    parser.add_argument(
        "--n-shuffles",
        type=int,
        default=20,
        help="Permutation-null draws for the text effect (0 to skip).",
    )
    args = parser.parse_args()

    spec = EvalSpec.model_validate(yaml.safe_load(args.spec.read_text()))
    pairs = origins_from_spec(args.spec)
    service = build_boc_service(statcan_cache_dir=args.statcan_cache_dir, fred_cache_dir=args.fred_cache_dir)

    directions = service.get_series(DIRECTION_SERIES_ID, as_of=pd.Timestamp(args.as_of))
    outcomes = pd.Series(
        directions["value"].astype(float).to_numpy(),
        index=pd.to_datetime(directions["timestamp"]).to_numpy(),
    )
    origins = pd.Series({meeting: origin for origin, meeting in pairs}, name="origin")

    stance = StanceTable.from_csv()
    shuffled = StanceTable.from_csv(shuffle_seed=SHUFFLE_SEED)
    print(f"stance table: {len(stance)} statements\n")

    arms = {
        "boc_logistic_macro": BoCTextLogisticPredictor(use_text=False),
        "boc_logistic_macro_text": BoCTextLogisticPredictor(stance),
        "boc_logistic_macro_text_shuffled": BoCTextLogisticPredictor(shuffled, predictor_suffix="shuffled"),
    }
    contexts = {meeting: _CachedContext(service.context(as_of=origin.to_pydatetime())) for origin, meeting in pairs}
    scores = {
        name: per_meeting_rps(_arm_probabilities(predictor, spec, contexts), outcomes)
        for name, predictor in arms.items()
    }

    market = load_market_table().reset_index().set_index("meeting_date")
    scores["market_implied"] = per_meeting_rps(
        market.rename(columns={f"p_{label}": label for label in CATEGORY_ORDER}).loc[:, CATEGORY_ORDER], outcomes
    )
    scores["categorical_frequency"] = per_meeting_rps(climatology_probabilities(directions, origins), outcomes)

    # The ablation arm must be the old baseline, or the text comparison is not
    # a text comparison.
    if A1_PER_MEETING.exists():
        stored = pd.read_csv(A1_PER_MEETING, index_col=0, parse_dates=True)["boc_logistic_macro"]
        shared = scores["boc_logistic_macro"].index.intersection(stored.index)
        gap = (scores["boc_logistic_macro"].loc[shared] - stored.loc[shared]).abs().max()
        print(f"ablation arm vs A1 baseline over {len(shared)} meetings: max |ΔRPS| = {gap:.6f}")
        if gap > 1e-9:
            raise SystemExit("Ablation arm does not reproduce the macro baseline; aborting.")

    per_meeting = pd.DataFrame(scores)
    per_meeting.index.name = "meeting_date"
    per_meeting.to_csv(DATA_DIR / "per_meeting_rps_a2.csv")

    board = skill_table(scores, references={"climatology": "categorical_frequency", "market": "market_implied"})
    board = board.merge(
        paired_deltas(scores, "market_implied").rename(
            columns={
                "delta_vs_reference": "delta_vs_market",
                "ci_lo": "market_ci_lo",
                "ci_hi": "market_ci_hi",
                "significant": "vs_market_significant",
            }
        ),
        on="predictor_id",
    )
    board = board.merge(
        paired_deltas(scores, "boc_logistic_macro").rename(
            columns={
                "delta_vs_reference": "delta_vs_macro_only",
                "ci_lo": "macro_ci_lo",
                "ci_hi": "macro_ci_hi",
                "significant": "vs_macro_significant",
            }
        ),
        on="predictor_id",
    )
    board.to_csv(DATA_DIR / "leaderboard_a2.csv", index=False)

    null = _permutation_null(
        args.n_shuffles,
        spec=spec,
        contexts=contexts,
        outcomes=outcomes,
        real_mean=float(scores["boc_logistic_macro_text"].mean()),
        macro_mean=float(scores["boc_logistic_macro"].mean()),
    )
    if null is not None:
        (DATA_DIR / "permutation_null_a2.json").write_text(json.dumps(null, indent=2))

    print()
    print(per_meeting.round(4).to_string())
    print()
    print(
        board.loc[
            :,
            [
                "predictor_id",
                "mean_rps",
                "n_meetings",
                "skill_vs_climatology",
                "skill_vs_market",
                "delta_vs_macro_only",
                "macro_ci_lo",
                "macro_ci_hi",
            ],
        ]
        .round(4)
        .to_string(index=False)
    )
    print(f"\nwrote {DATA_DIR / 'leaderboard_a2.csv'} and {DATA_DIR / 'per_meeting_rps_a2.csv'}")


if __name__ == "__main__":
    main()
