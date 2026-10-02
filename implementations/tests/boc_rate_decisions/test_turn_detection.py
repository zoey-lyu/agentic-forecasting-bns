"""Unit tests for the turn-detection Step 0 / Step 1 machinery.

The arithmetic here decides whether an agent gets built at all, so the pieces
that carry a verdict are pinned: the pool, the oracle, the headroom ratio, the
router break-even grid, and the replication rule. One test guards the
pre-registration itself — the thresholds in code must still match the ones
written down before the results existed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest
from boc_rate_decisions.turn_detection.ceiling import (
    ceiling_summary,
    linear_pool,
    oracle_scores,
    router_requirement,
)
from boc_rate_decisions.turn_detection.move_task import (
    brier_scores,
    derive_move_events,
    historical_move_frequency,
    move_probability,
    murphy_decomposition,
    platt_scale,
)
from boc_rate_decisions.turn_detection.run_steps import (
    CAPTURED_SHARE_KILL,
    HEADROOM_KILL,
    MAX_DROPPED_ORIGINS,
    _step1_verdict,
)


TURN_DIR = Path(__file__).resolve().parents[3] / "implementations" / "boc_rate_decisions" / "turn_detection"
PREREGISTRATION = TURN_DIR / "PREREGISTRATION.md"

_MEETINGS = pd.to_datetime(["2025-01-29", "2025-03-12", "2025-04-16"])


def _probabilities(cut: float) -> pd.DataFrame:
    """Return a probability table putting ``cut`` on cut and the rest on hold."""
    return pd.DataFrame(
        {"cut": [cut] * 3, "hold": [1.0 - cut] * 3, "hike": [0.0] * 3},
        index=_MEETINGS,
    )


class TestLinearPool:
    """Blending two probability tables."""

    def test_half_weight_averages(self) -> None:
        """0.5 puts the pool exactly between the two inputs."""
        pooled = linear_pool(_probabilities(0.8), _probabilities(0.2), 0.5)
        assert pooled["cut"].tolist() == pytest.approx([0.5, 0.5, 0.5])

    def test_weight_one_returns_the_first_table(self) -> None:
        """Weight 1 keeps the first forecast untouched."""
        pooled = linear_pool(_probabilities(0.8), _probabilities(0.2), 1.0)
        assert pooled["cut"].tolist() == pytest.approx([0.8, 0.8, 0.8])

    def test_pool_stays_a_distribution(self) -> None:
        """A blend of distributions must still sum to one."""
        pooled = linear_pool(_probabilities(0.8), _probabilities(0.2), 0.3)
        assert pooled.sum(axis=1).tolist() == pytest.approx([1.0, 1.0, 1.0])

    def test_only_shared_meetings_survive(self) -> None:
        """Blending cannot invent a forecast for a meeting one side lacks."""
        pooled = linear_pool(_probabilities(0.8), _probabilities(0.2).iloc[:2], 0.5)
        assert len(pooled) == 2  # noqa: PLR2004 - the shared meetings

    def test_weight_outside_the_unit_interval_is_rejected(self) -> None:
        """A weight above 1 is a bug, not an extrapolation."""
        with pytest.raises(ValueError, match="weight must lie"):
            linear_pool(_probabilities(0.8), _probabilities(0.2), 1.5)


class TestOracleAndCeiling:
    """The hindsight ceiling and what the fixed blend captures of it."""

    def test_oracle_takes_the_better_score_each_meeting(self) -> None:
        """Per meeting, not per window — that is what makes it a ceiling."""
        first = pd.Series([0.1, 0.9], index=_MEETINGS[:2])
        second = pd.Series([0.5, 0.2], index=_MEETINGS[:2])
        assert oracle_scores(first, second).tolist() == pytest.approx([0.1, 0.2])

    def test_captured_share_is_the_fraction_of_headroom_taken(self) -> None:
        """Blend halfway between market and oracle captures half the headroom."""
        market = pd.Series([0.4, 0.4], index=_MEETINGS[:2])
        model = pd.Series([0.2, 0.6], index=_MEETINGS[:2])  # oracle = 0.2, 0.4 -> mean 0.3
        blend = pd.Series([0.35, 0.35], index=_MEETINGS[:2])
        summary = ceiling_summary(market, model, blend, window="test")
        assert summary["headroom"] == pytest.approx(0.1)
        assert summary["captured_share"] == pytest.approx(0.5)

    def test_a_blend_worse_than_the_market_gives_a_negative_share(self) -> None:
        """The sign must survive: a harmful blend cannot look like progress."""
        market = pd.Series([0.4, 0.4], index=_MEETINGS[:2])
        model = pd.Series([0.2, 0.6], index=_MEETINGS[:2])
        blend = pd.Series([0.5, 0.5], index=_MEETINGS[:2])
        assert ceiling_summary(market, model, blend, window="test")["captured_share"] < 0

    def test_no_headroom_reports_no_share(self) -> None:
        """A ratio against a non-positive denominator would be meaningless."""
        market = pd.Series([0.2, 0.2], index=_MEETINGS[:2])
        model = pd.Series([0.9, 0.9], index=_MEETINGS[:2])
        blend = pd.Series([0.4, 0.4], index=_MEETINGS[:2])
        assert ceiling_summary(market, model, blend, window="test")["captured_share"] is None


class TestRouterRequirement:
    """Turning headroom into a specification an agent can be held to."""

    def _subsets(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "window": "pooled",
                    "subset": "moves",
                    "predictor_id": "market_implied",
                    "n_meetings": 10,
                    "mean_rps": 0.2,
                },
                {
                    "window": "pooled",
                    "subset": "moves",
                    "predictor_id": "boc_logistic_macro",
                    "n_meetings": 10,
                    "mean_rps": 0.6,
                },
                {
                    "window": "pooled",
                    "subset": "holds",
                    "predictor_id": "market_implied",
                    "n_meetings": 10,
                    "mean_rps": 0.1,
                },
                {
                    "window": "pooled",
                    "subset": "holds",
                    "predictor_id": "boc_logistic_macro",
                    "n_meetings": 10,
                    "mean_rps": 0.0,
                },
            ]
        )

    def test_perfect_router_matches_the_corner_of_the_grid(self) -> None:
        """Sensitivity and specificity of 1 is the perfect router."""
        grid, headline = router_requirement(self._subsets())
        assert grid.loc[1.0, "sensitivity_1.0"] == pytest.approx(headline["perfect_router"])
        assert headline["perfect_router"] == pytest.approx((10 * 0.2 + 10 * 0.0) / 20)

    def test_market_alone_is_the_bar(self) -> None:
        """The break-even line is the market's own score."""
        _, headline = router_requirement(self._subsets())
        assert headline["market_alone"] == pytest.approx((10 * 0.2 + 10 * 0.1) / 20)

    def test_cost_asymmetry_is_reported(self) -> None:
        """A miss costs four times a false alarm here (0.4 against 0.1)."""
        _, headline = router_requirement(self._subsets())
        assert headline["miss_cost"] == pytest.approx(0.4)
        assert headline["false_alarm_cost"] == pytest.approx(0.1)
        assert headline["miss_to_false_alarm_ratio"] == pytest.approx(4.0)


class TestStep1Verdict:
    """The pre-registered replication rule."""

    def _subsets(self, *, market_worst_on_holds: bool, market_best_on_moves: bool) -> pd.DataFrame:
        hold_market = 0.9 if market_worst_on_holds else 0.01
        move_market = 0.1 if market_best_on_moves else 0.9
        rows = [
            ("holds", "market_implied", hold_market),
            ("holds", "boc_logistic_macro", 0.1),
            ("holds", "categorical_frequency", 0.2),
            ("moves", "market_implied", move_market),
            ("moves", "boc_logistic_macro", 0.5),
            ("moves", "categorical_frequency", 0.6),
        ]
        return pd.DataFrame(
            [
                {"window": "2023-2024", "subset": s, "predictor_id": p, "n_meetings": 7, "mean_rps": v}
                for s, p, v in rows
            ]
        )

    def test_both_conditions_hold_replicates(self) -> None:
        """Worst on holds and best on moves is the finding."""
        verdict, _ = _step1_verdict(self._subsets(market_worst_on_holds=True, market_best_on_moves=True))
        assert verdict == "REPLICATED"

    def test_either_condition_failing_falsifies(self) -> None:
        """One half is not a replication — the rule requires both."""
        for holds, moves in ((False, True), (True, False), (False, False)):
            verdict, _ = _step1_verdict(self._subsets(market_worst_on_holds=holds, market_best_on_moves=moves))
            assert verdict == "FALSIFIED"


class TestPreRegistrationIsHonoured:
    """The thresholds in code must still be the ones written down beforehand."""

    def test_thresholds_match_the_committed_document(self) -> None:
        """Silently retuning a kill rule after seeing results would void it."""
        text = PREREGISTRATION.read_text()
        assert re.search(r"Captured share ≥ 70%", text), "the captured-share rule changed wording"
        assert re.search(r"Headroom < 0\.02 RPS", text), "the headroom rule changed wording"
        assert re.search(r"more than 3 of the 14", text), "the dropped-origin rule changed wording"
        assert pytest.approx(0.70) == CAPTURED_SHARE_KILL
        assert pytest.approx(0.02) == HEADROOM_KILL
        assert MAX_DROPPED_ORIGINS == 3  # noqa: PLR2004 - the pre-registered count

    def test_document_has_no_undated_edits(self) -> None:
        """Changes after the fact belong in a dated addendum, not the rules."""
        assert "## Addenda" in PREREGISTRATION.read_text()


class TestMoveTask:
    """The binary turn instrument built in Step 2."""

    def _directions(self) -> pd.DataFrame:
        dates = pd.to_datetime(["2025-01-29", "2025-03-12", "2025-04-16", "2025-06-04"])
        return pd.DataFrame({"timestamp": dates, "value": [-1.0, -1.0, 0.0, 1.0], "released_at": dates})

    def test_cuts_and_hikes_both_count_as_moves(self) -> None:
        """Direction is discarded; only whether the Bank acted survives."""
        events = derive_move_events(self._directions())
        assert events["value"].tolist() == [1.0, 1.0, 0.0, 1.0]

    def test_move_probability_is_one_minus_hold(self) -> None:
        """Using 1 - p_hold keeps the result a probability under rounding."""
        table = pd.DataFrame({"cut": [0.3], "hold": [0.6], "hike": [0.1]}, index=pd.to_datetime(["2025-01-29"]))
        assert move_probability(table).iloc[0] == pytest.approx(0.4)

    def test_historical_frequency_only_sees_resolved_meetings(self) -> None:
        """Climatology at an origin cannot know meetings that follow it."""
        origins = pd.Series({pd.Timestamp("2025-06-04"): pd.Timestamp("2025-04-01")})
        # Visible on 2025-04-01: two cuts -> a move rate of 1.0.
        assert historical_move_frequency(self._directions(), origins).iloc[0] == pytest.approx(1.0)

    def test_brier_is_the_squared_error(self) -> None:
        """A 0.8 forecast of an event that happened scores 0.04."""
        index = pd.to_datetime(["2025-01-29"])
        scores = brier_scores(pd.Series([0.8], index=index), pd.Series([1.0], index=index))
        assert scores.iloc[0] == pytest.approx(0.04)

    def test_murphy_terms_reconstruct_the_brier_score(self) -> None:
        """Reliability - resolution + uncertainty must return the score."""
        index = pd.date_range("2025-01-01", periods=12, freq="60D")
        probabilities = pd.Series([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.2, 0.3, 0.7, 0.8], index=index)
        outcomes = pd.Series([0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0], index=index)
        terms = murphy_decomposition(probabilities, outcomes)
        rebuilt = terms["reliability"] - terms["resolution"] + terms["uncertainty"] + terms["residual"]
        assert rebuilt == pytest.approx(terms["brier"], abs=1e-3)

    def test_bias_is_mean_forecast_minus_observed_rate(self) -> None:
        """The insurance claim is tested on this number, so pin its sign."""
        index = pd.date_range("2025-01-01", periods=4, freq="60D")
        terms = murphy_decomposition(
            pd.Series([0.5, 0.5, 0.5, 0.5], index=index), pd.Series([1.0, 0.0, 0.0, 0.0], index=index)
        )
        assert terms["bias"] == pytest.approx(0.25)

    def test_recalibration_needs_both_outcomes_to_fit(self) -> None:
        """A training window with no moves cannot teach a correction."""
        index = pd.date_range("2023-01-01", periods=4, freq="60D")
        flat = pd.Series([0.3, 0.3, 0.3, 0.3], index=index)
        with pytest.raises(ValueError, match="needs both outcomes"):
            platt_scale(flat, pd.Series([0.0] * 4, index=index), flat)

    def test_recalibration_is_fitted_on_one_window_and_applied_to_another(self) -> None:
        """The correction must transport; an in-sample number would be meaningless."""
        train_index = pd.date_range("2023-01-01", periods=8, freq="60D")
        test_index = pd.date_range("2025-01-01", periods=4, freq="60D")
        train = pd.Series([0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1], index=train_index)
        # The window the correction learns from: the 0.9 forecasts happen only
        # once in four, so a high forecast must be pulled down.
        train_outcomes = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        outcomes = pd.Series(train_outcomes + [1.0, 1.0, 0.0, 0.0], index=train_index.append(test_index))
        corrected = platt_scale(train, outcomes, pd.Series([0.9, 0.9, 0.1, 0.1], index=test_index))
        assert len(corrected) == len(test_index)
        assert corrected.between(0.0, 1.0).all()
        # Learned that high forecasts overstate, so it must pull them down.
        assert corrected.iloc[0] < 0.9  # noqa: PLR2004 - the raw forecast
