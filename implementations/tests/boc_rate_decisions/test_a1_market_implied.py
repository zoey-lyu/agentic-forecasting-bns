"""Unit tests for the A1 market-implied reference forecast.

Covers the three things that can silently go wrong: the futures arithmetic
(``implied``), the replay predictor's contract with the harness
(``market_implied``), and the two-reference skill table (``leaderboard``).
Synthetic settlement tables throughout — no network, no LLM. One integration
test checks the committed table against the protected eval spec so a spec that
gains an origin fails here instead of at eval time.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
import yaml
from aieng.forecasting.evaluation.task import ForecastingTask, TaskCategory
from boc_rate_decisions.a1_market_implied.implied import contract_symbol, implied_quote, move_to_probabilities
from boc_rate_decisions.a1_market_implied.leaderboard import (
    climatology_probabilities,
    per_meeting_rps,
    skill_table,
)
from boc_rate_decisions.a1_market_implied.market_implied import MarketImpliedPredictor, load_market_table


USE_CASE_DIR = Path(__file__).resolve().parents[3] / "implementations" / "boc_rate_decisions"
EVAL_SPEC = USE_CASE_DIR / "specs" / "boc_rate_direction_eval.yaml"

_CATEGORIES = [
    TaskCategory(label="cut", value=-1.0),
    TaskCategory(label="hold", value=0.0),
    TaskCategory(label="hike", value=1.0),
]


def _task() -> ForecastingTask:
    return ForecastingTask(
        task_id="boc_rate_direction_next_meeting",
        target_series_id="boc_rate_decision_direction",
        horizons=[28],
        frequency="D",
        description="3-way direction test task.",
        payload_type="categorical",
        categories=_CATEGORIES,
    )


class _StubContext:
    """Minimal stand-in for ``ForecastContext``: the predictor only reads ``as_of``."""

    def __init__(self, as_of: datetime) -> None:
        self.as_of = as_of


def _corra(level: float, start: str = "2025-01-01", end: str = "2025-04-30") -> pd.Series:
    """Flat daily CORRA at ``level`` over a date range."""
    index = pd.date_range(start, end, freq="D")
    return pd.Series([level] * len(index), index=index)


def _settlements(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """Build a settlement table from ``(date, symbol, price)`` triples."""
    return pd.DataFrame(
        [{"date": pd.Timestamp(d), "symbol": s, "settlement_price": p, "open_interest": 100.0} for d, s, p in rows]
    )


class TestMoveToProbabilities:
    """The expected-move to distribution mapping."""

    def test_full_step_down_is_a_certain_cut(self) -> None:
        """One whole step priced means probability 1 on the cut."""
        assert move_to_probabilities(-0.25) == (1.0, 0.0, 0.0)

    def test_half_step_splits_evenly(self) -> None:
        """Half a step priced means an even chance of a move."""
        p_cut, p_hold, p_hike = move_to_probabilities(-0.125)
        assert p_cut == pytest.approx(0.5)
        assert p_hold == pytest.approx(0.5)
        assert p_hike == 0.0

    def test_no_move_is_a_certain_hold(self) -> None:
        """A flat curve means the market expects nothing."""
        assert move_to_probabilities(0.0) == (0.0, 1.0, 0.0)

    def test_beyond_one_step_clamps(self) -> None:
        """A 50 bp move priced is still just 'a cut' on a direction task."""
        assert move_to_probabilities(-0.50) == (1.0, 0.0, 0.0)

    def test_upward_move_prices_a_hike(self) -> None:
        """Positive expected moves land on the hike category."""
        assert move_to_probabilities(0.25) == (0.0, 0.0, 1.0)


class TestContractSymbol:
    """Exchange symbol construction."""

    def test_month_code_and_two_digit_year(self) -> None:
        """January 2025 is the F contract."""
        assert contract_symbol("COA", pd.Timestamp("2025-01-01")) == "COAF25"

    def test_december_uses_z(self) -> None:
        """December is the Z contract."""
        assert contract_symbol("COA", pd.Timestamp("2025-12-01")) == "COAZ25"


class TestImpliedQuote:
    """Estimator selection and the day-count arithmetic."""

    def test_next_month_used_when_following_month_is_clean(self) -> None:
        """A late-month decision reads the post-decision rate straight off M+1."""
        # Meeting 2025-01-29, next meeting only in March: February is clean.
        quote = implied_quote(
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2025-01-29"),
            coa=_settlements(
                [
                    ("2024-12-31", "COAF25", 96.72),
                    ("2024-12-31", "COAG25", 96.875),  # implies 3.125 = 3.25 - 0.125
                ]
            ),
            corra=_corra(3.25, start="2024-12-01"),
            meeting_dates=[pd.Timestamp("2025-01-29"), pd.Timestamp("2025-03-12")],
        )
        assert quote.method == "next_month"
        assert quote.expected_move == pytest.approx(-0.125)
        assert quote.p_cut == pytest.approx(0.5)
        assert quote.quote_date == pd.Timestamp("2024-12-31")

    def test_meeting_month_used_when_following_month_has_a_decision(self) -> None:
        """A mid-month decision falls back to splitting its own month."""
        # Meeting 2025-03-12 (effective Thu 2025-03-13); March has 31 days, so
        # 12 days at 3.00 and 19 days at the post-decision rate. Pricing the
        # contract at an average of 2.9 implies a post-decision rate of
        # (2.9*31 - 12*3.00)/19 = 2.8368, i.e. a move of -0.1632.
        quote = implied_quote(
            pd.Timestamp("2025-02-12"),
            pd.Timestamp("2025-03-12"),
            coa=_settlements([("2025-02-12", "COAH25", 97.10), ("2025-02-12", "COAJ25", 97.20)]),
            corra=_corra(3.00, start="2025-01-01"),
            meeting_dates=[pd.Timestamp("2025-03-12"), pd.Timestamp("2025-04-16")],
        )
        assert quote.method == "meeting_month"
        assert quote.expected_move == pytest.approx(-0.1632, abs=1e-4)
        assert quote.p_cut == pytest.approx(0.6526, abs=1e-4)

    def test_intervening_meeting_is_rejected(self) -> None:
        """Another announcement between origin and target breaks attribution."""
        with pytest.raises(ValueError, match="between origin and meeting"):
            implied_quote(
                pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-03-12"),
                coa=_settlements([("2024-12-31", "COAH25", 97.10)]),
                corra=_corra(3.25, start="2024-12-01"),
                meeting_dates=[pd.Timestamp("2025-01-29"), pd.Timestamp("2025-03-12")],
            )

    def test_missing_settlement_raises(self) -> None:
        """No contract at the origin is an error, never a silent guess."""
        with pytest.raises(ValueError, match="No COA settlements"):
            implied_quote(
                pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-01-29"),
                coa=_settlements([("2025-01-15", "COAF25", 96.72)]),
                corra=_corra(3.25, start="2024-12-01"),
                meeting_dates=[pd.Timestamp("2025-01-29")],
            )


class TestMarketImpliedPredictor:
    """Replay behaviour and the harness contract."""

    def _table(self, tmp_path: Path, origin: str = "2025-01-01") -> Path:
        path = tmp_path / "table.csv"
        pd.DataFrame(
            [
                {
                    "origin_date": origin,
                    "meeting_date": "2025-01-29",
                    "p_cut": 0.78,
                    "p_hold": 0.22,
                    "p_hike": 0.0,
                    "method": "next_month",
                    "corra_now": 3.31,
                    "expected_move_pp": -0.195,
                    "source": "test",
                }
            ]
        ).to_csv(path, index=False)
        return path

    def test_emits_the_committed_row(self, tmp_path: Path) -> None:
        """The prediction is the table row, with the horizon applied."""
        predictor = MarketImpliedPredictor(self._table(tmp_path))
        (prediction,) = predictor.predict(_task(), _StubContext(datetime(2025, 1, 1)))
        assert prediction.payload.probabilities["cut"] == pytest.approx(0.78)
        assert prediction.forecast_date == datetime(2025, 1, 29)
        assert prediction.metadata["method"] == "next_month"

    def test_unknown_origin_raises_in_strict_mode(self, tmp_path: Path) -> None:
        """A missing quote must fail loudly rather than score as a guess."""
        predictor = MarketImpliedPredictor(self._table(tmp_path))
        with pytest.raises(ValueError, match="No market-implied quote"):
            predictor.predict(_task(), _StubContext(datetime(2025, 6, 1)))

    def test_unknown_origin_skipped_when_not_strict(self, tmp_path: Path) -> None:
        """Non-strict mode returns no prediction instead."""
        predictor = MarketImpliedPredictor(self._table(tmp_path), strict=False)
        assert predictor.predict(_task(), _StubContext(datetime(2025, 6, 1))) == []

    def test_probabilities_must_sum_to_one(self, tmp_path: Path) -> None:
        """A malformed table is rejected at load time."""
        path = tmp_path / "bad.csv"
        pd.DataFrame(
            [
                {
                    "origin_date": "2025-01-01",
                    "meeting_date": "2025-01-29",
                    "p_cut": 0.5,
                    "p_hold": 0.2,
                    "p_hike": 0.0,
                    "method": "next_month",
                }
            ]
        ).to_csv(path, index=False)
        with pytest.raises(ValueError, match="do not sum to 1"):
            load_market_table(path)


class TestLeaderboard:
    """Scoring helpers behind the two-reference table."""

    def test_per_meeting_rps_matches_hand_computation(self) -> None:
        """RPS uses the harness's unnormalised cumulative convention."""
        probabilities = pd.DataFrame({"cut": [0.8], "hold": [0.2], "hike": [0.0]}, index=[pd.Timestamp("2025-01-29")])
        outcomes = pd.Series([-1.0], index=[pd.Timestamp("2025-01-29")])
        # Realised cut: (0.8 - 1)^2 + (1.0 - 1)^2 = 0.04.
        assert per_meeting_rps(probabilities, outcomes).iloc[0] == pytest.approx(0.04)

    def test_climatology_uses_only_visible_history(self) -> None:
        """Frequencies are computed from meetings resolved before the origin."""
        directions = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-24", "2024-03-06", "2024-06-05"]),
                "value": [0.0, 0.0, -1.0],
            }
        )
        origins = pd.Series({pd.Timestamp("2024-07-24"): pd.Timestamp("2024-04-01")})
        frame = climatology_probabilities(directions, origins)
        # Only the first two meetings are visible on 2024-04-01: both holds.
        assert frame.loc[pd.Timestamp("2024-07-24"), "hold"] == pytest.approx(1.0)
        assert frame.loc[pd.Timestamp("2024-07-24"), "cut"] == pytest.approx(0.0)

    def test_skill_columns_are_relative_to_each_reference(self) -> None:
        """Halving the reference's score is a skill of 0.5 against it."""
        index = pd.to_datetime(["2025-01-29", "2025-03-12"])
        scores = {
            "market": pd.Series([0.1, 0.1], index=index),
            "climate": pd.Series([0.2, 0.2], index=index),
            "model": pd.Series([0.2, 0.2], index=index),
        }
        board = skill_table(scores, references={"climatology": "climate", "market": "market"}).set_index("predictor_id")
        assert board.loc["market", "skill_vs_climatology"] == pytest.approx(0.5)
        assert board.loc["model", "skill_vs_market"] == pytest.approx(-1.0)

    def test_means_are_restricted_to_common_meetings(self) -> None:
        """A predictor covering extra meetings is still compared like for like."""
        index = pd.to_datetime(["2025-01-29", "2025-03-12"])
        scores = {
            "market": pd.Series([0.1, 0.1, 0.9], index=index.append(pd.to_datetime(["2025-04-16"]))),
            "model": pd.Series([0.2, 0.2], index=index),
        }
        board = skill_table(scores, references={"market": "market"}).set_index("predictor_id")
        assert board.loc["market", "mean_rps"] == pytest.approx(0.1)
        assert board.loc["market", "n_meetings"] == 2
        assert board.loc["market", "n_meetings_available"] == 3


class TestCommittedTable:
    """The committed table must stay in step with the protected eval spec."""

    def test_every_eval_origin_has_a_quote(self) -> None:
        """Adding an origin to the spec without rebuilding the table fails here."""
        spec = yaml.safe_load(EVAL_SPEC.read_text())
        table = load_market_table()
        missing = [o for o in spec["origin_dates"] if pd.Timestamp(o) not in table.index]
        assert not missing, f"Origins without a market-implied quote: {missing}"

    def test_quotes_are_valid_distributions(self) -> None:
        """Every committed row is a probability distribution."""
        table = load_market_table()
        assert (table[["p_cut", "p_hold", "p_hike"]] >= 0).all().all()
        assert table[["p_cut", "p_hold", "p_hike"]].sum(axis=1).sub(1.0).abs().max() < 1e-6
