"""Unit tests for the A3 t+2 horizon target and its market reference.

Two things carry the whole experiment and both fail silently if wrong: the
shifted target series (a mis-shift would score every method against the wrong
decision, and nothing would look broken) and the two-leg futures decomposition
(an error in the first leg propagates into the second). Both are pinned here
against hand-computed values. Synthetic data throughout.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml
from boc_rate_decisions.a3_t2_horizon.build_table_t2 import origin_triples
from boc_rate_decisions.a3_t2_horizon.data_t2 import derive_t2_directions
from boc_rate_decisions.a3_t2_horizon.implied_t2 import implied_quote_t2


A3_DIR = Path(__file__).resolve().parents[3] / "implementations" / "boc_rate_decisions" / "a3_t2_horizon"
SPEC_PATH = A3_DIR / "specs" / "boc_rate_direction_t2_eval.yaml"
TABLE_PATH = A3_DIR / "data" / "market_implied_t2.csv"


def _directions() -> pd.DataFrame:
    """Four meetings: hold, cut, hold, cut."""
    dates = pd.to_datetime(["2025-01-29", "2025-03-12", "2025-04-16", "2025-06-04"])
    return pd.DataFrame({"timestamp": dates, "value": [0.0, -1.0, 0.0, -1.0], "released_at": dates})


def _corra(level: float) -> pd.Series:
    """Flat daily CORRA across the test window."""
    index = pd.date_range("2024-12-01", "2025-06-30", freq="D")
    return pd.Series([level] * len(index), index=index)


def _settlements(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """Build a settlement table from ``(date, symbol, price)`` triples."""
    return pd.DataFrame(
        [{"date": pd.Timestamp(d), "symbol": s, "settlement_price": p, "open_interest": 100.0} for d, s, p in rows]
    )


class TestDeriveT2Directions:
    """The shifted target series."""

    def test_each_row_carries_the_following_meetings_outcome(self) -> None:
        """Row timestamped at meeting N holds meeting N+1's decision."""
        shifted = derive_t2_directions(_directions())
        assert shifted.loc[0, "timestamp"] == pd.Timestamp("2025-01-29")
        assert shifted.loc[0, "value"] == pytest.approx(-1.0)  # the 2025-03-12 cut

    def test_released_at_is_the_later_meeting(self) -> None:
        """Cutoff enforcement must hide the outcome until it is announced."""
        shifted = derive_t2_directions(_directions())
        assert shifted.loc[0, "released_at"] == pd.Timestamp("2025-03-12")
        assert (pd.to_datetime(shifted["released_at"]) > pd.to_datetime(shifted["timestamp"])).all()

    def test_last_meeting_is_dropped(self) -> None:
        """The final meeting has no successor yet, so it carries no label."""
        shifted = derive_t2_directions(_directions())
        assert len(shifted) == len(_directions()) - 1
        assert pd.Timestamp("2025-06-04") not in set(shifted["timestamp"])

    def test_a_single_meeting_cannot_be_shifted(self) -> None:
        """Fail loudly rather than return an empty series."""
        with pytest.raises(ValueError, match="at least two resolved meetings"):
            derive_t2_directions(_directions().head(1))

    def test_origin_plus_horizon_lands_on_a_row_holding_the_later_outcome(self) -> None:
        """The harness contract: 28-day task, ten-week target."""
        shifted = derive_t2_directions(_directions()).set_index("timestamp")
        origin = pd.Timestamp("2025-01-29") - pd.Timedelta(days=28)
        resolved = origin + pd.Timedelta(days=28)
        assert shifted.loc[resolved, "value"] == pytest.approx(-1.0)


class TestImpliedQuoteT2:
    """The two-leg futures decomposition."""

    def test_meeting_month_leg_starts_from_the_first_legs_rate(self) -> None:
        """Days before the second decision sit at r1, not at today's CORRA."""
        # Leg 1: February holds no meeting, so r1 = 100 - 96.875 = 3.125.
        # Leg 2: April holds a meeting, so March is split at the 2025-03-13
        # effective date: 12 days at r1 and 19 at r2. A March contract implying
        # (12*3.125 + 19*3.00)/31 = 3.0483871 therefore means r2 = 3.00, i.e.
        # a further -0.125 move and an even chance of a cut.
        quote = implied_quote_t2(
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2025-01-29"),
            pd.Timestamp("2025-03-12"),
            coa=_settlements(
                [
                    ("2024-12-31", "COAF25", 96.72),
                    ("2024-12-31", "COAG25", 96.875),
                    ("2024-12-31", "COAH25", 96.9516129),
                    ("2024-12-31", "COAJ25", 97.00),
                ]
            ),
            corra=_corra(3.25),
            meeting_dates=[pd.Timestamp(d) for d in ("2025-01-29", "2025-03-12", "2025-04-16")],
        )
        assert quote.method == "meeting_month"
        assert quote.expected_rate_after_next == pytest.approx(3.125)
        assert quote.expected_rate_after_target == pytest.approx(3.00, abs=1e-4)
        assert quote.expected_move == pytest.approx(-0.125, abs=1e-4)
        assert quote.p_cut == pytest.approx(0.5, abs=1e-3)

    def test_next_month_leg_reads_the_rate_straight_off_the_contract(self) -> None:
        """A clean month after the target needs no day-count arithmetic."""
        # Leg 1 splits March (April has a meeting); leg 2 reads May, which is
        # meeting-free: r2 = 100 - 97.10 = 2.90 against r1 = 3.00.
        quote = implied_quote_t2(
            pd.Timestamp("2025-02-12"),
            pd.Timestamp("2025-03-12"),
            pd.Timestamp("2025-04-16"),
            coa=_settlements(
                [
                    ("2025-02-12", "COAH25", 97.0),
                    ("2025-02-12", "COAJ25", 97.05),
                    ("2025-02-12", "COAK25", 97.10),
                ]
            ),
            corra=_corra(3.00),
            meeting_dates=[pd.Timestamp(d) for d in ("2025-03-12", "2025-04-16", "2025-06-04")],
        )
        assert quote.method == "next_month"
        assert quote.expected_rate_after_target == pytest.approx(2.90)
        assert quote.diagnostics["leg1_method"] == "meeting_month"

    def test_two_meetings_in_one_month_are_rejected(self) -> None:
        """The decomposition assumes at most one decision per contract month."""
        with pytest.raises(ValueError, match="share a month"):
            implied_quote_t2(
                pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-03-05"),
                pd.Timestamp("2025-03-26"),
                coa=_settlements([("2024-12-31", "COAH25", 97.0)]),
                corra=_corra(3.0),
                meeting_dates=[pd.Timestamp("2025-03-05"), pd.Timestamp("2025-03-26")],
            )


class TestSpecAndTable:
    """The committed spec and table must stay in step with the calendar."""

    def test_every_origin_resolves_to_a_meeting_and_a_successor(self) -> None:
        """Origin + horizon is a scheduled meeting, and one follows it."""
        triples = origin_triples(SPEC_PATH)
        spec = yaml.safe_load(SPEC_PATH.read_text())
        assert len(triples) == len(spec["origin_dates"])
        for origin, next_meeting, target in triples:
            assert next_meeting == origin + pd.Timedelta(days=28)
            assert target > next_meeting

    def test_target_gap_is_longer_than_the_canonical_lead(self) -> None:
        """The point of A3: these forecasts are made much further ahead."""
        gaps = [(target - origin).days for origin, _, target in origin_triples(SPEC_PATH)]
        assert min(gaps) > 28  # noqa: PLR2004 - the canonical lead
        assert max(gaps) < 120  # noqa: PLR2004 - two meeting cycles, not three

    def test_committed_table_covers_every_origin_with_valid_distributions(self) -> None:
        """A spec that gains an origin without a rebuild fails here."""
        table = pd.read_csv(TABLE_PATH, parse_dates=["origin_date"])
        spec = yaml.safe_load(SPEC_PATH.read_text())
        missing = [o for o in spec["origin_dates"] if pd.Timestamp(o) not in set(table["origin_date"])]
        assert not missing, f"Origins without a t+2 quote: {missing}"
        probabilities = table[["p_cut", "p_hold", "p_hike"]]
        assert (probabilities >= 0).all().all()
        assert probabilities.sum(axis=1).sub(1.0).abs().max() < 1e-6
