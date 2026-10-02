"""Unit tests for the A2 statement-stance text features.

Three things need pinning: the verbatim-quote check that decides whether an
extraction is a measurement or a fabrication, the cutoff discipline in
:class:`StanceTable` (a stance must never be visible before the Bank published
it), and the predictor's feature wiring — including that the shuffled control
really is the same model on scrambled text. Synthetic data throughout; no LLM
call, no network.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from aieng.forecasting.evaluation.task import ForecastingTask, TaskCategory
from boc_rate_decisions.a2_text_features.features import TEXT_FEATURE_NAMES, StanceTable
from boc_rate_decisions.a2_text_features.stance import DRIVER_VOCABULARY, GUIDANCE_SCORES, quote_is_verbatim
from boc_rate_decisions.a2_text_features.text_logistic import BoCTextLogisticPredictor
from boc_rate_decisions.predictors.logistic_baseline import FEATURE_NAMES


STANCE_CSV = (
    Path(__file__).resolve().parents[3]
    / "implementations"
    / "boc_rate_decisions"
    / "a2_text_features"
    / "data"
    / "statement_stance.csv"
)

_CATEGORIES = [
    TaskCategory(label="cut", value=-1.0),
    TaskCategory(label="hold", value=0.0),
    TaskCategory(label="hike", value=1.0),
]


def _stance_frame() -> pd.DataFrame:
    """Three statements, hawkish then dovish then neutral."""
    return pd.DataFrame(
        {
            "statement_date": ["2024-01-24", "2024-03-06", "2024-06-05"],
            "hawk_dove": [2, -1, 0],
            "guidance_score": [2.0, -1.0, 0.0],
        }
    )


class TestQuoteCheck:
    """The fabrication guard on every extracted record."""

    def test_exact_quote_passes(self) -> None:
        """A quote copied from the text is accepted."""
        assert quote_is_verbatim("Governing Council judged", "... Governing Council judged that ...")

    def test_curly_quotes_and_spacing_are_normalised(self) -> None:
        """Typographic differences must not fail an otherwise real quote."""
        assert quote_is_verbatim("the Bank's  view", "In the Bank’s view, inflation ...")

    def test_invented_quote_fails(self) -> None:
        """A sentence that is not in the statement is a fabrication."""
        assert not quote_is_verbatim("the Bank will cut in July", "The Bank is proceeding carefully.")

    def test_empty_quote_fails(self) -> None:
        """An empty quote supports nothing."""
        assert not quote_is_verbatim("   ", "Any statement text.")


class TestGuidanceScale:
    """The guidance vocabulary must stay ordered and closed."""

    def test_scores_are_monotone_from_easing_to_tightening(self) -> None:
        """Ordering carries the meaning; it is what the feature encodes."""
        assert list(GUIDANCE_SCORES.values()) == sorted(GUIDANCE_SCORES.values())
        assert GUIDANCE_SCORES["explicit_easing"] < GUIDANCE_SCORES["none"] < GUIDANCE_SCORES["explicit_tightening"]

    def test_driver_vocabulary_has_no_duplicates(self) -> None:
        """A duplicated driver would double-count in any later audit."""
        assert len(DRIVER_VOCABULARY) == len(set(DRIVER_VOCABULARY))


class TestStanceTable:
    """Cutoff discipline and the delta feature."""

    def test_only_statements_published_by_the_origin_are_visible(self) -> None:
        """The June statement must be invisible at an April origin."""
        table = StanceTable(_stance_frame())
        row = table.feature_row(pd.Timestamp("2024-04-01"))
        assert row is not None
        assert row["stance_hawk_dove"] == pytest.approx(-1.0)  # March, not June
        assert row["stance_delta"] == pytest.approx(-3.0)  # -1 - 2

    def test_statement_published_on_the_origin_counts_as_visible(self) -> None:
        """Statements land at 09:45 ET, before any forecast issued that day."""
        table = StanceTable(_stance_frame())
        row = table.feature_row(pd.Timestamp("2024-06-05"))
        assert row is not None
        assert row["stance_hawk_dove"] == pytest.approx(0.0)

    def test_none_when_a_delta_cannot_be_formed(self) -> None:
        """One statement of history is not enough for a change feature."""
        assert StanceTable(_stance_frame()).feature_row(pd.Timestamp("2024-02-01")) is None

    def test_feature_row_keys_match_the_declared_names(self) -> None:
        """The predictor indexes by these names; drift would be silent."""
        row = StanceTable(_stance_frame()).feature_row(pd.Timestamp("2024-06-05"))
        assert row is not None
        assert sorted(row) == sorted(TEXT_FEATURE_NAMES)

    def test_shuffle_permutes_values_but_preserves_the_multiset(self) -> None:
        """The control must change the pairing, not the distribution."""
        frame = pd.DataFrame(
            {
                "statement_date": pd.date_range("2020-01-01", periods=12, freq="60D").strftime("%Y-%m-%d"),
                "hawk_dove": [2, -1, 0, 1, -2, 0, 1, 1, -1, 0, 2, -1],
                "guidance_score": [2.0, -1.0, 0.0, 1.0, -2.0, 0.0, 1.0, 1.0, -1.0, 0.0, 2.0, -1.0],
            }
        )
        plain = StanceTable(frame).as_frame()
        shuffled = StanceTable(frame, shuffle_seed=0).as_frame()
        assert sorted(plain["hawk_dove"]) == sorted(shuffled["hawk_dove"])
        assert list(plain["hawk_dove"]) != list(shuffled["hawk_dove"])
        # Dates are untouched: only the readings move.
        assert list(plain["statement_date"]) == list(shuffled["statement_date"])

    def test_missing_columns_are_rejected(self) -> None:
        """A malformed table fails at construction, not at fit time."""
        with pytest.raises(ValueError, match="missing columns"):
            StanceTable(pd.DataFrame({"statement_date": ["2024-01-24"]}))


class TestTextLogisticWiring:
    """Arm configuration, before any data is involved."""

    def test_text_arm_fits_seven_features(self) -> None:
        """Four macro plus three stance columns, in that order."""
        predictor = BoCTextLogisticPredictor(StanceTable(_stance_frame()))
        assert predictor.feature_names == [*FEATURE_NAMES, *TEXT_FEATURE_NAMES]

    def test_ablation_arm_fits_only_macro_features(self) -> None:
        """The ablation must be the old baseline's feature set exactly."""
        assert BoCTextLogisticPredictor(use_text=False).feature_names == list(FEATURE_NAMES)

    def test_arms_have_distinct_ids(self) -> None:
        """One leaderboard holds all three arms; ids must not collide."""
        ids = {
            BoCTextLogisticPredictor(use_text=False).predictor_id,
            BoCTextLogisticPredictor(StanceTable(_stance_frame())).predictor_id,
            BoCTextLogisticPredictor(StanceTable(_stance_frame()), predictor_suffix="shuffled").predictor_id,
        }
        assert len(ids) == 3  # noqa: PLR2004 - three arms

    def test_text_arm_requires_a_stance_table(self) -> None:
        """Asking for text features without a source is a configuration error."""
        with pytest.raises(ValueError, match="requires a StanceTable"):
            BoCTextLogisticPredictor(use_text=True)

    def test_binary_task_is_rejected(self) -> None:
        """Only the ordered direction task is served by this predictor."""
        task = ForecastingTask(
            task_id="boc_rate_cut_event",
            target_series_id="boc_rate_cut_event",
            horizons=[28],
            frequency="D",
            description="binary",
            payload_type="binary",
        )
        with pytest.raises(ValueError, match="requires a categorical task"):
            BoCTextLogisticPredictor(use_text=False).predict(task, _context())


def _context() -> object:
    """Return a context stand-in; the payload check fires before any series is read."""

    class _Stub:
        as_of = datetime(2025, 1, 1)

        def get_series(self, series_id: str) -> pd.DataFrame:
            raise AssertionError(f"No series should be read in this test ({series_id}).")

    return _Stub()


class TestCommittedStanceTable:
    """The committed extraction output must stay usable and honest."""

    def test_every_quote_is_verbatim(self) -> None:
        """A single fabricated quote invalidates the row it supports."""
        table = pd.read_csv(STANCE_CSV)
        bad = table.loc[~table["evidence_verbatim"].astype(bool), "statement_date"].tolist()
        assert not bad, f"Statements whose evidence quote is not verbatim: {bad}"

    def test_scores_are_inside_the_declared_range(self) -> None:
        """The schema bounds must hold in the committed data too."""
        table = pd.read_csv(STANCE_CSV)
        assert table["hawk_dove"].between(-2, 2).all()
        assert set(table["guidance"]).issubset(GUIDANCE_SCORES)

    def test_history_reaches_back_before_the_eval_window(self) -> None:
        """The fit-at-origin protocol trains on pre-window meetings."""
        table = pd.read_csv(STANCE_CSV)
        assert pd.Timestamp(table["statement_date"].min()) < pd.Timestamp("2015-01-01")
