"""Tests for the de-identified (arm B) payload and the news agent's directive.

Both of the things guarded here were measured, not assumed, and both would
regress silently:

- the v2 payload dropped the meeting counter and banded the policy rate after a
  re-identification probe found the model tracking the calendar year at a
  Spearman rho of +0.55; putting a raw level or a running count back would
  restore the leak with nothing failing;
- ``build_boc_news_config`` once attached ``search_web`` while reusing the
  tool-free instruction verbatim, so the agent never searched. Twenty-five runs
  produced zero retrieved facts before anyone noticed, because an unused tool
  looks exactly like a tool with nothing to find.
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import pytest
from aieng.forecasting.data.service import DataService
from aieng.forecasting.evaluation.task import ForecastingTask, TaskCategory
from boc_rate_decisions.analyst_agent import AnonymizedPromptBuilder
from boc_rate_decisions.analyst_agent.agent import build_boc_basic_config, build_boc_news_config
from boc_rate_decisions.analyst_agent.anonymized import _policy_rate_band
from boc_rate_decisions.data import (
    BOND_YIELD_2YR_SERIES_ID,
    CPI_SERIES_ID,
    TARGET_RATE_SERIES_ID,
    UNEMPLOYMENT_SERIES_ID,
)


AS_OF = datetime(2024, 1, 1)


def _daily(value: float, released_lag_days: int = 1) -> pd.DataFrame:
    idx = pd.date_range("2018-01-01", "2023-12-31", freq="D")
    return pd.DataFrame(
        {"timestamp": idx, "value": value, "released_at": idx + pd.Timedelta(days=released_lag_days)}
    )


def _monthly(value: float) -> pd.DataFrame:
    idx = pd.date_range("2018-01-01", "2023-12-01", freq="MS")
    return pd.DataFrame({"timestamp": idx, "value": value, "released_at": idx + pd.Timedelta(days=30)})


@pytest.fixture
def task() -> ForecastingTask:
    return ForecastingTask(
        task_id="boc_rate_direction_next_meeting",
        target_series_id="boc_rate_decision_direction",
        horizons=[28],
        frequency="D",
        payload_type="categorical",
        categories=[
            TaskCategory(label="cut", value=-1),
            TaskCategory(label="hold", value=0),
            TaskCategory(label="hike", value=1),
        ],
        description="test task",
    )


@pytest.fixture
def payload(task: ForecastingTask) -> dict:
    """The anonymised payload at one origin, parsed."""
    service = DataService()
    service._store.put(TARGET_RATE_SERIES_ID, _daily(5.0), None)
    service._store.put(BOND_YIELD_2YR_SERIES_ID, _daily(4.5), None)
    service._store.put(CPI_SERIES_ID, _monthly(120.0), None)
    service._store.put(UNEMPLOYMENT_SERIES_ID, _monthly(6.0), None)

    meetings = pd.date_range("2018-01-15", "2023-12-15", freq="45D")
    directions = pd.DataFrame(
        {"timestamp": meetings, "value": [0.0] * (len(meetings) - 2) + [-1.0, -1.0], "released_at": meetings}
    )
    service._store.put("boc_rate_decision_direction", directions, None)

    return json.loads(AnonymizedPromptBuilder()(task=task, context=service.context(AS_OF)))


class TestNoClock:
    """The payload must not carry anything that counts up with the origin."""

    def test_meeting_counter_is_absent(self, payload: dict) -> None:
        outcomes = payload["meeting_outcomes"]
        assert "n_meetings_observed" not in outcomes
        assert "counts" not in outcomes

    def test_no_explicit_list_lengths(self, payload: dict) -> None:
        # meetings_shown / window_meetings ramped 8 -> 40 across the early
        # origins, which is the counter again in a smaller font.
        assert "meetings_shown" not in payload["policy_rate"]
        assert "window_meetings" not in payload["meeting_outcomes"]

    def test_base_rates_are_rounded_to_two_decimals(self, payload: dict) -> None:
        # Four decimals divide back into the raw counts, handing the counter
        # back through the front door.
        for value in payload["meeting_outcomes"]["base_rates_over_window"].values():
            assert round(value, 2) == value


class TestNoIdentifyingLevel:
    """The policy rate is reported as a band, never as a level."""

    def test_payload_carries_a_band_not_a_level(self, payload: dict) -> None:
        policy_rate = payload["policy_rate"]
        assert "current_policy_rate_pct" not in policy_rate
        assert isinstance(policy_rate["level_relative_to_floor"], str)

    def test_serialised_payload_never_shows_the_rate(self, payload: dict) -> None:
        assert "5.0" not in json.dumps(payload["policy_rate"])

    @pytest.mark.parametrize(
        ("rate", "expected"),
        [
            (0.25, "at its effective floor"),
            (0.50, "at its effective floor"),
            (0.75, "slightly above its floor"),
            (1.50, "slightly above its floor"),
            (2.50, "moderately above its floor"),
            (3.25, "well above its floor"),
            (5.00, "well above its floor"),
        ],
    )
    def test_bands(self, rate: float, expected: str) -> None:
        assert _policy_rate_band(rate) == expected

    def test_near_floor_episodes_share_a_band(self) -> None:
        # 0.25 (2009-10 and 2020-21) and 0.50 (2015-17) land in one band on
        # purpose. Separating them is the kind of resolution that let the
        # re-identification probe date the payload, and the distinction buys
        # nothing: neither leaves room to cut.
        assert _policy_rate_band(0.25) == _policy_rate_band(0.50)

    def test_the_2022_tightening_peak_is_not_its_own_band(self) -> None:
        # 5.00 is unique to one period in this window; it must read the same
        # as any other clearly-restrictive level.
        assert _policy_rate_band(5.00) == _policy_rate_band(3.25)


class TestPredictiveContentSurvives:
    """De-identification must not quietly gut the payload."""

    def test_decision_history_is_kept_in_full(self, payload: dict) -> None:
        decisions = payload["policy_rate"]["recent_decisions_most_recent_last"]
        assert len(decisions) == AnonymizedPromptBuilder().max_history
        assert decisions[-1] == "cut"

    def test_macro_snapshot_keeps_raw_numbers(self, payload: dict) -> None:
        snapshot = payload["macro_snapshot"]
        assert set(snapshot) == {"yield_spread", "rate_momentum", "inflation_gap", "unemployment_momentum"}
        assert all(isinstance(v, float) for v in snapshot.values())


class TestNewsAgentIsToldToSearch:
    """A tool nothing asks for is a tool that goes unused."""

    def test_instruction_differs_from_the_tool_free_agent(self) -> None:
        assert build_boc_news_config().instruction != build_boc_basic_config().instruction

    def test_instruction_directs_the_agent_to_search(self) -> None:
        instruction = build_boc_news_config().instruction
        assert "search_web" in instruction
        assert "cutoff_date" in instruction

    def test_search_is_actually_enabled(self) -> None:
        assert build_boc_news_config().context_retrieval.enabled
        assert build_boc_news_config().context_retrieval.enforce_cutoff

    def test_output_schema_stays_last(self) -> None:
        # The research block is spliced in ahead of the schema so the schema is
        # still the last thing the model reads.
        instruction = build_boc_news_config().instruction
        assert instruction.index("Research step") < instruction.index("## Output schema")
