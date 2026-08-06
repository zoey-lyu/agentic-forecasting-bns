"""Tests for cached-news wiring on the free-form analyst path.

``build_wti_anchored_config`` could already swap live ``search_web`` for a
pre-cached briefing; the free-form ``build_wti_news_config`` could not. That
asymmetry made "anchored vs. free-form" comparisons confounded — the two
variants necessarily read different news. These tests pin the free-form path's
new ``news_source`` parameter to the same contract the anchored path already
honours:

1. the config drops context retrieval entirely (no live fallback), and
2. the prompt builder injects ``news_briefing`` into the payload,

so a future refactor can't reintroduce the confound by wiring only one half.
Tests are offline: the source points at a ``tmp_path`` holding one stub
briefing, never the committed curriculum cache.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from energy_oil_forecasting.analyst_agent.agent import (
    WtiPriceForecastPromptBuilder,
    build_wti_agent_predictor,
    build_wti_anchored_config,
    build_wti_news_config,
)
from energy_oil_forecasting.analyst_agent.anchor_lookup import AnchorSource
from energy_oil_forecasting.analyst_agent.anchored_predictor import build_wti_anchored_predictor
from energy_oil_forecasting.analyst_agent.news_cache import NewsCacheSource


STUB_ORIGIN = date(2026, 3, 2)
STUB_BRIEFING = "STUB BRIEFING body"


@pytest.fixture
def news_source(tmp_path: Path) -> NewsCacheSource:
    """Build a real ``NewsCacheSource`` over a temp dir holding one briefing."""
    (tmp_path / f"wti_news_{STUB_ORIGIN}.md").write_text(STUB_BRIEFING)
    return NewsCacheSource(context_dir=tmp_path)


# ---------------------------------------------------------------------------
# Config: cached news must remove the live search path, not sit alongside it
# ---------------------------------------------------------------------------


def test_news_config_without_source_keeps_live_search() -> None:
    """Default (no ``news_source``) is unchanged: named ``wti_analyst_news``, retrieval on."""
    config = build_wti_news_config()

    assert config.name == "wti_analyst_news"
    assert config.context_retrieval.enabled
    assert "search_web" in config.instruction


def test_news_config_with_source_disables_retrieval(news_source: NewsCacheSource) -> None:
    """A cached source removes context retrieval entirely — no live fallback."""
    config = build_wti_news_config(news_source=news_source)

    assert config.name == "wti_analyst_news_cached"
    assert config.context_retrieval is None or not config.context_retrieval.enabled


def test_cached_instruction_drops_search_guidance_and_explains_briefing(
    news_source: NewsCacheSource,
) -> None:
    """The agent is told to read ``news_briefing``, not to call ``search_web``."""
    instruction = build_wti_news_config(news_source=news_source).instruction

    assert "news_briefing" in instruction
    # The only surviving mention is the supplement stating the tool is absent.
    assert "Recommended queries" not in instruction
    assert "no `search_web` tool in this configuration" in instruction


# ---------------------------------------------------------------------------
# Prompt builder: the briefing has to actually reach the payload
# ---------------------------------------------------------------------------


def test_prompt_builder_defaults_to_no_briefing() -> None:
    """Omitting ``news_source`` leaves the builder exactly as it was before."""
    assert WtiPriceForecastPromptBuilder().news_source is None


def test_predictor_threads_news_source_into_prompt_builder(news_source: NewsCacheSource) -> None:
    """``build_wti_agent_predictor`` must pass the source down, or the payload loses the briefing."""
    predictor = build_wti_agent_predictor(
        build_wti_news_config(news_source=news_source), news_source=news_source
    )

    assert predictor.prompt_builder.news_source is news_source


def test_payload_carries_briefing(
    news_source: NewsCacheSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configured source lands in the serialised payload under ``news_briefing``."""
    import energy_oil_forecasting.analyst_agent.agent as agent_mod  # noqa: PLC0415

    monkeypatch.setattr(agent_mod, "_build_wti_payload", lambda **_: {"task": "stub"})
    builder = WtiPriceForecastPromptBuilder(news_source=news_source)

    class _Ctx:
        as_of = STUB_ORIGIN

    payload = json.loads(builder(task=object(), context=_Ctx()))

    assert payload["news_briefing"] == STUB_BRIEFING


# ---------------------------------------------------------------------------
# Symmetric-location prompt variant: framing-only, and separately cached
# ---------------------------------------------------------------------------


def test_symmetric_loc_is_off_by_default(news_source: NewsCacheSource) -> None:
    """Existing callers keep the original prompt and the original config name."""
    config = build_wti_anchored_config(news_source=news_source)

    assert config.name == "wti_analyst_anchored_cached_news"
    assert "BELOW" not in config.instruction


def test_symmetric_loc_renames_config_so_predictions_cache_separately(
    news_source: NewsCacheSource,
) -> None:
    """A shared name would overwrite the original run's artifact and destroy the comparison."""
    old = build_wti_anchored_config(news_source=news_source)
    new = build_wti_anchored_config(news_source=news_source, anchor_prompt="symloc")

    assert new.name == "wti_analyst_anchored_cached_news_symloc"
    assert new.name != old.name


def test_symmetric_loc_changes_framing_only_not_the_contract(
    news_source: NewsCacheSource,
) -> None:
    """Bounds and schema must be identical, or a difference isn't attributable to wording."""
    old = build_wti_anchored_config(news_source=news_source)
    new = build_wti_anchored_config(news_source=news_source, anchor_prompt="symloc")

    for instruction in (old.instruction, new.instruction):
        assert "[-1, +1]" in instruction  # signal_loc bounds unchanged
        assert "[0, +1]" in instruction  # signal_width bounds unchanged
    # ...but only the new one spells out the negative half as usable.
    assert "BELOW" in new.instruction
    assert "BELOW" not in old.instruction


def test_symmetric_loc_reaches_the_predictor(news_source: NewsCacheSource) -> None:
    """``build_wti_anchored_predictor`` must forward the flag, or the run silently uses the old prompt."""
    predictor = build_wti_anchored_predictor(
        AnchorSource.from_spec_id("energy_oil_eval"),
        news_source=news_source,
        anchor_prompt="symloc",
    )

    assert "symloc" in predictor.predictor_id
