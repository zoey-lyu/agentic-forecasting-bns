"""Tests for the public-data Scotiabank ALCO scenario overlay."""

import sys
from types import SimpleNamespace

import pytest
from boc_rate_decisions.scotiabank_alco import (
    build_alco_scenarios,
    complete_with_configured_model,
    scenarios_to_frame,
)


def test_public_sensitivities_scale_to_quarter_point_scenarios() -> None:
    scenarios = build_alco_scenarios({"cut": 0.3, "hold": 0.6, "hike": 0.1})
    frame = scenarios_to_frame(scenarios).set_index("decision")
    assert frame.loc["cut", "nii_12m_impact_cad_millions"] == pytest.approx(-47.25)
    assert frame.loc["cut", "eve_impact_cad_millions"] == pytest.approx(403.75)
    assert frame.loc["hike", "nii_12m_impact_cad_millions"] == pytest.approx(49.25)
    assert frame.loc["hike", "eve_impact_cad_millions"] == pytest.approx(-467.75)


def test_probabilities_must_form_a_distribution() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        build_alco_scenarios({"cut": 0.5, "hold": 0.5, "hike": 0.5})


@pytest.mark.parametrize(
    ("gateway", "expected_model"),
    [(True, "openai/claude-opus-5"), (False, "claude-opus-5")],
)
def test_completion_routes_logical_model_through_configured_gateway(
    monkeypatch: pytest.MonkeyPatch,
    gateway: bool,
    expected_model: str,
) -> None:
    captured: dict[str, object] = {}

    def fake_completion(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        message = SimpleNamespace(content="interpreted")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=fake_completion))
    if gateway:
        monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    else:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = complete_with_configured_model(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "Explain this."}],
    )

    assert result == "interpreted"
    assert captured["model"] == expected_model
    assert ("api_base" in captured) is gateway
    assert ("api_key" in captured) is gateway