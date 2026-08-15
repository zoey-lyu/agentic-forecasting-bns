"""Public-data ALCO scenario briefing helpers for the Scotiabank prototype.

This module does not reproduce Scotiabank's internal models or recommendations.
It linearly scales public Q2 2026 parallel-rate sensitivities into illustrative
25 bp decision scenarios and probability-weights them for decision support.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pandas as pd


SCOTIABANK_SENSITIVITY_AS_OF = "2026-04-30"
SCOTIABANK_PUBLIC_SENSITIVITIES = {
    "up_100bp": {"nii_cad_millions": 197.0, "eve_cad_millions": -1_871.0},
    "down_100bp": {"nii_cad_millions": -189.0, "eve_cad_millions": 1_615.0},
}
"""Q2 2026 Report to Shareholders, table T27, total major currencies."""


@dataclass(frozen=True)
class AlcoScenario:
    """One probability-weighted policy-decision scenario."""

    decision: str
    probability: float
    policy_move_bp: int
    nii_12m_impact_cad_millions: float
    eve_impact_cad_millions: float
    management_focus: str


_FOCUS = {
    "cut": "Review deposit floors, term-deposit migration, asset repricing, and prepayment assumptions.",
    "hold": "Maintain base plan; monitor funding mix, customer beta, and incoming inflation/labour data.",
    "hike": "Review duration exposure, borrower affordability, variable-rate credit, and hedge readiness.",
}


def complete_with_configured_model(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 1200,
) -> str:
    """Return text from either the configured gateway or a direct provider.

    ``model`` remains the logical model name selected by the notebook. When
    the project's OpenAI-compatible gateway variables are present, this helper
    supplies the gateway transport route required by LiteLLM; the route does
    not imply that the underlying model is supplied by OpenAI. Without gateway
    configuration, LiteLLM resolves ``model`` normally and uses the matching
    provider's shell-injected credentials.
    """
    import litellm  # noqa: PLC0415

    api_base = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if bool(api_base) != bool(api_key):
        raise RuntimeError(
            "OPENAI_BASE_URL and OPENAI_API_KEY must either both be set or both be absent."
        )

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if api_base and api_key:
        kwargs.update(
            model=model if model.startswith("openai/") else f"openai/{model}",
            api_base=api_base,
            api_key=api_key,
        )

    response = litellm.completion(**kwargs)
    return (response.choices[0].message.content or "No interpretation returned.").strip()


def validate_probabilities(probabilities: dict[str, float]) -> None:
    """Validate a complete cut/hold/hike distribution."""
    expected = {"cut", "hold", "hike"}
    if set(probabilities) != expected:
        raise ValueError(f"probabilities must contain exactly {sorted(expected)}")
    if any(value < 0.0 or value > 1.0 for value in probabilities.values()):
        raise ValueError("probabilities must be between 0 and 1")
    if abs(sum(probabilities.values()) - 1.0) > 1e-6:
        raise ValueError("probabilities must sum to 1")


def build_alco_scenarios(
    probabilities: dict[str, float],
    *,
    move_bp: int = 25,
) -> list[AlcoScenario]:
    """Scale public +/−100 bp sensitivities into cut/hold/hike scenarios."""
    validate_probabilities(probabilities)
    if move_bp <= 0:
        raise ValueError("move_bp must be positive")
    scale = move_bp / 100.0
    impacts = {
        "cut": (
            SCOTIABANK_PUBLIC_SENSITIVITIES["down_100bp"]["nii_cad_millions"] * scale,
            SCOTIABANK_PUBLIC_SENSITIVITIES["down_100bp"]["eve_cad_millions"] * scale,
        ),
        "hold": (0.0, 0.0),
        "hike": (
            SCOTIABANK_PUBLIC_SENSITIVITIES["up_100bp"]["nii_cad_millions"] * scale,
            SCOTIABANK_PUBLIC_SENSITIVITIES["up_100bp"]["eve_cad_millions"] * scale,
        ),
    }
    move_by_decision = {"cut": -move_bp, "hold": 0, "hike": move_bp}
    return [
        AlcoScenario(
            decision=decision,
            probability=probabilities[decision],
            policy_move_bp=move_by_decision[decision],
            nii_12m_impact_cad_millions=impacts[decision][0],
            eve_impact_cad_millions=impacts[decision][1],
            management_focus=_FOCUS[decision],
        )
        for decision in ("cut", "hold", "hike")
    ]


def scenarios_to_frame(scenarios: list[AlcoScenario]) -> pd.DataFrame:
    """Return scenarios plus probability-weighted NII and EVE contributions."""
    rows = []
    for scenario in scenarios:
        rows.append(
            {
                "decision": scenario.decision,
                "probability": scenario.probability,
                "policy_move_bp": scenario.policy_move_bp,
                "nii_12m_impact_cad_millions": scenario.nii_12m_impact_cad_millions,
                "eve_impact_cad_millions": scenario.eve_impact_cad_millions,
                "weighted_nii_cad_millions": scenario.probability * scenario.nii_12m_impact_cad_millions,
                "weighted_eve_cad_millions": scenario.probability * scenario.eve_impact_cad_millions,
                "management_focus": scenario.management_focus,
            }
        )
    return pd.DataFrame(rows)


def render_alco_brief(
    scenarios: list[AlcoScenario],
    *,
    meeting_date: str,
    forecast_as_of: str,
    evidence_titles: list[str],
) -> str:
    """Render a concise human-review ALCO scenario brief in Markdown."""
    frame = scenarios_to_frame(scenarios)
    expected_nii = frame["weighted_nii_cad_millions"].sum()
    expected_eve = frame["weighted_eve_cad_millions"].sum()
    base = max(scenarios, key=lambda scenario: scenario.probability)
    rows = "\n".join(
        f"| {row.decision.title()} | {row.probability:.1%} | {row.policy_move_bp:+d} | "
        f"{row.nii_12m_impact_cad_millions:+.1f} | {row.eve_impact_cad_millions:+.1f} |"
        for row in scenarios
    )
    sources = "\n".join(f"- {title}" for title in evidence_titles) or "- No cached source titles supplied"
    return f"""# Illustrative Scotiabank ALCO Monetary-Policy Scenario Brief

**Forecast as of:** {forecast_as_of}  
**BoC meeting:** {meeting_date}  
**Highest-probability decision:** {base.decision.upper()} ({base.probability:.1%})

## Scenario distribution and disclosed-sensitivity overlay

| Decision | Probability | Policy move (bp) | 12m NII impact (CAD mm) | EVE impact (CAD mm) |
|---|---:|---:|---:|---:|
{rows}

**Probability-weighted illustrative impact:** NII {expected_nii:+.1f} CAD mm; EVE {expected_eve:+.1f} CAD mm.

## Management focus by scenario

{chr(10).join(f'- **{scenario.decision.title()}:** {scenario.management_focus}' for scenario in scenarios)}

## Cached public evidence

{sources}

## Required interpretation controls

- Public-data prototype only; not Scotiabank internal ALCO analysis or advice.
- Sensitivities come from the Q2 2026 public disclosure and are linearly
  scaled from an immediate, sustained ±100 bp parallel shock.
- The disclosure assumes a constant balance sheet and no mitigating management
  action; an actual 25 bp policy move is not equivalent to a parallel curve shock.
- Probability weighting is a decision-support summary, not an instruction to
  price products, alter hedges, or take market positions.
- ALCO, Treasury, Finance, and independent risk/model validation must review assumptions and overlays.
"""


__all__ = [
    "SCOTIABANK_PUBLIC_SENSITIVITIES",
    "SCOTIABANK_SENSITIVITY_AS_OF",
    "AlcoScenario",
    "build_alco_scenarios",
    "complete_with_configured_model",
    "render_alco_brief",
    "scenarios_to_frame",
    "validate_probabilities",
]