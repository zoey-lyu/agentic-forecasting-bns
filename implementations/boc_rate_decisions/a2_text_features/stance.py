"""Extract a hawk/dove stance from one Bank of Canada rate statement.

The premise of A2 is a division of labour. Predicting a probability is what
LLMs are *bad* at on this task — the agent loses to a four-feature logistic on
the protected window, and everything it appeared to gain earlier turned out to
be recall of the answer. Reading a statement and scoring its tone is what LLMs
are *good* at: bounded, low-variance, checkable against the quoted sentence.
So the model never sees a probability question here. It returns a small
structured record, and a logistic regression — fit at the origin on past
meetings, exactly like the macro baseline — decides what that record is worth.

What is extracted, and why only this much:

``hawk_dove``
    Integer -2..+2 for the stance toward the *path ahead*, not the decision
    just taken. The decision itself is already in the macro features
    (``rate_momentum``), so re-encoding it would add noise, not information.
``guidance``
    Five ordered levels from explicit easing to explicit tightening. A
    statement can cut while signalling it is done cutting; that gap is the
    thing text might know and the yield curve might not.
``drivers``
    A fixed vocabulary of the Bank's stated reasons. Not fed to the model as
    features — with ~130 training rows, seven dummies would spend degrees of
    freedom the sample cannot pay for. Extracted because it is nearly free and
    makes the stance scores auditable by theme.
``evidence``
    One short quote supporting the score. Its only job is to be checkable:
    a score with a quote that is not in the statement is a fabrication, and
    :mod:`extract_stance` verifies every quote against the source text.
"""

from __future__ import annotations

import os
from typing import Any

from aieng.forecasting.methods.llm_processes._client import (
    make_json_schema_response_format,
    run_async,
    sample_n_async,
)
from aieng.forecasting.models import LITE_MODEL
from pydantic import BaseModel, Field


#: Ordered guidance levels, mapped to the numeric feature the logistic sees.
GUIDANCE_SCORES = {
    "explicit_easing": -2.0,
    "conditional_easing": -1.0,
    "none": 0.0,
    "conditional_tightening": 1.0,
    "explicit_tightening": 2.0,
}

#: Fixed driver vocabulary, from the plan's list plus the two themes that
#: dominate the 2009-2026 statements (global demand, excess supply).
DRIVER_VOCABULARY = [
    "inflation_expectations",
    "labour_slack",
    "housing",
    "trade_tariffs",
    "cad",
    "global_demand",
    "excess_supply",
]


class StatementStance(BaseModel):
    """Structured stance record for one rate statement."""

    hawk_dove: int = Field(
        ...,
        ge=-2,
        le=2,
        description="Stance toward the path ahead: -2 strongly dovish to +2 strongly hawkish.",
    )
    guidance: str = Field(..., description="One of the five ordered forward-guidance levels.")
    drivers: list[str] = Field(default_factory=list, description="Drivers cited, from the fixed vocabulary.")
    evidence: str = Field(..., description="A short verbatim quote from the statement supporting the score.")


_STANCE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hawk_dove": {"type": "integer", "minimum": -2, "maximum": 2},
        "guidance": {"type": "string", "enum": list(GUIDANCE_SCORES)},
        "drivers": {"type": "array", "items": {"type": "string", "enum": DRIVER_VOCABULARY}},
        "evidence": {"type": "string"},
    },
    "required": ["hawk_dove", "guidance", "drivers", "evidence"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """You are a central-bank communications analyst. You read one Bank of \
Canada interest-rate statement and record its stance in a fixed structure.

You are NOT forecasting. Do not consider, and do not let yourself be influenced by, what \
the Bank actually did at any later meeting, whether you happen to know it or not. Score \
only what this text says.

hawk_dove — the stance toward the PATH AHEAD, not the decision just announced:
  -2  clearly signals further easing, or urgency about weakness
  -1  leans toward easing, or open to it if conditions persist
   0  balanced; explicitly data-dependent with no lean
  +1  leans toward holding or tightening; concerned about inflation
  +2  clearly signals tightening, or urgency about inflation

A statement that cuts while saying it is close to done is NOT dovish; that is the \
distinction this field exists to capture.

guidance — how explicit the statement is about its next move:
  explicit_easing, conditional_easing, none, conditional_tightening, explicit_tightening

drivers — which of the listed reasons the statement actually cites. Omit rather than guess.

evidence — one quote of at most 200 characters, copied VERBATIM from the statement, that \
supports your hawk_dove score. It must appear in the text word for word."""


def _user_prompt(statement_text: str, *, max_chars: int) -> str:
    """Build the extraction prompt for one statement."""
    body = statement_text if len(statement_text) <= max_chars else statement_text[:max_chars]
    return f"Bank of Canada rate statement:\n\n---\n{body}\n---\n\nRecord the stance."


def extract_stance(
    statement_text: str,
    *,
    model: str = LITE_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout_s: float = 120.0,
    max_chars: int = 12000,
    reasoning_effort: str | None = None,
) -> tuple[StatementStance, float]:
    """Run one extraction call, returning ``(stance, cost_usd)``.

    Temperature is 0 by default: this is a measurement, and two runs of the
    same statement should give the same record.

    Raises
    ------
    RuntimeError
        If the model returns no schema-valid record.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _user_prompt(statement_text, max_chars=max_chars)},
    ]
    parsed, cost, _in, _out, _fails = run_async(
        sample_n_async(
            schema_cls=StatementStance,
            model=model,
            base_messages=messages,
            response_format=make_json_schema_response_format("StatementStance", _STANCE_JSON_SCHEMA),
            n_samples=1,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            reasoning_effort=reasoning_effort,
            api_base=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
    )
    if not parsed:
        raise RuntimeError("Stance extraction returned no schema-valid record.")
    return parsed[0], float(cost)


def quote_is_verbatim(evidence: str, statement_text: str) -> bool:
    """Check the evidence quote really appears in the statement.

    Whitespace and the Bank's curly quotation marks are normalised before
    comparison; nothing else is. A False here means the record is fabricated
    and the row should not be trusted.
    """
    if not evidence.strip():
        return False

    def normalise(text: str) -> str:
        replacements = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"}
        for old, new in replacements.items():
            text = text.replace(old, new)
        return " ".join(text.split()).lower()

    return normalise(evidence) in normalise(statement_text)


__all__ = [
    "DRIVER_VOCABULARY",
    "GUIDANCE_SCORES",
    "StatementStance",
    "extract_stance",
    "quote_is_verbatim",
]
