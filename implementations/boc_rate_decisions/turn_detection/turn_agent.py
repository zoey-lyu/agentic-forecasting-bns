"""The turn-detection agent — how active the Bank is, and whether it moves.

Everything before this established what the agent is for. The market's one
durable advantage is that it re-estimates the Bank's current activity level
continuously, while anything fitted to history carries the old regime with it —
three separate corrections failed on exactly that. An agent has no fitted
parameters to carry, so it is the right shape of tool for a base-rate problem.

The output is deliberately three things, not one:

``regime``
    What kind of period the Bank is in. This is the quantity the statistical
    arms cannot track and the one the whole investigation points at. Asking for
    it explicitly also makes the answer inspectable: a wrong probability with a
    sensible regime call is a different failure from a wrong regime.
``p_move``
    The probability the Bank changes the rate at the meeting in question —
    cut or hike, either counts. This is what gets scored.
``evidence``
    Verbatim quotes from the statement it was given. Mechanically checkable, as
    in A2: a quote that is not in the source means the assessment is invented.

What the prompt deliberately does **not** say: that missing a move costs about
five times what a false alarm does. That asymmetry is real, but it belongs to a
downstream decision that uses the probability, not to the probability itself.
Brier is a proper scoring rule — the best an honest forecaster can do is report
its actual belief, and calibration is the one thing the market reliably beats
us on. Telling the model to lean would throw that away.

Two controls are built in rather than bolted on, because both change the input
rather than the code:

``scramble_date``
    The origin is presented as a different, plausible date. If the assessment
    barely moves, the model is reading the statement; if it moves a lot, it may
    be recalling what happened around the real date.
``omit_statement``
    The statement is withheld. If the assessment does not change, the text is
    not being used and nothing here is about reading.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from aieng.forecasting.methods.llm_processes._client import (
    make_json_schema_response_format,
    run_async,
    sample_n_async,
)
from aieng.forecasting.models import LITE_MODEL
from pydantic import BaseModel, Field


#: The regime labels the agent must choose between.
REGIMES = (
    "active_easing",
    "active_tightening",
    "on_hold",
    "turning_point",
)


class TurnAssessment(BaseModel):
    """One meeting's assessment."""

    regime: str = Field(..., description="Which of the four regime labels the Bank is currently in.")
    regime_reason: str = Field(..., description="One sentence for the regime call.")
    p_move: float = Field(..., ge=0.0, le=1.0, description="Probability of a cut or hike at this meeting.")
    evidence: list[str] = Field(
        default_factory=list, description="Verbatim quotes from the statement supporting the assessment."
    )
    rationale: str = Field(..., description="Two or three sentences for the probability.")


_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "regime": {"type": "string", "enum": list(REGIMES)},
        "regime_reason": {"type": "string"},
        "p_move": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["regime", "regime_reason", "p_move", "evidence", "rationale"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """You assess Bank of Canada policy meetings. You answer two \
questions, in this order.

FIRST: what kind of period is the Bank in right now?
  active_easing      — cutting, and likely to keep cutting
  active_tightening  — raising, and likely to keep raising
  on_hold            — deliberately sitting still; moves are unlikely
  turning_point      — the stance is changing, or about to

This matters more than it looks. How often the Bank moves changes enormously \
between periods — in some stretches it moves at most meetings, in others it sits \
still for a year. Judge which one you are in from what the Bank says and from \
the data you are given, not from a long-run average.

SECOND: the probability the Bank changes its target rate at the specific meeting \
named. A cut and a hike both count as a move; you are not being asked which way.

Rules:
- Report the probability you actually believe. Do not round to comfortable \
numbers, do not hedge toward 50%, and do not shade it to be safe. You are scored \
by a rule that rewards honesty and punishes both overconfidence and vagueness.
- Base it on the information given plus general economic reasoning. You must NOT \
use any knowledge of what the Bank actually decided at or after the meeting in \
question, whether or not you happen to remember it.
- Quote the statement verbatim in `evidence`, word for word, at most 200 \
characters per quote. If you were given no statement, leave it empty."""


def _context_block(
    *,
    meeting_date: pd.Timestamp,
    origin: pd.Timestamp,
    current_rate: float,
    recent_decisions: list[tuple[pd.Timestamp, str]],
    trailing_move_rate: float,
    statement_date: pd.Timestamp | None,
    statement_text: str | None,
    max_chars: int,
    intervening_meeting: pd.Timestamp | None = None,
) -> str:
    """Assemble everything the agent sees for one meeting.

    ``intervening_meeting`` switches the question to the t+2 horizon: the named
    announcement is still the one to assess, but an earlier one now sits between
    it and today, and the agent is told so explicitly. Leaving it ``None``
    reproduces the frozen t+1 prompt byte for byte.
    """
    history = "\n".join(f"  {date:%Y-%m-%d}  {decision}" for date, decision in recent_decisions)
    next_line = (
        f"The next scheduled announcement is {meeting_date:%Y-%m-%d}."
        if intervening_meeting is None
        else (
            f"The next scheduled announcement is {intervening_meeting:%Y-%m-%d}, and the one after "
            f"that is {meeting_date:%Y-%m-%d}. You are assessing the LATER one, "
            f"{meeting_date:%Y-%m-%d}. The {intervening_meeting:%Y-%m-%d} decision has not been "
            "made yet either, so account for both."
        )
    )
    blocks = [
        f"Today is {origin:%Y-%m-%d}.",
        next_line,
        f"The target for the overnight rate is currently {current_rate:.2f}%.",
        f"\nThe last {len(recent_decisions)} announcements:\n{history}",
        f"\nOf the last 8 announcements, {trailing_move_rate:.0%} produced a change.",
    ]
    if statement_text is not None and statement_date is not None:
        body = statement_text if len(statement_text) <= max_chars else statement_text[:max_chars]
        blocks.append(f"\nThe Bank's most recent statement ({statement_date:%Y-%m-%d}):\n---\n{body}\n---")
    else:
        blocks.append("\nNo statement text is available for this assessment.")
    blocks.append(f"\nAssess the {meeting_date:%Y-%m-%d} announcement.")
    return "\n".join(blocks)


def assess_turn(
    *,
    meeting_date: pd.Timestamp,
    origin: pd.Timestamp,
    current_rate: float,
    recent_decisions: list[tuple[pd.Timestamp, str]],
    trailing_move_rate: float,
    statement_date: pd.Timestamp | None = None,
    statement_text: str | None = None,
    model: str = LITE_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout_s: float = 120.0,
    max_chars: int = 12000,
    reasoning_effort: str | None = None,
    intervening_meeting: pd.Timestamp | None = None,
    n_samples: int = 1,
) -> tuple[TurnAssessment, float]:
    """Run one assessment, returning ``(assessment, cost_usd)``.

    With ``n_samples > 1`` the model is sampled that many times and the
    probabilities are averaged; the returned regime and text come from the
    first sample, so the record stays readable while the number is the mean.
    Averaging cuts sampling variance, which is the only defensible way to touch
    this agent's calibration — a correction fitted to the observed bias would be
    fitting to the evaluation window (see ``PREREGISTRATION.md``).

    Raises
    ------
    RuntimeError
        If the model returns no schema-valid assessment.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _context_block(
                meeting_date=meeting_date,
                origin=origin,
                current_rate=current_rate,
                recent_decisions=recent_decisions,
                trailing_move_rate=trailing_move_rate,
                statement_date=statement_date,
                statement_text=statement_text,
                max_chars=max_chars,
                intervening_meeting=intervening_meeting,
            ),
        },
    ]
    parsed, cost, _in, _out, _fails = run_async(
        sample_n_async(
            schema_cls=TurnAssessment,
            model=model,
            base_messages=messages,
            response_format=make_json_schema_response_format("TurnAssessment", _JSON_SCHEMA),
            n_samples=n_samples,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            reasoning_effort=reasoning_effort,
            api_base=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
    )
    if not parsed:
        raise RuntimeError("Turn assessment returned no schema-valid result.")
    if len(parsed) == 1:
        return parsed[0], float(cost)

    averaged = parsed[0].model_copy(update={"p_move": float(sum(sample.p_move for sample in parsed) / len(parsed))})
    return averaged, float(cost)


__all__ = ["REGIMES", "TurnAssessment", "assess_turn"]
