"""Do the agent's stated numbers match the payload it was actually given?

The agent quotes payload values in two units without saying which: percentage
points (the unit the payload is actually in) and basis points. A quote of
"-4.88%" for a payload value of -0.049 is *correct* read as -4.88 bps and
alarming read as percent. Any check that ignores this reports mostly its own
confusion, so both readings are accepted here and the unit label is ignored.

What the check is really looking for is the residue: quotes that match under
neither reading, and quotes whose basis-point conversion is off by a power of
ten (the agent's pp -> bp conversion degrades for values below ~0.5).

It matters beyond tidiness: ``rationale_eval.py`` scores the agent's stated
reasoning against the Bank's published rationale, which silently assumes the
stated reasoning reflects the inputs. Where a number is invented, that
assumption fails.

    uv run python scripts/check_rationale_numbers.py [--arm B|C] [--spec ...]

Reads the stored recall-probe predictions and recomputes each origin's feature
row. Offline: no LLM calls, nothing written.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv


warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / ".env", override=False)

import boc_rate_decisions  # noqa: E402
from aieng.forecasting.evaluation import BacktestSpec  # noqa: E402
from boc_rate_decisions.analyst_agent import AnonymizedPromptBuilder  # noqa: E402
from boc_rate_decisions.data import build_boc_service  # noqa: E402


PKG = Path(boc_rate_decisions.__file__).parent

ARM_FILES = {
    "B": "boc_recall_probe_backtest__B_anonymized/agent_predictor_boc_analyst_anonymized_gemini-3.1-flash-lite-preview_categorical.yaml",
    "C": "boc_recall_probe_backtest__C_identified/agent_predictor_boc_analyst_basic_gemini-3.1-flash-lite-preview_categorical.yaml",
}

# Phrases the agent uses for each payload field, and how close a quoted number
# has to be to count as faithful. The tolerance absorbs rounding (the payload
# carries 4 decimals; rationales usually quote 2).
FIELD_PHRASES = {
    "inflation_gap": [r"inflation gap"],
    "yield_spread": [r"yield spread", r"2-?year (?:gov\w*\s+)?yield", r"2Y yield"],
    "unemployment_momentum": [r"unemployment momentum"],
    "rate_momentum": [r"rate momentum"],
}
TOLERANCE_PP = 0.06  # percentage-point reading: absorbs 2-decimal rounding
TOLERANCE_BP = 1.0  # basis-point reading: absorbs 1 bp of rounding
WINDOW = 90  # characters after the phrase to search for a number

# Numbers that are part of a name, not a value: "trailing 90-day change",
# "the 2-year yield", "the 2% target". Skipping these is what separates a real
# misquote from the checker reading the wrong token.
NOT_A_VALUE = re.compile(r"^\s*-?\s*(day|year|yr)\b|^\s*%\s*(target|inflation)", re.IGNORECASE)

# -1.04 / +1.11 / 1.11% / −4.88 (unicode minus) / – (en dash)
NUMBER = re.compile(r"[-+−–]?\d+(?:\.\d+)?")


def _first_number_after(text: str, start: int) -> float | None:
    """First numeric literal in the window after ``start``, normalised."""
    window = text[start : start + WINDOW]
    m = NUMBER.search(window)
    if m is None:
        return None
    # A window can hold several name-numbers before the real value
    # ("the 2-year yield spread over the 90-day window is +0.05").
    for _ in range(4):
        if not NOT_A_VALUE.match(window[m.end() :]):
            break
        m2 = NUMBER.search(window, m.end())
        if m2 is None:
            return None
        m = m2
    else:
        return None
    raw = m.group(0).replace("−", "-").replace("–", "-")
    try:
        return float(raw)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="B", choices=sorted(ARM_FILES))
    args = parser.parse_args()

    path = PKG / "data" / "recall_probe" / "predictions" / ARM_FILES[args.arm]
    preds = yaml.safe_load(path.read_text())["predictions"]

    service = build_boc_service()
    spec = BacktestSpec.model_validate(
        yaml.safe_load((PKG / "specs" / "boc_rate_direction_backtest.yaml").read_text())
    )
    # Rebuild the payload through the builder and the cutoff-scoped context —
    # the same path the agent's inputs took. Recomputing the feature row from
    # unfiltered frames instead gives a *different* CPI window (the context
    # filters on released_at, build_feature_row on timestamp) and turns a
    # faithful quote into an apparent fabrication.
    builder = AnonymizedPromptBuilder()

    rows = []
    for p in preds:
        as_of = pd.Timestamp(p["as_of"])
        payload = json.loads(builder(task=spec.task, context=service.context(as_of.to_pydatetime())))
        features = payload.get("macro_snapshot")
        if not isinstance(features, dict):
            continue
        text = p.get("metadata", {}).get("rationale") or ""
        for field, patterns in FIELD_PHRASES.items():
            for pattern in patterns:
                m = re.search(pattern, text, flags=re.IGNORECASE)
                if m is None:
                    continue
                quoted = _first_number_after(text, m.end())
                if quoted is None:
                    continue
                actual = float(features[field])
                as_pp = abs(quoted - actual) <= TOLERANCE_PP
                as_bp = abs(quoted - actual * 100.0) <= TOLERANCE_BP
                if as_pp:
                    reading = "pp"
                elif as_bp:
                    reading = "bp"
                elif abs(quoted - actual * 1000.0) <= 10.0 or abs(quoted - actual * 10000.0) <= 100.0:
                    reading = "bp off by 10x/100x"
                else:
                    reading = "no reading matches"
                rows.append(
                    {
                        "origin": as_of.date().isoformat(),
                        "field": field,
                        "quoted": quoted,
                        "actual": round(actual, 4),
                        "actual_bp": round(actual * 100, 1),
                        "reading": reading,
                        "faithful": as_pp or as_bp,
                    }
                )
                break  # one quote per field per origin is enough

    df = pd.DataFrame(rows)
    if df.empty:
        print("no quoted numbers matched the phrase patterns — widen FIELD_PHRASES")
        return

    print(f"arm {args.arm}: {len(df)} quoted numbers found across {df['origin'].nunique()} origins\n")
    per_field = df.groupby("field").agg(quotes=("faithful", "size"), faithful=("faithful", "sum"))
    per_field["faithful_pct"] = (100 * per_field["faithful"] / per_field["quotes"]).round(1)
    print(per_field.to_string())

    print("\nhow each quote reads:")
    print(df["reading"].value_counts().to_string())

    bad = df[~df["faithful"]]
    print(f"\n{len(bad)} / {len(df)} quotes match neither unit ({100 * len(bad) / len(df):.1f}%)")
    if not bad.empty:
        print("\nthe residue:")
        print(bad.sort_values("field").head(20).to_string(index=False))


if __name__ == "__main__":
    main()
