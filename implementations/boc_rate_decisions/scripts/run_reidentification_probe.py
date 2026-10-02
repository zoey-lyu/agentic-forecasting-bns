"""Can the model still work out which central bank and which year arm B is?

Arm B exists to make recall unavailable by removing every identifying handle.
That claim has so far been argued, not measured. This script measures it: it
feeds the model the exact anonymised payload arm B receives and simply asks
which institution and which calendar year it is looking at.

The headline number is the Spearman correlation between the guessed year and
the true year across origins. A correlation near zero means the payload carries
no usable clock; a high correlation means it does, whatever the payload looks
like to a reader.

Run it BEFORE changing the builder, to get the baseline leak, and again after,
to see what the change bought.

    uv run python scripts/run_reidentification_probe.py [--spec ...] [--out DIR]

Pre-registered pass marks (fixed before the first run, so a later result cannot
move them):

    |rho| < 0.2        guessed year carries no ordering information
    institution < 20%  naming the Bank of Canada is no better than a guess
                       among the handful of plausible inflation-targeting banks

Neither bar can reach zero: "policy rate is at its floor" is both identifying
and genuinely predictive, so some signal is irreducible without gutting the
task.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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
from aieng.forecasting.methods.llm_processes._client import (  # noqa: E402
    make_json_schema_response_format,
    sample_n_async,
)
from aieng.forecasting.models import LITE_MODEL  # noqa: E402
from boc_rate_decisions.analyst_agent import AnonymizedPromptBuilder  # noqa: E402
from boc_rate_decisions.data import build_boc_service  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from scipy import stats  # noqa: E402


PKG = Path(boc_rate_decisions.__file__).parent

PASS_RHO = 0.2
PASS_INSTITUTION = 0.20


class _Identification(BaseModel):
    """What the model thinks it is looking at."""

    institution: str = Field(description="Name of the central bank, best guess.")
    country: str = Field(description="Country or currency area, best guess.")
    year: int = Field(description="Calendar year of the snapshot, best guess.")
    confidence: int = Field(ge=0, le=100, description="0-100. 0 = pure guess, 100 = certain.")


_SCHEMA = {
    "type": "object",
    "properties": {
        "institution": {"type": "string"},
        "country": {"type": "string"},
        "year": {"type": "integer"},
        "confidence": {"type": "integer", "description": "Confidence 0-100, where 0 is a pure guess."},
    },
    "required": ["institution", "country", "year", "confidence"],
}

# The forecasting instruction tells the agent NOT to guess the institution.
# The probe asks for exactly that, so it uses its own system prompt.
_SYSTEM = (
    "You are a macroeconomic forensics analyst. You will be shown an anonymised "
    "snapshot of one central bank's policy state: its current policy rate, the "
    "sequence of its recent decisions, and a few macro indicators. Identifying "
    "details have been stripped.\n\n"
    "Your job is to work out WHICH central bank and WHICH calendar year this is. "
    "Use any signal available — the level of the policy rate, the shape of the "
    "decision sequence, the size of the inflation gap, the unemployment move.\n\n"
    "Always commit to a single best guess for institution, country and year, "
    "even when you are unsure; say so through `confidence` rather than by "
    "declining. Never answer 'unknown'."
)


async def _identify_all(
    payloads: list[tuple[str, str]], model: str, concurrency: int
) -> list[tuple[str, _Identification | None, float]]:
    """Run the identification call for every origin, `concurrency` at a time."""
    response_format = make_json_schema_response_format("Identification", _SCHEMA)
    out: list[tuple[str, _Identification | None, float]] = []

    async def one(origin: str, payload: str) -> tuple[str, _Identification | None, float]:
        parsed, cost, _, _, _ = await sample_n_async(
            schema_cls=_Identification,
            model=model,
            base_messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": payload},
            ],
            response_format=response_format,
            n_samples=1,
            temperature=0.0,
            max_tokens=2048,
            timeout_s=120.0,
            reasoning_effort=None,
            # Without these the call bypasses the Vector proxy and LiteLLM
            # routes the bare model name straight at Vertex AI, which this
            # project has no access to.
            api_base=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        return origin, (parsed[0] if parsed else None), cost

    for i in range(0, len(payloads), concurrency):
        chunk = payloads[i : i + concurrency]
        out.extend(await asyncio.gather(*(one(o, p) for o, p in chunk)))
        print(f"  {min(i + concurrency, len(payloads))}/{len(payloads)} done", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="boc_rate_direction_backtest.yaml")
    parser.add_argument("--model", default=LITE_MODEL)
    parser.add_argument("--out", default=str(PKG / "data" / "reidentification_probe"))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N origins (for a dry run).")
    args = parser.parse_args()

    spec = BacktestSpec.model_validate(yaml.safe_load((PKG / "specs" / args.spec).read_text()))
    service = build_boc_service()
    builder = AnonymizedPromptBuilder()

    origins = spec.origins()[: args.limit] if args.limit else spec.origins()
    payloads = []
    for origin in origins:
        ts = pd.Timestamp(origin)
        payloads.append((ts.date().isoformat(), builder(task=spec.task, context=service.context(ts.to_pydatetime()))))

    print(f"model = {args.model}   origins = {len(payloads)}\n")
    results = asyncio.run(_identify_all(payloads, args.model, args.concurrency))

    rows = []
    total_cost = 0.0
    for origin, ident, cost in results:
        total_cost += cost
        if ident is None:
            continue
        rows.append(
            {
                "origin": origin,
                "true_year": int(origin[:4]),
                "guessed_year": ident.year,
                "institution": ident.institution,
                "country": ident.country,
                "confidence": ident.confidence,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        print("no parseable identifications returned")
        return

    df["year_error"] = df["guessed_year"] - df["true_year"]
    rho, p_value = stats.spearmanr(df["true_year"], df["guessed_year"])
    hit = df["country"].str.contains("canada", case=False, na=False) | df[
        "institution"
    ].str.contains("canada", case=False, na=False)
    institution_rate = float(hit.mean())

    print("\n=== the measurement ===")
    print(f"  n parsed                 {len(df)} / {len(payloads)}")
    print(f"  Spearman rho(year)       {rho:+.3f}   (p = {p_value:.2g})   bar |rho| < {PASS_RHO}")
    print(f"  named Canada             {institution_rate:.1%}                  bar < {PASS_INSTITUTION:.0%}")
    print(f"  median |year error|      {df['year_error'].abs().median():.1f} years")
    print(f"  mean confidence          {df['confidence'].mean():.0f} / 100")
    print(f"  cost                     ${total_cost:.4f}")

    passed = abs(rho) < PASS_RHO and institution_rate < PASS_INSTITUTION
    print(f"\n  VERDICT: {'PASS — no usable clock' if passed else 'FAIL — payload still identifies the period'}")

    print("\n  most-named institutions:")
    print(df["institution"].value_counts().head(5).to_string())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "per_origin.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "spec": args.spec,
                "n": len(df),
                "spearman_rho": float(rho),
                "spearman_p": float(p_value),
                "institution_hit_rate": institution_rate,
                "median_abs_year_error": float(df["year_error"].abs().median()),
                "mean_confidence": float(df["confidence"].mean()),
                "cost_usd": total_cost,
                "bars": {"rho": PASS_RHO, "institution": PASS_INSTITUTION},
                "passed": bool(passed),
            },
            indent=2,
        )
    )
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
