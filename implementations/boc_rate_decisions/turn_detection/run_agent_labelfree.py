"""Stage 1: iterate the agent's prompt without ever looking at a score.

Fourteen scoreable meetings is not enough to tune a prompt against. Adjusting
wording until the score improves would repeat, by hand, the overfitting that
three fitted corrections already failed at — and unlike least squares it leaves
no trace. So prompt development happens here, against criteria that do not need
to know what the Bank actually did:

**Degeneracy.** Does the agent produce a range of probabilities, or the same
number every time? A forecaster with no resolution cannot help however well it
is calibrated.

**Evidence.** Does every quote appear verbatim in the statement it was given?
A fabricated quote means the assessment is not grounded in the text, whatever
its number says.

**Does the text matter?** Re-run with the statement withheld. If the
probabilities barely change, the agent is working from the rate history and its
priors, and nothing here is about reading statements.

**Is it recalling?** Re-run with the dates stripped from the statement and a
false origin date supplied. If the assessment survives that, it is reading
content rather than remembering the period — which is the one contamination
question this project cannot otherwise answer for an LLM arm.

Which window to run on takes some care. **2023-2024** is inside the model's
training data, so it cannot be scored and is the natural place to develop
against behaviour — but the first run there showed why it cannot answer the
"does the text matter" question either: withholding the statement changed
nothing, because the model recited the withheld statement's content from
memory. Recall substitutes for reading, and the check goes blind.

So that question is asked on **2025-2026** instead, where memory cannot stand
in. Running there is safe for the eventual scoring because nothing here reads
an outcome: "did the answer change when the input changed" carries no
information about what the Bank actually did. No score is computed until the
prompt is frozen.

Usage
-----
``python -m boc_rate_decisions.turn_detection.run_agent_labelfree``
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..a1_market_implied.build_table import origins_from_spec
from ..a2_text_features.extract_stance_table import DEFAULT_CACHE_DIR, statement_paths
from ..a2_text_features.stance import quote_is_verbatim
from ..a3_t2_horizon.build_table_t2 import DEFAULT_SPEC as SPEC_T2_2025_2026
from ..a3_t2_horizon.build_table_t2 import origin_triples
from ..data import DIRECTION_SERIES_ID, TARGET_RATE_SERIES_ID, build_boc_service
from .run_steps import SPEC_2023_2024, SPEC_2025_2026, USE_CASE_DIR
from .turn_agent import assess_turn


DATA_DIR = Path(__file__).resolve().parent / "data"

#: Offset applied to every date in the recall control. Large enough to land in a
#: different policy era, small enough to stay plausible.
_FALSE_DATE_OFFSET = pd.DateOffset(years=6)

_LABELS = {-1.0: "cut", 0.0: "hold", 1.0: "hike"}


@dataclass(frozen=True)
class _Arm:
    """One label-free variation of the agent's input.

    ``shift`` moves every date six years forward; paired with a date-stripped
    statement it is the recall control.
    """

    name: str
    statement_text: str | None
    statement_date: pd.Timestamp | None
    shift: bool


def strip_dates(text: str) -> str:
    """Remove years and month names so a statement cannot be placed in time."""
    without_years = re.sub(r"\b(19|20)\d{2}\b", "[year]", text)
    months = "January|February|March|April|May|June|July|August|September|October|November|December"
    return re.sub(rf"\b({months})\b", "[month]", without_years)


def quotes_are_verbatim(evidence: list[str], source: str | None) -> tuple[int, int]:
    """Return ``(verbatim, total)`` for one assessment's quotes.

    Reuses A2's check so both the stance extraction and the agent are held to
    one definition of "this quote is really in the statement".
    """
    if source is None:
        return 0, 0
    return sum(1 for quote in evidence if quote_is_verbatim(quote, source)), len(evidence)


def _parse_args() -> argparse.Namespace:
    """Build the CLI for the label-free runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statcan-cache-dir", type=Path, default=USE_CASE_DIR / "data" / "statcan")
    parser.add_argument("--fred-cache-dir", type=Path, default=USE_CASE_DIR / "data" / "fred")
    parser.add_argument("--limit", type=int, default=None, help="Assess only the first N meetings.")
    parser.add_argument(
        "--window",
        choices=("2023-2024", "2025-2026"),
        default="2025-2026",
        help="Which window to run the arms on (no scores are computed for either).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="Samples to average per assessment (pre-registered multi-sample test uses 5).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (the pre-registered multi-sample test uses 0.7).",
    )
    parser.add_argument(
        "--horizon",
        choices=("t+1", "t+2"),
        default="t+1",
        help="Assess the next announcement, or the one after it.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the three arms on the chosen window and report behaviour only."""
    args = _parse_args()

    service = build_boc_service(statcan_cache_dir=args.statcan_cache_dir, fred_cache_dir=args.fred_cache_dir)
    paths = statement_paths(DEFAULT_CACHE_DIR)
    if args.horizon == "t+2":
        if args.window != "2025-2026":
            raise SystemExit("The t+2 horizon is only set up for the 2025-2026 window.")
        triples = origin_triples(SPEC_T2_2025_2026)
    else:
        spec_path = SPEC_2023_2024 if args.window == "2023-2024" else SPEC_2025_2026
        triples = [(origin, None, meeting) for origin, meeting in origins_from_spec(spec_path)]
    if args.limit:
        triples = triples[: args.limit]

    rows: list[dict[str, object]] = []
    total_cost = 0.0
    for origin, intervening, meeting in triples:
        directions = service.get_series(DIRECTION_SERIES_ID, as_of=origin.to_pydatetime())
        rates = service.get_series(TARGET_RATE_SERIES_ID, as_of=origin.to_pydatetime())
        history = [
            (pd.Timestamp(t), _LABELS[float(v)])
            for t, v in zip(directions["timestamp"].tail(5), directions["value"].tail(5), strict=True)
        ]
        trailing = float((directions["value"].astype(float).tail(8) != 0.0).mean())
        current_rate = float(rates["value"].iloc[-1])

        visible = [date for date in paths if date <= origin]
        statement_date = max(visible) if visible else None
        statement = paths[statement_date].read_text() if statement_date else None

        arms = [
            _Arm("baseline", statement, statement_date, shift=False),
            _Arm("no_statement", None, None, shift=False),
            _Arm(
                "false_date",
                strip_dates(statement) if statement else None,
                statement_date + _FALSE_DATE_OFFSET if statement_date else None,
                shift=True,
            ),
        ]
        for arm in arms:
            shift = arm.shift
            assessment, cost = assess_turn(
                meeting_date=meeting + _FALSE_DATE_OFFSET if shift else meeting,
                origin=origin + _FALSE_DATE_OFFSET if shift else origin,
                current_rate=current_rate,
                recent_decisions=[(d + _FALSE_DATE_OFFSET, o) for d, o in history] if shift else history,
                trailing_move_rate=trailing,
                statement_text=arm.statement_text,
                statement_date=arm.statement_date,
                n_samples=args.n_samples,
                temperature=args.temperature,
                intervening_meeting=(
                    None if intervening is None else intervening + _FALSE_DATE_OFFSET if shift else intervening
                ),
            )
            total_cost += cost
            verbatim, quoted = quotes_are_verbatim(assessment.evidence, arm.statement_text)
            rows.append(
                {
                    "meeting": meeting.strftime("%Y-%m-%d"),
                    "arm": arm.name,
                    "regime": assessment.regime,
                    "p_move": round(assessment.p_move, 4),
                    "quotes": quoted,
                    "quotes_verbatim": verbatim,
                    "rationale": assessment.rationale.replace("\n", " ")[:200],
                }
            )
            print(
                f"  {meeting:%Y-%m-%d} {arm.name:<13} p_move {assessment.p_move:.2f}  {assessment.regime}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    tag = f"{args.window}_{args.horizon.replace('+', '')}"
    if args.n_samples > 1:
        tag += f"_k{args.n_samples}"
    frame.to_csv(DATA_DIR / f"agent_labelfree_{tag}.csv", index=False)
    wide = frame.pivot(index="meeting", columns="arm", values="p_move")

    checks = {
        "window": args.window,
        "horizon": args.horizon,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "n_meetings": int(len(triples)),
        "reported_cost_usd": round(total_cost, 4),
        "p_move_std_baseline": round(float(wide["baseline"].std()), 4),
        "p_move_distinct_values": int(wide["baseline"].nunique()),
        "regimes_used": sorted(frame[frame["arm"] == "baseline"]["regime"].unique()),
        "quotes_total": int(frame.loc[frame["arm"] == "baseline", "quotes"].sum()),
        "quotes_verbatim": int(frame.loc[frame["arm"] == "baseline", "quotes_verbatim"].sum()),
        "mean_abs_shift_without_statement": round(float((wide["baseline"] - wide["no_statement"]).abs().mean()), 4),
        "mean_abs_shift_with_false_date": round(float((wide["baseline"] - wide["false_date"]).abs().mean()), 4),
        "corr_baseline_vs_false_date": round(float(wide["baseline"].corr(wide["false_date"])), 4),
    }
    (DATA_DIR / f"agent_labelfree_checks_{tag}.json").write_text(json.dumps(checks, indent=2))

    print("\n=== p_move by arm ===")
    print(wide.round(2).to_string())
    print("\n=== label-free checks (no outcome was read) ===")
    for key, value in checks.items():
        print(f"  {key}: {value}")
    print(f"\nwrote agent_labelfree_{tag}.csv and its checks json in {DATA_DIR}")


if __name__ == "__main__":
    main()
