"""Record a prospective forecast, before the Bank has decided.

Every score in this folder comes from a window the model may have read. The
label-free checks measured what that costs — on 2023-24 the agent recited a
withheld statement from memory — and the answer is that contamination is not a
theoretical worry here, it is the dominant effect on any window inside training
data.

There is exactly one cure, and it is not a method: make the forecast before the
outcome exists. This script does that and writes it down. Each run appends one
immutable record for one announcement, containing every arm's probability, the
agent's reasoning, and the market quote as it stood that day. The records
accumulate at eight per year, which is slow — and is why the first one matters
more than the twentieth.

Discipline the file depends on:

- Run it **on or before the origin date**, never after. The script refuses to
  write a record for a meeting that has already happened.
- Append only. A record is never edited once written; a mistaken run is
  superseded by a later record with the same origin, and both stay.
- No tuning between runs. The prompt is frozen (see ``PREREGISTRATION.md``);
  changing it starts a new series rather than continuing this one.

Usage
-----
``python -m boc_rate_decisions.turn_detection.live_forecast``            # today
``python -m boc_rate_decisions.turn_detection.live_forecast --dry-run``  # no record
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..a1_market_implied.fetch_market_data import fetch_coa_settlements, fetch_valet_series
from ..a1_market_implied.implied import implied_quote
from ..a2_text_features.extract_stance_table import DEFAULT_CACHE_DIR, statement_paths
from ..data import DIRECTION_SERIES_ID, TARGET_RATE_SERIES_ID, build_boc_service, load_meeting_schedule
from .lexicon_v2 import lexicon_scores_v2
from .run_steps import USE_CASE_DIR
from .turn_agent import assess_turn


DATA_DIR = Path(__file__).resolve().parent / "data"
LIVE_RECORDS = DATA_DIR / "live_forecasts.jsonl"

#: Horizon of the canonical task, in days.
LEAD_DAYS = 28

_LABELS = {-1.0: "cut", 0.0: "hold", 1.0: "hike"}


def next_meeting_for(origin: pd.Timestamp) -> pd.Timestamp:
    """Return the announcement this origin forecasts, or raise.

    The canonical task predicts the announcement exactly ``LEAD_DAYS`` after the
    origin, so an origin that does not land on one is a mistake worth stopping
    for rather than quietly rounding to the nearest meeting.
    """
    schedule = {pd.Timestamp(d).normalize() for d in load_meeting_schedule()}
    meeting = origin.normalize() + pd.Timedelta(days=LEAD_DAYS)
    if meeting not in schedule:
        upcoming = sorted(d for d in schedule if d > origin)
        raise SystemExit(
            f"{origin:%Y-%m-%d} + {LEAD_DAYS}d = {meeting:%Y-%m-%d}, which is not a scheduled announcement.\n"
            f"The next announcements are {[f'{d:%Y-%m-%d}' for d in upcoming[:3]]}; "
            f"their origins are {[f'{d - pd.Timedelta(days=LEAD_DAYS):%Y-%m-%d}' for d in upcoming[:3]]}."
        )
    return meeting


def _market_quote(origin: pd.Timestamp, meeting: pd.Timestamp) -> dict[str, object]:
    """Fetch fresh futures data and price the meeting, as of the origin."""
    start = (origin - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    end = origin.strftime("%Y-%m-%d")
    coa = fetch_coa_settlements(start, end)
    coa["date"] = pd.to_datetime(coa["date"])
    corra_frame = fetch_valet_series("AVG.INTWO", start, end)
    corra = corra_frame.assign(date=pd.to_datetime(corra_frame["date"])).set_index("date")["value"]

    quote = implied_quote(origin, meeting, coa=coa, corra=corra, meeting_dates=load_meeting_schedule())
    return {
        "p_cut": round(quote.p_cut, 4),
        "p_hold": round(quote.p_hold, 4),
        "p_hike": round(quote.p_hike, 4),
        "p_move": round(quote.p_cut + quote.p_hike, 4),
        "method": quote.method,
        "quote_date": quote.quote_date.strftime("%Y-%m-%d"),
        "expected_move_pp": round(quote.expected_move, 4),
    }


def main() -> None:
    """Produce and record one prospective forecast."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", default=None, help="Forecast origin (default: today).")
    parser.add_argument("--dry-run", action="store_true", help="Print the forecast without recording it.")
    parser.add_argument("--statcan-cache-dir", type=Path, default=USE_CASE_DIR / "data" / "statcan")
    parser.add_argument("--fred-cache-dir", type=Path, default=USE_CASE_DIR / "data" / "fred")
    args = parser.parse_args()

    origin = pd.Timestamp(args.origin).normalize() if args.origin else pd.Timestamp.today().normalize()
    meeting = next_meeting_for(origin)
    today = pd.Timestamp.today().normalize()
    if meeting <= today and not args.dry_run:
        raise SystemExit(
            f"The {meeting:%Y-%m-%d} announcement has already happened; a record written now would not be "
            "prospective. Use --dry-run to inspect the forecast anyway."
        )

    service = build_boc_service(statcan_cache_dir=args.statcan_cache_dir, fred_cache_dir=args.fred_cache_dir)
    directions = service.get_series(DIRECTION_SERIES_ID, as_of=origin.to_pydatetime())
    rates = service.get_series(TARGET_RATE_SERIES_ID, as_of=origin.to_pydatetime())
    history = [
        (pd.Timestamp(t), _LABELS[float(v)])
        for t, v in zip(directions["timestamp"].tail(5), directions["value"].tail(5), strict=True)
    ]
    trailing = float((directions["value"].astype(float).tail(8) != 0.0).mean())
    current_rate = float(rates["value"].iloc[-1])

    paths = statement_paths(DEFAULT_CACHE_DIR)
    visible = [date for date in paths if date <= origin]
    if not visible:
        raise SystemExit(f"No cached statement published on or before {origin:%Y-%m-%d}.")
    statement_date = max(visible)
    statement = paths[statement_date].read_text()

    agent, agent_cost = assess_turn(
        meeting_date=meeting,
        origin=origin,
        current_rate=current_rate,
        recent_decisions=history,
        trailing_move_rate=trailing,
        statement_date=statement_date,
        statement_text=statement,
    )
    control, control_cost = assess_turn(
        meeting_date=meeting,
        origin=origin,
        current_rate=current_rate,
        recent_decisions=history,
        trailing_move_rate=trailing,
    )

    record = {
        "created_at": datetime.now(tz=timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"),
        "origin": origin.strftime("%Y-%m-%d"),
        "meeting": meeting.strftime("%Y-%m-%d"),
        "current_rate": current_rate,
        "trailing_move_rate": round(trailing, 4),
        "statement_date": statement_date.strftime("%Y-%m-%d"),
        "market": _market_quote(origin, meeting),
        "agent": {
            "p_move": round(agent.p_move, 4),
            "regime": agent.regime,
            "regime_reason": agent.regime_reason,
            "rationale": agent.rationale,
            "evidence": agent.evidence,
        },
        "agent_no_statement": {"p_move": round(control.p_move, 4), "regime": control.regime},
        "lexicon_v2": {k: round(v, 4) for k, v in lexicon_scores_v2(statement).items()},
        "cost_usd": round(agent_cost + control_cost, 4),
    }

    print(json.dumps(record, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("\n(dry run - nothing recorded)")
        return

    with LIVE_RECORDS.open("a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nappended one record to {LIVE_RECORDS}")


if __name__ == "__main__":
    main()
