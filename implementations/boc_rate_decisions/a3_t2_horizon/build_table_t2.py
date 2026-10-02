"""Build the committed ``origin_date -> (p_cut, p_hold, p_hike)`` table for t+2.

Mirrors :mod:`boc_rate_decisions.a1_market_implied.build_table`, one horizon
further out: for every origin in the A3 spec it prices the announcement after
the next one and writes ``data/market_implied_t2.csv``, diagnostics included so
a suspicious cell can be traced back to the two contracts behind it.

Usage
-----
``python -m boc_rate_decisions.a3_t2_horizon.build_table_t2``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from ..a1_market_implied.build_table import load_raw
from ..data import load_meeting_schedule
from .implied_t2 import implied_quote_t2


DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_SPEC = Path(__file__).resolve().parent / "specs" / "boc_rate_direction_t2_eval.yaml"
OUTPUT_PATH = DATA_DIR / "market_implied_t2.csv"


def origin_triples(spec_path: Path) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Return ``(origin, next_meeting, target_meeting)`` for every spec origin.

    The intervening meeting is ``origin + horizon`` (cross-checked against the
    committed calendar) and the target is the next scheduled announcement after
    it, so a drifting spec fails loudly here rather than pricing the wrong month.
    """
    spec = yaml.safe_load(spec_path.read_text())
    horizon = int(spec["task"]["horizons"][0])
    schedule = sorted(pd.Timestamp(d).normalize() for d in load_meeting_schedule())

    triples: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for raw_origin in spec["origin_dates"]:
        origin = pd.Timestamp(raw_origin).normalize()
        next_meeting = origin + pd.Timedelta(days=horizon)
        if next_meeting not in schedule:
            raise ValueError(f"Origin {origin:%Y-%m-%d} + {horizon}d is not a scheduled meeting.")
        later = [m for m in schedule if m > next_meeting]
        if not later:
            raise ValueError(f"No scheduled meeting after {next_meeting:%Y-%m-%d}.")
        triples.append((origin, next_meeting, later[0]))
    return triples


def build(spec_path: Path) -> pd.DataFrame:
    """Compute one implied-probability row per origin in ``spec_path``."""
    coa, corra = load_raw()
    schedule = load_meeting_schedule()

    rows: list[dict[str, object]] = []
    for origin, next_meeting, target in origin_triples(spec_path):
        quote = implied_quote_t2(origin, next_meeting, target, coa=coa, corra=corra, meeting_dates=schedule)
        rows.append(
            {
                "origin_date": origin.strftime("%Y-%m-%d"),
                "next_meeting": next_meeting.strftime("%Y-%m-%d"),
                "target_meeting": target.strftime("%Y-%m-%d"),
                "p_cut": round(quote.p_cut, 6),
                "p_hold": round(quote.p_hold, 6),
                "p_hike": round(quote.p_hike, 6),
                "method": quote.method,
                "corra_now": round(quote.corra_now, 4),
                "rate_after_next": round(quote.expected_rate_after_next, 4),
                "rate_after_target": round(quote.expected_rate_after_target, 4),
                "expected_move_pp": round(quote.expected_move, 4),
                "source": "TMX Montreal Exchange COA (One-Month CORRA Futures) settlement + BoC Valet AVG.INTWO",
                "diagnostics": json.dumps(quote.diagnostics, sort_keys=True, default=str),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Build the t+2 table for a spec and write it to ``data/``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    table = build(args.spec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)

    print(table.drop(columns=["diagnostics", "source"]).to_string(index=False))
    print(f"\nwrote {len(table)} rows -> {args.output}")


if __name__ == "__main__":
    main()
