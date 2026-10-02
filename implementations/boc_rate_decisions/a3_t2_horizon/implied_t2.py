"""Market-implied probabilities for the meeting after next.

A1 reads one decision out of the CORRA futures curve. Reaching the following
decision means reading two, in sequence, because the second one starts from
wherever the first one leaves the rate:

1. ``r1`` — the expected overnight rate after the **next** meeting. This is
   exactly A1's estimate, reused unchanged.
2. ``r2`` — the expected rate after the meeting **after** that, from the same
   two estimators A1 uses, with one change: inside the later meeting's own
   month the days *before* its decision sit at ``r1``, not at today's CORRA,
   because the intervening decision has already happened by then.

The implied move at t+2 is ``r2 - r1``, and it maps to probabilities the same
way. Errors in ``r1`` propagate into ``r2``, which is a real cost of the
horizon and is reported rather than hidden: ``diagnostics`` carries both legs.

Feasibility was checked before building this: across the 13 eval origins every
t+2 contract is listed at the origin, seven origins get the clean next-month
estimator, and the other seven fall back to a meeting-month split with 13-28
post-decision days — all well conditioned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..a1_market_implied.implied import POLICY_STEP, ImpliedQuote, contract_symbol, implied_quote, move_to_probabilities


@dataclass(frozen=True)
class ImpliedT2Quote:
    """Market-implied probabilities for the meeting after next.

    Attributes
    ----------
    origin : pd.Timestamp
        Forecast origin.
    next_meeting, target_meeting : pd.Timestamp
        The intervening announcement and the one being forecast.
    corra_now : float
        Rate in force at the origin (percent).
    expected_rate_after_next : float
        ``r1`` — expected rate once the intervening decision is in effect.
    expected_rate_after_target : float
        ``r2`` — expected rate once the target decision is in effect.
    expected_move : float
        ``r2 - r1``, in percentage points.
    p_cut, p_hold, p_hike : float
        The implied distribution over the task's ordered categories.
    method : str
        Which estimator produced ``r2``.
    diagnostics : dict
        Both legs, the contracts used, and the A1 quote's own diagnostics.
    """

    origin: pd.Timestamp
    next_meeting: pd.Timestamp
    target_meeting: pd.Timestamp
    corra_now: float
    expected_rate_after_next: float
    expected_rate_after_target: float
    expected_move: float
    p_cut: float
    p_hold: float
    p_hike: float
    method: str
    diagnostics: dict[str, object] = field(default_factory=dict)

    def probabilities(self) -> dict[str, float]:
        """Return the distribution keyed by the task's category labels."""
        return {"cut": self.p_cut, "hold": self.p_hold, "hike": self.p_hike}


def implied_quote_t2(
    origin: pd.Timestamp,
    next_meeting: pd.Timestamp,
    target_meeting: pd.Timestamp,
    *,
    coa: pd.DataFrame,
    corra: pd.Series,
    meeting_dates: list[pd.Timestamp],
    root: str = "COA",
) -> ImpliedT2Quote:
    """Price the meeting after next from the CORRA futures curve.

    Parameters
    ----------
    origin : pd.Timestamp
        Forecast origin; only data dated at or before it is read.
    next_meeting, target_meeting : pd.Timestamp
        The intervening announcement and the announcement being forecast.
    coa : pd.DataFrame
        Settlement table (``date``, ``symbol``, ``settlement_price``,
        ``open_interest``), dates parsed.
    corra : pd.Series
        Daily CORRA indexed by date (percent).
    meeting_dates : list[pd.Timestamp]
        Full announcement calendar.
    root : str
        Futures root symbol.

    Returns
    -------
    ImpliedT2Quote

    Raises
    ------
    ValueError
        If the two meetings share a calendar month (the decomposition assumes
        they do not), or if no usable contract is listed for the later month.
    """
    origin = pd.Timestamp(origin).normalize()
    next_meeting = pd.Timestamp(next_meeting).normalize()
    target_meeting = pd.Timestamp(target_meeting).normalize()

    leg1: ImpliedQuote = implied_quote(
        origin, next_meeting, coa=coa, corra=corra, meeting_dates=meeting_dates, root=root
    )
    r1 = leg1.expected_rate

    target_month = target_meeting.to_period("M").to_timestamp()
    if next_meeting.to_period("M") == target_meeting.to_period("M"):
        raise ValueError(
            f"Meetings {next_meeting:%Y-%m-%d} and {target_meeting:%Y-%m-%d} share a month; "
            "the two-leg decomposition assumes one decision per contract month."
        )

    quote_date = leg1.quote_date
    month_end = target_month + pd.offsets.MonthEnd(0)
    effective_date = target_meeting + pd.offsets.BDay(1)
    next_month = target_month + pd.offsets.MonthBegin(1)

    diagnostics: dict[str, object] = {
        "leg1_method": leg1.method,
        "leg1_expected_move": round(leg1.expected_move, 4),
        "leg1_diagnostics": leg1.diagnostics,
        "quote_date": quote_date.strftime("%Y-%m-%d"),
    }

    def settlement(symbol: str) -> tuple[float, float] | None:
        row = coa[(coa["symbol"] == symbol) & (coa["date"] == quote_date)]
        if row.empty:
            return None
        return float(row["settlement_price"].iloc[0]), float(row["open_interest"].iloc[0])

    # --- estimator 1: the target meeting's own month -----------------------
    move_meeting_month: float | None = None
    symbol_m = contract_symbol(root, target_month)
    quote_m = settlement(symbol_m)
    if quote_m is not None and effective_date <= month_end:
        settle_m, oi_m = quote_m
        days = pd.date_range(target_month, month_end, freq="D")
        pre_days = int((days < effective_date).sum())
        post_days = len(days) - pre_days
        # Days before the target decision sit at r1: the intervening meeting
        # has already happened by the time this month starts.
        expected_rate_m = ((100.0 - settle_m) * len(days) - pre_days * r1) / post_days
        move_meeting_month = expected_rate_m - r1
        diagnostics |= {
            "meeting_month_symbol": symbol_m,
            "meeting_month_settle": settle_m,
            "meeting_month_open_interest": oi_m,
            "meeting_month_post_days": post_days,
            "meeting_month_move": round(move_meeting_month, 4),
        }

    # --- estimator 2: the month after, when it holds no decision -----------
    move_next_month: float | None = None
    next_month_clean = not any(
        next_month <= pd.Timestamp(m).normalize() <= next_month + pd.offsets.MonthEnd(0) for m in meeting_dates
    )
    symbol_n = contract_symbol(root, next_month)
    quote_n = settlement(symbol_n)
    if quote_n is not None and next_month_clean and effective_date <= month_end:
        settle_n, oi_n = quote_n
        move_next_month = (100.0 - settle_n) - r1
        diagnostics |= {
            "next_month_symbol": symbol_n,
            "next_month_settle": settle_n,
            "next_month_open_interest": oi_n,
            "next_month_move": round(move_next_month, 4),
        }
    diagnostics["next_month_clean"] = next_month_clean

    if move_next_month is not None:
        method, expected_move = "next_month", move_next_month
    elif move_meeting_month is not None:
        method, expected_move = "meeting_month", move_meeting_month
    else:
        raise ValueError(
            f"No usable {root} contract at {origin:%Y-%m-%d} for target meeting {target_meeting:%Y-%m-%d}."
        )

    p_cut, p_hold, p_hike = move_to_probabilities(expected_move)
    diagnostics["clamped"] = abs(expected_move) > POLICY_STEP
    if move_meeting_month is not None and move_next_month is not None:
        diagnostics["estimator_gap"] = round(move_next_month - move_meeting_month, 4)

    return ImpliedT2Quote(
        origin=origin,
        next_meeting=next_meeting,
        target_meeting=target_meeting,
        corra_now=leg1.corra_now,
        expected_rate_after_next=r1,
        expected_rate_after_target=r1 + expected_move,
        expected_move=expected_move,
        p_cut=p_cut,
        p_hold=p_hold,
        p_hike=p_hike,
        method=method,
        diagnostics=diagnostics,
    )


__all__ = ["ImpliedT2Quote", "implied_quote_t2"]
