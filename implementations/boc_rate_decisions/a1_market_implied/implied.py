"""Turn One-Month CORRA futures settlements into cut/hold/hike probabilities.

The method is CME-FedWatch adapted to Canada. A ``COA`` contract for calendar
month *M* settles at ``100 - (average daily CORRA over M)``, so its settlement
price at a forecast origin states the market's expected average overnight rate
for that month. Splitting that month-average into the part before and after a
rate decision backs out the expected post-decision rate, and the expected move
divided by one 25 bp step is the implied probability of a move.

Two estimators, both computed, one chosen by a fixed rule:

``meeting_month``
    Use month *M* = the meeting's own month. Days before the decision takes
    effect are priced at the rate already in force (realised CORRA for days
    already past at the origin, the current CORRA level for the rest); solving
    the average for the remaining days gives the expected post-decision rate.
    Precision degrades as the decision moves toward month-end, because fewer
    and fewer days carry the whole signal.

``next_month``
    Use month *M+1*. When no BoC announcement falls in *M+1*, the entire month
    sits at the post-decision rate, so ``100 - settlement`` *is* the expected
    post-decision rate with no day-count arithmetic at all. Unavailable when
    the following month holds a meeting.

Selection rule (fixed in advance, never tuned on outcomes):
``next_month`` when it is available, otherwise ``meeting_month``. BoC meetings
are ~6 weeks apart, so a late-month decision — exactly the case that breaks
``meeting_month`` — always has a clean following month, and a mid-month
decision — where ``meeting_month`` is at its most precise — never does. Both
estimators are reported so the gap between them is visible.

Effective-date convention
-------------------------
Since 2021 a BoC decision takes effect the *business day after* the
announcement: the 2025-01-29 cut shows up in CORRA on 2025-01-30 (3.29 → 3.03)
and in the target rate series on the same day. ``effective_date`` therefore
defaults to the next business day.

Known approximations (all documented in the A1 README):
- CORRA sits a few bp above the target rate and spikes at month/quarter ends;
  the spread cancels in ``expected_move`` only if it is stable.
- A move is assumed to be exactly one 25 bp step. Pricing beyond a full step
  clamps at probability 1.0, which is correct for a *direction* task ("cut of
  any size") but discards the magnitude the market implies.
- All expectation is attributed to the single scheduled meeting; an
  inter-meeting move would be misread as meeting probability. The 28-day lead
  guarantees no other scheduled announcement falls in between.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


#: Month codes used by the exchange, in the standard futures convention.
MONTH_CODES = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M", 7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}

#: One policy step. The Bank has moved in 25 bp increments (and multiples) since
#: the 2020 emergency cuts; 50 bp moves clamp to probability 1.0 on a direction task.
POLICY_STEP = 0.25

#: Trailing window used to read "the rate in force at the origin" from CORRA.
#: A median over five business days is robust to month-end and quarter-end
#: spikes, which matter because two eval origins fall on 31 December.
_CORRA_WINDOW_DAYS = 5


def contract_symbol(root: str, month: pd.Timestamp) -> str:
    """Return the exchange symbol for a contract month, e.g. ``COAF25``."""
    return f"{root}{MONTH_CODES[month.month]}{month.year % 100:02d}"


@dataclass(frozen=True)
class ImpliedQuote:
    """Market-implied decision probabilities for one (origin, meeting) pair.

    Attributes
    ----------
    origin, meeting_date, effective_date : pd.Timestamp
        Forecast origin, the announcement being predicted, and the day the
        decision starts showing up in CORRA.
    quote_date : pd.Timestamp
        Trading day the settlement prices were read from: the latest session
        at or before ``origin``.
    corra_now : float
        Rate in force at the origin (trailing median CORRA, percent).
    expected_rate : float
        Expected post-decision CORRA implied by the chosen estimator.
    expected_move : float
        ``expected_rate - corra_now``, in percentage points.
    p_cut, p_hold, p_hike : float
        The implied distribution over the task's ordered categories.
    method : str
        ``"next_month"`` or ``"meeting_month"`` — which estimator was chosen.
    diagnostics : dict
        Both estimators' expected moves, the contracts and settlements used,
        open interest, and any clamping applied.
    """

    origin: pd.Timestamp
    meeting_date: pd.Timestamp
    effective_date: pd.Timestamp
    quote_date: pd.Timestamp
    corra_now: float
    expected_rate: float
    expected_move: float
    p_cut: float
    p_hold: float
    p_hike: float
    method: str
    diagnostics: dict[str, object] = field(default_factory=dict)

    def probabilities(self) -> dict[str, float]:
        """Return the distribution keyed by the task's category labels."""
        return {"cut": self.p_cut, "hold": self.p_hold, "hike": self.p_hike}


def move_to_probabilities(expected_move: float, step: float = POLICY_STEP) -> tuple[float, float, float]:
    """Map an expected rate move (percentage points) to ``(p_cut, p_hold, p_hike)``.

    A move of exactly one step down means a certain cut; half a step down means
    an even chance of a cut. Moves beyond one step clamp at 1.0, which is the
    right reading for a direction task that does not distinguish 25 from 50 bp.
    """
    if expected_move < 0:
        p_cut = min(1.0, -expected_move / step)
        return p_cut, 1.0 - p_cut, 0.0
    if expected_move > 0:
        p_hike = min(1.0, expected_move / step)
        return 0.0, 1.0 - p_hike, p_hike
    return 0.0, 1.0, 0.0


def _settlement_at(coa: pd.DataFrame, symbol: str, quote_date: pd.Timestamp) -> tuple[float, float] | None:
    """Return ``(settlement_price, open_interest)`` for a contract on a date."""
    row = coa[(coa["symbol"] == symbol) & (coa["date"] == quote_date)]
    if row.empty:
        return None
    return float(row["settlement_price"].iloc[0]), float(row["open_interest"].iloc[0])


def _corra_level(corra: pd.Series, origin: pd.Timestamp) -> float:
    """Rate in force at ``origin``: median of the trailing five CORRA prints."""
    visible = corra[corra.index <= origin]
    if visible.empty:
        raise ValueError(f"No CORRA observations at or before {origin:%Y-%m-%d}.")
    return float(visible.iloc[-_CORRA_WINDOW_DAYS:].median())


def implied_quote(
    origin: pd.Timestamp,
    meeting_date: pd.Timestamp,
    *,
    coa: pd.DataFrame,
    corra: pd.Series,
    meeting_dates: list[pd.Timestamp],
    root: str = "COA",
) -> ImpliedQuote:
    """Compute one origin's market-implied cut/hold/hike distribution.

    Parameters
    ----------
    origin : pd.Timestamp
        Forecast origin. Only data dated at or before it is read.
    meeting_date : pd.Timestamp
        The announcement being forecast.
    coa : pd.DataFrame
        Settlement table from :mod:`fetch_market_data` (``date``, ``symbol``,
        ``settlement_price``, ``open_interest``), dates parsed.
    corra : pd.Series
        Daily CORRA indexed by date (percent).
    meeting_dates : list[pd.Timestamp]
        Full announcement calendar, used to test whether the month after the
        meeting is free of decisions.
    root : str
        Futures root symbol; ``COA`` (One-Month CORRA Futures).

    Returns
    -------
    ImpliedQuote

    Raises
    ------
    ValueError
        If no contract settlement is available at the origin for either
        estimator, or if another announcement falls between origin and meeting
        (which would break the single-meeting attribution).
    """
    origin = pd.Timestamp(origin).normalize()
    meeting_date = pd.Timestamp(meeting_date).normalize()
    intervening = [m for m in meeting_dates if origin < pd.Timestamp(m).normalize() < meeting_date]
    if intervening:
        raise ValueError(f"Announcement(s) {intervening} fall between origin and meeting; attribution is ambiguous.")

    sessions = coa.loc[coa["date"] <= origin, "date"]
    if sessions.empty:
        raise ValueError(f"No {root} settlements at or before {origin:%Y-%m-%d}.")
    quote_date = sessions.max()

    effective_date = meeting_date + pd.offsets.BDay(1)
    corra_now = _corra_level(corra, origin)
    meeting_month = meeting_date.to_period("M").to_timestamp()
    month_end = meeting_month + pd.offsets.MonthEnd(0)
    next_month = meeting_month + pd.offsets.MonthBegin(1)

    diagnostics: dict[str, object] = {
        "quote_date": quote_date.strftime("%Y-%m-%d"),
        "corra_now": round(corra_now, 4),
    }

    # --- estimator 1: the meeting's own month -------------------------------
    move_meeting_month: float | None = None
    symbol_m = contract_symbol(root, meeting_month)
    quote_m = _settlement_at(coa, symbol_m, quote_date)
    if quote_m is not None and effective_date <= month_end:
        settle_m, oi_m = quote_m
        implied_avg = 100.0 - settle_m
        days = pd.date_range(meeting_month, month_end, freq="D")
        pre = days[days < effective_date]
        post_days = len(days) - len(pre)
        # Days already elapsed at the origin are realised CORRA, not a forecast.
        realised = corra.reindex(pd.date_range(corra.index.min(), corra.index.max(), freq="D")).ffill()
        pre_sum = sum(float(realised.loc[d]) if d <= quote_date else corra_now for d in pre)
        expected_rate_m = (implied_avg * len(days) - pre_sum) / post_days
        move_meeting_month = expected_rate_m - corra_now
        diagnostics |= {
            "meeting_month_symbol": symbol_m,
            "meeting_month_settle": settle_m,
            "meeting_month_open_interest": oi_m,
            "meeting_month_post_days": post_days,
            "meeting_month_move": round(move_meeting_month, 4),
        }

    # --- estimator 2: the month after, when it holds no decision ------------
    move_next_month: float | None = None
    next_month_clean = not any(
        next_month <= pd.Timestamp(m).normalize() <= next_month + pd.offsets.MonthEnd(0) for m in meeting_dates
    )
    symbol_n = contract_symbol(root, next_month)
    quote_n = _settlement_at(coa, symbol_n, quote_date)
    if quote_n is not None and next_month_clean and effective_date <= month_end:
        settle_n, oi_n = quote_n
        expected_rate_n = 100.0 - settle_n
        move_next_month = expected_rate_n - corra_now
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
        raise ValueError(f"No usable {root} contract at {origin:%Y-%m-%d} for meeting {meeting_date:%Y-%m-%d}.")

    p_cut, p_hold, p_hike = move_to_probabilities(expected_move)
    diagnostics["clamped"] = abs(expected_move) > POLICY_STEP
    if move_meeting_month is not None and move_next_month is not None:
        diagnostics["estimator_gap"] = round(move_next_month - move_meeting_month, 4)

    return ImpliedQuote(
        origin=origin,
        meeting_date=meeting_date,
        effective_date=effective_date,
        quote_date=quote_date,
        corra_now=corra_now,
        expected_rate=corra_now + expected_move,
        expected_move=expected_move,
        p_cut=p_cut,
        p_hold=p_hold,
        p_hike=p_hike,
        method=method,
        diagnostics=diagnostics,
    )


__all__ = ["MONTH_CODES", "POLICY_STEP", "ImpliedQuote", "contract_symbol", "implied_quote", "move_to_probabilities"]
