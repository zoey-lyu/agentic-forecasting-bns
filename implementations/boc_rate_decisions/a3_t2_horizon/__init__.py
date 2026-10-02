"""A3 — forecasting the meeting after next, and pricing it from the same curve.

A1 found nothing beats the futures curve four weeks before a decision. A3 tests
whether that is a fact about forecasting or about horizon, by moving the target
to the announcement after the next one — roughly ten weeks out — and extending
A1's futures decomposition to reach it.

Modules
-------
``data_t2``
    The shifted target series. Its timestamps sit on the intervening meeting and
    its values carry the following decision, so a fixed 28-day task resolves
    onto a ten-week-ahead outcome without bending the harness.
``implied_t2``
    Two-leg futures decomposition: the expected rate after the next meeting,
    then after the one following it.
``build_table_t2``
    Prices every origin in the A3 spec.
``run_a3``
    Leaderboard, the horizon comparison on matched meetings, and the split by
    what the Bank actually did.
"""
