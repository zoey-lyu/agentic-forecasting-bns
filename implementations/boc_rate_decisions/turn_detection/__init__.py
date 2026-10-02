"""Turn detection — the line of work that came out of A1 to A3.

Those three steps converged on one finding: the market's entire advantage lives
at turning points, and the quantity nothing fitted to history can track is how
active the Bank currently is. This package tests whether that gap can be closed,
and is organised as a sequence of pre-registered steps rather than a library —
``PREREGISTRATION.md`` is the spine, written before each result existed.

Steps, in the order they ran
---------------------------
``ceiling`` / ``run_steps``
    Step 0-1: how much could a router win, and does the structural finding
    replicate on a window nobody had scored?
``move_task`` / ``run_step2``
    Step 2: the binary P(move) instrument, its calibration, and the test that
    killed the "market sells insurance" hypothesis.
``run_calibration_test``
    Whether fixing the logistic's honesty alone beats the market. It does not,
    and the reason — base rates do not transport — recurs everywhere after.
``lexicon`` / ``lexicon_v2`` / ``regime_language`` / ``long_sample``
    Contamination-free word-count measures, and what they can and cannot do.
``turn_agent`` / ``run_agent_labelfree`` / ``run_agent_score``
    The agent, developed against criteria that read no outcome and scored once.
``live_forecast`` / ``score_live``
    The only evidence here that will never need a contamination caveat, and the
    slowest to arrive.
"""
