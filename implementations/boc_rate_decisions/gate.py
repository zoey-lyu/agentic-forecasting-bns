"""Validation gate for BoC calibration corrections.

A *correction* is a typed pair — a :class:`Condition` saying **when** it
applies and an :class:`Adjustment` saying **what** it does to the predicted
probability vector. The gate decides whether a correction is allowed to
become permanent by putting it through four checks.

Why four and not one
--------------------
Every check exists because a specific result died on it. On the WTI side of
this repo, four separate "the agent helps" findings each passed a
significance test and then collapsed under exactly one of these controls
(see ``planning-docs/anchored-agent-cross-regime-findings.md`` section 10).
A single "is it significant?" gate would have accepted all four.

=====  ==========================================  ==========================
bar    question                                    fires when
=====  ==========================================  ==========================
1      does it beat no adjustment at all?          always
2      does it beat its own constant?              adjustment varies per cell
3      does the condition earn its place?          condition is not ``always``
4      does it hold in another regime?             always
=====  ==========================================  ==========================

Bar 3 is why :data:`ConditionKind` includes ``always``: it is not a
placeholder, it is bar 3's control. A conditional rule must beat the *same*
adjustment applied unconditionally, or the condition is decoration. The one
hypothesis the adaptive agent ever recorded (``hyp-001``) fails exactly here
by its own logged evidence.

Bar 2 only bites when the adjustment carries per-cell judgement. A constant
adjustment *is* its own constant, so it passes trivially — recorded as
``None`` rather than a pass, to keep the distinction visible.

Scope
-----
Categorical (cut/hold/hike) forecasts scored with RPS. Origins are BoC fixed
announcement dates, which do not overlap, so the bootstrap resamples origins
directly — unlike the WTI side, where three horizons share one origin's
inputs and clustering was required.

This module never writes to the shared prediction store. Callers that want
artefacts should pass an explicit directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np
import pandas as pd


ConditionKind = Literal["always", "rate_momentum", "yield_spread", "n_train", "era", "last_decision"]
AdjustmentChannel = Literal["sharpness", "mass"]
AdjustmentOp = Literal["power", "mix_uniform", "mix_baserate", "shift_toward", "shift_away"]


# ---------------------------------------------------------------------------
# The typed correction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    """When a correction applies.

    Parameters
    ----------
    kind : ConditionKind
        ``always`` matches every cell and is the control bar 3 compares
        against. The others select on a column of the cell frame.
    value : str
        Bucket label to match, e.g. ``"easing"`` for ``rate_momentum``.
        Ignored when ``kind == "always"``.
    """

    kind: ConditionKind = "always"
    value: str = ""

    def mask(self, cells: pd.DataFrame) -> np.ndarray:
        """Boolean mask over ``cells`` selecting the rows this applies to."""
        if self.kind == "always":
            return np.ones(len(cells), dtype=bool)
        if self.kind not in cells.columns:
            raise KeyError(f"condition kind {self.kind!r} is not a column of the cell frame")
        return (cells[self.kind] == self.value).to_numpy()

    def unconditional(self) -> "Condition":
        """The bar-3 control: same adjustment, no condition."""
        return Condition(kind="always", value="")

    def __str__(self) -> str:
        return "always" if self.kind == "always" else f"{self.kind}={self.value}"


@dataclass(frozen=True)
class Adjustment:
    """What a correction does to a probability vector.

    ``sharpness``/``power`` raises each probability to ``value`` and
    renormalises: above 1 sharpens (more confident), below 1 flattens. This is
    the categorical analogue of scaling a prediction interval's width, and it
    is the first thing to test because the equivalent WTI finding — that the
    statistical model's uncertainty is miscalibrated by a *level* — was the
    only result that replicated across regimes.

    ``mass``/``mix_uniform`` blends toward the uniform vector by ``value``;
    ``mass``/``mix_baserate`` blends toward the supplied base rate. Both are
    pure flattening and cannot sharpen.

    ``mass``/``shift_toward`` moves ``value`` of the mass onto ``target``;
    ``mass``/``shift_away`` scales ``target`` down by ``value`` and
    redistributes proportionally over the others. These two move the
    distribution's *centre*, which no sharpness transform can do — a
    correction of the form "in easing cycles the model under-weights cut" is
    only expressible here.
    """

    channel: AdjustmentChannel = "sharpness"
    op: AdjustmentOp = "power"
    value: float = 1.0
    target: str = ""
    """Category label the mass ops act on. Unused by the other ops."""

    @property
    def is_noop(self) -> bool:
        """True when the adjustment provably cannot change any probability."""
        if self.op == "power":
            return self.value == 1.0
        return self.value == 0.0

    def apply(
        self,
        probs: np.ndarray,
        base_rate: np.ndarray | None = None,
        labels: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Return adjusted probabilities; ``probs`` is ``(n, k)``, rows sum to 1.

        ``labels`` gives the column order and is required by the mass-shift
        ops so ``target`` can be resolved to a column.
        """
        p = np.asarray(probs, dtype=float)
        if self.op == "power":
            out = np.power(np.clip(p, 1e-12, None), self.value)
            return out / out.sum(axis=1, keepdims=True)
        if self.op == "mix_uniform":
            u = np.full_like(p, 1.0 / p.shape[1])
            return (1.0 - self.value) * p + self.value * u
        if self.op == "mix_baserate":
            if base_rate is None:
                raise ValueError("mix_baserate requires base_rate")
            b = np.broadcast_to(np.asarray(base_rate, dtype=float), p.shape)
            return (1.0 - self.value) * p + self.value * b
        if self.op in ("shift_toward", "shift_away"):
            if labels is None:
                raise ValueError(f"{self.op} requires labels to resolve target {self.target!r}")
            if self.target not in labels:
                raise ValueError(f"target {self.target!r} not in labels {list(labels)}")
            j = list(labels).index(self.target)
            if self.op == "shift_toward":
                onehot = np.zeros_like(p)
                onehot[:, j] = 1.0
                return (1.0 - self.value) * p + self.value * onehot
            out = np.array(p, dtype=float, copy=True)
            out[:, j] *= 1.0 - self.value
            total = out.sum(axis=1, keepdims=True)
            # A row that was entirely on the target would sum to zero; fall
            # back to uniform there rather than dividing by zero.
            degenerate = (total <= 1e-12).ravel()  # noqa: PLR2004
            out[degenerate] = 1.0 / p.shape[1]
            total = out.sum(axis=1, keepdims=True)
            return out / total
        raise ValueError(f"unknown op {self.op!r}")

    def __str__(self) -> str:
        tgt = f", {self.target}" if self.target else ""
        return f"{self.channel}.{self.op}({self.value:g}{tgt})"


@dataclass(frozen=True)
class Correction:
    """A condition plus an adjustment — the unit the gate accepts or rejects."""

    condition: Condition
    adjustment: Adjustment

    def apply(
        self,
        cells: pd.DataFrame,
        probs: np.ndarray,
        base_rate: np.ndarray | None = None,
        labels: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Apply the adjustment to the rows the condition selects."""
        out = np.array(probs, dtype=float, copy=True)
        m = self.condition.mask(cells)
        if m.any():
            out[m] = self.adjustment.apply(out[m], base_rate=base_rate, labels=labels)
        return out

    def __str__(self) -> str:
        return f"[{self.condition}] -> {self.adjustment}"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def rps(probs: np.ndarray, outcome_idx: Sequence[int]) -> np.ndarray:
    """Per-row ranked probability score for an ordered categorical forecast.

    Lower is better. ``probs`` is ``(n, k)`` with columns in the task's
    declared category order; ``outcome_idx`` gives the realised column.

    Uses the **unnormalized** Epstein/Murphy convention — ``sum`` over the
    first ``k-1`` cumulative differences with no division by ``k-1`` — to
    match :func:`aieng.forecasting.evaluation.compute_rps`. Both
    normalizations appear in the literature; mixing them silently halves
    every number, so ``scripts/run_gate.py`` asserts that this function
    reproduces the library's mean score before any gate runs.
    """
    p = np.asarray(probs, dtype=float)
    n, k = p.shape
    obs = np.zeros_like(p)
    obs[np.arange(n), np.asarray(outcome_idx, dtype=int)] = 1.0
    return ((p.cumsum(axis=1) - obs.cumsum(axis=1)) ** 2)[:, : k - 1].sum(axis=1)


def bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 10_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap ``mean(a - b)`` over origins.

    Returns ``(point, lo, hi)`` for a 95% interval. Seeded so a gate verdict
    is reproducible — an unseeded gate would sometimes accept and sometimes
    reject the same correction.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    draws = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def _default_control_grid(adjustment: Adjustment) -> list[float]:
    """Strengths bar 3's unconditional control may choose from.

    Always contains the op's no-op value so the control can decline to act.
    """
    if adjustment.op == "power":
        return [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5]
    return [0.0, 0.05, 0.10, 0.20, 0.30]


@dataclass
class BarResult:
    """One bar's verdict. ``passed is None`` means the bar did not apply."""

    name: str
    passed: bool | None
    detail: str

    def __str__(self) -> str:
        mark = {True: "PASS", False: "FAIL", None: "n/a "}[self.passed]
        return f"  {mark}  bar {self.name}: {self.detail}"


@dataclass
class GateVerdict:
    """The full four-bar result for one correction."""

    correction: Correction
    bars: list[BarResult]

    @property
    def accepted(self) -> bool:
        """True only when no applicable bar failed."""
        return all(b.passed is not False for b in self.bars)

    def __str__(self) -> str:
        head = f"{self.correction}  ->  {'ACCEPTED' if self.accepted else 'REJECTED'}"
        return "\n".join([head, *(str(b) for b in self.bars)])


def run_gate(
    correction: Correction,
    windows: dict[str, pd.DataFrame],
    *,
    prob_cols: Sequence[str],
    outcome_col: str = "outcome_idx",
    per_cell_adjustment: Callable[[pd.DataFrame], np.ndarray] | None = None,
    base_rate: np.ndarray | None = None,
    labels: Sequence[str] | None = None,
    control_grid: Sequence[float] | None = None,
) -> GateVerdict:
    """Put one correction through all four bars.

    Parameters
    ----------
    correction : Correction
        The typed rule under test.
    windows : dict[str, pandas.DataFrame]
        Named cell frames, one per regime. Bar 4 requires at least two; the
        first is treated as the primary window for bars 1-3.
    prob_cols : sequence of str
        Probability columns, in the task's category order.
    outcome_col : str
        Column holding the realised category index.
    per_cell_adjustment : callable, optional
        Supply only when the adjustment varies per cell — bar 2 then replaces
        it with its own mean and rescores. Left ``None`` for constants, which
        makes bar 2 inapplicable rather than passed.
    base_rate : numpy.ndarray, optional
        Required by ``mix_baserate``.
    labels : sequence of str, optional
        Category labels in column order. Required by the mass-shift ops so
        ``Adjustment.target`` can be resolved; defaults to ``prob_cols`` with
        any ``p_`` prefix stripped.
    control_grid : sequence of float, optional
        Strengths bar 3's unconditional control is refit over. Defaults to
        :func:`_default_control_grid`, which always includes the no-op.

    Returns
    -------
    GateVerdict
    """
    if labels is None:
        labels = [c[2:] if c.startswith("p_") else c for c in prob_cols]
    names = list(windows)
    if len(names) < 2:  # noqa: PLR2004
        raise ValueError("bar 4 needs at least two windows")
    primary = windows[names[0]]

    def scored(cells: pd.DataFrame, corr: Correction) -> np.ndarray:
        p = cells[list(prob_cols)].to_numpy(dtype=float)
        return rps(corr.apply(cells, p, base_rate=base_rate, labels=labels), cells[outcome_col].to_numpy())

    def baseline(cells: pd.DataFrame) -> np.ndarray:
        return rps(cells[list(prob_cols)].to_numpy(dtype=float), cells[outcome_col].to_numpy())

    bars: list[BarResult] = []

    # --- bar 1: beats no adjustment ---------------------------------------
    adj, lo, hi = bootstrap_ci(scored(primary, correction), baseline(primary))
    ok1 = hi < 0
    bars.append(
        BarResult(
            "1 beats nothing",
            ok1,
            f"delta RPS {adj:+.4f} [{lo:+.4f}, {hi:+.4f}] on {names[0]} (n={len(primary)})",
        )
    )

    # --- bar 2: beats its own constant ------------------------------------
    if per_cell_adjustment is None:
        bars.append(BarResult("2 beats own constant", None, "adjustment is a constant, nothing to control"))
    else:
        per_cell = per_cell_adjustment(primary)
        const = np.full_like(per_cell, float(np.mean(per_cell)))
        p = primary[list(prob_cols)].to_numpy(dtype=float)
        outcome = primary[outcome_col].to_numpy()

        def _apply_vec(vals: np.ndarray) -> np.ndarray:
            out = np.array(p, dtype=float, copy=True)
            for i, v in enumerate(vals):
                out[i] = Adjustment(
                    correction.adjustment.channel,
                    correction.adjustment.op,
                    float(v),
                    correction.adjustment.target,
                ).apply(p[i : i + 1], base_rate=base_rate, labels=labels)[0]
            return rps(out, outcome)

        d, lo2, hi2 = bootstrap_ci(_apply_vec(per_cell), _apply_vec(const))
        ok2 = hi2 < 0
        bars.append(BarResult("2 beats own constant", ok2, f"real - constant {d:+.4f} [{lo2:+.4f}, {hi2:+.4f}]"))

    # --- bar 3: beats the BEST unconditional version ----------------------
    #
    # The control is refit, not copied. An earlier version reused the
    # conditional rule's own strength for the unconditional comparison, which
    # made bar 3 a straw man for any adjustment that only makes sense on a
    # minority of cells: "shift 20% toward cut everywhere" wrecks the 94 hold
    # cells out of 120, so "shift only in easing cycles" beat it easily while
    # still not beating no adjustment at all. Four candidates passed bar 3
    # that way and none of them were worth anything.
    #
    # The grid includes the adjustment's own no-op, so the control may choose
    # to do nothing. That makes bar 3 subsume bar 1 by construction — which is
    # deliberate: "does the condition earn its place?" is only a real question
    # against the best simpler alternative, and doing nothing is one. Bar 1 is
    # still reported separately because the two failures mean different things.
    if correction.condition.kind == "always":
        bars.append(BarResult("3 beats unconditional", None, "condition is already 'always'"))
    else:
        grid = list(control_grid) if control_grid is not None else _default_control_grid(correction.adjustment)
        best_value, best_scores, best_mean = None, None, np.inf
        for v in grid:
            cand = Correction(
                correction.condition.unconditional(),
                Adjustment(
                    correction.adjustment.channel,
                    correction.adjustment.op,
                    float(v),
                    correction.adjustment.target,
                ),
            )
            s = scored(primary, cand)
            if s.mean() < best_mean:
                best_value, best_scores, best_mean = float(v), s, float(s.mean())
        d, lo3, hi3 = bootstrap_ci(scored(primary, correction), best_scores)
        ok3 = hi3 < 0
        bars.append(
            BarResult(
                "3 beats unconditional",
                ok3,
                f"conditional - best unconditional (refit at {best_value:g}) {d:+.4f} [{lo3:+.4f}, {hi3:+.4f}]",
            )
        )

    # --- bar 4: holds in another regime -----------------------------------
    others = []
    for nm in names[1:]:
        cells = windows[nm]
        d, lo4, hi4 = bootstrap_ci(scored(cells, correction), baseline(cells))
        others.append((nm, d, lo4, hi4, len(cells)))
    ok4 = all(d < 0 for _, d, _, _, _ in others)
    detail = "; ".join(f"{nm} {d:+.4f} [{lo:+.4f}, {hi:+.4f}] (n={n})" for nm, d, lo, hi, n in others)
    bars.append(BarResult("4 holds in another regime", ok4, detail))

    return GateVerdict(correction, bars)
