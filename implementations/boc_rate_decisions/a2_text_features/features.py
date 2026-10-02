"""Turn extracted statement stances into leak-safe features at a forecast origin.

The feature row at origin *O* describes the most recent statement the Bank had
published by *O* — at a 28-day lead that is the previous meeting's statement,
roughly six weeks old. Three numbers reach the model:

``stance_hawk_dove``
    The stance score of that statement, -2..+2.
``stance_guidance``
    Its forward-guidance level on the same scale.
``stance_delta``
    Stance minus the stance of the statement before it. A level says where the
    Bank stands; the change says which way it is moving, which is what a
    statement can carry that a yield curve reads only slowly.

Deliberately *not* included: the driver flags. Seven dummies on ~130 training
rows would buy variance, not signal. They stay in the table for auditing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TEXT_FEATURE_NAMES = ["stance_hawk_dove", "stance_guidance", "stance_delta"]
"""Feature columns produced by :meth:`StanceTable.feature_row`, in order."""

DEFAULT_STANCE_PATH = Path(__file__).resolve().parent / "data" / "statement_stance.csv"


class StanceTable:
    """Cutoff-aware view over the extracted statement stances.

    Parameters
    ----------
    frame : pd.DataFrame
        Rows from ``statement_stance.csv``; needs ``statement_date``,
        ``hawk_dove`` and ``guidance_score``.
    shuffle_seed : int or None
        When set, the stance *values* are permuted across statement dates with
        this seed. This is the control arm: same features, same parameter
        count, same fitting procedure, but each meeting is handed some other
        meeting's reading. A gain that survives the shuffle was never about
        what the statements said.
    """

    def __init__(self, frame: pd.DataFrame, *, shuffle_seed: int | None = None) -> None:
        required = {"statement_date", "hawk_dove", "guidance_score"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Stance table is missing columns: {sorted(missing)}")

        table = frame.copy()
        table["statement_date"] = pd.to_datetime(table["statement_date"]).dt.normalize()
        table = table.sort_values("statement_date").reset_index(drop=True)

        if shuffle_seed is not None:
            rng = np.random.default_rng(shuffle_seed)
            order = rng.permutation(len(table))
            for column in ("hawk_dove", "guidance_score"):
                table[column] = table[column].to_numpy()[order]

        self._table = table

    @classmethod
    def from_csv(cls, path: Path | None = None, *, shuffle_seed: int | None = None) -> StanceTable:
        """Load the committed stance table.

        Raises
        ------
        FileNotFoundError
            If the table has not been extracted yet.
        """
        stance_path = path if path is not None else DEFAULT_STANCE_PATH
        if not stance_path.exists():
            raise FileNotFoundError(
                f"Stance table not found at {stance_path}. Build it with "
                "`python -m boc_rate_decisions.a2_text_features.extract_stance_table`."
            )
        return cls(pd.read_csv(stance_path), shuffle_seed=shuffle_seed)

    def __len__(self) -> int:
        """Return the number of statements in the table."""
        return len(self._table)

    def as_frame(self) -> pd.DataFrame:
        """Return a copy of the underlying stance rows, sorted by date.

        Useful for diagnostics (how much do the features actually move inside
        the eval window?) and for checking that a shuffled table is a genuine
        permutation of the real one.
        """
        return self._table.copy()

    def feature_row(self, origin: pd.Timestamp) -> dict[str, float] | None:
        """Return the text features visible at ``origin``, or ``None``.

        ``None`` means the origin has fewer than two statements behind it, so
        ``stance_delta`` cannot be formed — early-history origins only.
        Statements dated on ``origin`` itself count as visible: they are
        published at 09:45 ET, before any forecast issued that day.
        """
        visible = self._table[self._table["statement_date"] <= pd.Timestamp(origin).normalize()]
        if len(visible) < 2:  # noqa: PLR2004 - a delta needs two statements
            return None
        latest = visible.iloc[-1]
        previous = visible.iloc[-2]
        return {
            "stance_hawk_dove": float(latest["hawk_dove"]),
            "stance_guidance": float(latest["guidance_score"]),
            "stance_delta": float(latest["hawk_dove"]) - float(previous["hawk_dove"]),
        }


__all__ = ["DEFAULT_STANCE_PATH", "TEXT_FEATURE_NAMES", "StanceTable"]
