"""The dropped-pick variant — what an interceptable throw was worth, and how
likely the defence was to catch it.

This is the read side of `docs/research/49-dropped-pick-variant-prereg.md`, the
**labelled variant** of the adjudication that neutralizes a charted
interception-worthy throw at the throwing defence's posterior-sampled catch
probability. It sits beside the v1.3 ledger and never inside it: document 32's
closure stands, document 28's consistency argument stands, and FTN charting
starts in 2022 so 2016-2021 could never carry the component at all. Everything
here is reached only when a caller passes a fitted model in, and
`simulate_game`'s default is still `None`.

Two objects, and they answer different questions:

* :class:`SwingTable` — *what were the two branches worth?* Document 47 §3's
  six-cell bin table, keyed on pre-throw state (``yardline_100`` in thirds x
  ``down`` in {1-2, 3-4}), each cell the mean EPA of picked worthy throws minus
  the mean EPA of escaped ones. It is **built here, from the same data document
  47 §3 used**, and stored in the fit's summary so read time never recomputes
  it.
* :class:`DroppedPickModel` — *how likely was the catch?* The posterior of the
  round-2 conversion model, read one probability per draw exactly as
  :class:`~nfl_simulator.fg_model.FieldGoalModel` reads a make probability. The
  QB-season term is fitted and deliberately **not** read: a quarterback's own
  droppability is the offence's, and crediting it here would pay a passer for
  throwing catchable interceptions (document 49 §2).

The swing is negative by construction — an interception costs the offence EPA —
and the ledger's sign convention wants the *good* branch's value, so the event
builder takes its absolute value. A positive cell would mean a pick was worth
more than escaping it, which is a data fault rather than a finding, and
:func:`build_swing_table` records the sign so a caller can stop on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

# Document 47 §3's bins. `yardline_100` is distance to the opponent's goal line,
# so 1-33 is the offence deep in scoring range and 67-99 is backed up near its
# own end.
YARDLINE_BINS: tuple[tuple[int, int, str], ...] = (
    (1, 33, "1-33"),
    (34, 66, "34-66"),
    (67, 99, "67-99"),
)
DOWN_BINS: tuple[tuple[tuple[float, ...], str], ...] = (
    ((1.0, 2.0), "1-2"),
    ((3.0, 4.0), "3-4"),
)
MIN_PER_BRANCH = 30

# What a bin label is called on a ledger row and in a figure's event column.
DOWN_WORDS = {"1-2": "early down", "3-4": "late down"}
POOLED_EVENT_CLASS = "pooled swing"


def _yardline_bin(yardline_100: float | None) -> str | None:
    if yardline_100 is None:
        return None
    for low, high, label in YARDLINE_BINS:
        if low <= yardline_100 <= high:
            return label
    return None


def _down_bin(down: float | None) -> str | None:
    if down is None:
        return None
    for downs, label in DOWN_BINS:
        if float(down) in downs:
            return label
    return None


def cell_key(yardline_100: float | None, down: float | None) -> str | None:
    """``"34-66|3-4"`` for a readable pre-throw state, ``None`` otherwise.

    A throw whose ``down`` or ``yardline_100`` was never recorded has an unknown
    pre-throw state, and pricing it in the first-down cell would invent one. It
    gets no key, and the caller falls back to the pooled swing — which is the
    same mechanism document 47 §3 already gives a cell it cannot read.
    """
    yard, down_label = _yardline_bin(yardline_100), _down_bin(down)
    if yard is None or down_label is None:
        return None
    return f"{yard}|{down_label}"


def event_class_for(yardline_100: float | None, down: float | None) -> str:
    """The ledger's ``event_class`` for a throw — ``"34-66 yd, late down"``."""
    key = cell_key(yardline_100, down)
    if key is None:
        return POOLED_EVENT_CLASS
    yard, down_label = key.split("|")
    return f"{yard} yd, {DOWN_WORDS[down_label]}"


@dataclass(frozen=True)
class SwingTable:
    """Document 47 §3's bin table, and the pooled fallback beside it.

    ``cells`` maps every one of the six keys to a swing in EPA; ``counts`` keeps
    the branch counts and which of the two sources the cell used, so the record
    shows a pooled cell as pooled rather than hiding it behind a number.
    """

    cells: dict[str, float]
    counts: dict[str, dict]
    pooled: float

    def swing_for(self, yardline_100: float | None, down: float | None) -> float:
        """The picked-minus-escaped EPA for this pre-throw state (negative)."""
        key = cell_key(yardline_100, down)
        if key is None:
            return self.pooled
        return self.cells.get(key, self.pooled)

    def to_dict(self) -> dict:
        return {"cells": self.cells, "counts": self.counts, "pooled": self.pooled}

    @classmethod
    def from_dict(cls, payload: dict) -> SwingTable:
        return cls(
            cells={str(k): float(v) for k, v in payload["cells"].items()},
            counts=dict(payload["counts"]),
            pooled=float(payload["pooled"]),
        )


def with_bins(frame: pl.DataFrame) -> pl.DataFrame:
    """Add the ``swing_cell`` key column — Polars-native, so no per-row Python.

    A row whose pre-throw state is unreadable gets a null key rather than a
    guessed cell, which is what sends it to the pooled swing downstream.
    """
    yard = pl.when(pl.col("yardline_100").is_null()).then(None)
    for low, high, label in YARDLINE_BINS:
        yard = yard.when((pl.col("yardline_100") >= low) & (pl.col("yardline_100") <= high)).then(
            pl.lit(label)
        )
    yard = yard.otherwise(None)

    down = pl.when(pl.col("down").is_null()).then(None)
    for downs, label in DOWN_BINS:
        down = down.when(pl.col("down").cast(pl.Float64).is_in(downs)).then(pl.lit(label))
    down = down.otherwise(None)

    return frame.with_columns(
        pl.when(yard.is_null() | down.is_null())
        .then(None)
        .otherwise(pl.concat_str([yard, down], separator="|"))
        .alias("swing_cell")
    )


def build_swing_table(worthy: pl.DataFrame) -> SwingTable:
    """Price the two branches per bin, on charted worthy throws carrying ``epa``.

    ``worthy`` needs four columns — ``yardline_100``, ``down``, ``interception``
    and ``epa`` — and is the same frame document 47 §3 priced: every charted
    interception-worthy throw 2022-2025. A cell with fewer than
    ``MIN_PER_BRANCH`` throws on either branch cannot carry its own difference
    and takes the pooled one instead.
    """
    picked = worthy.filter(pl.col("interception") == 1)
    escaped = worthy.filter(pl.col("interception") == 0)
    if not picked.height or not escaped.height:
        raise ValueError("cannot price a swing table with an empty branch")
    pooled = float(picked["epa"].mean()) - float(escaped["epa"].mean())

    binned = with_bins(worthy)
    cells: dict[str, float] = {}
    counts: dict[str, dict] = {}
    for _low, _high, yard in YARDLINE_BINS:
        for _downs, down_label in DOWN_BINS:
            key = f"{yard}|{down_label}"
            cell = binned.filter(pl.col("swing_cell") == key)
            cell_picked = cell.filter(pl.col("interception") == 1)
            cell_escaped = cell.filter(pl.col("interception") == 0)
            n_picked, n_escaped = cell_picked.height, cell_escaped.height
            thin = n_picked < MIN_PER_BRANCH or n_escaped < MIN_PER_BRANCH

            mean_picked = float(cell_picked["epa"].mean()) if n_picked else None
            mean_escaped = float(cell_escaped["epa"].mean()) if n_escaped else None
            swing = pooled if thin else mean_picked - mean_escaped

            cells[key] = float(swing)
            counts[key] = {
                "n_picked": n_picked,
                "n_escaped": n_escaped,
                "mean_epa_picked": mean_picked,
                "mean_epa_escaped": mean_escaped,
                "source": "pooled" if thin else "cell",
            }
    return SwingTable(cells=cells, counts=counts, pooled=pooled)
