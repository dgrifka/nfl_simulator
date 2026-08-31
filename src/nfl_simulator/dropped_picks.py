"""The dropped-pick variant — what an interceptable throw was worth, and how
likely the defence was to catch it.

This is the read side of `docs/research/49-dropped-pick-variant-prereg.md`, the
**labelled variant** of the adjudication that neutralizes a charted
interception-worthy throw at the throwing defence's posterior-sampled catch
probability. It sits beside the Strict ledger and never inside it: document 32's
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

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from nfl_simulator.ingest import FTN_SEASONS

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


# --------------------------------------------------------------------------
# the adjudication frame
# --------------------------------------------------------------------------

# FTN charting starts in 2022, so 2016-2021 could never carry this component.
# Derived from the ingest layer rather than written as a literal, so a widened
# charting pull cannot leave a stale constant behind.
FIRST_CHARTED_SEASON = min(FTN_SEASONS)

# The pre-throw covariates the conversion model reads, from `research/61`'s
# document 43 §4 list. A caller assembling a frame for the variant needs these
# columns on the play-by-play; `simulate_game`'s ordinary column set does not
# carry them, which is why the audit script asks for them explicitly.
PBP_COVARIATE_COLUMNS: tuple[str, ...] = (
    "air_yards",
    "pass_location",
    "qb_hit",
    "down",
    "ydstogo",
    "yardline_100",
    "shotgun",
    "wp",
)
FTN_COVARIATE_COLUMNS: tuple[str, ...] = (
    "is_interception_worthy",
    "is_catchable_ball",
    "is_contested_ball",
    "is_qb_out_of_pocket",
    "is_play_action",
    "is_screen_pass",
    "n_pass_rushers",
)


def worthy_throw_frame(plays: pl.DataFrame, ftn: pl.DataFrame) -> pl.DataFrame:
    """Every charted interception-worthy throw in ``plays``, ready to price.

    The join is FTN's, on ``game_id`` and ``play_id``, and it is an inner join:
    a play with no charting row is not a throw the charter declined to call
    worthy, it is a play nobody charted, and the two are different facts.

    **Null covariates stay null and are flagged, not dropped.** Document 49 §4
    keeps them in the adjudication frame — a game's ledger cannot silently omit
    an interceptable throw because a charter left `air_yards` blank — and
    :meth:`DroppedPickModel.design_row` prices each null at its own reference
    level. ``covariates_imputed`` is how a reader tells the two apart. This is
    the opposite of the *fit* frame, which drops those 28 rows: a fit may
    restrict itself to complete cases, an adjudication of a real game may not.
    """
    charted = ftn.select(
        pl.col("nflverse_game_id").alias("game_id"),
        pl.col("nflverse_play_id").cast(pl.Float64).alias("play_id"),
        *[column for column in FTN_COVARIATE_COLUMNS if column in ftn.columns],
    )
    worthy = (
        plays.filter(pl.col("play_type") == "pass")
        .join(charted, on=["game_id", "play_id"], how="inner")
        .filter(pl.col("is_interception_worthy"))
    )
    imputed = [
        pl.col(column).is_null() for column in PBP_COVARIATE_COLUMNS if column in worthy.columns
    ]
    return worthy.with_columns(
        pl.concat_str([pl.col("season").cast(pl.String), pl.col("defteam")], separator="|").alias(
            "defence_season"
        ),
        (pl.any_horizontal(imputed) if imputed else pl.lit(False)).alias("covariates_imputed"),
    )


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DroppedPickModel:
    """Posterior draws of the catch-probability surface, per defence-season.

    Mirrors :class:`~nfl_simulator.fg_model.FieldGoalModel`: one probability per
    posterior draw, never a point estimate, because document 05 §4's layer 1
    requires the deserve-to-win interval to carry the uncertainty in ``p``
    itself. A model that collapsed to a mean here would quietly make the
    variant's intervals as tight as if the defence's hands were known.

    ``standardisation`` and ``reference_levels`` are **stored at fit time and
    read, never recomputed**. Round 3's fourth surprise was a standardisation
    recomputed on a frame that was 28 rows larger than the fitted one; the
    constants are properties of the fitted sample, so a caller scoring a game
    has to use the ones the fit used.

    A defence-season absent from the fit falls back to ``u_d = 0`` — the league
    surface. Under document 05 §1's rule that is the ``w = 0`` endpoint: no
    evidence about this entity, so no entity term. It cannot happen inside
    2022-2025, and it is guarded rather than assumed away.
    """

    alpha: np.ndarray
    beta: np.ndarray  # (draws, covariates)
    defence_effects: dict[str, np.ndarray]
    covariate_order: tuple[str, ...]
    standardisation: dict[str, dict[str, float]]
    reference_levels: dict[str, object]
    swing_table: SwingTable

    def __post_init__(self) -> None:
        if self.beta.ndim != 2:
            raise ValueError(f"beta must be (draws, covariates), got shape {self.beta.shape}")
        if self.beta.shape != (len(self.alpha), len(self.covariate_order)):
            raise ValueError(
                f"beta is {self.beta.shape} but alpha has {len(self.alpha)} draws and "
                f"the covariate order has {len(self.covariate_order)} names"
            )
        for name, effect in self.defence_effects.items():
            if len(effect) != len(self.alpha):
                raise ValueError(
                    f"defence effect for {name} has {len(effect)} draws, expected {len(self.alpha)}"
                )

    @property
    def n_draws(self) -> int:
        return len(self.alpha)

    # ------------------------------------------------------------------
    # the design row
    # ------------------------------------------------------------------

    def _standardised(self, column: str, value: float | None) -> float:
        constants = self.standardisation[column]
        if value is None:
            # The fitted mean standardises to exactly 0, so a null contributes
            # nothing to the linear predictor — the same "no information" an
            # omitted dummy level encodes (document 48 §6's rule, generalised in
            # `research/65` to every null covariate rather than `pass_location`
            # alone).
            return 0.0
        return (float(value) - constants["mean"]) / constants["sd"]

    def design_row(self, row: dict) -> np.ndarray:
        """The covariate vector for one throw, in the fit's own column order.

        Driven by ``covariate_order`` rather than by a hardcoded list, so a
        summary written by a different fit cannot be read into the wrong slots:
        a name this method does not recognise is an error, never a zero.
        """
        values = np.empty(len(self.covariate_order))
        for index, name in enumerate(self.covariate_order):
            values[index] = self._covariate(name, row)
        return values

    def _covariate(self, name: str, row: dict) -> float:
        if name == "air_yards_z_squared":
            return self._standardised("air_yards", row.get("air_yards")) ** 2
        if name.endswith("_z") and name[:-2] in self.standardisation:
            column = name[:-2]
            return self._standardised(column, row.get(column))
        if name.startswith("pass_location_"):
            level = row.get("pass_location") or self.reference_levels["pass_location"]
            return float(level == name.removeprefix("pass_location_"))
        if name.startswith("down_"):
            down = row.get("down")
            if down is None:
                down = self.reference_levels["down"]
            return float(float(down) == float(name.removeprefix("down_")))
        if name in row:
            value = row.get(name)
            return 0.0 if value is None else float(value)
        raise ValueError(
            f"`{name}` is not a covariate this model knows how to read — the "
            "summary's covariate order does not match this frame"
        )

    def covariates_are_complete(self, row: dict) -> bool:
        """Whether every covariate this model reads was actually recorded."""
        for column in self.standardisation:
            if row.get(column) is None:
                return False
        for name in self.covariate_order:
            if name.startswith(("pass_location_", "down_")):
                key = "pass_location" if name.startswith("pass_location_") else "down"
                if row.get(key) is None:
                    return False
            elif not name.endswith("_z") and name != "air_yards_z_squared":
                if row.get(name) is None:
                    return False
        return True

    # ------------------------------------------------------------------
    # the answer
    # ------------------------------------------------------------------

    def catch_probability(self, defence_season: str | None, row: dict) -> np.ndarray:
        """P(the defence catches this throw), one value per posterior draw.

        The QB-season term is deliberately absent: document 49 §2 keeps a
        passer's own droppability with the offence, so ``v_q`` is fitted and
        never read here.
        """
        logit = self.alpha + self.beta @ self.design_row(row)
        effect = self.defence_effects.get(defence_season) if defence_season else None
        if effect is not None:
            logit = logit + effect
        return _sigmoid(logit)

    def swing_for(self, yardline_100: float | None, down: float | None) -> float:
        """The picked-minus-escaped EPA for this pre-throw state (negative)."""
        return self.swing_table.swing_for(yardline_100, down)

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------

    @classmethod
    def from_posterior(cls, trace_path: str | Path, summary_path: str | Path) -> DroppedPickModel:
        """Load the pair `research/67_dropped_pick_model.py` writes.

        Both files are needed and neither substitutes for the other: the trace
        carries the posterior, the summary carries the constants the posterior
        was fitted under. Loading the trace alone and recomputing the rest is the
        defect document 30 corrected on the field-goal model, in a new place.
        """
        import arviz as az

        trace_path, summary_path = Path(trace_path), Path(summary_path)
        for path in (trace_path, summary_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"no fitted dropped-pick artifact at {path} — "
                    "run `uv run python research/67_dropped_pick_model.py`"
                )

        summary = json.loads(summary_path.read_text())
        posterior = az.from_netcdf(trace_path)["posterior"]
        alpha = posterior["alpha"].values.ravel()
        order = tuple(summary["covariate_order"])
        beta = posterior["beta"].values.reshape(len(alpha), len(order))

        effects = posterior["u_d"]
        levels = [str(level) for level in effects.coords["defence_season"].values]
        draws = effects.values.reshape(len(alpha), len(levels))
        return cls(
            alpha=alpha,
            beta=beta,
            defence_effects={level: draws[:, i] for i, level in enumerate(levels)},
            covariate_order=order,
            standardisation=summary["standardisation"],
            reference_levels=summary["reference_levels"],
            swing_table=SwingTable.from_dict(summary["swing_table"]),
        )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Clipped for the same reason the field-goal model clips: the simulator must
    # never book infinite luck, and a probability of exactly 0 or 1 would make a
    # `LedgerEntry` claim certainty the posterior does not have.
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))
