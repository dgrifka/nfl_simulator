"""The receiver-drop variant — what a catchable ball was worth, and how likely
the offence was to catch it.

This is the read side of `docs/research/56-receiver-drop-mirror-prereg.md`, the
other direction of amendment A-3's hands-on-the-ball class. It sits beside the
v1.3 ledger and never inside it, on the same terms
`nfl_simulator.dropped_picks` does: FTN charting starts in 2022, and everything
here is reached only when a caller passes a fitted model in.

Two objects, and they answer different questions:

* :class:`DropSwingTable` — *what were the two branches worth?* Unlike the
  dropped-pick table, which prices every throw from its bin, this one prices
  **each play from its own completion counterfactual** — nflfastR's
  ``air_epa + xyac_epa``, the EPA the play would have carried had the ball been
  caught — against the incompletion. The six-cell bin table is the fallback for
  a play whose counterfactual is missing, and it also supplies the incompletion
  side for a ball that *was* caught and so has no realised incompletion to use.
* :class:`ReceiverDropModel` — *how likely was the catch?* The posterior of
  document 56's arm 2, read one probability per draw. The model's ``p`` is the
  probability of a **drop**; the ledger books its complement.

**Two things this module is deliberate about, both from document 56 §2.**

*The charged entity is the team-season, not the receiver.* Document 56 §1's
clause-1 rule chose it: Gate C-3's power at the receiver-season grain came back
at 0.40 against a 0.80 bar, and at the team-season grain at 0.88. The rule's
second branch says the component charges the receiving corps and says so, which
is what ``entity_season`` names.

*The defence-season effect is excluded on read.* It is fitted, because the
offensive term would otherwise carry the receiver's schedule, and it is never
paid: how well a defence covers is the defence's football and stays in ``core``.
:attr:`ReceiverDropModel.defence_effects` is loaded so a reader can report it,
and :meth:`ReceiverDropModel.drop_probability` does not touch it. A test pins
that.

The swing is **positive** by construction — a catch is worth more than an
incompletion — which is the opposite sign from the dropped-pick table, where an
interception costs the offence EPA. A non-positive swing would mean a drop was
worth more than a catch, which is a data fault rather than a finding, and
:func:`build_drop_swing_table` refuses to return one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from nfl_simulator.ingest import FTN_SEASONS

# Document 47 §3's bins, shared with the dropped-pick table so a reader compares
# the two ledgers on one set of cells. `yardline_100` is distance to the
# opponent's goal line.
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
    """``"34-66|3-4"`` for a readable pre-throw state, ``None`` otherwise."""
    yard, down_label = _yardline_bin(yardline_100), _down_bin(down)
    if yard is None or down_label is None:
        return None
    return f"{yard}|{down_label}"


def event_class_for(yardline_100: float | None, down: float | None) -> str:
    """The ledger's ``event_class`` for a target — ``"34-66 yd, late down"``."""
    key = cell_key(yardline_100, down)
    if key is None:
        return POOLED_EVENT_CLASS
    yard, down_label = key.split("|")
    return f"{yard} yd, {DOWN_WORDS[down_label]}"


@dataclass(frozen=True)
class DropSwingTable:
    """The bin table, and the per-play pricing that usually replaces it.

    ``cells`` maps each of the six keys to a caught-minus-dropped EPA swing;
    ``incompletion_mean`` maps the same keys to that cell's mean realised EPA on
    the dropped branch, which is what a *caught* ball is priced against, since a
    catch has no realised incompletion of its own. ``counts`` keeps the branch
    counts and which source each cell used, so a pooled cell reads as pooled
    rather than hiding behind a number.
    """

    cells: dict[str, float]
    incompletion_mean: dict[str, float]
    counts: dict[str, dict]
    pooled: float
    pooled_incompletion_mean: float
    pooled_completion_mean: float

    def swing_for(self, yardline_100: float | None, down: float | None) -> float:
        """The caught-minus-dropped EPA for this pre-throw state (positive)."""
        key = cell_key(yardline_100, down)
        if key is None:
            return self.pooled
        return self.cells.get(key, self.pooled)

    def incompletion_mean_for(self, yardline_100: float | None, down: float | None) -> float:
        """What an incompletion was worth in this cell, on average."""
        key = cell_key(yardline_100, down)
        if key is None:
            return self.pooled_incompletion_mean
        return self.incompletion_mean.get(key, self.pooled_incompletion_mean)

    def swing_for_play(self, row: dict) -> float:
        """Document 56 §2's per-play swing, with its pre-registered fallback.

        ``|(air_epa + xyac_epa) − epa_incomplete|``, where ``epa_incomplete`` is
        the play's own realised EPA when it was dropped and the cell's
        dropped-branch mean when it was caught.

        **The fallback clause is read one step wider than document 56 §2 wrote
        it, and the direction is the safe one.** §2 says "both ``air_epa`` and
        ``xyac_epa`` null -> bin table". The completion counterfactual is a sum,
        so half of it is not a value: the bin table is taken whenever *either*
        term is missing. That uses the pre-registered fallback more often than
        the literal clause and never less, and it never invents a zero for a
        quantity nflfastR declined to supply. `research/72` reports how often
        each case fires.
        """
        air, xyac = row.get("air_epa"), row.get("xyac_epa")
        yardline, down = row.get("yardline_100"), row.get("down")
        if air is None or xyac is None:
            return self.swing_for(yardline, down)
        if cell_key(yardline, down) is None:
            # No readable pre-throw state, so no cell mean to stand in for the
            # incompletion a caught ball never had. The pooled swing is document
            # 47 §3's own answer for a cell it cannot read.
            return self.pooled
        completion = float(air) + float(xyac)
        incomplete = (
            float(row["epa"]) if row.get("is_drop") else self.incompletion_mean_for(yardline, down)
        )
        return abs(completion - incomplete)

    def to_dict(self) -> dict:
        return {
            "cells": self.cells,
            "incompletion_mean": self.incompletion_mean,
            "counts": self.counts,
            "pooled": self.pooled,
            "pooled_incompletion_mean": self.pooled_incompletion_mean,
            "pooled_completion_mean": self.pooled_completion_mean,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> DropSwingTable:
        return cls(
            cells={str(k): float(v) for k, v in payload["cells"].items()},
            incompletion_mean={str(k): float(v) for k, v in payload["incompletion_mean"].items()},
            counts=dict(payload["counts"]),
            pooled=float(payload["pooled"]),
            pooled_incompletion_mean=float(payload["pooled_incompletion_mean"]),
            pooled_completion_mean=float(payload["pooled_completion_mean"]),
        )


def with_bins(frame: pl.DataFrame) -> pl.DataFrame:
    """Add the ``swing_cell`` key column — Polars-native, so no per-row Python."""
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


def build_drop_swing_table(catchable: pl.DataFrame) -> DropSwingTable:
    """Price the two branches per bin, on charted catchable targets carrying ``epa``.

    ``catchable`` needs four columns — ``yardline_100``, ``down``, ``is_drop``
    and ``epa`` — and is the same frame document 56 §1 counts: every charted
    catchable target 2022-2025. A cell with fewer than ``MIN_PER_BRANCH`` targets
    on either branch cannot carry its own difference and takes the pooled one.
    """
    caught = catchable.filter(~pl.col("is_drop"))
    dropped = catchable.filter(pl.col("is_drop"))
    if not caught.height or not dropped.height:
        raise ValueError("cannot price a swing table with an empty branch")
    pooled_caught = float(caught["epa"].mean())
    pooled_dropped = float(dropped["epa"].mean())
    pooled = pooled_caught - pooled_dropped

    binned = with_bins(catchable)
    cells: dict[str, float] = {}
    incompletion: dict[str, float] = {}
    counts: dict[str, dict] = {}
    for _low, _high, yard in YARDLINE_BINS:
        for _downs, down_label in DOWN_BINS:
            key = f"{yard}|{down_label}"
            cell = binned.filter(pl.col("swing_cell") == key)
            cell_caught = cell.filter(~pl.col("is_drop"))
            cell_dropped = cell.filter(pl.col("is_drop"))
            n_caught, n_dropped = cell_caught.height, cell_dropped.height
            thin = n_caught < MIN_PER_BRANCH or n_dropped < MIN_PER_BRANCH

            mean_caught = float(cell_caught["epa"].mean()) if n_caught else None
            mean_dropped = float(cell_dropped["epa"].mean()) if n_dropped else None
            cells[key] = float(pooled if thin else mean_caught - mean_dropped)
            incompletion[key] = float(pooled_dropped if thin else mean_dropped)
            counts[key] = {
                "n_caught": n_caught,
                "n_dropped": n_dropped,
                "mean_epa_caught": mean_caught,
                "mean_epa_dropped": mean_dropped,
                "source": "pooled" if thin else "cell",
            }

    table = DropSwingTable(
        cells=cells,
        incompletion_mean=incompletion,
        counts=counts,
        pooled=pooled,
        pooled_incompletion_mean=pooled_dropped,
        pooled_completion_mean=pooled_caught,
    )
    if pooled <= 0 or any(value <= 0 for value in cells.values()):
        raise ValueError(
            "a swing cell is not positive — a catch would be worth no more than an "
            "incompletion, which is a data fault rather than a finding"
        )
    return table


# --------------------------------------------------------------------------
# the adjudication frame
# --------------------------------------------------------------------------

FIRST_CHARTED_SEASON = min(FTN_SEASONS)

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
    "is_catchable_ball",
    "is_drop",
    "is_contested_ball",
    "is_qb_out_of_pocket",
    "is_play_action",
    "is_screen_pass",
    "n_pass_rushers",
)
# The two counterfactual columns document 56 §2 prices a swing from, plus the
# realised EPA an incompletion carries. Post-branch by rule, so they never enter
# the design matrix — but the ledger cannot price a branch without them.
PBP_SWING_COLUMNS: tuple[str, ...] = ("epa", "air_epa", "xyac_epa")


def catchable_target_frame(plays: pl.DataFrame, ftn: pl.DataFrame) -> pl.DataFrame:
    """Every charted catchable target in ``plays``, ready to price.

    The join is FTN's, on ``game_id`` and ``play_id``, and it is an inner join:
    a play with no charting row is not a ball the charter judged uncatchable, it
    is a play nobody charted.

    ``play_type == "pass"`` is filtered here and **not** in the study frame, and
    the difference is deliberate. Document 56 §4's guard counts 56,211 catchable
    targets over every charted play, 1,865 of which are penalty-nullified
    ``no_play`` rows; a fit may take all the evidence there is about how often a
    ball is dropped, but an *adjudication* may only neutralize branches that
    moved the scoreboard, and a play wiped by a penalty did not. In practice the
    two frames agree anyway — every one of those rows has a null ``air_yards``
    and leaves the fit's complete-case step as well.

    **Null covariates stay null and are flagged, not dropped** (document 48 §6's
    rule, as `dropped_picks.worthy_throw_frame` applies it): a game's ledger
    cannot silently omit a catchable ball because a charter left a field blank.
    """
    charted = ftn.select(
        pl.col("nflverse_game_id").alias("game_id"),
        pl.col("nflverse_play_id").cast(pl.Float64).alias("play_id"),
        *[column for column in FTN_COVARIATE_COLUMNS if column in ftn.columns],
    )
    targets = (
        plays.filter(pl.col("play_type") == "pass")
        .join(charted, on=["game_id", "play_id"], how="inner")
        .filter(pl.col("is_catchable_ball"))
    )
    imputed = [
        pl.col(column).is_null() for column in PBP_COVARIATE_COLUMNS if column in targets.columns
    ]
    return targets.with_columns(
        pl.concat_str([pl.col("season").cast(pl.String), pl.col("posteam")], separator="|").alias(
            "entity_season"
        ),
        pl.concat_str([pl.col("season").cast(pl.String), pl.col("defteam")], separator="|").alias(
            "defence_season"
        ),
        (pl.any_horizontal(imputed) if imputed else pl.lit(False)).alias("covariates_imputed"),
    )


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReceiverDropModel:
    """Posterior draws of the drop-probability surface, per charged entity-season.

    Mirrors :class:`~nfl_simulator.dropped_picks.DroppedPickModel`: one
    probability per posterior draw, never a point estimate, because document 05
    §4's layer 1 requires the deserve-to-win interval to carry the uncertainty in
    ``p`` itself — and amendment A-3 clause 2 makes that a condition of the class
    rather than a preference.

    ``standardisation`` and ``reference_levels`` are **stored at fit time and
    read, never recomputed** (round 3's fourth surprise).

    ``defence_effects`` is loaded and never read. See the module docstring: the
    coverage's contribution to a drop is the defence's football.
    """

    alpha: np.ndarray
    beta: np.ndarray  # (draws, covariates)
    entity_effects: dict[str, np.ndarray]
    covariate_order: tuple[str, ...]
    standardisation: dict[str, dict[str, float]]
    reference_levels: dict[str, object]
    swing_table: DropSwingTable
    defence_effects: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.beta.ndim != 2:
            raise ValueError(f"beta must be (draws, covariates), got shape {self.beta.shape}")
        if self.beta.shape != (len(self.alpha), len(self.covariate_order)):
            raise ValueError(
                f"beta is {self.beta.shape} but alpha has {len(self.alpha)} draws and "
                f"the covariate order has {len(self.covariate_order)} names"
            )
        for name, effect in self.entity_effects.items():
            if len(effect) != len(self.alpha):
                raise ValueError(
                    f"entity effect for {name} has {len(effect)} draws, expected {len(self.alpha)}"
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
            # nothing — the same "no information" an omitted dummy level encodes.
            return 0.0
        return (float(value) - constants["mean"]) / constants["sd"]

    def design_row(self, row: dict) -> np.ndarray:
        """The covariate vector for one target, in the fit's own column order."""
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

    def drop_probability(self, entity_season: str | None, row: dict) -> np.ndarray:
        """P(this catchable ball is dropped), one value per posterior draw.

        The defence-season term is deliberately absent — it is fitted so this
        one is estimated free of schedule, and document 56 §2 keeps how well a
        defence covers with the defence.
        """
        logit = self.alpha + self.beta @ self.design_row(row)
        effect = self.entity_effects.get(entity_season) if entity_season else None
        if effect is not None:
            logit = logit + effect
        return _sigmoid(logit)

    def catch_probability(self, entity_season: str | None, row: dict) -> np.ndarray:
        """P(this catchable ball is caught) — what the ledger books as ``expected``."""
        return 1.0 - self.drop_probability(entity_season, row)

    def swing_for_play(self, row: dict) -> float:
        """What the catch was worth against the incompletion, on this play."""
        return self.swing_table.swing_for_play(row)

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------

    @classmethod
    def from_posterior(cls, trace_path: str | Path, summary_path: str | Path) -> ReceiverDropModel:
        """Load the pair `research/72_receiver_drop_confounds.py` writes.

        Both files are needed and neither substitutes for the other: the trace
        carries the posterior, the summary carries the constants the posterior
        was fitted under.
        """
        import arviz as az

        trace_path, summary_path = Path(trace_path), Path(summary_path)
        for path in (trace_path, summary_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"no fitted receiver-drop artifact at {path} — "
                    "run `uv run python research/72_receiver_drop_confounds.py`"
                )

        summary = json.loads(summary_path.read_text())
        posterior = az.from_netcdf(trace_path)["posterior"]
        alpha = posterior["alpha"].values.ravel()
        order = tuple(summary["covariate_order"])
        beta = posterior["beta"].values.reshape(len(alpha), len(order))

        def _effects(variable: str, coord: str) -> dict[str, np.ndarray]:
            values = posterior[variable]
            levels = [str(level) for level in values.coords[coord].values]
            draws = values.values.reshape(len(alpha), len(levels))
            return {level: draws[:, index] for index, level in enumerate(levels)}

        return cls(
            alpha=alpha,
            beta=beta,
            entity_effects=_effects("r_s", "entity_season"),
            covariate_order=order,
            standardisation=summary["standardisation"],
            reference_levels=summary["reference_levels"],
            swing_table=DropSwingTable.from_dict(summary["swing_table"]),
            defence_effects=_effects("d_d", "defence_season"),
        )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Clipped for the same reason the field-goal model clips: the simulator must
    # never book infinite luck, and a probability of exactly 0 or 1 would make a
    # `LedgerEntry` claim certainty the posterior does not have.
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))
