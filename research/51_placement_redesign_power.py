"""Product layer, task 2 redesign — power for the thresholds document 36 commits.

Gate M-4 failed on 2026-08-19 (document 35 §11): the placement meter's
team-season mean correlated **+0.5435** with offensive quality against a bound of
0.1065. Document 34 §6 routes that failure back to design, and the maintainer chose
redesign avenue (a) on 2026-08-20 — baseline the count channel, not just the cell
means.

This script measures everything document 36's thresholds are set from, and it
runs **before** document 36 is written, exactly as ``49_placement_power.py`` ran
before document 35. Nothing gated is computed here: every number below is either
a league-pooled design parameter, an identity, or a simulated null. The gated
statistics belong to ``research/52_placement_redesign.py``, which does not yet
exist.

    uv run python research/51_placement_redesign_power.py --part design
    uv run python research/51_placement_redesign_power.py --part all

Four parts:

* ``design``  — the redesigned score's design parameters, the contamination
  bound, and the **M-2 carry-forward proof**: that the redesign shifts the
  realized score and every null draw by the same per-team-game constant, so the
  band, its PIT and rung 4's adopted calibration survive untouched.
* ``m3``      — the persistence null and its power at the redesigned dispersion.
* ``m4``      — the skill-preservation null, its power grid, and the two new
  full-pipeline arms that measure how much of an M-4 pass is mechanical.
* ``m6``      — the rematch harness's width at the redesigned differential SD.

Results accumulate in ``research/outputs/51_redesign_power.json``;
``research/outputs/`` is gitignored, so this script and document 36 are the
committed artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_power = import_module("49_placement_power")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.placement import permutation_draws  # noqa: E402

RANDOM_SEED = 20260820
RESULTS = "51_redesign_power.json"

POINTS_PER_EPA = _power.POINTS_PER_EPA
CELL_RED_ZONE, CELL_LATE_DOWN, CELL_OTHER = 0, 1, 2

# Document 35 §10's ladder constants, unchanged. Part `design` proves they may be
# carried forward rather than assuming it.
LADDER = ("raw", "raw_var_matched", "down_stratified", "down_stratified_var_matched")
RUNG_SCALES = {
    "raw": (1.0, 1.0, 1.0),
    "raw_var_matched": (1.0892, 1.4116, 0.8237),
    "down_stratified": (1.0, 1.0, 1.0),
    "down_stratified_var_matched": (1.0892, 1.0, 1.0),
}
CARRY_FORWARD_TEAM_GAMES = 40
CARRY_FORWARD_DRAWS = 2000

# Document 08's split-half machinery, inherited through document 35 §7.
N_SPLITS = _power.N_SPLITS
MIN_GAMES = _power.MIN_GAMES
REFERENCE_R = _power.REFERENCE_R
M3_TARGETS = (0.05, 0.08, 0.10, 0.12, 0.20)
M4_LEAKS = (0.05, 0.10, 0.11, 0.15, 0.20, 0.30)

# The full-pipeline leak arms of part `m4`. Cheap because the score reads only
# each team-game's cell counts and cell EPA sums, so a synthetic league is drawn
# at the cell-sum grain rather than play by play.
N_PIPELINE_NULL_REPLICATES = 600
N_PIPELINE_REPLICATES = 300


# --------------------------------------------------------------------------
# part 0 — the redesigned score
# --------------------------------------------------------------------------


def team_game_table(plays: pl.DataFrame) -> pl.DataFrame:
    """One row per team-game: cell counts, cell EPA sums, and leave-one-game-out S0.

    ``s0_loo`` is the team's luck-priced offensive EPA per play over the *rest of
    its season*. The game being scored never enters its own baseline, which is
    the document 05 §5 contamination defence applied at the input rather than
    bounded after the fact.
    """
    meta = _power.team_game_arrays(plays)["meta"]
    return meta.with_columns(
        (pl.col("epa_all").sum().over("team_season") - pl.col("epa_all")).alias("epa_rest"),
        (pl.col("n_all").sum().over("team_season") - pl.col("n_all")).alias("n_rest"),
    ).with_columns((pl.col("epa_rest") / pl.col("n_rest")).alias("s0_loo"))


def cell_matrices(table: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(n_all, N, E, s0)`` — counts and EPA sums as (rows, 3) matrices."""
    n_all = table["n_all"].to_numpy().astype(float)
    n_rz = table["n_rz"].to_numpy().astype(float)
    n_ld = table["n_ld"].to_numpy().astype(float)
    epa_all = table["epa_all"].to_numpy().astype(float)
    epa_rz = table["epa_rz"].to_numpy().astype(float)
    epa_ld = table["epa_ld"].to_numpy().astype(float)
    counts = np.column_stack([n_rz, n_ld, n_all - n_rz - n_ld])
    sums = np.column_stack([epa_rz, epa_ld, epa_all - epa_rz - epa_ld])
    return n_all, counts, sums, table["s0_loo"].to_numpy().astype(float)


def loto_linear_fit(
    y_mean: np.ndarray, weight: np.ndarray, x: np.ndarray, group: np.ndarray
) -> np.ndarray:
    """Weighted least squares of ``y_mean`` on ``[1, x]``, leave-one-**team**-out.

    Returns the fitted value for every row, from coefficients estimated without
    any row belonging to that row's franchise. Two things ride on that.

    * **Contamination.** The baseline a team is scored against never saw a single
      play that team ran, in any season — a strictly stronger defence than
      leaving the game out, which ``s0_loo`` already does.
    * **The gate.** Fitting in-sample would make ``corr(residual, x)`` exactly
      zero by the normal equations, and M-4 would be reading its own arithmetic.
      Out-of-fold residuals carry no such guarantee, and §7 of document 36
      measures what is left.

    The weight is the count the score multiplies that cell by, which is what
    makes the fit's orthogonality condition *be* the leak condition rather than
    merely resemble it.
    """
    stats = np.array(
        [
            weight.sum(),
            (weight * x).sum(),
            (weight * x * x).sum(),
            (weight * y_mean).sum(),
            (weight * y_mean * x).sum(),
        ]
    )
    fitted = np.empty(len(weight))
    for held_out in np.unique(group):
        rows = group == held_out
        w, xi, yi = weight[rows], x[rows], y_mean[rows]
        s_w, s_wx, s_wxx, s_y, s_yx = stats - np.array(
            [w.sum(), (w * xi).sum(), (w * xi * xi).sum(), (w * yi).sum(), (w * yi * xi).sum()]
        )
        slope = (s_w * s_yx - s_wx * s_y) / (s_w * s_wxx - s_wx * s_wx)
        fitted[rows] = (s_y - slope * s_wx) / s_w + slope * xi
    return fitted


def expected_profile(
    counts: np.ndarray, sums: np.ndarray, s0: np.ndarray, group: np.ndarray
) -> np.ndarray:
    """``mu[row, cell]`` — the EPA per play a team of this quality produces there.

    Three two-parameter fits, one per cell, each weighted by that cell's play
    count and each estimated leave-one-team-out. This is the whole redesign: the
    incumbent scored a cell against the team's own game-wide mean and therefore
    against a bar that carried the league's structural profile, so an
    ``n_cell``-scaled score inherited every bit of the count endogeneity
    document 35 §11 found.
    """
    mean_cell = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return np.column_stack(
        [loto_linear_fit(mean_cell[:, c], counts[:, c], s0, group) for c in range(3)]
    )


def expected_shares(
    counts: np.ndarray, n_all: np.ndarray, s0: np.ndarray, group: np.ndarray
) -> np.ndarray:
    """``q[row, cell]`` — the play mix this team's quality predicts.

    **Not used by the committed score.** Document 36 §5 derives why baselining
    the mix cannot remove the leak on its own: the leak is the product of a
    structural cell offset and a quality-correlated cell count, and replacing the
    realised count with its expectation leaves the second factor exactly as
    quality-correlated as it was. Measured here because the reported
    ``expected_mix`` arm uses it, and because the spread of ``q / p`` is the
    arithmetic that disqualifies that arm from being the shipped score.

    Fitting all three cells with the same regressor and the same weights makes
    the intercepts sum to one and the slopes to zero exactly, so ``q`` is a mix
    without being renormalised into one.
    """
    shares = counts / n_all[:, None]
    return np.column_stack([loto_linear_fit(shares[:, c], n_all, s0, group) for c in range(3)])


def expected_mix_cell_points(
    n_all: np.ndarray,
    counts: np.ndarray,
    sums: np.ndarray,
    mu: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    """The reported ``expected_mix`` arm: every play reweighted by ``q_c / p_c``.

    The literal reading of redesign avenue (a). Kept as an arm rather than the
    committed score because of what document 36 §5 measures: the reweight is
    unbounded as a cell empties, so a team-game with one red-zone snap has that
    snap priced as nine, and ``max |score|`` runs half again as large as the
    committed construction's.
    """
    weight = np.where(counts > 0, n_all[:, None] * q, 0.0)
    centred = np.where(
        counts > 0,
        weight * np.divide(sums - counts * mu, counts, out=np.zeros_like(sums), where=counts > 0),
        0.0,
    )
    baseline = centred.sum(axis=1) / weight.sum(axis=1)
    return (centred - weight * baseline[:, None]) * POINTS_PER_EPA


def profile_shift(counts: np.ndarray, mu: np.ndarray, n_all: np.ndarray) -> np.ndarray:
    """The per-team-game constant the redesign subtracts from the incumbent score.

    Document 36 §6's carry-forward proof in one line: with the three cell sizes
    held fixed — which every rung of the ladder does — the profile a draw
    subtracts is the same whichever plays land where, so

        redesigned_score = incumbent_score - C

    for both the realized score and every null draw. The PIT is a rank inside the
    draws, ranks are invariant to a shift, and M-2's coverage therefore carries
    forward exactly.
    """
    leverage = counts[:, CELL_RED_ZONE] + counts[:, CELL_LATE_DOWN]
    weighted_mean_mu = (counts * mu).sum(axis=1) / n_all
    in_leverage = counts[:, CELL_RED_ZONE] * mu[:, CELL_RED_ZONE] + (
        counts[:, CELL_LATE_DOWN] * mu[:, CELL_LATE_DOWN]
    )
    return (in_leverage - leverage * weighted_mean_mu) * POINTS_PER_EPA


def redesigned_cell_points(
    n_all: np.ndarray, counts: np.ndarray, sums: np.ndarray, mu: np.ndarray
) -> np.ndarray:
    """``points[row, cell]`` for the committed construction.

    Written, like the incumbent, as a weighted sum minus a weight times the
    weighted mean, so an empty cell is exactly ``0.0`` rather than ``0/0`` and
    the three cells sum to zero by arithmetic rather than by decree.
    """
    centred = np.where(counts > 0, sums - counts * mu, 0.0)
    baseline = centred.sum(axis=1) / n_all
    return (centred - counts * baseline[:, None]) * POINTS_PER_EPA


def score_frame(table: pl.DataFrame, group_column: str = "posteam") -> pl.DataFrame:
    """The redesigned per-team-game score, its cells, and the incumbent beside it."""
    n_all, counts, sums, s0 = cell_matrices(table)
    group = table[group_column].to_numpy()
    mu = expected_profile(counts, sums, s0, group)
    points = redesigned_cell_points(n_all, counts, sums, mu)
    incumbent = _power.placement_scores(table)
    return table.with_columns(
        pl.Series("mu_rz", mu[:, 0]),
        pl.Series("mu_ld", mu[:, 1]),
        pl.Series("mu_other", mu[:, 2]),
        pl.Series("red_zone", points[:, 0]),
        pl.Series("late_down", points[:, 1]),
        pl.Series("other", points[:, 2]),
        pl.Series("score", points[:, 0] + points[:, 1]),
        pl.Series("identity_residual", points.sum(axis=1)),
        pl.Series("incumbent_score", incumbent["score"]),
        pl.Series("profile_shift", profile_shift(counts, mu, n_all)),
    )


# --------------------------------------------------------------------------
# part 1 — design parameters, contamination, and the M-2 carry-forward proof
# --------------------------------------------------------------------------


def design_part(plays: pl.DataFrame, rng: np.random.Generator) -> dict:
    table = score_frame(team_game_table(plays))
    n_all, counts, sums, s0 = cell_matrices(table)
    group = table["posteam"].to_numpy()

    score = table["score"].to_numpy()
    incumbent = table["incumbent_score"].to_numpy()
    shift = table["profile_shift"].to_numpy()

    # --- the fitted profile, league-pooled ---------------------------------
    mean_cell = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    mu_all = np.column_stack(
        [table["mu_rz"].to_numpy(), table["mu_ld"].to_numpy(), table["mu_other"].to_numpy()]
    )
    profile = {}
    for c, name in enumerate(("red_zone", "late_down", "other")):
        fitted = loto_linear_fit(mean_cell[:, c], counts[:, c], s0, group)
        design = np.column_stack([np.ones_like(s0), s0])
        w = counts[:, c]
        coefficients = np.linalg.lstsq(
            design * np.sqrt(w)[:, None], mean_cell[:, c] * np.sqrt(w), rcond=None
        )[0]
        profile[name] = {
            "intercept_epa_per_play": float(coefficients[0]),
            "slope_on_quality": float(coefficients[1]),
            "fitted_sd": float(fitted.std(ddof=1)),
        }
    # The play-count-weighted mean slope is the regression of the team-game's own
    # EPA per play on `s0_loo`: the reliability of a 16-game quality estimate.
    weights = counts / n_all[:, None]
    profile["mean_slope_reliability"] = float(
        sum(
            weights[:, c].mean() * profile[n]["slope_on_quality"]
            for c, n in enumerate(("red_zone", "late_down", "other"))
        )
    )

    # --- identities --------------------------------------------------------
    empty_rz = counts[:, CELL_RED_ZONE] == 0
    identities = {
        "three_cells_sum_to_zero_worst": float(np.abs(table["identity_residual"].to_numpy()).max()),
        "n_empty_red_zone_cells": int(empty_rz.sum()),
        "empty_red_zone_worst_abs_points": float(
            np.abs(table["red_zone"].to_numpy()[empty_rz]).max() if empty_rz.any() else 0.0
        ),
        "redesign_is_incumbent_minus_shift_worst": float(np.abs((incumbent - shift) - score).max()),
    }

    # --- contamination bound ----------------------------------------------
    # The fit already excludes the franchise entirely, so the only surviving path
    # from a game into its own baseline is the *opponent's* rows of that game,
    # which are a league row like any other. Bound it by refitting without the
    # whole game for a sample of team-games and reading the largest score move.
    sample = rng.choice(len(score), size=40, replace=False)
    game_ids = table["game_id"].to_numpy()
    worst = 0.0
    for row in sample:
        keep = game_ids != game_ids[row]
        mu_row = np.array(
            [
                _refit_one(
                    mean_cell[keep, c], counts[keep, c], s0[keep], group[keep], s0[row], group[row]
                )
                for c in range(3)
            ]
        )
        alt = redesigned_cell_points(
            n_all[row : row + 1], counts[row : row + 1], sums[row : row + 1], mu_row[None, :]
        )
        worst = max(worst, abs(float(alt[0, 0] + alt[0, 1]) - float(score[row])))
    contamination = {
        "sampled_team_games": int(len(sample)),
        "worst_abs_score_move_points": float(worst),
        "opponent_rows_share_of_the_fit": float(1.0 / len(score)),
    }

    # --- the reported expected-mix arm, and why it is not the shipped score ---
    q = expected_shares(counts, n_all, s0, group)
    reweight = np.where(counts > 0, q / np.where(counts > 0, counts / n_all[:, None], 1.0), np.nan)
    mix_points = expected_mix_cell_points(n_all, counts, sums, mu_all, q)
    mix_score = mix_points[:, 0] + mix_points[:, 1]
    expected_mix = {
        "identity_worst": float(np.abs(mix_points.sum(axis=1)).max()),
        "score_mean": float(mix_score.mean()),
        "score_sd": float(mix_score.std(ddof=1)),
        "max_abs_score": float(np.abs(mix_score).max()),
        "reweight_red_zone_p90": float(np.nanquantile(reweight[:, CELL_RED_ZONE], 0.90)),
        "reweight_red_zone_p99": float(np.nanquantile(reweight[:, CELL_RED_ZONE], 0.99)),
        "reweight_red_zone_max": float(np.nanmax(reweight[:, CELL_RED_ZONE])),
        "reweight_late_down_p99": float(np.nanquantile(reweight[:, CELL_LATE_DOWN], 0.99)),
        "expected_share_sd": [float(q[:, c].std(ddof=1)) for c in range(3)],
        "realized_share_sd": [float((counts[:, c] / n_all).std(ddof=1)) for c in range(3)],
    }

    # --- the M-2 carry-forward proof, numerically ---------------------------
    carry = _carry_forward(plays, table, rng)

    dispersion = {
        "score_mean": float(score.mean()),
        "score_sd": float(score.std(ddof=1)),
        "median_abs_score": float(np.median(np.abs(score))),
        "q95_abs_score": float(np.quantile(np.abs(score), 0.95)),
        "max_abs_score": float(np.abs(score).max()),
        "incumbent_score_mean": float(incumbent.mean()),
        "incumbent_score_sd": float(incumbent.std(ddof=1)),
        "profile_shift_mean": float(shift.mean()),
        "profile_shift_sd": float(shift.std(ddof=1)),
        "red_zone_cell_sd": float(table["red_zone"].to_numpy().std(ddof=1)),
        "late_down_cell_sd": float(table["late_down"].to_numpy().std(ddof=1)),
    }
    differential = _power.game_differentials(table, score)
    dispersion["differential_sd"] = float(differential["differential"].std(ddof=1))
    dispersion["n_games"] = int(differential.height)
    dispersion["n_team_games"] = int(table.height)

    return {
        "profile": profile,
        "identities": identities,
        "contamination": contamination,
        "expected_mix_arm": expected_mix,
        "m2_carry_forward": carry,
        "dispersion": dispersion,
    }


def _refit_one(
    y_mean: np.ndarray,
    weight: np.ndarray,
    x: np.ndarray,
    group: np.ndarray,
    x_row: float,
    group_row,
) -> float:
    """Leave-one-team-out fitted value for one held-out row, from a reduced sample."""
    rows = group != group_row
    w, xi, yi = weight[rows], x[rows], y_mean[rows]
    s_w, s_wx, s_wxx = w.sum(), (w * xi).sum(), (w * xi * xi).sum()
    s_y, s_yx = (w * yi).sum(), (w * yi * xi).sum()
    slope = (s_w * s_yx - s_wx * s_y) / (s_w * s_wxx - s_wx * s_wx)
    return float((s_y - slope * s_wx) / s_w + slope * x_row)


def _naive_assignment(
    rung: str, cell: np.ndarray, down: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """One draw's cell labels, written the slow obvious way.

    Deliberately independent of ``nfl_simulator.placement``, which computes the
    same nulls through algebraic shortcuts. A shortcut and the thing it is short
    for can only be checked against each other by writing both.
    """
    assigned = np.empty(len(cell), dtype=np.int64)
    n_rz = int(np.count_nonzero(cell == CELL_RED_ZONE))
    n_ld = int(np.count_nonzero(cell == CELL_LATE_DOWN))
    if rung in ("raw", "raw_var_matched"):
        order = rng.permutation(len(cell))
        assigned[order[:n_rz]] = CELL_RED_ZONE
        assigned[order[n_rz : n_rz + n_ld]] = CELL_LATE_DOWN
        assigned[order[n_rz + n_ld :]] = CELL_OTHER
        return assigned
    # The down-stratified rungs: a play keeps its down, so only which plays in a
    # down stratum wear the red-zone label moves. Everything else falls out of
    # the down — late downs to the late-down cell, early downs to the third.
    for stratum in np.unique(down):
        rows = np.flatnonzero(down == stratum)
        n_rz_d = int(np.count_nonzero(cell[rows] == CELL_RED_ZONE))
        shuffled = rows[rng.permutation(len(rows))]
        assigned[shuffled[:n_rz_d]] = CELL_RED_ZONE
        rest = shuffled[n_rz_d:]
        assigned[rest] = CELL_LATE_DOWN if stratum in (3, 4) else CELL_OTHER
    return assigned


def _naive_values(rung: str, epa: np.ndarray, assigned: np.ndarray, down: np.ndarray) -> np.ndarray:
    """The play values one draw sees, after that rung's variance stretch.

    Rung 4 stretches every play's deviation from the **team-game** mean by its
    assigned cell's factor; rung 3 stretches only the red-zone-assigned plays,
    and around their own **down stratum's** mean. Document 35 §5 fixes both.
    """
    if rung == "raw":
        return epa.copy()
    if rung == "raw_var_matched":
        scales = np.array(RUNG_SCALES["raw_var_matched"])
        return epa.mean() + scales[assigned] * (epa - epa.mean())
    if rung == "down_stratified":
        return epa.copy()
    values = epa.copy()
    stretch = RUNG_SCALES["down_stratified_var_matched"][CELL_RED_ZONE]
    for stratum in np.unique(down):
        rows = np.flatnonzero(down == stratum)
        block_mean = epa[rows].mean()
        chosen = rows[assigned[rows] == CELL_RED_ZONE]
        values[chosen] = block_mean + stretch * (epa[chosen] - block_mean)
    return values


def _naive_score(values: np.ndarray, assigned: np.ndarray, mu: np.ndarray | None) -> float:
    """One draw's placement points, with or without the redesign's re-centring."""
    centred = values if mu is None else values - mu[assigned]
    leverage = assigned != CELL_OTHER
    return float((centred[leverage].sum() - int(leverage.sum()) * centred.mean()) * POINTS_PER_EPA)


def _carry_forward(plays: pl.DataFrame, table: pl.DataFrame, rng: np.random.Generator) -> dict:
    """Every rung's null draws move by the same constant the realized score does.

    Document 36 §6 derives it: with the three cell sizes held fixed — which every
    rung does — the profile a draw subtracts is the same whichever plays land
    where, so ``redesigned = incumbent - C`` for the realized score *and* for
    every draw. A rank inside the draws is invariant to a shift, so the PIT does
    not move and M-2's coverage, its rung adoption and its power table all carry
    forward untouched.

    Checked here rather than asserted. Both scores are recomputed the slow way
    from a **shared** assignment, so the two sides of the claim are computed
    independently of each other and of the production module; the module's own
    draws are compared against the naive incumbent as well, so a shortcut that
    had drifted from its rung's definition would show up here too.
    """
    arrays = _power.team_game_arrays(plays)
    starts, sizes = arrays["starts"], arrays["sizes"]
    epa, cell, down = arrays["epa"], arrays["cell"], arrays["down"]
    mu = np.column_stack(
        [table["mu_rz"].to_numpy(), table["mu_ld"].to_numpy(), table["mu_other"].to_numpy()]
    )
    shift = table["profile_shift"].to_numpy()

    sample = rng.choice(len(starts), size=CARRY_FORWARD_TEAM_GAMES, replace=False)
    worst_gap = {rung: 0.0 for rung in LADDER}
    worst_pit = {rung: 0.0 for rung in LADDER}
    pit_moved = {rung: 0 for rung in LADDER}
    module_gap: dict[str, list[float]] = {rung: [] for rung in LADDER}
    checked = 0
    for row in sample:
        block = slice(starts[row], starts[row] + sizes[row])
        e, c, d = epa[block], cell[block], down[block]
        leverage = int(np.count_nonzero(c != CELL_OTHER))
        if leverage in (0, len(e)):
            continue
        checked += 1
        realized_incumbent = _naive_score(e, c, None)
        realized_redesign = _naive_score(e, c, mu[row])

        for rung in LADDER:
            draw_rng = np.random.default_rng(RANDOM_SEED + int(row))
            incumbent = np.empty(CARRY_FORWARD_DRAWS)
            redesign = np.empty(CARRY_FORWARD_DRAWS)
            for draw in range(CARRY_FORWARD_DRAWS):
                assigned = _naive_assignment(rung, c, d, draw_rng)
                values = _naive_values(rung, e, assigned, d)
                incumbent[draw] = _naive_score(values, assigned, None)
                redesign[draw] = _naive_score(values, assigned, mu[row])
            worst_gap[rung] = max(
                worst_gap[rung], float(np.abs((incumbent - shift[row]) - redesign).max())
            )
            pit_incumbent = _mid_p(realized_incumbent, incumbent)
            pit_redesign = _mid_p(realized_redesign, redesign)
            worst_pit[rung] = max(worst_pit[rung], abs(pit_incumbent - pit_redesign))
            pit_moved[rung] += int(pit_incumbent != pit_redesign)

            # The production module's shortcut against the naive rung, on the
            # distribution rather than draw for draw: the two use their randomness
            # differently, so only the null they describe is comparable. Signed,
            # because a one-directional gap is a construction difference and a
            # symmetric one is 300 draws' worth of noise.
            shortcut = permutation_draws(
                rung,
                e,
                c,
                d,
                n_draws=CARRY_FORWARD_DRAWS,
                rng=np.random.default_rng(RANDOM_SEED + int(row)),
                cell_scales=RUNG_SCALES[rung],
            )
            module_gap[rung].append(
                (float(np.std(shortcut)) - float(np.std(incumbent)))
                / max(float(np.std(incumbent)), 1e-9)
            )
    return {
        "team_games_checked": checked,
        "draws_per_team_game": CARRY_FORWARD_DRAWS,
        "worst_abs_draw_gap_points": worst_gap,
        "worst_abs_pit_gap": worst_pit,
        "team_games_whose_pit_moved_at_all": pit_moved,
        "module_vs_naive_mean_relative_sd_gap": {
            rung: float(np.mean(module_gap[rung])) for rung in LADDER
        },
        "module_vs_naive_sd_of_relative_sd_gap": {
            rung: float(np.std(module_gap[rung], ddof=1)) for rung in LADDER
        },
    }


def _mid_p(realized: float, draws: np.ndarray) -> float:
    below = float(np.count_nonzero(draws < realized))
    equal = float(np.count_nonzero(draws == realized))
    return (below + 0.5 * equal) / len(draws)


# --------------------------------------------------------------------------
# part 2 — M-3, persistence
# --------------------------------------------------------------------------


def m3_part(table: pl.DataFrame, rng: np.random.Generator) -> dict:
    rows, blocks = _power.season_blocks(table)
    values = table["score"].to_numpy().astype(float)[rows]
    mask = _power.split_masks(blocks, len(values), rng)

    null = _power.m3_permutation_null(values, mask, blocks, rng)
    threshold = float(np.quantile(null, 0.95))
    power = [_power.m3_power(values, mask, blocks, threshold, target, rng) for target in M3_TARGETS]
    return {
        "n_team_seasons": int(len(blocks)),
        "n_team_games": int(len(values)),
        "score_sd": float(values.std(ddof=1)),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "threshold_p95": threshold,
        "null_p99": float(np.quantile(null, 0.99)),
        "power": power,
    }


# --------------------------------------------------------------------------
# part 3 — M-4, skill preservation, and how much of a pass is mechanical
# --------------------------------------------------------------------------


def season_table(table: pl.DataFrame) -> pl.DataFrame:
    """Team-season means, their sampling SDs, and the two quality covariates."""
    season = (
        table.group_by("team_season", maintain_order=True)
        .agg(
            pl.col("posteam").first().alias("team"),
            pl.col("season").first(),
            pl.len().alias("games"),
            pl.col("score").mean().alias("mean_score"),
            pl.col("score").std(ddof=1).alias("sd_score"),
            pl.col("epa_all").sum().alias("epa_sum"),
            pl.col("n_all").sum().alias("plays"),
        )
        .with_columns((pl.col("epa_sum") / pl.col("plays")).alias("s0_quality"))
    )
    # M-4b's covariate: the franchise's quality in its *other* seasons. The
    # leave-one-team-out fit never saw a single play of this franchise, so no
    # part of the construction has been orthogonalised against this vector.
    franchise = season.group_by("team").agg(
        pl.col("epa_sum").sum().alias("fr_epa"), pl.col("plays").sum().alias("fr_plays")
    )
    return (
        season.join(franchise, on="team", how="left")
        .with_columns(
            ((pl.col("fr_epa") - pl.col("epa_sum")) / (pl.col("fr_plays") - pl.col("plays"))).alias(
                "s0_other_seasons"
            )
        )
        .with_columns((pl.col("sd_score") / pl.col("games").sqrt()).alias("mean_sampling_sd"))
    )


def m4_arm(quality: np.ndarray, sampling_sd: np.ndarray, rng: np.random.Generator) -> dict:
    null = _power.m4_null(quality, sampling_sd, rng)
    bound = float(np.quantile(np.abs(null), 0.95))
    return {
        "n_team_seasons": int(len(quality)),
        "null_abs_p50": float(np.quantile(np.abs(null), 0.50)),
        "bound_abs_p95": bound,
        "null_abs_p99": float(np.quantile(np.abs(null), 0.99)),
        "power": {
            f"{leak:.2f}": _power.m4_power(quality, sampling_sd, bound, leak, rng)
            for leak in M4_LEAKS
        },
    }


def _simulate_league(
    table: pl.DataFrame,
    counts: np.ndarray,
    per_game_truth: np.ndarray,
    leak_shape: str,
    magnitude: float,
    truth: np.ndarray,
    season_effect: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, pl.DataFrame]:
    """One synthetic league's cell EPA sums, on the real cell denominators.

    The score reads only each team-game's cell counts and cell EPA sums, so a
    league is drawn at that grain rather than play by play — which is what makes
    hundreds of *fit, score, correlate* replicates cheap. Cell variances and the
    structural profile are document 35 §3's, measured on the real stream.

    ``leak_shape`` says what, beyond the structural profile every league carries,
    is true of this one:

    * ``none``            — nothing. Teams differ in overall quality and in the
      play mix their quality drags them into, and placement is pure noise. This
      is the reference distribution, and the fact that it is *not* centred on
      zero is §7's finding.
    * ``profile_linear``  — a late-down premium linear in team quality. The fit
      removes this by construction, so the flag rate here measures how much of an
      M-4 pass is arithmetic rather than evidence.
    * ``profile_step``    — the same premium, given only to the top quality
      quartile. Nothing linear removes it; this is the power the gate keeps.
    """
    structural = np.array([0.01469, -0.04814, 0.01131])
    cell_sd = np.sqrt(np.array([2.1845, 3.6690, 1.2494]))

    premium = np.zeros((len(per_game_truth), 3))
    if leak_shape == "profile_linear":
        centred = per_game_truth - per_game_truth.mean()
        premium[:, CELL_LATE_DOWN] = magnitude * centred / centred.std()
    elif leak_shape == "profile_step":
        premium[:, CELL_LATE_DOWN] = magnitude * (per_game_truth > np.quantile(truth, 0.75))
    elif leak_shape == "placement_orthogonal":
        # A genuine, persistent, team-level placement tendency that has nothing
        # to do with quality. Nothing in the construction is aimed at it, so this
        # is the arm that says whether the redesign is a targeted correction or
        # an indiscriminate one — and it is the truth M-3 has to be able to see.
        premium[:, CELL_LATE_DOWN] = magnitude * season_effect
    elif leak_shape != "none":
        raise ValueError(leak_shape)

    cell_mean = per_game_truth[:, None] + structural[None, :] + premium
    drawn = np.where(
        counts > 0,
        counts * cell_mean
        + rng.normal(0.0, 1.0, size=counts.shape) * cell_sd[None, :] * np.sqrt(counts),
        0.0,
    )
    simulated = (
        table.with_columns(
            pl.Series("epa_rz", drawn[:, 0]),
            pl.Series("epa_ld", drawn[:, 1]),
            pl.Series("epa_all", drawn.sum(axis=1)),
        )
        .with_columns(
            (pl.col("epa_all").sum().over("team_season") - pl.col("epa_all")).alias("epa_rest"),
            (pl.col("n_all").sum().over("team_season") - pl.col("n_all")).alias("n_rest"),
        )
        .with_columns((pl.col("epa_rest") / pl.col("n_rest")).alias("s0_loo"))
    )
    return drawn, simulated


def pipeline_arm(
    table: pl.DataFrame,
    leak_shape: str,
    magnitude: float,
    replicates: int,
    rng: np.random.Generator,
) -> dict:
    """Fit, score and correlate a whole synthetic league, ``replicates`` times.

    Both scores are computed on every league — the redesign's and the incumbent's
    — because the contrast between them on the ``none`` arm is the cleanest
    statement available that the redesign fixes what document 35 §11 found, and
    it costs one extra line.

    Both covariates are read on every league too. ``same_season`` shares its
    plays with the score; ``other_seasons`` does not, and §7 turns on the gap.
    """
    n_all, counts, _, _ = cell_matrices(table)
    group = table["posteam"].to_numpy()
    season_key, inverse = np.unique(table["team_season"].to_numpy(), return_inverse=True)
    team_of_season = np.array([group[inverse == index][0] for index in range(len(season_key))])
    games = np.bincount(inverse).astype(float)
    plays_per_season = np.bincount(inverse, weights=n_all)

    quality_truth = table["epa_all"].to_numpy() / table["n_all"].to_numpy()
    truth = np.bincount(inverse, weights=quality_truth * n_all) / plays_per_season
    per_game_truth = truth[inverse]

    rows, blocks = _power.season_blocks(table)
    masks = _power.split_masks(blocks, len(rows), np.random.default_rng(RANDOM_SEED))

    out = {
        name: np.empty(replicates)
        for name in (
            "redesign_same",
            "redesign_other",
            "incumbent_same",
            "incumbent_other",
            "split_half_r",
        )
    }
    for replicate in range(replicates):
        season_effect = rng.normal(0.0, 1.0, size=len(season_key))[inverse]
        drawn, simulated = _simulate_league(
            table, counts, per_game_truth, leak_shape, magnitude, truth, season_effect, rng
        )
        _, counts_s, sums_s, s0_s = cell_matrices(simulated)
        mu = expected_profile(counts_s, sums_s, s0_s, group)
        redesign = redesigned_cell_points(n_all, counts_s, sums_s, mu)[:, :2].sum(axis=1)
        incumbent = redesigned_cell_points(n_all, counts_s, sums_s, np.zeros_like(mu))[:, :2].sum(
            axis=1
        )

        season_epa = np.bincount(inverse, weights=drawn.sum(axis=1))
        same = season_epa / plays_per_season
        other = np.array(
            [
                (season_epa[team_of_season == team].sum() - season_epa[index])
                / (plays_per_season[team_of_season == team].sum() - plays_per_season[index])
                for index, team in enumerate(team_of_season)
            ]
        )
        for name, score in (("redesign", redesign), ("incumbent", incumbent)):
            mean_score = np.bincount(inverse, weights=score) / games
            out[f"{name}_same"][replicate] = np.corrcoef(mean_score, same)[0, 1]
            out[f"{name}_other"][replicate] = np.corrcoef(mean_score, other)[0, 1]
        out["split_half_r"][replicate] = _power.split_half_r(redesign[rows], masks, blocks)
        if (replicate + 1) % 50 == 0:
            print(f"    pipeline {leak_shape}@{magnitude} {replicate + 1}/{replicates}", flush=True)

    return {
        "leak_shape": leak_shape,
        "magnitude_epa_per_play": magnitude,
        "replicates": replicates,
        "correlations": {
            name: {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "abs_p95": float(np.quantile(np.abs(values), 0.95)),
            }
            for name, values in out.items()
        },
        "_draws": {name: values.tolist() for name, values in out.items()},
    }


def m4_part(table: pl.DataFrame, rng: np.random.Generator) -> dict:
    season = season_table(table)
    sampling_sd = season["mean_sampling_sd"].to_numpy().astype(float)

    # Document 35 §7's null, kept for comparison: placement redrawn independent of
    # quality. §7 of document 36 records why it is not the reference the redesign
    # can use for the same-season covariate.
    independent = {
        "same_season": m4_arm(season["s0_quality"].to_numpy().astype(float), sampling_sd, rng),
        "other_seasons": m4_arm(
            season["s0_other_seasons"].to_numpy().astype(float), sampling_sd, rng
        ),
    }

    reference = pipeline_arm(table, "none", 0.0, N_PIPELINE_NULL_REPLICATES, rng)
    bounds = {
        "same_season": reference["correlations"]["redesign_same"]["abs_p95"],
        "other_seasons": reference["correlations"]["redesign_other"]["abs_p95"],
    }

    power = []
    for shape, magnitude in (
        ("profile_linear", 0.05),
        ("profile_linear", 0.10),
        ("profile_step", 0.05),
        ("profile_step", 0.10),
        ("profile_step", 0.20),
        ("placement_orthogonal", 0.02),
        ("placement_orthogonal", 0.05),
        ("placement_orthogonal", 0.10),
    ):
        arm = pipeline_arm(table, shape, magnitude, N_PIPELINE_REPLICATES, rng)
        for covariate, key in (
            ("same_season", "redesign_same"),
            ("other_seasons", "redesign_other"),
        ):
            draws = np.array(arm["_draws"][key])
            arm.setdefault("flag_rate", {})[covariate] = float(
                np.mean(np.abs(draws) > bounds[covariate])
            )
        arm.pop("_draws")
        power.append(arm)

    # The incumbent, read against the clean covariate on the real data. Not a gate
    # and not a rescue: document 35 §11 already failed that design at +0.5435, and
    # this number is here so the covariate document 36 promotes cannot be mistaken
    # for one chosen because it forgives the failure.
    incumbent_mean = (
        table.group_by("team_season", maintain_order=True)
        .agg(pl.col("incumbent_score").mean())["incumbent_score"]
        .to_numpy()
    )
    diagnostic = {
        "incumbent_vs_other_seasons_quality": float(
            np.corrcoef(incumbent_mean, season["s0_other_seasons"].to_numpy())[0, 1]
        ),
        "incumbent_vs_same_season_quality": float(
            np.corrcoef(incumbent_mean, season["s0_quality"].to_numpy())[0, 1]
        ),
    }

    reference.pop("_draws")
    return {
        "independent_null_doc35_style": independent,
        "pipeline_reference_no_leak": reference,
        "bounds": bounds,
        "power": power,
        "incumbent_diagnostic": diagnostic,
        "quality_sd": float(season["s0_quality"].to_numpy().std(ddof=1)),
        "other_seasons_quality_sd": float(season["s0_other_seasons"].to_numpy().std(ddof=1)),
    }


# --------------------------------------------------------------------------
# part 4 — M-6
# --------------------------------------------------------------------------


def m6_part(differential_sd: float, rng: np.random.Generator) -> dict:
    return _power.m6_power(differential_sd, rng)


# --------------------------------------------------------------------------


def load_results() -> dict:
    path = paths.RESEARCH_OUTPUT_DIR / RESULTS
    return json.loads(path.read_text()) if path.exists() else {"random_seed": RANDOM_SEED}


def save_results(results: dict) -> None:
    path = paths.RESEARCH_OUTPUT_DIR / RESULTS
    path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"\nwrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", default="all", choices=["design", "m3", "m4", "m6", "all"])
    args = parser.parse_args()

    rng = np.random.default_rng(RANDOM_SEED)
    plays, reconciliation = _power.load_luck_priced_plays()
    print(
        f"stream: {plays.height:,} plays, {reconciliation['n_plays_carrying_a_ledger_row']:,} re-priced"
    )

    results = load_results()
    results["reconciliation"] = reconciliation
    table = None

    if args.part in ("design", "all"):
        print("\n== design parameters, contamination, M-2 carry-forward ==")
        results["design"] = design_part(plays, rng)
        print(json.dumps(results["design"]["identities"], indent=2))
        print(json.dumps(results["design"]["m2_carry_forward"], indent=2))
        print(json.dumps(results["design"]["dispersion"], indent=2))

    if args.part in ("m3", "m4", "m6", "all"):
        table = score_frame(team_game_table(plays))

    if args.part in ("m3", "all"):
        print("\n== M-3, persistence ==")
        results["m3_persistence"] = m3_part(table, rng)
        print(json.dumps(results["m3_persistence"], indent=2))

    if args.part in ("m4", "all"):
        print("\n== M-4, skill preservation ==")
        results["m4_skill_preservation"] = m4_part(table, rng)
        print(json.dumps(results["m4_skill_preservation"], indent=2))

    if args.part in ("m6", "all"):
        print("\n== M-6, rematch width ==")
        differential_sd = results.get("design", {}).get("dispersion", {}).get(
            "differential_sd"
        ) or float(
            _power.game_differentials(table, table["score"].to_numpy())["differential"].std(ddof=1)
        )
        results["m6_rematch"] = m6_part(differential_sd, rng)
        print(json.dumps(results["m6_rematch"], indent=2))

    save_results(results)


if __name__ == "__main__":
    main()
