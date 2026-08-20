"""Product layer, task 2, step 1 — power calculation for the placement meter.

Document 04's closing lesson, restated as a process law and inherited by every
round since: a threshold set from an effect-size argument, with nobody asking
whether the data *could* achieve it, is a gate that fails for reasons unrelated
to the hypothesis. So this script runs **before**
`docs/research/35-placement-meter-prereg.md` commits any threshold, and that
document is committed before `research/50_placement_meter.py` exists.

It answers one question per gate of document 34 §6:

* **M-2 calibration.** With the team-games that actually exist, how far from
  uniform does a *correctly* calibrated band's PIT wander by chance — and how
  far does a *mis*calibrated one wander? The tolerance and the false-alarm rate
  at it come from here.
* **M-3 luck licence.** How large would a true team-level placement tendency
  have to be before the split-half test could see it, at the real
  team-season denominators?
* **M-4 skill preservation.** What correlation between a team-season's mean
  placement score and its offensive quality does a *genuinely clean* baseline
  produce by chance? That spread is the bound; anything below it is invisible.
* **M-6 rematch.** Does the document 06 harness have the width to pass a
  non-inferiority gate at all, when the quantity subtracted is placement luck
  at the meter's own magnitude?

### What this script deliberately does NOT compute

Not one gated statistic. It never computes the meter's split-half correlation,
its PIT against its own bands, its correlation with offensive quality, or its
rematch delta. Those live in `research/50_placement_meter.py`, which does not
exist yet.

What it does measure from real data are **design parameters** in document 08
§2's sense: league-pooled play-level moments, the real per-team-game cell
denominators, and the realized dispersion of the score. Every one of them is
pooled over all 32 teams and carries no team identity, and none of them can move
a threshold — the M-2 tolerance, the M-3 threshold and the M-4 bound are all
produced by simulation from a null. The meter's dispersion is measured because
M-6's power check is *defined* as "at the meter's real magnitude" and cannot be
run without it. This is disclosed here and in the pre-registration's defect
register rather than hidden, which is the only defence available.

    uv run python research/49_placement_power.py
"""

from __future__ import annotations

import json
import math
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_rematch_power = import_module("08_rematch_power")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.ingest import load_pbp  # noqa: E402

RANDOM_SEED = 20260819

# --------------------------------------------------------------------------
# the score's own constants — document 34 §3
# --------------------------------------------------------------------------

# Document 27 §13's fitted slope, the currency every luck number in this project
# is quoted in. Hard-coded rather than refitted so the meter and the ledger
# price a point identically.
POINTS_PER_EPA = 0.8389

RED_ZONE_YARDS = 20
LATE_DOWNS = (3, 4)

LEDGER_ARTIFACT = "dtw_ledger_v13.parquet"
GAMES_ARTIFACT = "dtw_games_v13.parquet"

# Cell codes, used as array labels throughout.
CELL_RED_ZONE = 0
CELL_LATE_DOWN = 1
CELL_OTHER = 2

# --------------------------------------------------------------------------
# band and gate machinery constants
# --------------------------------------------------------------------------

# Document 34 §4: ~2,000 draws, 89% equal-tailed, house convention.
N_BAND_DRAWS = 2000
BAND_LOW, BAND_HIGH = 5.5, 94.5

# The power simulations run at a reduced draw count so a scenario fits in
# seconds rather than minutes. The discreteness this introduces is carried into
# the PIT by construction (mid-P below), so the null it produces is the null of
# the *cheaper* statistic and is therefore conservative about calibration, not
# optimistic.
N_BAND_DRAWS_POWER = 300

# M-3 inherits document 08's split-half protocol wholesale so the number is
# directly comparable to the sequencing table already on record.
N_SPLITS = 200
MIN_GAMES = 8
N_NULL_REPLICATES = 500
N_POWER_REPLICATES = 500
REFERENCE_R = 0.12

# M-4 resampling depth, and M-2's simulated-league counts. The exchangeable
# truth sets the tolerance, so it gets the deeper run; the graded scenarios only
# need a flag rate, which is a mean rather than a tail quantile.
N_M4_REPLICATES = 4000
N_KS_REPLICATES_EXCHANGEABLE = 40
N_KS_REPLICATES_GRADED = 12
N_KS_RESAMPLES = 4000

# Two percentage points of coverage on the 89% band — the materiality floor the
# M-2 tolerance is not allowed to sit below.
MATERIAL_COVERAGE_SHIFT = 0.02

# M-6 depth. The harness refits a logistic model ten times per replicate on 531
# rows, so this is the expensive one.
N_M6_REPLICATES = 400

# The three rungs of document 34 §4's constraint ladder, in the order `main`
# simulates them. This tuple is frozen: the random stream is consumed rung by
# rung and then handed on to M-3, M-4 and M-6, so appending to it would move
# thresholds that document 35 has already committed.
LADDER = ("raw", "down_stratified", "down_stratified_var_matched")

# The fourth rung, approved by the maintainer on 2026-08-19 from the pre-registration's
# parked list and added by amendment before any calibration statistic ran. It
# fills the gap document 35 §5 derived: rung 1 re-randomizes the late-down cell
# but ignores its second moment (1.99x the league), and rungs 2 and 3 respect
# that moment only by freezing the cell entirely. Rung 4 is the only one that
# does both — a fully unconstrained shuffle whose leverage-assigned plays are
# stretched to their own cell's second moment.
#
# It is simulated by `ladder_addendum` on a *separate* generator and merged into
# the existing results file, so every number already committed in document 35
# reproduces bit-for-bit rather than merely being expected to.
LADDER_ADDENDUM = ("raw_var_matched",)

# The constraint order document 35 §6's adoption rule reads — least constrained
# first, where "constrained" means how much of the realized configuration the
# null holds fixed. Rung 4 holds no play's cell membership fixed and only
# corrects second moments, so it spends less exchangeability licence than the
# down-stratified rungs, which freeze every third- and fourth-down play's
# contribution outright.
ADOPTION_LADDER = (
    "raw",
    "raw_var_matched",
    "down_stratified",
    "down_stratified_var_matched",
)


# --------------------------------------------------------------------------
# part 0 — the luck-priced play stream and the score
# --------------------------------------------------------------------------


def load_luck_priced_plays() -> tuple[pl.DataFrame, dict]:
    """Scrimmage plays with each play's EPA re-priced at the v1.3 ledger.

    Document 34 §5: the meter's input is ``epa - luck_epa`` for any play carrying
    a ledger row, so a red-zone fumble's coin swing lives only in the DTW ledger
    and the meter sees that play at its expectation. The fix is at the input, not
    patched at the output.

    ``luck_epa`` arrives signed to the *home* team, because that is what makes
    the ledger's sum equal the whole adjustment the simulator applies. ``epa`` is
    nflverse's, signed to the *possessing* team. The two are reconciled here and
    nowhere else.
    """
    pbp = load_pbp(
        columns=[
            "game_id",
            "play_id",
            "season",
            "week",
            "posteam",
            "home_team",
            "away_team",
            "play_type",
            "epa",
            "yardline_100",
            "down",
        ]
    )

    scrimmage = pbp.filter(
        pl.col("posteam").is_not_null()
        & pl.col("play_type").is_in(["pass", "run"])
        & pl.col("epa").is_not_null()
        & pl.col("down").is_not_null()
    )

    ledger = (
        pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / LEDGER_ARTIFACT)
        .group_by(["game_id", "play_id"])
        .agg(pl.col("luck_epa").sum().alias("luck_epa_home"))
    )

    # Only games the shipped artifact adjudicates are in scope: the meter is a
    # panel on the product's own page, and a game with no DTW number has no page.
    adjudicated = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / GAMES_ARTIFACT).select("game_id")

    joined = (
        scrimmage.join(adjudicated, on="game_id", how="inner")
        .join(ledger, on=["game_id", "play_id"], how="left")
        .with_columns(pl.col("luck_epa_home").fill_null(0.0))
        .with_columns(
            pl.when(pl.col("posteam") == pl.col("home_team"))
            .then(pl.col("luck_epa_home"))
            .otherwise(-pl.col("luck_epa_home"))
            .alias("luck_epa_pos")
        )
        .with_columns(
            (pl.col("epa") - pl.col("luck_epa_pos")).alias("epa_priced"),
            pl.when(pl.col("yardline_100") <= RED_ZONE_YARDS)
            .then(CELL_RED_ZONE)
            .when(pl.col("down").is_in(LATE_DOWNS))
            .then(CELL_LATE_DOWN)
            .otherwise(CELL_OTHER)
            .alias("cell"),
            (pl.col("posteam") == pl.col("home_team")).alias("is_home"),
            pl.concat_str(
                [pl.col("season").cast(pl.String), pl.col("posteam")], separator="_"
            ).alias("team_season"),
        )
        .sort(["team_season", "game_id", "play_id"])
    )

    n_priced = int((joined["luck_epa_pos"] != 0.0).sum())
    reconciliation = {
        "n_scrimmage_plays": int(scrimmage.height),
        "n_plays_in_scope": int(joined.height),
        "n_plays_carrying_a_ledger_row": n_priced,
        "ledger_rows_total": int(ledger.height),
        "mean_abs_repricing_epa": float(joined["luck_epa_pos"].abs().mean()),
        "sum_repricing_epa": float(joined["luck_epa_pos"].sum()),
    }
    return joined, reconciliation


def team_game_arrays(plays: pl.DataFrame) -> dict:
    """Flat play arrays with per-team-game offsets, plus the team-game table.

    Groups are contiguous blocks in a single flat array rather than a list of
    per-game arrays: every routine below indexes into the same buffer, which is
    what keeps 2,000 permutation draws over 5,522 team-games inside a minute.
    """
    keyed = plays.with_columns(
        pl.concat_str([pl.col("game_id"), pl.col("posteam")], separator="|").alias("unit")
    ).sort(["unit", "play_id"])

    units = keyed["unit"].to_numpy()
    boundaries = np.flatnonzero(np.r_[True, units[1:] != units[:-1], True])
    starts = boundaries[:-1]
    sizes = np.diff(boundaries)

    epa = keyed["epa_priced"].to_numpy().astype(float)
    cell = keyed["cell"].to_numpy().astype(np.int64)
    down = keyed["down"].to_numpy().astype(np.int64)

    meta = keyed.group_by("unit", maintain_order=True).agg(
        pl.col("game_id").first(),
        pl.col("posteam").first(),
        pl.col("season").first(),
        pl.col("team_season").first(),
        pl.col("is_home").first(),
        pl.len().alias("n_all"),
        (pl.col("cell") == CELL_RED_ZONE).sum().alias("n_rz"),
        (pl.col("cell") == CELL_LATE_DOWN).sum().alias("n_ld"),
        pl.col("epa_priced").sum().alias("epa_all"),
        pl.col("epa_priced").filter(pl.col("cell") == CELL_RED_ZONE).sum().alias("epa_rz"),
        pl.col("epa_priced").filter(pl.col("cell") == CELL_LATE_DOWN).sum().alias("epa_ld"),
    )

    return {
        "epa": epa,
        "cell": cell,
        "down": down,
        "starts": starts,
        "sizes": sizes,
        "meta": meta,
    }


def cell_scores(
    n_all: np.ndarray,
    epa_all: np.ndarray,
    n_cell: np.ndarray,
    epa_cell: np.ndarray,
) -> np.ndarray:
    """One cell's placement points: ``(mean_cell - mean_all) * n_cell * ppe``.

    Written as ``epa_cell - n_cell * mean_all`` rather than as the literal
    difference of means, because that form is exactly zero when the cell is
    empty instead of ``0/0``. Document 34 §5: ~2% of team-games have no red-zone
    play and that cell contributes exactly 0.
    """
    mean_all = np.divide(epa_all, n_all, out=np.zeros_like(epa_all), where=n_all > 0)
    return (epa_cell - n_cell * mean_all) * POINTS_PER_EPA


def placement_scores(meta: pl.DataFrame) -> dict[str, np.ndarray]:
    """The realized score, its two cells, and the third-cell identity."""
    n_all = meta["n_all"].to_numpy().astype(float)
    epa_all = meta["epa_all"].to_numpy().astype(float)
    n_rz = meta["n_rz"].to_numpy().astype(float)
    n_ld = meta["n_ld"].to_numpy().astype(float)
    epa_rz = meta["epa_rz"].to_numpy().astype(float)
    epa_ld = meta["epa_ld"].to_numpy().astype(float)

    rz = cell_scores(n_all, epa_all, n_rz, epa_rz)
    ld = cell_scores(n_all, epa_all, n_ld, epa_ld)
    other = cell_scores(n_all, epa_all, n_all - n_rz - n_ld, epa_all - epa_rz - epa_ld)
    return {"red_zone": rz, "late_down": ld, "score": rz + ld, "other": other}


# --------------------------------------------------------------------------
# part 0b — the constraint ladder
# --------------------------------------------------------------------------
#
# The score is the sum over the two *leverage* cells, so under any label
# permutation only the leverage **union** matters, never the split between red
# zone and late down:
#
#     score = ppe * ( sum(epa over leverage plays) - k * mean_all ),   k = n_rz + n_ld
#
# That identity is what every rung below computes, and it is the reason the raw
# rung reduces to "draw k of n plays without replacement".


def _null_draws_raw(
    epa: np.ndarray, cell: np.ndarray, down: np.ndarray, n_draws: int, rng: np.random.Generator
) -> np.ndarray:
    """Rung 1 — plays are exchangeable across all three cells."""
    n = len(epa)
    k = int(np.count_nonzero(cell != CELL_OTHER))
    if k == 0 or k == n:
        return np.zeros(n_draws)
    tiled = np.broadcast_to(epa, (n_draws, n)).copy()
    shuffled = rng.permuted(tiled, axis=1)
    mean_all = epa.mean()
    return (shuffled[:, :k].sum(axis=1) - k * mean_all) * POINTS_PER_EPA


def _null_draws_down_stratified(
    epa: np.ndarray,
    cell: np.ndarray,
    down: np.ndarray,
    n_draws: int,
    rng: np.random.Generator,
    rz_scale: float = 1.0,
) -> np.ndarray:
    """Rungs 2 and 3 — plays keep their down; only field-position membership moves.

    **An arithmetic consequence worth stating before any data is seen.** The two
    leverage cells are "red zone, any down" and "late down, outside the red
    zone", so *every* third- and fourth-down play sits in the leverage union no
    matter where the red-zone labels land. Holding down fixed therefore freezes
    the entire late-down contribution, and the only quantity left random is which
    early-down plays were red-zone plays. Rungs 2 and 3 are consequently a null
    for red-zone placement alone, not for the meter as a whole, and their bands
    must be expected to be much narrower than rung 1's. This is derived, not
    measured, and it is recorded in the pre-registration's defect register.

    ``rz_scale`` is rung 3: the plays a draw assigns to the red-zone cell have
    their deviations from the team-game mean stretched by ``sqrt(var_rz /
    var_all)``, because a state that stretches outcomes makes a raw shuffle's
    red-zone band too narrow (document 34 §4). The stretch touches the *null*
    only; the realized score is never rescaled.
    """
    n = len(epa)
    k = int(np.count_nonzero(cell != CELL_OTHER))
    if k == 0 or k == n:
        return np.zeros(n_draws)

    mean_all = epa.mean()
    late = np.isin(down, LATE_DOWNS)
    is_rz = cell == CELL_RED_ZONE

    # Every late-down play is in the leverage union under this rung, whatever the
    # draw does, so its contribution is a constant.
    fixed = epa[late].sum()
    if rz_scale != 1.0:
        # A late-down play that lands in the red zone is stretched too, so the
        # correction is applied to the count of red-zone late-down plays drawn
        # from that stratum rather than to the whole block.
        fixed = 0.0

    totals = np.empty(n_draws)
    totals[:] = fixed

    for d in np.unique(down):
        idx = np.flatnonzero(down == d)
        n_d = len(idx)
        n_rz_d = int(np.count_nonzero(is_rz[idx]))
        block = epa[idx]
        late_d = bool(d in LATE_DOWNS)

        if rz_scale == 1.0:
            if late_d:
                continue  # already counted whole in `fixed`
            if n_rz_d == 0:
                continue
            if n_rz_d == n_d:
                totals += block.sum()
                continue
            tiled = np.broadcast_to(block, (n_draws, n_d)).copy()
            shuffled = rng.permuted(tiled, axis=1)
            totals += shuffled[:, :n_rz_d].sum(axis=1)
            continue

        # rung 3: red-zone-assigned plays are stretched about their own
        # stratum's mean, so the correction inflates the second moment and
        # leaves the first one exactly where it was.
        block_mean = block.mean()
        stretched = block_mean + rz_scale * (block - block_mean)
        if n_rz_d == 0:
            totals += block.sum() if late_d else 0.0
            continue
        if n_rz_d == n_d:
            totals += stretched.sum()
            continue
        order = np.argsort(rng.random((n_draws, n_d)), axis=1)
        chosen = order[:, :n_rz_d]
        rest = order[:, n_rz_d:]
        totals += stretched[chosen].sum(axis=1)
        if late_d:
            totals += block[rest].sum(axis=1)

    return (totals - k * mean_all) * POINTS_PER_EPA


def _null_draws_raw_var_matched(
    epa: np.ndarray,
    cell: np.ndarray,
    down: np.ndarray,
    n_draws: int,
    rng: np.random.Generator,
    cell_scales: tuple[float, float, float],
) -> np.ndarray:
    """Rung 4 — rung 1's unconstrained shuffle, with every cell's second moment restored.

    Approved by the maintainer 2026-08-19 from document 35 §5's parked list. The gap it
    fills is stated there: the score is the sum over the leverage *union*, so a
    rung that holds each play's down fixed freezes the whole late-down half of
    the meter (rungs 2 and 3), while the rung that does re-randomize it (rung 1)
    shuffles league-variance plays into a cell whose real play-level EPA variance
    is 1.99x the league. Neither is a null for the meter as a whole *and* honest
    about its second moment.

    Here every play is exchangeable across all three cells, exactly as in rung 1.
    A draw assigns ``n_rz`` plays to the red-zone cell, ``n_ld`` to the late-down
    cell and the rest to the third, and each play's deviation from the team-game
    mean is then stretched by *its assigned cell's* ``sqrt(var_cell / var_all)``.
    The third cell is stretched too — it shrinks, at 0.82 — because the score's
    baseline is the team-game mean over all plays, so a correction applied to the
    leverage cells alone lands in the baseline as well and inflates the null.

    Writing ``dev`` for deviations from the team-game mean and ``A``/``B`` for
    the drawn red-zone and late-down deviation sums, the third cell's sum is
    ``-(A + B)`` because deviations sum to zero, and the score reduces to

        ppe * [ (1 - k/n) * (s_rz * A + s_ld * B) + (k/n) * s_other * (A + B) ]

    which is exactly rung 1 when all three scales are 1 — the rung is a strict
    generalization of the one it sits above in the ladder, not an inflation of it.

    An earlier draft of this rung stretched only the two leverage cells and left
    the baseline alone. Measured before anything was committed, it was 1.27x too
    wide under an exchangeable truth (96.0% coverage of an 89% band) and still
    1.12x too wide under the real structure it was built for (94.5%), because the
    correction landed in ``mean_all`` as well as in the leverage sum. Recorded in
    document 35's defect register rather than silently replaced.

    As in rung 3 the stretch touches the *null* only; the realized score is never
    rescaled, so M-1's identities are unaffected.
    """
    n = len(epa)
    n_rz = int(np.count_nonzero(cell == CELL_RED_ZONE))
    n_ld = int(np.count_nonzero(cell == CELL_LATE_DOWN))
    k = n_rz + n_ld
    if k == 0 or k == n:
        return np.zeros(n_draws)

    s_rz, s_ld, s_other = cell_scales
    dev = epa - epa.mean()
    order = np.argsort(rng.random((n_draws, n)), axis=1)
    a = dev[order[:, :n_rz]].sum(axis=1) if n_rz else np.zeros(n_draws)
    b = dev[order[:, n_rz:k]].sum(axis=1) if n_ld else np.zeros(n_draws)
    share = k / n
    totals = (1.0 - share) * (s_rz * a + s_ld * b) + share * s_other * (a + b)
    return totals * POINTS_PER_EPA


def null_draws(
    rung: str,
    epa: np.ndarray,
    cell: np.ndarray,
    down: np.ndarray,
    n_draws: int,
    rng: np.random.Generator,
    rz_scale: float,
    cell_scales: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    if rung == "raw":
        return _null_draws_raw(epa, cell, down, n_draws, rng)
    if rung == "down_stratified":
        return _null_draws_down_stratified(epa, cell, down, n_draws, rng, rz_scale=1.0)
    if rung == "down_stratified_var_matched":
        return _null_draws_down_stratified(epa, cell, down, n_draws, rng, rz_scale=rz_scale)
    if rung == "raw_var_matched":
        return _null_draws_raw_var_matched(epa, cell, down, n_draws, rng, cell_scales)
    raise ValueError(f"unknown rung {rung!r}")


def realized_score(epa: np.ndarray, cell: np.ndarray) -> float:
    n = len(epa)
    k = int(np.count_nonzero(cell != CELL_OTHER))
    if n == 0:
        return 0.0
    lever = float(epa[cell != CELL_OTHER].sum())
    return (lever - k * epa.mean()) * POINTS_PER_EPA


def pit_of(realized: float, draws: np.ndarray) -> float:
    """Mid-P percentile of the realized score inside its own game's null.

    Mid-P rather than a plain rank because a permutation null is discrete: with
    ties broken to one side the PIT is not uniform even under exact
    exchangeability, and the calibration gate would then be testing the
    tie-breaking rule.
    """
    below = float(np.count_nonzero(draws < realized))
    equal = float(np.count_nonzero(draws == realized))
    return (below + 0.5 * equal) / len(draws)


def ks_statistic(pit: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance from uniform."""
    u = np.sort(pit)
    n = len(u)
    grid = np.arange(1, n + 1) / n
    return float(max(np.max(grid - u), np.max(u - (grid - 1.0 / n))))


# --------------------------------------------------------------------------
# part 1 — design parameters
# --------------------------------------------------------------------------


def design_parameters(plays: pl.DataFrame, arrays: dict, scores: dict) -> dict:
    """League-pooled moments and real denominators. No team identity survives."""
    epa = plays["epa_priced"].to_numpy().astype(float)
    cell = plays["cell"].to_numpy().astype(np.int64)
    meta = arrays["meta"]

    def moments(mask: np.ndarray) -> dict:
        block = epa[mask]
        return {
            "n": int(block.size),
            "mean": float(block.mean()),
            "var": float(block.var(ddof=1)),
        }

    rz = cell == CELL_RED_ZONE
    ld = cell == CELL_LATE_DOWN
    other = cell == CELL_OTHER

    n_rz = meta["n_rz"].to_numpy()
    n_ld = meta["n_ld"].to_numpy()
    n_all = meta["n_all"].to_numpy()
    score = scores["score"]

    params = {
        "all_plays": moments(np.ones_like(cell, dtype=bool)),
        "red_zone": moments(rz),
        "late_down_outside_rz": moments(ld),
        "other": moments(other),
        "cells_per_team_game": {
            "n_team_games": int(meta.height),
            "mean_n_all": float(n_all.mean()),
            "mean_n_rz": float(n_rz.mean()),
            "mean_n_ld": float(n_ld.mean()),
            "median_n_rz": float(np.median(n_rz)),
            "median_n_ld": float(np.median(n_ld)),
            "share_zero_red_zone": float(np.mean(n_rz == 0)),
            "share_zero_late_down": float(np.mean(n_ld == 0)),
            "share_zero_leverage": float(np.mean((n_rz + n_ld) == 0)),
            "share_all_leverage": float(np.mean((n_rz + n_ld) == n_all)),
        },
        "score_dispersion_points": {
            "mean": float(score.mean()),
            "sd": float(score.std(ddof=1)),
            "median_abs": float(np.median(np.abs(score))),
            "q95_abs": float(np.quantile(np.abs(score), 0.95)),
            "max_abs": float(np.abs(score).max()),
        },
        "identity_residual_max": float(
            np.abs(scores["red_zone"] + scores["late_down"] + scores["other"]).max()
        ),
        "points_per_epa": POINTS_PER_EPA,
    }
    params["variance_ratio_rz_over_all"] = params["red_zone"]["var"] / params["all_plays"]["var"]
    params["variance_ratio_ld_over_all"] = (
        params["late_down_outside_rz"]["var"] / params["all_plays"]["var"]
    )
    params["mean_offset_rz_points_per_game"] = (
        (params["red_zone"]["mean"] - params["all_plays"]["mean"])
        * params["cells_per_team_game"]["mean_n_rz"]
        * POINTS_PER_EPA
    )
    params["mean_offset_ld_points_per_game"] = (
        (params["late_down_outside_rz"]["mean"] - params["all_plays"]["mean"])
        * params["cells_per_team_game"]["mean_n_ld"]
        * POINTS_PER_EPA
    )
    return params


def game_differentials(meta: pl.DataFrame, score: np.ndarray) -> pl.DataFrame:
    """Home minus away placement points, one row per game."""
    frame = meta.select("game_id", "is_home").with_columns(pl.Series("score", score))
    home = frame.filter(pl.col("is_home")).select("game_id", pl.col("score").alias("home_score"))
    away = frame.filter(~pl.col("is_home")).select("game_id", pl.col("score").alias("away_score"))
    return home.join(away, on="game_id", how="inner").with_columns(
        (pl.col("home_score") - pl.col("away_score")).alias("differential")
    )


# --------------------------------------------------------------------------
# part 2 — M-2, the calibration instrument
# --------------------------------------------------------------------------


def synthetic_team_game(
    n_all: int,
    n_rz: int,
    n_ld: int,
    down_pool: np.ndarray,
    epa_pool: np.ndarray,
    var_inflation_rz: float,
    var_inflation_ld: float,
    mean_shift_rz: float,
    mean_shift_ld: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One synthetic team-game with a *known* calibration truth.

    Under the exchangeable truth every play is an independent draw from the same
    league pool, so placement is literally noise and a correctly built band must
    return a uniform PIT. The graded scenarios stretch and shift the leverage
    cells' play distributions, which is the miscalibration the ladder exists to
    survive: a state that widens outcomes makes a raw shuffle's band too narrow.

    Real denominators and the real fat-tailed play distribution are reused
    throughout — only the cell-conditional moments are manipulated.
    """
    pool_mean = epa_pool.mean()
    epa = epa_pool[rng.integers(0, len(epa_pool), size=n_all)]

    # Downs are drawn to be consistent with the cells: late-down cell plays must
    # carry a late down, red-zone plays take the league down mix, and the
    # everything-else cell is early downs outside the red zone by definition.
    down = down_pool[rng.integers(0, len(down_pool), size=n_all)]
    cell = np.full(n_all, CELL_OTHER, dtype=np.int64)
    cell[:n_rz] = CELL_RED_ZONE
    cell[n_rz : n_rz + n_ld] = CELL_LATE_DOWN
    late_values = np.array(LATE_DOWNS)
    down[n_rz : n_rz + n_ld] = late_values[rng.integers(0, len(late_values), size=n_ld)]
    n_tail = max(n_all - n_rz - n_ld, 0)
    early_values = np.array([d for d in np.unique(down_pool) if d not in LATE_DOWNS])
    down[n_rz + n_ld :] = early_values[rng.integers(0, len(early_values), size=n_tail)]

    rz = cell == CELL_RED_ZONE
    ld = cell == CELL_LATE_DOWN
    epa[rz] = pool_mean + mean_shift_rz + np.sqrt(var_inflation_rz) * (epa[rz] - pool_mean)
    epa[ld] = pool_mean + mean_shift_ld + np.sqrt(var_inflation_ld) * (epa[ld] - pool_mean)
    return epa, cell, down


def calibration_scenarios(design: dict, pool_mean: float) -> list[dict]:
    """The six simulated truths M-2's power is read at.

    `var_*` are multiples of the league play-level variance; `mean_*` are
    additive shifts in EPA per play. The "real" scenario carries the second- and
    first-moment structure actually measured from the play stream, which is the
    miscalibration the ladder has to survive in production — the truth the power
    column has to be read at.

    Extracted from `main` when rung 4 was added by amendment, so the addendum
    run and the original run cannot drift apart in what they simulate.
    """
    return [
        {"name": "exchangeable", "var_rz": 1.0, "var_ld": 1.0, "mean_rz": 0.0, "mean_ld": 0.0},
        {"name": "rz_var_1.10", "var_rz": 1.10, "var_ld": 1.0, "mean_rz": 0.0, "mean_ld": 0.0},
        {
            "name": "rz_var_real",
            "var_rz": design["variance_ratio_rz_over_all"],
            "var_ld": 1.0,
            "mean_rz": 0.0,
            "mean_ld": 0.0,
        },
        {"name": "rz_var_1.30", "var_rz": 1.30, "var_ld": 1.0, "mean_rz": 0.0, "mean_ld": 0.0},
        {
            "name": "ld_var_real",
            "var_rz": 1.0,
            "var_ld": design["variance_ratio_ld_over_all"],
            "mean_rz": 0.0,
            "mean_ld": 0.0,
        },
        {
            "name": "real_structure",
            "var_rz": design["variance_ratio_rz_over_all"],
            "var_ld": design["variance_ratio_ld_over_all"],
            "mean_rz": design["red_zone"]["mean"] - pool_mean,
            "mean_ld": design["late_down_outside_rz"]["mean"] - pool_mean,
        },
    ]


def pit_distribution_under(
    scenario: dict,
    rung: str,
    denominators: np.ndarray,
    down_pool: np.ndarray,
    epa_pool: np.ndarray,
    rz_scale: float,
    n_games: int,
    n_draws: int,
    rng: np.random.Generator,
    cell_scales: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """PIT values for `n_games` synthetic team-games under one calibration truth.

    Estimated once at high precision per (scenario, rung) rather than re-run per
    replicate: the sampling distribution of the KS statistic at any sample size
    then follows by resampling from this estimate, which is what makes a power
    table over six scenarios and three rungs affordable. The agreement between
    that resampled null and a direct simulation is checked in `main`.
    """
    rows = rng.integers(0, len(denominators), size=n_games)
    pit = np.empty(n_games)
    kept = 0
    for row in rows:
        n_all, n_rz, n_ld = denominators[row]
        if n_all < 2 or (n_rz + n_ld) in (0, n_all):
            continue
        epa, cell, down = synthetic_team_game(
            int(n_all),
            int(n_rz),
            int(n_ld),
            down_pool,
            epa_pool,
            scenario["var_rz"],
            scenario["var_ld"],
            scenario["mean_rz"],
            scenario["mean_ld"],
            rng,
        )
        draws = null_draws(rung, epa, cell, down, n_draws, rng, rz_scale, cell_scales)
        pit[kept] = pit_of(realized_score(epa, cell), draws)
        kept += 1
    return pit[:kept]


def direct_ks_null(
    scenario: dict,
    rung: str,
    denominators: np.ndarray,
    down_pool: np.ndarray,
    epa_pool: np.ndarray,
    rz_scale: float,
    n_units: int,
    replicates: int,
    rng: np.random.Generator,
    label: str = "",
    cell_scales: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """KS statistics from `replicates` *fresh* synthetic seasons of `n_units` games.

    An earlier draft estimated one PIT pool per scenario and then bootstrapped
    the KS statistic out of it. That was wrong in a way worth recording: drawing
    n values with replacement from a pool of the same size n convolves the pool's
    own sampling error into the answer, and it inflated the KS null by roughly
    40% — 0.0174 against a directly simulated 0.0122. A tolerance set from the
    inflated null would have been far too permissive, which is the dangerous
    direction for a calibration gate. Each replicate here is an independent
    league, which is the null the gate actually faces.

    Returns the per-replicate KS statistics and the pooled PIT values, the second
    only so the shape diagnostics (coverage, mean) read off a large sample.
    """
    ks = np.empty(replicates)
    pooled: list[np.ndarray] = []
    for replicate in range(replicates):
        pit = pit_distribution_under(
            scenario,
            rung,
            denominators,
            down_pool,
            epa_pool,
            rz_scale,
            n_games=n_units,
            n_draws=N_BAND_DRAWS_POWER,
            rng=rng,
            cell_scales=cell_scales,
        )
        ks[replicate] = ks_statistic(pit)
        pooled.append(pit)
        if replicates >= 20 and (replicate + 1) % 20 == 0:
            print(f"      {label} {replicate + 1}/{replicates}", flush=True)
    return ks, np.concatenate(pooled)


def resampled_ks_null(
    pit_pool: np.ndarray, n: int, replicates: int, rng: np.random.Generator
) -> np.ndarray:
    """KS statistics for `replicates` samples of size `n` drawn from a large pool.

    The tail quantile that becomes the tolerance needs thousands of draws, and
    thousands of *simulated leagues* is hours rather than minutes. Resampling
    from a pooled PIT estimate buys that precision — but only while the pool is
    much larger than the sample. An earlier draft used a pool of exactly `n` and
    inflated the null by roughly 40% (0.0174 against a directly simulated
    0.0122), because drawing n with replacement from n convolves the pool's own
    sampling error into the answer. A tolerance from the inflated null would have
    been far too permissive, which is the dangerous direction for a calibration
    gate. `main` prints the pool-to-sample ratio and the agreement with the
    direct simulation for exactly this reason.
    """
    draws = pit_pool[rng.integers(0, len(pit_pool), size=(replicates, n))]
    u = np.sort(draws, axis=1)
    grid = np.arange(1, n + 1) / n
    return np.maximum((grid - u).max(axis=1), (u - (grid - 1.0 / n)).max(axis=1))


def coverage_power(true_coverage: float, n: int, low: float, high: float) -> float:
    """P(a league of `n` team-games lands outside [low, high]) at a true coverage.

    Coverage — the share of team-games whose realized score falls inside its own
    89% band — is the functional of the PIT the band's copy actually claims, and
    unlike the KS distance it is a binomial with a closed form, so its power
    needs no simulation. The two team-games inside one NFL game are built from
    disjoint sets of plays, so independence is a good approximation here; the
    simulated leagues in `direct_ks_null` carry the exact dependence and their
    coverage spread is printed beside this for comparison.
    """
    sd = float(np.sqrt(true_coverage * (1.0 - true_coverage) / n))
    below = float(_normal_cdf((low - true_coverage) / sd))
    above = 1.0 - float(_normal_cdf((high - true_coverage) / sd))
    return below + above


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def coverage_of(pit_pool: np.ndarray) -> float:
    """Share of games whose realized score lands inside its own 89% band."""
    return float(np.mean((pit_pool > BAND_LOW / 100.0) & (pit_pool < BAND_HIGH / 100.0)))


# --------------------------------------------------------------------------
# part 3 — M-3, persistence power
# --------------------------------------------------------------------------


def season_blocks(meta: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Contiguous team-season starts and sizes, at document 08's 8-game floor.

    ``meta`` arrives in game-id order, so the rows of one team-season are
    scattered through it. The returned ``rows`` is the permutation that makes
    each team-season a contiguous block *and* drops the short ones, and every
    array downstream is indexed by it. Getting this wrong is silent: the
    split-half machinery would happily correlate halves of the wrong groups.
    """
    order = np.argsort(meta["team_season"].to_numpy(), kind="stable")
    keys = meta["team_season"].to_numpy()[order]
    boundaries = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1], True])
    spans = [
        (boundaries[i], boundaries[i + 1])
        for i in range(len(boundaries) - 1)
        if boundaries[i + 1] - boundaries[i] >= MIN_GAMES
    ]
    rows = order[np.concatenate([np.arange(lo, hi) for lo, hi in spans])]
    sizes = np.array([hi - lo for lo, hi in spans])
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    return rows, np.column_stack([starts, sizes])


def split_masks(blocks: np.ndarray, n_rows: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((N_SPLITS, n_rows), dtype=bool)
    for split in range(N_SPLITS):
        for start, size in blocks:
            chosen = rng.permutation(size)[: size // 2] + start
            mask[split, chosen] = True
    return mask


def split_half_r(values: np.ndarray, mask: np.ndarray, blocks: np.ndarray) -> float:
    """Mean split-half correlation of the per-team-game score's half means.

    The half statistic is the **mean placement points per game** over the games
    in that half. The score is already a per-game quantity in points, so a mean
    is the pooling document 08 §2 argued for on gaps: it is the estimator whose
    denominator is real rather than a ratio of two small counts.
    """
    starts = blocks[:, 0]
    counts_a = np.add.reduceat(mask.astype(float), starts, axis=1)
    sums_a = np.add.reduceat(mask * values[None, :], starts, axis=1)
    totals = np.add.reduceat(values, starts)
    sizes = blocks[:, 1].astype(float)
    a = sums_a / counts_a
    b = (totals[None, :] - sums_a) / (sizes[None, :] - counts_a)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    return float(np.mean(num / den))


def m3_permutation_null(
    values: np.ndarray, mask: np.ndarray, blocks: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Real team-game scores dealt at random into synthetic team-seasons.

    Destroys team identity while keeping the real score distribution, the real
    team-season sizes and the real split pattern. The pre-registered M-3
    threshold comes from this null and from nothing else.
    """
    draws = np.empty(N_NULL_REPLICATES)
    for replicate in range(N_NULL_REPLICATES):
        draws[replicate] = split_half_r(values[rng.permutation(len(values))], mask, blocks)
        if (replicate + 1) % 100 == 0:
            print(f"    M-3 permutation null {replicate + 1}/{N_NULL_REPLICATES}", flush=True)
    return draws


def m3_power(
    values: np.ndarray,
    mask: np.ndarray,
    blocks: np.ndarray,
    threshold: float,
    target_r: float,
    rng: np.random.Generator,
) -> dict:
    """Power against a true team-level placement tendency of correlation `target_r`.

    Simulation is at the **team-game** level, never the half level. The executed
    statistic averages over 200 random splits of the *same* games, so its split
    draws are heavily correlated; redrawing noise per half would make them
    independent, shrink the null spread by roughly sqrt(200), and hand the design
    power it does not have. Document 08 §5 records that error being caught.

    The generative story is placement in the literal sense: a team-season carries
    a persistent offset `theta` in placement points per game, and each game adds
    independent noise at the realized per-game dispersion. Total production is
    untouched by construction, because the score is already a within-game
    contrast against the team's own baseline.
    """
    sizes = blocks[:, 1]
    sigma = float(values.std(ddof=1))
    # r = tau^2 / (tau^2 + sigma^2 / games_per_half); solve at the mean half size.
    games_per_half = float(np.mean(sizes)) / 2.0
    tau = float(np.sqrt(target_r * sigma**2 / (games_per_half * (1.0 - target_r))))

    group_of_row = np.repeat(np.arange(len(sizes)), sizes)
    hits = 0
    achieved = np.empty(N_POWER_REPLICATES)
    for replicate in range(N_POWER_REPLICATES):
        theta = rng.normal(0.0, tau, size=len(sizes))
        simulated = theta[group_of_row] + rng.normal(0.0, sigma, size=len(group_of_row))
        r = split_half_r(simulated, mask, blocks)
        achieved[replicate] = r
        hits += int(r > threshold)
        if (replicate + 1) % 100 == 0:
            print(f"    M-3 power {replicate + 1}/{N_POWER_REPLICATES}", flush=True)
    return {
        "target_r": target_r,
        "tau_points_per_game": tau,
        "achieved_mean_r": float(achieved.mean()),
        "power": hits / N_POWER_REPLICATES,
    }


# --------------------------------------------------------------------------
# part 4 — M-4, skill preservation
# --------------------------------------------------------------------------


def m4_null(
    quality: np.ndarray,
    season_mean_sd: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Correlation a genuinely clean baseline produces against real quality.

    The placement scores are redrawn independent of quality at each team-season's
    own sampling SD, then correlated with the *real* offensive-quality vector. So
    the null carries the real quality distribution — its spread, its skew, its
    320 points — and only the relationship is removed. An analytic
    ``1/sqrt(n-3)`` bound would carry none of that.
    """
    n = len(quality)
    q = (quality - quality.mean()) / quality.std(ddof=1)
    draws = rng.normal(0.0, 1.0, size=(N_M4_REPLICATES, n)) * season_mean_sd[None, :]
    draws = draws - draws.mean(axis=1, keepdims=True)
    denom = np.sqrt((draws**2).sum(axis=1) * ((q - q.mean()) ** 2).sum())
    return (draws @ (q - q.mean())) / denom


def m4_power(
    quality: np.ndarray,
    season_mean_sd: np.ndarray,
    bound: float,
    leak: float,
    rng: np.random.Generator,
) -> float:
    """Power to flag a baseline that leaks `leak` worth of correlation with quality."""
    n = len(quality)
    q = (quality - quality.mean()) / quality.std(ddof=1)
    noise = rng.normal(0.0, 1.0, size=(N_M4_REPLICATES, n))
    scale = np.sqrt(max(1.0 - leak**2, 0.0))
    signal = leak * q[None, :] + scale * noise
    draws = signal * season_mean_sd[None, :]
    draws = draws - draws.mean(axis=1, keepdims=True)
    qc = q - q.mean()
    denom = np.sqrt((draws**2).sum(axis=1) * (qc**2).sum())
    r = (draws @ qc) / denom
    return float(np.mean(np.abs(r) > bound))


# --------------------------------------------------------------------------
# part 5 — M-6, does the rematch harness have the width
# --------------------------------------------------------------------------


def m6_power(differential_sd: float, rng: np.random.Generator) -> dict:
    """Two arms, because a non-inferiority gate needs both of its error rates.

    Document 12 measured this harness as nearly blind below roughly 20% damage,
    and the DQW% red-zone arm partly failed on interval width, so neither arm is
    a formality.

    * **Power arm — the meter's own truth.** Placement is exchangeable noise, so
      both the realized margin and the deserved margin carry it and subtracting
      it removes noise rather than signal. The challenger is genuinely no worse,
      and the question is purely whether the interval is narrow enough to show
      it. ``power_to_pass`` is the answer.
    * **False-pass arm — the meter subtracts something that is not there.** The
      challenger has pure noise of the same magnitude *added* to it, so it is
      genuinely worse. A gate that still passes here has no teeth at this
      magnitude no matter what the power arm says, and the two numbers only mean
      something together.

    The 531 real rematch pairs, the ten folds, the seed and the +0.010 margin are
    document 06's, untouched. Only the predictors are simulated.
    """
    _rematch = import_module("08_rematch")
    pairs = _rematch.build_pairs(GAMES_ARTIFACT)
    actual = pairs["margin_g1_a"].to_numpy().astype(float)
    deserved = pairs["deserved_margin"].to_numpy().astype(float)
    y = (pairs["margin_g2_a"].to_numpy() > 0).astype(float)
    a_home = pairs["a_home_g2"].to_numpy().astype(float)

    fold_rng = np.random.default_rng(_rematch.RANDOM_SEED)
    folds = fold_rng.permutation(pairs.height) % _rematch.N_FOLDS

    arms = {
        # incumbent, challenger — as functions of the drawn noise z
        "power_meter_truth": lambda z: (actual + z, deserved),
        "false_pass_noise_added": lambda z: (actual, deserved - z),
    }
    out: dict = {
        "n_pairs": int(pairs.height),
        "differential_sd_points": differential_sd,
        "noninferiority_margin": _rematch_power.NONINFERIORITY_MARGIN,
        "arms": {},
    }
    for name, build in arms.items():
        passes = 0
        means = np.empty(N_M6_REPLICATES)
        uppers = np.empty(N_M6_REPLICATES)
        for replicate in range(N_M6_REPLICATES):
            z = rng.normal(0.0, differential_sd, size=pairs.height)
            incumbent, challenger = build(z)
            per_pair = _rematch_power.paired_log_loss_diff(incumbent, challenger, y, a_home, folds)
            mean, se, _ = _rematch_power.decision(per_pair)
            means[replicate] = mean
            uppers[replicate] = mean + 1.96 * se
            passes += int(_rematch_power.passes_noninferiority(mean, se))
            if (replicate + 1) % 100 == 0:
                print(f"    M-6 {name} {replicate + 1}/{N_M6_REPLICATES}", flush=True)
        out["arms"][name] = {
            "mean_delta_log_loss": float(means.mean()),
            "mean_ci95_upper": float(uppers.mean()),
            "median_ci_half_width": float(np.median(uppers - means)),
            "pass_rate": passes / N_M6_REPLICATES,
        }
    out["power_to_pass"] = out["arms"]["power_meter_truth"]["pass_rate"]
    out["false_pass_rate"] = out["arms"]["false_pass_noise_added"]["pass_rate"]
    return out


# --------------------------------------------------------------------------


def main() -> None:
    paths.ensure_data_dirs()
    rng = np.random.default_rng(RANDOM_SEED)

    print(f"{'=' * 72}\nPART 0 — the luck-priced play stream\n{'=' * 72}")
    plays, reconciliation = load_luck_priced_plays()
    for key, value in reconciliation.items():
        print(f"  {key:<34} {value}")

    arrays = team_game_arrays(plays)
    meta = arrays["meta"]
    scores = placement_scores(meta)
    differentials = game_differentials(meta, scores["score"])

    print(f"\n{'=' * 72}\nPART 1 — design parameters (league-pooled, no team identity)\n{'=' * 72}")
    design = design_parameters(plays, arrays, scores)
    for cell in ("all_plays", "red_zone", "late_down_outside_rz", "other"):
        block = design[cell]
        print(
            f"  {cell:<22} n={block['n']:>7}  mean EPA {block['mean']:+.5f}  var {block['var']:.4f}"
        )
    print(
        f"  variance ratio  red zone / all {design['variance_ratio_rz_over_all']:.4f}   "
        f"late down / all {design['variance_ratio_ld_over_all']:.4f}"
    )
    print(
        f"  league mean offset, points per team-game:  red zone "
        f"{design['mean_offset_rz_points_per_game']:+.3f}   late down "
        f"{design['mean_offset_ld_points_per_game']:+.3f}"
    )
    cells = design["cells_per_team_game"]
    print(
        f"  team-games {cells['n_team_games']}   plays/game {cells['mean_n_all']:.1f}   "
        f"red zone {cells['mean_n_rz']:.2f}   late down {cells['mean_n_ld']:.2f}"
    )
    print(
        f"  share with no red-zone play {cells['share_zero_red_zone'] * 100:.2f}%   "
        f"no leverage play at all {cells['share_zero_leverage'] * 100:.2f}%"
    )
    disp = design["score_dispersion_points"]
    print(
        f"  score dispersion (points): sd {disp['sd']:.3f}  median|.| "
        f"{disp['median_abs']:.3f}  q95|.| {disp['q95_abs']:.3f}  max|.| {disp['max_abs']:.3f}"
    )
    print(f"  three-cell identity, worst residual: {design['identity_residual_max']:.3e} points")
    diff_sd = float(differentials["differential"].std(ddof=1))
    print(
        f"  game differential (home - away): sd {diff_sd:.3f} points over {differentials.height} games"
    )

    # ------------------------------------------------------------------ M-2
    print(f"\n{'=' * 72}\nPART 2 — M-2, the calibration instrument\n{'=' * 72}")
    rz_scale = float(np.sqrt(design["variance_ratio_rz_over_all"]))
    print(f"  rung 3's red-zone stretch factor sqrt(var_rz/var_all) = {rz_scale:.4f}")

    denominators = np.column_stack(
        [
            meta["n_all"].to_numpy(),
            meta["n_rz"].to_numpy(),
            meta["n_ld"].to_numpy(),
        ]
    ).astype(int)
    epa_pool = plays["epa_priced"].to_numpy().astype(float)
    down_pool = plays["down"].to_numpy().astype(np.int64)
    pool_mean = float(epa_pool.mean())

    scenarios = calibration_scenarios(design, pool_mean)

    n_units = int(cells["n_team_games"])
    m2 = {"rz_stretch": rz_scale, "n_units": n_units, "scenarios": {}}
    ks_nulls: dict[tuple[str, str], np.ndarray] = {}
    direct_ks: dict[tuple[str, str], np.ndarray] = {}
    pit_pools: dict[tuple[str, str], np.ndarray] = {}

    for rung in LADDER:
        print(f"\n  --- rung: {rung} ---")
        for scenario in scenarios:
            replicates = (
                N_KS_REPLICATES_EXCHANGEABLE
                if scenario["name"] == "exchangeable"
                else N_KS_REPLICATES_GRADED
            )
            ks, pit = direct_ks_null(
                scenario,
                rung,
                denominators,
                down_pool,
                epa_pool,
                rz_scale,
                n_units,
                replicates,
                rng,
                label=f"{rung}/{scenario['name']}",
            )
            resampled = resampled_ks_null(pit, n_units, N_KS_RESAMPLES, rng)
            ks_nulls[(rung, scenario["name"])] = resampled
            direct_ks[(rung, scenario["name"])] = ks
            pit_pools[(rung, scenario["name"])] = pit
            print(
                f"    {scenario['name']:<16} KS direct {ks.mean():.4f} / resampled "
                f"{resampled.mean():.4f}  95th {np.quantile(resampled, 0.95):.4f}  |  "
                f"89% coverage {coverage_of(pit) * 100:5.2f}%  PIT mean {pit.mean():.4f}  "
                f"(pool/sample {len(pit) / n_units:.0f}x)"
            )
            m2["scenarios"].setdefault(rung, {})[scenario["name"]] = {
                "leagues_simulated": int(replicates),
                "pool_over_sample": float(len(pit) / n_units),
                "ks_direct_mean": float(ks.mean()),
                "ks_resampled_mean": float(resampled.mean()),
                "ks_p95": float(np.quantile(resampled, 0.95)),
                "coverage_89": coverage_of(pit),
                "pit_mean": float(pit.mean()),
                "pit_sd": float(pit.std(ddof=1)),
            }

    # The tolerance is per rung, because each rung's exchangeable-truth null has
    # its own spread: a rung whose band is built from a narrower randomization
    # produces a coarser PIT and a wider KS null, and holding all three to one
    # number would be testing the rung's construction rather than its
    # calibration.
    print(f"\n  --- KS tolerance, from each rung's exchangeable truth at n = {n_units} ---")
    tolerances = {}
    for rung in LADDER:
        null_ks = ks_nulls[(rung, "exchangeable")]
        tolerances[rung] = {
            "p50": float(np.quantile(null_ks, 0.50)),
            "p95": float(np.quantile(null_ks, 0.95)),
            "p99": float(np.quantile(null_ks, 0.99)),
            "coverage_under_truth": coverage_of(pit_pools[(rung, "exchangeable")]),
        }
        print(
            f"    {rung:<30} median {tolerances[rung]['p50']:.4f}  "
            f"95th {tolerances[rung]['p95']:.4f}  99th {tolerances[rung]['p99']:.4f}  "
            f"coverage under truth {tolerances[rung]['coverage_under_truth'] * 100:.2f}%"
        )
    analytic = 1.358 / np.sqrt(n_units)
    print(f"    {'analytic uniform 95th (1.358/sqrt n)':<38} {analytic:.4f}")

    print("\n  --- two-nulls check: resampled KS null vs direct simulation ---")
    agreement = {}
    for rung in LADDER:
        direct = direct_ks[(rung, "exchangeable")]
        resampled = ks_nulls[(rung, "exchangeable")]
        agreement[rung] = {
            "direct_mean": float(direct.mean()),
            "direct_sd": float(direct.std(ddof=1)),
            "resampled_mean": float(resampled.mean()),
            "resampled_sd": float(resampled.std(ddof=1)),
            "ratio": float(resampled.mean() / direct.mean()),
        }
        print(
            f"    {rung:<30} direct {direct.mean():.4f} (sd {direct.std(ddof=1):.4f}, "
            f"{len(direct)} leagues)  resampled {resampled.mean():.4f} "
            f"(sd {resampled.std(ddof=1):.4f})  ratio {agreement[rung]['ratio']:.3f}"
        )
    m2["two_nulls_agreement"] = agreement

    # Materiality: the smallest simulated miscalibration whose consequence moves
    # the 89% band's real coverage by more than two percentage points — two
    # points on the only number the band's copy actually claims. A tolerance
    # below the KS that deviation produces would be a significance gate rather
    # than a materiality gate, which is the failure mode this project's floors
    # exist for.
    print("\n  --- materiality: KS at the smallest coverage-moving miscalibration ---")
    materiality = {}
    for rung in LADDER:
        rows = [
            {
                "scenario": scenario["name"],
                "coverage": coverage_of(pit_pools[(rung, scenario["name"])]),
                "ks_mean": float(ks_nulls[(rung, scenario["name"])].mean()),
            }
            for scenario in scenarios
        ]
        material = [r for r in rows if abs(r["coverage"] - 0.89) > MATERIAL_COVERAGE_SHIFT]
        floor = min((r["ks_mean"] for r in material), default=None)
        materiality[rung] = {"rows": rows, "ks_at_materiality": floor}
        print(f"    {rung:<30} smallest material KS {'n/a' if floor is None else f'{floor:.4f}'}")

    m2["tolerances"] = tolerances
    m2["analytic_uniform_p95"] = float(analytic)
    m2["materiality"] = materiality

    # Power: probability each rung's gate flags each scenario, at that rung's own
    # tolerance. The diagonal (exchangeable) is the false-alarm rate by
    # construction and must land near 0.05.
    print("\n  --- power at each rung's own tolerance ---")
    power_table = {}
    for rung in LADDER:
        tolerance = tolerances[rung]["p95"]
        power_table[rung] = {"tolerance": tolerance, "flag_rate": {}}
        line = []
        for scenario in scenarios:
            flagged = float(np.mean(ks_nulls[(rung, scenario["name"])] > tolerance))
            power_table[rung]["flag_rate"][scenario["name"]] = flagged
            line.append(f"{scenario['name']} {flagged:.2f}")
        print(f"    {rung:<30} tol {tolerance:.4f}  " + "  ".join(line))
    m2["power_at_own_tolerance"] = power_table

    print(f"\n{'=' * 72}\nPART 3 — M-3, persistence power at the team-season grain\n{'=' * 72}")
    rows, blocks = season_blocks(meta)
    values = scores["score"][rows]
    print(f"  {len(blocks)} team-seasons, {len(values)} team-games at the {MIN_GAMES}-game floor")
    mask = split_masks(blocks, len(values), rng)
    null = m3_permutation_null(values, mask, blocks, rng)
    threshold = float(np.quantile(null, 0.95))
    print(
        f"  permutation null: mean {null.mean():+.4f}  sd {null.std(ddof=1):.4f}  "
        f"95th {threshold:.4f}  99th {np.quantile(null, 0.99):.4f}"
    )
    m3 = {
        "n_team_seasons": int(len(blocks)),
        "n_team_games": int(len(values)),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "threshold_p95": threshold,
        "null_p99": float(np.quantile(null, 0.99)),
        "power": {},
    }
    for target in (0.05, 0.08, 0.10, REFERENCE_R, 0.20):
        result = m3_power(values, mask, blocks, threshold, target, rng)
        m3["power"][f"r_{target:.2f}"] = result
        print(
            f"    true r {target:.2f}  tau {result['tau_points_per_game']:.3f} pts/game  "
            f"achieved r {result['achieved_mean_r']:+.4f}  power {result['power']:.3f}"
        )

    # ------------------------------------------------------------------ M-4
    print(f"\n{'=' * 72}\nPART 4 — M-4, skill preservation\n{'=' * 72}")
    season = (
        meta.with_columns(pl.Series("score", scores["score"]))
        .group_by("team_season", maintain_order=True)
        .agg(
            pl.len().alias("games"),
            pl.col("score").mean().alias("mean_score"),
            pl.col("score").std(ddof=1).alias("sd_score"),
            (pl.col("epa_all").sum() / pl.col("n_all").sum()).alias("s0_quality"),
        )
        .filter(pl.col("games") >= MIN_GAMES)
    )
    quality = season["s0_quality"].to_numpy().astype(float)
    season_mean_sd = (season["sd_score"].to_numpy() / np.sqrt(season["games"].to_numpy())).astype(
        float
    )
    print(
        f"  {season.height} team-seasons; offensive quality sd {quality.std(ddof=1):.4f} EPA/play"
    )
    null_r = m4_null(quality, season_mean_sd, rng)
    bound = float(np.quantile(np.abs(null_r), 0.95))
    print(
        f"  true-zero null on |corr|: median {np.median(np.abs(null_r)):.4f}  "
        f"95th {bound:.4f}  99th {np.quantile(np.abs(null_r), 0.99):.4f}"
    )
    m4 = {
        "n_team_seasons": int(season.height),
        "null_abs_p50": float(np.median(np.abs(null_r))),
        "bound_abs_p95": bound,
        "null_abs_p99": float(np.quantile(np.abs(null_r), 0.99)),
        "power": {},
    }
    for leak in (0.05, 0.10, 0.11, 0.15, 0.20, 0.30):
        power = m4_power(quality, season_mean_sd, bound, leak, rng)
        m4["power"][f"leak_{leak:.2f}"] = power
        print(f"    true leak corr {leak:.2f}  power to flag {power:.3f}")

    # ------------------------------------------------------------------ M-6
    print(f"\n{'=' * 72}\nPART 5 — M-6, does the rematch harness have the width\n{'=' * 72}")
    m6 = m6_power(diff_sd, rng)
    print(
        f"  {m6['n_pairs']} pairs, placement differential sd "
        f"{m6['differential_sd_points']:.3f} points, margin "
        f"{m6['noninferiority_margin']:+.3f}"
    )
    for name, arm in m6["arms"].items():
        print(
            f"    {name:<24} mean delta log loss {arm['mean_delta_log_loss']:+.5f}  "
            f"mean 95% upper {arm['mean_ci95_upper']:+.5f}  "
            f"half-width {arm['median_ci_half_width']:.5f}  pass {arm['pass_rate']:.3f}"
        )
    print(f"  POWER TO PASS under the meter's own truth: {m6['power_to_pass']:.3f}")
    print(f"  FALSE-PASS RATE when the subtraction is pure harm: {m6['false_pass_rate']:.3f}")

    payload = {
        "random_seed": RANDOM_SEED,
        "reconciliation": reconciliation,
        "design": design,
        "game_differential_sd": diff_sd,
        "m2_calibration": m2,
        "m3_persistence": m3,
        "m4_skill_preservation": m4,
        "m6_rematch": m6,
        "constants": {
            "points_per_epa": POINTS_PER_EPA,
            "red_zone_yards": RED_ZONE_YARDS,
            "late_downs": list(LATE_DOWNS),
            "n_band_draws": N_BAND_DRAWS,
            "n_band_draws_power": N_BAND_DRAWS_POWER,
            "band_interval": [BAND_LOW, BAND_HIGH],
            "n_splits": N_SPLITS,
            "min_games": MIN_GAMES,
            "reference_r": REFERENCE_R,
            "ladder": list(LADDER),
            "adoption_ladder": list(ADOPTION_LADDER),
        },
    }
    out = paths.RESEARCH_OUTPUT_DIR / "49_placement_power.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")


def ladder_addendum() -> None:
    """Simulate the amendment rung's M-2 power and merge it into a finished run.

    Rung 4 was approved after `main` had already run and after document 35 had
    committed M-3's threshold and M-4's bound. `main` consumes one generator rung
    by rung and then hands it to M-3, M-4 and M-6, so simulating the new rung
    inside that loop would shift every downstream draw and move two thresholds
    that are already on the record — a goalpost move by accident rather than by
    intent.

    So the new rung runs here instead, on its own generator seeded
    ``RANDOM_SEED + 4``, and only *adds* keys to the results file. Nothing already
    written is recomputed, which makes the integrity claim checkable rather than
    promised: the file's existing numbers are the same bytes they were before.

        uv run python research/49_placement_power.py --ladder-addendum
    """
    out = paths.RESEARCH_OUTPUT_DIR / "49_placement_power.json"
    payload = json.loads(out.read_text())
    before = (
        json.dumps(payload["m3_persistence"], sort_keys=True),
        json.dumps(payload["m4_skill_preservation"], sort_keys=True),
    )

    rng = np.random.default_rng(RANDOM_SEED + 4)
    plays, _ = load_luck_priced_plays()
    arrays = team_game_arrays(plays)
    meta = arrays["meta"]
    scores = placement_scores(meta)
    design = design_parameters(plays, arrays, scores)

    rz_scale = float(np.sqrt(design["variance_ratio_rz_over_all"]))
    ld_scale = float(np.sqrt(design["variance_ratio_ld_over_all"]))
    other_scale = float(np.sqrt(design["other"]["var"] / design["all_plays"]["var"]))
    cell_scales = (rz_scale, ld_scale, other_scale)
    print(
        f"  rung 4 stretch factors: red zone {rz_scale:.4f}   late down {ld_scale:.4f}   "
        f"everything else {other_scale:.4f}"
    )

    denominators = np.column_stack(
        [meta["n_all"].to_numpy(), meta["n_rz"].to_numpy(), meta["n_ld"].to_numpy()]
    ).astype(int)
    epa_pool = plays["epa_priced"].to_numpy().astype(float)
    down_pool = plays["down"].to_numpy().astype(np.int64)
    scenarios = calibration_scenarios(design, float(epa_pool.mean()))

    m2 = payload["m2_calibration"]
    n_units = int(m2["n_units"])
    ks_nulls: dict[str, np.ndarray] = {}
    pit_pools: dict[str, np.ndarray] = {}

    for rung in LADDER_ADDENDUM:
        print(f"\n  --- rung: {rung} ---")
        for scenario in scenarios:
            replicates = (
                N_KS_REPLICATES_EXCHANGEABLE
                if scenario["name"] == "exchangeable"
                else N_KS_REPLICATES_GRADED
            )
            ks, pit = direct_ks_null(
                scenario,
                rung,
                denominators,
                down_pool,
                epa_pool,
                rz_scale,
                n_units,
                replicates,
                rng,
                label=f"{rung}/{scenario['name']}",
                cell_scales=cell_scales,
            )
            resampled = resampled_ks_null(pit, n_units, N_KS_RESAMPLES, rng)
            ks_nulls[scenario["name"]] = resampled
            pit_pools[scenario["name"]] = pit
            print(
                f"    {scenario['name']:<16} KS direct {ks.mean():.4f} / resampled "
                f"{resampled.mean():.4f}  95th {np.quantile(resampled, 0.95):.4f}  |  "
                f"89% coverage {coverage_of(pit) * 100:5.2f}%  PIT mean {pit.mean():.4f}"
            )
            m2["scenarios"].setdefault(rung, {})[scenario["name"]] = {
                "leagues_simulated": int(replicates),
                "pool_over_sample": float(len(pit) / n_units),
                "ks_direct_mean": float(ks.mean()),
                "ks_resampled_mean": float(resampled.mean()),
                "ks_p95": float(np.quantile(resampled, 0.95)),
                "coverage_89": coverage_of(pit),
                "pit_mean": float(pit.mean()),
                "pit_sd": float(pit.std(ddof=1)),
            }
            if scenario["name"] == "exchangeable":
                m2["two_nulls_agreement"][rung] = {
                    "direct_mean": float(ks.mean()),
                    "direct_sd": float(ks.std(ddof=1)),
                    "resampled_mean": float(resampled.mean()),
                    "resampled_sd": float(resampled.std(ddof=1)),
                    "ratio": float(resampled.mean() / ks.mean()),
                }

        tolerance = {
            "p50": float(np.quantile(ks_nulls["exchangeable"], 0.50)),
            "p95": float(np.quantile(ks_nulls["exchangeable"], 0.95)),
            "p99": float(np.quantile(ks_nulls["exchangeable"], 0.99)),
            "coverage_under_truth": coverage_of(pit_pools["exchangeable"]),
        }
        m2["tolerances"][rung] = tolerance
        print(
            f"\n    KS null under its own exchangeable truth: median {tolerance['p50']:.4f}  "
            f"95th {tolerance['p95']:.4f}  99th {tolerance['p99']:.4f}  "
            f"coverage under truth {tolerance['coverage_under_truth'] * 100:.2f}%"
        )

        rows = [
            {
                "scenario": scenario["name"],
                "coverage": coverage_of(pit_pools[scenario["name"]]),
                "ks_mean": float(ks_nulls[scenario["name"]].mean()),
            }
            for scenario in scenarios
        ]
        material = [r for r in rows if abs(r["coverage"] - 0.89) > MATERIAL_COVERAGE_SHIFT]
        m2["materiality"][rung] = {
            "rows": rows,
            "ks_at_materiality": min((r["ks_mean"] for r in material), default=None),
        }
        m2["power_at_own_tolerance"][rung] = {
            "tolerance": tolerance["p95"],
            "flag_rate": {
                scenario["name"]: float(np.mean(ks_nulls[scenario["name"]] > tolerance["p95"]))
                for scenario in scenarios
            },
        }

        # The gate with teeth is coverage, not KS — closed-form binomial, same
        # tolerance the other three rungs were graded against.
        low, high = 0.89 - MATERIAL_COVERAGE_SHIFT, 0.89 + MATERIAL_COVERAGE_SHIFT
        gate = m2.setdefault(
            "coverage_gate",
            {
                "tolerance_low": low,
                "tolerance_high": high,
                "binomial_sd_pp": float(np.sqrt(0.89 * 0.11 / n_units) * 100),
                "table": {},
            },
        )
        gate["table"][rung] = {}
        print("\n    coverage gate, the one with teeth:")
        for scenario in scenarios:
            coverage = coverage_of(pit_pools[scenario["name"]])
            flagged = coverage_power(coverage, n_units, low, high)
            gate["table"][rung][scenario["name"]] = {
                "true_coverage": coverage,
                "flag_rate": flagged,
            }
            verdict = "FLAGGED" if not low <= coverage <= high else "passes"
            print(
                f"      {scenario['name']:<16} true coverage {coverage * 100:6.2f}%  "
                f"P(flag a single league) {flagged:.3f}   [{verdict} at truth]"
            )

    m2["rung4_cell_stretch"] = {
        "red_zone": rz_scale,
        "late_down": ld_scale,
        "other": other_scale,
    }
    payload["constants"]["ladder_addendum"] = list(LADDER_ADDENDUM)
    payload["constants"]["adoption_ladder"] = list(ADOPTION_LADDER)
    payload["constants"]["addendum_seed"] = RANDOM_SEED + 4

    after = (
        json.dumps(payload["m3_persistence"], sort_keys=True),
        json.dumps(payload["m4_skill_preservation"], sort_keys=True),
    )
    assert before == after, "the addendum must not touch a committed threshold"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  M-3 and M-4 blocks verified untouched.\nupdated {out}")


def coverage_addendum() -> None:
    """Recompute M-2's coverage table from a finished run, without re-simulating.

    The expensive half of this script is the 300 simulated leagues behind Part 2.
    The coverage gate's power is a closed-form binomial in each scenario's true
    coverage, which that run already stored, so it is recovered here rather than
    paid for twice.

        uv run python research/49_placement_power.py --coverage-only
    """
    out = paths.RESEARCH_OUTPUT_DIR / "49_placement_power.json"
    payload = json.loads(out.read_text())
    n_units = int(payload["m2_calibration"]["n_units"])
    low, high = 0.89 - MATERIAL_COVERAGE_SHIFT, 0.89 + MATERIAL_COVERAGE_SHIFT

    print(
        f"  coverage gate: the 89% band must cover {low * 100:.1f}-{high * 100:.1f}% "
        f"of {n_units} team-games\n"
        f"  binomial sd at true 89%: {np.sqrt(0.89 * 0.11 / n_units) * 100:.3f} pp "
        f"— the tolerance is {MATERIAL_COVERAGE_SHIFT / np.sqrt(0.89 * 0.11 / n_units):.1f} "
        f"sd wide, so it is a materiality floor and not a significance test"
    )
    table = {}
    for rung, scenarios in payload["m2_calibration"]["scenarios"].items():
        table[rung] = {}
        print(f"\n  --- rung: {rung} ---")
        for name, block in scenarios.items():
            coverage = block["coverage_89"]
            flagged = coverage_power(coverage, n_units, low, high)
            table[rung][name] = {"true_coverage": coverage, "flag_rate": flagged}
            verdict = "FLAGGED" if not low <= coverage <= high else "passes"
            print(
                f"    {name:<16} true coverage {coverage * 100:6.2f}%  "
                f"P(flag a single league) {flagged:.3f}   [{verdict} at truth]"
            )
    payload["m2_calibration"]["coverage_gate"] = {
        "tolerance_low": low,
        "tolerance_high": high,
        "binomial_sd_pp": float(np.sqrt(0.89 * 0.11 / n_units) * 100),
        "table": table,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nupdated {out}")


if __name__ == "__main__":
    if "--coverage-only" in sys.argv:
        coverage_addendum()
    elif "--ladder-addendum" in sys.argv:
        ladder_addendum()
    else:
        main()
