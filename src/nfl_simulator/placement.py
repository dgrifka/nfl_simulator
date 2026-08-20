"""The placement meter — how many points a team's play placement was worth.

Designed in ``docs/research/34-placement-meter-design.md`` and pre-registered,
thresholds and all, in ``docs/research/35-placement-meter-prereg.md``. One
team-game produces one number: the EPA a team put into the two leverage cells —
the red zone, and late downs outside it — above what that same team's *own*
overall efficiency in that same game would have put there, converted to points.

    cell_points = ( sum(epa_priced in cell) - n_cell * mean_all ) * POINTS_PER_EPA

Three things about that line carry the design.

* **The baseline is the team's own game.** A team that is uniformly good scores
  zero. The skill-destruction channel that killed DQW% (document 08 §10-11) is
  closed at the definition rather than patched afterwards.
* **It is written as a sum minus a count times a mean**, not as a difference of
  means times a count, so an empty cell is exactly ``0.0`` instead of ``0/0``.
* **The three cells sum to zero**, which is this design's round-trip check.

The meter is descriptive. It books no ledger rows, so document 05's Gate A is
never in play, and it is displayed *beside* DTW% and never added to it — the
two-meters contract of document 08 §6. Nothing here re-simulates anything:
labels are permuted on realized plays, no play is replayed.

Special-teams placement is outside the meter, because it is outside the
S0-S2 filter and was never tested by document 08.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

# Document 27 §13's fitted slope, the currency every luck number in this project
# is quoted in. Hard-coded rather than refitted so the meter and the ledger price
# a point identically.
POINTS_PER_EPA = 0.8389

RED_ZONE_YARDS = 20
LATE_DOWNS = (3, 4)

CELL_RED_ZONE = 0
CELL_LATE_DOWN = 1
CELL_OTHER = 2

# Document 34 §4: ~2,000 draws, 89% equal-tailed, house convention.
N_BAND_DRAWS = 2000
BAND_LOW, BAND_HIGH = 5.5, 94.5

# The constraint ladder in **adoption order** — least constrained first, where
# "constrained" means how much of the realized configuration the null holds
# fixed. Document 35 §5 commits this order and §6's adoption rule walks it:
# adopt the first rung whose coverage lands inside the tolerance on real data.
LADDER = (
    "raw",
    "raw_var_matched",
    "down_stratified",
    "down_stratified_var_matched",
)


# --------------------------------------------------------------------------
# the input stream
# --------------------------------------------------------------------------


def assign_cells(plays: pl.DataFrame) -> pl.DataFrame:
    """Add the disjoint cell label, in document 35 §2's order of application.

    Red zone first and on any down, so a third-and-goal is a red-zone play and
    not a late-down one. Disjoint because the score has to sum: document 08's
    S1/S2 overlap was fine for two season-level tests, not for one number.
    """
    return plays.with_columns(
        pl.when(pl.col("yardline_100") <= RED_ZONE_YARDS)
        .then(CELL_RED_ZONE)
        .when(pl.col("down").is_in(LATE_DOWNS))
        .then(CELL_LATE_DOWN)
        .otherwise(CELL_OTHER)
        .alias("cell")
    )


def price_plays(plays: pl.DataFrame, ledger: pl.DataFrame) -> pl.DataFrame:
    """Add ``epa_priced`` = ``epa - luck_epa``, the luck-priced stream.

    Document 34 §5's double-counting fix, applied at the input rather than
    patched at the output: a fumble's coin swing lives only in the DTW ledger, so
    the meter must see that play at its expectation.

    The one place this project's two sign conventions meet. ``luck_epa`` arrives
    signed to the **home** team, because that is what makes the ledger's sum
    equal the whole adjustment the simulator applies; ``epa`` is nflverse's,
    signed to the **possessing** team. A play with several ledger rows is summed
    before the subtraction; a play with none is left exactly as it was.
    """
    per_play = ledger.group_by(["game_id", "play_id"]).agg(
        pl.col("luck_epa").sum().alias("luck_epa_home")
    )
    return (
        plays.join(per_play, on=["game_id", "play_id"], how="left")
        .with_columns(pl.col("luck_epa_home").fill_null(0.0))
        .with_columns(
            pl.when(pl.col("posteam") == pl.col("home_team"))
            .then(pl.col("luck_epa_home"))
            .otherwise(-pl.col("luck_epa_home"))
            .alias("luck_epa_pos")
        )
        .with_columns((pl.col("epa") - pl.col("luck_epa_pos")).alias("epa_priced"))
    )


# --------------------------------------------------------------------------
# the score
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PlacementScore:
    """One team-game's placement points, and the parts that must sum to zero."""

    red_zone: float
    late_down: float
    other: float
    n_all: int
    n_red_zone: int
    n_late_down: int

    @property
    def score(self) -> float:
        """The meter: the sum over the two leverage cells."""
        return self.red_zone + self.late_down

    @property
    def identity_residual(self) -> float:
        """All three cells sum to zero. M-1 reads this."""
        return self.red_zone + self.late_down + self.other


def score_team_game(epa_priced: np.ndarray, cell: np.ndarray) -> PlacementScore:
    """Placement points for one team-game."""
    n_all = len(epa_priced)
    if n_all == 0:
        return PlacementScore(0.0, 0.0, 0.0, 0, 0, 0)

    mean_all = float(epa_priced.mean())

    def cell_points(mask: np.ndarray) -> float:
        return (float(epa_priced[mask].sum()) - int(mask.sum()) * mean_all) * POINTS_PER_EPA

    is_rz = cell == CELL_RED_ZONE
    is_ld = cell == CELL_LATE_DOWN
    return PlacementScore(
        red_zone=cell_points(is_rz),
        late_down=cell_points(is_ld),
        other=cell_points(cell == CELL_OTHER),
        n_all=n_all,
        n_red_zone=int(is_rz.sum()),
        n_late_down=int(is_ld.sum()),
    )


def game_differential(home: PlacementScore, away: PlacementScore) -> float:
    """The game's headline: home placement points minus away.

    Margin scale, and never addable to the margin — the copy has to say so.
    """
    return home.score - away.score


# --------------------------------------------------------------------------
# the permutation band
# --------------------------------------------------------------------------
#
# The score is the sum over the two *leverage* cells, so under any label
# permutation only the leverage **union** matters, never the split between red
# zone and late down:
#
#     score = ppe * ( sum(epa over leverage plays) - k * mean_all ),  k = n_rz + n_ld
#
# Every rung below computes that, and it is why the raw rung reduces to "draw k
# of n plays without replacement".


def _draws_raw(epa: np.ndarray, cell: np.ndarray, n_draws: int, rng) -> np.ndarray:
    """Rung 1 — every play exchangeable across all three cells."""
    n = len(epa)
    k = int(np.count_nonzero(cell != CELL_OTHER))
    shuffled = rng.permuted(np.broadcast_to(epa, (n_draws, n)).copy(), axis=1)
    return (shuffled[:, :k].sum(axis=1) - k * epa.mean()) * POINTS_PER_EPA


def _draws_raw_var_matched(
    epa: np.ndarray, cell: np.ndarray, n_draws: int, rng, cell_scales
) -> np.ndarray:
    """Rung 4 — rung 1's shuffle with every cell's second moment restored.

    Approved by the maintainer 2026-08-19 and added to the ladder by amendment. It is the
    only rung that both re-randomizes the late-down cell — so it is a null for
    the meter as a whole — and prices that cell at its own second moment, which
    document 35 §3 measures at 1.99x the league.

    The third cell is stretched too, at 0.82, because the score's baseline is the
    team-game mean over *all* plays: a correction applied to the leverage cells
    alone lands in the baseline as well and inflates the null. With ``dev`` the
    deviations from the team-game mean and ``a``, ``b`` one draw's red-zone and
    late-down deviation sums, the third cell's sum is ``-(a + b)`` because
    deviations sum to zero, and the score reduces to

        ppe * [ (1 - k/n) * (s_rz * a + s_ld * b) + (k/n) * s_other * (a + b) ]

    which is rung 1 exactly when every scale is 1.
    """
    n = len(epa)
    n_rz = int(np.count_nonzero(cell == CELL_RED_ZONE))
    n_ld = int(np.count_nonzero(cell == CELL_LATE_DOWN))
    k = n_rz + n_ld
    s_rz, s_ld, s_other = cell_scales

    dev = epa - epa.mean()
    order = np.argsort(rng.random((n_draws, n)), axis=1)
    a = dev[order[:, :n_rz]].sum(axis=1) if n_rz else np.zeros(n_draws)
    b = dev[order[:, n_rz:k]].sum(axis=1) if n_ld else np.zeros(n_draws)
    share = k / n
    return ((1.0 - share) * (s_rz * a + s_ld * b) + share * s_other * (a + b)) * POINTS_PER_EPA


def _draws_down_stratified(
    epa: np.ndarray, cell: np.ndarray, down: np.ndarray, n_draws: int, rng, rz_scale: float
) -> np.ndarray:
    """Rungs 2 and 3 — plays keep their down; only field-position membership moves.

    Document 35 §5 derives what this costs, before any data was seen: the
    leverage union contains *every* third- and fourth-down play no matter where
    the red-zone labels land, so holding down fixed freezes the entire late-down
    contribution and leaves only "which early-down plays were red-zone plays"
    random. These two rungs are a null for red-zone placement alone, not for the
    meter as a whole, and that is reported wherever their band is displayed.

    ``rz_scale`` is rung 3: red-zone-assigned plays have their deviations from
    their own down-stratum's mean stretched, which inflates the second moment and
    leaves the first exactly where it was. The stretch touches the null only.
    """
    k = int(np.count_nonzero(cell != CELL_OTHER))
    mean_all = epa.mean()
    late = np.isin(down, LATE_DOWNS)
    is_rz = cell == CELL_RED_ZONE
    stretching = rz_scale != 1.0

    # Under rung 2 every late-down play sits in the leverage union whatever the
    # draw does, so its contribution is a constant. Under rung 3 a late-down play
    # that lands in the red zone is stretched, so it cannot be pre-summed.
    totals = np.full(n_draws, 0.0 if stretching else float(epa[late].sum()))

    for d in np.unique(down):
        idx = np.flatnonzero(down == d)
        block = epa[idx]
        n_d = len(idx)
        n_rz_d = int(np.count_nonzero(is_rz[idx]))
        late_d = bool(d in LATE_DOWNS)

        if not stretching:
            if late_d or n_rz_d == 0:
                continue  # already counted whole, or nothing to draw
            if n_rz_d == n_d:
                totals += block.sum()
                continue
            shuffled = rng.permuted(np.broadcast_to(block, (n_draws, n_d)).copy(), axis=1)
            totals += shuffled[:, :n_rz_d].sum(axis=1)
            continue

        block_mean = block.mean()
        stretched = block_mean + rz_scale * (block - block_mean)
        if n_rz_d == 0:
            totals += block.sum() if late_d else 0.0
            continue
        if n_rz_d == n_d:
            totals += stretched.sum()
            continue
        order = np.argsort(rng.random((n_draws, n_d)), axis=1)
        totals += stretched[order[:, :n_rz_d]].sum(axis=1)
        if late_d:
            totals += block[order[:, n_rz_d:]].sum(axis=1)

    return (totals - k * mean_all) * POINTS_PER_EPA


def permutation_draws(
    rung: str,
    epa_priced: np.ndarray,
    cell: np.ndarray,
    down: np.ndarray,
    n_draws: int = N_BAND_DRAWS,
    rng=None,
    cell_scales: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Null placement scores for one team-game under one rung of the ladder.

    The band answers *how big a placement-points number does chance produce with
    this same game's production?* — not a per-game hypothesis test and not an
    interval on an estimate. The realized score is exact; the band is the
    reference scale that stops +-2 points from being read as meaningful in a game
    where chance spans +-4.

    A team-game with no leverage plays — or with nothing but — has no placement
    to randomize, and its null is degenerate at zero.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(epa_priced)
    k = int(np.count_nonzero(cell != CELL_OTHER))
    if n == 0 or k == 0 or k == n:
        return np.zeros(n_draws)

    if rung == "raw":
        return _draws_raw(epa_priced, cell, n_draws, rng)
    if rung == "raw_var_matched":
        return _draws_raw_var_matched(epa_priced, cell, n_draws, rng, cell_scales)
    if rung == "down_stratified":
        return _draws_down_stratified(epa_priced, cell, down, n_draws, rng, 1.0)
    if rung == "down_stratified_var_matched":
        return _draws_down_stratified(epa_priced, cell, down, n_draws, rng, cell_scales[0])
    raise ValueError(f"unknown rung {rung!r}")


def band_from_draws(draws: np.ndarray) -> tuple[float, float]:
    """The 89% equal-tailed band, house convention."""
    return (
        float(np.quantile(draws, BAND_LOW / 100.0)),
        float(np.quantile(draws, BAND_HIGH / 100.0)),
    )


def pit_of(realized: float, draws: np.ndarray) -> float:
    """Mid-P percentile of the realized score inside its own game's null.

    Mid-P rather than a plain rank because a permutation null is discrete: with
    ties broken to one side the PIT is not uniform even under exact
    exchangeability, and M-2 would then be testing the tie-breaking rule.
    """
    below = float(np.count_nonzero(draws < realized))
    equal = float(np.count_nonzero(draws == realized))
    return (below + 0.5 * equal) / len(draws)
