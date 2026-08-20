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


def score_team_game(
    epa_priced: np.ndarray, cell: np.ndarray, mu: np.ndarray | None = None
) -> PlacementScore:
    """Placement points for one team-game.

    ``mu`` is document 36 §2's expected profile — the EPA per play a team of this
    quality produces in each cell, fitted league-wide and leave-one-team-out. Each
    cell is centred on it *before* the count multiplies anything, and the baseline
    is taken over the centred values:

        cell_points = ( sum(epa in cell) - n_cell*mu_cell - n_cell*baseline ) * ppe
        baseline    = sum over cells of ( sum(epa in cell) - n_cell*mu_cell ) / n_all

    With ``mu`` left at ``None`` — or set to zeros — this is document 35's score,
    character for character, because the baseline collapses to the team-game mean.
    The redesign is a strict generalisation of what it replaces, which is what
    lets it inherit that design's defences rather than re-argue them.
    """
    n_all = len(epa_priced)
    if n_all == 0:
        return PlacementScore(0.0, 0.0, 0.0, 0, 0, 0)

    profile = np.zeros(3) if mu is None else np.asarray(mu, dtype=float)
    masks = [cell == CELL_RED_ZONE, cell == CELL_LATE_DOWN, cell == CELL_OTHER]
    counts = np.array([int(mask.sum()) for mask in masks], dtype=float)
    sums = np.array([float(epa_priced[mask].sum()) for mask in masks])
    centred = sums - counts * profile
    baseline = centred.sum() / n_all
    points = (centred - counts * baseline) * POINTS_PER_EPA

    return PlacementScore(
        red_zone=float(points[0]),
        late_down=float(points[1]),
        other=float(points[2]),
        n_all=n_all,
        n_red_zone=int(counts[0]),
        n_late_down=int(counts[1]),
    )


# --------------------------------------------------------------------------
# the expected profile — document 36 §2
# --------------------------------------------------------------------------
#
# The incumbent scored a cell against the team's own game-wide mean. That bar
# carries the league's structural profile — late downs run 0.048 EPA per play
# below the league — and the score is n_cell-scaled, so every team-game was
# dragged in proportion to how often that team was in a situation its own quality
# had put it in. Bad offences face far more third downs, and gate M-4 read the
# resulting leak at +0.5435 against a bound of 0.1065.
#
# Document 36 §5 derives what can and cannot fix that. The leak is a product of
# two factors — a structural cell offset and a quality-correlated cell count — and
# replacing the realised count with the count quality predicts leaves the second
# factor exactly as quality-correlated as it was. Only the offset can be zeroed,
# and centring on the fitted profile zeroes it by construction.


def leave_one_out_rate(total: np.ndarray, count: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Each row's group rate computed *without* that row — ``s0_loo`` in §2.

    The game being scored never enters its own baseline. This is document 05 §5's
    contamination defence applied at the input rather than bounded after the fact.
    A group with a single row has no rest-of-group and returns ``nan`` rather than
    a quietly wrong zero.
    """
    _, inverse = np.unique(np.asarray(group), return_inverse=True)
    total_rest = np.bincount(inverse, weights=total)[inverse] - np.asarray(total, dtype=float)
    count_rest = np.bincount(inverse, weights=count)[inverse] - np.asarray(count, dtype=float)
    return np.divide(
        total_rest, count_rest, out=np.full(len(total_rest), np.nan), where=count_rest > 0
    )


def loto_weighted_linear_fit(
    y_mean: np.ndarray, weight: np.ndarray, x: np.ndarray, group: np.ndarray
) -> np.ndarray:
    """Weighted least squares of ``y_mean`` on ``[1, x]``, leave-one-**group**-out.

    Returns a fitted value for every row, from coefficients estimated without any
    row belonging to that row's group. Two things ride on the details, and both
    are deliberate.

    * **The weight is the count the score multiplies that cell by**, which makes
      the fit's own orthogonality condition *be* the leak condition rather than
      merely resemble it: the normal equations set ``sum(weight * residual * x)``
      to zero, and ``weight * residual`` is exactly what the score sums.
    * **The fold is the franchise, not the game.** A fit that included the team
      would make gate M-4 read its own arithmetic — in-sample residuals are
      orthogonal to ``x`` by the normal equations, so the gate would be reading a
      guarantee rather than evidence. Out-of-fold residuals carry no such
      guarantee, and document 36 §7 measures what is left.

    Held out one group at a time by *subtracting* that group's contribution from
    the pooled sufficient statistics, so the cost is one pass rather than one
    refit per franchise.
    """
    y_mean = np.asarray(y_mean, dtype=float)
    weight = np.asarray(weight, dtype=float)
    x = np.asarray(x, dtype=float)
    group = np.asarray(group)

    pooled = np.array(
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
        s_w, s_wx, s_wxx, s_y, s_yx = pooled - np.array(
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
    count and each estimated leave-one-team-out. A team-game with no play in a
    cell has a cell mean of 0/0; it enters at weight zero, so the undefined mean
    never reaches the arithmetic.
    """
    counts = np.asarray(counts, dtype=float)
    sums = np.asarray(sums, dtype=float)
    mean_cell = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return np.column_stack(
        [loto_weighted_linear_fit(mean_cell[:, c], counts[:, c], s0, group) for c in range(3)]
    )


def redesigned_cell_points(
    n_all: np.ndarray, counts: np.ndarray, sums: np.ndarray, mu: np.ndarray
) -> np.ndarray:
    """``points[row, cell]`` — ``score_team_game``'s arithmetic over many team-games.

    The score reads only each team-game's cell counts and cell EPA sums, so the
    gates run on this form rather than looping over play arrays. It is checked
    against the per-team-game function rather than assumed to agree with it.
    """
    counts = np.asarray(counts, dtype=float)
    centred = np.where(counts > 0, np.asarray(sums, dtype=float) - counts * mu, 0.0)
    baseline = centred.sum(axis=1) / np.asarray(n_all, dtype=float)
    return (centred - counts * baseline[:, None]) * POINTS_PER_EPA


def profile_shift(counts: np.ndarray, mu: np.ndarray, n_all: np.ndarray) -> np.ndarray:
    """The per-team-game constant the redesign subtracts from the incumbent score.

    Document 36 §6's carry-forward proof in one line: with the three cell sizes
    held fixed — which every rung of the ladder does — the profile a draw
    subtracts is the same whichever plays land where, so

        redesigned_score = incumbent_score - C

    for the realised score **and every null draw alike**. The PIT is a rank of the
    realised score inside its own draws, ranks are invariant to a common shift, so
    M-2's coverage, its rung adoption and its power table carry forward untouched.
    """
    counts = np.asarray(counts, dtype=float)
    mu = np.asarray(mu, dtype=float)
    leverage = counts[:, CELL_RED_ZONE] + counts[:, CELL_LATE_DOWN]
    in_leverage = (
        counts[:, CELL_RED_ZONE] * mu[:, CELL_RED_ZONE]
        + counts[:, CELL_LATE_DOWN] * mu[:, CELL_LATE_DOWN]
    )
    weighted_mean_mu = (counts * mu).sum(axis=1) / np.asarray(n_all, dtype=float)
    return (in_leverage - leverage * weighted_mean_mu) * POINTS_PER_EPA


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
