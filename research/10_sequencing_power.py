"""Phase 3, step 1 — power calculation for the sequencing-luck round.

Document 04's closing lesson, restated as a process law: a threshold set from an
effect-size argument, with nobody asking whether the data *could* achieve it, is
a gate that fails for reasons unrelated to the hypothesis. So this script runs
**before** `docs/research/08-sequencing-luck.md` commits any threshold.

It answers one question per measure: **with the plays that actually exist, how
large would a team's true sequencing tendency have to be before a split-half
test could see it?**

Nothing here computes the real split-half correlations. The design parameters
are league-pooled play-level variances and the real per-team-game denominators;
the persistence itself is simulated. The one quantity measured from the joint
distribution is the league ``wpa_per_epa`` slope, which is a property of the two
nflverse models and not of any team.

Two nulls, because one alone is arguable:

* **Permutation null** — real team-games dealt at random into synthetic
  team-seasons. Destroys team identity while preserving every within-game
  correlation, the real denominators and the real fat-tailed play distribution.
  The pre-registered thresholds come from this one.
* **Parametric null** — the same statistic built from analytic sampling noise at
  the same denominators. Used only to extend to a power curve at a true team
  spread ``tau > 0``, which a permutation cannot produce. Its agreement with the
  permutation null is reported; disagreement is a reason to distrust the power
  column, not the thresholds.

    uv run python research/10_sequencing_power.py
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from nfl_simulator import paths
from nfl_simulator.ingest import PBP_SEASONS, load_pbp

RANDOM_SEED = 20260817

# Split draws inside one split-half estimate. Document 02 used 200 and reported
# the mean across draws; the same protocol runs here, so the null distribution
# belongs to the *executed* statistic rather than to a cheaper cousin.
N_SPLITS = 200

N_NULL_REPLICATES = 500
N_POWER_REPLICATES = 500

RED_ZONE_YARDS = 20
LATE_DOWNS = (3, 4)

# A team-season needs enough games that both halves hold real plays. Document 02
# used 8 for the same reason.
MIN_GAMES = 8

# The smallest split-half correlation this project has ever called a real
# effect: document 02's middle three components sat at r = 0.12-0.16 and were
# described as "mostly noise, real skill inside". A measure whose design cannot
# detect 0.12 cannot distinguish "no sequencing skill" from "sequencing skill
# the size of penalty discipline", and must not be reported as if it could.
REFERENCE_R = 0.12

MEASURES: tuple[str, ...] = (
    "S0_overall_epa",
    "S1_redzone_gap",
    "S2_latedown_gap",
    "S3_wpa_epa_gap",
)

# Column order of the dense team-game matrix every routine below indexes into.
SUM_COLUMNS: tuple[str, ...] = (
    "n_all",
    "epa_all",
    "succ_all",
    "n_rz",
    "epa_rz",
    "n_ld",
    "succ_ld",
    "epa_valued",
    "wpa_valued",
    "n_valued",
    "games",
)

COLUMNS = [
    "game_id",
    "play_id",
    "season",
    "week",
    "posteam",
    "play_type",
    "epa",
    "wpa",
    "yardline_100",
    "down",
    "success",
]


# --------------------------------------------------------------------------
# sufficient statistics
# --------------------------------------------------------------------------


def team_game_statistics(pbp: pl.DataFrame) -> tuple[pl.DataFrame, float, dict]:
    """One row per team-game, carrying every sum the four measures need.

    Pooling sums within a half — rather than averaging a per-game rate as
    document 02 did — is a deliberate deviation, and the red-zone denominator is
    the reason: a team-game holds a median of 9 red-zone plays and 2% hold none
    at all, so a per-game red-zone mean is either undefined or estimated off a
    handful of snaps. Pooling numerator and denominator across the half is the
    honest estimator, and it is the one document 02 itself switched to for the
    fumble *rate* test for exactly this reason.
    """
    # S0/S1/S2 live on the offense's own scrimmage plays. Kicks, punts and
    # penalty-only rows are not "efficiency" in a sense a coach would accept.
    scrimmage = pbp.filter(
        pl.col("posteam").is_not_null()
        & pl.col("play_type").is_in(["pass", "run"])
        & pl.col("epa").is_not_null()
        & pl.col("down").is_not_null()
    )

    # S3 values the *whole* offensive record two ways, so it takes every play the
    # team had the ball for that carries both valuations.
    valued = pbp.filter(
        pl.col("posteam").is_not_null() & pl.col("epa").is_not_null() & pl.col("wpa").is_not_null()
    )

    epa = valued["epa"].to_numpy()
    wpa = valued["wpa"].to_numpy()
    slope = float(np.cov(epa, wpa)[0, 1] / np.var(epa, ddof=1))

    scrimmage_stats = scrimmage.group_by(["season", "posteam", "game_id"]).agg(
        pl.len().alias("n_all"),
        pl.col("epa").sum().alias("epa_all"),
        pl.col("success").sum().alias("succ_all"),
        (pl.col("yardline_100") <= RED_ZONE_YARDS).sum().alias("n_rz"),
        pl.col("epa").filter(pl.col("yardline_100") <= RED_ZONE_YARDS).sum().alias("epa_rz"),
        pl.col("down").is_in(LATE_DOWNS).sum().alias("n_ld"),
        pl.col("success").filter(pl.col("down").is_in(LATE_DOWNS)).sum().alias("succ_ld"),
    )

    valued_stats = valued.group_by(["season", "posteam", "game_id"]).agg(
        pl.len().alias("n_valued"),
        pl.col("epa").sum().alias("epa_valued"),
        pl.col("wpa").sum().alias("wpa_valued"),
    )

    frame = (
        scrimmage_stats.join(valued_stats, on=["season", "posteam", "game_id"], how="inner")
        .with_columns(
            pl.lit(1.0).alias("games"),
            pl.concat_str(
                [pl.col("season").cast(pl.String), pl.col("posteam")], separator="_"
            ).alias("team_season"),
        )
        .sort(["team_season", "game_id"])
    )

    rz_mask = scrimmage["yardline_100"] <= RED_ZONE_YARDS
    ld_mask = scrimmage["down"].is_in(LATE_DOWNS)
    p_all = float(scrimmage["success"].mean())
    p_ld = float(scrimmage.filter(ld_mask)["success"].mean())
    p_nonld = float(scrimmage.filter(~ld_mask)["success"].mean())

    design = {
        "wpa_per_epa_slope": slope,
        "wpa_epa_correlation": float(np.corrcoef(epa, wpa)[0, 1]),
        "mean_epa_all": float(scrimmage["epa"].mean()),
        "mean_epa_rz": float(scrimmage.filter(rz_mask)["epa"].mean()),
        "mean_epa_nonrz": float(scrimmage.filter(~rz_mask)["epa"].mean()),
        "var_epa_all": float(scrimmage["epa"].var(ddof=1)),
        "var_epa_rz": float(scrimmage.filter(rz_mask)["epa"].var(ddof=1)),
        "var_epa_nonrz": float(scrimmage.filter(~rz_mask)["epa"].var(ddof=1)),
        "success_rate_all": p_all,
        "success_rate_late_down": p_ld,
        "success_rate_non_late_down": p_nonld,
        "var_success_all": p_all * (1.0 - p_all),
        "var_success_late_down": p_ld * (1.0 - p_ld),
        "var_success_non_late_down": p_nonld * (1.0 - p_nonld),
        "mean_wpa_epa_residual": float(np.mean(wpa - slope * epa)),
        "var_wpa_epa_residual": float(np.var(wpa - slope * epa, ddof=1)),
        "n_scrimmage_plays": int(scrimmage.height),
        "n_valued_plays": int(valued.height),
        "n_team_games": int(frame.height),
    }
    return frame, slope, design


def to_dense(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense (team-game, len(SUM_COLUMNS)) matrix, group starts and group sizes.

    Team-seasons with fewer than ``MIN_GAMES`` games are dropped, and the matrix
    is returned already restricted to the kept rows so every routine downstream
    can treat groups as contiguous blocks.
    """
    keys = frame["team_season"].to_numpy()
    boundaries = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1], True])
    spans = [
        (boundaries[i], boundaries[i + 1])
        for i in range(len(boundaries) - 1)
        if boundaries[i + 1] - boundaries[i] >= MIN_GAMES
    ]
    rows = np.concatenate([np.arange(lo, hi) for lo, hi in spans])

    matrix = np.column_stack([frame[column].to_numpy().astype(float) for column in SUM_COLUMNS])[
        rows
    ]
    sizes = np.array([hi - lo for lo, hi in spans])
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    return matrix, starts, sizes


def split_masks(starts: np.ndarray, sizes: np.ndarray, n_rows: int, rng, n_splits: int):
    """Boolean (n_splits, n_rows): which team-games fall in half A of each split.

    Drawn once and reused, because the randomization that matters differs by
    caller — the permutation null randomizes *group membership*, the real run
    randomizes nothing else — and holding the split patterns fixed keeps the two
    comparable rather than adding a second source of difference.
    """
    mask = np.zeros((n_splits, n_rows), dtype=bool)
    for split in range(n_splits):
        for start, size in zip(starts, sizes, strict=True):
            chosen = rng.permutation(size)[: size // 2] + start
            mask[split, chosen] = True
    return mask


def half_sums(matrix: np.ndarray, mask: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """(n_splits, n_groups, n_columns) pooled sums for the masked half."""
    out = np.empty((mask.shape[0], len(starts), matrix.shape[1]))
    for column in range(matrix.shape[1]):
        weighted = mask * matrix[:, column][None, :]
        out[:, :, column] = np.add.reduceat(weighted, starts, axis=1)
    return out


def half_statistics(sums: np.ndarray, slope: float) -> np.ndarray:
    """The four measures from pooled column sums. Shape (..., len(MEASURES))."""
    n_all = sums[..., 0]
    epa_all = sums[..., 1]
    succ_all = sums[..., 2]
    n_rz = sums[..., 3]
    epa_rz = sums[..., 4]
    n_ld = sums[..., 5]
    succ_ld = sums[..., 6]
    epa_valued = sums[..., 7]
    wpa_valued = sums[..., 8]
    games = sums[..., 10]

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_all = epa_all / n_all
        s0 = mean_all
        s1 = epa_rz / n_rz - mean_all
        s2 = succ_ld / n_ld - succ_all / n_all
        s3 = (wpa_valued - slope * epa_valued) / games
    return np.stack([s0, s1, s2, s3], axis=-1)


def correlate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation along the group axis. Inputs (n_splits, n_groups)."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    numerator = (a * b).sum(axis=1)
    denominator = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    return numerator / denominator


def split_half_r(
    matrix: np.ndarray,
    mask: np.ndarray,
    starts: np.ndarray,
    totals: np.ndarray,
    slope: float,
) -> np.ndarray:
    """Mean split-half correlation per measure, averaged over the split draws.

    Half B is recovered as ``totals - half A`` rather than summed again: the two
    halves partition the team-season exactly, so one reduction does the work of
    two.
    """
    sums_a = half_sums(matrix, mask, starts)
    sums_b = totals[None, :, :] - sums_a
    a = half_statistics(sums_a, slope)
    b = half_statistics(sums_b, slope)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("a half statistic was undefined; a denominator hit zero")
    return np.array([correlate(a[:, :, m], b[:, :, m]).mean() for m in range(len(MEASURES))])


# --------------------------------------------------------------------------
# null 1 — permutation of real team-games
# --------------------------------------------------------------------------


def permutation_null(
    matrix: np.ndarray,
    mask: np.ndarray,
    starts: np.ndarray,
    slope: float,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Split-half r when team identity is destroyed but everything else is real.

    Whole team-games are dealt at random into synthetic team-seasons of the same
    sizes. Every within-game correlation, every real denominator and the real
    fat-tailed play distribution survive; the only thing removed is the thing
    being tested. That is the property an analytic null cannot claim.
    """
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, len(MEASURES)))
    for replicate in range(replicates):
        shuffled = matrix[rng.permutation(len(matrix))]
        totals = np.add.reduceat(shuffled, starts, axis=0)
        draws[replicate] = split_half_r(shuffled, mask, starts, totals, slope)
        if (replicate + 1) % 100 == 0:
            print(f"    permutation null {replicate + 1}/{replicates}", flush=True)
    return draws


# --------------------------------------------------------------------------
# null 2 and the power curve — parametric
# --------------------------------------------------------------------------


def half_noise_sd(
    matrix: np.ndarray, mask: np.ndarray, starts: np.ndarray, design: dict
) -> np.ndarray:
    """Analytic sampling SD of each half statistic. Shape (n_groups, n_splits, 2, n_measures).

    Used **only** to turn a target correlation into a ``tau`` in the measure's own
    units, so the power table can be read in both. The null and power draws
    themselves come from the game-level simulation, which is the faithful one;
    any mismatch between the tau this implies and the correlation it achieves is
    visible in the power table's "achieved mean r" column.

    Under the null every team shares the league process, so a half's statistic is
    a mean — or a difference of means — over that half's real denominators. The
    subset structure is carried explicitly: red-zone plays are a *subset* of all
    plays, so the two means covary, and

        Var(mean_sub - mean_all) = var_sub/n_sub + var_all/n_all - 2*var_sub/n_all

    rather than the sum of two independent variances. Dropping the covariance
    term would overstate the noise and hand the design more apparent power than
    it has.
    """
    sums_a = half_sums(matrix, mask, starts)
    totals = np.add.reduceat(matrix, starts, axis=0)
    sums_b = totals[None, :, :] - sums_a

    variances = []
    for sums in (sums_a, sums_b):
        n_all = sums[..., 0]
        n_rz = np.maximum(sums[..., 3], 1.0)
        n_ld = np.maximum(sums[..., 5], 1.0)
        n_valued = sums[..., 9]
        games = sums[..., 10]

        var_s0 = design["var_epa_all"] / n_all
        var_s1 = (
            design["var_epa_rz"] / n_rz
            + design["var_epa_all"] / n_all
            - 2.0 * design["var_epa_rz"] / n_all
        )
        var_s2 = (
            design["var_success_late_down"] / n_ld
            + design["var_success_all"] / n_all
            - 2.0 * design["var_success_late_down"] / n_all
        )
        # S3 is a per-game sum of a per-play residual, so its noise is
        # sqrt(plays * var) / games.
        var_s3 = design["var_wpa_epa_residual"] * n_valued / games**2
        variances.append(np.stack([var_s0, var_s1, var_s2, var_s3], axis=-1))

    # (n_splits, n_groups, 2, n_measures) -> (n_groups, n_splits, 2, n_measures)
    stacked = np.stack(variances, axis=2)
    return np.sqrt(np.maximum(stacked, 1e-14)).transpose(1, 0, 2, 3)


def simulate_matrix(
    matrix: np.ndarray,
    group_of_row: np.ndarray,
    design: dict,
    slope: float,
    measure: int,
    tau: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """A synthetic team-game matrix with a known true sequencing spread.

    Simulation happens at the **team-game** level, not the half level, and that
    is the whole point. The executed statistic averages over 200 random splits of
    the *same* games, so its split draws are heavily correlated; a simulation
    that redrew noise per half would make those draws independent, shrink the
    null spread by roughly sqrt(200), and hand the design power it does not have.

    The generative story for tau is sequencing in the literal sense — **the same
    total production, placed differently.** Whatever is added to the high-leverage
    subset is subtracted from its complement, so a team's overall efficiency is
    untouched and only *where* the production landed moves. That is the only
    generative story under which "sequencing is luck" is a meaningful hypothesis:
    if the total moved too, the measure would just be re-measuring offense.

    Real denominators are reused throughout. Only the numerators are simulated.
    """
    n_all = matrix[:, 0]
    n_rz = matrix[:, 3]
    n_ld = matrix[:, 5]
    n_valued = matrix[:, 9]
    n_nonrz = np.maximum(n_all - n_rz, 0.0)
    n_nonld = np.maximum(n_all - n_ld, 0.0)

    theta = (rng.standard_normal(group_of_row.max() + 1) * tau)[group_of_row]
    simulated = matrix.copy()

    def normal_sum(count, mean_per_unit, var_per_unit, shift=0.0):
        return rng.normal(
            count * mean_per_unit + shift, np.sqrt(np.maximum(var_per_unit * count, 1e-12))
        )

    if measure == 0:
        # A level effect on the offense itself: the positive control.
        simulated[:, 1] = normal_sum(n_all, design["mean_epa_all"] + theta, design["var_epa_all"])
    elif measure == 1:
        epa_rz = normal_sum(n_rz, design["mean_epa_rz"], design["var_epa_rz"], shift=theta * n_rz)
        epa_nonrz = normal_sum(
            n_nonrz, design["mean_epa_nonrz"], design["var_epa_nonrz"], shift=-theta * n_rz
        )
        simulated[:, 4] = epa_rz
        simulated[:, 1] = epa_rz + epa_nonrz
    elif measure == 2:
        succ_ld = normal_sum(
            n_ld,
            design["success_rate_late_down"],
            design["var_success_late_down"],
            shift=theta * n_ld,
        )
        succ_nonld = normal_sum(
            n_nonld,
            design["success_rate_non_late_down"],
            design["var_success_non_late_down"],
            shift=-theta * n_ld,
        )
        simulated[:, 6] = succ_ld
        simulated[:, 2] = succ_ld + succ_nonld
    elif measure == 3:
        # theta is per *game* here, because S3 is a per-game sum rather than a
        # per-play rate.
        residual = normal_sum(
            n_valued, design["mean_wpa_epa_residual"], design["var_wpa_epa_residual"], shift=theta
        )
        simulated[:, 8] = residual + slope * matrix[:, 7]
    else:
        raise ValueError(f"unknown measure index {measure}")

    return simulated


def parametric_draws(
    matrix: np.ndarray,
    group_of_row: np.ndarray,
    mask: np.ndarray,
    starts: np.ndarray,
    design: dict,
    slope: float,
    measure: int,
    tau: float,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Split-half r for one measure under a true team-level spread `tau`."""
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for replicate in range(replicates):
        simulated = simulate_matrix(matrix, group_of_row, design, slope, measure, tau, rng)
        totals = np.add.reduceat(simulated, starts, axis=0)
        draws[replicate] = split_half_r(simulated, mask, starts, totals, slope)[measure]
    return draws


def tau_for_target_r(sd_measure: np.ndarray, target_r: float) -> float:
    """The true spread that yields an expected split-half r of `target_r`.

    With ``half = theta + noise`` the population correlation of the two halves is
    ``tau^2 / (tau^2 + sigma^2)``, so inverting at the mean noise variance turns
    a correlation target into the measure's own units. It makes the power table
    readable in both.
    """
    if not 0.0 < target_r < 1.0:
        raise ValueError("target_r must be strictly between 0 and 1")
    mean_variance = float(np.mean(sd_measure**2))
    return float(np.sqrt(target_r * mean_variance / (1.0 - target_r)))


# --------------------------------------------------------------------------


def main() -> None:
    paths.ensure_data_dirs()
    pbp = load_pbp(PBP_SEASONS, columns=COLUMNS)
    frame, slope, design = team_game_statistics(pbp)

    matrix, starts, sizes = to_dense(frame)
    group_of_row = np.repeat(np.arange(len(sizes)), sizes)
    design["n_team_seasons_used"] = int(len(starts))
    design["n_team_games_used"] = int(len(matrix))

    print("=== Design parameters, league-pooled — no persistence measured here ===")
    for key, value in design.items():
        print(f"  {key:28s} {value:.6f}" if isinstance(value, float) else f"  {key:28s} {value}")
    print(
        f"\n  median per team-game: {np.median(matrix[:, 0]):.0f} scrimmage plays, "
        f"{np.median(matrix[:, 3]):.0f} red-zone, {np.median(matrix[:, 5]):.0f} late-down"
    )

    rng = np.random.default_rng(RANDOM_SEED)
    mask = split_masks(starts, sizes, len(matrix), rng, N_SPLITS)

    # ---- permutation null -------------------------------------------------
    print(f"\n=== Permutation null ({N_NULL_REPLICATES} replicates) ===")
    print("    real team-games dealt at random into synthetic team-seasons")
    null_draws = permutation_null(matrix, mask, starts, slope, N_NULL_REPLICATES, RANDOM_SEED)

    thresholds, null_rows = {}, []
    for m, name in enumerate(MEASURES):
        draws = null_draws[:, m]
        thresholds[name] = float(np.percentile(draws, 95))
        null_rows.append(
            {
                "measure": name,
                "null_mean_r": float(draws.mean()),
                "null_sd_r": float(draws.std(ddof=1)),
                "null_p95": thresholds[name],
                "null_p99": float(np.percentile(draws, 99)),
            }
        )
    with pl.Config(tbl_cols=-1, fmt_str_lengths=30):
        print(pl.DataFrame(null_rows))

    # ---- parametric null, as a cross-check on the simulated noise ---------
    print("\n=== Parametric null — simulated play sums at the same denominators ===")
    sd = half_noise_sd(matrix, mask, starts, design)
    agreement_rows = []
    for m, name in enumerate(MEASURES):
        parametric = parametric_draws(
            matrix,
            group_of_row,
            mask,
            starts,
            design,
            slope,
            m,
            0.0,
            N_NULL_REPLICATES // 2,
            RANDOM_SEED + 11 + m,
        )
        agreement_rows.append(
            {
                "measure": name,
                "permutation_sd": float(null_draws[:, m].std(ddof=1)),
                "parametric_sd": float(parametric.std(ddof=1)),
                "ratio_parametric_over_permutation": float(
                    parametric.std(ddof=1) / null_draws[:, m].std(ddof=1)
                ),
            }
        )
    with pl.Config(tbl_cols=-1, fmt_str_lengths=34):
        print(pl.DataFrame(agreement_rows))
    print(
        "    A ratio near 1 means the simulated noise reproduces the real one, which is\n"
        "    what licenses reading the power column below. A ratio below 1 means the\n"
        "    parametric arm understates noise and therefore OVERSTATES power."
    )

    # ---- power curve ------------------------------------------------------
    print(
        f"\n=== Power to clear the permutation null's 95th percentile "
        f"({N_POWER_REPLICATES} reps) ==="
    )
    power_rows = []
    for m, name in enumerate(MEASURES):
        for target_r in (0.05, 0.08, 0.10, REFERENCE_R, 0.20, 0.30, 0.50):
            tau = tau_for_target_r(sd[:, :, :, m], target_r)
            draws = parametric_draws(
                matrix,
                group_of_row,
                mask,
                starts,
                design,
                slope,
                m,
                tau,
                N_POWER_REPLICATES,
                RANDOM_SEED + 100 + m,
            )
            power = float((draws > thresholds[name]).mean())
            power_rows.append(
                {
                    "measure": name,
                    "target_true_r": target_r,
                    "tau": tau,
                    "mean_observed_r": float(draws.mean()),
                    "power": power,
                }
            )
            print(
                f"  {name:18s} nominal r {target_r:.2f}  tau {tau:.5f}  "
                f"achieved mean r {draws.mean():+.3f}  power {power:.3f}"
            )

    results = {
        "design": design,
        "n_splits": N_SPLITS,
        "n_null_replicates": N_NULL_REPLICATES,
        "n_power_replicates": N_POWER_REPLICATES,
        "reference_r": REFERENCE_R,
        "permutation_null": null_rows,
        "parametric_agreement": agreement_rows,
        "thresholds_p95": thresholds,
        "power": power_rows,
        "random_seed": RANDOM_SEED,
    }
    out = paths.RESEARCH_OUTPUT_DIR / "10_sequencing_power.json"
    with out.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
