"""Phase 4, step 1 — drive anatomy: the drive table, and the power behind its one gate.

Document 08 §11 recorded why DQW% failed: **depth is not a sufficient statistic
for a drive.** Conditioning drive points on how deep the drive got marked good
offenses down and bad ones up — ``corr(quality, adjustment) = -0.784`` — and
destroyed 29.4% of the real between-team spread in points per drive. The named
fix was *"a drive summary richer than depth — starting field position, plays,
yards, and explosive-play count"*, recorded as future work requiring its own
pre-registration.

This script builds that summary and measures the design parameters a
pre-registration needs. It runs **before** `docs/research/11-drive-anatomy.md`
commits any threshold, for the reason document 04's Gate 2 taught the project.

What it does **not** do: compute the real split-half correlation of any
finishing residual. That is the quantity document 11 gates, and it is produced by
`research/19_drive_anatomy.py` after the thresholds are committed. What it does
compute is the *permutation null* — real team-games dealt at random into
synthetic team-seasons, which destroys team identity while keeping every
denominator, every within-game correlation and the real residual distribution.

    uv run python research/19_drive_anatomy_power.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_seq = import_module("10_sequencing_power")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.ingest import PBP_SEASONS, load_pbp  # noqa: E402

RANDOM_SEED = 20260817

# Split draws inside one split-half estimate, matching documents 02 and 08 so
# every correlation this project reports is the same estimator.
N_SPLITS = 200
N_NULL_REPLICATES = 500
N_POWER_REPLICATES = 500

# A team-season needs enough games that both halves hold real drives. Documents
# 02 and 08 both used 8.
MIN_GAMES = 8

# The smallest split-half correlation this project has ever called real:
# document 02's middle three components sat at r = 0.12-0.16.
REFERENCE_R = 0.12

# Cross-validation folds for the out-of-fold conditional-mean models. Out-of-fold
# is not optional here: an in-sample residual from a flexible model is shrunk
# toward zero by the fit itself, which would understate exactly the quantity
# being tested.
N_FOLDS = 10

# Explosive-play convention, taken from football usage rather than fitted:
# 12+ yards on a run, 16+ on a pass. Choosing the cut after seeing which cut
# persists would be the goalpost-moving document 04 recorded.
EXPLOSIVE_RUN_YARDS = 12
EXPLOSIVE_PASS_YARDS = 16

RED_ZONE_YARDS = 20

# Same resampling universe document 08 §10 fixed, and for the same reasons: a
# drive that ended in a turnover, a safety or the clock is not an
# offensive-points outcome this project is allowed to redraw.
RESAMPLED_RESULTS: tuple[str, ...] = (
    "Touchdown",
    "Field goal",
    "Punt",
    "Missed field goal",
    "Turnover on downs",
)

# Nested feature sets, each a strict superset of the one above it. F1 is
# document 08's actual instrument; everything below it is the "richer summary"
# §11 named.
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "F1_depth": ("depth",),
    "F2_advance": ("depth", "start_yardline_100", "scrimmage_plays"),
    "F3_production": (
        "depth",
        "start_yardline_100",
        "scrimmage_plays",
        "explosive_plays",
        "max_gain",
        "first_downs",
        "penalty_aid_yards",
    ),
    "F4_yardage": (
        "depth",
        "start_yardline_100",
        "scrimmage_plays",
        "explosive_plays",
        "max_gain",
        "first_downs",
        "penalty_aid_yards",
        "net_yards",
    ),
}

COLUMNS = [
    "game_id",
    "play_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "posteam",
    "defteam",
    "fixed_drive",
    "fixed_drive_result",
    "yardline_100",
    "play_type",
    "yards_gained",
    "first_down",
    "penalty",
    "penalty_team",
    "penalty_yards",
    "extra_point_result",
    "two_point_conv_result",
]


# --------------------------------------------------------------------------
# the drive table
# --------------------------------------------------------------------------


def drive_table(pbp: pl.DataFrame) -> pl.DataFrame:
    """One row per offensive drive, with the rich summary document 08 §11 named.

    Every feature is a property of the drive's own scrimmage plays. **No team
    identity enters**, and that is a ruling rather than an oversight: the Phase 4
    plan excludes team-identity effects from EPA resampling on the grounds that
    entity abilities may set the expectation for a coin flip but must never
    revalue the football that was actually played. A team-quality covariate here
    would leak a power ranking into an adjudication.

    ``depth`` keeps document 08's definition exactly — the deepest yard line
    reached **at the snap** of a run or a pass. A touchdown drive necessarily
    *ends* at ``yardline_100 = 0``, so a depth defined by where the drive
    finished would encode the outcome into the conditioning variable.

    The features that are *not* clean of the outcome are named rather than
    hidden. ``net_yards`` is very nearly ``start_yardline_100`` on any touchdown
    drive, so the F4 set below is included precisely so the size of that
    entanglement is measurable instead of assumed. ``first_downs`` carries a
    weaker version of the same problem.
    """
    scrimmage = pbp.filter(
        pl.col("posteam").is_not_null()
        & pl.col("fixed_drive").is_not_null()
        & pl.col("play_type").is_in(["pass", "run"])
        & pl.col("yardline_100").is_not_null()
    ).with_columns(
        (
            ((pl.col("play_type") == "run") & (pl.col("yards_gained") >= EXPLOSIVE_RUN_YARDS))
            | ((pl.col("play_type") == "pass") & (pl.col("yards_gained") >= EXPLOSIVE_PASS_YARDS))
        ).alias("explosive"),
    )

    summary = (
        scrimmage.sort(["game_id", "fixed_drive", "play_id"])
        .group_by(["game_id", "fixed_drive"])
        .agg(
            pl.col("posteam").first().alias("posteam"),
            pl.col("yardline_100").first().alias("start_yardline_100"),
            pl.col("yardline_100").min().alias("depth"),
            pl.len().alias("scrimmage_plays"),
            pl.col("yards_gained").sum().alias("net_yards"),
            pl.col("yards_gained").max().alias("max_gain"),
            pl.col("explosive").sum().alias("explosive_plays"),
            pl.col("first_down").fill_null(0).sum().alias("first_downs"),
            (pl.col("yardline_100") <= RED_ZONE_YARDS).any().alias("reached_red_zone"),
        )
    )

    # Penalties that *helped* the offense: a flag on the defense. Counted over
    # every row of the drive, not just scrimmage plays, because a defensive
    # penalty on a punt still hands the offense the ball back.
    aids = (
        pbp.filter(pl.col("fixed_drive").is_not_null() & (pl.col("penalty") == 1))
        .group_by(["game_id", "fixed_drive"])
        .agg(
            pl.col("penalty_yards")
            .filter(pl.col("penalty_team") == pl.col("defteam"))
            .fill_null(0)
            .sum()
            .alias("penalty_aid_yards"),
        )
    )

    conversions = (
        pbp.filter(pl.col("fixed_drive").is_not_null())
        .group_by(["game_id", "fixed_drive"])
        .agg(
            pl.col("fixed_drive_result").drop_nulls().first().alias("result"),
            (pl.col("extra_point_result") == "good").any().alias("xp_good"),
            (pl.col("two_point_conv_result") == "success").any().alias("two_good"),
            pl.col("home_team").first().alias("home_team"),
            pl.col("season").first().alias("season"),
            pl.col("week").first().alias("week"),
        )
    )

    return (
        summary.join(conversions, on=["game_id", "fixed_drive"], how="inner")
        .join(aids, on=["game_id", "fixed_drive"], how="left")
        .with_columns(pl.col("penalty_aid_yards").fill_null(0.0))
        .with_columns(
            pl.when(pl.col("result") == "Touchdown")
            .then(6 + pl.col("xp_good").cast(pl.Int32) + 2 * pl.col("two_good").cast(pl.Int32))
            .when(pl.col("result") == "Field goal")
            .then(pl.lit(3))
            .otherwise(pl.lit(0))
            .cast(pl.Float64)
            .alias("points"),
            (pl.col("posteam") == pl.col("home_team")).alias("is_home"),
            pl.col("result").is_in(RESAMPLED_RESULTS).alias("resampled"),
            pl.concat_str(
                [pl.col("season").cast(pl.String), pl.col("posteam")], separator="_"
            ).alias("team_season"),
        )
        .with_columns(
            pl.col(
                [
                    "start_yardline_100",
                    "depth",
                    "scrimmage_plays",
                    "net_yards",
                    "max_gain",
                    "explosive_plays",
                    "first_downs",
                    "penalty_aid_yards",
                ]
            ).cast(pl.Float64)
        )
        .sort(["team_season", "game_id", "fixed_drive"])
    )


# --------------------------------------------------------------------------
# conditional means, out of fold
# --------------------------------------------------------------------------


def oof_predictions(features: np.ndarray, points: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Out-of-fold E[points | features] from a gradient-boosted regressor.

    The same estimator and the same hyper-parameters run on every nested feature
    set, so the comparison between them is a comparison of *information* rather
    than of model families. A booster rather than a linear fit because the
    incumbent instrument was a nonparametric cell table, and handing the richer
    sets a more flexible learner than the incumbent had would flatter them.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    out = np.empty(len(points))
    for fold in np.unique(folds):
        test = folds == fold
        model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.1,
            max_leaf_nodes=31,
            min_samples_leaf=100,
            random_state=RANDOM_SEED,
        )
        model.fit(features[~test], points[~test])
        out[test] = model.predict(features[test])
    return out


def cell_mean_predictions(depth: np.ndarray, points: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Document 08's actual instrument: the 5-yard depth-bin mean, out of fold.

    Reported alongside F1 so the richer sets are compared against what really
    shipped and failed, not against a reimplementation of it.
    """
    bins = (depth // 5).astype(int)
    out = np.empty(len(points))
    for fold in np.unique(folds):
        test = folds == fold
        train_bins, train_points = bins[~test], points[~test]
        overall = train_points.mean()
        table = {}
        for value in np.unique(train_bins):
            selected = train_points[train_bins == value]
            table[value] = selected.mean() if len(selected) >= 100 else overall
        out[test] = np.array([table.get(value, overall) for value in bins[test]])
    return out


# --------------------------------------------------------------------------
# the split-half statistic
# --------------------------------------------------------------------------


def team_game_matrix(
    drives: pl.DataFrame, residuals: dict[str, np.ndarray]
) -> tuple[pl.DataFrame, tuple[str, ...]]:
    """Team-game sums of every residual, plus points and drive counts.

    The split is by **team-game**, never by drive: two drives in the same game
    share a defense, a game script and a set of conditions, so splitting inside a
    game would break the independence the split-half estimator assumes and would
    inflate every correlation on the page.
    """
    frame = drives.with_columns(
        [pl.Series(f"resid_{name}", values) for name, values in residuals.items()]
    )
    measures = ("points", *(f"resid_{name}" for name in residuals))
    grouped = (
        frame.group_by(["team_season", "game_id"])
        .agg(
            pl.len().alias("drives"),
            pl.lit(1.0).alias("games"),
            *[pl.col(measure).sum().alias(measure) for measure in measures],
        )
        .sort(["team_season", "game_id"])
    )
    return grouped, measures


def to_dense(frame: pl.DataFrame, measures: tuple[str, ...]) -> tuple[np.ndarray, ...]:
    """Dense (team-game, 1 + len(measures)) matrix with contiguous team-season blocks."""
    keys = frame["team_season"].to_numpy()
    boundaries = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1], True])
    spans = [
        (boundaries[i], boundaries[i + 1])
        for i in range(len(boundaries) - 1)
        if boundaries[i + 1] - boundaries[i] >= MIN_GAMES
    ]
    rows = np.concatenate([np.arange(lo, hi) for lo, hi in spans])
    columns = ["drives", *measures]
    matrix = np.column_stack([frame[column].to_numpy().astype(float) for column in columns])[rows]
    sizes = np.array([hi - lo for lo, hi in spans])
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    return matrix, starts, sizes


def half_rates(sums: np.ndarray) -> np.ndarray:
    """Per-drive rate of every measure. Column 0 of ``sums`` is the drive count."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return sums[..., 1:] / sums[..., [0]]


def split_half_r(
    matrix: np.ndarray, mask: np.ndarray, starts: np.ndarray, totals: np.ndarray
) -> np.ndarray:
    """Mean split-half correlation per measure, averaged over the split draws."""
    sums_a = _seq.half_sums(matrix, mask, starts)
    sums_b = totals[None, :, :] - sums_a
    a, b = half_rates(sums_a), half_rates(sums_b)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("a half statistic was undefined; a drive count hit zero")
    return np.array([_seq.correlate(a[:, :, m], b[:, :, m]).mean() for m in range(a.shape[2])])


def split_half_distribution(matrix: np.ndarray, mask: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Per-split correlations, so the 5th-95th band document 02 published is available."""
    totals = np.add.reduceat(matrix, starts, axis=0)
    sums_a = _seq.half_sums(matrix, mask, starts)
    sums_b = totals[None, :, :] - sums_a
    a, b = half_rates(sums_a), half_rates(sums_b)
    return np.column_stack([_seq.correlate(a[:, :, m], b[:, :, m]) for m in range(a.shape[2])])


# --------------------------------------------------------------------------
# null and power
# --------------------------------------------------------------------------


def permutation_null(
    matrix: np.ndarray, mask: np.ndarray, starts: np.ndarray, replicates: int, seed: int
) -> np.ndarray:
    """Split-half r when team identity is destroyed and everything else is real."""
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, matrix.shape[1] - 1))
    for replicate in range(replicates):
        shuffled = matrix[rng.permutation(len(matrix))]
        totals = np.add.reduceat(shuffled, starts, axis=0)
        draws[replicate] = split_half_r(shuffled, mask, starts, totals)
        if (replicate + 1) % 100 == 0:
            print(f"    permutation null {replicate + 1}/{replicates}", flush=True)
    return draws


def parametric_draws(
    matrix: np.ndarray,
    group_of_row: np.ndarray,
    mask: np.ndarray,
    starts: np.ndarray,
    column: int,
    per_drive_sd: float,
    tau: float,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Split-half r for one measure under a known true team spread ``tau``.

    Simulation happens at the **team-game** level, never the half level. The
    executed statistic averages over 200 random splits of the *same* games, so its
    split draws are heavily correlated; redrawing noise per half would make them
    independent, shrink the null spread by roughly sqrt(200) and hand the design
    power it does not have. Document 08 §5 records that mistake being caught.
    """
    rng = np.random.default_rng(seed)
    drives = matrix[:, 0]
    draws = np.empty(replicates)
    simulated = matrix.copy()
    for replicate in range(replicates):
        theta = (rng.standard_normal(group_of_row.max() + 1) * tau)[group_of_row]
        simulated[:, column] = rng.normal(drives * theta, per_drive_sd * np.sqrt(drives))
        totals = np.add.reduceat(simulated, starts, axis=0)
        draws[replicate] = split_half_r(simulated, mask, starts, totals)[column - 1]
    return draws


def tau_for_target_r(per_drive_sd: float, drives_per_half: float, target_r: float) -> float:
    """The true team spread yielding an expected split-half r of ``target_r``."""
    noise_variance = per_drive_sd**2 / drives_per_half
    return float(np.sqrt(target_r * noise_variance / (1.0 - target_r)))


# --------------------------------------------------------------------------


def main() -> None:
    paths.ensure_data_dirs()
    pbp = load_pbp(PBP_SEASONS, columns=COLUMNS)
    drives = drive_table(pbp).filter(pl.col("resampled"))
    print(f"{drives.height:,} offensive drives in the resampling universe")

    points = drives["points"].to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    folds = rng.permutation(len(points)) % N_FOLDS

    print("\n=== Out-of-fold conditional means, nested feature sets ===")
    residuals: dict[str, np.ndarray] = {}
    fit_rows = []
    baseline_variance = float(np.var(points, ddof=1))

    cell = cell_mean_predictions(drives["depth"].to_numpy(), points, folds)
    residuals["F0_cellmean"] = points - cell
    fit_rows.append(
        {
            "feature_set": "F0_cellmean (document 08's shipped instrument)",
            "n_features": 1,
            "oof_r2": 1.0 - float(np.var(points - cell, ddof=1)) / baseline_variance,
            "residual_sd": float(np.std(points - cell, ddof=1)),
        }
    )

    for name, columns in FEATURE_SETS.items():
        features = np.column_stack([drives[column].to_numpy() for column in columns])
        predicted = oof_predictions(features, points, folds)
        residual = points - predicted
        residuals[name] = residual
        fit_rows.append(
            {
                "feature_set": name,
                "n_features": len(columns),
                "oof_r2": 1.0 - float(np.var(residual, ddof=1)) / baseline_variance,
                "residual_sd": float(np.std(residual, ddof=1)),
            }
        )
        print(
            f"  {name:16s} OOF R2 {fit_rows[-1]['oof_r2']:.4f}  "
            f"residual SD {fit_rows[-1]['residual_sd']:.4f}"
        )

    grouped, measures = team_game_matrix(drives, residuals)
    matrix, starts, sizes = to_dense(grouped, measures)
    group_of_row = np.repeat(np.arange(len(sizes)), sizes)
    drives_per_half = float(
        np.mean([matrix[s : s + n, 0].sum() / 2.0 for s, n in zip(starts, sizes, strict=True)])
    )

    design = {
        "n_drives": int(drives.height),
        "n_team_seasons": int(len(starts)),
        "n_team_games": int(len(matrix)),
        "mean_drives_per_team_game": float(matrix[:, 0].mean()),
        "mean_drives_per_half": drives_per_half,
        "points_per_drive": float(points.mean()),
        "var_points_per_drive": baseline_variance,
        "explosive_share": float((drives["explosive_plays"] > 0).mean()),
        "red_zone_share": float(drives["reached_red_zone"].mean()),
    }
    print("\n=== Design parameters, league-pooled — no persistence measured here ===")
    for key, value in design.items():
        print(f"  {key:28s} {value:.4f}" if isinstance(value, float) else f"  {key:28s} {value}")

    split_rng = np.random.default_rng(RANDOM_SEED)
    mask = _seq.split_masks(starts, sizes, len(matrix), split_rng, N_SPLITS)

    print(f"\n=== Permutation null ({N_NULL_REPLICATES} replicates) ===")
    null_draws = permutation_null(matrix, mask, starts, N_NULL_REPLICATES, RANDOM_SEED)

    thresholds, null_rows = {}, []
    for m, name in enumerate(measures):
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
    with pl.Config(tbl_cols=-1, fmt_str_lengths=34, tbl_rows=20):
        print(pl.DataFrame(null_rows))

    print(f"\n=== Power to clear the null's 95th percentile ({N_POWER_REPLICATES} reps) ===")
    power_rows = []
    for m, name in enumerate(measures):
        if name == "points":
            continue  # the positive control needs no power curve; it is known to persist
        column = m + 1
        per_drive_sd = float(np.std(residuals[name.removeprefix("resid_")], ddof=1))
        for target_r in (0.05, 0.08, 0.10, REFERENCE_R, 0.20, 0.30):
            tau = tau_for_target_r(per_drive_sd, drives_per_half, target_r)
            draws = parametric_draws(
                matrix,
                group_of_row,
                mask,
                starts,
                column,
                per_drive_sd,
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
                f"  {name:20s} nominal r {target_r:.2f}  tau {tau:.5f}  "
                f"achieved mean r {draws.mean():+.3f}  power {power:.3f}"
            )

    results = {
        "design": design,
        "nested_fits": fit_rows,
        "feature_sets": {name: list(columns) for name, columns in FEATURE_SETS.items()},
        "n_splits": N_SPLITS,
        "n_null_replicates": N_NULL_REPLICATES,
        "n_power_replicates": N_POWER_REPLICATES,
        "reference_r": REFERENCE_R,
        "permutation_null": null_rows,
        "thresholds_p95": thresholds,
        "power": power_rows,
        "random_seed": RANDOM_SEED,
    }
    out = paths.RESEARCH_OUTPUT_DIR / "19_drive_anatomy_power.json"
    with out.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
