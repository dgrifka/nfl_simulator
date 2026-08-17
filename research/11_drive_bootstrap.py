"""Phase 3, step 2 — the drive-outcome resampling, run against pre-registered gates.

Design, statistic and thresholds are fixed by `docs/research/08-sequencing-luck.md`
§10, committed at `1d91af4` before this script existed. Nothing here chooses
anything.

The measure it produces, **DQW%**, answers a different question from DTW%:

    DTW%   given the plays that happened, who deserved to win once the coin
           flips are set to their expectations?   (holds sequencing fixed)
    DQW%   given the drives this team produced, how often does that production
           win?                                    (holds the coin flips fixed)

Section 9 of that document found that where a team's production lands — relative
to the red zone, relative to third down — does not persist across the halves of
its own season. So this holds fixed how far each drive advanced the ball, which
is production, and re-draws whether that advance became points.

    Gate D-1  the resampling reproduces the league's own mean points per drive
    Gate D-2  non-inferiority on the rematch harness, same +0.010 log-loss
              margin document 06 pre-registered
    Gate D-3  DQW% and DTW% are distinct — reported descriptively, no pass rule

    uv run python research/11_drive_bootstrap.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_rematch = import_module("08_rematch_power")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.ingest import PBP_SEASONS, load_pbp  # noqa: E402

RANDOM_SEED = 20260817
N_REPLICATES = 2000
DEPTH_BIN_WIDTH = 5
MIN_BIN_DRIVES = 100

GATE_D1_TOLERANCE = 0.01  # points per drive
NONINFERIORITY_MARGIN = _rematch.NONINFERIORITY_MARGIN
N_FOLDS = _rematch.N_FOLDS

# Drives whose outcome is resampled. The exclusions are each a ruling:
#
#   Turnover / Opp touchdown  the drive ended in an interception or a lost
#                             fumble. Fumble luck is already neutralized inside
#                             DTW%, and interceptions are deliberately not
#                             neutralized anywhere in this project (doc 05 §3).
#                             Resampling them here would contradict both.
#   Safety                    two points to the defense; not an offensive-points
#                             outcome at all.
#   End of half / End of game the drive was terminated by the clock, not by
#                             finishing. That is a different kind of luck and it
#                             is out of scope.
RESAMPLED_RESULTS: tuple[str, ...] = (
    "Touchdown",
    "Field goal",
    "Punt",
    "Missed field goal",
    "Turnover on downs",
)

COLUMNS = [
    "game_id",
    "play_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "posteam",
    "fixed_drive",
    "fixed_drive_result",
    "yardline_100",
    "play_type",
    "extra_point_result",
    "two_point_conv_result",
    "result",
]


# --------------------------------------------------------------------------
# building the drive table
# --------------------------------------------------------------------------


def drive_table(pbp: pl.DataFrame) -> pl.DataFrame:
    """One row per offensive drive: how deep it got, and what it was worth.

    **Depth is the deepest yard line reached at the SNAP of a scrimmage play**,
    and that is the load-bearing definition. A touchdown drive necessarily *ends*
    at `yardline_100 = 0`, so defining depth by where the drive finished would
    encode the outcome into the conditioning variable and make the whole
    resampling vacuous. Taking the deepest snap of a run or a pass makes a drive
    that scored from the 3 and a drive that was stopped at the 3 exchangeable,
    which is exactly the comparison document 08 §9's null result on the red-zone
    gap licenses.

    Kickoffs, punts, field-goal attempts and extra points are excluded from the
    depth computation for the same reason: a two-point attempt is snapped from
    the 2, so leaving it in would hand every touchdown drive a depth of 2.
    """
    scrimmage = pbp.filter(
        pl.col("posteam").is_not_null()
        & pl.col("fixed_drive").is_not_null()
        & pl.col("play_type").is_in(["pass", "run"])
        & pl.col("yardline_100").is_not_null()
    )
    depth = scrimmage.group_by(["game_id", "fixed_drive"]).agg(
        pl.col("yardline_100").min().alias("depth"),
        pl.col("posteam").first().alias("posteam"),
        pl.len().alias("scrimmage_plays"),
    )

    # Points the offense put on the board. A touchdown is six plus whatever the
    # conversion attempt produced, read from the PAT row that nflverse keeps
    # inside the same drive.
    conversions = (
        pbp.filter(pl.col("fixed_drive").is_not_null())
        .group_by(["game_id", "fixed_drive"])
        .agg(
            pl.col("fixed_drive_result").drop_nulls().first().alias("result"),
            (pl.col("extra_point_result") == "good").any().alias("xp_good"),
            (pl.col("two_point_conv_result") == "success").any().alias("two_good"),
            pl.col("home_team").first().alias("home_team"),
            pl.col("season").first().alias("season"),
        )
    )

    return (
        depth.join(conversions, on=["game_id", "fixed_drive"], how="inner")
        .with_columns(
            pl.when(pl.col("result") == "Touchdown")
            .then(6 + pl.col("xp_good").cast(pl.Int32) + 2 * pl.col("two_good").cast(pl.Int32))
            .when(pl.col("result") == "Field goal")
            .then(pl.lit(3))
            .otherwise(pl.lit(0))
            .cast(pl.Float64)
            .alias("points"),
            (pl.col("posteam") == pl.col("home_team")).alias("is_home"),
            ((pl.col("depth") // DEPTH_BIN_WIDTH) * DEPTH_BIN_WIDTH)
            .cast(pl.Int32)
            .alias("depth_bin"),
            pl.col("result").is_in(RESAMPLED_RESULTS).alias("resampled"),
        )
        .sort(["game_id", "fixed_drive"])
    )


def conditional_table(drives: pl.DataFrame) -> pl.DataFrame:
    """League points-per-drive distribution, conditional on depth reached.

    Thin bins borrow from the nearest well-populated neighbour rather than from
    the league average, for the reason `components.py` gives about field-goal
    bins: points per drive is monotone in depth, so the pooled average is the
    worst possible stand-in for a drive that reached the 5.
    """
    universe = drives.filter(pl.col("resampled"))
    counts = (
        universe.group_by("depth_bin")
        .agg(pl.len().alias("n"), pl.col("points").mean().alias("mean_points"))
        .sort("depth_bin")
    )
    populated = set(counts.filter(pl.col("n") >= MIN_BIN_DRIVES)["depth_bin"].to_list())
    if not populated:
        raise ValueError(f"no depth bin holds {MIN_BIN_DRIVES}+ drives")

    donor_for = {}
    for depth_bin in counts["depth_bin"].to_list():
        donor_for[depth_bin] = (
            depth_bin
            if depth_bin in populated
            else min(populated, key=lambda candidate: abs(candidate - depth_bin))
        )
    return counts.with_columns(
        pl.col("depth_bin").replace_strict(donor_for, return_dtype=pl.Int32).alias("donor_bin")
    )


# --------------------------------------------------------------------------
# the bootstrap
# --------------------------------------------------------------------------


def resample(
    drives: pl.DataFrame,
    table: pl.DataFrame,
    replicates: int,
    seed: int,
    *,
    red_zone_only: bool = False,
) -> tuple[pl.DataFrame, dict]:
    """Per-game DQ margin and DQW%, plus the Gate D-1 identity check.

    Resampling is non-parametric: a drive draws a replacement points value
    uniformly from the observed points of every league drive in its donor bin.
    That respects the {0, 3, 6, 7, 8} support automatically, which a fitted
    distribution over points would not.

    ``red_zone_only`` is the variant document 08 §10's defect register named in
    advance. Document 09 §9's null result is defined on ``yardline_100 <= 20``,
    so red-zone drives are the part of this resampling the evidence *directly*
    licenses; everything outside the 20 is an extension by assumption. Running
    both arms keeps the licensed part separable, which is why the variant was
    written into the pre-registration rather than reached for after a failure.
    """
    rng = np.random.default_rng(seed)
    league = drives.filter(pl.col("resampled")).join(
        table.select("depth_bin", "donor_bin", "mean_points"), on="depth_bin", how="left"
    )

    # The pool is always built from every league drive in the bin, even when only
    # a subset is being resampled. Rebuilding it from the subset would truncate
    # the donor bin at the subset's own boundary — the 20-24 bin would hold only
    # its depth-20 drives — and quietly change the distribution being drawn from.
    pools: dict[int, np.ndarray] = {
        int(donor[0] if isinstance(donor, tuple) else donor): group["points"].to_numpy()
        for donor, group in league.group_by("donor_bin")
    }

    universe = league.filter(pl.col("depth") <= 20) if red_zone_only else league

    game_ids = drives["game_id"].unique(maintain_order=True).to_list()
    game_index = {game_id: i for i, game_id in enumerate(game_ids)}

    rows = universe.select("game_id", "is_home", "points", "donor_bin", "mean_points")
    row_game = np.array([game_index[g] for g in rows["game_id"].to_list()])
    row_sign = np.where(rows["is_home"].to_numpy(), 1.0, -1.0)
    row_points = rows["points"].to_numpy()
    row_donor = rows["donor_bin"].to_numpy()
    row_expected = rows["mean_points"].to_numpy()

    # Deterministic point estimate: the expected points the drive's depth implies,
    # rather than an average over draws. Same quantity, no Monte Carlo noise.
    expected_delta = np.bincount(
        row_game, weights=row_sign * (row_expected - row_points), minlength=len(game_ids)
    )

    order = {donor: np.flatnonzero(row_donor == donor) for donor in pools}
    deltas = np.empty((replicates, len(game_ids)))
    sampled_all = np.empty(replicates)
    for replicate in range(replicates):
        drawn = np.empty(len(row_points))
        for donor, positions in order.items():
            pool = pools[donor]
            drawn[positions] = pool[rng.integers(0, len(pool), size=len(positions))]
        sampled_all[replicate] = drawn.mean()
        deltas[replicate] = np.bincount(
            row_game, weights=row_sign * (drawn - row_points), minlength=len(game_ids)
        )
        if (replicate + 1) % 500 == 0:
            print(f"    replicate {replicate + 1}/{replicates}", flush=True)

    gate_d1 = {
        "observed_mean_points_per_drive": float(row_points.mean()),
        "resampled_mean_points_per_drive": float(sampled_all.mean()),
        "difference": float(sampled_all.mean() - row_points.mean()),
        "tolerance": GATE_D1_TOLERANCE,
        "pass": bool(abs(sampled_all.mean() - row_points.mean()) < GATE_D1_TOLERANCE),
        "n_drives_resampled": int(len(row_points)),
        "n_drives_held_observed": int(drives.height - len(row_points)),
    }
    print(
        f"\nGate D-1 (resampling is unbiased): {'PASS' if gate_d1['pass'] else 'FAIL'} — "
        f"observed {gate_d1['observed_mean_points_per_drive']:.4f} pts/drive, "
        f"resampled {gate_d1['resampled_mean_points_per_drive']:.4f}, "
        f"difference {gate_d1['difference']:+.5f} vs tolerance {GATE_D1_TOLERANCE}"
    )

    frame = pl.DataFrame(
        {
            "game_id": game_ids,
            "expected_delta": expected_delta,
            "delta_mean": deltas.mean(axis=0),
        }
    )
    return frame, {"gate_d1": gate_d1, "deltas": deltas, "game_ids": game_ids}


# --------------------------------------------------------------------------


def main() -> None:
    paths.ensure_data_dirs()
    pbp = load_pbp(PBP_SEASONS, columns=COLUMNS)
    drives = drive_table(pbp)
    print(f"{drives.height:,} offensive drives across {len(PBP_SEASONS)} seasons")

    table = conditional_table(drives)
    print("\n=== League points per drive, by deepest scrimmage snap ===")
    with pl.Config(tbl_rows=25):
        print(table)

    resampled_share = drives["resampled"].mean()
    print(
        f"\n{resampled_share:.1%} of drives are in the resampling universe; "
        f"the rest are held at their observed points"
    )
    print(
        "  held observed: "
        + ", ".join(
            f"{row['result']} ({row['count']:,})"
            for row in drives.filter(~pl.col("resampled"))["result"]
            .value_counts()
            .sort("count", descending=True)
            .to_dicts()
        )
    )

    games = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / "game_components.parquet")
    dtw = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / "dtw_games.parquet").select(
        "game_id", "deserved_margin", "dtw_home"
    )

    # Two arms, both named in document 08 section 10 before this script existed.
    # "all drives" is the primary; "red-zone drives only" is the variant the
    # defect register promised, so that the part section 9's null result
    # DIRECTLY licenses stays separable from the extension by assumption.
    arms = {}
    for label, red_zone_only in (("all drives", False), ("red-zone drives only", True)):
        print(f"\n{'#' * 72}\n### ARM: {label}\n{'#' * 72}")
        arms[label] = run_arm(drives, table, games, dtw, red_zone_only=red_zone_only, label=label)

    diagnosis = diagnose(drives, table, arms["all drives"]["frame"])

    results = {
        "n_drives": int(drives.height),
        "resampled_share": float(resampled_share),
        "n_replicates": N_REPLICATES,
        "depth_bin_width": DEPTH_BIN_WIDTH,
        "conditional_table": table.to_dicts(),
        "primary_arm": "all drives",
        "arms": {label: arm["report"] for label, arm in arms.items()},
        "diagnosis": diagnosis,
        "random_seed": RANDOM_SEED,
    }
    arms["all drives"]["frame"].write_parquet(paths.RESEARCH_OUTPUT_DIR / "dq_games.parquet")
    arms["red-zone drives only"]["frame"].write_parquet(
        paths.RESEARCH_OUTPUT_DIR / "dq_games_redzone.parquet"
    )
    out = paths.RESEARCH_OUTPUT_DIR / "11_drive_bootstrap.json"
    with out.open("w") as handle:
        json.dump(results, handle, indent=2, default=str)
    print(f"\nWrote {out}")


def diagnose(drives: pl.DataFrame, table: pl.DataFrame, per_game: pl.DataFrame) -> dict:
    """Why Gate D-2 failed — a direct test of the defect named before the build.

    Document 08 §10's defect register called this out in advance as the largest
    known weakness: *"conditioning on depth understates offenses that score from
    distance."* If that is what happened, the resampling is not removing luck, it
    is removing **skill** — and a predictor with less skill in it is exactly what
    a non-inferiority failure looks like.

    The test is direct. If depth is too coarse a summary of a drive, then teams
    whose drives are *better than their depth implies* will be adjusted downward
    systematically. So: does a team's DQ adjustment run against its offensive
    quality? A strong negative correlation is the defect made visible.
    """
    print(f"\n{'#' * 72}\n### DIAGNOSIS (exploratory) — why did Gate D-2 fail?\n{'#' * 72}")

    universe = drives.filter(pl.col("resampled")).join(
        table.select("depth_bin", "mean_points"), on="depth_bin", how="left"
    )
    team_season = (
        universe.with_columns(
            pl.concat_str(
                [pl.col("season").cast(pl.String), pl.col("posteam")], separator="_"
            ).alias("team_season"),
            (pl.col("mean_points") - pl.col("points")).alias("adjustment"),
        )
        .group_by("team_season")
        .agg(
            pl.col("adjustment").sum().alias("total_adjustment"),
            pl.col("points").sum().alias("points"),
            pl.len().alias("drives"),
        )
        .with_columns((pl.col("points") / pl.col("drives")).alias("points_per_drive"))
        .sort("team_season")
    )

    quality = team_season["points_per_drive"].to_numpy()
    adjustment = (team_season["total_adjustment"] / team_season["drives"]).to_numpy()
    correlation = float(np.corrcoef(quality, adjustment)[0, 1])

    print(
        f"  {team_season.height} team-seasons\n"
        f"  corr(offensive points per drive, DQ adjustment per drive) = {correlation:+.3f}"
    )
    print(
        "  A strong negative value means the resampling systematically marks good\n"
        "  offenses down and bad offenses up — that is skill being erased, not luck."
    )

    # How much of the between-team spread in scoring survives the adjustment?
    before = float(np.std(quality, ddof=1))
    after = float(np.std(quality + adjustment, ddof=1))
    print(
        f"\n  between-team SD of points per drive: {before:.4f} observed -> "
        f"{after:.4f} after adjustment ({after / before:.1%} retained)"
    )

    return {
        "note": "EXPLORATORY. Run after Gate D-2 failed, to test the mechanism "
        "document 08 §10 named in advance. Changes no verdict.",
        "n_team_seasons": int(team_season.height),
        "corr_quality_vs_adjustment": correlation,
        "between_team_sd_observed": before,
        "between_team_sd_after_adjustment": after,
        "share_of_spread_retained": after / before,
    }


def run_arm(
    drives: pl.DataFrame,
    table: pl.DataFrame,
    games: pl.DataFrame,
    dtw: pl.DataFrame,
    *,
    red_zone_only: bool,
    label: str,
) -> dict:
    """One resampling arm, end to end through all three gates."""
    print(f"Bootstrap, {N_REPLICATES} replicates")
    per_game, extras = resample(
        drives, table, N_REPLICATES, RANDOM_SEED, red_zone_only=red_zone_only
    )
    deltas = extras["deltas"]

    joined = (
        games.select("game_id", "season", "week", "home_team", "away_team", "margin")
        .join(per_game, on="game_id", how="inner")
        .join(dtw, on="game_id", how="inner")
        .drop_nulls("margin")
    )
    index = {game_id: i for i, game_id in enumerate(extras["game_ids"])}
    columns = np.array([index[g] for g in joined["game_id"].to_list()])
    actual = joined["margin"].to_numpy().astype(float)
    replicated_margins = actual[None, :] + deltas[:, columns]

    joined = joined.with_columns(
        (pl.col("margin") + pl.col("expected_delta")).alias("dq_margin"),
        pl.Series("dqw_home", (replicated_margins > 0).mean(axis=0)),
    )

    shift = (joined["dqw_home"] - (joined["margin"] > 0).cast(pl.Float64)).abs()
    flipped = joined.filter(
        ((pl.col("margin") > 0) & (pl.col("dqw_home") < 0.5))
        | ((pl.col("margin") < 0) & (pl.col("dqw_home") > 0.5))
    )
    adjustment = float((joined["dq_margin"] - joined["margin"]).abs().mean())
    print("\n=== What the resampling does to ten seasons ===")
    print(f"  games                          {joined.height:,}")
    print(f"  mean |DQ margin - actual|      {adjustment:.2f} points")
    print(f"  mean |DQW% - actual result|    {shift.mean():.3f}")
    print(
        f"  games where DQW% disagrees     {flipped.height:,} "
        f"({flipped.height / joined.height:.1%})"
    )

    correlation = float(np.corrcoef(joined["dqw_home"], joined["dtw_home"])[0, 1])
    disagree = joined.filter(
        ((pl.col("dqw_home") > 0.5) & (pl.col("dtw_home") < 0.5))
        | ((pl.col("dqw_home") < 0.5) & (pl.col("dtw_home") > 0.5))
    ).height
    print("\nGATE D-3 — are DQW% and DTW% distinct? (descriptive)")
    print(f"  corr(DQW%, DTW%) = {correlation:.3f}")
    print(
        f"  games where they name different winners: {disagree:,} ({disagree / joined.height:.1%})"
    )
    print(
        "  Document 08 section 10 recorded in advance that a correlation above ~0.95\n"
        "  would mean reporting both is redundant."
    )

    print("\nGATE D-2 — non-inferiority on the rematch harness")
    gate_d2 = rematch_validation(games, joined)

    return {
        "frame": joined,
        "report": {
            "label": label,
            "red_zone_only": red_zone_only,
            "gate_d1_unbiased": extras["gate_d1"],
            "gate_d2_rematch": gate_d2,
            "gate_d3_distinctness": {
                "corr_dqw_dtw": correlation,
                "games_naming_different_winners": disagree,
                "n_games": joined.height,
            },
            "summary": {
                "games": joined.height,
                "mean_abs_margin_adjustment": adjustment,
                "mean_abs_dqw_shift": float(shift.mean()),
                "games_dqw_disagrees_with_result": flipped.height,
            },
        },
    }


def rematch_validation(games: pl.DataFrame, dq: pl.DataFrame) -> dict:
    """Gate D-2, on the identical harness documents 06 and 07 pre-registered."""
    pairs = _rematch.rematch_pairs(games)
    keyed = (
        games.drop_nulls("margin")
        .with_columns(
            pl.min_horizontal("home_team", "away_team").alias("t1"),
            pl.max_horizontal("home_team", "away_team").alias("t2"),
        )
        .with_columns(
            pl.concat_str([pl.col("season").cast(pl.String), "t1", "t2"], separator="_").alias(
                "pair"
            )
        )
        .sort(["pair", "week"])
        .with_columns(pl.int_range(pl.len()).over("pair").alias("meeting"))
        .filter(pl.col("meeting") == 0)
        .select("pair", "game_id")
    )
    joined = pairs.join(keyed, on="pair", how="inner").join(
        dq.select("game_id", "dq_margin", "dqw_home", "deserved_margin"), on="game_id", how="inner"
    )
    print(f"  {joined.height} rematch pairs with a drive-quality margin")

    actual = joined["margin_g1_a"].to_numpy().astype(float)
    y = (joined["margin_g2_a"].to_numpy() > 0).astype(float)
    a_home = joined["a_home_g2"].to_numpy().astype(float)
    rng = np.random.default_rng(RANDOM_SEED)
    folds = rng.permutation(joined.height) % N_FOLDS

    def arm(predictor: np.ndarray, label: str) -> dict:
        per_pair = _rematch.paired_log_loss_diff(actual, predictor, y, a_home, folds)
        mean, se, superior = _rematch.decision(per_pair)
        ci = (mean - 1.96 * se, mean + 1.96 * se)
        passed = _rematch.passes_noninferiority(mean, se)
        print(
            f"  {label:26s} delta log loss {mean:+.5f} (SE {se:.5f}) "
            f"95% CI [{ci[0]:+.5f}, {ci[1]:+.5f}]  "
            f"{'PASS' if passed else 'FAIL'} vs margin {NONINFERIORITY_MARGIN:+.3f}"
        )
        return {
            "label": label,
            "mean_delta_log_loss": mean,
            "se": se,
            "ci95": list(ci),
            "noninferiority_pass": bool(passed),
            "favours_dq": bool(mean < 0),
            "superiority_would_reject": bool(superior),
        }

    # Order fixed by document 08 §10: margin primary, probability secondary,
    # because document 07 measured that a probability discards margin information.
    primary = arm(joined["dq_margin"].to_numpy().astype(float), "PRIMARY DQ margin")
    secondary = arm(joined["dqw_home"].to_numpy().astype(float), "SECONDARY DQW%")

    # Sanity check on the harness, exactly as document 06 Gate 3 specified.
    x = _rematch.design_matrix(joined["dq_margin"].to_numpy().astype(float), a_home)
    beta = _rematch.fit_logistic(x, y)
    p = 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -30, 30)))
    w = np.clip(p * (1.0 - p), 1e-9, None)
    se_b1 = float(np.sqrt(np.linalg.inv(x.T @ (x * w[:, None]))[1, 1]))
    coefficient = {
        "b1": float(beta[1]),
        "se": se_b1,
        "ci95": [beta[1] - 1.96 * se_b1, beta[1] + 1.96 * se_b1],
        "pass": bool(beta[1] > 0 and beta[1] - 1.96 * se_b1 > 0),
    }
    print(
        f"  coefficient sanity        b1 = {coefficient['b1']:+.4f} "
        f"[{coefficient['ci95'][0]:+.4f}, {coefficient['ci95'][1]:+.4f}]  "
        f"{'PASS' if coefficient['pass'] else 'FAIL'}"
    )

    return {
        "n_pairs": joined.height,
        "primary_dq_margin": primary,
        "secondary_dqw": secondary,
        "coefficient_sanity": coefficient,
        "noninferiority_margin": NONINFERIORITY_MARGIN,
    }


if __name__ == "__main__":
    main()
