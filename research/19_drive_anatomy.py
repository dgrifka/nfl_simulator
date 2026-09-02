"""Phase 4, step 1 — drive anatomy, run against the gates document 11 committed.

Design, statistics and thresholds are fixed by `docs/research/11-drive-anatomy.md`,
committed at `03c6a66` before this script produced a result. Nothing here chooses
anything; it executes what that document committed to.

    Gate DA-1   positive control — points per drive must persist far above the
                null, or the drive table is broken and nothing else is readable.
    Gate DA-1b  replication control — the reimplemented depth instrument must
                reproduce document 08 §11's 70.6% spread retention, the number
                that killed DQW%.
    Gate DA-2   per feature set — does the finishing residual persist? Reported
                against the permutation null's 95th percentile, with power.
    Gate DA-3   honesty — a null is only interpretable where power at r = 0.12
                is at least 0.80.
    Gate DA-4   descriptive — spread retention and the sign-flipped twin of
                document 08 §11's corr(quality, adjustment) = -0.784.

The power calculation's own machinery is imported rather than re-implemented, so
the executed statistic is literally the simulated one.

    uv run python research/19_drive_anatomy.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_power = import_module("19_drive_anatomy_power")
_seq = import_module("10_sequencing_power")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.ingest import PBP_SEASONS, load_pbp  # noqa: E402

RANDOM_SEED = _power.RANDOM_SEED
N_SPLITS = _power.N_SPLITS
N_FOLDS = _power.N_FOLDS
FEATURE_SETS = _power.FEATURE_SETS
REFERENCE_R = _power.REFERENCE_R
MIN_POWER = 0.80

# Pre-registered in document 11 §4 and §9, from the permutation null in
# `research/outputs/19_drive_anatomy_power.json`. Hard-coded here rather than read
# back, so a re-run of the power script cannot silently move a committed gate.
GATE_DA2_THRESHOLDS = {
    "resid_F0_cellmean": 0.0705,
    "resid_F1_depth": 0.0706,
    "resid_F2_advance": 0.0674,
    "resid_F3_production": 0.0695,
    "resid_F4_yardage": 0.0642,
}
GATE_DA1_THRESHOLD = 0.0891  # null 99th percentile, points per drive

# Document 08 §11: "Only 70.6% of the real spread between offenses survives."
GATE_DA1B_TARGET = 0.706
GATE_DA1B_TOLERANCE = 0.05


def spread_retention(
    drives: pl.DataFrame, predicted: np.ndarray
) -> tuple[float, float, float, float]:
    """How much between-team scoring spread a conditional mean retains.

    Document 08 §11 measured this as *"between-team SD of points per drive: 0.490
    observed -> 0.346 after adjustment"*, where the adjustment replaced each
    drive's points with its bin mean. Replacing points by ``predicted`` is the
    same operation, so a team-season's adjusted points per drive is exactly its
    mean predicted points per drive. The ratio of the two between-team SDs is the
    share of the real spread the summary can still see.

    Also returned is the sign-flipped twin of that document's headline
    diagnostic. It reported ``corr(quality, adjustment) = -0.784`` with
    ``adjustment = predicted - points``; the residual is ``points - predicted``,
    so a strong POSITIVE correlation here is the same finding.
    """
    frame = (
        drives.with_columns(pl.Series("predicted", predicted))
        .group_by("team_season")
        .agg(
            pl.col("points").mean().alias("points_per_drive"),
            pl.col("predicted").mean().alias("predicted_per_drive"),
            pl.len().alias("drives"),
        )
        .sort("team_season")
    )
    observed = frame["points_per_drive"].to_numpy()
    modelled = frame["predicted_per_drive"].to_numpy()
    residual_rate = observed - modelled

    sd_observed = float(np.std(observed, ddof=1))
    sd_modelled = float(np.std(modelled, ddof=1))
    return (
        sd_observed,
        sd_modelled,
        sd_modelled / sd_observed,
        float(np.corrcoef(observed, residual_rate)[0, 1]),
    )


def describe_drives(drives: pl.DataFrame) -> dict:
    """The per-drive summary table itself — the deliverable before any statistic."""
    print(f"\n{'=' * 72}\nTHE DRIVE SUMMARY TABLE\n{'=' * 72}")

    results = (
        drives["result"]
        .value_counts()
        .sort("count", descending=True)
        .with_columns((pl.col("count") / drives.height).alias("share"))
    )
    with pl.Config(tbl_rows=12):
        print(results)

    feature_columns = [
        "start_yardline_100",
        "depth",
        "scrimmage_plays",
        "net_yards",
        "max_gain",
        "explosive_plays",
        "first_downs",
        "penalty_aid_yards",
    ]
    summary = drives.select(
        [pl.col(column).mean().alias(f"{column}__mean") for column in feature_columns]
        + [pl.col(column).median().alias(f"{column}__median") for column in feature_columns]
    ).to_dicts()[0]
    print("\n  feature                mean    median")
    for column in feature_columns:
        print(
            f"  {column:22s} {summary[f'{column}__mean']:6.2f}  {summary[f'{column}__median']:6.1f}"
        )

    # Points by depth bin: the conditional table document 08 shipped, printed so a
    # reader can see the monotonicity that made it look reasonable.
    by_depth = (
        drives.with_columns(((pl.col("depth") // 5) * 5).cast(pl.Int32).alias("depth_bin"))
        .group_by("depth_bin")
        .agg(pl.len().alias("drives"), pl.col("points").mean().alias("points_per_drive"))
        .sort("depth_bin")
    )
    print("\n=== League points per drive by deepest scrimmage snap ===")
    with pl.Config(tbl_rows=25):
        print(by_depth)

    # The entanglement §2 named, measured rather than asserted.
    touchdowns = drives.filter(pl.col("result") == "Touchdown")
    yardage_gap = (touchdowns["net_yards"] - touchdowns["start_yardline_100"]).abs()
    entanglement = {
        "touchdown_drives": int(touchdowns.height),
        "share_net_yards_within_5_of_start_on_td": float((yardage_gap <= 5).mean()),
        "corr_net_yards_points": float(
            np.corrcoef(drives["net_yards"].to_numpy(), drives["points"].to_numpy())[0, 1]
        ),
        "corr_depth_points": float(
            np.corrcoef(drives["depth"].to_numpy(), drives["points"].to_numpy())[0, 1]
        ),
    }
    print(
        f"\n  net_yards entanglement: on {entanglement['touchdown_drives']:,} touchdown drives, "
        f"{entanglement['share_net_yards_within_5_of_start_on_td']:.1%} have net yards within "
        f"5 of the starting distance to the goal"
    )

    return {
        "results": results.to_dicts(),
        "feature_summary": summary,
        "points_by_depth_bin": by_depth.to_dicts(),
        "entanglement": entanglement,
    }


def team_game_quality(pbp: pl.DataFrame) -> pl.DataFrame:
    """Offensive EPA per scrimmage play, per team-game — the quality control.

    This is document 08's S0, the measure that round used as its positive control
    and measured at split-half ``r = +0.601``. It is the cleanest available
    statement of "how good was this offense", and it is computed on the same
    team-games the drive table uses so a half of a team-season carries both.
    """
    return (
        pbp.filter(
            pl.col("posteam").is_not_null()
            & pl.col("play_type").is_in(["pass", "run"])
            & pl.col("epa").is_not_null()
        )
        .group_by(["season", "posteam", "game_id"])
        .agg(pl.len().alias("n_plays"), pl.col("epa").sum().alias("epa_sum"))
        .with_columns(
            pl.concat_str(
                [pl.col("season").cast(pl.String), pl.col("posteam")], separator="_"
            ).alias("team_season")
        )
        .select("team_season", "game_id", "n_plays", "epa_sum")
    )


def quality_controlled_persistence(
    gated: np.ndarray,
    quality: np.ndarray,
    mask: np.ndarray,
    starts: np.ndarray,
    column: int,
) -> np.ndarray:
    """Does the finishing residual persist *beyond* the offense's overall quality?

    **EXPLORATORY, not pre-registered.** Added after Gate DA-2 found that every
    residual persists, to reconcile that with document 08 §9's finding that the
    red-zone gap does not (``r = -0.034`` at 87% power). The two are not the same
    quantity and the difference is the whole point: S1 is a **gap** — it subtracts
    the team's own overall efficiency, so a uniformly good offense scores exactly
    zero — while the finishing residual here subtracts only what a *league*
    conditional mean predicts from the drive's summary. A team that is simply
    better at football therefore scores positive on the residual and zero on S1.

    So the residual's persistence has two candidate explanations, and they imply
    opposite things for a successor measure:

    1. **The drive summary is an incomplete measure of production.** Good
       offenses out-score their own drive summaries because the summary does not
       capture everything they did. Then the persistence is ordinary offensive
       quality, already counted in ``core``, and a successor that resamples
       relative to the team's own within-game baseline is licensed.
    2. **Finishing given a fixed drive summary is a genuine, separate skill.**
       Then the DQW% line is dead at every summary, because resampling finishing
       always erases something real.

    The test separates them. Within each half, the team-season residual rate is
    regressed on that half's own offensive EPA per play and the regression
    residual is kept; the two halves' regression residuals are then correlated.
    Both controls are computed inside their own half, so nothing leaks across the
    split. If persistence survives the control, explanation 2 holds.
    """
    totals_gated = np.add.reduceat(gated, starts, axis=0)
    totals_quality = np.add.reduceat(quality, starts, axis=0)

    sums_a = _seq.half_sums(gated, mask, starts)
    sums_b = totals_gated[None, :, :] - sums_a
    quality_a = _seq.half_sums(quality, mask, starts)
    quality_b = totals_quality[None, :, :] - quality_a

    def controlled(sums: np.ndarray, quality_sums: np.ndarray) -> np.ndarray:
        rate = sums[:, :, column] / sums[:, :, 0]
        control = quality_sums[:, :, 2] / quality_sums[:, :, 1]
        rate_centred = rate - rate.mean(axis=1, keepdims=True)
        control_centred = control - control.mean(axis=1, keepdims=True)
        slope = (rate_centred * control_centred).sum(axis=1, keepdims=True) / (
            control_centred * control_centred
        ).sum(axis=1, keepdims=True)
        return rate_centred - slope * control_centred

    return _seq.correlate(controlled(sums_a, quality_a), controlled(sums_b, quality_b))


def main() -> None:
    paths.ensure_data_dirs()
    pbp = load_pbp(PBP_SEASONS, columns=[*_power.COLUMNS, "epa", "down"])
    drives = _power.drive_table(pbp).filter(pl.col("resampled"))
    print(f"{drives.height:,} offensive drives in the resampling universe")

    anatomy = describe_drives(drives)

    points = drives["points"].to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    folds = rng.permutation(len(points)) % N_FOLDS

    print(f"\n{'=' * 72}\nNESTED CONDITIONAL MEANS (out of fold)\n{'=' * 72}")
    baseline_variance = float(np.var(points, ddof=1))
    residuals: dict[str, np.ndarray] = {}
    predictions: dict[str, np.ndarray] = {}

    cell = _power.cell_mean_predictions(drives["depth"].to_numpy(), points, folds)
    predictions["F0_cellmean"] = cell
    residuals["F0_cellmean"] = points - cell

    for name, columns in FEATURE_SETS.items():
        features = np.column_stack([drives[column].to_numpy() for column in columns])
        predicted = _power.oof_predictions(features, points, folds)
        predictions[name] = predicted
        residuals[name] = points - predicted

    fit_rows = []
    for name, predicted in predictions.items():
        residual = residuals[name]
        observed_sd, modelled_sd, retained, correlation = spread_retention(drives, predicted)
        fit_rows.append(
            {
                "feature_set": name,
                "oof_r2": 1.0 - float(np.var(residual, ddof=1)) / baseline_variance,
                "residual_sd": float(np.std(residual, ddof=1)),
                "between_team_sd_observed": observed_sd,
                "between_team_sd_modelled": modelled_sd,
                "share_of_spread_retained": retained,
                "corr_quality_vs_residual": correlation,
            }
        )
        print(
            f"  {name:16s} OOF R2 {fit_rows[-1]['oof_r2']:.4f}   "
            f"spread retained {retained:6.1%}   "
            f"corr(quality, residual) {correlation:+.3f}"
        )

    # ---- the split-half machinery ----------------------------------------
    grouped, measures = _power.team_game_matrix(drives, residuals)
    matrix, starts, sizes = _power.to_dense(grouped, measures)
    split_rng = np.random.default_rng(RANDOM_SEED)
    mask = _seq.split_masks(starts, sizes, len(matrix), split_rng, N_SPLITS)
    per_split = _power.split_half_distribution(matrix, mask, starts)

    observed_r = {name: float(per_split[:, m].mean()) for m, name in enumerate(measures)}
    band = {
        name: (float(np.percentile(per_split[:, m], 5)), float(np.percentile(per_split[:, m], 95)))
        for m, name in enumerate(measures)
    }

    # ---- Gate DA-1 --------------------------------------------------------
    print(f"\n{'=' * 72}\nGATE DA-1 — positive control\n{'=' * 72}")
    gate_da1 = bool(observed_r["points"] > GATE_DA1_THRESHOLD)
    print(
        f"  points per drive split-half r = {observed_r['points']:+.4f} "
        f"[{band['points'][0]:+.4f}, {band['points'][1]:+.4f}] "
        f"vs null 99th pct {GATE_DA1_THRESHOLD:.4f}: {'PASS' if gate_da1 else 'FAIL'}"
    )
    if not gate_da1:
        print("\n  GATE DA-1 FAILED — the drive table is broken; nothing below is readable.")

    # ---- Gate DA-1b -------------------------------------------------------
    print(f"\n{'=' * 72}\nGATE DA-1b — replication control\n{'=' * 72}")
    f0 = next(row for row in fit_rows if row["feature_set"] == "F0_cellmean")
    gap = abs(f0["share_of_spread_retained"] - GATE_DA1B_TARGET)
    gate_da1b = bool(gap <= GATE_DA1B_TOLERANCE)
    print(
        f"  F0 (depth cell means) retains {f0['share_of_spread_retained']:.1%} of between-team "
        f"spread\n  document 08 §11 measured {GATE_DA1B_TARGET:.1%} for the shipped instrument; "
        f"gap {gap:.1%} vs tolerance {GATE_DA1B_TOLERANCE:.0%}: {'PASS' if gate_da1b else 'FAIL'}"
    )
    print(
        f"  document 08 §11 measured corr(quality, adjustment) = -0.784; the sign-flipped\n"
        f"  twin here is corr(quality, residual) = {f0['corr_quality_vs_residual']:+.3f}"
    )

    # ---- Gates DA-2 and DA-3 ---------------------------------------------
    power_lookup = {
        (row["measure"], row["target_true_r"]): row["power"]
        for row in json.load((paths.RESEARCH_OUTPUT_DIR / "19_drive_anatomy_power.json").open())[
            "power"
        ]
    }

    print(f"\n{'=' * 72}\nGATE DA-2 — does the finishing residual persist?\n{'=' * 72}")
    verdicts = []
    for name, threshold in GATE_DA2_THRESHOLDS.items():
        value = observed_r[name]
        persists = bool(value > threshold)
        power_at_reference = float(power_lookup[(name, REFERENCE_R)])
        interpretable = bool(power_at_reference >= MIN_POWER)

        if persists:
            reading = "SKILL SURVIVES this summary — resampling its residual would erase it"
        elif interpretable:
            reading = "residual does NOT persist — resampling it is licensed"
        else:
            reading = "UNRESOLVABLE — underpowered, no reading"

        print(
            f"  {name:20s} r = {value:+.4f} [{band[name][0]:+.4f}, {band[name][1]:+.4f}]  "
            f"threshold {threshold:.4f}  power at r={REFERENCE_R} {power_at_reference:.2f} "
            f"({'DA-3 pass' if interpretable else 'DA-3 FAIL'})"
        )
        print(f"      -> {reading}")
        verdicts.append(
            {
                "measure": name,
                "split_half_r": value,
                "r_p05": band[name][0],
                "r_p95": band[name][1],
                "gate_da2_threshold": threshold,
                "persists": persists,
                "power_at_reference_r": power_at_reference,
                "gate_da3_pass": interpretable,
                "reading": reading,
            }
        )

    # ---- Gate DA-4 --------------------------------------------------------
    print(f"\n{'=' * 72}\nGATE DA-4 — spread retention per feature set (descriptive)\n{'=' * 72}")
    print("  feature set        OOF R2   spread retained   corr(quality, residual)")
    for row in fit_rows:
        print(
            f"  {row['feature_set']:16s}  {row['oof_r2']:.4f}     "
            f"{row['share_of_spread_retained']:6.1%}            "
            f"{row['corr_quality_vs_residual']:+.3f}"
        )
    print(
        "\n  A summary that retains ~100% of between-team spread and leaves a residual\n"
        "  uncorrelated with quality is one whose resampling would not erase skill.\n"
        "  Document 08's shipped instrument retained 70.6% and left +0.784."
    )

    # ---- exploratory: reconciling DA-2 with document 08 §9's S1 null ------
    print(
        f"\n{'=' * 72}\nEXPLORATORY (not pre-registered) — is the residual just quality?\n{'=' * 72}"
    )
    quality_frame = (
        grouped.select("team_season", "game_id")
        .join(team_game_quality(pbp), on=["team_season", "game_id"], how="left")
        .with_columns(pl.col("n_plays").fill_null(0.0), pl.col("epa_sum").fill_null(0.0))
        .with_columns(pl.lit(1.0).alias("drives"))
        .sort(["team_season", "game_id"])
    )
    quality_matrix, quality_starts, _ = _power.to_dense(quality_frame, ("n_plays", "epa_sum"))
    if not np.array_equal(quality_starts, starts):
        raise ValueError("quality matrix rows do not align with the gated matrix")

    exploratory = []
    controlled_rng = np.random.default_rng(RANDOM_SEED + 7)
    for name in ("resid_F1_depth", "resid_F3_production", "resid_F4_yardage"):
        column = measures.index(name) + 1
        draws = quality_controlled_persistence(matrix, quality_matrix, mask, starts, column)
        # Exploratory threshold: the same permutation instrument, on the same
        # statistic, at a reduced replicate count because nothing is gated on it.
        null_draws = np.empty(200)
        for replicate in range(200):
            order = controlled_rng.permutation(len(matrix))
            null_draws[replicate] = quality_controlled_persistence(
                matrix[order], quality_matrix[order], mask, starts, column
            ).mean()
        threshold = float(np.percentile(null_draws, 95))
        exploratory.append(
            {
                "measure": name,
                "uncontrolled_r": observed_r[name],
                "quality_controlled_r": float(draws.mean()),
                "exploratory_null_p95": threshold,
                "survives_control": bool(draws.mean() > threshold),
            }
        )
        print(
            f"  {name:20s} r = {observed_r[name]:+.4f} uncontrolled -> "
            f"{draws.mean():+.4f} controlled for offensive EPA/play "
            f"(exploratory null 95th pct {threshold:.4f})"
        )
    print(
        "\n  Document 08 §9 measured the red-zone GAP at r = -0.034 with 87% power. A gap\n"
        "  subtracts the team's own overall efficiency; the residual above does not. If the\n"
        "  controlled column collapses toward the null, the two findings agree and the\n"
        "  residual's persistence was ordinary offensive quality the summary could not see."
    )

    # ---- exploratory: which scoring channel is the residual persisting in? --
    print(f"\n{'=' * 72}\nEXPLORATORY (not pre-registered) — which channel persists?\n{'=' * 72}")
    print(
        "  The residual survived the quality control, so it is not simply 'being good at\n"
        "  football'. A drive's POINTS bundle three things document 08 §9's S1 never\n"
        "  measured: whether the drive reached the end zone, whether a field-goal attempt\n"
        "  was made, and the fourth-down decision that chose between them. Kicker skill in\n"
        "  particular is a known, sized, persistent team property (document 05b: sigma 0.342)\n"
        "  and it is ALREADY neutralized inside DTW%. Valuing the same drives three ways\n"
        "  separates the channels."
    )
    is_touchdown = (drives["result"] == "Touchdown").to_numpy().astype(float)
    is_field_goal = (drives["result"] == "Field goal").to_numpy().astype(float)
    valuations = {
        "points (all channels)": points,
        "touchdown points only": 6.0 * is_touchdown,
        "field-goal points only": 3.0 * is_field_goal,
    }
    f3_features = np.column_stack(
        [drives[column].to_numpy() for column in FEATURE_SETS["F3_production"]]
    )
    channel_rows = []
    channel_rng = np.random.default_rng(RANDOM_SEED + 13)
    for label, values in valuations.items():
        residual = values - _power.oof_predictions(f3_features, values, folds)
        channel_grouped, channel_measures = _power.team_game_matrix(drives, {"channel": residual})
        channel_matrix, channel_starts, _ = _power.to_dense(channel_grouped, channel_measures)
        observed = float(
            _power.split_half_distribution(channel_matrix, mask, channel_starts)[:, 1].mean()
        )
        null_draws = np.empty(200)
        for replicate in range(200):
            shuffled = channel_matrix[channel_rng.permutation(len(channel_matrix))]
            totals = np.add.reduceat(shuffled, channel_starts, axis=0)
            null_draws[replicate] = _power.split_half_r(shuffled, mask, channel_starts, totals)[1]
        threshold = float(np.percentile(null_draws, 95))
        channel_rows.append(
            {
                "valuation": label,
                "split_half_r": observed,
                "exploratory_null_p95": threshold,
                "persists": bool(observed > threshold),
                "share_of_drives_scoring": float((values > 0).mean()),
            }
        )
        print(
            f"  {label:24s} residual split-half r = {observed:+.4f}  "
            f"(exploratory null 95th pct {threshold:.4f})  "
            f"{'PERSISTS' if observed > threshold else 'flat'}"
        )

    results = {
        "n_drives": int(drives.height),
        "n_team_seasons": int(len(starts)),
        "n_team_games": int(len(matrix)),
        "anatomy": anatomy,
        "nested_fits": fit_rows,
        "feature_sets": {name: list(columns) for name, columns in FEATURE_SETS.items()},
        "gate_da1_positive_control": {
            "split_half_r": observed_r["points"],
            "r_p05": band["points"][0],
            "r_p95": band["points"][1],
            "threshold": GATE_DA1_THRESHOLD,
            "pass": gate_da1,
        },
        "gate_da1b_replication": {
            "share_of_spread_retained": f0["share_of_spread_retained"],
            "target": GATE_DA1B_TARGET,
            "tolerance": GATE_DA1B_TOLERANCE,
            "gap": gap,
            "pass": gate_da1b,
            "corr_quality_vs_residual": f0["corr_quality_vs_residual"],
        },
        "gate_da2_verdicts": verdicts,
        "exploratory_quality_controlled": {
            "note": "EXPLORATORY, not pre-registered. Added after Gate DA-2 found every "
            "residual persists, to reconcile that with document 08 §9's S1 null result.",
            "rows": exploratory,
        },
        "exploratory_scoring_channels": {
            "note": "EXPLORATORY, not pre-registered. Added after the quality control failed "
            "to explain the residual's persistence. Same drives, three valuations.",
            "rows": channel_rows,
        },
        "n_splits": N_SPLITS,
        "random_seed": RANDOM_SEED,
    }
    out = paths.RESEARCH_OUTPUT_DIR / "19_drive_anatomy.json"
    with out.open("w") as handle:
        json.dump(results, handle, indent=2)

    # The drive table itself is the foundation step 2 builds on, so it is
    # persisted rather than rebuilt from scratch there.
    drives.write_parquet(paths.RESEARCH_OUTPUT_DIR / "drive_anatomy.parquet")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
