"""Phase 4, step 2 — the DQW% successor, run against the criteria document 12 committed.

Design, statistics and thresholds are fixed by `docs/research/12-dq-successor.md`,
committed at `b0bc656` before this script existed. Nothing here chooses anything.

    SC-1  the finishing residual must not persist       (binding; threshold 0.0669)
    SC-2  between-team spread must survive              (retention >= 0.95)
    SC-3  quality correlation net of its mechanical floor  (<= 0.0559)
    SC-4  the measure must not be vacuous               (>= 1.0 pt, >= 5% of games)

    Gate E-1  the resampling is unbiased on the league
    Gate E-2  non-inferiority on the rematch harness, +0.010 log loss
    Gate E-3  DQW% and DTW% are distinct — descriptive

**The sufficiency criteria run first and can stop the measure before it reaches
Gate E-2.** Document 12 §5 is why: the non-inferiority gate has zero power
against a predictor that erases 10% of true team strength, so a measure that
reaches a blind gate and passes has learned nothing.

    uv run python research/20_dq_successor.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_power = import_module("20_dq_successor_power")
_anatomy = import_module("19_drive_anatomy_power")
_seq = import_module("10_sequencing_power")
_rematch = import_module("08_rematch_power")

from nfl_simulator import paths  # noqa: E402

RANDOM_SEED = _power.RANDOM_SEED
N_SPLITS = _power.N_SPLITS
N_FOLDS = _power.N_FOLDS
FEATURES = _power.FEATURES
N_PREDICTION_BINS = _power.N_PREDICTION_BINS
REFERENCE_R = _power.REFERENCE_R
N_REPLICATES = 2000

# Pre-registered in document 12 §10. Hard-coded rather than read back, so a re-run
# of the power script cannot silently move a committed threshold.
SC1_THRESHOLD = 0.0669
SC2_THRESHOLD = 0.95
SC3_THRESHOLD = 0.0559
SC4_MIN_ADJUSTMENT = 1.0
SC4_MIN_FLIP_SHARE = 0.05
MIN_POWER = 0.80

GATE_E1_TOLERANCE = 0.01
NONINFERIORITY_MARGIN = _rematch.NONINFERIORITY_MARGIN


def equal_count_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin index by rank, so every bin holds the same number of drives.

    Equal-count rather than equal-width because the predicted-value distribution
    is heavily skewed — most drives predict close to zero points — and equal-width
    bins would put nearly everything in one cell.
    """
    order = np.argsort(values, kind="stable")
    bins = np.empty(len(values), dtype=int)
    bins[order] = (np.arange(len(values)) * n_bins) // len(values)
    return bins


def per_game_retention(drives: pl.DataFrame, adjustment: np.ndarray) -> tuple[float, float, float]:
    """SC-2: between-team spread in offensive points per game, before and after.

    Stated per **game** rather than per drive so it maps one-for-one onto the
    ``retention`` parameter of document 12 §5's skill-erasure instrument — which
    is the quantity the rematch gate was shown to be blind to.
    """
    frame = (
        drives.with_columns(pl.Series("adjustment", adjustment))
        .group_by("team_season")
        .agg(
            pl.col("points").sum().alias("points"),
            pl.col("adjustment").sum().alias("adjustment"),
            pl.col("game_id").n_unique().alias("games"),
        )
        .sort("team_season")
    )
    games = frame["games"].to_numpy()
    observed = frame["points"].to_numpy() / games
    adjusted = (frame["points"].to_numpy() + frame["adjustment"].to_numpy()) / games
    sd_observed = float(np.std(observed, ddof=1))
    sd_adjusted = float(np.std(adjusted, ddof=1))
    return sd_observed, sd_adjusted, sd_adjusted / sd_observed


def incumbent_reference(all_drives: pl.DataFrame) -> dict:
    """Document 08's shipped instrument, restated in SC-2's per-game units.

    The 70.6% that document 08 §11 published is a per-**drive** figure on its own
    universe, so it is not directly comparable to SC-2. Recomputing the incumbent
    in SC-2's units is what makes the successor's number readable against the
    design whose fate is known.
    """
    points = all_drives["points"].to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    folds = rng.permutation(len(points)) % N_FOLDS
    predicted = _anatomy.cell_mean_predictions(all_drives["depth"].to_numpy(), points, folds)
    sd_observed, sd_adjusted, retention = per_game_retention(all_drives, predicted - points)
    return {
        "label": "document 08's depth-bin instrument, in SC-2 units",
        "between_team_sd_observed": sd_observed,
        "between_team_sd_adjusted": sd_adjusted,
        "share_of_spread_retained": retention,
    }


def resample(
    drives: pl.DataFrame,
    bins: np.ndarray,
    bin_means: np.ndarray,
    mask: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-game deltas, one row per replicate, plus the Gate E-1 identity check.

    ``mask`` selects which drives are actually redrawn; the pools are always built
    from **every** league drive in the bin, even when only a subset is being
    resampled. Rebuilding the pool from the subset would truncate it at the
    subset's own boundary and quietly change the distribution being drawn from —
    the same ruling document 08 §10's implementation made for its red-zone arm.
    """
    rng = np.random.default_rng(seed)
    points = drives["points"].to_numpy()
    pools = {value: points[bins == value] for value in np.unique(bins)}

    game_ids = drives["game_id"].unique(maintain_order=True).to_list()
    index = {game_id: i for i, game_id in enumerate(game_ids)}
    row_game = np.array([index[g] for g in drives["game_id"].to_list()])
    row_sign = np.where(drives["is_home"].to_numpy(), 1.0, -1.0)

    active = np.flatnonzero(mask)
    positions = {value: active[bins[active] == value] for value in np.unique(bins)}

    expected_delta = np.bincount(
        row_game[active],
        weights=row_sign[active] * (bin_means[bins[active]] - points[active]),
        minlength=len(game_ids),
    )

    deltas = np.empty((replicates, len(game_ids)))
    sampled_means = np.empty(replicates)
    for replicate in range(replicates):
        drawn = points.copy()
        for value, rows in positions.items():
            pool = pools[value]
            drawn[rows] = pool[rng.integers(0, len(pool), size=len(rows))]
        sampled_means[replicate] = drawn[active].mean()
        deltas[replicate] = np.bincount(
            row_game, weights=row_sign * (drawn - points), minlength=len(game_ids)
        )
        if (replicate + 1) % 500 == 0:
            print(f"    replicate {replicate + 1}/{replicates}", flush=True)

    observed_mean = float(points[active].mean())
    resampled_mean = float(sampled_means.mean())
    gate_e1 = {
        "observed_mean_points_per_drive": observed_mean,
        "resampled_mean_points_per_drive": resampled_mean,
        "difference": resampled_mean - observed_mean,
        "tolerance": GATE_E1_TOLERANCE,
        "pass": bool(abs(resampled_mean - observed_mean) < GATE_E1_TOLERANCE),
        "n_drives_resampled": int(len(active)),
    }
    return deltas, expected_delta, {"gate_e1": gate_e1, "game_ids": game_ids}


def rematch_validation(games: pl.DataFrame, frame: pl.DataFrame, label: str) -> dict:
    """Gate E-2, on the identical harness documents 06, 07 and 08 §11 used."""
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
        frame.select("game_id", "dq_margin", "dqw_home"), on="game_id", how="inner"
    )
    print(f"  {joined.height} rematch pairs with a drive-quality margin")

    actual = joined["margin_g1_a"].to_numpy().astype(float)
    y = (joined["margin_g2_a"].to_numpy() > 0).astype(float)
    a_home = joined["a_home_g2"].to_numpy().astype(float)
    rng = np.random.default_rng(RANDOM_SEED)
    folds = rng.permutation(joined.height) % _rematch.N_FOLDS

    def arm(predictor: np.ndarray, name: str) -> dict:
        per_pair = _rematch.paired_log_loss_diff(actual, predictor, y, a_home, folds)
        mean, se, _ = _rematch.decision(per_pair)
        ci = (mean - 1.96 * se, mean + 1.96 * se)
        passed = _rematch.passes_noninferiority(mean, se)
        print(
            f"  {name:26s} delta log loss {mean:+.5f} (SE {se:.5f}) "
            f"95% CI [{ci[0]:+.5f}, {ci[1]:+.5f}]  "
            f"{'PASS' if passed else 'FAIL'} vs margin {NONINFERIORITY_MARGIN:+.3f}"
        )
        return {
            "arm": name,
            "mean_delta_log_loss": mean,
            "se": se,
            "ci95": list(ci),
            "noninferiority_pass": bool(passed),
            "favours_dq": bool(mean < 0),
        }

    primary = arm(joined["dq_margin"].to_numpy().astype(float), "PRIMARY DQ margin")
    secondary = arm(joined["dqw_home"].to_numpy().astype(float), "SECONDARY DQW%")

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
        "label": label,
        "n_pairs": joined.height,
        "primary_dq_margin": primary,
        "secondary_dqw": secondary,
        "coefficient_sanity": coefficient,
        "noninferiority_margin": NONINFERIORITY_MARGIN,
    }


def run_arm(
    drives: pl.DataFrame,
    bins: np.ndarray,
    bin_means: np.ndarray,
    mask: np.ndarray,
    games: pl.DataFrame,
    dtw: pl.DataFrame,
    label: str,
) -> dict:
    """One resampling arm, end to end through SC-4 and gates E-1 to E-3."""
    print(f"\n{'#' * 72}\n### ARM: {label}\n{'#' * 72}")
    deltas, expected_delta, extras = resample(
        drives, bins, bin_means, mask, N_REPLICATES, RANDOM_SEED
    )
    gate_e1 = extras["gate_e1"]
    print(
        f"\nGate E-1 (unbiased): {'PASS' if gate_e1['pass'] else 'FAIL'} — "
        f"observed {gate_e1['observed_mean_points_per_drive']:.4f} pts/drive, "
        f"resampled {gate_e1['resampled_mean_points_per_drive']:.4f}, "
        f"difference {gate_e1['difference']:+.5f} vs tolerance {GATE_E1_TOLERANCE}"
    )

    frame = (
        games.select("game_id", "season", "week", "home_team", "away_team", "margin")
        .join(
            pl.DataFrame({"game_id": extras["game_ids"], "expected_delta": expected_delta}),
            on="game_id",
            how="inner",
        )
        .join(dtw, on="game_id", how="inner")
        .drop_nulls("margin")
    )
    index = {game_id: i for i, game_id in enumerate(extras["game_ids"])}
    columns = np.array([index[g] for g in frame["game_id"].to_list()])
    actual = frame["margin"].to_numpy().astype(float)
    replicated = actual[None, :] + deltas[:, columns]
    frame = frame.with_columns(
        (pl.col("margin") + pl.col("expected_delta")).alias("dq_margin"),
        pl.Series("dqw_home", (replicated > 0).mean(axis=0)),
    )

    adjustment = float((frame["dq_margin"] - frame["margin"]).abs().mean())
    flipped = frame.filter(
        ((pl.col("margin") > 0) & (pl.col("dqw_home") < 0.5))
        | ((pl.col("margin") < 0) & (pl.col("dqw_home") > 0.5))
    ).height
    flip_share = flipped / frame.height
    sc4 = {
        "mean_abs_margin_adjustment": adjustment,
        "flip_share": flip_share,
        "min_adjustment": SC4_MIN_ADJUSTMENT,
        "min_flip_share": SC4_MIN_FLIP_SHARE,
        "pass": bool(adjustment >= SC4_MIN_ADJUSTMENT and flip_share >= SC4_MIN_FLIP_SHARE),
    }
    print(
        f"\nSC-4 (not vacuous): {'PASS' if sc4['pass'] else 'FAIL'} — "
        f"mean |DQ margin - actual| {adjustment:.2f} pts (floor {SC4_MIN_ADJUSTMENT}), "
        f"winner differs in {flip_share:.1%} of {frame.height:,} games "
        f"(floor {SC4_MIN_FLIP_SHARE:.0%})"
    )

    correlation = float(np.corrcoef(frame["dqw_home"], frame["dtw_home"])[0, 1])
    disagree = frame.filter(
        ((pl.col("dqw_home") > 0.5) & (pl.col("dtw_home") < 0.5))
        | ((pl.col("dqw_home") < 0.5) & (pl.col("dtw_home") > 0.5))
    ).height
    print(
        f"\nGate E-3 (distinct, descriptive): corr(DQW%, DTW%) = {correlation:.3f}; "
        f"different winners in {disagree:,} games ({disagree / frame.height:.1%})"
    )

    print("\nGate E-2 — non-inferiority on the rematch harness")
    gate_e2 = rematch_validation(games, frame, label)
    print(
        "  Document 12 §5: this gate has ZERO power against 10% skill erasure and\n"
        "  reaches 80% only between 20% and 29%. A pass means 'not catastrophically\n"
        "  broken', never 'validated'."
    )

    return {
        "frame": frame,
        "report": {
            "label": label,
            "gate_e1_unbiased": gate_e1,
            "sc4_not_vacuous": sc4,
            "gate_e2_rematch": gate_e2,
            "gate_e3_distinctness": {
                "corr_dqw_dtw": correlation,
                "games_naming_different_winners": disagree,
                "n_games": frame.height,
            },
        },
    }


def main() -> None:
    paths.ensure_data_dirs()
    all_drives = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / "drive_anatomy.parquet")
    drives = _power.universe(all_drives)
    points = drives["points"].to_numpy()
    print(
        f"{drives.height:,} drives resampled, "
        f"{all_drives.height - drives.height:,} field-goal drives held at observed points"
    )

    rng = np.random.default_rng(RANDOM_SEED)
    folds = rng.permutation(len(points)) % N_FOLDS
    features = np.column_stack([drives[column].to_numpy() for column in FEATURES])
    predicted = _anatomy.oof_predictions(features, points, folds)
    residual = points - predicted

    bins = equal_count_bins(predicted, N_PREDICTION_BINS)
    bin_means = np.array([points[bins == value].mean() for value in range(N_PREDICTION_BINS)])
    print("\n=== Conditional table: equal-count bins of the predicted value ===")
    for value in range(N_PREDICTION_BINS):
        selected = bins == value
        print(
            f"  bin {value:2d}  n {selected.sum():5d}  "
            f"predicted {predicted[selected].mean():5.2f}  "
            f"observed {points[selected].mean():5.2f}  "
            f"touchdown rate {(points[selected] > 0).mean():5.1%}"
        )

    # ---------------------------------------------------------------- SC-1
    print(f"\n{'=' * 72}\nSC-1 — does the finishing residual persist? (binding)\n{'=' * 72}")
    grouped, measures = _anatomy.team_game_matrix(drives, {"td_residual": residual})
    matrix, starts, sizes = _anatomy.to_dense(grouped, measures)
    split_rng = np.random.default_rng(RANDOM_SEED)
    mask_splits = _seq.split_masks(starts, sizes, len(matrix), split_rng, N_SPLITS)
    per_split = _anatomy.split_half_distribution(matrix, mask_splits, starts)[:, 1]

    power_at_reference = next(
        row["power"]
        for row in json.load((paths.RESEARCH_OUTPUT_DIR / "20_dq_successor_power.json").open())[
            "sc1_power"
        ]
        if row["target_true_r"] == REFERENCE_R
    )
    observed_r = float(per_split.mean())
    sc1 = {
        "split_half_r": observed_r,
        "r_p05": float(np.percentile(per_split, 5)),
        "r_p95": float(np.percentile(per_split, 95)),
        "threshold": SC1_THRESHOLD,
        "power_at_reference_r": power_at_reference,
        "interpretable": bool(power_at_reference >= MIN_POWER),
        "pass": bool(observed_r <= SC1_THRESHOLD),
    }
    print(
        f"  residual split-half r = {observed_r:+.4f} "
        f"[{sc1['r_p05']:+.4f}, {sc1['r_p95']:+.4f}] vs threshold {SC1_THRESHOLD:.4f}: "
        f"{'PASS' if sc1['pass'] else 'FAIL'}"
    )
    print(
        f"  power at r={REFERENCE_R} is {power_at_reference:.2f} "
        f"({'interpretable' if sc1['interpretable'] else 'UNRESOLVABLE'})"
    )

    # ---------------------------------------------------------------- SC-2
    print(f"\n{'=' * 72}\nSC-2 — does between-team spread survive?\n{'=' * 72}")
    sd_observed, sd_adjusted, retention = per_game_retention(drives, predicted - points)
    reference = incumbent_reference(all_drives)
    sc2 = {
        "between_team_sd_observed": sd_observed,
        "between_team_sd_adjusted": sd_adjusted,
        "share_of_spread_retained": retention,
        "threshold": SC2_THRESHOLD,
        "pass": bool(retention >= SC2_THRESHOLD),
        "incumbent_reference": reference,
    }
    print(
        f"  between-team SD of offensive points per game: {sd_observed:.4f} observed -> "
        f"{sd_adjusted:.4f} adjusted\n  retention {retention:.1%} vs threshold "
        f"{SC2_THRESHOLD:.0%}: {'PASS' if sc2['pass'] else 'FAIL'}"
    )
    print(
        f"  document 08's depth-bin instrument, in these same units: "
        f"{reference['share_of_spread_retained']:.1%}"
    )

    # ---------------------------------------------------------------- SC-3
    print(f"\n{'=' * 72}\nSC-3 — quality correlation, net of its mechanical floor\n{'=' * 72}")
    team_season = (
        drives.with_columns(pl.Series("residual", residual))
        .group_by("team_season")
        .agg(
            pl.col("points").mean().alias("points_per_drive"),
            pl.col("residual").mean().alias("residual_per_drive"),
        )
        .sort("team_season")
    )
    quality = team_season["points_per_drive"].to_numpy()
    residual_rate = team_season["residual_per_drive"].to_numpy()
    raw_correlation = float(np.corrcoef(quality, residual_rate)[0, 1])
    per_drive_retention = float(np.std(quality - residual_rate, ddof=1) / np.std(quality, ddof=1))
    floor = float(np.sqrt(max(1.0 - per_drive_retention**2, 0.0)))
    excess = abs(raw_correlation) - floor
    sc3 = {
        "corr_quality_vs_residual": raw_correlation,
        "per_drive_retention": per_drive_retention,
        "mechanical_floor": floor,
        "excess": excess,
        "threshold": SC3_THRESHOLD,
        "pass": bool(excess <= SC3_THRESHOLD),
    }
    print(
        f"  corr(quality, residual) = {raw_correlation:+.3f}, mechanical floor "
        f"sqrt(1 - retention^2) = {floor:.3f}\n  excess {excess:+.4f} vs threshold "
        f"{SC3_THRESHOLD:.4f}: {'PASS' if sc3['pass'] else 'FAIL'}"
    )
    print("  (document 08's instrument sat at -0.784 against a 0.709 floor: excess +0.075)")

    sufficiency_pass = sc1["pass"] and sc2["pass"] and sc3["pass"]
    print(f"\n{'=' * 72}")
    print(
        f"SUFFICIENCY SO FAR: SC-1 {'PASS' if sc1['pass'] else 'FAIL'}, "
        f"SC-2 {'PASS' if sc2['pass'] else 'FAIL'}, "
        f"SC-3 {'PASS' if sc3['pass'] else 'FAIL'}"
    )
    if not sufficiency_pass:
        print(
            "A sufficiency criterion failed. Per document 12 §8's decision rule the measure\n"
            "does not ship and does not reach Gate E-2. The arms below are still run and\n"
            "reported, because a failure a reader cannot see the size of is not a report —\n"
            "but nothing in them can license shipping."
        )
    print("=" * 72)

    # -------------------------------------------------------- arms and gates
    games = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / "game_components.parquet")
    dtw = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / "dtw_games_v11.parquet").select(
        "game_id", "deserved_margin", "dtw_home"
    )

    all_mask = np.ones(drives.height, dtype=bool)
    red_zone_mask = drives["reached_red_zone"].to_numpy()
    arms = {
        "all drives": run_arm(drives, bins, bin_means, all_mask, games, dtw, "all drives"),
        "red-zone trips only": run_arm(
            drives, bins, bin_means, red_zone_mask, games, dtw, "red-zone trips only"
        ),
    }

    verdict = (
        "SHIPS as a second reported measure"
        if sufficiency_pass
        and arms["all drives"]["report"]["sc4_not_vacuous"]["pass"]
        and arms["all drives"]["report"]["gate_e1_unbiased"]["pass"]
        and arms["all drives"]["report"]["gate_e2_rematch"]["primary_dq_margin"][
            "noninferiority_pass"
        ]
        else "DOES NOT SHIP"
    )
    print(f"\n{'=' * 72}\nVERDICT: {verdict}\n{'=' * 72}")

    results = {
        "n_drives_resampled": int(drives.height),
        "n_drives_held": int(all_drives.height - drives.height),
        "features": list(FEATURES),
        "n_prediction_bins": N_PREDICTION_BINS,
        "n_replicates": N_REPLICATES,
        "conditional_table": [
            {
                "bin": value,
                "n": int((bins == value).sum()),
                "mean_predicted": float(predicted[bins == value].mean()),
                "mean_observed": float(points[bins == value].mean()),
                "scoring_rate": float((points[bins == value] > 0).mean()),
            }
            for value in range(N_PREDICTION_BINS)
        ],
        "sc1_residual_persistence": sc1,
        "sc2_spread_retention": sc2,
        "sc3_excess_correlation": sc3,
        "sufficiency_pass": bool(sufficiency_pass),
        "arms": {label: arm["report"] for label, arm in arms.items()},
        "verdict": verdict,
        "random_seed": RANDOM_SEED,
    }
    out = paths.RESEARCH_OUTPUT_DIR / "20_dq_successor.json"
    with out.open("w") as handle:
        json.dump(results, handle, indent=2, default=str)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
