"""Phase 4, step 2 — power and instrument characterization for the DQW% successor.

Document 11 §10 refuted the successor document 08 §11 sketched. A richer drive
summary was **necessary** — it lifted between-team spread retention from 70.5% to
94.8% and cut residual persistence from r = +0.324 to +0.108 — and it was **not
sufficient**, because the residual still persisted. The exploratory diagnosis
named the channel: valuing the same drives three ways, *reaching the end zone*
does not persist (r = +0.042 against a 0.056 null, 89% power) while *turning a
drive into three points* does (+0.229). The persistence is kicking, which DTW%
already neutralizes.

So the successor resamples **touchdowns only**, on the drives that did not
attempt a field goal, conditioned on the rich summary. This script measures the
design parameters and characterizes the two instruments a pre-registration needs,
**before** `docs/research/12-dq-successor.md` commits any threshold:

1. **The persistence instrument** — the permutation null and power curve for the
   residual of the touchdown valuation on its own universe. This is sufficiency
   criterion SC-1, the binding one.
2. **The skill-erasure instrument** — how much erasure of true team strength the
   rematch harness's non-inferiority gate can actually catch. Document 06 §4
   measured the gate against *estimation noise*; nothing has ever measured it
   against the failure mode that killed DQW%, which was strength being erased.
   That number is what says how much work the sufficiency criteria have to do.

    uv run python research/20_dq_successor_power.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_anatomy = import_module("19_drive_anatomy_power")
_seq = import_module("10_sequencing_power")
_rematch = import_module("08_rematch_power")

from nfl_simulator import paths  # noqa: E402

RANDOM_SEED = 20260817

N_SPLITS = _anatomy.N_SPLITS
N_NULL_REPLICATES = 500
N_POWER_REPLICATES = 500
N_FOLDS = _anatomy.N_FOLDS
REFERENCE_R = _anatomy.REFERENCE_R

N_ERASURE_SIMULATIONS = 2000

# The successor's resampled universe: drives whose outcome was decided without a
# field-goal attempt. Field-goal and missed-field-goal drives are HELD AT THEIR
# OBSERVED POINTS, because document 11 §10 found the kicking channel is where the
# finishing residual's persistence lives — and because DTW% already prices field
# goals against the kicker hierarchy, with weather. Resampling them here would
# both erase a real skill and double-count a component the project has solved.
RESAMPLED_RESULTS: tuple[str, ...] = ("Touchdown", "Punt", "Turnover on downs")

# The rich summary document 11 §10 measured, unchanged. F3, not F4: `net_yards`
# is entangled with the outcome (78.1% of touchdown drives have net yards within
# 5 of the starting distance to the goal) and its residual persisted *more*.
FEATURES: tuple[str, ...] = _anatomy.FEATURE_SETS["F3_production"]

# Equal-count bins of the out-of-fold predicted value. A cell table cannot carry
# seven features, so the conditioning is on the predicted value itself and the
# resampling draws from the observed points of league drives in the same bin.
# 20 bins puts roughly 1,900 drives in each — the same order as the depth-bin
# populations document 08's instrument used.
N_PREDICTION_BINS = 20


def universe(drives: pl.DataFrame) -> pl.DataFrame:
    """Drives whose outcome the successor is allowed to redraw."""
    return drives.filter(pl.col("result").is_in(RESAMPLED_RESULTS))


def touchdown_points(drives: pl.DataFrame) -> np.ndarray:
    """A drive's points, on a universe where the only way to score is a touchdown.

    Taking the drive's realized points rather than a flat 6 keeps the extra-point
    and two-point outcomes on their real support, which is what makes the
    non-parametric resampling below reproduce the league's own distribution
    rather than a smoothed version of it.
    """
    return drives["points"].to_numpy()


def skill_erasure_catch_rate(params: dict, retention: float, simulations: int, seed: int) -> dict:
    """How often the non-inferiority gate catches a predictor that erased strength.

    **This instrument did not exist before.** Document 06 §4's harm curve is
    indexed by *estimation noise* — the neutralization removing a proportional
    slice and adding error on top. The failure that killed DQW% was different in
    kind: the resampling did not add noise so much as **shrink the true
    between-team strength signal**, which is what
    ``corr(quality, adjustment) = -0.784`` and 29.4% of spread destroyed describe.

    So the generator here erases strength and changes nothing else::

        actual = delta + luck + residual
        dq     = retention * delta + luck + residual

    Everything but the strength term is byte-identical between the two arms, so
    the catch rate is attributable to erasure alone. ``retention`` maps directly
    onto sufficiency criterion SC-2, which is stated in exactly these units.
    """
    rng = np.random.default_rng(seed)
    var_total = params["sd_margin"] ** 2
    var_delta = params["reliability"] * var_total
    var_luck = _rematch.LUCK_SHARE_FUMBLE_FG * var_total
    var_residual = max(var_total - var_delta - var_luck, 1e-6)
    n = params["n_pairs"]

    caught, means = [], []
    for _ in range(simulations):
        delta = rng.normal(0.0, np.sqrt(var_delta), n)
        luck = rng.normal(0.0, np.sqrt(var_luck), n)
        residual = rng.normal(0.0, np.sqrt(var_residual), n)
        actual = delta + luck + residual
        dq = retention * delta + luck + residual

        a_home_g2 = (rng.random(n) < params["p_a_home_g2"]).astype(float)
        hfa = params["hfa_points"] * (2.0 * a_home_g2 - 1.0)
        margin_g2 = delta + hfa + rng.normal(0.0, np.sqrt(var_total - var_delta), n)
        y = (margin_g2 > 0).astype(float)
        folds = rng.permutation(n) % _rematch.N_FOLDS

        mean, se, _ = _rematch.decision(
            _rematch.paired_log_loss_diff(actual, dq, y, a_home_g2, folds)
        )
        means.append(mean)
        caught.append(not _rematch.passes_noninferiority(mean, se))

    return {
        "retention": retention,
        "spread_erased": 1.0 - retention,
        "mean_delta_log_loss": float(np.mean(means)),
        "power_to_catch": float(np.mean(caught)),
    }


def main() -> None:
    paths.ensure_data_dirs()
    drives = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / "drive_anatomy.parquet")
    resampled = universe(drives)
    points = touchdown_points(resampled)

    print(
        f"{resampled.height:,} drives in the successor's universe "
        f"({resampled.height / drives.height:.1%} of document 08's), "
        f"{drives.height - resampled.height:,} field-goal drives held at their observed points"
    )
    print(f"  league points per drive on this universe: {points.mean():.4f}")
    print(f"  support: {sorted(set(points.tolist()))}")

    rng = np.random.default_rng(RANDOM_SEED)
    folds = rng.permutation(len(points)) % N_FOLDS
    features = np.column_stack([resampled[column].to_numpy() for column in FEATURES])
    predicted = _anatomy.oof_predictions(features, points, folds)
    residual = points - predicted

    baseline_variance = float(np.var(points, ddof=1))
    fit = {
        "oof_r2": 1.0 - float(np.var(residual, ddof=1)) / baseline_variance,
        "residual_sd": float(np.std(residual, ddof=1)),
        "points_per_drive": float(points.mean()),
        "var_points_per_drive": baseline_variance,
    }
    print(f"\n  OOF R2 on the touchdown valuation: {fit['oof_r2']:.4f}")

    # ---- instrument 1: the persistence null and power ---------------------
    grouped, measures = _anatomy.team_game_matrix(resampled, {"td_residual": residual})
    matrix, starts, sizes = _anatomy.to_dense(grouped, measures)
    group_of_row = np.repeat(np.arange(len(sizes)), sizes)
    drives_per_half = float(
        np.mean([matrix[s : s + n, 0].sum() / 2.0 for s, n in zip(starts, sizes, strict=True)])
    )

    design = {
        "n_drives": int(resampled.height),
        "n_team_seasons": int(len(starts)),
        "n_team_games": int(len(matrix)),
        "mean_drives_per_team_game": float(matrix[:, 0].mean()),
        "mean_drives_per_half": drives_per_half,
        **fit,
    }
    print("\n=== Design parameters — no persistence measured here ===")
    for key, value in design.items():
        print(f"  {key:28s} {value:.4f}" if isinstance(value, float) else f"  {key:28s} {value}")

    split_rng = np.random.default_rng(RANDOM_SEED)
    mask = _seq.split_masks(starts, sizes, len(matrix), split_rng, N_SPLITS)

    print(f"\n=== Permutation null for SC-1 ({N_NULL_REPLICATES} replicates) ===")
    null_draws = _anatomy.permutation_null(matrix, mask, starts, N_NULL_REPLICATES, RANDOM_SEED)
    null = {
        "measure": "td_residual",
        "null_mean_r": float(null_draws[:, 1].mean()),
        "null_sd_r": float(null_draws[:, 1].std(ddof=1)),
        "null_p95": float(np.percentile(null_draws[:, 1], 95)),
        "null_p99": float(np.percentile(null_draws[:, 1], 99)),
    }
    print(
        f"  null mean {null['null_mean_r']:+.4f}  SD {null['null_sd_r']:.4f}  "
        f"95th pct {null['null_p95']:.4f}  99th pct {null['null_p99']:.4f}"
    )

    print(f"\n=== Power for SC-1 ({N_POWER_REPLICATES} replicates) ===")
    per_drive_sd = float(np.std(residual, ddof=1))
    power_rows = []
    for target_r in (0.05, 0.08, 0.10, REFERENCE_R, 0.20, 0.30):
        tau = _anatomy.tau_for_target_r(per_drive_sd, drives_per_half, target_r)
        draws = _anatomy.parametric_draws(
            matrix,
            group_of_row,
            mask,
            starts,
            1,
            per_drive_sd,
            tau,
            N_POWER_REPLICATES,
            RANDOM_SEED + 200,
        )
        power = float((draws > null["null_p95"]).mean())
        power_rows.append(
            {
                "target_true_r": target_r,
                "tau": tau,
                "mean_observed_r": float(draws.mean()),
                "power": power,
            }
        )
        print(
            f"  nominal r {target_r:.2f}  tau {tau:.5f}  "
            f"achieved mean r {draws.mean():+.3f}  power {power:.3f}"
        )

    # ---- instrument 2: what the rematch gate can see ----------------------
    games = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / "game_components.parquet")
    params = _rematch.design_parameters(_rematch.rematch_pairs(games))
    print("\n=== Rematch design parameters (document 06 §2, re-measured) ===")
    for key, value in params.items():
        print(f"  {key:24s} {value:.4f}" if isinstance(value, float) else f"  {key:24s} {value}")

    print(
        f"\n=== Skill-erasure instrument — what Gate E-2 can catch "
        f"({N_ERASURE_SIMULATIONS} datasets) ==="
    )
    print("    a predictor that shrinks true team strength and changes nothing else")
    erasure_rows = []
    for offset, retention in enumerate([0.99, 0.97, 0.95, 0.90, 0.80, 0.706, 0.60], start=1):
        row = skill_erasure_catch_rate(
            params, retention, N_ERASURE_SIMULATIONS, RANDOM_SEED + 300 + offset
        )
        erasure_rows.append(row)
        print(
            f"  retention {retention:.3f} ({row['spread_erased']:5.1%} erased)  "
            f"mean delta log loss {row['mean_delta_log_loss']:+.5f}  "
            f"power to catch {row['power_to_catch']:.3f}"
        )
    print(
        "\n    0.706 is the retention document 08's shipped instrument achieved — the row\n"
        "    exists so the curve can be read against a design whose fate is known."
    )

    # ---- SC-3's sampling floor -------------------------------------------
    correlation_floor = float(1.0 / np.sqrt(len(starts)))
    print(
        f"\n  SC-3 sampling scale: 1/sqrt({len(starts)} team-seasons) = {correlation_floor:.4f}\n"
        "  A correlation this size is what pure sampling produces at this sample size."
    )

    results = {
        "design": design,
        "features": list(FEATURES),
        "resampled_results": list(RESAMPLED_RESULTS),
        "n_prediction_bins": N_PREDICTION_BINS,
        "n_splits": N_SPLITS,
        "n_null_replicates": N_NULL_REPLICATES,
        "n_power_replicates": N_POWER_REPLICATES,
        "reference_r": REFERENCE_R,
        "sc1_permutation_null": null,
        "sc1_power": power_rows,
        "rematch_design": params,
        "skill_erasure_catch_rate": erasure_rows,
        "sc3_sampling_scale": correlation_floor,
        "random_seed": RANDOM_SEED,
    }
    out = paths.RESEARCH_OUTPUT_DIR / "20_dq_successor_power.json"
    with out.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
