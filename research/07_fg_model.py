"""Step 4 — the kicker-hierarchical field-goal make model.

Fits the model pre-registered in `docs/research/05b-fg-model-foundations.md`,
committed at `d02395d` before this script produced anything, and checks all four
gates. Constants here are the ones in that document's appendix; changing one
means editing the doc too.

Writes the posterior the simulator consumes:

    research/outputs/trace_fg_model.nc        full posterior
    research/outputs/fg_kicker_effects.parquet  per-kicker-season summary

    uv run python research/07_fg_model.py
"""

from __future__ import annotations

import json

import arviz as az
import numpy as np
import polars as pl
import pymc as pm

from nfl_simulator import paths
from nfl_simulator.ingest import PBP_SEASONS, load_pbp

RANDOM_SEED = 20260817
CHAINS = 4
TUNE = 1000
DRAWS = 1000
TARGET_ACCEPT = 0.9

DISTANCE_CENTRE = 40.0
MIN_BIN_ATTEMPTS = 100
GATE_FG3_THRESHOLD = 0.2407  # pre-registered, from the null simulation

FG_COLUMNS = ["season", "play_type", "kick_distance", "field_goal_result", "kicker_player_id"]


def load_attempts() -> pl.DataFrame:
    """Field-goal attempts with a distance and a kicker. Blocks count as misses."""
    pbp = load_pbp(PBP_SEASONS, columns=FG_COLUMNS)
    return (
        pbp.filter((pl.col("play_type") == "field_goal") & pl.col("kick_distance").is_not_null())
        .drop_nulls("kicker_player_id")
        .select(
            pl.col("season"),
            pl.col("kicker_player_id"),
            pl.col("kick_distance").cast(pl.Float64).alias("distance"),
            (pl.col("field_goal_result") == "made").cast(pl.Int64).alias("made"),
            pl.concat_str(
                [pl.col("season").cast(pl.String), pl.col("kicker_player_id")], separator="_"
            ).alias("kicker_season"),
        )
    )


def build_model(
    attempts: pl.DataFrame,
    kicker_levels: list[str],
    kicker_idx: np.ndarray,
    *,
    quadratic: bool = False,
):
    """Hierarchical logistic: distance slope plus a kicker random intercept.

    ``quadratic=True`` is the fallback named in foundations doc section 5,
    reached only when Gate FG-2 fails on the pre-registered linear form.
    """
    centred = attempts["distance"].to_numpy() - DISTANCE_CENTRE
    made = attempts["made"].to_numpy()

    coords = {"kicker_season": kicker_levels, "attempt": np.arange(len(made))}
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", mu=2.0, sigma=1.5)
        beta = pm.Normal("beta", mu=0.0, sigma=0.2)
        sigma_kicker = pm.HalfNormal("sigma_kicker", sigma=1.0)

        # Non-centered: a ruling, not a default. See foundations doc section 5.
        z = pm.Normal("z", mu=0.0, sigma=1.0, dims="kicker_season")
        kicker = pm.Deterministic("kicker", sigma_kicker * z, dims="kicker_season")

        eta = alpha + beta * centred + kicker[kicker_idx]
        if quadratic:
            # Scaled by 100 so gamma sits on the same order as beta and the
            # prior below means something readable rather than being effectively
            # flat on a coefficient of size 1e-3.
            gamma = pm.Normal("gamma", mu=0.0, sigma=0.2)
            eta = eta + gamma * (centred**2) / 100.0
        pm.Bernoulli("made", logit_p=eta, observed=made, dims="attempt")
    return model


def make_probabilities(idata, attempts: pl.DataFrame, kicker_idx: np.ndarray) -> np.ndarray:
    """Per-attempt make probability on every posterior draw."""
    posterior = idata["posterior"]
    n_kickers = posterior["kicker"].shape[-1]
    alpha = posterior["alpha"].values.ravel()[:, None]
    beta = posterior["beta"].values.ravel()[:, None]
    kicker = posterior["kicker"].values.reshape(-1, n_kickers)
    centred = (attempts["distance"].to_numpy() - DISTANCE_CENTRE)[None, :]

    eta = alpha + beta * centred + kicker[:, kicker_idx]
    if "gamma" in posterior:
        eta = eta + posterior["gamma"].values.ravel()[:, None] * (centred**2) / 100.0
    return 1.0 / (1.0 + np.exp(-eta))


def gate_fg1(idata) -> dict:
    summary = az.summary(idata, var_names=["alpha", "beta", "sigma_kicker"])
    full = az.summary(idata)
    report = {
        "divergences": int(idata["sample_stats"]["diverging"].sum().item()),
        "max_r_hat": float(full["r_hat"].max()),
        "min_ess_bulk": float(full["ess_bulk"].min()),
        "min_ess_tail": float(full["ess_tail"].min()),
    }
    report["pass"] = (
        report["divergences"] == 0
        and report["max_r_hat"] < 1.01
        and report["min_ess_bulk"] > 400
        and report["min_ess_tail"] > 400
    )
    print(
        f"\nGate FG-1 (sampler health): {'PASS' if report['pass'] else 'FAIL'} — "
        f"divergences={report['divergences']}, max r_hat={report['max_r_hat']:.4f}, "
        f"min ess_bulk={report['min_ess_bulk']:.0f}, min ess_tail={report['min_ess_tail']:.0f}"
    )
    print(summary[["mean", "sd", "eti89_lb", "eti89_ub", "ess_bulk", "r_hat"]])
    return report


def calibration_statistic(masks: list[np.ndarray], made: np.ndarray, p_hat: np.ndarray) -> float:
    """Largest standardized bin miss. See foundations doc section 6, Gate FG-2."""
    worst = 0.0
    for mask in masks:
        expected = p_hat[mask]
        sd = np.sqrt((expected * (1.0 - expected)).sum()) / mask.sum()
        if sd > 0:
            worst = max(worst, abs(made[mask].mean() - expected.mean()) / sd)
    return worst


def gate_fg2(attempts: pl.DataFrame, p_draws: np.ndarray) -> dict:
    """Distance calibration: is the worst bin miss worse than the model's own?"""
    distance = attempts["distance"].to_numpy()
    made = attempts["made"].to_numpy().astype(float)
    bins = (distance // 5 * 5).astype(int)
    masks = [bins == edge for edge in np.unique(bins) if (bins == edge).sum() >= MIN_BIN_ATTEMPTS]

    p_hat = p_draws.mean(axis=0)
    observed = calibration_statistic(masks, made, p_hat)

    rng = np.random.default_rng(RANDOM_SEED)
    # Replicate from the posterior itself, so the reference distribution carries
    # parameter uncertainty rather than just Bernoulli noise at a point estimate.
    picks = rng.choice(len(p_draws), size=400, replace=False)
    replicated = np.array(
        [
            calibration_statistic(masks, rng.binomial(1, p_draws[i]).astype(float), p_draws[i])
            for i in picks
        ]
    )
    threshold = float(np.percentile(replicated, 94.5))
    report = {
        "observed_statistic": float(observed),
        "reference_94_5_pct": threshold,
        "pass": bool(observed <= threshold),
        "n_bins": len(masks),
    }
    print(
        f"\nGate FG-2 (distance calibration): {'PASS' if report['pass'] else 'FAIL'} — "
        f"worst standardized bin miss {observed:.3f}, reference 94.5th pct {threshold:.3f}, "
        f"{len(masks)} bins"
    )

    # The per-bin table is reported whatever the gate says; it is how a reader
    # sees *where* the curve is right rather than only whether it is.
    rows = []
    for mask in masks:
        rows.append(
            {
                "bin_start": int(bins[mask][0]),
                "attempts": int(mask.sum()),
                "observed": float(made[mask].mean()),
                "predicted": float(p_hat[mask].mean()),
            }
        )
    table = pl.DataFrame(rows).with_columns(
        ((pl.col("observed") - pl.col("predicted")) * 100).alias("miss_pp")
    )
    with pl.Config(tbl_rows=20):
        print(table)
    report["bins"] = rows
    return report


def gate_fg3(idata, attempts: pl.DataFrame) -> dict:
    """Is kicker skill resolvable above what a skill-free league would produce?"""
    sigma = idata["posterior"]["sigma_kicker"].values.ravel()
    bounds = [float(np.percentile(sigma, 5.5)), float(np.percentile(sigma, 94.5))]

    posterior = idata["posterior"]
    alpha = posterior["alpha"].values.ravel()
    beta = posterior["beta"].values.ravel()
    logit_45 = alpha + beta * (45.0 - DISTANCE_CENTRE)
    p_45 = 1.0 / (1.0 + np.exp(-logit_45))
    gap = 1.0 / (1.0 + np.exp(-(logit_45 + sigma))) - p_45

    report = {
        "sigma_kicker_mean": float(sigma.mean()),
        "sigma_kicker_eti89": bounds,
        "threshold": GATE_FG3_THRESHOLD,
        "pass": bool(bounds[1] > GATE_FG3_THRESHOLD),
        "make_rate_at_45yd": float(p_45.mean()),
        "gap_at_45yd_pp": float(gap.mean()) * 100,
        "gap_at_45yd_eti89_pp": [
            float(np.percentile(gap, 5.5)) * 100,
            float(np.percentile(gap, 94.5)) * 100,
        ],
    }
    print(
        f"\nGate FG-3 (kicker skill resolvable): {'PASS' if report['pass'] else 'FAIL'} — "
        f"sigma_kicker {report['sigma_kicker_mean']:.4f} "
        f"[{bounds[0]:.4f}, {bounds[1]:.4f}], threshold {GATE_FG3_THRESHOLD}"
    )
    print(
        f"  league make rate at 45 yd {report['make_rate_at_45yd']:.3f}; "
        f"one-SD kicker gap {report['gap_at_45yd_pp']:.2f} pp "
        f"[{report['gap_at_45yd_eti89_pp'][0]:.2f}, {report['gap_at_45yd_eti89_pp'][1]:.2f}]"
    )
    return report


def gate_fg4(attempts: pl.DataFrame, replicated: np.ndarray, kicker_idx: np.ndarray) -> dict:
    """Posterior predictive on the league make rate and the between-kicker spread."""
    made = attempts["made"].to_numpy().astype(float)
    n_kickers = kicker_idx.max() + 1
    counts = np.bincount(kicker_idx, minlength=n_kickers)
    keep = counts >= 10

    def between_kicker_variance(y: np.ndarray) -> float:
        totals = np.bincount(kicker_idx, weights=y, minlength=n_kickers)
        return float(np.var((totals / np.maximum(counts, 1))[keep]))

    observed_rate = float(made.mean())
    observed_var = between_kicker_variance(made)
    rates = replicated.mean(axis=1)
    variances = np.array([between_kicker_variance(row) for row in replicated])

    rate_p = float((rates >= observed_rate).mean())
    var_p = float((variances >= observed_var).mean())
    report = {
        "observed_make_rate": observed_rate,
        "make_rate_tail_p": rate_p,
        "observed_between_kicker_variance": observed_var,
        "variance_tail_p": var_p,
        "n_kickers_with_10plus": int(keep.sum()),
        "pass": bool(0.055 < rate_p < 0.945 and 0.055 < var_p < 0.945),
    }
    print(
        f"\nGate FG-4 (posterior predictive): {'PASS' if report['pass'] else 'FAIL'} — "
        f"make rate {observed_rate:.4f} (tail p {rate_p:.3f}), "
        f"between-kicker variance {observed_var:.5f} (tail p {var_p:.3f})"
    )
    return report


def main() -> None:
    paths.ensure_data_dirs()
    attempts = load_attempts()

    kicker_levels = sorted(attempts["kicker_season"].unique().to_list())
    lookup = {level: i for i, level in enumerate(kicker_levels)}
    kicker_idx = np.array([lookup[v] for v in attempts["kicker_season"].to_list()])

    print(
        f"{attempts.height:,} field-goal attempts, "
        f"league make rate {attempts['made'].mean():.3%}, "
        f"{len(kicker_levels)} kicker-seasons"
    )

    def fit_arm(label: str, *, quadratic: bool) -> tuple[object, np.ndarray, dict]:
        print(f"\n{'#' * 72}\n### {label}\n{'#' * 72}")
        model = build_model(attempts, kicker_levels, kicker_idx, quadratic=quadratic)
        with model:
            prior = pm.sample_prior_predictive(draws=500, random_seed=RANDOM_SEED)
            prior_rate = prior["prior_predictive"]["made"].values.mean(axis=-1)
            print(
                f"prior predictive make rate range: "
                f"[{prior_rate.min():.3f}, {prior_rate.max():.3f}]"
            )
            arm_idata = pm.sample(
                draws=DRAWS,
                tune=TUNE,
                chains=CHAINS,
                target_accept=TARGET_ACCEPT,
                random_seed=RANDOM_SEED,
                progressbar=False,
            )
        suffix = "quadratic" if quadratic else "linear"
        arm_idata.to_netcdf(paths.RESEARCH_OUTPUT_DIR / f"trace_fg_model_{suffix}.nc")

        arm_p = make_probabilities(arm_idata, attempts, kicker_idx)
        arm_report = {
            "gate_fg1_sampler_health": gate_fg1(arm_idata),
            "gate_fg2_calibration": gate_fg2(attempts, arm_p),
            "gate_fg3_kicker_skill": gate_fg3(arm_idata, attempts),
        }
        arm_rng = np.random.default_rng(RANDOM_SEED)
        picks = arm_rng.choice(len(arm_p), size=1000, replace=False)
        arm_report["gate_fg4_posterior_predictive"] = gate_fg4(
            attempts, arm_rng.binomial(1, arm_p[picks]).astype(float), kicker_idx
        )
        return arm_idata, arm_p, arm_report

    results = {"n_attempts": attempts.height, "n_kicker_seasons": len(kicker_levels)}

    # Attempt 1 is the pre-registered linear form. It is kept in the record
    # whatever it does — deleting a failed pre-registered arm would hide the
    # failure the fallback exists to answer.
    idata, p_draws, linear_report = fit_arm(
        "ATTEMPT 1 — pre-registered linear distance term", quadratic=False
    )
    results["attempt1_linear"] = linear_report
    adopted = "linear"

    if not linear_report["gate_fg2_calibration"]["pass"]:
        print(
            "\nGate FG-2 failed on the pre-registered form. Applying the fallback named in\n"
            "foundations doc section 5: add a quadratic term in centred distance, refit,\n"
            "report both."
        )
        idata, p_draws, quad_report = fit_arm("ATTEMPT 2 — quadratic fallback", quadratic=True)
        results["attempt2_quadratic"] = quad_report
        adopted = "quadratic"

    results["adopted_arm"] = adopted
    trace_path = paths.RESEARCH_OUTPUT_DIR / "trace_fg_model.nc"
    idata.to_netcdf(trace_path)
    kicker = idata["posterior"]["kicker"].values.reshape(-1, len(kicker_levels))

    # ---- the artifact the simulator consumes ------------------------------
    effects = pl.DataFrame(
        {
            "kicker_season": kicker_levels,
            "attempts": np.bincount(kicker_idx, minlength=len(kicker_levels)),
            "made": np.bincount(
                kicker_idx, weights=attempts["made"].to_numpy(), minlength=len(kicker_levels)
            ),
            "effect_mean": kicker.mean(axis=0),
            "effect_sd": kicker.std(axis=0),
        }
    ).with_columns((pl.col("made") / pl.col("attempts")).alias("observed_rate"))
    effects.write_parquet(paths.RESEARCH_OUTPUT_DIR / "fg_kicker_effects.parquet")

    print("\n=== Most and least effective kicker-seasons (shrunk) ===")
    extremes = pl.concat(
        [
            effects.filter(pl.col("attempts") >= 15).sort("effect_mean", descending=True).head(5),
            effects.filter(pl.col("attempts") >= 15).sort("effect_mean").head(5),
        ]
    )
    with pl.Config(tbl_rows=12, fmt_str_lengths=24):
        print(extremes)

    out = paths.RESEARCH_OUTPUT_DIR / "07_fg_model.json"
    with out.open("w") as handle:
        json.dump(results, handle, indent=2, default=str)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
