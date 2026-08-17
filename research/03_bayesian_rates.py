"""Step 5 — hierarchical beta-binomial rate models.

Fits the four models pre-registered in `docs/research/03-model-foundations.md`
and checks each gate. Constants here are the ones listed in that document's
appendix; changing one means editing the doc too.

    uv run python research/03_bayesian_rates.py
"""

from __future__ import annotations

import json

import arviz as az
import numpy as np
import polars as pl
import pymc as pm

from nfl_simulator import paths
from nfl_simulator.ingest import ANALYSIS_COLUMNS, FTN_SEASONS, PBP_SEASONS, load_ftn, load_pbp
from nfl_simulator.rates import (
    JUDGMENT_PENALTIES,
    PRE_SNAP_PENALTIES,
    fumble_recovery_counts,
    interception_conversion_counts,
    penalty_counts,
)

RANDOM_SEED = 20260817
CHAINS = 4
TUNE = 1000
DRAWS = 1000
TARGET_ACCEPT = 0.9

# Gate 2, pre-registered: the 89% interval for the population SD of true
# fumble-recovery rates must have an upper bound below this, in rate units.
GATE_2_THRESHOLD = 0.04


def build_model_centered(counts: pl.DataFrame, name: str) -> pm.Model:
    """Attempt 1, exactly as pre-registered: centered beta-binomial hierarchy.

    `log_kappa` rather than `kappa` because the likelihood is nearly flat in
    kappa once kappa is large — every value above a few hundred says "teams are
    identical" and looks the same to the data.

    This is retained after it failed Gate 1 so the failure stays reproducible.
    The log transform was not enough: sampling `p_team` alongside `kappa` leaves
    a funnel, because large kappa forces every p_team into a narrow band while
    small kappa lets them spread. NUTS cannot use one step size for both regimes.
    """
    coords = {"team_season": counts["team_season"].to_list()}
    n = counts["n"].to_numpy()
    k = counts["k"].to_numpy()

    with pm.Model(coords=coords, name="") as model:
        mu = pm.Beta("mu", alpha=2.0, beta=2.0)
        log_kappa = pm.Normal("log_kappa", mu=4.0, sigma=2.0)
        kappa = pm.Deterministic("kappa", pm.math.exp(log_kappa))

        p_team = pm.Beta("p_team", alpha=mu * kappa, beta=(1.0 - mu) * kappa, dims="team_season")

        # The reported answer: SD of the Beta over true team rates, in rate units.
        pm.Deterministic("population_sd", pm.math.sqrt(mu * (1.0 - mu) / (kappa + 1.0)))

        pm.Binomial("successes", n=n, p=p_team, observed=k, dims="team_season")

    model.name = name
    return model


def build_model_marginalized(counts: pl.DataFrame, name: str) -> pm.Model:
    """Attempt 2: the same generative story with `p_team` integrated out.

    ``BetaBinomial(n, mu*kappa, (1-mu)*kappa)`` is exactly
    ``Binomial(n, p)`` with ``p ~ Beta(mu*kappa, (1-mu)*kappa)`` marginalized.
    Identical model, but the 320 funnel-inducing `p_team` parameters no longer
    exist, leaving a clean two-parameter posterior for NUTS to explore.

    Per-team rates are recovered afterwards by conjugacy rather than sampled —
    see :func:`per_team_posterior`. Nothing is lost.
    """
    n = counts["n"].to_numpy()
    k = counts["k"].to_numpy()

    with pm.Model(coords={"team_season": counts["team_season"].to_list()}, name="") as model:
        mu = pm.Beta("mu", alpha=2.0, beta=2.0)
        log_kappa = pm.Normal("log_kappa", mu=4.0, sigma=2.0)
        kappa = pm.Deterministic("kappa", pm.math.exp(log_kappa))

        pm.Deterministic("population_sd", pm.math.sqrt(mu * (1.0 - mu) / (kappa + 1.0)))

        pm.BetaBinomial(
            "successes",
            n=n,
            alpha=mu * kappa,
            beta=(1.0 - mu) * kappa,
            observed=k,
            dims="team_season",
        )

    model.name = name
    return model


def per_team_posterior(counts: pl.DataFrame, idata) -> np.ndarray:
    """Posterior mean true rate per team-season, by conjugacy.

    Beta is conjugate to Binomial, so given (mu, kappa) the exact posterior for a
    team is ``Beta(mu*kappa + k, (1-mu)*kappa + n - k)`` with mean
    ``(mu*kappa + k) / (kappa + n)``. Averaging that over the (mu, kappa) draws
    gives the same answer sampling `p_team` would have, without the funnel.
    """
    mu = idata["posterior"]["mu"].values.ravel()[:, None]
    kappa = idata["posterior"]["kappa"].values.ravel()[:, None]
    n = counts["n"].to_numpy()[None, :]
    k = counts["k"].to_numpy()[None, :]
    return ((mu * kappa + k) / (kappa + n)).mean(axis=0)


def fit(counts: pl.DataFrame, name: str, builder=build_model_marginalized) -> tuple[object, dict]:
    """Prior predictive, sample, save, then run every gate check."""
    print(
        f"\n{'=' * 72}\n{name}: {counts.height} team-seasons, "
        f"n per team-season median {int(counts['n'].median())}, total k={counts['k'].sum():,}\n{'=' * 72}"
    )

    model = builder(counts, name)

    with model:
        prior = pm.sample_prior_predictive(draws=500, random_seed=RANDOM_SEED)
    prior_counts = prior["prior_predictive"]["successes"].values
    observed_rate = counts["k"].sum() / counts["n"].sum()
    print(
        f"prior predictive successes: [{prior_counts.min()}, {prior_counts.max()}]  "
        f"(observed total rate {observed_rate:.3f})"
    )

    with model:
        idata = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
            progressbar=False,
        )
    trace_path = paths.RESEARCH_OUTPUT_DIR / f"trace_{name}.nc"
    idata.to_netcdf(trace_path)  # save before any post-processing

    # ---- Gate 1: sampler health -----------------------------------------
    divergences = int(idata["sample_stats"]["diverging"].sum().item())
    summary = az.summary(idata, var_names=["mu", "kappa", "population_sd"])
    worst_rhat = float(az.summary(idata)["r_hat"].max())
    min_ess_bulk = float(az.summary(idata)["ess_bulk"].min())
    min_ess_tail = float(az.summary(idata)["ess_tail"].min())

    gate1 = divergences == 0 and worst_rhat < 1.01 and min_ess_bulk > 400 and min_ess_tail > 400
    print(
        f"\nGate 1 (sampler health): {'PASS' if gate1 else 'FAIL'} — "
        f"divergences={divergences}, max r_hat={worst_rhat:.4f}, "
        f"min ess_bulk={min_ess_bulk:.0f}, min ess_tail={min_ess_tail:.0f}"
    )
    print(summary[["mean", "sd", "eti89_lb", "eti89_ub", "ess_bulk", "r_hat"]])

    # ---- Gate 4: posterior predictive -----------------------------------
    with model:
        idata.update(pm.sample_posterior_predictive(idata, random_seed=RANDOM_SEED))
    idata.to_netcdf(trace_path)

    replicated = idata["posterior_predictive"]["successes"].values.reshape(-1, counts.height)
    # The variance check is the one that matters. A model that gets the mean
    # right and the between-team spread wrong is exactly a model that would
    # mislead about skill.
    observed_variance = float(np.var(counts["k"].to_numpy() / counts["n"].to_numpy()))
    replicated_variance = np.var(replicated / counts["n"].to_numpy(), axis=1)
    variance_pvalue = float((replicated_variance >= observed_variance).mean())
    gate4 = 0.055 < variance_pvalue < 0.945
    print(
        f"Gate 4 (posterior predictive): {'PASS' if gate4 else 'FAIL'} — "
        f"observed between-team variance of rates {observed_variance:.5f}, "
        f"posterior predictive 89% interval "
        f"[{np.percentile(replicated_variance, 5.5):.5f}, "
        f"{np.percentile(replicated_variance, 94.5):.5f}], "
        f"tail probability {variance_pvalue:.3f}"
    )

    posterior = idata["posterior"]
    population_sd = posterior["population_sd"].values.ravel()
    result = {
        "name": name,
        "n_team_seasons": counts.height,
        "total_n": int(counts["n"].sum()),
        "total_k": int(counts["k"].sum()),
        "observed_rate": observed_rate,
        "mu_mean": float(posterior["mu"].values.mean()),
        "mu_eti89": [
            float(np.percentile(posterior["mu"].values, 5.5)),
            float(np.percentile(posterior["mu"].values, 94.5)),
        ],
        "kappa_mean": float(posterior["kappa"].values.mean()),
        "population_sd_mean": float(population_sd.mean()),
        "population_sd_eti89": [
            float(np.percentile(population_sd, 5.5)),
            float(np.percentile(population_sd, 94.5)),
        ],
        "divergences": divergences,
        "max_r_hat": worst_rhat,
        "min_ess_bulk": min_ess_bulk,
        "gate1_sampler_health": gate1,
        "gate4_posterior_predictive": gate4,
        "ppc_variance_pvalue": variance_pvalue,
        "trace_path": str(trace_path.relative_to(paths.REPO_ROOT)),
    }
    print(
        f"\npopulation SD of true rates: {result['population_sd_mean']:.4f} "
        f"[{result['population_sd_eti89'][0]:.4f}, {result['population_sd_eti89'][1]:.4f}] "
        f"= {result['population_sd_mean'] * 100:.2f} percentage points"
    )
    return idata, result


def shrinkage_table(counts: pl.DataFrame, idata, top: int = 8) -> pl.DataFrame:
    """The most extreme observed rates next to their shrunk posteriors.

    This is the deliverable Phase 2 consumes: a team that recovered 14 of 20
    fumbles is not a 70% recovery team, and this table says what it is instead.
    """
    posterior_mean = per_team_posterior(counts, idata)
    table = counts.with_columns(
        (pl.col("k") / pl.col("n")).alias("observed_rate"),
        pl.Series("posterior_mean", posterior_mean),
    ).with_columns((pl.col("posterior_mean") - pl.col("observed_rate")).alias("shrinkage"))
    return pl.concat(
        [
            table.sort("observed_rate", descending=True).head(top),
            table.sort("observed_rate").head(top),
        ]
    )


def main() -> None:
    paths.ensure_data_dirs()
    pbp = load_pbp(PBP_SEASONS, columns=ANALYSIS_COLUMNS)
    ftn = load_ftn(FTN_SEASONS)

    datasets = {
        "fumble_recovery": fumble_recovery_counts(pbp, exclude_aborted=True),
        "fumble_recovery_with_aborted": fumble_recovery_counts(pbp, exclude_aborted=False),
        "interception_conversion": interception_conversion_counts(pbp, ftn),
        "penalty_pre_snap": penalty_counts(pbp, PRE_SNAP_PENALTIES),
        "penalty_judgment": penalty_counts(pbp, JUDGMENT_PENALTIES),
    }

    # Attempt 1 is the pre-registered centered parameterization. It is run on the
    # calibration model only, and kept in the record because it FAILED Gate 1 —
    # deleting it would hide the failure the fallback exists to answer.
    print("\n### ATTEMPT 1 — pre-registered centered parameterization ###")
    _, attempt1 = fit(
        datasets["fumble_recovery"], "fumble_recovery_centered", builder=build_model_centered
    )
    results = {"attempt1_centered_fumble_recovery": attempt1}

    print("\n### ATTEMPT 2 — marginalized BetaBinomial (Gate 1 fallback) ###")
    traces = {}
    for name, counts in datasets.items():
        idata, result = fit(counts, name, builder=build_model_marginalized)
        results[name] = result
        traces[name] = idata

    # ---- Gate 2: the calibration case ------------------------------------
    calibration = results["fumble_recovery"]
    upper = calibration["population_sd_eti89"][1]
    gate2 = upper < GATE_2_THRESHOLD
    print(f"\n{'=' * 72}")
    print(
        f"GATE 2 (calibration case): {'PASS' if gate2 else 'FAIL'} — "
        f"89% upper bound on population SD of fumble-recovery rates is "
        f"{upper:.4f} ({upper * 100:.2f} pp), pre-registered threshold "
        f"{GATE_2_THRESHOLD:.2f} ({GATE_2_THRESHOLD * 100:.0f} pp)"
    )
    print(f"{'=' * 72}")
    results["gate2_calibration"] = {
        "pass": gate2,
        "upper_bound": upper,
        "threshold": GATE_2_THRESHOLD,
    }

    print("\n=== Population SD of true team rates, all models ===")
    comparison = pl.DataFrame(
        [
            {
                "model": name,
                "league_rate": result["observed_rate"],
                "population_sd_pp": result["population_sd_mean"] * 100,
                "eti89_low_pp": result["population_sd_eti89"][0] * 100,
                "eti89_high_pp": result["population_sd_eti89"][1] * 100,
                "divergences": result["divergences"],
            }
            for name, result in results.items()
            if isinstance(result, dict) and "population_sd_mean" in result
        ]
    )
    with pl.Config(tbl_cols=-1, fmt_str_lengths=32):
        print(comparison)

    # Full shrinkage table for the figures; the printed one is just the extremes.
    full_shrinkage = (
        datasets["fumble_recovery"]
        .with_columns(
            (pl.col("k") / pl.col("n")).alias("observed_rate"),
            pl.Series(
                "posterior_mean",
                per_team_posterior(datasets["fumble_recovery"], traces["fumble_recovery"]),
            ),
        )
        .with_columns((pl.col("posterior_mean") - pl.col("observed_rate")).alias("shrinkage"))
    )
    full_shrinkage.write_parquet(paths.RESEARCH_OUTPUT_DIR / "fumble_shrinkage.parquet")

    print("\n=== Shrinkage, fumble recovery (most extreme team-seasons) ===")
    with pl.Config(tbl_rows=20):
        print(shrinkage_table(datasets["fumble_recovery"], traces["fumble_recovery"]))

    out = paths.RESEARCH_OUTPUT_DIR / "03_bayesian_rates.json"
    with out.open("w") as handle:
        json.dump(results, handle, indent=2, default=str)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
