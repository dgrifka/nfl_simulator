"""Step 3 — the attribution round.

Fits the three questions pre-registered in `docs/research/05-neutralization-principle.md`
§7, committed at `c1b454f` before any of these models existed:

    3a  whose skill is the 14.3% interception spread — quarterbacks' or defenses'?
    3b  is offensive holding random, even though pooled judgment calls persist?
    3c  does interception return yardage persist?

Every threshold used here was power-checked first by
`research/06_attribution_power.py`. Where a question turned out to be
underpowered, §7 committed to reporting it descriptively rather than gating it,
and this script follows that.

    uv run python research/06_attribution.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
import polars as pl
import pymc as pm

sys.path.insert(0, str(Path(__file__).parent))

from _betabinom_grid import fit_grid  # noqa: E402

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.ingest import (  # noqa: E402
    ANALYSIS_COLUMNS,
    FTN_SEASONS,
    PBP_SEASONS,
    load_ftn,
    load_pbp,
)

RANDOM_SEED = 20260817
CHAINS = 4
TUNE = 1000
DRAWS = 1000
TARGET_ACCEPT = 0.9

# The crossed model mixes slowly; see fit_crossed_attribution for the evidence.
CROSSED_TUNE = 3000
CROSSED_DRAWS = 3000

# Pre-registered in doc 05 §7, derived from the null distribution of the 89%
# upper bound at the real denominators — not from a football argument.
HOLDING_GATE_PP = 0.0837 / 100

SPLIT_HALF_DRAWS = 200
MIN_INTS_FOR_SPLIT = 6

COLUMNS = [*ANALYSIS_COLUMNS, "passer_player_id", "touchdown", "return_yards"]


def sampler_health(idata) -> dict:
    """Gate 1 from document 03: divergences, r_hat, ESS."""
    summary = az.summary(idata)
    report = {
        "divergences": int(idata["sample_stats"]["diverging"].sum().item()),
        "max_r_hat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "min_ess_tail": float(summary["ess_tail"].min()),
    }
    report["pass"] = (
        report["divergences"] == 0
        and report["max_r_hat"] < 1.01
        and report["min_ess_bulk"] > 400
        and report["min_ess_tail"] > 400
    )
    print(
        f"  Gate 1 (sampler health): {'PASS' if report['pass'] else 'FAIL'} — "
        f"divergences={report['divergences']}, max r_hat={report['max_r_hat']:.4f}, "
        f"min ess_bulk={report['min_ess_bulk']:.0f}"
    )
    return report


def eti89(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 5.5)), float(np.percentile(values, 94.5))]


# --------------------------------------------------------------------------
# 3a — crossed quarterback x defense attribution
# --------------------------------------------------------------------------


def implied_rate_sd(intercept: np.ndarray, sigma: np.ndarray, rng, size: int = 4000) -> np.ndarray:
    """Population SD of true rates implied by a log-odds scale SD.

    A sigma on the log-odds scale is not readable as football. This converts it
    by simulation rather than the delta method: for each posterior draw, sample
    entity effects from Normal(0, sigma), push them through the inverse logit,
    and take the SD of the resulting rates. Exact, and it does not assume the
    spread is small.
    """
    noise = rng.standard_normal((len(sigma), size))
    rates = 1.0 / (1.0 + np.exp(-(intercept[:, None] + sigma[:, None] * noise)))
    return rates.std(axis=1)


def fit_crossed_attribution(charted: pl.DataFrame) -> dict:
    """3a — is the interception spread the quarterback's or the defense's?"""
    print(f"\n{'=' * 72}\n3a — INT conversion, crossed QB x defense\n{'=' * 72}")

    frame = charted.drop_nulls(["passer_player_id", "defteam"]).with_columns(
        pl.concat_str(
            [pl.col("season").cast(pl.String), pl.col("passer_player_id")], separator="_"
        ).alias("qb_season"),
        pl.concat_str([pl.col("season").cast(pl.String), pl.col("defteam")], separator="_").alias(
            "def_season"
        ),
        (pl.col("interception") == 1).cast(pl.Int64).alias("picked"),
    )

    qb_levels = sorted(frame["qb_season"].unique().to_list())
    def_levels = sorted(frame["def_season"].unique().to_list())
    qb_index = {level: i for i, level in enumerate(qb_levels)}
    def_index = {level: i for i, level in enumerate(def_levels)}

    qb_idx = np.array([qb_index[v] for v in frame["qb_season"].to_list()])
    def_idx = np.array([def_index[v] for v in frame["def_season"].to_list()])
    y = frame["picked"].to_numpy()

    print(
        f"  {len(y):,} charted interception-worthy throws, "
        f"{len(qb_levels)} QB-seasons, {len(def_levels)} defense-seasons, "
        f"league rate {y.mean():.3%}"
    )

    coords = {"qb": qb_levels, "defense": def_levels, "obs": np.arange(len(y))}
    with pm.Model(coords=coords):
        # League log-odds. Normal(0, 1.5) spans essentially the whole rate range
        # without asserting a direction.
        intercept = pm.Normal("intercept", mu=0.0, sigma=1.5)

        # Non-centered on both factors. This is a ruling: document 04's Gate 1
        # failure was a centered hierarchy funnelling, and the median
        # quarterback-season here has only 10 throws — thinner than the fumble
        # model that failed.
        sigma_qb = pm.HalfNormal("sigma_qb", sigma=1.0)
        sigma_def = pm.HalfNormal("sigma_def", sigma=1.0)
        z_qb = pm.Normal("z_qb", mu=0.0, sigma=1.0, dims="qb")
        z_def = pm.Normal("z_def", mu=0.0, sigma=1.0, dims="defense")

        eta = intercept + sigma_qb * z_qb[qb_idx] + sigma_def * z_def[def_idx]
        pm.Bernoulli("picked", logit_p=eta, observed=y, dims="obs")

        prior = pm.sample_prior_predictive(draws=500, random_seed=RANDOM_SEED)
        rate_range = prior["prior_predictive"]["picked"].values.mean(axis=-1)
        print(
            f"  prior predictive INT rate range: [{rate_range.min():.3f}, {rate_range.max():.3f}]"
        )

        # Attempt 1 at the standard 1000/1000 returned ess_bulk 289 and r_hat
        # 1.0138 with ZERO divergences. That signature is slow mixing, not bad
        # geometry — the two crossed scales trade off against each other and the
        # chains explore that ridge slowly. The honest fix is more draws, not a
        # higher target_accept, which document 03 section 5 explicitly forbids as
        # a way of quieting warnings.
        idata = pm.sample(
            draws=CROSSED_DRAWS,
            tune=CROSSED_TUNE,
            chains=CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
            progressbar=False,
        )
    trace_path = paths.RESEARCH_OUTPUT_DIR / "trace_attribution_crossed.nc"
    idata.to_netcdf(trace_path)

    health = sampler_health(idata)

    posterior = idata["posterior"]
    intercept_draws = posterior["intercept"].values.ravel()
    rng = np.random.default_rng(RANDOM_SEED)
    sd_qb = implied_rate_sd(intercept_draws, posterior["sigma_qb"].values.ravel(), rng)
    sd_def = implied_rate_sd(intercept_draws, posterior["sigma_def"].values.ravel(), rng)

    league_rate = float(y.mean())
    result = {
        "n_throws": int(len(y)),
        "n_qb_seasons": len(qb_levels),
        "n_def_seasons": len(def_levels),
        "league_rate": league_rate,
        "sampler_health": health,
        "sigma_qb_logodds": float(posterior["sigma_qb"].values.mean()),
        "sigma_qb_logodds_eti89": eti89(posterior["sigma_qb"].values),
        "sigma_def_logodds": float(posterior["sigma_def"].values.mean()),
        "sigma_def_logodds_eti89": eti89(posterior["sigma_def"].values),
        "qb_rate_sd_pp": float(sd_qb.mean()) * 100,
        "qb_rate_sd_eti89_pp": [x * 100 for x in eti89(sd_qb)],
        "def_rate_sd_pp": float(sd_def.mean()) * 100,
        "def_rate_sd_eti89_pp": [x * 100 for x in eti89(sd_def)],
        "qb_relative": float(sd_qb.mean()) / league_rate,
        "def_relative": float(sd_def.mean()) / league_rate,
        # Pre-registered comparison: which factor's interval clears the bound the
        # design could reach under a true zero (doc 05 §7).
        "null_bound_qb_pp": 6.85,
        "null_bound_def_pp": 6.03,
    }

    print(
        f"\n  quarterback spread: {result['qb_rate_sd_pp']:.2f} pp "
        f"[{result['qb_rate_sd_eti89_pp'][0]:.2f}, {result['qb_rate_sd_eti89_pp'][1]:.2f}] "
        f"= {result['qb_relative']:.1%} of the league rate"
    )
    print(
        f"  defense spread:     {result['def_rate_sd_pp']:.2f} pp "
        f"[{result['def_rate_sd_eti89_pp'][0]:.2f}, {result['def_rate_sd_eti89_pp'][1]:.2f}] "
        f"= {result['def_relative']:.1%} of the league rate"
    )

    # P(quarterback spread exceeds defense spread), a direct posterior statement.
    result["p_qb_exceeds_def"] = float((sd_qb > sd_def).mean())
    print(f"  P(QB spread > defense spread) = {result['p_qb_exceeds_def']:.3f}")
    return result


# --------------------------------------------------------------------------
# 3b — offensive holding
# --------------------------------------------------------------------------


def fit_penalty_subtype(counts: pl.DataFrame, name: str) -> dict:
    """Marginalized beta-binomial, the parameterization document 04 settled on."""
    n = counts["n"].to_numpy()
    k = counts["k"].to_numpy()
    league_rate = float(k.sum() / n.sum())
    print(
        f"\n  --- {name}: {counts.height} team-seasons, "
        f"{k.sum():,} calls on {n.sum():,} plays, rate {league_rate:.4%}"
    )

    with pm.Model(coords={"team_season": counts["team_season"].to_list()}):
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
        idata = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
            progressbar=False,
        )
    idata.to_netcdf(paths.RESEARCH_OUTPUT_DIR / f"trace_attr_{name}.nc")
    health = sampler_health(idata)

    population_sd = idata["posterior"]["population_sd"].values.ravel()
    bounds = eti89(population_sd)

    # Cross-check against the exact grid posterior. Agreement is evidence the
    # NUTS fit converged, independent of its own diagnostics.
    grid = fit_grid(n, k).summary()

    result = {
        "name": name,
        "team_seasons": counts.height,
        "total_plays": int(n.sum()),
        "total_calls": int(k.sum()),
        "league_rate": league_rate,
        "sampler_health": health,
        "population_sd_pp": float(population_sd.mean()) * 100,
        "population_sd_eti89_pp": [b * 100 for b in bounds],
        "relative_spread": float(population_sd.mean()) / league_rate,
        "grid_population_sd_pp": grid["population_sd_mean"] * 100,
        "grid_eti89_pp": [b * 100 for b in grid["population_sd_eti89"]],
    }
    print(
        f"  population SD {result['population_sd_pp']:.4f} pp "
        f"[{result['population_sd_eti89_pp'][0]:.4f}, {result['population_sd_eti89_pp'][1]:.4f}] "
        f"= {result['relative_spread']:.1%} relative"
    )
    print(
        f"  grid cross-check: {result['grid_population_sd_pp']:.4f} pp "
        f"[{result['grid_eti89_pp'][0]:.4f}, {result['grid_eti89_pp'][1]:.4f}]"
    )
    return result


def penalty_subtype_counts(pbp: pl.DataFrame, penalty_type: str) -> pl.DataFrame:
    """Team-season counts of one penalty type over plays the team was on the field for."""
    plays = pbp.filter(pl.col("posteam").is_not_null() & pl.col("defteam").is_not_null())
    team_season = lambda col: pl.concat_str(  # noqa: E731
        [pl.col("season").cast(pl.String), pl.col(col)], separator="_"
    ).alias("team_season")

    on_field = (
        pl.concat([plays.select(team_season("posteam")), plays.select(team_season("defteam"))])
        .group_by("team_season")
        .agg(pl.len().alias("n"))
    )
    flagged = (
        pbp.filter((pl.col("penalty") == 1) & (pl.col("penalty_type") == penalty_type))
        .select(team_season("penalty_team"))
        .group_by("team_season")
        .agg(pl.len().alias("k"))
    )
    return (
        on_field.join(flagged, on="team_season", how="left")
        .with_columns(pl.col("k").fill_null(0))
        .sort("team_season")
    )


# --------------------------------------------------------------------------
# 3c — return yardage persistence
# --------------------------------------------------------------------------


def split_half_return_yards(ints: pl.DataFrame, rng: np.random.Generator) -> dict:
    """Split-half correlation of mean INT return yards, per defense-season.

    Same machinery as document 02, so the number is directly comparable to the
    +0.055 (fumble recovery) and +0.164 (interception EPA) already on record.
    """
    frame = (
        ints.drop_nulls("return_yards")
        .select(
            pl.concat_str(
                [pl.col("season").cast(pl.String), pl.col("defteam")], separator="_"
            ).alias("def_season"),
            pl.col("return_yards").cast(pl.Float64),
        )
        .drop_nulls("def_season")
    )
    groups = [
        group["return_yards"].to_numpy()
        for _, group in frame.group_by("def_season")
        if group.height >= MIN_INTS_FOR_SPLIT
    ]
    print(f"\n  {len(groups)} defense-seasons with {MIN_INTS_FOR_SPLIT}+ interceptions")

    correlations = []
    for _ in range(SPLIT_HALF_DRAWS):
        first, second = [], []
        for values in groups:
            order = rng.permutation(len(values))
            half = len(values) // 2
            first.append(values[order[:half]].mean())
            second.append(values[order[half : 2 * half]].mean())
        correlations.append(np.corrcoef(first, second)[0, 1])

    correlations = np.array(correlations)
    return {
        "n_defense_seasons": len(groups),
        "split_half_r": float(correlations.mean()),
        "pct5_95": [float(np.percentile(correlations, 5)), float(np.percentile(correlations, 95))],
        "detectability_floor": 0.11,
    }


def main() -> None:
    paths.ensure_data_dirs()
    pbp = load_pbp(PBP_SEASONS, columns=COLUMNS)
    ftn = load_ftn(FTN_SEASONS)
    results = {}

    # ---- 3a ---------------------------------------------------------------
    charted = (
        ftn.select(
            pl.col("nflverse_game_id").alias("game_id"),
            pl.col("nflverse_play_id").cast(pl.Float64).alias("play_id"),
            "is_interception_worthy",
        )
        .filter(pl.col("is_interception_worthy"))
        .join(pbp, on=["game_id", "play_id"], how="inner")
    )
    results["3a_crossed_attribution"] = fit_crossed_attribution(charted)

    # ---- 3b ---------------------------------------------------------------
    print(f"\n{'=' * 72}\n3b — penalty subtypes: is offensive holding random?\n{'=' * 72}")
    holding = fit_penalty_subtype(penalty_subtype_counts(pbp, "Offensive Holding"), "holding")
    false_start = fit_penalty_subtype(penalty_subtype_counts(pbp, "False Start"), "false_start")

    upper = holding["population_sd_eti89_pp"][1] / 100
    gate = upper < HOLDING_GATE_PP
    print(f"\n{'=' * 72}")
    print(
        f"GATE 3b (offensive holding is random): {'PASS' if gate else 'FAIL'} — "
        f"89% upper bound {upper * 100:.4f} pp, pre-registered threshold "
        f"{HOLDING_GATE_PP * 100:.4f} pp"
    )
    print(f"{'=' * 72}")
    results["3b_penalty_subtypes"] = {
        "offensive_holding": holding,
        "false_start": false_start,
        "gate_pass": bool(gate),
        "gate_threshold_pp": HOLDING_GATE_PP * 100,
        "observed_upper_bound_pp": upper * 100,
    }

    # ---- 3c ---------------------------------------------------------------
    print(f"\n{'=' * 72}\n3c — does interception return yardage persist?\n{'=' * 72}")
    ints = pbp.filter(pl.col("interception") == 1)
    rng = np.random.default_rng(RANDOM_SEED)
    persistence = split_half_return_yards(ints, rng)
    print(
        f"  split-half r = {persistence['split_half_r']:+.3f} "
        f"[{persistence['pct5_95'][0]:+.3f}, {persistence['pct5_95'][1]:+.3f}] "
        f"(detectability floor +/-{persistence['detectability_floor']})"
    )

    pick_six = (
        ints.group_by(
            pl.concat_str(
                [pl.col("season").cast(pl.String), pl.col("defteam")], separator="_"
            ).alias("team_season")
        )
        .agg(pl.len().alias("n"), (pl.col("touchdown") == 1).sum().alias("k"))
        .drop_nulls("team_season")
        .sort("team_season")
    )
    pick_six_fit = fit_penalty_subtype(pick_six, "pick_six")
    results["3c_return_yardage"] = {
        "split_half": persistence,
        "pick_six": pick_six_fit,
        "note": "No gate — doc 05 section 7 recorded this question as underpowered.",
    }

    out = paths.RESEARCH_OUTPUT_DIR / "06_attribution.json"
    with out.open("w") as handle:
        json.dump(results, handle, indent=2, default=str)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
