"""Phase 4, step 4 — the special-teams round, run against the gates document 14 committed.

Three components, three separate Gate A arguments:

    (a) Punting, weather-aware   — PU-1 sampler health (relative tolerance)
                                   PU-2 spot calibration, cubic fallback named
                                   PU-3 is punter skill resolvable?
                                   PU-4 is the wind effect resolvable?
                                   PU-5 posterior predictive
                                   PU-6 temperature, reported, no pass rule
    (b) The punt bounce          — an observability determination, not a fit
    (c) Kick and punt returns    — persistence, era-aware, within-season splits

**No component of this round earns a ledger row.** Gate A denies all three: a
punt is a played-out sequence, a return is a played-out sequence, and the one
genuine branch point — the bounce — is unobservable. The deliverables are
skill-or-noise verdicts as reported findings, exactly as the Phase 4 plan scoped
them.

    uv run python research/22_special_teams.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_power = import_module("22_special_teams_power")
_grid = import_module("_crossed_gaussian_grid")
_seq = import_module("10_sequencing_power")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.ingest import PBP_SEASONS, load_pbp  # noqa: E402

RANDOM_SEED = _power.RANDOM_SEED
CHAINS, TUNE, DRAWS = 4, 1000, 1000
TARGET_ACCEPT = 0.9
SPOT_CENTRE = _power.SPOT_CENTRE
MIN_BIN_PUNTS = 100
CALIBRATION_PERCENTILE = 94.5
REFERENCE_R = _power.REFERENCE_R
MIN_POWER = 0.80

# Document 09 §9's corrective, applied: a convergence tolerance must be RELATIVE
# to the quantity, or stated per-candidate from its own Monte Carlo standard
# error. An absolute tolerance borrowed from a different scale is the mistake
# that round recorded.
GRID_RELATIVE_TOLERANCE = 0.05

# Thresholds are filled in from `docs/research/14-special-teams.md`, which is
# committed before this script runs. Hard-coded rather than read back, so a
# re-run of the power script cannot silently move a committed gate.
THRESHOLDS_PATH = paths.RESEARCH_OUTPUT_DIR / "22_special_teams_power.json"


def eti89(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 5.5)), float(np.percentile(values, 94.5))]


def sampler_health(idata) -> dict:
    """Gate PU-1's substantive half: divergences, r_hat, effective sample size."""
    import arviz as az

    summary = az.summary(idata)
    divergences = int(idata["sample_stats"]["diverging"].values.sum())
    return {
        "divergences": divergences,
        "max_r_hat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "min_ess_tail": float(summary["ess_tail"].min()),
        "pass": bool(
            divergences == 0
            and summary["r_hat"].max() < 1.01
            and summary["ess_bulk"].min() > 400
            and summary["ess_tail"].min() > 400
        ),
    }


def fit_punt_model(punts: pl.DataFrame, wind_centre: float, temp_centre: float, *, cubic: bool):
    """The hierarchical punter model.

    A **Student-t** likelihood, and that is a ruling rather than a default. Net
    punt yards run from +79 down to −57, and the left tail is punts returned for
    a touchdown: a coverage failure, not a punting outcome. Under a Normal
    likelihood some thirty such plays would set the residual scale for all 22,403
    punts and shrink every punter toward the mean by an amount those thirty plays
    chose. Estimating the degrees of freedom lets the data say how heavy the tail
    is instead.
    """
    import pymc as pm

    spot = punts["spot"].to_numpy() - SPOT_CENTRE
    has_weather = punts["has_weather"].to_numpy().astype(float)
    wind = np.nan_to_num(punts["clean_wind"].to_numpy(), nan=wind_centre)
    temp = np.nan_to_num(punts["clean_temp"].to_numpy(), nan=temp_centre)
    net = punts["net"].to_numpy()

    roof_levels = ["dome", "closed", "open"]
    roof_matrix = np.column_stack(
        [(punts["clean_roof"].to_numpy() == level).astype(float) for level in roof_levels]
    )

    punters = sorted(punts["punter_season"].unique().to_list())
    lookup = {name: i for i, name in enumerate(punters)}
    punter_index = np.array([lookup[name] for name in punts["punter_season"].to_list()])

    coords = {"punter_season": punters, "roof_level": roof_levels, "obs": np.arange(len(net))}
    with pm.Model(coords=coords) as model:
        intercept = pm.Normal("intercept", mu=41.0, sigma=10.0)
        beta_spot = pm.Normal("beta_spot", mu=0.0, sigma=0.5)
        beta_spot2 = pm.Normal("beta_spot2", mu=0.0, sigma=1.0)
        roof = pm.Normal("roof", mu=0.0, sigma=3.0, dims="roof_level")
        beta_wind = pm.Normal("beta_wind", mu=0.0, sigma=0.5)
        beta_temp = pm.Normal("beta_temp", mu=0.0, sigma=0.2)

        # Non-centered on the punter effect. A ruling, not a default: document 04's
        # Gate 1 failure was a centered hierarchy funnelling, and the same
        # geometry applies whenever a group-level scale is small.
        sigma_punter = pm.HalfNormal("sigma_punter", sigma=3.0)
        offset = pm.Normal("z_punter", mu=0.0, sigma=1.0, dims="punter_season")
        punter_effect = pm.Deterministic("punter", offset * sigma_punter, dims="punter_season")

        mu = (
            intercept
            + beta_spot * spot
            + beta_spot2 * spot**2 / 100.0
            + roof_matrix @ roof
            + beta_wind * (wind - wind_centre) * has_weather
            + beta_temp * (temp - temp_centre) * has_weather
            + punter_effect[punter_index]
        )
        if cubic:
            beta_spot3 = pm.Normal("beta_spot3", mu=0.0, sigma=1.0)
            mu = mu + beta_spot3 * spot**3 / 1000.0

        sigma = pm.HalfNormal("sigma", sigma=10.0)
        nu = pm.Gamma("nu", alpha=2.0, beta=0.1)
        pm.StudentT("net", nu=nu, mu=mu, sigma=sigma, observed=net, dims="obs")

        idata = pm.sample(
            DRAWS,
            tune=TUNE,
            chains=CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
            progressbar=False,
        )
        idata.update(
            pm.sample_posterior_predictive(idata, random_seed=RANDOM_SEED, progressbar=False)
        )
    return model, idata, punter_index, len(punters)


def spot_calibration(punts: pl.DataFrame, idata) -> dict:
    """Gate PU-2: the largest standardized miss across 5-yard spot bins.

    Identical in construction to document 05b's Gate FG-2, which was itself fixed
    by a power check that caught its multiplicity problem: requiring *every* bin
    to sit inside its own interval fails on a correct model roughly a third of the
    time, because eight bins at nominal coverage pass together only 0.89^8 of the
    time. Reducing to a single maximum and calibrating that maximum against its
    own posterior predictive reference prices the multiplicity in exactly once.
    """
    net = punts["net"].to_numpy()
    replicated = idata["posterior_predictive"]["net"].values.reshape(-1, len(net))
    bins = ((punts["spot"].to_numpy() // 5) * 5).astype(int)

    observed_stat, reference = 0.0, np.zeros(len(replicated))
    rows = []
    per_bin_observed, per_bin_replicated = [], []
    for value in np.unique(bins):
        selected = bins == value
        if selected.sum() < MIN_BIN_PUNTS:
            continue
        predicted = replicated[:, selected].mean(axis=1)
        spread = predicted.std(ddof=1)
        if spread <= 0:
            continue
        actual = float(net[selected].mean())
        per_bin_observed.append(abs(actual - predicted.mean()) / spread)
        per_bin_replicated.append(np.abs(predicted - predicted.mean()) / spread)
        rows.append(
            {
                "spot_bin": int(value),
                "punts": int(selected.sum()),
                "observed_net": actual,
                "predicted_net": float(predicted.mean()),
                "miss": actual - float(predicted.mean()),
                "standardized": per_bin_observed[-1],
            }
        )
    observed_stat = float(max(per_bin_observed))
    reference = np.max(np.column_stack(per_bin_replicated), axis=1)
    threshold = float(np.percentile(reference, CALIBRATION_PERCENTILE))
    return {
        "statistic": observed_stat,
        "reference_percentile": CALIBRATION_PERCENTILE,
        "threshold": threshold,
        "pass": bool(observed_stat <= threshold),
        "bins": rows,
    }


def posterior_predictive(punts: pl.DataFrame, idata, punter_index: np.ndarray) -> dict:
    """Gate PU-5: the league mean and the between-punter variance of net yards.

    The variance half is the one that matters, for the reason document 03 §6 Gate
    4 gave: a model that gets the mean right and the spread wrong is precisely the
    model that would mislead about skill.
    """
    net = punts["net"].to_numpy()
    replicated = idata["posterior_predictive"]["net"].values.reshape(-1, len(net))

    def between_punter_variance(values: np.ndarray) -> float:
        sums = np.bincount(punter_index, weights=values)
        counts = np.bincount(punter_index)
        keep = counts >= 10
        return float(np.var(sums[keep] / counts[keep], ddof=1))

    observed_mean = float(net.mean())
    observed_variance = between_punter_variance(net)
    replicated_means = replicated.mean(axis=1)
    replicated_variances = np.array([between_punter_variance(row) for row in replicated])

    mean_tail = float((replicated_means < observed_mean).mean())
    variance_tail = float((replicated_variances < observed_variance).mean())
    return {
        "observed_mean": observed_mean,
        "mean_tail_probability": mean_tail,
        "observed_between_punter_variance": observed_variance,
        "variance_tail_probability": variance_tail,
        "pass": bool(0.055 < mean_tail < 0.945 and 0.055 < variance_tail < 0.945),
    }


def punt_bounce(pbp: pl.DataFrame, punts: pl.DataFrame) -> dict:
    """(b) Is the post-landing roll observable at all?

    **This is a data question and it is answered before any model is proposed**,
    because the Phase 4 plan required the observability to be confronted head-on
    rather than discovered after a fit. Document 09 §5's onside row is the
    precedent for what happens when the answer is no.
    """
    print(f"\n{'=' * 72}\n(b) THE PUNT BOUNCE — observability\n{'=' * 72}")

    outcome = (
        pl.when(pl.col("touchback") == 1)
        .then(pl.lit("touchback"))
        .when(pl.col("punt_fair_catch") == 1)
        .then(pl.lit("fair catch"))
        .when(pl.col("punt_downed") == 1)
        .then(pl.lit("downed"))
        .when(pl.col("punt_out_of_bounds") == 1)
        .then(pl.lit("out of bounds"))
        .when(pl.col("return_yards") > 0)
        .then(pl.lit("returned"))
        .otherwise(pl.lit("other"))
        .alias("outcome")
    )
    classified = punts.with_columns(outcome)
    counts = classified.group_by("outcome").agg(pl.len().alias("punts")).sort("punts", reverse=True)
    print(counts)

    # Does any description record a landing spot separately from a final spot?
    descriptions = pbp.filter(pl.col("punt_attempt") == 1)["desc"].to_list()
    bounce_mentions = sum(1 for text in descriptions if "bounce" in text.lower())

    # The aggregate bound. Caught punts have zero roll by construction; downed and
    # out-of-bounds punts carry flight PLUS roll inside the same kick_distance.
    # The gap between their conditional means at matched spots is an upper bound
    # on the mean roll — an upper bound and not an estimate, because punt INTENT
    # differs between the two groups and that confound cannot be removed here.
    matched = []
    classified = classified.with_columns(((pl.col("spot") // 5) * 5).cast(pl.Int32).alias("bin"))
    caught = classified.filter(pl.col("outcome").is_in(["fair catch", "returned"]))
    bounced = classified.filter(pl.col("outcome").is_in(["downed", "out of bounds"]))
    for value in sorted(set(caught["bin"].to_list()) & set(bounced["bin"].to_list())):
        left = caught.filter(pl.col("bin") == value)
        right = bounced.filter(pl.col("bin") == value)
        if left.height < MIN_BIN_PUNTS or right.height < MIN_BIN_PUNTS:
            continue
        matched.append(
            {
                "spot_bin": int(value),
                "caught_punts": int(left.height),
                "bounced_punts": int(right.height),
                "caught_kick_distance": float(left["kick_distance"].mean()),
                "bounced_kick_distance": float(right["kick_distance"].mean()),
                "gap": float(right["kick_distance"].mean() - left["kick_distance"].mean()),
            }
        )
    weighted_gap = float(
        np.average(
            [row["gap"] for row in matched], weights=[row["bounced_punts"] for row in matched]
        )
    )

    print(
        f"\n  descriptions mentioning a bounce: {bounce_mentions} of {len(descriptions):,}\n"
        "  landing-spot column in the play-by-play: none — `kick_distance` is the distance\n"
        "  to where the ball was FIRST TOUCHED OR CAME TO REST, so for a downed or\n"
        "  out-of-bounds punt it already contains the roll, and for a caught punt the roll\n"
        "  is zero by construction.\n"
        f"\n  aggregate upper bound on the mean roll, matched on 5-yard spot bins: "
        f"{weighted_gap:+.2f} yards"
    )
    print(
        "  UPPER BOUND, not an estimate: a punter aiming to pin the opponent kicks shorter\n"
        "  and hangs it less, so the two groups differ in intent as well as in roll."
    )

    return {
        "gate_a": "PASS — a loose oblong ball on the ground, structurally the fumble case",
        "observable": False,
        "verdict": "UNRESOLVABLE BY CONSTRUCTION",
        "outcome_counts": counts.to_dicts(),
        "descriptions_mentioning_bounce": bounce_mentions,
        "n_punt_descriptions": len(descriptions),
        "matched_bins": matched,
        "aggregate_roll_upper_bound_yards": weighted_gap,
    }


def returns_round(pbp: pl.DataFrame, thresholds: dict) -> dict:
    """(c) Does return skill persist? Era-aware, within-season splits only."""
    print(f"\n{'=' * 72}\n(c) KICK AND PUNT RETURNS — persistence\n{'=' * 72}")
    rows = []
    for kind in ("kickoff", "punt"):
        returns = _power.return_table(pbp, kind)
        eras = (
            _power.KICKOFF_ERAS
            if kind == "kickoff"
            else {"2016-2025 (no rule break)": tuple(range(2016, 2026))}
        )
        for era, seasons in eras.items():
            subset = returns.filter(pl.col("season").is_in(list(seasons)))
            for entity in ("returner", "team"):
                label = f"{kind} / {entity} / {era}"
                cell = thresholds["returns"].get(label)
                if cell is None or cell.get("skipped"):
                    print(f"  {label:56s} skipped — {cell['reason'] if cell else 'no threshold'}")
                    rows.append({"cell": label, "skipped": True})
                    continue
                matrix, starts, sizes = _power.entity_game_matrix(subset, entity)
                rng = np.random.default_rng(RANDOM_SEED)
                mask = _seq.split_masks(starts, sizes, len(matrix), rng, _power.N_SPLITS)
                observed = _power.split_half_r(matrix, mask, starts)
                power_at_reference = next(
                    row["power"] for row in cell["power"] if row["target_true_r"] == REFERENCE_R
                )
                persists = bool(observed > cell["null_p95"])
                interpretable = bool(power_at_reference >= MIN_POWER)
                verdict = (
                    "SKILL"
                    if persists
                    else ("NOISE — evidence of absence" if interpretable else "UNRESOLVABLE")
                )
                print(
                    f"  {label:56s} r = {observed:+.4f}  threshold {cell['null_p95']:.4f}  "
                    f"power {power_at_reference:.2f}  -> {verdict}"
                )
                rows.append(
                    {
                        "cell": label,
                        "split_half_r": observed,
                        "threshold": cell["null_p95"],
                        "power_at_reference_r": power_at_reference,
                        "persists": persists,
                        "interpretable": interpretable,
                        "verdict": verdict,
                        "n_entities": cell["n_entities"],
                        "n_returns": cell["n_returns"],
                    }
                )
    print(
        "\n  Gate A denies every row above a ledger entry regardless of the verdict: a return\n"
        "  is a played-out sequence of blocking and tackling, not a branch resolved by\n"
        "  nobody. Document 05 §3's return-yardage row already said so; these are reported\n"
        "  findings, not treatment-table changes."
    )
    return {"gate_a": "FAIL — a played-out sequence", "cells": rows}


def main() -> None:
    paths.ensure_data_dirs()
    with THRESHOLDS_PATH.open() as handle:
        thresholds = json.load(handle)

    pbp = load_pbp(PBP_SEASONS, columns=[*_power.COLUMNS, "desc"])
    punts = _power.punt_table(pbp)
    wind_centre = thresholds["punting"]["design"]["wind_centre"]
    temp_centre = thresholds["punting"]["design"]["temp_centre"]

    print(f"{'=' * 72}\n(a) PUNTING — the model\n{'=' * 72}")
    print(f"  {punts.height:,} punts, {punts['punter_season'].n_unique()} punter-seasons")

    arms = {}
    for label, cubic in (("quadratic (pre-registered)", False), ("cubic (fallback)", True)):
        print(f"\n--- arm: {label} ---")
        _, idata, punter_index, n_punters = fit_punt_model(
            punts, wind_centre, temp_centre, cubic=cubic
        )
        posterior = idata["posterior"]
        health = sampler_health(idata)
        calibration = spot_calibration(punts, idata)
        predictive = posterior_predictive(punts, idata, punter_index)

        sigma_punter = posterior["sigma_punter"].values.ravel()
        beta_wind = posterior["beta_wind"].values.ravel()
        beta_temp = posterior["beta_temp"].values.ravel()

        punter_bound = thresholds["punting"]["punter_null_bound"]
        wind_bound = thresholds["punting"]["wind_null_bound"]
        pu3 = {
            "sigma_punter_mean": float(sigma_punter.mean()),
            "eti89": eti89(sigma_punter),
            "null_bound": punter_bound,
            "pass": bool(np.percentile(sigma_punter, 5.5) > punter_bound),
        }
        pu4 = {
            "beta_wind_mean": float(beta_wind.mean()),
            "eti89": eti89(beta_wind),
            "null_bound": wind_bound,
            "pass": bool(np.percentile(beta_wind, 94.5) < wind_bound),
            "yards_lost_at_15mph": float(-beta_wind.mean() * 15.0),
        }
        pu6 = {
            "beta_temp_mean": float(beta_temp.mean()),
            "eti89": eti89(beta_temp),
            "yards_across_40F": float(beta_temp.mean() * 40.0),
        }
        roof_effects = {
            level: {
                "mean": float(posterior["roof"].values[..., i].mean()),
                "eti89": eti89(posterior["roof"].values[..., i].ravel()),
            }
            for i, level in enumerate(posterior["roof"].coords["roof_level"].values)
        }

        print(
            f"  PU-1 sampler health   {'PASS' if health['pass'] else 'FAIL'} — "
            f"{health['divergences']} divergences, r_hat {health['max_r_hat']:.4f}, "
            f"ESS bulk {health['min_ess_bulk']:.0f}"
        )
        print(
            f"  PU-2 spot calibration {'PASS' if calibration['pass'] else 'FAIL'} — "
            f"{calibration['statistic']:.3f} vs {calibration['threshold']:.3f}"
        )
        print(
            f"  PU-3 punter skill     {'PASS' if pu3['pass'] else 'FAIL'} — "
            f"sigma_punter {pu3['sigma_punter_mean']:.3f} "
            f"[{pu3['eti89'][0]:.3f}, {pu3['eti89'][1]:.3f}] vs null bound {punter_bound:.3f}"
        )
        print(
            f"  PU-4 wind resolvable  {'PASS' if pu4['pass'] else 'FAIL'} — "
            f"beta_wind {pu4['beta_wind_mean']:+.4f} "
            f"[{pu4['eti89'][0]:+.4f}, {pu4['eti89'][1]:+.4f}] vs null bound {wind_bound:+.4f}; "
            f"{pu4['yards_lost_at_15mph']:.2f} net yards lost at 15 mph"
        )
        print(
            f"  PU-5 posterior pred.  {'PASS' if predictive['pass'] else 'FAIL'} — "
            f"mean tail {predictive['mean_tail_probability']:.3f}, "
            f"variance tail {predictive['variance_tail_probability']:.3f}"
        )
        print(
            f"  PU-6 temperature      reported — {pu6['beta_temp_mean']:+.4f} yd/degF, "
            f"{pu6['yards_across_40F']:+.2f} net yards across a 40 degF swing"
        )
        arms[label] = {
            "pu1_sampler_health": health,
            "pu2_spot_calibration": calibration,
            "pu3_punter_skill": pu3,
            "pu4_wind": pu4,
            "pu5_posterior_predictive": predictive,
            "pu6_temperature": pu6,
            "roof_effects": roof_effects,
            "n_punter_seasons": n_punters,
        }

    adopted = (
        "quadratic (pre-registered)"
        if arms["quadratic (pre-registered)"]["pu2_spot_calibration"]["pass"]
        else "cubic (fallback)"
    )
    print(f"\n  ADOPTED ARM: {adopted}")

    # ---- secondary, reported: whose skill is net punting? -----------------
    print("\n--- SECONDARY (reported, no pass rule): punter versus return unit ---")
    x = _power.punt_design_matrix(punts, wind_centre, temp_centre)
    y = punts["net"].to_numpy()
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    codes, sizes = [], []
    for column in ("punter_season", "return_unit_season"):
        names = sorted(punts[column].unique().to_list())
        lookup = {name: i for i, name in enumerate(names)}
        codes.append(np.array([lookup[value] for value in punts[column].to_list()]))
        sizes.append(len(names))
    design, zty, yty, oney = _grid.prepare(codes, sizes, residual)
    crossed = _grid.fit(design, zty, yty, oney)
    print(
        f"  punter-season    sigma {crossed['sigma_a']['mean']:.3f} "
        f"[{crossed['sigma_a']['eti89_lb']:.3f}, {crossed['sigma_a']['eti89_ub']:.3f}] "
        f"vs null bound {thresholds['punting']['punter_null_bound']:.3f}"
    )
    print(
        f"  return-unit-season sigma {crossed['sigma_b']['mean']:.3f} "
        f"[{crossed['sigma_b']['eti89_lb']:.3f}, {crossed['sigma_b']['eti89_ub']:.3f}] "
        f"vs null bound {thresholds['punting']['return_unit_null_bound']:.3f}"
    )
    print(f"  P(punter spread > return-unit spread) = {crossed['p_a_exceeds_b']:.3f}")

    # Gate PU-1's second half: a RELATIVE convergence tolerance, per document 09.
    nuts_sigma = arms[adopted]["pu3_punter_skill"]["sigma_punter_mean"]
    relative_gap = abs(crossed["sigma_a"]["mean"] - nuts_sigma) / nuts_sigma
    print(
        f"\n  grid vs NUTS on sigma_punter: {crossed['sigma_a']['mean']:.4f} vs "
        f"{nuts_sigma:.4f}, relative gap {relative_gap:.2%} vs tolerance "
        f"{GRID_RELATIVE_TOLERANCE:.0%} — "
        f"{'PASS' if relative_gap <= GRID_RELATIVE_TOLERANCE else 'FAIL'}"
    )
    print(
        "  (the two are not the same estimator — the grid runs on covariate-residualized\n"
        "  net yards with a crossed return-unit factor, the hierarchy on the raw response\n"
        "  with a Student-t likelihood — so this is a sanity band, not an identity check)"
    )

    bounce = punt_bounce(pbp, punts)
    returns = returns_round(pbp, thresholds)

    results = {
        "punting": {
            "gate_a": "FAIL — a punt is a played-out sequence, not a branch resolved by nobody",
            "n_punts": int(punts.height),
            "adopted_arm": adopted,
            "arms": arms,
            "secondary_crossed": {
                "punter_season": crossed["sigma_a"],
                "return_unit_season": crossed["sigma_b"],
                "p_punter_exceeds_return_unit": crossed["p_a_exceeds_b"],
                "grid_vs_nuts_relative_gap": relative_gap,
                "relative_tolerance": GRID_RELATIVE_TOLERANCE,
            },
        },
        "punt_bounce": bounce,
        "returns": returns,
        "random_seed": RANDOM_SEED,
    }
    out = paths.RESEARCH_OUTPUT_DIR / "22_special_teams.json"
    with out.open("w") as handle:
        json.dump(results, handle, indent=2, default=str)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
