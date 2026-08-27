"""Part B of the dropped-interception confound study — the fits.

Every gate in `docs/research/43-dropped-pick-confounds-prereg.md` §7 gets a
verdict here, against thresholds that were computed and committed by
`research/61_dropped_pick_power.py` before this script existed. Nothing in
`src/nfl_simulator/` changes on any outcome: document 32's closure stands, and
the study's product is a **reported diagnostic**, not a ledger row.

Three arms, in document 43 §5's order:

    arm 1   worthy rate across QB-seasons and defence-seasons, beta-binomial
            grid. Gate D-1 — descriptive, no pass rule. How much of a
            dropped-pick count is the quarterback's own throwing.
    arm 2   conversion p(INT | worthy, X), the hierarchical logistic. Gate C-1
            for sampler health, and the beta that arm 3 residualises against.
            Arm 2b repeats it without the two hindsight-risk columns.
    arm 3   the gate arm. Persistence of the conditioned residual across
            defence-seasons and across pooled defences, with the crossed
            Gaussian grid. Gate C-2 against Part A's threshold, Gate C-3 from
            Part A's power.

    uv run python research/62_dropped_pick_confounds.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import arviz as az
import numpy as np
import polars as pl
import pymc as pm

sys.path.insert(0, str(Path(__file__).parent))

_crossed = import_module("_crossed_gaussian_grid")
_betabinom = import_module("_betabinom_grid")
_power = import_module("61_dropped_pick_power")

from nfl_simulator import paths  # noqa: E402

RANDOM_SEED = _power.RANDOM_SEED
LOGIT_SLOPE = _power.LOGIT_SLOPE
CROSS_CHECK_TOLERANCE_PP = 1.0  # document 43 §5 and §10

DRAWS = 1000
TUNE = 1000
CHAINS = 4

# The two columns document 43 §3(c) flags for charter hindsight: a charter who
# saw the interception may grade the throw with knowledge of how it ended.
HINDSIGHT_COLUMNS = ("is_catchable_ball", "is_contested_ball")


# --------------------------------------------------------------------------
# arm 1 — the worthy-rate spreads (Gate D-1)
# --------------------------------------------------------------------------


def worthy_rate_spread(name: str, counts: pl.DataFrame, power: dict) -> dict:
    """Population SD of the true entity worthy rate, with its Part A power."""
    n = counts["n"].to_numpy().astype(float)
    k = counts["k"].to_numpy().astype(float)
    posterior = _betabinom.fit_grid(n, k)
    summary = posterior.summary()
    lower, upper = (bound * 100 for bound in summary["population_sd_eti89"])
    report = {
        "name": name,
        "entities": int(counts.height),
        "opportunities": int(n.sum()),
        "league_rate": float(k.sum() / n.sum()),
        "median_entity_rate": float(np.median(k / n)),
        "median_n": float(np.median(n)),
        "population_sd_mean_pp": summary["population_sd_mean"] * 100,
        "population_sd_eti89_pp": [lower, upper],
        "relative_spread": summary["population_sd_mean"] / float(k.sum() / n.sum()),
        "edge_mass": posterior.edge_mass(),
        "power_at_reference": power["power_at_reference"],
        "resolvable": power["resolvable"],
    }
    print(f"\n  {name}: {report['entities']} entities, median n {report['median_n']:.0f}")
    print(
        f"    league worthy rate {report['league_rate']:.4%}, median entity rate "
        f"{report['median_entity_rate']:.4%}"
    )
    print(
        f"    population SD {report['population_sd_mean_pp']:.3f} pp "
        f"[{lower:.3f}, {upper:.3f}] 89% — {report['relative_spread']:.1%} of the league rate"
    )
    print(
        f"    Part A power at 12.5% relative {report['power_at_reference']:.3f} "
        f"({'resolvable' if report['resolvable'] else 'unresolvable'}); "
        f"grid edge mass {report['edge_mass']:.1e}"
    )
    return report


# --------------------------------------------------------------------------
# arm 2 — conversion by covariates (Gate C-1)
# --------------------------------------------------------------------------


def sampler_health(idata, variables: list[str]) -> dict:
    """Gate C-1's sampler half, in document 03's form.

    Named offenders as well as the extrema: the gate is over every parameter,
    non-centred offsets included, so a marginal failure on one ``z_d`` level
    reads very differently from one on ``sigma_d``. Which it is decides what a
    reader may still take from the fit, so the name is recorded, not just the
    number.
    """
    summary = az.summary(idata, var_names=variables)
    report = {
        "divergences": int(idata["sample_stats"]["diverging"].sum().item()),
        "max_r_hat": float(summary["r_hat"].max()),
        "max_r_hat_parameter": str(summary["r_hat"].idxmax()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "min_ess_bulk_parameter": str(summary["ess_bulk"].idxmin()),
        "min_ess_tail": float(summary["ess_tail"].min()),
        "min_ess_tail_parameter": str(summary["ess_tail"].idxmin()),
        "parameters_over_r_hat_bar": int((summary["r_hat"] >= 1.01).sum()),
        "parameters_under_ess_bar": int(
            ((summary["ess_bulk"] <= 400) | (summary["ess_tail"] <= 400)).sum()
        ),
        "parameters_checked": int(summary.shape[0]),
    }

    # The same three bars applied to the parameters this study actually reports,
    # separately from the 171 non-centred offsets it does not.
    reported = summary.loc[
        [name for name in summary.index if name.startswith(("alpha", "beta", "sigma"))]
    ]
    report["reported_parameters"] = {
        "count": int(reported.shape[0]),
        "max_r_hat": float(reported["r_hat"].max()),
        "min_ess_bulk": float(reported["ess_bulk"].min()),
        "min_ess_tail": float(reported["ess_tail"].min()),
        "pass": bool(
            reported["r_hat"].max() < 1.01
            and reported["ess_bulk"].min() > 400
            and reported["ess_tail"].min() > 400
        ),
    }
    report["pass"] = bool(
        report["divergences"] == 0
        and report["max_r_hat"] < 1.01
        and report["min_ess_bulk"] > 400
        and report["min_ess_tail"] > 400
    )
    print(
        f"    Gate C-1 sampler health: {'PASS' if report['pass'] else 'FAIL'} — "
        f"divergences {report['divergences']}, max r_hat {report['max_r_hat']:.4f} "
        f"({report['max_r_hat_parameter']}), min ess_bulk {report['min_ess_bulk']:.0f} "
        f"({report['min_ess_bulk_parameter']}), min ess_tail {report['min_ess_tail']:.0f} "
        f"({report['min_ess_tail_parameter']}) over {report['parameters_checked']} parameters"
    )
    reported = report["reported_parameters"]
    print(
        f"      of which alpha / beta / sigma only ({reported['count']} parameters): "
        f"{'PASS' if reported['pass'] else 'FAIL'} — max r_hat {reported['max_r_hat']:.4f}, "
        f"min ess_bulk {reported['min_ess_bulk']:.0f}, "
        f"min ess_tail {reported['min_ess_tail']:.0f}"
    )
    print(
        f"      {report['parameters_over_r_hat_bar']} parameters at or over the r_hat bar, "
        f"{report['parameters_under_ess_bar']} at or under an ESS bar"
    )
    return report


def fit_conversion(
    label: str,
    matrix: np.ndarray,
    names: tuple[str, ...],
    outcome: np.ndarray,
    defence_codes: np.ndarray,
    n_defence: int,
    qb_codes: np.ndarray,
    n_qb: int,
) -> tuple[dict, object]:
    """Document 43 §5's arm-2 model, fitted and summarised."""
    print(f"\n  arm {label}: {matrix.shape[0]:,} throws, {matrix.shape[1]} covariates")
    model = _power.build_conversion_model(matrix, outcome, defence_codes, n_defence, qb_codes, n_qb)
    with model:
        idata = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            random_seed=RANDOM_SEED,
            progressbar=False,
            nuts_sampler="nutpie",
        )

    health = sampler_health(idata, ["alpha", "beta", "sigma_d", "sigma_q", "z_d", "z_q"])
    posterior = idata["posterior"]
    beta_summary = az.summary(idata, var_names=["beta"])

    coefficients = []
    for index, name in enumerate(names):
        row = beta_summary.iloc[index]
        coefficients.append(
            {
                "name": name,
                "mean": float(row["mean"]),
                "eti89": [float(row["eti89_lb"]), float(row["eti89_ub"])],
                "excludes_zero": bool(row["eti89_lb"] > 0 or row["eti89_ub"] < 0),
                "odds_ratio": float(np.exp(row["mean"])),
            }
        )

    print("    beta (logit scale, standardised covariates; * = 89% interval excludes zero)")
    for row in sorted(coefficients, key=lambda item: -abs(item["mean"])):
        marker = "*" if row["excludes_zero"] else " "
        print(
            f"      {marker} {row['name']:22s} {row['mean']:+.3f} "
            f"[{row['eti89'][0]:+.3f}, {row['eti89'][1]:+.3f}]  "
            f"odds x{row['odds_ratio']:.2f}"
        )

    variances = {}
    for parameter, entity in (("sigma_d", "defence-season"), ("sigma_q", "QB-season")):
        draws = posterior[parameter].values.ravel()
        quantiles = np.quantile(draws, [0.055, 0.945])
        variances[parameter] = {
            "entity": entity,
            "logit_mean": float(draws.mean()),
            "logit_eti89": [float(quantiles[0]), float(quantiles[1])],
            "pp_mean": float(draws.mean()) * LOGIT_SLOPE * 100,
            "pp_eti89": [float(q) * LOGIT_SLOPE * 100 for q in quantiles],
        }
        entry = variances[parameter]
        print(
            f"    {parameter} ({entity}): logit {entry['logit_mean']:.3f} "
            f"[{entry['logit_eti89'][0]:.3f}, {entry['logit_eti89'][1]:.3f}]  "
            f"= {entry['pp_mean']:.2f} pp [{entry['pp_eti89'][0]:.2f}, "
            f"{entry['pp_eti89'][1]:.2f}] on the probability scale"
        )

    report = {
        "label": label,
        "throws": int(matrix.shape[0]),
        "covariates": list(names),
        "sampler": health,
        "beta": coefficients,
        "alpha_mean": float(posterior["alpha"].values.mean()),
        "variance_components": variances,
    }
    return report, idata


# --------------------------------------------------------------------------
# arm 3 — persistence of the conditioned residual (Gates C-2 and C-3)
# --------------------------------------------------------------------------


def residual_persistence(
    name: str,
    residual: np.ndarray,
    code_a: np.ndarray,
    size_a: int,
    code_b: np.ndarray,
    size_b: int,
    power: dict,
    arm2_upper_pp: float,
) -> dict:
    """One crossed-grid fit, its Gate C-2 verdict and its Gate C-3 licence."""
    design, zty, yty, oney = _crossed.prepare([code_a, code_b], [size_a, size_b], residual)
    fitted = _crossed.fit(design, zty, yty, oney)

    upper_pp = fitted["sigma_a"]["eti89_ub"] * 100
    threshold_pp = power["gate_threshold_pp"]
    c2_pass = upper_pp < threshold_pp
    c3_pass = power["resolvable"]
    gap = abs(upper_pp - arm2_upper_pp)

    report = {
        "name": name,
        "rows": int(len(residual)),
        "levels_a": int(size_a),
        "levels_b": int(size_b),
        "sigma_d_mean_pp": fitted["sigma_a"]["mean"] * 100,
        "sigma_d_eti89_pp": [
            fitted["sigma_a"]["eti89_lb"] * 100,
            fitted["sigma_a"]["eti89_ub"] * 100,
        ],
        "sigma_q_mean_pp": fitted["sigma_b"]["mean"] * 100,
        "sigma_q_eti89_pp": [
            fitted["sigma_b"]["eti89_lb"] * 100,
            fitted["sigma_b"]["eti89_ub"] * 100,
        ],
        "sigma_e_mean_pp": fitted["sigma_e"]["mean"] * 100,
        "edge_mass": fitted["edge_mass"],
        "gate_threshold_pp": threshold_pp,
        "gate_c2_pass": bool(c2_pass),
        "power_at_reference": power["power_at_reference"],
        "gate_c3_pass": bool(c3_pass),
        "reportable_as_finding": bool(c3_pass),
        "arm2_upper_bound_pp": arm2_upper_pp,
        "cross_check_gap_pp": gap,
        "cross_check_pass": bool(gap <= CROSS_CHECK_TOLERANCE_PP),
    }

    print(f"\n  {name}: {report['rows']:,} rows, {size_a} x {size_b} levels")
    print(
        f"    sigma_d {report['sigma_d_mean_pp']:.2f} pp "
        f"[{report['sigma_d_eti89_pp'][0]:.2f}, {report['sigma_d_eti89_pp'][1]:.2f}] 89%; "
        f"sigma_q {report['sigma_q_mean_pp']:.2f} pp "
        f"[{report['sigma_q_eti89_pp'][0]:.2f}, {report['sigma_q_eti89_pp'][1]:.2f}]; "
        f"residual SD {report['sigma_e_mean_pp']:.2f} pp; edge mass {report['edge_mass']:.1e}"
    )
    print(
        f"    Gate C-2, {name}: upper bound {upper_pp:.2f} pp vs threshold "
        f"{threshold_pp:.2f} pp -> {'PASS' if c2_pass else 'FAIL'}"
    )
    print(
        f"    Gate C-3: power at 12.5% = {report['power_at_reference']:.3f} -> "
        f"{'PASS' if c3_pass else 'FAIL'}"
        + ("" if c3_pass else " -> C-2 not reportable as a finding")
    )
    print(
        f"    Gate C-1 cross-check vs arm 2 ({arm2_upper_pp:.2f} pp): gap {gap:.2f} pp -> "
        f"{'PASS' if report['cross_check_pass'] else 'FAIL'} "
        f"(tolerance {CROSS_CHECK_TOLERANCE_PP:.1f} pp)"
    )
    return report


# --------------------------------------------------------------------------
# secondaries — continuity with document 32 §3
# --------------------------------------------------------------------------


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(left, right)[0, 1])


def split_half(frame: pl.DataFrame, value: str, keys: list[str], minimum: int = 3) -> dict:
    """Odd/even-week split-half correlation of a per-throw quantity.

    Document 32 §3 reported +0.140 for raw conversion at the defence-season
    grain. Reproducing it is the continuity check; running the same split on the
    conditioned residual is what says whether conditioning moved it.
    """
    halves = (
        frame.with_columns((pl.col("week") % 2 == 1).alias("odd"))
        .group_by([*keys, "odd"])
        .agg(pl.col(value).mean().alias("value"), pl.len().alias("n"))
    )
    wide = (
        halves.filter(pl.col("odd"))
        .join(halves.filter(~pl.col("odd")), on=keys, how="inner", suffix="_even")
        .filter((pl.col("n") >= minimum) & (pl.col("n_even") >= minimum))
    )
    if wide.height < 3:
        return {"entities": int(wide.height), "r": None, "min_chances_per_half": minimum}
    return {
        "entities": int(wide.height),
        "r": _pearson(wide["value"].to_numpy(), wide["value_even"].to_numpy()),
        "min_chances_per_half": minimum,
        "median_chances_per_half": float(np.median(wide["n"].to_numpy())),
    }


def season_to_season(effects: pl.DataFrame) -> dict:
    """Correlation of shrunk defence effects between adjacent seasons."""
    pairs = effects.join(
        effects.with_columns((pl.col("season") - 1).alias("season")),
        on=["defteam", "season"],
        how="inner",
        suffix="_next",
    )
    return {
        "pairs": int(pairs.height),
        "r": _pearson(pairs["effect"].to_numpy(), pairs["effect_next"].to_numpy())
        if pairs.height >= 3
        else None,
    }


# --------------------------------------------------------------------------


def main() -> None:
    paths.ensure_data_dirs()
    power_path = paths.RESEARCH_OUTPUT_DIR / "61_dropped_pick_power.json"
    if not power_path.exists():
        raise SystemExit(f"{power_path} is missing — Part A must run and commit before Part B")
    power = json.loads(power_path.read_text())["designs"]

    frame = _power.build_worthy_frame()

    print("\n=== arm 1 — worthy rate, two grains (Gate D-1) ===")
    qb_counts = _power.rate_counts(
        frame.charted.drop_nulls("passer_player_id"),
        ["season", "passer_player_id"],
        minimum=_power.MIN_QB_ATTEMPTS,
    )
    defence_counts = _power.rate_counts(frame.charted, ["season", "defteam"])
    arm1 = {
        "qb_season": worthy_rate_spread(
            "worthy rate, QB-season", qb_counts, power["worthy_rate_qb_season"]
        ),
        "defence_season": worthy_rate_spread(
            "worthy rate, defence-season", defence_counts, power["worthy_rate_defence_season"]
        ),
    }

    print("\n=== arm 2 — conversion by covariates ===")
    arm2, idata = fit_conversion(
        "2",
        frame.design_matrix,
        frame.feature_names,
        frame.outcome,
        frame.defence_season_codes,
        frame.n_defence_seasons,
        frame.qb_season_codes,
        frame.n_qb_seasons,
    )

    keep = [
        index for index, name in enumerate(frame.feature_names) if name not in HINDSIGHT_COLUMNS
    ]
    arm2b, _ = fit_conversion(
        "2b (no is_catchable_ball / is_contested_ball)",
        frame.design_matrix[:, keep],
        tuple(frame.feature_names[index] for index in keep),
        frame.outcome,
        frame.defence_season_codes,
        frame.n_defence_seasons,
        frame.qb_season_codes,
        frame.n_qb_seasons,
    )

    # Part A simulated its thresholds around a saved posterior-mean beta. Re-fitting
    # the same model with the same seed should reproduce it; the agreement is
    # reported so a reader can see the gate was judged at the p_hat it was built for.
    saved = json.loads((paths.RESEARCH_OUTPUT_DIR / "61_beta_hat.json").read_text())
    refit_beta = np.array([row["mean"] for row in arm2["beta"]])
    beta_gap = float(np.abs(refit_beta - np.asarray(saved["beta"])).max())
    alpha_gap = abs(arm2["alpha_mean"] - saved["alpha"])
    print(
        f"\n  arm 2 vs Part A's saved fixed effects: max |d beta| {beta_gap:.4f}, "
        f"|d alpha| {alpha_gap:.4f}"
    )

    print("\n=== arm 3 — persistence of the conditioned residual (Gates C-2, C-3) ===")
    residual_rows = _power.residual_frame(frame)
    eta = _power.linear_predictor(saved, residual_rows, frame.model)
    p_hat = 1.0 / (1.0 + np.exp(-eta))
    outcome = residual_rows["interception"].cast(pl.Float64).to_numpy()
    residual = outcome - p_hat

    defence_season_codes, n_defence_season = _power._codes(residual_rows, ["season", "defteam"])
    defence_pooled_codes, n_defence_pooled = _power._codes(residual_rows, ["defteam"])
    qb_codes, n_qb = _power._codes(residual_rows, ["season", "passer_player_id"])

    arm2_upper_pp = arm2["variance_components"]["sigma_d"]["pp_eti89"][1]
    arm3 = {
        "defence_season_x_qb_season": residual_persistence(
            "defence-season x QB-season",
            residual,
            defence_season_codes,
            n_defence_season,
            qb_codes,
            n_qb,
            power["residual_defence_season_x_qb_season"],
            arm2_upper_pp,
        ),
        "defence_pooled_x_qb_season": residual_persistence(
            "defence pooled x QB-season",
            residual,
            defence_pooled_codes,
            n_defence_pooled,
            qb_codes,
            n_qb,
            power["residual_defence_pooled_x_qb_season"],
            arm2_upper_pp,
        ),
    }
    # The QB-season row is the same crossed fit read on its other factor, so it
    # is reported against its own Part A threshold rather than re-fitted.
    qb_power = power["residual_qb_season_sigma_q"]
    qb_upper_pp = arm3["defence_season_x_qb_season"]["sigma_q_eti89_pp"][1]
    arm3["qb_season_sigma_q"] = {
        "name": "QB-season sigma_q (read from the defence-season x QB-season fit)",
        "upper_bound_pp": qb_upper_pp,
        "gate_threshold_pp": qb_power["gate_threshold_pp"],
        "gate_c2_pass": bool(qb_upper_pp < qb_power["gate_threshold_pp"]),
        "power_at_reference": qb_power["power_at_reference"],
        "gate_c3_pass": bool(qb_power["resolvable"]),
        "reportable_as_finding": bool(qb_power["resolvable"]),
    }
    entry = arm3["qb_season_sigma_q"]
    print(
        f"\n  QB-season sigma_q: upper bound {qb_upper_pp:.2f} pp vs threshold "
        f"{entry['gate_threshold_pp']:.2f} pp -> {'PASS' if entry['gate_c2_pass'] else 'FAIL'}; "
        f"power {entry['power_at_reference']:.3f} -> "
        f"{'PASS' if entry['gate_c3_pass'] else 'FAIL'}"
    )

    print("\n=== secondaries (reported, never gated) ===")
    conditioned = residual_rows.with_columns(pl.Series("residual", residual))
    secondaries = {
        "raw_conversion_split_half_defence_season": split_half(
            frame.worthy.with_columns(pl.col("interception").cast(pl.Float64)),
            "interception",
            ["season", "defteam"],
        ),
        "conditioned_residual_split_half_defence_season": split_half(
            conditioned, "residual", ["season", "defteam"]
        ),
        "raw_conversion_split_half_defence_pooled": split_half(
            frame.worthy.with_columns(pl.col("interception").cast(pl.Float64)),
            "interception",
            ["defteam"],
        ),
        "conditioned_residual_split_half_defence_pooled": split_half(
            conditioned, "residual", ["defteam"]
        ),
    }
    for name, entry in secondaries.items():
        value = "n/a" if entry["r"] is None else f"{entry['r']:+.3f}"
        print(f"  {name}: r = {value} on {entry['entities']} entities")

    # Shrunk defence-season effects from arm 2, correlated across adjacent seasons.
    labels = (
        frame.model.select("season", "defteam")
        .with_columns(pl.concat_str(["season", "defteam"], separator="|").alias("label"))["label"]
        .unique(maintain_order=True)
        .to_list()
    )
    effect_means = idata["posterior"]["u_d"].values.mean(axis=(0, 1))
    effects = pl.DataFrame(
        {
            "season": [int(label.split("|")[0]) for label in labels],
            "defteam": [label.split("|")[1] for label in labels],
            "effect": effect_means[: len(labels)],
        }
    )
    secondaries["shrunk_defence_effect_season_to_season"] = season_to_season(effects)
    entry = secondaries["shrunk_defence_effect_season_to_season"]
    value = "n/a" if entry["r"] is None else f"{entry['r']:+.3f}"
    print(f"  shrunk_defence_effect_season_to_season: r = {value} on {entry['pairs']} pairs")

    out = paths.RESEARCH_OUTPUT_DIR / "62_dropped_pick_confounds.json"
    out.write_text(
        json.dumps(
            {
                "random_seed": RANDOM_SEED,
                "draws": DRAWS,
                "tune": TUNE,
                "chains": CHAINS,
                "cross_check_tolerance_pp": CROSS_CHECK_TOLERANCE_PP,
                "guards": frame.guards,
                "residual_frame": {
                    "rows": int(residual_rows.height),
                    "defence_seasons": int(n_defence_season),
                    "defences_pooled": int(n_defence_pooled),
                    "qb_seasons": int(n_qb),
                    "conversion_rate": float(outcome.mean()),
                    "mean_p_hat": float(p_hat.mean()),
                },
                "arm1_worthy_rate": arm1,
                "arm2_conversion": arm2,
                "arm2b_no_hindsight_columns": arm2b,
                "arm2_vs_part_a_beta": {"max_abs_beta_gap": beta_gap, "alpha_gap": alpha_gap},
                "arm3_residual_persistence": arm3,
                "secondaries": secondaries,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
