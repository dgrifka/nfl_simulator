"""Phase 3, step 4 — refit the kicker model with weather and extra points.

Executes the change proposal in `docs/research/05b-fg-model-foundations.md` §10,
committed at `8e0b50e` before this script existed. Constants here are that
section's; changing one means editing the doc too.

Writes the posterior the simulator consumes:

    research/outputs/trace_fg_weather.nc          full posterior
    research/outputs/fg_weather_summary.json      gate report

    uv run python research/14_fg_weather_model.py
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

_power = import_module("13_fg_weather_power")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.fg_model import sanitize_weather  # noqa: E402
from nfl_simulator.ingest import PBP_SEASONS, load_pbp  # noqa: E402

RANDOM_SEED = 20260817
CHAINS = 4
TUNE = 1000
DRAWS = 1000
TARGET_ACCEPT = 0.9

DISTANCE_CENTRE = 40.0
MIN_CELL_ATTEMPTS = 100
GATE_W3_THRESHOLD = 0.00268  # pre-registered, from the null simulation
WIND_BUCKET = 5.0

# Outdoors is the reference level, so it gets no coefficient.
ROOF_LEVELS = ("dome", "closed", "open")

COLUMNS = [
    "season",
    "game_id",
    "play_type",
    "kick_distance",
    "field_goal_result",
    "extra_point_attempt",
    "extra_point_result",
    "kicker_player_id",
    "roof",
    "temp",
    "wind",
]


def load_kicks(exclude_blocked: bool = False, with_stadium: bool = False) -> pl.DataFrame:
    """Field goals and extra points, sanitized, in one table.

    ``exclude_blocked`` was added by document 27, which refits this model on the
    population that excludes blocked kicks. It defaults to ``False`` so the
    published posterior is reproduced exactly, and it lives here rather than in a
    copy of this loader so the fit and the refit cannot drift apart.

    ``with_stadium`` was added by document 66, the elevation round, for the same
    reason and with the same default: it carries ``stadium_id`` through so the
    elevation covariate can be joined on, and it is one extra column on the same
    loader rather than a second copy of these masks.


    Extra points carry their own `kick_distance` (33 yards for 98.5% of them,
    with the rest moved by a penalty), so they go through the same distance terms
    rather than being pinned at a constant. `delta_xp` then means what it says:
    the difference between an extra point and a field goal *from the same
    distance*.
    """
    columns = [*COLUMNS, "stadium_id"] if with_stadium else COLUMNS
    pbp = load_pbp(PBP_SEASONS, columns=columns)

    fg_mask = (pl.col("play_type") == "field_goal") & pl.col("kick_distance").is_not_null()
    xp_mask = (
        (pl.col("extra_point_attempt") == 1)
        & pl.col("extra_point_result").is_not_null()
        & pl.col("kick_distance").is_not_null()
    )
    if exclude_blocked:
        # Narrow the two masks, never the frame — the trap document 26 §8 names.
        fg_mask = fg_mask & (pl.col("field_goal_result") != "blocked")
        xp_mask = xp_mask & (pl.col("extra_point_result") != "blocked")

    shared = ["season", "kicker_player_id", "roof", "temp", "wind"]
    if with_stadium:
        shared = [*shared, "stadium_id", "game_id"]
    field_goals = pbp.filter(fg_mask).select(
        *shared,
        pl.col("kick_distance").cast(pl.Float64).alias("distance"),
        (pl.col("field_goal_result") == "made").cast(pl.Int64).alias("made"),
        pl.lit(0).alias("is_xp"),
    )
    extra_points = pbp.filter(xp_mask).select(
        *shared,
        pl.col("kick_distance").cast(pl.Float64).alias("distance"),
        (pl.col("extra_point_result") == "good").cast(pl.Int64).alias("made"),
        pl.lit(1).alias("is_xp"),
    )

    kicks = pl.concat([field_goals, extra_points]).drop_nulls("kicker_player_id")

    # One shared sanitize implementation with the simulator, so the model cannot
    # be trained on one definition of a windy day and applied to another.
    cleaned = [
        sanitize_weather(row["roof"], row["wind"], row["temp"])
        for row in kicks.select("roof", "wind", "temp").iter_rows(named=True)
    ]
    return kicks.with_columns(
        pl.Series("wind", [w.wind for w in cleaned], dtype=pl.Float64),
        pl.Series("temp", [w.temp for w in cleaned], dtype=pl.Float64),
        pl.Series("has_weather", [w.has_weather for w in cleaned]),
        pl.concat_str(
            [pl.col("season").cast(pl.String), pl.col("kicker_player_id")], separator="_"
        ).alias("kicker_season"),
    )


def build_model(
    kicks: pl.DataFrame,
    kicker_levels: list[str],
    kicker_idx: np.ndarray,
    centres,
    *,
    cubic: bool = False,
):
    """The §10 model: distance curve, roof, weather, and an extra-point arm.

    ``cubic=True`` is the fallback document 05b §9 named in advance — *"a spline
    or a monotone fit is the Phase 3 option"* — reached only when Gate W-4 fails
    on the adopted quadratic curve. It continues the polynomial ladder the model
    already climbs (linear failed FG-2, quadratic adopted) rather than being a
    shape chosen to fit the bin that missed.
    """
    centred = kicks["distance"].to_numpy() - DISTANCE_CENTRE
    made = kicks["made"].to_numpy()
    is_xp = kicks["is_xp"].to_numpy().astype(float)
    has_weather = kicks["has_weather"].to_numpy().astype(float)
    wind = np.nan_to_num(kicks["wind"].to_numpy().astype(float)) - centres["wind"]
    temp = np.nan_to_num(kicks["temp"].to_numpy().astype(float)) - centres["temp"]
    roof = kicks["roof"].to_numpy()
    roof_design = np.column_stack([(roof == level).astype(float) for level in ROOF_LEVELS])

    coords = {"kicker_season": kicker_levels, "roof_level": list(ROOF_LEVELS)}
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", mu=2.0, sigma=1.5)
        beta = pm.Normal("beta", mu=0.0, sigma=0.2)
        gamma = pm.Normal("gamma", mu=0.0, sigma=0.2)
        sigma_kicker = pm.HalfNormal("sigma_kicker", sigma=1.0)

        roof_effect = pm.Normal("roof", mu=0.0, sigma=0.5, dims="roof_level")
        beta_wind = pm.Normal("beta_wind", mu=0.0, sigma=0.05)
        beta_temp = pm.Normal("beta_temp", mu=0.0, sigma=0.02)
        delta_xp = pm.Normal("delta_xp", mu=0.0, sigma=1.0)
        # Centred on full transfer, so the data has to argue for anything else.
        lambda_xp = pm.Normal("lambda_xp", mu=1.0, sigma=0.5)

        # Non-centered: a ruling, not a default. See foundations doc section 5.
        z = pm.Normal("z", mu=0.0, sigma=1.0, dims="kicker_season")
        kicker = pm.Deterministic("kicker", sigma_kicker * z, dims="kicker_season")

        # A kicker's field-goal ability scaled by lambda_xp on extra points, so
        # the transfer is estimated rather than assumed.
        kicker_term = kicker[kicker_idx] * (1.0 + (lambda_xp - 1.0) * is_xp)

        eta = (
            alpha
            + beta * centred
            + gamma * centred**2 / 100.0
            + pm.math.dot(roof_design, roof_effect)
            + beta_wind * wind * has_weather
            + beta_temp * temp * has_weather
            + delta_xp * is_xp
            + kicker_term
        )
        if cubic:
            # Scaled by 1000 so the coefficient sits on the same order as beta
            # and gamma, and the prior below means something readable.
            delta_cubic = pm.Normal("delta_cubic", mu=0.0, sigma=0.2)
            eta = eta + delta_cubic * centred**3 / 1000.0
        pm.Bernoulli("made", logit_p=eta, observed=made)
    return model


def eti89(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 5.5)), float(np.percentile(values, 94.5))]


def make_probabilities(idata, kicks: pl.DataFrame, kicker_idx: np.ndarray, centres) -> np.ndarray:
    """Per-kick make probability on every posterior draw."""
    posterior = idata["posterior"]

    def flat(name):
        return posterior[name].values.ravel()[:, None]

    n_kickers = posterior["kicker"].shape[-1]
    kicker = posterior["kicker"].values.reshape(-1, n_kickers)
    roof_draws = posterior["roof"].values.reshape(-1, len(ROOF_LEVELS))

    centred = (kicks["distance"].to_numpy() - DISTANCE_CENTRE)[None, :]
    is_xp = kicks["is_xp"].to_numpy().astype(float)[None, :]
    has_weather = kicks["has_weather"].to_numpy().astype(float)[None, :]
    wind = (np.nan_to_num(kicks["wind"].to_numpy().astype(float)) - centres["wind"])[None, :]
    temp = (np.nan_to_num(kicks["temp"].to_numpy().astype(float)) - centres["temp"])[None, :]
    roof = kicks["roof"].to_numpy()
    roof_design = np.column_stack([(roof == level).astype(float) for level in ROOF_LEVELS])

    eta = (
        flat("alpha")
        + flat("beta") * centred
        + flat("gamma") * centred**2 / 100.0
        + roof_draws @ roof_design.T
        + flat("beta_wind") * wind * has_weather
        + flat("beta_temp") * temp * has_weather
        + flat("delta_xp") * is_xp
        + kicker[:, kicker_idx] * (1.0 + (flat("lambda_xp") - 1.0) * is_xp)
    )
    if "delta_cubic" in posterior:
        eta = eta + flat("delta_cubic") * centred**3 / 1000.0
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))


def standardized_miss(masks, made: np.ndarray, p_hat: np.ndarray) -> float:
    """Largest standardized cell miss. Gate FG-2's statistic, reused."""
    worst = 0.0
    for mask in masks:
        expected = p_hat[mask]
        sd = np.sqrt((expected * (1.0 - expected)).sum()) / mask.sum()
        if sd > 0:
            worst = max(worst, abs(made[mask].mean() - expected.mean()) / sd)
    return worst


def calibration_gate(
    label: str, masks, kicks: pl.DataFrame, p_draws: np.ndarray, seed: int = RANDOM_SEED
) -> dict:
    """The FG-2 construction: a maximum, calibrated against its own reference."""
    made = kicks["made"].to_numpy().astype(float)
    p_hat = p_draws.mean(axis=0)
    observed = standardized_miss(masks, made, p_hat)

    rng = np.random.default_rng(seed)
    picks = rng.choice(len(p_draws), size=400, replace=False)
    replicated = np.array(
        [
            standardized_miss(masks, rng.binomial(1, p_draws[i]).astype(float), p_draws[i])
            for i in picks
        ]
    )
    threshold = float(np.percentile(replicated, 94.5))
    report = {
        "observed_statistic": float(observed),
        "reference_94_5_pct": threshold,
        "n_cells": len(masks),
        "pass": bool(observed <= threshold),
    }
    print(
        f"\n{label}: {'PASS' if report['pass'] else 'FAIL'} — worst standardized miss "
        f"{observed:.3f}, reference 94.5th pct {threshold:.3f}, {len(masks)} cells"
    )
    return report


def fit_arm(
    label: str,
    kicks: pl.DataFrame,
    kicker_levels: list[str],
    kicker_idx: np.ndarray,
    centres: dict,
    thresholds: dict,
    *,
    cubic: bool,
    seed: int = RANDOM_SEED,
    trace_prefix: str = "trace_fg_weather",
) -> tuple[object, dict]:
    """One arm, end to end through all eight gates.

    ``seed`` and ``trace_prefix`` are document 27's, and both default to this
    round's values. The prefix exists so a refit writes alongside the published
    posterior rather than over it — the artifacts of a shipped version are never
    overwritten by a candidate, as v1.1's and v1.2's were preserved.
    """
    print(f"\n{'#' * 72}\n### {label}\n{'#' * 72}")
    model = build_model(kicks, kicker_levels, kicker_idx, centres, cubic=cubic)
    with model:
        idata = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=seed,
            progressbar=False,
        )
    suffix = "cubic" if cubic else "quadratic"
    idata.to_netcdf(paths.RESEARCH_OUTPUT_DIR / f"{trace_prefix}_{suffix}.nc")

    posterior = idata["posterior"]

    # ---- Gate W-1 --------------------------------------------------------
    summary = az.summary(idata)
    w1 = {
        "divergences": int(idata["sample_stats"]["diverging"].sum().item()),
        "max_r_hat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "min_ess_tail": float(summary["ess_tail"].min()),
    }
    w1["pass"] = bool(
        w1["divergences"] == 0
        and w1["max_r_hat"] < 1.01
        and w1["min_ess_bulk"] > 400
        and w1["min_ess_tail"] > 400
    )
    print(
        f"\nGate W-1 (sampler health): {'PASS' if w1['pass'] else 'FAIL'} — "
        f"divergences {w1['divergences']}, max r_hat {w1['max_r_hat']:.4f}, "
        f"min ess_bulk {w1['min_ess_bulk']:.0f}"
    )
    names = [
        "alpha",
        "beta",
        "gamma",
        "sigma_kicker",
        "roof",
        "beta_wind",
        "beta_temp",
        "delta_xp",
        "lambda_xp",
    ]
    if cubic:
        names.append("delta_cubic")
    print(
        az.summary(idata, var_names=names)[
            ["mean", "sd", "eti89_lb", "eti89_ub", "ess_bulk", "r_hat"]
        ]
    )

    p_draws = make_probabilities(idata, kicks, kicker_idx, centres)
    made = kicks["made"].to_numpy().astype(float)
    is_xp = kicks["is_xp"].to_numpy().astype(bool)

    # ---- Gate W-2: weather cells ----------------------------------------
    roof = kicks["roof"].to_numpy()
    wind = kicks["wind"].to_numpy().astype(float)
    bucket = np.where(np.isnan(wind), -1.0, np.floor(np.nan_to_num(wind) / WIND_BUCKET))
    cell = np.array([f"{r}|{int(b)}" for r, b in zip(roof, bucket, strict=True)])
    weather_masks = [
        cell == value for value in np.unique(cell) if (cell == value).sum() >= MIN_CELL_ATTEMPTS
    ]
    w2 = calibration_gate("Gate W-2 (weather calibration)", weather_masks, kicks, p_draws, seed)

    # ---- Gate W-4: distance bins ----------------------------------------
    distance = kicks["distance"].to_numpy()
    bins = (distance // 5 * 5).astype(int)
    distance_masks = [
        bins == edge for edge in np.unique(bins) if (bins == edge).sum() >= MIN_CELL_ATTEMPTS
    ]
    w4 = calibration_gate(
        "Gate W-4 (distance calibration, all kicks)", distance_masks, kicks, p_draws, seed
    )

    # Field goals only — the population Gate FG-2 was defined on. Section 7's
    # wording did not say which, so both are reported and the ambiguity is
    # recorded as a defect rather than resolved after seeing which one passes.
    fg_masks = [
        (bins == edge) & ~is_xp
        for edge in np.unique(bins[~is_xp])
        if ((bins == edge) & ~is_xp).sum() >= MIN_CELL_ATTEMPTS
    ]
    w4_fg = calibration_gate("Gate W-4 variant (field goals only)", fg_masks, kicks, p_draws, seed)

    per_bin = []
    p_hat = p_draws.mean(axis=0)
    for mask in distance_masks:
        expected = p_hat[mask]
        sd = np.sqrt((expected * (1.0 - expected)).sum()) / mask.sum()
        per_bin.append(
            {
                "bin_start": int(bins[mask][0]),
                "attempts": int(mask.sum()),
                "extra_points": int(is_xp[mask].sum()),
                "observed": float(made[mask].mean()),
                "predicted": float(expected.mean()),
                "miss_pp": float((made[mask].mean() - expected.mean()) * 100),
                "standardized": float(abs(made[mask].mean() - expected.mean()) / sd),
            }
        )
    with pl.Config(tbl_rows=20):
        print(pl.DataFrame(per_bin))

    # ---- Gate W-3: wind resolvable --------------------------------------
    beta_wind = posterior["beta_wind"].values.ravel()
    bounds = eti89(beta_wind)
    centred_45 = 45.0 - DISTANCE_CENTRE
    logit_45 = (
        posterior["alpha"].values.ravel()
        + posterior["beta"].values.ravel() * centred_45
        + posterior["gamma"].values.ravel() * centred_45**2 / 100.0
    )
    if cubic:
        logit_45 = logit_45 + posterior["delta_cubic"].values.ravel() * centred_45**3 / 1000.0
    calm = 1.0 / (1.0 + np.exp(-(logit_45 + beta_wind * (0.0 - centres["wind"]))))
    windy = 1.0 / (1.0 + np.exp(-(logit_45 + beta_wind * (15.0 - centres["wind"]))))
    drop = (calm - windy) * 100
    w3 = {
        "beta_wind_mean": float(beta_wind.mean()),
        "beta_wind_eti89": bounds,
        "threshold": thresholds["wind"],
        "pass": bool(bounds[1] < thresholds["wind"]),
        "drop_pp_calm_to_15mph_at_45yd": float(drop.mean()),
        "drop_pp_eti89": eti89(drop),
    }
    print(
        f"\nGate W-3 (wind resolvable): {'PASS' if w3['pass'] else 'FAIL'} — "
        f"beta_wind {w3['beta_wind_mean']:+.5f} [{bounds[0]:+.5f}, {bounds[1]:+.5f}], "
        f"threshold {thresholds['wind']:+.5f}"
    )
    print(
        f"  at 45 yd, calm -> 15 mph costs {w3['drop_pp_calm_to_15mph_at_45yd']:.2f} pp "
        f"[{w3['drop_pp_eti89'][0]:.2f}, {w3['drop_pp_eti89'][1]:.2f}]"
    )

    # ---- Gate W-5: posterior predictive ---------------------------------
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(p_draws), size=1000, replace=False)
    replicated = rng.binomial(1, p_draws[picks]).astype(float)
    counts = np.bincount(kicker_idx, minlength=len(kicker_levels))
    keep = counts >= 10

    def between_kicker_variance(y):
        totals = np.bincount(kicker_idx, weights=y, minlength=len(kicker_levels))
        return float(np.var((totals / np.maximum(counts, 1))[keep]))

    rate_p = float((replicated.mean(axis=1) >= made.mean()).mean())
    variances = np.array([between_kicker_variance(row) for row in replicated])
    var_p = float((variances >= between_kicker_variance(made)).mean())
    w5 = {
        "observed_make_rate": float(made.mean()),
        "make_rate_tail_p": rate_p,
        "observed_between_kicker_variance": between_kicker_variance(made),
        "variance_tail_p": var_p,
        "pass": bool(0.055 < rate_p < 0.945 and 0.055 < var_p < 0.945),
    }
    print(
        f"\nGate W-5 (posterior predictive): {'PASS' if w5['pass'] else 'FAIL'} — "
        f"make rate tail p {rate_p:.3f}, between-kicker variance tail p {var_p:.3f}"
    )

    # ---- Gates W-6 and W-7: reported, no pass rule -----------------------
    lambda_draws = posterior["lambda_xp"].values.ravel()
    delta_draws = posterior["delta_xp"].values.ravel()
    lambda_bounds = eti89(lambda_draws)
    w6 = {
        "lambda_xp_mean": float(lambda_draws.mean()),
        "lambda_xp_eti89": lambda_bounds,
        "excludes_full_transfer": bool(lambda_bounds[1] < 1.0 or lambda_bounds[0] > 1.0),
        "delta_xp_mean": float(delta_draws.mean()),
        "delta_xp_eti89": eti89(delta_draws),
    }
    beta_temp = posterior["beta_temp"].values.ravel()
    temp_bounds = eti89(beta_temp)
    w7 = {
        "beta_temp_mean": float(beta_temp.mean()),
        "beta_temp_eti89": temp_bounds,
        "null_bound": thresholds["temp"],
        "clears_null_bound": bool(temp_bounds[0] > thresholds["temp"]),
        "pp_per_40F_at_45yd": float(
            (
                1.0 / (1.0 + np.exp(-(logit_45 + beta_temp * 20.0)))
                - 1.0 / (1.0 + np.exp(-(logit_45 - beta_temp * 20.0)))
            ).mean()
        )
        * 100,
    }
    print(
        f"\nGate W-6 (extra-point transfer, reported): lambda_xp "
        f"{w6['lambda_xp_mean']:.3f} [{lambda_bounds[0]:.3f}, {lambda_bounds[1]:.3f}] — "
        f"interval {'EXCLUDES' if w6['excludes_full_transfer'] else 'contains'} 1"
    )
    print(
        f"  delta_xp {w6['delta_xp_mean']:+.3f} "
        f"[{w6['delta_xp_eti89'][0]:+.3f}, {w6['delta_xp_eti89'][1]:+.3f}] log-odds"
    )
    print(
        f"\nGate W-7 (temperature, reported): beta_temp {w7['beta_temp_mean']:+.6f} "
        f"[{temp_bounds[0]:+.6f}, {temp_bounds[1]:+.6f}] = "
        f"{w7['pp_per_40F_at_45yd']:+.2f} pp across a 40 F swing; "
        f"{'CLEARS' if w7['clears_null_bound'] else 'does NOT clear'} the null bound "
        f"{thresholds['temp']:+.6f}"
    )

    roof_draws = posterior["roof"].values.reshape(-1, len(ROOF_LEVELS))
    roof_report = {}
    print("\nRoof effects, as a make-rate change at 45 yards:")
    for i, level in enumerate(ROOF_LEVELS):
        pp = (
            1.0 / (1.0 + np.exp(-(logit_45 + roof_draws[:, i]))) - 1.0 / (1.0 + np.exp(-logit_45))
        ).mean() * 100
        roof_report[level] = {
            "log_odds_mean": float(roof_draws[:, i].mean()),
            "log_odds_eti89": eti89(roof_draws[:, i]),
            "pp_at_45yd": float(pp),
        }
        print(
            f"  {level:8s} {roof_report[level]['log_odds_mean']:+.3f} log-odds "
            f"[{roof_report[level]['log_odds_eti89'][0]:+.3f}, "
            f"{roof_report[level]['log_odds_eti89'][1]:+.3f}] = {pp:+.2f} pp"
        )

    report = {
        "label": label,
        "cubic": cubic,
        "gate_w1_sampler_health": w1,
        "gate_w2_weather_calibration": w2,
        "gate_w3_wind_resolvable": w3,
        "gate_w4_distance_calibration": w4,
        "gate_w4_variant_field_goals_only": w4_fg,
        "distance_bins": per_bin,
        "gate_w5_posterior_predictive": w5,
        "gate_w6_extra_point_transfer": w6,
        "gate_w7_temperature": w7,
        "roof_effects": roof_report,
        "sigma_kicker_mean": float(posterior["sigma_kicker"].values.mean()),
        "sigma_kicker_eti89": eti89(posterior["sigma_kicker"].values.ravel()),
    }
    return idata, report


def main() -> None:
    paths.ensure_data_dirs()
    kicks = load_kicks()
    centres = {
        "wind": float(kicks.filter(pl.col("has_weather"))["wind"].mean()),
        "temp": float(kicks.filter(pl.col("has_weather"))["temp"].mean()),
    }
    with (paths.RESEARCH_OUTPUT_DIR / "13_fg_weather_power.json").open() as handle:
        power = json.load(handle)
    thresholds = {"wind": power["gate_w3_threshold"], "temp": power["gate_w7_temp_threshold"]}

    kicker_levels = sorted(kicks["kicker_season"].unique().to_list())
    lookup = {level: i for i, level in enumerate(kicker_levels)}
    kicker_idx = np.array([lookup[v] for v in kicks["kicker_season"].to_list()])

    print(
        f"{kicks.height:,} kicks ({int(kicks.height - kicks['is_xp'].sum()):,} field goals, "
        f"{int(kicks['is_xp'].sum()):,} extra points), {len(kicker_levels)} kicker-seasons"
    )
    print(f"centres: wind {centres['wind']:.4f} mph, temp {centres['temp']:.4f} F")
    print(f"thresholds: wind {thresholds['wind']:+.5f}, temp {thresholds['temp']:+.6f}")

    results = {
        "n_kicks": int(kicks.height),
        "n_field_goals": int(kicks.height - kicks["is_xp"].sum()),
        "n_extra_points": int(kicks["is_xp"].sum()),
        "n_kicker_seasons": len(kicker_levels),
        "centres": centres,
        "thresholds": thresholds,
        "random_seed": RANDOM_SEED,
    }

    # Attempt 1 is the pre-registered quadratic curve. It stays in the record
    # whatever it does — deleting a failed pre-registered arm would defeat the
    # point of naming a fallback in advance.
    idata, quadratic = fit_arm(
        "ATTEMPT 1 — pre-registered quadratic distance curve",
        kicks,
        kicker_levels,
        kicker_idx,
        centres,
        thresholds,
        cubic=False,
    )
    results["attempt1_quadratic"] = quadratic
    adopted = "quadratic"

    if not quadratic["gate_w4_distance_calibration"]["pass"]:
        print(
            "\nGate W-4 failed on the pre-registered curve. Applying the fallback document\n"
            "05b section 9 named in advance — 'a spline or a monotone fit is the Phase 3\n"
            "option' — as a cubic term in centred distance. Both arms are reported."
        )
        idata, cubic_report = fit_arm(
            "ATTEMPT 2 — cubic distance fallback",
            kicks,
            kicker_levels,
            kicker_idx,
            centres,
            thresholds,
            cubic=True,
        )
        results["attempt2_cubic"] = cubic_report
        if cubic_report["gate_w4_distance_calibration"]["pass"]:
            adopted = "cubic"

    results["adopted_arm"] = adopted
    idata.to_netcdf(paths.RESEARCH_OUTPUT_DIR / "trace_fg_weather.nc")
    print(f"\nAdopted arm: {adopted}")

    out = paths.RESEARCH_OUTPUT_DIR / "fg_weather_summary.json"
    with out.open("w") as handle:
        json.dump(results, handle, indent=2, default=str)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
