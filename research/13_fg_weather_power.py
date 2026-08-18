"""Phase 3, step 4 — power calculation for adding weather to the field-goal model.

Runs **before** `docs/research/05b-fg-model-foundations.md` §10 commits any
threshold for the weather round, per the process law document 04 established.

Document 05b's defect register calls the absence of weather **"open, and the
largest defect"** of the field-goal model: *"a windy 50-yarder is priced as a
calm one, so the simulator overstates the kicker's bad luck outdoors in December
and understates it in a dome."* This script asks whether the data can size that
effect before a gate is written around it.

It also fixes the **sanitize** rules, which have to exist before a fit and which
turn out to matter more than expected — the raw `wind` column contains a 71 mph
reading, and 3,404 of 10,731 attempts have no weather at all because they were
kicked indoors.

    uv run python research/13_fg_weather_power.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

# Reuses the Newton-Raphson logistic fitter the rematch power calculation
# already validated, rather than adding a second implementation that could drift.
_rematch = import_module("08_rematch_power")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.ingest import PBP_SEASONS, load_pbp  # noqa: E402

RANDOM_SEED = 20260817
DATASETS = 400
DISTANCE_CENTRE = 40.0

# Published in document 05b §9, so using them here reveals nothing about the
# weather question. The null simulation needs a distance curve and a kicker
# spread; it takes both from the incumbent model rather than fitting new ones.
ALPHA = 1.898
BETA = -0.1148
GAMMA = 0.130
SIGMA_KICKER = 0.360

# --- sanitize rules, fixed here and reused by the fit -----------------------
#
# WIND_CAP: the raw column's maximum is 71 mph, on three attempts in one 2016
# game. That is not a credible sustained wind for a game that was played, and
# the bins above 25 mph hold 31 attempts in total — far too few to estimate a
# slope from. Leaving it uncapped would let three kicks lever the wind
# coefficient for the whole league. The cap is a data-quality guard, not a
# modelling choice, and the count of attempts it moves is reported.
WIND_CAP = 30.0

# Indoors there is no weather to record, and nflverse leaves temp/wind null on
# 3,196 of 3,200 dome and closed-roof attempts. The four exceptions carry a
# stadium-ambient reading (46 F, 2 mph) that leaked through; they are nulled, so
# "indoors" means one thing everywhere. This mirrors the sanitize_temp reasoning
# in the sibling baseball repo: a reading that cannot physically apply to the
# event is worse than a missing one, because it is silently used.
INDOOR_ROOFS = ("dome", "closed")

# Make-rate drops at 45 yards, between a calm day and a 15 mph wind, that the
# power table is computed against.
EFFECT_SCENARIOS_PP = (1.0, 2.0, 4.0, 6.0)

FG_COLUMNS = [
    "season",
    "game_id",
    "play_type",
    "kick_distance",
    "field_goal_result",
    "kicker_player_id",
    "roof",
    "temp",
    "wind",
]


# --------------------------------------------------------------------------
# data and sanitize
# --------------------------------------------------------------------------


def load_attempts(exclude_blocked: bool = False) -> tuple[pl.DataFrame, dict]:
    """Field-goal attempts with sanitized weather, plus a report of what moved.

    ``exclude_blocked`` was added by document 27 §7b, which re-derives this
    script's null bounds on the population the refit is fitted to. It defaults
    to ``False``, so the published `13_fg_weather_power.json` is reproduced
    exactly; there is one implementation of the sanitize rules and of the design
    matrix, and the two rounds cannot drift apart.
    """
    pbp = load_pbp(PBP_SEASONS, columns=FG_COLUMNS)
    attempt_mask = (pl.col("play_type") == "field_goal") & pl.col("kick_distance").is_not_null()
    if exclude_blocked:
        attempt_mask = attempt_mask & (pl.col("field_goal_result") != "blocked")
    attempts = (
        pbp.filter(attempt_mask)
        .drop_nulls("kicker_player_id")
        .select(
            pl.col("season"),
            pl.col("kicker_player_id"),
            pl.col("kick_distance").cast(pl.Float64).alias("distance"),
            (pl.col("field_goal_result") == "made").cast(pl.Int64).alias("made"),
            pl.col("roof"),
            pl.col("temp").cast(pl.Float64),
            pl.col("wind").cast(pl.Float64),
            pl.concat_str(
                [pl.col("season").cast(pl.String), pl.col("kicker_player_id")], separator="_"
            ).alias("kicker_season"),
        )
    )

    indoors = pl.col("roof").is_in(INDOOR_ROOFS)
    leaked = attempts.filter(
        indoors & (pl.col("wind").is_not_null() | pl.col("temp").is_not_null())
    )
    capped = attempts.filter(~indoors & (pl.col("wind") > WIND_CAP))

    sanitized = attempts.with_columns(
        pl.when(indoors).then(None).otherwise(pl.col("temp")).alias("temp"),
        pl.when(indoors)
        .then(None)
        .otherwise(pl.col("wind").clip(upper_bound=WIND_CAP))
        .alias("wind"),
    ).with_columns(
        (pl.col("wind").is_not_null() & pl.col("temp").is_not_null()).alias("has_weather")
    )

    report = {
        "attempts": int(sanitized.height),
        "indoor_readings_nulled": int(leaked.height),
        "wind_readings_capped": int(capped.height),
        "wind_cap": WIND_CAP,
        "max_raw_wind": float(attempts["wind"].max() or 0.0),
        "roof_counts": sanitized["roof"].value_counts().sort("count", descending=True).to_dicts(),
        "outdoor_without_weather": int(
            sanitized.filter(~pl.col("roof").is_in(INDOOR_ROOFS) & ~pl.col("has_weather")).height
        ),
        "mean_outdoor_wind": float(sanitized.filter(pl.col("has_weather"))["wind"].mean()),
        "mean_outdoor_temp": float(sanitized.filter(pl.col("has_weather"))["temp"].mean()),
    }
    return sanitized, report


def design_matrix(attempts: pl.DataFrame, report: dict) -> tuple[np.ndarray, list[str]]:
    """Distance curve, roof levels, and weather terms that are structurally zero indoors.

    Weather is **not imputed**. Indoors there is no wind, so the wind term is
    multiplied by an indicator and contributes exactly zero — a structural zero,
    not a guess. Outdoor attempts whose reading is missing (512 of them, 7% of
    outdoor kicks) are centred to the outdoor mean, which contributes zero for
    the same arithmetic reason and carries the honest meaning: *no information
    about this kick's conditions, so use the outdoor baseline.*
    """
    centred = attempts["distance"].to_numpy() - DISTANCE_CENTRE
    has_weather = attempts["has_weather"].to_numpy().astype(float)
    wind = np.nan_to_num(attempts["wind"].to_numpy().astype(float)) - report["mean_outdoor_wind"]
    temp = np.nan_to_num(attempts["temp"].to_numpy().astype(float)) - report["mean_outdoor_temp"]

    roof = attempts["roof"].to_numpy()
    columns = [
        np.ones(len(centred)),
        centred,
        centred**2 / 100.0,
        (roof == "dome").astype(float),
        (roof == "closed").astype(float),
        (roof == "open").astype(float),
        wind * has_weather,
        temp * has_weather,
    ]
    names = [
        "alpha",
        "beta_distance",
        "gamma_distance_sq",
        "roof_dome",
        "roof_closed",
        "roof_open",
        "beta_wind",
        "beta_temp",
    ]
    return np.column_stack(columns), names


# --------------------------------------------------------------------------
# the instrument
# --------------------------------------------------------------------------


def fit_and_intervals(x: np.ndarray, y: np.ndarray, indices: dict[str, int]) -> dict:
    """89% intervals for several coefficients from one fit.

    Gate W-7's pre-registered reporting rule requires a temperature null bound
    "built the same way Gate W-3's was", so the null simulation records both
    coefficients rather than only wind. Both are generated with a true value of
    zero, so neither bound is informed by the other.
    """
    beta = _rematch.fit_logistic(x, y)
    p = 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -30, 30)))
    w = np.clip(p * (1.0 - p), 1e-9, None)
    covariance = np.linalg.inv(x.T @ (x * w[:, None]) + 1e-9 * np.eye(x.shape[1]))
    z = 1.5982  # 89% equal-tailed, the project's convention
    return {
        name: (
            float(beta[i]),
            float(beta[i] - z * np.sqrt(covariance[i, i])),
            float(beta[i] + z * np.sqrt(covariance[i, i])),
        )
        for name, i in indices.items()
    }


def fit_and_interval(x: np.ndarray, y: np.ndarray, index: int) -> tuple[float, float, float]:
    """Coefficient, and its 89% equal-tailed interval from the observed information.

    A plain logistic fit, **without** kicker effects, is used here while the
    simulation generates data **with** them. That mismatch is deliberate and its
    direction is the safe one: unmodelled kicker spread inflates the residual, so
    the intervals this produces are wider than the real hierarchy's and the power
    reported below is at or under the truth. Document 05b §7 recorded the same
    trade for Gate FG-3, in the opposite direction, and said so.
    """
    beta = _rematch.fit_logistic(x, y)
    p = 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -30, 30)))
    w = np.clip(p * (1.0 - p), 1e-9, None)
    covariance = np.linalg.inv(x.T @ (x * w[:, None]) + 1e-9 * np.eye(x.shape[1]))
    se = float(np.sqrt(covariance[index, index]))
    # 89% equal-tailed, matching the project's convention throughout.
    z = 1.5982
    return float(beta[index]), float(beta[index] - z * se), float(beta[index] + z * se)


def simulate(
    x: np.ndarray,
    kicker_idx: np.ndarray,
    n_kickers: int,
    beta_wind: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One synthetic season set under a known true wind coefficient.

    Everything except the wind term is taken from the incumbent published model:
    document 05b's distance curve and its `sigma_kicker` of 0.360. Roof effects
    are set to zero under the null, which is a simplification — the wind
    coefficient's precision is driven by the spread of wind values across the
    6,815 outdoor attempts, not by roof levels, so this does not move the answer.
    """
    centred = x[:, 1]
    kicker = rng.normal(0.0, SIGMA_KICKER, n_kickers)[kicker_idx]
    eta = ALPHA + BETA * centred + GAMMA * centred**2 / 100.0 + kicker + beta_wind * x[:, 6]
    return (rng.random(len(eta)) < 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))).astype(float)


def beta_for_drop(drop_pp: float) -> float:
    """Wind coefficient producing a `drop_pp` fall at 45 yards between 0 and 15 mph."""
    centred = 45.0 - DISTANCE_CENTRE
    logit = ALPHA + BETA * centred + GAMMA * centred**2 / 100.0
    p = 1.0 / (1.0 + np.exp(-logit))
    target = p - drop_pp / 100.0
    return float((np.log(target / (1.0 - target)) - logit) / 15.0)


def main() -> None:
    paths.ensure_data_dirs()
    attempts, report = load_attempts()

    print("=== Sanitize report ===")
    for key, value in report.items():
        if key != "roof_counts":
            print(f"  {key:28s} {value}")
    print(f"  roof_counts                  {report['roof_counts']}")

    x, names = design_matrix(attempts, report)
    kicker_levels = sorted(attempts["kicker_season"].unique().to_list())
    lookup = {level: i for i, level in enumerate(kicker_levels)}
    kicker_idx = np.array([lookup[v] for v in attempts["kicker_season"].to_list()])
    wind_index = names.index("beta_wind")

    outdoor = attempts.filter(pl.col("has_weather"))
    print(
        f"\n{attempts.height:,} attempts, {len(kicker_levels)} kicker-seasons, "
        f"{outdoor.height:,} with usable weather "
        f"(wind mean {report['mean_outdoor_wind']:.2f} mph, "
        f"sd {outdoor['wind'].std():.2f}; temp mean {report['mean_outdoor_temp']:.1f} F)"
    )

    # ---- null: what does a true zero produce? ----------------------------
    print(f"\n=== Null: true wind and temperature effects = 0 ({DATASETS} datasets) ===")
    indices = {"wind": wind_index, "temp": names.index("beta_temp")}
    null_upper = np.empty(DATASETS)
    null_temp_lower = np.empty(DATASETS)
    for i in range(DATASETS):
        rng = np.random.default_rng(RANDOM_SEED + i)
        y = simulate(x, kicker_idx, len(kicker_levels), 0.0, rng)
        fitted = fit_and_intervals(x, y, indices)
        null_upper[i] = fitted["wind"][2]
        null_temp_lower[i] = fitted["temp"][1]
    threshold = float(np.percentile(null_upper, 10))
    # Wind is expected negative, so its gate is on the upper bound. Temperature
    # has no expected sign, so the mirror bound is recorded too: the 90th
    # percentile of the 89% LOWER bound, which a positive effect must clear.
    temp_threshold = float(np.percentile(null_temp_lower, 90))
    print(
        f"  89% upper bound on beta_wind under a true zero: "
        f"mean {null_upper.mean():+.5f}, 10th pct {threshold:+.5f}"
    )
    print(
        f"  89% lower bound on beta_temp under a true zero: "
        f"mean {null_temp_lower.mean():+.6f}, 90th pct {temp_threshold:+.6f}"
    )
    print(
        "  Each gate threshold is that percentile, so a true-zero design clears it\n"
        "  10% of the time by construction — the same construction Gate FG-3 used."
    )

    # ---- power ------------------------------------------------------------
    print(f"\n=== Power to clear the threshold ({DATASETS} datasets per scenario) ===")
    rows = []
    for drop_pp in EFFECT_SCENARIOS_PP:
        true_beta = beta_for_drop(drop_pp)
        uppers = np.empty(DATASETS)
        for i in range(DATASETS):
            rng = np.random.default_rng(RANDOM_SEED + 5000 + int(drop_pp * 10) * 911 + i)
            y = simulate(x, kicker_idx, len(kicker_levels), true_beta, rng)
            _, _, uppers[i] = fit_and_interval(x, y, wind_index)
        power = float((uppers < threshold).mean())
        rows.append(
            {
                "make_rate_drop_pp_at_15mph": drop_pp,
                "true_beta_wind": true_beta,
                "mean_upper_bound": float(uppers.mean()),
                "power": power,
            }
        )
        print(f"  {drop_pp:.0f} pp drop at 15 mph (beta_wind {true_beta:+.5f}): power {power:.3f}")

    results = {
        "sanitize": report,
        "datasets": DATASETS,
        "coefficient_names": names,
        "gate_w3_threshold": threshold,
        "gate_w7_temp_threshold": temp_threshold,
        "null_mean_upper_bound": float(null_upper.mean()),
        "null_mean_temp_lower_bound": float(null_temp_lower.mean()),
        "power": rows,
        "published_incumbent": {
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
            "sigma_kicker": SIGMA_KICKER,
        },
        "random_seed": RANDOM_SEED,
    }
    out = paths.RESEARCH_OUTPUT_DIR / "13_fg_weather_power.json"
    with out.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
