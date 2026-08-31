"""Round 24, step 1 — power and null bound for adding elevation to the FG model.

Runs **before** `docs/research/66-fg-elevation-prereg.md` commits any threshold,
per the process law document 04 established and document 05b §6 followed for
weather. Nothing here touches a real outcome: every fit in this file is on
simulated `y` generated from the incumbent posterior, so the gate this produces
cannot have been informed by the answer.

The incumbent is the document 27 refit — the posterior the product actually
reads (`trace_fg_refit.nc`, cubic arm, blocked kicks excluded). Its published
means are hard-coded below rather than read from the trace, so the threshold is
reproducible from the committed record alone.

It also fixes the **covariate construction** — elevation in thousands of feet,
centred on the kick-weighted mean — and exports it, so the fit in
`research/81_fg_elevation.py` cannot build a different column than the one the
power was computed for.

    uv run python research/81a_fg_elevation_power.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

# One loader with the fit and the incumbent refit, per document 27's rule.
_weather = import_module("14_fg_weather_model")
# The Newton-Raphson logistic fitter the rematch power calculation validated.
_rematch = import_module("08_rematch_power")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.data.stadium_elevation import elevation_kft  # noqa: E402

RANDOM_SEED = 20260831
DATASETS = 400
DISTANCE_CENTRE = _weather.DISTANCE_CENTRE
ROOF_LEVELS = _weather.ROOF_LEVELS

# --- incumbent, document 27 §14 / research/outputs/fg_refit_summary.json ----
# Posterior means of the adopted cubic arm. Published, so using them here
# reveals nothing about the elevation question: none of them was fitted with an
# elevation term in the model.
ALPHA = 1.906813
BETA = -0.115874
GAMMA = 0.248885
DELTA_CUBIC = -0.081130
SIGMA_KICKER = 0.385485
BETA_WIND = -0.022411
BETA_TEMP = 0.003407
DELTA_XP = 0.122232
ROOF_EFFECTS = {"dome": 0.245669, "closed": 0.250076, "open": 0.460447}
WIND_CENTRE = 8.021924
TEMP_CENTRE = 57.989814

DENVER_KFT = 5.280

# Make-rate gains at 45 yards, between a stadium at the league's mean elevation
# and Denver, that the power table is computed against.
EFFECT_SCENARIOS_PP = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

Z89 = 1.5982  # 89% equal-tailed, the project's convention


# --------------------------------------------------------------------------
# data and the covariate
# --------------------------------------------------------------------------


def load_elevation_kicks() -> pl.DataFrame:
    """The document 27 population, with the elevation of each kick attached.

    `elevation_kft` raises on an unknown stadium, so a season that opened a
    stadium nobody entered fails loudly here rather than pricing its kicks at
    sea level.
    """
    kicks = _weather.load_kicks(exclude_blocked=True, with_stadium=True)
    return kicks.with_columns(
        pl.Series(
            "elev_kft",
            [elevation_kft(s) for s in kicks["stadium_id"].to_list()],
            dtype=pl.Float64,
        )
    )


def elevation_centre(kicks: pl.DataFrame) -> float:
    """Kick-weighted mean elevation, in thousands of feet.

    Centring on the *kicks* rather than on the 42 stadiums keeps `alpha`'s
    meaning — the log-odds for an average kick — instead of re-anchoring it on a
    stadium average that a 12-kick site in Sao Paulo would drag upward.
    """
    return float(kicks["elev_kft"].mean())


def design_matrix(kicks: pl.DataFrame, elev_centre: float) -> tuple[np.ndarray, list[str]]:
    """The incumbent's design plus one elevation column.

    Cubic, because the cubic arm is what document 27 adopted and what the
    product reads. The elevation column is centred like wind and temperature so
    that adding it does not move the intercept.
    """
    centred = kicks["distance"].to_numpy() - DISTANCE_CENTRE
    has_weather = kicks["has_weather"].to_numpy().astype(float)
    wind = np.nan_to_num(kicks["wind"].to_numpy().astype(float)) - WIND_CENTRE
    temp = np.nan_to_num(kicks["temp"].to_numpy().astype(float)) - TEMP_CENTRE
    roof = kicks["roof"].to_numpy()
    is_xp = kicks["is_xp"].to_numpy().astype(float)
    elev = kicks["elev_kft"].to_numpy() - elev_centre

    columns = [
        np.ones(len(centred)),
        centred,
        centred**2 / 100.0,
        centred**3 / 1000.0,
        *[(roof == level).astype(float) for level in ROOF_LEVELS],
        wind * has_weather,
        temp * has_weather,
        is_xp,
        elev,
    ]
    names = [
        "alpha",
        "beta_distance",
        "gamma_distance_sq",
        "delta_cubic",
        *[f"roof_{level}" for level in ROOF_LEVELS],
        "beta_wind",
        "beta_temp",
        "delta_xp",
        "beta_elev",
    ]
    return np.column_stack(columns), names


# --------------------------------------------------------------------------
# the instrument
# --------------------------------------------------------------------------


def fit_and_interval(x: np.ndarray, y: np.ndarray, index: int) -> tuple[float, float, float]:
    """Coefficient and its 89% equal-tailed interval from the observed information.

    A plain logistic **without** kicker effects, fitted to data simulated
    **with** them. The mismatch is deliberate and its direction is the safe one:
    unmodelled kicker spread inflates the residual, so these intervals are wider
    than the real hierarchy's and the power below is at or under the truth.
    Document 05b §6 recorded the same trade for the wind coefficient, and said so.
    """
    beta = _rematch.fit_logistic(x, y)
    p = 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -30, 30)))
    w = np.clip(p * (1.0 - p), 1e-9, None)
    covariance = np.linalg.inv(x.T @ (x * w[:, None]) + 1e-9 * np.eye(x.shape[1]))
    se = float(np.sqrt(covariance[index, index]))
    return float(beta[index]), float(beta[index] - Z89 * se), float(beta[index] + Z89 * se)


def logit_at_45() -> float:
    """League log-odds at 45 yards, at the centre elevation, on the incumbent curve."""
    centred = 45.0 - DISTANCE_CENTRE
    return ALPHA + BETA * centred + GAMMA * centred**2 / 100.0 + DELTA_CUBIC * centred**3 / 1000.0


def beta_for_gain(gain_pp: float, elev_centre: float) -> float:
    """Elevation coefficient producing `gain_pp` at 45 yards, mean elevation -> Denver."""
    logit = logit_at_45()
    p = 1.0 / (1.0 + np.exp(-logit))
    target = p + gain_pp / 100.0
    return float((np.log(target / (1.0 - target)) - logit) / (DENVER_KFT - elev_centre))


def gain_for_beta(beta_elev: float, elev_centre: float) -> float:
    """The inverse of `beta_for_gain`, in percentage points."""
    logit = logit_at_45()
    base = 1.0 / (1.0 + np.exp(-logit))
    denver = 1.0 / (1.0 + np.exp(-(logit + beta_elev * (DENVER_KFT - elev_centre))))
    return float((denver - base) * 100.0)


def simulate(
    x: np.ndarray,
    kicker_idx: np.ndarray,
    n_kickers: int,
    beta_elev: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One synthetic set of kick outcomes under a known true elevation coefficient.

    Everything except the elevation term comes from the incumbent posterior
    means, roof levels and the extra-point offset included — unlike document
    05b §6's null, which zeroed the roofs. Here they are kept, because elevation
    and roof are *correlated* in this design: Allegiant is a dome at 2,030 feet
    and the two Atlanta stadiums and Glendale sit near 1,050. Simulating with the
    roof effects on is the faithful version of the collinearity the fit will face.
    """
    kicker = rng.normal(0.0, SIGMA_KICKER, n_kickers)[kicker_idx]
    truth = np.array(
        [
            ALPHA,
            BETA,
            GAMMA,
            DELTA_CUBIC,
            *[ROOF_EFFECTS[level] for level in ROOF_LEVELS],
            BETA_WIND,
            BETA_TEMP,
            DELTA_XP,
            beta_elev,
        ]
    )
    eta = x @ truth + kicker
    return (rng.random(len(eta)) < 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))).astype(float)


def main() -> None:
    paths.ensure_data_dirs()
    kicks = load_elevation_kicks()
    elev_centre = elevation_centre(kicks)
    x, names = design_matrix(kicks, elev_centre)
    elev_index = names.index("beta_elev")

    kicker_levels = sorted(kicks["kicker_season"].unique().to_list())
    lookup = {level: i for i, level in enumerate(kicker_levels)}
    kicker_idx = np.array([lookup[v] for v in kicks["kicker_season"].to_list()])

    by_stadium = (
        kicks.group_by("stadium_id")
        .agg(pl.len().alias("kicks"), pl.col("elev_kft").first())
        .sort("elev_kft", descending=True)
    )
    print(
        f"{kicks.height:,} kicks ({int(kicks.height - kicks['is_xp'].sum()):,} field goals, "
        f"{int(kicks['is_xp'].sum()):,} extra points), {len(kicker_levels)} kicker-seasons, "
        f"{by_stadium.height} stadiums"
    )
    print(
        f"elevation: kick-weighted mean {elev_centre:.4f} kft, "
        f"sd {kicks['elev_kft'].std():.4f} kft, "
        f"max {kicks['elev_kft'].max():.3f} kft"
    )
    print("\n=== Kicks above 1,500 ft (the leverage in the covariate) ===")
    with pl.Config(tbl_rows=12):
        print(by_stadium.filter(pl.col("elev_kft") >= 1.5))
    above_3k = int(kicks.filter(pl.col("elev_kft") >= 3.0).height)
    print(
        f"  kicks at or above 3,000 ft: {above_3k:,} "
        f"({above_3k / kicks.height * 100:.2f}% of the population)"
    )

    # ---- null: what does a true zero produce? ----------------------------
    print(f"\n=== Null: true elevation effect = 0 ({DATASETS} datasets) ===")
    null_lower = np.empty(DATASETS)
    null_point = np.empty(DATASETS)
    for i in range(DATASETS):
        rng = np.random.default_rng(RANDOM_SEED + i)
        y = simulate(x, kicker_idx, len(kicker_levels), 0.0, rng)
        point, lower, _upper = fit_and_interval(x, y, elev_index)
        null_lower[i] = lower
        null_point[i] = point
    threshold = float(np.percentile(null_lower, 90))
    print(
        f"  89% lower bound on beta_elev under a true zero: "
        f"mean {null_lower.mean():+.5f}, 90th pct {threshold:+.5f}"
    )
    print(
        f"  point estimate under a true zero: mean {null_point.mean():+.5f}, "
        f"sd {null_point.std():.5f}"
    )
    print(
        "  The gate threshold is that percentile, so a true-zero design clears it\n"
        "  10% of the time by construction — the same construction Gate W-3 used,\n"
        "  mirrored for a coefficient whose expected sign is positive."
    )
    print(
        f"  Threshold in readable units: a Denver gain at 45 yd of "
        f"{gain_for_beta(threshold, elev_centre):+.2f} pp"
    )

    # ---- power ------------------------------------------------------------
    print(f"\n=== Power ({DATASETS} datasets per scenario) ===")
    rows = []
    for gain_pp in EFFECT_SCENARIOS_PP:
        beta_elev = beta_for_gain(gain_pp, elev_centre)
        passes = 0
        for i in range(DATASETS):
            rng = np.random.default_rng(RANDOM_SEED + 100_000 + int(gain_pp * 10) * 1000 + i)
            y = simulate(x, kicker_idx, len(kicker_levels), beta_elev, rng)
            _point, lower, _upper = fit_and_interval(x, y, elev_index)
            passes += int(lower > threshold)
        power = passes / DATASETS
        rows.append(
            {
                "denver_gain_pp_at_45yd": gain_pp,
                "beta_elev": beta_elev,
                "power": power,
            }
        )
        print(f"  {gain_pp:>4.1f} pp   beta_elev {beta_elev:+.5f}   power {power:.3f}")

    detectable = [r for r in rows if r["power"] >= 0.8]
    minimum = detectable[0]["denver_gain_pp_at_45yd"] if detectable else None
    if minimum is None:
        print(
            f"\n  No scenario in {EFFECT_SCENARIOS_PP} reaches power 0.8. The design cannot\n"
            "  resolve an elevation effect of the size the literature suggests, and that\n"
            "  is the finding, recorded before the fit rather than after it."
        )
    else:
        print(f"\n  Minimum detectable Denver gain at 45 yards, at power 0.8: {minimum:.1f} pp")

    out = {
        "random_seed": RANDOM_SEED,
        "datasets": DATASETS,
        "n_kicks": int(kicks.height),
        "n_field_goals": int(kicks.height - kicks["is_xp"].sum()),
        "n_extra_points": int(kicks["is_xp"].sum()),
        "n_kicker_seasons": len(kicker_levels),
        "n_stadiums": int(by_stadium.height),
        "elevation_centre_kft": elev_centre,
        "elevation_sd_kft": float(kicks["elev_kft"].std()),
        "kicks_above_3000ft": above_3k,
        "kicks_by_stadium_above_1500ft": by_stadium.filter(pl.col("elev_kft") >= 1.5).to_dicts(),
        "gate_e3_threshold": threshold,
        "gate_e3_threshold_as_denver_gain_pp": gain_for_beta(threshold, elev_centre),
        "null_lower_bound_mean": float(null_lower.mean()),
        "null_point_sd": float(null_point.std()),
        "power_table": rows,
        "minimum_detectable_denver_gain_pp": minimum,
        "incumbent": {
            "source": "research/outputs/fg_refit_summary.json, arm2_cubic (document 27)",
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
            "delta_cubic": DELTA_CUBIC,
            "sigma_kicker": SIGMA_KICKER,
        },
    }
    path = paths.RESEARCH_OUTPUT_DIR / "81a_fg_elevation_power.json"
    with path.open("w") as handle:
        json.dump(out, handle, indent=2)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
