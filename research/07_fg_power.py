"""Step 4 — power checks for the field-goal model's pre-registered gates.

Runs before `docs/research/05b-fg-model-foundations.md` commits any threshold.

Two questions, because the FG model has two things that can go wrong and they
need different instruments:

    FG-2  if the distance curve is misspecified, would the calibration gate
          notice? (simulate a curved truth, fit the linear model, check)
    FG-3  can this many kicks resolve kicker skill at all, or would a null
          result just mean "not enough kicks"?

    uv run python research/07_fg_power.py
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from nfl_simulator import paths
from nfl_simulator.ingest import PBP_SEASONS, load_pbp

RANDOM_SEED = 20260817
DATASETS = 400

FG_COLUMNS = [
    "season",
    "play_type",
    "kick_distance",
    "field_goal_result",
    "kicker_player_id",
]

# Distance is centred here so the intercept means "log-odds of a 40-yarder"
# rather than "log-odds of a 0-yard kick", which does not exist.
DISTANCE_CENTRE = 40.0

# FG-2's alternative: a curvature large enough to move the make rate at 55 yards
# by this much. Chosen as a football-meaningful miss, not read off a fit.
CURVATURE_SHIFTS = [0.05, 0.10, 0.15]
CURVATURE_AT_DISTANCE = 55.0

# FG-3's alternatives, on the log-odds scale. sigma = 0.3 corresponds to roughly
# a 5-point make-rate gap at 45 yards between a one-SD-good and average kicker,
# which is about the size the public kicking literature reports.
SIGMA_ALTERNATIVES = [0.10, 0.20, 0.30, 0.40]


def load_attempts() -> pl.DataFrame:
    """Every field-goal attempt with a distance and a kicker.

    Blocked kicks count as misses, matching `components.py`. A block is partly a
    protection failure rather than a kicking outcome, but at 1.8% of attempts
    splitting it would add a class without changing a conclusion.
    """
    pbp = load_pbp(PBP_SEASONS, columns=FG_COLUMNS)
    return (
        pbp.filter((pl.col("play_type") == "field_goal") & pl.col("kick_distance").is_not_null())
        .drop_nulls("kicker_player_id")
        .select(
            pl.col("season"),
            pl.col("kick_distance").cast(pl.Float64).alias("distance"),
            (pl.col("field_goal_result") == "made").cast(pl.Int64).alias("made"),
            pl.concat_str(
                [pl.col("season").cast(pl.String), pl.col("kicker_player_id")], separator="_"
            ).alias("kicker_season"),
        )
    )


def fit_logistic(x: np.ndarray, y: np.ndarray, iterations: int = 50) -> np.ndarray:
    beta = np.zeros(x.shape[1])
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -30, 30)))
        w = np.clip(p * (1.0 - p), 1e-9, None)
        hessian = x.T @ (x * w[:, None]) + 1e-6 * np.eye(x.shape[1])
        step = np.linalg.solve(hessian, x.T @ (y - p))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def predict(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -30, 30)))


# --------------------------------------------------------------------------
# FG-2 — would a misspecified distance curve be caught?
# --------------------------------------------------------------------------


MIN_BIN_ATTEMPTS = 100
CALIBRATION_REPLICATES = 400


def _bin_masks(distance: np.ndarray) -> list[np.ndarray]:
    """Well-populated 5-yard distance bins.

    Bins under `MIN_BIN_ATTEMPTS` are skipped: their intervals are so wide the
    check would pass regardless, which is a gate that cannot fail rather than a
    gate that passes.
    """
    bins = (distance // 5 * 5).astype(int)
    masks = [bins == edge for edge in np.unique(bins)]
    return [m for m in masks if m.sum() >= MIN_BIN_ATTEMPTS]


def calibration_statistic(masks: list[np.ndarray], made: np.ndarray, p_hat: np.ndarray) -> float:
    """Largest standardized bin miss — one omnibus number, not a per-bin verdict.

    An earlier version of this gate required *every* bin to sit inside its own
    89% interval. Its power check said it failed 36% of the time on a correctly
    specified model, which is multiplicity: eight bins at nominal 89% coverage
    pass together only 0.89^8 = 39% of the time. Reducing the bins to a single
    maximum, and calibrating that maximum against its own reference
    distribution, prices the multiplicity in exactly once.
    """
    worst = 0.0
    for mask in masks:
        expected = p_hat[mask]
        sd = np.sqrt((expected * (1.0 - expected)).sum()) / mask.sum()
        if sd <= 0:
            continue
        worst = max(worst, abs(made[mask].mean() - expected.mean()) / sd)
    return worst


def calibration_gate(
    distance: np.ndarray, made: np.ndarray, p_hat: np.ndarray, rng: np.random.Generator
) -> bool:
    """True when the worst bin miss is no worse than the model itself produces."""
    masks = _bin_masks(distance)
    observed = calibration_statistic(masks, made, p_hat)
    replicated = np.array(
        [
            calibration_statistic(masks, rng.binomial(1, p_hat).astype(float), p_hat)
            for _ in range(CALIBRATION_REPLICATES)
        ]
    )
    return bool(observed <= np.percentile(replicated, 94.5))


def fg2_power(attempts: pl.DataFrame) -> dict:
    """Does the calibration gate catch a curved truth fitted with a linear model?"""
    print(f"\n{'=' * 72}\nFG-2 — power of the distance-calibration gate\n{'=' * 72}")
    distance = attempts["distance"].to_numpy()
    centred = distance - DISTANCE_CENTRE
    design_linear = np.column_stack([np.ones(len(distance)), centred])

    # A truth with curvature, calibrated so the make rate at 55 yards is shifted
    # by CURVATURE_TARGET_SHIFT relative to the linear model through the same
    # centre and slope.
    beta_linear = fit_logistic(design_linear, attempts["made"].to_numpy().astype(float))
    offset = CURVATURE_AT_DISTANCE - DISTANCE_CENTRE
    p_at = predict(np.array([[1.0, offset]]), beta_linear)[0]
    print(f"  linear make rate at {CURVATURE_AT_DISTANCE:.0f} yd: {p_at:.3f}")

    rng = np.random.default_rng(RANDOM_SEED)
    scenarios = [("well-specified", 0.0, 0.0)]
    for shift in CURVATURE_SHIFTS:
        target = np.clip(p_at - shift, 1e-4, 1 - 1e-4)
        quad = (np.log(target / (1 - target)) - np.log(p_at / (1 - p_at))) / offset**2
        scenarios.append((f"{shift:.0%} miss at {CURVATURE_AT_DISTANCE:.0f} yd", shift, quad))

    rows = []
    for label, shift, quad in scenarios:
        eta = design_linear @ beta_linear + quad * centred**2
        p_true = 1.0 / (1.0 + np.exp(-eta))
        outcomes = []
        for _ in range(DATASETS // 4):
            y = rng.binomial(1, p_true).astype(float)
            beta = fit_logistic(design_linear, y)
            outcomes.append(calibration_gate(distance, y, predict(design_linear, beta), rng))
        pass_rate = float(np.mean(outcomes))
        rows.append(
            {
                "scenario": label,
                "shift_at_55yd": shift,
                "curvature": float(quad),
                "gate_pass_rate": pass_rate,
                "power_to_catch": 1.0 - pass_rate,
            }
        )
        print(f"  {label:22s}: gate passes {pass_rate:.3f}, catches {1 - pass_rate:.3f}")

    false_alarm = 1.0 - rows[0]["gate_pass_rate"]
    print(f"\n  --> false-alarm rate on a well-specified model: {false_alarm:.3f} (nominal 0.055)")
    return {"false_alarm": false_alarm, "scenarios": rows}


# --------------------------------------------------------------------------
# FG-3 — can this many kicks resolve kicker skill?
# --------------------------------------------------------------------------


def kicker_information(attempts: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fisher information for each kicker-season's random intercept.

    For a logistic random intercept the information a kicker's attempts carry
    about their own effect is ``sum p(1-p)`` over those attempts, evaluated at
    the league distance curve. Two kickers with the same attempt count differ if
    one kicks more chip shots — a 20-yarder at p = 0.99 carries almost no
    information about anything.
    """
    distance = attempts["distance"].to_numpy()
    design = np.column_stack([np.ones(len(distance)), distance - DISTANCE_CENTRE])
    beta = fit_logistic(design, attempts["made"].to_numpy().astype(float))
    p = predict(design, beta)

    frame = attempts.with_columns(pl.Series("info", p * (1.0 - p)))
    grouped = frame.group_by("kicker_season").agg(
        pl.col("info").sum().alias("information"), pl.len().alias("attempts")
    )
    return grouped["information"].to_numpy(), grouped["attempts"].to_numpy()


def sigma_upper_bounds(
    information: np.ndarray, true_sigma: float, datasets: int, seed: int
) -> np.ndarray:
    """89% upper bounds on the kicker-effect SD, across simulated datasets.

    Uses the normal-normal reduction: a kicker's estimated effect is
    approximately Normal(true effect, 1/information), so the marginal likelihood
    for sigma is a product of Normal(0, sigma^2 + 1/information) terms. That is
    the standard random-effects meta-analysis form, and it makes hundreds of fits
    instant where fitting the full logistic hierarchy in PyMC would not be.

    It is an approximation — stated as one — and it is conservative in the right
    direction: it ignores the shrinkage the real hierarchy applies, so if
    anything it overstates how well the design resolves sigma.
    """
    rng = np.random.default_rng(seed)
    variance = 1.0 / information
    sigma_grid = np.linspace(1e-4, 1.5, 400)
    # HalfNormal(1), the prior the pre-registered model uses.
    log_prior = -0.5 * (sigma_grid / 1.0) ** 2

    bounds = np.empty(datasets)
    for i in range(datasets):
        effects = rng.normal(0.0, true_sigma, len(information)) if true_sigma > 0 else 0.0
        estimated = effects + rng.normal(0.0, np.sqrt(variance))

        total = sigma_grid[:, None] ** 2 + variance[None, :]
        log_lik = (-0.5 * (np.log(total) + estimated[None, :] ** 2 / total)).sum(axis=1)
        weights = np.exp(log_lik + log_prior - np.max(log_lik + log_prior))
        weights /= weights.sum()
        bounds[i] = sigma_grid[np.searchsorted(np.cumsum(weights), 0.945)]
    return bounds


def fg3_power(attempts: pl.DataFrame) -> dict:
    """Null and alternative distributions of the 89% upper bound on sigma."""
    print(f"\n{'=' * 72}\nFG-3 — power to resolve kicker skill\n{'=' * 72}")
    information, counts = kicker_information(attempts)
    print(
        f"  {len(information)} kicker-seasons, median {np.median(counts):.0f} attempts, "
        f"median information {np.median(information):.1f}"
    )

    null_bounds = sigma_upper_bounds(information, 0.0, DATASETS, RANDOM_SEED)
    achievable = float(np.percentile(null_bounds, 90))
    print(
        f"\n  under TRUE sigma = 0, the 89% upper bound lands at "
        f"{null_bounds.mean():.4f} on average, 90th pct {achievable:.4f}"
    )

    rows = []
    for offset, sigma in enumerate(SIGMA_ALTERNATIVES, start=1):
        bounds = sigma_upper_bounds(information, sigma, DATASETS, RANDOM_SEED + offset)
        power = float((bounds > achievable).mean())
        # Readable form: the make-rate gap this sigma implies at 45 yards.
        gap = _make_rate_gap(attempts, sigma)
        rows.append(
            {
                "true_sigma": sigma,
                "make_rate_gap_at_45yd_pp": gap * 100,
                "mean_upper_bound": float(bounds.mean()),
                "power": power,
            }
        )
        print(
            f"  sigma {sigma:.2f} (= {gap * 100:4.1f} pp gap at 45 yd) "
            f"-> mean upper bound {bounds.mean():.4f},  power {power:.3f}"
        )

    return {
        "kicker_seasons": int(len(information)),
        "median_attempts": float(np.median(counts)),
        "null_mean_upper_bound": float(null_bounds.mean()),
        "achievable_threshold": achievable,
        "alternatives": rows,
    }


def _make_rate_gap(attempts: pl.DataFrame, sigma: float) -> float:
    """Make-rate difference at 45 yards between a one-SD kicker and the average."""
    distance = attempts["distance"].to_numpy()
    design = np.column_stack([np.ones(len(distance)), distance - DISTANCE_CENTRE])
    beta = fit_logistic(design, attempts["made"].to_numpy().astype(float))
    baseline = predict(np.array([[1.0, 45.0 - DISTANCE_CENTRE]]), beta)[0]
    shifted = 1.0 / (1.0 + np.exp(-(np.log(baseline / (1 - baseline)) + sigma)))
    return float(shifted - baseline)


def main() -> None:
    paths.ensure_data_dirs()
    attempts = load_attempts()
    print(
        f"{attempts.height:,} field-goal attempts, "
        f"league make rate {attempts['made'].mean():.3%}, "
        f"{attempts['kicker_season'].n_unique()} kicker-seasons"
    )

    results = {
        "n_attempts": attempts.height,
        "league_make_rate": float(attempts["made"].mean()),
        "fg2_calibration": fg2_power(attempts),
        "fg3_kicker_skill": fg3_power(attempts),
    }

    out = paths.RESEARCH_OUTPUT_DIR / "07_fg_power.json"
    with out.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
