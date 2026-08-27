"""Part A of the dropped-interception confound study — power, before thresholds.

Runs **before** `docs/research/43-dropped-pick-confounds-prereg.md` §6 carries a
single number, per the process law documents 04 -> 05 §7 -> 09 §4 established.
Nothing here touches `src/nfl_simulator/`; document 32's closure stands whatever
the numbers say.

Five entity designs are powered, each with the instrument its gate will use:

    1. worthy rate, QB-season (>= 200 charted passes)   beta-binomial grid
    2. worthy rate, defence-season                      beta-binomial grid
    3. residual, defence-season x QB-season             crossed Gaussian grid
    4. residual, defence pooled x QB-season             crossed Gaussian grid
    5. residual, QB-season (sigma_q)                    design 3, reading sigma_b

Designs 1 and 2 simulate at the real denominators under a known population SD
and record the 89% upper bound, exactly as `research/12_coinflips_power.py` did.
Designs 3-5 need a fixed-effects prediction before the residual exists, so this
script fits document 43 §5's arm-2 model **once, for `beta` only**: the fitted
`alpha` and `beta` go to `research/outputs/61_beta_hat.json` and the `sigma_d` /
`sigma_q` posteriors are never read, printed or stored here. Part B re-fits the
same model and reports them. That ordering is what keeps the thresholds below
innocent of the result they will judge.

Direction of the test, which is the reverse of a skill hunt: a coin-like finish
is confirmed by showing the entity spread is **small**, so the gate reads "the
89% upper bound is below X". Power is the chance of correctly rejecting that
when a real effect exists, and a design with low power cannot report either
outcome as a finding (Gate C-3).

    uv run python research/61_dropped_pick_power.py
    uv run python research/61_dropped_pick_power.py --serial   # no worker pool

Parallel and serial runs produce identical numbers: every simulated dataset
draws from `np.random.default_rng` seeded by its own index.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_betabinom = import_module("_betabinom_grid")
_crossed = import_module("_crossed_gaussian_grid")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.ingest import FTN_SEASONS, load_ftn, load_pbp  # noqa: E402

# --- document 43 §10, constants ------------------------------------------------

RANDOM_SEED = 20260827
DATASETS = 400
RELATIVE_SCENARIOS = (0.05, 0.125, 0.25, 0.50)
REFERENCE_RELATIVE = 0.125
MIN_POWER = 0.80
NULL_PERCENTILE = 90

MIN_QB_ATTEMPTS = 200  # worthy-rate question, document 32 §3's floor
MIN_QB_WORTHY = 20  # residual question, document 43 §4

# Stop-and-ask guards. The study is a re-measurement of document 32 §3, so the
# frame it starts from has to be document 32's frame.
EXPECTED_WORTHY = 2997
WORTHY_TOLERANCE = 0.05
EXPECTED_PBAR = 0.485
PBAR_TOLERANCE_PP = 1.0
EXPECTED_DEFENCE_SEASONS = 128

# d/dx of the logistic at the league conversion rate: 0.485 * 0.515. Converts a
# logit-scale SD to the probability scale and back, and it is the same constant
# document 43 §5 uses for the arm-2 / arm-3 cross-check.
LOGIT_SLOPE = EXPECTED_PBAR * (1.0 - EXPECTED_PBAR)

PBP_COLUMNS = [
    "game_id",
    "play_id",
    "season",
    "week",
    "posteam",
    "defteam",
    "play_type",
    "interception",
    "passer_player_id",
    "air_yards",
    "pass_location",
    "qb_hit",
    "down",
    "ydstogo",
    "yardline_100",
    "shotgun",
    "wp",
]

FTN_COLUMNS = [
    "is_interception_worthy",
    "is_catchable_ball",
    "is_contested_ball",
    "is_qb_out_of_pocket",
    "is_play_action",
    "is_screen_pass",
    "n_pass_rushers",
]

# Standardised covariates, per document 43 §4. Everything else in X is an
# indicator and enters as a 0/1 dummy, so a Normal(0, 1) prior on its
# coefficient is already weakly informative on the logit scale.
STANDARDISED = ("air_yards", "n_pass_rushers", "ydstogo", "yardline_100", "wp")

# Excluded by rule (document 43 §4): anything recorded after the ball reaches
# the defender. Listed here so a reader can check the frame against the rule.
POST_BRANCH_EXCLUDED = (
    "is_drop",
    "is_created_reception",
    "complete_pass",
    "interception_player_id",
    "epa",
    "wpa",
)


# --------------------------------------------------------------------------
# the frame
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WorthyFrame:
    """Every frame the study needs, built once.

    ``charted`` is all 2022-2025 charted passes and supplies the worthy-rate
    denominators. ``worthy`` is the interception-worthy subset before
    complete-case filtering. ``model`` is the complete-case subset arm 2 fits,
    with its design matrix and level codes. ``residual`` is ``model`` further
    restricted to QB-seasons clearing ``MIN_QB_WORTHY``, which is the frame the
    gate arm's crossed grids see.
    """

    charted: pl.DataFrame
    worthy: pl.DataFrame
    model: pl.DataFrame
    design_matrix: np.ndarray
    feature_names: tuple[str, ...]
    outcome: np.ndarray
    defence_season_codes: np.ndarray
    qb_season_codes: np.ndarray
    n_defence_seasons: int
    n_qb_seasons: int
    guards: dict


def load_charted_passes() -> pl.DataFrame:
    """FTN charting joined to pbp, charted passes only.

    ``play_type == "pass"`` rather than ``pass_attempt == 1`` because that is
    the filter that reproduces document 32 §3's 80,785 charted passes and 2,997
    worthy throws exactly; ``pass_attempt`` adds 295 rows the memo did not have.
    """
    pbp = load_pbp(FTN_SEASONS, columns=PBP_COLUMNS)
    ftn = load_ftn(FTN_SEASONS)
    charted = ftn.select(
        pl.col("nflverse_game_id").alias("game_id"),
        pl.col("nflverse_play_id").cast(pl.Float64).alias("play_id"),
        *FTN_COLUMNS,
    ).join(pbp, on=["game_id", "play_id"], how="inner")
    return charted.filter(pl.col("play_type") == "pass")


def _codes(frame: pl.DataFrame, keys: list[str]) -> tuple[np.ndarray, int]:
    """Integer level codes for a grouping factor, and the level count."""
    labels = frame.select(pl.concat_str(keys, separator="|").alias("label"))["label"]
    unique = labels.unique(maintain_order=True).to_list()
    lookup = {label: index for index, label in enumerate(unique)}
    return np.array([lookup[label] for label in labels.to_list()], dtype=int), len(unique)


def design_matrix(frame: pl.DataFrame, reference: pl.DataFrame | None = None) -> tuple:
    """`X` and its column names, per document 43 §4's covariate list.

    Standardisation uses ``reference``'s mean and SD when one is given, so that a
    subset frame is measured on the same scale as the frame arm 2 was fitted on.
    ``air_yards`` enters as its standardised value and that value squared;
    ``pass_location`` is dummied against middle and ``down`` against first down.
    """
    scale_from = reference if reference is not None else frame
    columns: list[np.ndarray] = []
    names: list[str] = []

    for column in STANDARDISED:
        values = frame[column].cast(pl.Float64).to_numpy()
        mean = float(scale_from[column].cast(pl.Float64).mean())
        sd = float(scale_from[column].cast(pl.Float64).std())
        standardised = (values - mean) / sd
        columns.append(standardised)
        names.append(f"{column}_z")
        if column == "air_yards":
            columns.append(standardised**2)
            names.append("air_yards_z_squared")

    for location in ("left", "right"):  # middle is the reference level
        columns.append((frame["pass_location"] == location).to_numpy().astype(float))
        names.append(f"pass_location_{location}")

    for down in (2, 3, 4):  # first down is the reference level
        columns.append((frame["down"] == float(down)).to_numpy().astype(float))
        names.append(f"down_{down}")

    for column in (
        "is_contested_ball",
        "is_catchable_ball",
        "qb_hit",
        "is_qb_out_of_pocket",
        "is_play_action",
        "is_screen_pass",
        "shotgun",
    ):
        columns.append(frame[column].cast(pl.Float64).to_numpy())
        names.append(column)

    return np.column_stack(columns), tuple(names)


def build_worthy_frame(*, verbose: bool = True) -> WorthyFrame:
    """The study's frame, with document 43's guards checked before anything else."""
    charted = load_charted_passes()
    worthy = charted.filter(pl.col("is_interception_worthy"))

    n_worthy = worthy.height
    picked = int(worthy["interception"].sum())
    pbar = picked / n_worthy
    defence_seasons = worthy.select(pl.struct("season", "defteam").n_unique()).item()

    required = [*STANDARDISED, "pass_location", "down", "passer_player_id", "defteam"] + [
        "is_contested_ball",
        "is_catchable_ball",
        "qb_hit",
        "is_qb_out_of_pocket",
        "is_play_action",
        "is_screen_pass",
        "shotgun",
    ]
    model = worthy.drop_nulls(required)
    null_drops = n_worthy - model.height

    guards = {
        "charted_passes": int(charted.height),
        "worthy": int(n_worthy),
        "worthy_expected": EXPECTED_WORTHY,
        "picked": picked,
        "pbar": pbar,
        "pbar_expected": EXPECTED_PBAR,
        "defence_seasons": int(defence_seasons),
        "null_dropped_rows": int(null_drops),
        "model_rows": int(model.height),
    }

    worthy_off_by = abs(n_worthy - EXPECTED_WORTHY) / EXPECTED_WORTHY
    pbar_off_by_pp = abs(pbar - EXPECTED_PBAR) * 100
    guards["worthy_relative_error"] = worthy_off_by
    guards["pbar_error_pp"] = pbar_off_by_pp
    guards["ok"] = bool(
        worthy_off_by <= WORTHY_TOLERANCE
        and pbar_off_by_pp <= PBAR_TOLERANCE_PP
        and defence_seasons == EXPECTED_DEFENCE_SEASONS
    )

    if verbose:
        print("=== frame and guards (document 43 §4) ===")
        print(f"  charted passes            {charted.height:,}   (document 32: 80,785)")
        print(
            f"  interception-worthy       {n_worthy:,}   expected {EXPECTED_WORTHY:,} "
            f"(off by {worthy_off_by:.2%}, tolerance {WORTHY_TOLERANCE:.0%})"
        )
        print(
            f"  intercepted               {picked:,}   p(INT | worthy) = {pbar:.4f}  "
            f"expected {EXPECTED_PBAR} (off by {pbar_off_by_pp:.2f} pp, tolerance "
            f"{PBAR_TOLERANCE_PP:.1f} pp)"
        )
        print(
            f"  defence-seasons           {defence_seasons}   expected {EXPECTED_DEFENCE_SEASONS}"
        )
        print(f"  rows dropped for a null covariate  {null_drops}  -> {model.height:,} modelled")
        print(f"  guards: {'ok' if guards['ok'] else 'FAILED'}")

    if not guards["ok"]:
        raise SystemExit(
            "Guard failure (document 43 §10 / handoff constraint 8) — stop and ask before fitting."
        )

    matrix, names = design_matrix(model)
    outcome = model["interception"].cast(pl.Float64).to_numpy()
    defence_codes, n_defence = _codes(model, ["season", "defteam"])
    qb_codes, n_qb = _codes(model, ["season", "passer_player_id"])

    return WorthyFrame(
        charted=charted,
        worthy=worthy,
        model=model,
        design_matrix=matrix,
        feature_names=names,
        outcome=outcome,
        defence_season_codes=defence_codes,
        qb_season_codes=qb_codes,
        n_defence_seasons=n_defence,
        n_qb_seasons=n_qb,
        guards=guards,
    )


def residual_frame(frame: WorthyFrame) -> pl.DataFrame:
    """``model`` restricted to QB-seasons clearing ``MIN_QB_WORTHY``.

    Document 43 §4 sets the residual question's QB-season unit at >= 20 worthy
    throws. A crossed design gives every row a level on both factors, so rows
    thrown by a QB-season below the floor have no level to belong to and leave
    with it. The cost is recorded rather than worked around: the frame shrinks,
    and the chances per defence-season shrink with it.
    """
    counts = (
        frame.model.group_by(["season", "passer_player_id"])
        .agg(pl.len().alias("worthy_throws"))
        .filter(pl.col("worthy_throws") >= MIN_QB_WORTHY)
    )
    return frame.model.join(
        counts.select("season", "passer_player_id"),
        on=["season", "passer_player_id"],
        how="inner",
    )


# --------------------------------------------------------------------------
# arm 2's fixed effects, and nothing else
# --------------------------------------------------------------------------


def build_conversion_model(
    matrix: np.ndarray,
    outcome: np.ndarray,
    defence_codes: np.ndarray,
    n_defence: int,
    qb_codes: np.ndarray,
    n_qb: int,
):
    """Document 43 §5's arm-2 model, verbatim, non-centred.

        logit p_i = alpha + X_i beta + u_d[i] + v_q[i]
        alpha ~ Normal(0, 1.5)        beta_k ~ Normal(0, 1)
        u_d ~ Normal(0, sigma_d)      v_q ~ Normal(0, sigma_q)
        sigma_d, sigma_q ~ HalfNormal(0.5)

    Built here rather than in Part B's script so both parts fit the same object;
    Part A reads only ``alpha`` and ``beta`` out of it.
    """
    import pymc as pm

    coords = {
        "feature": [f"x{index}" for index in range(matrix.shape[1])],
        "defence": range(n_defence),
        "qb": range(n_qb),
    }
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", 0.0, 1.5)
        beta = pm.Normal("beta", 0.0, 1.0, dims="feature")
        sigma_d = pm.HalfNormal("sigma_d", 0.5)
        sigma_q = pm.HalfNormal("sigma_q", 0.5)
        offset_d = pm.Normal("z_d", 0.0, 1.0, dims="defence")
        offset_q = pm.Normal("z_q", 0.0, 1.0, dims="qb")
        u_d = pm.Deterministic("u_d", offset_d * sigma_d, dims="defence")
        v_q = pm.Deterministic("v_q", offset_q * sigma_q, dims="qb")
        eta = alpha + pm.math.dot(matrix, beta) + u_d[defence_codes] + v_q[qb_codes]
        pm.Bernoulli("y", logit_p=eta, observed=outcome)
    return model


def fit_fixed_effects(frame: WorthyFrame, *, seed: int = RANDOM_SEED) -> dict:
    """Posterior-mean ``alpha`` and ``beta`` from arm 2, and deliberately no more.

    The variance components this fit also estimates are the study's answer, and
    Part A must not know them: reading ``sigma_d`` here and then choosing a
    threshold would be the goalpost move documents 04 and 05 §7 wrote the
    power-first law to prevent. So the posterior is reduced to its fixed effects
    immediately and the InferenceData is dropped. Part B re-fits and reports.
    """
    import pymc as pm

    model = build_conversion_model(
        frame.design_matrix,
        frame.outcome,
        frame.defence_season_codes,
        frame.n_defence_seasons,
        frame.qb_season_codes,
        frame.n_qb_seasons,
    )
    with model:
        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            random_seed=seed,
            progressbar=False,
            nuts_sampler="nutpie",
        )
    alpha = float(idata["posterior"]["alpha"].values.mean())
    beta = idata["posterior"]["beta"].values.mean(axis=(0, 1)).astype(float)
    del idata
    return {
        "alpha": alpha,
        "beta": beta.tolist(),
        "feature_names": list(frame.feature_names),
        "rows": int(frame.model.height),
        "note": (
            "Posterior-mean fixed effects only. sigma_d and sigma_q from this fit "
            "were never read: Part A's thresholds must not know the answer they judge."
        ),
    }


def linear_predictor(beta_hat: dict, frame: pl.DataFrame, reference: pl.DataFrame) -> np.ndarray:
    """``alpha_hat + X beta_hat`` for a frame, on the fitted standardisation."""
    matrix, names = design_matrix(frame, reference=reference)
    if list(names) != list(beta_hat["feature_names"]):
        raise SystemExit("design matrix columns disagree with the saved beta — refusing to mix")
    return beta_hat["alpha"] + matrix @ np.asarray(beta_hat["beta"], dtype=float)


# --------------------------------------------------------------------------
# power: the two rate designs
# --------------------------------------------------------------------------


def rate_counts(frame: pl.DataFrame, keys: list[str], minimum: int = 0) -> pl.DataFrame:
    counts = (
        frame.group_by(keys)
        .agg(pl.len().alias("n"), pl.col("is_interception_worthy").sum().cast(pl.Int64).alias("k"))
        .drop_nulls()
        .sort(keys)
    )
    return counts.filter(pl.col("n") >= minimum) if minimum else counts


def _rate_task(spec: dict) -> dict:
    """One (design, scenario) beta-binomial cell, run in a worker."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    bounds = _betabinom.upper_bound_distribution(
        np.asarray(spec["n"], dtype=float),
        spec["league_rate"],
        spec["true_sd"],
        datasets=DATASETS,
        seed=spec["seed"],
    )
    return {"key": spec["key"], "bounds": bounds.tolist()}


def rate_specs(name: str, counts: pl.DataFrame, index: int) -> tuple[dict, list[dict]]:
    n = counts["n"].to_numpy().astype(float)
    k = counts["k"].to_numpy().astype(float)
    league_rate = float(k.sum() / n.sum())
    max_sd = float(np.sqrt(league_rate * (1.0 - league_rate)))

    meta = {
        "name": name,
        "instrument": "beta-binomial grid",
        "statistic": "89% upper bound on the population SD of the true entity rate (pp)",
        "entities": int(counts.height),
        "opportunities": int(n.sum()),
        "successes": int(k.sum()),
        "league_rate": league_rate,
        "median_n": float(np.median(n)),
    }
    specs = [
        {
            "key": (name, "null"),
            "n": n.tolist(),
            "league_rate": league_rate,
            "true_sd": 0.0,
            "seed": RANDOM_SEED + 100_000 * index,
        }
    ]
    for relative in RELATIVE_SCENARIOS:
        true_sd = relative * league_rate
        if true_sd >= max_sd:
            continue  # arithmetically impossible, reported as such in `collect`
        specs.append(
            {
                "key": (name, relative),
                "n": n.tolist(),
                "league_rate": league_rate,
                "true_sd": true_sd,
                "seed": RANDOM_SEED + 100_000 * index + int(relative * 1000),
            }
        )
    return meta, specs


# --------------------------------------------------------------------------
# power: the three residual designs
# --------------------------------------------------------------------------


def _residual_task(spec: dict) -> dict:
    """One (crossed design, scenario) cell: simulate, residualise, fit.

    Document 43 §6's simulation, verbatim::

        y_i ~ Bernoulli(logit^-1(logit p_hat_i + u_d[i] + v_q[i]))
        u_d ~ N(0, tau_d)     v_q ~ N(0, tau_q)

    with ``beta`` held at arm 2's posterior mean — a disclosed simplification
    that ignores ``beta`` uncertainty and so makes power very slightly
    optimistic (document 43 §9). The residual is recomputed against the same
    fixed ``p_hat`` every time, so the instrument sees exactly the object the
    gate will hand it.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    eta = np.asarray(spec["eta"], dtype=float)
    p_hat = 1.0 / (1.0 + np.exp(-eta))
    code_a = np.asarray(spec["code_a"], dtype=int)
    code_b = np.asarray(spec["code_b"], dtype=int)
    sizes = [int(spec["size_a"]), int(spec["size_b"])]
    design = _crossed.build_design([code_a, code_b], sizes)

    bounds_a = np.empty(DATASETS)
    bounds_b = np.empty(DATASETS)
    for index in range(DATASETS):
        rng = np.random.default_rng(spec["seed"] + index)
        shift = np.zeros(len(eta))
        if spec["tau_a"] > 0:
            shift += rng.normal(0.0, spec["tau_a"], sizes[0])[code_a]
        if spec["tau_b"] > 0:
            shift += rng.normal(0.0, spec["tau_b"], sizes[1])[code_b]
        probability = 1.0 / (1.0 + np.exp(-(eta + shift)))
        outcome = rng.binomial(1, probability).astype(float)
        residual = outcome - p_hat
        zty = _crossed.project([code_a, code_b], sizes, residual)
        fitted = _crossed.fit(design, zty, float(residual @ residual), float(residual.sum()))
        bounds_a[index] = fitted["sigma_a"]["eti89_ub"]
        bounds_b[index] = fitted["sigma_b"]["eti89_ub"]
    return {"key": spec["key"], "bounds_a": bounds_a.tolist(), "bounds_b": bounds_b.tolist()}


def residual_specs(
    name: str,
    eta: np.ndarray,
    code_a: np.ndarray,
    size_a: int,
    code_b: np.ndarray,
    size_b: int,
    *,
    index: int,
    factor: str,
    league_rate: float,
) -> tuple[dict, list[dict]]:
    """Meta and scenario specs for one crossed design.

    ``factor`` is which SD the design is powered for: ``"a"`` puts the simulated
    truth on the first factor (defence), ``"b"`` on the second (QB-season).
    """
    per_level = np.bincount(code_a, minlength=size_a)
    meta = {
        "name": name,
        "instrument": "crossed Gaussian grid",
        "statistic": (
            "89% upper bound on sigma_d (pp, probability scale)"
            if factor == "a"
            else "89% upper bound on sigma_q (pp, probability scale)"
        ),
        "rows": int(len(eta)),
        "levels_a": int(size_a),
        "levels_b": int(size_b),
        "median_chances_per_level_a": float(np.median(per_level)),
        "league_rate": league_rate,
        "powered_factor": factor,
    }
    specs = []
    for relative in (0.0, *RELATIVE_SCENARIOS):
        sd_probability = relative * league_rate
        tau = sd_probability / LOGIT_SLOPE
        specs.append(
            {
                "key": (name, "null" if relative == 0.0 else relative),
                "eta": eta.tolist(),
                "code_a": code_a.tolist(),
                "size_a": int(size_a),
                "code_b": code_b.tolist(),
                "size_b": int(size_b),
                "tau_a": tau if factor == "a" else 0.0,
                "tau_b": tau if factor == "b" else 0.0,
                "seed": RANDOM_SEED + 100_000 * index + int(relative * 1000),
            }
        )
    return meta, specs


# --------------------------------------------------------------------------
# assembling the table
# --------------------------------------------------------------------------


def summarise(meta: dict, bounds: dict[object, np.ndarray]) -> dict:
    """Threshold, power per scenario, and the Gate C-3 verdict for one design."""
    null_bounds = bounds["null"]
    threshold = float(np.percentile(null_bounds, NULL_PERCENTILE))
    max_sd = float(np.sqrt(meta["league_rate"] * (1.0 - meta["league_rate"])))

    rows = []
    for relative in RELATIVE_SCENARIOS:
        true_sd = relative * meta["league_rate"]
        if relative not in bounds:
            rows.append(
                {
                    "relative": relative,
                    "true_sd_pp": true_sd * 100,
                    "power": None,
                    "impossible": true_sd >= max_sd,
                }
            )
            continue
        scenario = bounds[relative]
        rows.append(
            {
                "relative": relative,
                "true_sd_pp": true_sd * 100,
                "mean_upper_bound_pp": float(scenario.mean()) * 100,
                "power": float((scenario >= threshold).mean()),
                "impossible": False,
            }
        )

    reference = next(row for row in rows if row["relative"] == REFERENCE_RELATIVE)
    resolvable = reference["power"] is not None and reference["power"] >= MIN_POWER
    return {
        **meta,
        "null_bound_mean_pp": float(null_bounds.mean()) * 100,
        "gate_threshold_pp": threshold * 100,
        "power": rows,
        "power_at_reference": reference["power"],
        "resolvable": bool(resolvable),
    }


def print_report(report: dict) -> None:
    print(f"\n--- {report['name']}")
    if "entities" in report:
        print(
            f"    {report['entities']} entities, {report['opportunities']:,} opportunities, "
            f"league rate {report['league_rate']:.4%}, median n {report['median_n']:.0f}"
        )
    else:
        print(
            f"    {report['rows']:,} rows, {report['levels_a']} x {report['levels_b']} levels, "
            f"median chances per level {report['median_chances_per_level_a']:.0f}, "
            f"powered on sigma_{report['powered_factor']}"
        )
    print(
        f"    null 89% upper bound: mean {report['null_bound_mean_pp']:.4f} pp, "
        f"{NULL_PERCENTILE}th pct {report['gate_threshold_pp']:.4f} pp = threshold"
    )
    for row in report["power"]:
        if row["power"] is None:
            reason = "arithmetically impossible" if row["impossible"] else "not simulated"
            print(f"    true SD {row['relative']:5.1%} ({row['true_sd_pp']:.3f} pp): {reason}")
            continue
        print(
            f"    true SD {row['relative']:5.1%} ({row['true_sd_pp']:6.3f} pp): "
            f"rejected {row['power']:.3f} of the time  "
            f"(mean bound {row['mean_upper_bound_pp']:.3f} pp)"
        )
    print(
        f"    => power at the {REFERENCE_RELATIVE:.1%} reference "
        f"{report['power_at_reference']:.3f}: "
        f"{'RESOLVABLE' if report['resolvable'] else 'UNRESOLVABLE'}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", action="store_true", help="run every cell in this process")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    paths.ensure_data_dirs()
    started = time.time()
    frame = build_worthy_frame()

    # --- the fixed effects the residual designs simulate around ---------------
    beta_path = paths.RESEARCH_OUTPUT_DIR / "61_beta_hat.json"
    if beta_path.exists():
        beta_hat = json.loads(beta_path.read_text())
        print(f"\nreusing arm-2 fixed effects from {beta_path.name}")
    else:
        print("\nfitting arm 2 once, for beta only (sigma_d / sigma_q not read here)")
        beta_hat = fit_fixed_effects(frame)
        beta_path.write_text(json.dumps(beta_hat, indent=2))
        print(f"wrote {beta_path}")

    residual = residual_frame(frame)
    eta = linear_predictor(beta_hat, residual, frame.model)
    p_hat = 1.0 / (1.0 + np.exp(-eta))
    defence_season_codes, n_defence_season = _codes(residual, ["season", "defteam"])
    defence_pooled_codes, n_defence_pooled = _codes(residual, ["defteam"])
    qb_codes, n_qb = _codes(residual, ["season", "passer_player_id"])
    conversion_rate = float(residual["interception"].mean())

    print(
        f"\nresidual frame: {residual.height:,} rows "
        f"(QB-seasons with >= {MIN_QB_WORTHY} worthy throws), "
        f"{n_defence_season} defence-seasons, {n_defence_pooled} defences, {n_qb} QB-seasons"
    )
    print(
        f"  conversion in the residual frame {conversion_rate:.4f}; mean p_hat {p_hat.mean():.4f}"
    )

    # --- specs ---------------------------------------------------------------
    qb_rate_counts = rate_counts(
        frame.charted.drop_nulls("passer_player_id"),
        ["season", "passer_player_id"],
        minimum=MIN_QB_ATTEMPTS,
    )
    defence_rate_counts = rate_counts(frame.charted, ["season", "defteam"])

    metas: dict[str, dict] = {}
    rate_jobs: list[dict] = []
    residual_jobs: list[dict] = []

    for index, (name, counts) in enumerate(
        (
            ("worthy_rate_qb_season", qb_rate_counts),
            ("worthy_rate_defence_season", defence_rate_counts),
        )
    ):
        meta, specs = rate_specs(name, counts, index)
        metas[name] = meta
        rate_jobs.extend(specs)

    crossed_designs = [
        (
            "residual_defence_season_x_qb_season",
            defence_season_codes,
            n_defence_season,
            qb_codes,
            n_qb,
            "a",
            2,
        ),
        (
            "residual_defence_pooled_x_qb_season",
            defence_pooled_codes,
            n_defence_pooled,
            qb_codes,
            n_qb,
            "a",
            3,
        ),
        (
            "residual_qb_season_sigma_q",
            defence_season_codes,
            n_defence_season,
            qb_codes,
            n_qb,
            "b",
            4,
        ),
    ]
    for name, code_a, size_a, code_b, size_b, factor, index in crossed_designs:
        meta, specs = residual_specs(
            name,
            eta,
            code_a,
            size_a,
            code_b,
            size_b,
            index=index,
            factor=factor,
            league_rate=conversion_rate,
        )
        metas[name] = meta
        # Design 5 reads sigma_q out of design 3's crossed fit, so its null is
        # design 3's null — the same simulation with both effects at zero. Only
        # its four non-null scenarios need their own datasets.
        if name == "residual_qb_season_sigma_q":
            specs = [spec for spec in specs if spec["key"][1] != "null"]
        residual_jobs.extend(specs)

    print(
        f"\n=== simulating {len(rate_jobs)} rate cells and {len(residual_jobs)} residual cells "
        f"at {DATASETS} datasets each ==="
    )

    def run(jobs: list[dict], worker) -> list[dict]:
        if args.serial or args.workers <= 1:
            return [worker(job) for job in jobs]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            return list(pool.map(worker, jobs))

    rate_results = run(rate_jobs, _rate_task)
    print(f"  rate cells done ({time.time() - started:.0f}s elapsed)")
    residual_results = run(residual_jobs, _residual_task)
    print(f"  residual cells done ({time.time() - started:.0f}s elapsed)")

    # --- collect -------------------------------------------------------------
    collected: dict[str, dict[object, np.ndarray]] = {name: {} for name in metas}
    for result in rate_results:
        name, scenario = result["key"]
        collected[name][scenario] = np.asarray(result["bounds"])
    for result in residual_results:
        name, scenario = result["key"]
        factor = metas[name]["powered_factor"]
        collected[name][scenario] = np.asarray(result[f"bounds_{factor}"])
        if name == "residual_defence_season_x_qb_season" and scenario == "null":
            collected["residual_qb_season_sigma_q"]["null"] = np.asarray(result["bounds_b"])

    reports = {name: summarise(metas[name], collected[name]) for name in metas}
    for report in reports.values():
        print_report(report)

    print("\n=== document 43 §6 table ===")
    for report in reports.values():
        cells = []
        for row in report["power"]:
            cells.append("*impossible*" if row["impossible"] else f"{row['power']:.2f}")
        print(
            f"| {report['name']} | {report['gate_threshold_pp']:.2f} pp | "
            + " | ".join(cells)
            + f" | {'Yes' if report['resolvable'] else 'No'} |"
        )

    out = paths.RESEARCH_OUTPUT_DIR / "61_dropped_pick_power.json"
    out.write_text(
        json.dumps(
            {
                "datasets_per_scenario": DATASETS,
                "relative_scenarios": list(RELATIVE_SCENARIOS),
                "reference_relative": REFERENCE_RELATIVE,
                "min_power": MIN_POWER,
                "null_percentile": NULL_PERCENTILE,
                "random_seed": RANDOM_SEED,
                "min_qb_attempts": MIN_QB_ATTEMPTS,
                "min_qb_worthy": MIN_QB_WORTHY,
                "logit_slope": LOGIT_SLOPE,
                "post_branch_excluded": list(POST_BRANCH_EXCLUDED),
                "guards": frame.guards,
                "residual_frame": {
                    "rows": int(residual.height),
                    "defence_seasons": int(n_defence_season),
                    "defences_pooled": int(n_defence_pooled),
                    "qb_seasons": int(n_qb),
                    "conversion_rate": conversion_rate,
                    "mean_p_hat": float(p_hat.mean()),
                },
                "designs": reports,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out}")
    print(f"guards: {'ok' if frame.guards['ok'] else 'FAILED'}")
    print(f"total {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
