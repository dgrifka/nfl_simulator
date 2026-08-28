"""Part A of round 7 — power on the receiver-drop frame, before any threshold.

The receiver side of amendment A-3, pre-registered in
`docs/research/56-receiver-drop-mirror-prereg.md`. This runs **before** document
56 §1's table carries a single number, per the process law documents 04 -> 05 §7
-> 09 §4 established and rounds 1-6 kept: the thresholds a gate is judged
against are simulated before the fit that will be judged.

Three grains are powered, at the instrument document 43 §6 fixed:

    1. residual, receiver-season x defence-season   crossed Gaussian grid
    2. residual, team-season x defence-season       crossed Gaussian grid
    3. residual, defence-season (sigma_d)           design 1, reading sigma_b

and the three raw drop-rate spreads arm 1 will report under Gate D-1 are
powered beside them on the beta-binomial grid, because document 43 §7 asks for
D-1's intervals to be quoted with their C-3 power and a number without its power
is what the whole ordering exists to prevent.

Direction of the test, which is the reverse of a skill hunt: a coin-like finish
is confirmed by showing the entity spread is **small**, so the gate reads "the
89% upper bound is below X". Power is the chance of correctly rejecting that
when a real effect exists, and a design with low power cannot report either
outcome as a finding (Gate C-3).

**One thing here is not round 6's, and it is an implementation, not an
instrument.** The gate arm crosses 1,931 receiver-seasons with 128
defence-seasons, and `research/_crossed_gaussian_grid.py` costs a 2,059-square
Cholesky at each of its 1,681 grid points — measured at **183.8 s for one fit**
on this machine, which is about 100 hours for the 2,000 fits this file needs at
that grain. `research/_crossed_block_grid.py` evaluates the same profiled
restricted likelihood on the same grid through a Schur complement on the
128-level block, at 0.17 s, and its `self_check` reproduces the original to
**1.2e-15** — that reproduction is printed first, before any number derived
from it, and the run stops if it fails.

    uv run python research/71_receiver_drop_power.py
    uv run python research/71_receiver_drop_power.py --serial   # no worker pool

Parallel and serial runs produce identical numbers: every simulated dataset draws
from ``np.random.default_rng(seed + index)``, so nothing depends on the pool.

Nothing in `src/nfl_simulator/` changes on any number below.
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

_power = import_module("61_dropped_pick_power")
_betabinom = import_module("_betabinom_grid")
_block = import_module("_crossed_block_grid")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.ingest import FTN_SEASONS, load_ftn, load_pbp  # noqa: E402

# --- document 56 §4, constants -------------------------------------------------

RANDOM_SEED = 20260827
DATASETS = 400
RELATIVE_SCENARIOS = (0.05, 0.125, 0.25, 0.50)
REFERENCE_RELATIVE = 0.125
MIN_POWER = 0.80
NULL_PERCENTILE = 90

# Arm 1's receiver unit, document 56 §1. It is a *unit definition for the rate
# question only* and never touches the gate arm: amendment A-1 (document 45)
# established that a floor in a crossed design is a row filter, and the gate arm
# below is floorless for exactly that reason.
MIN_RECEIVER_TARGETS = 100

# Stop-and-ask guards, document 56 §4.
EXPECTED_CATCHABLE = 56211
CATCHABLE_TOLERANCE = 0.02
EXPECTED_DROP_RATE = 0.0495
DROP_RATE_TOLERANCE_PP = 0.3
EXPECTED_DEFENCE_SEASONS = 128

# d/dx of the logistic at the league drop rate: 0.0495 * 0.9505. Converts a
# logit-scale SD to the probability scale and back, the same constant document
# 43 §5's cross-check uses on the dropped-pick side — recomputed here at *this*
# league rate, because a drop is a 1-in-20 event and a pick a coin.
LOGIT_SLOPE = EXPECTED_DROP_RATE * (1.0 - EXPECTED_DROP_RATE)

PBP_COLUMNS = [
    "game_id",
    "play_id",
    "season",
    "week",
    "posteam",
    "defteam",
    "play_type",
    "complete_pass",
    "incomplete_pass",
    "receiver_player_id",
    "receiver_player_name",
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
    "is_catchable_ball",
    "is_drop",
    "is_contested_ball",
    "is_qb_out_of_pocket",
    "is_play_action",
    "is_screen_pass",
    "n_pass_rushers",
]

# Standardised covariates, document 56 §1. Everything else in X is an indicator
# and enters as a 0/1 dummy, so a Normal(0, 1) prior on its coefficient is
# already weakly informative on the logit scale.
STANDARDISED = ("air_yards", "n_pass_rushers", "ydstogo", "yardline_100", "wp")

# Excluded by rule (document 56 §1): anything recorded after the ball reaches the
# receiver's hands. Listed here so a reader can check the frame against the rule.
# `is_catchable_ball` is not on it — it is the *selection*, so it cannot also be
# a covariate, which is the one structural difference from document 43 §4's list.
POST_BRANCH_EXCLUDED = (
    "is_drop",
    "is_created_reception",
    "complete_pass",
    "epa",
    "air_epa",
    "xyac_epa",
    "yards_after_catch",
)


# --------------------------------------------------------------------------
# the frame
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CatchableFrame:
    """Every frame the receiver-drop study needs, built once.

    ``charted`` is all 2022-2025 charted plays joined to pbp and supplies the
    per-entity target denominators. ``catchable`` is the catchable-ball subset
    before complete-case filtering — the object document 56 §4's guard counts.
    ``model`` is the complete-case subset arm 2 fits, with its design matrix and
    level codes, and it is also the gate arm's frame (amendment A-1: no floor).
    """

    charted: pl.DataFrame
    catchable: pl.DataFrame
    model: pl.DataFrame
    design_matrix: np.ndarray
    feature_names: tuple[str, ...]
    outcome: np.ndarray
    receiver_season_codes: np.ndarray
    team_season_codes: np.ndarray
    defence_season_codes: np.ndarray
    n_receiver_seasons: int
    n_team_seasons: int
    n_defence_seasons: int
    guards: dict


def load_charted_plays() -> pl.DataFrame:
    """FTN charting joined to pbp, every charted play.

    **No ``play_type`` filter**, and that is deliberate rather than an omission.
    Document 56 §4's guard is 56,211 catchable targets, which is FTN's own
    ``is_catchable_ball`` count over the charting table; filtering to
    ``play_type == "pass"`` gives 54,336 and would miss the guard by 3.3%
    against a 2% tolerance. The 1,865 rows the filter would remove are
    penalty-nullified (``play_type == "no_play"``, every one of them carrying a
    penalty), they are genuine charted catchable targets, and every one of them
    has a null ``air_yards``, ``pass_location`` and ``receiver_player_id`` — so
    they leave in the complete-case step below rather than by a filter written
    before anyone counted them. The difference is that the record says so.
    """
    pbp = load_pbp(FTN_SEASONS, columns=PBP_COLUMNS)
    ftn = load_ftn(FTN_SEASONS)
    return ftn.select(
        pl.col("nflverse_game_id").alias("game_id"),
        pl.col("nflverse_play_id").cast(pl.Float64).alias("play_id"),
        *FTN_COLUMNS,
    ).join(pbp, on=["game_id", "play_id"], how="inner")


def design_matrix(frame: pl.DataFrame, reference: pl.DataFrame | None = None) -> tuple:
    """`X` and its column names, per document 56 §1's covariate list.

    Standardisation uses ``reference``'s mean and SD when one is given, so that a
    subset frame is measured on the same scale as the frame arm 2 was fitted on —
    round 3's fourth surprise, and the reason the week-out folds pass a reference
    in. ``air_yards`` enters as its standardised value and that value squared;
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
        "qb_hit",
        "is_qb_out_of_pocket",
        "is_play_action",
        "is_screen_pass",
        "shotgun",
    ):
        columns.append(frame[column].cast(pl.Float64).to_numpy())
        names.append(column)

    return np.column_stack(columns), tuple(names)


REQUIRED_COLUMNS = [
    *STANDARDISED,
    "pass_location",
    "down",
    "receiver_player_id",
    "posteam",
    "defteam",
    "is_contested_ball",
    "qb_hit",
    "is_qb_out_of_pocket",
    "is_play_action",
    "is_screen_pass",
    "shotgun",
]


def build_catchable_frame(*, verbose: bool = True) -> CatchableFrame:
    """The study's frame, with document 56 §4's guards checked before anything else."""
    charted = load_charted_plays()
    catchable = charted.filter(pl.col("is_catchable_ball"))

    n_catchable = catchable.height
    dropped = int(catchable["is_drop"].sum())
    drop_rate = dropped / n_catchable
    defence_seasons = catchable.select(pl.struct("season", "defteam").n_unique()).item()

    model = catchable.drop_nulls(REQUIRED_COLUMNS)
    null_drops = n_catchable - model.height
    nullified = int(catchable.filter(pl.col("play_type") == "no_play").height)

    guards = {
        "charted_plays": int(charted.height),
        "catchable": int(n_catchable),
        "catchable_expected": EXPECTED_CATCHABLE,
        "dropped": dropped,
        "drop_rate": drop_rate,
        "drop_rate_expected": EXPECTED_DROP_RATE,
        "defence_seasons": int(defence_seasons),
        "null_dropped_rows": int(null_drops),
        "penalty_nullified_rows": nullified,
        "model_rows": int(model.height),
        "model_drop_rate": float(model["is_drop"].mean()),
    }

    catchable_off_by = abs(n_catchable - EXPECTED_CATCHABLE) / EXPECTED_CATCHABLE
    drop_off_by_pp = abs(drop_rate - EXPECTED_DROP_RATE) * 100
    guards["catchable_relative_error"] = catchable_off_by
    guards["drop_rate_error_pp"] = drop_off_by_pp
    guards["ok"] = bool(
        catchable_off_by <= CATCHABLE_TOLERANCE
        and drop_off_by_pp <= DROP_RATE_TOLERANCE_PP
        and defence_seasons == EXPECTED_DEFENCE_SEASONS
    )

    if verbose:
        print("=== frame and guards (document 56 §1, §4) ===")
        print(f"  charted plays             {charted.height:,}")
        print(
            f"  catchable targets         {n_catchable:,}   expected "
            f"{EXPECTED_CATCHABLE:,} (off by {catchable_off_by:.2%}, tolerance "
            f"{CATCHABLE_TOLERANCE:.0%})"
        )
        print(
            f"  dropped                   {dropped:,}   p(drop | catchable) = "
            f"{drop_rate:.4f}  expected {EXPECTED_DROP_RATE} (off by "
            f"{drop_off_by_pp:.2f} pp, tolerance {DROP_RATE_TOLERANCE_PP:.1f} pp)"
        )
        print(
            f"  defence-seasons           {defence_seasons}   expected {EXPECTED_DEFENCE_SEASONS}"
        )
        print(
            f"  rows dropped for a null covariate  {null_drops:,} "
            f"(of which {nullified:,} are penalty-nullified no_play rows) "
            f"-> {model.height:,} modelled, drop rate {guards['model_drop_rate']:.4f}"
        )
        print(f"  guards: {'ok' if guards['ok'] else 'FAILED'}")

    if not guards["ok"]:
        raise SystemExit(
            "Guard failure (document 56 §4 / handoff constraint 6) — stop and ask before fitting."
        )

    matrix, names = design_matrix(model)
    outcome = model["is_drop"].cast(pl.Float64).to_numpy()
    receiver_codes, n_receiver = _power._codes(model, ["season", "receiver_player_id"])
    team_codes, n_team = _power._codes(model, ["season", "posteam"])
    defence_codes, n_defence = _power._codes(model, ["season", "defteam"])

    if verbose:
        print(
            f"  levels: {n_receiver:,} receiver-seasons, {n_team} team-seasons, "
            f"{n_defence} defence-seasons"
        )

    return CatchableFrame(
        charted=charted,
        catchable=catchable,
        model=model,
        design_matrix=matrix,
        feature_names=names,
        outcome=outcome,
        receiver_season_codes=receiver_codes,
        team_season_codes=team_codes,
        defence_season_codes=defence_codes,
        n_receiver_seasons=n_receiver,
        n_team_seasons=n_team,
        n_defence_seasons=n_defence,
        guards=guards,
    )


# --------------------------------------------------------------------------
# arm 2's model, and its fixed effects
# --------------------------------------------------------------------------


def build_drop_model(
    matrix: np.ndarray,
    outcome: np.ndarray,
    receiver_codes: np.ndarray,
    n_receiver: int,
    defence_codes: np.ndarray,
    n_defence: int,
):
    """Document 56 §1's arm-2 model, verbatim, non-centred.

        logit p_i = alpha + X_i beta + r_s[i] + d_d[i]
        alpha ~ Normal(0, 1.5)        beta_k ~ Normal(0, 1)
        r_s ~ Normal(0, sigma_r)      d_d ~ Normal(0, sigma_d)
        sigma_r, sigma_d ~ HalfNormal(0.5)

    ``p_i`` is the probability the ball is **dropped**: document 56 §0 quotes the
    league rate as 4.95%, and §1's power scenarios are relative to it, so the
    modelled event is the drop. The read side takes the complement — a catch —
    exactly as the dropped-pick builder takes ``1 - catch`` for an escape.

    ``d_d`` is in the model and out of the read (document 56 §2): the coverage's
    contribution to a drop is the defence's football and stays in ``core``, but
    it has to be estimated or ``r_s`` would carry the receiver's schedule.

    Built here rather than in Part B's script so both parts fit the same object;
    Part A reads only ``alpha`` and ``beta`` out of it.
    """
    import pymc as pm

    coords = {
        "feature": [f"x{index}" for index in range(matrix.shape[1])],
        "receiver": range(n_receiver),
        "defence": range(n_defence),
    }
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", 0.0, 1.5)
        beta = pm.Normal("beta", 0.0, 1.0, dims="feature")
        sigma_r = pm.HalfNormal("sigma_r", 0.5)
        sigma_d = pm.HalfNormal("sigma_d", 0.5)
        offset_r = pm.Normal("z_r", 0.0, 1.0, dims="receiver")
        offset_d = pm.Normal("z_d", 0.0, 1.0, dims="defence")
        r_s = pm.Deterministic("r_s", offset_r * sigma_r, dims="receiver")
        d_d = pm.Deterministic("d_d", offset_d * sigma_d, dims="defence")
        eta = alpha + pm.math.dot(matrix, beta) + r_s[receiver_codes] + d_d[defence_codes]
        pm.Bernoulli("y", logit_p=eta, observed=outcome)
    return model


def fit_fixed_effects(frame: CatchableFrame, *, seed: int = RANDOM_SEED) -> dict:
    """Posterior-mean ``alpha`` and ``beta`` from arm 2, and deliberately no more.

    The variance components this fit also estimates are the study's answer, and
    Part A must not know them: reading ``sigma_r`` here and then choosing a
    threshold would be the goalpost move documents 04 and 05 §7 wrote the
    power-first law to prevent. So the posterior is reduced to its fixed effects
    immediately and the InferenceData is dropped. Part B re-fits and reports.
    """
    import pymc as pm

    model = build_drop_model(
        frame.design_matrix,
        frame.outcome,
        frame.receiver_season_codes,
        frame.n_receiver_seasons,
        frame.defence_season_codes,
        frame.n_defence_seasons,
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
            "Posterior-mean fixed effects only. sigma_r and sigma_d from this fit "
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
# power: the three rate designs (Gate D-1's power)
# --------------------------------------------------------------------------


def rate_counts(frame: pl.DataFrame, keys: list[str], minimum: int = 0) -> pl.DataFrame:
    counts = (
        frame.group_by(keys)
        .agg(pl.len().alias("n"), pl.col("is_drop").sum().cast(pl.Int64).alias("k"))
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
        "statistic": "89% upper bound on the population SD of the true entity drop rate (pp)",
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
            continue
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
# power: the three residual designs (Gate C-3)
# --------------------------------------------------------------------------


def _residual_task(spec: dict) -> dict:
    """One (crossed design, scenario) cell: simulate, residualise, fit.

    Document 43 §6's simulation, verbatim::

        y_i ~ Bernoulli(logit^-1(logit p_hat_i + a[i] + b[i]))
        a ~ N(0, tau_a)     b ~ N(0, tau_b)

    with ``beta`` held at arm 2's posterior mean — a disclosed simplification
    that ignores ``beta`` uncertainty and so makes power very slightly
    optimistic (document 43 §9). The residual is recomputed against the same
    fixed ``p_hat`` every time, so the instrument sees exactly the object the
    gate will hand it.

    The only difference from round 6's task is which module evaluates the grid;
    the module's own `self_check` is what licenses that, and `main` prints it
    before this function is ever called.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    eta = np.asarray(spec["eta"], dtype=float)
    p_hat = 1.0 / (1.0 + np.exp(-eta))
    code_a = np.asarray(spec["code_a"], dtype=int)
    code_b = np.asarray(spec["code_b"], dtype=int)
    size_a, size_b = int(spec["size_a"]), int(spec["size_b"])
    design = _block.build_blocks(code_a, code_b, size_a, size_b)

    bounds_a = np.empty(DATASETS)
    bounds_b = np.empty(DATASETS)
    for index in range(DATASETS):
        rng = np.random.default_rng(spec["seed"] + index)
        shift = np.zeros(len(eta))
        if spec["tau_a"] > 0:
            shift += rng.normal(0.0, spec["tau_a"], size_a)[code_a]
        if spec["tau_b"] > 0:
            shift += rng.normal(0.0, spec["tau_b"], size_b)[code_b]
        probability = 1.0 / (1.0 + np.exp(-(eta + shift)))
        outcome = rng.binomial(1, probability).astype(float)
        residual = outcome - p_hat
        zty_a, zty_b = _block.project_blocks(code_a, code_b, size_a, size_b, residual)
        fitted = _block.fit_blocked(
            design, zty_a, zty_b, float(residual @ residual), float(residual.sum())
        )
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
    truth on the first factor, ``"b"`` on the second.
    """
    per_level = np.bincount(code_a, minlength=size_a)
    meta = {
        "name": name,
        "instrument": "crossed Gaussian grid (blocked evaluation, `_crossed_block_grid`)",
        "statistic": (
            f"89% upper bound on sigma_{factor} (pp, probability scale) on the conditioned residual"
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


RECEIVER_DESIGN = "residual_receiver_season_x_defence_season"
TEAM_DESIGN = "residual_team_season_x_defence_season"
DEFENCE_DESIGN = "residual_defence_season_sigma_d"

GRAIN_OF = {
    RECEIVER_DESIGN: "receiver-season",
    TEAM_DESIGN: "team-season",
    DEFENCE_DESIGN: "defence-season",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", action="store_true", help="run every cell in this process")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    paths.ensure_data_dirs()
    started = time.time()

    # --- the licence for the fast grid, before anything uses it ---------------
    licence = _block.self_check()
    if not licence["pass"]:
        raise SystemExit(
            "the blocked grid does not reproduce `_crossed_gaussian_grid.fit`. "
            "Nothing below may be computed with it — stop and report."
        )

    frame = build_catchable_frame()

    # --- the fixed effects the residual designs simulate around ---------------
    beta_path = paths.RESEARCH_OUTPUT_DIR / "71_beta_hat.json"
    if beta_path.exists():
        beta_hat = json.loads(beta_path.read_text())
        print(f"\nreusing arm-2 fixed effects from {beta_path.name}")
    else:
        print("\nfitting arm 2 once, for beta only (sigma_r / sigma_d not read here)")
        fit_started = time.time()
        beta_hat = fit_fixed_effects(frame)
        beta_hat["wall_clock_seconds"] = time.time() - fit_started
        beta_path.write_text(json.dumps(beta_hat, indent=2))
        print(f"wrote {beta_path} ({beta_hat['wall_clock_seconds']:.0f} s)")

    residual = frame.model
    eta = linear_predictor(beta_hat, residual, frame.model)
    p_hat = 1.0 / (1.0 + np.exp(-eta))
    receiver_codes, n_receiver = _power._codes(residual, ["season", "receiver_player_id"])
    team_codes, n_team = _power._codes(residual, ["season", "posteam"])
    defence_codes, n_defence = _power._codes(residual, ["season", "defteam"])
    drop_rate = float(residual["is_drop"].mean())

    targets_per_receiver = np.bincount(receiver_codes, minlength=n_receiver)
    print(
        f"\ngate-arm frame: {residual.height:,} rows (no floor — amendment A-1), "
        f"{n_receiver:,} receiver-seasons, {n_team} team-seasons, "
        f"{n_defence} defence-seasons"
    )
    print(
        f"  targets per receiver-season: median "
        f"{np.median(targets_per_receiver):.0f}, min {targets_per_receiver.min()}, "
        f"max {targets_per_receiver.max()}"
    )
    print(f"  drop rate in the gate-arm frame {drop_rate:.4f}; mean p_hat {p_hat.mean():.4f}")

    # --- specs ---------------------------------------------------------------
    receiver_rate_counts = rate_counts(
        frame.charted.drop_nulls("receiver_player_id").filter(pl.col("is_catchable_ball")),
        ["season", "receiver_player_id"],
        minimum=MIN_RECEIVER_TARGETS,
    )
    team_rate_counts = rate_counts(frame.catchable, ["season", "posteam"])
    defence_rate_counts = rate_counts(frame.catchable, ["season", "defteam"])

    metas: dict[str, dict] = {}
    rate_jobs: list[dict] = []
    residual_jobs: list[dict] = []

    for index, (name, counts) in enumerate(
        (
            ("drop_rate_receiver_season", receiver_rate_counts),
            ("drop_rate_team_season", team_rate_counts),
            ("drop_rate_defence_season", defence_rate_counts),
        )
    ):
        meta, specs = rate_specs(name, counts, index)
        metas[name] = meta
        rate_jobs.extend(specs)

    crossed_designs = [
        (RECEIVER_DESIGN, receiver_codes, n_receiver, defence_codes, n_defence, "a", 3),
        (TEAM_DESIGN, team_codes, n_team, defence_codes, n_defence, "a", 4),
        (DEFENCE_DESIGN, receiver_codes, n_receiver, defence_codes, n_defence, "b", 5),
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
            league_rate=drop_rate,
        )
        metas[name] = meta
        # The defence-season design reads sigma_d out of the receiver design's
        # crossed fit, so its null is that design's null — the same simulation
        # with both effects at zero. Only its four non-null scenarios need their
        # own datasets.
        if name == DEFENCE_DESIGN:
            specs = [spec for spec in specs if spec["key"][1] != "null"]
        residual_jobs.extend(specs)

    print(
        f"\n=== simulating {len(rate_jobs)} rate cells and {len(residual_jobs)} residual "
        f"cells at {DATASETS} datasets each ==="
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
        if name == RECEIVER_DESIGN and scenario == "null":
            collected[DEFENCE_DESIGN]["null"] = np.asarray(result["bounds_b"])

    reports = {name: _power.summarise(metas[name], collected[name]) for name in metas}
    for report in reports.values():
        _power.print_report(report)

    print("\n=== document 56 §1 table — the gate arm, one row per grain ===")
    print("| grain | null bound | 5% | 12.5% | 25% | 50% | resolvable |")
    print("|---|---|---|---|---|---|---|")
    for name in (RECEIVER_DESIGN, TEAM_DESIGN, DEFENCE_DESIGN):
        report = reports[name]
        cells = [
            "*impossible*" if row["impossible"] else f"{row['power']:.2f}"
            for row in report["power"]
        ]
        print(
            f"| {GRAIN_OF[name]} | {report['gate_threshold_pp']:.3f} pp | "
            + " | ".join(cells)
            + f" | {'Yes' if report['resolvable'] else 'No'} |"
        )

    print("\n=== Gate D-1's power, on the raw drop-rate spreads ===")
    print("| grain | entities | null bound | 5% | 12.5% | 25% | 50% | resolvable |")
    print("|---|---|---|---|---|---|---|---|")
    for name in ("drop_rate_receiver_season", "drop_rate_team_season", "drop_rate_defence_season"):
        report = reports[name]
        cells = [
            "*impossible*" if row["impossible"] else f"{row['power']:.2f}"
            for row in report["power"]
        ]
        print(
            f"| {name.removeprefix('drop_rate_').replace('_', '-')} | "
            f"{report['entities']} | {report['gate_threshold_pp']:.3f} pp | "
            + " | ".join(cells)
            + f" | {'Yes' if report['resolvable'] else 'No'} |"
        )

    # --- the clause-1 grain rule, stated but not yet decided ------------------
    # Document 56 §1 pre-commits the order; Part B applies it once C-3 is read
    # off these thresholds together with arm 3's bounds. Printing it here makes
    # the rule visible before the numbers it will consume exist.
    resolvable = [name for name in (RECEIVER_DESIGN, TEAM_DESIGN) if reports[name]["resolvable"]]
    print(
        "\nclause-1 grain rule (document 56 §1): receiver-season if it clears C-3, "
        "else team-season, else G-4 fails.\n"
        f"  C-3 clears at: {[GRAIN_OF[name] for name in resolvable] or 'neither grain'}"
    )

    elapsed = time.time() - started
    out = paths.RESEARCH_OUTPUT_DIR / "71_receiver_drop_power.json"
    out.write_text(
        json.dumps(
            {
                "document": "56 — the receiver-drop mirror (A-3 gate G-4)",
                "part": "A — power, before thresholds",
                "datasets_per_scenario": DATASETS,
                "relative_scenarios": list(RELATIVE_SCENARIOS),
                "reference_relative": REFERENCE_RELATIVE,
                "min_power": MIN_POWER,
                "null_percentile": NULL_PERCENTILE,
                "random_seed": RANDOM_SEED,
                "min_receiver_targets_arm1": MIN_RECEIVER_TARGETS,
                "logit_slope": LOGIT_SLOPE,
                "post_branch_excluded": list(POST_BRANCH_EXCLUDED),
                "blocked_grid_licence": licence,
                "guards": frame.guards,
                "gate_arm_frame": {
                    "rows": int(residual.height),
                    "receiver_seasons": int(n_receiver),
                    "team_seasons": int(n_team),
                    "defence_seasons": int(n_defence),
                    "median_targets_per_receiver_season": float(np.median(targets_per_receiver)),
                    "min_targets_per_receiver_season": int(targets_per_receiver.min()),
                    "drop_rate": drop_rate,
                    "mean_p_hat": float(p_hat.mean()),
                },
                "beta_hat_rows": beta_hat["rows"],
                "beta_hat_fit_seconds": beta_hat.get("wall_clock_seconds"),
                "clause_1_grains_clearing_c3": [GRAIN_OF[name] for name in resolvable],
                "wall_clock_seconds": elapsed,
                "designs": reports,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out}")
    print(f"guards: {'ok' if frame.guards['ok'] else 'FAILED'}")
    print(f"total {elapsed:.0f}s")


if __name__ == "__main__":
    main()
