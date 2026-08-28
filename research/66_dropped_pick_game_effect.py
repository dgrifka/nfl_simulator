"""Round 3 Part B — is "seasons differ" game clustering?

The robustness check pre-registered in
`docs/research/47-dropped-pick-round3-prereg.md` §2, before any fit.

Round 2 found a defence-season spread in conditioned conversion that does not
carry to the next season (document 46 §4). A defence-season's ~22 chances come
from ~17 games, and throws inside one game share a quarterback, a weather, a
game script and a charter's mood. If conversion residuals are correlated within
a game, that spread can be inflated by *game* clustering with no defensive
contribution at all — and it would look exactly like round 2's pattern.

The design is round 2's arm 2 with one added node and nothing else moved:

    logit p_i = alpha + X_i beta + u_d[i] + v_q[i] + w_g[i]
    w_g ~ Normal(0, sigma_g)       sigma_g ~ HalfNormal(0.5)

non-centred like the other two, at amendment A-2's sampler spec (4 x 2,000
draws after 2,000 tuning, target_accept 0.9). One fit.

Document 47 §2 discloses that the crossed grid cannot take a third factor, so
this check runs in the confirmatory arm and the comparison below therefore mixes
instruments: `sigma_d` here is arm 2's parameter, read against arm 3's
season-grain null bound of 5.920 pp. That is the comparison document 47 §2
committed to, and it is the one printed.

    uv run python research/66_dropped_pick_game_effect.py

Nothing in `src/nfl_simulator/` changes on any outcome.
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm

sys.path.insert(0, str(Path(__file__).parent))

_power = import_module("61_dropped_pick_power")
_confounds = import_module("62_dropped_pick_confounds")

from nfl_simulator import paths  # noqa: E402

RANDOM_SEED = _power.RANDOM_SEED
LOGIT_SLOPE = _power.LOGIT_SLOPE

# Amendment A-2's spec, unchanged — document 47 §2 says "same spec as A-2".
DRAWS = 2000
TUNE = 2000
CHAINS = 4
TARGET_ACCEPT = 0.9

# Document 47 §5: the reference bound for §2's reading, from document 45 §4 row
# 1. Arm 3's season-grain null bound, quoted rather than recomputed.
REFERENCE_BOUND_PP = 5.920

EXPECTED_MODEL_ROWS = 2969
ROUND2_JSON = "64_dropped_pick_confounds_r2.json"


def build_game_effect_model(
    matrix: np.ndarray,
    outcome: np.ndarray,
    defence_codes: np.ndarray,
    n_defence: int,
    qb_codes: np.ndarray,
    n_qb: int,
    game_codes: np.ndarray,
    n_game: int,
):
    """`61`'s arm-2 model with document 47 §2's game effect added.

    Written out here rather than patched into ``build_conversion_model`` so that
    round 2's model object stays byte-identical for anyone re-running it: this
    file adds a node, it does not edit the incumbent.
    """
    coords = {
        "feature": [f"x{index}" for index in range(matrix.shape[1])],
        "defence": range(n_defence),
        "qb": range(n_qb),
        "game": range(n_game),
    }
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", 0.0, 1.5)
        beta = pm.Normal("beta", 0.0, 1.0, dims="feature")
        sigma_d = pm.HalfNormal("sigma_d", 0.5)
        sigma_q = pm.HalfNormal("sigma_q", 0.5)
        sigma_g = pm.HalfNormal("sigma_g", 0.5)
        offset_d = pm.Normal("z_d", 0.0, 1.0, dims="defence")
        offset_q = pm.Normal("z_q", 0.0, 1.0, dims="qb")
        offset_g = pm.Normal("z_g", 0.0, 1.0, dims="game")
        u_d = pm.Deterministic("u_d", offset_d * sigma_d, dims="defence")
        v_q = pm.Deterministic("v_q", offset_q * sigma_q, dims="qb")
        w_g = pm.Deterministic("w_g", offset_g * sigma_g, dims="game")
        eta = (
            alpha + pm.math.dot(matrix, beta) + u_d[defence_codes] + v_q[qb_codes] + w_g[game_codes]
        )
        pm.Bernoulli("y", logit_p=eta, observed=outcome)
    return model


def variance_component(posterior, parameter: str, entity: str) -> dict:
    """One sigma on both scales, in document 46 §3's form."""
    draws = posterior[parameter].values.ravel()
    quantiles = np.quantile(draws, [0.055, 0.945])
    return {
        "entity": entity,
        "logit_mean": float(draws.mean()),
        "logit_eti89": [float(quantiles[0]), float(quantiles[1])],
        "pp_mean": float(draws.mean()) * LOGIT_SLOPE * 100,
        "pp_eti89": [float(q) * LOGIT_SLOPE * 100 for q in quantiles],
    }


def main() -> None:
    print("=== Round 3 Part B — game-clustering check (document 47 §2) ===")

    # Round 2's arm 2, read from its own output rather than transcribed: the
    # side-by-side and the beta gap both have to be against the fit that ran.
    round2 = json.loads((paths.RESEARCH_OUTPUT_DIR / ROUND2_JSON).read_text())
    round2_sigma_d = round2["arm2_conversion"]["variance_components"]["sigma_d"]

    frame = _power.build_worthy_frame()

    # Document 47 §2 / handoff Part B step 1: the same 2,969-row frame.
    if frame.model.height != EXPECTED_MODEL_ROWS:
        raise SystemExit(
            f"frame is {frame.model.height:,} rows, expected {EXPECTED_MODEL_ROWS:,} — "
            "this is not round 2's frame; stop and ask before fitting."
        )
    print(f"\n  frame asserted identical to round 2: {frame.model.height:,} rows")

    game_codes, n_game = _power._codes(frame.model, ["game_id"])
    print(
        f"  games with >= 1 worthy throw: {n_game:,}  "
        f"(median chances per game {np.median(np.bincount(game_codes)):.0f})"
    )

    print(
        f"\n  fitting: {frame.design_matrix.shape[0]:,} throws, "
        f"{frame.design_matrix.shape[1]} covariates, defence-season + QB-season + game, "
        f"{CHAINS} x {DRAWS} draws after {TUNE} tuning, target_accept {TARGET_ACCEPT}"
    )
    model = build_game_effect_model(
        frame.design_matrix,
        frame.outcome,
        frame.defence_season_codes,
        frame.n_defence_seasons,
        frame.qb_season_codes,
        frame.n_qb_seasons,
        game_codes,
        n_game,
    )
    with model:
        idata = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            random_seed=RANDOM_SEED,
            progressbar=False,
            nuts_sampler="nutpie",
            nuts={"target_accept": TARGET_ACCEPT},
        )

    # --- Gate C-1, sampler half, over every parameter -------------------------
    health = _confounds.sampler_health(
        idata, ["alpha", "beta", "sigma_d", "sigma_q", "sigma_g", "z_d", "z_q", "z_g"]
    )
    print("\n=== Gate C-1 (sampler half) ===")
    print(
        f"  C-1: divergences {health['divergences']}; "
        f"max r_hat {health['max_r_hat']:.4f} ({health['max_r_hat_parameter']}); "
        f"min ess_bulk {health['min_ess_bulk']:.0f} ({health['min_ess_bulk_parameter']}); "
        f"min ess_tail {health['min_ess_tail']:.0f} ({health['min_ess_tail_parameter']}); "
        f"{health['parameters_over_r_hat_bar'] + health['parameters_under_ess_bar']} of "
        f"{health['parameters_checked']} over a bar -> "
        f"{'PASS' if health['pass'] else 'FAIL'}"
    )

    posterior = idata["posterior"]

    # --- beta, and the gap against round 2 ------------------------------------
    beta_summary = az.summary(idata, var_names=["beta"])
    coefficients = []
    for index, name in enumerate(frame.feature_names):
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

    print("\n=== beta (logit scale, standardised covariates; * = 89% excludes zero) ===")
    for row in sorted(coefficients, key=lambda item: -abs(item["mean"])):
        marker = "*" if row["excludes_zero"] else " "
        print(
            f"    {marker} {row['name']:22s} {row['mean']:+.3f} "
            f"[{row['eti89'][0]:+.3f}, {row['eti89'][1]:+.3f}]  "
            f"odds x{row['odds_ratio']:.2f}"
        )

    round2_beta = {entry["name"]: entry["mean"] for entry in round2["arm2_conversion"]["beta"]}
    gaps = {row["name"]: abs(row["mean"] - round2_beta[row["name"]]) for row in coefficients}
    worst_name = max(gaps, key=gaps.__getitem__)
    alpha_mean = float(posterior["alpha"].values.mean())
    alpha_gap = abs(alpha_mean - round2["arm2_conversion"]["alpha_mean"])
    print(
        f"\n  max |d_beta| vs round 2 arm 2: {gaps[worst_name]:.4f} ({worst_name}); "
        f"|d_alpha| {alpha_gap:.4f}"
    )

    # --- variance components --------------------------------------------------
    variances = {
        "sigma_d": variance_component(posterior, "sigma_d", "defence-season"),
        "sigma_q": variance_component(posterior, "sigma_q", "QB-season"),
        "sigma_g": variance_component(posterior, "sigma_g", "game"),
    }
    print("\n=== variance components (sigma_p ~ sigma_logit x pbar(1-pbar)) ===")
    for parameter, entry in variances.items():
        print(
            f"    {parameter} ({entry['entity']}): logit {entry['logit_mean']:.3f} "
            f"[{entry['logit_eti89'][0]:.3f}, {entry['logit_eti89'][1]:.3f}]  "
            f"= {entry['pp_mean']:.2f} pp [{entry['pp_eti89'][0]:.2f}, "
            f"{entry['pp_eti89'][1]:.2f}] on the probability scale"
        )

    sigma_d = variances["sigma_d"]
    sigma_g = variances["sigma_g"]
    print(
        f"\n  sigma_d side by side — round 2 (no game effect): "
        f"{round2_sigma_d['pp_mean']:.2f} pp "
        f"[{round2_sigma_d['pp_eti89'][0]:.2f}, {round2_sigma_d['pp_eti89'][1]:.2f}]"
        f"  ->  with game effect: {sigma_d['pp_mean']:.2f} pp "
        f"[{sigma_d['pp_eti89'][0]:.2f}, {sigma_d['pp_eti89'][1]:.2f}]"
    )

    # --- document 47 §2's pre-committed reading, applied ----------------------
    upper = sigma_d["pp_eti89"][1]
    sigma_g_excludes_zero = sigma_g["pp_eti89"][0] > 0.0
    if upper >= REFERENCE_BOUND_PP:
        row = "within-season spread survives clustering; R-2 wording stands"
        verdict = "survives"
    elif sigma_g_excludes_zero:
        row = "the within-season finding is game clustering, not the defence; wording reverts"
        verdict = "clustering"
    else:
        row = "the finding was fragile to a nuisance term; reported as such, wording reverts"
        verdict = "fragile"

    print("\n=== document 47 §2 reading ===")
    print(
        f"sigma_d with game effect: {sigma_d['pp_mean']:.2f} pp "
        f"[{sigma_d['pp_eti89'][0]:.2f}, {sigma_d['pp_eti89'][1]:.2f}]; "
        f"upper {upper:.2f} {'>=' if upper >= REFERENCE_BOUND_PP else '<'} "
        f"{REFERENCE_BOUND_PP:.2f} -> {row}"
    )
    print(
        f"sigma_g: {sigma_g['pp_mean']:.2f} pp "
        f"[{sigma_g['pp_eti89'][0]:.2f}, {sigma_g['pp_eti89'][1]:.2f}]"
        f" ({'excludes' if sigma_g_excludes_zero else 'includes'} zero)"
    )

    out = paths.RESEARCH_OUTPUT_DIR / "66_dropped_pick_game_effect.json"
    out.write_text(
        json.dumps(
            {
                "check": "document 47 §2 — game clustering",
                "random_seed": RANDOM_SEED,
                "draws": DRAWS,
                "tune": TUNE,
                "chains": CHAINS,
                "target_accept": TARGET_ACCEPT,
                "reference_bound_pp": REFERENCE_BOUND_PP,
                "frame": {
                    "rows": int(frame.model.height),
                    "defence_seasons": int(frame.n_defence_seasons),
                    "qb_seasons": int(frame.n_qb_seasons),
                    "games": int(n_game),
                    "identical_to_round2": True,
                },
                "guards": frame.guards,
                "sampler": health,
                "beta": coefficients,
                "alpha_mean": alpha_mean,
                "vs_round2_beta": {
                    "max_abs_beta_gap": gaps[worst_name],
                    "max_abs_beta_gap_parameter": worst_name,
                    "alpha_gap": alpha_gap,
                },
                "variance_components": variances,
                "round2_sigma_d": round2_sigma_d,
                "reading": {
                    "verdict": verdict,
                    "row": row,
                    "sigma_d_upper_pp": upper,
                    "sigma_g_excludes_zero": sigma_g_excludes_zero,
                },
            },
            indent=2,
        )
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
