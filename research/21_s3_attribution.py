"""Phase 4, step 3 — whose skill is leverage timing? Run against document 13's rules.

Design, statistics and null bounds are fixed by
`docs/research/13-s3-attribution.md`, committed before this script produced a
result. Nothing here chooses anything.

Document 08 §9 found that the gap between a team's win-probability contribution
and its expected-points contribution persists at split-half ``r = +0.180``, and
survives a control for playing close games at ``+0.144``. Its defect register
left the mechanism open: *"why teams differ — coaching, quarterback, situational
scheme — is untested."*

    Gate A-1  sampler health, with a RELATIVE grid-versus-NUTS tolerance
    Gate A-2  the reporting rule — no pass/fail, per document 05 §7's convention
    Gate A-3  the secondary arm: does the attribution survive the game-state
              control document 08 §9 used?

    uv run python research/21_s3_attribution.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_power = import_module("21_s3_attribution_power")
_grid = import_module("_crossed_gaussian_grid")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.ingest import PBP_SEASONS, load_pbp  # noqa: E402

RANDOM_SEED = _power.RANDOM_SEED
CHAINS, TUNE, DRAWS = 4, 3000, 3000
TARGET_ACCEPT = 0.9

# Document 09 §9's corrective: a convergence tolerance is RELATIVE to the
# quantity, never an absolute number borrowed from a different scale.
GRID_RELATIVE_TOLERANCE = 0.05

# Filled in from `docs/research/13-s3-attribution.md`. Hard-coded rather than read
# back, so a re-run of the power script cannot silently move a committed bound.
THRESHOLDS_PATH = paths.RESEARCH_OUTPUT_DIR / "21_s3_attribution_power.json"


def fit_nuts(codes: list[np.ndarray], sizes: list[int], y: np.ndarray) -> dict:
    """The crossed model in PyMC, as the convergence cross-check on the grid.

    3,000 tune / 3,000 draws rather than the usual 1,000, because document 05 §8
    recorded exactly this geometry mixing slowly: two crossed scales trade off
    along a ridge the chains cross slowly, which shows up as low effective sample
    size with **zero** divergences. More draws is the honest fix; raising
    ``target_accept`` to quiet the warning is what document 03 §5 forbids.
    """
    import arviz as az
    import pymc as pm

    with pm.Model(coords={"qb": range(sizes[0]), "coach": range(sizes[1])}):
        intercept = pm.Normal("intercept", mu=0.0, sigma=0.5)
        sigma_qb = pm.HalfNormal("sigma_qb", sigma=0.2)
        sigma_coach = pm.HalfNormal("sigma_coach", sigma=0.2)
        sigma_e = pm.HalfNormal("sigma_e", sigma=1.0)
        offset_qb = pm.Normal("z_qb", mu=0.0, sigma=1.0, dims="qb")
        offset_coach = pm.Normal("z_coach", mu=0.0, sigma=1.0, dims="coach")
        pm.Normal(
            "s3",
            mu=intercept
            + (offset_qb * sigma_qb)[codes[0]]
            + (offset_coach * sigma_coach)[codes[1]],
            sigma=sigma_e,
            observed=y,
        )
        idata = pm.sample(
            DRAWS,
            tune=TUNE,
            chains=CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
            progressbar=False,
        )

    summary = az.summary(idata)
    posterior = idata["posterior"]
    return {
        "divergences": int(idata["sample_stats"]["diverging"].values.sum()),
        "max_r_hat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "min_ess_tail": float(summary["ess_tail"].min()),
        "sigma_qb": float(posterior["sigma_qb"].values.mean()),
        "sigma_coach": float(posterior["sigma_coach"].values.mean()),
        "sigma_e": float(posterior["sigma_e"].values.mean()),
        "p_qb_exceeds_coach": float(
            (posterior["sigma_qb"].values > posterior["sigma_coach"].values).mean()
        ),
    }


def report_arm(frame: pl.DataFrame, label: str) -> dict:
    """The grid posterior for one arm, printed and returned."""
    codes, sizes, levels = _power.encode(frame)
    y = frame["s3"].to_numpy()
    design, zty, yty, oney = _grid.prepare(codes, sizes, y)
    result = _grid.fit(design, zty, yty, oney)
    print(f"  {label}: {frame.height:,} team-games, {sizes[0]} quarterbacks, {sizes[1]} coaches")
    for name, key in (("quarterback", "sigma_a"), ("coach", "sigma_b")):
        entry = result[key]
        print(
            f"    {name:12s} sigma {entry['mean']:.5f} "
            f"[{entry['eti89_lb']:.5f}, {entry['eti89_ub']:.5f}]"
        )
    print(f"    P(quarterback spread > coach spread) = {result['p_a_exceeds_b']:.3f}")
    print(f"    grid edge mass {result['edge_mass']:.2e}")
    return {"result": result, "codes": codes, "sizes": sizes, "levels": levels, "y": y}


def main() -> None:
    paths.ensure_data_dirs()
    with THRESHOLDS_PATH.open() as handle:
        thresholds = json.load(handle)
    bounds = thresholds["null_bounds"]
    power_rows = thresholds["power"]

    pbp = load_pbp(PBP_SEASONS, columns=_power.COLUMNS)

    print(f"{'=' * 72}\nPRIMARY ARM — every play the team had the ball for\n{'=' * 72}")
    frame, slope = _power.s3_dataset(pbp, competitive_only=False)
    primary = report_arm(frame, "primary")

    print(f"\n{'=' * 72}\nGATE A-1 — sampler health and the grid cross-check\n{'=' * 72}")
    nuts = fit_nuts(primary["codes"], primary["sizes"], primary["y"])
    health_pass = bool(
        nuts["divergences"] == 0
        and nuts["max_r_hat"] < 1.01
        and nuts["min_ess_bulk"] > 400
        and nuts["min_ess_tail"] > 400
    )
    gaps = {
        "quarterback": abs(primary["result"]["sigma_a"]["mean"] - nuts["sigma_qb"])
        / nuts["sigma_qb"],
        "coach": abs(primary["result"]["sigma_b"]["mean"] - nuts["sigma_coach"])
        / nuts["sigma_coach"],
        "residual": abs(primary["result"]["sigma_e"]["mean"] - nuts["sigma_e"]) / nuts["sigma_e"],
    }
    tolerance_pass = all(gap <= GRID_RELATIVE_TOLERANCE for gap in gaps.values())
    print(
        f"  NUTS: {nuts['divergences']} divergences, r_hat {nuts['max_r_hat']:.4f}, "
        f"ess_bulk {nuts['min_ess_bulk']:.0f}, ess_tail {nuts['min_ess_tail']:.0f} — "
        f"{'PASS' if health_pass else 'FAIL'}"
    )
    for name, gap in gaps.items():
        print(f"  grid vs NUTS, {name:12s} relative gap {gap:.2%}")
    print(
        f"  relative tolerance {GRID_RELATIVE_TOLERANCE:.0%}: "
        f"{'PASS' if tolerance_pass else 'FAIL'}"
    )

    print(f"\n{'=' * 72}\nGATE A-2 — the reporting rule\n{'=' * 72}")
    factors = []
    for name, key, bound_key in (
        ("Quarterback", "sigma_a", "quarterback"),
        ("Head coach", "sigma_b", "coach"),
    ):
        entry = primary["result"][key]
        bound = bounds[bound_key]
        clears = bool(entry["eti89_lb"] > bound)
        factors.append(
            {
                "label": name,
                "mean": entry["mean"],
                "eti89": [entry["eti89_lb"], entry["eti89_ub"]],
                "null_bound": bound,
                "clears_null_bound": clears,
            }
        )
        print(
            f"  {name:12s} sigma {entry['mean']:.5f} "
            f"[{entry['eti89_lb']:.5f}, {entry['eti89_ub']:.5f}]  "
            f"null bound {bound:.5f}  -> {'clears' if clears else 'does NOT clear'}"
        )

    qb_clears, coach_clears = factors[0]["clears_null_bound"], factors[1]["clears_null_bound"]
    if qb_clears and not coach_clears:
        verdict = "The leverage-timing gap belongs to the QUARTERBACK"
    elif coach_clears and not qb_clears:
        verdict = "The leverage-timing gap belongs to the COACH"
    elif qb_clears and coach_clears:
        verdict = "SHARED — both factors clear the design's null bound"
    else:
        verdict = "UNRESOLVED — neither factor clears the design's null bound"
    print(f"\n  VERDICT: {verdict}")
    print(
        "  (the rule was committed in document 13 before this ran: a claim that the gap\n"
        "  'belongs to' one factor requires that factor's interval to clear the bound AND\n"
        "  the other's to fail to)"
    )

    print(f"\n{'=' * 72}\nGATE A-3 — the game-state control (secondary arm)\n{'=' * 72}")
    print(
        "  Document 08 §9 found that a fifth of S3's persistence was playing close games.\n"
        "  If the attribution is real it should survive restricting to one-score states."
    )
    competitive_frame, competitive_slope = _power.s3_dataset(pbp, competitive_only=True)
    secondary = report_arm(competitive_frame, "competitive plays only")

    # Football units for the write-up: S3 is win-probability points per game.
    def readable(entry: dict) -> str:
        return f"{entry['mean'] * 100:.2f} percentage points of win probability per game"

    print("\n  In football:")
    print(f"    a one-SD quarterback: {readable(primary['result']['sigma_a'])}")
    print(f"    a one-SD coach:       {readable(primary['result']['sigma_b'])}")

    results = {
        "primary": {
            "n_team_games": int(frame.height),
            "quarterbacks": primary["sizes"][0],
            "coaches": primary["sizes"][1],
            "wpa_per_epa_slope": slope,
            "quarterback": primary["result"]["sigma_a"],
            "coach": primary["result"]["sigma_b"],
            "residual": primary["result"]["sigma_e"],
            "p_qb_exceeds_coach": primary["result"]["p_a_exceeds_b"],
            "grid_edge_mass": primary["result"]["edge_mass"],
        },
        "secondary_competitive_only": {
            "n_team_games": int(competitive_frame.height),
            "wpa_per_epa_slope": competitive_slope,
            "quarterback": secondary["result"]["sigma_a"],
            "coach": secondary["result"]["sigma_b"],
            "p_qb_exceeds_coach": secondary["result"]["p_a_exceeds_b"],
        },
        "gate_a1_sampler_health": {
            **nuts,
            "health_pass": health_pass,
            "relative_gaps": gaps,
            "relative_tolerance": GRID_RELATIVE_TOLERANCE,
            "tolerance_pass": tolerance_pass,
        },
        "gate_a2_reporting_rule": {
            "factors": factors,
            "verdict": verdict,
            "null_bounds": bounds,
        },
        "power": power_rows,
        "design": thresholds["design_primary"],
        "figure": {
            "headline": verdict,
            "factors": factors,
        },
        "random_seed": RANDOM_SEED,
    }
    out = paths.RESEARCH_OUTPUT_DIR / "21_s3_attribution.json"
    with out.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
