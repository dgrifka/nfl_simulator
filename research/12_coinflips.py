"""Phase 3, step 3 — the coin-flip candidate round, run against pre-registered gates.

Gate A, the branch-point argument, is settled in `docs/research/09-coinflip-candidates.md`
§2 and no code can change it — it is a mechanism argument, and document 05 §2 is
explicit that no statistic can detect the absence of a branch point. This script
runs Gate B for every candidate, including the four that failed Gate A, because
document 05 §3 prints a parenthesised `w` for penalties for the same reason: an
argument a reader can see the numbers behind is one they can disagree with.

Thresholds are fixed by §4 of that document, committed at `1c585e2` before this
script produced a result.

    Gate C-1  sampler health, plus agreement with the exact grid posterior
    Gate C-2  is the entity spread below what a skill-free league produces?
    Gate C-3  honesty — is either outcome interpretable at this sample size?

    uv run python research/12_coinflips.py
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

from _betabinom_grid import fit_grid  # noqa: E402

_power = import_module("12_coinflips_power")

from nfl_simulator import paths  # noqa: E402

RANDOM_SEED = 20260817
CHAINS = 4
TUNE = 1000
DRAWS = 1000
TARGET_ACCEPT = 0.9

# Pre-registered in document 09 §4, from the null simulation. Hard-coded rather
# than read back, so re-running the power script cannot silently move a
# committed gate.
GATE_C2_THRESHOLDS_PP = {
    "drops_team": 0.698,
    "drops_receiver": 1.081,
    "fourth_down": 4.977,
    "two_point": 10.538,
    "onside_recovery": 9.317,
    "extra_point_kicker": 1.840,
}

# Document 09 §2. Settled by mechanism, not by any number below.
GATE_A_PASS = {"onside_recovery", "extra_point_kicker"}

GRID_TOLERANCE_PP = 0.01


def eti89(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 5.5)), float(np.percentile(values, 94.5))]


def fit_candidate(name: str, counts: pl.DataFrame) -> dict:
    """Marginalized beta-binomial, the parameterization document 04 settled on."""
    n = counts["n"].to_numpy()
    k = counts["k"].to_numpy()
    league_rate = float(k.sum() / n.sum())
    print(
        f"\n{'=' * 72}\n{name}: {counts.height} entities, {int(k.sum()):,} of "
        f"{int(n.sum()):,}, rate {league_rate:.4%}\n{'=' * 72}"
    )

    with pm.Model():
        mu = pm.Beta("mu", alpha=2.0, beta=2.0)
        log_kappa = pm.Normal("log_kappa", mu=4.0, sigma=2.0)
        kappa = pm.Deterministic("kappa", pm.math.exp(log_kappa))
        pm.Deterministic("population_sd", pm.math.sqrt(mu * (1.0 - mu) / (kappa + 1.0)))
        pm.BetaBinomial("successes", n=n, alpha=mu * kappa, beta=(1.0 - mu) * kappa, observed=k)
        idata = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
            progressbar=False,
        )

    summary = az.summary(idata)
    health = {
        "divergences": int(idata["sample_stats"]["diverging"].sum().item()),
        "max_r_hat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "min_ess_tail": float(summary["ess_tail"].min()),
    }

    population_sd = idata["posterior"]["population_sd"].values.ravel()
    bounds_pp = [b * 100 for b in eti89(population_sd)]
    mean_pp = float(population_sd.mean()) * 100

    # Independent convergence check: the exact grid posterior of the same model.
    grid = fit_grid(n, k)
    grid_summary = grid.summary()
    grid_mean_pp = grid_summary["population_sd_mean"] * 100
    grid_agrees = abs(grid_mean_pp - mean_pp) < GRID_TOLERANCE_PP

    health["grid_agrees"] = bool(grid_agrees)
    health["grid_edge_mass"] = grid.edge_mass()
    health["pass"] = bool(
        health["divergences"] == 0
        and health["max_r_hat"] < 1.01
        and health["min_ess_bulk"] > 400
        and health["min_ess_tail"] > 400
        and grid_agrees
    )
    print(
        f"  Gate C-1 (sampler health): {'PASS' if health['pass'] else 'FAIL'} — "
        f"divergences {health['divergences']}, r_hat {health['max_r_hat']:.4f}, "
        f"ess_bulk {health['min_ess_bulk']:.0f}; "
        f"grid {grid_mean_pp:.4f} pp vs NUTS {mean_pp:.4f} pp"
    )

    threshold = GATE_C2_THRESHOLDS_PP[name]
    negligible = bool(bounds_pp[1] < threshold)
    power_report = _power_lookup()[name]
    interpretable = bool(power_report["resolvable"])

    print(
        f"  population SD {mean_pp:.4f} pp [{bounds_pp[0]:.4f}, {bounds_pp[1]:.4f}] "
        f"= {mean_pp / (league_rate * 100):.1%} relative"
    )
    print(
        f"  Gate C-2 (spread negligible): {'PASS' if negligible else 'FAIL'} — "
        f"89% upper bound {bounds_pp[1]:.4f} pp vs threshold {threshold:.4f} pp"
    )
    print(
        f"  Gate C-3 (interpretable):     "
        f"{'PASS' if interpretable else 'FAIL'} — power at the 12.5% reference "
        f"{power_report['power_at_reference']:.3f}"
    )

    gate_a = name in GATE_A_PASS
    if not gate_a:
        verdict = "NOT NEUTRALIZED — fails Gate A, no branch point"
    elif not interpretable:
        verdict = "NOT NEUTRALIZED — Gate A passes but the spread is unresolvable; deny by default"
    elif negligible:
        verdict = "NEUTRALIZE IN FULL at the league rate (w ~ 0)"
    else:
        verdict = "NEUTRALIZE PARTIALLY at the entity's shrunk rate"
    print(f"  => {verdict}")

    return {
        "name": name,
        "entities": int(counts.height),
        "opportunities": int(n.sum()),
        "successes": int(k.sum()),
        "league_rate": league_rate,
        "population_sd_pp": mean_pp,
        "population_sd_eti89_pp": bounds_pp,
        "relative_spread": mean_pp / (league_rate * 100),
        "grid_population_sd_pp": grid_mean_pp,
        "gate_c1_sampler_health": health,
        "gate_c2_threshold_pp": threshold,
        "gate_c2_spread_negligible": negligible,
        "gate_c3_interpretable": interpretable,
        "power_at_reference": power_report["power_at_reference"],
        "gate_a_branch_point": gate_a,
        "verdict": verdict,
    }


_POWER_CACHE: dict | None = None


def _power_lookup() -> dict:
    global _POWER_CACHE
    if _POWER_CACHE is None:
        with (paths.RESEARCH_OUTPUT_DIR / "12_coinflips_power.json").open() as handle:
            _POWER_CACHE = json.load(handle)["candidates"]
    return _POWER_CACHE


def main() -> None:
    paths.ensure_data_dirs()
    counts = _power.candidate_counts()
    results = {name: fit_candidate(name, frame) for name, frame in counts.items()}

    print(f"\n{'=' * 72}\nVERDICTS\n{'=' * 72}")
    table = pl.DataFrame(
        [
            {
                "candidate": name,
                "gate_a": report["gate_a_branch_point"],
                "sd_pp": round(report["population_sd_pp"], 4),
                "upper_pp": round(report["population_sd_eti89_pp"][1], 4),
                "threshold_pp": report["gate_c2_threshold_pp"],
                "c2": report["gate_c2_spread_negligible"],
                "c3": report["gate_c3_interpretable"],
                "verdict": report["verdict"],
            }
            for name, report in results.items()
        ]
    )
    with pl.Config(tbl_cols=-1, fmt_str_lengths=64, tbl_width_chars=250):
        print(table)

    out = paths.RESEARCH_OUTPUT_DIR / "12_coinflips.json"
    with out.open("w") as handle:
        json.dump(
            {"candidates": results, "random_seed": RANDOM_SEED}, handle, indent=2, default=str
        )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
