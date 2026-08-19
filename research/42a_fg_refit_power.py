"""Phase 7, task 1 — re-derive the two inherited null bounds at the refit's n.

Document 27 §7b commits this before `research/42_fg_refit.py` exists. Gates R-3
(wind) and R-7 (temperature) read against bounds that document 05b §10 obtained
by simulating 400 datasets under a true zero **at the incumbent's sample size**.
Those bounds are properties of the design, and the refit's design loses 192
blocked field goals. This script reruns the identical null simulation on the
non-blocked design matrix — same seed, same datasets, same simulating
parameters — so the thresholds match the design they are applied to.

It reuses `research/13_fg_weather_power.py` rather than reimplementing it: same
sanitize rules, same design matrix, same Newton-Raphson logistic fitter. The
only difference is one filter, and it is passed as an argument to that script's
own loader.

    uv run python research/42a_fg_refit_power.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_power = import_module("13_fg_weather_power")

from nfl_simulator import paths  # noqa: E402

RANDOM_SEED = _power.RANDOM_SEED  # the null simulation's seed is inherited too
DATASETS = _power.DATASETS


def null_bounds(attempts: pl.DataFrame, report: dict) -> dict:
    """The 10th percentile of the wind upper bound and the 90th of the temp lower.

    Both are exactly document 05b §10's construction: a true-zero design clears
    its own threshold 10% of the time, by definition of the percentile.
    """
    x, names = _power.design_matrix(attempts, report)
    kicker_levels = sorted(attempts["kicker_season"].unique().to_list())
    lookup = {level: i for i, level in enumerate(kicker_levels)}
    kicker_idx = np.array([lookup[v] for v in attempts["kicker_season"].to_list()])
    indices = {"wind": names.index("beta_wind"), "temp": names.index("beta_temp")}

    wind_upper = np.empty(DATASETS)
    temp_lower = np.empty(DATASETS)
    for i in range(DATASETS):
        rng = np.random.default_rng(RANDOM_SEED + i)
        y = _power.simulate(x, kicker_idx, len(kicker_levels), 0.0, rng)
        fitted = _power.fit_and_intervals(x, y, indices)
        wind_upper[i] = fitted["wind"][2]
        temp_lower[i] = fitted["temp"][1]

    return {
        "attempts": int(attempts.height),
        "kicker_seasons": len(kicker_levels),
        "wind_threshold": float(np.percentile(wind_upper, 10)),
        "temp_threshold": float(np.percentile(temp_lower, 90)),
        "null_mean_wind_upper": float(wind_upper.mean()),
        "null_mean_temp_lower": float(temp_lower.mean()),
    }


def main() -> None:
    paths.ensure_data_dirs()

    # Both arms are run, because a re-derived threshold is only readable next to
    # the one it replaces — and because reproducing the published bound on the
    # full population is the check that this script is running the same
    # instrument document 05b ran.
    full_attempts, full_report = _power.load_attempts()
    refit_attempts, refit_report = _power.load_attempts(exclude_blocked=True)

    print(
        f"full population   {full_attempts.height:,} attempts, "
        f"wind centre {full_report['mean_outdoor_wind']:.4f} mph, "
        f"temp centre {full_report['mean_outdoor_temp']:.4f} F"
    )
    print(
        f"refit population  {refit_attempts.height:,} attempts, "
        f"wind centre {refit_report['mean_outdoor_wind']:.4f} mph, "
        f"temp centre {refit_report['mean_outdoor_temp']:.4f} F"
    )
    print(f"  blocked field goals removed: {full_attempts.height - refit_attempts.height}\n")

    print(f"=== Null simulation, {DATASETS} datasets per arm ===")
    full = null_bounds(full_attempts, full_report)
    print(
        f"  full population : wind {full['wind_threshold']:+.7f}, "
        f"temp {full['temp_threshold']:+.7f}"
    )
    refit = null_bounds(refit_attempts, refit_report)
    print(
        f"  refit population: wind {refit['wind_threshold']:+.7f}, "
        f"temp {refit['temp_threshold']:+.7f}"
    )

    with (paths.RESEARCH_OUTPUT_DIR / "13_fg_weather_power.json").open() as handle:
        published = json.load(handle)
    inherited = {
        "wind_threshold": published["gate_w3_threshold"],
        "temp_threshold": published["gate_w7_temp_threshold"],
    }
    reproduced = (
        abs(full["wind_threshold"] - inherited["wind_threshold"]) < 1e-12
        and abs(full["temp_threshold"] - inherited["temp_threshold"]) < 1e-12
    )
    print(
        f"\n  published bounds : wind {inherited['wind_threshold']:+.7f}, "
        f"temp {inherited['temp_threshold']:+.7f}"
    )
    print(
        f"  full-population arm reproduces the published bounds exactly: "
        f"{'YES' if reproduced else 'NO'}"
    )
    if not reproduced:
        print(
            "  -> the instrument has drifted since document 05b ran it. The re-derived\n"
            "     refit bounds are still the ones document 27 §7b commits to, and this\n"
            "     line is the disclosure that the two rounds are not byte-identical."
        )

    print("\n=== Thresholds document 27's gates read against ===")
    print(
        f"  R-3 (beta_wind 89% upper bound must fall below): "
        f"{refit['wind_threshold']:+.7f}   "
        f"(inherited {inherited['wind_threshold']:+.7f})"
    )
    print(
        f"  R-7 (beta_temp 89% lower bound must clear):      "
        f"{refit['temp_threshold']:+.7f}   "
        f"(inherited {inherited['temp_threshold']:+.7f})"
    )

    payload = {
        "datasets": DATASETS,
        "random_seed": RANDOM_SEED,
        "full_population": full,
        "refit_population": refit,
        "inherited_published": inherited,
        "full_arm_reproduces_published": bool(reproduced),
        "gate_r3_threshold": refit["wind_threshold"],
        "gate_r7_threshold": refit["temp_threshold"],
    }
    out = paths.RESEARCH_OUTPUT_DIR / "42a_fg_refit_power.json"
    with out.open("w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
