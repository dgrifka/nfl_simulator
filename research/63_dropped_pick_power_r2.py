"""Part A of round 2 — power on the floorless frame, before any threshold.

Amendment A-1 of `docs/research/45-dropped-pick-pooling-prereg.md` removes
document 43 §4's ``MIN_QB_WORTHY = 20`` floor. Round 1 discovered that the floor,
written as a *unit definition*, acts as a **row filter** in a crossed design: a
throw by a QB-season under the floor has no level to belong to and leaves with
it. That cut 2,969 worthy throws to 1,145 and the median defence-season from 22
chances to 7, and it is the whole arithmetic reason round 1's residual designs
failed Gate C-3 (power 0.362 / 0.555 / 0.578 against the 0.80 bar). The floor was
never needed: a hierarchical model already pools thin levels toward the mean.

So this script re-runs round 1's power harness, unchanged, on the frame arm 2
actually fits — **every** QB-season with at least one worthy throw is a level:

    3. residual, defence-season x QB-season      crossed Gaussian grid
    4. residual, defence pooled x QB-season      crossed Gaussian grid
    5. residual, QB-season (sigma_q)             design 3, reading sigma_b

Arm 1's two rate designs are **not** re-run: their frames never saw the floor,
their numbers stand in document 44 §3, and document 45 §3 says so.

Everything else is imported from `research/61_dropped_pick_power.py` rather than
copied — the frame builder, the guards, the simulation, the crossed-grid task,
the summariser and the report printer are round 1's, so the only thing that
changes between the two power tables is the frame.

``beta_hat`` is **reused** from `research/outputs/61_beta_hat.json`, not refitted.
It came from the full 2,969-row frame (never from the floored one), and Part B of
round 1 re-fitted the same model with the same seed and reproduced it to
max |d beta| 0.0044. Refitting here would only risk moving the fixed effects the
thresholds are built around, for no gain.

    uv run python research/63_dropped_pick_power_r2.py
    uv run python research/63_dropped_pick_power_r2.py --serial   # no worker pool

Parallel and serial runs produce identical numbers: every simulated dataset draws
from ``np.random.default_rng(seed + index)``, so nothing depends on the pool.

Nothing in `src/nfl_simulator/` changes. Document 32's closure is untouched by any
number below.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from importlib import import_module
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

_power = import_module("61_dropped_pick_power")

from nfl_simulator import paths  # noqa: E402

DATASETS = _power.DATASETS
RANDOM_SEED = _power.RANDOM_SEED
REFERENCE_RELATIVE = _power.REFERENCE_RELATIVE
MIN_POWER = _power.MIN_POWER

# Document 45 §4: a fit slower than this is *reported*, never traded away.
# `DATASETS` stays 400 regardless — wall clock is not a reason to change an
# instrument.
SLOW_FIT_SECONDS = 5.0


def timed_residual_task(spec: dict) -> dict:
    """Round 1's `_residual_task`, wrapped in a clock.

    The task itself is imported unchanged; the wrapper exists only so document
    45 §4 can carry a measured per-fit cost instead of an estimate. One cell is
    ``DATASETS`` fits, so ``seconds / DATASETS`` is the mean cost of one crossed
    fit in that cell (the median *across cells* of that mean is what gets
    reported — a per-fit median inside a cell would need the task rewritten,
    which would stop it being round 1's task).
    """
    started = time.perf_counter()
    result = _power._residual_task(spec)
    result["seconds"] = time.perf_counter() - started
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", action="store_true", help="run every cell in this process")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    paths.ensure_data_dirs()
    started = time.time()
    frame = _power.build_worthy_frame()

    # --- the fixed effects the residual designs simulate around ---------------
    beta_path = paths.RESEARCH_OUTPUT_DIR / "61_beta_hat.json"
    if not beta_path.exists():
        raise SystemExit(
            f"{beta_path} is missing — round 2 reuses round 1's beta_hat (document 45 §4) "
            "and must not refit it"
        )
    beta_hat = json.loads(beta_path.read_text())
    print(f"\nreusing round 1's arm-2 fixed effects from {beta_path.name} (document 45 §4)")

    # --- A-1: the residual frame IS the model frame ---------------------------
    # No `residual_frame()` call, and that absence is the amendment. Every row arm
    # 2 fits is a row arm 3 sees, so the two frames are the same object.
    residual = frame.model
    same_rows = residual.height == frame.model.height
    print(f"\narm3 rows == arm2 rows: {same_rows}  ({residual.height:,} throws, no MIN_QB_WORTHY)")
    if not same_rows:
        raise SystemExit(
            "arm 3's frame differs from arm 2's — document 45 §2's stop-and-ask. Stop and ask."
        )

    eta = _power.linear_predictor(beta_hat, residual, frame.model)
    p_hat = 1.0 / (1.0 + np.exp(-eta))
    defence_season_codes, n_defence_season = _power._codes(residual, ["season", "defteam"])
    defence_pooled_codes, n_defence_pooled = _power._codes(residual, ["defteam"])
    qb_codes, n_qb = _power._codes(residual, ["season", "passer_player_id"])
    conversion_rate = float(residual["interception"].mean())

    chances_per_defence_season = np.bincount(defence_season_codes, minlength=n_defence_season)
    chances_per_defence_pooled = np.bincount(defence_pooled_codes, minlength=n_defence_pooled)
    throws_per_qb_season = np.bincount(qb_codes, minlength=n_qb)

    print(
        f"levels: {n_defence_season} defence-seasons, {n_defence_pooled} defences, "
        f"{n_qb} QB-seasons"
    )
    print(
        f"  chances per defence-season: median {np.median(chances_per_defence_season):.0f}, "
        f"min {chances_per_defence_season.min()}, max {chances_per_defence_season.max()}  "
        f"(round 1 with the floor: median 7)"
    )
    print(
        f"  chances per pooled defence: median {np.median(chances_per_defence_pooled):.0f}; "
        f"worthy throws per QB-season: median {np.median(throws_per_qb_season):.0f}, "
        f"min {throws_per_qb_season.min()}, max {throws_per_qb_season.max()}"
    )
    print(
        f"  conversion in the residual frame {conversion_rate:.4f}; mean p_hat {p_hat.mean():.4f}"
    )

    # --- specs ---------------------------------------------------------------
    # The design indices are round 1's (2, 3, 4), so every cell draws the same
    # seed stream it drew in round 1 and the two power tables differ only by the
    # frame they were simulated on.
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

    metas: dict[str, dict] = {}
    residual_jobs: list[dict] = []
    for name, code_a, size_a, code_b, size_b, factor, index in crossed_designs:
        meta, specs = _power.residual_specs(
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
        # design 3's null — the same simulation with both effects at zero.
        if name == "residual_qb_season_sigma_q":
            specs = [spec for spec in specs if spec["key"][1] != "null"]
        residual_jobs.extend(specs)

    print(f"\n=== simulating {len(residual_jobs)} residual cells at {DATASETS} datasets each ===")

    if args.serial or args.workers <= 1:
        results = [timed_residual_task(job) for job in residual_jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(timed_residual_task, residual_jobs))
    print(f"  residual cells done ({time.time() - started:.0f}s elapsed)")

    # --- per-fit cost, reported not traded away -------------------------------
    per_fit = sorted(result["seconds"] / DATASETS for result in results)
    median_per_fit = float(np.median(per_fit))
    print(
        f"\nper-fit seconds (cell mean): median {median_per_fit:.2f}s, "
        f"min {per_fit[0]:.2f}s, max {per_fit[-1]:.2f}s over {len(per_fit)} cells"
    )
    if median_per_fit > SLOW_FIT_SECONDS:
        print(
            f"  NOTE: median per-fit cost exceeds {SLOW_FIT_SECONDS:.0f}s. "
            f"DATASETS stays {DATASETS} (document 45 §4) — the cost is reported, not traded."
        )

    # --- collect -------------------------------------------------------------
    collected: dict[str, dict[object, np.ndarray]] = {name: {} for name in metas}
    for result in results:
        name, scenario = result["key"]
        factor = metas[name]["powered_factor"]
        collected[name][scenario] = np.asarray(result[f"bounds_{factor}"])
        if name == "residual_defence_season_x_qb_season" and scenario == "null":
            collected["residual_qb_season_sigma_q"]["null"] = np.asarray(result["bounds_b"])

    reports = {name: _power.summarise(metas[name], collected[name]) for name in metas}
    for report in reports.values():
        _power.print_report(report)

    print("\n=== document 45 §4 table ===")
    round1 = {
        "residual_defence_season_x_qb_season": (9.410, 0.362),
        "residual_defence_pooled_x_qb_season": (8.086, 0.555),
        "residual_qb_season_sigma_q": (8.075, 0.578),
    }
    for report in reports.values():
        cells = [
            "*impossible*" if row["impossible"] else f"{row['power']:.2f}"
            for row in report["power"]
        ]
        print(
            f"| {report['name']} | {report['gate_threshold_pp']:.2f} pp | "
            + " | ".join(cells)
            + f" | {'Yes' if report['resolvable'] else 'No'} |"
        )
    print("\n=== round 1 -> round 2, at the 12.5% reference ===")
    for name, report in reports.items():
        old_threshold, old_power = round1[name]
        print(
            f"  {name}: threshold {old_threshold:.2f} -> "
            f"{report['gate_threshold_pp']:.2f} pp; power {old_power:.3f} -> "
            f"{report['power_at_reference']:.3f} "
            f"({'RESOLVABLE' if report['resolvable'] else 'UNRESOLVABLE'})"
        )

    elapsed = time.time() - started
    out = paths.RESEARCH_OUTPUT_DIR / "63_dropped_pick_power_r2.json"
    out.write_text(
        json.dumps(
            {
                "amendment": "document 45 A-1 — MIN_QB_WORTHY removed; no floor on the gate arm",
                "datasets_per_scenario": DATASETS,
                "relative_scenarios": list(_power.RELATIVE_SCENARIOS),
                "reference_relative": REFERENCE_RELATIVE,
                "min_power": MIN_POWER,
                "null_percentile": _power.NULL_PERCENTILE,
                "random_seed": RANDOM_SEED,
                "logit_slope": _power.LOGIT_SLOPE,
                "beta_hat_source": "research/outputs/61_beta_hat.json (round 1, not refitted)",
                "guards": frame.guards,
                "arm3_rows_equal_arm2_rows": bool(same_rows),
                "residual_frame": {
                    "rows": int(residual.height),
                    "defence_seasons": int(n_defence_season),
                    "defences_pooled": int(n_defence_pooled),
                    "qb_seasons": int(n_qb),
                    "median_chances_per_defence_season": float(
                        np.median(chances_per_defence_season)
                    ),
                    "median_chances_per_defence_pooled": float(
                        np.median(chances_per_defence_pooled)
                    ),
                    "median_worthy_throws_per_qb_season": float(np.median(throws_per_qb_season)),
                    "min_worthy_throws_per_qb_season": int(throws_per_qb_season.min()),
                    "conversion_rate": conversion_rate,
                    "mean_p_hat": float(p_hat.mean()),
                },
                "cost": {
                    "wall_clock_seconds": elapsed,
                    "cells": len(per_fit),
                    "median_per_fit_seconds": median_per_fit,
                    "min_per_fit_seconds": per_fit[0],
                    "max_per_fit_seconds": per_fit[-1],
                    "slow_fit_threshold_seconds": SLOW_FIT_SECONDS,
                    "workers": 1 if (args.serial or args.workers <= 1) else args.workers,
                },
                "round1_comparison": {
                    name: {"threshold_pp": value[0], "power_at_reference": value[1]}
                    for name, value in round1.items()
                },
                "designs": reports,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out}")
    print(f"guards: {'ok' if frame.guards['ok'] else 'FAILED'}")
    print(f"arm3 rows == arm2 rows: {same_rows}")
    print(f"total {elapsed:.0f}s")


if __name__ == "__main__":
    main()
