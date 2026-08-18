"""Phase 5 candidate 3 — fumbles out of bounds: the Gate F-2 fit.

Runs the gate `docs/research/18-fumble-out-of-bounds.md` §5c committed at
`afae577`, before this file existed: is the entity spread on the widened
retention branch below the 5.260 pp null bound, so that full neutralization
survives?

Fitted with the exact grid posterior of `research/_betabinom_grid.py`, and
cross-checked against the incumbent live-only branch so the comparison document
04 made is visible beside it.

    uv run python research/30_fumble_oob.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from _betabinom_grid import fit_grid  # noqa: E402

_design = import_module("29_fumble_oob_power")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.components import _fumble_frame, live_fumble_mask  # noqa: E402
from nfl_simulator.ingest import PBP_SEASONS, load_pbp  # noqa: E402

GATE_F2_THRESHOLD_PP = 5.260  # docs/research/18 §4a, the null 90th percentile


def counts_for(frame: pl.DataFrame, success: pl.Expr) -> pl.DataFrame:
    return (
        frame.group_by(["season", "fumbled_1_team"])
        .agg(pl.len().alias("n"), success.sum().cast(pl.Int64).alias("k"))
        .drop_nulls()
        .sort(["season", "fumbled_1_team"])
    )


def fit(name: str, counts: pl.DataFrame) -> dict:
    n = counts["n"].to_numpy().astype(float)
    k = counts["k"].to_numpy().astype(float)
    posterior = fit_grid(n, k)
    summary = posterior.summary()
    low, high = summary["population_sd_eti89"]
    rate = float(k.sum() / n.sum())
    report = {
        "name": name,
        "entities": counts.height,
        "opportunities": int(n.sum()),
        "league_rate": rate,
        "population_sd_pp": float(summary["population_sd_mean"]) * 100,
        "population_sd_eti89_pp": [low * 100, high * 100],
        "relative": float(summary["population_sd_mean"]) / rate,
        "kappa_mean": float(summary["kappa_mean"]),
        "grid_edge_mass": posterior.edge_mass(),
        "w_median_entity": float(counts["n"].median())
        / (float(counts["n"].median()) + float(summary["kappa_mean"])),
    }
    print(
        f"  {name}: SD {report['population_sd_pp']:.4f} pp "
        f"[{low * 100:.4f}, {high * 100:.4f}], relative {report['relative']:.1%}, "
        f"kappa {report['kappa_mean']:.1f}, w(median n) {report['w_median_entity']:.4f}, "
        f"grid edge mass {report['grid_edge_mass']:.2e}"
    )
    return report


def main() -> None:
    pbp = load_pbp(PBP_SEASONS, columns=_design.FUMBLE_COLUMNS)

    widened = fit(
        "retention, all fumbles (widened)",
        counts_for(_design.widened_frame(pbp), pl.col("retained")),
    )
    incumbent = fit(
        "recovery, live only (incumbent)",
        counts_for(_fumble_frame(pbp.filter(live_fumble_mask())), pl.col("retained")),
    )

    upper = widened["population_sd_eti89_pp"][1]
    passed = upper < GATE_F2_THRESHOLD_PP
    print(
        f"\n[F-2] 89% upper bound {upper:.4f} pp vs threshold "
        f"{GATE_F2_THRESHOLD_PP:.3f} pp -> {'PASS' if passed else 'FAIL'}"
    )
    print(
        "      PASS means w is effectively zero and full neutralization survives the wider branch."
        if passed
        else "      FAIL means teams genuinely differ; document 18 §5f commits to shipping nothing."
    )

    with (paths.RESEARCH_OUTPUT_DIR / "29_fumble_oob_power.json").open() as handle:
        design = json.load(handle)

    payload = {
        "gate_f2_threshold_pp": GATE_F2_THRESHOLD_PP,
        "gate_f2_pass": bool(passed),
        "widened": widened,
        "incumbent": incumbent,
        "impact": design["impact"],
        "classes": design["classes"],
    }
    out = paths.RESEARCH_OUTPUT_DIR / "30_fumble_oob.json"
    with out.open("w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {out}")
    print(
        "\nVERDICT: "
        + ("SHIP as v1.2, pending approval" if passed else "SHIP NOTHING — open a partial round")
    )


if __name__ == "__main__":
    main()
