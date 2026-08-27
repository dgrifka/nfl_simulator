"""Product layer, round 2, avenue 2 — the luck-ledger waterfall.

Nothing is fitted or re-simulated here. The waterfall is arithmetic on two
committed artifacts — `dtw_ledger_v13.parquet` (one row per neutralized event)
and `dtw_games_v13.parquet` (the summary) — joined by the slope recorded in
`model_metadata_v13.json`. Because the simulator's identity is a sum,

    deserved_margin = actual_margin - total_luck_epa * points_per_epa

the bars have to reconcile the two ends of every game exactly. The script
checks that on all 2,761 games before drawing any of the three examples, and
`plot_luck_ledger` checks it again per game. A residual is a `SystemExit`, not
a rounding note.

    uv run python research/55_ledger_waterfall.py

Writes ``research/outputs/55_waterfall_<game_id>.png`` and
``research/outputs/55_ledger_waterfall.json``. Neither is committed —
``research/outputs/`` is gitignored, the script is the artifact and document 38
is the record of the numbers.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from nfl_simulator import paths
from nfl_simulator.plots import POINTS_FLOOR, luck_bars, plot_luck_ledger, verdict_from_row

GAMES_ARTIFACT = "dtw_games_v13.parquet"
LEDGER_ARTIFACT = "dtw_ledger_v13.parquet"
METADATA = "model_metadata_v13.json"
RESULTS = "55_ledger_waterfall.json"

# Handoff 2026-08-23 §1, the same three the distribution figure uses.
EXAMPLES = ("2018_05_GB_DET", "2021_14_LV_KC", "2025_17_DET_MIN")

# The ledger is a sum of doubles; anything above this is a real disagreement.
RECONCILE_TOLERANCE = 1e-9


def main() -> None:
    paths.ensure_data_dirs()
    games = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / GAMES_ARTIFACT)
    ledger = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / LEDGER_ARTIFACT)
    with (paths.RESEARCH_OUTPUT_DIR / METADATA).open() as handle:
        slope = json.load(handle)["points_per_epa"]
    print(f"points_per_epa = {slope:.10f}  (read from {METADATA}, not refitted)")

    print(
        f"\n{'=' * 72}\nRECONCILIATION — every game's bars must span actual to deserved\n{'=' * 72}"
    )
    summed = (
        ledger.group_by("game_id")
        .agg(pl.col("luck_epa").sum().alias("ledger_luck_epa"))
        .join(games, on="game_id", how="right")
        .with_columns(pl.col("ledger_luck_epa").fill_null(0.0))
        .with_columns(
            (
                pl.col("actual_margin")
                - pl.col("ledger_luck_epa") * slope
                - pl.col("deserved_margin")
            )
            .abs()
            .alias("residual")
        )
    )
    worst = float(summed["residual"].max())
    print(f"  {summed.height:,} games, max |residual| {worst:.2e}")
    if worst > RECONCILE_TOLERANCE:
        offenders = summed.filter(pl.col("residual") > RECONCILE_TOLERANCE)["game_id"].to_list()
        raise SystemExit(
            f"{len(offenders)} games do not reconcile ({offenders[:5]}). "
            "The ledger and the summary disagree — stop rather than draw them."
        )

    print(f"\n{'=' * 72}\nFIGURES\n{'=' * 72}")
    records = []
    for game_id in EXAMPLES:
        row = games.filter(pl.col("game_id") == game_id).to_dicts()[0]
        rows = ledger.filter(pl.col("game_id") == game_id).to_dicts()
        # The waterfall needs the game-level numbers, not the bootstrap draws;
        # a one-value array keeps `GameVerdict`'s point-mass property honest for
        # a game with no events and is unused by this figure otherwise.
        verdict = verdict_from_row(row, np.full(1, float(row["actual_margin"])))
        bars = luck_bars(rows, points_per_epa=slope)
        fig, _ax = plot_luck_ledger(verdict, rows, points_per_epa=slope)
        path = paths.RESEARCH_OUTPUT_DIR / f"55_waterfall_{game_id}.png"
        fig.savefig(path, bbox_inches="tight")

        folded = [bar for bar in bars if bar.n_events > 1]
        print(f"  {game_id:<18} {verdict.headline():<22} {verdict.bucket}")
        print(
            f"    {len(rows)} events -> {len(bars)} bars "
            f"({folded[0].n_events if folded else 0} folded under {POINTS_FLOOR:g} pt), "
            f"actual {verdict.actual_margin:+.0f} -> deserved {verdict.deserved_margin:+.2f}"
        )
        for bar in bars[:3]:
            print(f"      {bar.points:+6.2f}  {bar.label}")
        records.append(
            {
                "game_id": game_id,
                "headline": verdict.headline(),
                "bucket": verdict.bucket,
                "actual_margin": verdict.actual_margin,
                "deserved_margin": verdict.deserved_margin,
                "n_events": len(rows),
                "n_bars": len(bars),
                "n_folded": folded[0].n_events if folded else 0,
                "bars": [
                    {"label": bar.label, "points": bar.points, "n_events": bar.n_events}
                    for bar in bars
                ],
                "figure": path.name,
            }
        )

    with (paths.RESEARCH_OUTPUT_DIR / RESULTS).open("w") as handle:
        json.dump(
            {
                "points_per_epa": slope,
                "points_floor": POINTS_FLOOR,
                "games_reconciled": summed.height,
                "max_reconciliation_residual": worst,
                "games": records,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
