"""Product layer, round 2, item 4 — the overtime-toss sidebar.

Nothing is fitted. Document 16 measured the overtime coin toss at 2.05 points of
final margin, then **refused** to ship it: gate O-3 asked whether neutralizing it
moved the median overtime game by more than the 4.06 pp interval the product
already prints, and the answer was 3.93 pp. The component is therefore reported
and not neutralized, and this script is the annotation that says so beside a
figure rather than in a document a reader will not have.

Two checks run before anything is drawn:

* document 16 §8's impact numbers are **recomputed** from
  `26_overtime_games.parquet` — the median move, the side flips, the game count —
  and compared to the strings `plots.py` prints. A panel that quoted a figure the
  artifact no longer supports would be worse than no panel.
* the three fixed example games are confirmed **not** to be overtime games, which
  is why documents 37 and 38's figures carry no sidebar.

The two examples here are chosen and not typical, and document 40 says so:
`2016_14_NYJ_SF` is the largest per-game move in the window, and
`2025_13_DEN_WAS` is a regular-season game under the 2025 rulebook, which is the
one case where the panel has to say the era cannot be separated.

    uv run python research/57_overtime_sidebar.py

Writes ``research/outputs/57_overtime_<game_id>_{bootstrap,waterfall}.png`` and
``research/outputs/57_overtime_sidebar.json``. Neither is committed —
``research/outputs/`` is gitignored, the script is the artifact and document 40
is the record of the numbers.
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_read_side = import_module("44_read_side_fix")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.components import (  # noqa: E402
    build_game_table,
    fit_fg_baseline,
    fit_fumble_baseline,
    fit_xp_baseline,
)
from nfl_simulator.ingest import PBP_SEASONS, load_pbp  # noqa: E402
from nfl_simulator.plots import (  # noqa: E402
    OVERTIME_GAMES,
    OVERTIME_MEDIAN_MOVE,
    OVERTIME_SIDE_FLIPS,
    OvertimeToss,
    attach_overtime_sidebar,
    overtime_lines,
    plot_bootstrap_distribution,
    plot_luck_ledger,
    verdict_from_row,
)
from nfl_simulator.simulator import points_per_epa, simulate_game  # noqa: E402

GAMES_ARTIFACT = "dtw_games_v13.parquet"
LEDGER_ARTIFACT = "dtw_ledger_v13.parquet"
OVERTIME_ARTIFACT = "26_overtime_games.parquet"
METADATA = "model_metadata_v13.json"
RESULTS = "57_overtime_sidebar.json"

# v1.3's shipped settings, quoted from `research/46_simulator_v13.py`, as in 54.
RANDOM_SEED = 20260817
POSTERIOR_DRAWS = 200
COIN_DRAWS = 800
REPLAY_TOLERANCE = 1e-9

# Handoff 2026-08-23 §1's fixed three. None of them went to overtime, which is
# the reason documents 37 and 38 show no sidebar.
FIXED_EXAMPLES = ("2018_05_GB_DET", "2021_14_LV_KC", "2025_17_DET_MIN")

EXAMPLES = ("2016_14_NYJ_SF", "2025_13_DEN_WAS")


def main() -> None:
    paths.ensure_data_dirs()
    games = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / GAMES_ARTIFACT)
    ledger = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / LEDGER_ARTIFACT)
    overtime = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / OVERTIME_ARTIFACT)
    with (paths.RESEARCH_OUTPUT_DIR / METADATA).open() as handle:
        slope = json.load(handle)["points_per_epa"]

    print(f"{'=' * 72}\nREPRODUCTION — document 16 §8's impact run\n{'=' * 72}")
    delta = overtime["delta"].to_numpy()
    checks = {
        "overtime games": (f"{overtime.height}", OVERTIME_GAMES),
        "median |ΔDTW|": (f"{np.median(np.abs(delta)) * 100:.2f} pp", OVERTIME_MEDIAN_MOVE),
        "games whose side flips": (
            f"{int(overtime['flipped'].sum())} of {overtime.height}",
            OVERTIME_SIDE_FLIPS,
        ),
    }
    for name, (here, printed) in checks.items():
        status = "ok" if here == printed else "MISMATCH"
        print(f"  {name:<26} {here:>12}  vs the panel's {printed:>12}  {status}")
    if any(here != printed for here, printed in checks.values()):
        raise SystemExit(
            "the sidebar's constants no longer match the artifact they quote. "
            "Stop rather than annotate a figure with a number the data does not support."
        )

    overtime_ids = set(overtime["game_id"])
    caught = sorted(set(FIXED_EXAMPLES) & overtime_ids)
    print(f"\n  fixed examples that went to overtime: {caught or 'none'}")
    if caught:
        raise SystemExit(
            f"{caught} went to overtime, so documents 37 and 38's figures are missing a "
            "sidebar they should carry. Stop and report."
        )

    print("\nloading play-by-play and refitting v1.3's baselines ...")
    pbp = load_pbp(PBP_SEASONS, columns=_read_side.SIM_COLUMNS)
    refit, _ = _read_side.load_model("trace_fg_refit.nc", "fg_refit_summary.json")
    fumble_baseline = fit_fumble_baseline(pbp)
    fg_baseline = fit_fg_baseline(pbp)
    xp_baseline = fit_xp_baseline(pbp)
    replay_slope = points_per_epa(build_game_table(pbp).drop_nulls("margin"))
    print(f"  points_per_epa = {replay_slope:.4f} (metadata: {slope:.4f})")

    print(f"\n{'=' * 72}\nFIGURES\n{'=' * 72}")
    records = []
    for game_id in EXAMPLES:
        row = games.filter(pl.col("game_id") == game_id).to_dicts()[0]
        ot_row = overtime.filter(pl.col("game_id") == game_id).to_dicts()[0]
        toss = OvertimeToss(
            received=row["home_team"] if ot_row["home_received"] else row["away_team"],
            season=int(row["season"]),
            delta_dtw_home=float(ot_row["delta"]),
        )

        plays = pbp.filter(pl.col("game_id") == game_id)
        result = simulate_game(
            plays,
            fumble_baseline=fumble_baseline,
            fg_baseline=fg_baseline,
            xp_baseline=xp_baseline,
            fg_model=refit,
            points_per_epa=replay_slope,
            n_posterior_draws=POSTERIOR_DRAWS,
            n_coin_draws=COIN_DRAWS,
            seed=RANDOM_SEED,
            include_blocked=False,
        )
        worst = max(
            abs(result.deserved_margin - row["deserved_margin"]),
            abs(result.dtw_home - row["dtw_home"]),
            abs(result.dtw_interval[0] - row["dtw_low"]),
            abs(result.dtw_interval[1] - row["dtw_high"]),
        )
        if worst > REPLAY_TOLERANCE:
            raise SystemExit(
                f"{game_id} does not replay to its shipped summary ({worst:.2e}). "
                "Stop rather than draw it."
            )

        verdict = verdict_from_row(row, result.margin_draws)
        fig, ax = plot_bootstrap_distribution(verdict)
        attach_overtime_sidebar(fig, ax, verdict, toss)
        bootstrap_path = paths.RESEARCH_OUTPUT_DIR / f"57_overtime_{game_id}_bootstrap.png"
        fig.savefig(bootstrap_path, bbox_inches="tight")

        rows = ledger.filter(pl.col("game_id") == game_id).to_dicts()
        # The waterfall wants the game-level numbers, not the draws — the same
        # one-value array document 38's driver uses.
        flat = verdict_from_row(row, np.full(1, float(row["actual_margin"])))
        fig, ax = plot_luck_ledger(flat, rows, points_per_epa=slope)
        attach_overtime_sidebar(fig, ax, flat, toss)
        waterfall_path = paths.RESEARCH_OUTPUT_DIR / f"57_overtime_{game_id}_waterfall.png"
        fig.savefig(waterfall_path, bbox_inches="tight")

        lines = overtime_lines(verdict, toss)
        print(f"\n  {game_id:<18} {verdict.headline():<22} {verdict.bucket}")
        print(
            f"    replay max |Δ vs committed| {worst:.2e}; {toss.received} received first; "
            f"ΔDTW(home) {toss.delta_dtw_home * 100:+.1f} pp on simulator v1.1"
        )
        for line in lines:
            print(f"      · {line}")
        records.append(
            {
                "game_id": game_id,
                "headline": verdict.headline(),
                "bucket": verdict.bucket,
                "season": toss.season,
                "received": toss.received,
                "delta_dtw_home": toss.delta_dtw_home,
                "replay_gap": worst,
                "sidebar": lines,
                "figures": [bootstrap_path.name, waterfall_path.name],
            }
        )

    results_path = paths.RESEARCH_OUTPUT_DIR / RESULTS
    results_path.write_text(
        json.dumps(
            {
                "doc_16_checks": {
                    name: {"here": here, "printed": printed}
                    for name, (here, printed) in checks.items()
                },
                "fixed_examples_in_overtime": caught,
                "n_overtime_games": overtime.height,
                "points_per_epa": slope,
                "games": records,
            },
            indent=2,
        )
    )
    print(f"\n  wrote {results_path.name}")


if __name__ == "__main__":
    main()
