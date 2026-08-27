"""Product layer, round 2, avenue 1 — the bootstrap distribution figure.

Nothing is fitted here. The three example games are re-simulated under v1.3's
exact settings so the *draws* are available (the shipped parquet keeps only the
summary), and the regenerated summary is then checked against the committed
artifact before a single pixel is drawn. If the replay disagrees, the figure
would be showing a distribution that no longer belongs to the published number,
so the script stops instead.

    uv run python research/54_bootstrap_figures.py

Writes ``research/outputs/54_bootstrap_<game_id>.png`` and
``research/outputs/54_bootstrap_figures.json``. Neither is committed —
``research/outputs/`` is gitignored, the script is the artifact and document 37
is the record of the numbers.
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

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
from nfl_simulator.plots import plot_bootstrap_distribution, verdict_from_row  # noqa: E402
from nfl_simulator.simulator import points_per_epa, simulate_game  # noqa: E402

# v1.3's shipped settings, quoted from `research/46_simulator_v13.py`. Changing
# any of them changes the draws and the replay check below will say so.
RANDOM_SEED = 20260817
POSTERIOR_DRAWS = 200
COIN_DRAWS = 800

GAMES_ARTIFACT = "dtw_games_v13.parquet"
RESULTS = "54_bootstrap_figures.json"

# Handoff 2026-08-23 §1, fixed before this script existed — one game per bucket.
EXAMPLES = ("2018_05_GB_DET", "2021_14_LV_KC", "2025_17_DET_MIN")

# The replay must land on the committed number, not near it.
REPLAY_TOLERANCE = 1e-9


def main() -> None:
    paths.ensure_data_dirs()
    committed = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / GAMES_ARTIFACT)
    missing = set(EXAMPLES) - set(committed["game_id"])
    if missing:
        raise SystemExit(f"{sorted(missing)} are not in {GAMES_ARTIFACT}.")

    print("loading play-by-play and refitting v1.3's baselines ...")
    pbp = load_pbp(PBP_SEASONS, columns=_read_side.SIM_COLUMNS)
    refit, _ = _read_side.load_model("trace_fg_refit.nc", "fg_refit_summary.json")

    fumble_baseline = fit_fumble_baseline(pbp)
    fg_baseline = fit_fg_baseline(pbp)
    xp_baseline = fit_xp_baseline(pbp)
    slope = points_per_epa(build_game_table(pbp).drop_nulls("margin"))
    print(f"  points_per_epa = {slope:.4f}")

    print(
        f"\n{'=' * 72}\nREPLAY — the redrawn draws must belong to the shipped summary\n{'=' * 72}"
    )
    records = []
    for game_id in EXAMPLES:
        plays = pbp.filter(pl.col("game_id") == game_id)
        result = simulate_game(
            plays,
            fumble_baseline=fumble_baseline,
            fg_baseline=fg_baseline,
            xp_baseline=xp_baseline,
            fg_model=refit,
            points_per_epa=slope,
            n_posterior_draws=POSTERIOR_DRAWS,
            n_coin_draws=COIN_DRAWS,
            seed=RANDOM_SEED,
            include_blocked=False,
        )
        row = committed.filter(pl.col("game_id") == game_id).to_dicts()[0]
        gaps = {
            "deserved_margin": abs(result.deserved_margin - row["deserved_margin"]),
            "dtw_home": abs(result.dtw_home - row["dtw_home"]),
            "dtw_low": abs(result.dtw_interval[0] - row["dtw_low"]),
            "dtw_high": abs(result.dtw_interval[1] - row["dtw_high"]),
        }
        worst = max(gaps.values())
        print(
            f"  {game_id:<18} max |Δ vs committed| {worst:.2e}  {'ok' if worst <= REPLAY_TOLERANCE else 'MISMATCH'}"
        )
        if worst > REPLAY_TOLERANCE:
            raise SystemExit(
                f"{game_id} does not replay to its shipped summary ({gaps}). "
                "Stop and report rather than draw it."
            )

        verdict = verdict_from_row(row, result.margin_draws)
        fig, _ax = plot_bootstrap_distribution(verdict)
        path = paths.RESEARCH_OUTPUT_DIR / f"54_bootstrap_{game_id}.png"
        fig.savefig(path, bbox_inches="tight")
        records.append(
            {
                "game_id": game_id,
                "home_team": verdict.home_team,
                "away_team": verdict.away_team,
                "actual_margin": verdict.actual_margin,
                "deserved_margin": verdict.deserved_margin,
                "dtw_home": verdict.dtw_home,
                "dtw_interval": list(verdict.dtw_interval),
                "headline": verdict.headline(),
                "bucket": verdict.bucket,
                "deserved_winner": verdict.deserved_winner,
                "scoreboard_winner": verdict.scoreboard_winner,
                "is_degenerate": verdict.is_degenerate,
                "is_point_mass": verdict.is_point_mass,
                "n_draws": int(result.margin_draws.size),
                "n_luck_events": len(result.ledger),
                "replay_gaps": gaps,
                "figure": path.name,
            }
        )

    print(f"\n{'=' * 72}\nFIGURES\n{'=' * 72}")
    for record in records:
        print(f"  {record['game_id']:<18} {record['headline']:<22} {record['bucket']}")
        print(
            f"    {record['figure']}  ({record['n_draws']:,} draws, {record['n_luck_events']} events)"
        )

    with (paths.RESEARCH_OUTPUT_DIR / RESULTS).open("w") as handle:
        json.dump(
            {
                "points_per_epa": slope,
                "settings": {
                    "seed": RANDOM_SEED,
                    "posterior_draws": POSTERIOR_DRAWS,
                    "coin_draws": COIN_DRAWS,
                },
                "games": records,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
