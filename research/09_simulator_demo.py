"""Step 5 — run simulator v1 over every game and record what shipped.

Produces three things:

    research/outputs/dtw_games.parquet     one row per game
    research/outputs/dtw_ledger.parquet    one row per luck event
    research/outputs/model_metadata.json   ground truth for what v1 actually is

and runs the two smoke tests from the Phase 2 handoff's verification list.

    uv run python research/09_simulator_demo.py
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime

import polars as pl

from nfl_simulator import paths
from nfl_simulator.components import (
    build_game_table,
    fit_fg_baseline,
    fit_fumble_baseline,
)
from nfl_simulator.fg_model import FieldGoalModel
from nfl_simulator.ingest import ANALYSIS_COLUMNS, PBP_SEASONS, load_pbp
from nfl_simulator.simulator import points_per_epa, simulate_game

RANDOM_SEED = 20260817

# Smaller than the module defaults: 2,761 games at the full grid would cost
# minutes for precision the third decimal place of DTW does not need.
POSTERIOR_DRAWS = 200
COIN_DRAWS = 100

SIM_COLUMNS = [*ANALYSIS_COLUMNS, "kicker_player_id"]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=paths.REPO_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    paths.ensure_data_dirs()
    pbp = load_pbp(PBP_SEASONS, columns=SIM_COLUMNS)

    print("fitting league baselines ...")
    fumble_baseline = fit_fumble_baseline(pbp)
    fg_baseline = fit_fg_baseline(pbp)

    games = build_game_table(pbp)
    slope = points_per_epa(games.drop_nulls("margin"))
    print(f"points per EPA: {slope:.4f}")

    fg_model = FieldGoalModel.from_posterior(paths.RESEARCH_OUTPUT_DIR / "trace_fg_model.nc")
    print(f"field-goal posterior: {fg_model.n_draws} draws, {len(fg_model.kicker_effects)} kickers")

    margins = dict(zip(games["game_id"], games["margin"], strict=True))
    print(f"simulating {len(margins):,} games ...")

    rows, ledgers = [], []
    for i, (game_id, group) in enumerate(pbp.group_by("game_id")):
        game_id = game_id[0] if isinstance(game_id, tuple) else game_id
        if margins.get(game_id) is None:
            continue  # a tie, or a game the schedule never resolved
        result = simulate_game(
            group,
            fumble_baseline=fumble_baseline,
            fg_baseline=fg_baseline,
            fg_model=fg_model,
            points_per_epa=slope,
            n_posterior_draws=POSTERIOR_DRAWS,
            n_coin_draws=COIN_DRAWS,
            seed=RANDOM_SEED,
        )
        rows.append(
            {
                "game_id": result.game_id,
                "actual_margin": result.actual_margin,
                "deserved_margin": result.deserved_margin,
                "dtw_home": result.dtw_home,
                "dtw_low": result.dtw_interval[0],
                "dtw_high": result.dtw_interval[1],
                "total_luck_epa": result.total_luck_epa,
                "n_luck_events": len(result.ledger),
            }
        )
        frame = result.ledger.to_frame()
        if frame.height:
            ledgers.append(frame.with_columns(pl.lit(result.game_id).alias("game_id")))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1:,} games")

    table = pl.DataFrame(rows).join(
        games.select("game_id", "season", "week", "home_team", "away_team"), on="game_id"
    )
    table.write_parquet(paths.RESEARCH_OUTPUT_DIR / "dtw_games.parquet")
    ledger = pl.concat(ledgers)
    ledger.write_parquet(paths.RESEARCH_OUTPUT_DIR / "dtw_ledger.parquet")

    summary = describe(table, ledger)
    smoke = smoke_tests(table, ledger, pbp, fumble_baseline, fg_baseline, fg_model, slope)

    metadata = {
        "version": "simulator-v1",
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "seasons": list(PBP_SEASONS),
        "games_simulated": table.height,
        "random_seed": RANDOM_SEED,
        "posterior_draws": POSTERIOR_DRAWS,
        "coin_draws": COIN_DRAWS,
        "points_per_epa": slope,
        # The treatment table from docs/research/05 §3, as actually implemented.
        "component_treatment": {
            "fumble_recovery": {
                "treatment": "full",
                "expectation": "league rate for the fumble's class",
                "w": 0.011,
                "classes": fumble_baseline.table.select("fumble_class", "n", "p_own").to_dicts(),
            },
            "field_goal": {
                "treatment": "partial",
                "expectation": "kicker's shrunk make probability at that distance",
                "w_median": 0.285,
                "model": "docs/research/05b, quadratic arm",
            },
            "interception": {
                "treatment": "none",
                "reason": "step 3a could not attribute the spread to an entity",
            },
            "penalty": {"treatment": "none", "reason": "fails the branch-point gate (05 §2)"},
            "return_yardage": {
                "treatment": "none",
                "reason": "fails the branch-point gate; no persistence (r = -0.014)",
            },
        },
        "summary": summary,
        "smoke_tests": smoke,
    }
    out = paths.RESEARCH_OUTPUT_DIR / "model_metadata.json"
    with out.open("w") as handle:
        json.dump(metadata, handle, indent=2, default=str)
    print(f"\nWrote {out}")


def describe(table: pl.DataFrame, ledger: pl.DataFrame) -> dict:
    """What the simulator does to ten seasons, in aggregate."""
    shift = (table["dtw_home"] - (table["actual_margin"] > 0).cast(pl.Float64)).abs()
    flipped = table.filter(
        ((pl.col("actual_margin") > 0) & (pl.col("dtw_home") < 0.5))
        | ((pl.col("actual_margin") < 0) & (pl.col("dtw_home") > 0.5))
    )
    width = table["dtw_high"] - table["dtw_low"]

    print("\n=== Simulator v1 over ten seasons ===")
    print(f"  games                       {table.height:,}")
    print(
        f"  luck events                 {ledger.height:,} ({ledger.height / table.height:.1f}/game)"
    )
    print(f"  mean |DTW - actual result|  {shift.mean():.3f}")
    print(f"  games where DTW disagrees   {flipped.height:,} ({flipped.height / table.height:.1%})")
    print(f"  mean 89% interval width     {width.mean():.3f}")
    print(
        f"  mean |margin adjustment|    {(table['actual_margin'] - table['deserved_margin']).abs().mean():.2f} points"
    )

    print("\n  luck by component:")
    by_component = ledger.group_by("component").agg(
        pl.len().alias("events"),
        pl.col("luck_epa").abs().mean().alias("mean_abs_luck_epa"),
        pl.col("luck_epa").sum().alias("net_luck_epa"),
    )
    print(by_component)

    return {
        "games": table.height,
        "luck_events": ledger.height,
        "mean_abs_dtw_shift": float(shift.mean()),
        "games_dtw_disagrees_with_result": flipped.height,
        "mean_interval_width": float(width.mean()),
        "mean_abs_margin_adjustment": float(
            (table["actual_margin"] - table["deserved_margin"]).abs().mean()
        ),
        "by_component": by_component.to_dicts(),
    }


def smoke_tests(
    table: pl.DataFrame,
    ledger: pl.DataFrame,
    pbp: pl.DataFrame,
    fumble_baseline,
    fg_baseline,
    fg_model,
    slope: float,
) -> dict:
    """The two checks named in the Phase 2 handoff's verification list."""
    print("\n=== Smoke test 1: a game with no luck events ===")
    quiet = table.filter(pl.col("n_luck_events") == 0)
    if quiet.is_empty():
        print("  no completely quiet game in ten seasons; using the quietest instead")
        quiet = table.sort("n_luck_events").head(1)
    row = quiet.row(0, named=True)
    print(
        f"  {row['game_id']}: {row['n_luck_events']} luck events, "
        f"actual {row['actual_margin']:+.0f}, deserved {row['deserved_margin']:+.1f}, "
        f"DTW {row['dtw_home']:.3f}"
    )
    test1 = {
        "game_id": row["game_id"],
        "n_luck_events": row["n_luck_events"],
        "actual_margin": row["actual_margin"],
        "deserved_margin": row["deserved_margin"],
        "dtw_home": row["dtw_home"],
        "pass": bool(
            row["n_luck_events"] == 0
            and abs(row["deserved_margin"] - row["actual_margin"]) < 1e-9
            and row["dtw_home"] in (0.0, 1.0)
        ),
    }
    print(f"  PASS: {test1['pass']}")

    print("\n=== Smoke test 2: a fumble lottery (4+ lost fumbles by one team) ===")
    lost = (
        pbp.filter(
            (pl.col("fumble") == 1)
            & pl.col("fumbled_1_team").is_not_null()
            & pl.col("fumble_recovery_1_team").is_not_null()
            & (pl.col("fumble_recovery_1_team") != pl.col("fumbled_1_team"))
        )
        .group_by(["game_id", "fumbled_1_team"])
        .agg(pl.len().alias("lost"))
        .filter(pl.col("lost") >= 4)
        .sort("lost", descending=True)
    )
    print(f"  {lost.height} team-games with 4+ lost fumbles")

    lottery = lost.join(table, on="game_id").with_columns(
        (pl.col("actual_margin") - pl.col("deserved_margin")).abs().alias("adjustment"),
        (pl.col("dtw_home") - (pl.col("actual_margin") > 0).cast(pl.Float64))
        .abs()
        .alias("dtw_shift"),
    )
    with pl.Config(tbl_rows=20, fmt_str_lengths=24):
        print(
            lottery.select(
                "game_id",
                "fumbled_1_team",
                "lost",
                "actual_margin",
                "deserved_margin",
                "adjustment",
                "dtw_home",
                "dtw_shift",
            ).sort("dtw_shift", descending=True)
        )

    # Select the game where the luck plausibly decided the outcome, not the one
    # with the most fumbles. Four lost fumbles inside a 29-point blowout SHOULD
    # leave DTW at zero — a simulator that flipped it would be broken, so
    # picking by fumble count would be testing the wrong thing.
    candidate = lottery.sort("dtw_shift", descending=True).row(0, named=True)
    game_ledger = ledger.filter(pl.col("game_id") == candidate["game_id"])
    applied = candidate["actual_margin"] - candidate["deserved_margin"]
    sums = abs(applied - game_ledger["luck_epa"].sum() * slope) < 1e-9

    print(
        f"  {candidate['game_id']}: {candidate['fumbled_1_team']} lost {candidate['lost']} fumbles\n"
        f"    actual margin   {candidate['actual_margin']:+.0f}\n"
        f"    deserved margin {candidate['deserved_margin']:+.2f}\n"
        f"    DTW (home)      {candidate['dtw_home']:.3f} "
        f"[{candidate['dtw_low']:.3f}, {candidate['dtw_high']:.3f}]\n"
        f"    ledger entries  {game_ledger.height}, "
        f"total luck {game_ledger['luck_epa'].sum():+.2f} EPA"
    )
    print(f"    ledger sums to the applied adjustment: {sums}")
    with pl.Config(tbl_rows=20, fmt_str_lengths=20):
        print(game_ledger.select("component", "event_class", "actual", "expected", "luck_epa"))

    shift = candidate["dtw_shift"]
    mean_adjustment = float(lottery["adjustment"].mean())

    # Every one of these games must satisfy the ledger identity, not just the
    # one selected — the identity is a property of the simulator, not of a
    # lucky pick.
    all_sum = True
    for row in lottery.iter_rows(named=True):
        rows_for = ledger.filter(pl.col("game_id") == row["game_id"])
        expected = row["actual_margin"] - row["deserved_margin"]
        all_sum &= abs(expected - rows_for["luck_epa"].sum() * slope) < 1e-9

    test2 = {
        "game_id": candidate["game_id"],
        "team": candidate["fumbled_1_team"],
        "lost_fumbles": candidate["lost"],
        "actual_margin": candidate["actual_margin"],
        "deserved_margin": candidate["deserved_margin"],
        "dtw_home": candidate["dtw_home"],
        "dtw_shift_from_result": shift,
        "mean_margin_adjustment_across_lottery_games": mean_adjustment,
        "ledger_sums_on_selected_game": bool(sums),
        "ledger_sums_on_every_lottery_game": bool(all_sum),
        "pass": bool(sums and all_sum and shift > 0.05 and mean_adjustment > 3.0),
    }
    print(
        f"  DTW moved {shift:.3f} from the actual result; "
        f"mean margin adjustment across all {lottery.height} lottery games "
        f"{mean_adjustment:.2f} points"
    )
    print(f"  ledger identity holds on every lottery game: {all_sum}")
    print(f"  PASS: {test2['pass']}")

    return {"zero_luck_game": test1, "fumble_lottery": test2}


if __name__ == "__main__":
    main()
