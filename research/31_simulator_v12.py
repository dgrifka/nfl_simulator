"""Phase 5 — simulator v1.2: the approved fumble widening, rebuilt and compared.

One verdict reaches the simulator: `docs/research/18-fumble-out-of-bounds.md`
§5g, approved 2026-08-18. The fumble component's population widens from *fumbles
somebody recovered* to **all fumbles with a resolved disposition**, with a ball
that crosses the sideline counted as the fumbling team keeping it.

The v1.1 artifacts are left untouched — document 07's rematch validation was run
against them, and document 18 §6 accepted that re-running it would prove nothing
either way because document 12 measured the rematch test as nearly blind below
~20% damage. v1.2 writes alongside, exactly as v1.1 did.

    research/outputs/dtw_games_v12.parquet
    research/outputs/dtw_ledger_v12.parquet
    research/outputs/model_metadata_v12.json
    research/outputs/31_ledger_delta.json

Two arms are simulated. The second is not shipped: it is the same v1.2 code at a
different seed, and it exists because adding fumble rows shifts the shared random
stream every later component draws from. Without it, "how many games flipped
between v1.1 and v1.2" could not be separated from Monte Carlo reshuffling.

    uv run python research/31_simulator_v12.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.components import (  # noqa: E402
    _fumble_frame,
    build_game_table,
    fit_fg_baseline,
    fit_fumble_baseline,
    fit_xp_baseline,
)
from nfl_simulator.fg_model import FieldGoalModel  # noqa: E402
from nfl_simulator.ingest import ANALYSIS_COLUMNS, PBP_SEASONS, load_pbp  # noqa: E402
from nfl_simulator.simulator import points_per_epa, simulate_game  # noqa: E402

RANDOM_SEED = 20260817  # v1.1's seed, so the arms differ by the component alone
NOISE_SEED = 20260819  # the reshuffle arm; never shipped
POSTERIOR_DRAWS = 200
COIN_DRAWS = 800

SIM_COLUMNS = [
    *ANALYSIS_COLUMNS,
    "kicker_player_id",
    "extra_point_attempt",
    "extra_point_result",
    "roof",
    "temp",
    "wind",
]

# docs/research/18 §3 and §8, committed before this script existed.
EXPECTED = {
    "fumbles": 6505,
    "out_of_bounds": 602,
    "league_retention_rate": 0.5648,
    "pass/live": 0.5096,
    "run/aborted": 0.7690,
    "games_gaining_a_row": 536,
}


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=paths.REPO_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def check_against_document_18(pbp: pl.DataFrame, baseline) -> dict:
    """The reproduction gate. A mismatch here is a stop, not a reconciliation."""
    frame = _fumble_frame(pbp)
    table = baseline.table
    rates = dict(zip(table["fumble_class"], table["p_own"], strict=True))
    observed = {
        "fumbles": frame.height,
        "out_of_bounds": int(frame["out_of_bounds"].sum()),
        "league_retention_rate": float(frame["retained"].mean()),
        "pass/live": float(rates["pass/live"]),
        "run/aborted": float(rates["run/aborted"]),
        "games_gaining_a_row": frame.filter(pl.col("out_of_bounds"))["game_id"].n_unique(),
    }
    print(f"\n{'=' * 72}\nREPRODUCTION CHECK against docs/research/18 §3 and §8\n{'=' * 72}")
    disagreements = []
    for key, expected in EXPECTED.items():
        got = observed[key]
        agrees = abs(got - expected) <= (5e-4 if isinstance(expected, float) else 0)
        print(f"  {key:<24} {got!s:<12} expected {expected}  {'ok' if agrees else 'MISMATCH'}")
        if not agrees:
            disagreements.append(key)
    if disagreements:
        raise SystemExit(
            f"v1.2 disagrees with document 18 on {disagreements}. "
            "Document 18 §5g is what was approved; stop and report rather than reconcile."
        )
    return observed


def simulate_all(pbp, margins, *, seed, **baselines) -> tuple[pl.DataFrame, pl.DataFrame]:
    rows, ledgers = [], []
    for i, (game_id, group) in enumerate(pbp.group_by("game_id")):
        game_id = game_id[0] if isinstance(game_id, tuple) else game_id
        if margins.get(game_id) is None:
            continue
        result = simulate_game(
            group,
            n_posterior_draws=POSTERIOR_DRAWS,
            n_coin_draws=COIN_DRAWS,
            seed=seed,
            **baselines,
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
    return pl.DataFrame(rows), pl.concat(ledgers)


def flips(left: pl.DataFrame, right: pl.DataFrame) -> tuple[int, pl.DataFrame]:
    joined = left.select("game_id", "deserved_margin", "dtw_home").join(
        right.select(
            "game_id",
            pl.col("deserved_margin").alias("deserved_other"),
            pl.col("dtw_home").alias("dtw_other"),
        ),
        on="game_id",
        how="inner",
    )
    flipped = joined.filter(((pl.col("dtw_home") - 0.5) * (pl.col("dtw_other") - 0.5)) < 0)
    return flipped.height, joined


def main() -> None:
    paths.ensure_data_dirs()
    pbp = load_pbp(PBP_SEASONS, columns=SIM_COLUMNS)

    print("fitting league baselines ...")
    fumble_baseline = fit_fumble_baseline(pbp)
    fg_baseline = fit_fg_baseline(pbp)
    xp_baseline = fit_xp_baseline(pbp)
    print(
        fumble_baseline.table.select("fumble_class", "n", "p_own", "p_out_of_bounds", "swing_value")
    )

    reproduction = check_against_document_18(pbp, fumble_baseline)

    games = build_game_table(pbp)
    slope = points_per_epa(games.drop_nulls("margin"))
    print(f"\npoints per EPA: {slope:.4f}")

    with (paths.RESEARCH_OUTPUT_DIR / "fg_weather_summary.json").open() as handle:
        centres = json.load(handle)["centres"]
    fg_model = FieldGoalModel.from_posterior(
        paths.RESEARCH_OUTPUT_DIR / "trace_fg_weather.nc",
        wind_centre=centres["wind"],
        temp_centre=centres["temp"],
    )

    margins = dict(zip(games["game_id"], games["margin"], strict=True))
    baselines = {
        "fumble_baseline": fumble_baseline,
        "fg_baseline": fg_baseline,
        "fg_model": fg_model,
        "xp_baseline": xp_baseline,
        "points_per_epa": slope,
    }

    print(f"\nsimulating {len(margins):,} games (v1.2) ...")
    table, ledger = simulate_all(pbp, margins, seed=RANDOM_SEED, **baselines)
    table = table.join(
        games.select("game_id", "season", "week", "home_team", "away_team"), on="game_id"
    )
    table.write_parquet(paths.RESEARCH_OUTPUT_DIR / "dtw_games_v12.parquet")
    ledger.write_parquet(paths.RESEARCH_OUTPUT_DIR / "dtw_ledger_v12.parquet")

    print(f"\nsimulating the reshuffle arm at seed {NOISE_SEED} (not shipped) ...")
    noise_table, _ = simulate_all(pbp, margins, seed=NOISE_SEED, **baselines)

    delta = compare_to_v11(table, ledger, noise_table)
    delta["reproduction_check"] = reproduction

    metadata = {
        "version": "simulator-v1.2",
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "seasons": list(PBP_SEASONS),
        "games_simulated": table.height,
        "random_seed": RANDOM_SEED,
        "posterior_draws": POSTERIOR_DRAWS,
        "coin_draws": COIN_DRAWS,
        "points_per_epa": slope,
        "changes_from_v11": [
            "the fumble component's population widens from fumbles somebody recovered to "
            "all fumbles with a resolved disposition, with out of bounds counted as the "
            "fumbling team keeping the ball (docs/research/18 §5g)"
        ],
        "component_treatment": {
            "fumble_retention": {
                "treatment": "full",
                "expectation": "league retention rate for the fumble's class",
                "population": "all fumbles with a resolved disposition",
                "w": 0.015,
                "gate_f2_population_sd_pp": 2.370,
                "gate_f2_threshold_pp": 5.260,
            },
            "field_goal": {
                "treatment": "partial",
                "expectation": "kicker's shrunk make probability at that distance, "
                "adjusted for roof, wind and temperature",
                "w_median": 0.285,
                "model": "docs/research/05b §11, cubic arm with weather",
            },
            "extra_point": {
                "treatment": "partial",
                "expectation": "kicker's shrunk extra-point probability",
                "model": "docs/research/09 §8, folded into the kicker model",
                "league_rate": xp_baseline.p_make,
                "swing_epa": xp_baseline.swing_value,
            },
            "interception": {
                "treatment": "none",
                "reason": "step 3a could not attribute the spread to an entity",
            },
            "penalty": {"treatment": "none", "reason": "fails the branch-point gate (05 §2)"},
            "return_yardage": {"treatment": "none", "reason": "fails the branch-point gate"},
            "onside_recovery": {
                "treatment": "none",
                "reason": "passes the branch-point gate but 599 kicks cannot size the "
                "spread; denied by default (09 §8)",
            },
            "overtime_toss": {
                "treatment": "none",
                "reason": "measured at +2.05 points but below the materiality floor (16 §8)",
            },
            "deflected_interception": {
                "treatment": "none",
                "reason": "the denominator — deflections that stayed incomplete — is "
                "invisible in every available dataset (17 §1)",
            },
            "sequencing": {
                "treatment": "none",
                "reason": "no branch point; red-zone and late-down placement are luck "
                "but are reported separately, never as ledger rows (08 §6)",
            },
        },
        "rematch_revalidation": {
            "run": False,
            "reason": "docs/research/18 §6: document 12 measured the rematch test as nearly "
            "blind below ~20% damage, so re-running it on a change this size would prove "
            "nothing either way. v1.1's validated artifacts are preserved instead.",
        },
        "delta_from_v11": delta,
    }
    out = paths.RESEARCH_OUTPUT_DIR / "model_metadata_v12.json"
    with out.open("w") as handle:
        json.dump(metadata, handle, indent=2, default=str)
    with (paths.RESEARCH_OUTPUT_DIR / "31_ledger_delta.json").open("w") as handle:
        json.dump(delta, handle, indent=2, default=str)
    print(f"\nWrote {out}")


def compare_to_v11(table: pl.DataFrame, ledger: pl.DataFrame, noise_table: pl.DataFrame) -> dict:
    """What the widening actually changed, and what is merely the reshuffle."""
    v11_games = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / "dtw_games_v11.parquet")
    v11_ledger = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / "dtw_ledger_v11.parquet")

    print(f"\n{'=' * 72}\nWHAT CHANGED, v1.1 -> v1.2\n{'=' * 72}")
    counts = (
        ledger.group_by("component")
        .agg(pl.len().alias("v12"))
        .join(
            v11_ledger.group_by("component").agg(pl.len().alias("v11")),
            on="component",
            how="full",
            coalesce=True,
        )
        .with_columns(pl.col("v11").fill_null(0), (pl.col("v12") - pl.col("v11")).alias("added"))
        .sort("component")
    )
    print(counts)

    # The deterministic half: which games gained a fumble row, and how the row
    # count moved. Nothing here depends on a random draw.
    fumbles_v11 = v11_ledger.filter(pl.col("component") == "fumble")
    fumbles_v12 = ledger.filter(pl.col("component") == "fumble")
    new_rows = fumbles_v12.join(
        fumbles_v11.select("game_id", "play_id"), on=["game_id", "play_id"], how="anti"
    )
    dropped_rows = fumbles_v11.join(
        fumbles_v12.select("game_id", "play_id"), on=["game_id", "play_id"], how="anti"
    )
    print(
        f"\n  fumble rows: {fumbles_v11.height:,} -> {fumbles_v12.height:,} "
        f"({new_rows.height:,} new, {dropped_rows.height:,} dropped)"
    )
    print(f"  games gaining a fumble row: {new_rows['game_id'].n_unique():,}")

    duplicates = fumbles_v12.group_by(["game_id", "play_id"]).len().filter(pl.col("len") > 1)
    print(f"  Gate F-4, fumbles booked twice: {duplicates.height} (must be 0)")
    if duplicates.height:
        raise SystemExit("a fumble was booked twice; the widened mask double-counts")

    changed, joined = flips(table, v11_games)
    noise_flips, _ = flips(table, noise_table)
    margin_shift = (joined["deserved_margin"] - joined["deserved_other"]).abs()
    dtw_shift = (joined["dtw_home"] - joined["dtw_other"]).abs()

    touched = set(new_rows["game_id"].to_list())
    on_touched = joined.filter(pl.col("game_id").is_in(touched))
    print(
        f"\n  deserved margin: mean |change| {margin_shift.mean():.3f} points, "
        f"max {margin_shift.max():.2f}"
    )
    print(
        f"  DTW: mean |change| {dtw_shift.mean() * 100:.2f} pp across all games, "
        f"{(on_touched['dtw_home'] - on_touched['dtw_other']).abs().median() * 100:.2f} pp "
        f"median on the {on_touched.height:,} games that gained a row"
    )
    flipped_on_touched = on_touched.filter(
        ((pl.col("dtw_home") - 0.5) * (pl.col("dtw_other") - 0.5)) < 0
    ).height
    print(f"\n  games whose DTW side flipped, v1.1 -> v1.2: {changed} across all games")
    print(
        f"    of which {flipped_on_touched} are among the {on_touched.height:,} games "
        f"that gained a row"
    )
    print(f"  games whose DTW side flipped, v1.2 -> v1.2 at another seed: {noise_flips}")
    print(
        "  (the last line is the Monte Carlo floor. Adding fumble rows shifts the shared "
        "random\n   stream every later component draws from, so it had to be measured rather "
        "than assumed;\n   at one game it is negligible. Document 18 §4b's isolated "
        "comparison measured 48 flips\n   across all fumble games and 31 on the "
        "out-of-bounds games.)"
    )

    return {
        "ledger_entries_v11": int(v11_ledger.height),
        "ledger_entries_v12": int(ledger.height),
        "entries_by_component": counts.to_dicts(),
        "fumble_rows_v11": int(fumbles_v11.height),
        "fumble_rows_v12": int(fumbles_v12.height),
        "fumble_rows_new": int(new_rows.height),
        "fumble_rows_dropped": int(dropped_rows.height),
        "games_gaining_a_fumble_row": int(new_rows["game_id"].n_unique()),
        "fumbles_booked_twice": int(duplicates.height),
        "mean_abs_deserved_margin_change": float(margin_shift.mean()),
        "max_abs_deserved_margin_change": float(margin_shift.max()),
        "mean_abs_dtw_change": float(dtw_shift.mean()),
        "median_abs_dtw_change_on_games_gaining_a_row": float(
            (on_touched["dtw_home"] - on_touched["dtw_other"]).abs().median()
        ),
        "games_dtw_side_flipped": int(changed),
        "games_dtw_side_flipped_on_games_gaining_a_row": int(flipped_on_touched),
        "games_dtw_side_flipped_reshuffle_floor": int(noise_flips),
        "document_18_isolated_flip_counts": {"all_fumble_games": 48, "out_of_bounds_games": 31},
        "n_games_compared": int(joined.height),
    }


if __name__ == "__main__":
    main()
