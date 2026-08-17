"""Phase 3 — simulator v1.1: re-run every ledger, and record what changed.

Two Phase 3 verdicts reach the simulator:

* **Weather** (`docs/research/05b` §10/§11) — field-goal probabilities now carry
  roof, wind and temperature, so a windy 50-yarder is no longer priced as a calm
  one. This is the defect document 05b §7 called the model's largest.
* **Extra points** (`docs/research/09` §8) — a branch point with a resolvable
  kicker spread, so they become a neutralized component at the kicker's shrunk
  rate.

The v1 artifacts are left untouched: document 07's rematch validation was run
against them, and overwriting them would make that result unreproducible. v1.1
writes alongside, and the comparison between the two is the deliverable.

    research/outputs/dtw_games_v11.parquet
    research/outputs/dtw_ledger_v11.parquet
    research/outputs/model_metadata_v11.json
    research/outputs/15_ledger_delta.json

    uv run python research/15_simulator_v11.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_rematch = import_module("08_rematch_power")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.components import (  # noqa: E402
    build_game_table,
    fit_fg_baseline,
    fit_fumble_baseline,
    fit_xp_baseline,
)
from nfl_simulator.fg_model import FieldGoalModel  # noqa: E402
from nfl_simulator.ingest import ANALYSIS_COLUMNS, PBP_SEASONS, load_pbp  # noqa: E402
from nfl_simulator.simulator import points_per_epa, simulate_game  # noqa: E402

RANDOM_SEED = 20260817
POSTERIOR_DRAWS = 200
COIN_DRAWS = 100

SIM_COLUMNS = [
    *ANALYSIS_COLUMNS,
    "kicker_player_id",
    "extra_point_attempt",
    "extra_point_result",
    "roof",
    "temp",
    "wind",
]


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
    xp_baseline = fit_xp_baseline(pbp)
    print(
        f"extra-point baseline: {xp_baseline.n:,} attempts, make rate "
        f"{xp_baseline.p_make:.4%}, swing {xp_baseline.swing_value:.4f} EPA"
    )

    games = build_game_table(pbp)
    slope = points_per_epa(games.drop_nulls("margin"))
    print(f"points per EPA: {slope:.4f}")

    with (paths.RESEARCH_OUTPUT_DIR / "fg_weather_summary.json").open() as handle:
        weather_summary = json.load(handle)
    centres = weather_summary["centres"]

    fg_model = FieldGoalModel.from_posterior(
        paths.RESEARCH_OUTPUT_DIR / "trace_fg_weather.nc",
        wind_centre=centres["wind"],
        temp_centre=centres["temp"],
    )
    print(
        f"field-goal posterior: {fg_model.n_draws} draws, "
        f"{len(fg_model.kicker_effects)} kickers, "
        f"roof levels {sorted(fg_model.roof_effects)}, "
        f"beta_wind {fg_model.beta_wind.mean():+.5f}"
    )

    margins = dict(zip(games["game_id"], games["margin"], strict=True))
    print(f"simulating {len(margins):,} games ...")

    rows, ledgers = [], []
    for i, (game_id, group) in enumerate(pbp.group_by("game_id")):
        game_id = game_id[0] if isinstance(game_id, tuple) else game_id
        if margins.get(game_id) is None:
            continue
        result = simulate_game(
            group,
            fumble_baseline=fumble_baseline,
            fg_baseline=fg_baseline,
            fg_model=fg_model,
            xp_baseline=xp_baseline,
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
    ledger = pl.concat(ledgers)
    table.write_parquet(paths.RESEARCH_OUTPUT_DIR / "dtw_games_v11.parquet")
    ledger.write_parquet(paths.RESEARCH_OUTPUT_DIR / "dtw_ledger_v11.parquet")

    delta = compare_to_v1(table, ledger, pbp, slope)
    rematch = revalidate(games, table)

    metadata = {
        "version": "simulator-v1.1",
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "seasons": list(PBP_SEASONS),
        "games_simulated": table.height,
        "random_seed": RANDOM_SEED,
        "posterior_draws": POSTERIOR_DRAWS,
        "coin_draws": COIN_DRAWS,
        "points_per_epa": slope,
        "changes_from_v1": [
            "field-goal probabilities carry roof, wind and temperature (docs/research/05b §10-11)",
            "extra points are a neutralized component at the kicker's shrunk rate "
            "(docs/research/09 §8)",
        ],
        "component_treatment": {
            "fumble_recovery": {
                "treatment": "full",
                "expectation": "league rate for the fumble's class",
                "w": 0.011,
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
            "sequencing": {
                "treatment": "none",
                "reason": "no branch point; red-zone and late-down placement are luck "
                "but are reported separately, never as ledger rows (08 §6)",
            },
        },
        "delta_from_v1": delta,
        "rematch_revalidation": rematch,
    }
    out = paths.RESEARCH_OUTPUT_DIR / "model_metadata_v11.json"
    with out.open("w") as handle:
        json.dump(metadata, handle, indent=2, default=str)
    print(f"\nWrote {out}")


def compare_to_v1(table: pl.DataFrame, ledger: pl.DataFrame, pbp: pl.DataFrame, slope: float):
    """What the refit actually changed, entry by entry and game by game."""
    v1_games = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / "dtw_games.parquet")
    v1_ledger = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / "dtw_ledger.parquet")

    print(f"\n{'=' * 72}\nWHAT CHANGED, v1 -> v1.1\n{'=' * 72}")
    print(f"  ledger entries   {v1_ledger.height:,} -> {ledger.height:,}")
    print("  by component:")
    counts = (
        ledger.group_by("component")
        .agg(pl.len().alias("v11"))
        .join(
            v1_ledger.group_by("component").agg(pl.len().alias("v1")),
            on="component",
            how="full",
            coalesce=True,
        )
        .with_columns(pl.col("v1").fill_null(0))
        .sort("component")
    )
    print(counts)

    # Field-goal entries are the ones weather moves. Pair them on (game, play).
    fg_v1 = v1_ledger.filter(pl.col("component") == "field_goal").select(
        "game_id", "play_id", pl.col("expected").alias("expected_v1")
    )
    fg_v11 = ledger.filter(pl.col("component") == "field_goal").select(
        "game_id", "play_id", pl.col("expected").alias("expected_v11")
    )
    paired = fg_v1.join(fg_v11, on=["game_id", "play_id"], how="inner").with_columns(
        (pl.col("expected_v11") - pl.col("expected_v1")).alias("shift")
    )
    moved = paired.filter(pl.col("shift").abs() > 0.005)
    print(
        f"\n  field-goal entries repriced: {moved.height:,} of {paired.height:,} "
        f"moved by more than half a point of make probability"
    )
    print(
        f"    mean |shift| {paired['shift'].abs().mean() * 100:.2f} pp, "
        f"max {paired['shift'].abs().max() * 100:.2f} pp"
    )

    # Direction by roof: the whole point is that this should be systematic.
    roof = (
        pbp.filter(pl.col("play_type") == "field_goal")
        .select("game_id", "play_id", "roof", "wind")
        .unique(subset=["game_id", "play_id"])
    )
    by_roof = (
        paired.join(roof, on=["game_id", "play_id"], how="left")
        .group_by("roof")
        .agg(
            pl.len().alias("n"),
            (pl.col("shift").mean() * 100).alias("mean_shift_pp"),
        )
        .sort("mean_shift_pp", descending=True)
    )
    print("\n  mean change in make probability, by roof:")
    print(by_roof)

    joined = table.select("game_id", "deserved_margin", "dtw_home", "actual_margin").join(
        v1_games.select(
            "game_id",
            pl.col("deserved_margin").alias("deserved_v1"),
            pl.col("dtw_home").alias("dtw_v1"),
        ),
        on="game_id",
        how="inner",
    )
    margin_shift = (joined["deserved_margin"] - joined["deserved_v1"]).abs()
    dtw_shift = (joined["dtw_home"] - joined["dtw_v1"]).abs()
    flipped = joined.filter(
        ((pl.col("dtw_home") > 0.5) & (pl.col("dtw_v1") < 0.5))
        | ((pl.col("dtw_home") < 0.5) & (pl.col("dtw_v1") > 0.5))
    )
    print(
        f"\n  deserved margin: mean |change| {margin_shift.mean():.3f} points, "
        f"max {margin_shift.max():.2f}"
    )
    print(f"  DTW%:            mean |change| {dtw_shift.mean():.4f}, max {dtw_shift.max():.4f}")
    print(
        f"  games whose DTW winner flipped between v1 and v1.1: {flipped.height} "
        f"({flipped.height / joined.height:.2%})"
    )

    return {
        "ledger_entries_v1": int(v1_ledger.height),
        "ledger_entries_v11": int(ledger.height),
        "entries_by_component": counts.to_dicts(),
        "field_goal_entries_paired": int(paired.height),
        "field_goal_entries_moved_over_half_a_point": int(moved.height),
        "mean_abs_make_probability_shift_pp": float(paired["shift"].abs().mean() * 100),
        "max_abs_make_probability_shift_pp": float(paired["shift"].abs().max() * 100),
        "mean_make_probability_shift_by_roof_pp": by_roof.to_dicts(),
        "mean_abs_deserved_margin_change": float(margin_shift.mean()),
        "max_abs_deserved_margin_change": float(margin_shift.max()),
        "mean_abs_dtw_change": float(dtw_shift.mean()),
        "games_dtw_winner_flipped": int(flipped.height),
        "n_games_compared": int(joined.height),
    }


def revalidate(games: pl.DataFrame, table: pl.DataFrame) -> dict:
    """Re-run document 06's Gate 1 on v1.1's deserved margin.

    The treatment table changed, so the non-inferiority claim has to be re-earned
    rather than inherited. Same 531 pairs, same statistic, same +0.010 margin,
    same seed — nothing about the test moved.
    """
    print(f"\n{'=' * 72}\nREMATCH RE-VALIDATION (document 06 Gate 1, on v1.1)\n{'=' * 72}")
    pairs = _rematch.rematch_pairs(games)
    keyed = (
        games.drop_nulls("margin")
        .with_columns(
            pl.min_horizontal("home_team", "away_team").alias("t1"),
            pl.max_horizontal("home_team", "away_team").alias("t2"),
        )
        .with_columns(
            pl.concat_str([pl.col("season").cast(pl.String), "t1", "t2"], separator="_").alias(
                "pair"
            )
        )
        .sort(["pair", "week"])
        .with_columns(pl.int_range(pl.len()).over("pair").alias("meeting"))
        .filter(pl.col("meeting") == 0)
        .select("pair", "game_id")
    )
    joined = pairs.join(keyed, on="pair", how="inner").join(
        table.select("game_id", "deserved_margin"), on="game_id", how="inner"
    )

    actual = joined["margin_g1_a"].to_numpy().astype(float)
    deserved = joined["deserved_margin"].to_numpy().astype(float)
    y = (joined["margin_g2_a"].to_numpy() > 0).astype(float)
    a_home = joined["a_home_g2"].to_numpy().astype(float)
    rng = np.random.default_rng(RANDOM_SEED)
    folds = rng.permutation(joined.height) % _rematch.N_FOLDS

    per_pair = _rematch.paired_log_loss_diff(actual, deserved, y, a_home, folds)
    mean, se, _ = _rematch.decision(per_pair)
    passed = _rematch.passes_noninferiority(mean, se)
    print(f"  {joined.height} rematch pairs")
    print(
        f"  mean delta log loss {mean:+.5f} (SE {se:.5f})  "
        f"95% CI [{mean - 1.96 * se:+.5f}, {mean + 1.96 * se:+.5f}]"
    )
    print(
        f"  GATE 1 non-inferiority: {'PASS' if passed else 'FAIL'} vs margin "
        f"{_rematch.NONINFERIORITY_MARGIN:+.3f}"
    )
    print("  (document 07 measured -0.00159, SE 0.00273 on v1)")
    return {
        "n_pairs": int(joined.height),
        "mean_delta_log_loss": mean,
        "se": se,
        "ci95": [mean - 1.96 * se, mean + 1.96 * se],
        "noninferiority_pass": bool(passed),
        "v1_reference": {"mean_delta_log_loss": -0.00159, "se": 0.00273},
    }


if __name__ == "__main__":
    main()
