"""Step 3 — descriptive EDA.

Two questions:

1. How often does the team with the higher game EPA differential actually win —
   overall, split by home/away, and by season? This is the sanity check that EPA
   differential is the right currency to be decomposing at all.
2. Where does the variance in a game's outcome come from? Decompose the home
   EPA differential into core / interception / penalty / fumble_luck / fg_luck
   and report each component's share.

    uv run python research/01_descriptive_eda.py
"""

from __future__ import annotations

import json

import polars as pl

from nfl_simulator import paths
from nfl_simulator.components import (
    COMPONENTS,
    build_game_table,
    fit_fg_baseline,
    fit_fumble_baseline,
    variance_shares,
)
from nfl_simulator.ingest import PBP_SEASONS, load_pbp

PBP_COLUMNS = [
    "game_id",
    "play_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "posteam",
    "defteam",
    "play_type",
    "epa",
    "result",
    "fumble",
    "fumble_lost",
    "fumbled_1_team",
    "fumble_recovery_1_team",
    "fumble_out_of_bounds",
    "aborted_play",
    "interception",
    "penalty",
    "penalty_type",
    "penalty_team",
    "field_goal_result",
    "kick_distance",
    "spread_line",
]


def audit(pbp: pl.DataFrame) -> dict:
    """Step 2 of the EDA workflow: what do we actually have?"""
    per_season = (
        pbp.group_by("season")
        .agg(
            pl.len().alias("plays"),
            pl.col("game_id").n_unique().alias("games"),
            pl.col("epa").null_count().alias("epa_nulls"),
            (pl.col("fumble") == 1).sum().alias("fumbles"),
            (pl.col("interception") == 1).sum().alias("interceptions"),
            (pl.col("play_type") == "field_goal").sum().alias("fg_attempts"),
            (pl.col("penalty") == 1).sum().alias("penalties"),
        )
        .sort("season")
    )
    print("\n=== Per-season audit ===")
    print(per_season)

    ties = pbp.group_by("game_id").agg(pl.col("result").first()).filter(pl.col("result") == 0)
    print(f"\nTied games (excluded from win-rate tables): {ties.height}")
    return {"per_season": per_season.to_dicts(), "n_ties": ties.height}


def epa_wins_table(games: pl.DataFrame) -> dict:
    """P(win | higher EPA differential), overall / by venue / by season."""
    decided = games.filter(pl.col("margin") != 0).with_columns(
        (pl.col("epa_diff") > 0).alias("home_higher_epa"),
        (pl.col("margin") > 0).alias("home_won"),
    )
    # "The higher-EPA team won" is symmetric — it is true whether the higher-EPA
    # team was home or away, so one boolean covers both.
    decided = decided.with_columns(
        (pl.col("home_higher_epa") == pl.col("home_won")).alias("epa_leader_won")
    )

    overall = decided["epa_leader_won"].mean()
    print("\n=== P(win | higher game EPA differential) ===")
    print(f"Overall: {overall:.3%}  (n = {decided.height:,} decided games)")

    by_venue = (
        decided.group_by("home_higher_epa")
        .agg(pl.len().alias("n"), pl.col("epa_leader_won").mean().alias("win_rate"))
        .sort("home_higher_epa", descending=True)
        .with_columns(
            pl.when(pl.col("home_higher_epa"))
            .then(pl.lit("home team led EPA"))
            .otherwise(pl.lit("away team led EPA"))
            .alias("case")
        )
        .select("case", "n", "win_rate")
    )
    print("\nBy which side led:")
    print(by_venue)

    by_season = (
        decided.group_by("season")
        .agg(pl.len().alias("n"), pl.col("epa_leader_won").mean().alias("win_rate"))
        .sort("season")
    )
    print("\nBy season:")
    print(by_season)

    # Home-field advantage, for context on the venue split.
    home_win_rate = decided["home_won"].mean()
    print(f"\nHome win rate (all decided games): {home_win_rate:.3%}")

    return {
        "overall": overall,
        "n_decided": decided.height,
        "by_venue": by_venue.to_dicts(),
        "by_season": by_season.to_dicts(),
        "home_win_rate": home_win_rate,
    }


def baseline_tables(pbp: pl.DataFrame) -> dict:
    """The empirical coin-flip rates the decomposition rests on."""
    fumbles = fit_fumble_baseline(pbp)
    field_goals = fit_fg_baseline(pbp)

    print("\n=== Fumble recovery baseline (2016-2025) ===")
    print(fumbles.table)
    rate_ex_aborted = fumbles.league_recovery_rate(exclude_aborted=True)
    rate_all = fumbles.league_recovery_rate(exclude_aborted=False)
    print(f"\nLeague own-recovery rate, excluding aborted snaps: {rate_ex_aborted:.4f}")
    print(f"League own-recovery rate, all live fumbles:        {rate_all:.4f}")

    print("\n=== Field goal baseline ===")
    print(field_goals.table)

    return {
        "fumble_classes": fumbles.table.to_dicts(),
        "recovery_rate_ex_aborted": rate_ex_aborted,
        "recovery_rate_all": rate_all,
        "fg_bins": field_goals.table.to_dicts(),
    }


def variance_tables(games: pl.DataFrame) -> dict:
    print("\n=== Variance shares of home EPA differential ===")
    epa_shares = variance_shares(games, "epa_diff")
    print(epa_shares)

    print("\n=== Variance shares of points margin ===")
    margin_shares = variance_shares(games, "margin")
    print(margin_shares)

    correlation = (
        games.select(pl.corr(pl.col("epa_diff"), pl.col("margin")).alias("r")).item() or 0.0
    )
    print(
        f"\ncorr(EPA differential, points margin) = {correlation:.4f}  (r^2 = {correlation**2:.4f})"
    )

    print("\n=== Component spread, per game (EPA points) ===")
    spread = games.select(
        [pl.col(component).std().alias(component) for component in COMPONENTS]
    ).transpose(include_header=True, header_name="component", column_names=["sd_epa"])
    print(spread)

    return {
        "epa_diff_shares": epa_shares.to_dicts(),
        "margin_shares": margin_shares.to_dicts(),
        "corr_epa_margin": correlation,
        "component_sd": spread.to_dicts(),
    }


def main() -> None:
    paths.ensure_data_dirs()
    pbp = load_pbp(PBP_SEASONS, columns=PBP_COLUMNS)
    print(f"Loaded {pbp.height:,} plays across {len(PBP_SEASONS)} seasons")

    results = {"seasons": list(PBP_SEASONS)}
    results["audit"] = audit(pbp)
    results["baselines"] = baseline_tables(pbp)

    games = build_game_table(pbp)
    games.write_parquet(paths.RESEARCH_OUTPUT_DIR / "game_components.parquet")
    print(f"\nWrote game-level table: {games.height:,} games -> game_components.parquet")

    results["epa_wins"] = epa_wins_table(games)
    results["variance"] = variance_tables(games)

    out = paths.RESEARCH_OUTPUT_DIR / "01_descriptive_eda.json"
    with out.open("w") as handle:
        json.dump(results, handle, indent=2, default=str)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
