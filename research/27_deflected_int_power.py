"""Phase 5 candidate 2 — deflected-pass interceptions: channel, stakes, power.

Runs **before** `docs/research/17-deflected-interceptions.md` fixes any
threshold. Three questions:

1. **Which channel actually identifies a deflection?** Document 15 proposed
   FTN's `is_interception_worthy == False`. This script cross-tabulates that
   against an official-scoring alternative — an interception where the credited
   pass defense belongs to a *different* defender than the interceptor, which is
   the play-by-play's way of recording that somebody else got a hand on the ball
   first.
2. **What is at stake if it could be neutralized?** The most generous assumption
   the data permits — every deflected interception was pure luck, `p = 0` — run
   through the incumbent simulator to bound the impact from above.
3. **Is generating deflected interceptions a repeatable defensive skill?** The
   split-half instrument of document 15 §C5, measured at the real per-team-season
   sample sizes before any verdict is read off it.

    uv run python research/27_deflected_int_power.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_power = import_module("25_overtime_power")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.components import (  # noqa: E402
    build_game_table,
    fit_fg_baseline,
    fit_fumble_baseline,
    fit_xp_baseline,
)
from nfl_simulator.fg_model import FieldGoalModel  # noqa: E402
from nfl_simulator.ingest import PBP_SEASONS, load_pbp  # noqa: E402
from nfl_simulator.simulator import points_per_epa, simulate_game  # noqa: E402

RANDOM_SEED = 20260818
N_SPLIT_HALF_SIMS = 2000
SKILL_SHARES = (0.0, 0.05, 0.10, 0.20)
FTN_SEASONS = range(2022, 2026)

INT_COLUMNS = [
    *_power.SIM_COLUMNS,
    "pass_attempt",
    "sack",
    "complete_pass",
    "pass_defense_1_player_id",
    "interception_player_id",
    "air_yards",
]


def deflection_frame(pbp: pl.DataFrame) -> pl.DataFrame:
    """Interceptions, flagged by whether a second defender touched the ball.

    On an interception nflverse credits the pass defense to the interceptor
    themself unless somebody else got a hand on it first, so
    `pass_defense_1_player_id != interception_player_id` is the play-by-play's
    record of a deflection — an official scoring fact rather than a charting
    judgment.
    """
    return pbp.filter(pl.col("interception") == 1).with_columns(
        deflected=pl.col("pass_defense_1_player_id") != pl.col("interception_player_id")
    )


def channel_comparison(ints: pl.DataFrame) -> dict:
    """Cross-tabulate the deflection channel against FTN's throw judgment."""
    ftn = pl.concat(
        [pl.read_parquet(paths.ftn_path(y)) for y in FTN_SEASONS],
        how="diagonal_relaxed",
    ).select(
        pl.col("nflverse_game_id").alias("game_id"),
        pl.col("nflverse_play_id").cast(pl.Float64).alias("play_id"),
        "is_interception_worthy",
    )
    joined = (
        ints.filter(pl.col("season") >= min(FTN_SEASONS))
        .join(ftn, on=["game_id", "play_id"], how="left")
        .filter(pl.col("is_interception_worthy").is_not_null())
    )
    cells = {}
    for deflected in (False, True):
        for worthy in (False, True):
            subset = joined.filter(
                (pl.col("deflected") == deflected) & (pl.col("is_interception_worthy") == worthy)
            )
            cells[f"deflected={deflected},worthy={worthy}"] = {
                "n": subset.height,
                "mean_epa": float(subset["epa"].mean()) if subset.height else None,
            }
    n_ftn_channel = joined.filter(~pl.col("is_interception_worthy")).height
    n_deflected = joined.filter("deflected").height
    overlap = joined.filter(pl.col("deflected") & ~pl.col("is_interception_worthy")).height
    return {
        "matched_interceptions": joined.height,
        "cells": cells,
        "ftn_channel_n": n_ftn_channel,
        "deflection_channel_n": n_deflected,
        "overlap": overlap,
        "share_of_ftn_channel_that_is_deflected": overlap / n_ftn_channel,
        "share_of_deflections_ftn_channel_catches": overlap / n_deflected,
    }


def stakes_bound(ints: pl.DataFrame, pbp: pl.DataFrame, rng: np.random.Generator) -> dict:
    """Upper bound on the impact, under the most generous assumption available.

    `p = 0` says the interception was pure luck and deserved nothing, so the
    whole branch swing is booked as luck on every deflected interception. No
    real component could move DTW further than this, which is what makes it a
    bound rather than a proposal.
    """
    deflected = ints.filter("deflected")
    incompletions = pbp.filter(
        (pl.col("pass_attempt") == 1)
        & (pl.col("sack") != 1)
        & (pl.col("complete_pass") != 1)
        & (pl.col("interception") != 1)
        & pl.col("pass_defense_1_player_id").is_not_null()
    )
    swing_epa = float(incompletions["epa"].mean() - deflected["epa"].mean())
    print(
        f"  branch means: deflected INT {deflected['epa'].mean():.4f} EPA, "
        f"defended incompletion {incompletions['epa'].mean():.4f} EPA -> swing {swing_epa:.4f}"
    )

    print("  fitting league baselines ...")
    fumble_baseline = fit_fumble_baseline(pbp)
    fg_baseline = fit_fg_baseline(pbp)
    xp_baseline = fit_xp_baseline(pbp)
    games = build_game_table(pbp)
    slope = points_per_epa(games.drop_nulls("margin"))
    with (paths.RESEARCH_OUTPUT_DIR / "fg_weather_summary.json").open() as handle:
        centres = json.load(handle)["centres"]
    fg_model = FieldGoalModel.from_posterior(
        paths.RESEARCH_OUTPUT_DIR / "trace_fg_weather.nc",
        wind_centre=centres["wind"],
        temp_centre=centres["temp"],
    )

    # Signed to the home perspective: a deflected interception thrown by the
    # home team costs the home team, so removing it adds to the home margin.
    per_game = (
        deflected.with_columns(
            sign=pl.when(pl.col("posteam") == pl.col("home_team")).then(-1.0).otherwise(1.0)
        )
        .group_by("game_id")
        .agg(pl.col("sign").sum().alias("signed_events"), pl.len().alias("n_events"))
    )
    shift = dict(zip(per_game["game_id"], per_game["signed_events"], strict=True))

    rows = []
    for game_id, group in pbp.filter(pl.col("game_id").is_in(set(shift))).group_by("game_id"):
        game_id = game_id[0] if isinstance(game_id, tuple) else game_id
        result = simulate_game(
            group,
            fumble_baseline=fumble_baseline,
            fg_baseline=fg_baseline,
            fg_model=fg_model,
            xp_baseline=xp_baseline,
            points_per_epa=slope,
            n_posterior_draws=_power.SIM_POSTERIOR_DRAWS,
            n_coin_draws=_power.SIM_COIN_DRAWS,
            seed=_power.SIM_SEED,
        )
        margins = result.margin_draws
        dtw_old = float((margins > 0).mean())
        # p = 0 makes the replayed branch deterministic, so the shift is exact.
        moved = margins + shift[game_id] * swing_epa * slope
        dtw_new = float((moved > 0).mean())
        rows.append(
            {
                "game_id": game_id,
                "dtw_old": dtw_old,
                "dtw_new": dtw_new,
                "half_width_old": (result.dtw_interval[1] - result.dtw_interval[0]) / 2,
            }
        )
        if len(rows) % 200 == 0:
            print(f"    {len(rows)} games simulated")

    table = pl.DataFrame(rows).with_columns(
        delta=pl.col("dtw_new") - pl.col("dtw_old"),
        flipped=((pl.col("dtw_old") - 0.5) * (pl.col("dtw_new") - 0.5)) < 0,
    )
    return {
        "swing_epa": swing_epa,
        "swing_points": swing_epa * slope,
        "events": deflected.height,
        "games_touched": table.height,
        "median_abs_delta_dtw": float(table["delta"].abs().median()),
        "mean_abs_delta_dtw": float(table["delta"].abs().mean()),
        "side_flips": int(table["flipped"].sum()),
        "floor_median_half_width": float(table["half_width_old"].median()),
    }


def persistence_power(ints: pl.DataFrame, pbp: pl.DataFrame, rng: np.random.Generator) -> dict:
    """Split-half instrument for 'is generating deflected interceptions a skill?'

    Document 15 §C5's design: simulate team-seasons at the *observed* per-team
    denominators with a known true skill share, and record the split-half
    correlation the test would produce. A verdict read from an instrument that
    cannot separate 0% from 10% is not a verdict.
    """
    dropbacks = (
        pbp.filter((pl.col("pass_attempt") == 1) & (pl.col("sack") != 1))
        .group_by(["season", "defteam"])
        .agg(pl.len().alias("faced"))
    )
    made = (
        ints.filter("deflected")
        .group_by(["season", "defteam"])
        .agg(pl.len().alias("deflected_ints"))
    )
    team_seasons = dropbacks.join(made, on=["season", "defteam"], how="left").with_columns(
        pl.col("deflected_ints").fill_null(0)
    )
    n = team_seasons["faced"].to_numpy()
    counts = team_seasons["deflected_ints"].to_numpy()
    base_rate = counts.sum() / n.sum()

    rows = []
    for share in SKILL_SHARES:
        correlations = []
        for _ in range(N_SPLIT_HALF_SIMS // 4):
            # Team true rates with `share` of the total variance as real skill.
            total_var = base_rate * (1 - base_rate) / np.median(n)
            sd = np.sqrt(max(share, 0.0) * total_var / max(1 - share, 1e-9))
            true = np.clip(rng.normal(base_rate, sd, size=n.size), 1e-6, 0.5)
            half = np.maximum(n // 2, 1)
            a = rng.binomial(half, true) / half
            b = rng.binomial(n - half, true) / np.maximum(n - half, 1)
            correlations.append(float(np.corrcoef(a, b)[0, 1]))
        correlations = np.array(correlations)
        rows.append(
            {
                "true_skill_share": share,
                "median_r": float(np.median(correlations)),
                "p05": float(np.percentile(correlations, 5)),
                "p95": float(np.percentile(correlations, 95)),
            }
        )

    return {
        "team_seasons": int(n.size),
        "median_dropbacks_faced": float(np.median(n)),
        "median_deflected_ints": float(np.median(counts)),
        "base_rate": float(base_rate),
        "rows": rows,
        "note": "the observed split-half is computed in the fit script, not here",
    }


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    pbp = load_pbp(PBP_SEASONS, columns=INT_COLUMNS)
    ints = deflection_frame(pbp)
    print(f"interceptions 2016-2025: {ints.height:,}; deflected: {int(ints['deflected'].sum()):,}")
    print(
        "per season: "
        f"{ints.group_by('season').agg(pl.col('deflected').sum()).sort('season')['deflected'].to_list()}"
    )

    print("\n[1] channel comparison ...")
    channels = channel_comparison(ints)
    for key, cell in channels["cells"].items():
        print(f"    {key}: n={cell['n']}, mean EPA {cell['mean_epa']}")
    print(
        f"    FTN channel catches {channels['share_of_deflections_ftn_channel_catches']:.1%} "
        f"of deflections; only {channels['share_of_ftn_channel_that_is_deflected']:.1%} of the "
        "FTN channel is a deflection"
    )

    print("\n[2] stakes bound (p = 0, the most generous assumption) ...")
    stakes = stakes_bound(ints, pbp, rng)
    print(
        f"    swing {stakes['swing_epa']:.3f} EPA = {stakes['swing_points']:.3f} points; "
        f"{stakes['events']} events over {stakes['games_touched']} games"
    )
    print(
        f"    median |dDTW| {100 * stakes['median_abs_delta_dtw']:.2f} pp vs floor "
        f"{100 * stakes['floor_median_half_width']:.2f} pp; side flips {stakes['side_flips']}"
    )

    print("\n[3] persistence instrument ...")
    persistence = persistence_power(ints, pbp, rng)
    print(
        f"    {persistence['team_seasons']} team-seasons, median {persistence['median_deflected_ints']:.0f} "
        f"deflected INTs on {persistence['median_dropbacks_faced']:.0f} dropbacks faced "
        f"(base rate {persistence['base_rate']:.4%})"
    )
    for row in persistence["rows"]:
        print(
            f"    true skill share {row['true_skill_share']:.0%} -> split-half r "
            f"{row['median_r']:+.3f} [{row['p05']:+.3f}, {row['p95']:+.3f}]"
        )

    payload = {
        "random_seed": RANDOM_SEED,
        "interceptions": ints.height,
        "deflected": int(ints["deflected"].sum()),
        "channels": channels,
        "stakes": stakes,
        "persistence": persistence,
    }
    out = paths.RESEARCH_OUTPUT_DIR / "27_deflected_int_power.json"
    with out.open("w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
