"""Phase 6 candidate 2 — blocked-kick aftermath: identification, power, floor.

Runs **before** `docs/research/25-blocked-kick-aftermath.md` fixes any threshold,
and stops short of the treatment arm for document 20's reason: the materiality
*floor* is a property of shipped v1.2 and is computed here, the materiality
*statistic* is not.

Document 23 §C2 found 415 blocked kicks in ten seasons — 192 field goals, 110
extra points, 113 punts — of which five reach the fumble component. A blocked
kick puts a live ball on the turf, and the ball's fate afterwards is currently
booked entirely as deserved. Five questions:

1. **Identification.** Can the post-block recovery be read off the play text, and
   what is the branch? Every rejected row is printed, not counted.
2. **Is the block and the recovery one continuous defensive play?** Measured
   directly: how often is the player who blocked the kick the player who
   recovers it. This is the strongest argument against Gate A and it deserves a
   number rather than an opinion.
3. **The class table** — the branch rate, the branch EPA means and the swing.
4. **The document 24 §9 screen.** `p(1 - p) x swing` per event, beside the
   kickoff-muff candidate that failed its materiality floor at 0.395 EPA on 248
   games. Computed before any threshold is set, precisely so that a candidate
   the arithmetic already excludes is not dressed up in a full round.
5. **Entity power and the materiality floor** the fit would have to clear.

    uv run python research/38_blocked_kick_power.py
"""

from __future__ import annotations

import json
import re
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_coinflips = import_module("12_coinflips_power")
_power = import_module("25_overtime_power")
_oob = import_module("29_fumble_oob_power")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.components import (  # noqa: E402
    _fumble_frame,
    build_game_table,
    fit_fg_baseline,
    fit_fumble_baseline,
    fit_xp_baseline,
)
from nfl_simulator.fg_model import FieldGoalModel  # noqa: E402
from nfl_simulator.ingest import PBP_SEASONS, load_pbp  # noqa: E402
from nfl_simulator.simulator import (  # noqa: E402
    bootstrap_margins,
    extra_point_events,
    field_goal_events,
    fumble_events,
    points_per_epa,
)

RANDOM_SEED = 20260818
SIM_POSTERIOR_DRAWS = _power.SIM_POSTERIOR_DRAWS
SIM_COIN_DRAWS = _power.SIM_COIN_DRAWS

# Document 24 §8: the kickoff-muff candidate carried this much luck per event
# and still missed its floor. It is the benchmark the screen reads against.
MUFF_SCREEN_EPA = 0.3948
MUFF_SCREEN_GAMES = 248

BLOCK_COLUMNS = sorted(
    {
        *_power.SIM_COLUMNS,
        "desc",
        "touchdown",
        "punt_blocked",
    }
)

# The team abbreviation is all the disposition needs. Player names are matched
# separately and more loosely, because "Ni.Herbig" and "Th.Jackson" exist.
RECOVERED_BY_TEAM = re.compile(r"recovered by ([A-Z]{2,3})-", re.IGNORECASE)
RECOVERED_BY_NAME = re.compile(r"recovered by [A-Z]{2,3}-\d+-([A-Z][A-Za-z]?\.[A-Za-z'\-.]+)", re.I)
BLOCKER_NAME = re.compile(r"BLOCKED (?:by )?\(?\d+-([A-Z][A-Za-z]?\.[A-Za-z'\-.]+)\)?", re.I)


# --------------------------------------------------------------------------
# identification
# --------------------------------------------------------------------------


def blocked_kick_mask() -> pl.Expr:
    """Every blocked field goal, extra point and punt with a resolved kick type."""
    return (
        (pl.col("field_goal_result") == "blocked")
        | (pl.col("extra_point_result") == "blocked")
        | (pl.col("punt_blocked") == 1).fill_null(False)
    )


def _kick_class(row: dict) -> str:
    if row["field_goal_result"] == "blocked":
        return "field_goal"
    if row["extra_point_result"] == "blocked":
        return "extra_point"
    return "punt"


def blocked_frame(pbp: pl.DataFrame) -> pl.DataFrame:
    """One row per blocked kick, from the kicking team's point of view.

    ``retained`` is 1 when the **defense did not come up with the ball** — it
    died where it lay, went out of bounds, or the kicking team fell on it. All
    three leave the defense taking over at a spot with no chance to run, and the
    branch this component prices is whether the defense got the ball in hand.

    Note what ``retained`` deliberately does *not* mean: possession. A kicking
    team that recovers its own blocked field goal behind the line to gain has
    still lost the ball on downs. The EPA branches carry that; the coin does not
    have to.
    """
    frame = pbp.filter(blocked_kick_mask())
    classes, rec_teams, recoverers, blockers = [], [], [], []
    for row in frame.iter_rows(named=True):
        desc = row["desc"]
        tail = desc[max(desc.lower().find("blocked"), 0) :]
        team_match = RECOVERED_BY_TEAM.search(tail)
        if team_match:
            rec_teams.append(team_match.group(1))
        elif "recovered the blocked kick" in tail:
            # The defensive two-point phrasing names no team; on a try only the
            # defense can recover and advance, so the defense is the recoverer.
            rec_teams.append(row["defteam"])
        else:
            rec_teams.append(None)
        name_match = RECOVERED_BY_NAME.search(tail)
        recoverers.append(name_match.group(1) if name_match else None)
        blocker_match = BLOCKER_NAME.search(desc)
        blockers.append(blocker_match.group(1) if blocker_match else None)
        classes.append(_kick_class(row))
    return frame.with_columns(
        pl.Series("kick_class", classes),
        pl.Series("recovering_team", rec_teams, dtype=pl.String),
        pl.Series("recoverer", recoverers, dtype=pl.String),
        pl.Series("blocker", blockers, dtype=pl.String),
        pl.col("posteam").alias("kicking_team"),
        pl.col("epa").alias("epa_kicker"),
    ).with_columns(
        (
            pl.col("recovering_team").is_null()
            | (pl.col("recovering_team") == pl.col("kicking_team"))
        )
        .cast(pl.Int8)
        .alias("retained"),
    )


def identification(pbp: pl.DataFrame) -> dict:
    """Is the branch readable, and is it a branch at all?"""
    print("[1] identification")
    frame = blocked_frame(pbp)
    print(f"  blocked kicks 2016-2025: {frame.height}, in {frame['game_id'].n_unique()} games")
    print(
        frame.group_by(["kick_class", "retained"])
        .agg(pl.len().alias("n"), pl.col("epa").mean().round(3).alias("mean_epa"))
        .sort(["kick_class", "retained"])
    )

    # Plays whose text mentions a recovery the parser could not attribute.
    mentions = frame.filter(
        pl.col("recovering_team").is_null() & pl.col("desc").str.contains("(?i)recover")
    )
    print(f"\n  recovery mentioned but no team parsed: {mentions.height}")
    for row in mentions.iter_rows(named=True):
        print(f"    REJECT [{row['kick_class']}] epa={row['epa']:+.3f}: {row['desc'][:170]}")

    # Gate B-1's hardest question, answered with a number.
    both = frame.filter(
        (pl.col("retained") == 0)
        & pl.col("recoverer").is_not_null()
        & pl.col("blocker").is_not_null()
    )
    same = both.filter(pl.col("recoverer") == pl.col("blocker"))
    share = same.height / max(both.height, 1)
    print(
        f"\n  defensive recoveries with both names readable: {both.height}; "
        f"the blocker is the recoverer in {same.height} ({share:.0%})"
    )

    overlap = frame.join(
        _fumble_frame(pbp).select("game_id", "play_id"), on=["game_id", "play_id"], how="semi"
    )
    print(f"\n  blocked kicks already carrying a v1.2 fumble row: {overlap.height}")
    for row in overlap.iter_rows(named=True):
        print(f"    EXCLUDE [{row['kick_class']}]: {row['desc'][:170]}")

    return {
        "blocked_kicks": frame.height,
        "games": frame["game_id"].n_unique(),
        "unparsed_recovery_mentions": mentions.height,
        "defensive_recoveries_named": both.height,
        "blocker_is_recoverer": same.height,
        "blocker_is_recoverer_share": share,
        "overlap_with_fumble_population": overlap.height,
        "by_season": frame.group_by("season").agg(pl.len().alias("n")).sort("season").to_dicts(),
    }


# --------------------------------------------------------------------------
# the class table and the screen
# --------------------------------------------------------------------------


def eligible_frame(pbp: pl.DataFrame) -> pl.DataFrame:
    """The population the component would run on: blocked kicks minus the overlap."""
    return blocked_frame(pbp).join(
        _fumble_frame(pbp).select("game_id", "play_id"), on=["game_id", "play_id"], how="anti"
    )


def class_table(pbp: pl.DataFrame) -> pl.DataFrame:
    frame = eligible_frame(pbp)
    table = (
        frame.group_by("kick_class")
        .agg(
            pl.len().alias("n"),
            pl.col("retained").mean().alias("p_retain"),
            pl.col("epa_kicker").filter(pl.col("retained") == 1).mean().alias("epa_own"),
            pl.col("epa_kicker").filter(pl.col("retained") == 0).mean().alias("epa_lost"),
        )
        .sort("n", descending=True)
    )
    return table.with_columns(
        (pl.col("epa_own") - pl.col("epa_lost")).alias("swing_value")
    ).with_columns(
        (pl.col("p_retain") * (1 - pl.col("p_retain")) * pl.col("swing_value")).alias("screen_epa")
    )


def screen(pbp: pl.DataFrame) -> dict:
    """Document 24 §9's screen, applied before any threshold is written."""
    print("\n[2] class table and the document 24 screen")
    table = class_table(pbp)
    print(table)
    frame = eligible_frame(pbp)
    pooled_p = float(frame["retained"].mean())
    pooled_swing = float(
        frame.filter(pl.col("retained") == 1)["epa_kicker"].mean()
        - frame.filter(pl.col("retained") == 0)["epa_kicker"].mean()
    )
    weighted = float((table["n"] * table["screen_epa"]).sum() / table["n"].sum())
    print(
        f"  n-weighted screen across classes: {weighted:.4f} EPA per event "
        f"(kickoff muffs, which failed: {MUFF_SCREEN_EPA:.4f} on {MUFF_SCREEN_GAMES} games)"
    )
    print(f"  pooled p {pooled_p:.4f}, pooled swing {pooled_swing:.4f} EPA")
    return {
        "classes": table.to_dicts(),
        "weighted_screen_epa": weighted,
        "pooled_p": pooled_p,
        "pooled_swing": pooled_swing,
        "muff_benchmark_epa": MUFF_SCREEN_EPA,
        "muff_benchmark_games": MUFF_SCREEN_GAMES,
    }


# --------------------------------------------------------------------------
# entity spread and the floor
# --------------------------------------------------------------------------


def entity_power(pbp: pl.DataFrame) -> dict:
    print("\n[3] entity spread")
    frame = eligible_frame(pbp)
    counts = (
        frame.group_by(["season", "kicking_team"])
        .agg(pl.len().alias("n"), pl.col("retained").sum().cast(pl.Int64).alias("k"))
        .drop_nulls()
        .sort(["season", "kicking_team"])
    )
    return {
        "blocked kicks, kicking team": _coinflips.power_table(
            "blocked kicks, kicking team", counts, RANDOM_SEED
        )
    }


def materiality_floor(pbp: pl.DataFrame) -> dict:
    """v1.2's own median 89% DTW half-width on the games a blocked kick touches."""
    print("\n[4] materiality floor")
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
    touched = sorted(set(eligible_frame(pbp)["game_id"].to_list()))
    margins = dict(zip(games["game_id"], games["margin"], strict=True))
    half_widths = []
    for game_id, group in pbp.filter(pl.col("game_id").is_in(touched)).group_by("game_id"):
        game_id = game_id[0] if isinstance(game_id, tuple) else game_id
        actual = margins.get(game_id)
        if actual is None:
            continue
        events = [
            *fumble_events(
                group, fumble_baseline, SIM_POSTERIOR_DRAWS, np.random.default_rng(RANDOM_SEED)
            ),
            *field_goal_events(
                group,
                fg_baseline,
                fg_model,
                SIM_POSTERIOR_DRAWS,
                np.random.default_rng(RANDOM_SEED + 1),
            ),
            *extra_point_events(
                group,
                xp_baseline,
                fg_model,
                SIM_POSTERIOR_DRAWS,
                np.random.default_rng(RANDOM_SEED + 2),
            ),
        ]
        if not events:
            continue
        _, per_draw = bootstrap_margins(
            events, actual, slope, SIM_COIN_DRAWS, np.random.default_rng(RANDOM_SEED + 3)
        )
        low, high = np.percentile(per_draw, [5.5, 94.5])
        half_widths.append(float((high - low) / 2))

    floor = float(np.median(half_widths))
    print(f"  games touched by a blocked kick: {len(touched)}")
    print(f"  v1.2 median 89% DTW half-width on them: {floor * 100:.4f} pp")
    return {
        "games": len(touched),
        "games_scored": len(half_widths),
        "median_half_width": floor,
        "points_per_epa": slope,
    }


def main() -> None:
    pbp = load_pbp(PBP_SEASONS, columns=BLOCK_COLUMNS)
    report = {
        "identification": identification(pbp),
        "screen": screen(pbp),
        "entity_power": entity_power(pbp),
        "materiality_floor": materiality_floor(pbp),
    }
    out = paths.RESEARCH_OUTPUT_DIR / "38_blocked_kick_power.json"
    with out.open("w") as handle:
        json.dump(report, handle, indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
