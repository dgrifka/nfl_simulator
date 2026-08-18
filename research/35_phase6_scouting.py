"""Phase 6 candidate scouting — data-existence checks only.

No models are fit and no thresholds are set here. This is step 1 of the
candidate ladder for the next round, in the shape of `research/24_phase5_scouting.py`.

Four questions, all of them "is the data there":

1. **The conditioning audit document 18 §9 asked for.** Field goals condition on
   an attempt being made and extra points condition on a touchdown having been
   scored. What is the branch immediately upstream of each, and does it hide a
   coin the way the fumble component's recovery test hid the out-of-bounds
   branch?
2. **Blocked-kick aftermath.** A block is a defensive play — denied, like a
   drop — but the loose ball afterwards is fumble-family physics. Do those
   recoveries already flow through the fumble component, or are they invisible
   to it?
3. **Muffed punts.** They should already sit inside the fumble population, since
   a muff is a fumble by the returner. Verified here in writing.
4. **Replay and challenge.** A one-paragraph identification check on
   `replay_or_challenge` and `replay_or_challenge_result`.

    uv run python research/35_phase6_scouting.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.components import _fumble_frame, any_fumble_mask  # noqa: E402
from nfl_simulator.ingest import ANALYSIS_COLUMNS, PBP_SEASONS, load_pbp  # noqa: E402

SCOUT_COLUMNS = sorted(
    {
        *ANALYSIS_COLUMNS,
        "desc",
        "touchdown",
        "extra_point_attempt",
        "extra_point_result",
        "two_point_attempt",
        "two_point_conv_result",
        "punt_attempt",
        "punt_blocked",
        "blocked_player_id",
        "replay_or_challenge",
        "replay_or_challenge_result",
        "own_kickoff_recovery",
    }
)


def conditioning_audit(pbp: pl.DataFrame) -> dict:
    """What each shipped component conditions on, and what sits one branch up."""
    fg = pbp.filter(pl.col("play_type") == "field_goal")
    fg_results = fg.group_by("field_goal_result").agg(
        pl.len().alias("n"), pl.col("epa").mean().alias("mean_epa")
    )
    xp = pbp.filter((pl.col("extra_point_attempt") == 1).fill_null(False))
    xp_results = xp.group_by("extra_point_result").agg(
        pl.len().alias("n"), pl.col("epa").mean().alias("mean_epa")
    )
    touchdowns = pbp.filter((pl.col("touchdown") == 1).fill_null(False))
    two_point = pbp.filter((pl.col("two_point_attempt") == 1).fill_null(False))

    print("[1] conditioning audit")
    print("\n  field goals — component conditions on: an attempt was made")
    print(fg_results.sort("n", descending=True))
    print("\n  extra points — component conditions on: a touchdown was scored")
    print(xp_results.sort("n", descending=True))
    print(
        f"\n  touchdowns: {touchdowns.height:,}; extra-point attempts: {xp.height:,}; "
        f"two-point attempts: {two_point.height:,}"
    )
    print(
        f"  touchdowns with neither a kick nor a two-point try charted: "
        f"{touchdowns.height - xp.height - two_point.height:,}"
    )
    return {
        "field_goal_results": fg_results.to_dicts(),
        "extra_point_results": xp_results.to_dicts(),
        "touchdowns": touchdowns.height,
        "extra_point_attempts": xp.height,
        "two_point_attempts": two_point.height,
    }


def blocked_kicks(pbp: pl.DataFrame) -> dict:
    """How many blocks there are, and whether the loose ball afterwards is booked."""
    blocked = {
        "field_goal": pbp.filter(pl.col("field_goal_result") == "blocked"),
        "extra_point": pbp.filter(pl.col("extra_point_result") == "blocked"),
        "punt": pbp.filter((pl.col("punt_blocked") == 1).fill_null(False)),
    }
    report = {}
    print("\n[2] blocked kicks and what happens to the loose ball")
    for name, frame in blocked.items():
        in_fumble = frame.filter(any_fumble_mask())
        recovered = frame.filter(
            (pl.col("fumble") == 1).fill_null(False)
            & pl.col("fumble_recovery_1_team").is_not_null()
        )
        report[name] = {
            "n": frame.height,
            "carrying_a_fumble_row": in_fumble.height,
            "with_a_named_recovering_team": recovered.height,
            "mean_epa": float(frame["epa"].mean()) if frame.height else None,
            "games": frame["game_id"].n_unique(),
        }
        print(
            f"  {name:<12} {frame.height:>5,} blocked; {in_fumble.height:>4,} carry a fumble row "
            f"({in_fumble.height / max(frame.height, 1):.0%}); mean EPA "
            f"{report[name]['mean_epa'] if frame.height else float('nan'):.3f}"
        )
    return report


def muffed_punts(pbp: pl.DataFrame) -> dict:
    """Do muffs already sit inside the fumble population?"""
    muffs = pbp.filter(pl.col("desc").str.contains("MUFFS"))
    inside = muffs.filter(any_fumble_mask())
    resolved = _fumble_frame(muffs)
    outside = muffs.filter(~any_fumble_mask())
    print("\n[3] muffed punts and returns")
    print(
        f"  plays whose description says MUFFS: {muffs.height:,}; "
        f"inside the fumble population: {inside.height:,} "
        f"({inside.height / max(muffs.height, 1):.1%}); with a resolved disposition: "
        f"{resolved.height:,}"
    )
    if outside.height:
        print(f"  {outside.height} muffs sit OUTSIDE the fumble population, for example:")
        for text in outside["desc"].str.slice(0, 120).to_list()[:5]:
            print(f"    {text}")
    by_type = (
        muffs.with_columns(any_fumble_mask().alias("in_population"))
        .group_by(["play_type", "in_population"])
        .agg(pl.len().alias("n"))
        .sort(["play_type", "in_population"])
    )
    print(by_type)

    # The muffs that fall outside: did the receiving team keep the ball? If the
    # answer is "almost always", the population is being selected on the outcome
    # of the branch — the same shape document 18 corrected for fumbles.
    kept = outside.filter(
        pl.struct(["desc", "posteam"]).map_elements(
            lambda row: (
                f"RECOVERED by {row['posteam']}" in row["desc"] or "and recovers" in row["desc"]
            ),
            return_dtype=pl.Boolean,
        )
    )
    print(
        f"  of the {outside.height} outside, {kept.height} say the muffing side "
        f"recovered it ({kept.height / max(outside.height, 1):.0%})"
    )
    return {
        "muffs": muffs.height,
        "inside_fumble_population": inside.height,
        "resolved": resolved.height,
        "outside": outside.height,
        "outside_recovered_by_muffing_side": kept.height,
        "by_play_type": by_type.to_dicts(),
    }


def replay_and_challenge(pbp: pl.DataFrame) -> dict:
    """Identification check on the replay columns."""
    replay = pbp.filter((pl.col("replay_or_challenge") == 1).fill_null(False))
    results = (
        replay.group_by("replay_or_challenge_result")
        .agg(pl.len().alias("n"), pl.col("epa").mean().alias("mean_epa"))
        .sort("n", descending=True)
    )
    print("\n[4] replay and challenge")
    print(
        f"  plays flagged: {replay.height:,} of {pbp.height:,} ({replay.height / pbp.height:.2%})"
    )
    print(results)
    print(
        f"  games with at least one: {replay['game_id'].n_unique():,} of "
        f"{pbp['game_id'].n_unique():,}"
    )
    return {
        "flagged": replay.height,
        "plays": pbp.height,
        "results": results.to_dicts(),
        "games": replay["game_id"].n_unique(),
    }


def main() -> None:
    pbp = load_pbp(PBP_SEASONS, columns=SCOUT_COLUMNS)
    payload = {
        "conditioning": conditioning_audit(pbp),
        "blocked_kicks": blocked_kicks(pbp),
        "muffed_punts": muffed_punts(pbp),
        "replay": replay_and_challenge(pbp),
    }
    out = paths.RESEARCH_OUTPUT_DIR / "35_phase6_scouting.json"
    with out.open("w") as handle:
        json.dump(payload, handle, indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
