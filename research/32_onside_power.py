"""Phase 5 candidate 4 — onside kicks at the league rate: identification and power.

Runs **before** `docs/research/20-onside-kicks.md` fixes any threshold.

Document 09 passed onside kicks on mechanism — a loose ball in a scrum, the same
physics the fumble component already neutralizes — and denied them only because
the per-team trust dial `w` could not be estimated. The candidate here is the
`w = 0` variant, which needs no per-team estimate: the same choice fumble
recovery made, for the same reason.

Four questions, none of which sets a gate:

1. **Can an onside kick be identified?** Two channels exist — the play
   description and the `own_kickoff_recovery` flag — and neither is a definition.
2. **Is there class structure?** A surprise onside kick against a return team
   that is not expecting one is a different coin from a desperation kick in the
   last two minutes, and the fumble component's classes exist precisely because
   pooling coins of different weights is wrong.
3. **What is the branch worth?** The EPA gap between recovering and not.
4. **Could the entity spread be resolved if we wanted it?** Document 09 said no;
   this measures the instrument at the real denominators so the `w = 0` choice
   is made against a number rather than an impression.

    uv run python research/32_onside_power.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_coinflips = import_module("12_coinflips_power")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.ingest import PBP_SEASONS, load_pbp  # noqa: E402

RANDOM_SEED = 20260818

ONSIDE_COLUMNS = [
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
    "desc",
    "own_kickoff_recovery",
    "kick_distance",
    "qtr",
    "game_seconds_remaining",
    "score_differential",
]

# On a kickoff nflfastR puts `posteam` on the **receiving** team — the side that
# is about to have the ball — so the kicking team is `defteam`, and both `epa`
# and `score_differential` are signed from the receiver's point of view. Verified
# on the play text: an own-kickoff recovery carries epa -2.95 for `posteam`,
# which is only sensible if `posteam` is the team that just lost the ball.
KICKING_TEAM = pl.col("defteam")
EPA_KICKER = -pl.col("epa")
KICKER_DEFICIT = -pl.col("score_differential")

# The candidate class split, defined on game state rather than on the outcome.
# A kicking team that trails inside five minutes has no other option and the
# return team is lined up for it; a kick at any other moment is a surprise.
# Committed as a *candidate* here and fixed as the class definition in
# document 20 §3.
EXPECTED_ONSIDE = (KICKER_DEFICIT < 0) & (pl.col("game_seconds_remaining") <= 300)

# The identification channel. "kicks onside" is the scorer's phrase for an
# attempt; the bare word "onside" also appears in re-kick notes and in
# "(Onside Kick formation)" lines describing kicks that then went 43 and 54
# yards downfield.
ONSIDE_TEXT = pl.col("desc").str.contains("(?i)kicks onside")


def onside_frame(pbp: pl.DataFrame) -> pl.DataFrame:
    """Onside-kick attempts, annotated from the kicking team's point of view.

    The population is the play text — `kicks onside` — and **not** the union with
    `own_kickoff_recovery`. Document 15 read the flag as covering four attempts
    the text misses; `identification` below shows all four are deep kickoffs
    whose return was muffed and fell to the kicking team. That is a loose ball,
    but it is not an onside kick, and folding it in would price a different coin.

    The flag is used for the *outcome* instead, where it is the better channel:
    it agrees with the description on 99.5% of the population, and every
    disagreement is a recovery reversed on replay.
    """
    return (
        pbp.filter((pl.col("play_type") == "kickoff") & ONSIDE_TEXT)
        .with_columns(
            (pl.col("own_kickoff_recovery") == 1).fill_null(False).cast(pl.Int8).alias("recovered"),
            KICKING_TEAM.alias("kicking_team"),
            EPA_KICKER.alias("epa_kicker"),
            KICKER_DEFICIT.alias("kicker_deficit"),
            pl.when(EXPECTED_ONSIDE)
            .then(pl.lit("expected"))
            .otherwise(pl.lit("surprise"))
            .alias("onside_class"),
        )
        .drop_nulls("kicking_team")
    )


def identification(pbp: pl.DataFrame) -> dict:
    """Audit both channels before either is chosen, and show the rejects."""
    kicks = pbp.filter(pl.col("play_type") == "kickoff")
    loose = kicks.filter(pl.col("desc").str.contains("(?i)onside") & ~ONSIDE_TEXT)
    flag = kicks.filter((pl.col("own_kickoff_recovery") == 1).fill_null(False))
    flag_only = flag.filter(~ONSIDE_TEXT)
    population = onside_frame(pbp)

    # Does the flag agree with the play text about who ended up with the ball?
    text_recovery = pl.struct(["desc", "defteam"]).map_elements(
        lambda row: f"RECOVERED by {row['defteam']}" in row["desc"], return_dtype=pl.Boolean
    )
    checked = population.with_columns(text_recovery.alias("text_says_recovered"))
    disagree = checked.filter(pl.col("recovered").cast(pl.Boolean) != pl.col("text_says_recovered"))

    report = {
        "kickoffs": kicks.height,
        "population": population.height,
        "recovered": int(population["recovered"].sum()),
        "league_rate": float(population["recovered"].mean()),
        "rejected_loose_desc": loose.height,
        "rejected_flag_only": flag_only.height,
        "flag_text_disagreements": disagree.height,
        "flag_text_agreement": 1.0 - disagree.height / population.height,
        "median_kick_distance": float(population["kick_distance"].median()),
        "max_kick_distance": float(population["kick_distance"].max()),
        "rejected_examples": loose["desc"].str.slice(0, 110).to_list()
        + flag_only["desc"].str.slice(0, 110).to_list(),
        "disagreement_examples": disagree["desc"].str.slice(0, 130).to_list(),
    }
    print(f"[1] identification, {report['kickoffs']:,} kickoffs 2016-2025")
    print(
        f"    population ('kicks onside'):   {report['population']:,}, recovered "
        f"{report['recovered']} = {report['league_rate']:.2%}"
    )
    print(f"    rejected, 'onside' but not an attempt: {report['rejected_loose_desc']}")
    print(f"    rejected, flag fires on a deep kickoff: {report['rejected_flag_only']}")
    print(
        f"    flag vs play text on the population: {report['flag_text_agreement']:.1%} agreement, "
        f"{report['flag_text_disagreements']} disagreements"
    )
    for text in report["disagreement_examples"]:
        print(f"      {text}")
    print(
        f"    kick distance: median {report['median_kick_distance']:.0f} yards, "
        f"max {report['max_kick_distance']:.0f}"
    )
    return report


def classes(frame: pl.DataFrame) -> dict:
    """Recovery rate and branch EPA means, per candidate class."""
    table = (
        frame.group_by("onside_class")
        .agg(
            pl.len().alias("n"),
            pl.col("recovered").mean().alias("p_recover"),
            pl.col("epa_kicker").filter(pl.col("recovered") == 1).mean().alias("epa_recovered"),
            pl.col("epa_kicker").filter(pl.col("recovered") == 0).mean().alias("epa_lost"),
        )
        .with_columns((pl.col("epa_recovered") - pl.col("epa_lost")).alias("swing_value"))
        .sort("n", descending=True)
    )
    pooled = {
        "n": frame.height,
        "p_recover": float(frame["recovered"].mean()),
        "epa_recovered": float(frame.filter(pl.col("recovered") == 1)["epa_kicker"].mean()),
        "epa_lost": float(frame.filter(pl.col("recovered") == 0)["epa_kicker"].mean()),
    }
    pooled["swing_value"] = pooled["epa_recovered"] - pooled["epa_lost"]
    print("\n[2] class structure (game state, not outcome)")
    print(table)
    print(f"    pooled: {pooled}")

    by_quarter = (
        frame.group_by("qtr")
        .agg(pl.len().alias("n"), pl.col("recovered").mean().alias("p"))
        .sort("qtr")
    )
    print("\n    by quarter (the crude version of the same split):")
    print(by_quarter)

    by_season = (
        frame.group_by("season")
        .agg(pl.len().alias("n"), pl.col("recovered").mean().alias("p"))
        .sort("season")
    )
    print("\n    by season (the 2018 and 2024 kickoff rule changes are the worry):")
    print(by_season)
    return {
        "classes": table.to_dicts(),
        "pooled": pooled,
        "by_quarter": by_quarter.to_dicts(),
        "by_season": by_season.to_dicts(),
    }


def entity_power(frame: pl.DataFrame) -> dict:
    """Could a per-team spread be resolved at these denominators? (Document 09: no.)"""
    counts = (
        frame.group_by(["season", "kicking_team"])
        .agg(pl.len().alias("n"), pl.col("recovered").sum().cast(pl.Int64).alias("k"))
        .drop_nulls()
        .sort(["season", "kicking_team"])
    )
    print(
        f"\n[3] entity spread instrument: {counts.height} team-seasons, "
        f"median {counts['n'].median():.0f} kicks each"
    )
    return _coinflips.power_table("onside recovery", counts, RANDOM_SEED)


def stakes(frame: pl.DataFrame, pbp: pl.DataFrame) -> dict:
    """How many games carry one, and how big the swing is in points."""
    games = frame["game_id"].n_unique()
    per_game = frame.group_by("game_id").agg(pl.len().alias("kicks"))
    report = {
        "games_with_an_onside_kick": games,
        "games_total": pbp["game_id"].n_unique(),
        "max_kicks_in_a_game": int(per_game["kicks"].max()),
        "mean_kicks_per_such_game": float(per_game["kicks"].mean()),
    }
    print(
        f"\n[4] stakes: {games:,} of {report['games_total']:,} games carry at least one, "
        f"mean {report['mean_kicks_per_such_game']:.2f} per such game, "
        f"max {report['max_kicks_in_a_game']}"
    )
    return report


def main() -> None:
    pbp = load_pbp(PBP_SEASONS, columns=ONSIDE_COLUMNS)
    ident = identification(pbp)
    frame = onside_frame(pbp)
    payload = {
        "random_seed": RANDOM_SEED,
        "identification": ident,
        "classes": classes(frame),
        "entity_power": entity_power(frame),
        "stakes": stakes(frame, pbp),
    }
    out = paths.RESEARCH_OUTPUT_DIR / "32_onside_power.json"
    with out.open("w") as handle:
        json.dump(payload, handle, indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
