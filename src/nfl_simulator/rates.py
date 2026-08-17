"""Team-season success/opportunity counts for the hierarchical rate models.

Every builder returns the same three columns — ``team_season``, ``n`` (the
opportunity denominator), ``k`` (the successes) — so one model function can fit
all of them. Getting the denominator right is the entire job; a rate without an
opportunity count is the mistake these models exist to avoid.

See `docs/research/03-model-foundations.md` for the modeling rationale.
"""

from __future__ import annotations

import polars as pl

from nfl_simulator.components import live_fumble_mask

# Penalties that are unambiguous and entirely the committing team's doing.
PRE_SNAP_PENALTIES: tuple[str, ...] = (
    "False Start",
    "Delay of Game",
    "Illegal Formation",
    "Illegal Shift",
    "Illegal Motion",
    "Encroachment",
    "Neutral Zone Infraction",
    "Offside on Free Kick",
    "Defensive Offside",
    "Illegal Substitution",
)

# Penalties that are an official's judgment about a continuum. If these behave
# differently from the pre-snap set, the officiating-noise story has support.
JUDGMENT_PENALTIES: tuple[str, ...] = (
    "Offensive Holding",
    "Defensive Holding",
    "Defensive Pass Interference",
    "Offensive Pass Interference",
    "Unnecessary Roughness",
    "Roughing the Passer",
    "Face Mask",
    "Illegal Contact",
    "Illegal Use of Hands",
    "Unsportsmanlike Conduct",
    "Taunting",
)


def _team_season(season: pl.Expr, team: pl.Expr) -> pl.Expr:
    return pl.concat_str([season.cast(pl.String), team], separator="_").alias("team_season")


def fumble_recovery_counts(pbp: pl.DataFrame, *, exclude_aborted: bool = True) -> pl.DataFrame:
    """Live fumbles by each team-season, and how many it recovered.

    Aborted snaps are excluded by default. They are recovered by the offense
    about 76% of the time versus ~42% for normal fumbles, so pooling them would
    manufacture a between-team difference driven by how often a team's centre
    snaps badly — a real skill, but not the one being measured.
    """
    fumbles = pbp.filter(live_fumble_mask())
    if exclude_aborted:
        fumbles = fumbles.filter(~(pl.col("aborted_play") == 1).fill_null(False))

    return (
        fumbles.select(
            _team_season(pl.col("season"), pl.col("fumbled_1_team")),
            (pl.col("fumble_recovery_1_team") == pl.col("fumbled_1_team"))
            .cast(pl.Int64)
            .alias("kept"),
        )
        .group_by("team_season")
        .agg(pl.len().alias("n"), pl.col("kept").sum().alias("k"))
        .sort("team_season")
    )


def interception_conversion_counts(pbp: pl.DataFrame, ftn: pl.DataFrame) -> pl.DataFrame:
    """Charted interception-worthy throws per team-season, and how many were picked.

    Raw interception counts confound "throws bad passes" with "gets away with
    it". Conditioning on FTN's charted interception-worthiness isolates the
    second, which is the part that ought to be luck.

    The join key needs an explicit cast: FTN stores ``nflverse_play_id`` as i32
    while pbp stores ``play_id`` as f64.
    """
    charted = ftn.select(
        pl.col("nflverse_game_id").alias("game_id"),
        pl.col("nflverse_play_id").cast(pl.Float64).alias("play_id"),
        pl.col("is_interception_worthy"),
    ).filter(pl.col("is_interception_worthy"))

    joined = charted.join(
        pbp.select("game_id", "play_id", "season", "posteam", "interception"),
        on=["game_id", "play_id"],
        how="inner",
    )

    return (
        joined.select(
            _team_season(pl.col("season"), pl.col("posteam")),
            (pl.col("interception") == 1).cast(pl.Int64).alias("picked"),
        )
        .group_by("team_season")
        .agg(pl.len().alias("n"), pl.col("picked").sum().alias("k"))
        .sort("team_season")
    )


def penalty_counts(pbp: pl.DataFrame, penalty_types: tuple[str, ...]) -> pl.DataFrame:
    """Plays a team was on the field for, and penalties of the given class it committed.

    The denominator is every play the team participated in on either side of the
    ball, so pre-snap and judgment rates share a scale and can be compared
    directly.
    """
    plays = pbp.filter(pl.col("posteam").is_not_null() & pl.col("defteam").is_not_null())

    on_field = (
        pl.concat(
            [
                plays.select(_team_season(pl.col("season"), pl.col("posteam"))),
                plays.select(_team_season(pl.col("season"), pl.col("defteam"))),
            ]
        )
        .group_by("team_season")
        .agg(pl.len().alias("n"))
    )

    flagged = (
        pbp.filter((pl.col("penalty") == 1) & pl.col("penalty_type").is_in(penalty_types))
        .select(_team_season(pl.col("season"), pl.col("penalty_team")))
        .group_by("team_season")
        .agg(pl.len().alias("k"))
    )

    return (
        on_field.join(flagged, on="team_season", how="left")
        .with_columns(pl.col("k").fill_null(0))
        .sort("team_season")
    )
