"""Which finished games have enough play-by-play to score.

The rule is never the clock. A game is ready when the schedule says it is final
**and** the play-by-play carries enough scrimmage EPA rows for both teams.
nflverse publishes play-by-play about an hour after a final, and a partial pull —
mid-write, or a game still being ingested — can carry a handful of plays and
would otherwise score a final off a fraction of it.

**The play floor.** Both teams must clear :data:`MIN_SCRIMMAGE_PLAYS` pass-or-run
EPA rows. A game under the floor is not skipped, it is simply not ready yet, and
the next check picks it up. The floor counts exactly what the model's own loader
counts, so the number here is the number the scorer will see.

Polars in, Polars out: this is the one place in the subpackage that reads the
schedule and the raw pull rather than a built frame.
"""

from __future__ import annotations

import polars as pl

#: The smallest scrimmage-play count any team had in a completed 2016-2025
#: regular-season game is 30, measured over 5,278 team-games. The floor sits five
#: below it: low enough that no real game is held back, high enough that a partial
#: pull cannot look like a final.
MIN_SCRIMMAGE_PLAYS = 25

#: What the floor counts, matching the model's loader.
SCRIMMAGE_PLAY_TYPES = ("pass", "run")

_PLAY_COUNT_COLUMNS = ("game_id", "posteam", "play_type", "epa", "two_point_attempt")


def games_over_the_play_floor(pbp: pl.DataFrame, min_plays: int = MIN_SCRIMMAGE_PLAYS) -> pl.Series:
    """Game ids where both teams cleared ``min_plays`` scrimmage EPA rows."""
    missing = [c for c in _PLAY_COUNT_COLUMNS if c not in pbp.columns]
    if missing:
        raise ValueError(f"pbp is missing the column(s) the play floor reads: {missing}")
    per_side = (
        pbp.filter(
            pl.col("epa").is_not_null()
            & pl.col("posteam").is_not_null()
            & pl.col("play_type").is_in(SCRIMMAGE_PLAY_TYPES)
            & (pl.col("two_point_attempt") == 0)
        )
        .group_by(["game_id", "posteam"])
        .len()
    )
    per_game = per_side.group_by("game_id").agg(
        pl.col("len").min().alias("fewest"), pl.len().alias("sides")
    )
    return per_game.filter((pl.col("sides") == 2) & (pl.col("fewest") >= min_plays)).get_column(
        "game_id"
    )


def ready_finals(
    schedule: pl.DataFrame, pbp: pl.DataFrame, min_plays: int = MIN_SCRIMMAGE_PLAYS
) -> pl.DataFrame:
    """Schedule rows for finals whose play-by-play carries a full game for both teams.

    ``schedule`` needs a ``result`` column, which nflverse fills once a game is
    final. There is no charting gate: the process meter reads play-by-play only,
    so there is nothing else to wait for.
    """
    return schedule.filter(
        pl.col("result").is_not_null()
        & pl.col("game_id").is_in(games_over_the_play_floor(pbp, min_plays).implode())
    )


def readiness_report(
    schedule: pl.DataFrame,
    pbp: pl.DataFrame,
    week: int | None = None,
    min_plays: int = MIN_SCRIMMAGE_PLAYS,
) -> pl.DataFrame:
    """One row per final with the gate spelled out: game_id, week, has_epa, ready.

    ``has_epa`` is the play floor, not a single row. ``ready`` is the same rule
    :func:`ready_finals` applies, so the two never disagree.
    """
    finals = schedule.filter(pl.col("result").is_not_null())
    if week is not None:
        finals = finals.filter(pl.col("week") == week)
    over = games_over_the_play_floor(pbp, min_plays).implode()
    report = finals.select("game_id", "week", pl.col("game_id").is_in(over).alias("has_epa"))
    return report.with_columns(pl.col("has_epa").alias("ready"))


__all__ = [
    "MIN_SCRIMMAGE_PLAYS",
    "SCRIMMAGE_PLAY_TYPES",
    "games_over_the_play_floor",
    "readiness_report",
    "ready_finals",
]
