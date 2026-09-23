"""The play floor: which finished games have enough play-by-play to score."""

import polars as pl
import pytest

from nfl_simulator.process_meter.readiness import (
    MIN_SCRIMMAGE_PLAYS,
    games_over_the_play_floor,
    readiness_report,
    ready_finals,
)


def _pbp(**counts):
    """`counts` maps a game id to (home plays, away plays); every row is scrimmage EPA."""
    rows = []
    for game_id, sides in counts.items():
        for team, n in zip(("HOM", "AWY"), sides, strict=True):
            for _ in range(n):
                rows.append(
                    {
                        "game_id": game_id,
                        "posteam": team,
                        "play_type": "pass",
                        "epa": 0.1,
                        "two_point_attempt": 0,
                    }
                )
    return pl.DataFrame(
        rows,
        schema={
            "game_id": pl.String,
            "posteam": pl.String,
            "play_type": pl.String,
            "epa": pl.Float64,
            "two_point_attempt": pl.Int64,
        },
    )


def _schedule(*games):
    """`games` are (game_id, week, result); a null result is a game not yet final."""
    return pl.DataFrame(
        [{"game_id": g, "week": w, "result": r} for g, w, r in games],
        schema={"game_id": pl.String, "week": pl.Int64, "result": pl.Float64},
    )


def test_the_floor_is_five_below_the_smallest_real_game():
    assert MIN_SCRIMMAGE_PLAYS == 25


def test_both_sides_must_clear_the_floor():
    over = set(games_over_the_play_floor(_pbp(A=(30, 30), B=(30, 24), C=(24, 30))))
    assert over == {"A"}


def test_a_game_with_only_one_side_is_never_over_the_floor():
    assert list(games_over_the_play_floor(_pbp(A=(40, 0)))) == []


def test_the_floor_counts_only_scrimmage_epa_rows():
    pbp = _pbp(A=(30, 30)).with_columns(
        pl.when(pl.col("posteam") == "AWY")
        .then(pl.lit("punt"))
        .otherwise(pl.col("play_type"))
        .alias("play_type")
    )
    assert list(games_over_the_play_floor(pbp)) == []


def test_a_two_point_try_does_not_count_toward_the_floor():
    pbp = _pbp(A=(25, 25)).with_columns(
        pl.when(pl.col("posteam") == "AWY")
        .then(pl.lit(1))
        .otherwise(pl.col("two_point_attempt"))
        .alias("two_point_attempt")
    )
    assert list(games_over_the_play_floor(pbp)) == []


def test_a_missing_column_raises_and_names_it():
    with pytest.raises(ValueError, match="epa"):
        games_over_the_play_floor(_pbp(A=(30, 30)).drop("epa"))


def test_only_finals_over_the_floor_are_ready():
    schedule = _schedule(("A", 1, 7.0), ("B", 1, 3.0), ("C", 1, None))
    ready = ready_finals(schedule, _pbp(A=(30, 30), B=(30, 10), C=(30, 30)))
    assert list(ready.get_column("game_id")) == ["A"]


def test_a_custom_floor_is_honoured():
    schedule = _schedule(("A", 1, 7.0))
    assert len(ready_finals(schedule, _pbp(A=(10, 10)))) == 0
    assert len(ready_finals(schedule, _pbp(A=(10, 10)), min_plays=5)) == 1


def test_the_report_spells_the_gate_out_per_final_and_agrees_with_ready_finals():
    schedule = _schedule(("A", 1, 7.0), ("B", 1, 3.0), ("C", 2, None))
    pbp = _pbp(A=(30, 30), B=(30, 10), C=(30, 30))
    report = readiness_report(schedule, pbp).sort("game_id")
    assert list(report.get_column("game_id")) == ["A", "B"]  # C is not final
    assert list(report.get_column("has_epa")) == [True, False]
    assert list(report.get_column("ready")) == [True, False]
    assert set(report.filter(pl.col("ready")).get_column("game_id")) == set(
        ready_finals(schedule, pbp).get_column("game_id")
    )


def test_the_report_can_be_narrowed_to_one_week():
    schedule = _schedule(("A", 1, 7.0), ("B", 2, 3.0))
    report = readiness_report(schedule, _pbp(A=(30, 30), B=(30, 30)), week=2)
    assert list(report.get_column("game_id")) == ["B"]
