"""Denominators are the whole job in the rate models — pin them down."""

from __future__ import annotations

import polars as pl

from nfl_simulator.rates import (
    JUDGMENT_PENALTIES,
    PRE_SNAP_PENALTIES,
    fumble_recovery_counts,
    interception_conversion_counts,
    penalty_counts,
)


def play(**overrides) -> dict:
    row = {
        "game_id": "2024_01_AAA_BBB",
        "play_id": 1.0,
        "season": 2024,
        "posteam": "AAA",
        "defteam": "BBB",
        "play_type": "run",
        "epa": 0.0,
        "fumble": 0,
        "fumbled_1_team": None,
        "fumble_recovery_1_team": None,
        "aborted_play": 0,
        "interception": 0,
        "penalty": 0,
        "penalty_type": None,
        "penalty_team": None,
    }
    row.update(overrides)
    return row


class TestFumbleCounts:
    def test_counts_by_fumbling_team_not_offense(self):
        """A defender who fumbles after a pick is charged to the defense."""
        df = pl.DataFrame(
            [
                play(play_id=1.0, fumble=1, fumbled_1_team="BBB", fumble_recovery_1_team="BBB"),
                play(play_id=2.0, fumble=1, fumbled_1_team="AAA", fumble_recovery_1_team="BBB"),
            ]
        )
        counts = fumble_recovery_counts(df).sort("team_season")
        assert counts["team_season"].to_list() == ["2024_AAA", "2024_BBB"]
        assert counts["n"].to_list() == [1, 1]
        assert counts["k"].to_list() == [0, 1]

    def test_aborted_snaps_excluded_by_default(self):
        df = pl.DataFrame(
            [
                play(
                    play_id=1.0,
                    fumble=1,
                    aborted_play=1,
                    fumbled_1_team="AAA",
                    fumble_recovery_1_team="AAA",
                ),
                play(play_id=2.0, fumble=1, fumbled_1_team="AAA", fumble_recovery_1_team="AAA"),
            ]
        )
        assert fumble_recovery_counts(df)["n"].item() == 1
        assert fumble_recovery_counts(df, exclude_aborted=False)["n"].item() == 2

    def test_out_of_bounds_fumbles_are_not_opportunities(self):
        df = pl.DataFrame([play(fumble=1, fumbled_1_team="AAA", fumble_recovery_1_team=None)])
        assert fumble_recovery_counts(df).height == 0


class TestInterceptionCounts:
    def test_denominator_is_interception_worthy_throws_only(self):
        pbp = pl.DataFrame(
            [
                play(play_id=1.0, play_type="pass", interception=1),
                play(play_id=2.0, play_type="pass", interception=0),
                play(play_id=3.0, play_type="pass", interception=1),
            ]
        )
        ftn = pl.DataFrame(
            {
                "nflverse_game_id": ["2024_01_AAA_BBB"] * 3,
                "nflverse_play_id": [1, 2, 3],
                "is_interception_worthy": [True, True, False],
            }
        )
        counts = interception_conversion_counts(pbp, ftn)
        # Play 3 was picked but was not charted worthy, so it is out of both columns.
        assert counts["n"].item() == 2
        assert counts["k"].item() == 1

    def test_join_survives_the_play_id_dtype_mismatch(self):
        """FTN stores play_id as i32, pbp as f64. A silent empty join would look
        like 'no interception-worthy throws' rather than an error."""
        pbp = pl.DataFrame([play(play_id=7.0, play_type="pass", interception=1)])
        ftn = pl.DataFrame(
            {
                "nflverse_game_id": ["2024_01_AAA_BBB"],
                "nflverse_play_id": pl.Series([7], dtype=pl.Int32),
                "is_interception_worthy": [True],
            }
        )
        assert interception_conversion_counts(pbp, ftn)["n"].item() == 1


class TestPenaltyCounts:
    def test_denominator_counts_both_sides_of_the_ball(self):
        df = pl.DataFrame([play(play_id=float(i)) for i in range(5)])
        counts = penalty_counts(df, PRE_SNAP_PENALTIES).sort("team_season")
        # Five plays, two teams on the field for each.
        assert counts["n"].to_list() == [5, 5]
        assert counts["k"].to_list() == [0, 0]

    def test_penalty_charged_to_the_flagged_team(self):
        df = pl.DataFrame(
            [
                play(play_id=1.0, penalty=1, penalty_type="False Start", penalty_team="AAA"),
                play(play_id=2.0, penalty=1, penalty_type="Defensive Holding", penalty_team="BBB"),
            ]
        )
        pre_snap = penalty_counts(df, PRE_SNAP_PENALTIES).sort("team_season")
        assert pre_snap.filter(pl.col("team_season") == "2024_AAA")["k"].item() == 1
        assert pre_snap.filter(pl.col("team_season") == "2024_BBB")["k"].item() == 0

        judgment = penalty_counts(df, JUDGMENT_PENALTIES).sort("team_season")
        assert judgment.filter(pl.col("team_season") == "2024_BBB")["k"].item() == 1

    def test_penalty_classes_do_not_overlap(self):
        assert not set(PRE_SNAP_PENALTIES) & set(JUDGMENT_PENALTIES)
