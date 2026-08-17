"""Validation catches the failure modes an nflverse pull actually has."""

from __future__ import annotations

import polars as pl
import pytest

from nfl_simulator.validate import (
    expected_game_count,
    validate_ftn_season,
    validate_pbp_season,
)


class TestExpectedGameCount:
    @pytest.mark.parametrize(
        ("season", "expected"),
        [
            (2016, 267),  # 256 regular + 11 playoff (12-team field)
            (2019, 267),
            (2020, 269),  # playoff field expands to 14 teams
            (2021, 285),  # 17-game regular season
            (2022, 284),  # Bills-Bengals abandoned, never replayed
            (2025, 285),
        ],
    )
    def test_known_seasons(self, season, expected):
        assert expected_game_count(season) == expected


class TestPbpValidation:
    def test_clean_season_passes(self, synthetic_pbp):
        report = validate_pbp_season(synthetic_pbp(2024), 2024)
        assert report.ok, report.summary()
        assert report.n_games == expected_game_count(2024)
        assert report.n_rows == report.n_games * 160

    def test_missing_column_is_an_error(self, synthetic_pbp):
        df = synthetic_pbp(2024).drop("epa")
        report = validate_pbp_season(df, 2024)
        assert not report.ok
        assert "missing required columns" in report.errors[0]
        assert "epa" in report.errors[0]

    def test_empty_frame_is_an_error(self, synthetic_pbp):
        df = synthetic_pbp(2024).head(0)
        report = validate_pbp_season(df, 2024)
        assert not report.ok
        assert "zero rows" in report.errors[0]

    def test_truncated_pull_is_an_error(self, synthetic_pbp):
        report = validate_pbp_season(synthetic_pbp(2024, n_games=100), 2024)
        assert not report.ok
        assert any("100 games, expected 285" in e for e in report.errors)

    def test_duplicated_games_are_an_error(self, synthetic_pbp):
        """A doubled pull keeps the game count right but doubles every row."""
        df = synthetic_pbp(2024)
        doubled = pl.concat([df, df])
        report = validate_pbp_season(doubled, 2024)
        # game_id count is unchanged, so this has to be caught by plays-per-game.
        assert not report.ok
        assert any("over 300 plays" in e for e in report.errors)

    def test_truncated_game_is_an_error(self, synthetic_pbp):
        df = synthetic_pbp(2024)
        first_game = df.select(pl.col("game_id").first()).item()
        trimmed = pl.concat(
            [
                df.filter(pl.col("game_id") != first_game),
                df.filter(pl.col("game_id") == first_game).head(20),
            ]
        )
        report = validate_pbp_season(trimmed, 2024)
        assert not report.ok
        assert any("under 90 plays" in e for e in report.errors)

    def test_wrong_season_rows_are_an_error(self, synthetic_pbp):
        df = synthetic_pbp(2024).with_columns(
            pl.when(pl.int_range(pl.len()) < 10)
            .then(2023)
            .otherwise(pl.col("season"))
            .alias("season")
        )
        report = validate_pbp_season(df, 2024)
        assert not report.ok
        assert any("season other than 2024" in e for e in report.errors)

    def test_null_epa_on_real_plays_is_an_error(self, synthetic_pbp):
        df = synthetic_pbp(2024).with_columns(pl.lit(None, dtype=pl.Float64).alias("epa"))
        report = validate_pbp_season(df, 2024)
        assert not report.ok
        assert any("epa null" in e for e in report.errors)

    def test_null_epa_on_clock_rows_is_fine(self, synthetic_pbp):
        """Timeouts and end-of-quarter rows carry null EPA in every real season."""
        df = synthetic_pbp(2024, n_plays=140)
        clock_rows = df.head(2 * expected_game_count(2024)).with_columns(
            pl.lit("no_play").alias("play_type"),
            pl.lit(None, dtype=pl.Float64).alias("epa"),
        )
        report = validate_pbp_season(pl.concat([df, clock_rows]), 2024)
        assert report.ok, report.summary()

    def test_no_fumbles_is_only_a_warning(self, synthetic_pbp):
        df = synthetic_pbp(2024).with_columns(pl.lit(0).alias("fumble"))
        report = validate_pbp_season(df, 2024)
        assert report.ok
        assert any("no fumbles" in w for w in report.warnings)


class TestFtnValidation:
    def test_clean_season_passes(self, synthetic_ftn):
        report = validate_ftn_season(synthetic_ftn(2024), 2024)
        assert report.ok, report.summary()
        assert report.n_games == expected_game_count(2024)

    def test_small_shortfall_is_a_warning(self, synthetic_ftn):
        df = synthetic_ftn(2024, n_games=expected_game_count(2024) - 3)
        report = validate_ftn_season(df, 2024)
        assert report.ok
        assert report.warnings

    def test_large_shortfall_is_an_error(self, synthetic_ftn):
        report = validate_ftn_season(synthetic_ftn(2024, n_games=100), 2024)
        assert not report.ok

    def test_duplicate_join_keys_are_an_error(self, synthetic_ftn):
        """A fanned-out join would multiply every FTN-derived rate."""
        df = synthetic_ftn(2024)
        report = validate_ftn_season(pl.concat([df, df]), 2024)
        assert not report.ok
        assert any("duplicate" in e for e in report.errors)


class TestReportSerialisation:
    def test_to_dict_round_trips_findings(self, synthetic_pbp):
        report = validate_pbp_season(synthetic_pbp(2024, n_games=100), 2024)
        payload = report.to_dict()
        assert payload["ok"] is False
        assert payload["dataset"] == "pbp"
        assert payload["season"] == 2024
        assert payload["errors"]
