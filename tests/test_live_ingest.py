"""A season the frozen research window does not cover, pulled on demand.

The shipped artifacts are fit on 2016-2025 and `ingest.PBP_SEASONS` says so.
Adjudicating a game the day it goes final needs the *current* season's plays,
which is a different job with a different validation posture: a season three
weeks old is not truncated, it is three weeks old. Everything here is
network-free — the pull is stubbed, exactly as `test_ingest.py` stubs it.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from nfl_simulator import ingest, paths, validate

LIVE_SEASON = 2026


@pytest.fixture
def stub_live_network(monkeypatch, synthetic_pbp, synthetic_ftn):
    """A live season with three weeks played, and a schedule that covers it."""
    calls = {"pbp": 0, "ftn": 0, "schedules": 0}

    def fake_pbp(season):
        calls["pbp"] += 1
        # Far short of a full season, which is what "week 3" looks like.
        return synthetic_pbp(season, n_games=48)

    def fake_ftn(season):
        calls["ftn"] += 1
        raise FileNotFoundError(f"no FTN charting for {season}")

    def fake_schedules(seasons):
        calls["schedules"] += 1
        return pl.DataFrame(
            {
                "game_id": [f"{season}_01_DAL_PHI" for season in seasons],
                "season": list(seasons),
                "home_team": ["PHI"] * len(list(seasons)),
                "away_team": ["DAL"] * len(list(seasons)),
                "home_score": [24] * len(list(seasons)),
                "away_score": [20] * len(list(seasons)),
            }
        )

    monkeypatch.setattr(ingest.nfl, "load_pbp", fake_pbp)
    monkeypatch.setattr(ingest.nfl, "load_ftn_charting", fake_ftn)
    monkeypatch.setattr(ingest.nfl, "load_schedules", fake_schedules)
    return calls


class TestPartialSeasonValidation:
    def test_a_part_played_season_passes_as_partial(self, synthetic_pbp):
        frame = synthetic_pbp(LIVE_SEASON, n_games=48)
        report = validate.validate_pbp_season(frame, LIVE_SEASON, partial=True)
        assert report.ok
        assert any("48" in warning for warning in report.warnings)

    def test_the_same_shortfall_still_fails_a_finished_season(self, synthetic_pbp):
        frame = synthetic_pbp(LIVE_SEASON, n_games=48)
        report = validate.validate_pbp_season(frame, LIVE_SEASON)
        assert not report.ok

    def test_partial_still_refuses_more_games_than_a_season_holds(self, synthetic_pbp):
        """Fewer games is a young season; more is duplicate game ids, always."""
        frame = synthetic_pbp(LIVE_SEASON, n_games=400)
        report = validate.validate_pbp_season(frame, LIVE_SEASON, partial=True)
        assert not report.ok

    def test_partial_still_refuses_a_wrong_season(self, synthetic_pbp):
        frame = synthetic_pbp(2024, n_games=48)
        report = validate.validate_pbp_season(frame, LIVE_SEASON, partial=True)
        assert not report.ok


class TestLiveSeasonIngest:
    def test_a_season_outside_the_frozen_window_round_trips_through_the_cache(
        self, temp_data_dir, stub_live_network
    ):
        assert LIVE_SEASON not in ingest.PBP_SEASONS
        ingest.ingest_live_season(LIVE_SEASON)
        assert paths.pbp_path(LIVE_SEASON).exists()
        frame = ingest.load_pbp([LIVE_SEASON])
        assert set(frame["season"].unique()) == {LIVE_SEASON}

    def test_the_frozen_constants_are_not_widened_by_a_live_pull(
        self, temp_data_dir, stub_live_network
    ):
        ingest.ingest_live_season(LIVE_SEASON)
        assert tuple(range(2016, 2026)) == ingest.PBP_SEASONS
        assert LIVE_SEASON not in ingest.FTN_SEASONS

    def test_the_manifest_records_the_live_pull_as_partial(self, temp_data_dir, stub_live_network):
        ingest.ingest_live_season(LIVE_SEASON)
        entry = json.loads(paths.MANIFEST_PATH.read_text())["datasets"]["pbp"][str(LIVE_SEASON)]
        assert entry["validation"]["ok"] is True
        assert entry["partial"] is True

    def test_a_cached_live_season_is_re_pulled_because_it_is_still_growing(
        self, temp_data_dir, stub_live_network
    ):
        """A finished season is frozen; a live one gains a week every week."""
        ingest.ingest_live_season(LIVE_SEASON)
        ingest.ingest_live_season(LIVE_SEASON)
        assert stub_live_network["pbp"] == 2

    def test_a_season_that_has_not_started_is_one_clean_error(
        self, temp_data_dir, monkeypatch, stub_live_network
    ):
        monkeypatch.setattr(ingest.nfl, "load_pbp", lambda season: pl.DataFrame())
        with pytest.raises(ingest.IngestError, match="no play-by-play"):
            ingest.ingest_live_season(LIVE_SEASON)
        assert not paths.pbp_path(LIVE_SEASON).exists()


class TestSchedules:
    def test_refresh_adds_a_season_without_dropping_the_cached_ones(
        self, temp_data_dir, stub_live_network
    ):
        ingest.ingest_schedules([2024, 2025])
        ingest.refresh_schedules([LIVE_SEASON])
        seasons = set(ingest.load_schedules()["season"].unique())
        assert seasons == {2024, 2025, LIVE_SEASON}

    def test_refresh_replaces_a_season_rather_than_duplicating_it(
        self, temp_data_dir, stub_live_network
    ):
        ingest.refresh_schedules([LIVE_SEASON])
        ingest.refresh_schedules([LIVE_SEASON])
        frame = ingest.load_schedules()
        assert frame.height == 1

    def test_schedule_row_finds_the_game(self, temp_data_dir, stub_live_network):
        ingest.refresh_schedules([LIVE_SEASON])
        row = ingest.schedule_row(f"{LIVE_SEASON}_01_DAL_PHI")
        assert row["home_team"] == "PHI"
        assert row["home_score"] == 24

    def test_schedule_row_is_empty_for_a_game_not_on_file(self, temp_data_dir, stub_live_network):
        ingest.refresh_schedules([LIVE_SEASON])
        assert ingest.schedule_row(f"{LIVE_SEASON}_01_GB_DET") == {}


class TestOptionalCharting:
    def test_charting_that_does_not_exist_yet_reads_as_none(self, temp_data_dir, stub_live_network):
        assert ingest.load_ftn_if_cached([LIVE_SEASON]) is None

    def test_charting_that_is_cached_reads_as_a_frame(self, temp_data_dir, stub_network):
        ingest.ingest_ftn_season(2024)
        frame = ingest.load_ftn_if_cached([2024])
        assert frame is not None
        assert frame.height > 0
