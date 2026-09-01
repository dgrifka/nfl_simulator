"""Cache and manifest behaviour, with the network stubbed out."""

from __future__ import annotations

import json

import pytest

from nfl_simulator import ingest, paths


class TestSeasonIngest:
    def test_writes_parquet_and_reports_ok(self, temp_data_dir, stub_network):
        result = ingest.ingest_pbp_season(2024)
        assert result.downloaded
        assert result.report.ok
        assert paths.pbp_path(2024).exists()

    def test_existing_file_is_not_redownloaded(self, temp_data_dir, stub_network):
        ingest.ingest_pbp_season(2024)
        assert stub_network["pbp"] == 1
        second = ingest.ingest_pbp_season(2024)
        assert stub_network["pbp"] == 1
        assert not second.downloaded

    def test_force_redownloads(self, temp_data_dir, stub_network):
        ingest.ingest_pbp_season(2024)
        ingest.ingest_pbp_season(2024, force=True)
        assert stub_network["pbp"] == 2

    def test_failed_validation_refuses_to_write(self, temp_data_dir, monkeypatch, synthetic_pbp):
        monkeypatch.setattr(ingest.nfl, "load_pbp", lambda season: synthetic_pbp(season, n_games=5))
        with pytest.raises(ingest.IngestError, match="failed validation"):
            ingest.ingest_pbp_season(2024)
        assert not paths.pbp_path(2024).exists()


class TestManifest:
    def test_records_every_season(self, temp_data_dir, stub_network):
        ingest.ingest_all(pbp_seasons=(2023, 2024), ftn_seasons=(2024,))
        manifest = json.loads(paths.MANIFEST_PATH.read_text())
        assert set(manifest["datasets"]["pbp"]) == {"2023", "2024"}
        assert set(manifest["datasets"]["ftn"]) == {"2024"}

    def test_entry_carries_provenance(self, temp_data_dir, stub_network):
        ingest.ingest_all(pbp_seasons=(2024,), ftn_seasons=())
        entry = json.loads(paths.MANIFEST_PATH.read_text())["datasets"]["pbp"]["2024"]
        assert entry["nflreadpy_version"]
        assert entry["pulled_at"]
        assert entry["validation"]["ok"] is True
        assert entry["validation"]["n_games"] == 285

    def test_rerun_is_idempotent(self, temp_data_dir, stub_network):
        ingest.ingest_all(pbp_seasons=(2024,), ftn_seasons=(2024,))
        downloads_after_first = (stub_network["pbp"], stub_network["ftn"])
        results = ingest.ingest_all(pbp_seasons=(2024,), ftn_seasons=(2024,))
        assert (stub_network["pbp"], stub_network["ftn"]) == downloads_after_first
        assert results == []

    def test_deleted_parquet_is_a_cache_miss(self, temp_data_dir, stub_network):
        """A manifest entry whose file is gone must trigger a re-pull."""
        ingest.ingest_all(pbp_seasons=(2024,), ftn_seasons=())
        paths.pbp_path(2024).unlink()
        ingest.ingest_all(pbp_seasons=(2024,), ftn_seasons=())
        assert stub_network["pbp"] == 2


class TestLoaders:
    def test_load_pbp_concatenates_seasons(self, temp_data_dir, stub_network):
        ingest.ingest_all(pbp_seasons=(2023, 2024), ftn_seasons=())
        df = ingest.load_pbp([2023, 2024])
        assert set(df["season"].unique()) == {2023, 2024}

    def test_load_pbp_column_subset(self, temp_data_dir, stub_network):
        ingest.ingest_all(pbp_seasons=(2024,), ftn_seasons=())
        df = ingest.load_pbp([2024], columns=["game_id", "epa"])
        assert df.columns == ["game_id", "epa"]

    def test_missing_season_raises_with_instructions(self, temp_data_dir):
        with pytest.raises(FileNotFoundError, match="nfl_simulator.ingest"):
            ingest.load_pbp([2024])
