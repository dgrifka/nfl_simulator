"""The slope-provenance guard: cached data drift announces itself by name.

exp40 traced a 1e-06-scale margin mystery to upstream nflverse revising
pbp-2020 *values* after the cache was pulled — invisible until an afternoon of
byte-diffs found it. The guard stamps a hash per cached season into the model
metadata at corpus-build time, and the live entry point compares the cache it
is about to read against that stamp, warning with the drifted file's name so a
future refusal starts from the answer instead of the search.
"""

from __future__ import annotations

import warnings

import polars as pl
import pytest

from nfl_simulator.ingest import data_manifest, manifest_drift, warn_on_manifest_drift


@pytest.fixture()
def cache(tmp_path):
    (tmp_path / "pbp").mkdir()
    (tmp_path / "ftn").mkdir()
    pl.DataFrame({"play_id": [1.0, 2.0]}).write_parquet(tmp_path / "pbp" / "pbp_2020.parquet")
    pl.DataFrame({"play_id": [3.0]}).write_parquet(tmp_path / "pbp" / "pbp_2021.parquet")
    pl.DataFrame({"is_drop": [True]}).write_parquet(tmp_path / "ftn" / "ftn_2022.parquet")
    return tmp_path


def test_the_manifest_names_every_cached_season(cache):
    manifest = data_manifest(cache)
    assert set(manifest) == {
        "pbp/pbp_2020.parquet",
        "pbp/pbp_2021.parquet",
        "ftn/ftn_2022.parquet",
    }
    for digest in manifest.values():
        assert len(digest) == 16
        int(digest, 16)  # hex


def test_the_manifest_is_a_function_of_the_bytes(cache):
    before = data_manifest(cache)
    assert data_manifest(cache) == before
    (cache / "pbp" / "pbp_2020.parquet").write_bytes(b"revised upstream")
    after = data_manifest(cache)
    assert after["pbp/pbp_2020.parquet"] != before["pbp/pbp_2020.parquet"]
    assert after["pbp/pbp_2021.parquet"] == before["pbp/pbp_2021.parquet"]


def test_drift_is_empty_when_nothing_moved(cache):
    manifest = data_manifest(cache)
    assert manifest_drift(manifest, manifest) == []


def test_drift_names_the_revised_file_and_the_missing_ones(cache):
    recorded = data_manifest(cache)
    (cache / "pbp" / "pbp_2020.parquet").write_bytes(b"revised upstream")
    (cache / "pbp" / "pbp_2022.parquet").write_bytes(b"new season")
    current = data_manifest(cache)
    del current["pbp/pbp_2021.parquet"]
    assert manifest_drift(recorded, current) == [
        "pbp/pbp_2020.parquet (revised)",
        "pbp/pbp_2021.parquet (missing from cache)",
        "pbp/pbp_2022.parquet (not in the recorded manifest)",
    ]


def test_no_warning_without_a_recorded_manifest(cache):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_on_manifest_drift({"version": "simulator-v1.4"}, cache)


def test_no_warning_when_the_cache_matches(cache):
    metadata = {"data_manifest": data_manifest(cache)}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_on_manifest_drift(metadata, cache)


def test_drift_warns_once_naming_the_file(cache):
    metadata = {"data_manifest": data_manifest(cache)}
    (cache / "pbp" / "pbp_2020.parquet").write_bytes(b"revised upstream")
    with pytest.warns(UserWarning, match=r"pbp/pbp_2020\.parquet \(revised\)"):
        warn_on_manifest_drift(metadata, cache)
