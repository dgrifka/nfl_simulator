"""Where an installed wheel finds its cache and its fitted artifacts.

A checkout has `data/` and `research/outputs/` under the repo root, and every
path in the package hangs off that root. An installed wheel has neither: the
traces are gitignored `*.nc` files that are never package data, and the cache is
a 127 MB pull nobody ships. The 2026-09-01 ruling is that both come from named
environment variables, and that a missing one is one clean sentence naming the
variable rather than a `FileNotFoundError` from three frames inside a parquet
reader.
"""

from __future__ import annotations

import pytest

from nfl_simulator import paths


@pytest.fixture(autouse=True)
def no_inherited_overrides(monkeypatch):
    """The developer running this suite may have either variable exported."""
    monkeypatch.delenv(paths.DATA_DIR_ENV, raising=False)
    monkeypatch.delenv(paths.ARTIFACT_DIR_ENV, raising=False)


class TestDataDir:
    def test_defaults_to_the_repos_data_directory(self):
        assert paths.data_dir() == paths.DATA_DIR

    def test_the_environment_variable_moves_it(self, tmp_path, monkeypatch):
        monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path))
        assert paths.data_dir() == tmp_path

    def test_the_season_paths_follow_it(self, tmp_path, monkeypatch):
        monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path))
        assert paths.pbp_path(2026) == tmp_path / "pbp" / "pbp_2026.parquet"
        assert paths.ftn_path(2026) == tmp_path / "ftn" / "ftn_2026.parquet"
        assert paths.schedule_path() == tmp_path / "schedules.parquet"
        assert paths.manifest_path() == tmp_path / "manifest.json"
        assert paths.logo_path("GB") == tmp_path / "logos" / "GB.png"

    def test_a_user_path_is_expanded(self, monkeypatch):
        monkeypatch.setenv(paths.DATA_DIR_ENV, "~/nfl-cache")
        assert "~" not in str(paths.data_dir())


class TestArtifactDir:
    def test_defaults_to_the_repos_research_outputs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(paths, "RESEARCH_OUTPUT_DIR", tmp_path)
        assert paths.artifact_dir() == tmp_path

    def test_the_environment_variable_moves_it(self, tmp_path, monkeypatch):
        monkeypatch.setenv(paths.ARTIFACT_DIR_ENV, str(tmp_path))
        assert paths.artifact_dir() == tmp_path

    def test_an_absent_default_names_the_variable(self, tmp_path, monkeypatch):
        """The wheel case: no repo, no `research/outputs`, no variable set."""
        monkeypatch.setattr(paths, "RESEARCH_OUTPUT_DIR", tmp_path / "gone")
        with pytest.raises(paths.ArtifactDirMissing, match=paths.ARTIFACT_DIR_ENV):
            paths.artifact_dir()

    def test_a_variable_pointing_nowhere_names_itself_and_the_path(self, tmp_path, monkeypatch):
        missing = tmp_path / "not-synced-yet"
        monkeypatch.setenv(paths.ARTIFACT_DIR_ENV, str(missing))
        with pytest.raises(paths.ArtifactDirMissing) as error:
            paths.artifact_dir()
        assert paths.ARTIFACT_DIR_ENV in str(error.value)
        assert str(missing) in str(error.value)

    def test_it_is_not_a_file_not_found_error(self, tmp_path, monkeypatch):
        """`render` degrades to Strict on `FileNotFoundError`.

        A missing trace is a component the render does without. A missing
        artifact *directory* is a misconfigured deploy, and it must stop rather
        than render a quietly reduced edition — so the two cannot share a type.
        """
        monkeypatch.setattr(paths, "RESEARCH_OUTPUT_DIR", tmp_path / "gone")
        with pytest.raises(paths.ArtifactDirMissing):
            paths.artifact_dir()
        assert not issubclass(paths.ArtifactDirMissing, FileNotFoundError)
