"""The fitted artifacts, read through the package rather than off `research/`.

`render._read_side` used to `sys.path`-import `research/44_read_side_fix.py`
for two things: the play-by-play column list a replay needs, and the loader
that pairs a posterior with the centring constants it was fitted at. Both are
production reads, and an installed wheel has no `research/` directory to find
them in — so they live in the package now, and document 30's correction still
exists exactly once.
"""

from __future__ import annotations

import importlib
import json
import pathlib
import sys

import pytest

from nfl_simulator import fg_model, ingest, paths

RESEARCH_DIR = pathlib.Path(__file__).resolve().parents[1] / "research"


@pytest.fixture(autouse=True)
def no_inherited_overrides(monkeypatch):
    """Every test here sets its own artifact directory, or asserts there is none.

    Without this, running the suite with `NFL_SIM_ARTIFACT_DIR` exported — which
    is how the live path is exercised — hands the "no artifact directory" tests
    a real one, and they pass by finding what they are asserting is absent.
    """
    monkeypatch.delenv(paths.DATA_DIR_ENV, raising=False)
    monkeypatch.delenv(paths.ARTIFACT_DIR_ENV, raising=False)


def _research_read_side():
    """`research/44_read_side_fix.py`, imported the way its callers import it."""
    if not (RESEARCH_DIR / "44_read_side_fix.py").exists():
        pytest.skip("no research/ directory — this is an installed package")
    if str(RESEARCH_DIR) not in sys.path:
        sys.path.insert(0, str(RESEARCH_DIR))
    return importlib.import_module("44_read_side_fix")


class TestSimulationColumns:
    def test_the_column_list_lives_in_the_package(self):
        assert "kicker_player_id" in ingest.SIM_COLUMNS
        assert "stadium_id" in ingest.SIM_COLUMNS

    def test_it_extends_the_analysis_columns_rather_than_restating_them(self):
        assert set(ingest.ANALYSIS_COLUMNS) <= set(ingest.SIM_COLUMNS)

    def test_the_research_script_re_exports_the_same_object(self):
        """Move, not copy — `research/44` keeps its name and imports the list.

        Ten research scripts read `44_read_side_fix.SIM_COLUMNS` and
        `.load_model`, and document 30's correction is allowed to exist exactly
        once. So the file stays where those scripts import it from and its two
        production names are now the package's own objects, not second copies.
        """
        read_side = _research_read_side()
        assert read_side.SIM_COLUMNS is ingest.SIM_COLUMNS
        assert read_side.load_model is fg_model.load_fitted_model

    def test_render_reads_its_columns_without_touching_the_research_path(self):
        from nfl_simulator import render

        columns = render.simulation_columns()
        assert set(ingest.SIM_COLUMNS) <= set(columns)


class TestFittedModelLoader:
    def test_the_loader_reads_from_the_artifact_dir(self, tmp_path, monkeypatch):
        """The env-named directory, not `research/outputs` under a repo root."""
        monkeypatch.setenv(paths.ARTIFACT_DIR_ENV, str(tmp_path))
        (tmp_path / "summary.json").write_text(json.dumps({"centres": {"wind": 8.0, "temp": 60.0}}))
        with pytest.raises(FileNotFoundError, match=str(tmp_path)):
            fg_model.load_fitted_model("trace.nc", "summary.json")

    def test_a_missing_artifact_dir_names_the_variable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(paths, "RESEARCH_OUTPUT_DIR", tmp_path / "gone")
        with pytest.raises(paths.ArtifactDirMissing, match=paths.ARTIFACT_DIR_ENV):
            fg_model.load_fitted_model("trace.nc", "summary.json")

    def test_it_returns_the_model_and_the_centres_it_was_fitted_at(self, tmp_path, monkeypatch):
        monkeypatch.setenv(paths.ARTIFACT_DIR_ENV, str(tmp_path))
        centres = {"wind": 8.0, "temp": 60.0, "elevation": 0.569}
        (tmp_path / "summary.json").write_text(json.dumps({"centres": centres}))
        captured = {}

        def fake_from_posterior(path, **kwargs):
            captured.update(kwargs)
            captured["path"] = path
            return "model"

        monkeypatch.setattr(fg_model.FieldGoalModel, "from_posterior", fake_from_posterior)
        model, read_centres = fg_model.load_fitted_model("trace.nc", "summary.json")
        assert model == "model"
        assert read_centres == centres
        assert captured["path"] == tmp_path / "trace.nc"
        assert captured["wind_centre"] == 8.0
        assert captured["temp_centre"] == 60.0
        assert captured["elevation_centre"] == 0.569

    def test_a_summary_without_an_elevation_centre_passes_none(self, tmp_path, monkeypatch):
        """A v1.1/v1.2/v1.3 summary loads through this same call unchanged."""
        monkeypatch.setenv(paths.ARTIFACT_DIR_ENV, str(tmp_path))
        (tmp_path / "summary.json").write_text(json.dumps({"centres": {"wind": 8.0, "temp": 60.0}}))
        captured = {}
        monkeypatch.setattr(
            fg_model.FieldGoalModel,
            "from_posterior",
            lambda path, **kwargs: captured.update(kwargs) or "model",
        )
        fg_model.load_fitted_model("trace.nc", "summary.json")
        assert captured["elevation_centre"] is None
