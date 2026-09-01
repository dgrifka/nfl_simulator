"""The packaged entry point: one game in, four PNGs and a verdict out.

Everything above `TestAcceptance` is network-free and artifact-free — the
pieces the orchestration side's contract rests on, tested without the 4 GB of
posteriors. `TestAcceptance` is the real one, and it is gated on the cache the
way `test_stadium_elevation` gates its completeness check: a fresh clone has no
`data/` and no `research/outputs/`, and the check is about the machine that
holds them.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nfl_simulator import live, paths, render
from nfl_simulator.ingest import FTN_SEASONS, PBP_SEASONS

SHIPPED_GAME = "2025_01_DAL_PHI"
LIVE_SEASON = 2026


def _charting(game_ids: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "nflverse_game_id": game_ids,
            "nflverse_play_id": list(range(len(game_ids))),
            "is_interception_worthy": [False] * len(game_ids),
            "is_catchable_ball": [True] * len(game_ids),
        }
    )


class TestEditionResolution:
    def test_a_charted_game_with_charting_on_file_is_full(self):
        edition, note = live.resolve_edition(
            "2026_01_DAL_PHI", None, ftn=_charting(["2026_01_DAL_PHI"]), models_available=True
        )
        assert edition == "full"
        assert note == ""

    def test_a_game_whose_charting_has_not_landed_degrades_to_strict(self):
        edition, note = live.resolve_edition(
            "2026_01_DAL_PHI", None, ftn=None, models_available=True
        )
        assert edition == "strict"
        assert "charting" in note.lower()

    def test_charting_that_covers_other_games_but_not_this_one_degrades(self):
        edition, note = live.resolve_edition(
            "2026_01_DAL_PHI", None, ftn=_charting(["2026_01_GB_DET"]), models_available=True
        )
        assert edition == "strict"
        assert note

    def test_a_checkout_without_the_variant_traces_degrades(self):
        edition, note = live.resolve_edition(
            "2026_01_DAL_PHI", None, ftn=_charting(["2026_01_DAL_PHI"]), models_available=False
        )
        assert edition == "strict"
        assert note

    def test_a_pre_charting_season_is_strict_with_nothing_to_apologise_for(self):
        edition, note = live.resolve_edition("2018_05_GB_DET", None, ftn=None)
        assert edition == "strict"
        assert note == ""

    def test_an_explicit_full_request_on_an_uncharted_game_still_refuses(self):
        with pytest.raises(ValueError, match="FTN charting starts"):
            live.resolve_edition("2018_05_GB_DET", "full", ftn=None)

    def test_an_explicit_strict_request_is_honoured_on_a_charted_game(self):
        edition, note = live.resolve_edition(
            "2026_01_DAL_PHI", "strict", ftn=_charting(["2026_01_DAL_PHI"])
        )
        assert edition == "strict"
        assert note == ""


class TestHeadline:
    """The one-line biggest swing, and which club it went to."""

    def _row(self, luck_epa: float, **overrides) -> dict:
        row = {
            "play_id": 101.0,
            "component": "field_goal",
            "event_class": "40-44 yd field goal",
            "charged_team": "PHI",
            "opponent": "DAL",
            "actual": 1.0,
            "expected": 0.72,
            "swing": 3.0,
            "luck_epa": luck_epa,
            "kick_distance": 42.0,
            "kicker": "Elliott",
        }
        row.update(overrides)
        return row

    def test_luck_that_favoured_the_home_team_names_the_home_team(self):
        headline = live.headline_from_rows(
            [self._row(0.9)], points_per_epa=0.84, home_team="PHI", away_team="DAL"
        )
        assert headline is not None
        assert headline.endswith("to PHI")

    def test_luck_that_favoured_the_away_team_names_the_away_team(self):
        headline = live.headline_from_rows(
            [self._row(-0.9)], points_per_epa=0.84, home_team="PHI", away_team="DAL"
        )
        assert headline is not None
        assert headline.endswith("to DAL")

    def test_it_is_the_biggest_swing_and_not_the_first_row(self):
        rows = [self._row(0.2, play_id=1.0), self._row(-1.4, play_id=2.0)]
        headline = live.headline_from_rows(
            rows, points_per_epa=0.84, home_team="PHI", away_team="DAL"
        )
        assert "1.2 pt" in headline
        assert headline.endswith("to DAL")

    def test_the_figures_footnote_marker_is_not_carried_into_the_text(self):
        """`drop*` is a pointer to a footnote a one-line headline does not have."""
        row = self._row(
            -1.2,
            component="receiver_drop",
            event_class="receiver drop",
            expected=0.92,
            actual=0.0,
            receiver="Lamb",
        )
        headline = live.headline_from_rows(
            [row], points_per_epa=0.84, home_team="PHI", away_team="DAL"
        )
        assert "*" not in headline
        assert "drop" in headline

    def test_a_game_with_no_luck_events_has_no_headline(self):
        assert (
            live.headline_from_rows([], points_per_epa=0.84, home_team="PHI", away_team="DAL")
            is None
        )

    def test_a_ledger_that_nets_to_nothing_on_every_row_has_no_headline(self):
        headline = live.headline_from_rows(
            [self._row(0.0)], points_per_epa=0.84, home_team="PHI", away_team="DAL"
        )
        assert headline is None


class TestPublicSurface:
    def test_the_documented_fields_all_exist(self):
        """The orchestration side reads these names; they are the contract."""
        fields = set(live.LiveAdjudication.__dataclass_fields__)
        assert {
            "game_id",
            "figures",
            "dtw_home",
            "dtw_low",
            "dtw_high",
            "deserved_margin",
            "actual_margin",
            "home_team",
            "away_team",
            "home_points",
            "away_points",
            "edition",
            "headline",
        } <= fields

    def test_the_entry_point_is_importable_from_the_package_root(self):
        from nfl_simulator import adjudicate_live_game

        assert adjudicate_live_game is live.adjudicate_live_game


class TestMissingCache:
    def test_an_absent_baseline_cache_names_the_data_variable(self, tmp_path, monkeypatch):
        """The baselines are fit on 2016-2025; say where that cache should be."""
        monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path))
        monkeypatch.setenv(paths.ARTIFACT_DIR_ENV, str(tmp_path))
        with pytest.raises(FileNotFoundError, match=paths.DATA_DIR_ENV):
            live.adjudicate_live_game(SHIPPED_GAME, tmp_path / "out", pull=False)


@pytest.mark.slow
class TestAcceptance:
    """A 2025 game the shipped artifacts already cover, adjudicated the new way.

    Nothing in the live path reads this game's shipped row — that is the whole
    point of it — so agreement with the shipped row is a real check rather than
    a tautology.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def adjudication(tmp_path_factory):
        uncached = [s for s in PBP_SEASONS if not paths.pbp_path(s).exists()]
        if uncached:
            pytest.skip(
                f"play-by-play cache absent for {uncached[0]}-{uncached[-1]} — run "
                "`uv run python -m nfl_simulator.ingest` to enable this check"
            )
        try:
            artifacts = paths.artifact_dir()
        except paths.ArtifactDirMissing as error:  # pragma: no cover - fresh clone
            pytest.skip(str(error))
        if not (artifacts / "dtw_games_v14.parquet").exists():
            pytest.skip("the shipped v1.4 summary is not in the artifact directory")
        out = tmp_path_factory.mktemp("live")
        return live.adjudicate_live_game(SHIPPED_GAME, out, pull=False)

    @pytest.fixture(scope="class")
    @staticmethod
    def shipped():
        frame = pl.read_parquet(paths.artifact_dir() / "full_summary_v14.parquet")
        return frame.filter(pl.col("game_id") == SHIPPED_GAME).to_dicts()[0]

    def test_it_writes_the_same_four_figures(self, adjudication):
        assert len(adjudication.figures) == 4
        stem = f"{SHIPPED_GAME}_20-24--96-4_full_"
        assert [path.name for path in adjudication.figures] == [
            f"{stem}{suffix}.png" for suffix in render.SUFFIXES
        ]
        assert all(path.exists() and path.stat().st_size > 0 for path in adjudication.figures)

    def test_the_dtw_matches_the_shipped_row(self, adjudication, shipped):
        assert adjudication.dtw_home == pytest.approx(shipped["dtw_home"], abs=1e-9)

    def test_the_interval_matches_the_shipped_row(self, adjudication, shipped):
        assert adjudication.dtw_low == pytest.approx(shipped["dtw_low"], abs=1e-9)
        assert adjudication.dtw_high == pytest.approx(shipped["dtw_high"], abs=1e-9)

    def test_the_deserved_margin_matches_the_shipped_row(self, adjudication, shipped):
        assert adjudication.deserved_margin == pytest.approx(shipped["deserved_margin"], abs=1e-9)

    def test_it_carries_the_scoreboard_and_the_clubs(self, adjudication):
        assert adjudication.home_team == "PHI"
        assert adjudication.away_team == "DAL"
        assert adjudication.home_points - adjudication.away_points == pytest.approx(
            adjudication.actual_margin
        )

    def test_a_2025_game_is_adjudicated_full(self, adjudication):
        assert 2025 in FTN_SEASONS
        assert adjudication.edition == "full"

    def test_it_has_a_headline(self, adjudication):
        assert isinstance(adjudication.headline, str)
        assert adjudication.headline

    def test_it_is_deterministic(self, adjudication, tmp_path):
        again = live.adjudicate_live_game(SHIPPED_GAME, tmp_path, pull=False)
        assert again.dtw_home == adjudication.dtw_home
        assert again.deserved_margin == adjudication.deserved_margin
        assert again.headline == adjudication.headline

    def test_the_figures_land_in_the_directory_it_was_given(self, adjudication):
        assert all(isinstance(path, Path) for path in adjudication.figures)
        assert len({path.parent for path in adjudication.figures}) == 1
