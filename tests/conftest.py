"""Shared fixtures.

The suite is network-free by design: every test builds a small synthetic frame
rather than reading the parquet cache, so CI on a fresh checkout is green.
"""

from __future__ import annotations

import polars as pl
import pytest

from nfl_simulator.validate import expected_game_count


def _game_rows(game_id: str, season: int, n_plays: int, home: str, away: str) -> list[dict]:
    """One synthetic game: `n_plays` alternating-possession rows with valid EPA."""
    rows = []
    for i in range(n_plays):
        offense, defense = (home, away) if i % 2 == 0 else (away, home)
        rows.append(
            {
                "game_id": game_id,
                "season": season,
                "week": 1,
                "posteam": offense,
                "defteam": defense,
                "home_team": home,
                "away_team": away,
                "play_type": "pass" if i % 2 == 0 else "run",
                "epa": 0.1 if i % 2 == 0 else -0.1,
                "fumble": 1 if i == 3 else 0,
                "fumble_lost": 1 if i == 3 else 0,
                "fumble_recovery_1_team": defense if i == 3 else None,
                "interception": 0,
                "field_goal_result": None,
                "kick_distance": None,
                "penalty": 0,
                "penalty_team": None,
                "penalty_yards": None,
                "spread_line": -3.0,
            }
        )
    return rows


@pytest.fixture
def synthetic_pbp():
    """Build a full, valid season of play-by-play at whatever size is asked for."""

    def _build(season: int = 2024, n_games: int | None = None, n_plays: int = 160) -> pl.DataFrame:
        if n_games is None:
            n_games = expected_game_count(season)
        rows: list[dict] = []
        for game in range(n_games):
            home, away = f"H{game % 32:02d}", f"A{game % 32:02d}"
            rows.extend(_game_rows(f"{season}_{game:04d}", season, n_plays, home, away))
        return pl.DataFrame(rows)

    return _build


@pytest.fixture
def synthetic_ftn():
    """A valid FTN charting season keyed to match `synthetic_pbp`."""

    def _build(season: int = 2024, n_games: int | None = None, n_plays: int = 60) -> pl.DataFrame:
        if n_games is None:
            n_games = expected_game_count(season)
        rows = [
            {
                "nflverse_game_id": f"{season}_{game:04d}",
                "nflverse_play_id": play,
                "is_interception_worthy": play % 20 == 0,
                "is_catchable_ball": True,
                "n_pass_rushers": 4,
            }
            for game in range(n_games)
            for play in range(n_plays)
        ]
        return pl.DataFrame(rows)

    return _build
