"""The scoring seam, on a synthetic cache: no network, no shipped parquet.

`build_frames` reads a cache root, so these tests write a tiny one into tmp_path
— every season the scorer asks for, a handful of games each — and score through
it. That exercises the real loaders and the real draws; the pinned percents of
record need the full cache and live in the slow stage-0 file.
"""

import numpy as np
import pandas as pd
import pytest

from nfl_simulator.process_meter import score
from nfl_simulator.process_meter.posterior import COLUMNS, CoefficientPosterior
from nfl_simulator.process_meter.score import (
    BASELINE_SEASONS,
    SCORE_COLUMNS,
    build_frames,
    likely_points,
    margin_draws,
    score_games,
)
from nfl_simulator.process_meter.weights import load_weights

GAMES_PER_SEASON = 4
PLAYS_PER_SIDE = 30


def _game_plays(game_id, season, week, home, away, rng):
    """One synthetic game: `PLAYS_PER_SIDE` plays each, one takeaway for the home side."""
    rows = []
    for side, (offense, defense) in enumerate(((home, away), (away, home))):
        for i in range(PLAYS_PER_SIDE):
            picked = side == 1 and i == 5  # the away offense throws one pick
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "game_id": game_id,
                    "season_type": "REG",
                    "posteam": offense,
                    "defteam": defense,
                    "home_team": home,
                    "away_team": away,
                    "play_type": "pass" if i % 2 else "run",
                    "epa": float(rng.normal(0.02, 0.6)),
                    "success": int(rng.random() < 0.45),
                    "down": 1 + i % 4,
                    "yards_gained": float(rng.integers(-3, 18)),
                    "interception": int(picked),
                    "fumble_lost": 0,
                    "wp": 0.5,
                    "third_down_converted": 0,
                    "third_down_failed": 0,
                    "two_point_attempt": 0,
                    "return_yards": 22.0 if picked else 0.0,
                    "return_team": home if picked else None,
                    "fumble_recovery_1_team": None,
                    "fumble_recovery_1_yards": np.nan,
                    "fumble_recovery_2_team": None,
                    "fumble_recovery_2_yards": np.nan,
                    "touchdown": 0.0,
                    "td_team": None,
                }
            )
    return rows


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    """A cache root holding every season the scorer reads, `GAMES_PER_SEASON` each."""
    root = tmp_path_factory.mktemp("cache")
    (root / "pbp").mkdir()
    rng = np.random.default_rng(0)
    schedule = []
    for season in (*BASELINE_SEASONS, 2026):
        plays = []
        for g in range(GAMES_PER_SEASON):
            home, away = f"H{g:02d}", f"A{g:02d}"
            game_id = f"{season}_{g + 1:02d}_{away}_{home}"
            plays.extend(_game_plays(game_id, season, g + 1, home, away, rng))
            schedule.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": g + 1,
                    "game_type": "REG",
                    "home_team": home,
                    "away_team": away,
                    "home_score": 17 + g,
                    "away_score": 20 - g,
                }
            )
        pd.DataFrame(plays).to_parquet(root / "pbp" / f"pbp_{season}.parquet")
    pd.DataFrame(schedule).to_parquet(root / "schedules.parquet")
    return root


@pytest.fixture(scope="module")
def frames(cache):
    return build_frames(2026, cache)


@pytest.fixture(scope="module")
def weights():
    return load_weights()


def _ids(frames, n=3):
    return sorted(set(frames[0].game_id))[:n]


# --- the shape --------------------------------------------------------------------------


def test_build_frames_reads_the_baseline_seasons_and_the_live_one():
    assert score._seasons_for(2026) == (*BASELINE_SEASONS, 2026)
    assert score._seasons_for(2025) == BASELINE_SEASONS


def test_the_scored_table_is_score_columns_in_order(weights, frames):
    table = score_games(2026, _ids(frames, 1), weights, frames=frames)
    assert list(table.columns) == SCORE_COLUMNS


def test_the_scoreboard_comes_off_the_schedule(weights, frames):
    game_id = "2026_01_A00_H00"
    row = score_games(2026, [game_id], weights, frames=frames).iloc[0]
    assert (row.home, row.away) == ("H00", "A00")
    assert (row.home_score, row.away_score) == (17, 20)
    assert (row.season, row.week) == (2026, 1)


# --- the seam ---------------------------------------------------------------------------


def test_the_sampler_inputs_are_the_models_inputs(weights, frames):
    """The Beta draws' k / n is the success rate the weights multiply, and the
    Normal draws' mean is the yards per play."""
    table = score_games(2026, _ids(frames), weights, frames=frames)
    for side in ("home", "away"):
        assert np.allclose(table[f"{side}_y_mean"], table[f"{side}_ypp"])
        assert np.allclose(table[f"{side}_k"] / table[f"{side}_n"], table[f"{side}_success_rate"])


def test_the_yards_sample_size_is_the_teams_own_plays(weights, frames):
    table = score_games(2026, _ids(frames), weights, frames=frames)
    assert np.allclose(table.home_n_y, table.home_plays)
    assert (table.home_n >= table.home_n_y).all()


def test_deserved_points_are_the_plug_in_on_the_mean_weights(weights, frames):
    table = score_games(2026, _ids(frames), weights, frames=frames)
    intercept, b_rate, b_ypp = np.asarray(weights.mean, dtype=float)
    for side in ("home", "away"):
        plug_in = intercept + b_rate * table[f"{side}_success_rate"] + b_ypp * table[f"{side}_ypp"]
        assert np.allclose(table[f"{side}_deserved"], plug_in)


def test_the_deserved_margin_is_the_two_sides_apart(weights, frames):
    table = score_games(2026, _ids(frames), weights, frames=frames)
    assert np.allclose(table.deserved_margin, table.home_deserved - table.away_deserved)


# --- the per-game seeding -----------------------------------------------------------------


def test_a_game_scored_alone_equals_the_same_game_in_a_batch(weights, frames):
    ids = _ids(frames)
    batch = score_games(2026, ids, weights, frames=frames).set_index("game_id")
    alone = score_games(2026, [ids[1]], weights, frames=frames)
    assert abs(float(alone.p_home.iloc[0]) - float(batch.p_home[ids[1]])) < 1e-12


def test_reversing_the_batch_moves_nothing(weights, frames):
    ids = _ids(frames)
    forward = score_games(2026, ids, weights, frames=frames)
    backward = score_games(2026, ids[::-1], weights, frames=frames).set_index("game_id")
    for row in forward.itertuples():
        assert abs(row.p_home - backward.p_home[row.game_id]) < 1e-12


# --- the draws --------------------------------------------------------------------------


def test_with_draws_hands_back_the_matrix_beside_the_table(weights, frames):
    table, draws = score_games(
        2026, _ids(frames, 1), weights, frames=frames, n_draws=2_000, with_draws=True
    )
    assert len(table) == 1
    assert draws.shape == (1, 2_000)


def test_margin_draws_returns_one_row_per_game_and_the_draw_count(weights, frames):
    frame, prod_post, scores = frames
    games = score._games(frame, scores, _ids(frames, 2))
    assert margin_draws(games, weights, prod_post, n_draws=1_000).shape == (2, 1_000)


def test_the_draw_mean_is_the_plug_in_margin_plus_the_beta_smoothing(weights, frames):
    """The draws are not centred on the plug-in margin, and the offset is a
    modelling choice rather than noise: a rate is drawn from Beta(k + 1, n - k +
    1), whose mean is the smoothed (k + 1) / (n + 2), while the plug-in reads the
    raw k / n. Subtract the smoothing and what is left is Monte Carlo error,
    which is the check that would catch a real drift between the two."""
    table, draws = score_games(2026, _ids(frames), weights, frames=frames, with_draws=True)
    b_rate = float(np.asarray(weights.mean, dtype=float)[1])
    smoothing = b_rate * (
        ((table.home_k + 1) / (table.home_n + 2) - table.home_success_rate)
        - ((table.away_k + 1) / (table.away_n + 2) - table.away_success_rate)
    )
    residual = draws.mean(axis=1) - table.deserved_margin - smoothing
    assert np.abs(residual).max() < 0.15


def test_an_unknown_game_raises_naming_it(weights, frames):
    with pytest.raises(ValueError, match="2026_09_AAA_BBB"):
        score_games(2026, ["2026_09_AAA_BBB"], weights, frames=frames)


def test_scoring_refuses_weights_that_are_not_the_two_own_terms(weights, frames):
    wrong = CoefficientPosterior(
        columns=["intercept", "own_ypp_for"],
        mean=np.array([0.0, 1.0]),
        cov=np.eye(2),
        sigma2=1.0,
        n_rows=10,
    )
    with pytest.raises(ValueError):
        score_games(2026, _ids(frames, 1), wrong, frames=frames)


def test_likely_points_refuses_a_design_that_is_not_the_two_inputs(frames):
    wrong = CoefficientPosterior(
        columns=["intercept", "plays_for"],
        mean=np.array([0.0, 1.0]),
        cov=np.eye(2),
        sigma2=1.0,
        n_rows=10,
    )
    with pytest.raises(ValueError, match="own_success_rate"):
        likely_points(frames[0], wrong)


def test_the_design_columns_are_the_ones_the_weights_carry(weights):
    assert list(weights.columns) == COLUMNS
