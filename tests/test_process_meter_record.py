"""The draws of record: the per-game seed, the two sample sizes, the pooled scale."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from nfl_simulator.process_meter.posterior import (
    COLUMNS,
    CoefficientPosterior,
    LeagueMeans,
    noise_once_sd,
    percent_pooled,
)
from nfl_simulator.process_meter.record import (
    N_DRAWS_RECORD,
    SEED_OF_RECORD,
    SIDES_TOLERANCE,
    SIGMA_ONCE_RECORD,
    game_seed,
    margin_draws_per_game,
    percent_record,
    points_draws_per_game,
    share_draws,
    sides_gap,
)

# The arm of record's own numbers, measured per game at seed 0 on the 543 decided
# 2024-2025 games: the median variance of the margin draws and the fit's residual
# SD. `sigma_once` is sqrt(margin_sd**2 - median margin variance), so the two pin
# SIGMA_ONCE_RECORD. The residual SD is carried unrounded here because the
# published three decimals miss the scale by 6.4e-4, wider than this test's line;
# the convention of record is the rounded one, and doc 75 says so.
ARM_MEDIAN_MARGIN_VAR = 49.0324485506
ARM_MARGIN_SD_FULL = 8.8753920693
ARM_MARGIN_SD_PUBLISHED = 8.875


def _coef_post(seed=0):
    """A three-column posterior shaped like the arm's: intercept, rate, yards."""
    mean = np.array([-16.829, 35.126, 4.178])
    a = np.random.default_rng(seed).normal(size=(3, 3)) * 0.1
    cov = a @ a.T + np.diag([0.5, 4.0, 0.05])
    return CoefficientPosterior(columns=COLUMNS, mean=mean, cov=cov, sigma2=84.6, n_rows=100)


def _toy(n_games=3):
    """`n_games` toy games with a home and an away production row each."""
    games = pd.DataFrame(
        {
            "game_id": [f"2025_0{i + 1}_AAA_BBB" for i in range(n_games)],
            "home": [f"H{i}" for i in range(n_games)],
            "away": [f"A{i}" for i in range(n_games)],
        }
    )
    rows = []
    for i in range(n_games):
        rows.append(
            {
                "game_id": games.game_id[i],
                "posteam": games.home[i],
                "k": 30 + i,
                "n": 60 + 2 * i,
                "y_mean": 5.5,
                "y_sd": 8.0,
            }
        )
        rows.append(
            {
                "game_id": games.game_id[i],
                "posteam": games.away[i],
                "k": 24 + i,
                "n": 58 + i,
                "y_mean": 4.8,
                "y_sd": 7.4,
            }
        )
    return games, pd.DataFrame(rows)


def _draws_by_hand(games, coef_post, prod, n_draws, seed=SEED_OF_RECORD, yards_size=None):
    """The draw stream written out longhand, in the order the docstring promises.

    Deliberately a second implementation rather than a recorded hash of the
    first. A byte hash of these draws would be a hash of the platform, not of the
    behaviour: the multivariate Normal factorises its covariance through LAPACK,
    and macOS and CI's Linux runner do not agree bit for bit. Comparing two
    implementations on one machine pins the draw order and the two sample sizes
    exactly, and travels.
    """
    rows = prod.set_index(["game_id", "posteam"])
    out = np.empty((len(games), n_draws))
    for i, game in enumerate(games.itertuples()):
        rng = np.random.default_rng(game_seed(game.game_id, seed))
        beta = rng.multivariate_normal(coef_post.mean, coef_post.cov, size=n_draws)
        h, a = rows.loc[(game.game_id, game.home)], rows.loc[(game.game_id, game.away)]
        rate_h = rng.beta(h.k + 1, h.n - h.k + 1, size=n_draws)
        rate_a = rng.beta(a.k + 1, a.n - a.k + 1, size=n_draws)
        size_h = h.n if yards_size is None else h[yards_size]
        size_a = a.n if yards_size is None else a[yards_size]
        ypp_h = rng.normal(h.y_mean, h.y_sd / np.sqrt(size_h), size=n_draws)
        ypp_a = rng.normal(a.y_mean, a.y_sd / np.sqrt(size_a), size=n_draws)
        out[i] = beta[:, 1] * (rate_h - rate_a) + beta[:, 2] * (ypp_h - ypp_a)
    return out


# --- the seed ---------------------------------------------------------------------------


def test_game_seed_is_the_sha256_of_the_seed_and_the_game_id():
    want = int.from_bytes(hashlib.sha256(b"0:2025_01_CIN_CLE").digest()[:8], "big")
    assert game_seed("2025_01_CIN_CLE") == want
    assert game_seed("2025_01_CIN_CLE", seed=0) == want


def test_game_seed_is_the_same_on_every_call():
    assert game_seed("2025_04_CHI_LV") == game_seed("2025_04_CHI_LV")


def test_game_seed_differs_across_game_ids_and_across_seed_offsets():
    ids = ["2025_01_CIN_CLE", "2025_01_DAL_PHI", "2025_04_CHI_LV", "2025_18_CAR_TB"]
    assert len({game_seed(g) for g in ids}) == len(ids)
    assert len({game_seed("2025_01_CIN_CLE", seed=s) for s in range(10)}) == 10


def test_the_three_constants_of_record():
    assert SEED_OF_RECORD == 0
    assert N_DRAWS_RECORD == 40_000
    assert SIGMA_ONCE_RECORD == 5.4528


# --- the draws --------------------------------------------------------------------------


def test_margin_draws_return_one_row_per_game_and_n_draws_columns():
    games, prod = _toy(4)
    assert margin_draws_per_game(games, _coef_post(), prod, n_draws=50).shape == (4, 50)


def test_a_games_draws_are_identical_alone_in_a_batch_and_in_a_reversed_batch():
    games, prod = _toy(3)
    coef = _coef_post()
    batch = margin_draws_per_game(games, coef, prod, n_draws=100)
    alone = margin_draws_per_game(games.iloc[[1]], coef, prod, n_draws=100)
    reversed_batch = margin_draws_per_game(
        games.iloc[::-1].reset_index(drop=True), coef, prod, n_draws=100
    )
    assert np.array_equal(alone[0], batch[1])
    assert np.array_equal(reversed_batch[1], batch[1])


def test_a_different_seed_offset_moves_a_games_draws():
    games, prod = _toy(2)
    coef = _coef_post()
    assert not np.array_equal(
        margin_draws_per_game(games, coef, prod, n_draws=100),
        margin_draws_per_game(games, coef, prod, n_draws=100, seed=1),
    )


def test_a_posterior_that_is_not_the_three_design_columns_is_refused():
    games, prod = _toy(1)
    wrong = CoefficientPosterior(
        columns=["intercept", "a", "b"], mean=np.zeros(3), cov=np.eye(3), sigma2=1.0, n_rows=1
    )
    with pytest.raises(ValueError, match="design columns"):
        margin_draws_per_game(games, wrong, prod, n_draws=10)


def test_a_game_with_no_production_row_is_refused():
    games, prod = _toy(2)
    with pytest.raises(ValueError, match="no production"):
        margin_draws_per_game(games, _coef_post(), prod.iloc[1:], n_draws=10)


# --- the two sample sizes ---------------------------------------------------------------


def test_a_posterior_without_n_y_draws_exactly_what_it_drew_before_the_column_existed():
    games, prod = _toy(3)
    coef = _coef_post()
    assert np.array_equal(
        margin_draws_per_game(games, coef, prod, n_draws=200),
        _draws_by_hand(games, coef, prod, 200),
    )


def test_with_n_y_the_yards_draw_narrows_by_it_and_the_rest_of_the_stream_is_unmoved():
    games, prod = _toy(3)
    prod = prod.assign(n_y=prod.n - 4)
    coef = _coef_post()
    assert np.array_equal(
        margin_draws_per_game(games, coef, prod, n_draws=200),
        _draws_by_hand(games, coef, prod, 200, yards_size="n_y"),
    )


def test_n_y_equal_to_n_is_the_same_draws_as_no_n_y_at_all():
    games, prod = _toy(3)
    coef = _coef_post()
    assert np.array_equal(
        margin_draws_per_game(games, coef, prod, n_draws=200),
        margin_draws_per_game(games, coef, prod.assign(n_y=prod.n), n_draws=200),
    )


def test_the_yards_draw_narrows_by_n_y_not_by_n_when_the_column_is_there():
    games, prod = _toy(1)
    coef = CoefficientPosterior(
        columns=COLUMNS,
        mean=np.array([-16.829, 0.0, 4.178]),  # yards only: the rate draw cannot leak in
        cov=np.zeros((3, 3)),
        sigma2=1.0,
        n_rows=100,
    )
    narrow = margin_draws_per_game(games, coef, prod.assign(n_y=prod.n * 4), n_draws=20_000)
    wide = margin_draws_per_game(games, coef, prod.assign(n_y=prod.n), n_draws=20_000)
    assert narrow.std() < wide.std() / 1.9  # halving the SD needs four times the sample


def test_n_y_divides_the_yards_draw_by_its_own_square_root():
    """The identity the explicit column replaces: scaling `y_sd` gives the same draws."""
    games, prod = _toy(2)
    coef = _coef_post()
    explicit = margin_draws_per_game(games, coef, prod.assign(n_y=prod.n - 3), n_draws=400)
    scaled = margin_draws_per_game(
        games, coef, prod.assign(y_sd=prod.y_sd * np.sqrt(prod.n / (prod.n - 3))), n_draws=400
    )
    assert np.allclose(explicit, scaled, rtol=0, atol=1e-12)


def test_a_posterior_carrying_its_own_keys_beside_the_five_is_accepted():
    games, prod = _toy(2)
    full = prod.assign(n_y=prod.n, season=2025, week=1, defteam="ZZZ")
    assert np.array_equal(
        margin_draws_per_game(games, _coef_post(), full, n_draws=100),
        margin_draws_per_game(games, _coef_post(), prod.assign(n_y=prod.n), n_draws=100),
    )


def test_a_posterior_missing_one_of_the_draw_columns_raises():
    games, prod = _toy(2)
    with pytest.raises(KeyError):
        margin_draws_per_game(games, _coef_post(), prod.drop(columns=["y_sd"]), n_draws=10)


# --- the scale and the percent ----------------------------------------------------------


def test_the_record_scale_is_the_arms_own_pooled_noise_once_scale():
    pooled, floored = noise_once_sd(
        np.array([ARM_MEDIAN_MARGIN_VAR]), ARM_MARGIN_SD_PUBLISHED, mode="pooled"
    )
    assert round(pooled, 4) == SIGMA_ONCE_RECORD
    assert floored == 0


def test_the_full_residual_sd_gives_a_different_scale_which_is_why_the_convention_is_named():
    full, _ = noise_once_sd(np.array([ARM_MEDIAN_MARGIN_VAR]), ARM_MARGIN_SD_FULL, mode="pooled")
    assert round(full, 4) == 5.4535
    assert 1e-4 < abs(full - SIGMA_ONCE_RECORD) < 1e-3


def test_percent_record_is_the_pooled_percent_at_the_record_scale():
    games, prod = _toy(5)
    draws = margin_draws_per_game(games, _coef_post(), prod, n_draws=500)
    want = percent_pooled(games, draws, ARM_MARGIN_SD_FULL, sigma=SIGMA_ONCE_RECORD)
    assert np.array_equal(percent_record(games, draws), want.p_home)


def test_percent_record_is_one_number_per_game_strictly_inside_zero_and_one():
    games, prod = _toy(6)
    p = percent_record(games, margin_draws_per_game(games, _coef_post(), prod, n_draws=300))
    assert p.shape == (6,)
    assert ((p > 0) & (p < 1)).all()


# --- the two sides ----------------------------------------------------------------------


def test_home_minus_away_points_is_the_margin_of_record():
    games, prod = _toy(3)
    coef = _coef_post()
    home, away = points_draws_per_game(games, coef, prod, n_draws=500)
    margin = margin_draws_per_game(games, coef, prod, n_draws=500)
    assert home.shape == away.shape == margin.shape
    assert sides_gap(home, away, margin) < SIDES_TOLERANCE


def test_the_intercept_is_one_draw_shared_by_both_sides():
    """Both sides move together when only the intercept does, so the margin is unmoved."""
    games, prod = _toy(1)
    coef = CoefficientPosterior(
        columns=COLUMNS, mean=np.array([10.0, 0.0, 0.0]), cov=np.zeros((3, 3)), sigma2=1.0, n_rows=1
    )
    home, away = points_draws_per_game(games, coef, prod, n_draws=50)
    assert np.allclose(home, 10.0)
    assert np.allclose(away, 10.0)


def test_a_game_alone_draws_the_points_it_draws_in_a_batch():
    games, prod = _toy(3)
    coef = _coef_post()
    batch = points_draws_per_game(games, coef, prod, n_draws=100)
    alone = points_draws_per_game(games.iloc[[2]], coef, prod, n_draws=100)
    assert np.array_equal(alone[0][0], batch[0][2])
    assert np.array_equal(alone[1][0], batch[1][2])


# --- the offence/defence split ----------------------------------------------------------


def test_the_two_shares_sum_to_the_margin_of_record_on_every_draw():
    games, prod = _toy(3)
    coef = _coef_post()
    means = LeagueMeans(rate=0.45, ypp=5.4)
    offense, defense = share_draws(games, coef, prod, means, n_draws=400)
    margin = margin_draws_per_game(games, coef, prod, n_draws=400)
    # the split is the same arithmetic in a different order, so it agrees to
    # floating-point rounding rather than bit for bit
    assert np.max(np.abs(offense + defense - margin)) < SIDES_TOLERANCE


def test_moving_the_origin_moves_points_between_the_shares_and_never_their_sum():
    games, prod = _toy(2)
    coef = _coef_post()
    a = share_draws(games, coef, prod, LeagueMeans(rate=0.45, ypp=5.4), n_draws=200)
    b = share_draws(games, coef, prod, LeagueMeans(rate=0.50, ypp=6.0), n_draws=200)
    assert not np.allclose(a[0], b[0])
    assert np.allclose(a[0] + a[1], b[0] + b[1], rtol=0, atol=1e-12)
