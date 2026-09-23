"""The three closed-form posteriors: the weights, a team's rate, a team's yards."""

import numpy as np
import pandas as pd
import pytest
from process_meter_frames import (  # noqa: E402  (tests/ is on sys.path)
    _fumble,
    _game,
    _pick,
    _row,
    _scores,
)

from nfl_simulator.process_meter.features import (
    OWN_TERMS,
    team_game_features_foldn,
    team_game_features_srtake,
)
from nfl_simulator.process_meter.posterior import (
    COLUMNS,
    DRAW_COLUMNS,
    NOISE_ONCE_FLOOR,
    POSTERIOR_COLUMNS,
    YARDS_SIZE_COLUMN,
    CoefficientPosterior,
    coefficient_posterior,
    fit_arm,
    league_means,
    meter_from_margins,
    noise_once_sd,
    percent_pooled,
    points_columns,
    production_posterior,
    production_posterior_foldn,
    with_deserved,
)


def _training_frame(n_games=60, seed=0):
    """A frame with two team rows per game whose points follow the two inputs plus noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_games):
        game_id = f"2016_{g // 16 + 1:02d}_AWY_HOM{g}"
        rates = rng.uniform(0.35, 0.6, size=2)
        ypps = rng.uniform(4.0, 7.0, size=2)
        points = -16.8 + 35.1 * rates + 4.2 * ypps + rng.normal(0, 7.0, size=2)
        for side in (0, 1):
            rows.append(
                {
                    "season": 2016 + g % 8,
                    "week": g // 16 + 1,
                    "game_id": game_id,
                    "posteam": f"T{side}{g}",
                    "defteam": f"T{1 - side}{g}",
                    "home": 1 - side,
                    "own_success_rate_for": rates[side],
                    "own_ypp_for": ypps[side],
                    "own_success_rate_against": 0.0,
                    "own_ypp_against": 0.0,
                    "points_for": points[side],
                    "points_against": points[1 - side],
                }
            )
    return pd.DataFrame(rows)


def _posterior(plays, frame):
    in_frame = plays.merge(frame[["game_id", "posteam"]], on=["game_id", "posteam"], how="inner")
    return production_posterior_foldn(in_frame, frame)


# --- the design -------------------------------------------------------------------------


def test_the_design_is_the_intercept_and_the_two_own_inputs():
    assert COLUMNS == ["intercept", "own_success_rate_for", "own_ypp_for"]
    assert points_columns(OWN_TERMS) == [
        "own_success_rate_for",
        "own_ypp_for",
        "own_success_rate_against",
        "own_ypp_against",
    ]


def test_the_posterior_draw_columns_are_the_four_plus_the_optional_yards_size():
    assert POSTERIOR_COLUMNS == ["k", "n", "y_mean", "y_sd"]
    assert YARDS_SIZE_COLUMN == "n_y"
    assert DRAW_COLUMNS == ["k", "n", "y_mean", "y_sd", "n_y"]


# --- the coefficient posterior ----------------------------------------------------------


def test_the_coefficient_mean_is_the_least_squares_fit_on_the_same_rows():
    frame = _training_frame()
    seasons = tuple(range(2016, 2024))
    post = coefficient_posterior(frame, OWN_TERMS, seasons)
    arm = fit_arm(frame, OWN_TERMS, seasons)
    assert post.columns == COLUMNS
    for name, value in zip(post.columns, post.mean, strict=True):
        assert arm.points_coef[name] == pytest.approx(value, abs=5e-4)


def test_the_covariance_is_sigma_squared_times_the_inverse_gram_and_is_symmetric():
    post = coefficient_posterior(_training_frame(), OWN_TERMS, tuple(range(2016, 2024)))
    assert post.cov.shape == (3, 3)
    assert np.allclose(post.cov, post.cov.T, atol=0)
    assert np.all(np.linalg.eigvalsh(post.cov) > 0)
    assert post.sigma2 > 0


def test_ten_times_the_rows_gives_standard_errors_about_root_ten_smaller():
    seasons = tuple(range(2016, 2024))
    small = coefficient_posterior(_training_frame(40, seed=1), OWN_TERMS, seasons)
    big = coefficient_posterior(_training_frame(400, seed=1), OWN_TERMS, seasons)
    ratio = np.sqrt(np.diag(small.cov)) / np.sqrt(np.diag(big.cov))
    assert (ratio > 2.2).all()


def test_the_deserved_points_are_the_intercept_plus_the_weights_on_the_two_inputs():
    frame = _training_frame()
    arm = fit_arm(frame, OWN_TERMS, tuple(range(2016, 2024)))
    rows = with_deserved(frame, arm)
    one = rows.iloc[0]
    want = (
        arm.points_coef["intercept"]
        + arm.points_coef["own_success_rate_for"] * one.own_success_rate_for
        + arm.points_coef["own_ypp_for"] * one.own_ypp_for
    )
    assert one.deserved_for == pytest.approx(want)
    assert one.deserved_margin == pytest.approx(one.deserved_for - one.deserved_against)


def test_the_league_means_are_taken_over_the_rows_the_fit_trained_on():
    frame = _training_frame()
    seasons = tuple(range(2016, 2024))
    means = league_means(frame, seasons, OWN_TERMS)
    train = frame[frame.season.isin(list(seasons))]
    assert means.rate == pytest.approx(train.own_success_rate_for.mean())
    assert means.ypp == pytest.approx(train.own_ypp_for.mean())


# --- the production posterior -----------------------------------------------------------


def test_production_counts_successes_plays_and_per_play_yards_per_team_game():
    post = production_posterior(_game(ladder=True)).set_index("posteam")
    home = post.loc["HOM"]
    assert home.k == 6
    assert home.n == 10
    assert home.y_mean == pytest.approx(sum(5 + i for i in range(10)) / 10)
    assert home.y_sd == pytest.approx(np.std([5 + i for i in range(10)], ddof=1))


def test_one_play_has_no_spread_so_its_yards_sd_is_zero():
    plays = _game(ladder=True)
    one = plays[plays.posteam == "HOM"].head(1)
    assert production_posterior(one).y_sd.iloc[0] == 0.0


def test_production_refuses_an_unknown_filter():
    with pytest.raises(ValueError, match="garbage_filter"):
        production_posterior(_game(), "wp10_90")


# --- the arm of record's seam -----------------------------------------------------------


def test_the_beta_counts_are_the_folded_ones_and_the_yards_mean_is_the_arms():
    plays = _game(_pick(return_yards=30), _fumble(), ladder=True)
    frame = team_game_features_foldn(plays, _scores(), "all")
    post = _posterior(plays, frame).set_index("posteam")
    home_frame = _row(frame, "HOM")
    assert post.loc["HOM", "k"] == home_frame.own_k_for
    assert post.loc["HOM", "n"] == home_frame.own_n_for
    assert post.loc["HOM", "y_mean"] == pytest.approx(home_frame.own_ypp_for, abs=1e-12)


def test_the_posterior_names_the_own_plays_as_the_yards_sample_size():
    plays = _game(_pick(return_yards=30), _fumble(), ladder=True)
    frame = team_game_features_foldn(plays, _scores(), "all")
    post = _posterior(plays, frame).set_index("posteam")
    own = production_posterior(plays).set_index("posteam")
    assert post.loc["HOM", "n"] == 12  # 10 own plays plus 2 takeaways: the rate's denominator
    assert post.loc["HOM", "n_y"] == 10  # the own plays alone: the yards mean's denominator
    assert own.loc["HOM", "n"] == 10


def test_the_yards_sd_is_the_teams_own_one_with_no_scale_factor_left_in_it():
    plays = _game(_pick(return_yards=30), _fumble(), ladder=True)
    frame = team_game_features_foldn(plays, _scores(), "all")
    post = _posterior(plays, frame).set_index("posteam")
    own = production_posterior(plays).set_index("posteam")
    assert post.loc["HOM", "y_sd"] == pytest.approx(own.loc["HOM", "y_sd"], abs=1e-12)


def test_with_no_takeaway_the_two_sample_sizes_agree():
    plays = _game(_pick(), ladder=True)
    frame = team_game_features_foldn(plays, _scores(), "all")
    away = _posterior(plays, frame).set_index("posteam").loc["AWY"]
    assert away.n_y == away.n


def test_the_posterior_columns_are_the_four_plus_the_yards_size():
    plays = _game(_pick(), _fumble(), ladder=True)
    frame = team_game_features_foldn(plays, _scores(), "all")
    post = _posterior(plays, frame)
    assert list(post.columns)[-5:] == DRAW_COLUMNS


def test_a_team_game_with_no_frame_row_raises():
    plays = _game(_pick(), ladder=True)
    frame = team_game_features_foldn(plays, _scores(), "all")
    with pytest.raises(ValueError, match="no foldn_take_sr row"):
        production_posterior_foldn(plays, frame[frame.posteam == "HOM"])


def test_a_frame_whose_counts_miss_the_rate_raises():
    plays = _game(_pick(), ladder=True)
    frame = team_game_features_foldn(plays, _scores(), "all")
    with pytest.raises(ValueError, match="own_success_rate_for"):
        _posterior(plays, frame.assign(own_k_for=frame.own_k_for + 1))


def test_a_frame_whose_own_play_count_disagrees_with_the_posterior_raises():
    plays = _game(_pick(), ladder=True)
    frame = team_game_features_foldn(plays, _scores(), "all")
    broken = frame.assign(srtake_takeaways_for=frame.srtake_takeaways_for + 1)
    with pytest.raises(ValueError, match="own-play count"):
        _posterior(plays, broken)


def test_the_incumbent_fold_and_the_arm_share_every_count_but_the_yards_mean():
    """`fold_take_sr` and `foldn_take_sr` differ in one number and one number only."""
    plays = _game(_pick(return_yards=30), _fumble(), ladder=True)
    arm = team_game_features_foldn(plays, _scores(), "all")
    incumbent = team_game_features_srtake(plays, _scores(), "all", "fold_take_sr")
    post = _posterior(plays, arm).set_index("posteam")
    assert post.loc["HOM", "y_mean"] == pytest.approx(_row(arm, "HOM").own_ypp_for)
    assert post.loc["HOM", "y_mean"] != pytest.approx(_row(incumbent, "HOM").own_ypp_for)


# --- the residual scale -----------------------------------------------------------------


def test_pooled_noise_once_removes_the_median_production_variance():
    sigma, floored = noise_once_sd(np.array([9.0, 16.0, 25.0]), 8.0, mode="pooled")
    assert sigma == pytest.approx(np.sqrt(64.0 - 16.0))
    assert floored == 0


def test_per_game_noise_once_never_falls_below_the_floor_and_counts_the_hits():
    sigma, floored = noise_once_sd(np.array([1.0, 63.0]), 8.0, mode="per_game")
    assert sigma[0] == pytest.approx(np.sqrt(63.0))
    assert sigma[1] == NOISE_ONCE_FLOOR
    assert floored == 1


def test_noise_once_refuses_an_unknown_mode_and_a_variance_above_the_residual():
    with pytest.raises(ValueError, match="mode"):
        noise_once_sd(np.array([1.0]), 8.0, mode="per_team")
    with pytest.raises(ValueError, match="not below"):
        noise_once_sd(np.array([100.0]), 8.0, mode="pooled")


# --- the percent ------------------------------------------------------------------------


def test_the_percent_is_the_mean_of_phi_over_the_draws():
    from scipy import stats

    games = pd.DataFrame({"game_id": ["G"]})
    draws = np.array([[-3.0, 0.0, 4.0, 10.0]])
    got = meter_from_margins(games, draws, 5.0).p_home_bayes.iloc[0]
    assert got == pytest.approx(stats.norm.cdf(draws[0] / 5.0).mean())


def test_pooled_percent_takes_a_given_sigma_instead_of_measuring_its_own():
    games = pd.DataFrame({"game_id": ["G", "H"]})
    draws = np.random.default_rng(0).normal(2.0, 6.0, size=(2, 500))
    given = percent_pooled(games, draws, margin_sd=None, sigma=5.4528)
    assert given.sigma == 5.4528
    assert np.array_equal(given.p_home, meter_from_margins(games, draws, 5.4528).p_home_bayes)


def test_a_coefficient_posterior_is_a_plain_frozen_record():
    post = CoefficientPosterior(
        columns=COLUMNS, mean=np.zeros(3), cov=np.eye(3), sigma2=1.0, n_rows=10
    )
    assert post.columns == COLUMNS
    with pytest.raises(AttributeError):
        post.sigma2 = 2.0
