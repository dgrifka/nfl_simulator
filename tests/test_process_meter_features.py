"""The team-game frame: the plain inputs, the takeaway fold and the arm of record."""

import numpy as np
import pandas as pd
import pytest
from process_meter_frames import (  # noqa: E402  (tests/ is on sys.path)
    G1,
    _fumble,
    _game,
    _pick,
    _row,
    _scores,
)

from nfl_simulator.process_meter.features import (
    ARMS,
    FOLD_SR_ARM,
    FOLDN_ARM,
    OWN_TERMS,
    SR_ARM,
    SRTAKE_COLUMNS,
    folded_yards,
    offense_features,
    takeaway_successes,
    team_game_features,
    team_game_features_fold,
    team_game_features_foldn,
    team_game_features_own,
    team_game_features_srtake,
)

# --- the plain frame --------------------------------------------------------------------


def test_a_success_is_a_play_with_epa_above_zero_counted_by_the_success_column():
    """The meter never recounts the flag; nflverse's `success` is EPA above zero."""
    offense = offense_features(_game(home_successes=4)).set_index("posteam")
    assert offense.loc["HOM", "plays"] == 10
    assert offense.loc["HOM", "success_rate"] == pytest.approx(0.4)


def test_the_frame_carries_the_opponent_side_and_the_scoreboard():
    frame = team_game_features(_game(), _scores(), "all")
    home = _row(frame, "HOM")
    assert home.home == 1
    assert home.points_for == 17
    assert home.points_against == 21
    assert home.success_rate_against == pytest.approx(_row(frame, "AWY").success_rate_for)


def test_an_unknown_garbage_filter_raises():
    with pytest.raises(ValueError, match="garbage_filter"):
        team_game_features(_game(), _scores(), "wp10_90")


def test_the_own_terms_get_an_all_zero_against_column():
    frame = team_game_features_own(_game(), _scores(), "all")
    assert OWN_TERMS == ("own_success_rate", "own_ypp")
    for term in OWN_TERMS:
        assert (frame[f"{term}_against"] == 0.0).all()
    assert _row(frame, "HOM").own_ypp_for == pytest.approx(50 / 10)


# --- the fold ---------------------------------------------------------------------------


def test_folded_yards_credit_the_taking_team_and_count_the_takeaway_as_a_play():
    credit = folded_yards(_game(_pick(return_yards=30), _fumble(yards=5.0)), "take")
    assert credit.to_dict("records") == [
        {"game_id": G1, "team": "HOM", "fold_takeaways": 2, "fold_return_yards": 35.0}
    ]


def test_a_notd_fold_drops_a_takeaway_returned_for_a_touchdown_whole():
    scored = _pick(return_yards=60) | {"touchdown": 1.0, "td_team": "HOM"}
    plays = _game(pd.Series(scored).to_dict(), _pick(return_yards=10))
    assert folded_yards(plays, "take").fold_takeaways.iloc[0] == 2
    notd = folded_yards(plays, "take_notd")
    assert notd.fold_takeaways.iloc[0] == 1
    assert notd.fold_return_yards.iloc[0] == 10.0


def test_an_unknown_fold_raises():
    with pytest.raises(ValueError, match="arm"):
        folded_yards(_game(), "fold_take")


def test_the_fold_puts_the_takeaway_in_both_halves_of_yards_per_play():
    plays = _game(_pick(return_yards=30))
    home = _row(team_game_features_fold(plays, _scores(), "all", "take"), "HOM")
    assert home.fold_takeaways_for == 1
    assert home.own_ypp_for == pytest.approx((50 + 30) / 11)


# --- the success rate with the takeaway counted -----------------------------------------


def test_a_takeaway_with_negative_offense_epa_adds_one_play_and_one_success():
    frame = team_game_features_srtake(_game(_pick(epa=-4.0)), _scores(), "all", SR_ARM)
    home = _row(frame, "HOM")
    assert home.srtake_takeaways_for == 1
    assert home.srtake_successes_for == 1
    assert home.own_n_for == 11
    assert home.own_k_for == 7
    assert home.own_success_rate_for == pytest.approx(7 / 11)


def test_a_takeaway_with_positive_offense_epa_adds_one_play_and_no_success():
    frame = team_game_features_srtake(_game(_pick(epa=0.3)), _scores(), "all", SR_ARM)
    home = _row(frame, "HOM")
    assert home.srtake_successes_for == 0
    assert home.own_success_rate_for == pytest.approx(6 / 11)


def test_epa_exactly_zero_or_missing_on_a_takeaway_is_a_failure():
    plays = pd.DataFrame([_pick(epa=0.0), _pick(epa=np.nan), _pick(epa=-1.0)])
    assert takeaway_successes(plays).to_dict("records") == [
        {"game_id": G1, "posteam": "HOM", "srtake_takeaways_for": 3, "srtake_successes_for": 1}
    ]


def test_picks_and_lost_fumbles_both_count_and_the_taker_is_the_defense():
    frame = team_game_features_srtake(
        _game(_pick(), _fumble(), _pick(epa=0.2)), _scores(), "all", SR_ARM
    )
    home, away = _row(frame, "HOM"), _row(frame, "AWY")
    assert (home.srtake_takeaways_for, home.srtake_successes_for) == (3, 2)
    assert (away.srtake_takeaways_for, away.srtake_successes_for) == (0, 0)
    assert home.srtake_takeaways_for == home.fold_takeaways_for


@pytest.mark.parametrize("epas", [(-4.0,), (0.3,), (-4.0, -1.0), (-4.0, 0.2, -0.5), (0.0,)])
def test_the_shift_is_ts_minus_t_k_over_n_all_over_n_plus_t(epas):
    frame = team_game_features_srtake(
        _game(*(_pick(epa=e) for e in epas)), _scores(), "all", SR_ARM
    )
    home = _row(frame, "HOM")
    n, k = 10, 6
    t, ts = len(epas), sum(e < 0 for e in epas)
    assert home.srtake_shift_for == pytest.approx((ts - t * k / n) / (n + t), abs=1e-12)


def test_no_takeaway_leaves_the_plain_rate_exactly():
    plays = _game(_pick())
    plain = _row(team_game_features_own(plays, _scores(), "all"), "AWY")
    for arm in (SR_ARM, FOLD_SR_ARM):
        away = _row(team_game_features_srtake(plays, _scores(), "all", arm), "AWY")
        assert away.srtake_takeaways_for == 0
        assert away.own_success_rate_for == plain.own_success_rate_for
        assert away.srtake_shift_for == 0.0
        assert away.own_n_for == away.plays_for


def test_k_and_n_are_integers_and_k_over_n_is_the_rate():
    frame = team_game_features_srtake(_game(_pick(), _pick(epa=0.4)), _scores(), "all", FOLD_SR_ARM)
    assert frame.own_n_for.dtype.kind == "i"
    assert frame.own_k_for.dtype.kind == "i"
    assert np.abs(frame.own_k_for / frame.own_n_for - frame.own_success_rate_for).max() == 0.0


def test_sr_take_yards_per_play_is_plain_and_fold_take_sr_is_the_folds():
    plays = _game(_pick(return_yards=30), _fumble())
    plain = _row(team_game_features_srtake(plays, _scores(), "all", SR_ARM), "HOM")
    folded = _row(team_game_features_srtake(plays, _scores(), "all", FOLD_SR_ARM), "HOM")
    assert plain.own_ypp_for == pytest.approx(50 / 10)
    assert folded.own_ypp_for == pytest.approx((50 + 30) / 12)


def test_the_wp_filter_applies_to_the_takeaways_as_to_the_plays():
    plays = _game(_pick(), _pick())
    plays.loc[plays.interception == 1, "wp"] = [0.5, 0.99]
    home = _row(team_game_features_srtake(plays, _scores(), "wp05_95", SR_ARM), "HOM")
    assert home.srtake_takeaways_for == 1


def test_an_unknown_srtake_arm_raises():
    with pytest.raises(ValueError, match="arm"):
        team_game_features_srtake(_game(), _scores(), "all", "fold_take")


# --- the arm of record ------------------------------------------------------------------


def _own_yards():
    """HOM's own yards on the ladder game."""
    return sum(5 + i for i in range(10))


def test_the_return_yards_are_credited_but_the_takeaway_is_not_a_play():
    plays = _game(_pick(return_yards=30), ladder=True)
    home = _row(team_game_features_foldn(plays, _scores(), "all"), "HOM")
    assert home.plays_for == 10
    assert home.own_ypp_for == pytest.approx((_own_yards() + 30) / 10)


def test_every_credited_takeaway_adds_its_yards_and_no_play():
    plays = _game(
        _pick(return_yards=30), _fumble(yards=-4.0), _pick(epa=0.2, return_yards=12), ladder=True
    )
    home = _row(team_game_features_foldn(plays, _scores(), "all"), "HOM")
    assert home.fold_takeaways_for == 3
    assert home.own_ypp_for == pytest.approx((_own_yards() + 30 - 4 + 12) / 10)


def test_a_takeaway_lifts_the_yards_per_play_above_the_incumbent_fold():
    """(Y + R) / N is larger than (Y + R) / (N + T) whenever the numerator is positive."""
    plays = _game(_pick(return_yards=30), ladder=True)
    incumbent = _row(team_game_features_srtake(plays, _scores(), "all", FOLD_SR_ARM), "HOM")
    arm = _row(team_game_features_foldn(plays, _scores(), "all"), "HOM")
    assert arm.own_ypp_for == pytest.approx(incumbent.own_ypp_for * 11 / 10)


def test_the_success_rate_and_its_seam_are_the_incumbent_s_everywhere():
    plays = _game(_pick(), _pick(epa=0.2), _fumble(), ladder=True)
    incumbent = team_game_features_srtake(plays, _scores(), "all", FOLD_SR_ARM)
    arm = team_game_features_foldn(plays, _scores(), "all")
    for column in ("own_success_rate_for", "own_k_for", "own_n_for", "srtake_shift_for"):
        pd.testing.assert_series_equal(arm[column], incumbent[column])


def test_only_the_yards_per_play_moves_from_the_incumbent_frame():
    plays = _game(_pick(), _fumble(), ladder=True)
    incumbent = team_game_features_srtake(plays, _scores(), "all", FOLD_SR_ARM)
    arm = team_game_features_foldn(plays, _scores(), "all")
    assert list(arm.columns) == list(incumbent.columns)
    shared = [c for c in incumbent.columns if c != "own_ypp_for"]
    pd.testing.assert_frame_equal(arm[shared], incumbent[shared])
    assert list(incumbent.columns)[-len(SRTAKE_COLUMNS) :] == list(SRTAKE_COLUMNS)


def test_the_wp_filter_applies_to_the_credited_takeaways():
    plays = _game(_pick(return_yards=30), _pick(return_yards=30), ladder=True)
    plays.loc[plays.interception == 1, "wp"] = [0.5, 0.99]
    home = _row(team_game_features_foldn(plays, _scores(), "wp05_95"), "HOM")
    assert home.fold_takeaways_for == 1


def test_the_arms_carry_no_new_feature():
    assert (SR_ARM, FOLD_SR_ARM, FOLDN_ARM) == ("sr_take", "fold_take_sr", "foldn_take_sr")
    assert ARMS == {FOLDN_ARM: OWN_TERMS}
