"""The placement meter: a points-scaled score for where a team's production landed.

Document 34 designs it, document 35 pre-registers it. Two properties carry the
whole design and are therefore tested hardest:

* **the baseline defends against skill** — a team whose production is uniform
  across the field scores exactly zero, by construction rather than by
  correction. That is what killed DQW% (document 08 §10-11) and it is closed
  here at the definition;
* **the three cells sum to zero** — the round-trip check in this design's
  currency. A score whose parts do not add up is not a decomposition.

The rungs are tested for the properties document 35 §5 *derives*, not for their
sampling noise: rung 4 must reduce to rung 1 when its stretch factors are 1, and
the down-stratified rungs must freeze the late-down contribution outright.
"""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
import pytest

from nfl_simulator.placement import (
    CELL_LATE_DOWN,
    CELL_OTHER,
    CELL_RED_ZONE,
    LADDER,
    POINTS_PER_EPA,
    assign_cells,
    band_from_draws,
    game_differential,
    permutation_draws,
    pit_of,
    price_plays,
    score_team_game,
)


def plays(rows: list[dict]) -> pl.DataFrame:
    """A play frame with the columns the meter reads, defaults filled in."""
    default = {
        "game_id": "2024_01_AAA_BBB",
        "play_id": 1.0,
        "posteam": "AAA",
        "home_team": "AAA",
        "away_team": "BBB",
        "epa": 0.0,
        "yardline_100": 50,
        "down": 1,
    }
    return pl.DataFrame([default | row for row in rows])


# --------------------------------------------------------------------------
# cells — disjoint, and in the order document 35 §2 applies them
# --------------------------------------------------------------------------


def test_a_play_inside_the_twenty_is_a_red_zone_play_on_any_down():
    cells = assign_cells(plays([{"yardline_100": 20, "down": 1}]))["cell"]
    assert cells.to_list() == [CELL_RED_ZONE]


def test_a_late_down_inside_the_twenty_is_red_zone_not_late_down():
    cells = assign_cells(plays([{"yardline_100": 8, "down": 3}]))["cell"]
    assert cells.to_list() == [CELL_RED_ZONE]


def test_a_late_down_outside_the_twenty_is_a_late_down_play():
    cells = assign_cells(plays([{"yardline_100": 21, "down": 4}]))["cell"]
    assert cells.to_list() == [CELL_LATE_DOWN]


def test_an_early_down_outside_the_twenty_is_neither_leverage_cell():
    cells = assign_cells(plays([{"yardline_100": 75, "down": 2}]))["cell"]
    assert cells.to_list() == [CELL_OTHER]


# --------------------------------------------------------------------------
# the luck-priced input stream — document 35 §2
# --------------------------------------------------------------------------


def test_a_play_with_no_ledger_row_keeps_its_epa():
    priced = price_plays(plays([{"epa": 1.25}]), pl.DataFrame(schema=_ledger_schema()))
    assert priced["epa_priced"].to_list() == pytest.approx([1.25])


def test_a_home_play_has_its_home_signed_luck_subtracted():
    ledger = pl.DataFrame({"game_id": ["2024_01_AAA_BBB"], "play_id": [7.0], "luck_epa": [2.0]})
    priced = price_plays(plays([{"play_id": 7.0, "posteam": "AAA", "epa": 3.0}]), ledger)
    assert priced["epa_priced"].to_list() == pytest.approx([1.0])


def test_an_away_play_has_the_same_luck_row_resigned_before_subtraction():
    ledger = pl.DataFrame({"game_id": ["2024_01_AAA_BBB"], "play_id": [7.0], "luck_epa": [2.0]})
    priced = price_plays(plays([{"play_id": 7.0, "posteam": "BBB", "epa": 3.0}]), ledger)
    assert priced["epa_priced"].to_list() == pytest.approx([5.0])


def test_several_ledger_rows_on_one_play_are_summed_before_subtraction():
    ledger = pl.DataFrame(
        {
            "game_id": ["2024_01_AAA_BBB"] * 2,
            "play_id": [7.0, 7.0],
            "luck_epa": [1.5, -0.5],
        }
    )
    priced = price_plays(plays([{"play_id": 7.0, "epa": 3.0}]), ledger)
    assert priced["epa_priced"].to_list() == pytest.approx([2.0])


def _ledger_schema() -> dict:
    return {"game_id": pl.String, "play_id": pl.Float64, "luck_epa": pl.Float64}


# --------------------------------------------------------------------------
# the score — document 35 §2
# --------------------------------------------------------------------------


def test_a_uniformly_average_team_game_scores_exactly_zero():
    """The skill defence: a team that is equally good everywhere scores 0."""
    epa = np.full(12, 0.37)
    cell = np.array([CELL_RED_ZONE] * 3 + [CELL_LATE_DOWN] * 4 + [CELL_OTHER] * 5)
    assert score_team_game(epa, cell).score == pytest.approx(0.0, abs=1e-12)


def test_a_cell_is_priced_as_its_epa_above_the_team_s_own_game_mean():
    epa = np.array([2.0, 0.0, 0.0, 0.0])
    cell = np.array([CELL_RED_ZONE, CELL_OTHER, CELL_OTHER, CELL_OTHER])
    # one red-zone play 1.5 EPA above the team's own 0.5 mean
    assert score_team_game(epa, cell).red_zone == pytest.approx(1.5 * POINTS_PER_EPA)


def test_the_score_is_the_sum_of_the_two_leverage_cells():
    rng = np.random.default_rng(11)
    epa = rng.normal(0.0, 1.0, 40)
    cell = np.array([CELL_RED_ZONE] * 6 + [CELL_LATE_DOWN] * 9 + [CELL_OTHER] * 25)
    scored = score_team_game(epa, cell)
    assert scored.score == pytest.approx(scored.red_zone + scored.late_down)


def test_a_team_game_with_no_red_zone_plays_scores_exactly_zero_there():
    epa = np.array([1.0, -2.0, 0.5, 0.25])
    cell = np.array([CELL_LATE_DOWN, CELL_OTHER, CELL_OTHER, CELL_OTHER])
    assert score_team_game(epa, cell).red_zone == 0.0


def test_the_three_cells_sum_to_zero():
    rng = np.random.default_rng(3)
    epa = rng.normal(0.05, 1.4, 64)
    cell = rng.permutation([CELL_RED_ZONE] * 9 + [CELL_LATE_DOWN] * 12 + [CELL_OTHER] * 43)
    scored = score_team_game(epa, cell)
    assert abs(scored.red_zone + scored.late_down + scored.other) < 1e-9


def test_a_team_game_with_no_plays_at_all_scores_zero_rather_than_dividing_by_zero():
    scored = score_team_game(np.array([]), np.array([]))
    assert (scored.red_zone, scored.late_down, scored.score) == (0.0, 0.0, 0.0)


def test_the_game_headline_is_the_home_score_minus_the_away_score():
    home = score_team_game(np.array([2.0, 0.0]), np.array([CELL_RED_ZONE, CELL_OTHER]))
    away = score_team_game(np.array([0.0, 1.0]), np.array([CELL_RED_ZONE, CELL_OTHER]))
    assert game_differential(home, away) == pytest.approx(home.score - away.score)


# --------------------------------------------------------------------------
# the permutation band — document 34 §4, document 35 §5
# --------------------------------------------------------------------------


def _team_game(seed: int = 5, n: int = 60):
    rng = np.random.default_rng(seed)
    epa = rng.normal(0.0, 1.3, n)
    cell = np.array([CELL_RED_ZONE] * 8 + [CELL_LATE_DOWN] * 12 + [CELL_OTHER] * (n - 20))
    down = np.array([1, 2] * 4 + [3, 4] * 6 + list(rng.integers(1, 3, n - 20)))
    return epa, cell, down


@pytest.mark.parametrize("rung", LADDER)
def test_every_rung_returns_the_requested_number_of_draws(rung):
    epa, cell, down = _team_game()
    draws = permutation_draws(rung, epa, cell, down, n_draws=64, rng=np.random.default_rng(1))
    assert draws.shape == (64,)


@pytest.mark.parametrize("rung", LADDER)
def test_no_rung_can_move_the_realized_score(rung):
    """The stretch touches the null only — M-1's identities must survive it."""
    epa, cell, down = _team_game()
    before = score_team_game(epa, cell).score
    permutation_draws(rung, epa, cell, down, n_draws=32, rng=np.random.default_rng(2))
    assert score_team_game(epa, cell).score == before


@pytest.mark.parametrize("rung", LADDER)
def test_a_team_game_with_no_leverage_plays_has_a_degenerate_null(rung):
    epa = np.array([0.4, -1.1, 0.9, 0.2])
    cell = np.full(4, CELL_OTHER)
    down = np.array([1, 2, 1, 2])
    draws = permutation_draws(rung, epa, cell, down, n_draws=16, rng=np.random.default_rng(4))
    assert np.all(draws == 0.0)


def test_rung_four_reduces_to_rung_one_when_its_stretch_factors_are_one():
    """Document 35 §5: rung 4 is a strict generalization of the rung above it."""
    epa, cell, down = _team_game(seed=6)
    raw = permutation_draws("raw", epa, cell, down, 40_000, np.random.default_rng(21))
    flat = permutation_draws(
        "raw_var_matched",
        epa,
        cell,
        down,
        40_000,
        np.random.default_rng(22),
        cell_scales=(1.0, 1.0, 1.0),
    )
    assert flat.std() == pytest.approx(raw.std(), rel=0.02)
    assert flat.mean() == pytest.approx(raw.mean(), abs=0.05 * raw.std())


def test_rung_four_widens_the_null_when_a_leverage_cell_is_stretched():
    epa, cell, down = _team_game(seed=7)
    flat = permutation_draws(
        "raw_var_matched",
        epa,
        cell,
        down,
        20_000,
        np.random.default_rng(23),
        cell_scales=(1.0, 1.0, 1.0),
    )
    stretched = permutation_draws(
        "raw_var_matched",
        epa,
        cell,
        down,
        20_000,
        np.random.default_rng(23),
        cell_scales=(1.0892, 1.4116, 0.8237),
    )
    assert stretched.std() > flat.std()


def test_a_down_stratified_rung_freezes_the_late_down_contribution():
    """Document 35 §5, derived: holding down fixed leaves nothing random when
    every play is a late down, because the whole leverage union is then frozen."""
    rng = np.random.default_rng(8)
    epa = rng.normal(0.0, 1.0, 20)
    cell = np.array([CELL_OTHER] * 5 + [CELL_LATE_DOWN] * 15)
    down = np.array([1] * 5 + [3] * 15)
    draws = permutation_draws("down_stratified", epa, cell, down, 200, np.random.default_rng(9))
    assert draws.std() == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# band and PIT
# --------------------------------------------------------------------------


def test_the_band_is_the_eighty_nine_percent_equal_tailed_interval():
    draws = np.arange(1000, dtype=float)
    low, high = band_from_draws(draws)
    assert (low, high) == pytest.approx((np.quantile(draws, 0.055), np.quantile(draws, 0.945)))


def test_a_realized_score_at_the_centre_of_its_null_has_a_pit_of_a_half():
    assert pit_of(0.0, np.array([-2.0, -1.0, 1.0, 2.0])) == pytest.approx(0.5)


def test_the_pit_splits_ties_down_the_middle():
    """Mid-P: a permutation null is discrete, and a one-sided rank would not be
    uniform even under exact exchangeability."""
    assert pit_of(1.0, np.ones(10)) == pytest.approx(0.5)


def test_rung_four_matches_the_definition_of_a_stretched_relabelling():
    """The rung's algebra against the thing it is algebra *for*.

    Rung 4's closed form (document 35 §5) is a shortcut for: relabel the plays,
    stretch each play's deviation by its new cell's factor, and score the
    stretched game with the ordinary scorer — baseline included. A game small
    enough to enumerate pins the shortcut to that definition, which is what
    catches a correction applied to the leverage cells but not to the baseline.
    """
    epa = np.array([1.7, -0.4, 0.9, -1.2])
    cell = np.array([CELL_RED_ZONE, CELL_LATE_DOWN, CELL_OTHER, CELL_OTHER])
    down = np.array([1, 3, 1, 2])
    scales = (1.0892, 1.4116, 0.8237)

    mean_all = epa.mean()
    definitional = set()
    for order in itertools.permutations(range(4)):
        assigned = np.empty(4, dtype=int)
        assigned[list(order[:1])] = CELL_RED_ZONE
        assigned[list(order[1:2])] = CELL_LATE_DOWN
        assigned[list(order[2:])] = CELL_OTHER
        stretched = mean_all + np.array(
            [scales[assigned[i]] * (epa[i] - mean_all) for i in range(4)]
        )
        definitional.add(round(score_team_game(stretched, assigned).score, 6))

    drawn = {
        round(float(value), 6)
        for value in permutation_draws(
            "raw_var_matched", epa, cell, down, 4000, np.random.default_rng(31), scales
        )
    }
    # The 12 distinct relabellings are 0.09 points apart at the closest, so
    # rounding to 1e-6 separates them while absorbing the two routes' last bits.
    assert drawn == definitional  # 4,000 draws over 12 relabellings misses none


def test_rung_three_stretches_the_red_zone_draws_and_rung_two_does_not():
    """The only difference between the two down-stratified rungs is the stretch."""
    epa, cell, down = _team_game(seed=12)
    plain = permutation_draws("down_stratified", epa, cell, down, 5000, np.random.default_rng(41))
    stretched = permutation_draws(
        "down_stratified_var_matched",
        epa,
        cell,
        down,
        5000,
        np.random.default_rng(41),
        cell_scales=(1.0892, 1.0, 1.0),
    )
    assert stretched.std() > plain.std()
