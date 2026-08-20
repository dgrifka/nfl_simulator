"""The redesigned placement meter — document 36 §2's construction.

The incumbent scored a leverage cell against the team's *own game-wide mean*, so
the bar carried the league's structural profile and an ``n_cell``-scaled score
inherited every bit of the count endogeneity gate M-4 found at +0.5435. The
redesign centres each cell on **what a team of this quality produces there**,
fitted league-wide and leave-one-team-out, before the count multiplies anything.

Four properties carry the whole design, and they are what this module tests:

* **it reduces to the incumbent** — set every ``mu`` to zero and the score is
  document 35's, character for character. A redesign that is not a strict
  generalisation of what it replaces cannot inherit that design's defences;
* **the three cells still sum to zero** — the round-trip check in this design's
  currency, now with a fitted quantity subtracted inside it;
* **the profile shift depends only on the cell counts** — which is document 36
  §6's carry-forward proof, and the reason M-2's band survives untouched;
* **a team's own rows never enter its own baseline** — the fold is the franchise,
  so the fit that scores a team never saw a play that team ran.
"""

from __future__ import annotations

import numpy as np
import pytest

from nfl_simulator.placement import (
    CELL_LATE_DOWN,
    CELL_OTHER,
    CELL_RED_ZONE,
    POINTS_PER_EPA,
    expected_profile,
    leave_one_out_rate,
    loto_weighted_linear_fit,
    profile_shift,
    redesigned_cell_points,
    score_team_game,
)


def _random_team_game(rng, n=40):
    epa = rng.normal(0.0, 1.4, size=n)
    cell = rng.choice([CELL_RED_ZONE, CELL_LATE_DOWN, CELL_OTHER], size=n, p=[0.1, 0.2, 0.7])
    return epa, cell


# --------------------------------------------------------------------------
# the score — document 36 §2
# --------------------------------------------------------------------------


def test_a_zero_profile_reproduces_the_incumbent_score_exactly():
    """M-1's new reduction check, at the unit where it is cheapest to state."""
    rng = np.random.default_rng(36)
    for _ in range(20):
        epa, cell = _random_team_game(rng)
        incumbent = score_team_game(epa, cell)
        reduced = score_team_game(epa, cell, mu=np.zeros(3))
        assert reduced.red_zone == pytest.approx(incumbent.red_zone, abs=1e-12)
        assert reduced.late_down == pytest.approx(incumbent.late_down, abs=1e-12)
        assert reduced.other == pytest.approx(incumbent.other, abs=1e-12)


def test_the_three_cells_still_sum_to_zero_under_a_fitted_profile():
    rng = np.random.default_rng(37)
    for _ in range(20):
        epa, cell = _random_team_game(rng)
        mu = rng.normal(0.0, 0.05, size=3)
        assert score_team_game(epa, cell, mu=mu).identity_residual == pytest.approx(0.0, abs=1e-9)


def test_an_empty_cell_is_exactly_zero_under_a_fitted_profile():
    """Written as a sum minus a count times a mean, so an empty cell is 0, not 0/0."""
    epa = np.array([0.4, -0.2, 1.1, 0.0])
    cell = np.array([CELL_LATE_DOWN, CELL_OTHER, CELL_OTHER, CELL_OTHER])
    scored = score_team_game(epa, cell, mu=np.array([0.007, -0.042, 0.011]))
    assert scored.red_zone == 0.0
    assert scored.n_red_zone == 0


def test_a_team_that_matches_its_profile_everywhere_scores_zero():
    """The redesign's skill defence, in one line.

    The incumbent's was "uniformly good scores zero". This one is stronger and is
    the whole point: a team that is exactly as far above its expected profile in
    every cell has placed nothing anywhere, whatever the profile's shape.
    """
    mu = np.array([0.30, -0.20, 0.05])
    delta = 0.17
    cell = np.array([CELL_RED_ZONE] * 4 + [CELL_LATE_DOWN] * 6 + [CELL_OTHER] * 20)
    epa = mu[cell] + delta
    assert score_team_game(epa, cell, mu=mu).score == pytest.approx(0.0, abs=1e-12)


def test_a_cell_above_its_profile_is_priced_at_the_gap_times_the_count():
    """Hand arithmetic, so the definition is pinned rather than described."""
    mu = np.array([0.5, 0.0, 0.0])
    cell = np.array([CELL_RED_ZONE, CELL_OTHER, CELL_OTHER, CELL_OTHER])
    epa = np.array([2.0, 0.0, 0.0, 0.0])
    # centred values are 1.5, 0, 0, 0; the baseline is 1.5/4 = 0.375
    expected = (1.5 - 0.375) * POINTS_PER_EPA
    assert score_team_game(epa, cell, mu=mu).red_zone == pytest.approx(expected)


def test_the_matrix_form_and_the_per_team_game_form_agree():
    """``redesigned_cell_points`` is the batch path the gates run; it must agree."""
    rng = np.random.default_rng(38)
    n_all, counts, sums, mus, expected = [], [], [], [], []
    for _ in range(25):
        epa, cell = _random_team_game(rng)
        mu = rng.normal(0.0, 0.05, size=3)
        scored = score_team_game(epa, cell, mu=mu)
        n_all.append(len(epa))
        counts.append([int((cell == c).sum()) for c in (CELL_RED_ZONE, CELL_LATE_DOWN, CELL_OTHER)])
        sums.append(
            [float(epa[cell == c].sum()) for c in (CELL_RED_ZONE, CELL_LATE_DOWN, CELL_OTHER)]
        )
        mus.append(mu)
        expected.append([scored.red_zone, scored.late_down, scored.other])
    points = redesigned_cell_points(
        np.array(n_all, dtype=float), np.array(counts, dtype=float), np.array(sums), np.array(mus)
    )
    assert points == pytest.approx(np.array(expected), abs=1e-12)


# --------------------------------------------------------------------------
# the profile shift — document 36 §6's carry-forward proof
# --------------------------------------------------------------------------


def test_the_profile_shift_is_the_gap_between_the_incumbent_and_the_redesign():
    rng = np.random.default_rng(39)
    for _ in range(20):
        epa, cell = _random_team_game(rng)
        mu = rng.normal(0.0, 0.05, size=3)
        counts = np.array(
            [[int((cell == c).sum()) for c in (CELL_RED_ZONE, CELL_LATE_DOWN, CELL_OTHER)]],
            dtype=float,
        )
        shift = profile_shift(counts, mu[None, :], np.array([float(len(epa))]))[0]
        gap = score_team_game(epa, cell).score - score_team_game(epa, cell, mu=mu).score
        assert shift == pytest.approx(gap, abs=1e-12)


def test_the_profile_shift_does_not_move_when_the_labels_are_permuted():
    """Why M-2 carries forward: every rung holds the three cell sizes fixed.

    A null draw re-assigns which plays wear which label but never how many wear
    it, so the profile a draw subtracts is a constant — the same constant the
    realised score subtracts. The PIT is a rank inside the draws and ranks are
    invariant to a common shift.
    """
    rng = np.random.default_rng(40)
    epa, cell = _random_team_game(rng)
    mu = np.array([0.02, -0.05, 0.01])
    counts = np.array(
        [[int((cell == c).sum()) for c in (CELL_RED_ZONE, CELL_LATE_DOWN, CELL_OTHER)]], dtype=float
    )
    n_all = np.array([float(len(epa))])
    shift = profile_shift(counts, mu[None, :], n_all)[0]

    for _ in range(10):
        drawn = rng.permutation(cell)
        incumbent = score_team_game(epa, drawn).score
        redesigned = score_team_game(epa, drawn, mu=mu).score
        assert incumbent - redesigned == pytest.approx(shift, abs=1e-12)


# --------------------------------------------------------------------------
# the leave-one-team-out fit — document 36 §2
# --------------------------------------------------------------------------


def test_a_teams_fitted_value_comes_from_a_line_it_did_not_help_draw():
    """The fold is the franchise. Checked against the line fitted from the rest."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 10.0, 11.0])
    y = np.array([1.0, 3.0, 5.0, 7.0, 99.0, -99.0])  # team B's rows are wild
    w = np.ones(6)
    group = np.array(["A", "A", "A", "A", "B", "B"])

    fitted = loto_weighted_linear_fit(y, w, x, group)
    # A's fitted values must be B's line: through (10, 99) and (11, -99).
    slope = (-99.0 - 99.0) / (11.0 - 10.0)
    intercept = 99.0 - slope * 10.0
    assert fitted[:4] == pytest.approx(intercept + slope * x[:4])
    # and B's must be A's line, y = 1 + 2x, which A's four rows draw exactly.
    assert fitted[4:] == pytest.approx(1.0 + 2.0 * x[4:])


def test_a_perfect_line_is_recovered_whatever_the_folds_are():
    rng = np.random.default_rng(41)
    x = rng.normal(0.0, 1.0, size=60)
    y = 2.0 + 3.0 * x
    w = rng.uniform(1.0, 20.0, size=60)
    group = rng.choice(list("ABCDE"), size=60)
    assert loto_weighted_linear_fit(y, w, x, group) == pytest.approx(y, abs=1e-9)


def test_the_fit_is_weighted_by_the_count_the_score_multiplies():
    """A heavy row pulls the line; the same row weighted lightly does not.

    This is not a stylistic choice. The weight is the cell count the score
    multiplies that cell by, which makes the fit's own orthogonality condition
    *be* the leak condition rather than merely resemble it.
    """
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 20.0])
    group = np.array(["A", "A", "A", "B", "B", "B"])

    light = loto_weighted_linear_fit(y, np.ones(6), x, group)
    heavy = loto_weighted_linear_fit(y, np.array([1.0, 1.0, 1.0, 1.0, 1.0, 50.0]), x, group)
    # Group A is scored by B's line, and B's line is the one the outlier moves.
    assert heavy[0] != pytest.approx(light[0])
    assert abs(heavy[0]) > abs(light[0])


def test_a_zero_weight_row_cannot_move_anyone_elses_fitted_value():
    """An empty cell contributes no play, so it contributes no evidence either."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 3.0, 5.0, 7.0, 1000.0])
    group = np.array(["A", "A", "B", "B", "C"])
    w_zero = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    w_dropped = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    y_dropped = y.copy()
    y_dropped[4] = -1234.0  # any value at all, since its weight is zero

    a = loto_weighted_linear_fit(y, w_zero, x, group)
    b = loto_weighted_linear_fit(y_dropped, w_dropped, x, group)
    assert a[:4] == pytest.approx(b[:4])


def test_the_expected_profile_fits_one_line_per_cell():
    rng = np.random.default_rng(42)
    rows = 200
    s0 = rng.normal(0.0, 0.05, size=rows)
    counts = rng.integers(1, 12, size=(rows, 3)).astype(float)
    truth = np.array([[0.01, 0.8], [-0.04, 1.2], [0.01, 0.6]])  # intercept, slope per cell
    means = truth[:, 0][None, :] + np.outer(s0, truth[:, 1])
    sums = counts * means
    group = rng.choice(list("ABCDEFGH"), size=rows)

    mu = expected_profile(counts, sums, s0, group)
    assert mu.shape == (rows, 3)
    assert mu == pytest.approx(means, abs=1e-9)


def test_an_empty_cell_does_not_contaminate_its_own_profile_fit():
    """A team-game with no red-zone play has a red-zone mean of 0/0.

    It must enter the fit at weight zero rather than as a cell mean of zero,
    which would drag the whole profile toward the origin.
    """
    rng = np.random.default_rng(43)
    rows = 120
    s0 = rng.normal(0.0, 0.05, size=rows)
    counts = rng.integers(1, 10, size=(rows, 3)).astype(float)
    truth = np.array([[0.02, 0.9], [-0.03, 1.1], [0.00, 0.5]])
    sums = counts * (truth[:, 0][None, :] + np.outer(s0, truth[:, 1]))
    group = rng.choice(list("ABCDEFGH"), size=rows)

    empty = counts.copy()
    empty[:20, CELL_RED_ZONE] = 0.0
    empty_sums = sums.copy()
    empty_sums[:20, CELL_RED_ZONE] = 0.0

    mu = expected_profile(empty, empty_sums, s0, group)
    assert mu[20:, CELL_RED_ZONE] == pytest.approx(
        (truth[CELL_RED_ZONE, 0] + truth[CELL_RED_ZONE, 1] * s0)[20:], abs=1e-9
    )


# --------------------------------------------------------------------------
# leave-one-game-out quality — document 36 §2's one new input
# --------------------------------------------------------------------------


def test_quality_is_the_rest_of_the_season_and_never_the_game_being_scored():
    epa = np.array([10.0, 20.0, 30.0, 5.0])
    n = np.array([10.0, 10.0, 10.0, 5.0])
    group = np.array(["S1", "S1", "S1", "S2"])
    rate = leave_one_out_rate(epa, n, group)
    assert rate[0] == pytest.approx((20.0 + 30.0) / 20.0)
    assert rate[1] == pytest.approx((10.0 + 30.0) / 20.0)
    assert rate[2] == pytest.approx((10.0 + 20.0) / 20.0)
    assert np.isnan(rate[3])  # a one-game season has no rest of season
