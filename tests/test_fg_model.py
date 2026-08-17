"""The field-goal make model, as the simulator consumes it.

The research script fits it; this is the read side. Everything here is built
from small arrays rather than the fitted NetCDF, so the suite stays
network-free and does not depend on a regenerable artifact.
"""

from __future__ import annotations

import numpy as np
import pytest

from nfl_simulator.fg_model import FieldGoalModel

# Roughly the fitted values from docs/research/05b §9, at two posterior draws.
ALPHA = np.array([1.90, 1.88])
BETA = np.array([-0.115, -0.113])
GAMMA = np.array([0.130, 0.125])


def build_model(effects: dict[str, np.ndarray] | None = None, gamma=GAMMA) -> FieldGoalModel:
    return FieldGoalModel(
        alpha=ALPHA,
        beta=BETA,
        gamma=gamma,
        kicker_effects=effects or {},
        distance_centre=40.0,
    )


def test_make_probability_returns_one_value_per_posterior_draw():
    model = build_model()
    assert model.league_make_probability(45.0).shape == (2,)
    assert model.n_draws == 2


def test_make_probability_falls_with_distance():
    model = build_model()
    near = model.league_make_probability(30.0).mean()
    far = model.league_make_probability(55.0).mean()
    assert near > far


def test_league_curve_lands_near_the_fitted_values():
    """Sanity anchor against docs/research/05b §9: ~87% at 40 yd, ~62% at 55."""
    model = build_model()
    assert model.league_make_probability(40.0).mean() == pytest.approx(0.87, abs=0.02)
    assert model.league_make_probability(55.0).mean() == pytest.approx(0.62, abs=0.03)


def test_probabilities_stay_inside_zero_and_one_at_absurd_distances():
    model = build_model()
    for distance in (1.0, 200.0):
        p = model.make_probability("2024_KICKER", distance)
        assert np.all((p > 0.0) & (p < 1.0))


def test_a_better_kicker_gets_a_higher_probability_than_the_league():
    model = build_model({"2024_GOOD": np.array([0.5, 0.5])})
    assert (
        model.make_probability("2024_GOOD", 45.0).mean()
        > model.league_make_probability(45.0).mean()
    )


def test_a_worse_kicker_gets_a_lower_probability_than_the_league():
    model = build_model({"2024_BAD": np.array([-0.5, -0.5])})
    assert (
        model.make_probability("2024_BAD", 45.0).mean() < model.league_make_probability(45.0).mean()
    )


def test_an_unknown_kicker_falls_back_to_the_league_curve():
    """A kicker with no fitted effect must not silently get someone else's."""
    model = build_model({"2024_GOOD": np.array([0.5, 0.5])})
    np.testing.assert_allclose(
        model.make_probability("2024_NEVER_SEEN", 45.0),
        model.league_make_probability(45.0),
    )


def test_the_quadratic_term_changes_the_long_range_curve():
    """Gate FG-2 failed without it, so the simulator must actually apply it."""
    with_curve = build_model().league_make_probability(55.0).mean()
    without_curve = build_model(gamma=np.zeros(2)).league_make_probability(55.0).mean()
    assert with_curve != pytest.approx(without_curve, abs=1e-6)


def test_mismatched_parameter_lengths_are_rejected():
    with pytest.raises(ValueError, match="same number of posterior draws"):
        FieldGoalModel(
            alpha=np.array([1.9, 1.9]),
            beta=np.array([-0.115]),
            gamma=np.zeros(2),
            kicker_effects={},
        )


def test_a_kicker_effect_of_the_wrong_length_is_rejected():
    with pytest.raises(ValueError, match="2024_BAD"):
        build_model({"2024_BAD": np.array([0.1, 0.2, 0.3])})
