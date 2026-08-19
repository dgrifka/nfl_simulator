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


# --------------------------------------------------------------------------
# weather — docs/research/05b §10
# --------------------------------------------------------------------------


from nfl_simulator.fg_model import Weather, sanitize_weather  # noqa: E402

WEATHER_KWARGS = {
    "roof_effects": {"dome": np.array([0.30, 0.28]), "closed": np.array([0.25, 0.23])},
    "beta_wind": np.array([-0.015, -0.014]),
    "beta_temp": np.array([0.001, 0.001]),
    "wind_centre": 8.12,
    "temp_centre": 58.16,
}


def build_weather_model(**overrides) -> FieldGoalModel:
    kwargs = {**WEATHER_KWARGS, **overrides}
    return FieldGoalModel(
        alpha=ALPHA, beta=BETA, gamma=GAMMA, kicker_effects={}, distance_centre=40.0, **kwargs
    )


# ---- the sanitize rules --------------------------------------------------


def test_sanitize_drops_weather_readings_taken_under_a_closed_roof():
    """A stadium-ambient reading that leaked indoors is worse than a missing one."""
    assert sanitize_weather("closed", wind=2.0, temp=46.0) == Weather(
        roof="closed", wind=None, temp=None
    )


def test_sanitize_drops_weather_readings_taken_in_a_dome():
    assert sanitize_weather("dome", wind=2.0, temp=46.0) == Weather(
        roof="dome", wind=None, temp=None
    )


def test_sanitize_caps_an_implausible_wind_reading():
    """The raw column's maximum is 71 mph, on three kicks in one 2016 game."""
    assert sanitize_weather("outdoors", wind=71.0, temp=43.0).wind == 30.0


def test_sanitize_leaves_a_plausible_outdoor_reading_alone():
    assert sanitize_weather("outdoors", wind=12.0, temp=41.0) == Weather(
        roof="outdoors", wind=12.0, temp=41.0
    )


def test_sanitize_keeps_missing_outdoor_readings_missing():
    """512 outdoor attempts have no reading. Absent is not the same as calm."""
    assert sanitize_weather("outdoors", wind=None, temp=None).wind is None


def test_sanitize_treats_an_open_retractable_roof_as_having_no_usable_reading():
    """All 204 'open' attempts carry null temp and wind, so there is nothing to use."""
    assert sanitize_weather("open", wind=None, temp=None).has_weather is False


def test_weather_with_both_readings_is_usable():
    assert Weather("outdoors", wind=12.0, temp=41.0).has_weather is True


def test_weather_with_only_one_reading_is_not_usable():
    """The model centres both terms together, so a half-measurement is no measurement."""
    assert Weather("outdoors", wind=12.0, temp=None).has_weather is False


# ---- applying weather to a make probability ------------------------------


def test_a_kick_indoors_is_easier_than_the_same_kick_outdoors():
    model = build_weather_model()
    indoors = model.make_probability(None, 45.0, weather=Weather("dome", None, None)).mean()
    outdoors = model.make_probability(
        None, 45.0, weather=Weather("outdoors", wind=8.12, temp=58.16)
    ).mean()
    assert indoors > outdoors


def test_a_windy_kick_is_harder_than_a_calm_one_at_the_same_distance():
    model = build_weather_model()
    calm = model.make_probability(None, 45.0, weather=Weather("outdoors", 0.0, 58.16)).mean()
    windy = model.make_probability(None, 45.0, weather=Weather("outdoors", 20.0, 58.16)).mean()
    assert calm > windy


def test_wind_is_measured_against_its_league_centre_not_against_zero():
    """A kick at the outdoor mean wind must price exactly as the bare curve does."""
    model = build_weather_model()
    at_centre = model.make_probability(
        None, 45.0, weather=Weather("outdoors", wind=8.12, temp=58.16)
    )
    np.testing.assert_allclose(at_centre, model.league_make_probability(45.0), rtol=1e-12)


def test_an_outdoor_kick_with_no_reading_prices_as_the_outdoor_baseline():
    """No information about conditions must mean no adjustment, not a zero-wind day."""
    model = build_weather_model()
    np.testing.assert_allclose(
        model.make_probability(None, 45.0, weather=Weather("outdoors", None, None)),
        model.league_make_probability(45.0),
    )


def test_omitting_weather_entirely_reproduces_the_pre_weather_model():
    """The Phase 2 ledger must be reproducible, so no-weather has to stay a valid call."""
    model = build_weather_model()
    np.testing.assert_allclose(
        model.make_probability(None, 45.0), model.league_make_probability(45.0)
    )


def test_weather_and_kicker_effects_compose():
    model = FieldGoalModel(
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA,
        kicker_effects={"2024_GOOD": np.array([0.5, 0.5])},
        distance_centre=40.0,
        **WEATHER_KWARGS,
    )
    good_in_a_dome = model.make_probability("2024_GOOD", 45.0, weather=Weather("dome", None, None))
    good_outdoors = model.make_probability(
        "2024_GOOD", 45.0, weather=Weather("outdoors", 20.0, 30.0)
    )
    assert good_in_a_dome.mean() > good_outdoors.mean()


def test_an_unknown_roof_level_falls_back_to_the_outdoor_baseline():
    """A new stadium type must not borrow a dome's effect by accident."""
    model = build_weather_model()
    np.testing.assert_allclose(
        model.make_probability(None, 45.0, weather=Weather("retractable_v2", None, None)),
        model.league_make_probability(45.0),
    )


def test_a_model_without_weather_parameters_ignores_weather_it_is_given():
    """Loading a Phase 2 posterior and passing weather must not silently do nothing wrong."""
    model = build_model()
    np.testing.assert_allclose(
        model.make_probability(None, 45.0, weather=Weather("outdoors", 20.0, 30.0)),
        model.league_make_probability(45.0),
    )


def test_a_roof_effect_of_the_wrong_length_is_rejected():
    with pytest.raises(ValueError, match="dome"):
        build_weather_model(roof_effects={"dome": np.array([0.3, 0.3, 0.3])})


# ---- loading a fitted posterior ------------------------------------------


def _write_posterior(path, *, with_weather: bool, with_cubic: bool = False):
    """A tiny fitted posterior on disk, in the shape `research/14_*` writes.

    Built here rather than read from `research/outputs/`, so the suite never
    depends on a regenerable artifact — but written through arviz and read back
    through the real loader, so the round trip being tested is the real one.
    """
    import xarray as xr

    draws = {
        "alpha": (("chain", "draw"), np.array([[1.90, 1.88]])),
        "beta": (("chain", "draw"), np.array([[-0.115, -0.113]])),
        "gamma": (("chain", "draw"), np.array([[0.130, 0.125]])),
        "kicker": (
            ("chain", "draw", "kicker_season"),
            np.array([[[0.5, -0.5], [0.4, -0.4]]]),
        ),
    }
    coords = {"chain": [0], "draw": [0, 1], "kicker_season": ["2024_GOOD", "2024_BAD"]}
    if with_weather:
        draws["roof"] = (("chain", "draw", "roof_level"), np.array([[[0.30, 0.25], [0.28, 0.23]]]))
        draws["beta_wind"] = (("chain", "draw"), np.array([[-0.015, -0.014]]))
        draws["beta_temp"] = (("chain", "draw"), np.array([[0.001, 0.001]]))
        coords["roof_level"] = ["dome", "closed"]
    if with_cubic:
        # The adopted Phase 3 curve, plus the extra-point arm fitted with it.
        draws["delta_cubic"] = (("chain", "draw"), np.array([[-0.081, -0.079]]))
        draws["delta_xp"] = (("chain", "draw"), np.array([[0.167, 0.160]]))
        draws["lambda_xp"] = (("chain", "draw"), np.array([[1.263, 1.250]]))

    # ArviZ 1.x stores a fit as an xarray DataTree with one group per role, which
    # is what `az.from_netcdf` hands back and what the loader indexes into.
    xr.DataTree.from_dict({"posterior": xr.Dataset(draws, coords=coords)}).to_netcdf(path)
    return path


def test_loading_a_posterior_recovers_the_weather_parameters(tmp_path):
    path = _write_posterior(tmp_path / "trace.nc", with_weather=True)
    model = FieldGoalModel.from_posterior(path, wind_centre=8.12, temp_centre=58.16)

    assert set(model.roof_effects) == {"dome", "closed"}
    np.testing.assert_allclose(model.beta_wind, [-0.015, -0.014])
    assert model.wind_centre == 8.12


def test_a_loaded_weather_posterior_actually_prices_a_dome_higher(tmp_path):
    """Loading the numbers is not enough — they have to reach the probability."""
    path = _write_posterior(tmp_path / "trace.nc", with_weather=True)
    model = FieldGoalModel.from_posterior(path, wind_centre=8.12, temp_centre=58.16)
    in_dome = model.make_probability("2024_GOOD", 45.0, weather=Weather("dome", None, None))
    outdoors = model.make_probability("2024_GOOD", 45.0, weather=Weather("outdoors", 8.12, 58.16))
    assert in_dome.mean() > outdoors.mean()


def test_loading_a_phase_two_posterior_without_weather_still_works(tmp_path):
    """The pre-weather trace must keep loading, so Phase 2 stays reproducible."""
    path = _write_posterior(tmp_path / "trace.nc", with_weather=False)
    model = FieldGoalModel.from_posterior(path)
    assert model.roof_effects == {}
    assert model.beta_wind is None
    assert model.make_probability("2024_GOOD", 45.0).shape == (2,)


# --------------------------------------------------------------------------
# the cubic distance term and the extra-point arm — docs/research/30 §2
#
# Both were fitted in Phase 3 and neither reached the simulator, so every kick
# beyond ~50 yards was priced too generously and every extra point too harshly
# (document 27 §14f). These tests pin the read side to the fitted arithmetic.
# --------------------------------------------------------------------------

DELTA_CUBIC = np.array([-0.081, -0.079])
DELTA_XP = np.array([0.167, 0.160])
LAMBDA_XP = np.array([1.263, 1.250])


def build_full_model(**overrides) -> FieldGoalModel:
    """A posterior carrying every fitted parameter, weather included."""
    kwargs = {
        "alpha": ALPHA,
        "beta": BETA,
        "gamma": GAMMA,
        "delta_cubic": DELTA_CUBIC,
        "delta_xp": DELTA_XP,
        "lambda_xp": LAMBDA_XP,
        "kicker_effects": {"2024_GOOD": np.array([0.5, 0.5])},
        "distance_centre": 40.0,
        **WEATHER_KWARGS,
    }
    kwargs.update(overrides)
    return FieldGoalModel(**kwargs)


def fitted_logit(
    distance: float,
    *,
    kicker: np.ndarray | float = 0.0,
    roof: np.ndarray | float = 0.0,
    wind: float | None = None,
    temp: float | None = None,
    extra_point: bool = False,
) -> np.ndarray:
    """The linear predictor exactly as `research/14_fg_weather_model.py` builds it.

    Written out here rather than imported, so the test is an independent
    statement of what the fit means and not a shared helper that could drift
    with the code under test.
    """
    centred = distance - 40.0
    is_xp = 1.0 if extra_point else 0.0
    eta = (
        ALPHA
        + BETA * centred
        + GAMMA * centred**2 / 100.0
        + DELTA_CUBIC * centred**3 / 1000.0
        + roof
        + DELTA_XP * is_xp
        + kicker * (1.0 + (LAMBDA_XP - 1.0) * is_xp)
    )
    if wind is not None and temp is not None:
        eta = eta + WEATHER_KWARGS["beta_wind"] * (wind - WEATHER_KWARGS["wind_centre"])
        eta = eta + WEATHER_KWARGS["beta_temp"] * (temp - WEATHER_KWARGS["temp_centre"])
    return eta


def test_the_cubic_term_bends_the_long_range_curve():
    """It is negative, so a 55-yarder is harder than the quadratic alone says."""
    with_cubic = build_full_model().league_make_probability(55.0).mean()
    without = build_full_model(delta_cubic=None).league_make_probability(55.0).mean()
    assert with_cubic < without - 0.01


def test_the_read_side_reproduces_the_fitted_field_goal_probability():
    """Gate S-1 in miniature: the product must price what the model fitted."""
    model = build_full_model()
    expected = 1.0 / (
        1.0
        + np.exp(
            -fitted_logit(
                52.0,
                kicker=np.array([0.5, 0.5]),
                roof=WEATHER_KWARGS["roof_effects"]["dome"],
            )
        )
    )
    np.testing.assert_allclose(
        model.make_probability("2024_GOOD", 52.0, weather=Weather("dome", None, None)),
        expected,
        atol=1e-12,
    )


def test_the_read_side_reproduces_the_fitted_extra_point_probability():
    model = build_full_model()
    expected = 1.0 / (
        1.0
        + np.exp(
            -fitted_logit(
                33.0,
                kicker=np.array([0.5, 0.5]),
                wind=12.0,
                temp=41.0,
                extra_point=True,
            )
        )
    )
    np.testing.assert_allclose(
        model.make_probability(
            "2024_GOOD", 33.0, weather=Weather("outdoors", 12.0, 41.0), extra_point=True
        ),
        expected,
        atol=1e-12,
    )


def test_an_extra_point_is_easier_than_a_field_goal_from_the_same_distance():
    """`delta_xp` is what the fit means by that, and it must reach the product."""
    model = build_full_model()
    as_xp = model.make_probability(None, 33.0, extra_point=True).mean()
    as_fg = model.make_probability(None, 33.0).mean()
    assert as_xp > as_fg


def test_a_kickers_ability_transfers_to_extra_points_at_lambda_xp():
    """A half-transfer must halve the kicker's log-odds edge, not keep it whole."""
    model = build_full_model(lambda_xp=np.full(2, 0.5))
    edge_on_a_field_goal = _logit_of(model.make_probability("2024_GOOD", 33.0)) - _logit_of(
        model.make_probability(None, 33.0)
    )
    edge_on_an_extra_point = _logit_of(
        model.make_probability("2024_GOOD", 33.0, extra_point=True)
    ) - _logit_of(model.make_probability(None, 33.0, extra_point=True))
    np.testing.assert_allclose(edge_on_an_extra_point, edge_on_a_field_goal * 0.5, rtol=1e-9)


def _logit_of(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


def test_a_posterior_without_extra_point_terms_prices_an_extra_point_as_a_field_goal():
    """Gate S-2: a Phase 2 trace has no `delta_xp`, and must keep pricing as it did."""
    model = build_model()
    np.testing.assert_allclose(
        model.make_probability("2024_K", 33.0, extra_point=True),
        model.make_probability("2024_K", 33.0),
    )


def test_a_posterior_without_a_cubic_term_prices_exactly_as_the_quadratic_curve():
    """Gate S-2: the linear and quadratic arms must be unaffected by this change."""
    model = build_model()
    centred = 55.0 - 40.0
    expected = 1.0 / (1.0 + np.exp(-(ALPHA + BETA * centred + GAMMA * centred**2 / 100.0)))
    np.testing.assert_allclose(model.league_make_probability(55.0), expected, atol=1e-12)


def test_the_league_curve_can_be_asked_for_an_extra_point():
    model = build_full_model()
    np.testing.assert_allclose(
        model.league_make_probability(33.0, extra_point=True),
        model.make_probability(None, 33.0, extra_point=True),
    )


def test_a_cubic_term_of_the_wrong_length_is_rejected():
    with pytest.raises(ValueError, match="same number of posterior draws"):
        build_full_model(delta_cubic=np.array([-0.081, -0.079, -0.077]))


def test_loading_a_posterior_recovers_the_cubic_and_extra_point_terms(tmp_path):
    path = _write_posterior(tmp_path / "trace.nc", with_weather=True, with_cubic=True)
    model = FieldGoalModel.from_posterior(path, wind_centre=8.12, temp_centre=58.16)
    np.testing.assert_allclose(model.delta_cubic, [-0.081, -0.079])
    np.testing.assert_allclose(model.delta_xp, [0.167, 0.160])
    np.testing.assert_allclose(model.lambda_xp, [1.263, 1.250])


def test_a_loaded_posterior_actually_applies_its_cubic_term(tmp_path):
    """Loading the numbers is not enough — they have to reach the probability."""
    path = _write_posterior(tmp_path / "trace.nc", with_weather=True, with_cubic=True)
    model = FieldGoalModel.from_posterior(path, wind_centre=8.12, temp_centre=58.16)
    quadratic_only = FieldGoalModel(
        alpha=model.alpha, beta=model.beta, gamma=model.gamma, distance_centre=40.0
    )
    assert (
        model.league_make_probability(60.0).mean()
        < quadratic_only.league_make_probability(60.0).mean() - 0.01
    )


def test_a_loaded_posterior_actually_applies_its_extra_point_terms(tmp_path):
    path = _write_posterior(tmp_path / "trace.nc", with_weather=True, with_cubic=True)
    model = FieldGoalModel.from_posterior(path, wind_centre=8.12, temp_centre=58.16)
    assert (
        model.make_probability("2024_GOOD", 33.0, extra_point=True).mean()
        > model.make_probability("2024_GOOD", 33.0).mean()
    )


def test_a_phase_two_posterior_still_loads_without_the_new_parameters(tmp_path):
    path = _write_posterior(tmp_path / "trace.nc", with_weather=False)
    model = FieldGoalModel.from_posterior(path)
    assert model.delta_cubic is None
    assert model.delta_xp is None
    assert model.lambda_xp is None
