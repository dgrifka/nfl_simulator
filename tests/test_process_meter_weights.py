"""The committed weights artifact and the pin that refuses a wrong one."""

import json

import numpy as np
import pytest

from nfl_simulator.process_meter.features import FOLDN_ARM
from nfl_simulator.process_meter.posterior import COLUMNS
from nfl_simulator.process_meter.record import (
    N_DRAWS_RECORD,
    SEED_OF_RECORD,
    SIGMA_ONCE_RECORD,
)
from nfl_simulator.process_meter.weights import (
    WEIGHTS_PATH,
    WEIGHTS_PIN,
    check_weights,
    load_weights,
)


@pytest.fixture
def payload():
    return json.loads(WEIGHTS_PATH.read_text())


# --- the pin ----------------------------------------------------------------------------


def test_the_pin_is_the_record_constants_so_moving_one_breaks_the_other():
    assert WEIGHTS_PIN["columns"] == COLUMNS
    assert WEIGHTS_PIN["sigma_once"] == SIGMA_ONCE_RECORD
    assert WEIGHTS_PIN["n_draws"] == N_DRAWS_RECORD
    assert WEIGHTS_PIN["seed"] == SEED_OF_RECORD
    assert WEIGHTS_PIN["arm"] == FOLDN_ARM
    assert WEIGHTS_PIN["mean"] == [-16.829, 35.126, 4.178]
    assert WEIGHTS_PIN["margin_sd"] == 8.875


def test_the_committed_artifact_passes_its_own_pin(payload):
    assert check_weights(payload) is payload


def test_the_committed_artifact_names_a_public_document(payload):
    assert payload["source"] == "docs/research/75-process-meter-foundations.md"


def test_the_committed_artifact_records_the_scale_both_ways(payload):
    """The convention is the published residual SD; the other reading is recorded, not used."""
    assert round(payload["sigma_once_measured"], 4) == SIGMA_ONCE_RECORD
    assert round(payload["sigma_once_from_full_margin_sd"], 4) == 5.4535
    assert payload["sigma_once"] == SIGMA_ONCE_RECORD


def test_the_committed_artifact_was_fit_on_2016_to_2023(payload):
    assert payload["train_seasons"] == list(range(2016, 2024))
    assert payload["n_train_team_games"] == 4190


# --- what the pin refuses ---------------------------------------------------------------


def test_wrong_columns_are_refused(payload):
    with pytest.raises(ValueError, match="columns"):
        check_weights(payload | {"columns": ["intercept", "a", "b"]})


def test_a_weight_off_at_three_decimals_is_refused(payload):
    moved = list(payload["mean"])
    moved[1] += 0.01
    with pytest.raises(ValueError, match="own_success_rate_for"):
        check_weights(payload | {"mean": moved})


def test_a_weight_inside_three_decimals_is_accepted(payload):
    nudged = list(payload["mean"])
    nudged[1] += 1e-5
    assert check_weights(payload | {"mean": nudged})


def test_a_wrong_residual_scale_is_refused(payload):
    with pytest.raises(ValueError, match="margin_sd"):
        check_weights(payload | {"margin_sd": 9.199})


@pytest.mark.parametrize(
    ("field", "value"),
    [("sigma_once", 5.8741), ("n_draws", 4_000), ("seed", 1), ("arm", "fold_take_sr")],
)
def test_a_wrong_constant_is_refused(payload, field, value):
    with pytest.raises(ValueError, match=field):
        check_weights(payload | {field: value})


# --- the load ---------------------------------------------------------------------------


def test_load_weights_returns_the_posterior_the_draws_read():
    weights = load_weights()
    assert weights.columns == COLUMNS
    assert weights.mean.shape == (3,)
    assert weights.cov.shape == (3, 3)
    assert np.allclose(weights.cov, weights.cov.T, atol=0)
    assert np.all(np.linalg.eigvalsh(weights.cov) > 0)
    assert weights.n_rows == 4190


def test_load_weights_reads_a_named_file_and_still_checks_the_pin(tmp_path, payload):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(payload))
    assert np.array_equal(load_weights(good).mean, load_weights().mean)

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload | {"arm": "fold_take_sr"}))
    with pytest.raises(ValueError, match="arm"):
        load_weights(bad)
