"""The loaders, the relocation remap and the takeaway credit."""

import numpy as np
import pandas as pd
import pytest

from nfl_simulator.process_meter.plays import (
    RELOCATIONS,
    TAKEAWAY_ARMS,
    TRAIN_SEASONS,
    attach_return_columns,
    attach_touchdown_columns,
    credited_return_yards,
    takeaway_mask,
    touchdown_return_mask,
)

G1 = "2016_01_AWY_HOM"


def _plays(**columns):
    n = len(next(iter(columns.values())))
    base = {
        "game_id": [G1] * n,
        "epa": [0.1] * n,
        "posteam": ["AWY"] * n,
        "defteam": ["HOM"] * n,
        "interception": [0] * n,
        "fumble_lost": [0] * n,
        "return_yards": [0.0] * n,
        "fumble_recovery_1_team": [None] * n,
        "fumble_recovery_1_yards": [np.nan] * n,
        "fumble_recovery_2_team": [None] * n,
        "fumble_recovery_2_yards": [np.nan] * n,
        "touchdown": [0.0] * n,
        "td_team": [None] * n,
    }
    return pd.DataFrame(base | columns)


# --- the constants ----------------------------------------------------------------------


def test_the_training_window_is_2016_to_2023():
    assert tuple(range(2016, 2024)) == TRAIN_SEASONS


def test_the_relocations_map_the_old_code_onto_the_modern_franchise():
    assert RELOCATIONS == {"SD": "LAC", "OAK": "LV", "STL": "LA"}


# --- the takeaway mask ------------------------------------------------------------------


def test_int_counts_picks_only_and_take_adds_lost_fumbles():
    plays = _plays(interception=[1, 0, 0], fumble_lost=[0, 1, 0])
    assert list(takeaway_mask(plays, "int")) == [True, False, False]
    assert list(takeaway_mask(plays, "take")) == [True, True, False]


def test_an_unknown_takeaway_arm_raises():
    with pytest.raises(ValueError, match="arm"):
        takeaway_mask(_plays(interception=[0]), "fumbles")
    assert TAKEAWAY_ARMS == ("int", "take")


# --- the credited yards -----------------------------------------------------------------


def test_a_pick_is_credited_its_return_yards():
    plays = _plays(interception=[1], return_yards=[30.0])
    assert list(credited_return_yards(plays)) == [30.0]


def test_a_lost_fumble_reads_the_recovering_side_first_then_second():
    first = _plays(fumble_lost=[1], fumble_recovery_1_team=["HOM"], fumble_recovery_1_yards=[12.0])
    second = _plays(
        fumble_lost=[1],
        fumble_recovery_1_team=["AWY"],
        fumble_recovery_1_yards=[3.0],
        fumble_recovery_2_team=["HOM"],
        fumble_recovery_2_yards=[7.0],
    )
    assert list(credited_return_yards(first)) == [12.0]
    assert list(credited_return_yards(second)) == [7.0]


def test_a_fumble_neither_side_recovered_is_credited_nothing():
    plays = _plays(fumble_lost=[1])
    assert list(credited_return_yards(plays)) == [0.0]


def test_a_play_that_is_not_a_takeaway_is_credited_nothing():
    assert list(credited_return_yards(_plays(interception=[0]))) == [0.0]


def test_a_pick_the_interceptor_fumbles_back_keeps_the_picks_return():
    plays = _plays(
        interception=[1],
        fumble_lost=[1],
        return_yards=[25.0],
        fumble_recovery_1_team=["HOM"],
        fumble_recovery_1_yards=[4.0],
    )
    assert list(credited_return_yards(plays)) == [25.0]


def test_a_negative_return_stays_negative():
    plays = _plays(interception=[1], return_yards=[-6.0])
    assert list(credited_return_yards(plays)) == [-6.0]


# --- the touchdown return ---------------------------------------------------------------


def test_only_a_takeaway_the_defense_scored_on_is_a_touchdown_return():
    plays = _plays(
        interception=[1, 1, 0],
        touchdown=[1.0, 0.0, 1.0],
        td_team=["HOM", None, "AWY"],
    )
    assert list(touchdown_return_mask(plays)) == [True, False, False]


# --- the row-for-row attachments --------------------------------------------------------


def test_attach_return_columns_raises_when_the_rows_do_not_line_up():
    plays = _plays(epa=[0.1, 0.2])
    with pytest.raises(ValueError, match="do not line up"):
        attach_return_columns(plays, _plays(epa=[0.1]))


def test_attach_return_columns_relocates_the_team_columns():
    plays = _plays(epa=[0.1])
    extras = _plays(epa=[0.1], fumble_recovery_1_team=["OAK"], return_team=["STL"])
    out = attach_return_columns(plays, extras)
    assert out.fumble_recovery_1_team.iloc[0] == "LV"
    assert out.return_team.iloc[0] == "LA"


def test_attach_touchdown_columns_raises_when_the_rows_do_not_line_up():
    with pytest.raises(ValueError, match="do not line up"):
        attach_touchdown_columns(_plays(epa=[0.1, 0.2]), _plays(epa=[0.1]))


def test_attach_touchdown_columns_relocates_the_scoring_team():
    out = attach_touchdown_columns(_plays(epa=[0.1]), _plays(epa=[0.1], td_team=["SD"]))
    assert out.td_team.iloc[0] == "LAC"
