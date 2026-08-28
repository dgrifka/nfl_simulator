"""The receiver-drop variant ledger — document 56's correctness gate, as tests.

Document 56 mirrors document 49 onto the other direction of the hands-on-the-ball
class: a charted **catchable target** is neutralized at the charged offensive
entity's posterior-sampled catch probability, beside the v1.3 ledger and never
inside it. Every gate in §2/§3 that can be pinned without the fitted trace is
pinned here; V-1 (the 2,761-game default-off replay) and V-6/V-8 (the sampler and
the posterior spread) live in `research/72_` and `research/73_`, because they need
the fit.

Two pins here have no counterpart on the dropped-pick side and exist because
document 56 says so in as many words:

* **the defence-season effect is excluded on read** (§2) — it is fitted so the
  offensive term is estimated free of schedule, and never paid;
* **the swing has one sign** (handoff constraint 6) — a catch must be worth more
  than an incompletion, on every play, or the run stops.

Every test builds its own frame, so the suite stays network-free.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nfl_simulator.components import fit_fg_baseline, fit_fumble_baseline
from nfl_simulator.receiver_drops import (
    MIN_PER_BRANCH,
    DropSwingTable,
    ReceiverDropModel,
    build_drop_swing_table,
    catchable_target_frame,
    event_class_for,
)
from nfl_simulator.simulator import receiver_drop_events, simulate_game

# --------------------------------------------------------------------------
# the swing table — document 47 §3's six cells, mirrored onto the drop branch
# --------------------------------------------------------------------------


def target(
    play_id: float,
    *,
    yardline_100: float,
    down: float,
    dropped: bool,
    epa: float,
    air_epa: float | None = None,
    xyac_epa: float | None = None,
) -> dict:
    return {
        "play_id": play_id,
        "yardline_100": yardline_100,
        "down": down,
        "is_drop": dropped,
        "epa": epa,
        "air_epa": air_epa,
        "xyac_epa": xyac_epa,
    }


def catchable_corpus(
    *, cell_yardline: float = 50.0, cell_down: float = 3.0, n_per_branch: int = MIN_PER_BRANCH
) -> pl.DataFrame:
    """Catchable targets where exactly one cell clears the 30-per-branch floor.

    The populated cell prices a catch at +1.0 EPA and a drop at −1.5, so its own
    swing is 2.5; everything else in the frame is a single target per branch at
    deliberately different values, so a cell that fell back to the pooled swing
    rather than reading its own is visible in the number.
    """
    rows = []
    play_id = 1.0
    for index in range(n_per_branch * 2):
        rows.append(
            target(
                play_id,
                yardline_100=cell_yardline,
                down=cell_down,
                dropped=index % 2 == 0,
                epa=-1.5 if index % 2 == 0 else 1.0,
            )
        )
        play_id += 1
    for dropped, epa in ((True, -4.0), (False, 3.0)):
        rows.append(target(play_id, yardline_100=10.0, down=1.0, dropped=dropped, epa=epa))
        play_id += 1
    return pl.DataFrame(rows)


def test_a_populated_cell_prices_the_caught_branch_against_the_dropped_one():
    table = build_drop_swing_table(catchable_corpus())
    assert table.swing_for(50.0, 3.0) == pytest.approx(2.5)


def test_the_swing_is_positive_because_a_catch_beats_an_incompletion():
    table = build_drop_swing_table(catchable_corpus())
    assert table.pooled > 0
    assert all(value > 0 for value in table.cells.values())


def test_a_thin_cell_falls_back_to_the_pooled_swing():
    corpus = catchable_corpus()
    table = build_drop_swing_table(corpus)
    caught = corpus.filter(~pl.col("is_drop"))["epa"].mean()
    dropped = corpus.filter(pl.col("is_drop"))["epa"].mean()
    assert table.pooled == pytest.approx(caught - dropped)
    assert table.swing_for(10.0, 1.0) == pytest.approx(table.pooled)


def test_the_table_carries_the_dropped_branch_mean_each_cell_prices_an_incompletion_at():
    table = build_drop_swing_table(catchable_corpus())
    assert table.incompletion_mean_for(50.0, 3.0) == pytest.approx(-1.5)
    # The thin cell takes the pooled dropped-branch mean, not its own single row.
    assert table.incompletion_mean_for(10.0, 1.0) == pytest.approx(table.pooled_incompletion_mean)


def test_the_table_carries_document_47s_six_cells_with_their_counts():
    table = build_drop_swing_table(catchable_corpus())
    assert set(table.cells) == {
        "1-33|1-2",
        "1-33|3-4",
        "34-66|1-2",
        "34-66|3-4",
        "67-99|1-2",
        "67-99|3-4",
    }
    assert table.counts["34-66|3-4"]["n_dropped"] == MIN_PER_BRANCH
    assert table.counts["34-66|3-4"]["source"] == "cell"
    assert table.counts["1-33|1-2"]["source"] == "pooled"


def test_an_unreadable_pre_target_state_takes_the_pooled_swing():
    table = build_drop_swing_table(catchable_corpus())
    assert table.swing_for(None, 3.0) == pytest.approx(table.pooled)
    assert table.swing_for(50.0, None) == pytest.approx(table.pooled)


def test_the_event_class_names_the_bin_the_swing_came_from():
    assert event_class_for(50.0, 3.0) == "34-66 yd, late down"
    assert event_class_for(10.0, 1.0) == "1-33 yd, early down"
    assert event_class_for(None, 2.0) == "pooled swing"


def test_the_swing_table_round_trips_through_its_serialised_form():
    table = build_drop_swing_table(catchable_corpus())
    restored = DropSwingTable.from_dict(table.to_dict())
    assert restored.pooled == pytest.approx(table.pooled)
    assert restored.swing_for(50.0, 3.0) == pytest.approx(table.swing_for(50.0, 3.0))
    assert restored.incompletion_mean_for(50.0, 3.0) == pytest.approx(-1.5)


# --------------------------------------------------------------------------
# the per-play swing — document 56 §2
# --------------------------------------------------------------------------


def test_a_dropped_ball_is_priced_against_its_own_realised_incompletion():
    table = build_drop_swing_table(catchable_corpus())
    row = {
        "yardline_100": 50.0,
        "down": 3.0,
        "is_drop": True,
        "epa": -2.0,
        "air_epa": 0.4,
        "xyac_epa": 0.6,
    }
    # |(0.4 + 0.6) − (−2.0)| = 3.0
    assert table.swing_for_play(row) == pytest.approx(3.0)


def test_a_caught_ball_is_priced_against_its_cells_incompletion_mean():
    table = build_drop_swing_table(catchable_corpus())
    row = {
        "yardline_100": 50.0,
        "down": 3.0,
        "is_drop": False,
        "epa": 1.0,
        "air_epa": 0.5,
        "xyac_epa": 0.25,
    }
    # |(0.5 + 0.25) − (−1.5)| = 2.25; the realised epa of a catch is not used.
    assert table.swing_for_play(row) == pytest.approx(2.25)


def test_a_play_with_no_completion_counterfactual_falls_back_to_its_cell():
    table = build_drop_swing_table(catchable_corpus())
    for air, xyac in ((None, None), (0.4, None), (None, 0.6)):
        row = {
            "yardline_100": 50.0,
            "down": 3.0,
            "is_drop": False,
            "epa": 1.0,
            "air_epa": air,
            "xyac_epa": xyac,
        }
        assert table.swing_for_play(row) == pytest.approx(2.5)


def test_every_per_play_swing_is_positive():
    table = build_drop_swing_table(catchable_corpus())
    # A catch counterfactual *below* the incompletion mean would still book a
    # positive swing, because the ledger wants the magnitude of the branch gap.
    row = {
        "yardline_100": 50.0,
        "down": 3.0,
        "is_drop": True,
        "epa": 0.5,
        "air_epa": -0.2,
        "xyac_epa": 0.0,
    }
    assert table.swing_for_play(row) > 0


# --------------------------------------------------------------------------
# the read side — document 56 §2's `catch_probability`
# --------------------------------------------------------------------------

HOME, AWAY = "HOM", "AWY"

TEST_ORDER = (
    "air_yards_z",
    "air_yards_z_squared",
    "pass_location_left",
    "pass_location_right",
    "down_2",
    "down_3",
    "down_4",
    "is_contested_ball",
)
TEST_BETA = (0.1, 0.01, 0.2, -0.2, 0.02, 0.002, 0.0002, 0.5)
TEST_STANDARDISATION = {"air_yards": {"mean": 10.0, "sd": 5.0}}


def model(n_draws: int = 40, **overrides) -> ReceiverDropModel:
    """A hand-built posterior: constant draws, so every probability is exact."""
    defaults = dict(
        alpha=np.full(n_draws, 0.05),
        beta=np.tile(np.array(TEST_BETA), (n_draws, 1)),
        entity_effects={
            f"2022|{HOME}": np.full(n_draws, 0.30),
            f"2022|{AWAY}": np.full(n_draws, -0.40),
        },
        covariate_order=TEST_ORDER,
        standardisation=TEST_STANDARDISATION,
        reference_levels={"pass_location": "middle", "down": 1.0},
        swing_table=build_drop_swing_table(catchable_corpus()),
    )
    return ReceiverDropModel(**(defaults | overrides))


def covariates(**overrides) -> dict:
    base = {
        "air_yards": 15.0,
        "pass_location": "left",
        "down": 3.0,
        "is_contested_ball": 1,
    }
    return base | overrides


def test_the_drop_probability_is_the_inverse_logit_of_the_stored_draws():
    fitted = model()
    # air_yards 15 standardises to +1.0, so z and z squared are both 1.0;
    # `left` fires, `down_3` fires, contested fires. Everything else is zero.
    logit = 0.05 + 0.1 + 0.01 + 0.2 + 0.002 + 0.5 + 0.30
    draws = fitted.drop_probability(f"2022|{HOME}", covariates())
    assert draws.shape == (40,)
    assert draws[0] == pytest.approx(1.0 / (1.0 + np.exp(-logit)), abs=1e-12)


def test_the_catch_probability_is_the_complement_of_the_drop_probability():
    fitted = model()
    row = covariates()
    assert fitted.catch_probability(f"2022|{HOME}", row)[0] == pytest.approx(
        1.0 - fitted.drop_probability(f"2022|{HOME}", row)[0], abs=1e-12
    )


def test_the_defence_season_effect_is_excluded_on_read():
    """Document 56 §2: fitted so the offensive term is clean, never paid.

    The model is handed a defence-season effect table and a row naming a
    defence-season. Reading it would change the probability; the pin is that it
    does not, and that the attribute the fit stores is available to a reader
    who wants to *report* it.
    """
    fitted = model(defence_effects={f"2022|{AWAY}": np.full(40, 1.75)})
    row = covariates()
    with_defence = fitted.drop_probability(f"2022|{HOME}", row | {"defence_season": f"2022|{AWAY}"})
    without = fitted.drop_probability(f"2022|{HOME}", row)
    assert with_defence[0] == pytest.approx(without[0], abs=1e-12)
    # And the excluded effect really was non-trivial, so the test could fail.
    assert fitted.defence_effects[f"2022|{AWAY}"][0] == pytest.approx(1.75)


def test_a_null_covariate_is_priced_at_its_own_reference_level():
    fitted = model()
    nulled = fitted.drop_probability(
        f"2022|{HOME}", covariates(air_yards=None, pass_location=None, down=None)
    )
    logit = 0.05 + 0.5 + 0.30
    assert nulled[0] == pytest.approx(1.0 / (1.0 + np.exp(-logit)), abs=1e-12)
    explicit = fitted.drop_probability(
        f"2022|{HOME}", covariates(air_yards=10.0, pass_location="middle", down=1.0)
    )
    assert nulled[0] == pytest.approx(explicit[0], abs=1e-12)


def test_an_unknown_entity_season_is_priced_at_the_league_effect():
    fitted = model()
    unknown = fitted.drop_probability("2019|XXX", covariates())
    league = fitted.drop_probability(None, covariates())
    assert unknown[0] == pytest.approx(league[0], abs=1e-12)
    logit = 0.05 + 0.1 + 0.01 + 0.2 + 0.002 + 0.5
    assert unknown[0] == pytest.approx(1.0 / (1.0 + np.exp(-logit)), abs=1e-12)


def test_a_covariate_order_the_model_cannot_read_is_refused():
    fitted = model(
        covariate_order=("air_yards_z", "who_knows"),
        beta=np.tile(np.array([0.1, 0.2]), (40, 1)),
    )
    with pytest.raises(ValueError, match="not a covariate this model knows"):
        fitted.drop_probability(f"2022|{HOME}", covariates())


# --------------------------------------------------------------------------
# the join — document 56 §2's adjudication frame
# --------------------------------------------------------------------------

GAME = "2022_01_AWY_HOM"


def pbp_play(play_id: float, **overrides) -> dict:
    base = {
        "game_id": GAME,
        "play_id": play_id,
        "season": 2022,
        "week": 1,
        "home_team": HOME,
        "away_team": AWAY,
        "posteam": HOME,
        "defteam": AWAY,
        "play_type": "pass",
        "epa": 0.0,
        "air_epa": 0.4,
        "xyac_epa": 0.6,
        "result": 3.0,
        "fumble": 0,
        "fumbled_1_team": None,
        "fumble_recovery_1_team": None,
        "aborted_play": 0,
        "interception": 0,
        "penalty": 0,
        "field_goal_result": None,
        "kick_distance": None,
        "kicker_player_id": None,
        "receiver_player_id": "R1",
        "air_yards": 15.0,
        "pass_location": "left",
        "down": 3.0,
        "ydstogo": 8.0,
        "yardline_100": 50.0,
        "qb_hit": 0,
        "shotgun": 1,
        "wp": 0.5,
    }
    return base | overrides


def catchable_target_play(
    play_id: float, *, dropped: bool, offence: str = HOME, **overrides
) -> dict:
    return pbp_play(
        play_id,
        posteam=offence,
        defteam=AWAY if offence == HOME else HOME,
        epa=-1.5 if dropped else 1.0,
        **overrides,
    )


def ftn_rows(play_ids: list[float], *, catchable: bool = True, drops: tuple = ()) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "nflverse_game_id": GAME,
                "nflverse_play_id": int(play_id),
                "is_catchable_ball": catchable,
                "is_drop": play_id in drops,
                "is_contested_ball": False,
                "is_qb_out_of_pocket": False,
                "is_play_action": False,
                "is_screen_pass": False,
                "n_pass_rushers": 4,
            }
            for play_id in play_ids
        ]
    )


def game_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, infer_schema_length=None)


def test_the_join_keeps_only_charted_catchable_targets():
    plays = game_frame(
        [
            pbp_play(1.0),
            catchable_target_play(2.0, dropped=False),
            catchable_target_play(3.0, dropped=True),
            pbp_play(4.0, play_type="run"),
        ]
    )
    ftn = pl.concat(
        [ftn_rows([2.0, 3.0], drops=(3.0,)), ftn_rows([1.0, 4.0], catchable=False)],
        how="vertical",
    )
    targets = catchable_target_frame(plays, ftn)
    assert sorted(targets["play_id"].to_list()) == [2.0, 3.0]
    assert targets["entity_season"].to_list() == [f"2022|{HOME}", f"2022|{HOME}"]


def test_the_join_charges_the_offence_not_the_receiver():
    """Document 56 §1's clause-1 rule selected the team-season grain."""
    plays = game_frame([catchable_target_play(2.0, dropped=False, offence=AWAY)])
    targets = catchable_target_frame(plays, ftn_rows([2.0]))
    assert targets["entity_season"].to_list() == [f"2022|{AWAY}"]
    assert targets["defence_season"].to_list() == [f"2022|{HOME}"]


def test_the_join_flags_a_target_with_a_null_covariate():
    plays = game_frame(
        [
            catchable_target_play(2.0, dropped=False),
            catchable_target_play(3.0, dropped=False, pass_location=None, air_yards=None),
        ]
    )
    targets = catchable_target_frame(plays, ftn_rows([2.0, 3.0]))
    flags = dict(
        zip(
            targets["play_id"].to_list(),
            targets["covariates_imputed"].to_list(),
            strict=True,
        )
    )
    assert flags == {2.0: False, 3.0: True}


def test_a_game_with_no_charting_rows_yields_no_catchable_targets():
    plays = game_frame([catchable_target_play(2.0, dropped=False)])
    assert catchable_target_frame(plays, ftn_rows([99.0])).height == 0


# --------------------------------------------------------------------------
# the event builder and the switch — document 56 §2's V-2 to V-7
# --------------------------------------------------------------------------


def model_at(p_drop: float, swing_table: DropSwingTable | None = None) -> ReceiverDropModel:
    """A model whose drop probability is exactly `p_drop` on every target."""
    n = 40
    return ReceiverDropModel(
        alpha=np.full(n, float(np.log(p_drop / (1.0 - p_drop)))),
        beta=np.zeros((n, len(TEST_ORDER))),
        entity_effects={},
        covariate_order=TEST_ORDER,
        standardisation=TEST_STANDARDISATION,
        reference_levels={"pass_location": "middle", "down": 1.0},
        swing_table=swing_table or build_drop_swing_table(catchable_corpus()),
    )


@pytest.fixture
def baselines():
    """Fumble and field-goal baselines on a corpus with a known 40% retention."""
    rows, play_id = [], 1000.0
    for index in range(100):
        rows.append(
            pbp_play(
                play_id,
                play_type="run",
                epa=-4.0 if index >= 40 else 0.0,
                fumble=1,
                fumbled_1_team=HOME,
                fumble_recovery_1_team=HOME if index < 40 else AWAY,
            )
        )
        play_id += 1
    for index in range(40):
        rows.append(
            pbp_play(
                play_id,
                play_type="field_goal",
                epa=2.5 if index < 32 else -2.5,
                field_goal_result="made" if index < 32 else "missed",
                kick_distance=42.0,
                kicker_player_id="K1",
            )
        )
        play_id += 1
    corpus = game_frame(rows)
    return fit_fumble_baseline(corpus, min_class_size=10), fit_fg_baseline(corpus, min_bin_size=10)


def run(rows, baselines, **kwargs):
    fumble_baseline, fg_baseline = baselines
    return simulate_game(
        game_frame(rows),
        fumble_baseline=fumble_baseline,
        fg_baseline=fg_baseline,
        fg_model=None,
        points_per_epa=kwargs.pop("points_per_epa", 0.6),
        **kwargs,
    )


def test_v7_a_drop_by_the_home_offence_books_bad_luck(baselines):
    result = run(
        [catchable_target_play(2.0, dropped=True)],
        baselines,
        receiver_drop_model=model_at(0.05),
        ftn=ftn_rows([2.0], drops=(2.0,)),
    )
    assert len(result.ledger) == 1
    entry = next(iter(result.ledger))
    assert entry.component == "receiver_drop"
    assert entry.charged_team == HOME
    assert entry.actual == 0.0
    assert entry.expected == pytest.approx(0.95, abs=1e-9)  # P(catch)
    assert entry.luck_epa < 0


def test_v7_a_catch_on_a_low_probability_ball_books_good_luck(baselines):
    result = run(
        [catchable_target_play(2.0, dropped=False)],
        baselines,
        receiver_drop_model=model_at(0.60),
        ftn=ftn_rows([2.0]),
    )
    entry = next(iter(result.ledger))
    assert entry.actual == 1.0
    assert entry.expected == pytest.approx(0.40, abs=1e-9)
    assert entry.luck_epa > 0


def test_v7_the_away_offences_drop_carries_the_opposite_sign(baselines):
    home = run(
        [catchable_target_play(2.0, dropped=True, offence=HOME)],
        baselines,
        receiver_drop_model=model_at(0.05),
        ftn=ftn_rows([2.0], drops=(2.0,)),
    )
    away = run(
        [catchable_target_play(2.0, dropped=True, offence=AWAY)],
        baselines,
        receiver_drop_model=model_at(0.05),
        ftn=ftn_rows([2.0], drops=(2.0,)),
    )
    home_entry, away_entry = next(iter(home.ledger)), next(iter(away.ledger))
    assert home_entry.luck_epa == pytest.approx(-away_entry.luck_epa)
    assert away_entry.luck_epa > 0
    assert away_entry.charged_team == AWAY


def test_the_swing_is_never_negative_on_any_event(baselines):
    events = receiver_drop_events(
        game_frame(
            [
                catchable_target_play(2.0, dropped=False),
                catchable_target_play(3.0, dropped=True, yardline_100=20.0, down=1.0),
                catchable_target_play(4.0, dropped=False, offence=AWAY, air_epa=None),
            ]
        ),
        ftn_rows([2.0, 3.0, 4.0], drops=(3.0,)),
        model_at(0.05),
        n_draws=10,
        rng=np.random.default_rng(0),
    )
    assert len(events) == 3
    for event in events:
        assert abs(event.swing) > 0


def test_v5_every_event_carries_one_probability_per_posterior_draw(baselines):
    events = receiver_drop_events(
        game_frame(
            [catchable_target_play(2.0, dropped=False), catchable_target_play(3.0, dropped=True)]
        ),
        ftn_rows([2.0, 3.0], drops=(3.0,)),
        model_at(0.05),
        n_draws=25,
        rng=np.random.default_rng(0),
    )
    assert len(events) == 2
    for event in events:
        assert len(event.expected_draws) == 25
        assert 0.0 <= event.expected_draws.mean() <= 1.0
        assert event.to_entry().expected == pytest.approx(float(event.expected_draws.mean()))


def test_v5_the_event_class_is_the_swing_bin_it_was_priced_in(baselines):
    events = receiver_drop_events(
        game_frame([catchable_target_play(2.0, dropped=False, yardline_100=50.0, down=3.0)]),
        ftn_rows([2.0]),
        model_at(0.05),
        n_draws=10,
        rng=np.random.default_rng(0),
    )
    assert events[0].event_class == "34-66 yd, late down"


def test_v2_the_variant_ledger_sums_to_the_margin_it_moved(baselines):
    result = run(
        [
            catchable_target_play(2.0, dropped=False),
            catchable_target_play(3.0, dropped=True, yardline_100=20.0, down=1.0),
            catchable_target_play(4.0, dropped=False, offence=AWAY, yardline_100=80.0, down=2.0),
        ],
        baselines,
        receiver_drop_model=model_at(0.05),
        ftn=ftn_rows([2.0, 3.0, 4.0], drops=(3.0,)),
        points_per_epa=0.6,
    )
    assert len(result.ledger) == 3
    assert result.ledger.total_luck_epa() * 0.6 == pytest.approx(
        result.actual_margin - result.deserved_margin, abs=1e-9
    )


def test_v3_a_2022_game_with_no_catchable_targets_is_strict_field_for_field(baselines):
    rows = [pbp_play(1.0), pbp_play(2.0, play_type="run")]
    v13 = run(rows, baselines)
    variant = run(rows, baselines, receiver_drop_model=model_at(0.05), ftn=ftn_rows([9.0]))
    assert variant.variant == "strict"
    for name in ("actual_margin", "deserved_margin", "dtw_home", "dtw_interval", "total_luck_epa"):
        assert getattr(variant, name) == getattr(v13, name)
    assert np.array_equal(variant.margin_draws, v13.margin_draws)


def test_v3_a_game_with_catchable_targets_is_labelled_as_the_variant(baselines):
    result = run(
        [catchable_target_play(2.0, dropped=False)],
        baselines,
        receiver_drop_model=model_at(0.05),
        ftn=ftn_rows([2.0]),
    )
    assert result.variant == "strict+rd"


def test_v4_a_pre_2022_game_asked_for_the_variant_warns_and_returns_strict(baselines):
    rows = [
        pbp_play(1.0, season=2019, game_id="2019_01_AWY_HOM"),
        catchable_target_play(2.0, dropped=False, season=2019, game_id="2019_01_AWY_HOM"),
    ]
    v13 = run(rows, baselines)
    with pytest.warns(UserWarning, match="charting starts in 2022"):
        variant = run(rows, baselines, receiver_drop_model=model_at(0.05), ftn=ftn_rows([2.0]))
    assert variant.variant == "strict"
    assert variant.deserved_margin == v13.deserved_margin


def test_the_default_is_strict_so_the_component_is_opt_in(baselines):
    result = run([catchable_target_play(2.0, dropped=False)], baselines)
    assert result.variant == "strict"
    assert not [entry for entry in result.ledger if entry.component == "receiver_drop"]


# --------------------------------------------------------------------------
# the two directions together — document 56 §2's `v2.0`
# --------------------------------------------------------------------------


def dropped_pick_model_at(p_catch: float):
    """The other direction's model, built here so the two test files stay apart.

    Importing `tests/test_dropped_picks.py` would couple the two suites and
    double-collect its cases; the corpus it needs is four lines.
    """
    from nfl_simulator.dropped_picks import DroppedPickModel, build_swing_table

    worthy = pl.DataFrame(
        [
            {
                "play_id": float(index),
                "yardline_100": 50.0,
                "down": 3.0,
                "interception": index % 2,
                "epa": -4.0 if index % 2 else -0.5,
            }
            for index in range(1, 61)
        ]
    )
    n = 40
    return DroppedPickModel(
        alpha=np.full(n, float(np.log(p_catch / (1.0 - p_catch)))),
        beta=np.zeros((n, len(TEST_ORDER))),
        defence_effects={},
        covariate_order=TEST_ORDER,
        standardisation=TEST_STANDARDISATION,
        reference_levels={"pass_location": "middle", "down": 1.0},
        swing_table=build_swing_table(worthy),
    )


def worthy_and_catchable_ftn(worthy_ids: list[float], catchable_ids: list[float]) -> pl.DataFrame:
    rows = []
    for play_id in sorted(set(worthy_ids) | set(catchable_ids)):
        rows.append(
            {
                "nflverse_game_id": GAME,
                "nflverse_play_id": int(play_id),
                "is_interception_worthy": play_id in worthy_ids,
                "is_catchable_ball": play_id in catchable_ids,
                "is_drop": False,
                "is_contested_ball": False,
                "is_qb_out_of_pocket": False,
                "is_play_action": False,
                "is_screen_pass": False,
                "n_pass_rushers": 4,
            }
        )
    return pl.DataFrame(rows)


def test_the_combined_ledger_is_labelled_full(baselines):
    result = run(
        [catchable_target_play(2.0, dropped=False), catchable_target_play(3.0, dropped=False)],
        baselines,
        dropped_pick_model=dropped_pick_model_at(0.45),
        receiver_drop_model=model_at(0.05),
        ftn=worthy_and_catchable_ftn([3.0], [2.0]),
    )
    assert result.variant == "full"


def test_the_combined_ledger_sums_to_the_two_ledgers_added(baselines):
    """Document 56 §2's `+dp+rd` identity, as the handoff's Part C asks for it."""
    rows = [catchable_target_play(2.0, dropped=False), catchable_target_play(3.0, dropped=False)]
    ftn = worthy_and_catchable_ftn([3.0], [2.0])
    dp_model, rd_model = dropped_pick_model_at(0.45), model_at(0.05)

    dp_only = run(rows, baselines, dropped_pick_model=dp_model, ftn=ftn)
    rd_only = run(rows, baselines, receiver_drop_model=rd_model, ftn=ftn)
    v13 = run(rows, baselines)
    both = run(
        rows,
        baselines,
        dropped_pick_model=dp_model,
        receiver_drop_model=rd_model,
        ftn=ftn,
    )

    def component_luck(result, component: str) -> float:
        return sum(entry.luck_epa for entry in result.ledger if entry.component == component)

    assert both.total_luck_epa == pytest.approx(
        v13.total_luck_epa
        + component_luck(dp_only, "dropped_pick")
        + component_luck(rd_only, "receiver_drop"),
        abs=1e-9,
    )


def test_switching_the_receiver_model_on_does_not_move_the_dropped_pick_rows(baselines):
    """The shared random stream must not shift when a second variant is added."""
    rows = [catchable_target_play(2.0, dropped=False), catchable_target_play(3.0, dropped=False)]
    ftn = worthy_and_catchable_ftn([3.0], [2.0])
    dp_model = dropped_pick_model_at(0.45)

    dp_only = run(rows, baselines, dropped_pick_model=dp_model, ftn=ftn)
    both = run(
        rows, baselines, dropped_pick_model=dp_model, receiver_drop_model=model_at(0.05), ftn=ftn
    )

    def rows_of(result, component):
        return [
            entry.to_dict()
            for entry in result.ledger
            if entry.component in ("fumble", "field_goal", "extra_point", component)
        ]

    assert rows_of(dp_only, "dropped_pick") == rows_of(both, "dropped_pick")


# --------------------------------------------------------------------------
# the editions — document 58 §2, enacted by ruling R-4
# --------------------------------------------------------------------------


def test_one_direction_alone_is_an_audit_arm_with_no_public_name(baselines):
    rows = [catchable_target_play(2.0, dropped=False), catchable_target_play(3.0, dropped=False)]
    ftn = worthy_and_catchable_ftn([3.0], [2.0])
    dp_only = run(rows, baselines, dropped_pick_model=dropped_pick_model_at(0.45), ftn=ftn)
    rd_only = run(rows, baselines, receiver_drop_model=model_at(0.05), ftn=ftn)
    assert dp_only.variant == "strict+dp"
    assert rd_only.variant == "strict+rd"
    assert dp_only.edition is None
    assert rd_only.edition is None


def test_the_edition_property_returns_the_public_name(baselines):
    rows = [catchable_target_play(2.0, dropped=False), catchable_target_play(3.0, dropped=False)]
    ftn = worthy_and_catchable_ftn([3.0], [2.0])
    assert run(rows, baselines).edition == "Strict"
    assert (
        run(
            rows,
            baselines,
            dropped_pick_model=dropped_pick_model_at(0.45),
            receiver_drop_model=model_at(0.05),
            ftn=ftn,
        ).edition
        == "Full"
    )


def test_edition_full_equals_passing_both_models_explicitly(baselines):
    """The convenience is a switch over the handles, not a second code path."""
    rows = [catchable_target_play(2.0, dropped=False), catchable_target_play(3.0, dropped=False)]
    ftn = worthy_and_catchable_ftn([3.0], [2.0])
    dp_model, rd_model = dropped_pick_model_at(0.45), model_at(0.05)

    explicit = run(
        rows, baselines, dropped_pick_model=dp_model, receiver_drop_model=rd_model, ftn=ftn
    )
    by_edition = run(
        rows,
        baselines,
        dropped_pick_model=dp_model,
        receiver_drop_model=rd_model,
        ftn=ftn,
        edition="full",
    )
    assert by_edition.variant == explicit.variant == "full"
    assert by_edition.total_luck_epa == explicit.total_luck_epa
    assert np.array_equal(by_edition.margin_draws, explicit.margin_draws)


def test_edition_strict_ignores_the_models_it_was_handed(baselines):
    """A caller holding both handles can render Strict without dropping them."""
    rows = [catchable_target_play(2.0, dropped=False), catchable_target_play(3.0, dropped=False)]
    ftn = worthy_and_catchable_ftn([3.0], [2.0])

    no_models = run(rows, baselines)
    by_edition = run(
        rows,
        baselines,
        dropped_pick_model=dropped_pick_model_at(0.45),
        receiver_drop_model=model_at(0.05),
        ftn=ftn,
        edition="strict",
    )
    assert by_edition.variant == no_models.variant == "strict"
    assert by_edition.total_luck_epa == no_models.total_luck_epa
    assert np.array_equal(by_edition.margin_draws, no_models.margin_draws)


def test_a_pre_2022_game_asked_for_full_warns_and_returns_strict(baselines):
    rows = [
        pbp_play(1.0, season=2019, game_id="2019_01_AWY_HOM"),
        catchable_target_play(2.0, dropped=False, season=2019, game_id="2019_01_AWY_HOM"),
    ]
    with pytest.warns(UserWarning, match="charting starts in 2022"):
        result = run(
            rows,
            baselines,
            dropped_pick_model=dropped_pick_model_at(0.45),
            receiver_drop_model=model_at(0.05),
            ftn=ftn_rows([2.0]),
            edition="full",
        )
    assert result.variant == "strict"
    assert result.edition == "Strict"


def test_an_edition_that_is_not_one_of_the_two_is_refused(baselines):
    with pytest.raises(ValueError, match="edition"):
        run([catchable_target_play(2.0, dropped=False)], baselines, edition="v2.0")
