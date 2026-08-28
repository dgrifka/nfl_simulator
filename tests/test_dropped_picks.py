"""The dropped-pick variant ledger — document 49's correctness gate, as tests.

Document 49 neutralizes a charted interception-worthy throw at the defence's
posterior-sampled catch probability, **beside** the v1.3 ledger and never
instead of it. Every gate in §6 that can be pinned without the fitted trace is
pinned here; V-1 (the 2,761-game default-off replay) and V-6/V-8 (the sampler
and the posterior spread) live in `research/67_` and `research/68_`, because they
need the fit.

Every test builds its own frame, so the suite stays network-free.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nfl_simulator.components import fit_fg_baseline, fit_fumble_baseline
from nfl_simulator.dropped_picks import (
    MIN_PER_BRANCH,
    DroppedPickModel,
    SwingTable,
    build_swing_table,
    event_class_for,
    worthy_throw_frame,
)
from nfl_simulator.simulator import dropped_pick_events, simulate_game

# --------------------------------------------------------------------------
# the swing table — document 47 §3's six cells, recomputed in `src/`
# --------------------------------------------------------------------------


def throw(
    play_id: float,
    *,
    yardline_100: float,
    down: float,
    intercepted: bool,
    epa: float,
) -> dict:
    return {
        "play_id": play_id,
        "yardline_100": yardline_100,
        "down": down,
        "interception": 1 if intercepted else 0,
        "epa": epa,
    }


def worthy_corpus(
    *, cell_yardline: float = 50.0, cell_down: float = 3.0, n_per_branch: int = MIN_PER_BRANCH
) -> pl.DataFrame:
    """Worthy throws where exactly one cell clears the 30-per-branch floor.

    The populated cell prices a pick at −4.0 EPA and an escape at −0.5, so its
    own swing is −3.5; everything else in the frame is a single throw per branch
    at deliberately different values, so a cell that fell back to the pooled
    swing rather than reading its own is visible in the number.
    """
    rows = []
    play_id = 1.0
    for i in range(n_per_branch * 2):
        rows.append(
            throw(
                play_id,
                yardline_100=cell_yardline,
                down=cell_down,
                intercepted=i % 2 == 0,
                epa=-4.0 if i % 2 == 0 else -0.5,
            )
        )
        play_id += 1
    # One throw per branch in a different cell: below the floor, so pooled.
    for intercepted, epa in ((True, -8.0), (False, +2.0)):
        rows.append(throw(play_id, yardline_100=10.0, down=1.0, intercepted=intercepted, epa=epa))
        play_id += 1
    return pl.DataFrame(rows)


def test_a_populated_cell_prices_the_picked_branch_against_the_escaped_one():
    table = build_swing_table(worthy_corpus())
    assert table.swing_for(50.0, 3.0) == pytest.approx(-3.5)


def test_a_thin_cell_falls_back_to_the_pooled_swing():
    corpus = worthy_corpus()
    table = build_swing_table(corpus)
    picked = corpus.filter(pl.col("interception") == 1)["epa"].mean()
    escaped = corpus.filter(pl.col("interception") == 0)["epa"].mean()
    assert table.pooled == pytest.approx(picked - escaped)
    assert table.swing_for(10.0, 1.0) == pytest.approx(table.pooled)


def test_the_table_carries_document_47s_six_cells_with_their_counts():
    table = build_swing_table(worthy_corpus())
    assert set(table.cells) == {
        "1-33|1-2",
        "1-33|3-4",
        "34-66|1-2",
        "34-66|3-4",
        "67-99|1-2",
        "67-99|3-4",
    }
    assert table.counts["34-66|3-4"]["n_picked"] == MIN_PER_BRANCH
    assert table.counts["34-66|3-4"]["source"] == "cell"
    assert table.counts["1-33|1-2"]["source"] == "pooled"


def test_an_unreadable_pre_throw_state_takes_the_pooled_swing():
    table = build_swing_table(worthy_corpus())
    assert table.swing_for(None, 3.0) == pytest.approx(table.pooled)
    assert table.swing_for(50.0, None) == pytest.approx(table.pooled)


def test_the_event_class_names_the_bin_the_swing_came_from():
    assert event_class_for(50.0, 3.0) == "34-66 yd, late down"
    assert event_class_for(10.0, 1.0) == "1-33 yd, early down"
    assert event_class_for(80.0, 2.0) == "67-99 yd, early down"
    assert event_class_for(None, 2.0) == "pooled swing"


def test_the_swing_table_round_trips_through_its_serialised_form():
    table = build_swing_table(worthy_corpus())
    restored = SwingTable.from_dict(table.to_dict())
    assert restored.pooled == pytest.approx(table.pooled)
    assert restored.swing_for(50.0, 3.0) == pytest.approx(table.swing_for(50.0, 3.0))


# --------------------------------------------------------------------------
# the read side — document 49 §5's `catch_probability`
# --------------------------------------------------------------------------

HOME, AWAY = "HOM", "AWY"

# A small covariate order with one of each kind the design matrix carries: a
# standardised covariate, its square, a dummied factor and a plain indicator.
# Every coefficient is a different power of ten, so a term applied in the wrong
# slot is visible in the number rather than hidden by a coincidence.
TEST_ORDER = (
    "air_yards_z",
    "air_yards_z_squared",
    "pass_location_left",
    "pass_location_right",
    "down_2",
    "down_3",
    "down_4",
    "qb_hit",
)
TEST_BETA = (0.1, 0.01, 0.2, -0.2, 0.02, 0.002, 0.0002, 0.5)
TEST_STANDARDISATION = {"air_yards": {"mean": 10.0, "sd": 5.0}}


def model(n_draws: int = 40, **overrides) -> DroppedPickModel:
    """A hand-built posterior: constant draws, so every probability is exact."""
    defaults = dict(
        alpha=np.full(n_draws, 0.05),
        beta=np.tile(np.array(TEST_BETA), (n_draws, 1)),
        defence_effects={
            f"2022|{AWAY}": np.full(n_draws, 0.30),
            f"2022|{HOME}": np.full(n_draws, -0.40),
        },
        covariate_order=TEST_ORDER,
        standardisation=TEST_STANDARDISATION,
        reference_levels={"pass_location": "middle", "down": 1.0},
        swing_table=build_swing_table(worthy_corpus()),
    )
    return DroppedPickModel(**(defaults | overrides))


def covariates(**overrides) -> dict:
    base = {"air_yards": 15.0, "pass_location": "left", "down": 3.0, "qb_hit": 1}
    return base | overrides


def test_the_catch_probability_is_the_inverse_logit_of_the_stored_draws():
    fitted = model()
    # air_yards 15 standardises to +1.0, so z and z squared are both 1.0;
    # `left` fires, `down_3` fires, `qb_hit` fires. Everything else is zero.
    logit = 0.05 + 0.1 + 0.01 + 0.2 + 0.002 + 0.5 + 0.30
    draws = fitted.catch_probability(f"2022|{AWAY}", covariates())
    assert draws.shape == (40,)
    assert draws[0] == pytest.approx(1.0 / (1.0 + np.exp(-logit)), abs=1e-12)


def test_a_null_covariate_is_priced_at_its_own_reference_level():
    fitted = model()
    # A null standardised covariate goes to the fitted mean, which standardises
    # to exactly 0 and so contributes nothing; a null factor goes to the omitted
    # level. Both are "no information", which is what the reference level means.
    nulled = fitted.catch_probability(
        f"2022|{AWAY}", covariates(air_yards=None, pass_location=None, down=None)
    )
    logit = 0.05 + 0.5 + 0.30
    assert nulled[0] == pytest.approx(1.0 / (1.0 + np.exp(-logit)), abs=1e-12)
    explicit = fitted.catch_probability(
        f"2022|{AWAY}", covariates(air_yards=10.0, pass_location="middle", down=1.0)
    )
    assert nulled[0] == pytest.approx(explicit[0], abs=1e-12)


def test_an_unknown_defence_season_is_priced_at_the_league_effect():
    fitted = model()
    unknown = fitted.catch_probability("2019|XXX", covariates())
    league = fitted.catch_probability(None, covariates())
    assert unknown[0] == pytest.approx(league[0], abs=1e-12)
    # And it is the league curve, not a neighbour's effect: u_d = 0.
    logit = 0.05 + 0.1 + 0.01 + 0.2 + 0.002 + 0.5
    assert unknown[0] == pytest.approx(1.0 / (1.0 + np.exp(-logit)), abs=1e-12)


def test_a_covariate_order_the_model_cannot_read_is_refused():
    fitted = model(
        covariate_order=("air_yards_z", "who_knows"),
        beta=np.tile(np.array([0.1, 0.2]), (40, 1)),
    )
    with pytest.raises(ValueError, match="not a covariate this model knows"):
        fitted.catch_probability(f"2022|{AWAY}", covariates())


# --------------------------------------------------------------------------
# the join — document 49 §4's adjudication frame
# --------------------------------------------------------------------------

GAME = "2022_01_AWY_HOM"


def pbp_play(play_id: float, **overrides) -> dict:
    """One ordinary play with every column both builders read."""
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


def worthy_throw(play_id: float, *, intercepted: bool, offence: str = HOME, **overrides) -> dict:
    return pbp_play(
        play_id,
        posteam=offence,
        defteam=AWAY if offence == HOME else HOME,
        interception=1 if intercepted else 0,
        epa=-4.0 if intercepted else -0.5,
        **overrides,
    )


def ftn_rows(play_ids: list[float], *, worthy: bool = True) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "nflverse_game_id": GAME,
                "nflverse_play_id": int(play_id),
                "is_interception_worthy": worthy,
                "is_catchable_ball": True,
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
    # infer_schema_length=None scans every row: these frames start with null
    # kicking columns, which the default 100-row scan would type Null.
    return pl.DataFrame(rows, infer_schema_length=None)


def test_the_join_keeps_only_charted_interception_worthy_passes():
    plays = game_frame(
        [
            pbp_play(1.0),
            worthy_throw(2.0, intercepted=False),
            worthy_throw(3.0, intercepted=True),
            pbp_play(4.0, play_type="run"),
        ]
    )
    ftn = pl.concat([ftn_rows([2.0, 3.0]), ftn_rows([1.0, 4.0], worthy=False)], how="vertical")
    worthy = worthy_throw_frame(plays, ftn)
    assert sorted(worthy["play_id"].to_list()) == [2.0, 3.0]
    assert worthy["defence_season"].to_list() == [f"2022|{AWAY}", f"2022|{AWAY}"]


def test_the_join_flags_a_throw_with_a_null_covariate():
    plays = game_frame(
        [
            worthy_throw(2.0, intercepted=False),
            worthy_throw(3.0, intercepted=False, pass_location=None, air_yards=None),
        ]
    )
    worthy = worthy_throw_frame(plays, ftn_rows([2.0, 3.0]))
    flags = dict(
        zip(
            worthy["play_id"].to_list(),
            worthy["covariates_imputed"].to_list(),
            strict=True,
        )
    )
    assert flags == {2.0: False, 3.0: True}


def test_a_game_with_no_charting_rows_yields_no_worthy_throws():
    plays = game_frame([worthy_throw(2.0, intercepted=False)])
    assert worthy_throw_frame(plays, ftn_rows([99.0])).height == 0


# --------------------------------------------------------------------------
# the event builder and the switch — document 49 §6's V-2 to V-7
# --------------------------------------------------------------------------


def model_at(p_catch: float, swing_table: SwingTable | None = None) -> DroppedPickModel:
    """A model whose catch probability is exactly `p_catch` on every throw.

    Constant draws and zero coefficients, so a sign test reads the sign and
    nothing else: any luck the ledger books comes from `actual - expected`.
    """
    n = 40
    return DroppedPickModel(
        alpha=np.full(n, float(np.log(p_catch / (1.0 - p_catch)))),
        beta=np.zeros((n, len(TEST_ORDER))),
        defence_effects={},
        covariate_order=TEST_ORDER,
        standardisation=TEST_STANDARDISATION,
        reference_levels={"pass_location": "middle", "down": 1.0},
        swing_table=swing_table or build_swing_table(worthy_corpus()),
    )


@pytest.fixture
def baselines():
    """Fumble and field-goal baselines on a corpus with a known 40% retention."""
    rows, play_id = [], 1000.0
    for i in range(100):
        rows.append(
            pbp_play(
                play_id,
                play_type="run",
                epa=-4.0 if i >= 40 else 0.0,
                fumble=1,
                fumbled_1_team=HOME,
                fumble_recovery_1_team=HOME if i < 40 else AWAY,
            )
        )
        play_id += 1
    for i in range(40):
        rows.append(
            pbp_play(
                play_id,
                play_type="field_goal",
                epa=2.5 if i < 32 else -2.5,
                field_goal_result="made" if i < 32 else "missed",
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


def test_v7_an_escaped_worthy_throw_by_the_home_team_books_good_luck(baselines):
    result = run(
        [worthy_throw(2.0, intercepted=False)],
        baselines,
        dropped_pick_model=model_at(0.45),
        ftn=ftn_rows([2.0]),
    )
    assert len(result.ledger) == 1
    entry = next(iter(result.ledger))
    assert entry.component == "dropped_pick"
    assert entry.charged_team == HOME
    assert entry.luck_epa > 0


def test_v7_a_throw_picked_off_the_home_team_at_a_low_catch_rate_books_bad_luck(baselines):
    result = run(
        [worthy_throw(2.0, intercepted=True)],
        baselines,
        dropped_pick_model=model_at(0.30),
        ftn=ftn_rows([2.0]),
    )
    entry = next(iter(result.ledger))
    assert entry.expected == pytest.approx(0.70, abs=1e-9)  # P(escape)
    assert entry.luck_epa < 0


def test_v7_the_away_offences_escape_carries_the_opposite_sign(baselines):
    home = run(
        [worthy_throw(2.0, intercepted=False, offence=HOME)],
        baselines,
        dropped_pick_model=model_at(0.45),
        ftn=ftn_rows([2.0]),
    )
    away = run(
        [worthy_throw(2.0, intercepted=False, offence=AWAY)],
        baselines,
        dropped_pick_model=model_at(0.45),
        ftn=ftn_rows([2.0]),
    )
    home_entry, away_entry = next(iter(home.ledger)), next(iter(away.ledger))
    assert home_entry.luck_epa == pytest.approx(-away_entry.luck_epa)
    assert away_entry.luck_epa < 0
    assert away_entry.charged_team == AWAY


def test_v5_every_event_carries_one_probability_per_posterior_draw(baselines):
    events = dropped_pick_events(
        game_frame([worthy_throw(2.0, intercepted=False), worthy_throw(3.0, intercepted=True)]),
        ftn_rows([2.0, 3.0]),
        model_at(0.45),
        n_draws=25,
        rng=np.random.default_rng(0),
    )
    assert len(events) == 2
    for event in events:
        assert len(event.expected_draws) == 25
        assert 0.0 <= event.expected_draws.mean() <= 1.0
        assert event.to_entry().expected == pytest.approx(float(event.expected_draws.mean()))


def test_v5_the_event_class_is_the_swing_bin_it_was_priced_in(baselines):
    events = dropped_pick_events(
        game_frame([worthy_throw(2.0, intercepted=False, yardline_100=50.0, down=3.0)]),
        ftn_rows([2.0]),
        model_at(0.45),
        n_draws=10,
        rng=np.random.default_rng(0),
    )
    assert events[0].event_class == "34-66 yd, late down"


def test_v2_the_variant_ledger_sums_to_the_margin_it_moved(baselines):
    result = run(
        [
            worthy_throw(2.0, intercepted=False),
            worthy_throw(3.0, intercepted=True, yardline_100=20.0, down=1.0),
            worthy_throw(4.0, intercepted=False, offence=AWAY, yardline_100=80.0, down=2.0),
        ],
        baselines,
        dropped_pick_model=model_at(0.45),
        ftn=ftn_rows([2.0, 3.0, 4.0]),
        points_per_epa=0.6,
    )
    assert len(result.ledger) == 3
    assert result.ledger.total_luck_epa() * 0.6 == pytest.approx(
        result.actual_margin - result.deserved_margin, abs=1e-9
    )


def test_v3_a_2022_game_with_no_worthy_throws_is_strict_field_for_field(baselines):
    rows = [pbp_play(1.0), pbp_play(2.0, play_type="run")]
    v13 = run(rows, baselines)
    variant = run(rows, baselines, dropped_pick_model=model_at(0.45), ftn=ftn_rows([9.0]))
    assert variant.variant == "strict"
    assert v13.variant == "strict"
    for name in ("actual_margin", "deserved_margin", "dtw_home", "dtw_interval", "total_luck_epa"):
        assert getattr(variant, name) == getattr(v13, name)
    assert np.array_equal(variant.margin_draws, v13.margin_draws)
    assert len(variant.ledger) == len(v13.ledger)


def test_v3_a_game_with_worthy_throws_is_labelled_as_the_variant(baselines):
    result = run(
        [worthy_throw(2.0, intercepted=False)],
        baselines,
        dropped_pick_model=model_at(0.45),
        ftn=ftn_rows([2.0]),
    )
    assert result.variant == "strict+dp"


def test_v4_a_pre_2022_game_asked_for_the_variant_warns_and_returns_strict(baselines):
    rows = [
        pbp_play(1.0, season=2019, game_id="2019_01_AWY_HOM"),
        worthy_throw(2.0, intercepted=False, season=2019, game_id="2019_01_AWY_HOM"),
    ]
    v13 = run(rows, baselines)
    with pytest.warns(UserWarning, match="charting starts in 2022"):
        variant = run(rows, baselines, dropped_pick_model=model_at(0.45), ftn=ftn_rows([2.0]))
    assert variant.variant == "strict"
    assert len(variant.ledger) == len(v13.ledger)
    assert variant.deserved_margin == v13.deserved_margin


def test_the_default_is_strict_so_the_component_is_opt_in(baselines):
    result = run([worthy_throw(2.0, intercepted=False)], baselines)
    assert result.variant == "strict"
    assert not [entry for entry in result.ledger if entry.component == "dropped_pick"]
