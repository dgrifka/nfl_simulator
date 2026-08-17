"""Simulator v1 — deserve-to-win, its interval, and the ledger behind it.

Implements document 05: §1's one rule, §3's treatment table, §4's two-layer
bootstrap. Every test builds its own small frame, so the suite stays
network-free and independent of the fitted artifacts.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nfl_simulator.components import fit_fg_baseline, fit_fumble_baseline
from nfl_simulator.fg_model import FieldGoalModel
from nfl_simulator.simulator import points_per_epa, simulate_game

HOME, AWAY = "HOM", "AWY"


def play(play_id: float, **overrides) -> dict:
    """One ordinary, luck-free play with every column the simulator reads."""
    base = {
        "game_id": "2024_01_AWY_HOM",
        "play_id": play_id,
        "season": 2024,
        "week": 1,
        "home_team": HOME,
        "away_team": AWAY,
        "posteam": HOME,
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
    }
    return base | overrides


def fumble_play(play_id: float, *, fumbler: str, recoverer: str, epa: float = -2.0) -> dict:
    return play(
        play_id,
        posteam=fumbler,
        play_type="run",
        epa=epa,
        fumble=1,
        fumbled_1_team=fumbler,
        fumble_recovery_1_team=recoverer,
    )


def fg_play(
    play_id: float, *, kicker: str, distance: float, made: bool, team: str = HOME, **overrides
) -> dict:
    return play(
        play_id,
        posteam=team,
        play_type="field_goal",
        epa=2.5 if made else -2.5,
        field_goal_result="made" if made else "missed",
        kick_distance=distance,
        kicker_player_id=kicker,
        **overrides,
    )


def frame(rows: list[dict]) -> pl.DataFrame:
    # infer_schema_length=None scans every row. These frames start with null
    # field_goal_result and kicker columns, so the default 100-row scan would
    # type them Null and then fail when a real value arrives.
    return pl.DataFrame(rows, infer_schema_length=None)


@pytest.fixture
def baselines():
    """League baselines fit on a synthetic corpus with a known 40% run-fumble rate.

    Deliberately NOT 50%: document 05 §3 requires class-specific coins, and a
    test built on a coin that happens to be fair could not tell the two apart.
    """
    rows = []
    play_id = 1000.0
    for i in range(100):
        # 40 of 100 run fumbles recovered by the fumbling team.
        recoverer = HOME if i < 40 else AWAY
        rows.append(
            fumble_play(play_id, fumbler=HOME, recoverer=recoverer, epa=-4.0 if i >= 40 else 0.0)
        )
        play_id += 1
    # Field goals across every 5-yard bin a test might reach for, with a make
    # rate that falls with distance. A single-distance corpus would leave other
    # bins unpopulated, and those attempts get silently skipped.
    for bin_start, made_count in [
        (20, 39),
        (25, 38),
        (30, 36),
        (35, 34),
        (45, 28),
        (50, 24),
        (55, 20),
    ]:
        for i in range(40):
            rows.append(
                fg_play(play_id, kicker="K1", distance=bin_start + 2.0, made=i < made_count)
            )
            play_id += 1
    for i in range(100):
        rows.append(fg_play(play_id, kicker="K1", distance=42.0, made=i < 80))
        play_id += 1
    corpus = frame(rows)
    return fit_fumble_baseline(corpus, min_class_size=10), fit_fg_baseline(corpus, min_bin_size=10)


@pytest.fixture
def fg_model():
    return FieldGoalModel(
        alpha=np.full(50, 1.9),
        beta=np.full(50, -0.115),
        gamma=np.full(50, 0.13),
        kicker_effects={"2024_K_GOOD": np.full(50, 0.6), "2024_K_BAD": np.full(50, -0.6)},
    )


def run(rows, baselines, fg_model, **kwargs):
    fumble_baseline, fg_baseline = baselines
    return simulate_game(
        frame(rows),
        fumble_baseline=fumble_baseline,
        fg_baseline=fg_baseline,
        fg_model=fg_model,
        points_per_epa=kwargs.pop("points_per_epa", 0.6),
        **kwargs,
    )


# --------------------------------------------------------------------------
# points_per_epa
# --------------------------------------------------------------------------


def test_points_per_epa_recovers_a_known_slope():
    games = pl.DataFrame({"epa_diff": [-10.0, 0.0, 10.0, 20.0], "margin": [-5.0, 0.0, 5.0, 10.0]})
    assert points_per_epa(games) == pytest.approx(0.5)


def test_points_per_epa_rejects_a_degenerate_input():
    games = pl.DataFrame({"epa_diff": [1.0, 1.0], "margin": [3.0, 7.0]})
    with pytest.raises(ValueError, match="variance"):
        points_per_epa(games)


# --------------------------------------------------------------------------
# the zero-luck case — the handoff's first smoke test
# --------------------------------------------------------------------------


def test_a_game_with_no_luck_events_returns_its_actual_margin(baselines, fg_model):
    result = run([play(1.0), play(2.0, posteam=AWAY)], baselines, fg_model)

    assert len(result.ledger) == 0
    assert result.total_luck_epa == pytest.approx(0.0)
    assert result.deserved_margin == pytest.approx(result.actual_margin)


def test_a_game_with_no_luck_events_gives_the_home_winner_a_dtw_of_one(baselines, fg_model):
    result = run([play(1.0), play(2.0)], baselines, fg_model)
    assert result.actual_margin > 0
    assert result.dtw_home == pytest.approx(1.0)
    assert result.dtw_interval == (pytest.approx(1.0), pytest.approx(1.0))


def test_a_game_with_no_luck_events_gives_a_home_loser_a_dtw_of_zero(baselines, fg_model):
    result = run([play(1.0, result=-7.0), play(2.0, result=-7.0)], baselines, fg_model)
    assert result.dtw_home == pytest.approx(0.0)


# --------------------------------------------------------------------------
# the one rule
# --------------------------------------------------------------------------


def test_fumble_luck_uses_the_class_rate_not_a_fair_coin(baselines, fg_model):
    """The synthetic corpus has a 40% run-fumble recovery rate. A flat 50/50
    would book different luck, so this test distinguishes the two."""
    result = run([fumble_play(1.0, fumbler=HOME, recoverer=HOME)], baselines, fg_model)

    assert len(result.ledger) == 1
    assert result.ledger.entries[0].expected == pytest.approx(0.40, abs=0.02)


def test_recovering_your_own_fumble_books_good_luck_for_the_home_team(baselines, fg_model):
    result = run([fumble_play(1.0, fumbler=HOME, recoverer=HOME)], baselines, fg_model)
    assert result.total_luck_epa > 0
    assert result.deserved_margin < result.actual_margin


def test_losing_your_own_fumble_books_bad_luck_for_the_home_team(baselines, fg_model):
    result = run([fumble_play(1.0, fumbler=HOME, recoverer=AWAY)], baselines, fg_model)
    assert result.total_luck_epa < 0
    assert result.deserved_margin > result.actual_margin


def test_the_away_teams_fumble_luck_carries_the_opposite_sign(baselines, fg_model):
    home = run([fumble_play(1.0, fumbler=HOME, recoverer=HOME)], baselines, fg_model)
    away = run([fumble_play(1.0, fumbler=AWAY, recoverer=AWAY)], baselines, fg_model)
    assert home.total_luck_epa == pytest.approx(-away.total_luck_epa)


def test_the_ledger_sums_to_the_applied_margin_adjustment(baselines, fg_model):
    """The identity from document 05 §4. Asserted, not tolerated."""
    rows = [
        fumble_play(1.0, fumbler=HOME, recoverer=HOME),
        fumble_play(2.0, fumbler=AWAY, recoverer=HOME),
        fg_play(3.0, kicker="K_GOOD", distance=48.0, made=False),
    ]
    result = run(rows, baselines, fg_model, points_per_epa=0.6)

    applied = result.actual_margin - result.deserved_margin
    assert applied == pytest.approx(result.ledger.total_luck_epa() * 0.6)


# --------------------------------------------------------------------------
# field goals — partial neutralization against the kicker's own rate
# --------------------------------------------------------------------------


def test_a_good_kicker_is_charged_less_bad_luck_for_the_same_miss(baselines, fg_model):
    """Partial neutralization: a miss by a kicker who makes more of them is a
    bigger deviation from expectation, so it books MORE bad luck. The point is
    that the two differ at all — a league-average curve would treat them alike."""
    good = run([fg_play(1.0, kicker="K_GOOD", distance=48.0, made=False)], baselines, fg_model)
    bad = run([fg_play(1.0, kicker="K_BAD", distance=48.0, made=False)], baselines, fg_model)

    assert good.ledger.entries[0].expected > bad.ledger.entries[0].expected
    assert good.total_luck_epa < bad.total_luck_epa


def test_an_unknown_kicker_is_priced_at_the_league_curve(baselines, fg_model):
    known = run([fg_play(1.0, kicker="K_GOOD", distance=48.0, made=True)], baselines, fg_model)
    unknown = run([fg_play(1.0, kicker="K_NOBODY", distance=48.0, made=True)], baselines, fg_model)
    assert known.ledger.entries[0].expected != pytest.approx(
        unknown.ledger.entries[0].expected, abs=1e-6
    )


def test_field_goal_probability_falls_with_distance(baselines, fg_model):
    near = run([fg_play(1.0, kicker="K_GOOD", distance=30.0, made=True)], baselines, fg_model)
    far = run([fg_play(1.0, kicker="K_GOOD", distance=55.0, made=True)], baselines, fg_model)
    assert near.ledger.entries[0].expected > far.ledger.entries[0].expected


# --------------------------------------------------------------------------
# the two-layer bootstrap
# --------------------------------------------------------------------------


def test_dtw_stays_a_probability(baselines, fg_model):
    rows = [fumble_play(float(i), fumbler=HOME, recoverer=HOME) for i in range(1, 5)]
    result = run(rows, baselines, fg_model)
    assert 0.0 <= result.dtw_home <= 1.0
    assert result.dtw_interval[0] <= result.dtw_home <= result.dtw_interval[1]


def test_a_fumble_lottery_moves_dtw_materially(baselines, fg_model):
    """The handoff's second smoke test: a team that recovered everything it
    dropped should not keep all the credit."""
    rows = [play(1.0, epa=0.0, result=2.0)]
    rows += [fumble_play(float(i), fumbler=HOME, recoverer=HOME, epa=0.0) for i in range(10, 15)]
    lucky = run(rows, baselines, fg_model)

    assert lucky.actual_margin > 0
    assert lucky.dtw_home < 0.5, "five recovered own fumbles should flip a two-point win"


def test_the_interval_widens_when_the_rate_posterior_is_wider(baselines, fg_model):
    """Layer 1 has to actually do something. A model whose kicker effects vary
    across draws must produce a wider DTW interval than one that does not."""
    tight = FieldGoalModel(
        alpha=np.full(50, 1.9),
        beta=np.full(50, -0.115),
        gamma=np.zeros(50),
        kicker_effects={"2024_K_GOOD": np.full(50, 0.0)},
    )
    wide = FieldGoalModel(
        alpha=np.full(50, 1.9),
        beta=np.full(50, -0.115),
        gamma=np.zeros(50),
        kicker_effects={"2024_K_GOOD": np.linspace(-1.5, 1.5, 50)},
    )
    # The actual margin is chosen so the deserved margin lands near zero. DTW
    # saturated at 1.0 has no width for either arm to differ in, so a scenario
    # away from the decision boundary could not test this at all.
    rows = [
        fg_play(float(i), kicker="K_GOOD", distance=50.0, made=False, result=-6.0)
        for i in range(1, 4)
    ]
    rows.append(play(9.0, result=-6.0))

    narrow = run(rows, baselines, tight, n_coin_draws=400)
    broad = run(rows, baselines, wide, n_coin_draws=400)
    assert 0.05 < narrow.dtw_home < 0.95, "scenario must sit near the decision boundary"

    def width(result):
        return result.dtw_interval[1] - result.dtw_interval[0]

    assert width(broad) > width(narrow)


def test_the_same_seed_reproduces_the_same_answer(baselines, fg_model):
    rows = [fumble_play(float(i), fumbler=HOME, recoverer=HOME) for i in range(1, 4)]
    first = run(rows, baselines, fg_model, seed=7)
    second = run(rows, baselines, fg_model, seed=7)
    assert first.dtw_home == second.dtw_home
    assert first.dtw_interval == second.dtw_interval


def test_margin_draws_are_returned_for_plotting(baselines, fg_model):
    rows = [fumble_play(float(i), fumbler=HOME, recoverer=HOME) for i in range(1, 4)]
    result = run(rows, baselines, fg_model, n_coin_draws=100)
    assert result.margin_draws.ndim == 1
    assert len(result.margin_draws) > 100
    assert result.margin_draws.mean() == pytest.approx(result.deserved_margin, abs=1.0)


# --------------------------------------------------------------------------
# components the treatment table says NOT to touch
# --------------------------------------------------------------------------


def test_penalties_are_never_neutralized(baselines, fg_model):
    """Document 05 §3: penalties fail the branch-point gate."""
    rows = [play(1.0, penalty=1, epa=-1.5), play(2.0, penalty=1, epa=-2.0, posteam=AWAY)]
    result = run(rows, baselines, fg_model)
    assert len(result.ledger) == 0


def test_interceptions_are_never_neutralized_in_v1(baselines, fg_model):
    """Step 3a could not attribute the spread, so v1 leaves interceptions alone."""
    rows = [play(1.0, interception=1, epa=-4.0)]
    result = run(rows, baselines, fg_model)
    assert len(result.ledger) == 0
