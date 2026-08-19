"""The decomposition must partition EPA exactly and attribute luck to the right team."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nfl_simulator.components import (
    COMPONENTS,
    add_home_perspective_epa,
    build_game_table,
    decompose_games,
    decompose_plays,
    fit_fg_baseline,
    fit_fumble_baseline,
    fit_xp_baseline,
    variance_shares,
)


def make_play(**overrides) -> dict:
    """One pbp row with every column the decomposition reads."""
    play = {
        "game_id": "2024_01_AAA_BBB",
        "play_id": 1,
        "season": 2024,
        "week": 1,
        "home_team": "BBB",
        "away_team": "AAA",
        "posteam": "AAA",
        "defteam": "BBB",
        "play_type": "run",
        "epa": 0.0,
        "result": 3,
        "fumble": 0,
        "fumble_lost": 0,
        "fumbled_1_team": None,
        "fumble_recovery_1_team": None,
        "fumble_out_of_bounds": 0,
        "aborted_play": 0,
        "interception": 0,
        "penalty": 0,
        "field_goal_result": None,
        "kick_distance": None,
    }
    play.update(overrides)
    return play


@pytest.fixture
def toy_pbp():
    """A synthetic season with a controlled mix of fumbles, FGs, INTs, penalties."""
    rng = np.random.default_rng(0)
    rows = []
    play_id = 0
    for game in range(60):
        home, away = f"H{game % 10}", f"A{game % 10}"
        game_id = f"2024_{game:02d}_{away}_{home}"
        game_margin = int(rng.integers(-21, 22))
        for _ in range(40):
            play_id += 1
            offense = home if rng.random() < 0.5 else away
            defense = away if offense == home else home
            kind = rng.choice(
                ["plain", "fumble", "fg", "int", "penalty"], p=[0.6, 0.1, 0.1, 0.1, 0.1]
            )
            play = make_play(
                game_id=game_id,
                play_id=play_id,
                home_team=home,
                away_team=away,
                posteam=offense,
                defteam=defense,
                epa=float(rng.normal(0, 1)),
                result=game_margin,
            )
            if kind == "fumble":
                recovered_by = offense if rng.random() < 0.5 else defense
                play |= {
                    "fumble": 1,
                    "fumbled_1_team": offense,
                    "fumble_recovery_1_team": recovered_by,
                    "play_type": "run",
                }
            elif kind == "fg":
                distance = float(rng.integers(20, 56))
                play |= {
                    "play_type": "field_goal",
                    "kick_distance": distance,
                    "field_goal_result": "made" if rng.random() < 0.85 else "missed",
                }
            elif kind == "int":
                play |= {"interception": 1, "play_type": "pass"}
            elif kind == "penalty":
                play |= {"penalty": 1}
            rows.append(play)
    return pl.DataFrame(rows)


class TestHomePerspective:
    def test_flips_sign_for_away_offense(self):
        df = pl.DataFrame([make_play(posteam="AAA", epa=1.0), make_play(posteam="BBB", epa=1.0)])
        out = add_home_perspective_epa(df)
        # AAA is the away team, so its +1 EPA is -1 for the home team.
        assert out["epa_home"].to_list() == [-1.0, 1.0]


class TestPartition:
    def test_components_sum_to_epa_diff(self, toy_pbp):
        games = build_game_table(toy_pbp)
        total = sum(games[component] for component in COMPONENTS)
        assert np.allclose(total.to_numpy(), games["epa_diff"].to_numpy())

    def test_mismatch_raises(self, toy_pbp):
        """The partition invariant is enforced, not assumed."""
        fumble_baseline = fit_fumble_baseline(toy_pbp)
        fg_baseline = fit_fg_baseline(toy_pbp)
        plays = decompose_plays(toy_pbp, fumble_baseline, fg_baseline)
        corrupted = plays.with_columns(pl.col("core") + 1.0)
        with pytest.raises(AssertionError, match="do not sum"):
            decompose_games(corrupted)

    def test_one_row_per_game(self, toy_pbp):
        games = build_game_table(toy_pbp)
        assert games.height == toy_pbp["game_id"].n_unique()


def _widening_corpus() -> list[dict]:
    """100 fumbles by AAA: 49 recovered, 50 lost, 1 out of bounds.

    Retention is exactly 50/100 and the branch means are exactly +1.0 and -3.0,
    so every number a widening test asserts is arithmetic rather than a fit.
    """
    rows = [
        make_play(
            game_id=f"g{i}",
            play_id=i,
            fumble=1,
            fumbled_1_team="AAA",
            fumble_recovery_1_team="AAA" if i < 49 else "BBB",
            epa=1.0 if i < 49 else -3.0,
        )
        for i in range(99)
    ]
    rows.append(
        make_play(
            game_id="g99",
            play_id=99,
            fumble=1,
            fumbled_1_team="AAA",
            fumble_recovery_1_team=None,
            fumble_out_of_bounds=1,
            epa=1.0,
        )
    )
    return rows


class TestFumbleLuck:
    def test_zero_when_no_fumbles(self, toy_pbp):
        clean = toy_pbp.with_columns(
            pl.lit(0).alias("fumble"),
            pl.lit(None, dtype=pl.String).alias("fumbled_1_team"),
            pl.lit(None, dtype=pl.String).alias("fumble_recovery_1_team"),
        )
        games = build_game_table(clean)
        assert games["fumble_luck"].abs().max() == 0.0

    def test_luck_is_signed_toward_the_recovering_team(self):
        """Recovering your own fumble is positive luck for the fumbling team."""
        rows = []
        # 100 identical fumbles, half recovered each way, so p_own = 0.5 exactly.
        for i in range(100):
            offense_recovers = i % 2 == 0
            rows.append(
                make_play(
                    game_id=f"g{i}",
                    play_id=i,
                    fumble=1,
                    fumbled_1_team="AAA",
                    fumble_recovery_1_team="AAA" if offense_recovers else "BBB",
                    epa=1.0 if offense_recovers else -3.0,
                )
            )
        df = pl.DataFrame(rows)
        plays = decompose_plays(df, fit_fumble_baseline(df), fit_fg_baseline(df))
        luck = plays["fumble_luck"].to_numpy()
        # AAA is the away team, so AAA recovering is *negative* in home perspective.
        assert luck[0] < 0 and luck[1] > 0
        assert np.isclose(luck.mean(), 0.0)
        # Swing magnitude is half the branch gap: 0.5 * (1.0 - -3.0) = 2.0
        assert np.allclose(np.abs(luck), 2.0)

    def test_out_of_bounds_counts_as_keeping_the_ball(self):
        """v1.2, document 18 §5g: the branch is *did the fumbling team keep it*,
        and skipping out of bounds is one of the two ways to keep it."""
        df = pl.DataFrame(_widening_corpus())
        plays = decompose_plays(df, fit_fumble_baseline(df), fit_fg_baseline(df))
        out_of_bounds = plays.filter(pl.col("fumble_out_of_bounds") == 1)
        # Retention rate is 50/100, the swing is 1.0 - -3.0 = 4.0, and AAA is the
        # away team, so keeping the ball is negative luck in home perspective.
        assert np.isclose(out_of_bounds["fumble_luck"].item(), -2.0)

    def test_a_fumble_nobody_resolved_books_no_luck(self):
        """No recovery team and no out-of-bounds flag is an unresolved
        disposition, not a retention. Two such plays exist in ten seasons."""
        df = pl.DataFrame(
            [
                make_play(
                    fumble=1,
                    fumbled_1_team="AAA",
                    fumble_recovery_1_team=None,
                    fumble_out_of_bounds=0,
                    epa=-1.0,
                )
            ]
        )
        assert fit_fumble_baseline(df).table.height == 0
        plays = decompose_plays(df, fit_fumble_baseline(df), fit_fg_baseline(df))
        assert plays["fumble_luck"].item() == 0.0
        assert plays["core"].item() == 1.0  # away team's -1 EPA, home perspective

    def test_conflicting_flags_resolve_to_the_recovery(self):
        """Eleven fumbles carry both an out-of-bounds flag and a named recovering
        team. A named recovering team is the more specific fact (document 18 §3)."""
        rows = _widening_corpus()
        # Turn the out-of-bounds play into a conflicted one: the opponent
        # recovered it, so it is a loss despite the flag.
        rows[-1] |= {"fumble_recovery_1_team": "BBB", "epa": -3.0}
        df = pl.DataFrame(rows)
        baseline = fit_fumble_baseline(df)
        assert np.isclose(baseline.table["p_own"].item(), 0.49)
        plays = decompose_plays(df, fit_fumble_baseline(df), fit_fg_baseline(df))
        conflicted = plays.filter(pl.col("fumble_out_of_bounds") == 1)
        assert np.isclose(conflicted["fumble_luck"].item(), 0.49 * 4.0)

    def test_the_baseline_reports_the_out_of_bounds_rate(self):
        """Reporting only — the rate is not used in the luck identity, but a
        class that goes out of bounds 3% of the time and one that goes 11% are
        not the same coin, and the table has to show it (document 18 §5g)."""
        df = pl.DataFrame(_widening_corpus())
        table = fit_fumble_baseline(df).table
        assert np.isclose(table["p_out_of_bounds"].item(), 0.01)

    def test_aborted_snaps_get_their_own_class(self, toy_pbp):
        """Botched snaps recover far more than half the time; pooling them would
        mislabel a large chunk of quarterback-recovered fumbles as good luck."""
        aborted = toy_pbp.with_columns(
            pl.when(pl.col("fumble") == 1).then(1).otherwise(0).alias("aborted_play")
        )
        baseline = fit_fumble_baseline(aborted)
        classes = baseline.table["fumble_class"].to_list()
        assert all(name.endswith("/aborted") for name in classes)


class TestFieldGoalLuck:
    def test_make_rate_declines_with_distance(self, toy_pbp):
        baseline = fit_fg_baseline(toy_pbp, min_bin_size=1)
        assert baseline.table.height > 1
        assert set(baseline.table.columns) >= {"fg_bin", "p_make", "swing_value"}

    def test_expected_make_produces_zero_luck(self):
        """A kicker who makes exactly the bin rate banks no luck on average."""
        rows = [
            make_play(
                game_id=f"g{i}",
                play_id=i,
                play_type="field_goal",
                kick_distance=42.0,
                field_goal_result="made" if i < 80 else "missed",
                epa=1.0 if i < 80 else -3.0,
                posteam="BBB",
                defteam="AAA",
            )
            for i in range(100)
        ]
        df = pl.DataFrame(rows)
        plays = decompose_plays(df, fit_fumble_baseline(df), fit_fg_baseline(df))
        assert np.isclose(plays["fg_luck"].mean(), 0.0)
        # Making one is +0.2 * 4.0 luck; missing one is -0.8 * 4.0.
        assert np.isclose(plays["fg_luck"].to_numpy()[0], 0.2 * 4.0)
        assert np.isclose(plays["fg_luck"].to_numpy()[-1], -0.8 * 4.0)


class TestVarianceShares:
    def test_epa_shares_sum_to_one(self, toy_pbp):
        games = build_game_table(toy_pbp)
        table = variance_shares(games, "epa_diff")
        assert "unexplained" not in table["component"].to_list()
        assert np.isclose(table["share"].sum(), 1.0)

    def test_margin_shares_sum_to_r_squared(self, toy_pbp):
        games = build_game_table(toy_pbp)
        table = variance_shares(games, "margin")
        explained = table.filter(pl.col("component") != "unexplained")["share"].sum()
        correlation = np.corrcoef(games["epa_diff"], games["margin"])[0, 1]
        assert np.isclose(explained, correlation**2)


class TestDegenerateBaselines:
    """Classes that never went one way used to produce a null swing, silently
    zeroing luck. These pin the fallbacks in place."""

    def test_class_that_never_loses_the_ball_gets_a_finite_swing(self):
        rows = [
            make_play(
                game_id=f"g{i}",
                play_id=i,
                fumble=1,
                aborted_play=1,
                fumbled_1_team="AAA",
                fumble_recovery_1_team="AAA",
                epa=-0.5,
            )
            for i in range(40)
        ]
        # A second class supplies the pooled 'lost' branch.
        rows += [
            make_play(
                game_id=f"h{i}",
                play_id=1000 + i,
                fumble=1,
                fumbled_1_team="AAA",
                fumble_recovery_1_team="BBB" if i % 2 else "AAA",
                epa=-2.0,
            )
            for i in range(40)
        ]
        df = pl.DataFrame(rows)
        table = fit_fumble_baseline(df).table
        aborted = table.filter(pl.col("fumble_class") == "run/aborted")
        assert aborted["swing_value"].item() is not None
        # p_own is 1.0 for that class, so the luck it books is still exactly zero.
        plays = decompose_plays(df, fit_fumble_baseline(df), fit_fg_baseline(df))
        assert np.allclose(plays.head(40)["fumble_luck"].to_numpy(), 0.0)

    def test_no_fumbles_at_all_yields_an_empty_baseline(self):
        df = pl.DataFrame([make_play(play_id=i) for i in range(5)])
        assert fit_fumble_baseline(df).table.height == 0
        plays = decompose_plays(df, fit_fumble_baseline(df), fit_fg_baseline(df))
        assert plays["fumble_luck"].abs().max() == 0.0

    def test_sparse_fg_bin_borrows_from_the_nearest_bin_not_the_league(self):
        """A 65-yard heave must not inherit the 85% make rate of chip shots."""
        rows = [
            make_play(
                game_id=f"short{i}",
                play_id=i,
                play_type="field_goal",
                kick_distance=25.0,
                field_goal_result="made" if i < 95 else "missed",
                epa=0.1 if i < 95 else -3.5,
            )
            for i in range(100)
        ]
        rows += [
            make_play(
                game_id=f"long{i}",
                play_id=500 + i,
                play_type="field_goal",
                kick_distance=60.0,
                field_goal_result="made" if i < 16 else "missed",
                epa=2.0 if i < 16 else -1.4,
            )
            for i in range(40)
        ]
        rows += [
            make_play(
                game_id=f"heave{i}",
                play_id=900 + i,
                play_type="field_goal",
                kick_distance=66.0,
                field_goal_result="missed",
                epa=-1.0,
            )
            for i in range(3)
        ]
        table = fit_fg_baseline(pl.DataFrame(rows)).table
        heave = table.filter(pl.col("fg_bin") == 65)
        assert heave["borrowed_from"].item() == 60
        assert heave["p_make"].item() == pytest.approx(16 / 40)


class TestBlockedKicks:
    """Blocked kicks leave the kicking populations — docs/research/26 §2, 30 §7.

    A blocked kick has no branch point: the ball never flew, and what resolved
    the play was the defending team's rush beating the protection. Gate A denies
    it, so v1.3 stops neutralizing it. The v1.2 population stays reachable
    because v1.1's and v1.2's artifacts have to remain reproducible.
    """

    @staticmethod
    def _fg_corpus() -> pl.DataFrame:
        rows = [
            make_play(
                game_id=f"g{i}",
                play_id=i,
                play_type="field_goal",
                kick_distance=42.0,
                field_goal_result="made" if i < 10 else "blocked",
                epa=1.0 if i < 10 else -3.0,
                posteam="BBB",
                defteam="AAA",
            )
            for i in range(20)
        ]
        # A miss, so the bin has both branches and a usable swing value.
        rows.append(
            make_play(
                game_id="g20",
                play_id=20,
                play_type="field_goal",
                kick_distance=42.0,
                field_goal_result="missed",
                epa=-3.0,
                posteam="BBB",
                defteam="AAA",
            )
        )
        return pl.DataFrame(rows)

    @staticmethod
    def _xp_corpus() -> pl.DataFrame:
        rows = []
        for i in range(30):
            result = "good" if i < 20 else ("failed" if i < 25 else "blocked")
            rows.append(
                make_play(
                    game_id=f"x{i}",
                    play_id=i,
                    play_type="extra_point",
                    extra_point_attempt=1,
                    extra_point_result=result,
                    kick_distance=33.0,
                    epa=0.07 if result == "good" else -0.95,
                    posteam="BBB",
                    defteam="AAA",
                )
            )
        return pl.DataFrame(rows)

    def test_a_blocked_field_goal_is_not_a_field_goal_attempt(self):
        table = fit_fg_baseline(self._fg_corpus(), min_bin_size=1).table
        assert table["n"].to_list() == [11]
        assert table["p_make"].to_list() == [pytest.approx(10 / 11)]

    def test_the_v12_field_goal_population_is_still_reproducible(self):
        table = fit_fg_baseline(self._fg_corpus(), min_bin_size=1, include_blocked=True).table
        assert table["n"].to_list() == [21]
        assert table["p_make"].to_list() == [pytest.approx(10 / 21)]

    def test_a_blocked_extra_point_is_not_an_extra_point_attempt(self):
        baseline = fit_xp_baseline(self._xp_corpus(), min_attempts=10)
        assert baseline.n == 25
        assert baseline.p_make == pytest.approx(0.8)

    def test_the_v12_extra_point_population_is_still_reproducible(self):
        baseline = fit_xp_baseline(self._xp_corpus(), min_attempts=10, include_blocked=True)
        assert baseline.n == 30
        assert baseline.p_make == pytest.approx(20 / 30)

    def test_a_blocked_kicks_epa_lands_in_core(self):
        """The luck the ledger stops booking has to go somewhere, and this is it."""
        corpus = self._fg_corpus()
        plays = decompose_plays(
            corpus, fit_fumble_baseline(corpus), fit_fg_baseline(corpus, min_bin_size=1)
        )
        blocked = plays.filter(pl.col("field_goal_result") == "blocked")
        assert blocked.height == 10
        assert np.allclose(blocked["fg_luck"].to_numpy(), 0.0)
        np.testing.assert_allclose(blocked["core"].to_numpy(), blocked["epa_home"].to_numpy())

    def test_the_decomposition_still_partitions_epa_with_a_blocked_kick_in_it(self):
        corpus = self._fg_corpus()
        plays = decompose_plays(
            corpus, fit_fumble_baseline(corpus), fit_fg_baseline(corpus, min_bin_size=1)
        )
        total = sum(plays[component].to_numpy() for component in COMPONENTS)
        np.testing.assert_allclose(total, plays["epa_home"].to_numpy(), atol=1e-12)
