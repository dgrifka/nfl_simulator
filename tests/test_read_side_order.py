"""The read side is a pure function of ledger values, never of row order.

Document 73 made the adjudication row-order invariant; this file closes the
read side behind it. Every case feeds the same ledger values in two physical
orders and requires the same figure inputs back — the rows a figure walks,
the bars a waterfall draws, and the five events a share card prints by name.
A tie in ``abs(points)`` is routine (two extra points by the same kicker), so
"biggest first" alone is not an order at all; play then component is.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import polars as pl
import pytest

from nfl_simulator.plots import (
    GameVerdict,
    LuckBar,
    band_sweep,
    fold_to_frame,
    group_rows,
    luck_bars,
    table_rows,
    team_ledgers,
)
from nfl_simulator.render import Sources, prepare_rows

PPE = 0.8389495557652871


def verdict(**overrides) -> GameVerdict:
    base = dict(
        game_id="2018_05_GB_DET",
        home_team="DET",
        away_team="GB",
        actual_margin=8.0,
        deserved_margin=-8.28,
        dtw_home=0.05,
        dtw_interval=(0.03, 0.08),
        margin_draws=np.linspace(-20.0, 6.0, 80),
        home_score=31,
        away_score=23,
        game_date="2018-10-07",
    )
    base.update(overrides)
    return GameVerdict(**base)


def ledger_frame() -> pl.DataFrame:
    """Four rows in a deliberately scrambled physical order.

    Play 900 is a blocked kick booking two components on one ``play_id`` —
    the pairing document 73's read-side audit flagged as the reason a sort on
    the play alone is not total on a ledger.
    """
    return pl.DataFrame(
        {
            "play_id": [900.0, 834.0, 900.0, 324.0],
            "component": ["fumble", "field_goal", "field_goal", "fumble"],
            "event_class": ["fg/live", "40-44 yd", "55-59 yd", "punt/live"],
            "charged_team": ["DET", "GB", "DET", "GB"],
            "expected": [0.5, 0.8769803817228944, 0.55, 0.6840206759729129],
            "swing": [2.0, -4.290041955688844, 3.0, -5.000382690077953],
            "luck_epa": [-1.0, 3.762282631907235, -1.65, 3.4203651477903745],
        }
    )


def card_row(
    luck_epa: float,
    *,
    play_id: float,
    component: str = "field_goal",
    event_class: str = "40-44 yd",
    charged_team: str = "GB",
    actual: float = 0.0,
    opponent: str = "DET",
) -> dict:
    """A ledger row with the keys ``render.prepare_rows`` adds."""
    return {
        "play_id": play_id,
        "component": component,
        "event_class": event_class,
        "charged_team": charged_team,
        "luck_epa": luck_epa,
        "actual": actual,
        "expected": None,
        "opponent": opponent,
        "kick_distance": None,
        "kicker": None,
    }


# --------------------------------------------------------------------------
# render.prepare_rows — the one place a frame becomes the rows list
# --------------------------------------------------------------------------


def test_prepare_rows_is_row_order_invariant():
    frame = ledger_frame()
    game = verdict()
    assert prepare_rows(frame, game) == prepare_rows(frame.reverse(), game)


def test_prepare_rows_sorts_to_play_then_component():
    rows = prepare_rows(ledger_frame(), verdict())
    assert [(row["play_id"], row["component"]) for row in rows] == [
        (324.0, "fumble"),
        (834.0, "field_goal"),
        (900.0, "field_goal"),
        (900.0, "fumble"),
    ]


# --------------------------------------------------------------------------
# plots.luck_bars — biggest first, ties broken by play then component
# --------------------------------------------------------------------------


def test_luck_bars_breaks_an_abs_points_tie_by_play():
    rows = [
        card_row(2.0, play_id=9.0),
        card_row(2.0, play_id=3.0),
        card_row(-2.5, play_id=5.0, charged_team="DET", opponent="GB", actual=1.0),
    ]
    bars = luck_bars(rows, points_per_epa=PPE, floor=0.0)
    assert [bar.play_id for bar in bars] == [5.0, 3.0, 9.0]
    assert bars == luck_bars(list(reversed(rows)), points_per_epa=PPE, floor=0.0)


def test_luck_bars_breaks_a_same_play_tie_by_component():
    # A blocked kick's two bookings can carry the same abs(points).
    rows = [
        card_row(2.0, play_id=7.0, component="fumble", event_class="fg/live"),
        card_row(-2.0, play_id=7.0, actual=1.0, charged_team="DET", opponent="GB"),
    ]
    bars = luck_bars(rows, points_per_epa=PPE, floor=0.0)
    assert [bar.component for bar in bars] == ["field_goal", "fumble"]
    assert bars == luck_bars(list(reversed(rows)), points_per_epa=PPE, floor=0.0)


# --------------------------------------------------------------------------
# plots.group_rows and fold_to_frame — the drawn set is order-free
# --------------------------------------------------------------------------


def test_group_rows_output_is_input_order_invariant_on_ties():
    bars = [
        LuckBar(label="kick A", points=2.0, play_id=10.0, team="GB", component="field_goal"),
        LuckBar(label="kick B", points=-2.0, play_id=20.0, team="DET", component="field_goal"),
        LuckBar(label="fumble", points=1.5, play_id=30.0, team="GB", component="fumble"),
    ]
    ordered = group_rows(bars)
    assert [bar.play_id for bar in ordered] == [10.0, 20.0, 30.0]
    assert group_rows(list(reversed(bars))) == ordered


def test_fold_to_frame_settles_on_the_same_rows_for_permuted_ledgers():
    rows = [
        card_row(3.8, play_id=1.0),
        card_row(3.8, play_id=2.0),
        card_row(-3.8, play_id=3.0, charged_team="DET", opponent="GB", actual=1.0),
        card_row(0.05, play_id=4.0, component="fumble", event_class="punt/live"),
        card_row(0.05, play_id=5.0, component="fumble", event_class="punt/live"),
    ]
    game = verdict()
    folded_a, frame_a = fold_to_frame(game, luck_bars(rows, points_per_epa=PPE))
    folded_b, frame_b = fold_to_frame(
        game, luck_bars(list(reversed(rows)), points_per_epa=PPE)
    )
    assert frame_a == frame_b
    assert folded_a == folded_b


# --------------------------------------------------------------------------
# the share card — a tie straddling the top-5 cut names the same event
# --------------------------------------------------------------------------


def straddling_rows() -> list[dict]:
    """Seven GB rows; plays 60 and 61 tie exactly across the 5th/6th slot.

    The tied pair are different events — a kick and a fumble — so which one
    the cut names is visible content, exactly the case the audit flagged.
    """
    return [
        card_row(5.0, play_id=10.0),
        card_row(4.0, play_id=20.0),
        card_row(3.0, play_id=30.0),
        card_row(2.0, play_id=40.0),
        card_row(1.5, play_id=60.0),
        card_row(1.5, play_id=61.0, component="fumble", event_class="punt/live"),
        card_row(1.0, play_id=70.0),
    ]


def test_share_card_tie_straddling_the_cut_resolves_identically():
    game = verdict()
    rows = straddling_rows()
    permuted = [rows[5], rows[2], rows[0], rows[6], rows[4], rows[1], rows[3]]
    away_a, home_a = team_ledgers(game, rows, points_per_epa=1.0)
    away_b, home_b = team_ledgers(game, permuted, points_per_epa=1.0)
    assert table_rows(away_a) == table_rows(away_b)
    assert home_a == home_b


def test_share_card_tie_at_the_cut_prints_the_earlier_play():
    away, _home = team_ledgers(verdict(), straddling_rows(), points_per_epa=1.0)
    printed = table_rows(away)
    assert len(printed) == 6  # five named rows and the fold
    assert printed[-1].event == "and 2 more"
    # The earlier play wins the fifth slot: the kick is named, the fumble is
    # folded — and the fold still carries the exact sum of what it absorbed.
    # GB is the away side, so its rows are signed away from home: negative.
    assert "field goal" in printed[4].event
    assert printed[-1].points == pytest.approx(-(1.5 + 1.0))


# --------------------------------------------------------------------------
# the LOW guards — single-row invariants asserted, not assumed
# --------------------------------------------------------------------------


def sources_with(games: pl.DataFrame, **overrides) -> Sources:
    empty = pl.DataFrame({"game_id": []}, schema={"game_id": pl.String})
    base = dict(games=games, ledger=empty, schedule=empty, overtime=empty, slope=1.0)
    base.update(overrides)
    return Sources(**base)


def test_a_summary_with_two_rows_for_one_game_is_refused():
    games = pl.DataFrame({"game_id": ["2018_05_GB_DET", "2018_05_GB_DET"], "dtw_home": [0.1, 0.9]})
    with pytest.raises(AssertionError):
        sources_with(games).game_row("2018_05_GB_DET")


def test_a_schedule_with_two_rows_for_one_game_is_refused():
    schedule = pl.DataFrame({"game_id": ["2018_05_GB_DET"] * 2, "gameday": ["a", "b"]})
    with pytest.raises(AssertionError):
        sources_with(pl.DataFrame({"game_id": []}, schema={"game_id": pl.String}), schedule=schedule).schedule_row(
            "2018_05_GB_DET"
        )


def test_two_overtime_rows_for_one_game_are_refused():
    overtime = pl.DataFrame(
        {"game_id": ["2018_05_GB_DET"] * 2, "home_received": [True, False], "delta": [0.1, 0.2]}
    )
    game = verdict(went_to_overtime=True)
    with pytest.raises(AssertionError):
        sources_with(
            pl.DataFrame({"game_id": []}, schema={"game_id": pl.String}), overtime=overtime
        ).toss(game)


def test_band_sweep_refuses_a_duplicated_half_width():
    with pytest.raises(AssertionError):
        band_sweep([0.4, 0.6], [3.0, -3.0], half_widths=[0.10, 0.10])
