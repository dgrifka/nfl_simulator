"""The product layer's first figure: the bootstrap distribution.

Two things are under test and they are different in kind. The **verdict** is
arithmetic on committed numbers — which bucket a game lands in, which team the
headline names, whether the interval means anything — and it is tested the way
any arithmetic is. The **figure** is tested only for the marks a reader would
notice missing: the actual margin, the deserved margin, the zero line that
separates a home win from an away one, and the caveat that stops the interval
from being read as a plain 89%.

Nothing here re-derives DTW%. The simulator already owns that; this module
presents it.
"""

from __future__ import annotations

import re
from dataclasses import replace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib.offsetbox import AnnotationBbox  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from PIL import Image  # noqa: E402

from nfl_simulator.plots import (  # noqa: E402
    BAND_HIGH,
    BAND_LOW,
    CLEAR_FLIP,
    LEDGER_BOX_HEIGHT,
    LEDGER_BOXES_Y,
    OVERTIME_TITLE,
    REPORTED_NOT_NEUTRALIZED,
    SCOREBOARD_HOLDS,
    TOO_CLOSE,
    GameVerdict,
    OvertimeToss,
    attach_overtime_sidebar,
    band_sweep,
    bucket_label,
    luck_bars,
    margin_sentence,
    net_luck,
    overtime_lines,
    pill_colour,
    plain_label,
    plot_band_sweep,
    plot_bootstrap_distribution,
    plot_game_card,
    plot_luck_ledger,
    plot_luck_ledger_card,
    running_totals,
    table_rows,
    team_ledgers,
)
from nfl_simulator.style import (  # noqa: E402
    CLASH_DISTANCE,
    PALETTE,
    colour_distance,
    finalize,
)


def verdict(
    dtw_home: float = 0.75,
    actual_margin: float = 7.0,
    deserved_margin: float = 4.0,
    interval: tuple[float, float] = (0.70, 0.80),
    draws: np.ndarray | None = None,
    **kwargs,
) -> GameVerdict:
    defaults = {
        "game_id": "2025_17_DET_MIN",
        "home_team": "MIN",
        "away_team": "DET",
    }
    if draws is None:
        draws = np.linspace(deserved_margin - 10, deserved_margin + 10, 512)
    return GameVerdict(
        actual_margin=actual_margin,
        deserved_margin=deserved_margin,
        dtw_home=dtw_home,
        dtw_interval=interval,
        margin_draws=draws,
        **(defaults | kwargs),
    )


# --------------------------------------------------------------------------
# degeneracy — document 10 Gate V-3's convention, unchanged
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtw", [0.0, 0.001, 0.999, 1.0])
def test_a_verdict_at_or_past_the_threshold_is_degenerate(dtw):
    assert verdict(dtw_home=dtw).is_degenerate


@pytest.mark.parametrize("dtw", [0.002, 0.5, 0.998])
def test_a_verdict_inside_the_threshold_is_not_degenerate(dtw):
    assert not verdict(dtw_home=dtw).is_degenerate


def test_a_game_with_no_luck_events_is_a_point_mass():
    """No coin was flipped, so every draw is the game's own result."""
    assert verdict(draws=np.full(1, 7.0)).is_point_mass


def test_a_game_with_luck_events_is_not_a_point_mass():
    assert not verdict().is_point_mass


def test_a_degenerate_game_with_events_is_not_a_point_mass():
    """`2021_14_LV_KC`: 15 coins, none of which reach the other side of zero."""
    spread = np.linspace(20.0, 36.0, 512)
    assert verdict(dtw_home=1.0, draws=spread).is_degenerate
    assert not verdict(dtw_home=1.0, draws=spread).is_point_mass


# --------------------------------------------------------------------------
# the three buckets — document 33 §2a, band 0.40–0.60
# --------------------------------------------------------------------------


def test_dtw_inside_the_band_is_too_close_to_call():
    assert verdict(dtw_home=0.548, actual_margin=13.0).bucket == TOO_CLOSE


@pytest.mark.parametrize("dtw", [BAND_LOW, BAND_HIGH])
def test_the_band_edges_are_inside_the_band(dtw):
    assert verdict(dtw_home=dtw).bucket == TOO_CLOSE


def test_dtw_against_the_home_winner_and_outside_the_band_is_a_clear_flip():
    """`2018_05_GB_DET`: Detroit won by 8 at home, DTW% 0.054."""
    assert verdict(dtw_home=0.054, actual_margin=8.0).bucket == CLEAR_FLIP


def test_dtw_against_the_away_winner_and_outside_the_band_is_a_clear_flip():
    assert verdict(dtw_home=0.93, actual_margin=-8.0).bucket == CLEAR_FLIP


def test_dtw_with_the_scoreboard_winner_holds():
    assert verdict(dtw_home=0.88, actual_margin=10.0).bucket == SCOREBOARD_HOLDS


def test_an_away_win_the_bootstrap_agrees_with_holds():
    assert verdict(dtw_home=0.12, actual_margin=-10.0).bucket == SCOREBOARD_HOLDS


def test_a_actual_tie_outside_the_band_is_a_flip_because_the_scoreboard_named_nobody():
    tie = verdict(dtw_home=0.85, actual_margin=0.0)
    assert tie.scoreboard_winner is None
    assert tie.bucket == CLEAR_FLIP


# --------------------------------------------------------------------------
# who the headline names
# --------------------------------------------------------------------------


def test_the_deserved_winner_is_the_side_the_bootstrap_favours():
    assert verdict(dtw_home=0.054).deserved_winner == "DET"  # away side wins 94.6%


def test_the_deserved_winner_is_home_when_dtw_is_above_a_half():
    assert verdict(dtw_home=0.548).deserved_winner == "MIN"


def test_the_headline_names_the_favoured_team_first_with_both_shares():
    assert verdict(dtw_home=0.78).headline() == "MIN 78% / DET 22%"


def test_the_headline_rounds_rather_than_truncates():
    assert verdict(dtw_home=0.548).headline() == "MIN 55% / DET 45%"


# --------------------------------------------------------------------------
# the interval caveat — never a bare 89%
# --------------------------------------------------------------------------


def test_a_live_interval_quotes_both_the_nominal_and_the_measured_coverage():
    note = verdict(dtw_home=0.548, interval=(0.49, 0.599)).interval_note()
    assert "89%" in note
    assert "91.5%" in note


def test_a_degenerate_interval_says_it_collapsed_and_how_common_that_is():
    note = verdict(dtw_home=1.0, interval=(1.0, 1.0)).interval_note()
    assert "44.4%" in note
    assert "89%" not in note


# --------------------------------------------------------------------------
# the figure
# --------------------------------------------------------------------------


def _vline_positions(ax) -> list[float]:
    """x positions of the figure's vertical rules."""
    positions = []
    for line in ax.lines:
        xs = line.get_xdata()
        if len(xs) == 2 and xs[0] == xs[1]:
            positions.append(float(xs[0]))
    return positions


def test_the_figure_marks_the_actual_margin():
    fig, ax = plot_bootstrap_distribution(verdict(actual_margin=8.0, deserved_margin=-8.28))
    assert 8.0 in _vline_positions(ax)


def test_the_figure_marks_the_deserved_margin():
    fig, ax = plot_bootstrap_distribution(verdict(actual_margin=8.0, deserved_margin=-8.28))
    assert pytest.approx(-8.28) in _vline_positions(ax)


def test_the_figure_marks_the_zero_line_that_separates_the_two_winners():
    fig, ax = plot_bootstrap_distribution(verdict())
    assert 0.0 in _vline_positions(ax)


def test_the_figure_titles_carry_the_headline():
    fig, ax = plot_bootstrap_distribution(verdict(dtw_home=0.78))
    text = " ".join(t.get_text() for t in fig.findobj(matplotlib.text.Text))
    assert "DTW: MIN 78% • DET 22%" in text


def test_the_figure_carries_the_interval_caveat():
    fig, ax = plot_bootstrap_distribution(verdict(dtw_home=0.548, interval=(0.49, 0.599)))
    text = " ".join(t.get_text() for t in fig.findobj(matplotlib.text.Text))
    assert "91.5%" in text


def test_a_point_mass_game_is_drawn_as_a_note_rather_than_a_density():
    """One value has no density. Drawing a histogram of it would invent a shape."""
    fig, ax = plot_bootstrap_distribution(verdict(draws=np.full(1, 7.0), actual_margin=7.0))
    text = " ".join(t.get_text() for t in fig.findobj(matplotlib.text.Text))
    assert "no luck events" in text.lower()
    assert not ax.patches


def test_a_game_with_draws_is_drawn_as_a_density():
    fig, ax = plot_bootstrap_distribution(verdict())
    assert ax.patches


def test_the_shaded_home_win_region_is_the_share_the_bootstrap_reports():
    """The fill right of zero is the DTW% the headline claims, not a redrawing."""
    draws = np.array([-3.0] * 250 + [5.0] * 750)
    fig, ax = plot_bootstrap_distribution(verdict(dtw_home=0.75, draws=draws))
    shaded = [p for p in ax.patches if p.get_x() >= 0.0]
    assert shaded


def test_the_figure_legends_both_teams_so_the_fills_are_not_colour_alone():
    fig, ax = plot_bootstrap_distribution(verdict())
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("MIN" in label for label in labels)
    assert any("DET" in label for label in labels)


def test_a_point_mass_game_has_no_legend_because_it_has_no_fills():
    fig, ax = plot_bootstrap_distribution(verdict(draws=np.full(1, 7.0)))
    assert ax.get_legend() is None


def test_bins_are_one_point_of_margin_wide_and_aligned_to_zero():
    """Margins are lumpy — bins off the integer grid comb the distribution."""
    draws = np.concatenate([np.full(400, -8.3), np.full(600, 3.1)])
    fig, ax = plot_bootstrap_distribution(verdict(dtw_home=0.6, draws=draws))
    edges = sorted({round(p.get_x(), 6) for p in ax.patches})
    assert all(abs(edge - round(edge)) < 1e-9 for edge in edges)
    assert all(abs(p.get_width() - 1.0) < 1e-9 for p in ax.patches)


def test_the_interval_is_stated_for_the_team_the_headline_names():
    """A bare "49–60%" beside a headline naming the away side reads as theirs."""
    note = verdict(dtw_home=0.548, interval=(0.49, 0.599)).interval_note()
    assert "MIN's share runs 49–60%" in note


def test_the_interval_is_mirrored_when_the_away_side_is_favoured():
    note = verdict(dtw_home=0.054, interval=(0.036, 0.075)).interval_note()
    assert "DET's share runs 92–96%" in note


# --------------------------------------------------------------------------
# the figure's layout — measured, because a collision is invisible to an
# assertion on the text itself
# --------------------------------------------------------------------------


CALLOUT = re.compile(r"^(Actual|Deserved) [+-]")


def _callouts(ax) -> list:
    """The two boxed rule labels.

    Matched on the signed number, not on the word: the header's subtitle also
    opens with "Actual", and a looser match picked it up as a third rule."""
    return [text for text in ax.texts if CALLOUT.match(text.get_text())]


def _rule_label_boxes(fig, ax) -> list:
    """The two named rule labels' bounding boxes, in display pixels."""
    fig.canvas.draw()
    return [text.get_window_extent() for text in _callouts(ax)]


@pytest.mark.parametrize("gap", [0.0, 0.4, 1.0, 2.3])
def test_the_two_rule_labels_never_overprint_when_the_margins_are_close(gap):
    """`2025_13_DEN_WAS` printed "deserved -3.3" straight through "actual -1".

    Both labels hang off the top of their own rule, so a game whose two margins
    are within a label's width of each other stacks one on the other and neither
    can be read."""
    fig, ax = plot_bootstrap_distribution(
        verdict(actual_margin=-1.0, deserved_margin=-1.0 - gap, draws=np.linspace(-12, 6, 512))
    )
    boxes = _rule_label_boxes(fig, ax)
    assert len(boxes) == 2
    assert not boxes[0].overlaps(boxes[1])
    # A label moved out of the plot has somewhere else to be wrong: the subtitle
    # sits just above the top spine.
    subtitle = next(text for text in ax.texts if text.get_text().startswith("Actual: "))
    assert not any(box.overlaps(subtitle.get_window_extent()) for box in boxes)


@pytest.mark.parametrize(
    "deserved, actual",
    [(-8.28, 8.0), (27.93, 39.0), (0.70, 13.0)],
    ids=["2018_05_GB_DET", "2021_14_LV_KC", "2025_17_DET_MIN"],
)
def test_document_37_example_games_keep_their_rule_labels_clear(deserved, actual):
    """The three shipped examples are far apart and were never the defect. They
    are here so a fix for the close case cannot regress the common one."""
    span = max(abs(deserved), abs(actual)) + 8
    fig, ax = plot_bootstrap_distribution(
        verdict(
            actual_margin=actual,
            deserved_margin=deserved,
            draws=np.linspace(deserved - span / 2, deserved + span / 2, 512),
        )
    )
    boxes = _rule_label_boxes(fig, ax)
    assert not boxes[0].overlaps(boxes[1])


# --------------------------------------------------------------------------
# the luck ledger — arithmetic
# --------------------------------------------------------------------------


PPE = 0.8389495557652871  # v1.3's shipped points-per-EPA slope


def ledger_row(
    luck_epa: float,
    component: str = "fumble",
    event_class: str = "pass/live",
    charged_team: str = "DET",
    play_id: float = 100.0,
) -> dict:
    return {
        "play_id": play_id,
        "component": component,
        "event_class": event_class,
        "charged_team": charged_team,
        "luck_epa": luck_epa,
    }


def test_a_bar_is_the_points_neutralising_the_event_takes_off_the_margin():
    """Luck that favoured the home team comes off the home team's margin."""
    (bar,) = luck_bars([ledger_row(2.0)], points_per_epa=PPE)
    assert bar.points == pytest.approx(-2.0 * PPE)


def test_luck_against_the_home_team_moves_the_margin_the_other_way():
    (bar,) = luck_bars([ledger_row(-2.0)], points_per_epa=PPE)
    assert bar.points == pytest.approx(2.0 * PPE)


def test_the_bars_sum_to_the_gap_between_the_actual_and_deserved_margins():
    """The waterfall has to reconcile, or it is not the ledger it claims to be."""
    rows = [ledger_row(3.42, play_id=1.0), ledger_row(-0.80, play_id=2.0)]
    bars = luck_bars(rows, points_per_epa=PPE)
    total = sum(bar.points for bar in bars)
    assert total == pytest.approx((3.42 - 0.80) * -PPE)


def test_bars_are_ordered_biggest_mover_first():
    rows = [
        ledger_row(0.4, play_id=1.0),
        ledger_row(-3.0, play_id=2.0),
        ledger_row(1.5, play_id=3.0),
    ]
    sizes = [abs(bar.points) for bar in luck_bars(rows, points_per_epa=PPE, floor=0.0)]
    assert sizes == sorted(sizes, reverse=True)


def test_chronological_order_follows_the_play_clock_instead():
    rows = [
        ledger_row(0.4, play_id=3000.0),
        ledger_row(-3.0, play_id=100.0),
        ledger_row(1.5, play_id=900.0),
    ]
    bars = luck_bars(rows, points_per_epa=PPE, floor=0.0, chronological=True)
    assert [bar.play_id for bar in bars] == [100.0, 900.0, 3000.0]


def test_events_below_the_floor_fold_into_one_row_that_says_how_many():
    """A 0.03-point bar is an invisible sliver; four of them are four blank rows."""
    rows = [ledger_row(3.0, play_id=1.0)] + [
        ledger_row(0.03, component="extra_point", event_class="extra point", play_id=float(i))
        for i in range(2, 6)
    ]
    bars = luck_bars(rows, points_per_epa=PPE, floor=0.1)
    assert len(bars) == 2
    assert bars[-1].n_events == 4
    assert "4" in bars[-1].label


def test_the_folded_row_carries_their_exact_sum_so_nothing_is_dropped():
    rows = [ledger_row(3.0, play_id=1.0)] + [
        ledger_row(0.03, play_id=float(i)) for i in range(2, 6)
    ]
    bars = luck_bars(rows, points_per_epa=PPE, floor=0.1)
    assert sum(bar.points for bar in bars) == pytest.approx(3.12 * -PPE)


def test_the_folded_row_is_last_even_though_it_can_outweigh_a_real_one():
    rows = [ledger_row(0.4, play_id=1.0)] + [
        ledger_row(0.09, play_id=float(i)) for i in range(2, 8)
    ]
    bars = luck_bars(rows, points_per_epa=PPE, floor=0.1)
    assert bars[-1].n_events == 6


def test_a_floor_of_zero_shows_every_event():
    rows = [ledger_row(0.01, play_id=float(i)) for i in range(5)]
    assert len(luck_bars(rows, points_per_epa=PPE, floor=0.0)) == 5


def test_a_bar_is_labelled_in_the_plain_words_the_figure_prints():
    (bar,) = luck_bars(
        [ledger_row(2.0, component="field_goal", event_class="45-49 yd", charged_team="MIN")],
        points_per_epa=PPE,
    )
    assert bar.label == "MIN 45-49 yd field goal"


def test_an_extra_point_does_not_repeat_itself_in_its_label():
    (bar,) = luck_bars(
        [ledger_row(2.0, component="extra_point", event_class="extra point", charged_team="GB")],
        points_per_epa=PPE,
    )
    assert bar.label == "GB extra point"


def test_running_totals_start_at_the_actual_margin_and_land_on_the_deserved_one():
    rows = [ledger_row(3.42, play_id=1.0), ledger_row(-0.80, play_id=2.0)]
    bars = luck_bars(rows, points_per_epa=PPE)
    spans = running_totals(bars, start=8.0)
    assert spans[0][0] == pytest.approx(8.0)
    assert spans[-1][1] == pytest.approx(8.0 - (3.42 - 0.80) * PPE)


def test_each_step_begins_where_the_previous_one_ended():
    rows = [ledger_row(3.42, play_id=1.0), ledger_row(-0.80, play_id=2.0)]
    spans = running_totals(luck_bars(rows, points_per_epa=PPE), start=8.0)
    for before, after in zip(spans, spans[1:], strict=False):
        assert before[1] == pytest.approx(after[0])


# --------------------------------------------------------------------------
# the luck ledger — the figure
# --------------------------------------------------------------------------


def test_the_waterfall_refuses_a_ledger_that_does_not_reconcile():
    """A ledger from another game, or a different slope, would draw a lie."""
    rows = [ledger_row(3.42, play_id=1.0)]
    with pytest.raises(ValueError, match="reconcile"):
        plot_luck_ledger(
            verdict(actual_margin=8.0, deserved_margin=-8.28), rows, points_per_epa=PPE
        )


def test_the_waterfall_draws_one_bar_per_row_plus_the_two_anchors():
    rows = [ledger_row(3.42, play_id=1.0), ledger_row(-0.80, play_id=2.0)]
    gap = (3.42 - 0.80) * PPE
    fig, ax = plot_luck_ledger(
        verdict(actual_margin=8.0, deserved_margin=8.0 - gap), rows, points_per_epa=PPE
    )
    assert len(ax.patches) == 4


def test_the_waterfall_names_both_ends_with_their_margins():
    rows = [ledger_row(3.42, play_id=1.0)]
    fig, ax = plot_luck_ledger(
        verdict(actual_margin=8.0, deserved_margin=8.0 - 3.42 * PPE), rows, points_per_epa=PPE
    )
    labels = [label.get_text() for label in ax.get_yticklabels()]
    assert any("Actual" in label for label in labels)
    assert any("Deserved" in label for label in labels)


def test_the_waterfall_legends_both_directions_so_colour_is_never_alone():
    rows = [ledger_row(3.42, play_id=1.0), ledger_row(-0.80, play_id=2.0)]
    gap = (3.42 - 0.80) * PPE
    fig, ax = plot_luck_ledger(
        verdict(actual_margin=8.0, deserved_margin=8.0 - gap), rows, points_per_epa=PPE
    )
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("MIN" in label for label in labels)
    assert any("DET" in label for label in labels)


def test_a_game_with_no_luck_events_is_drawn_as_a_note_rather_than_a_waterfall():
    fig, ax = plot_luck_ledger(
        verdict(actual_margin=7.0, deserved_margin=7.0), [], points_per_epa=PPE
    )
    text = " ".join(t.get_text() for t in fig.findobj(matplotlib.text.Text))
    assert "no luck events" in text.lower()


def test_the_waterfall_carries_the_headline_and_the_bucket():
    rows = [ledger_row(3.42, play_id=1.0)]
    game = verdict(dtw_home=0.78, actual_margin=8.0, deserved_margin=8.0 - 3.42 * PPE)
    fig, ax = plot_luck_ledger(game, rows, points_per_epa=PPE)
    text = " ".join(t.get_text() for t in fig.findobj(matplotlib.text.Text))
    assert "DTW: MIN 78% • DET 22%" in text
    assert game.bucket in text


def test_the_waterfall_says_the_order_of_the_bars_does_not_change_the_total():
    """A waterfall looks sequential; this one is a sum, and readers assume wrong."""
    rows = [ledger_row(3.42, play_id=1.0)]
    fig, ax = plot_luck_ledger(
        verdict(actual_margin=8.0, deserved_margin=8.0 - 3.42 * PPE), rows, points_per_epa=PPE
    )
    text = " ".join(t.get_text() for t in fig.findobj(matplotlib.text.Text))
    assert "order" in text.lower()


def test_the_value_labels_stay_inside_the_frame():
    """The last bar of a lopsided game ends at the axis edge, and its label ran off it."""
    rows = [ledger_row(9.0, play_id=1.0), ledger_row(0.15, play_id=2.0)]
    gap = 9.15 * PPE
    fig, ax = plot_luck_ledger(
        verdict(actual_margin=8.0, deserved_margin=8.0 - gap), rows, points_per_epa=PPE
    )
    fig.canvas.draw()
    frame = ax.get_window_extent()
    # The bars' own value labels. The header and footers are anchored to the
    # frame and are meant to run its full width; they are not what ran off it.
    values = [text for text in ax.texts if re.fullmatch(r"[+-]\d+\.\d+", text.get_text())]
    assert values
    for text in values:
        box = text.get_window_extent()
        assert frame.x0 <= box.x0 and box.x1 <= frame.x1, f"{text.get_text()!r} runs off the frame"


def test_the_legend_names_only_the_directions_the_game_actually_has():
    """A blue key beside a figure with no blue bar sends the reader hunting for one."""
    rows = [ledger_row(3.42, play_id=1.0), ledger_row(0.80, play_id=2.0)]
    gap = 4.22 * PPE
    fig, ax = plot_luck_ledger(
        verdict(actual_margin=8.0, deserved_margin=8.0 - gap), rows, points_per_epa=PPE
    )
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["luck that helped MIN"]


# --------------------------------------------------------------------------
# the flip-band sweep — is 0.40–0.60 load-bearing?
# --------------------------------------------------------------------------

# A miniature league, chosen so each band width moves a known game between
# buckets: dtw_home, actual_margin.
SWEEP_GAMES = [
    (0.054, 8.0),  # clear flip at every band — the bootstrap is nowhere near 0.5
    (0.38, 6.0),  # a flip at the shipped band, too close once the band reaches 0.12
    (0.548, 13.0),  # too close at the shipped band, holds once the band is under 0.048
    (0.88, 10.0),  # holds at every band
    (0.85, 0.0),  # an actual tie the scoreboard never named a winner for
]


def sweep_inputs():
    return (
        np.array([dtw for dtw, _ in SWEEP_GAMES]),
        np.array([margin for _, margin in SWEEP_GAMES]),
    )


def test_the_sweep_labels_a_game_the_same_way_the_product_does():
    """The sweep and the headline must not be able to disagree at the shipped band."""
    for dtw, margin in SWEEP_GAMES:
        assert bucket_label(dtw, margin) == verdict(dtw_home=dtw, actual_margin=margin).bucket


def test_a_wider_band_swallows_a_game_the_shipped_band_calls_a_flip():
    assert bucket_label(0.38, 6.0) == CLEAR_FLIP
    assert bucket_label(0.38, 6.0, low=0.35, high=0.65) == TOO_CLOSE


def test_a_narrower_band_lets_a_too_close_game_back_out():
    assert bucket_label(0.548, 13.0) == TOO_CLOSE
    assert bucket_label(0.548, 13.0, low=0.47, high=0.53) == SCOREBOARD_HOLDS


def test_every_game_lands_in_exactly_one_bucket_at_every_width():
    dtw, margin = sweep_inputs()
    for row in band_sweep(dtw, margin):
        assert row.clear_flip + row.too_close + row.scoreboard_holds == len(SWEEP_GAMES)


def test_widening_the_band_can_only_add_to_too_close_to_call():
    dtw, margin = sweep_inputs()
    counts = [row.too_close for row in band_sweep(dtw, margin)]
    assert counts == sorted(counts)


def test_widening_the_band_can_only_take_away_from_the_two_decided_buckets():
    dtw, margin = sweep_inputs()
    rows = band_sweep(dtw, margin)
    assert [r.clear_flip for r in rows] == sorted((r.clear_flip for r in rows), reverse=True)
    assert [r.scoreboard_holds for r in rows] == sorted(
        (r.scoreboard_holds for r in rows), reverse=True
    )


def test_a_zero_width_band_calls_nothing_too_close():
    """The band's own null: with no band, the label is the binary DTW% flip."""
    dtw, margin = sweep_inputs()
    row = band_sweep(dtw, margin, half_widths=[0.0])[0]
    assert (row.low, row.high) == (0.5, 0.5)
    assert row.too_close == 0


def test_the_default_sweep_lands_exactly_on_the_shipped_band():
    """A robustness display whose grid misses the shipped choice cannot place it."""
    dtw, margin = sweep_inputs()
    shipped = [r for r in band_sweep(dtw, margin) if r.half_width == pytest.approx(0.10)]
    assert len(shipped) == 1
    assert (shipped[0].low, shipped[0].high) == (BAND_LOW, BAND_HIGH)


def test_the_sweep_spans_the_range_the_round_asked_for():
    dtw, margin = sweep_inputs()
    rows = band_sweep(dtw, margin)
    assert (rows[0].low, rows[0].high) == (0.5, 0.5)
    assert (rows[-1].low, rows[-1].high) == pytest.approx((0.35, 0.65))


def test_the_sweep_counts_the_ties_so_document_33s_count_can_be_recovered():
    """Document 33 excluded actual ties from its flip counts; the product labels
    them. Both readings have to be available from one row or they will diverge."""
    dtw, margin = sweep_inputs()
    row = band_sweep(dtw, margin, half_widths=[0.10])[0]
    assert row.ties_outside_band == 1
    assert row.clear_flip - row.ties_outside_band == 2


def test_the_sweep_refuses_inputs_that_are_not_the_same_games():
    with pytest.raises(ValueError, match="same length"):
        band_sweep(np.array([0.5, 0.6]), np.array([3.0]))


def test_the_sweep_figure_gives_each_bucket_its_own_panel():
    dtw, margin = sweep_inputs()
    fig, axes = plot_band_sweep(band_sweep(dtw, margin))
    assert len(axes) == 3
    titles = [ax.get_title(loc="left") for ax in axes]
    assert titles == [CLEAR_FLIP, TOO_CLOSE, SCOREBOARD_HOLDS]


def test_each_panel_scales_to_its_own_bucket_rather_than_to_the_largest():
    """`scoreboard holds` is an order of magnitude bigger than the other two; one
    shared axis would flatten the movement the figure exists to show."""
    dtw, margin = sweep_inputs()
    _fig, axes = plot_band_sweep(band_sweep(dtw, margin))
    assert axes[1].get_ylim() != axes[2].get_ylim()


def test_every_panel_marks_the_shipped_band():
    dtw, margin = sweep_inputs()
    _fig, axes = plot_band_sweep(band_sweep(dtw, margin))
    for ax in axes:
        rules = [line for line in ax.lines if len(set(line.get_xdata())) == 1]
        assert any(x == pytest.approx(0.10) for line in rules for x in line.get_xdata())


def test_the_shipped_counts_are_written_on_the_figure():
    """A reader comparing this to the headline needs the number, not a pixel."""
    dtw, margin = sweep_inputs()
    rows = band_sweep(dtw, margin)
    shipped = next(r for r in rows if r.half_width == pytest.approx(0.10))
    _fig, axes = plot_band_sweep(rows)
    written = [text.get_text() for ax in axes for text in ax.texts]
    for count in (shipped.clear_flip, shipped.too_close, shipped.scoreboard_holds):
        assert any(str(count) in text for text in written)


def test_the_shipped_label_sits_clear_of_the_line_it_annotates():
    """`too close to call` rises across the sweep, so a label above its point is a
    label the line climbs through."""
    dtw, margin = sweep_inputs()
    _fig, axes = plot_band_sweep(band_sweep(dtw, margin))
    falling, rising = axes[0].texts[0], axes[1].texts[0]
    assert falling.get_va() == "bottom"
    assert rising.get_va() == "top"


def test_a_single_series_panel_carries_no_legend_box():
    dtw, margin = sweep_inputs()
    _fig, axes = plot_band_sweep(band_sweep(dtw, margin))
    assert all(ax.get_legend() is None for ax in axes)


def test_the_figure_says_what_is_being_swept():
    dtw, margin = sweep_inputs()
    fig, axes = plot_band_sweep(band_sweep(dtw, margin))
    labels = " ".join(ax.get_xlabel() for ax in axes).lower()
    assert "band" in labels
    caption = " ".join(text.get_text() for text in fig.texts).lower()
    assert "presentation" in caption


# --------------------------------------------------------------------------
# the overtime toss — reported, not neutralized (document 16)
# --------------------------------------------------------------------------


def toss(**kwargs) -> OvertimeToss:
    defaults = {"received": "MIN", "season": 2019}
    return OvertimeToss(**(defaults | kwargs))


def _joined(verdict_, toss_) -> str:
    return " ".join(overtime_lines(verdict_, toss_))


def test_the_sidebar_is_labelled_reported_not_neutralized():
    """The label is the whole point: this is a luck event the ledger declines."""
    assert REPORTED_NOT_NEUTRALIZED in OVERTIME_TITLE


def test_the_sidebar_names_the_team_that_took_the_first_possession():
    assert "MIN" in _joined(verdict(), toss(received="MIN"))


def test_the_sidebar_says_first_possession_rather_than_won_the_toss():
    """Document 16 §6's most serious open defect: nflverse has no coin-toss
    field, so first possession is a proxy and has to be labelled as one."""
    text = _joined(verdict(), toss())
    assert "first overtime possession" in text
    assert "no coin-toss field" in text


def test_the_swing_is_never_quoted_without_its_interval():
    text = _joined(verdict(), toss())
    assert "2.05" in text
    assert "1.04" in text and "3.07" in text


def test_the_sidebar_says_why_the_component_was_left_out():
    """A caveat that does not say it was measured and refused reads as an
    oversight rather than as document 16's decision."""
    text = _joined(verdict(), toss())
    assert "3.93" in text and "4.06" in text
    assert "14 of 155" in text


def test_the_sidebar_states_that_the_swing_is_a_league_average():
    """Document 16 §6 registers this defect as `stated wherever the component is
    reported`, which makes it a requirement on this panel and not a nicety."""
    assert "league average" in _joined(verdict(), toss())


def test_a_per_game_move_is_quoted_on_the_share_the_headline_names():
    text = _joined(verdict(dtw_home=0.63), toss(delta_dtw_home=-0.214))
    assert "MIN" in text
    assert "-21 pp" in text


def test_a_per_game_move_is_mirrored_when_the_headline_names_the_away_team():
    """The stored delta is on the home share. Printing it beside a headline that
    names the away team would hand one team's movement to the other."""
    text = _joined(verdict(dtw_home=0.30), toss(delta_dtw_home=0.12))
    assert "-12 pp" in text
    assert "+12 pp" not in text


def test_a_per_game_move_carries_the_simulator_version_it_was_measured_on():
    """Document 16's impact run was simulator v1.1; the printed share is v1.3.
    Without the version the two numbers look subtractable, and they are not."""
    assert "v1.1" in _joined(verdict(), toss(delta_dtw_home=-0.214))


def test_no_per_game_move_is_invented_when_none_was_supplied():
    assert "v1.1" not in _joined(verdict(), toss())


def test_a_new_rules_game_says_the_rulebook_cannot_be_separated():
    """Document 16 §4d pre-registered that no era split may be read as a finding."""
    text = _joined(verdict(), toss(season=2025))
    assert "0.243" in text
    assert "60" in text


def test_a_game_under_the_old_rules_does_not_raise_the_rulebook_question():
    assert "0.243" not in _joined(verdict(), toss(season=2019))


def test_a_game_that_did_not_go_to_overtime_gets_no_sidebar():
    """Silence is the honest annotation for a mechanism that never happened."""
    fig, ax = plot_bootstrap_distribution(verdict())
    assert attach_overtime_sidebar(fig, ax, verdict(), None) is None


def test_attaching_a_sidebar_does_not_shrink_the_figure_it_annotates():
    """The figure grows to the right. A panel that squeezed the plot would
    re-scale a distribution in order to say something beside it."""
    fig, ax = plot_bootstrap_distribution(verdict())
    before = ax.get_position().width * fig.get_size_inches()[0]
    attach_overtime_sidebar(fig, ax, verdict(), toss())
    after = ax.get_position().width * fig.get_size_inches()[0]
    assert after == pytest.approx(before, abs=1e-6)


def test_the_sidebar_panel_draws_no_axes_of_its_own():
    fig, ax = plot_bootstrap_distribution(verdict())
    panel = attach_overtime_sidebar(fig, ax, verdict(), toss())
    assert not panel.axison


def test_the_sidebar_wears_no_team_colour():
    """A caveat is not an entity. Colour here would read as a third side."""
    fig, ax = plot_bootstrap_distribution(verdict())
    panel = attach_overtime_sidebar(fig, ax, verdict(), toss())
    assert {text.get_color() for text in panel.texts} <= {PALETTE["text"], PALETTE["text_muted"]}


def test_the_sidebar_attaches_to_the_waterfall_as_well_as_the_distribution():
    rows = [ledger_row(3.0, play_id=1.0)]
    game = verdict(actual_margin=7.0, deserved_margin=7.0 - 3.0 * PPE)
    fig, ax = plot_luck_ledger(game, rows, points_per_epa=PPE)
    panel = attach_overtime_sidebar(fig, ax, game, toss())
    assert any(REPORTED_NOT_NEUTRALIZED in text.get_text() for text in panel.texts)


def test_the_interval_caveat_stays_clear_of_the_sidebar_panel():
    """`2025_13_DEN_WAS`: six paragraphs of panel, and the footnote ran under them.

    The caveat wrapped to the figure's width, so widening the figure for a
    sidebar widened the caveat with it and the two overprinted. The panel's own
    text overflows the panel axes on a long note, so the test measures both."""
    game = verdict()
    fig, ax = plot_bootstrap_distribution(game)
    panel = attach_overtime_sidebar(fig, ax, game, toss(season=2025, delta_dtw_home=0.136))
    fig.canvas.draw()

    caveats = [text for text in ax.texts if "89% interval" in text.get_text()]
    assert len(caveats) == 1
    occupied = matplotlib.transforms.Bbox.union(
        [panel.get_window_extent()] + [text.get_window_extent() for text in panel.texts]
    )
    assert not caveats[0].get_window_extent().overlaps(occupied)


def test_a_five_paragraph_sidebar_also_clears_the_interval_caveat():
    """`2016_14_NYJ_SF`: an old-rules game, one paragraph shorter, already clear."""
    game = verdict(dtw_home=0.36)
    fig, ax = plot_bootstrap_distribution(game)
    panel = attach_overtime_sidebar(fig, ax, game, toss(season=2016, delta_dtw_home=-0.214))
    fig.canvas.draw()

    caveat = next(text for text in ax.texts if "89% interval" in text.get_text())
    occupied = matplotlib.transforms.Bbox.union(
        [panel.get_window_extent()] + [text.get_window_extent() for text in panel.texts]
    )
    assert not caveat.get_window_extent().overlaps(occupied)


# --------------------------------------------------------------------------
# the shared header grammar — the brand round, 2026-08-26
# --------------------------------------------------------------------------


def branded(**kwargs) -> GameVerdict:
    """A verdict carrying the scoreboard facts a header needs."""
    defaults = {
        "game_id": "2018_05_GB_DET",
        "home_team": "DET",
        "away_team": "GB",
        "home_score": 31,
        "away_score": 23,
        "game_date": "2018-10-07",
        "dtw_home": 0.05,
        "actual_margin": 8.0,
        "deserved_margin": -8.28,
    }
    return verdict(**(defaults | kwargs))


def figure_text(fig) -> str:
    return " ".join(t.get_text() for t in fig.findobj(matplotlib.text.Text))


def test_the_subtitle_states_the_scoreboard_the_date_and_both_shares():
    line = branded().subtitle_line()
    assert "Actual: GB 23 - DET 31" in line
    assert "(10/07/2018)" in line
    assert "DTW: DET 5% • GB 95%" in line or "DTW: GB 95% • DET 5%" in line


def test_the_subtitle_falls_back_to_the_margin_when_no_score_is_known():
    """A verdict built from the summary alone still has a header to fill."""
    line = verdict(home_team="MIN", away_team="DET", actual_margin=13.0).subtitle_line()
    assert "MIN" in line and "13" in line
    assert "None" not in line


def test_the_verdict_pill_is_red_on_a_flip_and_green_when_the_scoreboard_holds():
    assert pill_colour(CLEAR_FLIP) == PALETTE["bad"]
    assert pill_colour(SCOREBOARD_HOLDS) == PALETTE["good"]
    assert pill_colour(TOO_CLOSE) == PALETTE["text_muted"]


def test_the_verdict_pill_is_drawn_as_a_filled_shape_not_bare_text():
    fig, ax = plot_bootstrap_distribution(branded())
    pill = next(t for t in ax.texts if t.get_text() == CLEAR_FLIP)
    assert pill.get_bbox_patch() is not None


@pytest.mark.parametrize("bucket_dtw", [0.05, 0.55, 0.86], ids=["flip", "too-close", "holds"])
def test_the_pill_clears_the_heading_and_the_subtitle_it_shares_a_row_with(bucket_dtw):
    """`LV_KC_9-48--0-100_dtw.png`: "scoreboard holds" printed through the heading.

    Drawn at the real draw count, so the heading is the full-width
    "Deserve-to-Win - 160,000 simulations" the shipped settings produce rather
    than a short one a fixture happened to make."""
    game = branded(dtw_home=bucket_dtw, draws=np.linspace(-20, 6, 160_000))
    fig, ax = plot_bootstrap_distribution(game)
    fig.canvas.draw()
    pill = next(t for t in ax.texts if t.get_text() == game.bucket)
    heading = next(t for t in ax.texts if t.get_text().startswith("Deserve-to-Win"))
    subtitle = next(t for t in ax.texts if t.get_text().startswith("Actual: "))
    box = pill.get_window_extent()
    assert not box.overlaps(heading.get_window_extent()), "the pill runs into the heading"
    assert not box.overlaps(subtitle.get_window_extent()), "the pill runs into the subtitle"


def test_the_pill_stays_clear_of_the_corner_the_watermark_is_stamped_into(tmp_path):
    """The credit is stamped on the saved pixels, so the pill has to leave it room."""
    game = branded(draws=np.linspace(-20, 6, 160_000))
    fig, ax = plot_bootstrap_distribution(game)
    path = finalize(fig, tmp_path / "corner.png")
    pixels = np.asarray(Image.open(path).convert("RGB"), dtype=float)
    height, width = pixels.shape[:2]
    corner = pixels[: int(height * 0.055), int(width * 0.86) :]
    # The pill is a saturated fill; the credit is mid-grey on cream. Nothing in
    # the corner may be as dark or as coloured as a pill.
    spread = corner.max(axis=2) - corner.min(axis=2)
    assert spread.max() < 40, "something coloured is in the watermark's corner"


def test_a_degenerate_game_legends_only_the_side_that_has_bars():
    """`LV_KC`: KC wins every re-flip, and a key for an absent colour sends a
    reader hunting the figure for it."""
    fig, ax = plot_bootstrap_distribution(branded(dtw_home=1.0, draws=np.linspace(15, 40, 512)))
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["DET wins"]


def test_the_distribution_heading_counts_the_simulations_it_actually_drew():
    fig, ax = plot_bootstrap_distribution(branded(draws=np.linspace(-12, 6, 1000)))
    assert "1,000 simulations" in figure_text(fig)


def test_the_word_re_flips_is_gone_from_the_distribution():
    """Round 1's review: "re-flips" is the simulator's word, not a reader's."""
    fig, ax = plot_bootstrap_distribution(branded(draws=np.linspace(-12, 6, 1000)))
    assert "re-flips" not in figure_text(fig)


# --------------------------------------------------------------------------
# the rule callouts
# --------------------------------------------------------------------------


def test_the_rule_callouts_are_boxed_so_they_read_over_the_bars():
    fig, ax = plot_bootstrap_distribution(branded())
    callouts = _callouts(ax)
    assert len(callouts) == 2
    assert all(t.get_bbox_patch() is not None for t in callouts)


def test_the_callouts_name_the_two_margins_in_derek_s_wording():
    fig, ax = plot_bootstrap_distribution(branded())
    text = figure_text(fig)
    assert "Actual +8" in text
    assert "Deserved -8.3" in text


def test_the_bars_wear_the_team_colours_they_are_handed():
    fig, ax = plot_bootstrap_distribution(branded(), colors=("#0076B6", "#203731"))
    faces = {p.get_facecolor()[:3] for p in ax.patches}
    assert matplotlib.colors.to_rgb("#0076B6") in faces
    assert matplotlib.colors.to_rgb("#203731") in faces


# --------------------------------------------------------------------------
# plain-word event labels
# --------------------------------------------------------------------------


def test_a_made_field_goal_reads_as_a_sentence_with_its_distance():
    row = ledger_row(-1.0, component="field_goal", event_class="40-44 yd", charged_team="GB")
    label = plain_label(row | {"actual": 1.0, "kick_distance": 42.0})
    assert label == "GB 42-yd field goal, made"


def test_a_field_goal_without_a_distance_falls_back_to_its_class():
    row = ledger_row(3.7, component="field_goal", event_class="40-44 yd", charged_team="GB")
    assert plain_label(row | {"actual": 0.0}) == "GB 40-44 yd field goal, missed"


def test_a_missed_extra_point_does_not_repeat_the_word_extra_point():
    row = ledger_row(0.95, component="extra_point", event_class="extra point", charged_team="GB")
    assert plain_label(row | {"actual": 0.0}) == "GB extra point, missed"


def test_a_lost_fumble_names_who_recovered_it():
    row = ledger_row(2.17, component="fumble", event_class="pass/live", charged_team="DET")
    label = plain_label(row | {"actual": 0.0, "opponent": "GB"})
    assert label == "DET fumble on a pass, recovered by GB"


def test_a_fumble_the_fumbling_team_recovered_reads_as_retained():
    """ "DET fumble, recovered by DET" says the same thing twice."""
    row = ledger_row(-1.0, component="fumble", event_class="run/live", charged_team="DET")
    assert plain_label(row | {"actual": 1.0, "opponent": "GB"}) == "DET fumble on a run, retained"


def test_an_aborted_snap_says_so_rather_than_calling_itself_a_run():
    row = ledger_row(2.7, component="fumble", event_class="run/aborted", charged_team="DET")
    label = plain_label(row | {"actual": 0.0, "opponent": "GB"})
    assert label == "DET fumble on an aborted run, recovered by GB"


def test_a_label_with_no_branch_recorded_still_names_the_event():
    """Never invent an outcome: a row without its branch gets no outcome clause."""
    row = ledger_row(2.17, component="fumble", event_class="punt/live", charged_team="DET")
    assert plain_label(row) == "DET fumble on a punt"


# --------------------------------------------------------------------------
# the waterfall, restyled
# --------------------------------------------------------------------------


HOW_TO_READ = (
    "Start at the actual margin. Each bar is one luck event re-priced at its "
    "expectation. The last bar is the margin the game deserved."
)


def waterfall(game=None, rows=None, **kwargs):
    """A waterfall whose single bar reconciles with whatever game it is handed."""
    game = game or branded()
    if rows is None:
        luck_epa = (game.actual_margin - game.deserved_margin) / PPE
        rows = [ledger_row(luck_epa, play_id=1.0)]
    return plot_luck_ledger(game, rows, points_per_epa=PPE, **kwargs)


def test_the_waterfall_says_how_to_read_itself():
    fig, ax = waterfall()
    assert HOW_TO_READ in figure_text(fig).replace("\n", " ")


def test_the_verdict_pill_never_comes_down_onto_the_how_to_read_caption():
    """`DET_MIN_10-23--45-55_luck_ledger.png`: the pill's box sat on the caption's
    last words, so the caption is wrapped to the room the pill leaves."""
    fig, ax = waterfall(branded(dtw_home=0.55))
    fig.canvas.draw()
    pill = next(t for t in ax.texts if t.get_text() == TOO_CLOSE)
    caption = next(t for t in ax.texts if t.get_text().startswith("Start at the actual"))
    assert not pill.get_window_extent().overlaps(caption.get_window_extent())


def test_the_waterfall_end_bars_are_a_neutral_rather_than_a_team_colour():
    """The two ends are totals, not luck, so they never wear a side's colour."""
    fig, ax = waterfall(colors=("#0076B6", "#203731"))
    ends = [p.get_facecolor()[:3] for p in ax.patches[:1]] + [
        p.get_facecolor()[:3] for p in ax.patches[-1:]
    ]
    assert set(ends) == {matplotlib.colors.to_rgb(PALETTE["anchor"])}


def test_a_team_whose_primary_is_black_is_still_told_apart_from_the_totals():
    """`LV_KC_9-48--0-100_luck_ledger.png`: the Raiders' #000000 event bars and
    the waterfall's totals were the same colour, and the figure could not say
    which bar was a total."""
    fig, ax = waterfall(colors=("#E31837", "#000000"))
    faces = {matplotlib.colors.to_hex(p.get_facecolor()) for p in ax.patches}
    anchor = matplotlib.colors.to_hex(PALETTE["anchor"])
    assert anchor in faces
    assert all(colour_distance(colour, anchor) >= CLASH_DISTANCE for colour in faces - {anchor})


def test_the_waterfall_names_its_ends_in_derek_s_wording():
    fig, ax = waterfall()
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels[0].startswith("Actual")
    assert labels[-1].startswith("Deserved")


def test_an_overtime_game_says_the_toss_is_reported_not_neutralized():
    fig, ax = waterfall(branded(went_to_overtime=True))
    assert "Went to overtime; the coin toss is reported, not neutralized." in figure_text(fig)


def test_a_regulation_game_carries_no_overtime_footer():
    fig, ax = waterfall(branded(went_to_overtime=False))
    assert "Went to overtime" not in figure_text(fig)


# --------------------------------------------------------------------------
# the share card
# --------------------------------------------------------------------------


def test_the_card_is_square():
    fig = plot_game_card(branded())[0]
    width, height = fig.get_size_inches()
    assert width == pytest.approx(height)


def test_the_card_carries_the_score_the_verdict_and_the_deserved_margin():
    fig, _ax = plot_game_card(branded())
    text = figure_text(fig)
    assert "23" in text and "31" in text
    assert CLEAR_FLIP in text
    assert "Deserved margin: GB by 8.3" in text


def test_the_card_prints_both_shares_at_the_ends_of_one_bar():
    fig, _ax = plot_game_card(branded())
    text = figure_text(fig)
    assert "95%" in text and "5%" in text


def test_a_degenerate_card_says_the_bootstrap_never_changed_its_mind():
    fig, _ax = plot_game_card(branded(dtw_home=1.0, interval=(1.0, 1.0)))
    assert "every re-flip" in figure_text(fig).lower()


def test_the_card_draws_no_axes_because_it_is_not_a_plot():
    fig, ax = plot_game_card(branded())
    assert not ax.axison


# --------------------------------------------------------------------------
# the distribution's variant options — figure workshop round 2, Part A
# --------------------------------------------------------------------------

DET_BLUE, GB_GREEN = "#0076B6", "#203731"


def synthetic_mark(colour=(20, 40, 200)) -> np.ndarray:
    """A 4x6 fully opaque RGBA block standing in for a club's mark."""
    mark = np.zeros((4, 6, 4), dtype=np.uint8)
    mark[:, :, :3] = colour
    mark[:, :, 3] = 255
    return mark


VARIANTS = [
    {"bin_width": 1.0, "callout": True},
    {"bin_width": 3.0, "callout": True},
    {"bin_width": 3.0, "callout": True, "arrow": True},
    {"bin_width": 3.0, "callout": True, "arrow": True, "legend_logos": True},
]


def test_a_wider_bin_draws_fewer_bars_and_every_bar_is_that_wide():
    game = branded(draws=np.linspace(-20.0, 6.0, 4000))
    narrow = plot_bootstrap_distribution(game, bin_width=1.0)[1]
    wide = plot_bootstrap_distribution(game, bin_width=3.0)[1]
    assert len(wide.patches) < len(narrow.patches)
    assert all(abs(patch.get_width() - 1.0) < 1e-9 for patch in narrow.patches)
    assert all(abs(patch.get_width() - 3.0) < 1e-9 for patch in wide.patches)


def test_a_wider_bin_still_puts_an_edge_on_zero():
    """Zero decides the winner, so no bar may straddle it at any width."""
    game = branded(draws=np.linspace(-20.0, 6.0, 4000))
    _fig, ax = plot_bootstrap_distribution(game, bin_width=3.0)
    assert all(abs(patch.get_x() % 3.0) < 1e-9 for patch in ax.patches)


def test_the_y_axis_is_a_percentage_of_the_simulations():
    fig, ax = plot_bootstrap_distribution(branded())
    fig.canvas.draw()
    assert ax.get_ylabel() == "% of simulations"
    labels = [tick.get_text() for tick in ax.get_yticklabels() if tick.get_text()]
    assert labels, "the y axis is worth reading now, so it carries ticks"
    assert all(label.endswith("%") for label in labels)


def test_the_bar_heights_are_the_share_of_simulations_in_each_bin():
    """Percent, not density: over a one-point grid the bars sum to 100."""
    _fig, ax = plot_bootstrap_distribution(branded(draws=np.linspace(-20.0, 6.0, 4000)))
    assert sum(patch.get_height() for patch in ax.patches) == pytest.approx(100.0)


def test_the_axis_label_names_the_two_teams_the_margin_runs_between():
    _fig, ax = plot_bootstrap_distribution(branded())
    assert ax.get_xlabel() == "final margin (DET \u2212 GB)"


# --- the callout ----------------------------------------------------------


def test_the_callout_says_who_deserved_to_win_and_how_often():
    _fig, ax = plot_bootstrap_distribution(branded(), colors=(DET_BLUE, GB_GREEN), callout=True)
    hits = [text for text in ax.texts if "deserved to win" in text.get_text()]
    assert len(hits) == 1
    assert hits[0].get_text() == "GB deserved to win 95% of simulations"


def test_the_callout_is_written_in_the_favoured_team_s_own_colour():
    _fig, ax = plot_bootstrap_distribution(branded(), colors=(DET_BLUE, GB_GREEN), callout=True)
    hit = next(text for text in ax.texts if "deserved to win" in text.get_text())
    assert matplotlib.colors.to_hex(hit.get_color()) == GB_GREEN.lower()


def test_the_callout_follows_the_favoured_side_when_it_is_the_home_team():
    _fig, ax = plot_bootstrap_distribution(
        branded(dtw_home=0.86), colors=(DET_BLUE, GB_GREEN), callout=True
    )
    hit = next(text for text in ax.texts if "deserved to win" in text.get_text())
    assert hit.get_text() == "DET deserved to win 86% of simulations"
    assert matplotlib.colors.to_hex(hit.get_color()) == DET_BLUE.lower()


@pytest.mark.parametrize("dtw", [0.05, 0.86], ids=["away-favoured", "home-favoured"])
def test_the_callout_never_prints_through_a_rule_label(dtw):
    fig, ax = plot_bootstrap_distribution(branded(dtw_home=dtw), callout=True)
    fig.canvas.draw()
    hit = next(text for text in ax.texts if "deserved to win" in text.get_text())
    boxes = _rule_label_boxes(fig, ax)
    assert boxes
    assert not any(box.overlaps(hit.get_window_extent()) for box in boxes)


def test_there_is_no_callout_unless_the_figure_asks_for_one():
    _fig, ax = plot_bootstrap_distribution(branded())
    assert not [text for text in ax.texts if "deserved to win" in text.get_text()]


# --- the arrow ------------------------------------------------------------


def _spans(ax) -> list:
    return [
        text for text in ax.texts if isinstance(getattr(text, "arrow_patch", None), FancyArrowPatch)
    ]


def test_the_arrow_runs_from_the_actual_margin_to_the_deserved_one():
    _fig, ax = plot_bootstrap_distribution(branded(), arrow=True)
    spans = _spans(ax)
    assert len(spans) == 1
    assert spans[0].xyann[0] == pytest.approx(8.0)
    assert spans[0].xy[0] == pytest.approx(-8.28)


def test_the_arrow_says_how_far_luck_moved_the_margin_and_toward_whom():
    _fig, ax = plot_bootstrap_distribution(branded(), arrow=True)
    label = next(t for t in ax.texts if t.get_text().startswith("luck moved the margin"))
    assert label.get_text() == "luck moved the margin 16.3 points toward DET"


def test_the_arrow_names_the_other_team_when_luck_ran_the_other_way():
    game = branded(actual_margin=-3.0, deserved_margin=6.0, draws=np.linspace(-10.0, 14.0, 512))
    _fig, ax = plot_bootstrap_distribution(game, arrow=True)
    label = next(t for t in ax.texts if t.get_text().startswith("luck moved the margin"))
    assert label.get_text() == "luck moved the margin 9.0 points toward GB"


def test_the_arrow_is_drawn_in_ink_rather_than_in_either_team_s_colour():
    _fig, ax = plot_bootstrap_distribution(branded(), colors=(DET_BLUE, GB_GREEN), arrow=True)
    label = next(t for t in ax.texts if t.get_text().startswith("luck moved the margin"))
    assert matplotlib.colors.to_hex(label.get_color()) == PALETTE["text_muted"].lower()


def test_a_degenerate_game_is_drawn_without_an_arrow():
    """The bootstrap never changed its mind, so there is nothing to measure."""
    game = branded(dtw_home=1.0, draws=np.linspace(15.0, 40.0, 512))
    _fig, ax = plot_bootstrap_distribution(game, arrow=True)
    assert not _spans(ax)
    assert not [t for t in ax.texts if t.get_text().startswith("luck moved")]


def test_there_is_no_arrow_unless_the_figure_asks_for_one():
    _fig, ax = plot_bootstrap_distribution(branded())
    assert not _spans(ax)


# --- the logo legend ------------------------------------------------------


def test_the_legend_can_be_the_two_clubs_marks_instead_of_two_swatches():
    logos = {"GB": synthetic_mark(), "DET": synthetic_mark((200, 30, 30))}
    _fig, ax = plot_bootstrap_distribution(branded(), logos=logos, legend_logos=True)
    assert ax.get_legend() is None
    marks = [artist for artist in ax.artists if isinstance(artist, AnnotationBbox)]
    assert len(marks) == 2
    assert {"GB wins", "DET wins"} <= {text.get_text() for text in ax.texts}


def test_the_swatch_legend_is_what_a_figure_gets_by_default():
    _fig, ax = plot_bootstrap_distribution(branded())
    assert ax.get_legend() is not None


def test_a_club_without_a_mark_still_gets_its_name_in_the_logo_legend():
    _fig, ax = plot_bootstrap_distribution(
        branded(), logos={"DET": synthetic_mark()}, legend_logos=True
    )
    assert {"GB wins", "DET wins"} <= {text.get_text() for text in ax.texts}
    assert len([a for a in ax.artists if isinstance(a, AnnotationBbox)]) == 1


# --- every variant, on the awkward games ----------------------------------


@pytest.mark.parametrize("options", VARIANTS, ids=["V1", "V2", "V3", "V4"])
def test_every_variant_draws_a_degenerate_game_without_raising(options):
    game = branded(dtw_home=1.0, draws=np.linspace(15.0, 40.0, 512))
    _fig, ax = plot_bootstrap_distribution(game, logos={"DET": synthetic_mark()}, **options)
    assert ax.patches


@pytest.mark.parametrize("options", VARIANTS, ids=["V1", "V2", "V3", "V4"])
def test_every_variant_draws_a_point_mass_game_without_raising(options):
    game = branded(dtw_home=1.0, actual_margin=8.0, deserved_margin=8.0, draws=np.full(1, 8.0))
    _fig, ax = plot_bootstrap_distribution(game, logos={"DET": synthetic_mark()}, **options)
    assert "no luck events" in " ".join(text.get_text() for text in ax.texts).lower()


def test_the_arrow_and_its_label_stay_clear_of_everything_else_up_there():
    """Three annotations share the top of the plot; none may print through another."""
    fig, ax = plot_bootstrap_distribution(branded(), arrow=True, callout=True)
    fig.canvas.draw()
    label = next(t for t in ax.texts if t.get_text().startswith("luck moved the margin"))
    callout = next(t for t in ax.texts if "deserved to win" in t.get_text())
    others = _rule_label_boxes(fig, ax) + [callout.get_window_extent()]
    assert not any(box.overlaps(label.get_window_extent()) for box in others)


def test_the_annotated_figure_reserves_a_clear_band_above_its_tallest_bar():
    """The callout is placed by rule, so the room it needs is made, not found.

    `59_2018_05_GB_DET_V1.png` printed "GB deserved to win 95% of simulations"
    straight across the three tallest bars, because the histogram filled the
    frame and the corner the callout is put in was not empty."""
    game = branded(draws=np.linspace(-20.0, 6.0, 4000))
    fig, ax = plot_bootstrap_distribution(game, callout=True, arrow=True)
    fig.canvas.draw()
    tallest = max(patch.get_height() for patch in ax.patches)
    callout = next(t for t in ax.texts if "deserved to win" in t.get_text())
    floor = ax.transData.inverted().transform((0, callout.get_window_extent().y0))[1]
    assert floor > tallest, "the callout sits on a bar"


# --------------------------------------------------------------------------
# the luck ledger card — figure workshop round 2, Part B
# --------------------------------------------------------------------------


def card_row(
    luck_epa: float,
    *,
    component: str = "field_goal",
    event_class: str = "40-44 yd",
    charged_team: str = "GB",
    play_id: float = 1.0,
    actual: float = 0.0,
    opponent: str = "DET",
    kick_distance: float | None = None,
) -> dict:
    """A committed ledger row with the three keys `render.prepare_rows` adds."""
    return {
        "play_id": play_id,
        "component": component,
        "event_class": event_class,
        "charged_team": charged_team,
        "luck_epa": luck_epa,
        "actual": actual,
        "opponent": opponent,
        "kick_distance": kick_distance,
    }


def gb_det_rows() -> list[dict]:
    """Seven events: five charged to GB, two to DET, none of them tiny."""
    return [
        card_row(3.8, play_id=1.0, kick_distance=41.0),
        card_row(3.7, play_id=2.0, kick_distance=42.0),
        card_row(3.6, play_id=3.0, kick_distance=38.0),
        card_row(3.4, play_id=4.0, component="fumble", event_class="punt/live"),
        card_row(2.9, play_id=5.0, component="fumble", event_class="pass/live"),
        card_row(-3.1, play_id=6.0, charged_team="DET", opponent="GB", kick_distance=55.0),
        card_row(
            -0.6, play_id=7.0, charged_team="DET", opponent="GB", kick_distance=39.0, actual=1.0
        ),
    ]


def reconciling(rows, game=None) -> GameVerdict:
    """The verdict those rows actually add up to."""
    game = game or branded()
    return replace(
        game, deserved_margin=game.actual_margin - sum(r["luck_epa"] for r in rows) * PPE
    )


def ledger_card(rows=None, game=None, **kwargs):
    rows = gb_det_rows() if rows is None else rows
    return plot_luck_ledger_card(reconciling(rows, game), rows, points_per_epa=PPE, **kwargs)


# --- the arithmetic -------------------------------------------------------


def test_the_two_headline_numbers_are_one_number_with_two_signs():
    """Luck is zero-sum: a point the scoreboard gave one team it took from the
    other, so the two nets are the gap between the two margins, mirrored."""
    rows = gb_det_rows()
    game = reconciling(rows)
    away, home = team_ledgers(game, rows, points_per_epa=PPE)
    assert away.team == "GB"
    assert home.team == "DET"
    assert away.net_points + home.net_points == pytest.approx(0.0, abs=1e-9)
    assert abs(home.net_points) == pytest.approx(abs(game.actual_margin - game.deserved_margin))


def test_the_team_the_scoreboard_flattered_is_the_one_with_the_positive_luck():
    """DET won by 8 and deserved less; every point of that gap is DET's luck."""
    away, home = team_ledgers(reconciling(gb_det_rows()), gb_det_rows(), points_per_epa=PPE)
    assert home.net_points > 0
    assert away.net_points < 0


def test_a_row_is_signed_toward_the_team_whose_table_it_sits_in():
    """A missed field goal is points GB did not get, so GB's table prints it red."""
    away, _home = team_ledgers(reconciling(gb_det_rows()), gb_det_rows(), points_per_epa=PPE)
    missed = next(row for row in away.rows if row.event == "41-yd field goal")
    assert missed.points == pytest.approx(-3.8 * PPE)
    assert missed.outcome == "missed"


def test_each_team_s_table_holds_its_own_events_biggest_first():
    away, home = team_ledgers(reconciling(gb_det_rows()), gb_det_rows(), points_per_epa=PPE)
    assert [row.event for row in home.rows] == ["55-yd field goal", "39-yd field goal"]
    assert away.n_events == 5
    sizes = [abs(row.points) for row in away.rows]
    assert sizes == sorted(sizes, reverse=True)


def test_a_row_label_is_the_plain_words_the_rest_of_the_product_prints():
    away, _home = team_ledgers(reconciling(gb_det_rows()), gb_det_rows(), points_per_epa=PPE)
    assert away.rows[0].label == plain_label(gb_det_rows()[0])


def test_a_table_shows_five_rows_and_folds_whatever_is_left_into_one():
    rows = gb_det_rows() + [
        card_row(1.1, play_id=8.0, kick_distance=44.0),
        card_row(0.9, play_id=9.0, kick_distance=47.0),
    ]
    away, _home = team_ledgers(reconciling(rows), rows, points_per_epa=PPE)
    shown = table_rows(away)
    assert len(shown) == 6
    assert shown[-1].event == "and 2 more"
    assert shown[-1].points == pytest.approx(sum(row.points for row in away.rows[5:]))


def test_a_team_with_five_or_fewer_events_has_nothing_to_fold():
    away, home = team_ledgers(reconciling(gb_det_rows()), gb_det_rows(), points_per_epa=PPE)
    assert len(table_rows(away)) == 5
    assert len(table_rows(home)) == 2


# --- the figure -----------------------------------------------------------


def test_the_card_is_portrait_because_it_is_a_share_image():
    fig, _ax = ledger_card()
    assert fig.get_figheight() > fig.get_figwidth()


def headline_texts(ax) -> list:
    """The two big numbers in the team boxes, away box first."""
    found = [t for t in ax.texts if t.get_text().endswith(" points")]
    return sorted(found, key=lambda t: t.get_position()[0])


def middle_column(ax) -> str:
    """The lane between the two boxes, read top line down, wrapping unwound."""
    band = (LEDGER_BOXES_Y - LEDGER_BOX_HEIGHT, LEDGER_BOXES_Y)
    lane = [
        t
        for t in ax.texts
        if t.get_position()[0] == pytest.approx(4.0) and band[0] < t.get_position()[1] <= band[1]
    ]
    lines = [t.get_text() for t in sorted(lane, key=lambda t: -t.get_position()[1])]
    return " ".join(" ".join(line.split()) for line in lines)


def test_a_box_headline_is_the_luck_on_that_team_s_own_plays():
    """Round 2 shipped a card whose headline was the game's net and whose table
    was that team's own plays, so a red headline sat over a green column."""
    rows = gb_det_rows()
    away, home = team_ledgers(reconciling(rows), rows, points_per_epa=PPE)
    for luck in (away, home):
        assert luck.own_points == pytest.approx(
            sum(row.points for row in table_rows(luck)), abs=1e-9
        )
    fig, ax = ledger_card(rows)
    printed = [t.get_text() for t in headline_texts(ax)]
    assert printed == [f"{away.own_points:+.1f} points", f"{home.own_points:+.1f} points"]


def test_a_folded_table_still_adds_up_to_its_own_headline():
    """`and n more` carries the exact sum it replaces, so the column adds up."""
    rows = gb_det_rows() + [
        card_row(1.1, play_id=8.0, kick_distance=44.0),
        card_row(0.9, play_id=9.0, kick_distance=47.0),
    ]
    away, _home = team_ledgers(reconciling(rows), rows, points_per_epa=PPE)
    assert len(table_rows(away)) == 6
    assert away.own_points == pytest.approx(sum(row.points for row in table_rows(away)), abs=1e-9)


def test_a_headline_wears_green_when_the_team_gained_and_red_when_it_paid():
    rows = [
        card_row(-3.0, play_id=1.0, charged_team="GB", kick_distance=41.0),
        card_row(-3.1, play_id=2.0, charged_team="DET", opponent="GB", kick_distance=55.0),
    ]
    fig, ax = ledger_card(rows)
    away_headline, home_headline = headline_texts(ax)
    assert away_headline.get_text().startswith("+")
    assert matplotlib.colors.to_hex(away_headline.get_color()) == PALETTE["good"].lower()
    assert home_headline.get_text().startswith("-")
    assert matplotlib.colors.to_hex(home_headline.get_color()) == PALETTE["bad"].lower()


def test_both_boxes_name_the_quantity_their_headline_is():
    fig, _ax = ledger_card()
    assert figure_text(fig).count("LUCK ON OWN PLAYS") == 2


def test_the_old_net_luck_headline_is_gone():
    text = figure_text(ledger_card()[0])
    assert "NET LUCK" not in text
    assert "points of luck" not in text


# --- the middle column ----------------------------------------------------


def test_the_net_luck_is_the_gap_between_the_two_margins():
    """Net luck is home own-plays minus away own-plays, and that is the gap."""
    rows = gb_det_rows()
    game = reconciling(rows)
    away, home = team_ledgers(game, rows, points_per_epa=PPE)
    size, favoured = net_luck(away, home)
    assert size == pytest.approx(abs(game.actual_margin - game.deserved_margin))
    assert size == pytest.approx(abs(home.own_points - away.own_points))
    assert favoured == "DET"


def test_the_middle_column_states_the_net_luck_in_the_favoured_team_s_colour():
    fig, ax = ledger_card(colors=(DET_BLUE, GB_GREEN))
    net = next(t for t in ax.texts if t.get_text().startswith("Net luck"))
    assert re.fullmatch(r"Net luck: (GB|DET) \+\d+\.\d", net.get_text())
    assert net.get_text() == "Net luck: DET +11.5"
    assert matplotlib.colors.to_hex(net.get_color()) == DET_BLUE.lower()
    assert net.get_fontweight() == "bold"
    assert "Net luck: DET +11.5" in middle_column(ax)


def test_the_middle_column_reads_vs_above_the_two_numbers():
    fig, ax = ledger_card()
    assert middle_column(ax).startswith("vs ")
    assert "VS" not in figure_text(fig)


def test_the_margin_sentence_reads_from_the_scoreboard_winner():
    """DET won by 8 and deserved to lose, so the sentence says both about DET."""
    fig, ax = ledger_card()
    game = reconciling(gb_det_rows())
    sentence = f"DET won by 8, deserved to lose by {abs(game.deserved_margin):.1f}"
    assert margin_sentence(game) == sentence
    assert sentence in middle_column(ax)


def test_the_margin_sentence_breaks_at_its_comma_rather_than_mid_clause():
    """Wrapped to the lane it read "deserved to lose / by 8.3", which orphans the
    number from the clause it belongs to. The comma is where a reader breaks."""
    fig, ax = ledger_card()
    sentence = next(t for t in ax.texts if t.get_text().startswith("DET won by 8"))
    assert sentence.get_text().split("\n")[0] == "DET won by 8,"


def test_the_margin_sentence_says_deserved_to_win_when_the_winner_deserved_it():
    game = verdict(
        home_team="DEN",
        away_team="WAS",
        home_score=27,
        away_score=26,
        actual_margin=1.0,
        deserved_margin=3.3,
    )
    assert margin_sentence(game) == "DEN won by 1, deserved to win by 3.3"


def test_the_margin_sentence_names_the_favoured_team_on_a_tie():
    game = verdict(
        home_team="MIN",
        away_team="DET",
        home_score=20,
        away_score=20,
        actual_margin=0.0,
        deserved_margin=2.1,
    )
    assert margin_sentence(game) == "tied 20\u201320, MIN deserved to win by 2.1"


def test_a_game_with_no_luck_says_the_two_margins_are_the_same_one():
    game = replace(branded(), deserved_margin=branded().actual_margin)
    assert margin_sentence(game) == "The deserved margin is the actual one."
    fig, ax = plot_luck_ledger_card(game, [], points_per_epa=PPE)
    lane = middle_column(ax)
    assert "The deserved margin is the actual one." in lane
    assert "Net luck" not in lane, "a game with no luck has no net to state"
    normalised = " ".join(figure_text(fig).split())
    assert normalised.count("The deserved margin is the actual one.") == 1


def test_the_old_margin_arrow_is_gone():
    assert "Actual margin" not in figure_text(ledger_card()[0])


def test_the_card_states_the_sign_convention_its_points_column_uses():
    fig, _ax = ledger_card()
    assert (
        "Points are what the scoreboard gave the team beyond what the play deserved."
        in figure_text(fig)
    )


def test_both_boxes_count_their_own_luck_events():
    """The event count is the one line both boxes carry, so neither is a stub."""
    text = figure_text(ledger_card()[0])
    assert "5 luck events" in text
    assert "2 luck events" in text


def test_the_card_carries_the_header_the_baseball_ledger_carries():
    fig, _ax = ledger_card()
    text = figure_text(fig)
    assert "Luck Ledger" in text
    assert "GB @ DET" in text
    assert "Final: GB 23, DET 31" in text
    assert "DTW: GB 95% / DET 5%" in text


def test_the_card_names_its_rows_in_plain_words():
    fig, _ax = ledger_card()
    text = figure_text(fig)
    assert "41-yd field goal" in text
    assert "recovered by DET" in text
    assert "missed" in text


def test_the_card_refuses_a_ledger_that_does_not_reconcile():
    """A decomposition under a headline it does not explain is worse than none."""
    with pytest.raises(ValueError, match="reconcile"):
        plot_luck_ledger_card(branded(), gb_det_rows(), points_per_epa=PPE)


def test_a_game_with_no_luck_events_is_a_sentence_rather_than_two_empty_tables():
    game = replace(branded(), deserved_margin=branded().actual_margin)
    fig, _ax = plot_luck_ledger_card(game, [], points_per_epa=PPE)
    assert "no luck events" in figure_text(fig).lower()


def test_a_team_with_fewer_than_five_events_renders_without_error():
    rows = gb_det_rows()[:3]
    fig, _ax = plot_luck_ledger_card(reconciling(rows), rows, points_per_epa=PPE)
    assert "41-yd field goal" in figure_text(fig)


# --------------------------------------------------------------------------
# the waterfall, re-read — round 2, Part B
# --------------------------------------------------------------------------


def test_a_bar_wears_the_colour_of_the_team_the_lucky_break_helped():
    """Round 1: GB missing a field goal was drawn in GB's colour, because
    neutralising it moves the margin toward GB. It is DET's lucky break, and a
    reader who knows that reads the old figure backwards."""
    rows = [ledger_row(3.42, charged_team="GB", play_id=1.0)]
    game = replace(branded(), deserved_margin=8.0 - 3.42 * PPE)
    _fig, ax = plot_luck_ledger(game, rows, points_per_epa=PPE, colors=(DET_BLUE, GB_GREEN))
    assert matplotlib.colors.to_hex(ax.patches[1].get_facecolor()) == DET_BLUE.lower()


def test_the_waterfall_legend_says_who_the_luck_helped():
    rows = [ledger_row(3.42, play_id=1.0), ledger_row(-0.80, play_id=2.0)]
    game = replace(branded(), deserved_margin=8.0 - (3.42 - 0.80) * PPE)
    _fig, ax = plot_luck_ledger(game, rows, points_per_epa=PPE)
    labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert labels == {"luck that helped GB", "luck that helped DET"}


def test_the_waterfall_carries_the_same_luck_arrow_the_distribution_does():
    _fig, ax = waterfall()
    label = next(t for t in ax.texts if t.get_text().startswith("luck moved the margin"))
    assert label.get_text() == "luck moved the margin 16.3 points toward DET"


def test_the_waterfall_title_starts_at_the_figure_margin_not_at_the_axes():
    """The row labels are sentences, so the axes starts a third of the way in
    and the title went with it, leaving a hole where the title should be."""
    fig, ax = waterfall()
    fig.canvas.draw()
    heading = next(t for t in fig.findobj(matplotlib.text.Text) if t.get_text() == "Luck Ledger")
    assert heading.get_window_extent().x0 < ax.get_window_extent().x0


def test_a_row_carries_its_club_s_mark_beside_its_label():
    """Two end bars and one event row, each marked."""
    logos = {"GB": synthetic_mark(), "DET": synthetic_mark((200, 30, 30))}
    _fig, ax = waterfall(logos=logos)
    assert len([a for a in ax.artists if isinstance(a, AnnotationBbox)]) == 3


def test_a_folded_row_has_no_club_to_mark():
    """ "4 events under 0.1 pt" is not one team's event, so it gets no mark."""
    rows = [ledger_row(3.42, charged_team="GB", play_id=1.0)] + [
        ledger_row(0.02, charged_team="GB", play_id=float(index)) for index in (2, 3)
    ]
    game = replace(branded(), deserved_margin=8.0 - (3.42 + 0.04) * PPE)
    _fig, ax = plot_luck_ledger(
        game, rows, points_per_epa=PPE, logos={"GB": synthetic_mark(), "DET": synthetic_mark()}
    )
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert any("events under" in label for label in labels)
    assert len([a for a in ax.artists if isinstance(a, AnnotationBbox)]) == 3


def test_the_two_tables_never_run_into_each_other():
    """`GB_DET_23-31--95-5_luck_ledger.png`: a nine-event team fills six rows,
    and the home team's accent bar was drawn straight through the last of them."""
    rows = gb_det_rows() + [
        card_row(1.1, play_id=8.0, kick_distance=44.0),
        card_row(0.9, play_id=9.0, kick_distance=47.0),
    ]
    _fig, ax = plot_luck_ledger_card(reconciling(rows), rows, points_per_epa=PPE)
    wide = [patch.get_bbox() for patch in ax.patches if patch.get_bbox().width > 5.0]
    accents = [box for box in wide if box.height < 0.2]
    row_rects = [box for box in wide if box.height >= 0.2]
    assert len(accents) == 2, "one accent bar per team section"
    assert len(row_rects) >= 8
    for accent in accents:
        straddling = [box for box in row_rects if box.y0 < accent.y1 and box.y1 > accent.y0]
        assert not straddling, "a table row runs through a section's accent bar"
    assert min(box.y0 for box in row_rects) > 0.0, "the last row runs off the card"


def test_the_vs_between_the_two_boxes_is_not_buried_under_them():
    fig, ax = plot_luck_ledger_card(reconciling(gb_det_rows()), gb_det_rows(), points_per_epa=PPE)
    fig.canvas.draw()
    versus = next(t for t in ax.texts if t.get_text() == "vs").get_window_extent()
    boxes = [p.get_window_extent() for p in ax.patches if p.get_window_extent().width > 200]
    assert boxes
    assert not any(box.overlaps(versus) for box in boxes)


def test_a_row_worth_less_than_a_tenth_of_a_point_is_not_printed_as_zero():
    """ "+0.0" reads as a rounding failure rather than as a small number."""
    rows = [
        card_row(0.02, play_id=1.0, component="extra_point", event_class="xp", actual=1.0),
        card_row(3.0, play_id=2.0, kick_distance=41.0),
    ]
    fig, _ax = plot_luck_ledger_card(reconciling(rows), rows, points_per_epa=PPE)
    text = figure_text(fig)
    assert "+0.0 " not in text and not text.endswith("+0.0")
    assert "-0.02" in text


def test_the_two_tables_sit_together_when_one_team_had_few_events():
    """`NYJ_SF_23-17--36-64_luck_ledger.png`: the Jets had three luck events, so
    their table stopped a third of the way down and the 49ers' section stayed
    pinned at its fixed height. The hole between them read as a bug."""
    rows = [
        card_row(2.0, play_id=1.0, kick_distance=41.0),
        card_row(1.0, play_id=2.0, kick_distance=36.0),
        card_row(0.5, play_id=3.0, kick_distance=30.0),
        *[
            card_row(-1.0 - index, play_id=10.0 + index, charged_team="DET", opponent="GB")
            for index in range(5)
        ],
    ]
    _fig, ax = plot_luck_ledger_card(reconciling(rows), rows, points_per_epa=PPE)
    wide = [patch.get_bbox() for patch in ax.patches if patch.get_bbox().width > 5.0]
    accents = sorted((box for box in wide if box.height < 0.2), key=lambda box: box.y0)
    row_rects = [box for box in wide if box.height >= 0.2]
    away_floor = min(box.y0 for box in row_rects if box.y0 > accents[0].y1)
    assert away_floor - accents[0].y1 < 1.0, "a hole between the two tables"
    assert min(box.y0 for box in row_rects) > 0.3, "the last row runs off the card"


def test_the_card_says_which_quantity_each_number_is():
    """`DEN_WAS_27-26--86-14_luck_ledger.png`: Denver's headline read -2.3 while
    every row of Denver's own table was a green positive summing to +2.3. Round 2
    labelled the two quantities; round 3 made them one quantity, so the headline
    and the column under it are now the same number."""
    fig, _ax = ledger_card()
    text = figure_text(fig)
    assert text.count("LUCK ON OWN PLAYS") == 2, "each headline names the quantity it is"
    assert "on each team's own plays" in text, "and the tables name theirs"
