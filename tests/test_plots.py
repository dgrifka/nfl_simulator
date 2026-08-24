"""The product layer's first figure: the bootstrap distribution.

Two things are under test and they are different in kind. The **verdict** is
arithmetic on committed numbers — which bucket a game lands in, which team the
headline names, whether the interval means anything — and it is tested the way
any arithmetic is. The **figure** is tested only for the marks a reader would
notice missing: the realized margin, the deserved margin, the zero line that
separates a home win from an away one, and the caveat that stops the interval
from being read as a plain 89%.

Nothing here re-derives DTW%. The simulator already owns that; this module
presents it.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from nfl_simulator.plots import (  # noqa: E402
    BAND_HIGH,
    BAND_LOW,
    CLEAR_FLIP,
    INK,
    INK_MUTED,
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
    overtime_lines,
    plot_band_sweep,
    plot_bootstrap_distribution,
    plot_luck_ledger,
    running_totals,
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


def test_a_realized_tie_outside_the_band_is_a_flip_because_the_scoreboard_named_nobody():
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


def test_the_figure_marks_the_realized_margin():
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
    assert "MIN 78% / DET 22%" in text


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


def _rule_label_boxes(fig, ax) -> list:
    """The two named rule labels' bounding boxes, in display pixels."""
    fig.canvas.draw()
    return [
        text.get_window_extent()
        for text in ax.texts
        if text.get_text().startswith(("deserved", "realized"))
    ]


@pytest.mark.parametrize("gap", [0.0, 0.4, 1.0, 2.3])
def test_the_two_rule_labels_never_overprint_when_the_margins_are_close(gap):
    """`2025_13_DEN_WAS` printed "deserved -3.3" straight through "realized -1".

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
    subtitle = next(text for text in ax.texts if "deserve-to-win across" in text.get_text())
    assert not any(box.overlaps(subtitle.get_window_extent()) for box in boxes)


@pytest.mark.parametrize(
    "deserved, realized",
    [(-8.28, 8.0), (27.93, 39.0), (0.70, 13.0)],
    ids=["2018_05_GB_DET", "2021_14_LV_KC", "2025_17_DET_MIN"],
)
def test_document_37_example_games_keep_their_rule_labels_clear(deserved, realized):
    """The three shipped examples are far apart and were never the defect. They
    are here so a fix for the close case cannot regress the common one."""
    span = max(abs(deserved), abs(realized)) + 8
    fig, ax = plot_bootstrap_distribution(
        verdict(
            actual_margin=realized,
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


def test_the_bars_sum_to_the_gap_between_the_realized_and_deserved_margins():
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


def test_a_bar_is_labelled_by_its_class_component_and_the_team_charged():
    (bar,) = luck_bars(
        [ledger_row(2.0, component="field_goal", event_class="45-49 yd", charged_team="MIN")],
        points_per_epa=PPE,
    )
    assert bar.label == "45-49 yd field goal — MIN"


def test_an_extra_point_does_not_repeat_itself_in_its_label():
    (bar,) = luck_bars(
        [ledger_row(2.0, component="extra_point", event_class="extra point", charged_team="GB")],
        points_per_epa=PPE,
    )
    assert bar.label == "extra point — GB"


def test_running_totals_start_at_the_realized_margin_and_land_on_the_deserved_one():
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
    assert any("realized" in label for label in labels)
    assert any("deserved" in label for label in labels)


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
    fig, ax = plot_luck_ledger(
        verdict(dtw_home=0.78, actual_margin=8.0, deserved_margin=8.0 - 3.42 * PPE),
        rows,
        points_per_epa=PPE,
    )
    text = " ".join(t.get_text() for t in fig.findobj(matplotlib.text.Text))
    assert "MIN 78% / DET 22%" in text


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
    for text in ax.texts:
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
    assert labels == ["moves the margin toward DET"]


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
    (0.85, 0.0),  # a realized tie the scoreboard never named a winner for
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
    """Document 33 excluded realized ties from its flip counts; the product labels
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
    assert {text.get_color() for text in panel.texts} <= {INK, INK_MUTED}


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
