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
    SCOREBOARD_HOLDS,
    TOO_CLOSE,
    GameVerdict,
    plot_bootstrap_distribution,
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
