"""Rendering one game to four PNGs.

Nothing here reads a committed artifact or a network. `render_game` itself is
exercised by `research/58_brand_figures.py`, which has the artifacts; what is
tested here is everything around it that can be wrong without the artifacts
noticing — the filename, the row preparation that turns ledger vocabulary into
plain words, and the fact that every figure this module writes lands on the
house cream with its data credit on it.
"""

from __future__ import annotations

import re
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")

import numpy as np
import polars as pl
import pytest
from PIL import Image

from nfl_simulator.plots import (
    OVERTIME_FOOTER,
    GameVerdict,
    OvertimeToss,
    attach_overtime_sidebar,
    plot_bootstrap_distribution,
    plot_game_card,
    plot_luck_ledger,
    plot_luck_ledger_card,
)
from nfl_simulator.render import ARTICLE_SUFFIX, SUFFIXES, figure_filename, prepare_rows
from nfl_simulator.style import PALETTE, finalize

PPE = 0.8389495557652871
CREAM = (252, 250, 246)


@pytest.fixture
def game() -> GameVerdict:
    """2018_05_GB_DET, the round's worked example: a clear flip."""
    return GameVerdict(
        game_id="2018_05_GB_DET",
        home_team="DET",
        away_team="GB",
        actual_margin=8.0,
        deserved_margin=-8.28,
        dtw_home=0.05,
        dtw_interval=(0.03, 0.08),
        margin_draws=np.linspace(-20.0, 6.0, 800),
        home_score=31,
        away_score=23,
        game_date="2018-10-07",
    )


def ledger_frame() -> pl.DataFrame:
    """Two real 2018_05_GB_DET rows, in the committed artifact's own schema."""
    return pl.DataFrame(
        {
            "play_id": [834.0, 324.0],
            "component": ["field_goal", "fumble"],
            "event_class": ["40-44 yd", "punt/live"],
            "charged_team": ["GB", "GB"],
            "expected": [0.8769803817228944, 0.6840206759729129],
            "swing": [-4.290041955688844, -5.000382690077953],
            "luck_epa": [3.762282631907235, 3.4203651477903745],
        }
    )


# --------------------------------------------------------------------------
# the filename
# --------------------------------------------------------------------------


def test_the_filename_follows_the_baseball_pattern(game):
    assert figure_filename(game, "dtw") == "GB_DET_23-31--95-5_dtw.png"


def test_every_suffix_names_a_different_file(game):
    names = {figure_filename(game, suffix) for suffix in SUFFIXES}
    assert len(names) == len(SUFFIXES) == 4
    assert names == {
        "GB_DET_23-31--95-5_dtw.png",
        "GB_DET_23-31--95-5_luck_ledger.png",
        "GB_DET_23-31--95-5_card.png",
        "GB_DET_23-31--95-5_waterfall.png",
    }


def test_the_share_ledger_and_the_article_waterfall_are_different_figures():
    """Round 1 shipped the waterfall under the `luck_ledger` name, and the maintainer
    needed help reading it. The share image is the card; the waterfall stays."""
    assert SUFFIXES == ("dtw", "luck_ledger", "card", "waterfall")


def test_the_shares_in_the_filename_sum_to_a_hundred(game):
    """`95-5`, not `95-5.2`: the two are rounded once, together, as in the headline."""
    away, home = figure_filename(game, "dtw").split("--")[1].split("_")[0].split("-")
    assert int(away) + int(home) == 100


def test_a_game_with_no_score_on_file_is_named_by_its_game_id(game):
    """Never invent a scoreline to fill a filename."""
    unscored = replace(game, home_score=None, away_score=None)
    assert figure_filename(unscored, "card") == "2018_05_GB_DET_card.png"


# --------------------------------------------------------------------------
# preparing the ledger rows
# --------------------------------------------------------------------------


def test_prepared_rows_read_as_sentences(game):
    from nfl_simulator.plots import plain_label

    rows = prepare_rows(ledger_frame(), game, distances={834.0: 42.0})
    assert [plain_label(row) for row in rows] == [
        "GB 42-yd field goal, missed",
        "GB fumble on a punt, recovered by DET",
    ]


def test_the_branch_is_recovered_for_an_artifact_that_does_not_carry_it(game):
    rows = prepare_rows(ledger_frame(), game)
    assert [row["actual"] for row in rows] == [0.0, 0.0]


def test_the_opponent_is_the_other_team_in_the_game_not_the_home_team(game):
    """A fumble charged to the home team is recovered by the away team."""
    frame = ledger_frame().with_columns(pl.lit("DET").alias("charged_team"))
    assert {row["opponent"] for row in prepare_rows(frame, game)} == {"GB"}


def test_a_distance_that_is_not_known_leaves_the_label_on_its_class(game):
    rows = prepare_rows(ledger_frame(), game, distances={})
    assert rows[0].get("kick_distance") is None


# --------------------------------------------------------------------------
# what lands on disk
# --------------------------------------------------------------------------


def figures(game):
    """The three figures for a verdict the two-row fixture ledger reconciles with.

    The real game has fifteen luck events; the fixture carries two of them, so
    the verdict drawn here is given the deserved margin *those two* imply. The
    waterfall refuses a ledger that does not span its own two ends, and rightly
    — the point of these tests is what lands on disk, not the arithmetic."""
    rows = prepare_rows(ledger_frame(), game, distances={834.0: 42.0})
    reconciling = replace(
        game, deserved_margin=game.actual_margin - sum(r["luck_epa"] for r in rows) * PPE
    )
    colours = ("#0076B6", "#203731")
    return {
        "dtw": plot_bootstrap_distribution(reconciling, colors=colours)[0],
        "luck_ledger": plot_luck_ledger_card(reconciling, rows, points_per_epa=PPE, colors=colours)[
            0
        ],
        "card": plot_game_card(reconciling, colors=colours)[0],
        "waterfall": plot_luck_ledger(reconciling, rows, points_per_epa=PPE, colors=colours)[0],
    }


@pytest.mark.parametrize("suffix", ["dtw", "luck_ledger", "card", "waterfall"])
def test_every_saved_figure_lands_on_the_house_cream(game, tmp_path, suffix):
    path = finalize(figures(game)[suffix], tmp_path / f"{suffix}.png")
    assert Image.open(path).convert("RGB").getpixel((2, 2)) == CREAM


@pytest.mark.parametrize("suffix", ["dtw", "luck_ledger", "card", "waterfall"])
def test_every_saved_figure_carries_its_data_credit(game, tmp_path, suffix):
    """nflverse asks for credit, so no figure leaves this module without it."""
    path = finalize(figures(game)[suffix], tmp_path / f"{suffix}.png")
    pixels = np.asarray(Image.open(path).convert("RGB"), dtype=float)
    height, width = pixels.shape[:2]
    corner = pixels[: int(height * 0.06), int(width * 0.88) :]
    assert corner.mean(axis=2).min() < 200


def test_the_card_is_square_on_disk_and_survives_a_four_hundred_pixel_preview(game, tmp_path):
    path = finalize(figures(game)["card"], tmp_path / "card.png", bbox_inches=None)
    image = Image.open(path)
    assert image.width == image.height
    assert image.width >= 1200


def test_the_two_rule_labels_are_still_clear_on_the_branded_figure(game):
    """The restyle boxed the callouts, which made them wider — the fix still holds."""
    fig, ax = plot_bootstrap_distribution(game)
    fig.canvas.draw()
    boxes = [
        text.get_window_extent()
        for text in ax.texts
        if re.match(r"^(Actual|Deserved) [+-]", text.get_text())
    ]
    assert len(boxes) == 2
    assert not boxes[0].overlaps(boxes[1])


def test_the_palette_the_card_paints_is_the_house_one():
    assert PALETTE["bg"] == "#FCFAF6"


# --------------------------------------------------------------------------
# overtime — a footer on the share images, the sidebar for the article
# --------------------------------------------------------------------------


def overtime(game) -> GameVerdict:
    return replace(game, went_to_overtime=True)


def share_figures(game):
    """The three figures a timeline sees, as `render_game` builds them."""
    return {suffix: figures(game)[suffix] for suffix in ("dtw", "luck_ledger", "card")}


@pytest.mark.parametrize("suffix", ["dtw", "luck_ledger", "card"])
def test_every_share_figure_says_the_toss_is_reported_not_neutralized(game, suffix):
    fig = share_figures(overtime(game))[suffix]
    text = " ".join(t.get_text() for t in fig.findobj(matplotlib.text.Text))
    assert OVERTIME_FOOTER in text.replace("\n", " ")


@pytest.mark.parametrize("suffix", ["dtw", "luck_ledger", "card"])
def test_a_regulation_game_carries_no_overtime_line(game, suffix):
    fig = share_figures(game)[suffix]
    text = " ".join(t.get_text() for t in fig.findobj(matplotlib.text.Text))
    assert "Went to overtime" not in text


@pytest.mark.parametrize("suffix", ["dtw", "luck_ledger", "card"])
def test_no_share_figure_carries_the_sidebar(game, suffix):
    """Round 1's review: the sidebar is overwhelming on a share image. Six
    paragraphs of methodology beside a card is an article, not a post."""
    assert len(share_figures(overtime(game))[suffix].axes) == 1


@pytest.mark.parametrize("suffix", ["dtw", "luck_ledger", "card"])
def test_an_overtime_share_figure_is_the_size_a_regulation_one_is(game, suffix):
    """The sidebar grew the figure. Two games annotated differently were then
    drawn at two different widths, and a timeline crops them differently."""
    regulation = share_figures(game)[suffix].get_size_inches()
    assert list(share_figures(overtime(game))[suffix].get_size_inches()) == list(regulation)


def test_the_card_puts_the_overtime_line_under_the_interval_line(game):
    """The card's layout is frozen; this is the one line added to it."""
    fig = share_figures(overtime(game))["card"]
    ax = fig.axes[0]
    footer = next(t for t in ax.texts if t.get_text() == OVERTIME_FOOTER)
    interval = next(t for t in ax.texts if "interval on" in t.get_text())
    assert footer.get_position()[1] < interval.get_position()[1]
    assert footer.get_fontsize() == interval.get_fontsize()


def test_the_article_file_is_named_for_the_figure_it_is_a_version_of(game):
    assert figure_filename(game, ARTICLE_SUFFIX) == "GB_DET_23-31--95-5_dtw_article.png"
    assert ARTICLE_SUFFIX not in SUFFIXES, "the article is an extra, not a fourth share image"


def test_the_sidebar_is_what_the_article_version_adds(game):
    """`render_game(..., article=True)` is the only path that attaches it."""
    fig, ax = plot_bootstrap_distribution(overtime(game))
    before = fig.get_size_inches()[0]
    panel = attach_overtime_sidebar(
        fig, ax, overtime(game), OvertimeToss(received="GB", season=2016, delta_dtw_home=-0.21)
    )
    assert panel is not None
    assert fig.get_size_inches()[0] > before


# --------------------------------------------------------------------------
# the editions — document 58 §2
# --------------------------------------------------------------------------


def test_the_context_maps_each_edition_to_the_handles_it_simulates_with():
    """Strict pays for no model; Full pays for both. The map is what a caller reads."""
    from nfl_simulator.render import edition_handles

    handles = edition_handles("dp-model", "rd-model")
    assert set(handles) == {"strict", "full"}
    assert handles["strict"] == {"dropped_pick_model": None, "receiver_drop_model": None}
    assert handles["full"] == {
        "dropped_pick_model": "dp-model",
        "receiver_drop_model": "rd-model",
    }


def test_a_missing_trace_leaves_its_edition_handle_none_rather_than_failing():
    """`_dropped_pick_pieces` degrades to `None`; the map must carry that through."""
    from nfl_simulator.render import edition_handles

    handles = edition_handles(None, None)
    assert handles["full"] == {"dropped_pick_model": None, "receiver_drop_model": None}
