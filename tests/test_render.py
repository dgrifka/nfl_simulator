"""Rendering one game to four PNGs.

Nothing here reads a committed artifact or a network. `render_game` itself is
exercised by `research/58_brand_figures.py`, which has the artifacts; what is
tested here is everything around it that can be wrong without the artifacts
noticing — the filename, the row preparation that turns ledger vocabulary into
plain words, and the fact that every figure this module writes lands on the
house cream with its data credit on it.
"""

from __future__ import annotations

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")

import numpy as np
import polars as pl
import pytest
from PIL import Image

from nfl_simulator.plots import (
    MEASURED_COVERAGE,
    OVERTIME_FOOTER,
    GameVerdict,
    OvertimeToss,
    attach_overtime_sidebar,
    plot_bootstrap_distribution,
    plot_game_card,
    plot_luck_ledger,
    plot_luck_ledger_card,
)
from nfl_simulator.render import (
    ARTICLE_SUFFIX,
    SUFFIXES,
    figure_filename,
    kicker_surname,
    prepare_rows,
)
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


def test_the_filename_opens_with_the_game_id(game):
    """Round 10: the game id leads, and the rest of the name is unchanged.

    Document 63 §3a: the name used to open with the two clubs, which carries no
    season and no week, so two meetings of the same pair with the same scoreline
    and the same split wrote the same file and one of them was lost.
    """
    assert figure_filename(game, "dtw") == "2018_05_GB_DET_23-31--95-5_strict_dtw.png"


def test_every_suffix_names_a_different_file(game):
    names = {figure_filename(game, suffix) for suffix in SUFFIXES}
    assert len(names) == len(SUFFIXES) == 4
    assert names == {
        "2018_05_GB_DET_23-31--95-5_strict_dtw.png",
        "2018_05_GB_DET_23-31--95-5_strict_luck_ledger.png",
        "2018_05_GB_DET_23-31--95-5_strict_card.png",
        "2018_05_GB_DET_23-31--95-5_strict_waterfall.png",
    }


def test_the_two_pairs_that_collided_in_the_corpus_now_write_four_names(game):
    """Document 63 §3a's eight lost PNGs, as four distinct names.

    `2016_15_MIA_NYJ` and `2023_12_MIA_NYJ` share a scoreline and a split;
    `2018_16_DEN_OAK` and `2023_18_DEN_LV` share those *and* a pair of clubs,
    because the relocation alias resolves Oakland to Las Vegas. The season and
    the week are the only things that ever told either pair apart.
    """
    pairs = [
        ("2016_15_MIA_NYJ", "NYJ", "MIA", 34, 13),
        ("2023_12_MIA_NYJ", "NYJ", "MIA", 34, 13),
        ("2018_16_DEN_OAK", "LV", "DEN", 14, 27),
        ("2023_18_DEN_LV", "LV", "DEN", 14, 27),
    ]
    names = {
        figure_filename(
            replace(
                game,
                game_id=game_id,
                home_team=home,
                away_team=away,
                away_score=away_score,
                home_score=home_score,
                dtw_home=1.0,
            ),
            "dtw",
        )
        for game_id, home, away, away_score, home_score in pairs
    }
    assert len(names) == 4
    assert "2018_16_DEN_OAK_14-27--0-100_strict_dtw.png" in names
    assert "2023_18_DEN_LV_14-27--0-100_strict_dtw.png" in names


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
    assert figure_filename(unscored, "card") == "2018_05_GB_DET_strict_card.png"


# --------------------------------------------------------------------------
# preparing the ledger rows
# --------------------------------------------------------------------------


def test_prepared_rows_read_as_sentences(game):
    from nfl_simulator.plots import plain_label

    rows = prepare_rows(ledger_frame(), game, distances={834.0: 42.0})
    assert [plain_label(row) for row in rows] == [
        "GB 42-yd field goal, missed (88% kick)",
        "GB fumble on a punt, recovered by DET",
    ]


def test_a_prepared_kick_row_carries_the_kicker_the_play_by_play_names(game):
    from nfl_simulator.plots import plain_label

    rows = prepare_rows(ledger_frame(), game, distances={834.0: 42.0}, kickers={834.0: "Crosby"})
    assert plain_label(rows[0]) == "GB 42-yd field goal · Crosby, missed (88% kick)"


def test_a_play_with_no_kicker_on_file_prepares_without_one(game):
    rows = prepare_rows(ledger_frame(), game, distances={834.0: 42.0})
    assert rows[0]["kicker"] is None


def test_a_kicker_is_read_down_to_the_surname_the_figure_prints():
    """nflverse writes `M.Crosby`; a card row has no room for the initial."""
    assert kicker_surname("M.Crosby") == "Crosby"
    assert kicker_surname("G.Tavecchio") == "Tavecchio"


def test_a_kicker_name_that_is_not_on_file_stays_absent():
    assert kicker_surname(None) is None
    assert kicker_surname("") is None


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
    # Round 5: the `dtw` share image is the margin distribution again, with the
    # unsigned "wins by" axis. The per-team scoreline figure is withdrawn from
    # the render path — a margin swing is not a per-team points swing.
    return {
        "dtw": plot_bootstrap_distribution(reconciling, colors=colours, coverage=False)[0],
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
    """nflverse asks for credit, so no figure leaves this module without it.

    Bottom-right since round 10 — the corner document 63 measured the title
    running into when the stamp was in the top one.
    """
    path = finalize(figures(game)[suffix], tmp_path / f"{suffix}.png")
    pixels = np.asarray(Image.open(path).convert("RGB"), dtype=float)
    height, width = pixels.shape[:2]
    corner = pixels[int(height * 0.94) :, int(width * 0.80) :]
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
    boxes = [text.get_window_extent() for text in ax.texts if text.get_gid() == "rule-label"]
    assert len(boxes) == 2
    assert not boxes[0].overlaps(boxes[1])


def test_the_share_and_the_article_are_the_same_margin_figure(game):
    """Round 5 withdrew the scoreline swap: both are the "wins by" margin plot."""
    share = figures(game)["dtw"].axes[0]
    article, _ax = plot_bootstrap_distribution(game)
    for axes in (share, article.axes[0]):
        assert axes.get_xlabel() == ""
        assert {"\u2190 GB wins by", "DET wins by \u2192"} <= {
            text.get_text() for text in axes.texts
        }


def test_the_share_drops_the_coverage_sentence_the_article_keeps(game):
    """Round 4 §A: a second percentage beside the share reads as a competing one."""
    share = " ".join(t.get_text() for t in figures(game)["dtw"].findobj(matplotlib.text.Text))
    article = " ".join(
        t.get_text() for t in plot_bootstrap_distribution(game)[0].findobj(matplotlib.text.Text)
    )
    assert MEASURED_COVERAGE not in share
    assert MEASURED_COVERAGE in article


def test_the_four_share_suffixes_did_not_change(game):
    """The `dtw` file is the same name for a different figure, not a fifth file."""
    assert SUFFIXES == ("dtw", "luck_ledger", "card", "waterfall")


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
    assert (
        figure_filename(game, ARTICLE_SUFFIX) == "2018_05_GB_DET_23-31--95-5_strict_dtw_article.png"
    )
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
    assert handles["strict"] == {
        "dropped_pick_model": None,
        "receiver_drop_model": None,
        "edition": "strict",
    }
    assert handles["full"] == {
        "dropped_pick_model": "dp-model",
        "receiver_drop_model": "rd-model",
        "edition": "full",
    }


def test_each_handle_names_its_own_edition_so_the_possession_cap_switches_on():
    """Round 8's defect. `simulate_game` keys document 61's cap on its `edition`
    **argument** — not on the variant the ledger came out carrying — and this
    map never passed one, so a Full render replayed uncapped and stopped against
    the capped summary it was checked against.

    `tests/test_simulator.py` owns what the argument then does; what this map
    owes is the argument itself.
    """
    from nfl_simulator.render import edition_handles

    handles = edition_handles(None, None)
    assert handles["full"]["edition"] == "full"
    assert handles["strict"]["edition"] == "strict"


def test_a_missing_trace_leaves_its_edition_handle_none_rather_than_failing():
    """`_dropped_pick_pieces` degrades to `None`; the map must carry that through."""
    from nfl_simulator.render import edition_handles

    handles = edition_handles(None, None)
    assert handles["full"] == {
        "dropped_pick_model": None,
        "receiver_drop_model": None,
        "edition": "full",
    }


# --------------------------------------------------------------------------
# a summary and a replay per edition — figure round 6, part A
# --------------------------------------------------------------------------


def summary_frame(rows: list[dict]) -> pl.DataFrame:
    """A `dtw_games_*.parquet`-shaped frame, at whatever numbers a test needs.

    No rows is the shape a checkout that never ran the Full pass actually holds
    — `render._empty_summary` — rather than a frame with no columns at all,
    which nothing on disk can produce."""
    from nfl_simulator.render import _empty_summary

    return pl.DataFrame(rows) if rows else _empty_summary()


def sources_with(strict: pl.DataFrame, full: pl.DataFrame):
    """A `Sources` carrying two summaries and nothing else that touches disk."""
    from nfl_simulator.render import Sources

    empty = summary_frame([])
    return Sources(
        games=strict,
        ledger=empty,
        schedule=empty,
        overtime=empty,
        slope=PPE,
        full=full,
    )


STRICT_ROW = {
    "game_id": "2025_17_DET_MIN",
    "home_team": "MIN",
    "away_team": "DET",
    "actual_margin": 13.0,
    "deserved_margin": 1.2,
    "dtw_home": 0.55,
    "dtw_low": 0.50,
    "dtw_high": 0.61,
}
FULL_ROW = {
    **STRICT_ROW,
    "deserved_margin": 3.4,
    "dtw_home": 0.63,
    "dtw_low": 0.57,
    "dtw_high": 0.70,
}


def test_the_default_edition_is_full_from_the_first_charted_season_and_strict_before_it():
    """FTN charting starts in 2022; before it there is only one adjudication."""
    from nfl_simulator.render import default_edition

    assert default_edition("2022_13_WAS_NYG") == "full"
    assert default_edition("2025_17_DET_MIN") == "full"
    assert default_edition("2018_05_GB_DET") == "strict"
    assert default_edition("2016_14_NYJ_SF") == "strict"


def test_a_pre_charting_game_asked_for_full_is_refused_before_anything_is_simulated():
    """The error names the season and the reason, and it costs no simulation.

    `replay` would otherwise load the play-by-play and both traces before
    discovering that the game predates the charting the Full edition is built
    from — and would then return a Strict ledger under a Full headline, because
    both variant builders warn and return an empty list on a pre-2022 game."""
    from nfl_simulator.render import replay

    with pytest.raises(ValueError, match="2018.*charting"):
        replay("2018_05_GB_DET", STRICT_ROW, edition="full")


def test_an_edition_nobody_named_is_refused(game):
    from nfl_simulator.render import replay

    with pytest.raises(ValueError, match="deluxe"):
        replay("2025_17_DET_MIN", STRICT_ROW, edition="deluxe")


def test_each_edition_reads_its_own_summary():
    """A Full render must not check itself against Strict's published numbers."""
    sources = sources_with(summary_frame([STRICT_ROW]), summary_frame([FULL_ROW]))
    assert sources.game_row("2025_17_DET_MIN")["dtw_home"] == 0.55
    assert sources.game_row("2025_17_DET_MIN", edition="strict")["dtw_home"] == 0.55
    assert sources.game_row("2025_17_DET_MIN", edition="full")["dtw_home"] == 0.63


def test_a_game_missing_from_the_full_summary_names_the_artifact_to_build():
    """A checkout that has not run the Full pass says so, rather than falling
    back to Strict's numbers under a Full stamp."""
    from nfl_simulator.render import FULL_ARTIFACT

    sources = sources_with(summary_frame([STRICT_ROW]), summary_frame([]))
    with pytest.raises(SystemExit, match=FULL_ARTIFACT):
        sources.game_row("2025_17_DET_MIN", edition="full")


def test_the_replay_gaps_are_measured_against_the_row_it_was_handed():
    """The four numbers a redrawn distribution has to land on, and no others."""
    from nfl_simulator.ledger import Ledger
    from nfl_simulator.render import replay_gaps
    from nfl_simulator.simulator import SimulationResult

    result = SimulationResult(
        game_id="2025_17_DET_MIN",
        actual_margin=13.0,
        deserved_margin=3.4,
        dtw_home=0.63,
        dtw_interval=(0.57, 0.70),
        margin_draws=np.zeros(4),
        ledger=Ledger(()),
        total_luck_epa=0.0,
        variant="full",
    )
    assert max(replay_gaps(result, FULL_ROW).values()) == 0.0
    gaps = replay_gaps(result, STRICT_ROW)
    assert set(gaps) == {"deserved_margin", "dtw_home", "dtw_low", "dtw_high"}
    assert gaps["dtw_home"] == pytest.approx(0.08)


def test_the_simulation_context_loads_the_columns_both_variant_models_price_on():
    """A Full replay reads covariates v1.3 never needed.

    `44_read_side_fix.SIM_COLUMNS` is the v1.3 frame, and the two
    hands-on-the-ball models price on columns that are not in it. Loading the
    narrow frame and asking for Full would raise deep inside a model rather
    than at the edge — or worse, price a null the frame simply never fetched.
    Document 49 §6's V-1 proved the wide frame inert on Strict, which is what
    makes loading it unconditionally safe."""
    from nfl_simulator.dropped_picks import PBP_COVARIATE_COLUMNS as DROPPED_PICK_COLUMNS
    from nfl_simulator.receiver_drops import PBP_COVARIATE_COLUMNS as RECEIVER_COLUMNS
    from nfl_simulator.receiver_drops import PBP_SWING_COLUMNS
    from nfl_simulator.render import simulation_columns

    columns = simulation_columns()
    assert len(columns) == len(set(columns)), "a duplicated column is a wasted read"
    for needed in (DROPPED_PICK_COLUMNS, RECEIVER_COLUMNS, PBP_SWING_COLUMNS, ["defteam"]):
        assert set(needed) <= set(columns)
    # And v1.3's own frame is still all there: Strict must not lose a column.
    assert {"game_id", "play_id", "kicker_player_name", "extra_point_result"} <= set(columns)


def test_the_filename_carries_the_edition_so_two_of_them_never_collide(game):
    """One game has two adjudications and they are two different images."""
    strict = replace(game, edition="strict")
    full = replace(game, edition="full", dtw_home=0.37, dtw_interval=(0.30, 0.44))
    assert figure_filename(strict, "card") == "2018_05_GB_DET_23-31--95-5_strict_card.png"
    assert figure_filename(full, "card") == "2018_05_GB_DET_23-31--63-37_full_card.png"
    assert figure_filename(strict, "card") != figure_filename(full, "card")


def test_a_game_with_no_score_still_names_its_edition(game):
    unscored = replace(game, home_score=None, away_score=None, edition="full")
    assert figure_filename(unscored, "card") == "2018_05_GB_DET_full_card.png"


def test_a_prepared_variant_row_carries_the_thrower_and_the_receiver(game):
    """The two names the Full edition's rows are read by, from the play-by-play.

    Presentation only, exactly as the kicker's name is: the dropped pick was
    priced at the defence's shrunk rate and the drop at the receiving corps'.
    Neither name moves a number, and a play without one keeps its bare label."""
    from nfl_simulator.plots import plain_label

    frame = pl.DataFrame(
        {
            "play_id": [55.0, 61.0],
            "component": ["dropped_pick", "receiver_drop"],
            "event_class": ["34-66 yd, early down", "1-33 yd, late down"],
            "charged_team": ["GB", "GB"],
            "actual": [1.0, 0.0],
            "expected": [0.52, 0.96],
            "swing": [-2.0, -1.4],
            "luck_epa": [0.96, 1.34],
        }
    )
    rows = prepare_rows(frame, game, passers={55.0: "Goff"}, receivers={61.0: "Watson"})
    # Detroit is the club that dropped Green Bay's interceptable throw, and
    # `opponent` — which `prepare_rows` is what adds — is how the label knows.
    assert [plain_label(row) for row in rows] == [
        "DET dropped pick · thrown by Goff (48% catch)",
        "GB drop · Watson (96% catch)",
    ]


def test_a_variant_row_with_nobody_on_file_keeps_its_bare_label(game):
    from nfl_simulator.plots import event_phrase

    frame = pl.DataFrame(
        {
            "play_id": [55.0],
            "component": ["receiver_drop"],
            "event_class": ["1-33 yd, late down"],
            "charged_team": ["GB"],
            "actual": [0.0],
            "expected": [0.96],
            "swing": [-1.4],
            "luck_epa": [1.34],
        }
    )
    (row,) = prepare_rows(frame, game)
    assert row["receiver"] is None
    assert event_phrase(row) == "drop"


def test_the_simulation_frame_carries_the_two_names_those_rows_are_read_by():
    """Added the way `kicker_player_name` was in round 4 — presentation only."""
    from nfl_simulator.render import simulation_columns

    assert {"passer_player_name", "receiver_player_name"} <= set(simulation_columns())


def test_the_simulation_keeps_the_events_the_intervals_are_read_from():
    """`LedgerEntry` stores the posterior mean; the draws live on the event.

    Nothing on disk carries them — the shipped ledger has one number per row —
    so the only place a figure can get an interval is the replay it is already
    running, and the result has to hand the events back for that."""
    from nfl_simulator.simulator import LuckEvent, SimulationResult

    assert "events" in SimulationResult.__dataclass_fields__
    event = LuckEvent(
        play_id=1.0,
        component="field_goal",
        event_class="40-44 yd",
        charged_team="GB",
        actual=0.0,
        expected_draws=np.linspace(0.80, 0.95, 200),
        swing=-4.0,
    )
    assert event.to_entry().expected == pytest.approx(event.expected_draws.mean())


def test_the_intervals_are_keyed_by_the_play_and_the_component_on_it():
    """A blocked field goal books a fumble row on the same play id. Keying on
    the play alone would give one of the two rows the other's probability."""
    from nfl_simulator.render import expected_intervals
    from nfl_simulator.simulator import LuckEvent, SimulationResult

    def event(component, draws):
        return LuckEvent(
            play_id=7.0,
            component=component,
            event_class="x",
            charged_team="GB",
            actual=0.0,
            expected_draws=draws,
            swing=-1.0,
        )

    result = SimulationResult(
        game_id="2018_05_GB_DET",
        actual_margin=8.0,
        deserved_margin=8.0,
        dtw_home=0.5,
        dtw_interval=(0.4, 0.6),
        margin_draws=np.zeros(3),
        ledger=None,
        total_luck_epa=0.0,
        events=(
            event("field_goal", np.linspace(0.80, 0.96, 200)),
            event("fumble", np.linspace(0.40, 0.60, 200)),
        ),
    )
    intervals = expected_intervals(result)
    assert set(intervals) == {(7.0, "field_goal"), (7.0, "fumble")}
    assert intervals[(7.0, "field_goal")][0] == pytest.approx(0.8088, abs=1e-3)


def test_a_prepared_row_takes_the_interval_of_its_own_play_and_component(game):
    frame = pl.DataFrame(
        {
            "play_id": [834.0],
            "component": ["field_goal"],
            "event_class": ["40-44 yd"],
            "charged_team": ["GB"],
            "expected": [0.88],
            "swing": [-4.29],
            "luck_epa": [3.76],
        }
    )
    (row,) = prepare_rows(frame, game, intervals={(834.0, "field_goal"): (0.83, 0.92)})
    assert (row["expected_low"], row["expected_high"]) == (0.83, 0.92)


def test_a_prepared_row_with_no_interval_on_file_carries_none(game):
    (row,) = prepare_rows(ledger_frame().head(1), game)
    assert row["expected_low"] is None


# --------------------------------------------------------------------------
# the annotation band — figure round 8
# --------------------------------------------------------------------------


def test_the_share_figure_asks_for_no_callout():
    """`HOU 55% · LAC 45% — too close to call` restated the subtitle's own
    `DTW:` line and the verdict pill beside it, in the strip the two rule labels
    and the luck arrow were already competing for."""
    from nfl_simulator.render import DTW_FIGURE

    assert DTW_FIGURE.get("callout", False) is False
    assert DTW_FIGURE["arrow"] is True


def test_the_article_figure_keeps_the_callout_the_share_image_drops():
    """A reader who has come for the methodology has room for the sentence, and
    the article figure is the one with the sidebar beside it."""
    from nfl_simulator.render import DTW_ARTICLE_FIGURE, DTW_FIGURE

    assert DTW_ARTICLE_FIGURE["callout"] is True
    assert DTW_ARTICLE_FIGURE["bin_width"] == DTW_FIGURE["bin_width"]
    assert DTW_ARTICLE_FIGURE["arrow"] == DTW_FIGURE["arrow"]
