"""The house style — the parts of it a reader can check without looking.

A style module is mostly taste, and taste is not testable. What *is* testable is
the handful of promises the rest of the product builds on: the surface a figure
is saved on, the fact that a saved PNG carries its data credit, and the blend
the clash rule uses. Those are the ones that break silently — a watermark that
stops being stamped looks exactly like a figure nobody has looked at yet.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image, ImageDraw

from nfl_simulator.style import (
    BRAND_HANDLE,
    DATA_CREDIT,
    PALETTE,
    WATERMARK,
    apply_base_style,
    draw_title_block,
    finalize,
    lighten,
    title_axes,
)

# The top-right corner the watermark is stamped into, as fractions of the saved
# image. `apply_watermark` places the block at 2% in from the right and ~1.2%
# down from the top, so a box this size contains all of it and nothing else.
# Round 10 moved the stamp to the *bottom* right because document 63 measured
# the title running under it on 84-89% of distribution figures; write-up round 3
# moves it back — the MLB simulator's corner — and buys the room with
# `reserve_stamp_strip` rather than by conceding the corner.
CORNER_W, CORNER_H = 0.20, 0.12

# Cream is (252, 250, 246); the watermark ink is mid-grey. Anything whose mean
# channel is below this is text rather than background.
DARK = 200


def corner(path) -> np.ndarray:
    pixels = np.asarray(Image.open(path).convert("RGB"), dtype=float)
    height, width = pixels.shape[:2]
    return pixels[: int(height * CORNER_H), int(width * (1 - CORNER_W)) :]


@pytest.fixture
def blank():
    """A figure with nothing in it, so anything found in a corner is the stamp."""
    apply_base_style()
    fig = plt.figure(figsize=(6, 4))
    yield fig
    plt.close(fig)


def test_the_surface_is_the_warm_cream_the_house_style_uses():
    assert PALETTE["bg"] == "#FCFAF6"


def test_the_brand_constants_are_importable_and_the_handle_is_the_placeholder():
    assert BRAND_HANDLE == "@nfl_simulator"
    assert DATA_CREDIT == "Data: nflverse"
    assert WATERMARK == "Data: nflverse | @nfl_simulator"


def test_a_finalized_png_is_saved_on_the_cream_surface(blank, tmp_path):
    path = tmp_path / "surface.png"
    finalize(blank, path)
    top_left = Image.open(path).convert("RGB").getpixel((2, 2))
    assert top_left == (252, 250, 246)


def test_finalize_stamps_the_watermark_into_the_top_right_corner(blank, tmp_path):
    bare, stamped = tmp_path / "bare.png", tmp_path / "stamped.png"
    blank.savefig(bare, dpi=200, facecolor=PALETTE["bg"])

    assert corner(bare).mean(axis=2).min() >= DARK, "the bare save should have an empty corner"

    finalize(blank, stamped)
    assert corner(stamped).mean(axis=2).min() < DARK


def test_the_data_credit_travels_with_every_saved_figure(blank, tmp_path):
    """nflverse's licence asks for credit, so it is stamped rather than optional."""
    path = tmp_path / "credit.png"
    finalize(blank, path)
    assert corner(path).mean(axis=2).min() < DARK


def test_lighten_blends_halfway_to_white():
    assert lighten("#000000", 0.5) == "#808080"
    assert lighten("#000000", 0.0) == "#000000"
    assert lighten("#000000", 1.0) == "#ffffff"


def test_apply_base_style_puts_the_cream_on_the_figure_and_the_axes():
    apply_base_style()
    assert matplotlib.rcParams["figure.facecolor"] == PALETTE["bg"]
    assert matplotlib.rcParams["axes.facecolor"] == PALETTE["bg"]
    assert matplotlib.rcParams["savefig.facecolor"] == PALETTE["bg"]


def test_the_title_strip_is_an_axis_less_band_across_the_top(blank):
    strip = title_axes(blank)
    assert strip.get_xticks().size == 0
    assert not any(spine.get_visible() for spine in strip.spines.values())
    # It sits above the middle of the figure — it is a header, not a panel.
    assert strip.get_position().y0 > 0.5


def test_the_title_block_writes_its_heading_and_every_subtitle_line(blank):
    strip = title_axes(blank)
    draw_title_block(strip, "Luck Ledger", ["GB 23 - DET 31", "DTW: GB 95%"])
    written = [text.get_text() for text in strip.texts]
    assert written == ["Luck Ledger", "GB 23 - DET 31", "DTW: GB 95%"]
    assert strip.texts[0].get_fontweight() == "bold"
    assert strip.texts[1].get_color() == PALETTE["text_muted"]


# --------------------------------------------------------------------------
# round 10 — the stamp's corner
# --------------------------------------------------------------------------

# Anything darker than this is ink somebody drew. The surface is cream at 249
# mean and the stamp's own line is painted at 140, so a threshold between the
# two separates "the title ran under the stamp" from "the stamp is the only
# thing here".
FOREIGN_INK = 120


def ink_mask(path, threshold: int = FOREIGN_INK) -> np.ndarray:
    pixels = np.asarray(Image.open(path).convert("RGB"), dtype=float).mean(axis=2)
    return pixels < threshold


def test_the_stamp_is_painted_into_the_top_right_corner(blank, tmp_path):
    """Write-up round 3: back to the MLB simulator's corner.

    Round 10 conceded the top-right because document 63 measured the title
    running under the stamp there on 2,325 of 2,759 Strict distribution
    figures. the maintainer reversed that on 2026-08-30: the corner is the one the
    sibling product uses, and the collision is bought off with
    `reserve_stamp_strip` — which now grows the canvas at the *top* — rather
    than by moving the stamp somewhere no reader looks first.
    """
    from nfl_simulator.style import stamp_box

    path = tmp_path / "corner.png"
    finalize(blank, path)
    image = Image.open(path)
    width, height = image.size
    left, top, right, bottom = stamp_box(image.size, WATERMARK)

    assert bottom < height * 0.5, "the stamp is in the top half of the image"
    assert left > width * 0.5, "and in the right half of it"
    assert top >= 0 and right <= width, "the whole block is on the canvas"
    # And it is actually there: the block's box is the only dark thing on a
    # blank figure, and it is not empty.
    painted = np.asarray(image.convert("RGB"), dtype=float).mean(axis=2)
    assert painted[top:bottom, left:right].min() < 200


def test_the_mark_is_anchored_to_the_top_right_corner_itself(blank, tmp_path):
    """The badge's own box, not the block's: 2% in from the right, 2% down.

    The credit line hangs *below* the mark, so the pixel nearest the corner is
    the badge. Both tolerances are the same 2% because the margin the geometry
    is written against is 2% of the width and ~1.2% of the height.
    """
    from nfl_simulator.style import stamp_box

    path = tmp_path / "anchored.png"
    finalize(blank, path)
    width, height = Image.open(path).size
    _left, _top, right, _bottom = stamp_box((width, height), WATERMARK)
    saturated = spread(path) > SATURATED
    rows = np.nonzero(saturated.any(axis=1))[0]
    columns = np.nonzero(saturated.any(axis=0))[0]

    assert rows.size and columns.size, "the badge was not drawn at all"
    assert rows.min() <= height * 0.02, "the badge's top edge is not at the top"
    # the maintainer 2026-08-31: the badge is centred over the credit's text, so its
    # column centre matches the credit box's centre rather than its right edge.
    from nfl_simulator.style import stamp_box as _sb

    box_left, _t, box_right, _b = _sb((width, height), WATERMARK)
    badge_cx = (columns.min() + columns.max()) / 2
    assert abs(badge_cx - (box_left + box_right) / 2) <= 4, "the badge is not centred on the credit"


def test_the_mark_is_a_share_of_the_image_width(blank, tmp_path):
    """the maintainer 2026-08-30 (second pass): 2.5 credit lines still read as a
    thumbnail, because the credit line itself is small. The mark now scales
    with the image — `STAMP_LOGO_WIDTH_SHARE` of its width — with the credit
    lines as a floor so a tiny image never loses the mark entirely.

    the maintainer 2026-08-31 (round 5): 0.045 of the width made the mark sit *over* the
    title band rather than in it. The share drops to 0.035 so the stamp slots
    into the band the way `15_defense_shrinkage.png` already did — smaller, and
    the same block on every figure. The credit-line floor is untouched, so a
    narrow image still keeps its mark.
    """
    from nfl_simulator.style import STAMP_LOGO_WIDTH_SHARE, _mark_height

    assert STAMP_LOGO_WIDTH_SHARE == 0.035
    # Unit: on a wide image the width share wins; on a narrow one the floor.
    assert _mark_height(10, 2000) == 70
    assert _mark_height(10, 100) == 25

    path = tmp_path / "scale.png"
    finalize(blank, path)
    image = Image.open(path)
    width = image.size[0]

    badge_rows = np.nonzero((spread(path) > SATURATED).any(axis=1))[0]
    badge_h = badge_rows.max() - badge_rows.min() + 1
    # The measured ink can run a little under the box (transparent badge edge).
    assert badge_h >= 0.029 * width


def test_the_strip_is_reserved_when_a_figure_reaches_into_the_stamp_s_corner(blank, tmp_path):
    """A header in the top-right corner buys the stamp a strip of its own.

    `bbox_inches="tight"` crops to what the figure drew, so how close anything
    comes to the top edge is a per-game fact. Rather than ask a figure to leave
    room for a stamp it cannot see, the room is made on the saved pixels — and
    since round 3 of the write-up it is made *above* the figure rather than
    below it, because that is the side the stamp is anchored to.
    """
    from nfl_simulator.style import stamp_box

    blank.text(0.99, 0.99, "a header in the corner", ha="right", va="top", color="#1A1A1A")
    bare = tmp_path / "bare.png"
    # The card's path — `bbox_inches=None`, because its square shape is the
    # point of it — which is also the one that crops nothing away and so leaves
    # a header wherever the figure put it.
    blank.savefig(bare, dpi=200, bbox_inches=None, facecolor=PALETTE["bg"], edgecolor="none")
    stamped = finalize(blank, tmp_path / "reserved.png", bbox_inches=None, close=False)

    grown = Image.open(stamped)
    assert grown.height > Image.open(bare).height, "the strip was borrowed, not made"
    left, top, right, bottom = stamp_box(grown.size, WATERMARK)
    under = np.asarray(grown.convert("RGB"), dtype=float).mean(axis=2)[top:bottom, left:right]
    assert (under >= FOREIGN_INK).all(), "the header is still under the stamp"


def test_a_wide_title_does_not_run_under_the_stamp(blank, tmp_path):
    """The collision document 63 measured, on the widest title the suite draws.

    Checked over the block :func:`apply_watermark` says it painted, with the
    mark suppressed, so the only ink that box may contain is the credit's own
    mid-grey at 140. The box comes from the painter itself because recomputing
    it on the stamped PNG would anchor on the stamp's own ink.
    """
    from nfl_simulator.style import apply_watermark, reserve_stamp_strip

    draw_title_block(
        title_axes(blank, height_frac=0.2),
        "Los Angeles Chargers at Houston Texans — wild-card round, 2024",
        ["Deserved-to-win share across 160,000 re-simulations of the Full edition"],
    )
    path = tmp_path / "wide_title.png"
    blank.savefig(path, dpi=200, bbox_inches="tight", facecolor=PALETTE["bg"])
    reserve_stamp_strip(path, WATERMARK)
    left, top, right, bottom = apply_watermark(path, text=WATERMARK)

    image = Image.open(path)
    block = np.asarray(image.convert("RGB"), dtype=float).mean(axis=2)[
        max(0, top) : bottom, left:right
    ]
    assert top >= 0, "the block ran off the canvas"
    assert block.min() < 240, "the credit itself was not painted"
    assert (block >= FOREIGN_INK).all(), "the title is under the stamp"


# --------------------------------------------------------------------------
# the brand logo — the packaged asset
# --------------------------------------------------------------------------

# 60 KB is the budget a repo can carry in every wheel without anybody noticing,
# and 400 px is more than twice the width the mark is ever drawn at (the stamp
# asks for a logo about 1.6 text lines tall — tens of pixels, not hundreds).
LOGO_BYTE_BUDGET = 60 * 1024
LOGO_MAX_WIDTH = 400


def test_the_brand_logo_ships_inside_the_package():
    """It lives under `src/nfl_simulator/assets/`, so the wheel carries it.

    A logo read from a path outside the package is a logo that exists on the
    machine that made the figures and nowhere else.
    """
    from nfl_simulator import style
    from nfl_simulator.style import BRAND_LOGO

    assert BRAND_LOGO.exists()
    assert BRAND_LOGO.parent.parent == Path(style.__file__).parent


def test_the_brand_logo_is_rgba_with_a_transparent_background():
    """The badge is pasted onto a cream figure, so its corners must be see-through."""
    from nfl_simulator.style import BRAND_LOGO

    logo = Image.open(BRAND_LOGO)
    assert logo.mode == "RGBA"
    assert logo.getpixel((0, 0))[3] == 0


def test_the_brand_logo_stays_inside_its_size_budget():
    from nfl_simulator.style import BRAND_LOGO

    assert BRAND_LOGO.stat().st_size <= LOGO_BYTE_BUDGET
    assert Image.open(BRAND_LOGO).width <= LOGO_MAX_WIDTH


# --------------------------------------------------------------------------
# the brand logo — the stamp
# --------------------------------------------------------------------------

# The cream is (252, 250, 246) and the credit's ink is a neutral (140, 140, 140):
# both have a channel spread under 10. The badge is navy, blue and green, and
# even at the 85% the stamp pastes it with it keeps a spread of tens. So
# "saturated" separates the mark from everything else the corner can contain.
SATURATED = 30


def spread(path) -> np.ndarray:
    """Per-pixel channel spread — 0 for grey and cream, large for the badge."""
    pixels = np.asarray(Image.open(path).convert("RGB"), dtype=float)
    return pixels.max(axis=2) - pixels.min(axis=2)


def test_the_brand_logo_is_stamped_on_every_figure_by_default(blank, tmp_path):
    """`finalize` needs no argument to carry the mark — that is the point of it.

    A logo that has to be passed is a logo that is on the figures whoever
    remembered it rendered, and missing from the rest.
    """
    path = tmp_path / "default.png"
    finalize(blank, path)
    assert spread(path).max() > SATURATED


def test_passing_logo_path_false_suppresses_the_mark(blank, tmp_path):
    """The sentinel exists so a figure can opt out without opting out of the credit."""
    path = tmp_path / "suppressed.png"
    finalize(blank, path, logo_path=False)
    assert spread(path).max() <= SATURATED
    # The credit line is still there.
    assert corner(path).mean(axis=2).min() < DARK


def test_the_mark_sits_above_the_credit_line(blank, tmp_path):
    """The MLB simulator's stack, right-aligned rather than centred.

    Beside the credit was round 10's arrangement, and it only worked because
    the block was anchored to the bottom edge, where a mark 1.6 lines tall
    could overhang the text rows into empty pixels. Anchored to the *top* edge
    the mark has to go above the line it belongs to, or the corner it is
    supposed to occupy is occupied by the word "Data".
    """
    from nfl_simulator.style import stamp_box

    path = tmp_path / "above_text.png"
    finalize(blank, path)
    image = Image.open(path)
    _left, top, _right, _bottom = stamp_box(image.size, WATERMARK)
    rows = np.nonzero((spread(path) > SATURATED).any(axis=1))[0]

    assert rows.size, "the badge was not drawn at all"
    assert rows.max() < top, "the badge runs into the credit line's rows"


def test_the_mark_does_not_move_the_credit_line(blank, tmp_path):
    """Constraint 3: adding the logo changes nothing about where the text lands.

    Compared over every *row* from the credit's own top edge down, which is all
    of the text and none of the badge. It was a column comparison until the
    stack replaced the row — the badge shares the credit's columns now and has
    its own rows, which is the mirror image of the arrangement round 10 shipped.

    What makes this hold is that `stamp_box` reserves the mark's rows whether or
    not a mark is drawn: the credit hangs at a fixed offset from the top edge,
    and `logo_path` decides only whether anything is painted above it.
    """
    from nfl_simulator.style import stamp_box

    with_logo = finalize(blank, tmp_path / "with.png", close=False)
    without = finalize(blank, tmp_path / "without.png", logo_path=False)

    a = np.asarray(Image.open(with_logo).convert("RGB"))
    b = np.asarray(Image.open(without).convert("RGB"))
    assert a.shape == b.shape
    top = stamp_box(Image.open(with_logo).size, WATERMARK)[1]
    assert np.array_equal(a[top:], b[top:])


def test_the_reserved_strip_covers_the_rows_the_mark_is_pasted_into(tmp_path):
    """The mark gets the same clean surface under it that the credit does.

    `reserve_stamp_strip` grows the canvas when the figure's ink reaches the
    stamp's box. The box grows *upward* for the badge now that the block is
    anchored to the top edge, so ink that only reaches the badge's rows — not
    the text's — has to buy the strip too.
    """
    from nfl_simulator.style import BRAND_LOGO, reserve_stamp_strip, stamp_box

    size = (1200, 800)
    image = Image.new("RGB", size, (252, 250, 246))
    text_top = stamp_box(size, WATERMARK)[1]
    badge_top = stamp_box(size, WATERMARK, logo_path=BRAND_LOGO)[1]
    assert badge_top < text_top, "the box did not grow upward for the badge"

    # Ink in the badge's rows only, hard against the right edge.
    ImageDraw.Draw(image).rectangle(
        [size[0] - 60, badge_top, size[0] - 1, text_top - 1], fill=(26, 26, 26)
    )
    path = tmp_path / "under_the_badge.png"
    image.save(path)
    reserve_stamp_strip(path, WATERMARK, logo_path=BRAND_LOGO)

    assert Image.open(path).height > size[1]


def test_the_mark_and_the_credit_line_do_not_touch(blank, tmp_path):
    """A badge sitting on the "D" of "Data" reads as one smudged glyph.

    The air between them is measured against the mark's own height rather than
    fixed, because both sides of the gap scale with the image. Vertical since
    the stack replaced the row.
    """
    from nfl_simulator.style import stamp_box

    path = tmp_path / "gap.png"
    finalize(blank, path)
    text_top = stamp_box(Image.open(path).size, WATERMARK)[1]
    saturated = spread(path) > SATURATED
    badge_rows = np.nonzero(saturated.any(axis=1))[0]
    badge_h = badge_rows.max() - badge_rows.min() + 1

    assert text_top - badge_rows.max() - 1 >= badge_h / 8


def test_the_stamp_sits_beside_a_title_not_above_it(blank, tmp_path):
    """the maintainer 2026-08-31 round 6: the mark's top aligns with the title band's
    top, and the divider rule is cut where the block crosses it."""
    from nfl_simulator.style import BRAND_LOGO, apply_watermark, reserve_stamp_strip

    ax = blank.axes[0] if blank.axes else blank.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.axis("off")
    blank.add_artist(
        __import__("matplotlib").lines.Line2D(
            [0.02, 0.98], [0.88, 0.88], color="#3a3a3a", lw=2, transform=blank.transFigure
        )
    )
    path = tmp_path / "titled.png"
    blank.savefig(path, dpi=200, bbox_inches="tight", facecolor=PALETTE["bg"])
    reserve_stamp_strip(path, WATERMARK, logo_path=BRAND_LOGO)

    image = Image.open(path)
    width, height = image.size
    dark = np.asarray(image.convert("RGB"), dtype=float).mean(axis=2)
    rule_top = int(np.nonzero((dark < 240).sum(axis=1) > width * 0.7)[0].min())

    left, top, right, bottom = apply_watermark(path, logo_path=BRAND_LOGO, text=WATERMARK)
    assert top >= 0, "the block ran off the canvas"
    # The block's top is the title band's top (the rule is the only title ink
    # on this synthetic figure).
    assert abs(top - rule_top) <= 4, "the mark's top is not in line with the title's"

    # The rule is cut inside the block's columns and intact to their left.
    after = np.asarray(Image.open(path).convert("RGB"), dtype=float).mean(axis=2)
    assert (after[rule_top - 1 : rule_top + 2, left + 4 : width - 4] > 240).all(), (
        "the rule still runs under the mark"
    )
    assert (after[rule_top : rule_top + 2, : left // 2] < 240).any(), (
        "the rule was erased everywhere, not just under the mark"
    )
