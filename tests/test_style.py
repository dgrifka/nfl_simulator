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
from PIL import Image

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

# The bottom-right corner the watermark is stamped into, as fractions of the
# saved image. `apply_watermark` places the block at 2% in from the right and
# ~1.2% up from the bottom, so a box this size contains all of it and nothing
# else. It was the *top* right until round 10 — document 63 measured the title
# running under it on 84-89% of distribution figures.
CORNER_W, CORNER_H = 0.20, 0.06

# Cream is (252, 250, 246); the watermark ink is mid-grey. Anything whose mean
# channel is below this is text rather than background.
DARK = 200


def corner(path) -> np.ndarray:
    pixels = np.asarray(Image.open(path).convert("RGB"), dtype=float)
    height, width = pixels.shape[:2]
    return pixels[int(height * (1 - CORNER_H)) :, int(width * (1 - CORNER_W)) :]


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
    assert BRAND_HANDLE == "@[TBD]"
    assert DATA_CREDIT == "Data: nflverse"
    assert WATERMARK == "Data: nflverse | @[TBD]"


def test_a_finalized_png_is_saved_on_the_cream_surface(blank, tmp_path):
    path = tmp_path / "surface.png"
    finalize(blank, path)
    top_left = Image.open(path).convert("RGB").getpixel((2, 2))
    assert top_left == (252, 250, 246)


def test_finalize_stamps_the_watermark_into_the_bottom_right_corner(blank, tmp_path):
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


def test_the_stamp_is_painted_into_the_bottom_right_corner(blank, tmp_path):
    """Round 10: the corner no artist is laid out in.

    Document 63 measured the title running under the top-right stamp on 2,325
    of 2,759 Strict distribution figures. The stamp is painted on the saved
    pixels after layout, so the title cannot see it and move; the corner it
    lives in is the only thing that can change.
    """
    from nfl_simulator.style import stamp_box

    path = tmp_path / "corner.png"
    finalize(blank, path)
    image = Image.open(path)
    width, height = image.size
    left, top, right, bottom = stamp_box(image.size, WATERMARK)

    assert top > height * 0.5, "the stamp is in the bottom half of the image"
    assert left > width * 0.5, "and in the right half of it"
    assert bottom <= height and right <= width, "the whole block is on the canvas"
    # And it is actually there: the block's box is the only dark thing on a
    # blank figure, and it is not empty.
    painted = np.asarray(image.convert("RGB"), dtype=float).mean(axis=2)
    assert painted[top:bottom, left:right].min() < 200


def test_the_strip_is_reserved_when_a_figure_reaches_into_the_stamp_s_corner(blank, tmp_path):
    """A footer in the bottom-right corner buys the stamp a strip of its own.

    `bbox_inches="tight"` crops to what the figure drew, so how close anything
    comes to the bottom edge is a per-game fact. Rather than ask a figure to
    leave room for a stamp it cannot see, the room is made on the saved pixels.
    """
    from nfl_simulator.style import stamp_box

    blank.text(0.99, 0.01, "a footer in the corner", ha="right", va="bottom", color="#1A1A1A")
    bare = tmp_path / "bare.png"
    # The card's path — `bbox_inches=None`, because its square shape is the
    # point of it — which is also the one that crops nothing away and so leaves
    # a footer wherever the figure put it.
    blank.savefig(bare, dpi=200, bbox_inches=None, facecolor=PALETTE["bg"], edgecolor="none")
    stamped = finalize(blank, tmp_path / "reserved.png", bbox_inches=None, close=False)

    grown = Image.open(stamped)
    assert grown.height > Image.open(bare).height, "the strip was borrowed, not made"
    left, top, right, bottom = stamp_box(grown.size, WATERMARK)
    under = np.asarray(grown.convert("RGB"), dtype=float).mean(axis=2)[top:bottom, left:right]
    assert (under >= FOREIGN_INK).all(), "the footer is still under the stamp"


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
