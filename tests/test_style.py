"""The house style — the parts of it a reader can check without looking.

A style module is mostly taste, and taste is not testable. What *is* testable is
the handful of promises the rest of the product builds on: the surface a figure
is saved on, the fact that a saved PNG carries its data credit, and the blend
the clash rule uses. Those are the ones that break silently — a watermark that
stops being stamped looks exactly like a figure nobody has looked at yet.
"""

from __future__ import annotations

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

# The top-right corner the watermark is stamped into, as fractions of the saved
# image. `_apply_watermark` places the block at 2% in from the right and ~1.2%
# down from the top, so a box this size contains all of it and nothing else.
CORNER_W, CORNER_H = 0.12, 0.06

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
    assert BRAND_HANDLE == "@[TBD]"
    assert DATA_CREDIT == "Data: nflverse"
    assert WATERMARK == "Data: nflverse | @[TBD]"


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
