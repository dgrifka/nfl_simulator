"""Optional palette hooks in `nfl_simulator.style`, and the stamp that reads them.

`PALETTE` carries seven keys a downstream caller may set: `band`, `title_ink`,
`stamp_ink`, `stamp_disc`, `mark_disc`, `accent`, `accent_2`. Each defaults to
`None` or `False`, and with the defaults a figure renders exactly as it did
before the keys existed. These tests pin both halves.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex
from PIL import Image

from nfl_simulator import style
from nfl_simulator.style import (
    BRAND_LOGO,
    PALETTE,
    _rgb255,
    apply_watermark,
    draw_title_block,
    finalize,
    place_mark,
    title_axes,
)

HOOK_DEFAULTS = {
    "band": None,
    "title_ink": None,
    "stamp_ink": None,
    "stamp_disc": False,
    "mark_disc": False,
    "accent": None,
    "accent_2": None,
}
BAND = "#336619"


@pytest.fixture
def palette():
    """Hand the test `PALETTE`; restore it afterwards."""
    saved = dict(PALETTE)
    yield PALETTE
    PALETTE.clear()
    PALETTE.update(saved)


def _strip(figsize=(4, 3)):
    fig = plt.figure(figsize=figsize)
    tax = title_axes(fig)
    draw_title_block(tax, "Title", ["Subtitle"])
    return fig, tax


def test_the_hook_keys_exist_with_behave_as_before_defaults():
    for key, default in HOOK_DEFAULTS.items():
        assert key in PALETTE, f"PALETTE is missing hook key {key!r}"
        assert PALETTE[key] is default, f"PALETTE[{key!r}] default is {PALETTE[key]!r}"


def test_the_defaults_render_cream_with_no_band(palette, tmp_path):
    fig, _ = _strip()
    assert fig.patches == [], "a default figure must not gain a band patch"
    out = finalize(fig, tmp_path / "default.png", dpi=50)
    assert Image.open(out).convert("RGB").getpixel((5, 5)) == (0xFC, 0xFA, 0xF6)


def test_a_band_is_one_full_width_patch_and_the_title_wears_title_ink(palette):
    palette.update(band=BAND, title_ink="#FFFFFF")
    fig, tax = _strip()
    try:
        assert len(fig.patches) == 1
        band = fig.patches[0]
        assert band.get_x() == 0 and band.get_width() == 1
        assert band.get_y() + band.get_height() == pytest.approx(1.0)
        assert band.get_zorder() < 0
        inks = {t.get_text(): to_hex(t.get_color()) for t in tax.texts}
        assert inks == {"Title": "#ffffff", "Subtitle": "#ffffff"}
    finally:
        plt.close(fig)


def test_the_stamp_disc_puts_white_behind_the_mark(palette):
    plain = np.asarray(style._prepared_mark(BRAND_LOGO))
    ys, xs = np.nonzero(plain[..., 3])
    top, bottom, left, right = ys.min(), ys.max(), xs.min(), xs.max()

    palette["stamp_disc"] = True
    disc = np.asarray(style._prepared_mark(BRAND_LOGO))
    h, w = plain.shape[:2]
    assert disc.shape[0] == disc.shape[1] > max(h, w)
    oy, ox = (disc.shape[0] - h) // 2, (disc.shape[1] - w) // 2
    cy, cx = oy + (top + bottom) // 2, ox + (left + right) // 2
    # A 2-px ring just outside the mark's alpha bbox, on the four midlines
    # (the bbox corners fall outside a round disc).
    ring = np.array([
        disc[oy + top - 2, cx], disc[oy + top - 1, cx],
        disc[oy + bottom + 1, cx], disc[oy + bottom + 2, cx],
        disc[cy, ox + left - 2], disc[cy, ox + left - 1],
        disc[cy, ox + right + 1], disc[cy, ox + right + 2],
    ])
    assert (ring[:, :3] == 255).all(), "ring outside the mark is not white"
    assert (ring[:, 3] > 0).all(), "ring outside the mark is transparent"


def test_the_stamp_ink_is_read_at_call_time(palette, tmp_path):
    path = tmp_path / "ink.png"
    Image.new("RGB", (1200, 800), (0x33, 0x66, 0x19)).save(path)
    palette.update(band=BAND, stamp_ink="#FFFF00")
    apply_watermark(path)
    pixels = np.asarray(Image.open(path).convert("RGB")).astype(int)
    yellow = (pixels[..., 0] > 200) & (pixels[..., 1] > 200) & (pixels[..., 2] < 80)
    assert yellow.sum() > 30


def test_the_band_is_surface_not_ink_for_the_stamp_anchor(palette):
    image = Image.new("RGB", (400, 300), (0x33, 0x66, 0x19))
    assert style._top_ink(image, 300).any(), "without a band hook the green reads as ink"
    palette["band"] = BAND
    assert not style._top_ink(image, 300).any()


def test_the_rule_erase_paints_the_band_not_the_body(palette, tmp_path):
    # A band-coloured top with one dark full-width rule through the stamp's rows.
    width, height = 1200, 800
    image = Image.new("RGB", (width, height), _rgb255(PALETTE["bg"]))
    band_rgb = (0x33, 0x66, 0x19)
    image.paste(Image.new("RGB", (width, 120), band_rgb), (0, 0))
    rule_row = 40
    image.paste(Image.new("RGB", (width, 1), (20, 20, 20)), (0, rule_row))
    path = tmp_path / "rule.png"
    image.save(path)

    palette.update(band=BAND)
    left, _top, _right, _bottom = apply_watermark(path)
    out = np.asarray(Image.open(path).convert("RGB")).astype(int)
    erased = out[rule_row, left - 10 : left - 2]
    assert (erased == band_rgb).all(), f"rule erased to {erased[0]}, not the band"


def test_place_mark_puts_its_disc_one_below_the_image(palette):
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    try:
        rgba = np.zeros((40, 40, 4), dtype=np.uint8)
        ab = place_mark(ax, rgba, (0.5, 0.5), 30, disc=True, zorder=5)
        assert len(ax.collections) == 1
        assert ax.collections[0].get_zorder() == 4
        assert ab.get_zorder() == 5
        assert ab.offsetbox.get_zoom() == pytest.approx(30 / 40 * 72 / 100)
        ax2 = fig.add_subplot(2, 1, 2)
        place_mark(ax2, rgba, (0.5, 0.5), 30)
        assert len(ax2.collections) == 0
    finally:
        plt.close(fig)
