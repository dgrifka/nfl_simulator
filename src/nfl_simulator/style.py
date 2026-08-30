"""The house style — one cream surface, one title grammar, one watermark.

Ported from the baseball simulator's ``Simulator/style.py`` so the two projects
read as one hand. Only the style crosses over; nothing here imports from that
repo, and the outcome-semantic keys it carries (``single``/``xbh``/``hr``) are
dropped because football has no such ladder.

Three things are load-bearing rather than decorative:

* **The surface is warm cream, not white.** Every figure is saved on
  ``PALETTE['bg']`` and every fill is separated from its neighbour by a hairline
  of it, so adjacent bars never bleed into one shape.
* **The watermark is stamped after ``savefig``, in PIL, not drawn into the
  figure.** A figure-coordinate watermark moves with figure size, dpi and
  ``bbox_inches="tight"``; a pixel-coordinate one lands in the same corner of
  every PNG the product ships.
* **The data credit is part of the watermark, not an option.** nflverse asks for
  credit for the play-by-play this whole repo runs on, so it travels with the
  image rather than with the post it was attached to.

The handle is deliberately a placeholder. ``BRAND_HANDLE`` reads ``@[TBD]``
until the maintainer names the account; a figure that shipped with an invented handle
would be pointing readers at somebody else.
"""

from __future__ import annotations

import contextlib
from functools import cache
from pathlib import Path

import numpy as np
from matplotlib import font_manager, rcParams
from matplotlib.colors import to_hex, to_rgb

PALETTE = {
    "bg": "#FCFAF6",  # subtle warm cream — figure & axes facecolor
    "text": "#1A1A1A",  # primary text
    "text_muted": "#6B6258",  # subtitle / footer / muted labels
    "grid": "#D8D3C8",  # gridlines (dashed, alpha 0.6)
    "spine": "#C7C0B4",  # the spines that survive
    # Verdict semantics. `bad` is the flip — the scoreboard and the adjudication
    # disagree — and `good` is the scoreboard holding. Neither is ever used for a
    # team: a team is an identity, and identity never borrows a status colour.
    "good": "#2E7D32",
    "bad": "#C03A2B",
    # Totals and other not-a-team marks. Chosen by measurement, not by taste:
    # it is at least 0.281 in RGB from every one of the 32 club primaries (the
    # nearest is Minnesota's #4F2683), well past the 0.20 the clash rule calls a
    # collision. Ink would be simpler and is wrong — #1A1A1A is 0.087 from
    # Chicago's navy, 0.124 from Houston's and 0.177 from the Raiders' black, so
    # a total drawn in ink is a colour four clubs also wear.
    "anchor": "#5E5B55",
    "row_alt": "#F4EFE6",
}

# Bar alphas. The home team's fill is the solid one and the away team's the
# lighter, which is a second, non-colour cue for which side is which — and the
# same cue the baseball run-distribution chart uses.
HOME_ALPHA = 0.78
AWAY_ALPHA = 0.55

# Brand. `WATERMARK` is the one string stamped on every PNG.
BRAND_HANDLE = "@nfl_simulator"
DATA_CREDIT = "Data: nflverse"
WATERMARK = f"{DATA_CREDIT} | {BRAND_HANDLE}"

# The mark itself, packaged rather than referenced. It sits inside
# ``src/nfl_simulator`` so hatchling puts it in the wheel: a logo read from a
# path outside the package is a logo that exists on the machine that made the
# figures and nowhere else. RGBA with a transparent field, 400 px on its long
# side and quantised to 128 colours, which is the whole of this flat badge and
# a quarter of the bytes a truecolour save costs.
BRAND_LOGO = Path(__file__).parent / "assets" / "logo.png"

# Ruling R-4's two editions (document 58 §2), by the public name each wears.
# The audit arms `"strict+dp"` and `"strict+rd"` are deliberately absent: they
# are callable in the simulator and were never named, so an image cannot claim
# to be one.
EDITION_NAMES = {"strict": "Strict", "full": "Full"}


def edition_stamp(edition: str) -> str:
    """`"Full edition · Data: nflverse | @[TBD]"` — the corner of every PNG.

    The edition goes **before** the credit rather than after it because the
    credit is the fixed part: a reader who has seen one of these images already
    knows where nflverse's name sits, and the word that changes between two
    images of the same game is the one worth reading first.
    """
    if edition not in EDITION_NAMES:
        raise ValueError(
            f"{edition!r} is not an edition anybody named; they are "
            f"{list(EDITION_NAMES)} (document 58 §2)."
        )
    return f"{EDITION_NAMES[edition]} edition \u00b7 {WATERMARK}"


# Font preference order. Matplotlib walks the list and uses the first family
# present on the system; DejaVu Sans ships with matplotlib, so the last entry
# always resolves and a machine without Inter still gets a readable figure.
_BODY_PREFERENCE = ["Inter", "IBM Plex Sans", "Helvetica Neue", "DejaVu Sans"]
_HEADING_PREFERENCE = ["Oswald", "Barlow Condensed", "DejaVu Sans"]

_BASE_STYLE_APPLIED = False


@cache
def _installed(preference: tuple[str, ...]) -> tuple[str, ...]:
    """The families from ``preference`` this machine actually has, plus DejaVu.

    Matplotlib is happy to be handed families it cannot find — it walks the list
    — but it logs a warning for *every text object* drawn, which buries a
    driver's real output under a thousand `findfont` lines. Resolving the list
    once here means the rcParam only ever names fonts that exist.
    """
    installed = {font.name for font in font_manager.fontManager.ttflist}
    found = tuple(family for family in preference if family in installed)
    return found or ("DejaVu Sans",)


def body_font() -> list[str]:
    """The body family list, resolved against what is installed."""
    return list(_installed(tuple(_BODY_PREFERENCE)))


def apply_base_style() -> None:
    """Set rcParams for the cream surface and the body font.

    Safe to call repeatedly — rcParams overwrite idempotently. The first call
    also nudges ``font_manager`` into scanning the system so an installed Inter
    or Oswald is visible to the family lists above.
    """
    global _BASE_STYLE_APPLIED

    if not _BASE_STYLE_APPLIED:
        # A machine without Inter is not an error — the preference list falls
        # through to DejaVu — so a failed probe is suppressed rather than raised.
        with contextlib.suppress(Exception):
            font_manager.fontManager.findfont(_BODY_PREFERENCE[0], fallback_to_default=True)
        _BASE_STYLE_APPLIED = True

    rcParams["figure.facecolor"] = PALETTE["bg"]
    rcParams["axes.facecolor"] = PALETTE["bg"]
    rcParams["savefig.facecolor"] = PALETTE["bg"]

    rcParams["font.family"] = body_font()
    rcParams["font.size"] = 11
    rcParams["text.color"] = PALETTE["text"]

    rcParams["axes.edgecolor"] = PALETTE["spine"]
    rcParams["axes.labelcolor"] = PALETTE["text"]
    rcParams["axes.titlecolor"] = PALETTE["text"]
    rcParams["axes.titleweight"] = "bold"
    rcParams["axes.spines.top"] = False
    rcParams["axes.spines.right"] = False
    rcParams["axes.linewidth"] = 0.8

    rcParams["xtick.color"] = PALETTE["text_muted"]
    rcParams["ytick.color"] = PALETTE["text_muted"]
    rcParams["xtick.labelsize"] = 10
    rcParams["ytick.labelsize"] = 10

    rcParams["grid.color"] = PALETTE["grid"]
    rcParams["grid.linestyle"] = "--"
    rcParams["grid.linewidth"] = 0.8
    rcParams["grid.alpha"] = 0.6

    rcParams["legend.frameon"] = False
    rcParams["legend.fontsize"] = 10
    rcParams["legend.labelcolor"] = PALETTE["text"]


def heading_font() -> list[str]:
    """The heading family list, for use as ``fontfamily=...``."""
    return list(_installed(tuple(_HEADING_PREFERENCE)))


def rc_style() -> dict:
    """The style as an rcParams mapping, for ``mpl.rc_context``.

    ``apply_base_style`` mutates global rcParams, which is what a script wants
    and what a library must not do. The figure functions take this instead, so a
    caller's own rcParams survive the import.
    """
    return {
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor": PALETTE["bg"],
        "savefig.facecolor": PALETTE["bg"],
        "font.family": body_font(),
        "font.size": 10,
        "text.color": PALETTE["text"],
        "axes.labelcolor": PALETTE["text_muted"],
        "axes.edgecolor": PALETTE["spine"],
        "xtick.color": PALETTE["text_muted"],
        "ytick.color": PALETTE["text_muted"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "figure.dpi": 160,
    }


# Euclidean distance in RGB, and the blend applied when a pair falls under it.
# Both are the baseball run-distribution chart's numbers
# (`Simulator/visualizations.py:453-457`). `CLASH_DARKEN` is this project's
# own: the baseball chart never ran a contrast check, and on a cream surface
# blending toward white costs the very contrast a last resort has to pass.
CLASH_DISTANCE = 0.20
CLASH_LIGHTEN = 0.45
CLASH_DARKEN = 0.45


def colour_distance(first: str, second: str) -> float:
    """Euclidean distance between two colours in RGB, each channel on 0-1."""
    a, b = to_rgb(first), to_rgb(second)
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) ** 0.5


# --------------------------------------------------------------------------
# colour-vision separation — the `dataviz` skill's computable checks, ported
# --------------------------------------------------------------------------

# The RGB rule above is cheap and blind. It cannot see that the Jets' #003F2D
# and the 49ers' #AA0000 — 0.42 apart in RGB, comfortably "separate" — collapse
# onto each other for a reader with protanopia. What follows is the `dataviz`
# skill's own validator (`scripts/validate_palette.js`), ported so a matchup can
# be checked before it is drawn rather than after it has shipped.
#
# Machado, Oliveira & Fernandes (2009) at severity 1.0, in **linear** RGB. The
# simulation model is part of the calibration, not an implementation detail:
# swapping in another one moves borderline pairs and would need the floors below
# re-derived, which is why the matrices are quoted here rather than pulled from
# whatever a colour library happens to ship.
MACHADO = {
    "protan": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deutan": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
    "tritan": (
        (1.255528, -0.076749, -0.178779),
        (-0.078411, 0.930809, 0.147602),
        (0.004733, 0.691367, 0.303900),
    ),
}
CVD_KINDS = ("protan", "deutan", "tritan")

# OKLab ΔE ×100. The skill's numbers, unchanged: 8 is the target for a
# colourblind reader and 6 is the floor, legal only where a second, non-colour
# encoding carries the same identity. Every figure in this product carries one —
# a legend, a direct label, or the club's own mark — so the floor is the gate
# and the band between 6 and 8 is reported rather than refused. 15 is a hard
# gate under normal vision: below it a full-colour reader cannot tell the two
# apart either, and no amount of labelling makes two identical bars two bars.
CVD_TARGET = 8.0
CVD_FLOOR = 6.0
NORMAL_FLOOR = 15.0
# WCAG, against whatever surface the figure is drawn on.
CONTRAST_MIN = 3.0


def _linear(colour: str) -> tuple[float, float, float]:
    """sRGB to linear RGB, the space every transform below works in."""

    def channel(value: float) -> float:
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    return tuple(channel(component) for component in to_rgb(colour))


def _oklab(rgb) -> tuple[float, float, float]:
    """Linear RGB to OKLab."""
    red, green, blue = rgb
    long_ = np.cbrt(0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue)
    medium = np.cbrt(0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue)
    short = np.cbrt(0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue)
    return (
        0.2104542553 * long_ + 0.7936177850 * medium - 0.0040720468 * short,
        1.9779984951 * long_ - 2.4285922050 * medium + 0.4505937099 * short,
        0.0259040371 * long_ + 0.7827717662 * medium - 0.8086757660 * short,
    )


def simulate_cvd(colour: str, kind: str) -> tuple[float, float, float]:
    """``colour`` as a reader with ``kind`` sees it, in linear RGB."""
    rgb = _linear(colour)
    matrix = MACHADO[kind]
    return tuple(
        min(1.0, max(0.0, sum(row[index] * rgb[index] for index in range(3)))) for row in matrix
    )


def delta_e(first: str, second: str, kind: str | None = None) -> float:
    """Euclidean distance between two colours in OKLab, ×100.

    ``kind=None`` is unsimulated vision; otherwise the pair is put through the
    named colour-vision simulation first.
    """
    a = _oklab(simulate_cvd(first, kind) if kind else _linear(first))
    b = _oklab(simulate_cvd(second, kind) if kind else _linear(second))
    return 100.0 * float(np.hypot(np.hypot(a[0] - b[0], a[1] - b[1]), a[2] - b[2]))


def separations(first: str, second: str) -> dict[str, float]:
    """The four readings of how far apart two colours are.

    ``normal`` plus one per colour-vision simulation. Returned together because
    the decision is on the whole set — a pair that separates beautifully for a
    full-colour reader and not at all for a protan reader is not a usable pair,
    and a single worst-case number hides which reader it failed.
    """
    return {
        "normal": delta_e(first, second),
        **{kind: delta_e(first, second, kind) for kind in CVD_KINDS},
    }


def relative_luminance(colour: str) -> float:
    red, green, blue = _linear(colour)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(first: str, second: str) -> float:
    """WCAG contrast ratio, which is what a mark needs against its surface."""
    high, low = sorted((relative_luminance(first), relative_luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def separated(first: str, second: str) -> bool:
    """Whether every reader can tell two marks apart.

    All four readings, not their average: a pair is only as usable as its worst
    reader. Contrast against the surface is a separate question — it is a
    property of one mark rather than of a pair — and is checked by the caller
    that is free to move a colour.
    """
    readings = separations(first, second)
    return readings["normal"] >= NORMAL_FLOOR and all(
        readings[kind] >= CVD_FLOOR for kind in CVD_KINDS
    )


def reads_on(colour: str, surface: str | None = None) -> bool:
    """Whether a mark clears the WCAG floor against the surface it is drawn on."""
    return contrast_ratio(colour, PALETTE["bg"] if surface is None else surface) >= CONTRAST_MIN


def lighten(color: str, amount: float = 0.5) -> str:
    """Blend ``color`` toward white by ``amount`` (0 = unchanged, 1 = white).

    Returns a hex string rather than the baseball version's RGB tuple: every
    caller here hands the result straight back to matplotlib as a colour, and a
    tuple that has to be re-hexed at each call site is where a stray
    ``to_hex`` gets forgotten.
    """
    r, g, b = to_rgb(color)
    return to_hex((r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount))


def darken(color: str, amount: float = 0.45) -> str:
    """Blend ``color`` toward black by ``amount`` (0 = unchanged, 1 = black).

    The mirror of :func:`lighten`, and the one the cream surface usually wants.
    Lightening a colour toward white moves it *toward* the background: Kansas
    City's red against San Francisco's cannot be separated by lightening either
    of them, because every candidate light enough to separate is too light to
    read on ``PALETTE["bg"]``. Darkening separates in lightness and gains
    contrast at the same time.
    """
    red, green, blue = to_rgb(color)
    keep = 1.0 - amount
    return to_hex((red * keep, green * keep, blue * keep))


def title_axes(
    fig, *, height_frac: float = 0.14, top_pad: float = 0.015, right_reserve: float = 0.12
):
    """Reserve an axis-less strip across the top of ``fig`` for a title block.

    Titles placed in a plot's own coordinate space collide with whatever is
    tallest in it. A dedicated strip gives them their own space, and
    ``right_reserve`` keeps the corner the watermark is stamped into clear.
    """
    width = max(0.50, 1.0 - 0.04 - right_reserve)
    ax = fig.add_axes([0.04, 1.0 - height_frac - top_pad, width, height_frac])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("none")
    return ax


def draw_title_block(
    ax,
    title: str,
    subtitle_lines: str | list[str] | None = None,
    *,
    title_size: float = 20,
    subtitle_size: float = 11,
    rule: bool = True,
):
    """Fill a strip from :func:`title_axes`: heading, divider rule, subtitles.

    Subtitle lines are a list rather than one newline-joined string so each line
    can be measured and spaced on its own — the score line and the caption under
    it are different weights of the same block.
    """
    if subtitle_lines is None:
        subtitle_lines = []
    elif isinstance(subtitle_lines, str):
        subtitle_lines = [subtitle_lines]

    ax.text(
        0.0,
        0.92,
        title,
        fontsize=title_size,
        fontweight="bold",
        color=PALETTE["text"],
        ha="left",
        va="top",
        fontfamily=heading_font(),
        transform=ax.transAxes,
    )

    cursor_y = 0.55
    if rule:
        ax.plot(
            [0.0, 1.0],
            [cursor_y, cursor_y],
            color=PALETTE["grid"],
            linewidth=0.8,
            transform=ax.transAxes,
            clip_on=False,
        )
        cursor_y -= 0.10

    for line in subtitle_lines:
        ax.text(
            0.0,
            cursor_y,
            line,
            fontsize=subtitle_size,
            color=PALETTE["text_muted"],
            ha="left",
            va="top",
            transform=ax.transAxes,
        )
        cursor_y -= 0.30
    return ax


# Where the stamp lives on the saved pixels, as fractions of the image. The
# horizontal inset is the one it has always had; ``STAMP_INSET`` is measured
# down from the **top** edge again — round 10 measured it up from the bottom,
# and the maintainer reversed that on 2026-08-30 (document 60 §15).
STAMP_MARGIN = 0.02
STAMP_INSET = 0.012
# The font, as a fraction of the image's short side, so a card and a
# distribution wear the same stamp at their own sizes. The other two numbers of
# the stamp hang off it rather than off the image: the mark is
# `STAMP_LOGO_RATIO` credit lines tall, and the air under it is
# `STAMP_GAP_RATIO` of the mark. Chaining them means the three parts stay one
# block however the font resolves on the machine doing the rendering — a
# machine with Inter and one without would otherwise space them differently.
#
# 2.5 rather than round 10's 1.6, same ruling. At 1.6 the badge read as a bullet
# in front of the credit rather than as a mark, which is the whole reason it is
# on the image.
STAMP_FONT_SCALE = 0.0095
STAMP_LOGO_RATIO = 2.5
STAMP_GAP_RATIO = 0.25

# the maintainer 2026-08-30, second pass: 2.5 credit lines still read as a thumbnail,
# because the credit line itself is small. The mark scales with the image —
# this share of its width — and the credit-line ratio above becomes the floor.
STAMP_LOGO_WIDTH_SHARE = 0.045

# Anything darker than this, on the cream, is an artist rather than the surface.
# Used only to decide whether the stamp's corner is occupied.
_SURFACE_INK = 240


def _stamp_font(reference: int):
    """The stamp's face at the size the image asks for, or matplotlib's own."""
    from PIL import ImageFont

    size = max(10, int(reference * STAMP_FONT_SCALE))
    try:
        import matplotlib.font_manager as fm

        return ImageFont.truetype(fm.findfont(fm.FontProperties(family="DejaVu Sans")), size)
    except Exception:  # pragma: no cover - a machine with no findable font
        return ImageFont.load_default()


def _stamp_gap(logo_height: int) -> int:
    """The air between the mark and the credit line. Never less than two pixels."""
    return max(2, round(logo_height * STAMP_GAP_RATIO))


def _mark_height(text_height: int, image_width: int) -> int:
    """The mark's pixel height above a credit line ``text_height`` tall.

    ``STAMP_LOGO_WIDTH_SHARE`` of the image's width, floored at
    ``STAMP_LOGO_RATIO`` credit lines so a small image never loses the mark.

    Split out of :func:`_logo_geometry` because the *height* is a fact about the
    credit line and the *width* is a fact about the file: :func:`stamp_box`
    reserves the mark's rows on every figure, including the ones that will never
    open a logo, and it must not have to name a file to do it.
    """
    return max(
        1,
        round(text_height * STAMP_LOGO_RATIO),
        round(image_width * STAMP_LOGO_WIDTH_SHARE),
    )


def _logo_geometry(logo_path: str | Path, text_height: int, image_width: int) -> tuple[int, int]:
    """The pixels the mark is drawn at beside a credit line ``text_height`` tall.

    Only the header of the PNG is read — the aspect ratio is all this needs, and
    it is asked for twice per figure (once to reserve the strip, once to paint).
    """
    from PIL import Image

    height = _mark_height(text_height, image_width)
    with Image.open(logo_path) as probe:
        width = max(1, round(height * probe.width / probe.height))
    return width, height


def stamp_box(
    size: tuple[int, int], text: str = WATERMARK, *, logo_path: str | Path | None = None
) -> tuple[int, int, int, int]:
    """The pixels the credit stamp will take, as ``(left, top, right, bottom)``.

    Public because two callers outside :func:`apply_watermark` need it: the
    strip reservation below, which has to know whether anything was drawn there
    before the stamp covers it, and the corpus read, which counts foreign ink
    inside it on a written PNG.

    **Top-right again since write-up round 3.** Round 10 moved the stamp to the
    bottom-right because document 63 measured the title running under it in the
    top-right on 2,325 of 2,759 Strict distribution figures and 1,016 of 1,139
    Full ones. the maintainer reversed that on 2026-08-30: the corner is the sibling
    MLB simulator's, and the collision document 63 measured is now bought off
    by :func:`reserve_stamp_strip`, which grows the canvas *above* the figure
    rather than by conceding the corner a reader looks at first.

    The mark stacks **above** the credit, right-aligned with it — the MLB
    simulator's arrangement — rather than beside it. Anchored to the top edge a
    mark 2.5 lines tall cannot overhang the credit's rows the way round 10's
    1.6-line mark overhung them at the bottom: it would run off the canvas.

    The credit's own rows do not depend on ``logo_path``. They hang a fixed
    mark-plus-gap below the top inset whether a mark is drawn or not, so a
    figure rendered with ``logo_path=False`` puts its credit in exactly the
    pixels a marked one does. ``logo_path`` grows the returned box **upward**
    to cover the mark's rows, which is what the strip reservation and the
    corpus read need and what the painter must not use.
    """
    from PIL import Image, ImageDraw

    width, height = size
    font = _stamp_font(min(width, height))
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    mark_h = _mark_height(text_h, width)
    block_top = int(height * STAMP_INSET)
    text_top = block_top + mark_h + _stamp_gap(mark_h)
    right = width - int(width * STAMP_MARGIN)
    left = right - text_w
    top = block_top if logo_path is not None else text_top
    return (left, top, right, text_top + text_h)


def reserve_stamp_strip(
    filepath: str | Path, text: str = WATERMARK, *, logo_path: str | Path | None = None
) -> None:
    """Grow the canvas at the top if the figure reaches into the stamp's box.

    ``bbox_inches="tight"`` crops to whatever the figure drew, so how close the
    title comes to the top edge is a per-game fact — a two-line subtitle and a
    one-line one crop to two different images. Rather than ask every figure to
    leave room for a stamp it cannot see, the room is made here, in the same
    pixels the stamp is painted into.

    **This is what buys back the corner round 10 gave up.** Document 63's
    finding was not that the top-right is a bad corner; it was that a title
    laid out before the stamp exists will run under it. Making the room on the
    saved pixels answers that without moving the mark somewhere no reader looks
    first, and it is the same trick round 10 used at the bottom, mirrored.

    Only the stamp's own columns are consulted. Growing every image to clear a
    title the stamp is nowhere near would change two fixed shapes for no
    legibility gain. ``logo_path`` is passed through to :func:`stamp_box` so
    those columns include the rows the brand mark is pasted into: the mark is
    stamped over whatever is under it at 85% opacity, so it needs the same clean
    surface the credit line does.
    """
    from PIL import Image

    filepath = Path(filepath)
    image = Image.open(filepath).convert("RGB")
    width, height = image.size
    left, top, _right, bottom = stamp_box(image.size, text, logo_path=logo_path)
    block_h = bottom - top
    # One block of air between the stamp and whatever is below it, so the credit
    # reads as its own line rather than as the title's first word.
    clearance = block_h

    columns = np.asarray(image, dtype=float)[:, max(0, left - clearance) :].mean(axis=2)
    occupied = np.nonzero((columns < _SURFACE_INK).any(axis=1))[0]
    highest = int(occupied.min()) if occupied.size else height
    if highest > bottom + clearance:
        return

    # How far the figure has to move down for the box's bottom to clear the ink.
    # The box is anchored to the top edge, so the added rows push the ink away
    # from it; `STAMP_INSET` of the added height follows the box down, and the
    # two spare pixels absorb that rounding.
    shift = bottom + clearance - highest
    shift = int(np.ceil(shift / (1.0 - STAMP_INSET))) + 2
    # A square image stays square: the two card figures are square by design
    # (their preview crop depends on it), so the same shift is added to the
    # width. The stamp is re-anchored to the new right edge when it is painted,
    # which also moves it clear of a centred title horizontally.
    new_width = width + shift if width == height else width
    canvas = Image.new("RGB", (new_width, height + shift), tuple(_rgb255(PALETTE["bg"])))
    canvas.paste(image, (0, shift))
    canvas.save(filepath)


def _rgb255(colour: str) -> tuple[int, int, int]:
    return tuple(int(round(channel * 255)) for channel in to_rgb(colour))


def apply_watermark(
    filepath: str | Path,
    *,
    logo_path: str | Path | None = None,
    text: str = WATERMARK,
) -> None:
    """Stamp the credit line — and optionally a logo — onto a saved PNG.

    Done in PIL on the saved pixels rather than in figure coordinates, because
    ``bbox_inches="tight"`` crops a figure by an amount that depends on how long
    its longest tick label happened to be. A corner in pixels is the same corner
    on every image the product ships.

    The corner is the **top-right** again since write-up round 3 — see
    :func:`stamp_box` for what the corpus measured there and what now answers
    it. ``position`` and ``y_pct`` are still gone: the corner is settled, and a
    parameter nothing passes is a corner a figure could still be stamped into
    by accident.

    ``logo_path=None`` draws the text line alone. This is the low-level exit and
    it stays explicit about the mark; the product's default — every figure wears
    :data:`BRAND_LOGO` — lives one level up, in :func:`finalize`.

    The mark is pasted **above** the credit and right-aligned with it, at 2.5
    line heights — the MLB simulator's stack, which right-aligns rather than
    centring the narrower element so the corner itself is the mark. Above
    rather than beside because the block is anchored to the top edge now: a
    mark 2.5 lines tall centred on the credit's ink would reach a line and a
    half above it, which at a 1.2% inset is off the canvas.

    A failure here is a warning rather than an exception: the figure is already
    on disk and correct, and losing the whole render over a missing font would
    be the worse outcome. The credit's absence is visible on the image itself.
    """
    from PIL import Image, ImageDraw

    filepath = Path(filepath)
    try:
        image = Image.open(filepath).convert("RGBA")
        width, height = image.size
        reference = min(width, height)

        font = _stamp_font(reference)
        draw = ImageDraw.Draw(image)
        # The credit's own box, asked for without a logo: where the text lands
        # is the one thing the mark is not allowed to change.
        left, top, right, _bottom = stamp_box(image.size, text)

        if logo_path is not None:
            bbox = draw.textbbox((0, 0), text, font=font)
            logo_w, logo_h = _logo_geometry(logo_path, bbox[3] - bbox[1], width)
            logo = Image.open(logo_path).convert("RGBA")
            pixels = np.array(logo)
            near_white = (pixels[:, :, 0] > 240) & (pixels[:, :, 1] > 240) & (pixels[:, :, 2] > 240)
            pixels[near_white, 3] = 0
            pixels[:, :, 3] = (pixels[:, :, 3].astype(float) * 0.85).astype(np.uint8)
            logo = Image.fromarray(pixels).resize((logo_w, logo_h), Image.LANCZOS)
            # Sat on the credit's ink rather than on its box: `draw.text`
            # anchors at the font's ascender line, so the ink starts `bbox[1]`
            # below `top` and a gap measured to the box would look uneven.
            image.paste(logo, (right - logo_w, top + bbox[1] - _stamp_gap(logo_h) - logo_h), logo)

        draw.text((left, top), text, fill=(140, 140, 140), font=font)
        image.convert("RGB").save(filepath)
    except Exception as error:  # pragma: no cover - the figure is already saved
        print(f"Warning: could not watermark {filepath}: {error}")


def finalize(
    fig,
    filepath: str | Path,
    *,
    dpi: int = 200,
    logo_path: str | Path | bool | None = None,
    bbox_inches: str | None = "tight",
    close: bool = True,
    edition: str | None = None,
) -> Path:
    """Save, stamp and close — the one exit every figure in the product takes.

    Centralised so no figure can be shipped without its data credit: the way to
    write a PNG here is this function, and this function always watermarks.

    ``edition`` puts ruling R-4's edition name in front of the credit, so the
    corner of the image says which of the two adjudications it is. Left
    ``None`` the stamp is the bare credit, which is what every figure outside
    the per-game product — the band sweep, a diagnostic — should carry: those
    are not an adjudication of a game and have no edition to name.

    The brand mark is **on by default**: ``logo_path=None`` means
    :data:`BRAND_LOGO`, a path means that file, and ``logo_path=False`` is the
    one way to ship a figure without it. Default-on rather than opt-in for the
    same reason the credit is not optional — a mark that has to be asked for is
    a mark on the figures whoever remembered it rendered, and missing from the
    rest.
    """
    import matplotlib.pyplot as plt

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        filepath, dpi=dpi, bbox_inches=bbox_inches, facecolor=PALETTE["bg"], edgecolor="none"
    )
    stamp = WATERMARK if edition is None else edition_stamp(edition)
    mark = None if logo_path is False else BRAND_LOGO if logo_path is None else logo_path
    # Room first, then the stamp: the strip the credit lives in is reserved on
    # the saved pixels when the figure reached into it, so the credit is never
    # painted over a footer and a footer is never painted over.
    reserve_stamp_strip(filepath, stamp, logo_path=mark)
    apply_watermark(filepath, logo_path=mark, text=stamp)
    if close:
        plt.close(fig)
    return filepath
