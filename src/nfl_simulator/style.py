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
BRAND_HANDLE = "@[TBD]"
DATA_CREDIT = "Data: nflverse"
WATERMARK = f"{DATA_CREDIT} | {BRAND_HANDLE}"

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
# (`Simulator/visualizations.py:453-457`).
CLASH_DISTANCE = 0.20
CLASH_LIGHTEN = 0.45


def colour_distance(first: str, second: str) -> float:
    """Euclidean distance between two colours in RGB, each channel on 0-1."""
    a, b = to_rgb(first), to_rgb(second)
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) ** 0.5


def lighten(color: str, amount: float = 0.5) -> str:
    """Blend ``color`` toward white by ``amount`` (0 = unchanged, 1 = white).

    Returns a hex string rather than the baseball version's RGB tuple: every
    caller here hands the result straight back to matplotlib as a colour, and a
    tuple that has to be re-hexed at each call site is where a stray
    ``to_hex`` gets forgotten.
    """
    r, g, b = to_rgb(color)
    return to_hex((r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount))


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


def apply_watermark(
    filepath: str | Path,
    *,
    logo_path: str | Path | None = None,
    position: str = "top-right",
    y_pct: float | None = None,
    text: str = WATERMARK,
) -> None:
    """Stamp the credit line — and optionally a logo — onto a saved PNG.

    Done in PIL on the saved pixels rather than in figure coordinates, because
    ``bbox_inches="tight"`` crops a figure by an amount that depends on how long
    its longest tick label happened to be. A corner in pixels is the same corner
    on every image the product ships.

    ``logo_path=None`` draws the text line alone. The project has no mark yet,
    so that is the shipped path; the slot exists so adding one later does not
    move the text.

    A failure here is a warning rather than an exception: the figure is already
    on disk and correct, and losing the whole render over a missing font would
    be the worse outcome. The credit's absence is visible on the image itself.
    """
    from PIL import Image, ImageDraw, ImageFont

    filepath = Path(filepath)
    try:
        image = Image.open(filepath).convert("RGBA")
        width, height = image.size
        reference = min(width, height)

        logo = None
        logo_w = logo_h = 0
        if logo_path is not None:
            logo = Image.open(logo_path).convert("RGBA")
            pixels = np.array(logo)
            near_white = (pixels[:, :, 0] > 240) & (pixels[:, :, 1] > 240) & (pixels[:, :, 2] > 240)
            pixels[near_white, 3] = 0
            pixels[:, :, 3] = (pixels[:, :, 3].astype(float) * 0.85).astype(np.uint8)
            logo = Image.fromarray(pixels)
            logo_h = max(20, int(reference * 0.028))
            logo_w = int(logo_h * logo.width / logo.height)
            logo = logo.resize((logo_w, logo_h), Image.LANCZOS)

        font_size = max(10, int(reference * 0.0095))
        try:
            import matplotlib.font_manager as fm

            font = ImageFont.truetype(
                fm.findfont(fm.FontProperties(family="DejaVu Sans")), font_size
            )
        except Exception:
            font = ImageFont.load_default()

        draw = ImageDraw.Draw(image)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]

        margin = int(width * 0.02)
        gap = int(reference * 0.003)
        block_w = max(logo_w, text_w)
        center_x = (
            margin + block_w // 2 if position == "top-left" else width - margin - block_w // 2
        )

        top_y = int(height * y_pct) if y_pct is not None else int(height * 0.012)
        if logo is not None:
            image.paste(logo, (center_x - logo_w // 2, top_y), logo)
        draw.text(
            (center_x - text_w // 2, top_y + logo_h + (gap if logo is not None else 0)),
            text,
            fill=(140, 140, 140),
            font=font,
        )
        image.convert("RGB").save(filepath)
    except Exception as error:  # pragma: no cover - the figure is already saved
        print(f"Warning: could not watermark {filepath}: {error}")


def finalize(
    fig,
    filepath: str | Path,
    *,
    dpi: int = 200,
    logo_path: str | Path | None = None,
    bbox_inches: str | None = "tight",
    close: bool = True,
) -> Path:
    """Save, stamp and close — the one exit every figure in the product takes.

    Centralised so no figure can be shipped without its data credit: the way to
    write a PNG here is this function, and this function always watermarks.
    """
    import matplotlib.pyplot as plt

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        filepath, dpi=dpi, bbox_inches=bbox_inches, facecolor=PALETTE["bg"], edgecolor="none"
    )
    apply_watermark(filepath, logo_path=logo_path)
    if close:
        plt.close(fig)
    return filepath
