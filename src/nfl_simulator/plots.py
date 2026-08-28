"""The product layer's figures — presentation of committed numbers, nothing fitted.

The simulator decides who deserved to win. This module only shows it, and the
showing has its own rules:

* **The bucket, not a bare number.** Document 33 §2a settled a three-way label
  with a "too close to call" band at DTW% 0.40–0.60, because the two available
  flip definitions disagreed on 56 games and all 56 sat inside 0.363–0.626. The
  band is a presentation choice made before that reconciliation, not a threshold
  fitted to minimise the residual.
* **The interval is never quoted bare.** It is nominally 89% (document 03's
  5.5/94.5 convention). Document 10 measured its coverage at 91.5% on games with
  something to adjudicate, so it runs about two points wide — and on the 44.4% of
  games that are degenerate it collapses to a point and means nothing at all.
  Both facts travel with every interval this module draws.
* **A single value has no density.** A game with no luck events returns one
  margin draw. Histogramming it would invent a shape, so it is drawn as a note.

Surface, ink and grid come from ``style.PALETTE`` — the house style shared with
the baseball simulator. Only the two teams wear colour; every rule and label is
ink, so identity is never carried by colour alone. The default pair
(``#2a78d6`` / ``#eb6834``) is the validated light-mode categorical pair, used
when a caller supplies no team colours. Output is PNG for print — no hover
layer, no dark mode.
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.text import Annotation, Text

from nfl_simulator.style import (
    CLASH_DISTANCE,
    PALETTE,
    colour_distance,
    heading_font,
    rc_style,
)

# Document 10 Gate V-3's convention: a game whose verdict never changes across
# the bootstrap. Its interval is a point, and reporting one is misleading.
DEGENERATE_EPS = 0.001

# Document 33 §2a. Inclusive on both edges — a game at exactly 0.40 is as
# undecided as one at 0.401.
BAND_LOW, BAND_HIGH = 0.40, 0.60

CLEAR_FLIP = "clear flip"
TOO_CLOSE = "too close to call"
SCOREBOARD_HOLDS = "scoreboard holds"

# Measured, not asserted. Document 10 §"What this establishes": 0.9152 informative
# coverage at the shipped 800 coin draws. Document 33 §3: 1,226 of 2,761 games.
# Document 16 measured the overtime toss and refused it, so every figure that
# draws a ledger has to say the ledger is one event short on purpose. Silence
# would let a reader take the figure for the whole story.
OVERTIME_FOOTER = "Went to overtime; the coin toss is reported, not neutralized."

NOMINAL_COVERAGE = "89%"
MEASURED_COVERAGE = "91.5%"
DEGENERATE_SHARE = "44.4%"

# --------------------------------------------------------------------------
# style
# --------------------------------------------------------------------------

# The palette is the house style's, shared with the baseball simulator, so the
# two projects' figures read as one hand. Everything that is not a team is ink
# from `PALETTE`; the two team hues below are the fallback a caller gets when it
# does not pass a pair from `teams.py`.
HOME_HUE, AWAY_HUE = "#2a78d6", "#eb6834"

STYLE = rc_style()


# --------------------------------------------------------------------------
# the verdict
# --------------------------------------------------------------------------


def bucket_label(
    dtw_home: float,
    actual_margin: float,
    *,
    low: float = BAND_LOW,
    high: float = BAND_HIGH,
) -> str:
    """Document 33 §2a's three-way label, for any band.

    The band is a parameter rather than a constant so the robustness sweep and
    the headline are the *same* label — a sweep with its own copy of this rule
    could report bucket counts the product would not agree with.

    An actual tie falls out as a clear flip whenever it is outside the band,
    which is the honest reading: the scoreboard declined to name a winner and the
    bootstrap does not. Document 33 excluded the 10 ties from its flip *counts*;
    a product that has to render one still has to say something.
    """
    if low <= dtw_home <= high:
        return TOO_CLOSE
    if actual_margin == 0:
        return CLEAR_FLIP
    deserved_home = dtw_home > 0.5
    return SCOREBOARD_HOLDS if deserved_home == (actual_margin > 0) else CLEAR_FLIP


@dataclass(frozen=True)
class GameVerdict:
    """One game's adjudication, as the product states it.

    Every field is read from the simulator's committed output. Nothing here
    recomputes DTW% — a presentation layer that re-derived the headline could
    drift from the number the research record carries.
    """

    game_id: str
    home_team: str
    away_team: str
    actual_margin: float
    deserved_margin: float
    dtw_home: float
    dtw_interval: tuple[float, float]
    margin_draws: np.ndarray
    # The scoreboard facts, for the header. Optional because the simulator's
    # summary does not carry them — a verdict built from `dtw_games_v13.parquet`
    # alone still has to draw, it just states the margin instead of the score.
    home_score: int | None = None
    away_score: int | None = None
    game_date: str | None = None
    went_to_overtime: bool = False

    @property
    def is_degenerate(self) -> bool:
        """The bootstrap never changes its mind, so the interval is a point."""
        return self.dtw_home <= DEGENERATE_EPS or self.dtw_home >= 1.0 - DEGENERATE_EPS

    @property
    def is_point_mass(self) -> bool:
        """No coin was flipped at all — every draw is the game's own result."""
        return len(np.unique(self.margin_draws)) == 1

    @property
    def scoreboard_winner(self) -> str | None:
        """None on an actual tie: the scoreboard named nobody."""
        if self.actual_margin > 0:
            return self.home_team
        if self.actual_margin < 0:
            return self.away_team
        return None

    @property
    def deserved_winner(self) -> str:
        return self.home_team if self.dtw_home > 0.5 else self.away_team

    @property
    def bucket(self) -> str:
        """Document 33 §2a's three-way label, at the shipped band."""
        return bucket_label(self.dtw_home, self.actual_margin)

    def headline(self) -> str:
        """`"MIN 55% / DET 45%"` — the favoured side first, shares summing to 100."""
        home_share = round(self.dtw_home * 100)
        away_share = 100 - home_share
        if self.dtw_home > 0.5:
            return f"{self.home_team} {home_share}% / {self.away_team} {away_share}%"
        return f"{self.away_team} {away_share}% / {self.home_team} {home_share}%"

    def interval_note(self, *, coverage: bool = True) -> str:
        """The interval, with the two facts that stop it being read as a plain 89%.

        ``coverage=False`` drops document 10's measured-coverage sentence and
        keeps the interval. Round 4 took that sentence off the share card for a
        reason that survives round 5's return of the margin plot to the share
        image: beside the share the figure is about, a second percentage reads
        as a competing answer rather than as a note on the first. The sentence
        is not dropped from the product — the article figure still carries
        it, for a reader who has already asked for the methodology.
        """
        if self.is_degenerate:
            return (
                "Every re-flip lands the same way, so the interval collapses to a point. "
                f"{DEGENERATE_SHARE} of games are degenerate this way (doc 33 §3)."
            )
        low, high = self.dtw_interval
        favoured = self.deserved_winner
        # The stored interval is on the home team's share. Quoting it beside a
        # headline that names the away team would attribute the home team's
        # bounds to the away team's number, so it is mirrored to match.
        share_low, share_high = (low, high) if favoured == self.home_team else (1 - high, 1 - low)
        interval = (
            f"The {NOMINAL_COVERAGE} interval on {favoured}'s share runs "
            f"{share_low * 100:.0f}–{share_high * 100:.0f}%."
        )
        if not coverage:
            return interval
        return (
            f"{interval} Document 10 measured that interval's coverage at "
            f"{MEASURED_COVERAGE} on games with something to adjudicate, so it "
            "runs about two points wide."
        )

    def score_line(self) -> str:
        """`"Actual: GB 23 - DET 31"`, away first, the way a scoreboard reads.

        A verdict built from the summary alone has no score, so it states the
        margin instead. Printing `None - None` would be worse than either.
        """
        if self.home_score is None or self.away_score is None:
            leader = self.scoreboard_winner
            if leader is None:
                return "Actual: tied"
            return f"Actual: {leader} by {abs(self.actual_margin):.0f}"
        return (
            f"Actual: {self.away_team} {self.away_score:.0f} - "
            f"{self.home_team} {self.home_score:.0f}"
        )

    def dtw_line(self) -> str:
        """`"DTW: GB 95% • DET 5%"` — the favoured side first, as in `headline`."""
        return f"DTW: {self.headline().replace(' / ', ' • ')}"

    def date_line(self) -> str:
        """`"(10/07/2018)"`, or nothing at all when the date is not known."""
        if not self.game_date:
            return ""
        year, month, day = str(self.game_date)[:10].split("-")
        return f"({month}/{day}/{year})"

    def subtitle_line(self) -> str:
        """The one muted line under every figure's heading."""
        return "   ".join(part for part in (self.score_line(), self.date_line()) if part) + (
            f"    {self.dtw_line()}"
        )

    def deserved_line(self) -> str:
        """`"Deserved margin: DET -8.3"` — the side it favours, then the size."""
        if self.deserved_margin == 0:
            return "Deserved margin: dead level"
        side = self.home_team if self.deserved_margin > 0 else self.away_team
        return f"Deserved margin: {side} by {abs(self.deserved_margin):.1f}"


def verdict_from_row(
    row: dict, margin_draws: np.ndarray, schedule: dict | None = None
) -> GameVerdict:
    """Build a verdict from a `dtw_games_v13.parquet` row plus its bootstrap draws.

    ``schedule`` is the game's nflverse schedule row, and it supplies only
    presentation facts — the two scores, the date, whether the game went to
    overtime. Nothing in it can change the adjudication; the summary row is
    still the sole source of every number the figure states.
    """
    schedule = schedule or {}
    return GameVerdict(
        game_id=row["game_id"],
        home_team=row["home_team"],
        away_team=row["away_team"],
        actual_margin=float(row["actual_margin"]),
        deserved_margin=float(row["deserved_margin"]),
        dtw_home=float(row["dtw_home"]),
        dtw_interval=(float(row["dtw_low"]), float(row["dtw_high"])),
        margin_draws=margin_draws,
        home_score=schedule.get("home_score"),
        away_score=schedule.get("away_score"),
        game_date=schedule.get("gameday"),
        went_to_overtime=bool(schedule.get("overtime") or False),
    )


# --------------------------------------------------------------------------
# the shared header
# --------------------------------------------------------------------------

# The verdict pill's fill. A flip is the alarming reading and wears the status
# red; the scoreboard holding is the reassuring one and wears the green. "Too
# close to call" is neither, so it wears ink — a third status colour would
# imply a third kind of finding.
PILL_COLOURS = {
    CLEAR_FLIP: "bad",
    SCOREBOARD_HOLDS: "good",
    TOO_CLOSE: "text_muted",
}

# Offsets in points above the axes, measured from the top spine. Points rather
# than axes fractions because the waterfall's height grows with its row count,
# and a fixed fraction of a changing height is not a fixed gap.
HEADING_OFFSET = 62
RULE_OFFSET = 52
SUBTITLE_OFFSET = 36
# The second subtitle line, when a figure has one. The team-points figure puts
# the most-likely scoreline here, under the actual one it is being compared to,
# which is where the baseball run-distribution chart puts its own.
SUBTITLE_EXTRA_OFFSET = 22
# Two lines' room: the caption is wrapped clear of the verdict pill, which puts
# it on two lines on a narrow figure.
CAPTION_OFFSET = 12
CAPTION_PILL_GAP = 14


def pill_colour(bucket: str) -> str:
    """The fill for a verdict pill, from the palette rather than from taste."""
    return PALETTE[PILL_COLOURS.get(bucket, "text_muted")]


def draw_header(
    ax,
    verdict: GameVerdict,
    heading: str,
    *,
    caption: str | None = None,
    left_points: float = 0.0,
    subtitle_extra: str | None = None,
):
    """The title block every per-game figure wears: heading, rule, subtitle, pill.

    Drawn in the band above ``ax`` rather than in a separate strip axes. A strip
    is what the baseball style does, and it is the better shape when a figure is
    a figure — but ``attach_overtime_sidebar`` widens the figure and rescales the
    host axes to hold the plot at the inches it was drawn at, and a second axes
    would stretch across the growth. Everything here is anchored to ``ax``, so
    the sidebar moves the header with the plot it belongs to.

    ``left_points`` moves the heading, the rule and the subtitle left of the
    axes. The waterfall's row labels are sentences, so its axes starts a third of
    the way across the figure; a title anchored to it left a hole where the title
    should be. The pill does not move — it belongs to the plot's right edge,
    which is where the figure's right edge is.

    Returns ``(heading_text, pill_text)`` so a caller can measure them.
    """
    left_px = left_points / 72.0 * ax.figure.dpi
    plot_width = ax.get_window_extent().width

    def at(y_points, text, **kwargs):
        return ax.annotate(
            text,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(left_points, y_points),
            textcoords="offset points",
            va="bottom",
            **kwargs,
        )

    heading_text = at(
        HEADING_OFFSET,
        heading,
        fontsize=16,
        fontweight="bold",
        color=PALETTE["text"],
        fontfamily=heading_font(),
    )

    # The divider rule, drawn the width of the plot so the header reads as one
    # block with the figure under it rather than as a caption floating above.
    # Offset in points off the top spine, like everything else here.
    lifted = ax.transAxes + mpl.transforms.ScaledTranslation(
        0, RULE_OFFSET / 72, ax.figure.dpi_scale_trans
    )
    ax.plot(
        [left_px / plot_width if plot_width else 0.0, 1],
        [1, 1],
        transform=lifted,
        color=PALETTE["grid"],
        linewidth=0.8,
        clip_on=False,
        zorder=1,
    )

    at(SUBTITLE_OFFSET, verdict.subtitle_line(), fontsize=9.5, color=PALETTE["text_muted"])
    if subtitle_extra:
        at(SUBTITLE_EXTRA_OFFSET, subtitle_extra, fontsize=9.5, color=PALETTE["text_muted"])

    # The pill sits on the **subtitle** row, not beside the heading. `finalize`
    # stamps the data credit into the top-right corner of the saved pixels, and
    # a pill on the heading row lands under it — measured on
    # `LV_KC_9-48--0-100_dtw.png`, where "scoreboard holds" printed through both
    # the watermark and the last word of the heading.
    pill_text = ax.annotate(
        verdict.bucket,
        xy=(1, 1),
        xycoords="axes fraction",
        xytext=(0, SUBTITLE_OFFSET - 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color=PALETTE["bg"],
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": pill_colour(verdict.bucket),
            "edgecolor": "none",
        },
        zorder=6,
    )

    if caption:
        caption_text = at(CAPTION_OFFSET, caption, fontsize=8, color=PALETTE["text_muted"])
        # Wrapped to the room the pill leaves rather than to the plot's width.
        # On `DET_MIN_10-23--45-55_luck_ledger.png` the pill's rounded box came
        # down onto the end of the caption's single long line; measuring the
        # pill is the only way to know how much room is actually left, since
        # both the pill's text and the figure's width vary per game.
        pill_width = pill_text.get_window_extent(_renderer(ax.figure)).width
        room = plot_width - left_px - pill_width - CAPTION_PILL_GAP
        _wrap_to_width(ax.figure, caption_text, room)

    return heading_text, pill_text


# --------------------------------------------------------------------------
# the figure
# --------------------------------------------------------------------------


def _renderer(fig):
    """A renderer to measure text with.

    Layout rules that guess how wide a string is are wrong on the first game
    that proves them wrong, so this module measures instead. The figures are
    drawn to PNG, so the canvas can always supply one.
    """
    fig.canvas.draw()
    return fig.canvas.get_renderer()


def _wrap_to_width(fig, text: Text, width_px: float) -> None:
    """Re-wrap ``text`` to ``width_px``, in place, by measuring each candidate line.

    Matplotlib's own ``wrap=True`` wraps to the *figure* edge, which is not a
    boundary this module controls: ``attach_overtime_sidebar`` widens the figure,
    and a caveat wrapped to it widens with it and runs under the sidebar.

    A break the caller already put in is kept and wrapped inside, rather than
    collapsed and re-broken wherever the measuring lands. The ledger card's
    margin sentence breaks at its comma for that reason: re-flowed as one run it
    read "deserved to lose / by 8.3", which parts a number from its clause.
    """
    renderer = _renderer(fig)
    font = text.get_fontproperties()

    def too_wide(line: str) -> bool:
        return renderer.get_text_width_height_descent(line, font, False)[0] > width_px

    lines: list[str] = []
    for source in text.get_text().split("\n"):
        current = ""
        for word in source.split():
            candidate = f"{current} {word}" if current else word
            if current and too_wide(candidate):
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)

    text.set_wrap(False)
    text.set_text("\n".join(lines))


def _lift_colliding_label(fig, first: Annotation, second: Annotation) -> None:
    """Separate two rule labels that print on top of each other.

    Both labels hang inside the top of the plot off their own rule, so a game
    whose deserved and actual margins are close prints one through the other —
    `2025_13_DEN_WAS` at −3.3 and −1 was unreadable. The **left-hand** label moves
    *above* the top spine, into the empty band between the plot and its subtitle.

    Two choices are load-bearing. It is the left-hand label because a label runs
    to the right of its own rule: lifting that one also takes it off the other
    rule, which it was otherwise struck through by. And it goes above the spine
    rather than onto a second row inside the plot, because the rules stop at the
    spine and the band above it is empty, whereas a second row lands the text on
    whatever bar is tallest at that margin.
    """
    renderer = _renderer(fig)
    if not first.get_window_extent(renderer).overlaps(second.get_window_extent(renderer)):
        return
    lifted = min((first, second), key=lambda label: label.xy[0])
    lifted.set_verticalalignment("bottom")
    lifted.xyann = (4, 3)


# The gap a rule label keeps from a corner label, in points. A bare
# non-overlap is not enough: on `2025_17_DET_MIN` the flipped `Actual: MIN by
# 13` stopped a pixel short of `MIN wins`, and a box edge against a capital M
# reads as a clipped letter.
CORNER_CLEARANCE = 8.0


def _clear_corner_labels(fig, label: Annotation, corners: Sequence[Text]) -> None:
    """Move a rule label out from under the corner label it landed on.

    The two corner labels are the axis's key — `GB wins` over one half of it,
    `DET wins` over the other — and they sit in the band the two rule labels
    have hung in since document 37. A rule at either extreme of the axis puts
    its label straight through one of them; `2021_14_LV_KC`, whose actual margin
    is 39 on an axis that stops at 51, is the case that found this.

    The label is flipped to the other side of its own rule first. That is the
    cheaper move: it keeps the label on the row it belongs to and beside the
    rule it names. A rule so close to a frame edge that the flip does not clear
    the corner either is lifted above the top spine instead — where
    :func:`_lift_colliding_label` already sends a label with nowhere else to be.
    """
    renderer = _renderer(fig)
    clearance = CORNER_CLEARANCE / 72.0 * fig.dpi

    def collides() -> bool:
        box = label.get_window_extent(renderer)
        return any(
            box.overlaps(corner.get_window_extent(renderer).padded(clearance)) for corner in corners
        )

    if not collides():
        return
    label.set_horizontalalignment("right")
    label.xyann = (-4, label.xyann[1])
    if not collides():
        return
    label.set_horizontalalignment("left")
    label.set_verticalalignment("bottom")
    label.xyann = (4, 3)


def _rule(
    ax, x: float, label: str, *, color: str, dashes, weight: float, boxed: bool = False
) -> Annotation:
    """A vertical reference rule with its label attached, never colour alone.

    ``boxed`` puts the label in a cream-filled rounded box edged in the rule's
    own colour — the baseball chart's `(Actual)` callout. The fill matters: a
    bare label printed over the tallest part of a histogram is unreadable, and
    the box gives it a surface without hiding the bar it sits on.
    """
    ax.plot(
        [x, x],
        [0, 1],
        transform=ax.get_xaxis_transform(),
        color=color,
        linewidth=weight,
        dashes=dashes,
        zorder=4,
        clip_on=False,
    )
    annotation = ax.annotate(
        label,
        xy=(x, 1.0),
        xycoords=ax.get_xaxis_transform(),
        xytext=(4, -2),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=9,
        color=color,
        zorder=5,
        bbox=(
            {
                "boxstyle": "round,pad=0.3",
                "facecolor": PALETTE["bg"],
                "edgecolor": color,
                "linewidth": 0.8,
            }
            if boxed
            else None
        ),
    )
    # Round 5 made these labels read `Actual: DET by 8`, which is also how a
    # verdict with no scoreboard on file opens its own subtitle. A gid is the
    # only way left to tell the figure's two rules from the header's words. The
    # zero rule carries no label and is not one of them.
    if label:
        annotation.set_gid("rule-label")
    return annotation


# Where the three annotations that share the top of the plot sit, in axes
# fractions. The rule callouts hang off the top spine and own everything above
# 0.95, so the arrow and the deserved-to-win line are stacked under them rather
# than beside them: a game whose two rules are far apart would otherwise put the
# arrow straight through both labels.
ARROW_Y = 0.855
CALLOUT_Y = 0.76

# How much taller than its tallest bar the plot is drawn. The annotations above
# are placed by rule rather than by inspection, so the room they need is made
# rather than hoped for: an annotated figure reserves the whole band they sit
# in, and a bare one reserves only what the two rule callouts want.
ANNOTATED_HEADROOM = 1.62
PLAIN_HEADROOM = 1.18

# The logo legend's two entries, in axes fractions: how far apart their centres
# sit and how far below the plot they hang. The swatch legend's own anchor is
# -0.20, and the two are alternatives, so they share the row.
LOGO_LEGEND_Y = -0.20
LOGO_LEGEND_GAP = 0.34


def _shielded() -> dict:
    """A cream backing for a label that may land on a bar.

    Invisible on the surface and opaque over a fill, which is what a callout
    placed by rule rather than by inspection needs: the corner it is put in is
    empty on most games and is not on all of them."""
    return {
        "boxstyle": "square,pad=0.25",
        "facecolor": PALETTE["bg"],
        "edgecolor": "none",
    }


def deserved_share(verdict: GameVerdict) -> int:
    """The favoured team's share as a whole number, rounded as the headline is.

    Rounded from the *home* share and subtracted, so the callout and the
    headline can never disagree by a point on a share that rounds both ways.
    """
    home_share = round(verdict.dtw_home * 100)
    return home_share if verdict.deserved_winner == verdict.home_team else 100 - home_share


def _draw_callout(ax, verdict: GameVerdict, home_colour: str, away_colour: str) -> Text | None:
    """`"GB deserved to win 95% of simulations"`, in that team's own colour.

    The baseball run-distribution chart's device. It is the sentence the figure
    exists to say, and a reader who takes nothing else off the plot should take
    this. It goes in the upper corner on the favoured team's own side, so the
    words sit over the bars they describe.

    Two games get something else, because on them that sentence is not the one
    the figure is entitled to say:

    * **Inside the band**, the pill declines to name a winner and the callout
      must not name one either — it states both shares and says so. It is ink
      rather than a club colour, since it belongs to neither team.
    * **A degenerate game** gets no callout. "100% of simulations" repeats the
      title, and a bootstrap that never changed its mind is the last place to
      put a sentence about how often something happened. The luck arrow is
      already suppressed here for the same reason.
    """
    if verdict.is_degenerate:
        return None
    at_home = verdict.deserved_winner == verdict.home_team
    if verdict.bucket == TOO_CLOSE:
        message = f"{verdict.headline().replace(' / ', ' \u00b7 ')} \u2014 {TOO_CLOSE}"
        colour = PALETTE["text"]
    else:
        message = (
            f"{verdict.deserved_winner} deserved to win {deserved_share(verdict)}% of simulations"
        )
        colour = home_colour if at_home else away_colour
    return ax.text(
        0.985 if at_home else 0.015,
        CALLOUT_Y,
        message,
        transform=ax.transAxes,
        ha="right" if at_home else "left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=colour,
        zorder=7,
        bbox=_shielded(),
    )


def _draw_luck_arrow(ax, verdict: GameVerdict):
    """The span between the two rules, labelled with the luck it measures.

    The patch runs from the actual margin to the deserved one, and its head is
    at the **actual** end — that is the direction luck pushed the game, and the
    label says so in the same words. An arrowhead on the deserved end would
    point one way while the sentence above it pointed the other.

    Sign convention: ``actual - deserved`` is what luck added to the home team's
    margin, so a positive gap is luck that helped the home team.
    """
    gap = verdict.actual_margin - verdict.deserved_margin
    toward = verdict.home_team if gap > 0 else verdict.away_team
    span = ax.annotate(
        "",
        xy=(verdict.deserved_margin, ARROW_Y),
        xycoords=("data", "axes fraction"),
        xytext=(verdict.actual_margin, ARROW_Y),
        textcoords=("data", "axes fraction"),
        arrowprops={
            "arrowstyle": "<-",
            "color": PALETTE["text_muted"],
            "linewidth": 1.2,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
        zorder=6,
        annotation_clip=False,
    )
    label = ax.text(
        (verdict.actual_margin + verdict.deserved_margin) / 2.0,
        ARROW_Y + 0.015,
        f"luck moved the margin {abs(gap):.1f} points toward {toward}",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=9,
        color=PALETTE["text_muted"],
        zorder=7,
        bbox=_shielded(),
    )
    return span, label


def _draw_logo_legend(ax, entries, *, template: str = "{team} wins") -> None:
    """The clubs' marks under the plot, in place of two coloured swatches.

    A mark is the identity a reader already knows, so it does the job a swatch
    does without asking them to hold a colour in their head while they look at
    the bars. The abbreviation stays beside it: a club without a cached mark
    still has to be named, and identity is never carried by an image alone.
    """
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    count = len(entries)
    centres = [0.5 + (index - (count - 1) / 2.0) * LOGO_LEGEND_GAP for index in range(count)]
    for centre, (team, logo) in zip(centres, entries, strict=True):
        if logo is not None:
            ax.add_artist(
                AnnotationBbox(
                    OffsetImage(
                        logo,
                        zoom=logo_zoom(logo, ax.figure, max_width_in=0.34, max_height_in=0.17),
                    ),
                    (centre - 0.05, LOGO_LEGEND_Y),
                    xycoords="axes fraction",
                    frameon=False,
                    annotation_clip=False,
                    box_alignment=(0.5, 0.5),
                )
            )
        ax.text(
            centre - 0.02,
            LOGO_LEGEND_Y,
            template.format(team=team),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=10,
            color=PALETTE["text"],
            clip_on=False,
        )


# How far under the axis the two direction labels sit, in points: clear of the
# tick labels, and close enough to them to read as a key to the numbers rather
# than as a caption under the figure.
WINS_BY_OFFSET = -30


def _clamp_into_axes(fig, label: Annotation) -> None:
    """Pull a label back inside the plot when its anchor pushed it out.

    The two direction labels are anchored to zero, which is where they mean the
    most. A one-sided game puts zero hard against a frame edge, and the label
    hanging off that side runs into the y axis or off the figure — measured on
    `2021_14_LV_KC`, whose margins are all one team's.
    """
    box = label.get_window_extent(_renderer(fig))
    frame = label.axes.get_window_extent()
    shift = 0.0
    if box.x0 < frame.x0:
        shift = frame.x0 - box.x0
    elif box.x1 > frame.x1:
        shift = frame.x1 - box.x1
    if shift:
        x_points, y_points = label.xyann
        label.xyann = (x_points + shift / fig.dpi * 72.0, y_points)


def _part_labels(fig, left: Annotation, right: Annotation) -> None:
    """Push the right-hand label clear of the left-hand one.

    The two direction labels are both anchored to zero, so a game whose margins
    all fall one side of it has both of them at the same pinned edge — and the
    clamp that saved the first from the y axis walked it into the second.
    `2021_14_LV_KC` printed `← LV wins by` and `KC wins by →` on top of
    each other.
    """
    renderer = _renderer(fig)
    gap = CORNER_CLEARANCE / 72.0 * fig.dpi
    overlap = left.get_window_extent(renderer).x1 + gap - right.get_window_extent(renderer).x0
    if overlap > 0:
        x_points, y_points = right.xyann
        right.xyann = (x_points + overlap / fig.dpi * 72.0, y_points)


def _draw_wins_by_labels(fig, ax, verdict: GameVerdict, home_colour: str, away_colour: str) -> list:
    """`← GB wins by` and `DET wins by →`, flanking the zero they are measured from.

    The axis title was `final margin (DET − GB)` and the ticks were signed, so
    the size of a win was only readable through a subtraction the reader had to
    perform. With the ticks unsigned there is nothing left to subtract: each
    half of the axis is one club's winning margin, and these two labels say
    whose is whose and which way it grows.

    They are anchored to zero rather than centred in their own halves so that
    each arrow reads off the line the numbers count up from — `0` is one tick
    above, between them.
    """
    low, high = ax.get_xlim()
    zero = min(max(0.0, low), high)
    labels = []
    for colour, empty, text, align, gap in (
        (away_colour, low >= 0.0, f"\u2190 {verdict.away_team} wins by", "right", -6),
        (home_colour, high <= 0.0, f"{verdict.home_team} wins by \u2192", "left", 6),
    ):
        if empty:
            continue
        labels.append(
            ax.annotate(
                text,
                xy=(zero, 0.0),
                xycoords=ax.get_xaxis_transform(),
                xytext=(gap, WINS_BY_OFFSET),
                textcoords="offset points",
                ha=align,
                va="top",
                fontsize=9,
                fontweight="bold",
                color=colour,
                annotation_clip=False,
            )
        )
    for label in labels:
        _clamp_into_axes(fig, label)
    if len(labels) == 2:
        _part_labels(fig, *labels)
    return labels


def plot_bootstrap_distribution(
    verdict: GameVerdict,
    *,
    bin_width: float = 1.0,
    colors: tuple[str, str] | None = None,
    logos: dict | None = None,
    callout: bool = False,
    arrow: bool = False,
    coverage: bool = True,
):
    """Deserved margin across the bootstrap, with the actual margin marked.

    The x axis is the home team's margin, so everything right of zero is a home
    win and everything left of it is an away win. The two fills are the two
    teams; the share of the distribution on each side *is* the DTW% in the title.

    **Round 5: the axis is read, not computed.** The ticks are unsigned, the
    axis title is gone, and in its place two direction labels flank zero —
    `← GB wins by` on one side and `DET wins by →` on the other, each in its own
    club's colour. Each half of the axis carries that club's faint tint and its
    name and mark in the corner above it, reusing the waterfall's
    :func:`_draw_side_tints`; the two rule labels name a team rather than
    carrying a sign, reusing its :func:`anchor_label`. There is no legend row:
    the tints and the corner labels are the key, and a second one under the plot
    would have named the same two clubs twice.

    ``coverage=False`` drops document 10's measured-coverage sentence from the
    footnote — see :meth:`GameVerdict.interval_note`. The share image passes it;
    the article figure does not.

    Bins are one point of margin wide and aligned to the integer grid. A
    neutralised margin is a sum of a handful of EPA swings, so the distribution
    is genuinely lumpy — clusters a field goal apart, with extra-point structure
    inside them. Bin edges that fall between those clusters comb the histogram
    into alternating spikes and gaps, which reads as a rendering artifact rather
    than as the clustering it actually is. A one-point grid puts the edges where
    a reader already thinks in margins, and zero lands on an edge, so no bar
    straddles the line that decides the winner.

    ``colors`` is the game's ``(home, away)`` pair from :mod:`teams`. It is a
    parameter rather than a lookup so this module stays a presentation layer
    with no data dependency of its own — and so a test never reaches a network.

    Returns ``(figure, axes)`` so a caller can add a panel beside it.
    """
    home_colour, away_colour = colors or (HOME_HUE, AWAY_HUE)
    logos = logos or {}
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.6, 4.0))

        if verdict.is_point_mass:
            ax.text(
                0.5,
                0.5,
                "This game had no luck events to re-flip.\nThe deserved margin is the actual one.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color=PALETTE["text_muted"],
            )
            ax.set_yticks([])
            span = max(abs(verdict.actual_margin), 1.0) * 1.6
            ax.set_xlim(verdict.actual_margin - span, verdict.actual_margin + span)
        else:
            draws = np.asarray(verdict.margin_draws, dtype=float)
            lower = np.floor(draws.min() / bin_width) * bin_width
            upper = np.ceil(draws.max() / bin_width) * bin_width + bin_width
            edges = np.arange(lower, upper, bin_width)
            counts, edges = np.histogram(draws, bins=edges)
            # Per cent of the simulations, not a density. A density's height
            # depends on the bin width, so the same game drawn at one point and
            # at three would carry two different y axes for the same fact; a
            # share of the runs is the number a reader can actually state.
            counts = counts / counts.sum() * 100.0
            left = edges[:-1]
            # A bar is the home team's when its whole span is a home win. The
            # bin starting at exactly zero is the first one, since a margin of
            # zero is a tie rather than a home win.
            # Full-strength team colour on both sides, no alpha. The baseball
            # chart dilutes its away fill because its two histograms overlap and
            # one would hide the other; these two never overlap — every bar is
            # wholly one side's — so alpha buys nothing and costs the identity
            # the colour is there to carry. Drawn at 0.55, Green Bay's #203731
            # reads as grey.
            colours = [home_colour if edge >= 0 else away_colour for edge in left]
            ax.bar(
                left,
                counts,
                width=bin_width,
                align="edge",
                color=colours,
                edgecolor=PALETTE["bg"],
                linewidth=0.5,
                zorder=2,
            )
            # Headroom first: everything placed in axes fractions above depends
            # on where the bars stop, and `set_ylim` after the fact would move
            # them relative to a plot they were measured against.
            ax.set_ylim(
                0.0, counts.max() * (ANNOTATED_HEADROOM if callout or arrow else PLAIN_HEADROOM)
            )
            ax.set_ylabel("% of simulations", fontsize=9, color=PALETTE["text_muted"])
            # Round 1's review: the figure "makes sense the more you read it".
            # A y axis with nothing on it is one of the reasons — the reader is
            # shown a shape and left to guess what its height means.
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4, min_n_ticks=3))
            ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(decimals=0))
            # No legend row. Round 5: the two half-plane tints and the two
            # corner labels below name the same two clubs, and a legend under
            # the plot said it a second time — in the row the two direction
            # labels now occupy.

        _rule(ax, 0.0, "", color=PALETTE["text_muted"], dashes=(2, 3), weight=1.0)
        deserved_label = _rule(
            ax,
            verdict.deserved_margin,
            anchor_label("Deserved", verdict.deserved_margin, verdict),
            color=PALETTE["text_muted"],
            dashes=(5, 3),
            weight=1.6,
            boxed=True,
        )
        actual_label = _rule(
            ax,
            verdict.actual_margin,
            anchor_label("Actual", verdict.actual_margin, verdict),
            color=PALETTE["text"],
            dashes=(1, 0),
            weight=2.0,
            boxed=True,
        )

        # Pinned before anything is measured against it. The bars and the three
        # rules above are what set the limits — the rules are drawn with
        # `clip_on=False` and still count toward the autoscale, which is why the
        # zero line is always in frame — and a span added under an unresolved
        # autoscale moves the limits the span itself was measured from.
        ax.set_xlim(*ax.get_xlim())
        corners = _draw_side_tints(ax, verdict, home_colour, away_colour, logos, shield=True)

        ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        # Unsigned ticks, and no axis title. `final margin (DET − GB)` is a
        # subtraction the reader had to perform before `-15` meant anything;
        # `15` under `← GB wins by` is the same fact with the arithmetic done.
        ax.xaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda value, _pos: f"{abs(value):g}")
        )
        ax.set_xlabel("")
        _draw_wins_by_labels(fig, ax, verdict, home_colour, away_colour)

        # The count is the number of re-adjudications actually drawn — 200
        # posterior draws x 800 coin draws on the shipped settings — not the
        # coin constant alone. A heading that said "800" over a histogram of
        # 160,000 values would be describing a different figure.
        heading = "Deserve-to-Win"
        if not verdict.is_point_mass:
            heading = f"{heading} — {len(verdict.margin_draws):,} simulations"
        draw_header(ax, verdict, heading)

        caveat = ax.text(
            0,
            -0.42,
            verdict.interval_note(coverage=coverage),
            transform=ax.transAxes,
            fontsize=8,
            color=PALETTE["text_muted"],
            va="top",
        )
        # The caveat is a footnote to the plot and is wrapped to the plot's own
        # width. It used to wrap to the figure's, which `attach_overtime_sidebar`
        # widens — the footnote then ran the full width of the widened figure and
        # under the sidebar's paragraphs.
        _wrap_to_width(fig, caveat, ax.get_window_extent().width)
        # Appended *after* the wrap rather than drawn as a second object below
        # it: the caveat wraps to one line or two depending on the game, and an
        # annotation at a fixed offset under it lands on the second line half
        # the time. Document 16 refused the toss, so a game that had one has to
        # say the ledger is one event short on purpose.
        if verdict.went_to_overtime:
            caveat.set_text(f"{caveat.get_text()}\n{OVERTIME_FOOTER}")
        # Corners first, then each other: a label moved off a corner can land on
        # the other rule's label, and the lift is the move that always works.
        for rule_label in (deserved_label, actual_label):
            _clear_corner_labels(fig, rule_label, corners)
        _lift_colliding_label(fig, deserved_label, actual_label)

        if callout:
            _draw_callout(ax, verdict, home_colour, away_colour)
        # Nothing to span on a degenerate game: the bootstrap never changed its
        # mind, so an arrow there would measure a gap the figure is not about.
        if arrow and not verdict.is_degenerate:
            _draw_luck_arrow(ax, verdict)
        return fig, ax


# --------------------------------------------------------------------------
# the team-points distribution — the share image
# --------------------------------------------------------------------------

# Three points is a field goal, and a field goal is the smallest step a
# scoreboard usually takes. Bins narrower than that comb the histogram into the
# gaps between reachable scores; bins wider than that pool a touchdown with a
# field goal. Aligned to the three-point grid so the same score is the same bar
# in both teams' fills.
TEAM_POINTS_BIN = 3.0

# The two fills overlap, which the margin histogram's never did — every margin
# bar was wholly one side's. An overlap drawn at full strength hides whichever
# team is drawn second, so both are drawn translucent and the shared region
# reads as a third, darker shade rather than as one team's colour.
TEAM_POINTS_ALPHA = 0.6

# ...and every translucent fill is then a diluted colour, which is the trap
# `plot_bootstrap_distribution` already records: at 0.55 Green Bay's #203731
# reads as grey, and a grey fill under a green mark in the legend is identity
# lost. So the silhouette is traced at full strength in the club's own colour
# and the dilution is confined to the interior, where its only job is to let the
# team behind show through.
TEAM_POINTS_OUTLINE = 1.8


def point_bin_edges(values, *, bin_width: float = TEAM_POINTS_BIN) -> np.ndarray:
    """Bin edges on the three-point grid, spanning ``values`` and never negative.

    ``values`` is everything the axis has to hold — both teams' draws and both
    actual scores, so a club whose deserved points sit nowhere near its own
    scoreline still has its rule inside the frame.

    A scoreboard has no negative side, so the grid is anchored at zero and the
    axis stops there. It stops *below* zero only if a distribution actually
    reaches there, because hiding a draw is worse than showing a score no club
    ever put on a board.
    """
    values = np.asarray(values, dtype=float)
    lower = min(0.0, np.floor(values.min() / bin_width) * bin_width)
    if values.min() >= 0.0:
        lower = max(0.0, np.floor(values.min() / bin_width) * bin_width)
    upper = np.ceil(values.max() / bin_width) * bin_width + bin_width
    return np.arange(lower, upper, bin_width)


def most_likely_score(draws, edges) -> int:
    """The centre of the fullest bin, to the nearest point.

    The mode rather than the mean, because a scoreline is a thing that happened
    and a mean of scorelines is not one. Rounded half up rather than with
    numpy's round-half-to-even, so 40.5 is 41 on every game rather than on every
    other one.
    """
    counts, _edges = np.histogram(np.asarray(draws, dtype=float), bins=edges)
    index = int(np.argmax(counts))
    centre = (edges[index] + edges[index + 1]) / 2.0
    return int(np.floor(centre + 0.5))


def most_likely_line(verdict: GameVerdict, home: int, away: int) -> str:
    """`"Most likely: GB 27 – DET 23"` — away first, as the score line reads."""
    return f"Most likely: {verdict.away_team} {away} \u2013 {verdict.home_team} {home}"


def _check_reconciles(verdict: GameVerdict, home_draws: np.ndarray, away_draws: np.ndarray) -> None:
    """The two point distributions must be this game's margin distribution.

    They are a split of one replay, not a second one, so the subtraction is an
    identity rather than an approximation. Drawing a pair that fails it would
    put two teams' scores under a headline computed from different coins.
    """
    if len(home_draws) != len(away_draws) or len(home_draws) != len(verdict.margin_draws):
        raise ValueError(
            f"{verdict.game_id}'s point distributions do not reconcile with its margin "
            f"draws: {len(home_draws)} home and {len(away_draws)} away against "
            f"{len(verdict.margin_draws)} margins."
        )
    drift = float(np.abs((home_draws - away_draws) - np.asarray(verdict.margin_draws)).max())
    if drift > 1e-9:
        raise ValueError(
            f"{verdict.game_id}'s point distributions do not reconcile with its margin "
            f"draws ({drift:.2e} apart at worst). Stop rather than draw them."
        )


def plot_team_points_distribution(
    verdict: GameVerdict,
    home_draws,
    away_draws,
    *,
    bin_width: float = TEAM_POINTS_BIN,
    colors: tuple[str, str] | None = None,
    logos: dict | None = None,
    callout: bool = False,
    legend_logos: bool = False,
):
    """Each team's deserved points, as two overlapping histograms on one points axis.

    The margin histogram answers "by how much"; this answers "what would the
    scoreboard have said", which is the question a reader scrolling past a share
    image actually has. It is the baseball simulator's run-distribution chart in
    this repo's style: two fills, each team's actual score as a dashed rule in
    its own colour, and the most likely scoreline stated in words above the plot
    so nobody has to read a mode off a histogram.

    Nothing here is a new statistic. The two distributions are one replay of the
    same coins the margin distribution is drawn from, split by the team each
    luck event is charged to, and the split is checked against the margin draws
    before a bar is drawn.

    ``colors`` and ``logos`` are the game's ``(home, away)`` pair and marks,
    supplied by the caller — see :func:`plot_bootstrap_distribution`.

    Returns ``(figure, axes)``.
    """
    home_colour, away_colour = colors or (HOME_HUE, AWAY_HUE)
    logos = logos or {}
    home_draws = np.asarray(home_draws, dtype=float)
    away_draws = np.asarray(away_draws, dtype=float)
    _check_reconciles(verdict, home_draws, away_draws)

    scores = [
        float(score) for score in (verdict.home_score, verdict.away_score) if score is not None
    ]
    edges = point_bin_edges(
        np.concatenate([home_draws, away_draws, np.array(scores or [], dtype=float)]),
        bin_width=bin_width,
    )

    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.6, 4.0))

        # Away first, so the home team's fill lands on top of it — the same
        # order the score line and the legend read in, so a reader who checks
        # which fill is in front is checking one convention, not two.
        tallest = 0.0
        for draws, colour in ((away_draws, away_colour), (home_draws, home_colour)):
            counts, _edges = np.histogram(draws, bins=edges)
            # Per cent of the simulations, not a density: a density's height
            # depends on the bin width, so the same game at one point and at
            # three would carry two different y axes for the same fact.
            counts = counts / counts.sum() * 100.0
            tallest = max(tallest, counts.max())
            ax.bar(
                edges[:-1],
                counts,
                width=bin_width,
                align="edge",
                # An RGBA face rather than `alpha=`, which would dilute the
                # outline with the fill and put the identity back where it was.
                color=mpl.colors.to_rgba(colour, TEAM_POINTS_ALPHA),
                linewidth=0,
                zorder=2,
            )
            ax.stairs(
                counts,
                edges,
                color=colour,
                linewidth=TEAM_POINTS_OUTLINE,
                zorder=3,
            )

        ax.set_ylim(0.0, tallest * (ANNOTATED_HEADROOM if callout else PLAIN_HEADROOM))
        ax.set_ylabel("% of simulations", fontsize=9, color=PALETTE["text_muted"])
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4, min_n_ticks=3))
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(decimals=0))
        ax.set_xlim(edges[0], edges[-1])
        ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("points scored", fontsize=9, color=PALETTE["text_muted"])

        # The two scorelines, each in its own club's colour and each boxed, so a
        # reader can see at a glance how far the scoreboard sat from the fill.
        rules = [
            _rule(
                ax,
                float(score),
                f"{team} {score:.0f} (Actual)",
                color=colour,
                dashes=(5, 3),
                weight=2.0,
                boxed=True,
            )
            for team, score, colour in (
                (verdict.away_team, verdict.away_score, away_colour),
                (verdict.home_team, verdict.home_score, home_colour),
            )
            if score is not None
        ]

        heading = "Deserve-to-Win"
        if len(home_draws) > 1:
            heading = f"{heading} \u2014 {len(home_draws):,} simulations"
        draw_header(
            ax,
            verdict,
            heading,
            subtitle_extra=most_likely_line(
                verdict,
                most_likely_score(home_draws, edges),
                most_likely_score(away_draws, edges),
            ),
        )

        entries = [
            (verdict.away_team, logos.get(verdict.away_team)),
            (verdict.home_team, logos.get(verdict.home_team)),
        ]
        if legend_logos:
            _draw_logo_legend(ax, entries, template="{team}")
        else:
            ax.legend(
                handles=[
                    Patch(
                        facecolor=mpl.colors.to_rgba(colour, TEAM_POINTS_ALPHA),
                        edgecolor=colour,
                        linewidth=TEAM_POINTS_OUTLINE,
                        label=team,
                    )
                    for team, colour in (
                        (verdict.away_team, away_colour),
                        (verdict.home_team, home_colour),
                    )
                ],
                loc="upper center",
                bbox_to_anchor=(0.5, -0.20),
                ncol=2,
                frameon=False,
                fontsize=9,
                handlelength=1.1,
                handleheight=0.9,
            )

        caveat = ax.text(
            0,
            -0.42,
            verdict.interval_note(),
            transform=ax.transAxes,
            fontsize=8,
            color=PALETTE["text_muted"],
            va="top",
        )
        _wrap_to_width(fig, caveat, ax.get_window_extent().width)
        if verdict.went_to_overtime:
            caveat.set_text(f"{caveat.get_text()}\n{OVERTIME_FOOTER}")
        if len(rules) == 2:
            _lift_colliding_label(fig, rules[0], rules[1])

        if callout:
            _draw_callout(ax, verdict, home_colour, away_colour)
        return fig, ax


# --------------------------------------------------------------------------
# the luck ledger
# --------------------------------------------------------------------------

# Below this many points a bar is a sliver a reader cannot see, and a game with
# five extra points in it would spend five rows drawing nothing. Folding is not
# dropping: the folded row carries their exact sum, so the waterfall still
# reconciles. Presentation only — the ledger itself keeps every event.
POINTS_FLOOR = 0.1

COMPONENT_NAMES = {
    "fumble": "fumble",
    "field_goal": "field goal",
    "extra_point": "extra point",
}

# The ledger's fumble classes are `{play type}/{live|aborted}`, which is the
# simulator's vocabulary rather than a reader's. "aborted" is nflverse's word
# for a snap that never got away cleanly, and it is kept because "fumble on a
# run" would describe a botched exchange as a play that was actually run.
PLAY_WORDS = {
    "pass": "a pass",
    "run": "a run",
    "punt": "a punt",
    "kickoff": "a kickoff",
    "field_goal": "a field goal",
    "aborted pass": "an aborted pass",
    "aborted run": "an aborted run",
    "aborted punt": "an aborted punt",
    "aborted field_goal": "an aborted field goal",
}


@dataclass(frozen=True)
class LuckBar:
    """One signed step from the actual margin toward the deserved one."""

    label: str
    points: float
    play_id: float | None = None
    n_events: int = 1
    # The team the event is charged to, so the row can wear that club's mark.
    # ``None`` on the folded row: a sum of several teams' slivers is nobody's
    # event, and stamping one club on it would say it was.
    team: str | None = None


def _fumble_phrase(event_class: str) -> str:
    """`"run/aborted"` -> `"an aborted run"`, falling back to the class as written."""
    play, _, liveness = str(event_class).partition("/")
    key = f"aborted {play}" if liveness == "aborted" else play
    return PLAY_WORDS.get(key, f"a {key.replace('_', ' ')}")


KICKS = ("field_goal", "extra_point")


def _with_kicker(phrase: str, row: dict) -> str:
    """`"41-yd field goal"` -> `"41-yd field goal · Crosby"`, when a name is known.

    Round 4: three of Green Bay's five 2018 misses were the same kicker, and a
    column of anonymous field goals hides that. The name is presentation only —
    the pricing already used that kicker's shrunk rate — and it is never
    invented: a play-by-play row without a kicker keeps the label it had.
    """
    kicker = row.get("kicker")
    return f"{phrase} \u00b7 {kicker}" if kicker else phrase


def event_phrase(row: dict) -> str:
    """The event itself, in plain words and without the team it is charged to.

    `"42-yd field goal · Crosby"`, `"fumble on a punt"`. The team is dropped
    because the ledger card puts each event in its own team's table, where a
    prefix would repeat the heading on every row; :func:`plain_label` puts it
    back for the figures whose rows are not grouped.

    ``kick_distance`` is optional and is never invented: the ledger stores a
    five-yard class, and printing the class midpoint as if it were the distance
    would be making up a number, so the class is printed instead.
    """
    component = str(row["component"])
    if component == "field_goal":
        distance = row.get("kick_distance")
        where = f"{float(distance):.0f}-yd" if distance is not None else str(row["event_class"])
        return _with_kicker(f"{where} field goal", row)
    if component == "extra_point":
        return _with_kicker("extra point", row)
    if component == "fumble":
        return f"fumble on {_fumble_phrase(row['event_class'])}"
    # An unfamiliar component still gets a row rather than a crash: the ledger
    # is allowed to grow a fourth kind of event before this function knows it.
    name = COMPONENT_NAMES.get(component, component.replace("_", " "))
    event_class = str(row["event_class"])
    return name if event_class == name else f"{event_class} {name}"


def outcome_phrase(row: dict) -> str:
    """What happened: `"missed (88% kick)"`, `"retained"`, `"recovered by DET"`.

    Empty when the ledger row does not carry its branch — :func:`ledger.with_actual`
    recovers ``actual`` from the identity where it can, and where it cannot the
    outcome is unknown and is left unsaid rather than guessed at.

    A kick also states what it was expected to do. Round 4's note: a 41-yard
    miss costs 3.2 points and a 42-yard miss 3.1, and without the make
    probability beside them the two points look like a rounding difference
    rather than two kicks of different difficulty. The percentage is the row's
    own ``expected`` — the shrunk rate the luck was priced at, not a new number
    — and a row that does not carry one keeps the bare word.
    """
    branch = row.get("actual")
    if branch is None:
        return ""
    made = bool(round(float(branch)))
    component = str(row["component"])
    if component in KICKS:
        happened = "made" if made else "missed"
        expected = row.get("expected")
        if expected is None:
            return happened
        return f"{happened} ({round(float(expected) * 100)}% kick)"
    if component == "fumble":
        # Asymmetric on purpose. A fumble the fumbling team recovered is
        # "retained" — "DET fumble, recovered by DET" says the same thing twice
        # and reads as a mistake. A fumble it lost names who got it, because
        # that is the fact a reader wants and the ledger does not record it.
        if made:
            return "retained"
        opponent = row.get("opponent")
        return f"recovered by {opponent}" if opponent else "lost"
    return ""


def plain_label(row: dict) -> str:
    """One luck event in plain words: `"GB 42-yd field goal, made"`.

    The ledger's own vocabulary is the simulator's — `"40-44 yd field goal — GB"`,
    `"run/aborted fumble — DET"` — and it is exactly right for auditing a ledger
    and exactly wrong on a figure somebody reads once. This is the same row said
    the way it would be said out loud: the team, the event, and what happened.
    """
    outcome = outcome_phrase(row)
    head = f"{row['charged_team']} {event_phrase(row)}"
    return f"{head}, {outcome}" if outcome else head


def luck_bars(
    rows,
    *,
    points_per_epa: float,
    floor: float = POINTS_FLOOR,
    chronological: bool = False,
) -> list[LuckBar]:
    """One bar per ledger row: the points neutralizing that event takes off the margin.

    The simulator's identity is ``deserved = actual - total_luck_epa * points_per_epa``,
    so a bar is ``-luck_epa * points_per_epa`` — luck that favoured the home team
    comes *off* the home team's margin. Because the identity is a sum, the bars
    reconcile the two ends exactly no matter what order they are drawn in.

    Default order is biggest mover first, which answers the question the figure is
    asked ("what moved the verdict"). ``chronological=True`` orders by play instead,
    which tells the game's story rather than the adjudication's.
    """
    bars = [
        LuckBar(
            label=plain_label(row),
            points=-float(row["luck_epa"]) * points_per_epa,
            play_id=float(row["play_id"]),
            team=row.get("charged_team"),
        )
        for row in rows
    ]
    small = [bar for bar in bars if abs(bar.points) < floor]
    # A lone sliver is left where it is — "1 events under 0.1 pt" is a worse row
    # than the event itself.
    if len(small) < 2:
        small = []
    kept = [bar for bar in bars if bar not in small]

    if chronological:
        kept.sort(key=lambda bar: bar.play_id)
    else:
        kept.sort(key=lambda bar: abs(bar.points), reverse=True)

    if small:
        kept.append(
            LuckBar(
                label=f"{len(small)} events under {floor:g} pt",
                points=sum(bar.points for bar in small),
                play_id=None,
                n_events=len(small),
            )
        )
    return kept


def running_totals(bars: Sequence[LuckBar], start: float) -> list[tuple[float, float]]:
    """Where each step begins and ends, walking from the actual margin."""
    spans, running = [], start
    for bar in bars:
        spans.append((running, running + bar.points))
        running += bar.points
    return spans


def _favoured(margin: float, verdict: GameVerdict) -> str:
    """Which side a signed home-perspective margin favours."""
    return verdict.home_team if margin > 0 else verdict.away_team


def logo_zoom(logo, fig, *, max_width_in: float, max_height_in: float) -> float:
    """The ``OffsetImage`` zoom that fits ``logo`` inside a box, in inches.

    Both dimensions are bounded, and that is the point. nflverse's marks are not
    one shape: most clubs ship a squarish shield, but the Jets' is a wide, short
    wordmark. Scaled to a target *width* the wordmark comes out the right width
    and far too tall; scaled to a fixed zoom it came out four inches wide and
    printed straight through the waterfall's first bar. Fitting to a box makes
    every club's mark the same visual weight whatever its aspect ratio.

    ``OffsetImage`` sizes in points, so the zoom for a given pixel size depends
    on the figure's dpi; that is solved for here rather than tuned by eye.
    """
    height, width = logo.shape[:2]
    scale = min(max_width_in * fig.dpi / width, max_height_in * fig.dpi / height)
    return scale * 72.0 / fig.dpi


def _stamp_logo(ax, logo, x: float, y: float) -> None:
    """Put a club's mark on a bar end, or nothing when there is no mark to put."""
    if logo is None:
        return
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    zoom = logo_zoom(logo, ax.figure, max_width_in=0.46, max_height_in=0.22)
    box = AnnotationBbox(
        OffsetImage(logo, zoom=zoom),
        (x, y),
        xybox=(20 if x >= 0 else -20, 0),
        xycoords="data",
        boxcoords="offset points",
        frameon=False,
        annotation_clip=False,
        zorder=6,
    )
    ax.add_artist(box)


def _left_edge_points(ax) -> float:
    """How far left of the axes the figure's content already runs, in points.

    Negative. A waterfall's row labels are sentences drawn outside the axes, and
    ``bbox_inches="tight"`` crops the saved PNG to include them — so the visual
    left margin of the image is the leftmost label, not the axes and not the
    figure's own x=0. Measuring it is the only way to put a title on it.
    """
    labels = [label for label in ax.get_yticklabels() if label.get_text()]
    if not labels:
        return 0.0
    renderer = _renderer(ax.figure)
    leftmost = min(label.get_window_extent(renderer).x0 for label in labels)
    return (leftmost - ax.get_window_extent().x0) * 72.0 / ax.figure.dpi


def _stamp_row_logos(ax, bars, rows_y, logos) -> None:
    """A club's mark immediately left of each row's own label.

    Left of *its own* label rather than at a shared column: the labels are right
    aligned on the axis and vary in length, so a fixed column would leave a short
    row's mark stranded halfway across the figure.
    """
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    renderer = _renderer(ax.figure)
    box = ax.get_window_extent()
    labels = ax.get_yticklabels()
    for y, bar in zip(rows_y[1:-1], bars, strict=True):
        logo = logos.get(bar.team)
        if logo is None:
            continue
        left = labels[int(round(y))].get_window_extent(renderer).x0
        ax.add_artist(
            AnnotationBbox(
                OffsetImage(
                    logo, zoom=logo_zoom(logo, ax.figure, max_width_in=0.20, max_height_in=0.14)
                ),
                ((left - box.x0) / box.width - 0.012, y),
                xycoords=("axes fraction", "data"),
                frameon=False,
                annotation_clip=False,
                box_alignment=(1.0, 0.5),
                zorder=6,
            )
        )


def _draw_ledger_arrow(ax, verdict: GameVerdict, rows_y, x_rail: float) -> None:
    """The same span the distribution draws, run down the waterfall's right side.

    Head at the **actual** end, because that is the direction luck pushed the
    game; the label is the distribution's, word for word, so the two figures say
    the same sentence about the same game.
    """
    gap = verdict.actual_margin - verdict.deserved_margin
    toward = verdict.home_team if gap > 0 else verdict.away_team
    top, bottom = float(rows_y[0]), float(rows_y[-1])
    for y, x_end in ((top, verdict.actual_margin), (bottom, verdict.deserved_margin)):
        ax.plot([x_end, x_rail], [y, y], color=PALETTE["grid"], linewidth=0.8, zorder=1)
    ax.annotate(
        "",
        xy=(x_rail, top),
        xytext=(x_rail, bottom),
        arrowprops={
            "arrowstyle": "->",
            "color": PALETTE["text_muted"],
            "linewidth": 1.1,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
        annotation_clip=False,
        zorder=5,
    )
    ax.text(
        x_rail,
        (top + bottom) / 2.0,
        f"luck moved the margin {abs(gap):.1f} points toward {toward}",
        rotation=90,
        rotation_mode="anchor",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=PALETTE["text_muted"],
        zorder=5,
    )


# How much taller the two totals are than a luck event's bar. Round 4: ink
# alone cannot separate a total from Las Vegas's #000000 — the two are 0.18
# apart on the clash scale, under the 0.20 floor — so the ends carry a second,
# non-colour channel. Height is the one that survives a black club, a very dark
# green one, and a greyscale print.
ANCHOR_HEIGHT = 1.4

# Below this a bar is narrower than its own label, so the label is pushed clear
# of the bar and joined to it by a leader. Document 42's D-4: on
# `2025_17_DET_MIN` a -0.3 bar's label sat against the bar's edge and printed
# through the dashed zero rule beside it.
LEADER_FLOOR = 0.5

# The side-of-zero tints. Faint enough to be a background and not a fill: the
# bars are the data, and a wash that competes with them would say the halves of
# the axis matter more than the events on it.
SIDE_TINT_ALPHA = 0.06

# Verbatim, and the first thing a reader of the waterfall needs. A waterfall
# is a chart type most people have not been taught; three sentences is cheaper
# than losing them.
HOW_TO_READ = (
    "Start at the actual margin. Each bar is one luck event re-priced at its "
    "expectation. The last bar is the margin the game deserved."
)


def anchor_colour(home_colour: str, away_colour: str) -> str:
    """Ink for the two totals, stepping to the neutral when ink is a club's colour.

    Round 4 asked for the ends in ink, so that Actual and Deserved read as the
    figure's two anchors rather than as two more events. On most matchups that
    is exactly right. On three of the round's five example games it is not: ink
    is 0.18 from the Jets' and the Raiders' ``#000000`` and 0.15 from Green
    Bay's ``#203731``, both inside document 42 §3's 0.20 clash floor, and
    `NYJ_SF`'s totals came out the same colour as its luck bars — the defect
    document 42 §6 closed in round 2.

    So the ink is stepped by the clash rule the module already owns rather than
    by taste: a game whose clubs leave ink legible gets ink, and one that does
    not gets the neutral the ends wore before. The two totals are also 1.4x the
    height of every event bar and named `Actual:` / `Deserved:` in their row
    labels, so the reading never rests on the colour alone either way.
    """
    ink = PALETTE["text"]
    if all(colour_distance(ink, colour) >= CLASH_DISTANCE for colour in (home_colour, away_colour)):
        return ink
    return PALETTE["anchor"]


def anchor_label(kind: str, margin: float, verdict: GameVerdict) -> str:
    """`"Actual: DET by 8"`, `"Deserved: GB by 8.3"`, `"Deserved: even"`.

    Round 4: `Actual +8` asks a reader to hold the axis's subtraction in their
    head and work out whose +8 it is. Naming the team does the arithmetic for
    them, which is the whole brief for this figure.

    The actual margin prints whole because it is a scoreboard and the deserved
    one to a tenth because it is an estimate — the same rule
    :func:`margin_sentence` already states.
    """
    if abs(margin) < 0.05:
        return f"{kind}: even"
    size = f"{abs(margin):.0f}" if kind == "Actual" else f"{abs(margin):.1f}"
    return f"{kind}: {_favoured(margin, verdict)} by {size}"


def _draw_side_tints(
    ax, verdict: GameVerdict, home_colour: str, away_colour: str, logos, *, shield: bool = False
) -> list[Annotation]:
    """Whose half of the axis is whose, said in a wash and in two words.

    A margin axis has a meaning either side of zero that the axis label states
    as a subtraction — `final margin (DET − GB)` — and that a reader has to
    unpack before any bar means anything. Two faint tints and two corner labels
    say it directly, so the figure can be read from the top down without ever
    parsing the subtraction.

    ``shield`` backs each corner label in the module's cream, which the
    distribution needs and the waterfall does not: the distribution's three
    rules run the full height of the plot and cross this band, and on
    `2018_05_GB_DET` the solid actual rule at +8 printed straight through
    `DET wins`. It is document 42 D-4's device, applied to the same failure.

    Returns the two corner labels, so a caller with its own furniture up there
    can measure against them — see :func:`_clear_corner_labels`.
    """
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    low, high = ax.get_xlim()
    for span_low, span_high, colour in ((low, 0.0, away_colour), (0.0, high, home_colour)):
        if span_high <= span_low:
            continue
        patch = ax.axvspan(span_low, span_high, facecolor=colour, alpha=SIDE_TINT_ALPHA, zorder=0)
        patch.set_gid("side-span")

    # The gap clears the club's mark, which is anchored at the corner itself.
    corners = []
    for x, team, align, gap in (
        (0.015, verdict.away_team, "left", 26),
        (0.985, verdict.home_team, "right", -26),
    ):
        corner = ax.annotate(
            f"{team} wins",
            xy=(x, 1.0),
            xycoords="axes fraction",
            xytext=(gap, -8),
            textcoords="offset points",
            ha=align,
            va="top",
            fontsize=9,
            fontweight="bold",
            color=PALETTE["text_muted"],
            zorder=5,
            bbox=_shielded() if shield else None,
        )
        corners.append(corner)
        logo = logos.get(team)
        if logo is None:
            continue
        box = AnnotationBbox(
            OffsetImage(
                logo, zoom=logo_zoom(logo, ax.figure, max_width_in=0.22, max_height_in=0.15)
            ),
            (x, 1.0),
            xybox=(0, -8),
            xycoords="axes fraction",
            boxcoords="offset points",
            frameon=False,
            annotation_clip=False,
            # Anchored by its outer edge, so the mark sits inside the frame
            # rather than half of it hanging over the figure's own edge.
            box_alignment=(0.0 if align == "left" else 1.0, 1.0),
            zorder=5,
        )
        box.set_gid("side-header-logo")
        ax.add_artist(box)
    return corners


def plot_luck_ledger(
    verdict: GameVerdict,
    rows,
    *,
    points_per_epa: float,
    floor: float = POINTS_FLOOR,
    chronological: bool = False,
    colors: tuple[str, str] | None = None,
    logos: dict | None = None,
):
    """The luck ledger as a waterfall: actual margin at one end, deserved at the other.

    Every bar is one neutralized event, signed to the home team's margin, and the
    bars are checked against the verdict before anything is drawn — a ledger that
    does not reconcile belongs to another game or another slope, and drawing it
    would put a decomposition under a headline it does not explain.

    ``colors`` and ``logos`` are the game's, supplied by the caller rather than
    looked up here — see :func:`plot_bootstrap_distribution`.

    Returns ``(figure, axes)``.
    """
    home_colour, away_colour = colors or (HOME_HUE, AWAY_HUE)
    ends_colour = anchor_colour(home_colour, away_colour)
    logos = logos or {}
    bars = luck_bars(rows, points_per_epa=points_per_epa, floor=floor, chronological=chronological)
    gap = verdict.deserved_margin - verdict.actual_margin
    drift = abs(sum(bar.points for bar in bars) - gap)
    if drift > 1e-6:
        raise ValueError(
            f"the ledger does not reconcile with {verdict.game_id}: its bars move the margin "
            f"by {sum(bar.points for bar in bars):+.4f} but the verdict moves it by {gap:+.4f} "
            f"({drift:.2e} apart). Stop rather than draw it."
        )

    with mpl.rc_context(STYLE):
        # One row's worth of extra height pays for the band the two side headers
        # sit in, so adding them does not squeeze the rows.
        fig, ax = plt.subplots(figsize=(7.6, 1.9 + 0.34 * (len(bars) + 3)))

        if not bars:
            ax.text(
                0.5,
                0.5,
                "This game had no luck events to neutralise.\n"
                "The deserved margin is the actual one.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color=PALETTE["text_muted"],
            )
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            spans = running_totals(bars, verdict.actual_margin)
            rows_y = np.arange(len(bars) + 2, dtype=float)

            ax.barh(
                rows_y[0],
                abs(verdict.actual_margin),
                left=min(0.0, verdict.actual_margin),
                height=0.62 * ANCHOR_HEIGHT,
                color=ends_colour,
                zorder=2,
            )
            for y, bar, (begin, end) in zip(rows_y[1:-1], bars, spans, strict=True):
                ax.barh(
                    y,
                    abs(bar.points),
                    left=min(begin, end),
                    height=0.62,
                    # By the team the break *helped*, not by the direction
                    # neutralising it moves the margin. GB missing a field goal
                    # moves the margin toward GB, and is DET's lucky break; drawn
                    # the first way a reader who knows the game reads it
                    # backwards, which is exactly what happened in round 1.
                    color=home_colour if bar.points < 0 else away_colour,
                    zorder=2,
                )
                # The number goes at the bar's **tip** — the end away from the
                # running total the next bar picks up — so it lands on the side
                # of zero belonging to the team the break helped. A bar that
                # helped the home team runs leftward as it is neutralised, and
                # its number therefore sits to its right, over the home team's
                # half of the axis. Round 3 put every number at the running
                # total instead, which put a DET-helping bar's value on GB's
                # side and made the reader check the colour to know whose it was.
                helped_home = bar.points < 0
                small = abs(bar.points) < LEADER_FLOOR
                ax.annotate(
                    # A folded row can land under half a tenth, and "-0.0" reads
                    # as a rounding failure rather than as a small number.
                    f"{bar.points:+.2f}" if abs(bar.points) < 0.1 else f"{bar.points:+.1f}",
                    xy=(begin, y),
                    # D-4: a bar under half a point is narrower than its own
                    # label, so the label is pushed clear of it and joined back
                    # by a leader. Without the push it sat against the bar's
                    # edge and, on `2025_17_DET_MIN`, through the zero rule.
                    xytext=((22 if small else 6) * (1 if helped_home else -1), 0),
                    textcoords="offset points",
                    ha="left" if helped_home else "right",
                    va="center",
                    fontsize=8,
                    color=PALETTE["text_muted"],
                    # The shield is what stops the dashed zero rule striking the
                    # number through when the running total sits close to zero.
                    bbox=_shielded(),
                    arrowprops=(
                        {
                            "arrowstyle": "-",
                            "color": PALETTE["grid"],
                            "linewidth": 0.8,
                            "shrinkA": 1.0,
                            "shrinkB": 1.0,
                        }
                        if small
                        else None
                    ),
                    zorder=5,
                )
            ax.barh(
                rows_y[-1],
                abs(verdict.deserved_margin),
                left=min(0.0, verdict.deserved_margin),
                height=0.62 * ANCHOR_HEIGHT,
                color=ends_colour,
                zorder=2,
            )

            # A club's mark at each end, so the two totals are read as "this
            # team's game" without having to parse a sign. The actual end wears
            # the side the scoreboard gave it, the deserved end the side the
            # adjudication gives it — on a flip they are visibly different marks.
            _stamp_logo(
                ax,
                logos.get(_favoured(verdict.actual_margin, verdict)),
                verdict.actual_margin,
                rows_y[0],
            )
            _stamp_logo(
                ax,
                logos.get(_favoured(verdict.deserved_margin, verdict)),
                verdict.deserved_margin,
                rows_y[-1],
            )

            # Connectors, so a step is visibly picked up where the last one left off.
            for y, (_begin, end) in zip(rows_y[1:-1], spans, strict=True):
                ax.plot(
                    [end, end],
                    [y - 0.31, y + 0.69],
                    color=PALETTE["grid"],
                    linewidth=1.0,
                    zorder=1,
                )

            ax.set_yticks(rows_y)
            ax.set_yticklabels(
                [anchor_label("Actual", verdict.actual_margin, verdict)]
                + [bar.label for bar in bars]
                + [anchor_label("Deserved", verdict.deserved_margin, verdict)],
                fontsize=9,
            )
            # Inverted by hand rather than by `invert_yaxis`, because the band
            # the two side headers live in has to be reserved above the first
            # row and an auto-scaled limit has no room in it.
            ax.set_ylim(float(rows_y[-1]) + 0.9, float(rows_y[0]) - 1.3)
            ax.axvline(0.0, color=PALETTE["text_muted"], linewidth=1.0, dashes=(2, 3), zorder=1)
            # Only the directions the game actually has. A key for a colour that
            # appears nowhere sends a reader hunting the figure for it.
            handles = []
            if any(bar.points > 0 for bar in bars):
                handles.append(
                    Patch(facecolor=away_colour, label=f"luck that helped {verdict.away_team}")
                )
            if any(bar.points < 0 for bar in bars):
                handles.append(
                    Patch(facecolor=home_colour, label=f"luck that helped {verdict.home_team}")
                )
            ax.legend(
                handles=handles,
                loc="upper center",
                # Same reason as the titles: a fixed fraction of a height that
                # changes with the row count is not a fixed gap.
                bbox_to_anchor=(0.5, -34 / (ax.get_position().height * fig.get_figheight() * 72)),
                ncol=2,
                frameon=False,
                fontsize=9,
                handlelength=1.1,
                handleheight=0.9,
            )
            # The outermost bar ends at the outermost x, and its value label sits
            # beyond that — without room reserved for it the label runs out of the
            # frame and lands on the row names.
            xs = [0.0, verdict.actual_margin, verdict.deserved_margin]
            xs += [x for span in spans for x in span]
            low, high = min(xs), max(xs)
            # The end bars' club marks hang outside their own ends, so a game
            # with logos needs more room at both edges than one without.
            pad = max((0.20 if logos else 0.12) * (high - low), 0.5)
            # The luck arrow runs down a rail outside the bars, and its label is
            # rotated against that rail, so the frame reserves a lane for both.
            rail_room = max(0.18 * (high - low), 0.8)
            x_rail = high + pad * 0.9
            ax.set_xlim(low - pad, high + pad + rail_room)
            _draw_side_tints(ax, verdict, home_colour, away_colour, logos)

            ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8)
            ax.set_axisbelow(True)
            ax.set_xlabel(
                f"final margin ({verdict.home_team} \u2212 {verdict.away_team})",
                fontsize=9,
                color=PALETTE["text_muted"],
            )
            _draw_ledger_arrow(ax, verdict, rows_y, x_rail)
            _stamp_row_logos(ax, bars, rows_y, logos)

        # A waterfall's height grows with its row count, so anything placed in
        # axes fractions drifts further from the plot the more events a game had.
        # These are offsets in points, which hold still.
        draw_header(
            ax, verdict, "Luck Waterfall", caption=HOW_TO_READ, left_points=_left_edge_points(ax)
        )

        footer = [
            "The bars are a sum, not a sequence: their order does not change where the "
            "waterfall lands."
        ]
        # Document 16 measured the overtime toss and refused it, so a game that
        # went to overtime has to say the ledger is one event short on purpose.
        # Silence would let a reader take the waterfall for the whole story.
        if verdict.went_to_overtime:
            footer.append(OVERTIME_FOOTER)
        ax.annotate(
            "\n".join(footer),
            xy=(0, 0),
            xycoords="axes fraction",
            xytext=(0, -58),
            textcoords="offset points",
            va="top",
            fontsize=8,
            color=PALETTE["text_muted"],
        )
        return fig, ax


# --------------------------------------------------------------------------
# the luck ledger card — the share image
# --------------------------------------------------------------------------

# Top 5 per team, the baseball ledger's own number. Past it the rows are folded
# into one that carries their exact sum, so the card never quietly drops luck.
LEDGER_TOP_ROWS = 5

# The one line that makes the Points column readable. Without it a red "-3.2"
# beside a missed field goal is ambiguous: it could as easily mean the kick cost
# the *other* team three points.
SIGN_CONVENTION = "Points are what the scoreboard gave the team beyond what the play deserved."

# Portrait, and tall enough for two full tables. The height is not a taste: a
# team with more than five luck events spends six rows, two teams spend twelve,
# and the first draft at 12 inches drew the home team's accent bar straight
# through the away team's folded row.
LEDGER_CARD_SIZE_IN = (8.0, 13.0)

# The card's vertical grid, in inches from the bottom. Named because the two
# table sections have to be a fixed distance apart, and that distance follows
# from the row height rather than from a guess.
LEDGER_TITLE_Y = 12.62
LEDGER_HEADER_Y = 12.15
LEDGER_BOXES_Y = 11.42
LEDGER_BOX_HEIGHT = 2.15
# Narrower than the first draft's 3.10, because the lane between the two boxes
# stopped being a two-letter "vs" and became the game's net luck and the sentence
# that states the two margins. A box wide enough for "Washington Commanders" at
# 13 pt is wide enough; the inches it gives back are what make the lane readable.
LEDGER_BOX_WIDTH = 2.55
LEDGER_BOX_X = (1.83, 6.17)
LEDGER_LANE_WIDTH = LEDGER_BOX_X[1] - LEDGER_BOX_X[0] - LEDGER_BOX_WIDTH
LEDGER_SWINGS_Y = 8.90
# The band the two tables live in, and the air between them. Where each section
# starts is computed from how many rows the one above it has: pinning the second
# to a fixed height left a hole down the middle of a card whose first team had
# three luck events, and a hole reads as a bug rather than as margin.
LEDGER_TABLES_TOP = 8.00
LEDGER_TABLES_BOTTOM = 0.34
LEDGER_SECTION_GAP = 0.44


@dataclass(frozen=True)
class LuckRow:
    """One event as the card's table prints it, signed toward one team."""

    event: str
    outcome: str
    points: float
    label: str


@dataclass(frozen=True)
class TeamLuck:
    """One team's side of the ledger: what it netted, and what it netted it on."""

    team: str
    net_points: float
    rows: list[LuckRow]

    @property
    def n_events(self) -> int:
        return len(self.rows)

    @property
    def own_points(self) -> float:
        """The luck on this team's own plays — the sum of its own table.

        Not ``net_points``: that is the *game's* net, which the away team's own
        plays move as surely as the home team's. Denver's round-2 card headlined
        −2.3 over a column of green positives adding to +2.3, and both numbers
        were right. This is the one the column adds up to.
        """
        return sum(row.points for row in self.rows)


def team_ledgers(verdict: GameVerdict, rows, *, points_per_epa: float) -> tuple[TeamLuck, TeamLuck]:
    """The two sides of the ledger, away first, as the card reads them.

    **Luck is zero-sum.** There is one scoreboard, so a point it gave one team
    beyond what the play deserved is a point it took from the other. The two net
    figures are therefore the same number with opposite signs, and that number is
    the gap between the actual margin and the deserved one — the quantity the
    waterfall walks across, stated as a property of a team rather than of a
    margin.

    A **row** is signed toward the team it is charged to, which is why GB missing
    a field goal prints red in GB's table: the scoreboard gave GB fewer points
    than the kick deserved. ``luck_epa`` is signed to the home team's margin
    throughout the ledger, so the sign flips for the away team's rows and not
    for the home team's.
    """
    signed = [(row["charged_team"], float(row["luck_epa"]) * points_per_epa) for row in rows]
    net_home = sum(points for _team, points in signed)

    ledgers = []
    for team in (verdict.away_team, verdict.home_team):
        sign = 1.0 if team == verdict.home_team else -1.0
        own = [
            LuckRow(
                event=event_phrase(row),
                outcome=outcome_phrase(row),
                points=toward_home * sign,
                label=plain_label(row),
            )
            for row, (charged, toward_home) in zip(rows, signed, strict=True)
            if charged == team
        ]
        own.sort(key=lambda entry: abs(entry.points), reverse=True)
        ledgers.append(TeamLuck(team=team, net_points=net_home * sign, rows=own))
    return ledgers[0], ledgers[1]


def table_rows(luck: TeamLuck, *, top: int = LEDGER_TOP_ROWS) -> list[LuckRow]:
    """The rows the card prints, with everything past ``top`` folded into one.

    Folding is not dropping: the folded row carries the exact sum of what it
    replaces, so a reader adding the column still lands on the headline.
    """
    if len(luck.rows) <= top:
        return list(luck.rows)
    kept, rest = luck.rows[:top], luck.rows[top:]
    return [
        *kept,
        LuckRow(
            event=f"and {len(rest)} more",
            outcome="",
            points=sum(entry.points for entry in rest),
            label="",
        ),
    ]


def card_header_lines(verdict: GameVerdict) -> list[str]:
    """The two muted lines under the card's title: the matchup, then the facts."""
    matchup = f"{verdict.away_team} @ {verdict.home_team}"
    date = verdict.date_line().strip("()")
    if date:
        matchup = f"{matchup}  \u2022  {date}"
    if verdict.home_score is not None and verdict.away_score is not None:
        final = (
            f"Final: {verdict.away_team} {verdict.away_score:.0f}, "
            f"{verdict.home_team} {verdict.home_score:.0f}"
        )
    else:
        final = verdict.score_line()
    return [matchup, f"{final}  |  DTW: {verdict.headline()}"]


def net_luck(away: TeamLuck, home: TeamLuck) -> tuple[float, str]:
    """The game's net luck and the team it favoured, from the two own-plays sums.

    Home own-plays minus away own-plays is the gap between the actual margin and
    the deserved one, which is why the card can state it as one line instead of
    printing a signed number in each box and leaving the reader to subtract.
    Returned unsigned, with the name of whoever it favoured, because a signed
    number in the lane between two boxes has no side to belong to.
    """
    net = home.own_points - away.own_points
    return abs(net), (home.team if net >= 0 else away.team)


def margin_sentence(verdict: GameVerdict) -> str:
    """`"DET won by 8, deserved to lose by 8.3"` — both margins, one subject.

    The subject is always the scoreboard winner, so the two clauses are about the
    same team and the reader never has to flip a sign mid-sentence. The actual
    margin prints whole because it is a scoreboard, the deserved one to a tenth
    because it is an estimate.
    """
    gap = verdict.actual_margin - verdict.deserved_margin
    if abs(gap) < 0.05:
        return "The deserved margin is the actual one."

    deserved = verdict.deserved_margin
    winner = verdict.scoreboard_winner
    if winner is None:
        favoured = verdict.home_team if deserved > 0 else verdict.away_team
        scores = ""
        if verdict.home_score is not None and verdict.away_score is not None:
            scores = f" {verdict.away_score:.0f}\u2013{verdict.home_score:.0f}"
        return f"tied{scores}, {favoured} deserved to win by {abs(deserved):.1f}"

    toward_winner = deserved if winner == verdict.home_team else -deserved
    actual = f"{winner} won by {abs(verdict.actual_margin):.0f}"
    if abs(toward_winner) < 0.05:
        return f"{actual}, deserved dead level."
    verb = "win" if toward_winner > 0 else "lose"
    return f"{actual}, deserved to {verb} by {abs(toward_winner):.1f}"


def _luck_colour(points: float) -> str:
    """Green for luck a team received, red for luck it paid, ink for neither."""
    if points > 0:
        return PALETTE["good"]
    if points < 0:
        return PALETTE["bad"]
    return PALETTE["text_muted"]


def points_label(points: float) -> str:
    """A signed points figure, to a tenth — or to a hundredth when a tenth is 0.

    "+0.0" beside an event that really did move the margin reads as a rounding
    failure rather than as a small number. The rule holds for a box headline as
    well as for a row: a headline that rounded a real +0.04 to +0.0 would stop
    matching the column it is the sum of, which is the whole point of round 3.
    """
    return f"{points:+.2f}" if abs(points) < 0.1 else f"{points:+.1f}"


def _headline_colour(points: float) -> str:
    """The box headline's colour, which unlike a row has no neutral state.

    A row printing "+0.00" is a real event that moved nothing, and muting it says
    so. A headline is a verdict on a whole team's luck, and a muted one reads as
    missing data rather than as nothing owed, so zero goes green with the rest of
    the not-in-debt.
    """
    return PALETTE["good"] if points >= 0 else PALETTE["bad"]


def _rounded(ax, xy, width, height, *, facecolor, edgecolor="none", pad=0.06, zorder=1):
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad={pad}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.8,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _card_logo(ax, logo, x, y, *, max_width_in, max_height_in):
    if logo is None:
        return
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    ax.add_artist(
        AnnotationBbox(
            OffsetImage(
                logo,
                zoom=logo_zoom(
                    logo, ax.figure, max_width_in=max_width_in, max_height_in=max_height_in
                ),
            ),
            (x, y),
            frameon=False,
            annotation_clip=False,
            box_alignment=(0.5, 0.5),
            zorder=4,
        )
    )


def _draw_team_box(ax, luck: TeamLuck, names, *, x_centre, y_top, colour, logo, detail):
    """One club's summary panel: mark, name, its own-plays luck, one muted line."""
    width, height = LEDGER_BOX_WIDTH, LEDGER_BOX_HEIGHT
    left = x_centre - width / 2.0
    _rounded(
        ax,
        (left, y_top - height),
        width,
        height,
        facecolor=PALETTE["row_alt"],
        edgecolor=PALETTE["grid"],
        pad=0.10,
        zorder=1,
    )
    # The accent bar names the club before the reader has read anything.
    _rounded(ax, (left, y_top - 0.10), width, 0.08, facecolor=colour, pad=0.02, zorder=2)

    _card_logo(ax, logo, x_centre, y_top - 0.52, max_width_in=1.05, max_height_in=0.52)
    ax.text(
        x_centre,
        y_top - 0.95,
        team_or_abbr(luck.team, names),
        fontsize=13,
        fontweight="bold",
        ha="center",
        va="top",
        color=colour,
        zorder=3,
    )
    # The headline is the sum of the table underneath it, to the last decimal.
    # Round 2 headlined the game's net here and labelled the difference; round 3
    # made the two one quantity, so there is nothing left to reconcile. The
    # game's net moved to the lane between the boxes, where it belongs to both.
    ax.text(
        x_centre,
        y_top - 1.18,
        "LUCK ON OWN PLAYS",
        fontsize=7.5,
        fontweight="bold",
        ha="center",
        va="top",
        color=PALETTE["text_muted"],
        zorder=3,
    )
    ax.text(
        x_centre,
        y_top - 1.34,
        f"{points_label(luck.own_points)} points",
        fontsize=19,
        fontweight="bold",
        ha="center",
        va="top",
        color=_headline_colour(luck.own_points),
        zorder=3,
    )
    ax.plot(
        [x_centre - width / 2 + 0.28, x_centre + width / 2 - 0.28],
        [y_top - 1.68, y_top - 1.68],
        color=PALETTE["grid"],
        linewidth=0.6,
        zorder=2,
    )
    ax.text(
        x_centre,
        y_top - 1.80,
        detail,
        fontsize=9.5,
        ha="center",
        va="top",
        color=PALETTE["text_muted"],
        zorder=3,
    )


def team_or_abbr(team: str, names: dict | None) -> str:
    """The club's full name if the caller supplied one, else its abbreviation.

    Names come from `teams.team_name`, and this module never looks one up — it
    is a presentation layer with no data dependency of its own, exactly as the
    colours and the marks are handed to it.
    """
    return (names or {}).get(team, team)


LEDGER_ROW_HEIGHT = 0.42


def section_height(luck: TeamLuck) -> float:
    """How tall one team's table is, in inches: header, columns, then its rows.

    Measured rather than assumed, because everything below a section is placed
    from it — and a team's row count is a property of the game, not of the card.
    """
    rows = max(1, len(table_rows(luck)))
    return 0.76 + 0.48 + (rows - 1) * LEDGER_ROW_HEIGHT + LEDGER_ROW_HEIGHT / 2


def section_tops(away: TeamLuck, home: TeamLuck) -> tuple[float, float]:
    """Where the two tables start, centred in the band they share.

    Two full tables fill the band; two short ones sit in the middle of it with
    equal air above and below, which reads as a margin. Either way the second
    follows the first rather than waiting at a fixed height for it.
    """
    heights = (section_height(away), section_height(home))
    total = heights[0] + LEDGER_SECTION_GAP + heights[1]
    available = LEDGER_TABLES_TOP - LEDGER_TABLES_BOTTOM
    top = LEDGER_TABLES_TOP - max(0.0, (available - total) / 2.0)
    return top, top - heights[0] - LEDGER_SECTION_GAP


def _draw_team_table(ax, luck: TeamLuck, names, *, y_top, colour, logo, columns):
    """One club's table: accent bar, header, column names, then striped rows."""
    _rounded(ax, (0.55, y_top - 0.08), 6.90, 0.08, facecolor=colour, pad=0.02, zorder=2)
    _card_logo(ax, logo, 0.92, y_top - 0.36, max_width_in=0.40, max_height_in=0.26)
    ax.text(
        1.25,
        y_top - 0.36,
        team_or_abbr(luck.team, names),
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="center",
        color=colour,
        zorder=3,
    )
    # Not the event count — the box above already carries it, and a card that
    # prints the same fact twice reads as a mistake.
    ax.text(
        7.45,
        y_top - 0.36,
        "biggest first",
        fontsize=9,
        ha="right",
        va="center",
        color=PALETTE["text_muted"],
        zorder=3,
    )

    header_y = y_top - 0.76
    for key, label, align in columns:
        ax.text(
            key,
            header_y,
            label,
            fontsize=9.5,
            fontweight="bold",
            ha=align,
            va="center",
            color=PALETTE["text"],
            zorder=3,
        )
    ax.plot(
        [0.55, 7.45],
        [header_y - 0.20, header_y - 0.20],
        color=PALETTE["grid"],
        linewidth=0.8,
        zorder=2,
    )

    for index, row in enumerate(table_rows(luck)):
        y = header_y - 0.48 - index * LEDGER_ROW_HEIGHT
        _rounded(
            ax,
            (0.55, y - LEDGER_ROW_HEIGHT / 2 + 0.06),
            6.90,
            LEDGER_ROW_HEIGHT - 0.12,
            facecolor=PALETTE["row_alt"] if index % 2 == 0 else PALETTE["bg"],
            edgecolor=PALETTE["grid"],
            pad=0.04,
            zorder=1,
        )
        ax.text(
            columns[0][0],
            y,
            row.event,
            fontsize=10,
            ha=columns[0][2],
            va="center",
            color=PALETTE["text"],
            zorder=3,
        )
        ax.text(
            columns[1][0],
            y,
            row.outcome or "\u2014",
            fontsize=10,
            ha=columns[1][2],
            va="center",
            color=PALETTE["text_muted"],
            zorder=3,
        )
        ax.text(
            columns[2][0],
            y,
            points_label(row.points),
            fontsize=12,
            fontweight="bold",
            ha=columns[2][2],
            va="center",
            color=_luck_colour(row.points),
            zorder=3,
        )


def plot_luck_ledger_card(
    verdict: GameVerdict,
    rows,
    *,
    points_per_epa: float,
    colors: tuple[str, str] | None = None,
    logos: dict | None = None,
    names: dict | None = None,
):
    """The luck ledger as a portrait share image, in the baseball card's shape.

    The waterfall answers "how did the margin get from there to here", which is
    an article's question. This answers the one a reader scrolling past actually
    asks — *who got the breaks, and on what* — and answers it in two numbers and
    ten rows.

    The ledger is checked against the verdict before anything is drawn, exactly
    as the waterfall checks it: a decomposition printed under a headline it does
    not explain is worse than no decomposition at all.

    ``colors`` and ``logos`` are the game's, supplied by the caller rather than
    looked up here. Returns ``(figure, axes)``.
    """
    home_colour, away_colour = colors or (HOME_HUE, AWAY_HUE)
    logos = logos or {}
    away, home = team_ledgers(verdict, rows, points_per_epa=points_per_epa)

    gap = verdict.actual_margin - verdict.deserved_margin
    drift = abs(home.net_points - gap)
    if drift > 1e-6:
        raise ValueError(
            f"the ledger does not reconcile with {verdict.game_id}: its rows give the home "
            f"team {home.net_points:+.4f} points of luck but the verdict gives it {gap:+.4f} "
            f"({drift:.2e} apart). Stop rather than draw it."
        )

    width_in, height_in = LEDGER_CARD_SIZE_IN
    with mpl.rc_context(STYLE | {"figure.dpi": CARD_DPI}):
        fig = plt.figure(figsize=(width_in, height_in))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, width_in)
        ax.set_ylim(0, height_in)
        ax.axis("off")

        ax.text(
            4.0,
            LEDGER_TITLE_Y,
            "Luck Ledger",
            fontsize=30,
            fontweight="bold",
            ha="center",
            va="top",
            color=PALETTE["text"],
            fontfamily=heading_font(),
        )
        header_y = LEDGER_HEADER_Y
        for line, size in zip(card_header_lines(verdict), (14, 11), strict=True):
            ax.text(
                4.0,
                header_y,
                line,
                fontsize=size,
                ha="center",
                va="top",
                color=PALETTE["text_muted"],
            )
            header_y -= 0.30
        ax.plot([0.55, 7.45], [LEDGER_BOXES_Y + 0.20] * 2, color=PALETTE["grid"], linewidth=0.8)

        # Both boxes carry the same three lines, because they now state the same
        # quantity about two teams. What is true of the matchup rather than of a
        # club — the game's net luck, and which way the two margins ran — goes in
        # the lane between them.
        for luck, x_centre, colour in zip(
            (away, home), LEDGER_BOX_X, (away_colour, home_colour), strict=True
        ):
            _draw_team_box(
                ax,
                luck,
                names,
                x_centre=x_centre,
                y_top=LEDGER_BOXES_Y,
                colour=colour,
                logo=logos.get(luck.team),
                detail=f"{luck.n_events} luck events",
            )
        ax.text(
            4.0,
            LEDGER_BOXES_Y - 0.66,
            "vs",
            fontsize=13,
            ha="center",
            va="center",
            style="italic",
            color=PALETTE["text_muted"],
        )
        size, favoured = net_luck(away, home)
        if size >= 0.05:
            ax.text(
                4.0,
                LEDGER_BOXES_Y - 1.06,
                f"Net luck: {favoured} +{size:.1f}",
                fontsize=10.5,
                fontweight="bold",
                ha="center",
                va="center",
                color=home_colour if favoured == verdict.home_team else away_colour,
            )
        sentence = ax.text(
            4.0,
            LEDGER_BOXES_Y - 1.32,
            margin_sentence(verdict).replace(", ", ",\n"),
            fontsize=8.5,
            ha="center",
            va="top",
            linespacing=1.35,
            color=PALETTE["text_muted"],
        )
        _wrap_to_width(fig, sentence, LEDGER_LANE_WIDTH * fig.dpi)

        ax.plot([0.55, 7.45], [LEDGER_SWINGS_Y + 0.23] * 2, color=PALETTE["grid"], linewidth=0.8)
        ax.text(
            4.0,
            LEDGER_SWINGS_Y,
            "Biggest luck swings",
            fontsize=17,
            fontweight="bold",
            ha="center",
            va="top",
            color=PALETTE["text"],
            fontfamily=heading_font(),
        )
        ax.text(
            4.0,
            LEDGER_SWINGS_Y - 0.34,
            f"Top {LEDGER_TOP_ROWS} on each team's own plays",
            fontsize=10,
            ha="center",
            va="top",
            color=PALETTE["text_muted"],
        )
        ax.text(
            4.0,
            LEDGER_SWINGS_Y - 0.58,
            SIGN_CONVENTION,
            fontsize=9,
            ha="center",
            va="top",
            style="italic",
            color=PALETTE["text_muted"],
        )

        if not rows:
            ax.text(
                4.0,
                4.60,
                "This game had no luck events to re-price.",
                fontsize=14,
                ha="center",
                va="center",
                color=PALETTE["text_muted"],
            )
            return fig, ax

        if verdict.went_to_overtime:
            ax.text(
                4.0,
                0.12,
                OVERTIME_FOOTER,
                fontsize=10,
                ha="center",
                va="center",
                color=PALETTE["text_muted"],
            )

        columns = (
            (1.00, "Event", "left"),
            (4.55, "What happened", "center"),
            (7.40, "Points", "right"),
        )
        for luck, y_top, colour in zip(
            (away, home), section_tops(away, home), (away_colour, home_colour), strict=True
        ):
            _draw_team_table(
                ax,
                luck,
                names,
                y_top=y_top,
                colour=colour,
                logo=logos.get(luck.team),
                columns=columns,
            )
        return fig, ax


# --------------------------------------------------------------------------
# the share card
# --------------------------------------------------------------------------

# Square, because the card exists to be posted and every timeline crops a
# rectangle. 8 in at dpi 200 is 1,600 px, which downsamples cleanly to the
# 400 px the smallest preview gives it.
CARD_SIZE_IN = 8.0
CARD_DPI = 200


def plot_game_card(verdict: GameVerdict, *, colors: tuple[str, str] | None = None, logos=None):
    """The whole adjudication on one square: who won, who deserved to, by how much.

    This is the figure for somebody who will not open the other two. It carries
    no distribution and no ledger — five facts, each large enough to survive a
    400 px preview: the score, the two shares, the verdict, the deserved margin,
    and the one line that stops the share being read as a plain 89%.

    Away on the left and home on the right throughout, which is the order the
    score line already reads in. Returns ``(figure, axes)``.
    """
    home_colour, away_colour = colors or (HOME_HUE, AWAY_HUE)
    logos = logos or {}

    with mpl.rc_context(STYLE | {"figure.dpi": CARD_DPI}):
        fig = plt.figure(figsize=(CARD_SIZE_IN, CARD_SIZE_IN))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        def centred(y, text, **kwargs):
            return ax.text(0.5, y, text, ha="center", va="center", **kwargs)

        centred(
            0.945,
            "Deserve-to-Win",
            fontsize=26,
            fontweight="bold",
            color=PALETTE["text"],
            fontfamily=heading_font(),
        )
        ax.plot([0.08, 0.92], [0.905, 0.905], color=PALETTE["grid"], linewidth=1.0, clip_on=False)

        # The scoreboard: mark, then abbreviation, then the two numbers between.
        for x, team in ((0.16, verdict.away_team), (0.84, verdict.home_team)):
            logo = logos.get(team)
            if logo is not None:
                from matplotlib.offsetbox import AnnotationBbox, OffsetImage

                ax.add_artist(
                    AnnotationBbox(
                        OffsetImage(
                            logo,
                            zoom=logo_zoom(
                                logo,
                                fig,
                                max_width_in=CARD_SIZE_IN * 0.20,
                                max_height_in=CARD_SIZE_IN * 0.155,
                            ),
                        ),
                        (x, 0.780),
                        frameon=False,
                        annotation_clip=False,
                    )
                )
            ax.text(
                x,
                0.660,
                team,
                ha="center",
                va="center",
                fontsize=22,
                fontweight="bold",
                color=PALETTE["text"],
                fontfamily=heading_font(),
            )

        if verdict.home_score is not None and verdict.away_score is not None:
            score_text = f"{verdict.away_score:.0f} – {verdict.home_score:.0f}"
        else:
            score_text = f"{verdict.actual_margin:+.0f}"
        centred(
            0.780,
            score_text,
            fontsize=46,
            fontweight="bold",
            color=PALETTE["text"],
            fontfamily=heading_font(),
        )
        if verdict.date_line():
            centred(0.660, verdict.date_line(), fontsize=13, color=PALETTE["text_muted"])

        # One bar, split where the bootstrap splits it. The split *is* the
        # headline, so there is nothing else on this row to read instead.
        away_share = 1.0 - verdict.dtw_home
        left, width, height, base = 0.08, 0.84, 0.055, 0.500
        ax.barh(base, away_share * width, left=left, height=height, color=away_colour, zorder=2)
        ax.barh(
            base,
            verdict.dtw_home * width,
            left=left + away_share * width,
            height=height,
            color=home_colour,
            zorder=2,
        )
        centred(0.573, "deserve-to-win share", fontsize=11, color=PALETTE["text_muted"])
        home_share = round(verdict.dtw_home * 100)
        ax.text(
            left,
            0.425,
            f"{verdict.away_team} {100 - home_share}%",
            ha="left",
            va="center",
            fontsize=17,
            fontweight="bold",
            color=PALETTE["text"],
        )
        ax.text(
            left + width,
            0.425,
            f"{verdict.home_team} {home_share}%",
            ha="right",
            va="center",
            fontsize=17,
            fontweight="bold",
            color=PALETTE["text"],
        )

        centred(
            0.290,
            verdict.bucket,
            fontsize=20,
            fontweight="bold",
            color=PALETTE["bg"],
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": pill_colour(verdict.bucket),
                "edgecolor": "none",
            },
        )
        centred(0.180, verdict.deserved_line(), fontsize=19, color=PALETTE["text"])

        if verdict.is_degenerate:
            note = (
                "Every re-flip lands the same way, so the interval collapses to a point — "
                f"true of {DEGENERATE_SHARE} of games."
            )
        else:
            low, high = verdict.dtw_interval
            favoured = verdict.deserved_winner
            share_low, share_high = (
                (low, high) if favoured == verdict.home_team else (1 - high, 1 - low)
            )
            # The interval, and nothing else. Round 4: the card is the figure
            # for somebody who will not open the other two, and `(measured
            # coverage 91.5%)` is a methodological aside they cannot act on —
            # it reads as a second, competing percentage beside the one the
            # card is about. Document 10's number is not dropped: it stays on
            # the article figure, where a reader has already asked for the
            # methodology, via `GameVerdict.interval_note`.
            note = (
                f"{NOMINAL_COVERAGE} interval on {favoured}'s share: "
                f"{share_low * 100:.0f}–{share_high * 100:.0f}%."
            )
        centred(0.098, note, fontsize=11, color=PALETTE["text_muted"])

        if verdict.went_to_overtime:
            centred(0.048, OVERTIME_FOOTER, fontsize=11, color=PALETTE["text_muted"])
        return fig, ax


# --------------------------------------------------------------------------
# the flip-band sweep
# --------------------------------------------------------------------------

# 0.00 to 0.15 in hundredths: the band runs from empty (a binary flip label) to
# 0.35–0.65, the widest the round asked for. The shipped 0.10 is on the grid, so
# the display can place the choice rather than interpolate to it.
SWEEP_HALF_WIDTHS = tuple(round(0.01 * step, 4) for step in range(16))


@dataclass(frozen=True)
class BandRow:
    """The three bucket counts at one candidate band.

    ``ties_outside_band`` is carried because document 33 excluded actual ties
    from its flip counts and this module labels them. Without it the sweep's
    ``clear_flip`` and document 33's number differ by an unexplained handful.
    """

    half_width: float
    low: float
    high: float
    clear_flip: int
    too_close: int
    scoreboard_holds: int
    ties_outside_band: int


def band_sweep(
    dtw_home,
    actual_margin,
    *,
    half_widths: Sequence[float] = SWEEP_HALF_WIDTHS,
) -> list[BandRow]:
    """Bucket counts as the "too close to call" band opens from nothing to 0.35–0.65.

    The band at 0.40–0.60 is a presentation choice made before document 33's
    reconciliation, not a threshold fitted to anything, and the only way to show
    that is to show what the neighbouring choices would have said.

    Every count comes from :func:`bucket_label`, so the row at the shipped band is
    the product's own label by construction rather than by agreement.
    """
    dtw_home = np.asarray(dtw_home, dtype=float)
    actual_margin = np.asarray(actual_margin, dtype=float)
    if dtw_home.shape != actual_margin.shape:
        raise ValueError(
            "dtw_home and actual_margin must be the same length — they are one row "
            f"per game ({dtw_home.shape} vs {actual_margin.shape})."
        )

    rows = []
    for half_width in half_widths:
        low, high = round(0.5 - half_width, 4), round(0.5 + half_width, 4)
        labels = [
            bucket_label(dtw, margin, low=low, high=high)
            for dtw, margin in zip(dtw_home, actual_margin, strict=True)
        ]
        ties_outside = sum(
            1
            for margin, label in zip(actual_margin, labels, strict=True)
            if margin == 0 and label != TOO_CLOSE
        )
        rows.append(
            BandRow(
                half_width=half_width,
                low=low,
                high=high,
                clear_flip=labels.count(CLEAR_FLIP),
                too_close=labels.count(TOO_CLOSE),
                scoreboard_holds=labels.count(SCOREBOARD_HOLDS),
                ties_outside_band=ties_outside,
            )
        )
    return rows


def plot_band_sweep(rows: Sequence[BandRow], *, shipped_half_width: float = 0.10):
    """The three bucket counts against the width of the band, one panel each.

    **One panel each, and each on its own scale.** "Scoreboard holds" is an order
    of magnitude larger than the two buckets the band trades between; on one
    shared axis the movement this figure exists to show — a hundred games sliding
    between buckets — is a flat line at the bottom of the frame. Three panels on
    a shared x axis keep every series legible without a second y scale.

    No series wears a colour. A bucket is not an entity the way a team is, and
    with one line to a panel the title already names it — so the ink stays ink and
    nothing here needs a legend.

    Returns ``(figure, axes)``.
    """
    widths = [row.half_width for row in rows]
    panels = (
        (CLEAR_FLIP, [row.clear_flip for row in rows]),
        (TOO_CLOSE, [row.too_close for row in rows]),
        (SCOREBOARD_HOLDS, [row.scoreboard_holds for row in rows]),
    )
    shipped = min(rows, key=lambda row: abs(row.half_width - shipped_half_width))

    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.2), sharex=True)

        for ax, (bucket, counts) in zip(axes, panels, strict=True):
            ax.plot(widths, counts, color=PALETTE["text"], linewidth=1.8, zorder=3)
            ax.plot(widths, counts, "o", color=PALETTE["text"], markersize=2.6, zorder=4)
            ax.axvline(
                shipped.half_width,
                color=PALETTE["text_muted"],
                linewidth=1.0,
                dashes=(2, 3),
                zorder=1,
            )

            at_shipped = counts[rows.index(shipped)]
            ax.plot(
                shipped.half_width, at_shipped, "o", color=PALETTE["text"], markersize=5.5, zorder=5
            )
            # The label sits beside the marker rather than above it: the shipped
            # band's own rule runs vertically through "above", and a number with a
            # dashed line through it reads as struck out. It then goes to whichever
            # side of the point the series is *leaving*, so a rising line does not
            # climb through its own label.
            rising = counts[-1] > counts[0]
            ax.annotate(
                f"{at_shipped:,}",
                xy=(shipped.half_width, at_shipped),
                xytext=(8, -6 if rising else 6),
                textcoords="offset points",
                ha="left",
                va="top" if rising else "bottom",
                fontsize=9,
                color=PALETTE["text"],
                zorder=6,
            )

            ax.set_title(bucket, fontsize=10, color=PALETTE["text"], pad=8, loc="left")
            ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
            ax.set_axisbelow(True)
            ax.margins(y=0.22)
            # Counts of games have no negative side, and a "-50 games" tick is a
            # margin artifact rather than a reading. The top is left free.
            ax.set_ylim(bottom=max(0.0, ax.get_ylim()[0]))
            ax.set_xticks([0.0, 0.05, 0.10, 0.15])
            ax.set_xticklabels(["0.50\nonly", "0.45\nto 0.55", "0.40\nto 0.60", "0.35\nto 0.65"])
            ax.tick_params(labelsize=8)

        axes[0].set_ylabel("games", fontsize=9, color=PALETTE["text_muted"])
        axes[1].set_xlabel(
            "width of the “too close to call” band", fontsize=9, color=PALETTE["text_muted"]
        )

        fig.text(
            0.0,
            1.34,
            "The band is a presentation choice, not a fitted threshold",
            fontsize=13,
            fontweight="bold",
            color=PALETTE["text"],
            va="bottom",
            transform=axes[0].transAxes,
        )
        fig.text(
            0.0,
            1.24,
            f"{len(rows)} candidate bands, "
            f"{panels[0][1][0] + panels[1][1][0] + panels[2][1][0]:,} games — "
            "the shipped 0.40–0.60 is marked",
            fontsize=9,
            color=PALETTE["text_muted"],
            va="bottom",
            transform=axes[0].transAxes,
        )
        fig.text(
            0.5,
            -0.30,
            "Every game lands in exactly one bucket at every width, so the three panels "
            "always sum to the same total.",
            fontsize=8,
            color=PALETTE["text_muted"],
            ha="center",
            va="top",
            transform=axes[1].transAxes,
        )
        return fig, axes


# --------------------------------------------------------------------------
# the overtime toss — reported, not neutralized
# --------------------------------------------------------------------------

# Document 16's verdict: the toss is a real branch worth about two points, and it
# was measured and left out of the ledger because it missed the materiality floor
# by 0.13 pp. The product therefore has to say it out loud rather than let a
# reader assume the simulator never noticed. Figures are strings for the same
# reason the coverage constants above are: they are quotations from a committed
# document, not quantities this module is free to recompute or re-round.
REPORTED_NOT_NEUTRALIZED = "reported, not neutralized"
OVERTIME_TITLE = f"Overtime — {REPORTED_NOT_NEUTRALIZED}"

OVERTIME_GAMES = "155"  # §3, 2016–2025
OVERTIME_SWING = "2.05"  # §8 Gate O-1, points of final margin
OVERTIME_SWING_ETI = "+1.04 to +3.07"
OVERTIME_MEDIAN_MOVE = "3.93 pp"  # §8 Gate O-3
OVERTIME_FLOOR = "4.06 pp"  # the incumbent's own median 89% half-width
OVERTIME_SIDE_FLIPS = "14 of 155"
# §8's impact run was simulator v1.1. The share this module prints is v1.3, so a
# per-game move quoted beside it is a size and not a correction.
OVERTIME_IMPACT_VERSION = "v1.1"
# §4d, pre-registered: the era split cannot be read as a finding either way.
OVERTIME_NEW_RULES_SEASON = 2025
OVERTIME_NEW_RULES_POWER = "0.243"
OVERTIME_NEW_RULES_TRIGGER = "60"

SIDEBAR_WIDTH_IN = 2.7
SIDEBAR_GAP_IN = 0.30
SIDEBAR_WRAP = 42


@dataclass(frozen=True)
class OvertimeToss:
    """One game's overtime toss, as a fact reported beside the adjudication.

    ``received`` is the team that took the first overtime possession, which is a
    *proxy* for winning the toss — nflverse has no coin-toss field, and document
    16 §6 registers that as the component's most serious open defect. The panel
    says so rather than letting the label pass as a measurement.

    ``delta_dtw_home`` is optional and, when given, is the move in the **home**
    team's share from `research/outputs/26_overtime_games.parquet`.
    """

    received: str
    season: int
    delta_dtw_home: float | None = None


def overtime_lines(verdict: GameVerdict, toss: OvertimeToss) -> list[str]:
    """The sidebar's paragraphs, in order.

    Kept separate from the drawing so the wording is testable as wording. Four
    paragraphs always, plus one for a per-game move and one for a game played
    under the 2025 rulebook.
    """
    lines = [
        f"{toss.received} took the first overtime possession. Across "
        f"{OVERTIME_GAMES} overtime games 2016–2025 that is worth "
        f"{OVERTIME_SWING} points of final margin ({NOMINAL_COVERAGE} interval "
        f"{OVERTIME_SWING_ETI}), and the figure books all of it as deserved.",
        "nflverse carries no coin-toss field, so first possession stands in for "
        "winning the toss — document 16 §6's most serious open defect.",
        "The component was measured and refused, not overlooked: neutralizing it "
        f"moves the median overtime game's share by {OVERTIME_MEDIAN_MOVE}, under "
        f"the {OVERTIME_FLOOR} interval the product already prints on those games. "
        f"It changes the deserved winner in {OVERTIME_SIDE_FLIPS}.",
        f"The swing is a league average. {OVERTIME_GAMES} games cannot say which "
        "offenses gain more from receiving, so it is not this game's own number.",
    ]

    if toss.delta_dtw_home is not None:
        favoured = verdict.deserved_winner
        # The stored move is on the home share, and the headline may name the
        # away team. Mirroring it is the same correction `interval_note` makes.
        move = toss.delta_dtw_home if favoured == verdict.home_team else -toss.delta_dtw_home
        lines.append(
            f"Here the toss is worth {move * 100:+.0f} pp of {favoured}'s share — "
            f"measured on simulator {OVERTIME_IMPACT_VERSION} against the "
            "v1.3 share above, so it sizes the toss rather than correcting it."
        )

    if toss.season >= OVERTIME_NEW_RULES_SEASON:
        lines.append(
            f"Played under the {OVERTIME_NEW_RULES_SEASON} overtime rules, which the "
            "swing above pools with the earlier ones. Sixteen games cannot separate "
            f"them — the design has power {OVERTIME_NEW_RULES_POWER} to detect that "
            "the new rules removed the effect entirely — so the era is revisited at "
            f"{OVERTIME_NEW_RULES_TRIGGER} new-rule games and read as nothing before "
            "then."
        )
    return lines


def attach_overtime_sidebar(
    fig,
    ax,
    verdict: GameVerdict,
    toss: OvertimeToss | None,
    *,
    width_in: float = SIDEBAR_WIDTH_IN,
    gap_in: float = SIDEBAR_GAP_IN,
):
    """Put document 16's overtime note beside a figure, or nothing at all.

    ``toss=None`` draws nothing and returns ``None``. Silence is the honest
    annotation for a mechanism that did not occur: a panel explaining that this
    game had no overtime toss would put a caveat where there is no event.

    The figure **grows to the right** rather than the plot shrinking. A sidebar
    that squeezed the axes would re-scale a distribution in order to say
    something beside it, and two games annotated differently would then be drawn
    at two different scales.

    The panel spans the figure's full height rather than the host axes' box,
    because the waterfall's height changes with its row count and a note pinned
    to that box would start lower on a game with more luck events.

    Returns the sidebar's axes.
    """
    if toss is None:
        return None

    with mpl.rc_context(STYLE):
        width, height = fig.get_size_inches()
        box = ax.get_position()
        grown = width + gap_in + width_in
        fig.set_size_inches(grown, height)
        # `set_size_inches` holds axes at their figure *fractions*, which would
        # stretch the plot across the new width. Rescaling by the growth factor
        # holds it at the inches it was drawn at.
        shrink = width / grown
        ax.set_position([box.x0 * shrink, box.y0, box.width * shrink, box.height])

        panel = fig.add_axes([(width + gap_in) / grown, 0.0, width_in / grown, 1.0])
        panel.axis("off")
        # A rule, not a box: the panel is an aside to the figure rather than a
        # second figure, and a full border would read as the latter.
        panel.plot(
            [0.0, 0.0],
            [0.02, 0.98],
            transform=panel.transAxes,
            color=PALETTE["grid"],
            linewidth=1.0,
            clip_on=False,
        )
        panel.text(
            0.07,
            0.98,
            OVERTIME_TITLE,
            transform=panel.transAxes,
            fontsize=9,
            fontweight="bold",
            color=PALETTE["text"],
            va="top",
        )
        panel.text(
            0.07,
            0.94,
            "\n\n".join(
                textwrap.fill(line, SIDEBAR_WRAP) for line in overtime_lines(verdict, toss)
            ),
            transform=panel.transAxes,
            fontsize=7.5,
            color=PALETTE["text_muted"],
            va="top",
            linespacing=1.45,
        )
        return panel
