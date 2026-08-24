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

Palette is the validated light-mode categorical pair on surface ``#fcfcfb``
(``#2a78d6`` / ``#eb6834``: lightness, chroma, CVD separation, normal-vision and
contrast all PASS). Only the two teams wear colour; every rule and label is ink,
so identity is never carried by colour alone. Output is PNG for print — no hover
layer, no dark mode.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

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
NOMINAL_COVERAGE = "89%"
MEASURED_COVERAGE = "91.5%"
DEGENERATE_SHARE = "44.4%"

# --------------------------------------------------------------------------
# style
# --------------------------------------------------------------------------

HOME_HUE, AWAY_HUE = "#2a78d6", "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK_MUTED,
    "axes.edgecolor": GRID,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "figure.dpi": 160,
}


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

    A realized tie falls out as a clear flip whenever it is outside the band,
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
        """None on a realized tie: the scoreboard named nobody."""
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

    def interval_note(self) -> str:
        """The interval, with the two facts that stop it being read as a plain 89%."""
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
        return (
            f"The {NOMINAL_COVERAGE} interval on {favoured}'s share runs "
            f"{share_low * 100:.0f}–{share_high * 100:.0f}%. Document 10 measured that "
            f"interval's coverage at {MEASURED_COVERAGE} on games with something to "
            "adjudicate, so it runs about two points wide."
        )


def verdict_from_row(row: dict, margin_draws: np.ndarray) -> GameVerdict:
    """Build a verdict from a `dtw_games_v13.parquet` row plus its bootstrap draws."""
    return GameVerdict(
        game_id=row["game_id"],
        home_team=row["home_team"],
        away_team=row["away_team"],
        actual_margin=float(row["actual_margin"]),
        deserved_margin=float(row["deserved_margin"]),
        dtw_home=float(row["dtw_home"]),
        dtw_interval=(float(row["dtw_low"]), float(row["dtw_high"])),
        margin_draws=margin_draws,
    )


# --------------------------------------------------------------------------
# the figure
# --------------------------------------------------------------------------


def _rule(ax, x: float, label: str, *, color: str, dashes, weight: float) -> None:
    """A vertical reference rule with its label attached, never colour alone."""
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
    ax.annotate(
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
    )


def plot_bootstrap_distribution(verdict: GameVerdict, *, bin_width: float = 1.0):
    """Deserved margin across the bootstrap, with the realized margin marked.

    The x axis is the home team's margin, so everything right of zero is a home
    win and everything left of it is an away win. The two fills are the two
    teams; the share of the distribution on each side *is* the DTW% in the title.

    Bins are one point of margin wide and aligned to the integer grid. A
    neutralised margin is a sum of a handful of EPA swings, so the distribution
    is genuinely lumpy — clusters a field goal apart, with extra-point structure
    inside them. Bin edges that fall between those clusters comb the histogram
    into alternating spikes and gaps, which reads as a rendering artifact rather
    than as the clustering it actually is. A one-point grid puts the edges where
    a reader already thinks in margins, and zero lands on an edge, so no bar
    straddles the line that decides the winner.

    Returns ``(figure, axes)`` so a caller can add a panel beside it.
    """
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.6, 4.0))

        if verdict.is_point_mass:
            ax.text(
                0.5,
                0.5,
                "This game had no luck events to re-flip.\n"
                "The deserved margin is the realized one.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color=INK_MUTED,
            )
            ax.set_yticks([])
            span = max(abs(verdict.actual_margin), 1.0) * 1.6
            ax.set_xlim(verdict.actual_margin - span, verdict.actual_margin + span)
        else:
            draws = np.asarray(verdict.margin_draws, dtype=float)
            lower = np.floor(draws.min() / bin_width) * bin_width
            upper = np.ceil(draws.max() / bin_width) * bin_width + bin_width
            edges = np.arange(lower, upper, bin_width)
            counts, edges = np.histogram(draws, bins=edges, density=True)
            left = edges[:-1]
            # A bar is the home team's when its whole span is a home win. The
            # bin starting at exactly zero is the first one, since a margin of
            # zero is a tie rather than a home win.
            colours = [HOME_HUE if edge >= 0 else AWAY_HUE for edge in left]
            ax.bar(
                left,
                counts,
                width=bin_width,
                align="edge",
                color=colours,
                edgecolor=SURFACE,
                linewidth=0.5,
                zorder=2,
            )
            ax.set_yticks([])
            ax.set_ylabel("share of re-flips", fontsize=9, color=INK_MUTED)
            ax.legend(
                handles=[
                    Patch(facecolor=AWAY_HUE, label=f"{verdict.away_team} wins"),
                    Patch(facecolor=HOME_HUE, label=f"{verdict.home_team} wins"),
                ],
                loc="upper center",
                bbox_to_anchor=(0.5, -0.20),
                ncol=2,
                frameon=False,
                fontsize=9,
                handlelength=1.1,
                handleheight=0.9,
            )

        _rule(ax, 0.0, "", color=INK_MUTED, dashes=(2, 3), weight=1.0)
        _rule(
            ax,
            verdict.deserved_margin,
            f"deserved {verdict.deserved_margin:+.1f}",
            color=INK_MUTED,
            dashes=(5, 3),
            weight=1.6,
        )
        _rule(
            ax,
            verdict.actual_margin,
            f"realized {verdict.actual_margin:+.0f}",
            color=INK,
            dashes=(1, 0),
            weight=2.0,
        )

        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel(f"margin, {verdict.home_team} perspective", fontsize=9, color=INK_MUTED)

        ax.text(
            0,
            1.30,
            f"{verdict.headline()}    ·    {verdict.bucket}",
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            color=INK,
            va="bottom",
        )
        ax.text(
            0,
            1.20,
            f"{verdict.game_id} — deserve-to-win across the luck-neutralised bootstrap",
            transform=ax.transAxes,
            fontsize=9,
            color=INK_MUTED,
            va="bottom",
        )
        ax.text(
            0,
            -0.42,
            verdict.interval_note(),
            transform=ax.transAxes,
            fontsize=8,
            color=INK_MUTED,
            va="top",
            wrap=True,
        )
        return fig, ax


# --------------------------------------------------------------------------
# the luck ledger
# --------------------------------------------------------------------------

# Below this many points a bar is a sliver a reader cannot see, and a game with
# five extra points in it would spend five rows drawing nothing. Folding is not
# dropping: the folded row carries their exact sum, so the waterfall still
# reconciles. Presentation only — the ledger itself keeps every event.
POINTS_FLOOR = 0.1

# Totals are not luck, so they do not wear a team's colour.
ANCHOR = "#8a8985"

COMPONENT_NAMES = {
    "fumble": "fumble",
    "field_goal": "field goal",
    "extra_point": "extra point",
}


@dataclass(frozen=True)
class LuckBar:
    """One signed step from the realized margin toward the deserved one."""

    label: str
    points: float
    play_id: float | None = None
    n_events: int = 1


def _bar_label(row: dict) -> str:
    """`"45-49 yd field goal — MIN"`, or `"extra point — GB"` when the class is
    the component said twice."""
    component = COMPONENT_NAMES.get(row["component"], str(row["component"]).replace("_", " "))
    event_class = str(row["event_class"])
    head = component if event_class == component else f"{event_class} {component}"
    return f"{head} — {row['charged_team']}"


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
            label=_bar_label(row),
            points=-float(row["luck_epa"]) * points_per_epa,
            play_id=float(row["play_id"]),
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
    """Where each step begins and ends, walking from the realized margin."""
    spans, running = [], start
    for bar in bars:
        spans.append((running, running + bar.points))
        running += bar.points
    return spans


def plot_luck_ledger(
    verdict: GameVerdict,
    rows,
    *,
    points_per_epa: float,
    floor: float = POINTS_FLOOR,
    chronological: bool = False,
):
    """The luck ledger as a waterfall: realized margin at one end, deserved at the other.

    Every bar is one neutralized event, signed to the home team's margin, and the
    bars are checked against the verdict before anything is drawn — a ledger that
    does not reconcile belongs to another game or another slope, and drawing it
    would put a decomposition under a headline it does not explain.

    Returns ``(figure, axes)``.
    """
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
        fig, ax = plt.subplots(figsize=(7.6, 1.9 + 0.34 * (len(bars) + 2)))

        if not bars:
            ax.text(
                0.5,
                0.5,
                "This game had no luck events to neutralise.\n"
                "The deserved margin is the realized one.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color=INK_MUTED,
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
                height=0.62,
                color=ANCHOR,
                zorder=2,
            )
            for y, bar, (begin, end) in zip(rows_y[1:-1], bars, spans, strict=True):
                ax.barh(
                    y,
                    abs(bar.points),
                    left=min(begin, end),
                    height=0.62,
                    color=HOME_HUE if bar.points > 0 else AWAY_HUE,
                    zorder=2,
                )
                ax.annotate(
                    # A folded row can land under half a tenth, and "-0.0" reads
                    # as a rounding failure rather than as a small number.
                    f"{bar.points:+.2f}" if abs(bar.points) < 0.1 else f"{bar.points:+.1f}",
                    xy=(end, y),
                    xytext=(6 if bar.points > 0 else -6, 0),
                    textcoords="offset points",
                    ha="left" if bar.points > 0 else "right",
                    va="center",
                    fontsize=8,
                    color=INK_MUTED,
                    zorder=5,
                )
            ax.barh(
                rows_y[-1],
                abs(verdict.deserved_margin),
                left=min(0.0, verdict.deserved_margin),
                height=0.62,
                color=ANCHOR,
                zorder=2,
            )

            # Connectors, so a step is visibly picked up where the last one left off.
            for y, (_begin, end) in zip(rows_y[1:-1], spans, strict=True):
                ax.plot(
                    [end, end],
                    [y - 0.31, y + 0.69],
                    color=GRID,
                    linewidth=1.0,
                    zorder=1,
                )

            ax.set_yticks(rows_y)
            ax.set_yticklabels(
                [f"realized {verdict.actual_margin:+.0f}"]
                + [bar.label for bar in bars]
                + [f"deserved {verdict.deserved_margin:+.1f}"],
                fontsize=9,
            )
            ax.invert_yaxis()
            ax.axvline(0.0, color=INK_MUTED, linewidth=1.0, dashes=(2, 3), zorder=1)
            # Only the directions the game actually has. A key for a colour that
            # appears nowhere sends a reader hunting the figure for it.
            handles = []
            if any(bar.points < 0 for bar in bars):
                handles.append(
                    Patch(facecolor=AWAY_HUE, label=f"moves the margin toward {verdict.away_team}")
                )
            if any(bar.points > 0 for bar in bars):
                handles.append(
                    Patch(facecolor=HOME_HUE, label=f"moves the margin toward {verdict.home_team}")
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
            pad = max(0.12 * (high - low), 0.5)
            ax.set_xlim(low - pad, high + pad)

            ax.grid(axis="x", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.set_xlabel(f"margin, {verdict.home_team} perspective", fontsize=9, color=INK_MUTED)

        # A waterfall's height grows with its row count, so anything placed in
        # axes fractions drifts further from the plot the more events a game had.
        # These are offsets in points, which hold still.
        ax.annotate(
            f"{verdict.headline()}    ·    {verdict.bucket}",
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, 28),
            textcoords="offset points",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color=INK,
        )
        ax.annotate(
            f"{verdict.game_id} — every luck event, and what neutralising it was worth",
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, 14),
            textcoords="offset points",
            va="bottom",
            fontsize=9,
            color=INK_MUTED,
        )
        ax.annotate(
            "The bars are a sum, not a sequence: their order does not change where the "
            "waterfall lands.",
            xy=(0, 0),
            xycoords="axes fraction",
            xytext=(0, -58),
            textcoords="offset points",
            va="top",
            fontsize=8,
            color=INK_MUTED,
        )
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

    ``ties_outside_band`` is carried because document 33 excluded realized ties
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
            ax.plot(widths, counts, color=INK, linewidth=1.8, zorder=3)
            ax.plot(widths, counts, "o", color=INK, markersize=2.6, zorder=4)
            ax.axvline(shipped.half_width, color=INK_MUTED, linewidth=1.0, dashes=(2, 3), zorder=1)

            at_shipped = counts[rows.index(shipped)]
            ax.plot(shipped.half_width, at_shipped, "o", color=INK, markersize=5.5, zorder=5)
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
                color=INK,
                zorder=6,
            )

            ax.set_title(bucket, fontsize=10, color=INK, pad=8, loc="left")
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.margins(y=0.22)
            # Counts of games have no negative side, and a "-50 games" tick is a
            # margin artifact rather than a reading. The top is left free.
            ax.set_ylim(bottom=max(0.0, ax.get_ylim()[0]))
            ax.set_xticks([0.0, 0.05, 0.10, 0.15])
            ax.set_xticklabels(["0.50\nonly", "0.45\nto 0.55", "0.40\nto 0.60", "0.35\nto 0.65"])
            ax.tick_params(labelsize=8)

        axes[0].set_ylabel("games", fontsize=9, color=INK_MUTED)
        axes[1].set_xlabel("width of the “too close to call” band", fontsize=9, color=INK_MUTED)

        fig.text(
            0.0,
            1.34,
            "The band is a presentation choice, not a fitted threshold",
            fontsize=13,
            fontweight="bold",
            color=INK,
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
            color=INK_MUTED,
            va="bottom",
            transform=axes[0].transAxes,
        )
        fig.text(
            0.5,
            -0.30,
            "Every game lands in exactly one bucket at every width, so the three panels "
            "always sum to the same total.",
            fontsize=8,
            color=INK_MUTED,
            ha="center",
            va="top",
            transform=axes[1].transAxes,
        )
        return fig, axes
