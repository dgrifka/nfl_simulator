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
        """Document 33 §2a's three-way label.

        A realized tie falls out as a clear flip whenever it is outside the band,
        which is the honest reading: the scoreboard declined to name a winner and
        the bootstrap does not. Document 33 excluded the 10 ties from its flip
        *counts*; a product that has to render one still has to say something.
        """
        if BAND_LOW <= self.dtw_home <= BAND_HIGH:
            return TOO_CLOSE
        return SCOREBOARD_HOLDS if self.deserved_winner == self.scoreboard_winner else CLEAR_FLIP

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
