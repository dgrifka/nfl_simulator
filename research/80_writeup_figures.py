"""Community write-up, draft 1 — every image the article references.

One script, one output directory (``docs/writeup/figures/``), so a reader of the
article can regenerate every picture in it with a single command:

    uv run python research/80_writeup_figures.py

Three kinds of figure come out of it.

* **Six per-game figures** are copies of what ``render.render_game`` already
  ships. The article does not draw its own version of a product figure — if the
  waterfall in the write-up disagreed with the waterfall in the product, one of
  them would be wrong and a reader could not tell which.
* **Six explanatory figures** are drawn here, in the house style, from committed
  research artifacts (``research/outputs/``). Each one reproduces a number a
  numbered document already published, and the reproduction is checked in code
  rather than eyeballed: :data:`DOC_CHECKS` fails the run if a redraw drifts.
* **Five formula plates** render the article's LaTeX to PNG, because Medium
  renders neither LaTeX nor Mermaid and the markdown has to carry both forms.

``research/outputs/`` is gitignored; ``docs/writeup/figures/`` is not, because
these images ship with the article. They are the one place in this repo where a
PNG is a committed artifact, which is why the script caps each at 500 KB
(:data:`SIZE_LIMIT`) and the whole set at eighteen files.
"""

from __future__ import annotations

import argparse
import json
import shutil

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from nfl_simulator import paths
from nfl_simulator.plots import BAND_HIGH, BAND_LOW
from nfl_simulator.render import render_game
from nfl_simulator.style import PALETTE, apply_base_style, draw_title_block, finalize, title_axes

FIGURE_DIR = paths.REPO_ROOT / "docs" / "writeup" / "figures"
OUTPUTS = paths.RESEARCH_OUTPUT_DIR

# A committed PNG is a file somebody clones. 500 KB is the article's budget per
# image and the script refuses to leave a heavier one on disk.
SIZE_LIMIT = 500_000
MAX_FILES = 18

# --------------------------------------------------------------------------
# the six per-game figures, copied from the product renderer
# --------------------------------------------------------------------------

# (article filename stem, game id, edition, which of `render.SUFFIXES`).
# "ledger" in an article filename means the **waterfall** — the row-by-row
# figure section 9 walks through — not the square card, which is the share
# image and has no rows to read.
GAME_FIGURES = [
    ("01_gb_det_2018_wk5_dtw_strict", "2018_05_GB_DET", "strict", "dtw"),
    ("02_gb_det_2018_wk5_ledger_strict", "2018_05_GB_DET", "strict", "waterfall"),
    ("03_lv_kc_2021_wk14_dtw_strict", "2021_14_LV_KC", "strict", "dtw"),
    ("04_lac_hou_2024_wk19_ledger_full", "2024_19_LAC_HOU", "full", "waterfall"),
    ("05_den_was_2025_wk13_dtw_full", "2025_13_DEN_WAS", "full", "dtw"),
    ("06_den_was_2025_wk13_ledger_full", "2025_13_DEN_WAS", "full", "waterfall"),
]

# --------------------------------------------------------------------------
# what the redraws have to reproduce
# --------------------------------------------------------------------------

# Every explanatory figure below is drawn from a committed artifact and has to
# land on a number a numbered document already published. These are those
# numbers, with their document, and `check` raises rather than warns.
DOC_CHECKS = {
    # document 02 §1, the pooled recovery-rate split half
    "fumble_rate_split_half_r": (0.051, 0.001),
    # document 02, test 1 — the six split-half correlations
    "split_half_fumble": (0.055, 0.001),
    "split_half_int": (0.164, 0.001),
    # document 04 via document 05 §3 — Buffalo's 10-of-12 shrinking to 48.0%
    "buffalo_2024_posterior": (0.480, 0.001),
    # document 33 §1, §2a, §3
    "n_games_strict": (2761, 0),
    "too_close": (186, 0),
    "sign_flips": (255, 0),
    "degenerate": (1226, 0),
    # document 62 §4
    "n_games_full": (1139, 0),
    "full_bucket_moves": (190, 0),
}


# --------------------------------------------------------------------------
# document 64 — the one-simulator summary
# --------------------------------------------------------------------------

# Round 2 presents **one** simulator: the Full edition over the seasons FTN
# charting reaches. Every headline number the article quotes about that corpus
# is computed here and written into `docs/research/64-one-simulator-summary.md`.
# These are document 64's own numbers rather than a reproduction of an older
# document's, so they are computed and published, not `check`ed — the checks
# below still bind every figure that redraws a *published* number.
ONE_SIM_ARTIFACT = "full_summary.parquet"

# Document 33's degeneracy definition, quoted from `research/48_magnitude_audit.py`
# (which took it from document 10, gate V-3) rather than restated: a DTW% outside
# the open interval is a game the bootstrap never moves off its verdict.
DEGENERATE_LOW = 0.001
DEGENERATE_HIGH = 0.999

# Document 33 §5's margin-movement thresholds, of which the article quotes one.
MARGIN_THRESHOLD = 3.0


def one_simulator_summary() -> dict:
    """Every number document 64 publishes about the 1,139-game corpus.

    Definitions are document 33's, taken from `research/48_magnitude_audit.py`
    verbatim so the two documents can be read side by side: a **sign flip** is a
    sign disagreement between the two margins with both kinds of tie excluded, a
    game is **too close to call** inside DTW% 0.40-0.60, and it is **degenerate**
    outside (0.001, 0.999).

    The three buckets partition the corpus. `clear flip` uses the DTW%
    definition — the bootstrap put the realized winner below even money — with
    the band taking precedence, which is document 33 §2a's resolution and the
    one `full_vs_strict`'s `bucket` already draws.
    """
    games = pl.read_parquet(OUTPUTS / ONE_SIM_ARTIFACT)
    actual = games["actual_margin"].to_numpy().astype(float)
    deserved = games["deserved_margin"].to_numpy().astype(float)
    dtw = games["dtw_home"].to_numpy().astype(float)
    n_events = games["n_events"].to_numpy().astype(int)
    n_dropped = games["n_dropped_picks"].to_numpy().astype(int)
    n_drops = games["n_receiver_drops"].to_numpy().astype(int)
    check("n_games_full", games.height)

    home_won = actual > 0
    realized_tie = actual == 0
    deserved_tie = deserved == 0
    sign_flip = (home_won != (deserved > 0)) & ~realized_tie & ~deserved_tie
    # A drawn game the ledger gives a winner to. Document 33 counts it in its
    # own category rather than as a flip, because a tie has no winner to flip.
    tie_broken = realized_tie & ~deserved_tie

    too_close = (dtw >= BAND_LOW) & (dtw <= BAND_HIGH)
    dtw_flip = (home_won != (dtw > 0.5)) & ~realized_tie
    clear_flip = dtw_flip & ~too_close
    # The band absorbs the two definitions' quarrel — but that is a claim about
    # two label *sets*, so it is compared element-wise. Equal totals would not
    # have shown it: document 33's defect register is a round that thought they
    # would.
    clear_sign_flip = sign_flip & ~too_close
    agrees = ~(clear_flip | too_close)
    degenerate = (dtw <= DEGENERATE_LOW) | (dtw >= DEGENERATE_HIGH)

    shift = np.abs(deserved - actual)
    largest = int(np.argmax(shift))

    return {
        "n_games": games.height,
        "seasons": sorted({int(gid[:4]) for gid in games["game_id"]}),
        "sign_flips": int(sign_flip.sum()),
        "sign_flip_share": float(sign_flip.mean()),
        "sign_flips_naive": int(((deserved > 0) != (actual > 0)).sum()),
        "realized_ties": int(realized_tie.sum()),
        "tie_broken": int(tie_broken.sum()),
        "dtw_flips": int(dtw_flip.sum()),
        # Element-wise, never a difference of two totals: document 33's defect
        # register records the round that quoted a net as a disagreement count.
        "flip_disagreements": int((sign_flip != dtw_flip).sum()),
        "clear_flip_disagreements": int((clear_sign_flip != clear_flip).sum()),
        "non_degenerate": int((~degenerate).sum()),
        "sign_flips_non_degenerate": int((sign_flip & ~degenerate).sum()),
        "clear_flips": int(clear_flip.sum()),
        "too_close": int(too_close.sum()),
        "agrees": int(agrees.sum()),
        "degenerate": int(degenerate.sum()),
        "degenerate_share": float(degenerate.mean()),
        "median_abs_shift": float(np.median(shift)),
        "over_threshold": int((shift > MARGIN_THRESHOLD).sum()),
        "over_threshold_share": float((shift > MARGIN_THRESHOLD).mean()),
        "largest_shift": float(shift[largest]),
        "largest_shift_game": str(games["game_id"][largest]),
        "largest_shift_actual": float(actual[largest]),
        "largest_shift_deserved": float(deserved[largest]),
        "median_events": float(np.median(n_events)),
        "mean_events": float(n_events.mean()),
        "share_with_dropped_pick": float((n_dropped >= 1).mean()),
        "share_with_receiver_drop": float((n_drops >= 1).mean()),
        "median_dropped_picks": float(np.median(n_dropped)),
        "median_receiver_drops": float(np.median(n_drops)),
    }


def print_summary(summary: dict) -> None:
    """Document 64's table, printed so every number in it has a run behind it."""
    n = summary["n_games"]
    seasons = summary["seasons"]

    def share(count: int) -> str:
        return f"{count:5d}  ({count / n * 100:5.2f}%)"

    print(
        f"\n{'=' * 72}\nDOCUMENT 64 — the one simulator, {n:,} games "
        f"{seasons[0]}-{seasons[-1]}\n{'=' * 72}"
    )
    print(f"  games                                {n:5d}")
    print(f"  sign flips (deserved != scoreboard)  {share(summary['sign_flips'])}")
    print(f"  realized ties given a winner         {summary['tie_broken']:5d}")
    print(
        f"  the same count with no tie rule      {summary['sign_flips_naive']:5d}  "
        "(what the handoff's spot check reported)"
    )
    print(f"  DTW% flips (the other definition)    {share(summary['dtw_flips'])}")
    print(
        f"  the two definitions disagree on      {summary['flip_disagreements']:5d}  "
        "(element-wise, never a difference of totals)"
    )
    print(f"  non-degenerate games                 {share(summary['non_degenerate'])}")
    print(
        f"  sign flips among those               {summary['sign_flips_non_degenerate']:5d}  "
        f"({summary['sign_flips_non_degenerate'] / summary['non_degenerate'] * 100:5.2f}% "
        "of the non-degenerate remainder)"
    )
    print(f"  clear flips (DTW%, outside the band) {share(summary['clear_flips'])}")
    print(
        f"  ... on which the two definitions      "
        f"disagree about {summary['clear_flip_disagreements']} games (element-wise)"
    )
    print(f"  too close to call (0.40-0.60)        {share(summary['too_close'])}")
    print(f"  scoreboard holds                     {share(summary['agrees'])}")
    print(f"  degenerate (outside 0.001-0.999)     {share(summary['degenerate'])}")
    print(f"  median |deserved - actual|           {summary['median_abs_shift']:5.2f} pt")
    print(
        f"  moving more than {MARGIN_THRESHOLD:.0f} pt              "
        f"{share(summary['over_threshold'])}"
    )
    print(
        f"  largest swing                        {summary['largest_shift']:5.2f} pt  "
        f"{summary['largest_shift_game']} "
        f"({summary['largest_shift_actual']:+.0f} -> "
        f"{summary['largest_shift_deserved']:+.2f})"
    )
    print(
        f"  luck events per game                 median "
        f"{summary['median_events']:.0f}, mean {summary['mean_events']:.1f}"
    )
    print(
        f"  games with >= 1 dropped-pick chance  {summary['share_with_dropped_pick'] * 100:5.2f}%"
    )
    print(
        f"  games with >= 1 catchable ball       {summary['share_with_receiver_drop'] * 100:5.2f}%"
    )


# The game the article walks through, and the one fumble row document 64 §7
# prints term by term. The play is named rather than picked by a rule so the
# document and the article can never quote two different events: it is the
# Washington sack-fumble at 6:43 of the second quarter.
WALKTHROUGH_GAME = "2025_13_DEN_WAS"
WORKED_FUMBLE_PLAY = 1438.0


def worked_fumble_example() -> dict:
    """One fumble row of the walk-through game, with every term of `luck(e)`.

    The row comes from the Full edition's own replay — the same one the game
    figures are drawn from, and checked against the published summary before it
    is read — so the article's worked example is the event the ledger booked
    rather than an illustration built to match it.
    """
    from nfl_simulator.render import load_sources, replay

    sources = load_sources()
    row = sources.game_row(WALKTHROUGH_GAME, edition="full")
    result, _gaps = replay(
        WALKTHROUGH_GAME, row, sources.schedule_row(WALKTHROUGH_GAME), edition="full"
    )
    ledger = result.ledger.to_frame().filter(
        (pl.col("component") == "fumble") & (pl.col("play_id") == WORKED_FUMBLE_PLAY)
    )
    if ledger.height != 1:
        raise AssertionError(
            f"{WALKTHROUGH_GAME} play {WORKED_FUMBLE_PLAY:.0f} is not one fumble row "
            f"in the Full ledger ({ledger.height} rows). Pick the event again."
        )
    event = ledger.to_dicts()[0]
    quarters = (
        pl.read_parquet(paths.pbp_path(int(WALKTHROUGH_GAME[:4])))
        .filter((pl.col("game_id") == WALKTHROUGH_GAME) & (pl.col("play_id") == WORKED_FUMBLE_PLAY))
        .select("qtr", "time")
        .to_dicts()
    )
    return {
        "game_id": WALKTHROUGH_GAME,
        "play_id": event["play_id"],
        "quarter": int(quarters[0]["qtr"]) if quarters else None,
        "clock": quarters[0]["time"] if quarters else None,
        "event_class": event["event_class"],
        "charged_team": event["charged_team"],
        "y": event["actual"],
        "p": event["expected"],
        "swing": event["swing"],
        "luck_epa": event["luck_epa"],
        "luck_points": event["luck_epa"] * sources.slope,
        "slope": sources.slope,
    }


def print_worked_fumble(event: dict) -> None:
    """Document 64 §7's five lines, each one an arithmetic step a reader can redo."""
    print(f"\n{'=' * 72}\nDOCUMENT 64 §7 — one fumble, term by term\n{'=' * 72}")
    print(f"  game            {event['game_id']}, Q{event['quarter']} {event['clock']}")
    print(f"  class           {event['event_class']}, charged to {event['charged_team']}")
    print(f"  y  (what happened)          {event['y']:.0f}   (1 = the charged team recovered)")
    print(f"  p  (the class's shrunk rate) {event['p']:.4f}")
    print(f"  swing (both branches apart) {event['swing']:+.4f} EPA")
    print(
        f"  luck = (y - p) x swing      {event['luck_epa']:+.4f} EPA"
        f"   = {event['luck_points']:+.3f} pt at {event['slope']:.4f} pt/EPA"
    )


def check(name: str, value: float) -> float:
    """Fail the run when a redraw drifts off the number its document published."""
    expected, tolerance = DOC_CHECKS[name]
    if abs(value - expected) > tolerance:
        raise AssertionError(
            f"{name}: recomputed {value!r}, but the record says {expected!r} "
            f"(tolerance {tolerance}). Stop and reconcile before drawing anything."
        )
    return value


def new_figure(width: float, height: float, *, title: str, subtitle: list[str]):
    """A figure wearing the house title block, with the plot axis under it."""
    fig = plt.figure(figsize=(width, height))
    fig.patch.set_facecolor(PALETTE["bg"])
    draw_title_block(title_axes(fig, height_frac=0.13), title, subtitle, title_size=17)
    ax = fig.add_axes([0.10, 0.14, 0.84, 0.66])
    ax.set_facecolor(PALETTE["bg"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["spine"])
    ax.tick_params(colors=PALETTE["text_muted"], labelsize=9)
    return fig, ax


# --------------------------------------------------------------------------
# figure 7 — fumble recovery does not carry over
# --------------------------------------------------------------------------


def fumble_year_over_year() -> dict:
    """Each team's own-recovery rate in season *t* against season *t+1*.

    Document 02 measured persistence by splitting each team-season in half; this
    is the same claim asked across seasons instead, which is the form a reader
    can check against their own memory of a team. Both numbers go in the
    caption: the doc-02 split-half r, and this scatter's own Pearson r, which
    this script is the trace for.
    """
    frame = pl.read_parquet(OUTPUTS / "fumble_shrinkage.parquet").with_columns(
        pl.col("team_season").str.slice(0, 4).cast(pl.Int32).alias("season"),
        pl.col("team_season").str.slice(5).alias("team"),
    )
    nxt = frame.select(
        pl.col("team"),
        (pl.col("season") - 1).alias("season"),
        pl.col("observed_rate").alias("rate_next"),
        pl.col("n").alias("n_next"),
    )
    pairs = frame.join(nxt, on=["team", "season"], how="inner")
    this_year = pairs["observed_rate"].to_numpy()
    next_year = pairs["rate_next"].to_numpy()
    r = float(np.corrcoef(this_year, next_year)[0, 1])
    league = float(
        frame["k"].sum() / frame["n"].sum()  # pooled, not a mean of rates
    )

    fig, ax = new_figure(
        7.6,
        6.0,
        title="Fumble recovery does not carry over",
        subtitle=[
            f"Own-recovery rate, season t vs season t+1 · {pairs.height} team pairs, 2016–2025",
            "Each dot is one club. If recovery were a skill the dots would line up.",
        ],
    )
    ax.axhline(league, color=PALETTE["anchor"], linewidth=1.4, zorder=2)
    ax.axvline(league, color=PALETTE["anchor"], linewidth=1.4, alpha=0.35, zorder=2)
    ax.scatter(
        this_year,
        next_year,
        s=42,
        facecolor=PALETTE["bad"],
        edgecolor=PALETTE["bg"],
        linewidth=1.2,
        alpha=0.72,
        zorder=3,
    )
    ax.text(
        0.985,
        league + 0.012,
        f"league rate {league:.1%}",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=9,
        color=PALETTE["anchor"],
    )
    ax.set_xlabel("recovery rate, season t", color=PALETTE["text_muted"], fontsize=10)
    ax.set_ylabel("recovery rate, season t+1", color=PALETTE["text_muted"], fontsize=10)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.05, 0.95)
    ax.set_xticks(np.arange(0.1, 0.91, 0.2))
    ax.set_yticks(np.arange(0.1, 0.91, 0.2))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(color=PALETTE["grid"], linestyle="--", alpha=0.6, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.text(
        0.03,
        0.96,
        f"r = {r:+.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=PALETTE["text"],
    )
    finalize(fig, FIGURE_DIR / "07_eda_fumble_recovery.png")
    return {"pairs": pairs.height, "r": r, "league_rate": league}


# --------------------------------------------------------------------------
# figure 8 — the persistence bars
# --------------------------------------------------------------------------

# Document 02 test 1, in the order the document prints them. The luck side is
# the one component whose interval contains zero.
PERSISTENCE_ORDER = [
    ("epa_diff", "EPA differential\n(the whole game)", "skill"),
    ("core", "core offence\n& defence", "skill"),
    ("interception", "interceptions", "skill"),
    ("fg_luck", "field goals", "skill"),
    ("penalty", "penalties", "skill"),
    ("fumble_luck", "fumble recovery", "luck"),
]


def persistence_bars() -> dict:
    """One bar per component: how much of it repeats within a season."""
    results = json.loads((OUTPUTS / "02_skill_vs_luck.json").read_text())
    by_metric = {row["metric"]: row for row in results["persistence"]}
    check("split_half_fumble", round(by_metric["fumble_luck"]["split_half_r"], 3))
    check("split_half_int", round(by_metric["interception"]["split_half_r"], 3))

    labels = [label for _, label, _ in PERSISTENCE_ORDER]
    values = [by_metric[key]["split_half_r"] for key, _, _ in PERSISTENCE_ORDER]
    lows = [by_metric[key]["r_p05"] for key, _, _ in PERSISTENCE_ORDER]
    highs = [by_metric[key]["r_p95"] for key, _, _ in PERSISTENCE_ORDER]
    # Identity, not rank: the colour says which side of the classification a
    # component landed on, so it never moves when the bars are reordered.
    colours = [
        PALETTE["anchor"] if side == "skill" else PALETTE["bad"] for *_, side in PERSISTENCE_ORDER
    ]

    fig, ax = new_figure(
        8.2,
        6.2,
        title="Only one thing in football does not repeat",
        subtitle=[
            "Split-half correlation within a team-season · 320 team-seasons, 200 random splits",
            "Bars are the mean across splits; the line is the 5th–95th percentile (document 02).",
        ],
    )
    positions = np.arange(len(labels))[::-1]
    ax.barh(
        positions,
        values,
        height=0.62,
        color=colours,
        edgecolor=PALETTE["bg"],
        linewidth=2.0,
        zorder=3,
    )
    for pos, low, high in zip(positions, lows, highs, strict=True):
        ax.plot([low, high], [pos, pos], color=PALETTE["text"], linewidth=1.6, zorder=4)
    # The value sits clear of the whole interval, not of the bar: at r = +0.055
    # the 95th percentile runs past the bar's end and struck the label through.
    for pos, value, high in zip(positions, values, highs, strict=True):
        ax.text(
            max(value, high) + 0.018,
            pos,
            f"{value:+.3f}",
            va="center",
            ha="left",
            fontsize=10,
            color=PALETTE["text"],
        )
    ax.axvline(0.0, color=PALETTE["spine"], linewidth=1.2, zorder=2)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9.5, color=PALETTE["text"])
    ax.set_xlim(-0.06, 0.68)
    ax.set_xlabel("split-half correlation r", color=PALETTE["text_muted"], fontsize=10)
    ax.grid(axis="x", color=PALETTE["grid"], linestyle="--", alpha=0.6, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    # Under the axis, not inside it: in the plot area the key landed on the
    # fumble row's interval line, which is the one row a reader goes looking for.
    ax.text(
        0.0,
        -0.155,
        "red = neutralized as luck   ·   grey = kept as skill",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=PALETTE["text_muted"],
    )
    finalize(fig, FIGURE_DIR / "08_eda_int_vs_fumble_persistence.png")
    return {key: by_metric[key]["split_half_r"] for key, _, _ in PERSISTENCE_ORDER}


# --------------------------------------------------------------------------
# figure 9 — shrinkage
# --------------------------------------------------------------------------


def shrinkage() -> dict:
    """Five team-seasons' raw recovery rates, and where the model puts them.

    Real posterior means from the fumble beta-binomial of documents 03–04
    (``fumble_shrinkage.parquet``), not a synthetic illustration. The dropped-pick
    trace was the handoff's first choice and is the wrong object for this
    picture: that model is a crossed logistic regression, and the arrow being
    drawn here is the beta-binomial's.
    """
    frame = pl.read_parquet(OUTPUTS / "fumble_shrinkage.parquet")
    chosen = ["2024_BUF", "2025_KC", "2018_ARI", "2018_MIN", "2021_TB"]
    rows = (
        frame.filter(pl.col("team_season").is_in(chosen))
        .unique(subset="team_season")
        .sort("observed_rate", descending=True)
    )
    buffalo = rows.filter(pl.col("team_season") == "2024_BUF")
    check("buffalo_2024_posterior", round(float(buffalo["posterior_mean"][0]), 3))
    league = float(frame["posterior_mean"].mean())

    fig, ax = new_figure(
        8.0,
        5.6,
        title="What the model does with a lucky season",
        subtitle=[
            "Fumble own-recovery rate: what a club recorded, and what the model believes",
            "Real posterior means from the beta-binomial hierarchy of documents 03–04.",
        ],
    )
    positions = np.arange(rows.height)[::-1]
    ax.axvline(league, color=PALETTE["anchor"], linewidth=1.6, zorder=2)
    ax.text(
        league,
        len(positions) - 0.30,
        f"  league {league:.1%}",
        ha="left",
        va="center",
        fontsize=9.5,
        color=PALETTE["anchor"],
    )
    for pos, row in zip(positions, rows.iter_rows(named=True), strict=True):
        ax.annotate(
            "",
            xy=(row["posterior_mean"], pos),
            xytext=(row["observed_rate"], pos),
            arrowprops={
                "arrowstyle": "-|>,head_width=0.22,head_length=0.5",
                "color": PALETTE["text_muted"],
                "linewidth": 1.5,
                "shrinkB": 1.0,
            },
            zorder=3,
        )
        ax.scatter(
            [row["observed_rate"]],
            [pos],
            s=90,
            facecolor=PALETTE["bad"],
            edgecolor=PALETTE["bg"],
            linewidth=1.6,
            zorder=4,
        )
        ax.scatter(
            [row["posterior_mean"]],
            [pos],
            s=90,
            facecolor=PALETTE["anchor"],
            edgecolor=PALETTE["bg"],
            linewidth=1.6,
            zorder=5,
        )
    ax.set_yticks(positions)
    ax.set_yticklabels(
        [
            f"{row['team_season'][5:]} {row['team_season'][:4]}  ({row['k']} of {row['n']})"
            for row in rows.iter_rows(named=True)
        ],
        fontsize=10,
        color=PALETTE["text"],
    )
    ax.set_xlim(0.0, 0.95)
    ax.set_ylim(-0.7, len(positions) - 0.15)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("own-recovery rate", color=PALETTE["text_muted"], fontsize=10)
    ax.grid(axis="x", color=PALETTE["grid"], linestyle="--", alpha=0.6, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(
        0.98,
        0.04,
        "red = what happened   ·   grey = what the model believes",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=PALETTE["text_muted"],
    )
    finalize(fig, FIGURE_DIR / "09_shrinkage.png")
    return {"league": league, "buffalo": float(buffalo["posterior_mean"][0])}


# --------------------------------------------------------------------------
# figure 10 — the DAG
# --------------------------------------------------------------------------


def dag() -> dict:
    """Boxes and arrows: where the simulator cuts the graph.

    Medium renders no Mermaid, so the markdown carries the Mermaid source and
    this PNG, and they have to say the same thing.
    """
    fig = plt.figure(figsize=(9.0, 4.6))
    fig.patch.set_facecolor(PALETTE["bg"])
    draw_title_block(
        title_axes(fig, height_frac=0.17),
        "Where the simulator cuts",
        [
            "The play is kept. The bounce is replaced by its average. Everything downstream follows.",
        ],
        title_size=17,
    )
    ax = fig.add_axes([0.03, 0.05, 0.94, 0.68])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    boxes = [
        (4, 20, "the play\nboth teams ran", PALETTE["anchor"]),
        (29, 20, "the luck event\nbounce · drift · hands", PALETTE["bad"]),
        (56, 20, "realized EPA\nfor that play", PALETTE["anchor"]),
        (81, 20, "the scoreboard\nmargin", PALETTE["anchor"]),
    ]
    # 15 wide clipped "bounce · drift · hands" out of its own box.
    width, height = 18, 11
    for x, y, label, colour in boxes:
        ax.add_patch(
            plt.Rectangle(
                (x, y - height / 2),
                width,
                height,
                facecolor=PALETTE["bg"],
                edgecolor=colour,
                linewidth=2.0,
                zorder=3,
                joinstyle="round",
            )
        )
        ax.text(
            x + width / 2,
            y,
            label,
            ha="center",
            va="center",
            fontsize=9.5,
            color=PALETTE["text"],
            zorder=4,
        )
    for start, end in ((4, 29), (29, 56), (56, 81)):
        ax.annotate(
            "",
            xy=(end - 1.0, 20),
            xytext=(start + width + 1.0, 20),
            arrowprops={
                "arrowstyle": "-|>,head_width=0.25,head_length=0.6",
                "color": PALETTE["text_muted"],
                "linewidth": 1.6,
            },
            zorder=2,
        )
    # The cut: the edge into the luck event is severed and an expectation is
    # substituted for it. Drawn as a scissors line, labelled in words.
    ax.plot(
        [27.5, 27.5], [8, 32], color=PALETTE["bad"], linewidth=2.0, linestyle=(0, (4, 3)), zorder=5
    )
    ax.text(
        25.5,
        34.0,
        "the one cut",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=PALETTE["bad"],
    )
    ax.text(
        38.0,
        9.5,
        "replaced by the average\nof both branches",
        ha="center",
        va="top",
        fontsize=9.5,
        color=PALETTE["bad"],
    )
    ax.text(
        2,
        1.5,
        "Nothing to the left of the cut is touched: the play calls, the drive, the clock all stand as played.",
        ha="left",
        va="center",
        fontsize=9.5,
        color=PALETTE["text_muted"],
    )
    finalize(fig, FIGURE_DIR / "10_dag.png")
    return {}


# --------------------------------------------------------------------------
# figure 11 — the DTW% distribution and its band
# --------------------------------------------------------------------------


def flip_distribution() -> dict:
    """Every Strict game's deserved-win probability, with the coin-flip band.

    Recomputed from ``dtw_games_v13.parquet`` — the same artifact
    ``research/48_magnitude_audit.py`` reads — rather than copied from document
    33, and checked against document 33's counts before it is drawn.
    """
    games = pl.read_parquet(OUTPUTS / "dtw_games_v13.parquet")
    dtw = games["dtw_home"].to_numpy()
    actual = games["actual_margin"].to_numpy()
    deserved = games["deserved_margin"].to_numpy()

    check("n_games_strict", games.height)
    sign_flips = int((((deserved > 0) != (actual > 0)) & (actual != 0)).sum())
    check("sign_flips", sign_flips)
    too_close = int(((dtw >= BAND_LOW) & (dtw <= BAND_HIGH)).sum())
    check("too_close", too_close)
    degenerate = int(((dtw <= 0.001) | (dtw >= 0.999)).sum())
    check("degenerate", degenerate)

    fig, ax = new_figure(
        8.6,
        5.8,
        title="There is no typical game",
        subtitle=[
            f"Deserved-win probability for the home team · {games.height:,} games, "
            "2016–2025, Strict edition",
            f"{degenerate:,} games ({degenerate / games.height:.0%}) are decided beyond luck's reach — the two end bars.",
        ],
    )
    edges = np.linspace(0.0, 1.0, 41)
    counts, _ = np.histogram(dtw, bins=edges)
    centres = (edges[:-1] + edges[1:]) / 2
    inside = (centres >= BAND_LOW) & (centres <= BAND_HIGH)
    ax.axvspan(BAND_LOW, BAND_HIGH, color=PALETTE["row_alt"], zorder=1)
    ax.bar(
        centres,
        counts,
        width=(edges[1] - edges[0]) * 0.86,
        color=np.where(inside, PALETTE["bad"], PALETTE["anchor"]),
        edgecolor=PALETTE["bg"],
        linewidth=0.8,
        zorder=3,
    )
    ax.set_xlim(-0.01, 1.01)
    ax.set_xlabel("deserved-win probability (DTW%)", color=PALETTE["text_muted"], fontsize=10)
    ax.set_ylabel("games", color=PALETTE["text_muted"], fontsize=10)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", color=PALETTE["grid"], linestyle="--", alpha=0.6, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.annotate(
        f"too close to call\n{too_close} games ({too_close / games.height:.1%})",
        xy=(0.5, counts[inside].max()),
        xytext=(0.5, max(counts) * 0.62),
        ha="center",
        va="bottom",
        fontsize=10,
        color=PALETTE["bad"],
        arrowprops={
            "arrowstyle": "-|>,head_width=0.2,head_length=0.5",
            "color": PALETTE["bad"],
            "linewidth": 1.3,
        },
    )
    finalize(fig, FIGURE_DIR / "11_flip_distribution.png")
    return {"too_close": too_close, "sign_flips": sign_flips, "degenerate": degenerate}


# --------------------------------------------------------------------------
# figure 12 — Strict against Full
# --------------------------------------------------------------------------


def full_vs_strict() -> dict:
    """The games the Full edition moves, Strict DTW% against Full DTW%.

    A slope plot: one line per moved game, from where Strict put it to where
    Full does, with the three buckets drawn as bands so a crossing is visible
    as a crossing rather than as a number.
    """
    strict = pl.read_parquet(OUTPUTS / "dtw_games_v13.parquet").select(
        "game_id", pl.col("dtw_home").alias("strict_dtw"), "actual_margin"
    )
    full = pl.read_parquet(OUTPUTS / "full_summary.parquet").select(
        "game_id", pl.col("dtw_home").alias("full_dtw")
    )
    check("n_games_full", full.height)
    paired = full.join(strict, on="game_id", how="inner")

    def bucket(dtw: np.ndarray, margin: np.ndarray) -> np.ndarray:
        """Clear flip / too close to call / scoreboard holds, per document 33 §2a."""
        home_won = margin > 0
        label = np.where((dtw > 0.5) != home_won, "flip", "holds")
        return np.where((dtw >= BAND_LOW) & (dtw <= BAND_HIGH), "close", label)

    strict_dtw = paired["strict_dtw"].to_numpy()
    full_dtw = paired["full_dtw"].to_numpy()
    margin = paired["actual_margin"].to_numpy()
    moved = bucket(strict_dtw, margin) != bucket(full_dtw, margin)
    check("full_bucket_moves", int(moved.sum()))

    fig, ax = new_figure(
        7.2,
        7.0,
        title="What the second edition moves",
        subtitle=[
            f"Deserved-win probability, Strict against Full · {paired.height:,} games, 2022–2025",
            f"Red are the {int(moved.sum())} that change verdict bucket. Bands are 'too close to call'.",
        ],
    )
    # The bands are the buckets themselves, on both axes: a game that leaves the
    # band on one axis and not the other is a game the second edition re-labels.
    ax.axhspan(BAND_LOW, BAND_HIGH, color=PALETTE["row_alt"], zorder=1)
    ax.axvspan(BAND_LOW, BAND_HIGH, color=PALETTE["row_alt"], zorder=1)
    ax.plot([0, 1], [0, 1], color=PALETTE["spine"], linewidth=1.0, zorder=2)
    ax.scatter(
        strict_dtw[~moved],
        full_dtw[~moved],
        s=9,
        color=PALETTE["anchor"],
        alpha=0.28,
        edgecolor="none",
        zorder=3,
    )
    ax.scatter(
        strict_dtw[moved],
        full_dtw[moved],
        s=26,
        facecolor=PALETTE["bad"],
        edgecolor=PALETTE["bg"],
        linewidth=0.6,
        zorder=4,
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Strict edition DTW%", color=PALETTE["text_muted"], fontsize=10)
    ax.set_ylabel("Full edition DTW%", color=PALETTE["text_muted"], fontsize=10)
    ax.grid(color=PALETTE["grid"], linestyle="--", alpha=0.6, linewidth=0.7)
    ax.set_axisbelow(True)
    # Under the axis: inside the panel this line lay across the top-left cloud,
    # which is where the biggest moves are.
    ax.text(
        0.0,
        -0.105,
        "above the diagonal: the Full edition likes the home team more than Strict does",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=PALETTE["text_muted"],
    )
    # A dot plot of 1,139 points compresses to a fraction of what 190 translucent
    # slope strokes cost, so this one keeps the house 200 dpi.
    finalize(fig, FIGURE_DIR / "12_full_vs_strict_moves.png")
    return {"moved": int(moved.sum()), "n": paired.height}


# --------------------------------------------------------------------------
# the formula plates
# --------------------------------------------------------------------------

# Medium renders no LaTeX. Each entry is one fenced block in the article and one
# PNG beside it; the two must always be edited together.
FORMULAE = {
    "formula_01": (
        r"$\mathrm{luck}(e) = \left(y(e) - p(e)\right) \cdot \mathrm{swing}(e)$"
        "\n\n"
        r"$\mathrm{EPA}^{*}_{\mathrm{diff}} = \mathrm{EPA}_{\mathrm{diff}}"
        r" - \sum_{e} \mathrm{luck}(e)$"
    ),
    "formula_02": (
        r"$p(e) = w\,\hat{r}_{\mathrm{entity}} + (1-w)\,\bar{r}_{\mathrm{league}}$"
        "\n\n"
        r"$w = \frac{n}{n+\kappa}$"
    ),
    "formula_03": (
        r"$M_{\mathrm{deserved}} = M_{\mathrm{actual}}"
        r" - 0.8389 \sum_{e} \mathrm{luck}(e)$"
        "\n\n"
        r"$\mathrm{DTW}\% = P\left(M^{*}_{\mathrm{deserved}} > 0\right)$"
    ),
    "formula_04": (
        r"$\mathrm{logit}\, p_{\mathrm{make}} = \alpha + \beta d"
        r" + \gamma \frac{d^{2}}{100} + \delta \frac{d^{3}}{1000}"
        r" + \rho_{\mathrm{roof}} + \beta_{w} W + \beta_{t} T + \sigma_{k} z_{k}$"
    ),
    "formula_05": (
        r"$r_{t} \sim \mathrm{Beta}\left(\mu\kappa,\; (1-\mu)\kappa\right)$"
        "\n\n"
        r"$k_{t} \sim \mathrm{Binomial}\left(n_{t},\; r_{t}\right)$"
    ),
}


def formula_plates() -> list[str]:
    """Render each fenced formula to a transparent 300-dpi PNG."""
    written = []
    for name, latex in FORMULAE.items():
        lines = latex.split("\n\n")
        fig = plt.figure(figsize=(8.4, 0.85 * len(lines) + 0.25))
        fig.patch.set_alpha(0.0)
        for index, line in enumerate(lines):
            fig.text(
                0.5,
                1.0 - (index + 0.62) / len(lines),
                line,
                ha="center",
                va="center",
                fontsize=17,
                color=PALETTE["text"],
            )
        target = FIGURE_DIR / f"{name}.png"
        # Not `finalize`: a formula is typography, not a figure, and stamping a
        # data credit on an equation would credit nflverse with the algebra.
        fig.savefig(target, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)
        written.append(target.name)
    return written


# --------------------------------------------------------------------------


def copy_game_figures() -> list[str]:
    """Render each example game and copy the one figure the article uses."""
    written = []
    scratch = OUTPUTS / "80_writeup_renders"
    scratch.mkdir(parents=True, exist_ok=True)
    for stem, game_id, edition, suffix in GAME_FIGURES:
        rendered = render_game(game_id, scratch, edition=edition)
        source = next(path for path in rendered if path.stem.endswith(f"_{suffix}"))
        target = FIGURE_DIR / f"{stem}.png"
        shutil.copyfile(source, target)
        written.append(target.name)
    return written


def audit() -> None:
    """The two rules a committed image has to obey, checked rather than trusted."""
    files = sorted(FIGURE_DIR.glob("*.png"))
    heavy = [path.name for path in files if path.stat().st_size > SIZE_LIMIT]
    if heavy:
        raise AssertionError(f"over {SIZE_LIMIT} bytes: {heavy}")
    if len(files) > MAX_FILES:
        raise AssertionError(f"{len(files)} images, and the article's budget is {MAX_FILES}")
    print(f"\n{len(files)} images in {FIGURE_DIR.relative_to(paths.REPO_ROOT)}, none over 500 KB")


def summary_only() -> dict:
    """`--summary`: document 64's numbers, and nothing drawn.

    Kept separate from :func:`main` because the document is checked far more
    often than the figures are redrawn, and because a run that writes no PNG
    cannot leave a half-updated figure directory behind.
    """
    summary = one_simulator_summary()
    print_summary(summary)
    event = worked_fumble_example()
    print_worked_fumble(event)
    return {"summary": summary, "worked_fumble": event}


def main() -> None:
    apply_base_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    summary_only()

    print("game figures ...")
    for name in copy_game_figures():
        print(f"  {name}")

    print("explanatory figures ...")
    print("  07 fumble recovery  ", fumble_year_over_year())
    print("  08 persistence bars ", persistence_bars())
    print("  09 shrinkage        ", shrinkage())
    print("  10 dag              ", dag())
    print("  11 flip distribution", flip_distribution())
    print("  12 full vs strict   ", full_vs_strict())

    print("formula plates ...")
    for name in formula_plates():
        print(f"  {name}")

    audit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print document 64's numbers and draw nothing",
    )
    if parser.parse_args().summary:
        summary_only()
    else:
        main()
