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
(:data:`SIZE_LIMIT`), caps the whole set at twenty files, and refuses to finish
unless the set on disk is exactly the set the article links to.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import ticker as mticker

from nfl_simulator import paths
from nfl_simulator.plots import BAND_HIGH, BAND_LOW
from nfl_simulator.render import render_game
from nfl_simulator.style import PALETTE, apply_base_style, draw_title_block, finalize, title_axes

FIGURE_DIR = paths.REPO_ROOT / "docs" / "writeup" / "figures"
OUTPUTS = paths.RESEARCH_OUTPUT_DIR

# A committed PNG is a file somebody clones. 500 KB is the article's budget per
# image and the script refuses to leave a heavier one on disk.
SIZE_LIMIT = 500_000
# Round 5's budget, two above round 3's: figures 22 and 23 join the set and
# nothing is withdrawn, so twenty-two are drawn against a cap of twenty-four.
MAX_FILES = 24

# --------------------------------------------------------------------------
# round 3's five new figures, and the constants that pick their subjects
# --------------------------------------------------------------------------

# Figure 18's kicker is **chosen by a rule, not by name**: the largest posterior
# mean effect among 2025 kicker-seasons with at least `KICKER_FLOOR` attempts.
# The rule is here and the name it resolves to is in document 64 §11, so a refit
# that changed the answer would change the document rather than silently draw a
# different kicker under the old caption.
KICKER_SEASON = 2025
KICKER_FLOOR = 20
# 45 yards because document 05b §9 reports the kicker spread there, and because
# it is the distance at which the league is neither automatic nor hopeless —
# 79.9%, so a kicker's own ability has room to show.
KICKER_DISTANCE = 45.0
# Spelled as he spells it. Round 3 wrote "Pineiro" because the effects parquet
# keys on a player id and the name was typed from memory; round 5's audit
# (document 65) is the reason it is now typed from the man's name.
KICKER_NAME = "E. Pi\u00f1eiro"
KICKER_TEAM = "SF"
# Untracked and gitignored — a headshot is not this repository's to redistribute.
# Absent, figure 18 falls back to the team mark, which is what round 3 drew.
KICKER_HEADSHOT = "headshots/pineiro.png"

# Figure 19's defence-season. `worthy_throw_frame` keys these `"season|team"`.
DENVER_KEY = "2024|DEN"

# Figure 22 overlays two defence-seasons so the reader sees that a replay draws
# from the *unit's own* curve rather than from one league curve. Denver 2024 is
# figure 19's strong unit; the Jets' 2025 defence is the other end — nine
# interceptable throws and none of them caught — and the pair shows shrinkage
# working in both directions from one league middle.
SAMPLING_KEYS = ("2024|DEN", "2025|NYJ")

# Figure 23 is figure 18's mirror on the offence: one receiving corps' drop rate
# before and after shrinking. **Chosen by a rule, like the kicker.** Among
# corps-seasons with at least `CORPS_FLOOR` catchable targets, the one whose raw
# rate sits furthest from its own posterior — the largest visible shrink, which
# is what the figure is for. It resolves to Jacksonville 2025.
CORPS_FLOOR = 200
CORPS_KEY = "2025|JAX"

# Figure 20's two numbers, and they are **two different statistics** — see
# `persistent_share`. The variance share is document 48 through document 52 §2;
# the shrinkage weight is document 05 §3's `w` for fumble recovery.
DROPPED_PICK_SHARE = 0.014
RECEIVER_DROP_SHARE = 0.00088  # doc 57 §1b — same statistic as the dropped pick's 1.4%
FUMBLE_SHRINKAGE_W = 0.011

# --------------------------------------------------------------------------
# the six per-game figures, copied from the product renderer
# --------------------------------------------------------------------------

# (article filename stem, game id, edition, which of `render.SUFFIXES`).
# "ledger" in an article filename means the **waterfall** — the row-by-row
# figure section 9 walks through — not the square card, which is the share
# image and has no rows to read.
GAME_FIGURES = [
    ("04_lac_hou_2024_wk19_ledger_full", "2024_19_LAC_HOU", "full", "waterfall"),
    ("06_den_was_2025_wk13_ledger_full", "2025_13_DEN_WAS", "full", "waterfall"),
]

# The distribution figure is not in that list. It is the same product figure
# drawn by the same function, but the article renames one label — see
# :func:`article_dtw_figure` — and a caller that renames a label cannot go
# through `render.render_game`, which writes the PNG itself.
ARTICLE_DTW = ("05_den_was_2025_wk13_dtw_full", "2025_13_DEN_WAS", "full")

# --------------------------------------------------------------------------
# what the redraws have to reproduce
# --------------------------------------------------------------------------

# Every explanatory figure below is drawn from a committed artifact and has to
# land on a number a numbered document already published. These are those
# numbers, with their document, and `check` raises rather than warns.
DOC_CHECKS = {
    "den_2025_worthy": (21, 0.001),
    "den_2025_caught": (8, 0.001),
    # document 02 §1, the pooled recovery-rate split half
    "fumble_rate_split_half_r": (0.051, 0.001),
    # document 02, test 1 — the six split-half correlations
    "split_half_fumble": (0.055, 0.001),
    "split_half_int": (0.164, 0.001),
    # document 04 via document 05 §3 — Buffalo's 10-of-12 shrinking to 48.0%
    "buffalo_2024_posterior": (0.480, 0.001),
    # document 01 §1 and §4 — the EPA-to-points conversion figure 13 redraws
    "epa_points_slope": (0.8389, 0.0001),
    # Document 01's own artifact (`01_descriptive_eda.json`, `variance
    # /corr_epa_margin`), not its prose. The prose reports r² = 0.991, which is
    # r = 0.99576 squared — 0.99154 — rounded the wrong way at the third
    # decimal. The correlation is checked here because it is the number the
    # study actually stored; the figure prints its square, 0.992.
    "epa_points_r": (0.9958, 0.0001),
    "n_games_epa": (2761, 0),
    # document 62 §4
    "n_games_full": (1139, 0),
    # Document 50 §2 — the defence-season spread the dropped-pick model believes
    # in, `sigma_d` on the logit scale carried to the probability scale at the
    # league rate. Figure 15's subtitle prints it.
    "defence_spread_pp": (6.4, 0.1),
    # The three materiality refusals figure 14 draws, each read from the study's
    # own artifact rather than typed from its document: measured median |ΔDTW|
    # against the floor pre-registered before it was computed.
    "muff_effect_pp": (0.404, 0.001),
    "muff_floor_pp": (0.7222, 0.0001),
    "blocked_effect_pp": (0.222, 0.001),
    "blocked_floor_pp": (1.4392, 0.0001),
    "toss_effect_pp": (3.93, 0.005),
    "toss_floor_pp": (4.06, 0.005),
    # Document 64 §7a — the six measured fumble classes, at v1.3 settings. These
    # were handed to this round as the rates the article may cite, so they are
    # `check`ed rather than published: the table below is the arithmetic, and
    # this is the record it has to land on.
    "fumble_rate_pass_live": (0.5096, 0.00005),
    "fumble_rate_run_live": (0.4611, 0.00005),
    "fumble_rate_run_aborted": (0.7690, 0.00005),
    "fumble_rate_punt_live": (0.6843, 0.00005),
    "fumble_rate_kickoff_live": (0.5124, 0.00005),
    "fumble_rate_pass_aborted": (1.0, 0.00005),
    # Document 64 §8 — the walk-through game with and without the
    # hands-on-the-ball rows — **at v1.4**, where document 68 §7c republishes
    # all four. The game is Denver *at* Washington, played at Northwest
    # Stadium's 180 feet: below the model's 569-foot centre, so every kick in it
    # is now priced slightly harder and each of the four numbers moves a little.
    # v1.3's values were 0.1449 / 0.4058 / -3.32 / -1.35.
    "den_was_dtw_strict": (0.1497, 0.0001),
    "den_was_dtw_full": (0.4094, 0.0001),
    "den_was_deserved_strict": (-3.27, 0.005),
    "den_was_deserved_full": (-1.31, 0.005),
    # --- round 3 --------------------------------------------------------
    # Figure 18. **Not document 05b §9's 5.35 pp**, and the difference is the
    # point: §9 reports the Phase 2 posterior, whose `sigma_kicker` is 0.360.
    # The shipped model is the refit, which is what every ledger row is priced
    # with, so the figure draws the refit. Document 64 §11 carries the
    # reconciliation, and round 5 moves both numbers to **v1.4**: adding the
    # elevation term re-centres the league curve (79.88 -> 79.92 at 45 yards)
    # and narrows `sigma_kicker` from 0.3855 to 0.3837, which carries the
    # one-standard-deviation kicker from 5.48 pp to 5.45.
    "kicker_one_sigma_pp": (5.45, 0.05),
    "fg_league_p45": (79.92, 0.05),
    # Figure 19. The posterior mean is document 64's 55.2%; the league figure
    # and the two counts are computed here and published in document 64 §11.
    "denver_worthy_throws": (17, 0),
    "denver_worthy_caught": (13, 0),
    "denver_posterior_pct": (55.2, 0.05),
    "denver_league_pct": (49.8, 0.05),
    # Figure 20 — both are transcriptions, `check`ed so a typo in the constant
    # cannot reach the figure. Document 48 via 52 §2, and document 05 §3.
    "dropped_pick_share_pct": (1.4, 0.001),
    "fumble_shrinkage_pct": (1.1, 0.001),
    "receiver_drop_share_pct": (0.088, 0.001),
    # --- round 5 --------------------------------------------------------
    # Figure 22's second unit, the other end of figure 19's axis. Nine
    # interceptable throws, none caught, and the model still says 42.6% — which
    # is the whole of what the figure is drawn to show.
    "nyj_worthy_throws": (9, 0),
    "nyj_worthy_caught": (0, 0),
    "nyj_posterior_pct": (42.6, 0.05),
    # Figure 23, the offence mirror. Jacksonville's 2025 corps dropped 40 of the
    # 420 balls FTN charted catchable — 9.52%, nearly twice the league's rate on
    # those same targets — and the model keeps about a third of that record.
    "corps_targets": (420, 0),
    "corps_drops": (40, 0),
    "corps_raw_pct": (9.52, 0.01),
    "corps_posterior_pct": (6.34, 0.01),
    "corps_league_pct": (5.14, 0.01),
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
# **v1.4's artifact**, not v1.3's. Simulator v1.4 put stadium elevation into the
# shipped field-goal model (document 68); `render.py` reads the v1.4 files, and a
# figure drawn from `full_summary.parquet` beside a waterfall drawn from
# `full_summary_v14.parquet` would be two adjudications wearing one article.
# Document 68 §6 is the list of what moved between them.
ONE_SIM_ARTIFACT = "full_summary_v14.parquet"

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


# --------------------------------------------------------------------------
# document 64 §7a — the fumble classes the simulator prices with
# --------------------------------------------------------------------------

# The order document 64 §7a prints, which is the order the classes come out of
# `fit_fumble_baseline` — commonest first. The two classes below
# `min_class_size` are printed last and marked, because they do not carry a
# rate of their own.
FUMBLE_CLASS_ORDER = [
    ("pass/live", "fumble_rate_pass_live", "a fumble on a pass play, ball live"),
    ("run/live", "fumble_rate_run_live", "a fumble on a run, ball live"),
    ("run/aborted", "fumble_rate_run_aborted", "a botched snap on a run play"),
    ("punt/live", "fumble_rate_punt_live", "a muffed or fumbled punt, ball live"),
    ("kickoff/live", "fumble_rate_kickoff_live", "a fumble on a kickoff, ball live"),
    ("pass/aborted", "fumble_rate_pass_aborted", "a botched snap on a pass play"),
]


def fumble_class_rates() -> dict:
    """Every fumble class the shipped simulator prices with, and its rate.

    Read off the **same fitted baseline the product replays with** —
    `render._simulation_context()["fumble_baseline"]` — rather than recomputed
    from a research artifact, so a class rate the article quotes is the rate a
    ledger row was actually priced at. The six measured classes are `check`ed
    against the record; the two below `min_class_size` carry the pooled rate
    instead of one of their own and are reported as such.

    ``p_own`` is a **retention** rate, not the narrower recovery rate document
    05 §3 published: since v1.2 a ball that crosses the sideline counts as kept
    (`components._fumble_frame`). The two are different populations and the
    article must quote this one, because it is the one the ledger prices with.
    """
    from nfl_simulator.render import _simulation_context

    table = _simulation_context()["fumble_baseline"].table
    rows = {row["fumble_class"]: row for row in table.iter_rows(named=True)}
    measured = []
    for name, check_name, gloss in FUMBLE_CLASS_ORDER:
        row = rows[name]
        check(check_name, round(float(row["p_own"]), 4))
        measured.append({**row, "gloss": gloss, "pooled": False})
    pooled = [
        {**row, "gloss": "below the 30-fumble floor — priced at the pooled rate", "pooled": True}
        for name, row in rows.items()
        if name not in {entry[0] for entry in FUMBLE_CLASS_ORDER}
    ]
    return {
        "measured": measured,
        "pooled": sorted(pooled, key=lambda row: -row["n"]),
        "n_fumbles": int(table["n"].sum()),
    }


def print_fumble_class_rates(rates: dict) -> None:
    """Document 64 §7a's table, in the order the document prints it."""
    print(f"\n{'=' * 88}\nDOCUMENT 64 §7a — the fumble classes v1.3 prices with\n{'=' * 88}")
    print(f"  {'class':<16}{'n':>7}{'p (kept)':>11}{'swing EPA':>12}   what it is")
    for row in rates["measured"] + rates["pooled"]:
        print(
            f"  {row['fumble_class']:<16}{row['n']:>7}{row['p_own']:>10.4f} "
            f"{row['swing_value']:>11.4f}   {row['gloss']}"
        )
    print(f"  {'total':<16}{rates['n_fumbles']:>7}")


# --------------------------------------------------------------------------
# document 64 §8 — the walk-through game, row by row
# --------------------------------------------------------------------------

# The two adjudications document 64 §8 sets side by side. Strict **is** v1.3:
# fumbles, field goals and extra points, and nothing else. Full is that plus
# amendment A-3's hands-on-the-ball class and document 61's possession cap.
V13_COMPONENTS = ("fumble", "field_goal", "extra_point")


def walkthrough_ledger() -> dict:
    """`2025_13_DEN_WAS` under both adjudications, with every v1.3 row.

    Both replays are checked against their own published summary by
    :func:`render.replay` before anything is read, so the rows below belong to
    the same adjudication the article's figures are drawn from. The four
    headline numbers are `check`ed against the values this round was handed.
    """
    from nfl_simulator.render import load_sources, replay

    sources = load_sources()
    schedule = sources.schedule_row(WALKTHROUGH_GAME)
    editions = {}
    for edition in ("strict", "full"):
        row = sources.game_row(WALKTHROUGH_GAME, edition=edition)
        result, _gaps = replay(WALKTHROUGH_GAME, row, schedule, edition=edition)
        editions[edition] = {
            "ledger": result.ledger.to_frame(),
            "margin_draws": np.asarray(result.margin_draws, dtype=float),
            "dtw_home": float(result.dtw_home),
            "dtw_interval": tuple(float(bound) for bound in result.dtw_interval),
            "deserved_margin": float(result.deserved_margin),
            "actual_margin": float(result.actual_margin),
            "total_luck_epa": float(result.total_luck_epa),
        }

    for edition in ("strict", "full"):
        check(f"den_was_dtw_{edition}", round(editions[edition]["dtw_home"], 4))
        check(f"den_was_deserved_{edition}", round(editions[edition]["deserved_margin"], 2))

    rows = editions["strict"]["ledger"].sort("component", "play_id")
    per_component = (
        editions["full"]["ledger"]
        .group_by("component")
        .agg(pl.len().alias("rows"), pl.col("luck_epa").sum().alias("luck_epa"))
        .sort("component")
    )
    return {
        "game_id": WALKTHROUGH_GAME,
        "home_team": schedule["home_team"],
        "away_team": schedule["away_team"],
        "home_score": schedule["home_score"],
        "away_score": schedule["away_score"],
        "slope": sources.slope,
        "v13_rows": rows.to_dicts(),
        "full_components": per_component.to_dicts(),
        "editions": editions,
    }


def print_walkthrough_ledger(walkthrough: dict) -> None:
    """Document 64 §8's two tables and its four headline numbers.

    ``luck_epa`` is already **home-signed** (`simulator` line 377): a positive
    row is EPA the game handed the home team, whichever team the event is
    charged to. Points are that times the slope, and the column sums to the gap
    between the actual margin and the deserved one by construction.
    """
    slope = walkthrough["slope"]
    home, away = walkthrough["home_team"], walkthrough["away_team"]
    print(f"\n{'=' * 88}\nDOCUMENT 64 §8 — {walkthrough['game_id']}, every v1.3 row\n{'=' * 88}")
    print(
        f"  {away} {walkthrough['away_score']} - {home} {walkthrough['home_score']}, "
        f"overtime; every row is signed for {home}, the home team"
    )
    header = f"  {'play':>7}  {'component':<13}{'class':<22}{'chg':<5}{'y':>3}"
    print(f"{header}{'p':>9}{'swing':>10}{'luck EPA':>11}{'points':>9}")
    for row in walkthrough["v13_rows"]:
        print(
            f"  {row['play_id']:>7.0f}  {row['component']:<13}{row['event_class']:<22}"
            f"{row['charged_team']:<5}{row['actual']:>3.0f}{row['expected']:>9.4f}"
            f"{row['swing']:>10.4f}{row['luck_epa']:>11.4f}"
            f"{row['luck_epa'] * slope:>9.3f}"
        )
    total = walkthrough["editions"]["strict"]["total_luck_epa"]
    print(
        f"  {'':>7}  {'total':<13}{'':<22}{'':<5}{'':>3}{'':>9}{'':>10}{total:>11.4f}{total * slope:>9.3f}"
    )

    print("\n  the Full edition's rows, by component (v1.3's three, plus A-3's two and the cap)")
    print(f"  {'component':<18}{'rows':>6}{'luck EPA':>12}{'points':>9}")
    for row in walkthrough["full_components"]:
        print(
            f"  {row['component']:<18}{row['rows']:>6}{row['luck_epa']:>12.4f}"
            f"{row['luck_epa'] * slope:>9.3f}"
        )
    full_total = walkthrough["editions"]["full"]["total_luck_epa"]
    print(f"  {'total':<18}{'':>6}{full_total:>12.4f}{full_total * slope:>9.3f}")

    print(f"\n  {'':<26}{'without A-3':>14}{'with A-3':>12}")
    strict, full = walkthrough["editions"]["strict"], walkthrough["editions"]["full"]
    print(f"  {f'DTW% for {home}':<26}{strict['dtw_home']:>14.4f}{full['dtw_home']:>12.4f}")
    print(f"  {f'DTW% for {away}':<26}{1 - strict['dtw_home']:>14.4f}{1 - full['dtw_home']:>12.4f}")
    print(
        f"  {'deserved margin':<26}{strict['deserved_margin']:>14.4f}"
        f"{full['deserved_margin']:>12.4f}"
    )
    print(f"  {'actual margin':<26}{strict['actual_margin']:>14.1f}{full['actual_margin']:>12.1f}")


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
        title="Fumble recoveries don't repeat year to year",
        subtitle=[
            f"Own-recovery rate, season t vs season t+1 · {pairs.height} team pairs, 2016–2025",
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
# figure 11 — the DTW% distribution and its band
# --------------------------------------------------------------------------


def flip_distribution(summary: dict) -> dict:
    """Every game's deserved-win probability, with the coin-flip band.

    Round 2 redraws this on the **one simulator** — the Full edition's 1,139
    games — so the picture and the counts under it come from the same corpus
    document 64 publishes. ``summary`` is that document's own dictionary rather
    than a second read of the parquet: a figure and a document that each count
    their own games are two corpora wearing one caption.

    Draft 1 printed the band's share inside the band. Round 2 does not: the
    annotation sat over the bars it was describing, and the subtitle already
    carries the count.
    """
    games = pl.read_parquet(OUTPUTS / ONE_SIM_ARTIFACT)
    check("n_games_full", games.height)
    dtw = games["dtw_home"].to_numpy()

    too_close = summary["too_close"]
    degenerate = summary["degenerate"]
    if too_close != int(((dtw >= BAND_LOW) & (dtw <= BAND_HIGH)).sum()):
        raise AssertionError("the band count in document 64 is not the band count on this axis")

    fig, ax = new_figure(
        8.6,
        5.8,
        title="There is no typical game",
        subtitle=[
            f"Deserved-win probability for the home team · {games.height:,} games, 2022–2025",
            f"{degenerate:,} games ({degenerate / games.height:.0%}) are decided beyond "
            "luck's reach — DTW% pinned at 0 or 100.",
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
    # The band is named above the plot rather than inside it: at 40 bins the
    # tallest bar in the band is most of the axis, and draft 1's callout printed
    # over the bars it was describing.
    ax.text(
        (BAND_LOW + BAND_HIGH) / 2,
        1.02,
        f"too close to call · {too_close} games",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=PALETTE["bad"],
    )
    finalize(fig, FIGURE_DIR / "11_flip_distribution.png")
    return {"too_close": too_close, "degenerate": degenerate, "games": games.height}


# --------------------------------------------------------------------------
# figure 13 — EPA differential against final margin
# --------------------------------------------------------------------------


def epa_to_points() -> dict:
    """Every game's EPA differential against its final margin, and the slope.

    This is document 01's regression, redrawn: the single number that turns a
    ledger of EPA into points of margin. Both the slope and the r² are `check`ed
    against document 01 before the figure is written, because the whole article
    quotes 0.8389 and a redraw that produced 0.83 would make every points figure
    in it wrong by a percent.
    """
    from nfl_simulator.components import build_game_table
    from nfl_simulator.render import _simulation_context
    from nfl_simulator.simulator import points_per_epa

    games = build_game_table(_simulation_context()["pbp"]).drop_nulls("margin")
    check("n_games_epa", games.height)
    epa = games["epa_diff"].to_numpy()
    margin = games["margin"].to_numpy()
    slope = check("epa_points_slope", round(points_per_epa(games), 4))
    correlation = float(np.corrcoef(epa, margin)[0, 1])
    check("epa_points_r", round(correlation, 4))
    r_squared = round(correlation**2, 3)
    intercept = float(margin.mean() - slope * epa.mean())

    fig, ax = new_figure(
        7.8,
        6.0,
        title="EPA and points are nearly the same quantity",
        subtitle=[
            f"Final margin against EPA differential · {games.height:,} games, 2016–2025",
            "One dot per game. The line is the conversion the simulator uses.",
        ],
    )
    ax.scatter(
        epa,
        margin,
        s=9,
        facecolor=PALETTE["anchor"],
        edgecolor="none",
        alpha=0.28,
        zorder=2,
    )
    grid = np.linspace(epa.min(), epa.max(), 2)
    ax.plot(grid, intercept + slope * grid, color=PALETTE["bad"], linewidth=2.0, zorder=3)
    ax.set_xlabel("EPA differential (home − away)", color=PALETTE["text_muted"], fontsize=10)
    ax.set_ylabel("final margin (home − away)", color=PALETTE["text_muted"], fontsize=10)
    ax.grid(color=PALETTE["grid"], linestyle="--", alpha=0.6, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.text(
        0.03,
        0.96,
        f"{slope:.4f} points per EPA\nr² = {r_squared:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["text"],
        linespacing=1.4,
    )
    finalize(fig, FIGURE_DIR / "13_epa_to_points.png")
    return {"slope": slope, "r_squared": r_squared, "games": games.height}


# --------------------------------------------------------------------------
# figure 14 — the materiality refusals
# --------------------------------------------------------------------------

# Each row is (label, artifact, how to reach the measured median, the floor's
# key, the document that pre-registered it). The floors were committed before
# the effect beside them was computed, which is the point of the figure — so
# both numbers are read from the study's own JSON rather than typed from its
# prose, and both are `check`ed against the document afterwards.
REFUSALS = [
    (
        "kickoff muffs",
        "37_kickoff_muffs.json",
        ("impact", "by_w", "0.00", "median_abs_delta_dtw_pp"),
        ("impact", "pre_registered_floor_pp"),
        "doc 24 §8",
        "muff_effect_pp",
        "muff_floor_pp",
    ),
    (
        "blocked-kick aftermath",
        "39_blocked_kicks.json",
        ("impact", "by_w", "0.00", "median_abs_delta_dtw_pp"),
        ("impact", "pre_registered_floor_pp"),
        "doc 25 §8",
        "blocked_effect_pp",
        "blocked_floor_pp",
    ),
    (
        "the overtime coin toss",
        "26_overtime.json",
        ("impact", "median_abs_delta_dtw"),
        ("gate_o3_floor",),
        "doc 16 §8",
        "toss_effect_pp",
        "toss_floor_pp",
    ),
]

# The other refusals, which never reached a floor because they have no branch
# point to neutralize (document 05 §3). Text, not marks: a bar of length zero
# would say "measured and found small", and none of these was measured.
MECHANISM_REFUSALS = (
    "Refused earlier still, for want of a branch point to neutralize (doc 05 §3):\n"
    "penalties · interceptions · return yardage · fourth-down calls · two-point conversions"
)

# Where every row's annotation starts, in the axis's own units — past the
# longest bar, which is the overtime toss's 4.06 pp floor.
ANNOTATION_X = 4.4


def _dig(payload: dict, keys: tuple) -> float:
    """`payload["a"]["b"]` for a tuple of keys, so a path can be data."""
    for key in keys:
        payload = payload[key]
    return float(payload)


def refused_floors() -> dict:
    """What each refused component moved, against the floor it had to clear.

    The overtime toss's artifact stores its two numbers as shares and the other
    two store theirs in percentage points, which is why the toss row is scaled
    here — a difference in units between two studies is not a difference in
    what they measured.
    """
    rows = []
    for label, artifact, effect_path, floor_path, source, effect_check, floor_check in REFUSALS:
        payload = json.loads((OUTPUTS / artifact).read_text())
        effect = _dig(payload, effect_path)
        floor = _dig(payload, floor_path)
        scale = 100.0 if artifact.startswith("26_") else 1.0
        rows.append(
            {
                "label": label,
                "effect": check(effect_check, round(effect * scale, 3)),
                "floor": check(floor_check, round(floor * scale, 4)),
                "source": source,
            }
        )

    fig, ax = new_figure(
        8.8,
        5.2,
        title="Measured, and refused anyway",
        subtitle=[
            "Median move in deserve-to-win share, against the floor each had to clear",
            "Grey bar = the floor. Red dot = what the component actually moved.",
        ],
    )
    # The mechanism refusals go under the axis, so the axis is lifted to make
    # room for them rather than printed over.
    ax.set_position([0.24, 0.22, 0.72, 0.58])
    positions = np.arange(len(rows))[::-1]
    for pos, row in zip(positions, rows, strict=True):
        ax.plot(
            [0, row["floor"]],
            [pos, pos],
            color=PALETTE["grid"],
            linewidth=9,
            solid_capstyle="butt",
            zorder=2,
        )
        ax.plot(
            [row["floor"], row["floor"]],
            [pos - 0.30, pos + 0.30],
            color=PALETTE["anchor"],
            linewidth=2.0,
            zorder=4,
        )
        ax.plot([0, row["effect"]], [pos, pos], color=PALETTE["bad"], linewidth=2.0, zorder=3)
        ax.scatter(
            [row["effect"]],
            [pos],
            s=110,
            facecolor=PALETTE["bad"],
            edgecolor=PALETTE["bg"],
            linewidth=1.6,
            zorder=5,
        )
        # All three annotations start at one x, so they read as a column rather
        # than as three labels that happen to follow three different bars.
        ax.text(
            ANNOTATION_X,
            pos,
            f"{row['effect']:.2f} pp moved, {row['floor']:.2f} pp needed  ({row['source']})",
            ha="left",
            va="center",
            fontsize=9.5,
            color=PALETTE["text_muted"],
        )
    ax.set_yticks(positions)
    ax.set_yticklabels([row["label"] for row in rows], fontsize=11, color=PALETTE["text"])
    ax.set_xlim(0.0, 9.6)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(
        "median |change in deserve-to-win share|, percentage points",
        color=PALETTE["text_muted"],
        fontsize=10,
        loc="left",
    )
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.grid(axis="x", color=PALETTE["grid"], linestyle="--", alpha=0.6, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    # The axis stops where the data stops. The annotation column lives past it,
    # and a spine running under the words would read as more axis to come.
    ax.spines["bottom"].set_bounds(0.0, 4.2)
    ax.tick_params(axis="y", length=0)
    fig.text(
        0.045,
        0.055,
        MECHANISM_REFUSALS,
        ha="left",
        va="center",
        fontsize=9,
        color=PALETTE["text_muted"],
        linespacing=1.35,
    )
    finalize(fig, FIGURE_DIR / "14_refused_floors.png")
    return {row["label"]: (row["effect"], row["floor"]) for row in rows}


# --------------------------------------------------------------------------
# figure 15 — the same shrinkage, on the dropped-pick model
# --------------------------------------------------------------------------

# How many defence-seasons figure 15 draws, and how they are chosen: the two
# highest and the two lowest observed conversion rates, plus the one nearest the
# median, with ties broken by the most interception-worthy throws faced. The
# tie-break is what keeps a two-throw season out of the picture.
SHRINKAGE_ROWS = 5


def denver_2025_followup() -> dict:
    """Denver the year after: the shrunk 55.2% beat the raw 76.5% as a forecast.

    the maintainer 2026-08-31. Raw 2025 rate on interception-worthy throws against the
    Denver defense, from the same charting join every drop figure uses. The
    check pins the counts so the article's sentence cannot drift from the data.
    """
    import glob

    pbp = pl.read_parquet("data/pbp/pbp_2025.parquet").with_columns(
        pl.col("play_id").cast(pl.Int32)
    )
    ftn = pl.read_parquet(glob.glob("data/ftn/*2025*")[0])
    joined = pbp.join(
        ftn,
        left_on=["game_id", "play_id"],
        right_on=["nflverse_game_id", "nflverse_play_id"],
        how="inner",
    )
    worthy = joined.filter(pl.col("is_interception_worthy"))
    denver = worthy.filter(pl.col("defteam") == "DEN")
    caught = int(denver["interception"].sum())
    check("den_2025_worthy", denver.height)
    check("den_2025_caught", caught)
    return {
        "worthy": denver.height,
        "caught": caught,
        "rate": caught / denver.height,
        "league": float(worthy["interception"].mean()),
    }


def defence_shrinkage() -> dict:
    """Five defence-seasons' raw dropped-pick rate, and where the model puts it.

    Figure 9's arrow, drawn from a different model. Fumble recovery is a
    beta-binomial on a team-season's own count; this is a **crossed logistic
    regression** whose defence-season term is one intercept among eighteen
    covariates, so the arrow is "the same shrinkage idea from a different
    model" rather than the same estimator applied twice. The model's rate for a
    defence-season is the posterior mean catch probability over that
    defence-season's **own** interception-worthy throws, so the covariate mix a
    defence actually faced is what it is scored on.
    """
    from nfl_simulator.dropped_picks import DroppedPickModel, worthy_throw_frame
    from nfl_simulator.render import _simulation_context

    context = _simulation_context()
    model = DroppedPickModel.from_posterior(
        OUTPUTS / "trace_dropped_pick.nc", OUTPUTS / "dropped_pick_summary.json"
    )
    worthy = worthy_throw_frame(context["pbp"], context["ftn"])
    counts = (
        worthy.group_by("defence_season")
        .agg(
            pl.len().alias("n"),
            pl.col("interception").cast(pl.Int64).sum().alias("k"),
        )
        .with_columns((pl.col("k") / pl.col("n")).alias("observed_rate"))
        # Three sort keys, not two. `group_by` does not promise an order, and a
        # sort on rate and count alone leaves the several defence-seasons tied
        # at exactly 0.5 in whatever order the grouping happened to emit — which
        # drew a different median row on two consecutive runs of this script.
        # The name is the total order that makes the figure reproducible.
        .sort("observed_rate", "n", "defence_season", descending=[True, True, False])
    )
    middle = counts.height // 2
    chosen = [0, 1, middle, counts.height - 2, counts.height - 1]
    picked = counts[chosen]

    rows = []
    for row in picked.iter_rows(named=True):
        season = row["defence_season"]
        plays = worthy.filter(pl.col("defence_season") == season).to_dicts()
        posterior = float(np.mean([model.catch_probability(season, play).mean() for play in plays]))
        rows.append({**row, "posterior_mean": posterior})
    league = float(counts["k"].sum() / counts["n"].sum())
    # Document 50 §2's `sigma_d`, carried from the logit scale to the
    # probability scale at the league rate: how far apart the model believes
    # defences actually are, which is what sets how hard each dot is pulled.
    spread_pp = check(
        "defence_spread_pp",
        round(
            json.loads((OUTPUTS / "dropped_pick_summary.json").read_text())["sigma_d_mean"]
            * league
            * (1 - league)
            * 100,
            1,
        ),
    )

    fig, ax = new_figure(
        8.0,
        5.8,
        # the maintainer, round 5. The old title said "what a defense recorded" while the
        # x-axis said "actually picked", and document 65's finding A-3 names
        # that mixed framing as the soil the article's reversed caption (W-1)
        # grew in. One frame now, in the title and on the axis: this is a
        # **catch** rate, and high is good.
        title="Modeling each defense's interception catch rate",
        subtitle=[
            "What a defense caught on interceptable throws, and where the model puts it",
            f"A crossed logistic regression, not the beta-binomial above — and it believes "
            f"defenses differ by about {spread_pp:.1f} percentage points.",
        ],
    )
    # The key goes under the axis. At the top right it printed across the league
    # rule, which on this figure sits in the middle of the axis.
    ax.set_position([0.22, 0.20, 0.74, 0.60])
    positions = np.arange(len(rows))[::-1]
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
    for pos, row in zip(positions, rows, strict=True):
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
            f"{row['defence_season'].split('|')[1]} {row['defence_season'].split('|')[0]}"
            f"  ({row['k']} of {row['n']})"
            for row in rows
        ],
        fontsize=10,
        color=PALETTE["text"],
    )
    ax.set_xlim(0.0, 0.95)
    ax.set_ylim(-0.7, len(positions) - 0.15)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel(
        "share of interception-worthy throws the defense caught",
        color=PALETTE["text_muted"],
        fontsize=10,
    )
    ax.grid(axis="x", color=PALETTE["grid"], linestyle="--", alpha=0.6, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(
        0.22,
        0.055,
        "red = what the defense caught   ·   grey = what the model believes",
        ha="left",
        va="center",
        fontsize=9,
        color=PALETTE["text_muted"],
    )
    finalize(fig, FIGURE_DIR / "15_defense_shrinkage.png")
    return {
        "league": league,
        "rows": [
            (row["defence_season"], row["observed_rate"], row["posterior_mean"]) for row in rows
        ],
    }


# --------------------------------------------------------------------------
# figure 16 — one game, with and without the hands-on-the-ball rows
# --------------------------------------------------------------------------


def den_was_with_without(walkthrough: dict) -> dict:
    """The walk-through game's distribution, before and after amendment A-3.

    Two panels on one x axis, so the second is read as a movement of the first
    rather than as a different game. ``walkthrough`` is document 64 §8's own
    dictionary — the same two replays, each already checked against its
    published summary — so the picture and the document cannot disagree.
    """
    from nfl_simulator.teams import pair_colors

    home, away = walkthrough["home_team"], walkthrough["away_team"]
    home_colour, away_colour = pair_colors(home, away, int(walkthrough["game_id"][:4]))
    strict = walkthrough["editions"]["strict"]
    full = walkthrough["editions"]["full"]
    panels = [
        (strict, "without them — v1.3 alone", f"{away} {1 - strict['dtw_home']:.0%}"),
        (
            full,
            "with them — dropped picks and receiver drops",
            f"{away} {1 - full['dtw_home']:.0%}",
        ),
    ]

    # One set of bin edges for both panels, anchored so that **zero is an edge**:
    # zero is the line that decides the winner, and a bar straddling it would put
    # some of one team's wins inside the other team's colour.
    every_draw = np.concatenate([strict["margin_draws"], full["margin_draws"]])
    width = 3.0
    lo = np.floor(np.percentile(every_draw, 0.2) / width - 1) * width
    hi = np.ceil(np.percentile(every_draw, 99.8) / width + 1) * width
    edges = np.arange(lo, hi + width, width)
    # And one y limit, so the two panels' bar heights mean the same thing.
    tallest = max(
        (np.histogram(edition["margin_draws"], bins=edges)[0] / len(edition["margin_draws"])).max()
        for edition, _caption, _share in panels
    )

    fig = plt.figure(figsize=(8.4, 6.4))
    fig.patch.set_facecolor(PALETTE["bg"])
    draw_title_block(
        title_axes(fig, height_frac=0.15),
        "What the hands-on-the-ball rows do to one game",
        [
            f"{away} at {home}, week 13 of 2025 — {away} won by "
            f"{abs(int(walkthrough['away_score'] - walkthrough['home_score']))} in overtime",
            "Same axis, same coins. Only the list of priced events changes.",
        ],
        title_size=17,
    )
    axes = [
        fig.add_axes([0.10, 0.48, 0.86, 0.30]),
        fig.add_axes([0.10, 0.12, 0.86, 0.30]),
    ]
    for ax, (edition, caption, share) in zip(axes, panels, strict=True):
        ax.set_facecolor(PALETTE["bg"])
        counts, _ = np.histogram(edition["margin_draws"], bins=edges)
        centres = (edges[:-1] + edges[1:]) / 2
        share_of_draws = counts / counts.sum()
        ax.bar(
            centres,
            share_of_draws,
            width=(edges[1] - edges[0]) * 0.9,
            color=np.where(centres < 0, away_colour, home_colour),
            edgecolor=PALETTE["bg"],
            linewidth=0.8,
            zorder=3,
        )
        ax.axvline(0.0, color=PALETTE["text"], linewidth=1.4, dashes=(2, 3), zorder=4)
        ax.axvline(
            edition["deserved_margin"],
            color=PALETTE["text_muted"],
            linewidth=1.6,
            dashes=(5, 3),
            zorder=4,
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(0.0, tallest * 1.30)
        ax.set_ylabel("% of simulations", color=PALETTE["text_muted"], fontsize=9)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, min_n_ticks=2))
        ax.xaxis.set_major_formatter(lambda v, _: f"{abs(v):g}")
        ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8, alpha=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(PALETTE["spine"])
        ax.tick_params(colors=PALETTE["text_muted"], labelsize=9)
        # The panels share an axis, so only the lower one prints its ticks —
        # the upper panel's numbers landed in the lower panel's caption row.
        if ax is not axes[-1]:
            ax.tick_params(axis="x", labelbottom=False, length=0)
        ax.text(
            0.0,
            1.06,
            caption,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.5,
            color=PALETTE["text"],
        )
        ax.text(
            1.0,
            1.06,
            f"{share} of the simulations",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=away_colour,
        )
        # The label sits on the far side of its own rule from zero, so it never
        # prints across the line that decides the winner.
        leans_home = edition["deserved_margin"] > 0
        ax.text(
            edition["deserved_margin"] + (0.4 if leans_home else -0.4),
            tallest * 1.26,
            f"expected margin {abs(edition['deserved_margin']):.1f} to "
            f"{home if leans_home else away}",
            ha="left" if leans_home else "right",
            va="top",
            fontsize=9,
            color=PALETTE["text_muted"],
        )
    axes[-1].set_xlabel(
        f"← {away} wins by          ·          {home} wins by →",
        color=PALETTE["text_muted"],
        fontsize=10,
    )
    finalize(fig, FIGURE_DIR / "16_den_was_with_without.png", edition="full")
    return {
        "strict_away_share": 1 - strict["dtw_home"],
        "full_away_share": 1 - full["dtw_home"],
    }


# --------------------------------------------------------------------------
# the annotated formula plates
# --------------------------------------------------------------------------

# Round 2's plates are annotated: the equation is laid out **one text object per
# term**, so every term has a measured bounding box, and the terms a reader has
# to be told about get a plain-English label directly under them. Draft 1's
# plates were one centred string apiece, which is a picture of an equation and
# not an explanation of one.
#
# Every plate is built from three ingredients:
#
# * ``terms`` — ``(key, mathtext, gap after it in points)``, laid out left to
#   right. Operators and brackets are terms too, with their own keys, because
#   the layout measures whatever it is handed and a bracket that shares a text
#   object with its contents cannot be spaced independently of them.
# * ``labels`` — ``key -> words``. A label carries **no position of its own**:
#   it is centred on its term and joined to it by a short vertical tick, and the
#   gaps in ``terms`` are what make room for it. Two labels that would touch
#   fail the render rather than shipping an overlap — see
#   :func:`_check_labels_clear`.
# * the plate's own size.
#
# `usetex=False` throughout — matplotlib's mathtext, which is what every other
# figure in this repo renders and what a machine without a TeX install has.

# The gap that follows a term, in points, when nothing says otherwise.
TERM_GAP = 9.0

# The label apparatus, **in inches**. The spike's labels spread along the bottom
# edge and reached their terms on a bent two-segment leader, which asks a reader
# to trace a line before they can read a word; a label centred under its own
# term needs no line at all, only a tick short enough to read as furniture.
# `TICK_GAP_IN` clears the highlight wash, `TICK_LENGTH_IN` is the tick,
# `LABEL_GAP_IN` is the air under it. Inches rather than figure fractions
# because a two-row plate is twice as tall as a one-row plate and the same
# fraction would draw it a tick twice as long.
TICK_GAP_IN = 0.06
TICK_LENGTH_IN = 0.23
LABEL_GAP_IN = 0.06

# How much clear plate has to sit between two neighbouring labels.
LABEL_CLEARANCE_IN = 0.14

# How far the highlight wash extends past its term, vertically. Named because
# the ticks start below it and would otherwise be measured off the wrong edge.
HIGHLIGHT_PAD_IN = 0.09

# One row of a plate: an annotated row carries an equation, a tick and up to
# three lines of label; a bare row carries the equation alone. `EQUATION_OFFSET`
# is where the equation's centre line sits below the row's top.
ANNOTATED_ROW_IN = 1.90
BARE_ROW_IN = 0.95
EQUATION_OFFSET_IN = 0.40

RULE_TERMS = [
    ("lhs", r"$\mathrm{luck}(e)$", 13.0),
    ("equals", r"$=$", 13.0),
    ("open", r"$\left(\right.$", 4.0),
    ("y", r"$y(e)$", 20.0),
    ("minus", r"$-$", 20.0),
    ("p", r"$p(e)$", 4.0),
    ("close", r"$\left.\right)$", 15.0),
    ("times", r"$\times$", 15.0),
    ("swing", r"$\mathrm{impact}(e)$", 0.0),
]

RULE_LABELS = {
    "y": "what happened\n1 if the ball fell\ntheir way, 0 if not",
    "p": "what was expected\nthe shrunk rate\nfor events like it",
    "swing": "what it was worth\nthe gap between\nthe two outcomes",
}


def _fraction_boxes(fig, terms: list[tuple[str, str, float]], y: float, size: float) -> dict:
    """Lay the terms out left to right, centred, and return each one's box.

    Two passes, because a term's width is not knowable until it is drawn: the
    first places every term at the left edge and measures it, the second moves
    it to where the measurement says it belongs. The returned boxes are in
    figure fractions — ``(left, right, bottom, top)`` — which is what the ticks
    and the highlight patches are drawn in.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width_px, height_px = fig.get_size_inches() * fig.dpi
    drawn = [
        fig.text(0.0, y, text, fontsize=size, color=PALETTE["text"], ha="left", va="center")
        for _key, text, _gap in terms
    ]
    widths = [text.get_window_extent(renderer).width for text in drawn]
    heights = [text.get_window_extent(renderer).height for text in drawn]
    gaps = [gap / 72.0 * fig.dpi for _key, _text, gap in terms]
    total = sum(widths) + sum(gaps[:-1]) if len(terms) > 1 else sum(widths)
    cursor = (width_px - total) / 2.0

    boxes = {}
    for (key, _text, _gap), text, width, height, gap in zip(
        terms, drawn, widths, heights, gaps, strict=True
    ):
        text.set_position((cursor / width_px, y))
        boxes[key] = (
            cursor / width_px,
            (cursor + width) / width_px,
            y - height / 2.0 / height_px,
            y + height / 2.0 / height_px,
        )
        cursor += width + gap
    return boxes


def _check_labels_clear(fig, keys: list[str], drawn: list) -> None:
    """Refuse to save a plate whose labels touch each other.

    A label sits under its own term, so the only thing keeping two of them apart
    is the gap between those terms — which is hand-set per plate and can
    therefore be wrong. An overlap is invisible in a thumbnail and unreadable in
    the article, so it is measured rather than eyeballed.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width_px = fig.get_size_inches()[0] * fig.dpi
    spans = [
        (
            text.get_window_extent(renderer).x0 / width_px,
            text.get_window_extent(renderer).x1 / width_px,
        )
        for text in drawn
    ]
    order = sorted(range(len(spans)), key=lambda index: spans[index][0])
    for left, right in zip(order, order[1:], strict=False):
        clear = (spans[right][0] - spans[left][1]) * fig.get_size_inches()[0]
        if clear < LABEL_CLEARANCE_IN:
            raise AssertionError(
                f"the labels under {keys[left]!r} and {keys[right]!r} are {clear:.3f} in "
                f"apart and need {LABEL_CLEARANCE_IN} in. Widen those terms' gaps or "
                "shorten the words — do not ship the overlap."
            )


def _annotate_terms(fig, boxes: dict, labels: dict, *, size: float) -> None:
    """A plain-English label directly under each named term, on a short tick.

    Every tick hangs from one line — the lowest of the annotated terms' washes —
    so the ticks are all one length and the labels share a top edge. Both are
    the figure's muted grey: the words under an equation are apparatus, and
    apparatus that competes with the equation for attention has the emphasis
    backwards.
    """
    keys = list(labels)
    height = fig.get_size_inches()[1]
    tick_top = min(boxes[key][2] for key in keys) - (HIGHLIGHT_PAD_IN + TICK_GAP_IN) / height
    tick_bottom = tick_top - TICK_LENGTH_IN / height
    drawn = []
    for key in keys:
        left, right, _bottom, _top = boxes[key]
        centre = (left + right) / 2.0
        fig.lines.append(
            plt.Line2D(
                [centre, centre],
                [tick_top, tick_bottom],
                transform=fig.transFigure,
                color=PALETTE["spine"],
                linewidth=0.9,
                zorder=1,
            )
        )
        drawn.append(
            fig.text(
                centre,
                tick_bottom - LABEL_GAP_IN / height,
                labels[key],
                fontsize=size,
                color=PALETTE["text_muted"],
                ha="center",
                va="top",
                linespacing=1.32,
                zorder=2,
            )
        )
    _check_labels_clear(fig, keys, drawn)


def _highlight_terms(fig, boxes: dict, keys) -> None:
    """A faint warm wash behind each annotated term.

    The wash is the house `row_alt`, which is the same tint the ledger's
    alternating rows wear — so on a white page it reads as the figure's own
    paper rather than as a colour that means something. It is what makes "this
    word, not that one" visible at a glance before any tick is followed.
    """
    from matplotlib.patches import FancyBboxPatch

    # The wash has to stop short of the bracket either side of it. At 0.006 the
    # padded box swallowed the opening bracket into `y(e)` and the closing one
    # into `p(e)`, and a highlight that includes a bracket is highlighting the
    # wrong thing.
    pad_x = 0.0035
    pad_y = HIGHLIGHT_PAD_IN / fig.get_size_inches()[1]
    for key in keys:
        left, right, bottom, top = boxes[key]
        fig.patches.append(
            FancyBboxPatch(
                (left - pad_x, bottom - pad_y),
                (right - left) + 2 * pad_x,
                (top - bottom) + 2 * pad_y,
                boxstyle="round,pad=0.004,rounding_size=0.012",
                transform=fig.transFigure,
                facecolor=PALETTE["row_alt"],
                edgecolor="none",
                zorder=0,
            )
        )


def annotated_plate(
    name: str,
    rows: list[tuple[list[tuple[str, str, float]], dict]],
    *,
    width: float = 9.0,
    term_size: float = 25.0,
    label_size: float = 10.5,
) -> str:
    """One annotated formula, written transparent at 300 dpi.

    ``rows`` is one ``(terms, labels)`` pair per line of the plate. A row whose
    labels are empty is an unannotated line — the second half of a two-line
    definition, where the first line is where the reader needs the words — and
    is given a shorter band because it has nothing hanging under it.

    Not `finalize`: a formula is typography, not a figure, and stamping a data
    credit on an equation would credit nflverse with the algebra.
    """
    heights = [ANNOTATED_ROW_IN if labels else BARE_ROW_IN for _terms, labels in rows]
    height = sum(heights)
    fig = plt.figure(figsize=(width, height))
    fig.patch.set_alpha(0.0)

    top = height
    for (terms, labels), row_height in zip(rows, heights, strict=True):
        boxes = _fraction_boxes(fig, terms, (top - EQUATION_OFFSET_IN) / height, term_size)
        if labels:
            _highlight_terms(fig, boxes, labels)
            _annotate_terms(fig, boxes, labels, size=label_size)
        top -= row_height

    target = FIGURE_DIR / f"{name}.png"
    fig.savefig(target, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return target.name


# --------------------------------------------------------------------------
# the four remaining plates
# --------------------------------------------------------------------------

# Medium renders neither LaTeX nor Mermaid, so every equation in the article is
# an image. Each plate below is one `![...](figures/formula_NN_*.png)` line in
# `docs/writeup/community-writeup.md`, and the two are edited together: the
# words under a term here and the sentence that unpacks it there have to agree.

POINTS_TERMS = [
    ("lhs", r"$M_{\mathrm{deserved}}$", 13.0),
    ("equals", r"$=$", 13.0),
    ("actual", r"$M_{\mathrm{actual}}$", 15.0),
    ("minus", r"$-$", 15.0),
    ("slope", r"$0.8389$", 12.0),
    ("times", r"$\times$", 12.0),
    ("sum", r"$\sum_{e}\,\mathrm{luck}(e)$", 0.0),
]

POINTS_LABELS = {
    "slope": "points per EPA\nmeasured, not chosen\n(slope of margin on EPA)",
    "sum": "every luck event\nin the game,\nadded up",
}

DTW_TERMS = [
    ("dtw", r"$\mathrm{DTW}\%$", 13.0),
    ("equals", r"$=$", 13.0),
    ("prob", r"$P($", 4.0),
    ("inside", r"$M^{*}_{\mathrm{deserved}} > 0$", 4.0),
    ("close", r"$)$", 0.0),
]

DTW_LABELS = {
    "dtw": "deserve-to-win share",
    "inside": "one re-simulation of the game\nleaves this team ahead",
}

FG_TERMS = [
    ("lhs", r"$\mathrm{logit}\,p_{\mathrm{make}}$", 12.0),
    ("equals", r"$=$", 12.0),
    ("alpha", r"$\alpha$", 11.0),
    ("plus1", r"$+$", 11.0),
    ("distance", r"$\beta d$", 15.0),
    ("plus2", r"$+$", 15.0),
    ("curve", r"$\gamma\frac{d^{2}}{100} + \delta\frac{d^{3}}{1000}$", 15.0),
    ("plus3", r"$+$", 15.0),
    ("roof", r"$\rho_{\mathrm{roof}}$", 15.0),
    ("plus4", r"$+$", 15.0),
    ("wind", r"$\beta_{w}W$", 15.0),
    ("plus5", r"$+$", 15.0),
    ("temp", r"$\beta_{t}T$", 15.0),
    ("plus6", r"$+$", 15.0),
    ("elev", r"$\beta_{e}E$", 26.0),
    ("plus7", r"$+$", 15.0),
    ("kicker", r"$\sigma_{k}z_{k}$", 15.0),
]

FG_LABELS = {
    "distance": "how far,\nfrom 40 yards",
    "curve": "the distance curve —\nlong kicks fall away faster",
    "roof": "indoors\nor out",
    "wind": "wind",
    "temp": "temperature",
    "elev": "elevation, in\nthousands of feet",
    "kicker": "this kicker,\nthis season",
}

BETA_TERMS = [
    ("rate", r"$r_{t}$", 13.0),
    ("tilde", r"$\sim$", 13.0),
    ("beta", r"$\mathrm{Beta}\left(\mu\kappa,\; (1-\mu)\kappa\right)$", 0.0),
]

BETA_LABELS = {
    "rate": "unit t's true rate,\nwhich is never observed",
    "beta": "the league's spread: mean μ,\nand κ for how tightly\nthe units cluster around it",
}

BINOMIAL_TERMS = [
    ("count", r"$k_{t}$", 13.0),
    ("tilde", r"$\sim$", 13.0),
    ("binomial", r"$\mathrm{Binomial}\left(n_{t},\; r_{t}\right)$", 0.0),
]

BINOMIAL_LABELS = {
    "count": "what it recorded",
    "binomial": "out of the chances it had,\nat that true rate",
}

TRUST_TERMS = [
    ("lhs", r"$p(e)$", 13.0),
    ("equals", r"$=$", 13.0),
    ("weight", r"$w$", 30.0),
    ("dot1", r"$\cdot$", 30.0),
    ("unit", r"$\hat{r}_{\mathrm{unit}}$", 15.0),
    ("plus", r"$+$", 15.0),
    ("rest", r"$(1-w)$", 13.0),
    ("dot2", r"$\cdot$", 13.0),
    ("league", r"$\bar{r}_{\mathrm{league}}$", 0.0),
]

TRUST_LABELS = {
    "weight": "how much to trust\nits own rate",
    "unit": "what this\nunit did",
    "league": "what the\nleague does",
}

WEIGHT_TERMS = [
    ("lhs", r"$w$", 13.0),
    ("equals", r"$=$", 13.0),
    ("fraction", r"$\dfrac{n}{n+\kappa}$", 0.0),
]

WEIGHT_LABELS = {
    "fraction": "how many chances the unit has been seen to take,\nagainst how tightly the league clusters — measured, never chosen",
}

# name -> (rows, plate width in inches, term size, label size). One entry per
# `formula_*.png` the article references, and nothing else: a plate nobody links
# to is a file in the budget doing no work.
PLATES = {
    "formula_01_rule": ([(RULE_TERMS, RULE_LABELS)], 9.0, 25.0, 10.5),
    "formula_02_points": (
        [(POINTS_TERMS, POINTS_LABELS), (DTW_TERMS, DTW_LABELS)],
        9.6,
        23.0,
        10.5,
    ),
    "formula_03_fg": ([(FG_TERMS, FG_LABELS)], 14.6, 20.0, 9.5),
    "formula_04_betabinomial": (
        [(BETA_TERMS, BETA_LABELS), (BINOMIAL_TERMS, BINOMIAL_LABELS)],
        9.6,
        23.0,
        10.5,
    ),
    "formula_05_trust": (
        [(TRUST_TERMS, TRUST_LABELS), (WEIGHT_TERMS, WEIGHT_LABELS)],
        10.2,
        23.0,
        10.5,
    ),
}


def formula_plates() -> list[str]:
    """Every annotated plate, in the order the article meets them."""
    return [
        annotated_plate(name, rows, width=width, term_size=term_size, label_size=label_size)
        for name, (rows, width, term_size, label_size) in PLATES.items()
    ]


# --------------------------------------------------------------------------
# figure 17 — the six fumble classes the simulator actually prices with
# --------------------------------------------------------------------------


def _align_title_with_ticks(fig, ax) -> None:
    """Left-align the title block with the y tick labels (the maintainer 2026-08-31).

    `barh` tick labels hang outside the axes, further left than the title
    block's own margin, and the mismatch reads as a layout accident. Measured
    after a draw because the labels' width is a font fact, not a layout one.
    """
    fig.canvas.draw()
    labels = ax.get_yticklabels()
    if not labels:
        return
    left = min(label.get_window_extent().x0 for label in labels) / fig.get_window_extent().width
    title_ax = fig.axes[0]
    box = title_ax.get_position()
    title_ax.set_position([left, box.y0, box.x1 - left, box.height])


def fumble_retention_bars() -> str:
    """Document 64 §7a as a chart: the rate each class of fumble is kept at.

    The article's one claim about fumbles is that a loose ball is a coin, and
    §7a is where that claim stops being a slogan — a botched snap on a run play
    is kept 77% of the time and a fumble on a run 46%, so "a coin" is a
    statement about a *class*, not about fumbles. Six bars say that in one
    look; the sentence took the article a paragraph and lost the arithmetic.

    Only the six **measured** classes are drawn. `field_goal/live` and
    `punt/aborted` hold six plays between them across ten seasons and wear the
    pooled rate under `fit_fumble_baseline`'s 30-fumble floor, so a bar for
    them would be the pooled rate drawn twice under two labels a reader would
    take for measurements.
    """
    rates = fumble_class_rates()
    measured = sorted(rates["measured"], key=lambda row: row["p_own"])
    # Both sub-floor classes carry the same pooled rate, which is the line.
    pooled_rate = float(rates["pooled"][0]["p_own"])

    fig, ax = new_figure(
        8.4,
        5.4,
        title="Not all fumbles are the same",
        subtitle=[
            "How often the team that fumbled keeps the ball, by the kind of play it happened on.",
            "2016–2025, the fitted baseline the simulator prices every ledger row with.",
        ],
    )

    positions = np.arange(len(measured))
    values = [row["p_own"] * 100 for row in measured]
    # One hue, one meaning: these are eight readings of a single quantity, so
    # they are one colour and the *length* carries the difference. Colouring
    # each class would say the classes are categories a reader must tell apart
    # by hue, which the labels already do.
    ax.barh(positions, values, height=0.62, color=PALETTE["anchor"], alpha=0.85)
    ax.set_yticks(positions)
    ax.set_yticklabels([row["gloss"] for row in measured], fontsize=9.5)
    ax.set_xlim(0, 112)
    ax.set_xlabel("kept by the fumbling team (%)", fontsize=9, color=PALETTE["text_muted"])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: f"{v:g}%"))
    ax.grid(axis="x", color=PALETTE["grid"], linestyle="--", alpha=0.6, linewidth=0.8)
    ax.set_axisbelow(True)

    for position, row in zip(positions, measured, strict=True):
        # The rate outside the bar and the count inside it. Both outside and
        # they read as one number with a strange unit; both inside and the
        # 46% bar has no room for either.
        ax.text(
            row["p_own"] * 100 + 1.6,
            position,
            f"{row['p_own'] * 100:.0f}%",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=PALETTE["text"],
        )
        ax.text(
            1.5,
            position,
            f"{row['n']:,} fumbles",
            va="center",
            fontsize=8.5,
            color=PALETTE["bg"],
        )

    league = pooled_rate * 100
    # Above the bars, below the labels. At `zorder=4` the dashed rule was drawn
    # straight through "51.2%" and "51.0%", the two values it passes nearest —
    # which are exactly the two a reader is checking it against.
    ax.axvline(league, color=PALETTE["bad"], linewidth=1.6, linestyle=(0, (4, 2)), zorder=2.5)
    ax.annotate(
        f"all fumbles pooled: {league:.0f}%",
        xy=(league, len(measured) - 0.42),
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=9,
        color=PALETTE["bad"],
        va="center",
    )

    ax.annotate(
        f"{rates['n_fumbles']:,} fumbles, 2016–2025. A pooled 50/50 would book a fake "
        "bad-luck charge against every offense that fell on its own botched snap.",
        xy=(0, 0),
        xycoords="axes fraction",
        xytext=(0, -42),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8,
        color=PALETTE["text_muted"],
    )
    _align_title_with_ticks(fig, ax)
    return finalize(fig, FIGURE_DIR / "17_fumble_retention_bars.png").name


# --------------------------------------------------------------------------
# figures 18 and 19 — a prior and a posterior, twice
# --------------------------------------------------------------------------


def _prior_posterior_axes(title: str, subtitle: list[str]):
    """The shared shape of figures 18 and 19: two curves on one probability axis."""
    fig, ax = new_figure(8.2, 5.2, title=title, subtitle=subtitle)
    ax.set_ylabel("posterior density", fontsize=9, color=PALETTE["text_muted"])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=PALETTE["grid"], linestyle="--", alpha=0.6, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _density(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """A Gaussian kernel density, normalised to a peak of one.

    Normalised to its own peak rather than to unit area because the two curves
    on these figures have very different spreads: at equal area the wide one is
    a flat smear beside a spike, and the figure's question — where does each
    curve sit — is answered by position, not by height.
    """
    from scipy.stats import gaussian_kde

    curve = gaussian_kde(values)(grid)
    return curve / curve.max()


def _draw_logo(ax, logo, x: float, y: float, *, width_in: float = 0.42) -> None:
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    if logo is None:
        return
    from nfl_simulator.plots import logo_zoom

    ax.add_artist(
        AnnotationBbox(
            OffsetImage(
                logo,
                zoom=logo_zoom(
                    logo, ax.figure, max_width_in=width_in, max_height_in=width_in * 0.62
                ),
            ),
            (x, y),
            xycoords="axes fraction",
            frameon=False,
            annotation_clip=False,
            box_alignment=(0.5, 0.5),
        )
    )


def kicker_prior_posterior() -> dict:
    """Figure 18 — one kicker-season's own record, priced.

    The best 2025 kicker-season with at least twenty attempts, on the
    **make-probability-at-45-yards** scale rather than on the logit scale the
    model fits: nobody has an intuition for +0.45 log-odds, and everybody has
    one for "makes four more of a hundred 45-yarders".

    Both curves are pushed through the same league curve, so the distance
    between them is the whole of what this kicker's record bought.
    """
    import arviz as az

    from nfl_simulator.fg_model import _sigmoid
    from nfl_simulator.render import _read_side
    from nfl_simulator.teams import team_colors, team_logo

    model, _ = _read_side().load_model("trace_fg_v14.nc", "fg_v14_summary.json")
    posterior = az.from_netcdf(OUTPUTS / "trace_fg_v14.nc")["posterior"]
    sigma = posterior["sigma_kicker"].values.ravel()

    effects = pl.read_parquet(OUTPUTS / "fg_kicker_effects.parquet").with_columns(
        pl.col("kicker_season").str.slice(0, 4).cast(pl.Int32).alias("season")
    )
    best = (
        effects.filter((pl.col("season") == KICKER_SEASON) & (pl.col("attempts") >= KICKER_FLOOR))
        .sort("effect_mean", descending=True)
        .row(0, named=True)
    )
    key = best["kicker_season"]

    league_logit = model._logit(KICKER_DISTANCE)
    league = _sigmoid(league_logit)
    # The prior, on the make-probability scale: a kicker drawn from
    # `Normal(0, sigma_kicker)` before anybody has watched them kick. Paired
    # draw for draw with the curve, so the spread is the kicker's and not the
    # league curve's uncertainty leaking in.
    rng = np.random.default_rng(20260830)
    prior = _sigmoid(league_logit + rng.normal(0.0, sigma))
    shrunk = _sigmoid(league_logit + model.kicker_effects[key])

    one_sigma_pp = check(
        "kicker_one_sigma_pp",
        round(float((_sigmoid(league_logit + sigma) - league).mean() * 100), 2),
    )
    posterior_pp = round(float((shrunk - league).mean() * 100), 2)
    league_pct = check("fg_league_p45", round(float(league.mean() * 100), 2))

    team = KICKER_TEAM
    primary, _secondary = team_colors(team, KICKER_SEASON)
    # the maintainer, round 5. The subtitle has to say, in those words, which of the two
    # numbers on this figure is a record and which is a model: readers were
    # taking the 31 of 32 for the curve's own claim, when the curve is the
    # shrunk posterior and sits nowhere near 97%.
    fig, ax = _prior_posterior_axes(
        f"{KICKER_NAME}'s FG probability vs the league",
        [
            f"His {KICKER_SEASON} record is {int(best['made'])} of {best['attempts']} — raw, "
            "unshrunk, and not what the simulator prices with.",
            f"The colored curve is his shrunk posterior at {KICKER_DISTANCE:.0f} yards, "
            f"against a league that makes {league.mean() * 100:.1f}%.",
        ],
    )

    # Trimmed to the 0.5th-99.5th percentile of the two curves together. The
    # prior's full support runs down past 45%, which is thirty points of empty
    # axis bought to show a tail nobody is reading.
    both = np.concatenate([prior, shrunk])
    grid = np.linspace(np.quantile(both, 0.005) - 0.01, np.quantile(both, 0.995) + 0.01, 512)
    prior_curve, shrunk_curve = _density(prior, grid), _density(shrunk, grid)

    ax.fill_between(grid * 100, 0, prior_curve, color=PALETTE["anchor"], alpha=0.22, linewidth=0)
    ax.plot(grid * 100, prior_curve, color=PALETTE["anchor"], linewidth=1.6)
    ax.fill_between(grid * 100, 0, shrunk_curve, color=primary, alpha=0.32, linewidth=0)
    ax.plot(grid * 100, shrunk_curve, color=primary, linewidth=2.2)

    ax.axvline(league_pct, color=PALETTE["text_muted"], linewidth=1.2, linestyle=":", zorder=1.5)
    # Above the curves. Beside the line is ink — both curves are tall where it
    # crosses — and below the axis is the x-label's row, which is where the
    # second render put it and where it printed through "make probability".
    ax.annotate(
        f"league {league_pct:.1f}%",
        xy=(league_pct, 1.15),
        xytext=(-5, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=8.5,
        color=PALETTE["text_muted"],
    )
    ax.annotate(
        f"{KICKER_NAME}: {shrunk.mean() * 100:.1f}%   (+{posterior_pp:.2f} pp)",
        xy=(shrunk.mean() * 100, shrunk_curve.max()),
        xytext=(0, 16),
        textcoords="offset points",
        ha="center",
        fontsize=10.5,
        fontweight="bold",
        color=primary,
    )
    ax.annotate(
        f"before the season: any kicker,\n±{one_sigma_pp:.2f} pp at one standard deviation",
        xy=(0.02, 0.70),
        xycoords="axes fraction",
        fontsize=8.5,
        color=PALETTE["text_muted"],
    )
    # the maintainer, round 5: the man rather than the badge, when the file is there.
    # Drawn **outside** the posterior's right tail rather than on it — a face
    # over the curve is ink on the quantity the figure exists to show. The file
    # is untracked and gitignored, so a fresh clone falls back to the team mark
    # and the figure still draws.
    headshot = OUTPUTS / KICKER_HEADSHOT
    if headshot.exists():
        _draw_logo(ax, plt.imread(headshot), 0.93, 0.62, width_in=0.60)
    else:
        _draw_logo(ax, team_logo(team, KICKER_SEASON), 0.855, 0.30)

    ax.set_xlabel(
        f"make probability from {KICKER_DISTANCE:.0f} yards (%)",
        fontsize=9,
        color=PALETTE["text_muted"],
    )
    ax.set_ylim(0, 1.18)
    ax.annotate(
        "Each of the 200 replays draws one value from the colored curve, and flips "
        "that kick at it.",
        xy=(0, 0),
        xycoords="axes fraction",
        xytext=(0, -42),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8,
        color=PALETTE["text_muted"],
    )
    name = finalize(fig, FIGURE_DIR / "18_kicker_prior_posterior.png").name
    return {
        "file": name,
        "kicker": KICKER_NAME,
        "kicker_season": key,
        "team": team,
        "attempts": int(best["attempts"]),
        "made": int(best["made"]),
        "observed_rate": round(float(best["observed_rate"]), 4),
        "league_p45_pct": league_pct,
        "posterior_p45_pct": round(float(shrunk.mean() * 100), 2),
        "posterior_gain_pp": posterior_pp,
        "one_sigma_pp": one_sigma_pp,
        "sigma_kicker": round(float(sigma.mean()), 4),
    }


def denver_prior_posterior() -> dict:
    """Figure 19 — the same picture for a defence's hands.

    Denver's 2024 defence caught 13 of the 17 throws FTN charted as
    interceptable. The league surface, scored on those same seventeen throws,
    says 49.8%. The model lands at 55.2% — it moved, and it moved a fifth of
    the way, because seventeen throws is seventeen throws.
    """
    from nfl_simulator.dropped_picks import DroppedPickModel, worthy_throw_frame
    from nfl_simulator.render import _simulation_context
    from nfl_simulator.teams import team_colors, team_logo

    context = _simulation_context()
    model = DroppedPickModel.from_posterior(
        OUTPUTS / "trace_dropped_pick.nc", OUTPUTS / "dropped_pick_summary.json"
    )
    worthy = worthy_throw_frame(context["pbp"], context["ftn"])
    plays = worthy.filter(pl.col("defence_season") == DENVER_KEY).to_dicts()

    attempts = check("denver_worthy_throws", len(plays))
    caught = check("denver_worthy_caught", int(sum(play["interception"] for play in plays)))
    # Per posterior draw, the defence-season's rate over its **own** throws —
    # the same quantity figure 15 puts a dot on, kept as a distribution here
    # rather than collapsed to its mean.
    with_denver = np.array([model.catch_probability(DENVER_KEY, play) for play in plays]).mean(0)
    league = np.array([model.catch_probability(None, play) for play in plays]).mean(0)

    posterior_pct = check("denver_posterior_pct", round(float(with_denver.mean() * 100), 1))
    league_pct = check("denver_league_pct", round(float(league.mean() * 100), 1))

    # the maintainer 2026-08-31: navy reads like the league grey — use the orange.
    _navy, primary = team_colors("DEN", 2024)
    fig, ax = _prior_posterior_axes(
        "Modeling Denver's interception catch rate",
        [
            f"Denver's 2024 defense caught {caught} of the {attempts} throws the charters "
            "called interceptable.",
            "Both curves are the rate the model gives those same seventeen throws.",
        ],
    )

    both = np.concatenate([league, with_denver])
    grid = np.linspace(np.quantile(both, 0.002) - 0.01, np.quantile(both, 0.998) + 0.01, 512)
    league_curve, denver_curve = _density(league, grid), _density(with_denver, grid)

    ax.fill_between(grid * 100, 0, league_curve, color=PALETTE["anchor"], alpha=0.22, linewidth=0)
    ax.plot(grid * 100, league_curve, color=PALETTE["anchor"], linewidth=1.6)
    ax.fill_between(grid * 100, 0, denver_curve, color=primary, alpha=0.32, linewidth=0)
    ax.plot(grid * 100, denver_curve, color=primary, linewidth=2.2)

    observed = caught / attempts * 100
    ax.axvline(observed, color=PALETTE["bad"], linewidth=1.6, linestyle=(0, (4, 2)))
    # The grid is trimmed to the curves' own support, and 76.5% is outside it —
    # which is the finding. The axis is widened to reach the line rather than
    # the grid, so the gap between what happened and what the model believes is
    # drawn to scale instead of being cropped out of the picture.
    ax.set_xlim(grid.min() * 100, max(grid.max() * 100, observed + 3.5))
    ax.annotate(
        f"what actually happened\n{caught} of {attempts} = {observed:.1f}%",
        xy=(observed, 0.95),
        xytext=(-7, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=9,
        color=PALETTE["bad"],
    )
    # Above the curves, for the reason figure 18 gives: the two overlap where
    # this label belongs, and the row below the axis belongs to the x-label.
    ax.annotate(
        f"league interception-worthy catch rate: {league_pct:.1f}%",
        xy=(league_pct, 1.15),
        xytext=(-5, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=8.5,
        color=PALETTE["text_muted"],
    )
    ax.annotate(
        f"Denver: {posterior_pct:.1f}%",
        xy=(posterior_pct, denver_curve.max()),
        xytext=(0, 16),
        textcoords="offset points",
        ha="center",
        fontsize=10.5,
        fontweight="bold",
        color=primary,
    )
    _draw_logo(ax, team_logo("DEN", 2024), 0.60, 0.30)

    ax.set_xlabel(
        "rate the defense finishes an interceptable throw (%)",
        fontsize=9,
        color=PALETTE["text_muted"],
    )
    ax.set_ylim(0, 1.18)
    ax.annotate(
        "The model moves a fifth of the way to 76% and stops. Seventeen throws is "
        "not enough evidence to move it further.",
        xy=(0, 0),
        xycoords="axes fraction",
        xytext=(0, -42),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8,
        color=PALETTE["text_muted"],
    )
    name = finalize(fig, FIGURE_DIR / "19_denver_prior_posterior.png").name
    return {
        "file": name,
        "attempts": attempts,
        "caught": caught,
        "observed_pct": round(observed, 1),
        "league_pct": league_pct,
        "posterior_pct": posterior_pct,
    }


# --------------------------------------------------------------------------
# figure 20 — how much of it is skill
# --------------------------------------------------------------------------


def persistent_share() -> dict:
    """Figure 20 — the share of each near-coin that is somebody's persistent skill.

    **The two bars are two different statistics and the figure says so.** The
    dropped pick's 1.4% is a share of per-throw *variance* (document 48 via
    document 52 §2). The fumble's 1.1% is a shrinkage weight `w` (document 05
    §3) — how much of the information about a team's true recovery rate its own
    record carries. Both answer "how much of this is the team rather than the
    bounce?" on the same scale, and neither is the other's number; the axis
    label and the footer both name the difference rather than letting a reader
    assume one statistic drawn twice.
    """
    rows = [
        {
            "label": "Dropped pick",
            "detail": "share of per-throw variance the\ndefense's persistent skill explains",
            "value": DROPPED_PICK_SHARE,
            "source": "document 48 · 52 §2",
        },
        {
            "label": "Receiver drops",
            "detail": "share of per-target variance the\ncorps' persistent skill explains",
            "value": RECEIVER_DROP_SHARE,
            "source": "document 57 §1b",
        },
        {
            "label": "Fumble recovery",
            "detail": "shrinkage weight — how much of its own\nrecord a team-season keeps",
            "value": FUMBLE_SHRINKAGE_W,
            "source": "document 05 §3",
        },
    ]
    check("dropped_pick_share_pct", round(DROPPED_PICK_SHARE * 100, 1))
    check("fumble_shrinkage_pct", round(FUMBLE_SHRINKAGE_W * 100, 1))
    check("receiver_drop_share_pct", round(RECEIVER_DROP_SHARE * 100, 3))

    # 4.8 rather than the 3.6 a two-row chart looks like it needs. `new_figure`
    # gives the title block a fixed 13% of the figure's height, so shrinking the
    # canvas to fit the bars printed the subtitle through the title's rule. The
    # bars are made to fill the space instead.
    # the maintainer, round 5. "Almost none of it is skill" was read as a claim about the
    # *play*, and it is not: forcing an interceptable throw is real, persistent
    # defensive skill and the simulator never touches it. What these bars
    # measure is the **finish** — whether the forced ball is held — and that is
    # the part the ledger re-flips. The title now says which half is which, and
    # the subtitle names what is kept before it names what is re-priced.
    fig, ax = new_figure(
        8.2,
        5.6,
        title="Finishing the play is mostly random",
        subtitle=[
            "The skill that forced the play is kept exactly as earned. Only its finish "
            "is re-flipped,",
            "and this is how little of that finish is the unit — on two measurements the "
            "labels tell apart.",
        ],
    )

    positions = np.arange(len(rows))
    for position, row in zip(positions, rows, strict=True):
        skill = row["value"] * 100
        ax.barh(position, skill, height=0.9, color=PALETTE["bad"], alpha=0.9, zorder=3)
        # The 2% gap of surface between the two segments is the house rule: two
        # fills that touch read as one shape with a colour change in it.
        ax.barh(
            position,
            100 - skill,
            left=skill + 0.6,
            height=0.9,
            color=PALETTE["anchor"],
            alpha=0.28,
            zorder=3,
        )
        ax.text(
            skill + 2.4,
            position,
            f"{skill:.2g}%  the team",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=PALETTE["bad"],
        )
        ax.text(
            99,
            position,
            f"{100 - skill:.4g}%  everything else",
            va="center",
            ha="right",
            fontsize=9.5,
            color=PALETTE["text_muted"],
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(
        [f"{row['label']}\n{row['detail']}" for row in rows], fontsize=9, linespacing=1.35
    )
    ax.set_xlim(0, 101)
    ax.set_xticks([])
    # Inverted so the dropped pick — the component the article is about — is the
    # top row. `barh` counts up from the bottom, and the reading order of a
    # two-row chart is the order the subtitle names them in.
    ax.set_ylim(len(rows) - 0.42, -0.58)
    for side in ("bottom", "left"):
        ax.spines[side].set_visible(False)

    ax.annotate(
        "The drop rows are one statistic (a variance share) and the fumble row another "
        "(a shrinkage weight); both answer\nhow much of this is the team, and every "
        "answer is about one percent or less.",
        xy=(0, 0),
        xycoords="axes fraction",
        xytext=(0, -30),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8,
        color=PALETTE["text_muted"],
    )
    _align_title_with_ticks(fig, ax)
    name = finalize(fig, FIGURE_DIR / "20_persistent_share.png").name
    return {"file": name, "rows": rows}


# --------------------------------------------------------------------------
# figure 22 — two defenses, two curves, and the draw that picks between them
# --------------------------------------------------------------------------


def _draw_ticks(ax, values: np.ndarray, colour: str, y: float, *, n: int = 12) -> None:
    """A few of the draws a replay actually takes, as ticks along a curve's foot.

    Evenly spaced *quantiles* rather than a random subsample: the point is
    "this is the spread a replay samples from", and a random twelve of 8,000
    draws would land wherever the seed put them and read as noise.

    ``y`` is an **axes fraction** and the ticks are drawn inside the axes on a
    blended transform. The first render hung them below the axis, where two rows
    of twelve printed straight through the tick labels and the x-axis title.
    """
    from matplotlib.transforms import blended_transform_factory

    picks = np.quantile(values, np.linspace(0.06, 0.94, n))
    ax.plot(
        picks * 100,
        np.full(n, y),
        marker="|",
        markersize=8,
        markeredgewidth=1.4,
        linestyle="none",
        color=colour,
        alpha=0.9,
        transform=blended_transform_factory(ax.transData, ax.transAxes),
        zorder=6,
    )


def defence_sampling() -> dict:
    """Figure 22 — every replay draws from the *unit's own* curve.

    Figure 19 shows one defence-season moving off the league. This shows two of
    them at once, at opposite ends, because the sentence the article needs is
    not "the model shrinks" but **"a strong unit keeps its edge in every
    re-run"** — which is a claim about what each of the 200 posterior draws is
    drawn from, and is invisible on a figure holding one unit.

    Denver 2024 caught 13 of 17. The Jets' 2025 defence caught none of 9. The
    model puts them 12 percentage points apart and leaves them there, so no
    replay of a Denver game prices its hands at the Jets' rate.
    """
    from nfl_simulator.dropped_picks import DroppedPickModel, worthy_throw_frame
    from nfl_simulator.render import _simulation_context
    from nfl_simulator.teams import team_colors, team_logo

    context = _simulation_context()
    model = DroppedPickModel.from_posterior(
        OUTPUTS / "trace_dropped_pick.nc", OUTPUTS / "dropped_pick_summary.json"
    )
    worthy = worthy_throw_frame(context["pbp"], context["ftn"])

    units = []
    for key in SAMPLING_KEYS:
        plays = worthy.filter(pl.col("defence_season") == key).to_dicts()
        season, team = key.split("|")
        curve = np.array([model.catch_probability(key, play) for play in plays]).mean(0)
        units.append(
            {
                "key": key,
                "team": team,
                "season": int(season),
                "n": len(plays),
                "caught": int(sum(play["interception"] for play in plays)),
                "curve": curve,
                "mean": float(curve.mean()),
            }
        )
    league_curve = np.array(
        [
            model.catch_probability(None, play)
            for play in worthy.filter(pl.col("defence_season") == SAMPLING_KEYS[0]).to_dicts()
        ]
    ).mean(0)

    check("denver_worthy_throws", units[0]["n"])
    check("denver_worthy_caught", units[0]["caught"])
    check("nyj_worthy_throws", units[1]["n"])
    check("nyj_worthy_caught", units[1]["caught"])
    check("denver_posterior_pct", round(units[0]["mean"] * 100, 1))
    check("nyj_posterior_pct", round(units[1]["mean"] * 100, 1))

    fig, ax = _prior_posterior_axes(
        "Every replay draws from the unit's own curve",
        [
            f"{units[0]['team']} {units[0]['season']} caught {units[0]['caught']} of "
            f"{units[0]['n']} interceptable throws; {units[1]['team']} {units[1]['season']} "
            f"caught {units[1]['caught']} of {units[1]['n']}.",
            "The ticks are draws a replay takes. A strong unit keeps its edge in every "
            "re-run — it is never re-priced at the league's hands.",
        ],
    )

    both = np.concatenate([unit["curve"] for unit in units] + [league_curve])
    grid = np.linspace(np.quantile(both, 0.002) - 0.02, np.quantile(both, 0.998) + 0.02, 512)

    league_density = _density(league_curve, grid)
    ax.plot(grid * 100, league_density, color=PALETTE["text_muted"], linewidth=1.2, linestyle=":")
    ax.annotate(
        f"the league's hands: {league_curve.mean() * 100:.1f}%",
        xy=(league_curve.mean() * 100, 1.15),
        xytext=(0, 0),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=8.5,
        color=PALETTE["text_muted"],
    )

    # Two tick rows, one per unit, at the foot of the axes rather than under it,
    # so twelve ticks never print through the tick labels. Each unit's mark sits
    # **over its own curve** — the first render placed them by list order, which
    # put Denver's badge above the Jets' curve and the Jets' above Denver's, the
    # one mistake this figure cannot afford to make.
    for offset, unit in enumerate(units):
        primary, _secondary = team_colors(unit["team"], unit["season"])
        density = _density(unit["curve"], grid)
        ax.fill_between(grid * 100, 0, density, color=primary, alpha=0.30, linewidth=0)
        ax.plot(grid * 100, density, color=primary, linewidth=2.2)
        ax.annotate(
            f"{unit['team']}: {unit['mean'] * 100:.1f}%",
            xy=(unit["mean"] * 100, density.max()),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            fontsize=10.5,
            fontweight="bold",
            color=primary,
        )
        _draw_ticks(ax, unit["curve"], primary, 0.045 + 0.052 * offset)
        unit["density_peak"] = float(density.max())

    ax.set_xlim(grid.min() * 100, grid.max() * 100)
    low, high = ax.get_xlim()
    midpoint = float(np.mean([unit["mean"] for unit in units]))
    for unit in units:
        # Data value -> axes fraction, because `_draw_logo` anchors in axes
        # coordinates. Half-way up the unit's own curve, pushed to whichever
        # shoulder faces away from the other unit so the two marks never meet
        # where the curves cross.
        centre = (unit["mean"] * 100 - low) / (high - low)
        outward = 0.11 if unit["mean"] >= midpoint else -0.11
        _draw_logo(
            ax,
            team_logo(unit["team"], unit["season"]),
            min(0.95, max(0.05, centre + outward)),
            0.62,
        )

    ax.set_xlabel(
        "rate the defense catches an interceptable throw (%)",
        fontsize=9,
        color=PALETTE["text_muted"],
    )
    ax.set_ylim(0, 1.18)
    ax.annotate(
        "Neither unit is priced at the league rate, and neither is priced at its own raw "
        "record. Each replay takes one\nvalue from its own curve, so the gap between two "
        "defenses survives all 160,000 of them.",
        xy=(0, 0),
        xycoords="axes fraction",
        xytext=(0, -52),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8,
        color=PALETTE["text_muted"],
    )
    name = finalize(fig, FIGURE_DIR / "22_defense_sampling.png").name
    return {
        "file": name,
        "units": [{k: unit[k] for k in ("key", "n", "caught", "mean")} for unit in units],
        "league_pct": round(float(league_curve.mean() * 100), 1),
    }


# --------------------------------------------------------------------------
# figure 23 — the offense's mirror of figure 18
# --------------------------------------------------------------------------


def corps_prior_posterior() -> dict:
    """Figure 23 — one receiving corps' drop rate, raw and shrunk.

    Figure 18 does this for a kicker and figure 19 for a defence; the offence's
    hands had no picture at all, which left the article's "each event is priced
    at the unit's own shrunk rate" resting on two examples that were both
    somebody else's mistake. **Chosen by a rule** — the corps-season with the
    largest gap between its raw rate and its posterior, among those with at
    least `CORPS_FLOOR` catchable targets — so a refit that moved the answer
    would move the figure rather than draw a different corps under this caption.
    """
    from nfl_simulator.receiver_drops import ReceiverDropModel, catchable_target_frame
    from nfl_simulator.render import _simulation_context
    from nfl_simulator.teams import team_colors, team_logo

    context = _simulation_context()
    model = ReceiverDropModel.from_posterior(
        OUTPUTS / "trace_receiver_drop.nc", OUTPUTS / "receiver_drop_summary.json"
    )
    targets = catchable_target_frame(context["pbp"], context["ftn"])
    plays = targets.filter(pl.col("entity_season") == CORPS_KEY).to_dicts()
    season, team = CORPS_KEY.split("|")

    n = check("corps_targets", len(plays))
    drops = check("corps_drops", int(sum(bool(play["is_drop"]) for play in plays)))
    posterior = np.array([model.drop_probability(CORPS_KEY, play) for play in plays]).mean(0)
    league = np.array([model.drop_probability(None, play) for play in plays]).mean(0)

    raw_pct = check("corps_raw_pct", round(drops / n * 100, 2))
    posterior_pct = check("corps_posterior_pct", round(float(posterior.mean() * 100), 2))
    league_pct = check("corps_league_pct", round(float(league.mean() * 100), 2))

    primary, _secondary = team_colors(team, int(season))
    # Titled like figure 18, because it is figure 18's sentence about the other
    # side of the ball and the two are read as a pair.
    fig, ax = _prior_posterior_axes(
        f"{team}'s {season} drop rate vs the league",
        [
            f"{team}'s {season} receiving corps dropped {drops} of {n} balls the charters "
            f"called catchable — {raw_pct:.2f}%, raw and unshrunk.",
            "Grey is the league's rate on those same targets; the colored curve is where "
            "the model puts this corps.",
        ],
    )

    both = np.concatenate([league, posterior])
    grid = np.linspace(np.quantile(both, 0.002) - 0.004, np.quantile(both, 0.998) + 0.004, 512)
    league_density, corps_density = _density(league, grid), _density(posterior, grid)

    ax.fill_between(grid * 100, 0, league_density, color=PALETTE["anchor"], alpha=0.22, linewidth=0)
    ax.plot(grid * 100, league_density, color=PALETTE["anchor"], linewidth=1.6)
    ax.fill_between(grid * 100, 0, corps_density, color=primary, alpha=0.32, linewidth=0)
    ax.plot(grid * 100, corps_density, color=primary, linewidth=2.2)

    # The raw rate is off the right end of both curves, which is the finding.
    # The axis is widened to reach it, for figure 19's reason: cropping it out
    # would hide the distance the figure exists to draw.
    ax.axvline(raw_pct, color=PALETTE["bad"], linewidth=1.6, linestyle=(0, (4, 2)))
    ax.set_xlim(grid.min() * 100, max(grid.max() * 100, raw_pct + 0.7))
    ax.annotate(
        f"what actually happened\n{drops} of {n} = {raw_pct:.2f}%",
        xy=(raw_pct, 0.95),
        xytext=(-7, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=9,
        color=PALETTE["bad"],
    )
    ax.annotate(
        f"the league: {league_pct:.2f}%",
        xy=(league_pct, 1.15),
        xytext=(0, 0),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=8.5,
        color=PALETTE["text_muted"],
    )
    ax.annotate(
        f"{team}: {posterior_pct:.2f}%",
        xy=(posterior_pct, corps_density.max()),
        xytext=(0, 16),
        textcoords="offset points",
        ha="center",
        fontsize=10.5,
        fontweight="bold",
        color=primary,
    )
    _draw_logo(ax, team_logo(team, int(season)), 0.80, 0.30)

    ax.set_xlabel("drop rate on catchable targets (%)", fontsize=9, color=PALETTE["text_muted"])
    ax.set_ylim(0, 1.18)
    ax.annotate(
        "The model keeps about a third of the record and gives the rest back to the league. "
        'The caveat travels with it:\n"catchable" is a human charter\'s judgment, and the '
        "article's section 6 says why that is open rather than settled.",
        xy=(0, 0),
        xycoords="axes fraction",
        xytext=(0, -52),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8,
        color=PALETTE["text_muted"],
    )
    name = finalize(fig, FIGURE_DIR / "23_corps_prior_posterior.png").name
    return {
        "file": name,
        "corps": CORPS_KEY,
        "targets": n,
        "drops": drops,
        "raw_pct": raw_pct,
        "posterior_pct": posterior_pct,
        "league_pct": league_pct,
    }


# --------------------------------------------------------------------------
# figure 21 — where 160,000 comes from
# --------------------------------------------------------------------------


def bootstrap_buildup(walkthrough: dict) -> dict:
    """Figure 21 — the two layers, built up one panel at a time.

    Panel (c) is not a redraw. `SimulationResult.margin_draws` is
    ``margins.ravel()`` of the shipped ``(200, 800)`` bootstrap, so reshaping it
    recovers the two layers exactly and panel (c) is the same 160,000 numbers
    the product's own histogram bins. A figure that re-simulated the game to
    illustrate the simulation could disagree with it and no reader could tell.
    """
    from nfl_simulator.render import COIN_DRAWS, POSTERIOR_DRAWS

    draws = walkthrough["editions"]["full"]["margin_draws"]
    grid = draws.reshape(POSTERIOR_DRAWS, COIN_DRAWS)

    # `simulator._bootstrap` line 627: `dtw_per_draw = (margins > 0).mean(axis=1)`
    # and `dtw_home` is its mean. Every row has the same 800 coin draws, so the
    # share over the flattened grid is that number exactly — which is what makes
    # this a check on the reshape rather than a restatement of it.
    share = check("den_was_dtw_full", round(float((draws > 0).mean()), 4))

    # **The middle panel draws ten outlines, not one pooled histogram.**
    # Document 65's finding W-7: round 3 passed `grid[:10].ravel()` to a single
    # `ax.hist`, which flattens the ten draws into 8,000 numbers and bins them
    # together. The caption promised "ten draws overlaid, visibly disagreeing"
    # and the panel drew one smooth histogram with nothing to disagree — in the
    # one figure whose entire job is to make layer one visible. Each draw now
    # gets its own outline, and the disagreement between them *is* layer one.
    panels = [
        (grid[:1], "1 × 800", "one draw of every probability, flipped 800 times"),
        (grid[:10], "10 × 800", "ten draws, each outlined — they disagree"),
        (draws.reshape(1, -1), "200 × 800 = 160,000", "all of them — the figure the product ships"),
    ]
    edges = np.histogram_bin_edges(draws, bins=34)

    fig = plt.figure(figsize=(9.6, 5.0))
    fig.patch.set_facecolor(PALETTE["bg"])
    draw_title_block(
        title_axes(fig, height_frac=0.15),
        "Where 160,000 comes from",
        [
            "DEN at WAS, week 13 of 2025, Full edition. Each panel adds posterior draws; "
            "the axis never moves.",
            "Layer 1 draws the probabilities, layer 2 flips the coins.",
        ],
        title_size=17,
    )

    axes = []
    # The panels sit low enough to leave two lines of heading above each one.
    # The first render put the caption 26 points above the axes, which on a
    # 4.4-inch figure is where the title block's second line already was.
    for index, (rows_, heading, caption) in enumerate(panels):
        ax = fig.add_axes([0.07 + index * 0.315, 0.22, 0.265, 0.44])
        ax.set_facecolor(PALETTE["bg"])
        colour = PALETTE["anchor"] if index < 2 else PALETTE["bad"]
        values = rows_.ravel()
        if index == 1:
            # One outline per posterior draw. `histtype="step"` so ten curves
            # can sit on one axis without ten fills hiding each other, and each
            # is scaled by its own row length so the panel's y axis stays the
            # "% of simulations" every other panel uses.
            for row in rows_:
                ax.hist(
                    row,
                    bins=edges,
                    weights=np.full(row.size, 100.0 / row.size),
                    histtype="step",
                    linewidth=1.0,
                    color=colour,
                    alpha=0.65,
                )
        else:
            ax.hist(
                values,
                bins=edges,
                weights=np.full(values.size, 100.0 / values.size),
                color=colour,
                alpha=0.55 if index < 2 else 0.8,
            )
        ax.axvline(0.0, color=PALETTE["text"], linewidth=1.0, linestyle=":")
        ax.set_title(heading, fontsize=11, fontweight="bold", color=PALETTE["text"], pad=22)
        ax.annotate(
            caption,
            xy=(0.5, 1.0),
            xycoords="axes fraction",
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=PALETTE["text_muted"],
        )
        # Each panel's own share, so "it is already there by the second panel"
        # is a number a reader can check rather than a claim the caption makes.
        ax.annotate(
            f"WAS {(values > 0).mean():.3f}",
            xy=(0.97, 0.93),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=9,
            fontweight="bold",
            color=colour if index == 2 else PALETTE["text_muted"],
        )
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(PALETTE["spine"])
        ax.tick_params(colors=PALETTE["text_muted"], labelsize=8)
        ax.set_xlim(edges[0], edges[-1])
        if index:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("% of simulations", fontsize=8.5, color=PALETTE["text_muted"])
        axes.append(ax)

    top = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, top)

    fig.text(
        0.5,
        0.105,
        "deserved margin, home-signed  (← DEN wins by · WAS wins by →)",
        ha="center",
        fontsize=9,
        color=PALETTE["text_muted"],
    )
    fig.text(
        0.07,
        0.035,
        f"Washington's deserved-win share is the area right of zero, printed on each "
        f"panel. All 200 draws give {share:.4f}, which is the number the product reports.",
        ha="left",
        fontsize=8,
        color=PALETTE["text_muted"],
    )
    name = finalize(fig, FIGURE_DIR / "21_bootstrap_buildup.png").name
    return {"file": name, "dtw_home": share, "shape": [POSTERIOR_DRAWS, COIN_DRAWS]}


# --------------------------------------------------------------------------


def copy_game_figures() -> list[str]:
    """Render each example game and copy the one figure the article uses.

    The article does not draw its own version of a product figure. If the
    waterfall in the write-up disagreed with the waterfall in the product, one
    of them would be wrong and a reader could not tell which.
    """
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


# What the article calls the dashed rule on the distribution. `plots.py` hands
# `anchor_label` the word "Deserved", and round 2's article says "Expected
# margin" everywhere — including under figure 16, which is drawn here. The
# rename happens on the finished figure rather than in `plots.py`: the product's
# own wording is not this article's to change, and one word is a caller's
# business.
DESERVED_LABEL = "Expected:"
ARTICLE_LABEL = "Expected:"


def article_dtw_figure() -> str:
    """`05_...dtw_full.png` — the product's distribution, with one word renamed.

    Everything but the rename is `render.render_game`'s own `dtw` branch,
    reached through the same `replay`-checked verdict, so the histogram is the
    product's and not a redraw of it.
    """
    from nfl_simulator.plots import plot_bootstrap_distribution, verdict_from_row
    from nfl_simulator.render import (
        DTW_FIGURE,
        counterpart_verdict,
        load_sources,
        replay,
        season_of,
    )
    from nfl_simulator.teams import pair_colors, team_logo

    stem, game_id, edition = ARTICLE_DTW
    sources = load_sources()
    row = sources.game_row(game_id, edition=edition)
    schedule = sources.schedule_row(game_id)
    result, _gaps = replay(game_id, row, schedule, edition=edition)
    verdict = verdict_from_row(
        row,
        result.margin_draws,
        schedule,
        edition=edition,
        counterpart=counterpart_verdict(sources, game_id, edition, schedule),
    )
    season = season_of(game_id)
    colours = pair_colors(verdict.home_team, verdict.away_team, season)
    logos = {team: team_logo(team, season) for team in (verdict.home_team, verdict.away_team)}
    fig, _ax = plot_bootstrap_distribution(
        verdict, colors=colours, logos=logos, coverage=False, **DTW_FIGURE
    )

    renamed = 0
    for artist in fig.findobj(matplotlib.text.Text):
        words = artist.get_text()
        if words.startswith(DESERVED_LABEL):
            artist.set_text(words.replace(DESERVED_LABEL, ARTICLE_LABEL, 1))
            renamed += 1
    if renamed != 1:
        raise AssertionError(
            f"expected exactly one {DESERVED_LABEL!r} label on the distribution, found {renamed}. "
            "`plots.anchor_label` has moved; re-read it before renaming anything."
        )
    return finalize(fig, FIGURE_DIR / f"{stem}.png", edition=edition).name


# --------------------------------------------------------------------------
# what ships
# --------------------------------------------------------------------------

CAPTIONS = {
    "04_lac_hou_2024_wk19_ledger_full.png": (
        "The luck ledger for LAC at HOU, wild-card round 2024 — the article's drops "
        "showcase, and the corpus's largest swing at 19.03 points."
    ),
    "05_den_was_2025_wk13_dtw_full.png": (
        "Deserved margin across 160,000 re-simulations of DEN at WAS, week 13 of 2025, "
        "with the actual margin and the expected margin marked."
    ),
    "06_den_was_2025_wk13_ledger_full.png": (
        "The same game's luck ledger, row by row — the walk-through figure section 9 reads."
    ),
    "07_eda_fumble_recovery.png": (
        "Each club's own-recovery rate in one season against the next, 2016–2025. "
        "If recovery were a skill the dots would line up; they do not."
    ),
    "09_shrinkage.png": (
        "Five team-seasons' fumble own-recovery rate and where the beta-binomial puts it. "
        "Buffalo's 10 of 12 in 2024 shrinks to 48.0%."
    ),
    "11_flip_distribution.png": (
        "Deserved-win probability for the home team across all 1,139 games, 2022–2025, "
        "with the too-close-to-call band shaded."
    ),
    "13_epa_to_points.png": (
        "Final margin against EPA differential, 2,761 games — the 0.8389 points-per-EPA "
        "conversion the whole article quotes, r² = 0.992."
    ),
    "14_refused_floors.png": (
        "Three components that passed the mechanism test and failed on size, each against "
        "the floor committed before its effect was computed."
    ),
    "15_defense_shrinkage.png": (
        "Five defense-seasons' interception catch rate on interceptable throws, and where "
        "the model puts each — a logistic regression, not the beta-binomial figure 3 draws."
    ),
    "16_den_was_with_without.png": (
        "One game with and without the hands-on-the-ball rows, on one axis. Denver's share "
        "falls from 86% to 59%; the verdict did not flip, it stopped being a verdict."
    ),
    "17_fumble_retention_bars.png": (
        "How often the fumbling team keeps the ball, by class of play, on the fitted "
        "baseline the simulator prices with. A botched snap on a run is kept 76.9% of the "
        "time and a fumble on a run 46.1% — a coin, but not the same coin."
    ),
    "18_kicker_prior_posterior.png": (
        "One kicker-season priced: E. Pi\u00f1eiro's 2025 record of 31 of 32 is raw, and "
        "the colored curve is his shrunk posterior — both read as make probability from "
        "45 yards."
    ),
    "19_denver_prior_posterior.png": (
        "The same picture for a defense's hands: Denver's 2024 defense caught 13 of 17 "
        "interceptable throws, and the model moves from the league's 49.8% to 55.2%."
    ),
    "20_persistent_share.png": (
        "How much of the *finish* of a dropped pick, a receiver drop and a fumble recovery "
        "is the unit rather than the bounce — one percent or less of each, on two "
        "measurements the figure names apart. The skill that forced the throw is untouched."
    ),
    "21_bootstrap_buildup.png": (
        "Where 160,000 comes from: one posterior draw's 800 replays, then ten draws each "
        "outlined separately, then all 200 — the histogram the product ships."
    ),
    "22_defense_sampling.png": (
        "Two defense-seasons' catch-rate curves on one axis — Denver 2024's 13 of 17 and "
        "the Jets' 2025 none of 9 — with the draws a replay takes ticked beneath each. "
        "A strong unit keeps its edge in every re-run."
    ),
    "23_corps_prior_posterior.png": (
        "The offense's mirror of the kicker figure: Jacksonville's 2025 receiving corps "
        "dropped 40 of 420 catchable balls, 9.52%, and the model puts them at 6.34% "
        "against a league rate of 5.14% on those same targets."
    ),
    "formula_01_rule.png": "The neutralization rule, term by term.",
    "formula_02_points.png": "EPA to points of margin, and the deserve-to-win share.",
    "formula_03_fg.png": "The field-goal make model, every covariate labeled — elevation included since v1.4.",
    "formula_04_betabinomial.png": "The beta-binomial hierarchy that shrinks a unit's rate.",
    "formula_05_trust.png": "The trust dial, and what sets it.",
}

ARTICLE = paths.REPO_ROOT / "docs" / "writeup" / "community-writeup.md"


def write_captions() -> Path:
    """One line per shipped image, for worker 2 to read after the merge."""
    lines = ["# Figure captions — community write-up, round 2", ""]
    lines += [
        "*Written by `research/80_writeup_figures.py`. Every image in "
        "`docs/writeup/figures/` is listed; every one was opened before its line "
        "was written.*",
        "",
    ]
    for name in sorted(CAPTIONS):
        lines.append(f"- **`{name}`** — {CAPTIONS[name]}")
    target = FIGURE_DIR / "CAPTIONS.md"
    target.write_text("\n".join(lines) + "\n")
    return target


def audit() -> None:
    """The rules a committed image has to obey, checked rather than trusted.

    **`CAPTIONS` is the manifest, and round 3 moved it there from the article.**
    Round 2 made the article the source of truth for which images ship, which
    worked while one worker wrote both. Round 3 splits them: worker 2 writes the
    article against the filenames in the handoff, on a branch this one never
    sees, so a figure this script has just drawn is legitimately unlinked until
    those two branches meet. Enforcing "every image is linked" here would make
    that ordinary state a crash.

    So the direction that can still be fatal is the one that is always a defect:
    **linked but absent** is a broken image in a published article, whoever
    wrote which half. Present-but-unlinked is reported and does not stop the
    run, because on this branch it is the expected reading.
    """
    files = sorted(path.name for path in FIGURE_DIR.glob("*.png"))
    heavy = [name for name in files if (FIGURE_DIR / name).stat().st_size > SIZE_LIMIT]
    if heavy:
        raise AssertionError(f"over {SIZE_LIMIT} bytes: {heavy}")
    if len(files) > MAX_FILES:
        raise AssertionError(f"{len(files)} images, and the article's budget is {MAX_FILES}")
    if set(CAPTIONS) != set(files):
        raise AssertionError(
            f"CAPTIONS covers {sorted(set(CAPTIONS) ^ set(files))} differently from the directory"
        )

    linked = set(re.findall(r"\(figures/([^)]+\.png)\)", ARTICLE.read_text()))
    missing = sorted(linked - set(files))
    if missing:
        raise AssertionError(f"the article links images that do not exist: {missing}")
    unlinked = sorted(set(files) - linked)

    print(
        f"\n{len(files)} images in {FIGURE_DIR.relative_to(paths.REPO_ROOT)}, "
        "none over 500 KB, every one captioned"
    )
    if unlinked:
        print(
            f"  not yet linked from this branch's article ({len(unlinked)}): "
            f"{', '.join(unlinked)}\n"
            "  Expected while worker 2's prose is on its own branch; check again after "
            "the merge."
        )


def sweep() -> list[str]:
    """Delete images this script no longer draws.

    Keyed on `CAPTIONS` rather than on the article's links, for the reason
    :func:`audit` gives: on this branch the article is a round behind the
    figures by design, and sweeping against it would delete each figure the
    moment it was drawn.
    """
    removed = []
    for path_ in sorted(FIGURE_DIR.glob("*.png")):
        if path_.name not in CAPTIONS:
            path_.unlink()
            removed.append(path_.name)
    return removed


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
    rates = fumble_class_rates()
    print_fumble_class_rates(rates)
    walkthrough = walkthrough_ledger()
    print_walkthrough_ledger(walkthrough)
    return {
        "summary": summary,
        "worked_fumble": event,
        "fumble_class_rates": rates,
        "walkthrough": walkthrough,
    }


def main() -> None:
    apply_base_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    computed = summary_only()

    print("\ngame figures ...")
    for name in copy_game_figures():
        print(f"  {name}")
    print(f"  {article_dtw_figure()}")

    print("explanatory figures ...")
    print("  07 fumble recovery  ", fumble_year_over_year())
    print("  09 shrinkage        ", shrinkage())
    print("  11 flip distribution", flip_distribution(computed["summary"]))
    print("  13 epa to points    ", epa_to_points())
    print("  14 refused floors   ", refused_floors())
    print("  15 defense shrinkage", defence_shrinkage())
    print("  den 2025 follow-up  ", denver_2025_followup())
    print("  16 with and without ", den_was_with_without(computed["walkthrough"]))
    print("  17 fumble retention ", fumble_retention_bars())
    print("  18 kicker curves    ", kicker_prior_posterior())
    print("  19 denver curves    ", denver_prior_posterior())
    print("  20 persistent share ", persistent_share())
    print("  21 bootstrap buildup", bootstrap_buildup(computed["walkthrough"]))
    print("  22 defense sampling ", defence_sampling())
    print("  23 corps curves     ", corps_prior_posterior())

    print("formula plates ...")
    for name in formula_plates():
        print(f"  {name}")

    withdrawn = sweep()
    if withdrawn:
        print(f"\nwithdrawn from the article, deleted: {', '.join(withdrawn)}")
    print(f"captions: {write_captions().relative_to(paths.REPO_ROOT)}")
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
