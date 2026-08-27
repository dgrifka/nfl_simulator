"""One game in, three PNGs out.

This is the product layer's front door. Everything under it — the simulator,
the ledger, the figures — already exists and is already tested; what is here is
the assembly: find the game's committed numbers, fetch the colours and the
marks, put plain words on the events, and write three files named the way the
baseball simulator names its own.

**Nothing here recomputes a published number.** The deserved margin, the DTW%
and the interval are read from `dtw_games_v13.parquet`; the events are read
from `dtw_ledger_v13.parquet`; the slope is read from `model_metadata_v13.json`.
The one thing that *is* recomputed is the bootstrap's margin draws, because the
shipped summary does not keep them — and it is recomputed under v1.3's exact
settings and checked against the summary before a pixel is drawn, exactly as
`research/54_bootstrap_figures.py` does. A replay that disagrees is a stop, not
a footnote.

The schedule contributes presentation facts only: the two scores, the date,
and whether the game went to overtime. None of them can move the adjudication.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import polars as pl

from nfl_simulator import paths
from nfl_simulator.ledger import with_actual
from nfl_simulator.plots import (
    GameVerdict,
    OvertimeToss,
    attach_overtime_sidebar,
    plot_bootstrap_distribution,
    plot_game_card,
    plot_luck_ledger,
    plot_luck_ledger_card,
    plot_team_points_distribution,
    verdict_from_row,
)
from nfl_simulator.style import finalize
from nfl_simulator.teams import pair_colors, team_logo, team_name

GAMES_ARTIFACT = "dtw_games_v13.parquet"
LEDGER_ARTIFACT = "dtw_ledger_v13.parquet"
OVERTIME_ARTIFACT = "26_overtime_games.parquet"
METADATA = "model_metadata_v13.json"

# v1.3's shipped settings, quoted from `research/46_simulator_v13.py` the same
# way drivers 54 and 57 quote them. Changing any of them changes the draws, and
# the replay check below is what says so.
RANDOM_SEED = 20260817
POSTERIOR_DRAWS = 200
COIN_DRAWS = 800
REPLAY_TOLERANCE = 1e-9

# `luck_ledger` is the portrait share image; `waterfall` is the same ledger as
# an article figure. Round 1 shipped the waterfall under the `luck_ledger` name
# and the maintainer needed help reading it — a waterfall is a chart type most people have
# not been taught, which is fine in an article and wrong on a timeline.
SUFFIXES = ("dtw", "luck_ledger", "card", "waterfall")

# Not a fifth share image — an extra, written only when a caller asks for it and
# only when the game actually went to overtime. Round 1's review: document 16's
# six-paragraph panel is the right amount of methodology for an article and far
# too much beside a card somebody scrolls past.
ARTICLE_SUFFIX = "dtw_article"

# The distribution's shipped reading, chosen by looking at the eight variants
# `research/59_dtw_variants.py` renders (document 42 §1). Three-point bins pool
# the reachable margins without smoothing between them; the callout states the
# verdict in a sentence; the arrow says what luck moved and toward whom; and the
# clubs' marks carry identity in place of two coloured swatches.
DTW_FIGURE = {"bin_width": 3.0, "callout": True, "arrow": True, "legend_logos": True}

# Round 4: the share image is the two teams' deserved points rather than the
# margin. Same three-point bins, same callout, same marks in the legend — and no
# luck arrow, because an arrow measures a distance along a margin axis and this
# axis is a scoreboard. The margin histogram is not retired: it is what
# `ARTICLE_SUFFIX` draws, where the overtime panel already lives.
TEAM_POINTS_FIGURE = {"bin_width": 3.0, "callout": True, "legend_logos": True}


# --------------------------------------------------------------------------
# naming
# --------------------------------------------------------------------------


def figure_filename(verdict: GameVerdict, suffix: str) -> str:
    """`"GB_DET_23-31--95-5_dtw.png"` — the baseball simulator's pattern, verbatim.

    Away team first, then home, then the scoreline, then the two shares, then
    what the figure is. The name is the caption for anyone scrolling a folder
    of them, which is why the verdict's own numbers are in it.

    The two shares are rounded **once, together** — the home share is rounded
    and the away share is 100 minus it — so the pair in the filename always sums
    to 100 and always agrees with the headline on the figure.

    A game whose scoreline is not on file is named by its game id instead. A
    filename is not a place to invent a score.
    """
    if verdict.home_score is None or verdict.away_score is None:
        return f"{verdict.game_id}_{suffix}.png"
    home_share = round(verdict.dtw_home * 100)
    return (
        f"{verdict.away_team}_{verdict.home_team}_"
        f"{verdict.away_score:.0f}-{verdict.home_score:.0f}--"
        f"{100 - home_share}-{home_share}_{suffix}.png"
    )


# --------------------------------------------------------------------------
# the rows a waterfall reads
# --------------------------------------------------------------------------


def prepare_rows(
    frame: pl.DataFrame, verdict: GameVerdict, distances: dict | None = None
) -> list[dict]:
    """Ledger rows with the three extra keys :func:`plots.plain_label` can use.

    ``actual`` is recovered from the ledger's own identity when the artifact
    predates the column (see :func:`ledger.with_actual`). ``opponent`` is the
    other team in *this* game, which is what makes "recovered by GB" sayable —
    the ledger records who fumbled, not who ended up with the ball. And
    ``kick_distance`` is the kick's real yardage from the play-by-play, left
    absent rather than guessed when the play is not found: the ledger stores a
    five-yard class, and printing its midpoint as the distance would be making
    a number up.
    """
    distances = distances or {}
    rows = with_actual(frame).to_dicts()
    for row in rows:
        charged = row.get("charged_team")
        row["opponent"] = verdict.away_team if charged == verdict.home_team else verdict.home_team
        row["kick_distance"] = distances.get(row.get("play_id"))
    return rows


# --------------------------------------------------------------------------
# the committed sources
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Sources:
    """Everything on disk a render needs, loaded once for a whole batch."""

    games: pl.DataFrame
    ledger: pl.DataFrame
    schedule: pl.DataFrame
    overtime: pl.DataFrame
    slope: float

    def game_row(self, game_id: str) -> dict:
        rows = self.games.filter(pl.col("game_id") == game_id).to_dicts()
        if not rows:
            raise SystemExit(f"{game_id} is not in {GAMES_ARTIFACT}.")
        return rows[0]

    def schedule_row(self, game_id: str) -> dict:
        rows = self.schedule.filter(pl.col("game_id") == game_id).to_dicts()
        return rows[0] if rows else {}

    def toss(self, verdict: GameVerdict) -> OvertimeToss | None:
        """Document 16's overtime note for this game, or nothing at all."""
        rows = self.overtime.filter(pl.col("game_id") == verdict.game_id).to_dicts()
        if not rows:
            return None
        row = rows[0]
        return OvertimeToss(
            received=verdict.home_team if row["home_received"] else verdict.away_team,
            season=int(str(verdict.game_id)[:4]),
            delta_dtw_home=float(row["delta"]),
        )


@cache
def load_sources() -> Sources:
    """Read the committed artifacts and the schedule. Cached for a batch render."""
    output = paths.RESEARCH_OUTPUT_DIR
    with (output / METADATA).open() as handle:
        slope = float(json.load(handle)["points_per_epa"])
    return Sources(
        games=pl.read_parquet(output / GAMES_ARTIFACT),
        ledger=pl.read_parquet(output / LEDGER_ARTIFACT),
        schedule=pl.read_parquet(paths.SCHEDULE_PATH),
        overtime=pl.read_parquet(output / OVERTIME_ARTIFACT),
        slope=slope,
    )


@cache
def _simulation_context():
    """v1.3's fitted pieces, built once so a batch pays for them once.

    Imported through `research/44_read_side_fix.py` because that is where the
    read-side fix and the refitted field-goal model live, and re-implementing
    either here would be a second copy of the thing document 30 corrected.
    """
    research = Path(__file__).resolve().parents[2] / "research"
    if str(research) not in sys.path:
        sys.path.insert(0, str(research))
    from importlib import import_module

    from nfl_simulator.components import (
        build_game_table,
        fit_fg_baseline,
        fit_fumble_baseline,
        fit_xp_baseline,
    )
    from nfl_simulator.ingest import PBP_SEASONS, load_pbp
    from nfl_simulator.simulator import points_per_epa

    read_side = import_module("44_read_side_fix")
    pbp = load_pbp(PBP_SEASONS, columns=read_side.SIM_COLUMNS)
    fg_model, _ = read_side.load_model("trace_fg_refit.nc", "fg_refit_summary.json")
    return {
        "pbp": pbp,
        "fg_model": fg_model,
        "fumble_baseline": fit_fumble_baseline(pbp),
        "fg_baseline": fit_fg_baseline(pbp),
        "xp_baseline": fit_xp_baseline(pbp),
        "slope": points_per_epa(build_game_table(pbp).drop_nulls("margin")),
    }


def replay(game_id: str, row: dict, schedule: dict | None = None):
    """The game's bootstrap, re-simulated and checked against the summary.

    The shipped parquet keeps the summary, not the 160,000 draws, so a figure
    that shows the distribution has to redraw it. Redrawing it is only safe if
    the redraw lands on the published number — otherwise the histogram belongs
    to a different adjudication than the headline over it.

    ``schedule`` supplies the two scores. Given them the result also carries
    each team's deserved points, split out of the same replay — which is why
    this returns the whole :class:`SimulationResult` rather than the margins
    alone: two figures drawn from two replays of the same coins would be two
    adjudications wearing one headline.
    """
    from nfl_simulator.simulator import simulate_game

    schedule = schedule or {}
    scores = {
        "home_points": schedule.get("home_score"),
        "away_points": schedule.get("away_score"),
    }
    if scores["home_points"] is None or scores["away_points"] is None:
        scores = {}

    context = _simulation_context()
    result = simulate_game(
        context["pbp"].filter(pl.col("game_id") == game_id),
        fumble_baseline=context["fumble_baseline"],
        fg_baseline=context["fg_baseline"],
        xp_baseline=context["xp_baseline"],
        fg_model=context["fg_model"],
        points_per_epa=context["slope"],
        n_posterior_draws=POSTERIOR_DRAWS,
        n_coin_draws=COIN_DRAWS,
        seed=RANDOM_SEED,
        include_blocked=False,
        **scores,
    )
    gaps = {
        "deserved_margin": abs(result.deserved_margin - row["deserved_margin"]),
        "dtw_home": abs(result.dtw_home - row["dtw_home"]),
        "dtw_low": abs(result.dtw_interval[0] - row["dtw_low"]),
        "dtw_high": abs(result.dtw_interval[1] - row["dtw_high"]),
    }
    if max(gaps.values()) > REPLAY_TOLERANCE:
        raise SystemExit(
            f"{game_id} does not replay to its shipped summary ({gaps}). Stop rather than draw it."
        )
    return result, gaps


def kick_distances(game_id: str) -> dict:
    """`play_id -> kick_distance` for the game, from the cached play-by-play.

    Failure is quiet and total: without it the labels fall back to the ledger's
    five-yard class, which is correct, just less specific.
    """
    try:
        plays = _simulation_context()["pbp"].filter(pl.col("game_id") == game_id)
        if "kick_distance" not in plays.columns:
            return {}
        kicks = plays.select("play_id", "kick_distance").drop_nulls("kick_distance")
        return {float(p): float(d) for p, d in kicks.iter_rows()}
    except Exception as error:  # pragma: no cover - the labels degrade, not the render
        print(f"Warning: no kick distances for {game_id}: {error}")
        return {}


# --------------------------------------------------------------------------
# the render
# --------------------------------------------------------------------------


def render_game(game_id: str, out_dir: Path | None = None, *, article: bool = False) -> list[Path]:
    """Write this game's four PNGs and return their paths, in ``SUFFIXES`` order.

    **No share image carries the sidebar.** Document 16's refusal is a fact about
    every one of these figures' ledgers, and every one of them states it — but as
    one muted line, not as six paragraphs of methodology down the side. The
    sidebar also grows the figure, so an overtime game and a regulation one came
    out at two different widths and a timeline crops them differently.

    ``article=True`` adds one more file for an overtime game,
    ``{...}_dtw_article.png``: the distribution with the panel attached. That is
    where the six paragraphs belong.
    """
    out_dir = Path(out_dir) if out_dir is not None else paths.RESEARCH_OUTPUT_DIR
    sources = load_sources()

    row = sources.game_row(game_id)
    schedule = sources.schedule_row(game_id)
    result, _gaps = replay(game_id, row, schedule)
    verdict = verdict_from_row(row, result.margin_draws, schedule)

    ledger = sources.ledger.filter(pl.col("game_id") == game_id).drop("game_id")
    rows = prepare_rows(ledger, verdict, kick_distances(game_id))
    colours = pair_colors(verdict.home_team, verdict.away_team)
    sides = (verdict.home_team, verdict.away_team)
    logos = {team: team_logo(team) for team in sides}
    names = {team: team_name(team) for team in sides}
    toss = sources.toss(verdict) if verdict.went_to_overtime else None

    written = []
    for suffix in SUFFIXES:
        if suffix == "dtw":
            # The scoreline figure when the scoreboard is on file, which it is
            # for every game with a schedule row. A game without one still gets
            # a share image: the margin histogram it got before round 4.
            if result.home_point_draws is not None:
                fig, _ax = plot_team_points_distribution(
                    verdict,
                    result.home_point_draws,
                    result.away_point_draws,
                    colors=colours,
                    logos=logos,
                    **TEAM_POINTS_FIGURE,
                )
            else:
                fig, _ax = plot_bootstrap_distribution(
                    verdict, colors=colours, logos=logos, **DTW_FIGURE
                )
        elif suffix == "luck_ledger":
            fig, _ax = plot_luck_ledger_card(
                verdict,
                rows,
                points_per_epa=sources.slope,
                colors=colours,
                logos=logos,
                names=names,
            )
        elif suffix == "waterfall":
            fig, _ax = plot_luck_ledger(
                verdict, rows, points_per_epa=sources.slope, colors=colours, logos=logos
            )
        else:
            fig, _ax = plot_game_card(verdict, colors=colours, logos=logos)
        written.append(
            finalize(
                fig,
                out_dir / figure_filename(verdict, suffix),
                # The card's square shape is the point of it, and `tight` would
                # crop it to whatever its content happened to fill.
                bbox_inches=None if suffix in ("card", "luck_ledger") else "tight",
            )
        )

    if article and toss is not None:
        fig, ax = plot_bootstrap_distribution(verdict, colors=colours, logos=logos, **DTW_FIGURE)
        attach_overtime_sidebar(fig, ax, verdict, toss)
        written.append(finalize(fig, out_dir / figure_filename(verdict, ARTICLE_SUFFIX)))
    return written
