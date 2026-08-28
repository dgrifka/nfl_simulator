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
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl

from nfl_simulator import paths
from nfl_simulator.ingest import FTN_SEASONS
from nfl_simulator.ledger import with_actual
from nfl_simulator.plots import (
    GameVerdict,
    OvertimeToss,
    attach_overtime_sidebar,
    plot_bootstrap_distribution,
    plot_game_card,
    plot_luck_ledger,
    plot_luck_ledger_card,
    verdict_from_row,
)
from nfl_simulator.simulator import EDITIONS
from nfl_simulator.style import finalize
from nfl_simulator.teams import pair_colors, team_logo, team_name

GAMES_ARTIFACT = "dtw_games_v13.parquet"
LEDGER_ARTIFACT = "dtw_ledger_v13.parquet"
OVERTIME_ARTIFACT = "26_overtime_games.parquet"
METADATA = "model_metadata_v13.json"

# Ruling R-4's second edition (document 58 §2), written by
# `research/76_full_edition_summary.py`. It is not a v1.3 artifact and it is not
# shipped: a checkout that has not run that script renders Strict and says so
# when it is asked for Full, rather than quietly checking a Full render against
# Strict's published numbers.
FULL_ARTIFACT = "full_summary.parquet"

# The first season FTN charting reaches, which is the first season the Full
# edition exists at all. Read from `ingest` rather than written as 2022 so the
# two cannot drift.
FIRST_CHARTED_SEASON = min(FTN_SEASONS)

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
# verdict in a sentence; and the arrow says what luck moved and toward whom.
# Round 5 dropped the legend switch: the axis's own tints and corner labels are
# the key now, on every figure this function draws.
#
# The share image and the article figure are the same figure at this reading and
# differ in one thing — `coverage`, below — so the two call sites pass it
# explicitly rather than leaving the difference to a default.
DTW_FIGURE = {"bin_width": 3.0, "callout": True, "arrow": True}


# --------------------------------------------------------------------------
# naming
# --------------------------------------------------------------------------


def figure_filename(verdict: GameVerdict, suffix: str) -> str:
    """`"GB_DET_23-31--95-5_strict_dtw.png"` — the baseball simulator's pattern.

    Away team first, then home, then the scoreline, then the two shares, then
    the edition, then what the figure is. The name is the caption for anyone
    scrolling a folder of them, which is why the verdict's own numbers are in it.

    **The edition is in the name, not only in the stamp.** One game has two
    adjudications and they differ in exactly the numbers the rest of the name is
    built from, so without it a Full render would overwrite the Strict one of
    any game whose two shares happened to round the same way — silently, and
    with the wrong image left on disk.

    The two shares are rounded **once, together** — the home share is rounded
    and the away share is 100 minus it — so the pair in the filename always sums
    to 100 and always agrees with the headline on the figure.

    A game whose scoreline is not on file is named by its game id instead. A
    filename is not a place to invent a score.
    """
    if verdict.home_score is None or verdict.away_score is None:
        return f"{verdict.game_id}_{verdict.edition}_{suffix}.png"
    home_share = round(verdict.dtw_home * 100)
    return (
        f"{verdict.away_team}_{verdict.home_team}_"
        f"{verdict.away_score:.0f}-{verdict.home_score:.0f}--"
        f"{100 - home_share}-{home_share}_{verdict.edition}_{suffix}.png"
    )


# --------------------------------------------------------------------------
# the rows a waterfall reads
# --------------------------------------------------------------------------


def kicker_surname(name: str | None) -> str | None:
    """`"M.Crosby"` -> `"Crosby"`, and nothing at all when there is no name.

    nflverse writes a kicker as an initial and a surname. A ledger row has room
    for one word, and the surname is the one a reader recognises — so the
    initial is dropped rather than the row being widened for it. A play without
    a name on file gets no name invented for it.
    """
    if not name:
        return None
    return str(name).split(".")[-1].strip() or None


def prepare_rows(
    frame: pl.DataFrame,
    verdict: GameVerdict,
    distances: dict | None = None,
    kickers: dict | None = None,
    passers: dict | None = None,
    receivers: dict | None = None,
) -> list[dict]:
    """Ledger rows with the three extra keys :func:`plots.plain_label` can use.

    ``actual`` is recovered from the ledger's own identity when the artifact
    predates the column (see :func:`ledger.with_actual`). ``opponent`` is the
    other team in *this* game, which is what makes "recovered by GB" sayable —
    the ledger records who fumbled, not who ended up with the ball. And
    ``kick_distance`` is the kick's real yardage from the play-by-play, left
    absent rather than guessed when the play is not found: the ledger stores a
    five-yard class, and printing its midpoint as the distance would be making
    a number up. ``kicker`` is the surname on the same play, and is absent the
    same way — as are ``passer`` and ``receiver``, the two names the Full
    edition's rows are read by.
    """
    distances = distances or {}
    kickers = kickers or {}
    passers = passers or {}
    receivers = receivers or {}
    rows = with_actual(frame).to_dicts()
    for row in rows:
        charged = row.get("charged_team")
        row["opponent"] = verdict.away_team if charged == verdict.home_team else verdict.home_team
        play_id = row.get("play_id")
        row["kick_distance"] = distances.get(play_id)
        row["kicker"] = kickers.get(play_id)
        row["passer"] = passers.get(play_id)
        row["receiver"] = receivers.get(play_id)
    return rows


# --------------------------------------------------------------------------
# the two editions
# --------------------------------------------------------------------------


def season_of(game_id: str) -> int:
    """The season an nflverse game id starts with."""
    return int(str(game_id)[:4])


def default_edition(game_id: str) -> str:
    """Which edition this game is headlined in when a caller does not choose.

    Ruling R-4 (document 58 §2): Full is the headline wherever it exists, and it
    exists from the first charted season on. A 2018 game has one adjudication,
    so it is not offered a choice it cannot have.
    """
    return "full" if season_of(game_id) >= FIRST_CHARTED_SEASON else "strict"


def check_edition(game_id: str, edition: str) -> str:
    """Refuse an impossible pairing **before** anything is loaded or simulated.

    Two ways to ask for a figure nobody can draw. The first is an edition that
    was never named — `"strict+dp"` is callable in the simulator and has no
    public name, so it cannot be rendered. The second is the Full edition on a
    game that predates FTN charting: both variant builders warn and return an
    empty list there, so the render would come back with a Strict ledger under a
    Full stamp. That is the one failure mode a stamp is supposed to prevent, and
    it is worth a stop rather than a warning.
    """
    if edition not in EDITIONS:
        raise ValueError(f"unknown edition {edition!r}; the editions are {list(EDITIONS)}")
    season = season_of(game_id)
    if edition == "full" and season < FIRST_CHARTED_SEASON:
        raise ValueError(
            f"{game_id} is a {season} game and FTN charting starts in "
            f"{FIRST_CHARTED_SEASON}, so it has no Full edition. Render it Strict."
        )
    return edition


# --------------------------------------------------------------------------
# the committed sources
# --------------------------------------------------------------------------


def _empty_summary() -> pl.DataFrame:
    return pl.DataFrame({"game_id": []}, schema={"game_id": pl.String})


@dataclass(frozen=True)
class Sources:
    """Everything on disk a render needs, loaded once for a whole batch."""

    games: pl.DataFrame
    ledger: pl.DataFrame
    schedule: pl.DataFrame
    overtime: pl.DataFrame
    slope: float
    # The Full edition's summary, and it is allowed to be empty. `games` is the
    # shipped v1.3 artifact and is always there; this one is written by
    # `research/76_full_edition_summary.py` and a checkout that has not run it
    # still renders every Strict figure.
    full: pl.DataFrame = field(default_factory=_empty_summary)

    def summary(self, edition: str = "strict") -> pl.DataFrame:
        """The summary the named edition's numbers are published in."""
        return self.full if edition == "full" else self.games

    def game_row(self, game_id: str, edition: str = "strict") -> dict:
        """This game's published row **in the edition asked for**.

        Never falls back to the other edition. A Full figure checked against
        Strict's numbers would replay clean and then print a headline the replay
        does not belong to, which is the whole reason the check exists.
        """
        artifact = FULL_ARTIFACT if edition == "full" else GAMES_ARTIFACT
        rows = self.summary(edition).filter(pl.col("game_id") == game_id).to_dicts()
        if not rows:
            raise SystemExit(
                f"{game_id} is not in {artifact}."
                + (
                    "  Run `uv run python research/76_full_edition_summary.py` to build it."
                    if edition == "full"
                    else ""
                )
            )
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
    full = output / FULL_ARTIFACT
    return Sources(
        games=pl.read_parquet(output / GAMES_ARTIFACT),
        ledger=pl.read_parquet(output / LEDGER_ARTIFACT),
        schedule=pl.read_parquet(paths.SCHEDULE_PATH),
        overtime=pl.read_parquet(output / OVERTIME_ARTIFACT),
        slope=slope,
        # Absence is not an error, for the reason on the field itself: every
        # Strict figure still draws on a checkout that has not run the Full pass.
        full=pl.read_parquet(full) if full.exists() else _empty_summary(),
    )


def _read_side():
    """`research/44_read_side_fix.py`, imported off the research path.

    That is where the read-side fix and the refitted field-goal model live, and
    re-implementing either here would be a second copy of the thing document 30
    corrected.
    """
    research = Path(__file__).resolve().parents[2] / "research"
    if str(research) not in sys.path:
        sys.path.insert(0, str(research))
    from importlib import import_module

    return import_module("44_read_side_fix")


def simulation_columns() -> list[str]:
    """The play-by-play columns a replay of **either** edition needs.

    v1.3's own frame plus what the two hands-on-the-ball models price on — the
    same list `research/68_dropped_pick_variant_audit.py` builds, and for the
    same reason. It is loaded unconditionally rather than per edition because
    document 49 §6's V-1 replayed all 2,761 shipped games on the wide frame at
    0.00e+00: the extra columns are proven inert on Strict, not assumed to be,
    so one frame serves both editions and there is no second cache to keep
    consistent with the first.
    """
    from nfl_simulator.dropped_picks import PBP_COVARIATE_COLUMNS as DROPPED_PICK_COLUMNS
    from nfl_simulator.receiver_drops import PBP_COVARIATE_COLUMNS as RECEIVER_COLUMNS
    from nfl_simulator.receiver_drops import PBP_SWING_COLUMNS

    return list(
        dict.fromkeys(
            [
                *_read_side().SIM_COLUMNS,
                # The defence the dropped pick is charged against; v1.3 never
                # needed it, because a fumble is charged to whoever fumbled.
                "defteam",
                # Presentation only, added in figure round 6 the way
                # `kicker_player_name` was added in round 4: a Full-edition row
                # is read by who threw the ball and who it was thrown to.
                # Nothing prices on either — the dropped pick was priced at the
                # defence's shrunk rate and the drop at the receiving corps' —
                # and the seven example games still replay at 0.00e+00 with both
                # loaded.
                "passer_player_name",
                "receiver_player_name",
                *DROPPED_PICK_COLUMNS,
                *RECEIVER_COLUMNS,
                *PBP_SWING_COLUMNS,
            ]
        )
    )


@cache
def _simulation_context():
    """Both editions' fitted pieces, built once so a batch pays for them once."""
    from nfl_simulator.components import (
        build_game_table,
        fit_fg_baseline,
        fit_fumble_baseline,
        fit_xp_baseline,
    )
    from nfl_simulator.ingest import PBP_SEASONS, load_pbp
    from nfl_simulator.simulator import points_per_epa

    read_side = _read_side()
    pbp = load_pbp(PBP_SEASONS, columns=simulation_columns())
    fg_model, _ = read_side.load_model("trace_fg_refit.nc", "fg_refit_summary.json")
    dropped_pick_model, ftn = _dropped_pick_pieces()
    receiver_drop_model = _receiver_drop_pieces()
    return {
        "pbp": pbp,
        "fg_model": fg_model,
        "fumble_baseline": fit_fumble_baseline(pbp),
        "fg_baseline": fit_fg_baseline(pbp),
        "xp_baseline": fit_xp_baseline(pbp),
        "slope": points_per_epa(build_game_table(pbp).drop_nulls("margin")),
        # Amendment A-3's two components, loaded once for a batch and **used by
        # nothing in this module**. `replay` below reproduces the shipped Strict
        # summary, so it cannot pass these; they are here so a caller that wants
        # the Full edition does not pay for the traces and the charting pull a
        # second time. Nothing renders them yet — that is figure round 6.
        "dropped_pick_model": dropped_pick_model,
        "receiver_drop_model": receiver_drop_model,
        "ftn": ftn,
        # Ruling R-4's two editions (document 58 §2), as the handles each one
        # simulates with, so a caller renders both from one context.
        "editions": edition_handles(dropped_pick_model, receiver_drop_model),
    }


def edition_handles(dropped_pick_model, receiver_drop_model) -> dict[str, dict]:
    """What each edition passes to `simulate_game` (document 58 §2).

    Strict pays for no model — it is v1.3 byte for byte, and that is the whole
    point of the name. Full pays for both directions of the hands-on-the-ball
    class, which amendment A-3 clause 3 admits together or not at all. A handle
    that failed to load stays `None`: a checkout without the traces still
    renders Strict, and asks for Full at whatever coverage it has.
    """
    return {
        "strict": {"dropped_pick_model": None, "receiver_drop_model": None},
        "full": {
            "dropped_pick_model": dropped_pick_model,
            "receiver_drop_model": receiver_drop_model,
        },
    }


def _dropped_pick_pieces():
    """The variant's fitted model and the charting frame, or ``(None, None)``.

    Absence is not an error. The v1.3 figures are the product, and a checkout
    that has not run `research/67_dropped_pick_model.py` must still render every
    one of them — so a missing trace degrades to "no variant available" rather
    than taking the render down with it.
    """
    from nfl_simulator.dropped_picks import DroppedPickModel
    from nfl_simulator.ingest import FTN_SEASONS, load_ftn

    try:
        model = DroppedPickModel.from_posterior(
            paths.RESEARCH_OUTPUT_DIR / "trace_dropped_pick.nc",
            paths.RESEARCH_OUTPUT_DIR / "dropped_pick_summary.json",
        )
        return model, load_ftn(FTN_SEASONS)
    except FileNotFoundError as error:
        print(f"Note: the dropped-pick variant is unavailable ({error}).")
        return None, None


def _receiver_drop_pieces():
    """The receiver direction's fitted model, or ``None``.

    Absence is not an error, for `_dropped_pick_pieces`'s reason: the charting
    frame is shared, so only the trace is loaded here.
    """
    from nfl_simulator.receiver_drops import ReceiverDropModel

    try:
        return ReceiverDropModel.from_posterior(
            paths.RESEARCH_OUTPUT_DIR / "trace_receiver_drop.nc",
            paths.RESEARCH_OUTPUT_DIR / "receiver_drop_summary.json",
        )
    except FileNotFoundError as error:
        print(f"Note: the receiver-drop component is unavailable ({error}).")
        return None


def replay_gaps(result, row: dict) -> dict[str, float]:
    """How far a re-simulation lands from the summary row it is checked against.

    The four published numbers, and only those four: a figure states the
    deserved margin, the DTW share and the interval's two ends, so those are
    what a redraw has to reproduce before a pixel is drawn.
    """
    return {
        "deserved_margin": abs(result.deserved_margin - row["deserved_margin"]),
        "dtw_home": abs(result.dtw_home - row["dtw_home"]),
        "dtw_low": abs(result.dtw_interval[0] - row["dtw_low"]),
        "dtw_high": abs(result.dtw_interval[1] - row["dtw_high"]),
    }


def replay(game_id: str, row: dict, schedule: dict | None = None, *, edition: str = "strict"):
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

    ``edition`` names which adjudication is replayed, and ``row`` must be that
    edition's published row — `Sources.game_row` is what pairs them. Strict is
    v1.3 with both variant models switched off and replays at 0.00e+00 against
    the shipped summary, exactly as it did before this argument existed.
    """
    from nfl_simulator.simulator import simulate_game

    check_edition(game_id, edition)
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
        # The charting frame is handed over whole: `worthy_throw_frame` and
        # `catchable_target_frame` both join it on game id and play id, so a
        # frame carrying every 2022-2025 game contributes exactly this game's
        # rows. Strict discards both handles inside `simulate_game`.
        ftn=context["ftn"],
        **context["editions"][edition],
        **scores,
    )
    gaps = replay_gaps(result, row)
    if max(gaps.values()) > REPLAY_TOLERANCE:
        raise SystemExit(
            f"{game_id} does not replay to its {edition} summary ({gaps}). "
            "Stop rather than draw it."
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


def kicker_names(game_id: str) -> dict:
    """`play_id -> surname` for the game's kicks, from the cached play-by-play.

    Degrades exactly as :func:`kick_distances` does: no name is a label without
    a name on it, never a render that stops.
    """
    try:
        plays = _simulation_context()["pbp"].filter(pl.col("game_id") == game_id)
        if "kicker_player_name" not in plays.columns:
            return {}
        kicks = plays.select("play_id", "kicker_player_name").drop_nulls("kicker_player_name")
        return {float(p): kicker_surname(name) for p, name in kicks.iter_rows()}
    except Exception as error:  # pragma: no cover - the labels degrade, not the render
        print(f"Warning: no kicker names for {game_id}: {error}")
        return {}


def _player_names(game_id: str, column: str) -> dict:
    """`play_id -> surname` for one play-by-play name column.

    Degrades exactly as :func:`kick_distances` does — a missing column or an
    unreadable cache costs the labels their names, never the render.
    """
    try:
        plays = _simulation_context()["pbp"].filter(pl.col("game_id") == game_id)
        if column not in plays.columns:
            return {}
        named = plays.select("play_id", column).drop_nulls(column)
        return {float(play): kicker_surname(name) for play, name in named.iter_rows()}
    except Exception as error:  # pragma: no cover - the labels degrade, not the render
        print(f"Warning: no {column} for {game_id}: {error}")
        return {}


def passer_names(game_id: str) -> dict:
    """`play_id -> the quarterback's surname`, for the dropped-pick rows."""
    return _player_names(game_id, "passer_player_name")


def receiver_names(game_id: str) -> dict:
    """`play_id -> the target's surname`, for the receiver-drop rows."""
    return _player_names(game_id, "receiver_player_name")


# --------------------------------------------------------------------------
# the render
# --------------------------------------------------------------------------


def counterpart_verdict(sources: Sources, game_id: str, edition: str, schedule: dict):
    """The *other* edition's verdict, for the one muted line that quotes it.

    Read from that edition's published summary and never replayed. The replay
    check exists for the distribution a figure **draws**; this verdict is drawn
    by nothing — it contributes a headline and a deserved margin to a footer —
    and re-simulating a second adjudication to print two numbers would double
    every render's cost for no guarantee the committed record does not already
    give.

    ``None`` when there is no other edition: a game before the first charted
    season has one adjudication, and `GameVerdict.edition_note` says so.
    """
    if season_of(game_id) < FIRST_CHARTED_SEASON:
        return None
    other = "strict" if edition == "full" else "full"
    return verdict_from_row(
        sources.game_row(game_id, edition=other), np.empty(0), schedule, edition=other
    )


def render_game(
    game_id: str,
    out_dir: Path | None = None,
    *,
    article: bool = False,
    edition: str | None = None,
) -> list[Path]:
    """Write this game's four PNGs and return their paths, in ``SUFFIXES`` order.

    **No share image carries the sidebar.** Document 16's refusal is a fact about
    every one of these figures' ledgers, and every one of them states it — but as
    one muted line, not as six paragraphs of methodology down the side. The
    sidebar also grows the figure, so an overtime game and a regulation one came
    out at two different widths and a timeline crops them differently.

    ``article=True`` adds one more file for an overtime game,
    ``{...}_dtw_article.png``: the distribution with the panel attached. That is
    where the six paragraphs belong.

    ``edition`` names which of ruling R-4's two adjudications is drawn. Left
    ``None`` it is :func:`default_edition` — Full from the first charted season,
    Strict before it — which is the headline reading. Every file the call writes
    carries the edition in its name and in its stamp, and each of them states
    the other edition's verdict in one muted line.
    """
    out_dir = Path(out_dir) if out_dir is not None else paths.RESEARCH_OUTPUT_DIR
    edition = check_edition(game_id, edition or default_edition(game_id))
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

    # Strict reads the shipped ledger artifact, which is the committed record of
    # v1.3's rows. Full has no shipped ledger and takes the rows from the replay
    # that was just checked against its summary — the same coins the histogram
    # above them is drawn from, which is the rule this module opens with.
    ledger = (
        result.ledger.to_frame()
        if edition == "full"
        else sources.ledger.filter(pl.col("game_id") == game_id).drop("game_id")
    )
    rows = prepare_rows(
        ledger,
        verdict,
        kick_distances(game_id),
        kicker_names(game_id),
        passer_names(game_id),
        receiver_names(game_id),
    )
    colours = pair_colors(verdict.home_team, verdict.away_team)
    sides = (verdict.home_team, verdict.away_team)
    logos = {team: team_logo(team) for team in sides}
    names = {team: team_name(team) for team in sides}
    toss = sources.toss(verdict) if verdict.went_to_overtime else None

    written = []
    for suffix in SUFFIXES:
        if suffix == "dtw":
            # The margin distribution, with round 5's unsigned "wins by" axis.
            # Round 4's per-team scoreline figure is withdrawn from the render
            # path — `plot_team_points_distribution` is correct arithmetic and
            # stays in the module, but a margin swing is not a per-team points
            # swing, and drawing it as one put `GB 44 - DET 35` on a share
            # image. Nothing renders it.
            fig, _ax = plot_bootstrap_distribution(
                verdict, colors=colours, logos=logos, coverage=False, **DTW_FIGURE
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
                edition=edition,
                # The card's square shape is the point of it, and `tight` would
                # crop it to whatever its content happened to fill.
                bbox_inches=None if suffix in ("card", "luck_ledger") else "tight",
            )
        )

    if article and toss is not None:
        # The one difference from the share image: document 10's measured
        # coverage, for a reader who has already asked for the methodology.
        fig, ax = plot_bootstrap_distribution(
            verdict, colors=colours, logos=logos, coverage=True, **DTW_FIGURE
        )
        attach_overtime_sidebar(fig, ax, verdict, toss)
        written.append(
            finalize(fig, out_dir / figure_filename(verdict, ARTICLE_SUFFIX), edition=edition)
        )
    return written
