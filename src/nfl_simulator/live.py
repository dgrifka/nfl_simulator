"""One final game in, four PNGs and a verdict out — with nothing shipped for it.

`render.render_game` is the batch front door and it reads a game's numbers from
the committed artifacts: the deserved margin, the DTW% and the interval all come
out of `dtw_games_v14.parquet`, and the replay exists to prove the figure belongs
to them. That is exactly right for the 2,761 games the research record covers and
useless for a game that went final tonight, which has no row anywhere.

This module is the other door. It adjudicates the game from its own play-by-play
and the same fitted pieces the batch renderer uses — the fumble, field-goal and
extra-point baselines fit on 2016-2025, v1.4's field-goal posterior, amendment
A-3's two hands-on-the-ball models — and then draws the same four figures from
the result rather than from an artifact row.

Two things it deliberately does not do. It does not consult the shipped summary
for the game it is adjudicating, so agreement with that summary on a 2025 game is
a real check rather than a tautology — `tests/test_live.py::TestAcceptance` is
that check. And it changes no model and no simulator setting: the seed, the draw
counts and the blocked-kick exclusion are `render`'s own constants, imported
rather than restated.

**What it needs on disk.** Two directories, both nameable from the environment
(see :mod:`nfl_simulator.paths`):

``$NFL_SIM_DATA_DIR``
    the cached pulls. The baselines are fit on the whole 2016-2025 play-by-play,
    so that cache has to be present even to adjudicate a 2026 game.

``$NFL_SIM_ARTIFACT_DIR``
    the fitted artifacts — v1.4's field-goal posterior and its summary, and the
    dropped-pick and receiver-drop traces the Full edition needs.

**Runtime**, measured on the maintainer's machine (Apple M4 Pro) on
2026-09-01: 1.3 s to build the fitted context and 2.4 s to adjudicate and draw
one game, so **3.7 s cold and 2.4 s per game after that** — the context is
cached for the life of the process, which is why a caller adjudicating a slate
should do it in one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from nfl_simulator import paths, render
from nfl_simulator.ingest import (
    FTN_SEASONS,
    PBP_SEASONS,
    ingest_live_season,
    load_ftn_if_cached,
    load_pbp,
    schedule_row,
    warn_on_manifest_drift,
)
from nfl_simulator.plots import GameVerdict, plain_label
from nfl_simulator.render import FIRST_CHARTED_SEASON, season_of
from nfl_simulator.teams import era_code


@dataclass(frozen=True)
class LiveAdjudication:
    """One game's verdict, as the caller downstream of this package reads it.

    Every field here is part of the published contract — the orchestration side
    reads them by name — so they are named for what they are rather than for
    where they came from.
    """

    game_id: str
    #: `"strict"` or `"full"`. Full where FTN charting reaches and Strict
    #: everywhere else, including a charted game whose charting has not landed
    #: yet; `edition_note` says which of those two a Strict reading is.
    edition: str
    home_team: str
    away_team: str
    #: The scoreboard, or ``None`` when the game is in no cached schedule.
    home_points: float | None
    away_points: float | None
    actual_margin: float
    deserved_margin: float
    dtw_home: float
    dtw_low: float
    dtw_high: float
    #: The biggest single luck event in plain words, or ``None`` for a game
    #: whose ledger is empty or nets to nothing.
    headline: str | None
    #: The four PNGs, in `render.SUFFIXES` order.
    figures: list[Path] = field(default_factory=list)
    #: Empty when the edition is the one the season allows, and one sentence
    #: when this reading was reduced — which is how a caller tells "2018, so
    #: Strict" from "2026 and the charting has not landed, so Strict".
    edition_note: str = ""

    @property
    def deserved_winner(self) -> str:
        return self.home_team if self.dtw_home > 0.5 else self.away_team

    @property
    def scoreboard_winner(self) -> str | None:
        """``None`` on a tie: the scoreboard named nobody."""
        if self.actual_margin > 0:
            return self.home_team
        if self.actual_margin < 0:
            return self.away_team
        return None


def resolve_edition(
    game_id: str,
    requested: str | None = None,
    *,
    ftn: pl.DataFrame | None = None,
    models_available: bool = True,
) -> tuple[str, str]:
    """Which adjudication this game can actually have, and why if it is reduced.

    Ruling R-4 makes Full the headline wherever it exists, and it exists where
    FTN charting reaches. Live, that is a question about *this game* rather than
    about its season: charting lands days after the play-by-play does, so a game
    that went final tonight is a charted season's game with no charting. The
    honest reading then is Strict — the same numbers v1.4 published, on a
    complete ledger of what it prices — and the note says the Full reading is
    pending rather than absent.

    An explicit ``requested`` edition is honoured where it is possible and
    refused where it is not: `render.check_edition` already refuses Full on a
    pre-charting game, and asking for Full on a game whose charting has not
    arrived would draw a Strict ledger under a Full stamp.
    """
    if requested is not None:
        render.check_edition(game_id, requested)
    edition = requested or render.default_edition(game_id)

    if edition != "full":
        return edition, ""

    if not models_available:
        return "strict", (
            "adjudicated Strict: the dropped-pick and receiver-drop posteriors "
            "are not in the artifact directory, so the Full edition's two "
            "components cannot be priced."
        )

    charted = ftn is not None and (ftn.filter(pl.col("nflverse_game_id") == game_id).height > 0)
    if not charted:
        if requested is not None:
            raise ValueError(
                f"{game_id} has no FTN charting on file, so it has no Full "
                "edition yet. Adjudicate it Strict, or wait for the charting."
            )
        return "strict", (
            f"adjudicated Strict: FTN charting for {game_id} has not landed "
            "yet, so the Full edition's two components have nothing to price."
        )
    return "full", ""


def headline_from_rows(
    rows,
    *,
    points_per_epa: float,
    home_team: str,
    away_team: str,
) -> str | None:
    """The biggest single luck event in the ledger, in one line.

    `plots.luck_bars` is the figure's answer to the same question and folds
    every sliver under a per-club heap, which is right on a waterfall and wrong
    here: a headline names an event somebody watched. So this reads the ledger
    rows directly and takes the largest ``|luck_epa|``.

    The club named is the one the luck *favoured*, which is the sign of
    ``luck_epa`` — positive is good fortune for the home team, by the ledger's
    own convention — and never the club charged with the event. A dropped
    interception is charged to the defence that dropped it and favoured the
    offence that threw it, and the sentence a reader wants says the second.
    """
    scored = [row for row in rows if float(row.get("luck_epa") or 0.0) != 0.0]
    if not scored:
        return None
    row = max(scored, key=lambda entry: abs(float(entry["luck_epa"])))
    luck_epa = float(row["luck_epa"])
    beneficiary = home_team if luck_epa > 0 else away_team
    points = abs(luck_epa) * points_per_epa
    # `drop*` and `dropped pick*` carry the figures' footnote marker, and the
    # footnote it points at is on the figure. A line of text has nowhere to
    # send the reader, so the marker comes off rather than dangling.
    label = plain_label(row).replace("*", "")
    return f"{label} — {points:.1f} pt of luck to {beneficiary}"


def _ensure_plays(game_id: str, *, pull: bool, refresh: bool) -> pl.DataFrame:
    """This game's rows, pulling the season first when it is not the frozen one.

    A 2016-2025 game is read from the same frame the baselines were fit on. Any
    other season is the live case: it is pulled unless it is already cached and
    the caller did not ask for a refresh, and `ingest.ingest_live_season`
    validates it as the part-played season it is.
    """
    season = season_of(game_id)
    columns = render.simulation_columns()

    if season in PBP_SEASONS:
        return render.game_plays(game_id, render._simulation_context()["pbp"])

    cached = paths.pbp_path(season).exists()
    if pull and (refresh or not cached):
        ingest_live_season(season)
    elif not cached:
        raise FileNotFoundError(
            f"{season} play-by-play is not cached and `pull=False` — set "
            f"{paths.DATA_DIR_ENV} to the cache, or allow the pull."
        )

    plays = load_pbp([season], columns=columns).filter(pl.col("game_id") == game_id)
    if plays.is_empty():
        raise ValueError(
            f"{game_id} has no play-by-play in the {season} pull. Check the game "
            "id, or wait for nflverse to publish the game."
        )
    return plays


def _baseline_cache_present() -> None:
    """One clean error naming the variable when the baselines have nothing to fit on."""
    uncached = [season for season in PBP_SEASONS if not paths.pbp_path(season).exists()]
    if uncached:
        raise FileNotFoundError(
            f"the {PBP_SEASONS[0]}-{PBP_SEASONS[-1]} play-by-play cache is "
            f"incomplete ({uncached[0]} missing from {paths.data_dir()}). The "
            "fumble, field-goal and extra-point baselines are fit on that whole "
            f"window, so it is needed even for a {PBP_SEASONS[-1] + 1} game — set "
            f"{paths.DATA_DIR_ENV} to the synced cache, or run "
            "`python -m nfl_simulator.ingest`."
        )


def _charting_for(game_id: str) -> pl.DataFrame | None:
    """The charting frame that could cover this game, or ``None``."""
    season = season_of(game_id)
    if season < FIRST_CHARTED_SEASON:
        return None
    if season in FTN_SEASONS:
        return render._simulation_context()["ftn"]
    return load_ftn_if_cached([season])


def adjudicate_live_game(
    game_id: str,
    out_dir: Path | None = None,
    *,
    edition: str | None = None,
    pull: bool = True,
    refresh: bool = False,
) -> LiveAdjudication:
    """Adjudicate one game from its play-by-play and write its four figures.

    ``game_id`` is nflverse's — `"2026_01_DAL_PHI"`. ``out_dir`` is where the
    PNGs go, and defaults to the artifact directory the way `render.render_game`
    defaults to it. ``edition`` forces one of ruling R-4's two adjudications and
    is normally left alone: see :func:`resolve_edition`. ``pull`` allows the
    network for a season outside the frozen window, and ``refresh`` re-pulls
    that season even when it is already cached — which is what a caller
    adjudicating tonight's game wants when it pulled this afternoon's.

    Deterministic: the seed, the 200 posterior draws and the 800 coin draws are
    `render`'s shipped constants, so two calls on the same game return the same
    numbers and write the same pixels.

    Runtime is ~2.4 s per game after a ~1.3 s first call that fits the
    baselines. Both directories in this module's docstring must be readable; a
    missing one is one error naming the variable rather than a crash inside a
    parquet reader.
    """
    from nfl_simulator.simulator import simulate_game

    _baseline_cache_present()
    out_dir = Path(out_dir) if out_dir is not None else paths.artifact_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = paths.artifact_dir() / render.METADATA
    if metadata_path.exists():
        import json

        with metadata_path.open() as handle:
            warn_on_manifest_drift(json.load(handle))

    context = render._simulation_context()
    plays = _ensure_plays(game_id, pull=pull, refresh=refresh)
    charting = _charting_for(game_id)
    edition, note = resolve_edition(
        game_id,
        edition,
        ftn=charting,
        models_available=(
            context["dropped_pick_model"] is not None and context["receiver_drop_model"] is not None
        ),
    )

    schedule = schedule_row(game_id)
    scores = {
        "home_points": schedule.get("home_score"),
        "away_points": schedule.get("away_score"),
    }
    if scores["home_points"] is None or scores["away_points"] is None:
        scores = {}

    result = simulate_game(
        plays,
        fumble_baseline=context["fumble_baseline"],
        fg_baseline=context["fg_baseline"],
        xp_baseline=context["xp_baseline"],
        fg_model=context["fg_model"],
        points_per_epa=context["slope"],
        n_posterior_draws=render.POSTERIOR_DRAWS,
        n_coin_draws=render.COIN_DRAWS,
        seed=render.RANDOM_SEED,
        include_blocked=False,
        ftn=charting,
        **context["editions"][edition],
        **scores,
    )

    verdict = _verdict_from_result(result, plays, schedule, edition=edition)
    ledger = result.ledger.to_frame()
    figures = render.write_figures(
        verdict,
        result,
        ledger=ledger,
        points_per_epa=context["slope"],
        out_dir=out_dir,
        plays=plays,
        # A live game is in no overtime artifact — document 16's note is a
        # measured 2016-2025 quantity — so there is no sidebar to attach.
        toss=None,
    )

    rows = render.prepare_rows(ledger, verdict)
    return LiveAdjudication(
        game_id=game_id,
        edition=edition,
        edition_note=note,
        home_team=verdict.home_team,
        away_team=verdict.away_team,
        home_points=_optional_float(schedule.get("home_score")),
        away_points=_optional_float(schedule.get("away_score")),
        actual_margin=float(result.actual_margin),
        deserved_margin=float(result.deserved_margin),
        dtw_home=float(result.dtw_home),
        dtw_low=float(result.dtw_interval[0]),
        dtw_high=float(result.dtw_interval[1]),
        headline=headline_from_rows(
            rows,
            points_per_epa=context["slope"],
            home_team=verdict.home_team,
            away_team=verdict.away_team,
        ),
        figures=figures,
    )


def _optional_float(value) -> float | None:
    return None if value is None else float(value)


def _verdict_from_result(result, plays: pl.DataFrame, schedule: dict, *, edition: str):
    """A `GameVerdict` built from the replay rather than from a summary row.

    `plots.verdict_from_row` is the batch path's builder and takes a row out of
    the shipped artifact. There is no such row for a game that has just gone
    final, so the same fields come from the simulation result — which is the
    only source that exists for them, and is the same source the shipped rows
    were themselves written from.

    ``counterpart`` is left ``None`` on purpose. It is the other edition's
    *published* verdict, and a live game has no published anything; a verdict
    without one prints nothing rather than a claim it cannot support (see
    `GameVerdict.edition_note`).
    """
    season = season_of(result.game_id)
    home = era_code(str(plays["home_team"][0]), season)
    away = era_code(str(plays["away_team"][0]), season)
    return GameVerdict(
        game_id=result.game_id,
        home_team=home,
        away_team=away,
        actual_margin=float(result.actual_margin),
        deserved_margin=float(result.deserved_margin),
        dtw_home=float(result.dtw_home),
        dtw_interval=(float(result.dtw_interval[0]), float(result.dtw_interval[1])),
        margin_draws=np.asarray(result.margin_draws),
        home_score=schedule.get("home_score"),
        away_score=schedule.get("away_score"),
        game_date=schedule.get("gameday"),
        went_to_overtime=bool(schedule.get("overtime") or False),
        edition=edition,
        counterpart=None,
    )
