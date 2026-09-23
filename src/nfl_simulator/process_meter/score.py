"""Cache plus a list of game ids in, one row per game out.

This is the seam between the model and whatever shows it. Nothing here re-derives
a filter, a feature or a fit: the loaders, the frame, the posterior and the
percent are the same functions the research record ran, so a published number is
the research number by construction rather than by resemblance. The weights are
never fit here — they arrive from :func:`.weights.load_weights`.

Two things a reader of a scored row should know.

The per-side ``ypp`` and ``success_rate`` are the model's inputs, not the box
score: they fold a credited takeaway in, and they fold it differently. The
success rate counts a takeaway as one more play for the taking team, so its
``n`` is the folded denominator, while yards per play credits the return yards
and counts the team's own plays only, so ``ypp`` is at or above ``yards /
plays`` and its denominator is ``n_y`` = ``plays``. The two agree only on a
team-game with no takeaway.

The per-side deserved points are the plug-in on the mean weights, because only
the margin is drawn and its intercept cancels there. A side's own uncertainty is
:func:`~nfl_simulator.process_meter.record.points_draws_per_game`.

nflverse play-by-play only: the process meter reads no charting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_simulator.process_meter.features import OWN_TERMS, team_game_features_foldn
from nfl_simulator.process_meter.plays import load_plays_fold, load_scores
from nfl_simulator.process_meter.posterior import production_posterior_foldn
from nfl_simulator.process_meter.record import (
    N_DRAWS_RECORD,
    SEED_OF_RECORD,
    margin_draws_per_game,
    percent_record,
    sides,
)

#: The seasons the frame is built over. The live season is added by
#: :func:`score_games` when it is not one of them.
BASELINE_SEASONS = tuple(range(2016, 2026))

#: The per-side numbers a scored row carries: the four a reader sees, then the
#: five the sampler draws from. ``n_y`` is one of the five, because the yards draw
#: narrows by the team's own plays and the rate draw by the folded count; anything
#: that redraws a side needs both or it draws a different game.
SIDE_STATS = ("plays", "yards", "ypp", "success_rate", "k", "n", "y_mean", "y_sd", "n_y")

#: One row per game: the scoreboard and the verdict, then the inputs per side, so
#: nothing downstream has to recompute what the scorer already knows.
SCORE_COLUMNS = [
    "game_id",
    "season",
    "week",
    "home",
    "away",
    "home_score",
    "away_score",
    "home_deserved",
    "away_deserved",
    "deserved_margin",
    "p_home",
    *[f"{side}_{stat}" for side in ("home", "away") for stat in SIDE_STATS],
]


def _seasons_for(season) -> tuple:
    return tuple(sorted({*BASELINE_SEASONS, int(season)}))


def build_frames(season, data_dir=None) -> tuple:
    """The frame of record, its production posterior and the schedule, for one season.

    Building these is the expensive part of a score — it reads every cached
    season — so a caller scoring several sets of games should build once and pass
    the result to :func:`score_games` as ``frames``.
    """
    seasons = _seasons_for(season)
    plays = load_plays_fold(seasons, data_dir)
    scores = load_scores(seasons, data_dir)
    frame = team_game_features_foldn(plays, scores, "all")
    in_frame = plays.merge(frame[["game_id", "posteam"]], on=["game_id", "posteam"], how="inner")
    prod_post = production_posterior_foldn(in_frame, frame)
    return frame, prod_post, scores


def likely_points(frame, weights) -> pd.Series:
    """The plug-in likely points: the intercept plus the weights on the two inputs.

    This is the point estimate, not a draw. The draws carry the uncertainty and
    the margin's intercept cancels, so a per-side number has to come from the
    mean weights.
    """
    columns = list(weights.columns)
    if columns[0] != "intercept" or tuple(columns[1:]) != tuple(f"{t}_for" for t in OWN_TERMS):
        raise ValueError(f"expected the intercept and {OWN_TERMS} columns, got {columns}")
    mean = np.asarray(weights.mean, dtype=float)
    design = np.column_stack(
        [np.ones(len(frame))] + [frame[name].to_numpy(float) for name in columns[1:]]
    )
    return pd.Series(design @ mean, index=frame.index)


def _games(frame, scores, game_ids) -> pd.DataFrame:
    """One row per wanted game, home side first: the keys the draws read."""
    wanted = list(dict.fromkeys(game_ids))
    known = set(frame.game_id)
    unknown = [g for g in wanted if g not in known]
    if unknown:
        raise ValueError(f"{len(unknown)} game(s) have no scored rows in the cache: {unknown[:5]}")
    keyed = scores.set_index("game_id")
    return pd.DataFrame(
        {
            "game_id": wanted,
            "season": keyed.season.reindex(wanted).to_numpy(int),
            "week": keyed.week.reindex(wanted).to_numpy(int),
            "home": keyed.home.reindex(wanted).to_numpy(),
            "away": keyed.away.reindex(wanted).to_numpy(),
            "home_score": keyed.home_score.reindex(wanted).to_numpy(int),
            "away_score": keyed.away_score.reindex(wanted).to_numpy(int),
        }
    )


def margin_draws(
    games, weights, prod_post, *, n_draws=N_DRAWS_RECORD, seed=SEED_OF_RECORD
) -> np.ndarray:
    """The deserved-margin draws per game, rows in ``games`` order.

    Straight through :func:`~nfl_simulator.process_meter.record.margin_draws_per_game`:
    each game gets its own generator, seeded from the game id, so a game scored
    alone reads the same as a game scored inside a week's batch.
    """
    return margin_draws_per_game(games, weights, prod_post, n_draws, seed)


def score_games(
    season,
    game_ids,
    weights,
    data_dir=None,
    *,
    n_draws=N_DRAWS_RECORD,
    seed=SEED_OF_RECORD,
    frames=None,
    with_draws=False,
):
    """One row per game in :data:`SCORE_COLUMNS` order: the scoreboard, the verdict, the inputs.

    ``weights`` is the artifact (:func:`.weights.load_weights`), never a fit made
    here. ``frames`` lets a caller reuse :func:`build_frames`' output across
    calls; ``with_draws`` returns the margin-draw matrix beside the table.
    """
    frame, prod_post, scores = frames if frames is not None else build_frames(season, data_dir)
    games = _games(frame, scores, game_ids)

    draws = margin_draws(games, weights, prod_post, n_draws=n_draws, seed=seed)
    games["p_home"] = percent_record(games, draws)

    rows_by_key = frame.set_index(["game_id", "posteam"])
    rows_by_key = rows_by_key.assign(deserved=likely_points(rows_by_key, weights))
    per_side = sides(games, prod_post)  # k, n, y_mean, y_sd, n_y; home first
    draw_stats = ("k", "n", "y_mean", "y_sd", "n_y")
    for position, side in enumerate(("home", "away")):
        keys = pd.MultiIndex.from_arrays([games.game_id, games[side]])
        rows = rows_by_key.loc[keys]
        games[f"{side}_deserved"] = rows.deserved.to_numpy(float)
        games[f"{side}_plays"] = rows.plays_for.to_numpy(float)
        games[f"{side}_yards"] = rows.yards_for.to_numpy(float)
        games[f"{side}_ypp"] = rows.own_ypp_for.to_numpy(float)
        games[f"{side}_success_rate"] = rows.own_success_rate_for.to_numpy(float)
        for column, values in zip(draw_stats, per_side[position].T, strict=True):
            games[f"{side}_{column}"] = values
    games["deserved_margin"] = games.home_deserved - games.away_deserved

    table = games[SCORE_COLUMNS].reset_index(drop=True)
    return (table, draws) if with_draws else table


__all__ = [
    "BASELINE_SEASONS",
    "SCORE_COLUMNS",
    "SIDE_STATS",
    "build_frames",
    "likely_points",
    "margin_draws",
    "score_games",
]
