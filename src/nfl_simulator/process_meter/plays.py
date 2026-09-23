"""The plays the process meter reads, and the takeaway credit taken off them.

One filter, applied once: a pass or a run, carrying EPA, with both sides named,
not a two-point try, regular season. Every feature in :mod:`.features` is built
from the frame these loaders return, so a filter is never written twice.

Three loaders, each adding columns to the one below it without re-filtering:

``load_plays``
    the filtered plays and the columns the success rate and yards per play read;
``load_plays_returns``
    plus the columns a takeaway's return yards live in;
``load_plays_fold``
    plus ``touchdown`` and ``td_team``, so a takeaway returned for a score can be
    told from one that was not.

**Where a takeaway's yards live** (checked on 2016-2025): a pick's return is
``return_yards``, whose ``return_team`` is always the defense; a lost fumble's is
``fumble_recovery_1_yards`` when the defense recovered first, else
``fumble_recovery_2_yards`` when it recovered second (``return_team`` is blank on
almost every fumble). A fumble out of the end zone has no recovery and no yards.
Negative returns stay negative.

nflverse play-by-play only: the process meter reads no charting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nfl_simulator import paths

#: The seasons the weights are fit on. 2016 is the first nflverse season with
#: the EPA columns the meter reads.
TRAIN_SEASONS = tuple(range(2016, 2024))

#: Franchises that moved inside the 2016-2025 window, old code to modern. nflverse
#: keeps the code of the day, so a feature keyed on the raw code would split one
#: franchise in two at the relocation. This is the reverse of
#: :data:`nfl_simulator.teams.RELOCATIONS`, which answers "what was this club
#: called that season" for the figures; here the mapping only ever runs forwards.
RELOCATIONS = {"SD": "LAC", "OAK": "LV", "STL": "LA"}

# Explosive = strictly above these gains, classed by play_type (a scramble is a run).
EXPLOSIVE_PASS_YARDS = 20
EXPLOSIVE_RUN_YARDS = 12

#: Pre-snap win probability band for the garbage-time filter, both ends kept.
WP_BAND = (0.05, 0.95)
GARBAGE_FILTERS = ("all", "wp05_95")

#: The takeaway definitions: picks alone, or picks plus lost fumbles. The arm of
#: record folds ``take``.
TAKEAWAY_ARMS = ("int", "take")

_PBP_COLUMNS = [
    "season",
    "week",
    "game_id",
    "season_type",
    "posteam",
    "defteam",
    "home_team",
    "away_team",
    "play_type",
    "epa",
    "success",
    "down",
    "yards_gained",
    "interception",
    "fumble_lost",
    "wp",
    "third_down_converted",
    "third_down_failed",
    "two_point_attempt",
]

#: The columns the filter itself reads. Extras ride along beside them so the rows
#: an attachment hands back line up with the rows ``load_plays`` kept.
_FILTER_COLUMNS = [
    "game_id",
    "season_type",
    "play_type",
    "epa",
    "posteam",
    "defteam",
    "two_point_attempt",
]

RETURN_COLUMNS = (
    "return_yards",
    "return_team",
    "fumble_recovery_1_team",
    "fumble_recovery_1_yards",
    "fumble_recovery_2_team",
    "fumble_recovery_2_yards",
)
TEAM_COLUMNS = ("return_team", "fumble_recovery_1_team", "fumble_recovery_2_team")
TD_COLUMNS = ("touchdown", "td_team")

_SCHEDULE_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_type",
    "home_team",
    "away_team",
]


def cache_root(override=None) -> Path:
    """The nflverse cache root: ``override`` when given, else the library's.

    :func:`nfl_simulator.paths.data_dir` is ``$NFL_SIM_DATA_DIR`` when it is set
    and the checkout's own ``data/`` otherwise, so a caller that sets neither is
    on the repo layout.
    """
    return Path(override) if override is not None else paths.data_dir()


def _keep(pbp) -> pd.Series:
    """The one filter: a pass or a run with EPA, both sides named, regular season."""
    return (
        pbp.play_type.isin(["pass", "run"])
        & pbp.epa.notna()
        & pbp.posteam.notna()
        & pbp.defteam.notna()
        & (pbp.two_point_attempt == 0)
        & (pbp.season_type == "REG")
    )


def _read_seasons(seasons, root, columns) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = root / "pbp" / f"pbp_{season}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"no pbp cache for {season} at {path}")
        frames.append(pd.read_parquet(path, columns=columns))
    return pd.concat(frames, ignore_index=True)


# --- the three loaders ------------------------------------------------------------------


def load_plays(seasons, data_dir=None) -> pd.DataFrame:
    """The filtered pass and run plays, relocated franchises remapped."""
    pbp = _read_seasons(seasons, cache_root(data_dir), _PBP_COLUMNS)
    pbp = pbp[_keep(pbp)].copy()
    for column in ("posteam", "defteam", "home_team", "away_team"):
        pbp[column] = pbp[column].replace(RELOCATIONS)
    return pbp.drop(columns=["season_type", "two_point_attempt"]).reset_index(drop=True)


def load_scores(seasons, data_dir=None) -> pd.DataFrame:
    """Final scores of played regular-season games, franchises remapped."""
    root = cache_root(data_dir)
    schedules = pd.read_parquet(
        root / "schedules.parquet", columns=_SCHEDULE_COLUMNS + ["home_score", "away_score"]
    )
    games = schedules[
        schedules.season.isin(list(seasons))
        & (schedules.game_type == "REG")
        & schedules.home_score.notna()
    ].copy()
    for column in ("home_team", "away_team"):
        games[column] = games[column].replace(RELOCATIONS)
    games = games.rename(columns={"home_team": "home", "away_team": "away"})
    return games[["game_id", "season", "week", "home", "away", "home_score", "away_score"]]


def attach_return_columns(plays, extras) -> pd.DataFrame:
    """``plays`` plus the return columns of ``extras``, row for row; teams relocated.

    ``extras`` must be the same filtered plays in the same order: game_id and epa
    are checked on every row, and a mismatch raises instead of misaligning.
    """
    lined_up = (
        len(plays) == len(extras)
        and np.array_equal(plays.game_id.to_numpy(), extras.game_id.to_numpy())
        and np.allclose(plays.epa.to_numpy(float), extras.epa.to_numpy(float))
    )
    if not lined_up:
        raise ValueError(
            f"return columns do not line up with the plays ({len(extras)} rows "
            f"against {len(plays)})"
        )
    out = plays.copy()
    for column in RETURN_COLUMNS:
        out[column] = extras[column].to_numpy()
    for column in TEAM_COLUMNS:
        out[column] = out[column].replace(RELOCATIONS)
    return out


def load_plays_returns(seasons, data_dir=None) -> pd.DataFrame:
    """:func:`load_plays` plus the return columns, under the same filter."""
    plays = load_plays(seasons, data_dir)
    pbp = _read_seasons(seasons, cache_root(data_dir), [*_FILTER_COLUMNS, *RETURN_COLUMNS])
    return attach_return_columns(plays, pbp[_keep(pbp)].reset_index(drop=True))


def attach_touchdown_columns(plays, extras) -> pd.DataFrame:
    """``plays`` plus ``touchdown`` and ``td_team`` of ``extras``, row for row.

    ``extras`` must be the same filtered plays in the same order: game_id is
    checked on every row, and a mismatch raises instead of misaligning. This
    mirrors :func:`attach_return_columns`, which owns the return columns.
    """
    if len(plays) != len(extras) or not np.array_equal(
        plays.game_id.to_numpy(), extras.game_id.to_numpy()
    ):
        raise ValueError(
            f"touchdown columns do not line up with the plays "
            f"({len(extras)} rows against {len(plays)})"
        )
    out = plays.copy()
    for column in TD_COLUMNS:
        out[column] = extras[column].to_numpy()
    out["td_team"] = out.td_team.replace(RELOCATIONS)
    return out


def load_plays_fold(seasons, data_dir=None) -> pd.DataFrame:
    """:func:`load_plays_returns` plus ``touchdown`` and ``td_team``, same filter.

    This is the loader the scoring path uses: it carries every column the arm of
    record reads.
    """
    plays = load_plays_returns(seasons, data_dir)
    pbp = _read_seasons(seasons, cache_root(data_dir), [*_FILTER_COLUMNS, *TD_COLUMNS])
    return attach_touchdown_columns(plays, pbp[_keep(pbp)].reset_index(drop=True))


# --- the takeaway credit ----------------------------------------------------------------


def takeaway_mask(plays, arm) -> pd.Series:
    """Per play, True where the defense took the ball under ``arm``'s definition."""
    if arm not in TAKEAWAY_ARMS:
        raise ValueError(f"arm must be one of {TAKEAWAY_ARMS}, got {arm!r}")
    pick = plays.interception.fillna(0) == 1
    return pick if arm == "int" else pick | (plays.fumble_lost.fillna(0) == 1)


def credited_return_yards(plays) -> np.ndarray:
    """Per play, the return yards credited to the defense; 0 where it took nothing."""
    pick = (plays.interception.fillna(0) == 1).to_numpy()
    lost = (plays.fumble_lost.fillna(0) == 1).to_numpy()
    first = (plays.fumble_recovery_1_team == plays.defteam).to_numpy()
    second = (plays.fumble_recovery_2_team == plays.defteam).to_numpy()
    fumble = np.where(
        first,
        plays.fumble_recovery_1_yards.fillna(0).to_numpy(float),
        np.where(second, plays.fumble_recovery_2_yards.fillna(0).to_numpy(float), 0.0),
    )
    # a pick the interceptor fumbles back carries both flags; its return is the pick's
    return np.where(pick, plays.return_yards.fillna(0).to_numpy(float), np.where(lost, fumble, 0.0))


def touchdown_return_mask(plays) -> pd.Series:
    """Per play, True where the defense both took the ball and scored on the return.

    ``touchdown == 1`` alone catches the offense's own scores; ``td_team ==
    defteam`` is what makes it the defense's six points, which the scoreboard
    already holds.
    """
    took = takeaway_mask(plays, "take")
    return took & (plays.touchdown.fillna(0) == 1) & (plays.td_team == plays.defteam)


__all__ = [
    "EXPLOSIVE_PASS_YARDS",
    "EXPLOSIVE_RUN_YARDS",
    "GARBAGE_FILTERS",
    "RELOCATIONS",
    "RETURN_COLUMNS",
    "TAKEAWAY_ARMS",
    "TD_COLUMNS",
    "TEAM_COLUMNS",
    "TRAIN_SEASONS",
    "WP_BAND",
    "attach_return_columns",
    "attach_touchdown_columns",
    "cache_root",
    "credited_return_yards",
    "load_plays",
    "load_plays_fold",
    "load_plays_returns",
    "load_scores",
    "takeaway_mask",
    "touchdown_return_mask",
]
