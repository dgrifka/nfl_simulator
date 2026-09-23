"""One row per team per game: the two inputs the meter weights, and the fold.

The arm of record is ``foldn_take_sr``. With N own plays, K own successes, T
takeaways (T_s of them successful for the taking team), Y own yards and R
credited return yards, a team's two inputs in a game are::

    own_success_rate_for = (K + T_s) / (N + T)
    own_ypp_for          = (Y + R) / N

A takeaway is an interception or a lost fumble. It counts as one more play for
the taking team in the success rate, successful when the offense lost expected
points on it (``epa < 0``; exactly 0 or missing is a failure, there is no
tiebreak). Its return yards are credited in yards per play, but it is **not** a
play in that denominator: a team is not charged a play for intercepting the ball.
That asymmetry is the whole of Model v2.1, and it is why the posterior carries
two sample sizes (see :mod:`.posterior`).

A **success** is a play whose expected points added is above zero — nflverse's
own ``success`` flag, never recounted here.

The frame is built in layers, each one adding columns to the last and moving
exactly the column it is named for:

===========================================  ==================================
:func:`team_game_features`                   the plain rate, for and against
:func:`team_game_features_with_yards`        plus yards and the two yards terms
:func:`team_game_features_own`               plus the own-side pair
:func:`team_game_features_fold`              folds the takeaway into both halves
:func:`team_game_features_srtake`            recounts the rate with the takeaway
:func:`team_game_features_foldn`             takes the takeaway back out of N
===========================================  ==================================

An own-side term gets an all-zero ``_against`` column, so the least-squares fit
in :mod:`.posterior` gives it a coefficient of exactly 0 and the design stays
intercept plus two weights.

nflverse play-by-play only: the process meter reads no charting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_simulator.process_meter.plays import (
    EXPLOSIVE_PASS_YARDS,
    EXPLOSIVE_RUN_YARDS,
    GARBAGE_FILTERS,
    WP_BAND,
    credited_return_yards,
    takeaway_mask,
    touchdown_return_mask,
)

#: The per-offense columns the plain frame carries.
OFFENSE_FEATURES = [
    "plays",
    "success_rate",
    "early_down_epa",
    "non_explosive_epa",
    "third_down_rate",
]
KEYS = ["season", "week", "game_id", "posteam", "defteam"]
TEAM_GAME = ["game_id", "posteam"]

FEATURE_SETS = {
    "s1": ("success_rate",),
    "s2": ("success_rate", "early_down_epa", "non_explosive_epa"),
}

#: Game-level magnitude terms: already one side minus the other, so each carries
#: the game value in ``_for`` and a zero pad in ``_against``.
YARDS_TERMS = ("yards_logratio", "yards_ypp_diff")

#: The two inputs of record.
OWN_TERMS = ("own_success_rate", "own_ypp")

# fold -> (the takeaway definition, drop takeaways returned for a touchdown?)
FOLD_RULES = {
    "int": ("int", False),
    "take": ("take", False),
    "int_notd": ("int", True),
    "take_notd": ("take", True),
}
FOLD_COLUMNS = ("fold_takeaways_for", "fold_return_yards_for")

SR_ARM = "sr_take"
FOLD_SR_ARM = "fold_take_sr"
FOLDN_ARM = "foldn_take_sr"

#: The columns :func:`team_game_features_srtake` appends, the sampler's seam
#: among them: integer ``own_n_for`` = N + T and ``own_k_for`` = K + T_s.
SRTAKE_COLUMNS = (
    "srtake_takeaways_for",
    "srtake_successes_for",
    "own_n_for",
    "own_k_for",
    "srtake_shift_for",
)

#: The arm of record and the two inputs it weights. There is no other arm in the
#: library: the research arms that lost live in the research record.
ARMS = {FOLDN_ARM: OWN_TERMS}


def _filtered(plays, garbage_filter):
    """The plays a credit is taken over: the whole game, or the win-probability band."""
    if garbage_filter == "wp05_95":
        return plays[plays.wp.between(*WP_BAND)]
    if garbage_filter != "all":
        raise ValueError(f"garbage_filter must be one of {GARBAGE_FILTERS}, got {garbage_filter!r}")
    return plays


# --- the plain frame --------------------------------------------------------------------


def offense_features(plays) -> pd.DataFrame:
    """One row per offense per game: the process numbers the meter is allowed to read."""
    turnover = (plays.interception.fillna(0) == 1) | (plays.fumble_lost.fillna(0) == 1)
    explosive = ((plays.play_type == "pass") & (plays.yards_gained > EXPLOSIVE_PASS_YARDS)) | (
        (plays.play_type == "run") & (plays.yards_gained > EXPLOSIVE_RUN_YARDS)
    )
    work = plays.assign(
        early_epa=plays.epa.where(plays.down.isin([1, 2])),
        calm_epa=plays.epa.where(~(turnover | explosive)),
        conv=plays.third_down_converted.fillna(0),
        fail=plays.third_down_failed.fillna(0),
    )
    grouped = work.groupby(KEYS, as_index=False).agg(
        plays=("epa", "size"),
        success_rate=("success", "mean"),
        early_down_epa=("early_epa", "mean"),
        non_explosive_epa=("calm_epa", "mean"),
        conv=("conv", "sum"),
        fail=("fail", "sum"),
    )
    attempts = grouped.conv + grouped.fail
    grouped["third_down_rate"] = (grouped.conv / attempts).where(attempts > 0)
    return grouped[KEYS + OFFENSE_FEATURES]


def team_game_features(plays, scores, garbage_filter) -> pd.DataFrame:
    """Team rows: own offense (``_for``), the opponent's (``_against``), the scoreboard."""
    plays = _filtered(plays, garbage_filter)

    offense = offense_features(plays)
    own = offense.rename(columns={f: f"{f}_for" for f in OFFENSE_FEATURES})
    opponent = offense.rename(
        columns={f: f"{f}_against" for f in OFFENSE_FEATURES}
        | {"posteam": "defteam", "defteam": "posteam"}
    )
    frame = own.merge(opponent, on=KEYS, how="left")

    scored = scores[["game_id", "home", "home_score", "away_score"]].rename(
        columns={"home": "home_team"}
    )
    frame = frame.merge(scored, on="game_id", how="inner")
    is_home = frame.posteam == frame.home_team
    frame["home"] = is_home.astype(int)
    frame["points_for"] = np.where(is_home, frame.home_score, frame.away_score).astype(float)
    frame["points_against"] = np.where(is_home, frame.away_score, frame.home_score).astype(float)
    frame["garbage_filter"] = garbage_filter
    frame = frame.drop(columns=["home_team", "home_score", "away_score"])
    return frame.sort_values(["season", "week", "game_id", "posteam"], ignore_index=True)


def team_game_features_with_yards(plays, scores, garbage_filter) -> pd.DataFrame:
    """:func:`team_game_features` plus yards for and against and the two game terms."""
    frame = team_game_features(plays, scores, garbage_filter)
    plays = _filtered(plays, garbage_filter)

    yards = plays.groupby(TEAM_GAME, as_index=False).yards_gained.sum()
    own = yards.rename(columns={"yards_gained": "yards_for"})
    opponent = yards.rename(columns={"posteam": "defteam", "yards_gained": "yards_against"})
    frame = frame.merge(own, on=TEAM_GAME, how="left").merge(
        opponent, on=["game_id", "defteam"], how="left"
    )

    sides = frame[["yards_for", "yards_against"]].to_numpy(float)
    if (sides[~np.isnan(sides)] <= 0).any():
        bad = frame[(frame.yards_for <= 0) | (frame.yards_against <= 0)]
        raise ValueError(
            f"non-positive yards on {len(bad)} team-game rows, the log ratio is undefined: "
            f"{bad[['game_id', 'posteam', 'yards_for', 'yards_against']].head().to_dict('records')}"
        )

    frame["yards_logratio_for"] = np.log(frame.yards_for / frame.yards_against)
    frame["yards_ypp_diff_for"] = (
        frame.yards_for / frame.plays_for - frame.yards_against / frame.plays_against
    )
    for term in YARDS_TERMS:
        frame[f"{term}_against"] = 0.0
    return frame


def team_game_features_own(plays, scores, garbage_filter) -> pd.DataFrame:
    """The frame plus the two own-only terms, each with an all-zero ``_against``."""
    frame = team_game_features_with_yards(plays, scores, garbage_filter)
    frame["own_success_rate_for"] = frame.success_rate_for
    frame["own_ypp_for"] = frame.yards_for / frame.plays_for
    for term in OWN_TERMS:
        frame[f"{term}_against"] = 0.0
    return frame


# --- the fold ---------------------------------------------------------------------------


def folded_yards(plays, arm) -> pd.DataFrame:
    """One row per defense per game with a credited takeaway.

    ``arm`` is a fold name: ``int``, ``take``, ``int_notd`` or ``take_notd``.
    Under a ``_notd`` fold a takeaway returned for a touchdown is dropped whole,
    so it contributes neither a play nor a yard — those six points are already on
    the scoreboard the fit is trained against.
    """
    if arm not in FOLD_RULES:
        raise ValueError(f"arm must be one of {tuple(FOLD_RULES)}, got {arm!r}")
    takeaway_kind, drop_touchdowns = FOLD_RULES[arm]
    mask = takeaway_mask(plays, takeaway_kind)
    if drop_touchdowns:
        mask = mask & ~touchdown_return_mask(plays)
    mask = mask.to_numpy()
    credited = plays.loc[mask, ["game_id", "defteam"]].assign(
        fold_return_yards=credited_return_yards(plays)[mask]
    )
    grouped = credited.groupby(["game_id", "defteam"], as_index=False).agg(
        fold_takeaways=("fold_return_yards", "size"),
        fold_return_yards=("fold_return_yards", "sum"),
    )
    return grouped.rename(columns={"defteam": "team"})[
        ["game_id", "team", "fold_takeaways", "fold_return_yards"]
    ]


def team_game_features_fold(plays, scores, garbage_filter, arm) -> pd.DataFrame:
    """The own frame with ``own_ypp_for`` refolded, plus the two counts behind it.

    The takeaway enters **both** halves here: ``(Y + R) / (N + T)``. The arm of
    record takes it back out of the denominator; see
    :func:`team_game_features_foldn`.
    """
    credit = folded_yards(_filtered(plays, garbage_filter), arm).rename(
        columns={
            "team": "posteam",
            "fold_takeaways": "fold_takeaways_for",
            "fold_return_yards": "fold_return_yards_for",
        }
    )
    frame = team_game_features_own(plays, scores, garbage_filter)
    frame = frame.merge(credit, on=TEAM_GAME, how="left")
    frame["fold_takeaways_for"] = frame.fold_takeaways_for.fillna(0).astype(int)
    frame["fold_return_yards_for"] = frame.fold_return_yards_for.fillna(0.0).astype(float)
    frame["own_ypp_for"] = (frame.yards_for + frame.fold_return_yards_for) / (
        frame.plays_for + frame.fold_takeaways_for
    )
    return frame


# --- the success rate with the takeaway counted -----------------------------------------


def takeaway_successes(plays) -> pd.DataFrame:
    """One row per taking team per game: its takeaways, and those the offense lost EPA on."""
    took = plays[takeaway_mask(plays, "take").to_numpy()]
    counted = took.assign(taker_success=(took.epa < 0).astype(int))
    grouped = counted.groupby(["game_id", "defteam"], as_index=False).agg(
        srtake_takeaways_for=("taker_success", "size"),
        srtake_successes_for=("taker_success", "sum"),
    )
    return grouped.rename(columns={"defteam": "posteam"})[
        [*TEAM_GAME, "srtake_takeaways_for", "srtake_successes_for"]
    ]


def own_success_counts(plays) -> pd.DataFrame:
    """One row per offense per game: its own plays N and own successes K, as integers."""
    return (
        plays.assign(_k=plays.success.fillna(0).astype(int))
        .groupby(TEAM_GAME, as_index=False)
        .agg(own_n_for=("_k", "size"), own_k_for=("_k", "sum"))
    )


def team_game_features_srtake(plays, scores, garbage_filter, arm) -> pd.DataFrame:
    """The fold's frame with the success rate recounted over N + T plays.

    On a team-game with no takeaway the rate is the plain ``success_rate_for``
    exactly, so the shift is exactly 0 there. ``arm`` picks the yards half:
    ``sr_take`` leaves it plain, ``fold_take_sr`` keeps the fold's.
    """
    if arm not in (SR_ARM, FOLD_SR_ARM):
        raise ValueError(f"arm must be one of {(SR_ARM, FOLD_SR_ARM)}, got {arm!r}")
    count_plays = _filtered(plays, garbage_filter)
    frame = team_game_features_fold(plays, scores, garbage_filter, "take")
    frame = frame.merge(takeaway_successes(count_plays), on=TEAM_GAME, how="left").merge(
        own_success_counts(count_plays), on=TEAM_GAME, how="left", validate="one_to_one"
    )
    for column in ("srtake_takeaways_for", "srtake_successes_for"):
        frame[column] = frame[column].fillna(0).astype(int)
    frame["own_n_for"] = (frame.own_n_for + frame.srtake_takeaways_for).astype(int)
    frame["own_k_for"] = (frame.own_k_for + frame.srtake_successes_for).astype(int)
    plain = frame.success_rate_for
    frame["own_success_rate_for"] = (frame.own_k_for / frame.own_n_for).where(
        frame.srtake_takeaways_for > 0, plain
    )
    frame["srtake_shift_for"] = frame.own_success_rate_for - plain
    if arm == SR_ARM:
        frame["own_ypp_for"] = frame.yards_for / frame.plays_for
    return frame


# --- the arm of record ------------------------------------------------------------------


def team_game_features_foldn(plays, scores, garbage_filter) -> pd.DataFrame:
    """The frame of record: the folded rate, and yards per play over own plays only.

    The credit is the fold's — the same takeaway mask, the same credited return
    yards, the same win-probability filter — so only the denominator differs from
    ``fold_take_sr``. Every other column, the success rate and its seam included,
    is untouched.
    """
    frame = team_game_features_srtake(plays, scores, garbage_filter, FOLD_SR_ARM)
    frame["own_ypp_for"] = (frame.yards_for + frame.fold_return_yards_for) / frame.plays_for
    return frame


__all__ = [
    "ARMS",
    "FEATURE_SETS",
    "FOLD_COLUMNS",
    "FOLD_RULES",
    "FOLD_SR_ARM",
    "FOLDN_ARM",
    "KEYS",
    "OFFENSE_FEATURES",
    "OWN_TERMS",
    "SR_ARM",
    "SRTAKE_COLUMNS",
    "TEAM_GAME",
    "YARDS_TERMS",
    "folded_yards",
    "offense_features",
    "own_success_counts",
    "takeaway_successes",
    "team_game_features",
    "team_game_features_fold",
    "team_game_features_foldn",
    "team_game_features_own",
    "team_game_features_srtake",
    "team_game_features_with_yards",
]
