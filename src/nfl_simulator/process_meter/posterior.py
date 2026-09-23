"""Three closed-form posteriors, and the residual scale that turns them into a percent.

The generative story, with beta-hat the least-squares fit on 2016-2023::

    weights  ~ Normal(beta-hat, sigma^2 (X'X)^-1)   fit once, never on the scored game
    rate     ~ Beta(k + 1, n - k + 1)               a team's true within-game success rate
    yards    ~ Normal(y_mean, y_sd / sqrt(n_y))     its true within-game yards per play

    margin draw = home (b_rate * rate + b_ypp * yards) - away (the same)
    percent     = mean over draws of Phi(margin draw / sigma_once)

Every posterior is closed form, so there is no sampler and no convergence to
check. Inference is cut twice on purpose: the weights never see the game being
scored, and a game's production never feeds back into the weights.

**Two sample sizes, not one.** The rate is taken over N + T plays (the team's own
plays plus its takeaways) and the yards mean over N (its own plays alone), so the
posterior carries both: ``n`` for the rate draw and ``n_y`` for the yards draw. A
posterior with no ``n_y`` column narrows both by ``n``, which is what one sample
size always meant, so an older posterior draws exactly what it drew before.

**The residual scale.** ``margin_sd`` was fit to realized rates, so it already
holds their sampling noise; the draws above add that noise a second time.
:func:`noise_once_sd` takes it back out — ``sqrt(margin_sd^2 - the production
variance the draws carry)`` — and the pooled form of that number is the meter's
scale. See ``docs/research/76-sampler-of-record.md``.

nflverse play-by-play only: the process meter reads no charting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from nfl_simulator.process_meter.features import (
    KEYS,
    OWN_TERMS,
    TEAM_GAME,
    _filtered,
)
from nfl_simulator.process_meter.plays import TRAIN_SEASONS

#: The design of record: an intercept and the two own-side inputs. The all-zero
#: ``_against`` pads never enter, because least squares gives them a coefficient
#: of exactly 0.
COLUMNS = ["intercept", "own_success_rate_for", "own_ypp_for"]

#: The four draw columns every production posterior carries.
POSTERIOR_COLUMNS = ["k", "n", "y_mean", "y_sd"]

#: The optional fifth: the number of plays the yards mean was taken over. An arm
#: whose rate and yards are taken over different play counts emits it; one whose
#: counts agree may leave it out.
YARDS_SIZE_COLUMN = "n_y"

#: Every column the draws read, in order. A caller rebuilding a posterior from a
#: scored row must carry all five: dropping ``n_y`` silently narrows the yards
#: draw by the wrong sample size.
DRAW_COLUMNS = [*POSTERIOR_COLUMNS, YARDS_SIZE_COLUMN]

#: The interval reported beside a draw summary: the 89% one.
INTERVAL = (5.5, 94.5)

NOISE_ONCE_MODES = ("pooled", "per_game")
#: Points: the per-game scale never falls below this. The pooled form is never
#: floored, so the floor is reported rather than applied there.
NOISE_ONCE_FLOOR = 5.0

#: How far ``k / n`` may miss the frame's own success rate before the seam raises.
SEAM_LINE = 1e-12


@dataclass(frozen=True)
class CoefficientPosterior:
    """The points regression's weights as a Normal: mean, covariance, residual variance."""

    columns: list
    mean: np.ndarray
    cov: np.ndarray
    sigma2: float
    n_rows: int


@dataclass
class ArmFit:
    """One least-squares fit: the points weights, the margin weights and their scales."""

    features: tuple
    points_coef: dict
    points_r2: float
    margin_coef: dict
    margin_sd: float
    margin_r2: float
    n_train_games: int

    def as_dict(self) -> dict:
        return {
            "features": list(self.features),
            "points_coef": {k: round(v, 3) for k, v in self.points_coef.items()},
            "points_r2": round(self.points_r2, 4),
            "margin_coef": {k: round(v, 3) for k, v in self.margin_coef.items()},
            "margin_sd": round(self.margin_sd, 3),
            "margin_r2": round(self.margin_r2, 4),
            "n_train_games": self.n_train_games,
        }


# --- the design and the fit -------------------------------------------------------------


def points_columns(features) -> list:
    """The ``_for`` columns then the ``_against`` ones, the order the design is built in."""
    return [f"{f}_for" for f in features] + [f"{f}_against" for f in features]


def _live_columns(frame, features) -> list:
    """The design columns that actually vary; the rest are the all-zero pads."""
    columns = points_columns(features)
    variance = frame[columns].astype(float).var()
    return [column for column in columns if variance[column] > 0]


def _ols(design, target):
    design = np.column_stack([np.ones(len(design)), design])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    return beta, target - design @ beta


def _game_pairs(frame) -> pd.DataFrame:
    """Home row beside away row, one line per game."""
    home = frame[frame.home == 1]
    away = frame[frame.home == 0].drop(columns=["season", "week"])
    pairs = home.merge(away, on="game_id", suffixes=("_h", "_a"), validate="one_to_one")
    pairs["result"] = pairs.points_for_h - pairs.points_for_a
    return pairs.rename(columns={"posteam_h": "home", "posteam_a": "away"})


def _training_rows(frame, features, train_seasons) -> pd.DataFrame:
    """The training seasons, then the games whose two team rows are both complete."""
    columns = points_columns(features)
    train = frame[frame.season.isin(list(train_seasons))]
    complete = train.dropna(subset=columns + ["points_for"]).groupby("game_id").size()
    return train[train.game_id.isin(complete[complete == 2].index)]


def fit_arm(frame, features=OWN_TERMS, train_seasons=TRAIN_SEASONS) -> ArmFit:
    """Least squares of points on the two inputs, and of margin on their difference.

    The margin regression is where ``margin_sd`` comes from: the home margin on
    the home-minus-away input differences, so the two teams' errors are never
    assumed independent.
    """
    features = tuple(features)
    columns = points_columns(features)
    train = _training_rows(frame, features, train_seasons)

    target = train.points_for.to_numpy(float)
    beta, resid = _ols(train[columns].to_numpy(float), target)

    pairs = _game_pairs(train)
    diffs = np.column_stack([pairs[f"{f}_for_h"] - pairs[f"{f}_for_a"] for f in features])
    margin = pairs.result.to_numpy(float)
    margin_beta, margin_resid = _ols(diffs, margin)
    dof = len(margin) - (len(features) + 1)

    return ArmFit(
        features=features,
        points_coef=dict(zip(["intercept", *columns], map(float, beta), strict=True)),
        points_r2=float(1 - resid.var() / target.var()),
        margin_coef=dict(
            zip(
                ["intercept", *(f"{f}_diff" for f in features)],
                map(float, margin_beta),
                strict=True,
            )
        ),
        margin_sd=float(np.sqrt(np.sum(margin_resid**2) / dof)),
        margin_r2=float(1 - margin_resid.var() / margin.var()),
        n_train_games=int(len(pairs)),
    )


def with_deserved(frame, arm) -> pd.DataFrame:
    """Team rows plus ``deserved_for``, ``deserved_against`` and ``deserved_margin``."""
    columns = points_columns(arm.features)
    coef = np.array([arm.points_coef[c] for c in columns])
    deserved = arm.points_coef["intercept"] + frame[columns].to_numpy(float) @ coef
    out = frame.assign(deserved_for=deserved)
    opponent = out[["game_id", "posteam", "deserved_for"]].rename(
        columns={"posteam": "defteam", "deserved_for": "deserved_against"}
    )
    out = out.merge(opponent, on=["game_id", "defteam"], how="left")
    out["deserved_margin"] = out.deserved_for - out.deserved_against
    out["actual_margin"] = out.points_for - out.points_against
    return out


def coefficient_posterior(
    frame, features=OWN_TERMS, train_seasons=TRAIN_SEASONS
) -> CoefficientPosterior:
    """The weights as a Normal: the least-squares mean and ``sigma^2 (X'X)^-1``."""
    train = _training_rows(frame, features, train_seasons)
    live = _live_columns(train, features)
    design = np.column_stack([np.ones(len(train)), train[live].to_numpy(float)])
    target = train.points_for.to_numpy(float)
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ beta
    sigma2 = float(resid @ resid / (len(target) - design.shape[1]))
    cov = sigma2 * np.linalg.inv(design.T @ design)
    return CoefficientPosterior(
        columns=["intercept", *live],
        mean=beta,
        cov=(cov + cov.T) / 2,
        sigma2=sigma2,
        n_rows=int(len(target)),
    )


@dataclass(frozen=True)
class LeagueMeans:
    """The average team-game a split is measured against: rate-bar and ypp-bar."""

    rate: float
    ypp: float


def league_means(frame, train_seasons=TRAIN_SEASONS, features=OWN_TERMS) -> LeagueMeans:
    """The two input means over exactly the rows the fit trained on.

    Using the fit's own rows is what makes ``intercept + b_rate * rate-bar +
    b_ypp * ypp-bar`` the league-average deserved points: with an intercept,
    least squares puts the fitted value at the mean input on the mean target.
    """
    train = _training_rows(frame, features, train_seasons)
    return LeagueMeans(
        rate=float(train["own_success_rate_for"].mean()),
        ypp=float(train["own_ypp_for"].mean()),
    )


# --- the production posterior -----------------------------------------------------------


def production_posterior(plays, garbage_filter="all") -> pd.DataFrame:
    """Per team-game ``k`` successes of ``n`` plays, and per-play yards mean and SD.

    The plays and the filter are the frame's, so ``k / n`` is its plain success
    rate and ``y_mean`` its plain yards per play. One play has no spread, so its
    ``y_sd`` is 0.
    """
    plays = _filtered(plays, garbage_filter)
    grouped = plays.groupby(KEYS, as_index=False).agg(
        k=("success", "sum"),
        n=("epa", "size"),
        yards=("yards_gained", "sum"),
        y_sd=("yards_gained", "std"),
    )
    grouped["y_mean"] = grouped.yards / grouped.n
    grouped["y_sd"] = grouped.y_sd.fillna(0.0)
    return grouped[KEYS + POSTERIOR_COLUMNS]


def production_posterior_foldn(plays, frame) -> pd.DataFrame:
    """The arm of record's posterior: folded Beta counts, own-play yards mean and scale.

    ``frame`` is
    :func:`~nfl_simulator.process_meter.features.team_game_features_foldn` on the
    same plays. ``k`` and ``n`` are its ``own_k_for`` and ``own_n_for``, so
    ``k / n`` is the arm's success rate; ``y_mean`` is its ``own_ypp_for``;
    ``n_y`` is the team's own play count, the plays that mean is taken over; and
    ``y_sd`` is the team's own yards SD with no scale factor folded into it.

    Raises when a posterior team-game has no frame row, when ``k / n`` misses the
    frame's success rate, when a count is not a whole number, or when the frame's
    folded count less its takeaways is not the own-play count the posterior read.
    """
    own = production_posterior(plays)
    seam = frame[
        [
            *TEAM_GAME,
            "own_k_for",
            "own_n_for",
            "srtake_takeaways_for",
            "own_success_rate_for",
            "own_ypp_for",
        ]
    ]
    merged = own.merge(seam, on=TEAM_GAME, how="left", validate="one_to_one")
    if merged.own_n_for.isna().any():
        missing = merged.loc[merged.own_n_for.isna(), TEAM_GAME].head(3).to_dict("records")
        raise ValueError(
            f"no foldn_take_sr row for {int(merged.own_n_for.isna().sum())} team-games, "
            f"e.g. {missing}"
        )
    k, n = merged.own_k_for.to_numpy(float), merged.own_n_for.to_numpy(float)
    if not (np.array_equal(k, np.round(k)) and np.array_equal(n, np.round(n))):
        raise ValueError("own_k_for and own_n_for must be whole numbers")
    gap = np.abs(k / n - merged.own_success_rate_for.to_numpy(float))
    if gap.max() > SEAM_LINE:
        raise ValueError(
            f"k / n misses own_success_rate_for by {gap.max():.2e} (line {SEAM_LINE:.0e})"
        )
    own_plays = n - merged.srtake_takeaways_for.to_numpy(float)
    if not np.array_equal(own_plays, merged.n.to_numpy(float)):
        raise ValueError("own_n_for less the takeaways is not the own-play count of the posterior")
    merged[YARDS_SIZE_COLUMN] = own_plays.astype(int)
    merged["y_mean"] = merged.own_ypp_for.to_numpy(float)
    merged["k"], merged["n"] = k.astype(int), n.astype(int)
    return merged[[*own.columns, YARDS_SIZE_COLUMN]]


# --- the residual scale and the percent -------------------------------------------------


def noise_once_sd(margin_var, margin_sd, mode="pooled", floor=NOISE_ONCE_FLOOR):
    """The residual SD with the production variance the draws already carry removed.

    ``pooled`` is one number, ``sqrt(margin_sd^2 - median(margin_var))``, never
    floored. ``per_game`` is ``sqrt(max(margin_sd^2 - margin_var, floor^2))`` per
    game, returned with the count of games that hit the floor.
    """
    if mode not in NOISE_ONCE_MODES:
        raise ValueError(f"mode must be one of {NOISE_ONCE_MODES}, got {mode!r}")
    margin_var = np.asarray(margin_var, dtype=float)
    total = float(margin_sd) ** 2
    if mode == "pooled":
        removed = float(np.median(margin_var))
        if removed >= total:
            raise ValueError(
                f"median production variance {removed:.2f} is not below margin_sd^2 {total:.2f}"
            )
        return float(np.sqrt(total - removed)), 0
    left = total - margin_var
    return np.sqrt(np.maximum(left, floor**2)), int((left < floor**2).sum())


def meter_from_margins(games, margin, residual_sd) -> pd.DataFrame:
    """The home percent and the margin summaries off a kept matrix of margin draws.

    ``residual_sd`` is one number or one per game; it enters only through Phi, so
    it never moves the draws.
    """
    margin = np.asarray(margin, dtype=float)
    scale = np.broadcast_to(np.asarray(residual_sd, dtype=float), (len(margin),))
    lo, hi = np.percentile(margin, INTERVAL, axis=1)
    return pd.DataFrame(
        {
            "game_id": games.game_id.to_numpy(),
            "p_home_bayes": stats.norm.cdf(margin / scale[:, None]).mean(axis=1),
            "margin_mean": margin.mean(axis=1),
            "margin_lo89": lo,
            "margin_hi89": hi,
            "margin_var": margin.var(axis=1),
        }
    )


@dataclass(frozen=True)
class Percent:
    """One residual choice on one matrix of margin draws: the home percent, its scale."""

    p_home: np.ndarray
    sigma: object  # one number (pooled) or one per game
    floor_hits: int = 0


def percent_per_game(games, margin_draws, margin_sd) -> Percent:
    """Each game's own noise-once scale, floored, then the mean of Phi."""
    margin = np.asarray(margin_draws, dtype=float)
    sigma, floored = noise_once_sd(margin.var(axis=1), margin_sd, mode="per_game")
    p = meter_from_margins(games, margin, sigma).p_home_bayes.to_numpy()
    return Percent(p_home=p, sigma=sigma, floor_hits=floored)


def percent_pooled(games, margin_draws, margin_sd, sigma=None) -> Percent:
    """The meter of record: one scale for every game, this matrix's own unless given."""
    margin = np.asarray(margin_draws, dtype=float)
    if sigma is None:
        sigma, _ = noise_once_sd(margin.var(axis=1), margin_sd, mode="pooled")
    p = meter_from_margins(games, margin, sigma).p_home_bayes.to_numpy()
    return Percent(p_home=p, sigma=float(sigma))


def draw_interval(margin_draws, sigma):
    """The 89% interval of each draw's own percent, per game.

    A draw's percent is ``Phi(margin / sigma)``, monotone in the margin, so its
    interval is Phi of the margin interval.
    """
    lo, hi = np.percentile(np.asarray(margin_draws, dtype=float), INTERVAL, axis=1)
    scale = np.broadcast_to(np.asarray(sigma, dtype=float), lo.shape)
    return stats.norm.cdf(lo / scale), stats.norm.cdf(hi / scale)


__all__ = [
    "COLUMNS",
    "DRAW_COLUMNS",
    "INTERVAL",
    "NOISE_ONCE_FLOOR",
    "NOISE_ONCE_MODES",
    "POSTERIOR_COLUMNS",
    "SEAM_LINE",
    "YARDS_SIZE_COLUMN",
    "ArmFit",
    "CoefficientPosterior",
    "LeagueMeans",
    "Percent",
    "coefficient_posterior",
    "draw_interval",
    "fit_arm",
    "league_means",
    "meter_from_margins",
    "noise_once_sd",
    "percent_per_game",
    "percent_pooled",
    "points_columns",
    "production_posterior",
    "production_posterior_foldn",
    "with_deserved",
]
