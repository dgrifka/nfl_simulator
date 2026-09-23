"""The draws of record: one generator per game, 40,000 draws, one pooled scale.

Three constants are rulings, not defaults, and a change to any of them is a
change to every number the meter has ever published:

``SEED_OF_RECORD`` = 0
    with :func:`game_seed`, which hashes the seed and the game id together. A
    game's draws therefore depend only on (seed, game id, draw count) — never on
    its row position or on which games sit beside it, so a game scored alone
    reads the same as the same game scored inside a week's batch.
``N_DRAWS_RECORD`` = 40,000
    the smallest draw count that met the pre-registered stability conditions;
    4,000 and 16,000 were not stable across seeds. See
    ``docs/research/76-sampler-of-record.md``.
``SIGMA_ONCE_RECORD`` = 5.4528
    the pooled noise-once scale measured on this arm's own draws over the 543
    decided 2024-2025 games, from the published three-decimal ``margin_sd``. The
    convention is stated in ``docs/research/75-process-meter-foundations.md``;
    :mod:`.fit` re-measures it on every refit and refuses to write if it moved.

``hashlib.sha256`` and never Python's ``hash()``, which is salted per process.

The weights are drawn per game too, off the game's own generator. Same
expectation, no cross-game correlation.

nflverse play-by-play only: the process meter reads no charting.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from nfl_simulator.process_meter.posterior import (
    COLUMNS,
    DRAW_COLUMNS,
    POSTERIOR_COLUMNS,
    YARDS_SIZE_COLUMN,
    percent_pooled,
)

SEED_OF_RECORD = 0
N_DRAWS_RECORD = 40_000
SIGMA_ONCE_RECORD = 5.4528

#: Points: the largest gap the two-sided points draws may leave against the
#: margin they are meant to sum to.
SIDES_TOLERANCE = 1e-9


def game_seed(game_id: str, seed: int = SEED_OF_RECORD) -> int:
    """A game's generator seed: the first 8 bytes of ``sha256(f"{seed}:{game_id}")``.

    Stable across processes and machines, unlike Python's ``hash()``, which is
    salted per process. The seed offset folds into the same digest, so seeds 0-9
    give one game ten unrelated streams.
    """
    digest = hashlib.sha256(f"{seed}:{game_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def sides(games, prod_post):
    """Rows in ``games`` order, home first then away: (k, n, y_mean, y_sd, n_y).

    A posterior that does not carry ``n_y`` has it filled with ``n``, which is
    what one sample size always meant. Raises on a team-game with no production
    row, and ignores any other column the posterior carries.
    """
    if YARDS_SIZE_COLUMN not in prod_post.columns:
        prod_post = prod_post.assign(**{YARDS_SIZE_COLUMN: prod_post["n"]})
    production = prod_post.set_index(["game_id", "posteam"])[DRAW_COLUMNS]
    out = []
    for side in ("home", "away"):
        keys = pd.MultiIndex.from_arrays([games.game_id, games[side]])
        missing = ~keys.isin(production.index)
        if missing.any():
            raise ValueError(
                f"no production for {int(missing.sum())} {side} team-games, "
                f"e.g. {list(keys[missing][:3])}"
            )
        out.append(production.loc[keys].to_numpy(float))
    return out


def _check_columns(coef_post):
    if list(coef_post.columns) != COLUMNS:
        raise ValueError(f"expected the design columns {COLUMNS}, got {coef_post.columns}")


def margin_draws_per_game(
    games, coef_post, prod_post, n_draws=N_DRAWS_RECORD, seed=SEED_OF_RECORD
) -> np.ndarray:
    """The deserved-margin draws, one row per game in ``games`` order.

    Per game, off ``default_rng(game_seed(game_id, seed))`` and in this order:
    the weight draws, then the home and away rate draws ``Beta(k + 1, n - k +
    1)``, then the home and away yards draws ``Normal(y_mean, y_sd /
    sqrt(n_y))``, then the linear combination. The intercept cancels in the
    margin.

    The rate draw always narrows by ``n``, the plays its rate was taken over. The
    yards draw narrows by ``n_y``, the plays its mean was taken over, which is a
    separate number for the arm of record.
    """
    _check_columns(coef_post)
    home, away = sides(games, prod_post)
    margin = np.empty((len(games), n_draws))
    for i, game_id in enumerate(games.game_id):
        rng = np.random.default_rng(game_seed(game_id, seed))
        beta = rng.multivariate_normal(coef_post.mean, coef_post.cov, size=n_draws)
        b_rate, b_ypp = beta[:, 1], beta[:, 2]
        h, a = home[i], away[i]
        rate_h = rng.beta(h[0] + 1, h[1] - h[0] + 1, size=n_draws)
        rate_a = rng.beta(a[0] + 1, a[1] - a[0] + 1, size=n_draws)
        ypp_h = rng.normal(h[2], h[3] / np.sqrt(h[4]), size=n_draws)
        ypp_a = rng.normal(a[2], a[3] / np.sqrt(a[4]), size=n_draws)
        margin[i] = b_rate * (rate_h - rate_a) + b_ypp * (ypp_h - ypp_a)
    return margin


def points_draws_per_game(
    games, coef_post, prod_post, n_draws=N_DRAWS_RECORD, seed=SEED_OF_RECORD
) -> tuple[np.ndarray, np.ndarray]:
    """(home, away) likely-point draws, each of shape (games, n_draws).

    The same stream as :func:`margin_draws_per_game`, kept as each side's points
    instead of only their difference::

        points = intercept + b_rate * rate + b_ypp * yards per play

    The intercept is the same draw on both sides, so it cancels in ``home -
    away``: :func:`sides_gap` holds that difference to the margin of record
    within :data:`SIDES_TOLERANCE`.
    """
    _check_columns(coef_post)
    home_prod, away_prod = sides(games, prod_post)
    home = np.empty((len(games), n_draws))
    away = np.empty((len(games), n_draws))
    for i, game_id in enumerate(games.game_id):
        rng = np.random.default_rng(game_seed(game_id, seed))
        beta = rng.multivariate_normal(coef_post.mean, coef_post.cov, size=n_draws)
        h, a = home_prod[i], away_prod[i]
        rate_h = rng.beta(h[0] + 1, h[1] - h[0] + 1, size=n_draws)
        rate_a = rng.beta(a[0] + 1, a[1] - a[0] + 1, size=n_draws)
        ypp_h = rng.normal(h[2], h[3] / np.sqrt(h[4]), size=n_draws)
        ypp_a = rng.normal(a[2], a[3] / np.sqrt(a[4]), size=n_draws)
        home[i] = beta[:, 0] + beta[:, 1] * rate_h + beta[:, 2] * ypp_h
        away[i] = beta[:, 0] + beta[:, 1] * rate_a + beta[:, 2] * ypp_a
    return home, away


def sides_gap(home, away, margin) -> float:
    """The largest ``|(home - away) - margin|`` over every game and draw, in points."""
    return float(np.max(np.abs((np.asarray(home) - np.asarray(away)) - np.asarray(margin))))


def share_draws(games, coef_post, prod_post, means, n_draws=N_DRAWS_RECORD, seed=SEED_OF_RECORD):
    """The margin of record split into what the home offense earned and what its defense held.

    Writing rate-bar and ypp-bar for the league means the fit trained on
    (:func:`~nfl_simulator.process_meter.posterior.league_means`), each draw of
    the margin is split at those means::

        offense = b_rate (rate_h - rate-bar) + b_ypp (ypp_h - ypp-bar)
        defense = b_rate (rate-bar - rate_a) + b_ypp (ypp-bar - ypp_a)
        offense + defense = the margin draw, exactly

    The away side is the mirror: its offense share is minus the home defense
    share, and its defense share minus the home offense share. Choosing a
    different origin moves points between the two shares and never moves their
    sum, which is why the means are published beside the numbers.

    This is a **display on the fit of record**, not a model change: the draws are
    :func:`margin_draws_per_game`'s own, element for element, so the split can
    never disagree with the percent. Returns ``(offense_home, defense_home)``.
    """
    _check_columns(coef_post)
    home_prod, away_prod = sides(games, prod_post)
    offense = np.empty((len(games), n_draws))
    defense = np.empty((len(games), n_draws))
    for i, game_id in enumerate(games.game_id):
        rng = np.random.default_rng(game_seed(game_id, seed))
        beta = rng.multivariate_normal(coef_post.mean, coef_post.cov, size=n_draws)
        b_rate, b_ypp = beta[:, 1], beta[:, 2]
        h, a = home_prod[i], away_prod[i]
        rate_h = rng.beta(h[0] + 1, h[1] - h[0] + 1, size=n_draws)
        rate_a = rng.beta(a[0] + 1, a[1] - a[0] + 1, size=n_draws)
        ypp_h = rng.normal(h[2], h[3] / np.sqrt(h[4]), size=n_draws)
        ypp_a = rng.normal(a[2], a[3] / np.sqrt(a[4]), size=n_draws)
        offense[i] = b_rate * (rate_h - means.rate) + b_ypp * (ypp_h - means.ypp)
        defense[i] = b_rate * (means.rate - rate_a) + b_ypp * (means.ypp - ypp_a)
    return offense, defense


def percent_record(games, margin_draws) -> np.ndarray:
    """The percent of record: the pooled percent at :data:`SIGMA_ONCE_RECORD`.

    The scale is given, so this never measures one off the draws in front of it.
    """
    return percent_pooled(games, margin_draws, margin_sd=None, sigma=SIGMA_ONCE_RECORD).p_home


__all__ = [
    "DRAW_COLUMNS",
    "N_DRAWS_RECORD",
    "POSTERIOR_COLUMNS",
    "SEED_OF_RECORD",
    "SIDES_TOLERANCE",
    "SIGMA_ONCE_RECORD",
    "YARDS_SIZE_COLUMN",
    "game_seed",
    "margin_draws_per_game",
    "percent_record",
    "points_draws_per_game",
    "share_draws",
    "sides",
    "sides_gap",
]
