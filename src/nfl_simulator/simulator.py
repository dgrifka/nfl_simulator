"""Simulator v1 — who deserved to win, and by how much.

Implements `docs/research/05-neutralization-principle.md`:

* **§1, the one rule.** Luck is the realized outcome minus its expectation at
  the responsible entity's shrunk rate. Every component uses the same identity;
  only where `p` comes from differs.
* **§3, the treatment table.** Fumble recovery is neutralized in full at the
  league rate for the fumble's *class*. Field goals are neutralized partially,
  against that kicker's shrunk make probability. Nothing else is touched —
  penalties and return yardage failed the branch-point gate, and step 3a could
  not attribute the interception spread to an entity.
* **§4, the two-layer bootstrap.** Layer 1 draws `p` from its posterior; layer 2
  flips the coin at that `p`. Reporting a deserve-to-win interval from layer 2
  alone would hide the fact that `p` is itself estimated.

The output is anchored on what actually happened:

    deserved_margin = actual_margin - total_luck_epa * points_per_epa

so a game with no luck events returns its own result exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nfl_simulator.components import (
    FieldGoalBaseline,
    FumbleBaseline,
    fg_attempt_mask,
    live_fumble_mask,
)
from nfl_simulator.fg_model import FieldGoalModel
from nfl_simulator.ledger import Ledger, LedgerEntry

DEFAULT_POSTERIOR_DRAWS = 400
DEFAULT_COIN_DRAWS = 200
DEFAULT_SEED = 20260817

# Document 03's convention, carried through Phase 2.
ETI_LOW, ETI_HIGH = 5.5, 94.5


@dataclass(frozen=True)
class LuckEvent:
    """One neutralizable branch, with `p` as a vector of posterior draws."""

    play_id: float
    component: str
    event_class: str
    charged_team: str
    realized: float
    expected_draws: np.ndarray
    swing: float  # already signed to home perspective

    def to_entry(self) -> LedgerEntry:
        """Ledger row at the posterior mean of `p`."""
        return LedgerEntry(
            play_id=self.play_id,
            component=self.component,
            event_class=self.event_class,
            charged_team=self.charged_team,
            realized=self.realized,
            expected=float(self.expected_draws.mean()),
            swing=self.swing,
        )


@dataclass(frozen=True)
class SimulationResult:
    game_id: str
    actual_margin: float
    deserved_margin: float
    dtw_home: float
    dtw_interval: tuple[float, float]
    margin_draws: np.ndarray
    ledger: Ledger
    total_luck_epa: float


def points_per_epa(games: pl.DataFrame) -> float:
    """OLS slope of points margin on EPA differential.

    Document 01 measured r = 0.996 between the two, so this is a tight
    conversion rather than a fitted relationship doing real work. It exists so
    the answer can be stated on the scoreboard's scale.
    """
    variance = games["epa_diff"].var(ddof=1)
    if not variance:
        raise ValueError("`epa_diff` has zero variance; cannot fit a points-per-EPA slope")
    covariance = games.select(pl.cov(pl.col("epa_diff"), pl.col("margin")).alias("c")).item()
    return float(covariance / variance)


# --------------------------------------------------------------------------
# event extraction
# --------------------------------------------------------------------------


def _class_rate_draws(n: float, p: float, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """Posterior draws for a league class rate estimated from `n` events.

    Jeffreys prior, so a class observed 946 times contributes a much tighter
    rate than one observed 68 times. Document 05 §4 layer 1 asks for the *rate*
    to be drawn, and a league class rate is still an estimate even though it is
    not a team's.
    """
    successes = max(n * p, 0.0)
    failures = max(n - successes, 0.0)
    return rng.beta(successes + 0.5, failures + 0.5, size=n_draws)


def fumble_events(
    plays: pl.DataFrame,
    baseline: FumbleBaseline,
    n_draws: int,
    rng: np.random.Generator,
) -> list[LuckEvent]:
    """Live fumbles, neutralized in full at the league rate for their class.

    Full neutralization because document 04 measured `w = 0.011` — a
    team-season's own recovery record carries about one percent of the
    information about its true rate, so the entity term vanishes and `p` is the
    league's. Class-specific because the classes run from 40% to 76%, and a flat
    coin would be wrong by up to 26 points on a botched snap.
    """
    from nfl_simulator.components import _fumble_frame

    fumbles = _fumble_frame(plays.filter(live_fumble_mask())).join(
        baseline.table.select("fumble_class", "n", "p_own", "swing_value"),
        on="fumble_class",
        how="left",
    )

    events = []
    for row in fumbles.iter_rows(named=True):
        if row["p_own"] is None or row["swing_value"] is None:
            continue
        home_sign = 1.0 if row["fumbled_1_team"] == row["home_team"] else -1.0
        events.append(
            LuckEvent(
                play_id=float(row["play_id"]),
                component="fumble",
                event_class=row["fumble_class"],
                charged_team=row["fumbled_1_team"],
                realized=float(row["recovered_own"]),
                expected_draws=_class_rate_draws(
                    float(row["n"]), float(row["p_own"]), n_draws, rng
                ),
                swing=float(row["swing_value"]) * home_sign,
            )
        )
    return events


def field_goal_events(
    plays: pl.DataFrame,
    baseline: FieldGoalBaseline,
    fg_model: FieldGoalModel | None,
    n_draws: int,
    rng: np.random.Generator,
) -> list[LuckEvent]:
    """Field goals, neutralized partially against the kicker's shrunk rate.

    `p` comes from the hierarchical model of `docs/research/05b`, which puts the
    shrinkage weight at 0.285 for a median kicker-season — so a kicker keeps
    roughly a quarter of their own record and borrows the rest from the league
    curve. The EPA swing still comes from the empirical distance bin, because
    document 05 §4's layer 1 draws probabilities, not EPA values.
    """
    if fg_model is None:
        return []

    from nfl_simulator.components import _fg_frame

    attempts = _fg_frame(plays.filter(fg_attempt_mask())).join(
        baseline.table.select("fg_bin", "swing_value"), on="fg_bin", how="left"
    )

    events = []
    for row in attempts.iter_rows(named=True):
        if row["swing_value"] is None:
            continue
        kicker_season = (
            f"{row['season']}_{row['kicker_player_id']}" if row.get("kicker_player_id") else None
        )
        draws = fg_model.make_probability(kicker_season, float(row["kick_distance"]))
        home_sign = 1.0 if row["posteam"] == row["home_team"] else -1.0
        events.append(
            LuckEvent(
                play_id=float(row["play_id"]),
                component="field_goal",
                event_class=f"{int(row['fg_bin'])}-{int(row['fg_bin']) + 4} yd",
                charged_team=row["posteam"],
                realized=float(row["made"]),
                expected_draws=_resample(draws, n_draws, rng),
                swing=float(row["swing_value"]) * home_sign,
            )
        )
    return events


def _resample(draws: np.ndarray, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """Match a posterior to the bootstrap's draw count.

    Subsampling rather than taking a mean: collapsing here would discard exactly
    the layer-1 uncertainty the bootstrap exists to propagate.
    """
    if len(draws) == n_draws:
        return draws
    return draws[rng.integers(0, len(draws), size=n_draws)]


# --------------------------------------------------------------------------
# the simulator
# --------------------------------------------------------------------------


def simulate_game(
    plays: pl.DataFrame,
    *,
    fumble_baseline: FumbleBaseline,
    fg_baseline: FieldGoalBaseline,
    fg_model: FieldGoalModel | None,
    points_per_epa: float,
    n_posterior_draws: int = DEFAULT_POSTERIOR_DRAWS,
    n_coin_draws: int = DEFAULT_COIN_DRAWS,
    seed: int = DEFAULT_SEED,
) -> SimulationResult:
    """Deserve-to-win for one game.

    `plays` must be the plays of a single game, carrying a `result` column with
    the realized home margin.
    """
    if plays.is_empty():
        raise ValueError("cannot simulate a game with no plays")

    game_id = plays["game_id"][0]
    actual_margin = float(plays["result"][0])
    rng = np.random.default_rng(seed)

    events = fumble_events(plays, fumble_baseline, n_posterior_draws, rng)
    events += field_goal_events(plays, fg_baseline, fg_model, n_posterior_draws, rng)

    ledger = Ledger(tuple(event.to_entry() for event in events))
    total_luck_epa = ledger.total_luck_epa()
    deserved_margin = actual_margin - total_luck_epa * points_per_epa

    if not events:
        # Nothing to adjudicate. The distribution is degenerate at the actual
        # result, and DTW is 1 or 0 — correctly, since no coin was involved.
        dtw = 1.0 if actual_margin > 0 else 0.0
        return SimulationResult(
            game_id=game_id,
            actual_margin=actual_margin,
            deserved_margin=deserved_margin,
            dtw_home=dtw,
            dtw_interval=(dtw, dtw),
            margin_draws=np.full(1, actual_margin),
            ledger=ledger,
            total_luck_epa=total_luck_epa,
        )

    # (posterior draws, events)
    p = np.column_stack([event.expected_draws for event in events])
    swing = np.array([event.swing for event in events])
    realized = np.array([event.realized for event in events])

    # Layer 2: flip every coin `n_coin_draws` times at each posterior draw's p.
    uniforms = rng.random((n_posterior_draws, n_coin_draws, len(events)))
    replayed = (uniforms < p[:, None, :]).astype(float)

    # The adjustment is `realized - replayed`, NOT `replayed - p`. We are
    # replacing the branch that happened with one drawn fairly, so the margin
    # moves by the difference between the two branches. Using the deviation
    # from expectation instead would have mean zero, which would recentre the
    # whole distribution on the actual result and quietly neutralize nothing.
    adjustment = ((realized[None, None, :] - replayed) * swing[None, None, :]).sum(axis=2)
    margins = actual_margin - adjustment * points_per_epa

    # DTW per posterior draw, so the interval is a genuine credible interval on
    # the probability rather than a spread of coin-flip noise.
    dtw_per_draw = (margins > 0).mean(axis=1)

    return SimulationResult(
        game_id=game_id,
        actual_margin=actual_margin,
        deserved_margin=deserved_margin,
        dtw_home=float(dtw_per_draw.mean()),
        dtw_interval=(
            float(np.percentile(dtw_per_draw, ETI_LOW)),
            float(np.percentile(dtw_per_draw, ETI_HIGH)),
        ),
        margin_draws=margins.ravel(),
        ledger=ledger,
        total_luck_epa=total_luck_epa,
    )
