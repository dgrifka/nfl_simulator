"""Simulator v1 — who deserved to win, and by how much.

Implements `docs/research/05-neutralization-principle.md`:

* **§1, the one rule.** Luck is the actual outcome minus its expectation at
  the responsible entity's shrunk rate. Every component uses the same identity;
  only where `p` comes from differs.
* **§3, the treatment table.** Fumble retention is neutralized in full at the
  league rate for the fumble's *class*, on the widened population of document
  18 — every fumble with a resolved disposition, with out of bounds counted as
  the fumbling team keeping the ball. Field goals are neutralized partially,
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

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from nfl_simulator.components import (
    ExtraPointBaseline,
    FieldGoalBaseline,
    FumbleBaseline,
    fg_attempt_mask,
    xp_attempt_mask,
)
from nfl_simulator.dropped_picks import (
    FIRST_CHARTED_SEASON,
    DroppedPickModel,
    event_class_for,
    worthy_throw_frame,
)
from nfl_simulator.fg_model import FieldGoalModel, Weather, sanitize_weather
from nfl_simulator.ledger import Ledger, LedgerEntry
from nfl_simulator.receiver_drops import (
    FIRST_CHARTED_SEASON as RECEIVER_FIRST_CHARTED_SEASON,
)
from nfl_simulator.receiver_drops import (
    ReceiverDropModel,
    catchable_target_frame,
)
from nfl_simulator.receiver_drops import event_class_for as receiver_event_class_for

DEFAULT_POSTERIOR_DRAWS = 400

# Not a performance knob. `dtw_per_draw` is an average over this many coin
# flips, so its spread across posterior draws mixes real uncertainty about `p`
# with Monte Carlo noise from a finite flip count — and the second does not
# belong in a credible interval. `docs/research/10` §8 measured coverage against
# this constant: at 100 the nominal 89% interval covered 97% of informative
# games, and 800 is the smallest swept value that lands inside the
# pre-registered band. Lowering it widens every reported interval.
DEFAULT_COIN_DRAWS = 800

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
    actual: float
    expected_draws: np.ndarray
    swing: float  # already signed to home perspective

    def to_entry(self) -> LedgerEntry:
        """Ledger row at the posterior mean of `p`."""
        return LedgerEntry(
            play_id=self.play_id,
            component=self.component,
            event_class=self.event_class,
            charged_team=self.charged_team,
            actual=self.actual,
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
    # Which adjudication produced these numbers. `"v1.3"` is the shipped ledger;
    # `"v1.3+dp"` says at least one dropped-pick row is in it (document 49 §5),
    # `"v1.3+rd"` at least one receiver-drop row (document 56 §2), and
    # `"v1.3+dp+rd"` — amendment A-3's `v2.0` — both. The label describes the
    # ledger, not the code path: a 2022+ game whose charting held no
    # interceptable throw and no catchable ball is v1.3, because its numbers are.
    variant: str = "v1.3"


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
    """Fumbles, neutralized in full at the league retention rate for their class.

    Full neutralization because document 18 §8 measured `w = 0.015` on the
    widened branch — a team-season's own record of keeping fumbled balls carries
    about one and a half percent of the information about its true rate, so the
    entity term vanishes and `p` is the league's. Class-specific because the
    classes run from 46% to 77%, and a flat coin would be wrong by up to 26
    points on a botched snap.

    The branch is **retention**, not recovery: v1.2 asks whether the fumbling
    team ended up with the ball, so a fumble that crosses the sideline is a kept
    ball rather than a play with no coin in it.
    """
    from nfl_simulator.components import _fumble_frame

    fumbles = _fumble_frame(plays).join(
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
                actual=float(row["retained"]),
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
    *,
    include_blocked: bool = False,
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

    attempts = _fg_frame(plays.filter(fg_attempt_mask(include_blocked)), include_blocked).join(
        baseline.table.select("fg_bin", "swing_value"), on="fg_bin", how="left"
    )

    events = []
    for row in attempts.iter_rows(named=True):
        if row["swing_value"] is None:
            continue
        kicker_season = (
            f"{row['season']}_{row['kicker_player_id']}" if row.get("kicker_player_id") else None
        )
        weather = _weather_for(row)
        draws = fg_model.make_probability(
            kicker_season, float(row["kick_distance"]), weather=weather
        )
        home_sign = 1.0 if row["posteam"] == row["home_team"] else -1.0
        events.append(
            LuckEvent(
                play_id=float(row["play_id"]),
                component="field_goal",
                event_class=f"{int(row['fg_bin'])}-{int(row['fg_bin']) + 4} yd",
                charged_team=row["posteam"],
                actual=float(row["made"]),
                expected_draws=_resample(draws, n_draws, rng),
                swing=float(row["swing_value"]) * home_sign,
            )
        )
    return events


def _weather_for(row: dict) -> Weather | None:
    """Sanitized conditions for one kick, or None when the frame has no weather.

    A frame without `roof`/`wind`/`temp` is not an error — it is a Phase 2 replay,
    and it must reproduce the Phase 2 ledger exactly. Returning None there means
    the model's weather terms never fire.
    """
    if "roof" not in row:
        return None
    return sanitize_weather(row.get("roof"), row.get("wind"), row.get("temp"))


def extra_point_events(
    plays: pl.DataFrame,
    baseline: ExtraPointBaseline | None,
    fg_model: FieldGoalModel | None,
    n_draws: int,
    rng: np.random.Generator,
    *,
    include_blocked: bool = False,
) -> list[LuckEvent]:
    """Extra points, neutralized partially against the kicker's shrunk rate.

    Document 09 §2 gave extra points a branch point — a ball in flight, the same
    structure as a field goal — and §8 measured a 2.422 pp population SD in
    kicker rates against a 1.840 pp null bound, so kickers genuinely differ and
    the treatment is partial rather than full.

    `p` comes from the same hierarchical model the field goals use, which is
    what "folded into the kicker model" means: one `sigma_kicker`, one set of
    per-kicker effects, and an extra-point intercept offset — and since v1.3 the
    offset and the transfer coefficient are actually applied. The EPA swing still
    comes from the empirical branch means, because document 05 §4's layer 1
    draws probabilities, not EPA values.
    """
    if baseline is None or "extra_point_attempt" not in plays.columns:
        return []

    attempts = plays.filter(xp_attempt_mask(include_blocked))
    events = []
    for row in attempts.iter_rows(named=True):
        kicker_season = (
            f"{row['season']}_{row['kicker_player_id']}" if row.get("kicker_player_id") else None
        )
        if fg_model is not None and row.get("kick_distance") is not None:
            # `extra_point=True` selects the fitted extra-point arm — the
            # `delta_xp` offset and the `lambda_xp` transfer of kicker ability.
            # Both were fitted in Phase 3 and neither reached the ledger until
            # v1.3; document 27 §14f sized the omission at −0.98 pp on every
            # extra point in the sample.
            draws = fg_model.make_probability(
                kicker_season,
                float(row["kick_distance"]),
                weather=_weather_for(row),
                extra_point=True,
            )
            expected = _resample(draws, n_draws, rng)
        else:
            # No fitted model: fall back to the league rate, drawn rather than
            # fixed, so layer 1 still carries the rate's own uncertainty.
            expected = _class_rate_draws(baseline.n, baseline.p_make, n_draws, rng)
        home_sign = 1.0 if row["posteam"] == row["home_team"] else -1.0
        events.append(
            LuckEvent(
                play_id=float(row["play_id"]),
                component="extra_point",
                event_class="extra point",
                charged_team=row["posteam"],
                actual=1.0 if row["extra_point_result"] == "good" else 0.0,
                expected_draws=expected,
                swing=baseline.swing_value * home_sign,
            )
        )
    return events


def dropped_pick_events(
    plays: pl.DataFrame,
    ftn: pl.DataFrame | None,
    model: DroppedPickModel | None,
    n_draws: int,
    rng: np.random.Generator,
) -> list[LuckEvent]:
    """The **variant** component: interceptable throws, at the defence's rate.

    Document 49, and nothing about it is v1.3. It fires only when a caller hands
    in both a fitted model and a charting frame, which is why every v1.3 number
    in the repo is reproduced by this function returning an empty list.

    The branch is **escape**, and the offence is charged: ``actual`` is 1 when
    the throw got away, ``expected`` is the posterior probability that it would,
    and ``swing`` is what escaping was worth against being picked. Signs follow
    `fumble_events` exactly — ``actual`` is the good branch for the charged team,
    ``swing`` is its EPA value times the home sign — so a positive ``luck_epa``
    still means good fortune for the home team no matter who threw the ball.

    Coverage is a warning, never an error (document 49 §6, V-4). A pre-2022 game
    asked for the variant gets v1.3, because FTN charting does not reach it; a
    2022+ game whose charting has no worthy throws gets v1.3 too, because there
    was nothing interceptable to adjudicate.
    """
    if model is None or ftn is None:
        return []

    season = int(plays["season"][0])
    if season < FIRST_CHARTED_SEASON:
        warnings.warn(
            f"{plays['game_id'][0]} is a {season} game and FTN charting starts in "
            f"{FIRST_CHARTED_SEASON}; the dropped-pick variant cannot be built for it "
            "and the v1.3 adjudication is returned unchanged.",
            UserWarning,
            stacklevel=2,
        )
        return []

    events = []
    for row in worthy_throw_frame(plays, ftn).iter_rows(named=True):
        catch = model.catch_probability(row["defence_season"], row)
        home_sign = 1.0 if row["posteam"] == row["home_team"] else -1.0
        swing = abs(model.swing_for(row["yardline_100"], row["down"]))
        events.append(
            LuckEvent(
                play_id=float(row["play_id"]),
                component="dropped_pick",
                event_class=event_class_for(row["yardline_100"], row["down"]),
                charged_team=row["posteam"],
                # The escape branch. `1 - catch` rather than a second model
                # call, so the two branches cannot drift apart by a draw.
                actual=0.0 if row["interception"] else 1.0,
                expected_draws=_resample(1.0 - catch, n_draws, rng),
                swing=swing * home_sign,
            )
        )
    return events


def receiver_drop_events(
    plays: pl.DataFrame,
    ftn: pl.DataFrame | None,
    model: ReceiverDropModel | None,
    n_draws: int,
    rng: np.random.Generator,
) -> list[LuckEvent]:
    """The **variant** component: catchable targets, at the receiving corps' rate.

    Document 56, the other direction of amendment A-3's hands-on-the-ball class,
    and nothing about it is v1.3. It fires only when a caller hands in both a
    fitted model and a charting frame.

    The branch is **the catch**, and the offence is charged: ``actual`` is 1 when
    the ball was caught, ``expected`` is the posterior probability that it would
    be, and ``swing`` is what catching it was worth against the incompletion —
    priced from this play's own completion counterfactual where nflfastR supplies
    one. Signs follow `fumble_events` exactly, so a positive ``luck_epa`` still
    means good fortune for the home team no matter who was throwing.

    Note the asymmetry with `dropped_pick_events`, which is the honest outcome of
    amendment A-3 clause 2 rather than a defect: a drop is a 1-in-20 event, so a
    dropped ball books close to a whole swing of bad fortune and a routine catch
    books a twentieth of one the other way.

    Coverage is a warning, never an error (document 56 §2's V-4). A pre-2022 game
    asked for the variant gets v1.3, because FTN charting does not reach it.
    """
    if model is None or ftn is None:
        return []

    season = int(plays["season"][0])
    if season < RECEIVER_FIRST_CHARTED_SEASON:
        warnings.warn(
            f"{plays['game_id'][0]} is a {season} game and FTN charting starts in "
            f"{RECEIVER_FIRST_CHARTED_SEASON}; the receiver-drop variant cannot be "
            "built for it and the v1.3 adjudication is returned unchanged.",
            UserWarning,
            stacklevel=2,
        )
        return []

    events = []
    for row in catchable_target_frame(plays, ftn).iter_rows(named=True):
        catch = model.catch_probability(row["entity_season"], row)
        home_sign = 1.0 if row["posteam"] == row["home_team"] else -1.0
        swing = abs(model.swing_for_play(row))
        events.append(
            LuckEvent(
                play_id=float(row["play_id"]),
                component="receiver_drop",
                event_class=receiver_event_class_for(row["yardline_100"], row["down"]),
                charged_team=row["posteam"],
                actual=0.0 if row["is_drop"] else 1.0,
                expected_draws=_resample(catch, n_draws, rng),
                swing=swing * home_sign,
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


def bootstrap_margins(
    events: Sequence[LuckEvent],
    actual_margin: float,
    points_per_epa: float,
    n_coin_draws: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Document 05 §4's two-layer bootstrap, as a callable.

    Layer 1 is already done — it is the `expected_draws` vector each event
    carries. This runs layer 2 on top of it: at every posterior draw of `p`,
    flip every coin `n_coin_draws` times and recompute the margin.

    Returns ``(margins, dtw_per_draw)`` with shapes
    ``(n_posterior_draws, n_coin_draws)`` and ``(n_posterior_draws,)``.

    Extracted so the interval-coverage check in `docs/research/10` exercises the
    simulator's own arithmetic rather than a re-implementation that could drift
    from it — a calibration check on a copy of the code would prove nothing about
    the code that ships.
    """
    # (posterior draws, events)
    p = np.column_stack([event.expected_draws for event in events])
    swing = np.array([event.swing for event in events])
    actual = np.array([event.actual for event in events])

    uniforms = rng.random((p.shape[0], n_coin_draws, len(events)))
    replayed = (uniforms < p[:, None, :]).astype(float)

    # The adjustment is `actual - replayed`, NOT `replayed - p`. We are
    # replacing the branch that happened with one drawn fairly, so the margin
    # moves by the difference between the two branches. Using the deviation
    # from expectation instead would have mean zero, which would recentre the
    # whole distribution on the actual result and quietly neutralize nothing.
    adjustment = ((actual[None, None, :] - replayed) * swing[None, None, :]).sum(axis=2)
    margins = actual_margin - adjustment * points_per_epa

    # DTW per posterior draw, so the interval is a genuine credible interval on
    # the probability rather than a spread of coin-flip noise.
    return margins, (margins > 0).mean(axis=1)


def simulate_game(
    plays: pl.DataFrame,
    *,
    fumble_baseline: FumbleBaseline,
    fg_baseline: FieldGoalBaseline,
    fg_model: FieldGoalModel | None,
    points_per_epa: float,
    xp_baseline: ExtraPointBaseline | None = None,
    dropped_pick_model: DroppedPickModel | None = None,
    receiver_drop_model: ReceiverDropModel | None = None,
    ftn: pl.DataFrame | None = None,
    n_posterior_draws: int = DEFAULT_POSTERIOR_DRAWS,
    n_coin_draws: int = DEFAULT_COIN_DRAWS,
    seed: int = DEFAULT_SEED,
    include_blocked: bool = False,
) -> SimulationResult:
    """Deserve-to-win for one game.

    `plays` must be the plays of a single game, carrying a `result` column with
    the actual home margin.
    """
    if plays.is_empty():
        raise ValueError("cannot simulate a game with no plays")

    game_id = plays["game_id"][0]
    actual_margin = float(plays["result"][0])
    rng = np.random.default_rng(seed)

    # The fumble builder always receives the unfiltered frame. Four blocked field
    # goals also carry a fumble row, and dropping them here would leave the
    # ledger short — document 26 §8's trap, pinned by a test.
    events = fumble_events(plays, fumble_baseline, n_posterior_draws, rng)
    events += field_goal_events(
        plays, fg_baseline, fg_model, n_posterior_draws, rng, include_blocked=include_blocked
    )
    events += extra_point_events(
        plays, xp_baseline, fg_model, n_posterior_draws, rng, include_blocked=include_blocked
    )
    # Last, and after the shared random stream has been drawn from by every v1.3
    # builder. Appending rather than interleaving is what keeps `None` byte-for-
    # byte identical to v1.3: the draws the fumble and kicking coins consume do
    # not move when the variant is switched on.
    dropped_picks = dropped_pick_events(plays, ftn, dropped_pick_model, n_posterior_draws, rng)
    events += dropped_picks
    # And the receiver direction after that, for the same reason: appending keeps
    # `+dp` byte-for-byte identical whether or not `+rd` is switched on, so the
    # two variants can be read alone and together without one moving the other.
    receiver_drops = receiver_drop_events(plays, ftn, receiver_drop_model, n_posterior_draws, rng)
    events += receiver_drops
    variant = "v1.3" + ("+dp" if dropped_picks else "") + ("+rd" if receiver_drops else "")

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
            variant=variant,
        )

    margins, dtw_per_draw = bootstrap_margins(
        events, actual_margin, points_per_epa, n_coin_draws, rng
    )

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
        variant=variant,
    )
