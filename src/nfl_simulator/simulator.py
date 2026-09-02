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
from collections.abc import Callable, Sequence
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

# Document 73 §3's rule, as a key. The two variant builders below read their
# frame's *row position* twice over: `_resample` hands out one block of posterior
# indices per event in iteration order, and `_replayed_adjustment` then
# column-indexes its coin uniforms by each event's position in the sequence. The
# frames they iterate come out of an inner join, and an inner join makes no order
# promise — Polars' hash join emits the order the charting frame arrived in, so
# two charting pulls that agree on every value row for row still adjudicate
# differently (document 73 §1: 0 of 47,316 rows differ, the margin moves
# 1.14e-06 pt). Sorting to a key with no ties makes the adjudication a function
# of the data's values rather than of the order they were handed over in.
TOTAL_ORDER = ("game_id", "play_id")


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


# The two editions ruling R-4 named (document 58 §2), mapped to the public name
# each carries on a figure. `"strict+dp"` and `"strict+rd"` are deliberately
# absent: they are audit arms, and `SimulationResult.edition` returns `None` for
# them so nothing can render an arm the maintainer never named.
PUBLIC_EDITIONS = {"strict": "Strict", "full": "Full"}
EDITIONS = tuple(PUBLIC_EDITIONS)


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
    # The same bootstrap, split into the two teams' deserved points. Optional
    # and defaulted to ``None`` so nothing that already reads this dataclass
    # changes shape: a caller with no scoreboard — the summary artifacts carry
    # margins, not scores — still gets every field it had before. When they are
    # present, ``home_point_draws - away_point_draws`` is ``margin_draws``
    # exactly, because the split comes from the same replay rather than from a
    # second one.
    home_point_draws: np.ndarray | None = None
    away_point_draws: np.ndarray | None = None
    # Which adjudication produced these numbers. `"strict"` is v1.4's ledger,
    # renamed by ruling R-4 (document 58 §2); `"full"` is the other edition —
    # at least one dropped-pick row (document 49 §5) *and* at least one
    # receiver-drop row (document 56 §2), the two directions amendment A-3
    # clause 3 admits together or not at all. `"strict+dp"` and `"strict+rd"`
    # are the audit-only arms: callable, but they have no public name and never
    # render. The label describes the ledger, not the code path — a 2022+ game
    # whose charting held no interceptable throw and no catchable ball is
    # `"strict"`, because its numbers are.
    variant: str = "strict"
    # The events this adjudication was built from, kept because they carry the
    # one thing the ledger drops: `expected_draws`, the whole posterior on each
    # branch. `LedgerEntry` stores its mean, which is the number the arithmetic
    # needs and is all the shipped artifacts hold — so a figure that wants to
    # say how *sure* a probability was has nowhere else to read it. Defaulted to
    # empty so nothing that already reads this dataclass changes shape.
    events: tuple[LuckEvent, ...] = ()

    @property
    def edition(self) -> str | None:
        """The public name of this adjudication, or ``None`` for an audit arm."""
        return PUBLIC_EDITIONS.get(self.variant)


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
            kicker_season,
            float(row["kick_distance"]),
            weather=weather,
            stadium_id=row.get("stadium_id"),
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

    `stadium_id` — v1.4's elevation covariate — is read the same way, with
    `row.get`, and for the same reason: a frame that predates the column prices
    every kick at the fitted mean elevation, which is exactly what v1.3 did.
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
                stadium_id=row.get("stadium_id"),
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

    Document 49, and nothing about it is in the Strict edition. It fires only when
    a caller hands in both a fitted model and a charting frame, which is why every
    Strict number in the repo — v1.1 through v1.4 — is reproduced by this function
    returning an empty list.

    The branch is **escape**, and the offence is charged: ``actual`` is 1 when
    the throw got away, ``expected`` is the posterior probability that it would,
    and ``swing`` is what escaping was worth against being picked. Signs follow
    `fumble_events` exactly — ``actual`` is the good branch for the charged team,
    ``swing`` is its EPA value times the home sign — so a positive ``luck_epa``
    still means good fortune for the home team no matter who threw the ball.

    Coverage is a warning, never an error (document 49 §6, V-4). A pre-2022 game
    asked for the variant gets the Strict adjudication, because FTN charting does
    not reach it; a 2022+ game whose charting has no worthy throws gets it too,
    because there was nothing interceptable to adjudicate.
    """
    if model is None or ftn is None:
        return []

    season = int(plays["season"][0])
    if season < FIRST_CHARTED_SEASON:
        warnings.warn(
            f"{plays['game_id'][0]} is a {season} game and FTN charting starts in "
            f"{FIRST_CHARTED_SEASON}; the dropped-pick variant cannot be built for it "
            "and the Strict adjudication is returned unchanged.",
            UserWarning,
            stacklevel=2,
        )
        return []

    events = []
    # Sorted before a single draw is taken — document 73 §3, and see `TOTAL_ORDER`.
    for row in worthy_throw_frame(plays, ftn).sort(TOTAL_ORDER).iter_rows(named=True):
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
    and nothing about it is in the Strict edition. It fires only when a caller
    hands in both a fitted model and a charting frame.

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
    asked for the variant gets the Strict adjudication, because FTN charting does
    not reach it.
    """
    if model is None or ftn is None:
        return []

    season = int(plays["season"][0])
    if season < RECEIVER_FIRST_CHARTED_SEASON:
        warnings.warn(
            f"{plays['game_id'][0]} is a {season} game and FTN charting starts in "
            f"{RECEIVER_FIRST_CHARTED_SEASON}; the receiver-drop variant cannot be "
            "built for it and the Strict adjudication is returned unchanged.",
            UserWarning,
            stacklevel=2,
        )
        return []

    events = []
    # Sorted before a single draw is taken — document 73 §3, and see `TOTAL_ORDER`.
    for row in catchable_target_frame(plays, ftn).sort(TOTAL_ORDER).iter_rows(named=True):
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


def _replayed_adjustment(
    events: Sequence[LuckEvent], n_coin_draws: int, rng: np.random.Generator
) -> np.ndarray:
    """One replay of every coin, kept per event rather than summed.

    Shape ``(n_posterior_draws, n_coin_draws, n_events)``. Layer 1 is already
    done — it is the `expected_draws` vector each event carries. This is layer
    2: at every posterior draw of `p`, flip every coin `n_coin_draws` times.

    The per-event shape is what lets the margin and the two teams' deserved
    points come out of **one** replay. Summing here and re-drawing for the
    split would run two RNG streams over the same coins, and the two figures
    would then be showing two different adjudications of the same game.

    The adjustment is `actual - replayed`, NOT `replayed - p`. We are replacing
    the branch that happened with one drawn fairly, so the margin moves by the
    difference between the two branches. Using the deviation from expectation
    instead would have mean zero, which would recentre the whole distribution
    on the actual result and quietly neutralize nothing.
    """
    # (posterior draws, events)
    p = np.column_stack([event.expected_draws for event in events])
    swing = np.array([event.swing for event in events])
    actual = np.array([event.actual for event in events])

    uniforms = rng.random((p.shape[0], n_coin_draws, len(events)))
    replayed = (uniforms < p[:, None, :]).astype(float)
    return (actual[None, None, :] - replayed) * swing[None, None, :]


# Document 61 §5's row: the possession cap is a ledger entry like any other, so
# the waterfall can draw it and the round trip still closes on it.
POSSESSION_CAP_COMPONENT = "possession_cap"


def _cap_entry(label: str, luck_epa: float, *, offence: str, play_id: float) -> LedgerEntry:
    """One possession's clip, as a ledger row.

    A cap row is not a branch — nothing was flipped, a sum was bounded — so the
    three columns `LedgerEntry` insists on are used to state exactly that: the
    clip is certain (`actual = 1`) against a counterfactual in which the drive
    kept everything it booked (`expected = 0`), and the whole row is worth
    `swing`. The module's identity, `luck_epa = (actual − expected) × swing`,
    then reads literally true of a cap row as it does of a fumble, which is what
    keeps `Ledger.total_luck_epa` a sum nobody has to special-case.

    `play_id` is the drive's largest-swing event — the play that *set* `C_d` —
    so a reader who follows the row back lands on the "what if" the bound was
    taken from rather than on an arbitrary snap.
    """
    return LedgerEntry(
        play_id=play_id,
        component=POSSESSION_CAP_COMPONENT,
        event_class=label,
        charged_team=offence,
        actual=1.0,
        expected=0.0,
        swing=luck_epa,
    )


def _apply_possession_cap(
    adjustment: np.ndarray,
    events: Sequence[LuckEvent],
    drive_of: Callable[[LuckEvent], object],
) -> dict[object, float]:
    """Document 61 §2's clip, applied **in place** to a replayed adjustment.

    Per replicate, a possession's booked luck is bounded by the largest single
    "what if" on it::

        A_d = clip( Σ_i a_i , −C_d , +C_d )      C_d = max_i |swing_i|

    Two facts make this the right shape rather than an arbitrary shrink.

    **It is applied per replicate, not to the point estimate.** Document 61 §0's
    defect is two-headed: the sum over-counts, *and* flipping every event
    independently makes the deserved-margin distribution wider than a possession
    could ever be. Clipping inside the bootstrap answers both at once.

    **It is applied proportionally within the drive.** The excess is not taken
    off one nominated event; every event on the possession is scaled by the same
    per-replicate factor ``clipped / total`` in ``[0, 1]``. That is what lets
    :func:`_split_points` keep working on the result — ``home − away`` is still
    the margin this same replay produced — and it is why the clip can never grow
    a replicate or turn one team's good fortune into the other's.

    A one-event possession is skipped outright: ``|a_i| ≤ |swing_i| = C_d`` is
    arithmetic, so document 61 §6's P-5 is a property of the code rather than a
    result it happens to produce. A possession no replicate clips books no row.

    Returns ``{drive key: mean(clipped) − mean(unclipped)}`` — document 61 §5's
    cap row, one per bitten possession, in event order.
    """
    members: dict[object, list[int]] = {}
    for index, event in enumerate(events):
        members.setdefault(drive_of(event), []).append(index)

    cap_rows: dict[object, float] = {}
    for key, indices in members.items():
        if len(indices) == 1:
            continue
        cap = max(abs(events[index].swing) for index in indices)
        total = adjustment[:, :, indices].sum(axis=2)
        clipped = np.clip(total, -cap, cap)
        if np.array_equal(clipped, total):
            continue
        scale = np.divide(clipped, total, out=np.ones_like(total), where=total != 0.0)
        adjustment[:, :, indices] *= scale[:, :, None]
        cap_rows[key] = float(clipped.mean() - total.mean())
    return cap_rows


def _bootstrap(
    events: Sequence[LuckEvent],
    actual_margin: float,
    points_per_epa: float,
    n_coin_draws: int,
    rng: np.random.Generator,
    *,
    drive_of: Callable[[LuckEvent], object] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[object, float]]:
    """The bootstrap, plus the per-event adjustment it was computed from."""
    adjustment = _replayed_adjustment(events, n_coin_draws, rng)
    # The cap runs after the whole replay and consumes no draws of its own, which
    # is what makes `drive_of=None` the function this was before it (P-2).
    cap_rows = {} if drive_of is None else _apply_possession_cap(adjustment, events, drive_of)
    margins = actual_margin - adjustment.sum(axis=2) * points_per_epa
    # DTW per posterior draw, so the interval is a genuine credible interval on
    # the probability rather than a spread of coin-flip noise.
    return margins, (margins > 0).mean(axis=1), adjustment, cap_rows


def bootstrap_margins(
    events: Sequence[LuckEvent],
    actual_margin: float,
    points_per_epa: float,
    n_coin_draws: int,
    rng: np.random.Generator,
    *,
    drive_of: Callable[[LuckEvent], object] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Document 05 §4's two-layer bootstrap, as a callable.

    Returns ``(margins, dtw_per_draw)`` with shapes
    ``(n_posterior_draws, n_coin_draws)`` and ``(n_posterior_draws,)``.

    Extracted so the interval-coverage check in `docs/research/10` exercises the
    simulator's own arithmetic rather than a re-implementation that could drift
    from it — a calibration check on a copy of the code would prove nothing about
    the code that ships.

    ``drive_of`` maps an event to the possession it sat on and switches on
    document 61's cap (:func:`_apply_possession_cap`). Left ``None`` — the
    default, and what the Strict edition always passes — this is the function it
    was before round 9, draw for draw.
    """
    margins, dtw_per_draw, _adjustment, _cap_rows = _bootstrap(
        events, actual_margin, points_per_epa, n_coin_draws, rng, drive_of=drive_of
    )
    return margins, dtw_per_draw


def _split_points(
    adjustment: np.ndarray,
    events: Sequence[LuckEvent],
    home_team: str,
    home_points: float,
    away_points: float,
    points_per_epa: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The two teams' deserved points, from an adjustment already replayed.

    A team's deserved points are its actual points minus the luck booked on
    **its own** plays. `swing` is signed to the home team's margin throughout,
    so a home-charged event's adjustment comes off the home score and an
    away-charged one's goes onto the away score — the two signs that make
    ``home - away`` the margin the same replay produced.
    """
    home_mask = np.array([event.charged_team == home_team for event in events], dtype=bool)
    home_luck = adjustment[:, :, home_mask].sum(axis=2)
    away_luck = adjustment[:, :, ~home_mask].sum(axis=2)
    return (
        (home_points - home_luck * points_per_epa).ravel(),
        (away_points + away_luck * points_per_epa).ravel(),
    )


def team_point_draws(
    events: Sequence[LuckEvent],
    *,
    home_team: str,
    home_points: float,
    away_points: float,
    points_per_epa: float,
    n_coin_draws: int,
    rng: np.random.Generator,
    drive_of: Callable[[LuckEvent], object] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Each team's deserved points across the bootstrap, on the scoreboard's scale.

    The same two-layer bootstrap :func:`bootstrap_margins` runs, split by the
    team each event is charged to. Called with the same ``rng`` state it
    reproduces that function's margins draw for draw, because both consume one
    replay of the same coins in the same order.

    A game with nothing to adjudicate deserved the score it got, so it returns
    the two actual scores as single draws — the degenerate case
    :func:`simulate_game` already draws for the margin.
    """
    if not events:
        return np.full(1, float(home_points)), np.full(1, float(away_points))
    adjustment = _replayed_adjustment(events, n_coin_draws, rng)
    if drive_of is not None:
        _apply_possession_cap(adjustment, events, drive_of)
    return _split_points(adjustment, events, home_team, home_points, away_points, points_per_epa)


def possessions(plays: pl.DataFrame) -> tuple[dict[float, str], dict[str, str]]:
    """One game's plays, read as possessions: `play_id -> label` and `label -> offence`.

    Document 61 §2 groups events by ``(game_id, fixed_drive)``; `plays` is
    already one game, so `fixed_drive` alone is the key. The label a cap row
    wears is `"Q3 drive 7"`, and it is computed once per drive from that drive's
    **first** play rather than per play — a drive that crosses a quarter break
    is one possession, and labelling its plays separately would split it.

    The offence is likewise the team possessing on the drive's first play — its
    first play that **has** one. A kickoff carries a null `posteam`, so reading
    the first row outright leaves the opening drive of a game charged to nobody,
    and `LedgerEntry` would then carry a `None` where a club belongs. It is also
    not always the only `posteam` on the drive: a would-be touchdown dropped and
    returned for a score puts the other team's extra point on the same
    `fixed_drive`. A drive with no `posteam` at all is left out, and the caller
    falls back to the charged team of the event that set `C_d`.

    A frame without `fixed_drive` returns empty maps, and the caller then leaves
    every event on a possession of its own — document 61 §2's guard, under which
    the cap is inert by P-5 rather than silently wrong.
    """
    if "fixed_drive" not in plays.columns:
        return {}, {}
    columns = ["play_id", "fixed_drive", "posteam"]
    if "qtr" in plays.columns:
        columns.append("qtr")
    ordered = plays.select(columns).sort("play_id")

    label_by_drive: dict[object, str] = {}
    offence: dict[str, str] = {}
    for row in ordered.iter_rows(named=True):
        drive = row["fixed_drive"]
        if drive is None:
            continue
        if drive not in label_by_drive:
            quarter = row.get("qtr")
            label_by_drive[drive] = (
                f"drive {int(drive)}" if quarter is None else f"Q{int(quarter)} drive {int(drive)}"
            )
        label = label_by_drive[drive]
        if label not in offence and row["posteam"] is not None:
            offence[label] = row["posteam"]

    labels = {
        float(row["play_id"]): label_by_drive[row["fixed_drive"]]
        for row in ordered.iter_rows(named=True)
        if row["fixed_drive"] is not None
    }
    return labels, offence


def _possession_cap_handles(
    plays: pl.DataFrame, events: Sequence[LuckEvent]
) -> tuple[Callable[[LuckEvent], object], dict[str, str]]:
    """`drive_of` for :func:`_apply_possession_cap`, and the offence each label names."""
    labels, offence = possessions(plays)
    if not labels:
        warnings.warn(
            f"{plays['game_id'][0]} carries no `fixed_drive`, so the Full edition's "
            "possession cap (document 61) has no possessions to group by and every "
            "event is left on one of its own. Load `fixed_drive` with the play-by-play "
            "to switch the cap on.",
            UserWarning,
            stacklevel=3,
        )

    def drive_of(event: LuckEvent) -> object:
        # An event whose play is not in the frame's drive map keeps a key of its
        # own — document 61 §2's guard. Two events on the *same* play share one,
        # because they are on the same possession by construction.
        return labels.get(float(event.play_id), ("no-drive", float(event.play_id)))

    return drive_of, offence


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
    home_points: float | None = None,
    away_points: float | None = None,
    edition: str | None = None,
) -> SimulationResult:
    """Deserve-to-win for one game.

    `plays` must be the plays of a single game, carrying a `result` column with
    the actual home margin.

    ``home_points`` and ``away_points`` are the scoreboard, and they are
    optional because the adjudication does not need them: the margin is what
    the simulator decides. Given them, the result also carries each team's
    deserved points, split out of the same replay the margin came from.
    `edition` is a switch over the model handles, not a second code path. A
    caller holding both fitted models — `render._simulation_context` does —
    asks for `"strict"` to get v1.4 without dropping them, and for `"full"` to
    use both. Left `None`, whichever models were passed are used, which is how
    the audit arms `"strict+dp"` and `"strict+rd"` stay reachable.
    """
    if edition is not None and edition not in EDITIONS:
        raise ValueError(f"unknown edition {edition!r}; the editions are {list(EDITIONS)}")
    if edition == "strict":
        dropped_pick_model = None
        receiver_drop_model = None
    if plays.is_empty():
        raise ValueError("cannot simulate a game with no plays")

    game_id = plays["game_id"][0]
    actual_margin = float(plays["result"][0])
    rng = np.random.default_rng(seed)

    # The two scores and the margin are the same fact stated twice. A caller
    # that hands over a pair which does not subtract to `result` has fetched the
    # wrong game's scoreboard, and the per-team distributions would then be
    # drawn around scores the margin underneath them does not belong to.
    if home_points is not None and away_points is not None:
        gap = float(home_points) - float(away_points) - actual_margin
        if abs(gap) > 1e-9:
            raise ValueError(
                f"the scoreboard {away_points}-{home_points} does not subtract to "
                f"{game_id}'s margin of {actual_margin:+.0f} ({gap:+.4f} apart)"
            )

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
    # Last, and after the shared random stream has been drawn from by every Strict
    # builder. Appending rather than interleaving is what keeps `None` byte-for-
    # byte identical to Strict: the draws the fumble and kicking coins consume do
    # not move when the variant is switched on.
    dropped_picks = dropped_pick_events(plays, ftn, dropped_pick_model, n_posterior_draws, rng)
    events += dropped_picks
    # And the receiver direction after that, for the same reason: appending keeps
    # `+dp` byte-for-byte identical whether or not `+rd` is switched on, so the
    # two variants can be read alone and together without one moving the other.
    receiver_drops = receiver_drop_events(plays, ftn, receiver_drop_model, n_posterior_draws, rng)
    events += receiver_drops
    if dropped_picks and receiver_drops:
        variant = "full"
    else:
        variant = "strict" + ("+dp" if dropped_picks else "") + ("+rd" if receiver_drops else "")

    entries = tuple(event.to_entry() for event in events)

    if not events:
        # Nothing to adjudicate. The distribution is degenerate at the actual
        # result, and DTW is 1 or 0 — correctly, since no coin was involved.
        ledger = Ledger(entries)
        total_luck_epa = ledger.total_luck_epa()
        deserved_margin = actual_margin - total_luck_epa * points_per_epa
        dtw = 1.0 if actual_margin > 0 else 0.0
        has_score = home_points is not None and away_points is not None
        return SimulationResult(
            game_id=game_id,
            actual_margin=actual_margin,
            deserved_margin=deserved_margin,
            dtw_home=dtw,
            dtw_interval=(dtw, dtw),
            margin_draws=np.full(1, actual_margin),
            ledger=ledger,
            total_luck_epa=total_luck_epa,
            home_point_draws=np.full(1, float(home_points)) if has_score else None,
            away_point_draws=np.full(1, float(away_points)) if has_score else None,
            variant=variant,
            events=tuple(events),
        )

    # Document 61's possession cap, on the Full edition and nowhere else — hard
    # constraint P-1. It is keyed on the `edition` **argument**, not on the
    # `variant` the ledger came out carrying: `"full"` is the adjudication the maintainer
    # named, and the audit arms reached with `edition=None` are deliberately the
    # uncapped comparison the cap has to be measured against.
    drive_of, offence = (None, {})
    if edition == "full":
        drive_of, offence = _possession_cap_handles(plays, events)

    # One replay. `bootstrap_margins` is still the public arithmetic document 10
    # checks; this asks the same helper for the per-event adjustment as well, so
    # the two teams' point distributions come out of the coins the margin was
    # already computed from rather than out of a second stream.
    margins, dtw_per_draw, adjustment, cap_rows = _bootstrap(
        events, actual_margin, points_per_epa, n_coin_draws, rng, drive_of=drive_of
    )

    # The cap rows join the ledger before the total is taken, because the
    # deserved margin is the ledger's sum by definition (document 61 §5's
    # reconciliation, and gate P-3's round trip).
    largest = {}
    for event in events:
        key = drive_of(event) if drive_of is not None else None
        if key is not None and (key not in largest or abs(event.swing) > abs(largest[key].swing)):
            largest[key] = event
    ledger = Ledger(
        entries
        + tuple(
            _cap_entry(
                str(label),
                luck_epa,
                offence=offence.get(str(label), largest[label].charged_team),
                play_id=largest[label].play_id,
            )
            for label, luck_epa in cap_rows.items()
        )
    )
    total_luck_epa = ledger.total_luck_epa()
    deserved_margin = actual_margin - total_luck_epa * points_per_epa

    home_draws = away_draws = None
    if home_points is not None and away_points is not None:
        home_draws, away_draws = _split_points(
            adjustment,
            events,
            plays["home_team"][0],
            float(home_points),
            float(away_points),
            points_per_epa,
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
        home_point_draws=home_draws,
        away_point_draws=away_draws,
        variant=variant,
        events=tuple(events),
    )
