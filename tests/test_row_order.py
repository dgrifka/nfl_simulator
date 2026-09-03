"""Row-order invariance of the adjudication — document 73's gate G-1.

Document 73 §1: two FTN charting pulls of the same season are identical on a
keyed join — 0 of 47,316 rows differ in any value — but arrive in different
physical row order, and the adjudicated deserved margin moves 1.14e-06 points
with that order. Values equal, order different, output different: something in
the pipeline is reading meaning from row position.

The rule document 73 §3 pre-registers is one line: **every frame is sorted to a
total order before any step that reads row position.** These tests are that
rule stated as a gate. They permute the FTN frame — shuffled under a fixed
seed, and reversed — and demand the adjudication come back *exactly* equal:
every ledger row, the deserved margin, the whole bootstrap, byte-for-byte. No
`pytest.approx` anywhere in this file is deliberate; a tolerance here would
pass the very defect the round exists to close.

Two of the positional steps are pinned separately, because a fix that closed
only one would still leave the round trip refusing:

* `_resample` draws one index block per event **in the frame's iteration
  order**, so with a posterior that varies across draws each event's
  `expected_draws` — and therefore the ledger's `expected` column — follows
  whatever order the join happened to emit. `posterior_spread` switches that
  face on; a model whose `alpha` is constant would hide it.
* `_replayed_adjustment` draws `uniforms` with one column per event and
  column-indexes them by the event's position in the sequence, so the event
  order alone moves the replay even when every `expected_draws` is identical.

Every frame is built here, so the file stays network-free like the rest of the
suite.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nfl_simulator.components import fit_fg_baseline, fit_fumble_baseline, fit_xp_baseline
from nfl_simulator.dropped_picks import DroppedPickModel, build_swing_table, worthy_throw_frame
from nfl_simulator.fg_model import FieldGoalModel
from nfl_simulator.receiver_drops import (
    ReceiverDropModel,
    build_drop_swing_table,
    catchable_target_frame,
)
from nfl_simulator.simulator import simulate_game

GAME = "2022_01_AWY_HOM"
HOME, AWAY = "HOM", "AWY"
SEASON = 2022

# The covariate order both variant models are scored on. Kept minimal: this
# file is about row order, not about the surfaces.
COVARIATE_ORDER = (
    "air_yards_z",
    "air_yards_z_squared",
    "pass_location_left",
    "pass_location_right",
    "down_2",
    "down_3",
    "down_4",
    "is_contested_ball",
)
STANDARDISATION = {"air_yards": {"mean": 10.0, "sd": 5.0}}
REFERENCE_LEVELS = {"pass_location": "middle", "down": 1.0}

# A game big enough that the join has something to reorder. Below roughly a
# dozen charted rows Polars' hash join happens to emit the left frame's order
# whatever the right frame looks like, and the defect hides.
N_PLAYS = 60
N_POSTERIOR = 40

# The Strict components, appended after the passes. Enough of each that a
# permutation has something to reorder within the component as well as between.
N_FUMBLES = 12
N_KICKS = 10


# --------------------------------------------------------------------------
# the frames
# --------------------------------------------------------------------------


def pbp_play(play_id: float, **overrides) -> dict:
    base = {
        "game_id": GAME,
        "play_id": play_id,
        "season": SEASON,
        "week": 1,
        "home_team": HOME,
        "away_team": AWAY,
        "posteam": HOME,
        "defteam": AWAY,
        "play_type": "pass",
        "epa": 0.0,
        "air_epa": 0.4,
        "xyac_epa": 0.6,
        "result": 3.0,
        "fumble": 0,
        "fumbled_1_team": None,
        "fumble_recovery_1_team": None,
        "aborted_play": 0,
        "interception": 0,
        "penalty": 0,
        "field_goal_result": None,
        "kick_distance": None,
        "kicker_player_id": None,
        "receiver_player_id": "R1",
        "air_yards": 15.0,
        "pass_location": "left",
        "down": 3.0,
        "ydstogo": 8.0,
        "yardline_100": 50.0,
        "qb_hit": 0,
        "shotgun": 1,
        "wp": 0.5,
        "extra_point_attempt": 0,
        "extra_point_result": None,
    }
    return base | overrides


def game_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, infer_schema_length=None)


def plays_frame() -> pl.DataFrame:
    """One 2022 game: charted passes, plus the three Strict components.

    The passes carry the charting frame's variant events; the fumbles, field
    goals and extra points after them are what makes the Strict arm — every
    published v1.1 to v1.4 number — testable here at all. Strict never sees the
    FTN frame, so without these rows a permutation of `plays` would have nothing
    to move.
    """
    rows = []
    for index in range(N_PLAYS):
        dropped = index % 5 == 0
        rows.append(
            pbp_play(
                float(index + 1),
                posteam=HOME if index % 2 else AWAY,
                defteam=AWAY if index % 2 else HOME,
                epa=-1.5 if dropped else 1.0,
                yardline_100=float(10 + (index * 7) % 80),
                down=float(1 + index % 4),
                fixed_drive=float(1 + index // 4),
                qtr=float(1 + index // 16),
                interception=1 if index % 11 == 0 else 0,
            )
        )

    play_id = float(N_PLAYS + 1)
    for index in range(N_FUMBLES):
        offence = HOME if index % 2 else AWAY
        rows.append(
            pbp_play(
                play_id,
                play_type="run",
                posteam=offence,
                defteam=AWAY if offence == HOME else HOME,
                epa=-3.0 if index % 3 == 0 else 0.5,
                fumble=1,
                fumbled_1_team=offence,
                fumble_recovery_1_team=offence
                if index % 3
                else (AWAY if offence == HOME else HOME),
                fixed_drive=float(20 + index // 2),
                qtr=4.0,
            )
        )
        play_id += 1
    for index in range(N_KICKS):
        offence = HOME if index % 2 else AWAY
        rows.append(
            pbp_play(
                play_id,
                play_type="field_goal",
                posteam=offence,
                defteam=AWAY if offence == HOME else HOME,
                epa=2.5 if index % 4 else -2.5,
                field_goal_result="made" if index % 4 else "missed",
                kick_distance=float(40 + index % 5),
                kicker_player_id="K1" if index % 2 else "K2",
                fixed_drive=float(40 + index),
                qtr=4.0,
            )
        )
        play_id += 1
    for index in range(N_KICKS):
        offence = HOME if index % 2 else AWAY
        rows.append(
            pbp_play(
                play_id,
                play_type="extra_point",
                posteam=offence,
                defteam=AWAY if offence == HOME else HOME,
                epa=0.6 if index % 5 else -2.4,
                extra_point_attempt=1,
                extra_point_result="good" if index % 5 else "failed",
                kick_distance=33.0,
                kicker_player_id="K1" if index % 2 else "K2",
                fixed_drive=float(60 + index),
                qtr=4.0,
            )
        )
        play_id += 1
    return game_frame(rows)


def ftn_frame() -> pl.DataFrame:
    """The charting rows for that game, in play order."""
    return pl.DataFrame(
        [
            {
                "nflverse_game_id": GAME,
                "nflverse_play_id": index + 1,
                "is_interception_worthy": index % 3 == 0,
                "is_catchable_ball": True,
                "is_drop": index % 5 == 0,
                "is_contested_ball": False,
                "is_qb_out_of_pocket": False,
                "is_play_action": False,
                "is_screen_pass": False,
                "n_pass_rushers": 4,
            }
            for index in range(N_PLAYS)
        ]
    )


# --------------------------------------------------------------------------
# the models
# --------------------------------------------------------------------------


def _alpha(p: float, spread: float, seed: int) -> np.ndarray:
    """Posterior draws of an intercept centred on `p`.

    `spread = 0` is the degenerate posterior every other test in the suite uses;
    a positive spread is what makes `_resample`'s per-event index block visible
    in the numbers, because the draws it picks between are no longer all equal.
    """
    centre = float(np.log(p / (1.0 - p)))
    if spread == 0.0:
        return np.full(N_POSTERIOR, centre)
    return centre + np.random.default_rng(seed).normal(0.0, spread, size=N_POSTERIOR)


def swing_corpus(*, drop_branch: float, keep_branch: float) -> pl.DataFrame:
    rows, play_id = [], 1.0
    for index in range(80):
        rows.append(
            {
                "play_id": play_id,
                "yardline_100": 50.0,
                "down": 3.0,
                "is_drop": index % 2 == 0,
                "interception": index % 2,
                "epa": drop_branch if index % 2 == 0 else keep_branch,
                "air_epa": None,
                "xyac_epa": None,
            }
        )
        play_id += 1
    return pl.DataFrame(rows)


def dropped_pick_model(spread: float) -> DroppedPickModel:
    return DroppedPickModel(
        alpha=_alpha(0.45, spread, seed=11),
        beta=np.zeros((N_POSTERIOR, len(COVARIATE_ORDER))),
        defence_effects={},
        covariate_order=COVARIATE_ORDER,
        standardisation=STANDARDISATION,
        reference_levels=REFERENCE_LEVELS,
        swing_table=build_swing_table(swing_corpus(drop_branch=-4.0, keep_branch=-0.5)),
    )


def receiver_drop_model(spread: float) -> ReceiverDropModel:
    return ReceiverDropModel(
        alpha=_alpha(0.05, spread, seed=12),
        beta=np.zeros((N_POSTERIOR, len(COVARIATE_ORDER))),
        entity_effects={},
        covariate_order=COVARIATE_ORDER,
        standardisation=STANDARDISATION,
        reference_levels=REFERENCE_LEVELS,
        swing_table=build_drop_swing_table(swing_corpus(drop_branch=-1.5, keep_branch=1.0)),
    )


def kicking_model(spread: float) -> FieldGoalModel:
    """A make-probability surface with two kickers and a real posterior spread.

    `spread = 0` reproduces the constant-`alpha` fixture the rest of the suite
    uses; a positive spread is what makes `_resample`'s per-kick index block
    visible, exactly as it does on the two charting models.
    """
    rs = np.random.default_rng(21)
    jitter = rs.normal(0.0, spread, size=N_POSTERIOR) if spread else np.zeros(N_POSTERIOR)
    return FieldGoalModel(
        alpha=1.9 + jitter,
        beta=np.full(N_POSTERIOR, -0.115),
        gamma=np.full(N_POSTERIOR, 0.13),
        kicker_effects={
            f"{SEASON}_K1": np.full(N_POSTERIOR, 0.6),
            f"{SEASON}_K2": np.full(N_POSTERIOR, -0.6),
        },
        delta_xp=np.full(N_POSTERIOR, 0.9),
        lambda_xp=np.full(N_POSTERIOR, 0.5),
    )


@pytest.fixture(scope="module")
def baselines():
    """Fumble, field-goal and extra-point baselines, off a corpus of their own."""
    rows, play_id = [], 1000.0
    for index in range(100):
        rows.append(
            pbp_play(
                play_id,
                play_type="run",
                epa=-4.0 if index >= 40 else 0.0,
                fumble=1,
                fumbled_1_team=HOME,
                fumble_recovery_1_team=HOME if index < 40 else AWAY,
            )
        )
        play_id += 1
    for index in range(60):
        rows.append(
            pbp_play(
                play_id,
                play_type="field_goal",
                epa=2.5 if index < 48 else -2.5,
                field_goal_result="made" if index < 48 else "missed",
                kick_distance=float(40 + index % 5),
                kicker_player_id="K1",
            )
        )
        play_id += 1
    for index in range(40):
        rows.append(
            pbp_play(
                play_id,
                play_type="extra_point",
                epa=0.6 if index < 36 else -2.4,
                extra_point_attempt=1,
                extra_point_result="good" if index < 36 else "failed",
                kick_distance=33.0,
                kicker_player_id="K1",
            )
        )
        play_id += 1
    corpus = game_frame(rows)
    return (
        fit_fumble_baseline(corpus, min_class_size=10),
        fit_fg_baseline(corpus, min_bin_size=10),
        fit_xp_baseline(corpus, min_attempts=10),
    )


def adjudicate(
    baselines,
    ftn: pl.DataFrame,
    *,
    posterior_spread: float = 0.0,
    edition="full",
    plays: pl.DataFrame | None = None,
):
    fumble_baseline, fg_baseline, xp_baseline = baselines
    return simulate_game(
        plays_frame() if plays is None else plays,
        fumble_baseline=fumble_baseline,
        fg_baseline=fg_baseline,
        xp_baseline=xp_baseline,
        fg_model=kicking_model(posterior_spread),
        points_per_epa=0.6,
        dropped_pick_model=dropped_pick_model(posterior_spread),
        receiver_drop_model=receiver_drop_model(posterior_spread),
        ftn=ftn,
        edition=edition,
        home_points=24.0,
        away_points=21.0,
    )


def _permute(frame: pl.DataFrame) -> list[tuple[str, pl.DataFrame]]:
    return [
        ("shuffled seed 7", frame.sample(fraction=1.0, shuffle=True, seed=7)),
        ("shuffled seed 1984", frame.sample(fraction=1.0, shuffle=True, seed=1984)),
        ("reversed", frame.reverse()),
    ]


def permutations() -> list[tuple[str, pl.DataFrame]]:
    return _permute(ftn_frame())


def play_permutations() -> list[tuple[str, pl.DataFrame]]:
    """The same three permutations, applied to the play-by-play frame.

    `simulate_game` reads `plays["game_id"][0]`, `plays["result"][0]` and
    `plays["home_team"][0]` off the first row, so a permutation is only a
    permutation if every row agrees on those three — which they do here, being
    one game. A frame that mixed games could not be shuffled this way.
    """
    return _permute(plays_frame())


# --------------------------------------------------------------------------
# the premise: the permutations really are the same data in another order
# --------------------------------------------------------------------------


def test_the_permutations_carry_the_same_values_document_73_section_1():
    """0 rows differ on the keyed join — the whole point of document 73 §1."""
    key = ["nflverse_game_id", "nflverse_play_id"]
    base = ftn_frame().sort(key)
    for name, permuted in permutations():
        assert permuted.sort(key).equals(base), name
        assert not permuted.equals(ftn_frame()), f"{name} is not actually a permutation"


def test_the_join_output_order_follows_the_charting_frames_order():
    """The order the adjudication is handed is not the order it asked for.

    Not a defect on its own — an inner join makes no order promise — but it is
    the door row order walks in through, so it is pinned rather than assumed.
    """
    plays = plays_frame()
    ftn = ftn_frame()
    for name, permuted in permutations():
        for frame_of in (worthy_throw_frame, catchable_target_frame):
            base = frame_of(plays, ftn)
            other = frame_of(plays, permuted)
            key = ["game_id", "play_id"]
            assert base.sort(key).equals(other.sort(key)), f"{frame_of.__name__} {name}"


# --------------------------------------------------------------------------
# G-1 — the gate
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spread", [0.0, 0.6], ids=["flat posterior", "posterior spread"])
def test_g1_the_adjudication_is_exactly_invariant_to_ftn_row_order(baselines, spread):
    """Document 73 G-1. Exact equality, no tolerance — see the module docstring."""
    base = adjudicate(baselines, ftn_frame(), posterior_spread=spread)
    for name, permuted in permutations():
        other = adjudicate(baselines, permuted, posterior_spread=spread)

        assert other.deserved_margin == base.deserved_margin, name
        assert other.total_luck_epa == base.total_luck_epa, name
        assert other.actual_margin == base.actual_margin, name
        assert other.dtw_home == base.dtw_home, name
        assert other.dtw_interval == base.dtw_interval, name
        assert other.variant == base.variant, name
        assert np.array_equal(other.margin_draws, base.margin_draws), name
        assert np.array_equal(other.home_point_draws, base.home_point_draws), name
        assert np.array_equal(other.away_point_draws, base.away_point_draws), name


@pytest.mark.parametrize("spread", [0.0, 0.6], ids=["flat posterior", "posterior spread"])
def test_g1_every_ledger_row_is_exactly_invariant_to_ftn_row_order(baselines, spread):
    """The ledger row for row, in the order it is written — cap rows included."""
    base = adjudicate(baselines, ftn_frame(), posterior_spread=spread)
    base_rows = [entry.to_dict() for entry in base.ledger]
    for name, permuted in permutations():
        other = adjudicate(baselines, permuted, posterior_spread=spread)
        rows = [entry.to_dict() for entry in other.ledger]
        assert rows == base_rows, name


@pytest.mark.parametrize("spread", [0.0, 0.6], ids=["flat posterior", "posterior spread"])
def test_g1_the_event_sequence_and_its_posterior_draws_are_invariant(baselines, spread):
    """The two positional steps, pinned where they read.

    Sequence position is what `_replayed_adjustment` column-indexes; the
    `expected_draws` vector is what `_resample` hands out one index block at a
    time, in iteration order. Both have to be order-free or the round trip
    refuses on one of them.
    """
    base = adjudicate(baselines, ftn_frame(), posterior_spread=spread)
    for name, permuted in permutations():
        other = adjudicate(baselines, permuted, posterior_spread=spread)
        assert [(e.component, e.play_id) for e in other.events] == [
            (e.component, e.play_id) for e in base.events
        ], name
        for mine, theirs in zip(base.events, other.events, strict=True):
            assert np.array_equal(mine.expected_draws, theirs.expected_draws), (
                f"{name}: {theirs.component} on play {theirs.play_id}"
            )
            assert theirs.swing == mine.swing, name
            assert theirs.actual == mine.actual, name


def test_g1_holds_with_the_possession_cap_switched_off(baselines):
    """`edition=None` is the uncapped audit arm, and it must be invariant too.

    Otherwise a passing `full` would only mean the cap happened to absorb the
    movement, not that the adjudication stopped reading row position.
    """
    base = adjudicate(baselines, ftn_frame(), posterior_spread=0.6, edition=None)
    for name, permuted in permutations():
        other = adjudicate(baselines, permuted, posterior_spread=0.6, edition=None)
        assert other.total_luck_epa == base.total_luck_epa, name
        assert np.array_equal(other.margin_draws, base.margin_draws), name


def test_g1_the_strict_edition_never_saw_the_charting_frame_at_all(baselines):
    """A control: Strict must be untouched by FTN order, before and after the fix."""
    base = adjudicate(baselines, ftn_frame(), edition="strict")
    for name, permuted in permutations():
        other = adjudicate(baselines, permuted, edition="strict")
        assert other.total_luck_epa == base.total_luck_epa, name
        assert np.array_equal(other.margin_draws, base.margin_draws), name


# --------------------------------------------------------------------------
# the play-by-play frame — the same rule, on the other input
# --------------------------------------------------------------------------
#
# The charting frame was where document 73 §1 saw the defect, but nothing in
# §3's rule is specific to it: "every frame is sorted to a total order before
# any step that reads row position". The play-by-play frame feeds the three
# Strict builders, and every published v1.1 to v1.4 number is a Strict number,
# so if `plays` order is load-bearing the blast radius is the whole record
# rather than the charted seasons.
#
# `fumble_events` and `extra_point_events` draw a fresh Beta per event
# (`_class_rate_draws`) and `field_goal_events` takes a `_resample` block per
# kick, all three in the frame's iteration order; then every event's position
# in the sequence indexes `_replayed_adjustment`'s uniforms, exactly as on the
# charting side.


def test_the_play_permutations_carry_the_same_values():
    """The premise again: same rows, different order, nothing added or dropped."""
    key = ["game_id", "play_id"]
    base = plays_frame().sort(key)
    for name, permuted in play_permutations():
        assert permuted.sort(key).equals(base), name
        assert not permuted.equals(plays_frame()), f"{name} is not actually a permutation"


def test_the_game_has_something_of_every_component_to_reorder(baselines):
    """Guard: a green permutation test proves nothing if the ledger is empty."""
    result = adjudicate(baselines, ftn_frame(), posterior_spread=0.6)
    counts: dict[str, int] = {}
    for entry in result.ledger:
        counts[entry.component] = counts.get(entry.component, 0) + 1
    for component in ("fumble", "field_goal", "extra_point", "dropped_pick", "receiver_drop"):
        assert counts.get(component, 0) > 1, f"{component}: {counts}"


@pytest.mark.parametrize("spread", [0.0, 0.6], ids=["flat posterior", "posterior spread"])
@pytest.mark.parametrize("edition", ["strict", "full"], ids=["strict", "full"])
def test_the_adjudication_is_exactly_invariant_to_play_by_play_row_order(
    baselines, edition, spread
):
    """Document 73 §3's rule, applied to `plays` instead of the charting frame."""
    base = adjudicate(baselines, ftn_frame(), posterior_spread=spread, edition=edition)
    for name, permuted in play_permutations():
        other = adjudicate(
            baselines, ftn_frame(), posterior_spread=spread, edition=edition, plays=permuted
        )
        label = f"{edition} / {name}"

        assert other.deserved_margin == base.deserved_margin, label
        assert other.total_luck_epa == base.total_luck_epa, label
        assert other.actual_margin == base.actual_margin, label
        assert other.dtw_home == base.dtw_home, label
        assert other.dtw_interval == base.dtw_interval, label
        assert other.variant == base.variant, label
        assert np.array_equal(other.margin_draws, base.margin_draws), label
        assert np.array_equal(other.home_point_draws, base.home_point_draws), label
        assert np.array_equal(other.away_point_draws, base.away_point_draws), label


@pytest.mark.parametrize("edition", ["strict", "full"], ids=["strict", "full"])
def test_every_ledger_row_is_exactly_invariant_to_play_by_play_row_order(baselines, edition):
    base = adjudicate(baselines, ftn_frame(), posterior_spread=0.6, edition=edition)
    base_rows = [entry.to_dict() for entry in base.ledger]
    for name, permuted in play_permutations():
        other = adjudicate(
            baselines, ftn_frame(), posterior_spread=0.6, edition=edition, plays=permuted
        )
        assert [entry.to_dict() for entry in other.ledger] == base_rows, f"{edition} / {name}"


def test_both_frames_permuted_at_once_still_lands_on_the_same_adjudication(baselines):
    """The two inputs together, since production reorders both independently."""
    base = adjudicate(baselines, ftn_frame(), posterior_spread=0.6)
    for (name, permuted_ftn), (_, permuted_plays) in zip(
        permutations(), play_permutations(), strict=True
    ):
        other = adjudicate(baselines, permuted_ftn, posterior_spread=0.6, plays=permuted_plays)
        assert other.deserved_margin == base.deserved_margin, name
        assert other.total_luck_epa == base.total_luck_epa, name
        assert np.array_equal(other.margin_draws, base.margin_draws), name
