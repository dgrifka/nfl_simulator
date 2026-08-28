"""The possession-level luck cap — document 61.

The defect: the ledger prices every event from the game state at its own play
and then adds the rows up, so two events on one possession are counted as
though the second would still have happened had the first gone the other way.
Document 61 §2 bounds a possession's luck by the largest single "what if" on it,

    A_d = clip( Σ_i a_i , −C_d , +C_d )      C_d = max_i |swing_i| on drive d

per replicate, inside the bootstrap, because the width of the deserved-margin
distribution is the second half of the same defect.

Every gate document 61 §6 pins in a test lives here: P-2 (the no-cap identity),
P-3 (the round trip with cap rows in it), P-4 (direction), P-5 (a one-event
drive is never capped), P-6 (the two-drops case) and the cap row's schema. P-1
and P-7 are population gates and are read in `research/78`.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nfl_simulator.ledger import LEDGER_SCHEMA
from nfl_simulator.simulator import (
    POSSESSION_CAP_COMPONENT,
    LuckEvent,
    bootstrap_margins,
    team_point_draws,
)

HOME, AWAY = "HOM", "AWY"


def coin(
    play_id: float,
    p: float,
    *,
    actual: float,
    swing: float,
    draws: int = 40,
    team: str = HOME,
) -> LuckEvent:
    return LuckEvent(
        play_id=play_id,
        component="test",
        event_class="test",
        charged_team=team,
        actual=actual,
        expected_draws=np.full(draws, p),
        swing=swing,
    )


def on_drive(*assignments: tuple[float, str]):
    """`drive_of` from a literal `play_id -> drive` table."""
    table = dict(assignments)
    return lambda event: table[event.play_id]


def bootstrap(events, *, drive_of=None, seed: int = 5, margin: float = 3.0, ppe: float = 1.0):
    return bootstrap_margins(
        events,
        actual_margin=margin,
        points_per_epa=ppe,
        n_coin_draws=25,
        rng=np.random.default_rng(seed),
        drive_of=drive_of,
    )


# --------------------------------------------------------------------------
# P-2 — the no-cap identity
# --------------------------------------------------------------------------


def test_p2_drive_of_none_reproduces_the_bootstrap_at_the_same_seed():
    """Document 61 §6's P-2 at the unit level: the default is the old function.

    The population half of P-2 — the whole pre-cap Full pass, game by game — is
    `research/78`'s job. What a unit test can pin is the thing that would break
    it: the cap must not consume, reorder or perturb a single random draw.
    """
    events = [coin(1.0, 0.5, actual=1.0, swing=4.0), coin(2.0, 0.5, actual=0.0, swing=-3.0)]
    margins, dtw = bootstrap(events)
    again, again_dtw = bootstrap(events, drive_of=None)
    assert np.array_equal(margins, again)
    assert np.array_equal(dtw, again_dtw)


def test_p2_a_cap_that_never_bites_is_the_uncapped_bootstrap_exactly():
    """The machinery itself is inert, not merely close: same draws, same answer."""
    events = [coin(1.0, 0.5, actual=1.0, swing=4.0), coin(2.0, 0.5, actual=0.0, swing=-3.0)]
    uncapped, _ = bootstrap(events)
    # Each event on its own possession, so no drive can ever exceed its own C_d.
    capped, _ = bootstrap(events, drive_of=on_drive((1.0, "Q1 drive 1"), (2.0, "Q1 drive 2")))
    assert np.array_equal(uncapped, capped)


# --------------------------------------------------------------------------
# P-5 — a one-event drive is never capped
# --------------------------------------------------------------------------


def test_p5_a_single_event_drive_books_no_cap_row():
    """|a_i| ≤ |swing_i| = C_d, so the clip has nothing to do. Arithmetic, not luck."""
    from nfl_simulator.simulator import _apply_possession_cap, _replayed_adjustment

    events = [coin(1.0, 0.5, actual=1.0, swing=9.0)]
    adjustment = _replayed_adjustment(events, 25, np.random.default_rng(1))
    before = adjustment.copy()
    cap_rows = _apply_possession_cap(adjustment, events, on_drive((1.0, "Q1 drive 1")))
    assert cap_rows == {}
    assert np.array_equal(adjustment, before)


def test_p5_holds_even_when_the_one_event_is_the_biggest_in_the_game():
    events = [
        coin(1.0, 0.0, actual=1.0, swing=12.0),
        coin(2.0, 0.5, actual=1.0, swing=1.0),
    ]
    _, dtw = bootstrap(events, drive_of=on_drive((1.0, "Q1 drive 1"), (2.0, "Q2 drive 5")))
    uncapped, uncapped_dtw = bootstrap(events)
    assert np.array_equal(dtw, uncapped_dtw)


# --------------------------------------------------------------------------
# P-6 — the two-drops case
# --------------------------------------------------------------------------


def test_p6_two_nine_epa_drops_on_one_possession_book_at_most_nine():
    """Document 61 §3's worked case, at the certainty that makes it exact.

    Both branches are certain (`p = 1`, `actual = 0`), so every replicate books
    both events at their full swing: −9 and −9, eighteen EPA of luck on one
    possession that could only ever have produced one touchdown. The cap is 9.
    """
    drops = [
        coin(1.0, 1.0, actual=0.0, swing=-9.0),
        coin(2.0, 1.0, actual=0.0, swing=-9.0),
    ]
    uncapped, _ = bootstrap(drops)
    capped, _ = bootstrap(drops, drive_of=on_drive((1.0, "Q4 drive 22"), (2.0, "Q4 drive 22")))
    # `a_i = (0 − 1) × −9 = +9` each, so the margin moves *down* by the total.
    np.testing.assert_allclose(uncapped, 3.0 - 18.0)
    np.testing.assert_allclose(capped, 3.0 - 9.0)


def test_p6_the_cap_is_the_largest_swing_not_the_average_of_them():
    """A big drop beside a small one is bounded by the big one, not by their mean."""
    drops = [
        coin(1.0, 1.0, actual=0.0, swing=-9.0),
        coin(2.0, 1.0, actual=0.0, swing=-1.0),
    ]
    capped, _ = bootstrap(drops, drive_of=on_drive((1.0, "Q4 drive 22"), (2.0, "Q4 drive 22")))
    np.testing.assert_allclose(capped, 3.0 - 9.0)


# --------------------------------------------------------------------------
# P-4 — direction
# --------------------------------------------------------------------------


def test_p4_the_clip_never_grows_a_replicate_or_flips_its_sign():
    """The invariant that is exactly true, replicate by replicate.

    `clipped = sign(u)·min(|u|, C)`, so no replicate's drive total can grow and
    none can change side. This is the statement document 61 §6's P-4 rests on;
    the cap-row version below is that statement carried to the mean.
    """
    from nfl_simulator.simulator import _apply_possession_cap, _replayed_adjustment

    events = [
        coin(1.0, 0.4, actual=1.0, swing=6.0),
        coin(2.0, 0.6, actual=0.0, swing=-5.0),
        coin(3.0, 0.5, actual=1.0, swing=2.0),
    ]
    drive_of = on_drive((1.0, "Q2 drive 7"), (2.0, "Q2 drive 7"), (3.0, "Q2 drive 7"))
    adjustment = _replayed_adjustment(events, 200, np.random.default_rng(3))
    before = adjustment.sum(axis=2).copy()
    _apply_possession_cap(adjustment, events, drive_of)
    after = adjustment.sum(axis=2)
    assert np.all(np.abs(after) <= np.abs(before) + 1e-12)
    assert np.all(after * before >= -1e-12)


def test_p4_a_cap_row_reduces_the_luck_the_possession_booked():
    """On a possession whose events all pull one way, the cap row pulls back."""
    from nfl_simulator.simulator import _apply_possession_cap, _replayed_adjustment

    events = [
        coin(1.0, 1.0, actual=0.0, swing=-9.0),
        coin(2.0, 1.0, actual=0.0, swing=-4.0),
    ]
    drive_of = on_drive((1.0, "Q4 drive 22"), (2.0, "Q4 drive 22"))
    adjustment = _replayed_adjustment(events, 50, np.random.default_rng(9))
    booked = adjustment.sum(axis=2).mean()
    cap_rows = _apply_possession_cap(adjustment, events, drive_of)
    assert set(cap_rows) == {"Q4 drive 22"}
    (cap_row,) = cap_rows.values()
    # Booked +13, cap 9, so the row gives back 4 and points the other way.
    assert cap_row < 0.0 < booked
    assert abs(booked + cap_row) < abs(booked)
    np.testing.assert_allclose(cap_row, -4.0)


# --------------------------------------------------------------------------
# the two teams still add up
# --------------------------------------------------------------------------


def test_the_two_team_point_draws_still_subtract_to_the_capped_margin():
    """The clip is proportional inside a possession, so the split survives it.

    A drive's events are scaled by one factor per replicate rather than one of
    them being trimmed, which is what keeps `home − away` the margin the same
    replay produced.
    """
    events = [
        coin(1.0, 1.0, actual=0.0, swing=-9.0, team=HOME),
        coin(2.0, 1.0, actual=0.0, swing=-4.0, team=AWAY),
    ]
    drive_of = on_drive((1.0, "Q4 drive 22"), (2.0, "Q4 drive 22"))
    margins, _ = bootstrap_margins(
        events,
        actual_margin=3.0,
        points_per_epa=1.0,
        n_coin_draws=25,
        rng=np.random.default_rng(4),
        drive_of=drive_of,
    )
    home, away = team_point_draws(
        events,
        home_team=HOME,
        home_points=10.0,
        away_points=7.0,
        points_per_epa=1.0,
        n_coin_draws=25,
        rng=np.random.default_rng(4),
        drive_of=drive_of,
    )
    np.testing.assert_allclose(home - away, margins.ravel())


# --------------------------------------------------------------------------
# the cap row's schema
# --------------------------------------------------------------------------


def cap_entry(label: str = "Q3 drive 7", luck: float = -1.25):
    from nfl_simulator.simulator import _cap_entry

    return _cap_entry(label, luck, offence=AWAY, play_id=1763.0)


def test_a_cap_row_books_exactly_the_luck_it_was_built_from():
    entry = cap_entry(luck=-1.25)
    assert entry.luck_epa == pytest.approx(-1.25)


def test_a_cap_row_names_the_possession_and_the_offence():
    entry = cap_entry()
    assert entry.component == POSSESSION_CAP_COMPONENT
    assert entry.event_class == "Q3 drive 7"
    assert entry.charged_team == AWAY
    assert entry.play_id == 1763.0


def test_a_cap_row_fits_the_ledger_frame_every_other_row_fits():
    from nfl_simulator.ledger import Ledger

    frame = Ledger((cap_entry(),)).to_frame()
    assert frame.schema == pl.Schema(LEDGER_SCHEMA)
    assert frame["component"].to_list() == [POSSESSION_CAP_COMPONENT]
