"""The luck ledger: one row per neutralizable event, and it must add up.

The sum identity is the whole reason this module exists separately. A
deserve-to-win number nobody can decompose into "this fumble, that field goal"
is not an adjudication, it is an assertion.
"""

from __future__ import annotations

import polars as pl
import pytest

from nfl_simulator.ledger import Ledger, LedgerEntry, with_actual


def entry(actual: float = 1.0, expected: float = 0.45, swing: float = 4.0, **kwargs):
    defaults = {
        "play_id": 101.0,
        "component": "fumble",
        "event_class": "run/live",
        "charged_team": "BUF",
    }
    return LedgerEntry(actual=actual, expected=expected, swing=swing, **(defaults | kwargs))


def test_luck_epa_is_deviation_from_expectation_times_swing():
    assert entry(actual=1.0, expected=0.40, swing=5.0).luck_epa == pytest.approx(3.0)


def test_an_outcome_exactly_at_its_expectation_books_no_luck():
    assert entry(actual=0.45, expected=0.45, swing=9.9).luck_epa == pytest.approx(0.0)


def test_an_unfavourable_outcome_books_negative_luck():
    assert entry(actual=0.0, expected=0.60, swing=5.0).luck_epa == pytest.approx(-3.0)


def test_swing_sign_carries_the_home_perspective():
    """The away team's good fortune is negative in home perspective."""
    away_luck = entry(actual=1.0, expected=0.40, swing=-5.0)
    assert away_luck.luck_epa == pytest.approx(-3.0)


def test_empty_ledger_totals_zero():
    assert Ledger([]).total_luck_epa() == pytest.approx(0.0)


def test_total_is_the_sum_of_its_entries():
    ledger = Ledger(
        [
            entry(actual=1.0, expected=0.40, swing=5.0),  # +3.0
            entry(actual=0.0, expected=0.25, swing=8.0),  # -2.0
        ]
    )
    assert ledger.total_luck_epa() == pytest.approx(1.0)


def test_to_frame_has_one_row_per_entry_with_the_luck_column():
    ledger = Ledger([entry(play_id=1.0), entry(play_id=2.0)])
    frame = ledger.to_frame()

    assert isinstance(frame, pl.DataFrame)
    assert frame.height == 2
    assert "luck_epa" in frame.columns
    assert frame["luck_epa"].sum() == pytest.approx(ledger.total_luck_epa())


def test_to_frame_on_an_empty_ledger_still_has_the_schema():
    """A game with no luck events must not blow up the caller's concat."""
    frame = Ledger([]).to_frame()
    assert frame.height == 0
    assert "luck_epa" in frame.columns
    assert "component" in frame.columns


def test_ledger_reports_its_length():
    assert len(Ledger([entry(), entry()])) == 2


def test_expected_probability_outside_zero_one_is_rejected():
    """A probability is a probability. Catching this here stops a silent
    nonsense ledger entry from propagating into a DTW number."""
    with pytest.raises(ValueError, match="expected"):
        entry(expected=1.4)


# --------------------------------------------------------------------------
# reading an artifact written before the column was named `actual`
# --------------------------------------------------------------------------


def legacy_frame() -> pl.DataFrame:
    """`dtw_ledger_v13.parquet`'s shape: the identity's three terms, no branch."""
    return pl.DataFrame(
        {
            "play_id": [1.0, 2.0],
            "component": ["field_goal", "fumble"],
            "expected": [0.8769803817228944, 0.6840206759729129],
            "swing": [-4.290041955688844, -5.000382690077953],
            "luck_epa": [3.762282631907235, 3.4203651477903745],
        }
    )


def test_the_branch_is_recovered_exactly_from_the_identity():
    """`luck_epa = (actual - expected) * swing` determines `actual` given the rest.

    Both rows above are real `dtw_ledger_v13.parquet` rows from 2018_05_GB_DET:
    a missed 40-44 yd field goal and a lost punt fumble, so both branches are 0.
    """
    recovered = with_actual(legacy_frame())["actual"].to_list()
    assert recovered == [0.0, 0.0]


def test_a_frame_that_already_names_the_branch_is_returned_untouched():
    frame = legacy_frame().with_columns(pl.lit(1.0).alias("actual"))
    assert with_actual(frame)["actual"].to_list() == [1.0, 1.0]


def test_the_recovered_branch_is_the_one_the_entry_was_built_from():
    """Round trip: write an entry out, drop the column, and get the same branch back."""
    frame = Ledger([entry(actual=1.0), entry(actual=0.0, play_id=2.0)]).to_frame()
    assert with_actual(frame.drop("actual"))["actual"].to_list() == [1.0, 0.0]
