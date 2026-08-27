"""The dropped-pick variant ledger — document 49's correctness gate, as tests.

Document 49 neutralizes a charted interception-worthy throw at the defence's
posterior-sampled catch probability, **beside** the v1.3 ledger and never
instead of it. Every gate in §6 that can be pinned without the fitted trace is
pinned here; V-1 (the 2,761-game default-off replay) and V-6/V-8 (the sampler
and the posterior spread) live in `research/67_` and `research/68_`, because they
need the fit.

Every test builds its own frame, so the suite stays network-free.
"""

from __future__ import annotations

import polars as pl
import pytest

from nfl_simulator.dropped_picks import (
    MIN_PER_BRANCH,
    SwingTable,
    build_swing_table,
    event_class_for,
)

# --------------------------------------------------------------------------
# the swing table — document 47 §3's six cells, recomputed in `src/`
# --------------------------------------------------------------------------


def throw(
    play_id: float,
    *,
    yardline_100: float,
    down: float,
    intercepted: bool,
    epa: float,
) -> dict:
    return {
        "play_id": play_id,
        "yardline_100": yardline_100,
        "down": down,
        "interception": 1 if intercepted else 0,
        "epa": epa,
    }


def worthy_corpus(
    *, cell_yardline: float = 50.0, cell_down: float = 3.0, n_per_branch: int = MIN_PER_BRANCH
) -> pl.DataFrame:
    """Worthy throws where exactly one cell clears the 30-per-branch floor.

    The populated cell prices a pick at −4.0 EPA and an escape at −0.5, so its
    own swing is −3.5; everything else in the frame is a single throw per branch
    at deliberately different values, so a cell that fell back to the pooled
    swing rather than reading its own is visible in the number.
    """
    rows = []
    play_id = 1.0
    for i in range(n_per_branch * 2):
        rows.append(
            throw(
                play_id,
                yardline_100=cell_yardline,
                down=cell_down,
                intercepted=i % 2 == 0,
                epa=-4.0 if i % 2 == 0 else -0.5,
            )
        )
        play_id += 1
    # One throw per branch in a different cell: below the floor, so pooled.
    for intercepted, epa in ((True, -8.0), (False, +2.0)):
        rows.append(throw(play_id, yardline_100=10.0, down=1.0, intercepted=intercepted, epa=epa))
        play_id += 1
    return pl.DataFrame(rows)


def test_a_populated_cell_prices_the_picked_branch_against_the_escaped_one():
    table = build_swing_table(worthy_corpus())
    assert table.swing_for(50.0, 3.0) == pytest.approx(-3.5)


def test_a_thin_cell_falls_back_to_the_pooled_swing():
    corpus = worthy_corpus()
    table = build_swing_table(corpus)
    picked = corpus.filter(pl.col("interception") == 1)["epa"].mean()
    escaped = corpus.filter(pl.col("interception") == 0)["epa"].mean()
    assert table.pooled == pytest.approx(picked - escaped)
    assert table.swing_for(10.0, 1.0) == pytest.approx(table.pooled)


def test_the_table_carries_document_47s_six_cells_with_their_counts():
    table = build_swing_table(worthy_corpus())
    assert set(table.cells) == {
        "1-33|1-2",
        "1-33|3-4",
        "34-66|1-2",
        "34-66|3-4",
        "67-99|1-2",
        "67-99|3-4",
    }
    assert table.counts["34-66|3-4"]["n_picked"] == MIN_PER_BRANCH
    assert table.counts["34-66|3-4"]["source"] == "cell"
    assert table.counts["1-33|1-2"]["source"] == "pooled"


def test_an_unreadable_pre_throw_state_takes_the_pooled_swing():
    table = build_swing_table(worthy_corpus())
    assert table.swing_for(None, 3.0) == pytest.approx(table.pooled)
    assert table.swing_for(50.0, None) == pytest.approx(table.pooled)


def test_the_event_class_names_the_bin_the_swing_came_from():
    assert event_class_for(50.0, 3.0) == "34-66 yd, late down"
    assert event_class_for(10.0, 1.0) == "1-33 yd, early down"
    assert event_class_for(80.0, 2.0) == "67-99 yd, early down"
    assert event_class_for(None, 2.0) == "pooled swing"


def test_the_swing_table_round_trips_through_its_serialised_form():
    table = build_swing_table(worthy_corpus())
    restored = SwingTable.from_dict(table.to_dict())
    assert restored.pooled == pytest.approx(table.pooled)
    assert restored.swing_for(50.0, 3.0) == pytest.approx(table.swing_for(50.0, 3.0))
