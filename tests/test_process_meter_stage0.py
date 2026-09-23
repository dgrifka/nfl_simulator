"""Stage 0: every number of record reproduces through the library's own seam.

Three checks, and all three need the 2016-2025 play-by-play cache, so the file is
marked ``slow``. Point ``NFL_SIM_DATA_DIR`` at the cache (or run from a checkout
whose ``data/`` holds it) and::

    uv run pytest tests/test_process_meter_stage0.py -m slow -q

1. **the six reference games** at four decimals;
2. **the Brier score** on the decided 2024-2025 regular-season games;
3. **the log loss** on the same.

The claim being checked is the one the whole design rests on: the library scores
a final through the same code the research record ran, so a published number is a
gate number rather than something that resembles one. The expected calibration
error is measured and printed beside them, never gated.

The two 2026 games skip by name when the cache's week-1 file does not carry them.
"""

from __future__ import annotations

import numpy as np
import pytest

from nfl_simulator.process_meter.fit import EXPECTED_GATE_GAMES, gate_games
from nfl_simulator.process_meter.score import build_frames, score_games
from nfl_simulator.process_meter.weights import load_weights

pytestmark = pytest.mark.slow

#: The percents of record, at four decimals.
REFERENCE_PERCENTS = {
    "2026_01_NE_SEA": 0.9121,
    "2026_01_SF_LA": 0.1707,
    "2025_01_CIN_CLE": 0.7488,
    "2025_01_DAL_PHI": 0.5363,
    "2025_04_CHI_LV": 0.8913,
    "2025_18_CAR_TB": 0.2835,
}

EXPECTED_BRIER = 0.1436
EXPECTED_LOG_LOSS = 0.4376

#: Log loss only; the Brier score reads the unclipped share.
PROBABILITY_FLOOR = 1e-15

SEASON = 2026


@pytest.fixture(scope="module")
def weights():
    return load_weights()


@pytest.fixture(scope="module")
def frames():
    try:
        return build_frames(SEASON)
    except FileNotFoundError as exc:  # pragma: no cover - environment, not behaviour
        pytest.skip(f"no play-by-play cache: {exc}")


def _losses(p, outcome):
    """Per-game Brier and log loss of the home share against the home win."""
    p = np.asarray(p, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    clipped = np.clip(p, PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
    brier = (p - outcome) ** 2
    log_loss = -(outcome * np.log(clipped) + (1 - outcome) * np.log(1 - clipped))
    return brier, log_loss


# --- the six reference games ------------------------------------------------------------


@pytest.mark.parametrize(("game_id", "expected"), sorted(REFERENCE_PERCENTS.items()))
def test_a_reference_game_scores_to_its_percent_of_record(weights, frames, game_id, expected):
    if game_id not in set(frames[0].game_id):
        pytest.skip(f"{game_id} is not in this cache")
    table = score_games(SEASON, [game_id], weights, frames=frames)
    assert round(float(table.p_home.iloc[0]), 4) == expected


# --- the gate window --------------------------------------------------------------------


@pytest.fixture(scope="module")
def gate_table(weights, frames):
    frame, _, scores = frames
    gate = gate_games(scores, frame)
    assert len(gate) == EXPECTED_GATE_GAMES
    table = score_games(SEASON, list(gate.game_id), weights, frames=frames)
    outcome = (gate.set_index("game_id").result.reindex(table.game_id) > 0).to_numpy(float)
    return table, outcome


def test_the_brier_score_on_the_gate_window(gate_table):
    brier, _ = _losses(gate_table[0].p_home.to_numpy(float), gate_table[1])
    assert round(float(brier.mean()), 4) == EXPECTED_BRIER


def test_the_log_loss_on_the_gate_window(gate_table):
    _, log_loss = _losses(gate_table[0].p_home.to_numpy(float), gate_table[1])
    assert round(float(log_loss.mean()), 4) == EXPECTED_LOG_LOSS


def test_the_calibration_error_is_measured_and_reported_never_gated(gate_table, capsys):
    """Printed beside the two gated losses, with no line of its own to clear."""
    table, outcome = gate_table
    p = table.p_home.to_numpy(float)
    edges = np.linspace(0.0, 1.0, 11)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
    total, n_total = 0.0, 0
    for b in range(10):
        rows = which == b
        if not rows.any():
            continue
        total += rows.sum() * abs(p[rows].mean() - outcome[rows].mean())
        n_total += int(rows.sum())
    ece = total / n_total
    with capsys.disabled():
        print(f"\nexpected calibration error on {n_total} games: {ece:.4f} (reported, not gated)")
    assert 0.0 <= ece < 0.1
