# 74 — Simulator v1.4.1: the corpus at canonical order, shipped

*Written 2026-09-03, the day of the tag. The ship record for `v1.4.1` — the
release document 73 pre-registered. Inputs: document 73 (the pre-registration,
its gates, and its C-1 correction), document 68 (v1.4, the baseline this
release re-prices), document 61 (the possession cap, whose grouping path first
drew suspicion).*

## 1. What shipped

Both coverage levels were re-adjudicated under the row-order-invariant code:
every frame sorts to the total order `(game_id, play_id)` before any step
that reads row position — seeded draws, join consumers, figure sorts — and
the corpus artifacts are themselves written in that order. The read side
(figures and share cards) is order-invariant and regression-tested. No
formula, prior, rate, or cap rule changed; every fitted posterior is
byte-identical to v1.4's.

## 2. The blast radius, measured

Re-pricing the full record against the shipped v1.4 artifacts:

- **11 of 3,900 adjudications moved**: 4 of 2,761 games at 2016–2025
  coverage, 7 of 1,139 at 2022–2025 coverage.
- The largest move is `2016_10_ATL_PHI`, **1.9058e-02 points** of deserved
  margin — one 55–59 yd field goal whose make probability shifts 0.51
  percentage points once its draw comes from the sorted order. The other
  three (`2021_08_JAX_SEA`, `2018_02_OAK_DEN`, `2019_16_NYG_WAS`) move
  2.6e-03, 1.8e-03 and 1.1e-03 points, all through extra points. The seven
  charted-coverage games top out at **2.03e-03 points**.
- **No game changes its verdict bucket and none changes the sign of its
  deserved margin.** Every other game reprices at 0.00e+00.
- The moved set and its deltas are asserted on every test run by
  `tests/test_strict_movers.py`, from committed fixtures — the numbers above
  are defensible without a 3,900-game replay.

## 3. The gates

The round-trip check passed on all 23,247 kicks (max reprice gap 6.0e-15);
the ledger sums to zero; the V-1 replay gate now expects exactly the four
moved games (an impossibility of 0.00e+00 having been settled by document
73's diagnosis) and stops on a fifth mover or a missing fourth. The full
suite shipped green, and the permutation tests of `tests/test_row_order.py`
and `tests/test_read_side_order.py` hold the invariance itself.

## 4. Alongside the release

- The quickstart builds the locked environment (`uv sync --extra dev`), so a
  fresh clone runs the pinned dependency set the invariance work validated.
- The model metadata now records a hash per cached data season, and the live
  entry point warns — naming the file — when the cache it reads no longer
  matches. That guard exists because of document 73's C-1 finding: the
  observation that started this work was upstream revising 2020 play-by-play
  values under the cache, not row order.
