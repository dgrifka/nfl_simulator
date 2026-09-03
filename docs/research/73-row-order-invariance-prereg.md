# 73 — Row-order invariance of the round trip, pre-registered

*Written 2026-09-02, before the fix exists. No model changes, no fits, no new
statistics — this is a robustness pre-registration for a deterministic defect,
so the change-proposal template does not apply. Gate A is not triggered
(nothing is estimated). The gates below were written before the diagnosis ran;
they are committed to git before any code changes.*

*Inputs: document 61 (the possession cap, whose grouping path is the prime
suspect), document 30 §5a (the round-trip rule), document 68 (simulator v1.4,
the shipped configuration this must not silently move), document 71 (the
research-code audit whose fresh-clone pass surfaced the observation).*

---

## 1. The observation

The 2026-09-02 fresh-clone audit rebuilt the pipeline from nothing but the
README and compared its full-coverage round trip against the maintained cache.
On a keyed join — `game_id`, `play_id` — the two FTN charting pulls are
identical: **0 of 47,316 rows differ in any value**. But the rows arrive in a
different physical order (2,064 rows reordered within seasons), and with that
order the adjudicated deserved margin moves by **1.14e-06 points** on the
checked game. The exact round-trip guard refuses, as it should.

> **Correction (amendment C-1, 2026-09-03):** the reordering described above
> is inter-game block movement only — no game's charting rows differ
> within-game between a fresh pull and the cache, a fresh play-by-play pull is
> identical to the cache in both values and physical order, and block layout
> alone does not change the builders' join emission order — so the 1.14e-06 pt
> move cannot be attributed to charting row order and the input that produced
> it remains unidentified (the leading remaining candidate is the fresh
> clone's independently re-fitted posterior). The positional-read defect this
> document's fix closed is established independently by the permutation tests
> of `tests/test_row_order.py`.

Values equal, order different, output different: the pipeline is reading
meaning from row position somewhere. The repo already documents the two known
mechanisms — Polars `group_by` returns groups in thread-completion order, and
a seeded draw that indexes by row position binds the seed to whatever order
the frame happened to have.

The maintainer ruled 2026-09-02 that fitted artifacts are not distributed and
the quickstart rebuilds from scratch, so no outside user can reach this
refusal today. The defect is internal robustness — but the 2026 week-1 dry
run will ingest fresh data whose order nobody controls, so it is fixed before
then.

## 2. Hypothesis

Somewhere between the FTN frame and the bootstrap margin, an operation is
order-sensitive: either a `group_by` whose output order feeds a positional
step, or a seeded row-position draw taken before the frame is sorted to a
total order. Sorting to a deterministic total key at that boundary makes the
adjudication a pure function of the data's *values*.

## 3. The pre-registered change

One rule, applied at whatever site the diagnosis finds: **every frame is
sorted to a total order — a key that leaves no ties, such as
(`game_id`, `play_id`) — before any step that reads row position**, including
seeded draws and any consumer of `group_by` output order. The fix is a sort
(or a keyed, order-stable reformulation), not a change to any formula, prior,
rate, or cap rule.

Out of scope, pre-committed: no change to the cap arithmetic of document 61,
no change to any fitted artifact, no re-fit, no rendered-output change.

## 4. Gates

- **G-1, invariance (the point of the round).** A new test permutes the FTN
  frame's row order under a fixed seed and asserts the adjudicated output is
  *exactly* equal — every ledger row, every bootstrap margin, byte-for-byte —
  across permutations. This test must fail before the fix and pass after it.
- **G-2, the suite.** The full test suite passes; the fix adds tests and
  deletes none.
- **G-3, V-1 replay.** The shipped v1.4 outputs are replayed under the fixed
  code against the maintained cache. Expected: agreement at 0.00e+00, because
  the sort is inserted where the maintained cache's order already equals the
  total order. **If the replay is nonzero, the round stops and reports the
  magnitude and the count of affected games** — whether a canonical-order
  re-ship is worth a version bump is the maintainer's call, not the round's.
- **G-4, the fresh-clone repro.** The originally observed refusal — fresh
  pull versus maintained cache — closes: both orders now produce the same
  margin to the last bit.

## 5. Disposition

G-1 through G-4 all pass: merge, and the week-1 dry run proceeds on order-
independent footing. G-3 nonzero: stop and report per the gate. Diagnosis
finds the sensitivity somewhere other than §2's two mechanisms: stop, write
up what was found, and amend this document before any fix is written.
