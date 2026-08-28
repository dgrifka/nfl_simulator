# 61 — Possession-level luck cap, pre-registered

*Written 2026-08-28 in a Fable 5 brainstorm after figure round 6, **before
any measurement**. the maintainer raised it from the LAC @ HOU waterfall: two
would-be-touchdown drops on one possession book about −7 points each, ~−14
in total, though the drive could have produced one touchdown. This document
is the change proposal (model change: it edits the bootstrap) and the gate.
Full replay of drives stays out of scope (document 08); this is a bound, not
a simulation.*

*Inputs: documents 05 §4 (the two-layer bootstrap), 08 (sequencing out of
scope; "the bars are a sum, not a sequence"), 49/56 (the two Full
components), 59/60 (Full's audit and figures).*

---

## 0. The defect, plainly

The ledger prices every event from the game state **at that play** against
its own counterfactual, then sums. Two events on one possession are not
independent: had the first drop been caught, the second play never happens.
The sum therefore over-counts within a drive, and — because the bootstrap
flips every event independently — the deserved-margin distribution is wider
than it should be. Strict rarely hits this (two fumbles on one drive is
uncommon). Full, at ~48 events a game, hits it routinely. The same applies
across directions: a dropped pick followed by a receiver drop on the same
possession.

## 1. Tier declaration

**Model change** — `bootstrap_margins` gains a per-possession clip. It applies
to the **Full edition only**: a cap on Strict would change v1.3's numbers,
which is a separate decision; how often it *would* bite on Strict is
measured and reported (§3, M-4).

## 2. DAG edit

```
events on drive d ──► per-replicate adjustments a_i = (actual_i − replayed_i) × swing_i
                              │
                              ▼  (new)
                      A_d = clip( Σ_i a_i , −C_d , +C_d )      C_d = max_i |swing_i| on drive d
                              │
                              ▼
                      margin = actual − points_per_epa × Σ_d A_d
```

Events are grouped by `(game_id, fixed_drive)`. Kickoff/punt-return plays
carry the drive they start. Events with no drive (should be none; guard)
are their own group.

## 3. Mechanism story, and the cap chosen

**Why this cap.** A possession's luck cannot exceed the largest single
"what if" on it: if the biggest event had gone the other way, the drive's
outcome would have changed by at most that event's swing, and the smaller
events on the same drive are contingent on it. `C_d = max_i |swing_i|` is
the simplest bound with that reading; it is exact for the two-end-zone-
drops case (−14 → −7.4), it does not bite on a drive with one event, and it
does not bite on opposite-signed events unless their net exceeds the
largest single swing. It is deliberately conservative in the sense of
*understating* luck on multi-event drives; a tighter, state-aware bound
(best attainable drive EPA minus realised) is the parked alternative and
needs drive-level EP modelling this project has scoped out.

**Why it must be applied per replicate**, not to the point estimate: the
distribution's width is the second defect. Clipping inside the bootstrap
fixes both at once, and the ledger's point-estimate rows are then
reconciled to the mean of the clipped replicate sums (§5).

**What would make it fail.** (a) The cap bites so rarely that the change is
cosmetic — fine, then it ships as correctness and the measurement says so.
(b) It bites so often that Full's audit numbers move materially — then
document 59's 17.6% was partly double counting, which is exactly what the
measurement exists to expose, and the new numbers replace it in the
write-up. (c) The reconciliation of ledger rows to the clipped total breaks
the round-trip identity — pinned by test.

## 4. Measurement first (M-1 … M-4), before the cap is built

`research/77_possession_cap_measure.py`, on the Full pass over 2022–2025:

- **M-1** share of Full events that share a drive with at least one other
  event; share of drives with ≥ 2 events; distribution of events per drive.
- **M-2** per drive, `Σ|luck_i|` against `C_d`: share of drives where the
  summed |luck| exceeds `C_d` (the cap would bite), and by how much (EPA and
  points).
- **M-3** the 200 Full bucket-move games: how many contain at least one
  drive where the cap bites; the expected magnitude of the clipped
  reduction on those games (point-estimate arithmetic, before the bootstrap
  change).
- **M-4** the same M-2 on Strict, reported: how often a Strict drive would
  be capped, and the largest reduction it would have taken.
- The three named Full games (`2025_17_DET_MIN`, `2022_13_WAS_NYG`,
  `2024_19_LAC_HOU`) drive by drive.

**Pre-committed expectations, so a surprise is recognisable:** M-1 finds a
majority of Full events share a drive with another (48 events over ~24
drives); M-2 finds the cap bites on a minority of drives, mostly those
holding a ≥ 5 EPA event plus smaller ones; M-3 finds the cap reduces, not
eliminates, Full's bucket moves — if more than a third of the 200 vanish,
document 59's headline was substantially double counting and the write-up
must say so. M-4 finds Strict almost never capped (< 1% of drives).

## 5. The build, TDD

- `bootstrap_margins(events, ..., drive_of=None)`: when `drive_of` (event →
  drive key) is given, clip per-drive per-replicate sums to `±C_d`. With
  `drive_of=None` the function is byte-for-byte the old one (test: identical
  draws at the same seed).
- Ledger reconciliation: each event's `luck_epa` is unchanged in its row;
  a **cap row** per bitten drive — `component="possession_cap"`,
  `charged_team` = the drive's offence, `event_class` = `"Q3 drive 7"`,
  `luck_epa` = clipped mean − unclipped mean for that drive — is appended so
  the ledger still sums to the margin shift (V-2) and the waterfall can draw
  the cap as its own bar (`Possession cap · Q3 drive 7 (LAC): +6.6`).
- `simulate_game(..., edition="full")` passes `drive_of`; Strict never does.
- `SimulationResult.variant` unchanged; `full_summary.parquet` regenerated.

## 6. Gates

| Gate | Statement | Bar |
|---|---|---|
| **P-1 Strict untouched** | V-1 replay | 0.00e+00 over 2,761 games |
| **P-2 no-cap identity** | `drive_of=None` reproduces the pre-cap Full pass | 0.00e+00 on every game |
| **P-3 round trip** | ledger (with cap rows) × slope = margin shift | ≤ 1e-9 |
| **P-4 direction** | every cap row reduces `\|Σ drive luck\|`; never flips its sign | test |
| **P-5 single-event drives** | a drive with one event is never capped | test |
| **P-6 two-drops case** | a synthetic drive with two 9-EPA drops books at most 9 EPA | test |
| **P-7 audit** | Full vs Strict re-run with the cap: bucket moves, median |ΔDTW|, interval width, beside document 59's 200 / 3.85 pp / 0.0516 | reported; the new numbers **replace** doc 59's in the write-up |

## 7. Register

| Item | Status |
|---|---|
| Cap is a bound, not a drive simulation | By design; document 08 stands |
| Strict not capped | Separate decision; M-4 measures the stakes |
| Cross-drive dependence (a turnover ends one drive and starts another) | Out of scope; noted |
| Tighter state-aware cap | Parked |
