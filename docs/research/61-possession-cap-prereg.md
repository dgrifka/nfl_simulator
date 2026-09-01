# 61 — Possession-level luck cap, pre-registered

*Written 2026-08-28 in a Fable 5 brainstorm after figure round 6, **before
any measurement**. The maintainer raised it from the LAC @ HOU waterfall: two
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

---

### Results, filled in 2026-08-28 from `research/77_possession_cap_measure.py`

*Full pass over 2022-2025, 1,139 games, at v1.3's settings (seed 20260817, 200
posterior draws, 800 coin draws, slope 0.8389). V-1 was 0.00e+00 over 2,761
games on the same run. `research/outputs/77_possession_cap_measure.json` holds
every number below; `research/outputs/` is gitignored, the script is the
artifact.*

**One statistic had to be added before M-3 could be read, and it is named
here rather than swapped in quietly.** §4 as written asks whether
`Σ|luck_i| > C_d`, a question about the *point estimate*. That question turned
out to be nearly blind to what the cap does. In the bootstrap an event
contributes either nothing or its **whole swing** — `a_i = (actual_i −
replayed_i) × swing_i` with both terms in {0, 1} — so a possession's replicate
sum ranges over subset sums of the drive's swings, which are several times
larger than the expectation-weighted `luck_i` the ledger prints. The
pre-registered statistic says the cap bites on 0.9% of drives; the clip
actually engages in at least one replicate on 69.3% of them. Both are reported
below, together with the quantity §5 actually books — `mean(clipped) −
mean(unclipped)` over replicates, the cap row — estimated at 2,000 layer-2
draws with layer 1 held at the posterior mean. The pre-registered statistic is
reported first and was not replaced.

**M-1 — sharing.** 69,419 Full events sit on 21,327 drives. **64,640 of them
(93.1%) share a drive with at least one other event.** 16,548 drives carry two
or more — 77.6% of the drives that carry any, and 66.9% of all 24,752 drives
played. Mean 3.25 events per event-carrying drive, max 15. Distribution: 4,779
drives hold one event, 4,748 two, 3,557 three, 2,962 four, 2,128 five, and a
tail to 15. No event was missing a `fixed_drive`. **As expected, and more so.**

**M-2 — how often, and by how much.**

| statistic | drives bitten | share |
|---|---|---|
| §4 as pre-registered, `Σ\|luck\| > C_d` | 634 of 21,327 | 3.0% |
| the clip at the point estimate, `\|Σ luck\| > C_d` | 183 | 0.9% |
| clipped in ≥ 1 bootstrap replicate | 14,779 | 69.3% |
| books a cap row (`mean(clipped) ≠ mean(unclipped)`) | 14,223 | 66.7% |

Size: the cap row is **0.013 EPA (0.011 pt) at the median** and 0.068 EPA at
the mean, with a maximum of 5.73 EPA (4.81 pt). Over the whole Full population
the cap moves **+46 EPA (+39 pt)** of booked luck. **Not as expected**: the cap
touches two thirds of possessions rather than a minority, and does almost
nothing to nearly all of them.

**M-3 — the 200 bucket moves.** All 200 hold a drive the cap clips, and all 200
take at least one cap row, so the exact upper bound on moves that could vanish
is 200 and carries no information. The cap gives back a **median 0.31 pt** of
deserved margin on them (mean 0.56, max 5.10) — a **median 9.1% of the
Full-minus-Strict margin shift**. Walking DTW back by that same fraction and
re-bucketing, a linear proxy, **29 of the 200 moves vanish (14.5%)**, against
the one-third threshold. **Below the threshold; §4's stop condition did not
trip, and part B proceeds.** The proxy is a point-estimate walk-back, not a
bootstrap result — part C recomputes it with the cap inside the bootstrap, and
the true number can differ because the cap also narrows the interval.

**M-4 — Strict.** 120 of 28,070 Strict drives (0.4%) bite on the pre-registered
statistic, 14 at the point estimate, 1,368 (4.9%) in at least one replicate, and
1,357 (4.8%) would book a cap row worth a median 0.029 EPA and at most 2.19 EPA
(1.84 pt). **Roughly as expected** on the pre-registered statistic; the
replicate-level figure is five times the "< 1% of drives" guess. Not applied —
P-1 stands — and this is what leaving it unapplied costs.

**The three named games, and a finding about the one that prompted this
document.** The full drive tables are in the JSON. The totals: DET @ MIN
+0.10 EPA of cap rows (−0.08 pt), WAS @ NYG +2.39 EPA (−2.00 pt), LAC @ HOU
−1.41 EPA (+1.18 pt).

**The two would-be-touchdown drops in LAC @ HOU are on two different
possessions.** They are `play_id` 3221 on `fixed_drive` 21 (swing −9.07,
booking +8.60 EPA) and `play_id` 3368 on `fixed_drive` 22 (swing −9.28, booking
+8.80). Drive 21 is Los Angeles's; the drop on it was returned for a Houston
score, which is why Houston's extra point sits on the same drive. Drive 22 is
Los Angeles's next possession. Neither drive is clipped — each holds one large
swing that *is* its own `C_d`, which is P-5 — so **the cap document 61 proposes
does not bite on the case that motivated document 61.** That is a finding, not a
failure: §3(a) pre-registered this outcome ("the cap bites so rarely that the
change is cosmetic — fine, then it ships as correctness and the measurement says
so"), and the cross-drive dependence §7 parks is exactly the thing that would
have to be priced to reach these two drops.

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

---

## 8. Outcome

**Enacted, 2026-08-28**, on branch `feat/possession-cap`. Record: document 62.
Every gate in §6 passed: P-1 at 0.00e+00 over 2,761 games on the widened frame,
P-2 at 0.00e+00 over 1,139 with the uncapped arm still reproducing document 59
§4 exactly, P-3 at 0.00e+00, and P-4 through P-6 pinned in
`tests/test_possession_cap.py`. P-7, reported: the Full edition's bucket moves
fall from **200 to 190** (17.6% → **16.7%**), the median |ΔDTW| on affected
games from 3.85 pp to **3.39 pp**, and the mean 89% interval from 0.0516 to
**0.0495**.

Two things §3's "what would make it fail" did not anticipate, both recorded in
document 62 rather than smoothed over:

- **(a) and (b) are both true at once.** The cap touches nearly every possession
  — 14,747 rows over 1,139 games, one in every game — and does almost nothing to
  nearly all of them, a median 0.012 EPA per row. It is cosmetic 14,000 times
  and material a few dozen.
- **The case that raised this document is not reached by the cap it proposed.**
  The two would-be-touchdown drops in `2024_19_LAC_HOU` are on consecutive
  possessions, not one, and each is its own possession's `C_d`. §7's cross-drive
  dependence, which this document put out of scope, is what that case needs.

§4's pre-registered M-2 statistic was kept and reported first; a second
statistic was added beside it because the first is nearly blind to what the cap
does inside the bootstrap, and the addition is named as an addition in §4 and
in document 62 §2.
