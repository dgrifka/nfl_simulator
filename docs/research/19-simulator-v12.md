# 19 — Simulator v1.2: the fumble widening, shipped

*Written 2026-08-18, after the code landed. This is a **ship record**, not a
pre-registration: every threshold this change had to clear was fixed in document
18 and the fit was run before the maintainer approved it on 2026-08-18. What is new here
is the production implementation and the rebuilt artifacts.*

*Script: `research/31_simulator_v12.py`. Artifacts:
`research/outputs/dtw_games_v12.parquet`, `dtw_ledger_v12.parquet`,
`model_metadata_v12.json`, `31_ledger_delta.json`.*

---

## 1. The answer, stated first

> **The fumble component now runs on every fumble with a resolved disposition
> rather than only the ones somebody recovered, and the rebuilt simulator
> reproduces document 18's fit exactly.** 591 new ledger rows across 529 games,
> no fumble booked twice, and the ledger still sums.

| Check | Document 18 | v1.2 production | |
|---|---|---|---|
| Fumbles in the population | 6,505 | **6,505** | exact |
| Out of bounds | 602 | **602** | exact |
| League retention rate | 56.48% | **56.48%** | exact |
| `pass/live` p | 0.5096 | **0.5096** | exact |
| `run/aborted` p | 0.7690 | **0.7690** | exact |
| Games containing an out-of-bounds fumble | 536 | **536** | exact |
| DTW side flips, all fumble games | 48 | **49** | agrees |
| DTW side flips, out-of-bounds games | 31 | **32** | agrees |
| Median \|ΔDTW\| on the games that gained a row | 1.65 pp | **1.75 pp** | agrees |

The first six are deterministic and match to every published digit. The last
three involve replayed coins and land within a game or a tenth of a point of
document 18's isolated arm study — which is the agreement the design predicted,
not a coincidence, because the Monte Carlo floor was measured rather than
assumed (§4).

## 2. What changed in the code

`src/nfl_simulator/components.py` and `src/nfl_simulator/simulator.py`, under
test-driven development; branch `feat/phase5-v12-fumble-widening`.

| Before (v1.1) | After (v1.2) |
|---|---|
| `live_fumble_mask()` — fumble, identified fumbling team, **and a recovering team** | `any_fumble_mask()` — fumble and an identified fumbling team |
| `recovered_own` — did the fumbling team recover it | `retained` — did the fumbling team **end up with** it |
| Out-of-bounds fumbles are excluded, and book zero luck | Out of bounds is a retention branch and books a ledger row |
| `FumbleBaseline.league_recovery_rate()` | `FumbleBaseline.league_retention_rate()`, and the table gains `p_out_of_bounds` |

Three implementation facts worth stating in writing:

1. **Disposition is resolved in order, not added up.** A named recovering team
   is checked first, then the out-of-bounds flag. That is what makes the eleven
   contradictory plays produce exactly one row each instead of two, which is
   Gate F-4's whole concern (document 18 §5e). It is pinned by a test.
2. **`live_fumble_mask` was kept, not deleted.** Document 04's recovery-rate
   model (`rates.fumble_recovery_counts`) and the incumbent arm of documents
   18/29/30 are all defined on the narrower population, and filtering a frame
   through the old mask before the new frame builder reproduces them exactly.
   Deleting it would have made the v1.1 record unreproducible for no gain.
3. **A frame without the `fumble_out_of_bounds` column still works**, treating
   every fumble as resolved by its recovery. That is the v1.1 replay path, and
   it is why the Phase 2 tests still pass unchanged.

## 3. Two numbers that need their exact wording

**529 games gained a row, not 536.** 536 games contain an out-of-bounds fumble
and that is the population document 18's materiality floor was read on. But 11
of the 602 out-of-bounds fumbles also carry a recovering team and were therefore
*already* in the v1.1 ledger, so 591 rows are genuinely new and they fall in 529
games. Seven games contain an out-of-bounds fumble and gain nothing. Document 18
§4b called the 536 "games where a new ledger row appears"; the precise statement
is **games containing an out-of-bounds fumble**, and the two differ by seven.

**The pooling rule for branch means changed with the widening.** v1.1 replaced a
thin class's `epa_own`/`epa_lost` with the pooled value whenever the class had
fewer than 30 plays. v1.2 falls back only when a branch mean is genuinely
missing. This is the behaviour document 18 §8's impact figures were computed
with, so shipping it is what reproduces the approved numbers — but it means two
classes totalling **six plays in ten seasons** (`field_goal/live`, n = 4;
`punt/aborted`, n = 2) now carry branch means estimated from a handful of plays.
Document 18 §6 registered this as an open defect and named the fix: pool the
swing on class size, which would change v1.1's numbers too and therefore needs
its own round. It is carried forward unchanged, and the code says so at the
line.

## 4. The Monte Carlo floor, measured

Adding fumble rows shifts the shared random stream that the field-goal and
extra-point draws are pulled from, so a raw v1.1-to-v1.2 comparison mixes the
component's effect with a reshuffle. Document 18 avoided this by holding the
other components' generators fixed; a full rebuild cannot.

Rather than assume the reshuffle was small, `research/31_simulator_v12.py`
simulates a second arm — the same v1.2 code at a different seed — and counts how
many games flip between the two. **One game.** The floor is negligible, so the
49 flips between v1.1 and v1.2 are the component's doing, and their agreement
with document 18's isolated 48 is not luck.

## 5. What was deliberately not re-run

**The rematch validation.** Document 18 §6 accepted in advance that re-running
document 06's Gate 1 on v1.2 would prove nothing: document 12 measured the
rematch test as nearly blind below roughly 20% damage, and this change moves the
mean deserved margin by half a point. `model_metadata_v12.json` records the
non-run and its reason rather than silently omitting the field. v1.1's validated
artifacts are preserved untouched.

## 6. What v1.2 is

| | v1.1 | v1.2 |
|---|---|---|
| Fumble population | 5,914 recovered fumbles | **6,505 fumbles** |
| Fumble branch | recovery | **retention** |
| League rate | 52.13% | **56.48%** |
| Ledger rows | 29,463 | **30,054** |
| Mean \|Δ deserved margin\| vs v1.1 | — | 0.53 points |
| Max \|Δ deserved margin\| vs v1.1 | — | 6.33 points |
| Field goals, extra points, everything else | unchanged | unchanged |

Document 05 §3's treatment table gains no row. The component's name, its
`event_class` values and the ledger schema are all unchanged; the population it
runs on is wider.
