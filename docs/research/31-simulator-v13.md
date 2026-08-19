# 31 — Simulator v1.3: three corrections, shipped and decomposed

*Written 2026-08-18, after the code landed. This is a **ship record**, not a
pre-registration: every gate this release had to clear was fixed in documents
27 and 30 before any of it was measured. What is new here is the production
implementation, the rebuilt artifacts, and the decomposition that says which of
the three corrections moved what.*

*Scripts: `research/44_read_side_fix.py` (Part A's gates and isolated impact),
`research/45_blocked_exclusion_c1.py` (Part B under Gate C),
`research/46_simulator_v13.py` (the build and the decomposition),
`research/47_rematch_v13.py` (the rematch check). Artifacts:
`research/outputs/dtw_games_v13.parquet`, `dtw_ledger_v13.parquet`,
`model_metadata_v13.json`, `46_ledger_delta.json`.*

---

## 1. The answer, stated first

> **The simulator now prices kicks with the model it actually fitted, on the
> population Gate A allows.** Three corrections ship together: the
> make-probability refit, the read side that had never applied `delta_cubic`,
> `delta_xp` or `lambda_xp`, and the exclusion of 302 blocked kicks. The median
> game with a kick moves **0.108 points of deserved margin**, 33 of 2,761
> verdicts change hands, and the ledger loses exactly 302 rows and gains none.

| Check | Source | v1.3 production | |
|---|---|---|---|
| Field-goal rows, v1.2 population | 10,731 | **10,731** | exact |
| Extra-point rows, v1.2 population | 12,818 | **12,818** | exact |
| Field-goal rows, v1.3 | 10,539 | **10,539** | exact |
| Extra-point rows, v1.3 | 12,708 | **12,708** | exact |
| Fumble rows | 6,505 | **6,505** | exact |
| Kicks in the refit population | 23,247 | **23,247** | exact |
| Kicker-seasons in the refit | 432 | **432** | exact |
| `points_per_epa` | 0.8389 | **0.8389** | exact |
| `alpha` / `beta` / `gamma` | 1.9068 / −0.11587 / 0.2489 | **identical to 5 dp** | exact |
| `delta_cubic` / `delta_xp` / `lambda_xp` | −0.0811 / +0.1222 / 1.2472 | **identical to 5 dp** | exact |
| v1.2 replayed under v1.3's code | 2,761 games | max \|Δ deserved margin\| **0.00e+00** | exact |
| Round trip, read side vs fit | ≤ 1e-9 | **5.7e-15** refit, **6.8e-15** incumbent | passes |
| Ledger identity | 0 | **0.00e+00** | exact |

Everything above is deterministic and reproduces to every published digit. One
published number did **not** reproduce, and §4 is about it.

## 2. What changed in the code

`src/nfl_simulator/fg_model.py`, `simulator.py` and `components.py`, under
test-driven development; branch `feat/phase8-v13`.

| Before (v1.2) | After (v1.3) |
|---|---|
| `_logit` computes `alpha + beta·d + gamma·d²/100` and stops | `+ delta_cubic·d³/1000` when the posterior carries one |
| No extra-point path at all; a PAT is priced as a 33-yard field goal | `make_probability(..., extra_point=True)` adds `delta_xp` and scales the kicker effect by `lambda_xp` |
| `from_posterior` reads five variables | reads eight; each is optional and **absent means absent**, never zero |
| `fg_attempt_mask()` — every charted field goal | `& field_goal_result != "blocked"`, with `include_blocked=True` preserving the v1.2 population |
| `xp_attempt_mask()` — every charted extra point | the same narrowing |
| The simulator reads `trace_fg_weather.nc` | v1.3 artifacts read **`trace_fg_refit.nc`** |

Three implementation facts worth stating in writing:

1. **The masks narrow; the play frame never does.** Four blocked field goals
   also carry a fumble row, and a frame-level filter would delete them silently
   — document 26 §8's trap. The test that pins this implements the trap and
   watches the fumble row vanish, so it fails for the right reason.
2. **`include_blocked=True` was kept, not deleted**, exactly as `live_fumble_mask`
   was when the fumble component widened (document 19 §2). v1.1's and v1.2's
   ledgers are artifacts of this repository and they must replay; `research/46`
   proves they still do, to 0.00e+00.
3. **The blocked kick's EPA lands in `core`** — or in `penalty`, on the six of
   192 that also carry a penalty flag. `decompose_plays` shares the mask, so the
   EPA the ledger stops adjudicating goes back to the bucket for things nobody
   is being given credit or blame for. The invariant checked is the five-way
   partition, play by play, at 4.4e-16.

## 3. The decomposition — which correction moved what

All five arms run at v1.2's seed with the same draws, against v1.2 as the
reference, over the same 2,761 games.

| Arm | Posterior | Read side | Blocked kicks | Median \|ΔDTW\| | Median \|Δ margin\| | Mean signed Δ margin | Max \|ΔDTW\| | Flips |
|---|---|---|---|---|---|---|---|---|
| **A — read-side fix alone** | incumbent | fixed | in | 0.034 pp | 0.047 pts | +0.003 | 20.19 pp | **13** |
| **B — refit alone** | refit | shipped | in | 0.076 pp | 0.081 pts | +0.002 | 9.75 pp | **14** |
| **C — exclusion alone** | incumbent | shipped | **out** | 0.000 pp | 0.019 pts | −0.029 | 53.56 pp | **25** |
| **v1.3 — combined** | refit | fixed | **out** | **0.104 pp** | **0.108 pts** | **−0.024** | 53.66 pp | **33** |
| *Monte Carlo floor* | refit | fixed | out | 0.012 pp | 0.015 pts | −0.001 | 1.76 pp | **1** |

**How to read it.** The three corrections have completely different shapes.

- **The read-side fix and the refit are broad and shallow.** Both touch every
  kick and move the typical game by a twentieth of a point. Neither is visible
  in a single game unless that game turned on a long field goal — where A's max
  reaches 20 pp of DTW.
- **The exclusion is narrow and deep.** Its median across all games is *exactly
  zero*, because 2,474 games contain no blocked kick at all; on the 287 that do,
  it moves the median game by **2.68 points** and flips 22. That is the shape
  document 26 §9 described and the reason amendment C-1 exists: a materiality
  floor reads the 0.000 and stops, and a correctness gate reads the 2.68.
- **The combined row is not the sum**, and document 30 §8 said so before the
  numbers existed. 33 flips against 13 + 14 + 25 = 52. DTW is a bounded
  non-linear function of luck; the refit raises the `p_make` the exclusion then
  removes; and dropping rows reshuffles the shared random stream.
- **The Monte Carlo floor is one game**, measured rather than assumed, exactly
  as v1.2's was. Every number above it is the corrections' doing.

### The two isolated studies, in their own terms

**Part A** (`research/44_read_side_fix.py`, incumbent posterior on both arms):
`p_make` falls a mean of **0.80 pp** on field goals and rises **0.98 pp** on
every extra point. On the 1,393 games carrying a 50+ yard attempt the median
game moves 0.145 points and 12 of the 13 flips are there. Per row the fix agrees
with document 27 §14f to four decimals — **0.0447 EPA** of misbooked luck per
field goal against §14f's 0.0446.

**Part B** (`research/45_blocked_exclusion_c1.py`, on v1.3's arithmetic):

| | 287 games with a blocked kick | All 2,761 games with a kick |
|---|---|---|
| Median \|ΔDTW\| | **0.983 pp** *(document 26 measured 1.167 pp on v1.2)* | **0.000 pp** |
| Mean \|ΔDTW\| | 8.202 pp | 0.963 pp |
| Max \|ΔDTW\| | 60.70 pp | 60.70 pp |
| Median \|Δ deserved margin\| | **2.682 pts** *(document 26: 2.688)* | 0.018 pts |
| DTW side flips | **22** *(document 26: 22)* | 26 |

**Neither number is a pass rule**, and both are printed, which is the whole
content of amendment C-1. Document 26's floor was 1.6250 pp and its refit
counterpart 1.4409 pp; under Gate C they are context, not bars. The
reconciliation Gate C requires — removed luck × events against game movement —
agrees to a **median 0.019 points** and a maximum of 0.316, the residual being
the class tables re-pricing every remaining kick in the game.

## 4. One published number did not reproduce, and it should not have

`research/46_simulator_v13.py` checks the shipped posterior against document 27
§14c. Every fitted parameter matched to five decimals. **The league make-rate
curve did not:**

| Distance | Document 27 §14c printed | v1.3 prices |
|---|---|---|
| 30 yd | 96.48% | **96.74%** |
| 40 yd | 87.05% | 87.05% |
| 45 yd | 80.04% | **79.88%** |
| 50 yd | 73.03% | **71.41%** |
| 55 yd | 67.39% | **61.17%** |

**The explanation is the defect this release fixes, applied to §14c's own
numbers.** `league_curves` in `research/42_fg_refit.py` prices through
`FieldGoalModel.league_make_probability`, which at the time discarded
`delta_cubic` — so §14c's table is a *quadratic* reading of a *cubic* posterior,
in both its columns. The check proves it rather than asserting it: with the
cubic term stripped, the production read side reproduces the published table to
the digit, on both the refit and the incumbent.

**What actually changes.** §14c's argument was about the *difference* between
the two posteriors, and that barely moves: +2.41 pp at 55 yards on the corrected
curve against the +3.21 pp printed. What moves is the level — a 55-yarder is
61.2%, not 67.4%. Document 27 §14c is annotated as superseded rather than
edited, because the pre-registration is the record.

**This was reported before it was resolved**, per the rule this project runs on.
It is a documented consequence of a defect already registered in document 27
§14h, not a new disagreement, and it does not touch any gate.

## 5. Two checks that caught their own author

Both are recorded because a check that only ever passes is not a check.

1. **The `core` check was wrong, not the code.** Document 30 §7b asserted that a
   blocked kick's EPA lands in `core`. Six of the 192 also carry a penalty flag,
   so their EPA lands in `penalty` — a bucket that is equally not neutralized.
   The invariant that actually holds is the five-way partition, and that is what
   the shipped check asserts.
2. **The reconciliation had its sign inverted.** Removing a row whose luck was
   `L` raises deserved margin by `+L × slope`; the first implementation used a
   minus and produced a median gap of 5.354 points, larger than the movement it
   was reconciling. The sign is the whole content of the check — it is what
   would make a correction that moved games the wrong way visible — and the
   corrected version lands at 0.019.

## 6. The rematch validation, re-run as a check

*Script: `research/47_rematch_v13.py`. Document 27 §10 committed to re-running
it; document 12 measured the test as nearly blind below roughly 20% damage.*

| | Document 07, on v1.0 | **v1.3** |
|---|---|---|
| Rematch pairs | 531 | 531 |
| Mean Δ log loss (deserved − actual) | −0.00159 | **−0.00357** |
| 95% CI upper bound | +0.00377 | **+0.00218** |
| Non-inferiority margin | +0.010 | +0.010 |
| Harness check (both predictors) | pass | pass |

Deserve-to-win remains non-inferior to the actual result at predicting a
rematch, and the point estimate still favours it. **This is reported as a check
and not as a gate**, and it is not evidence that v1.3 is better than v1.2: the
design has 7.2% power against superiority, and the movement here is well inside
what a nearly blind instrument produces by chance.

## 7. The ship template, amended

Document 19 established the template. **v1.3 adds one permanent item, and it is
Phase 7's lesson:**

> **The round trip is part of every ship.** Fit → artifact → production read →
> reprice, agreeing to numerical tolerance on **every kick in the fitted
> population**, on **every posterior the release ships or replays**, pinned by a
> test in the suite as well as measured in the build script.

The defect this release fixes lived through v1.1 and v1.2 and was found by a
check document 27 §9d asked for in one sentence as a formality. The checklist a
version must now clear:

1. Every deterministic number reproduced against the document that published it,
   with a mismatch treated as a stop rather than a reconciliation.
2. **The round trip**, above.
3. The previous version replayed exactly under the new code.
4. The ledger identity, and row-count arithmetic with the population change
   named.
5. The Monte Carlo floor measured at a second seed, never assumed.
6. A decomposition when a release carries more than one change.
7. `uv run pytest -q` and `uv run ruff check .` green before the merge commit.

## 8. What v1.3 is

| | v1.2 | v1.3 |
|---|---|---|
| Field-goal posterior | `trace_fg_weather.nc` | **`trace_fg_refit.nc`** |
| Training population | 23,549 kicks | **23,247** — blocked kicks removed |
| Curve the product prices | quadratic (a cubic fit, read wrong) | **cubic, as fitted** |
| Extra points | priced as 33-yard field goals | **`delta_xp` + `lambda_xp` transfer** |
| Field-goal ledger rows | 10,731 | **10,539** |
| Extra-point ledger rows | 12,818 | **12,708** |
| Fumble ledger rows | 6,505 | 6,505 |
| Total ledger rows | 30,054 | **29,752** |
| `sigma_kicker` | 0.342 | **0.385** |
| Field-goal `w` at the median kicker-season | 0.285 | **≈0.336** *(implied; the 0.285 has no derivation in this repo)* |
| Median \|Δ deserved margin\| vs v1.2 | — | 0.108 points |
| Games whose verdict changes | — | 33 of 2,761 |
| Rematch validation | not re-run | **re-run as a check; passes** |

Document 05 §3's treatment table gains **no row**. Two rows have their
populations narrowed and both record that a blocked kick is `core`.

## 9. Known defects, carried forward

| Defect | Status |
|---|---|
| `w = 0.285` has no derivation in this repository | **Open.** v1.3's metadata records the implied 0.336 and says it is implied |
| Document 05b's 50–54 yard bin, the missing wind direction, the non-independent kicker-seasons, the cubic's absent mechanism story | **Unchanged** |
| `research/42c_read_side_defect.py` cannot reproduce after this release | **Accepted.** Once the read side agrees with the fit it reports ~0; the measurement is preserved in its JSON and document 27 §14f |
| Document 01's published variance shares are not regenerated | **Open, deliberate.** 192 blocked kicks moved EPA from `fg_luck` to `core`, in the direction Gate A implies |
| The empirical class tables were fitted on the narrowed population | **Closed by this release** — the masks are shared, so the table and the ledger cannot drift apart |
| Everything document 05 §5 and 05b §7 already carry | **Unchanged** |

## 10. What the maintainer is being asked

**Nothing.** All four decisions were made on 2026-08-18 and this is the record of
executing them. The one thing worth his eye is §4: a table document 27 published
turned out to have been printed through the defect this release fixes, and the
correction is annotated in place rather than silently rewritten.
