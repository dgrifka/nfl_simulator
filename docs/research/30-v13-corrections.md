# 30 — The two corrections v1.3 carries, pre-registered

*Written 2026-08-18, **before either correction has been measured**. The maintainer
approved four things on this date: amendment C-1 (document 28), the
make-probability refit (document 27), the fix to the read-side defect (document
27 §14f) and a re-measurement of document 26's blocked-kick exclusion under C-1
on the refit's arithmetic. The refit is already measured — document 27 §14 is
its record and nothing here re-opens it. **The other two are unmeasured, and
this document fixes what will be measured and how it will be read before a
number exists.** Committed to git before the measuring scripts are written.*

*Inputs: document 27 §14f (the read-side defect and its size in `p_make`),
document 26 §2, §5h and §8 (the exclusion, its masks and the fumble-overlap
trap), document 05 §2 Gate C (the rule this round is the first to run under),
document 18 §4b (the two-population reporting rule), document 19 (the ship
record this feeds).*

*Tier: **production code change** on both parts. Part A changes what the
simulator computes from an already-adopted posterior; Part B changes which plays
the ledger books. Neither touches a prior, a likelihood or a sampler constant.*

---

## 1. One-page story

### The question

Two corrections are approved and neither has a game-level number attached.

> **Part A.** The shipped simulator never reads `delta_cubic`, `delta_xp` or
> `lambda_xp`. It prices kicks on a quadratic curve whose `gamma` was fitted
> jointly with a cubic term it then discards, and prices extra points as plain
> field goals from 33 yards. **What does fixing that do to a deserve-to-win
> number?**
>
> **Part B.** The field-goal and extra-point components neutralize 302 kicks
> that were blocked — plays Gate A denies, because the ball never flew.
> Document 26 measured the exclusion at 1.167 pp against a materiality floor it
> missed. That floor no longer governs it, and the arithmetic underneath has
> changed. **What does the exclusion do on the refit, under Gate C?**

### The answer

**Unseen, both parts.** §4 and §7 predict directions and register first-order
sizes so the predictions can be wrong in public. No arm has been run.

### Five things to hold onto

1. **Part A is not governed by Gate C and never was.** Document 27 §14f said so
   explicitly: it is a plumbing defect, not a Gate A violation. No materiality
   threshold applies to it either, and that is not an exemption invented here —
   document 05b's ship rule for this model was gate-based from the start. Part A
   is governed by a correctness gate of its own (§5), and its size is reported.
2. **Part B is the first candidate ever measured under Gate C.** Document 26 is
   its pre-registration for clauses 1–4; §6 shows the clauses against it line by
   line rather than asserting compliance. What is new here is the arithmetic it
   runs on and the reconciliation C-1 requires.
3. **The three corrections must decompose.** v1.3 carries the refit, Part A and
   Part B together. Each is measured **alone against v1.2** as well as in
   combination, so a reader can attribute the movement. The combined number is
   not the sum of the three, and §8 says so in advance rather than explaining it
   afterwards.
4. **Part A moves `p_make` in both directions at once.** Long field goals are
   priced too generously today and every extra point is priced too harshly. The
   two errors book luck of opposite sign, so the game-level effect may be much
   smaller than either. That is a prediction, and §4 commits to it.
5. **Nothing here re-opens the refit.** Document 27's gates were run and passed
   on a fixed pre-registration. This document consumes that verdict.

### Statistic convention

Posterior means with 89% equal-tailed intervals. Movement statistics are medians
and means of absolute change with the max stated, on both populations document
18 §4b requires.

---

## 2. Part A — the defect, stated as code

`src/nfl_simulator/fg_model.py`, in the shipped version:

```python
def _logit(self, distance):
    centred = distance - self.distance_centre
    return self.alpha + self.beta * centred + self.gamma * centred**2 / 100.0
```

Three fitted parameters never arrive:

| Parameter | What the fit means by it | What the read side does |
|---|---|---|
| `delta_cubic` | The cubic distance term of the adopted Phase 3 curve, `delta_cubic · d³ / 1000` | Never read by `from_posterior`, never applied by `_logit` |
| `delta_xp` | The log-odds offset of an extra point over a field goal from the same distance | No extra-point path exists at all |
| `lambda_xp` | How much of a kicker's field-goal ability transfers to extra points | Same |

The consequence, sized in document 27 §14f on the shipped population: field
goals of 55–59 yards are priced **6.80 pp too generously**, 2,117 field goals are
off by more than a point, and all 12,818 extra points are off by **−0.98 pp** on
average.

### The correction, and why it is the only one available

The fitted model is the definition of `p_make`. The read side is supposed to be
an evaluation of it, and it is not. **The correction is to make the read side
evaluate the fitted linear predictor exactly** — no new parameter, no choice of
form, nothing to tune. Concretely:

- `FieldGoalModel` gains `delta_cubic`, `delta_xp` and `lambda_xp`, each
  optional and each defaulting to *absent*.
- `_logit` gains the cubic term when `delta_cubic` is present.
- `make_probability` gains an `extra_point` flag. When it is set, `delta_xp` is
  added to the log-odds and the kicker's effect is scaled by `lambda_xp` —
  `kicker · (1 + (lambda_xp − 1) · is_xp)`, which is the fit's own expression.
- `simulator.extra_point_events` passes `extra_point=True`. Nothing else calls
  the read side with an extra point.
- **Absent parameters stay absent.** A pre-weather Phase 2 posterior, and the
  quadratic-arm traces, load and price exactly as they do today.

---

## 3. Part A — DAG edit

**None.** No node, no edge, no parameter and no prior changes. The generative
model is document 05b §10's, unchanged; this makes the consumer agree with it.

---

## 4. Part A — mechanism story and prediction

`delta_cubic` is negative (−0.0685 incumbent, −0.0811 refit), and it multiplies
`d³/1000` on centred distance. Below 40 yards `d³` is negative, so dropping the
term makes short kicks look **harder** than fitted; above 40 yards it makes long
kicks look **easier**. `delta_xp` is positive (+0.167 incumbent), so dropping it
makes every extra point look harder than fitted.

**What that does to booked luck.** Luck is `(realized − p) × swing`. Overstating
`p` on a long field goal books too much bad luck on a miss and too little good
luck on a make. Understating `p` on an extra point does the reverse. Document 27
§14f measured the signed totals across ten seasons: **−385 EPA on field goals
and +128 EPA on extra points**, which partly cancel.

### The prediction, registered so it can be wrong

- **Direction.** Fixing the read side lowers `p_make` on kicks beyond ~45 yards,
  raises it slightly under ~35, and raises it on every extra point.
- **First-order size.** −385 + 128 = **−257 EPA** of misbooked luck across
  2,761 games with a kick, or about −0.093 EPA per game. At `points_per_epa`
  0.8389 that is **≈0.078 points of deserved margin on the mean game**, with the
  effect concentrated on games carrying a 50+ yard attempt.
- **Median.** Because the two errors cancel within a game, the *median* game is
  predicted to move **less than 0.1 points**, and the DTW side-flip count is
  predicted to be **under 25** — the scale of the refit's own 18.
- **What would make this prediction wrong.** If the field-goal and extra-point
  errors do not cancel within games — for example if long attempts cluster in
  the same games as many extra points — the per-game movement would be larger
  than the signed totals suggest.

---

## 5. Part A — pre-registered gates

Three gates, all binding, all correctness rather than size. **A failure on any
of them stops the ship and is reported to the maintainer rather than reconciled.**

### 5a. Gate S-1 — the round trip *(the binding gate)*

> Priced through `FieldGoalModel`, every kick in the fitted population must
> reproduce the fit script's own `make_probabilities` to
> **max |Δp| ≤ 1e-9**, on **both** posteriors — the incumbent
> (`trace_fg_weather.nc`, 23,549 kicks) and the refit (`trace_fg_refit.nc`,
> 23,247 kicks) — and on **both** kick kinds.

This is the check document 27 §9d asked for as a formality and that found the
defect. It becomes a gate here, and §9 makes it a permanent item on the ship
template so no future version can ship a read side that disagrees with its own
fit.

**Pinned by a test**, on a synthetic posterior written through ArviZ and read
back through the real loader, so the suite does not depend on a regenerable
artifact.

### 5b. Gate S-2 — backward compatibility

> A posterior carrying none of the three parameters must price **bit-identically**
> to the shipped code. The Phase 2 trace must still load, and passing weather to
> a model fitted without weather must stay a no-op.

v1.1's and v1.2's ledgers are reproducible artifacts of this repository, and a
change that quietly re-prices them destroys the record. Pinned by tests.

### 5c. Gate S-3 — the ledger must still sum

> Part A re-prices rows. It must add none and remove none: 10,731 field-goal
> rows and 12,818 extra-point rows on the incumbent population, unchanged, and
> `deserved = actual − total_luck × points_per_epa` still true of every game.

### 5d. The impact report — required, no threshold

Measured with the **incumbent posterior on both arms**, so Part A is isolated
from the refit. Reported on both populations:

| Population | Why |
|---|---|
| **All games with a kick** (~2,761) | Every kick is repriced; this is the primary number |
| **Games with a 50+ yard field-goal attempt** | Where document 27 §14f localizes the error |

Median, mean and max |ΔDTW|; median |Δ deserved margin|; mean **signed** Δ
deserved margin; DTW side flips. Plus a **reconciliation**: the signed luck
correction per component times its row count, converted at `points_per_epa`,
read against the measured mean movement. A correction whose game-level size
cannot be explained by its own per-row arithmetic is a red flag, and this is
where it would show.

**No number in the impact report can fail Part A.** The gates are S-1 to S-3.

### 5e. Decision rule, committed in advance

- **S-1, S-2 and S-3 all pass** → Part A ships as part of v1.3, with the impact
  report attached.
- **Any of them fails** → nothing ships, the branch keeps the code, v1.2 stays
  authoritative, and §10 is the write-up. **Stop and ask the maintainer.**

---

## 6. Part B — the exclusion, qualified against Gate C clause by clause

Gate C (document 05 §2, amendment C-1) admits a correction only if all four
clauses hold, *each stated in a pre-registration committed before the correction
is measured*. Document 26 is that pre-registration, committed at `e33edc6`. The
clauses are checked against it here rather than asserted.

| Clause | Where document 26 satisfies it |
|---|---|
| **1. A correction, not an addition** | §2 "what exclusion means for the play" and §5h: the candidate removes 192 field-goal and 110 extra-point rows and books **no** row on any play that carries none today. Row-count arithmetic in §8's Gate P-4 |
| **2. A violation memo** | §2: the branch document 05 §2 admitted for the kicking components is *a kick in flight*, and on a blocked kick there is none. What resolved it is named — **the defending team's rush beat the protection**, a football act by a specific side |
| **3. The memo argues the other side, and measures it** | §2's "why a separate blocked class is inadmissible", resting on document 25 §2's measured answer to the strongest objection: the blocker and the recoverer are the same man on only **16 of 144** blocked-kick aftermaths |
| **4. The correction Gate A implies, not a free parameter** | §2: exclusion, because the play has no branch at all. The alternative — a separate blocked class — is argued down on mechanism in the same section, before any size was known |

**Clause 1 is the one worth re-checking rather than inheriting**, because it is
the clause that decides whether the materiality floor governs. It is re-verified
numerically in §7's ledger-sum check, not taken on the document's word.

### What is new, and therefore pre-registered here

Document 26 measured on v1.2's arithmetic. Everything under the candidate has
moved:

- The **posterior** is the refit, so `p_make` is 1.33 pp higher on average.
- The **read side** is Part A's, so long kicks are no longer over-priced.
- The **floor is gone**. Document 26's 1.6250 pp is not a bar any more; document
  27 §14g measured its refit counterpart at 1.4409 pp and it is reported as
  context, never as a pass rule.

---

## 7. Part B — pre-registered gates

Gate C's list, unchanged, with the threshold removed and the report mandatory.

### 7a. Identification — rejected rows printed, not counted

> The 192 blocked field goals and 110 blocked extra points are identified from
> `field_goal_result == "blocked"` and `extra_point_result == "blocked"`, on the
> already-narrowed attempt masks. **Every removed row is printed** — game id,
> play id, season, distance, kicker — per document 20 §9, and written to the
> results JSON.

### 7b. The ledger must still sum — including the trap

> Field-goal rows 10,731 → **10,539**; extra-point rows 12,818 → **12,708**;
> exactly −192 and −110 and nothing else. **Fumble rows unchanged**, because
> four blocked field goals also carry a v1.2 fumble row and a frame-level filter
> would silently delete them (document 26 §8). The production implementation
> narrows `fg_attempt_mask` and `xp_attempt_mask`, **never the play frame**, and
> a test pins that the four fumble rows survive.
>
> **The removed luck must land in `core`.** `decompose_plays` shares
> `fg_attempt_mask`, so a blocked kick's EPA stops being `fg_luck` and becomes
> `core` on the same play. The five components must still sum to `epa_home` on
> every row, and the game-level `fg_luck` drop must equal the `core` rise to
> floating-point tolerance.

### 7c. The dial gate — absent by design

No `w` is assumed. The correction removes rows rather than re-weighting them,
which is document 26 §5e's reasoning, unchanged.

### 7d. The materiality **report** — mandatory, no pass rule

On both populations, both printed in the verdict, always:

| Population | n |
|---|---|
| **Games containing a blocked kick** | 287 |
| **All games with a kick** | ~2,761 |

Median and mean |ΔDTW|, max |ΔDTW|, median |Δ deserved margin|, mean signed Δ
deserved margin, and DTW side flips. Reported against document 26's v1.2
numbers (1.167 pp median on the blocked games, 0.000 pp on all games) so the
movement caused by the new arithmetic is visible.

### 7e. The reconciliation Gate C requires

> Per-event luck removed × number of events, converted at `points_per_epa`, read
> against the measured game-level movement.

Document 27 §14g measured the refit's mean |luck| on a blocked field goal at
**3.486 EPA** and on a blocked extra point at **0.961 EPA**. That is
`192 × 3.486 + 110 × 0.961 ≈ 775 EPA` of luck removed in total, or about **2.70
EPA per affected game** across 287 games — **≈2.27 points of deserved margin**
at `points_per_epa` 0.8389. The measured median must be readable against that
number. **This is a prediction, and a large disagreement with it is a finding to
report, not a number to reconcile away.**

### 7f. Decision rule, committed in advance

- **Identification, ledger-sum and the dial gate pass** → the correction is
  correct, and it ships as part of v1.3 **at whatever size it turns out to
  have**, with the materiality report attached. That is Gate C's verdict rule
  quoted, not a new one.
- **Any of the three fails** → nothing ships from Part B, and the failure is
  reported as prominently as a pass would have been.
- The materiality report **cannot** fail it. If the numbers come out small, they
  are printed and it ships anyway; if they come out large, they are printed and
  it ships. That is the whole content of amendment C-1.

---

## 8. What the ship must show — the decomposition

v1.3 carries three corrections. Measured alone against v1.2, on the same seeds:

| Arm | Posterior | Read side | Kicking population |
|---|---|---|---|
| **v1.2** *(reference)* | incumbent | shipped | includes blocked |
| **A — read-side fix alone** | incumbent | fixed | includes blocked |
| **B — refit alone** | refit | shipped | includes blocked |
| **C — exclusion alone** | incumbent | shipped | excludes blocked |
| **v1.3 — combined** | refit | fixed | excludes blocked |

**The combined movement will not equal A + B + C, and that is expected**, for
three reasons stated before the numbers exist:

1. DTW is a bounded non-linear function of luck, so shifts do not add.
2. The corrections interact by construction — the refit raises `p_make`, which
   raises the luck the exclusion then removes (document 27 §14g).
3. Removing rows shifts the shared random stream later components draw from.
   Document 19 §4 measured that Monte Carlo floor at one game for v1.2, and
   **it is re-measured for v1.3** rather than inherited.

Arm B is already measured — document 27 §14e — and is re-run inside the same
harness so that all four rows come from one instrument.

---

## 9. What adoption means, fixed now

- `src/nfl_simulator/fg_model.py`, `simulator.py` and `components.py` change
  under test-driven development on branch `feat/phase8-v13`.
- The v1.3 artifacts are `dtw_games_v13.parquet`, `dtw_ledger_v13.parquet`,
  `model_metadata_v13.json`. **v1.1's and v1.2's are left untouched.**
- The v1.3 build reads `trace_fg_refit.nc` and the refit's centres.
- **Gate S-1 joins the ship-record checklist permanently** (document 31 §7):
  fit → artifact → production read → reprice, agreeing to numerical tolerance on
  every kick, pinned by a test. That is Phase 7's lesson written into the
  template so it cannot be forgotten.
- Document 05b §2 loses the blocked-kicks bullet and gains its replacement; §7's
  register loses that row; §11's fitted table gains a refit column and its
  extra-point claim is corrected (document 27 §10 and §14c).
- Document 05 §3's treatment table: **no new row.** The field-goal row's `w`
  moves from 0.285 to about 0.336 (document 27 §14c) and both kicking rows
  record that a blocked kick is `core`.
- **The rematch validation is re-run and reported as a check, not a gate**
  (document 27 §10, document 12's blindness caveat).

## 10. Kill and rollback

On any gate failure in §5e or §7f: no production module changes, v1.2 stays
authoritative, the branch keeps the code, and §11 is the record. On a pass,
nothing merges until the ship record is written and the suite is green.

---

## 11. Known-defect register, carried into this round

| Defect | Evidence | Status |
|---|---|---|
| **Document 05b's published `w = 0.285` has no derivation in the repository** | Document 27 §14c could only recompute it as an arithmetic implication | **Open.** v1.3's metadata records the implied 0.336 and says it is implied |
| **Blocked kicks stay in `components.py`'s empirical class tables until Part B lands** | Document 27 §12 | **Closed by Part B**, if Part B ships |
| **The Phase 7 defect-sizing script cannot reproduce after Part A** | `research/42c_read_side_defect.py` measures the gap between the read side and the fit; once they agree it reports ~0 | **Accepted and stated.** The measurement is preserved in `42c_read_side_defect.json` and document 27 §14f. It is a record, not a regression test |
| **Narrowing `fg_attempt_mask` also moves the Phase 1 decomposition** | `decompose_plays` shares the mask, so 192 plays move EPA from `fg_luck` to `core` | **Deliberate and checked** (§7b). Document 01's published variance shares are not regenerated; the change is in the direction Gate A implies |
| Everything document 05 §5 and 05b §7 already carry | — | **Unchanged** |

---

## 12. Constants appendix

| Constant | Value | Where it lives |
|---|---|---|
| Blocked field goals / extra points removed | **192 / 110** | document 26 §3 |
| Blocked field goals also carrying a fumble row | **4** | document 26 §8 |
| Field-goal / extra-point rows, v1.2 | 10,731 / 12,818 | document 05b §11 |
| Field-goal / extra-point rows, v1.3 | **10,539 / 12,708** | §7b |
| Games with a blocked kick / with any kick | 287 / 2,761 | document 26 §8 |
| Gate S-1 tolerance | **1e-9** on `p_make` | §5a |
| `points_per_epa` | 0.8389 | `research/outputs/model_metadata_v12.json` |
| Document 26's v1.2 statistic | 1.167 pp median on 287 games | document 26 §8 |
| Document 26's floor, incumbent / refit | 1.6250 pp / 1.4409 pp | documents 26 §4, 27 §14g |
| Refit mean \|luck\| on a blocked FG / XP | 3.486 / 0.961 EPA | document 27 §14g |
| Read-side signed luck error, FG / XP | −385 / +128 EPA | document 27 §14f |
| Simulation draws (posterior / coin) | 200 / 800 | `research/31_simulator_v12.py` |
| v1.3 build seed / reshuffle seed | 20260817 / 20260819 | inherited from v1.2 so the arms differ by the change alone |
| Measurement seed, Parts A and B | 20260818 | documents 26 and 27's rounds |

Results are written back into this document as §13.

---

## 13. Results

*Scripts: `research/44_read_side_fix.py` (Part A) and
`research/45_blocked_exclusion_c1.py` (Part B). The gates were committed at
`e6920d0` before either script existed. The ship record is document 31.*

### The verdict, stated first

> **Both corrections pass everything that could fail them, and both are
> smaller in the median game and larger in the games they touch than the
> headline numbers suggested. Part A's round trip now agrees to 7e-15 where it
> once disagreed by 40 percentage points; Part B removes 302 rows, moves the
> median blocked-kick game by 2.68 points, and moves the median kick game by
> nothing at all.**

| Gate | Statistic | Verdict |
|---|---|---|
| **S-1** — the round trip | max \|read − fitted\| **5.7e-15** (refit), **6.8e-15** (incumbent), against a 1e-9 tolerance | **Pass** |
| **S-2** — backward compatibility | `dtw_games_v12.parquet` replayed on 2,761 games, max \|Δ\| **0.00e+00** | **Pass** |
| **S-3** — the ledger must sum | rows unchanged at 10,731 / 12,818 / 6,505; identity residual 0.00e+00 | **Pass** |
| **§7a** — identification | 192 + 110 rows, every one printed | **Pass** |
| **§7b** — ledger-sum | −192 and −110 exactly; the four fumble-overlap plays keep their row; partition residual 4.4e-16 | **Pass** |
| **§7c** — the dial gate | no `w` is assumed | **Absent by design** |
| **§7d** — the materiality report | printed on both populations, below | **Reported, no pass rule** |
| **§7e** — the reconciliation | median gap **0.019 points**, max 0.316 | **Reported** |

### Part A — what the fix does

| Population | n | Median \|ΔDTW\| | Median \|Δ margin\| | Mean signed | Flips |
|---|---|---|---|---|---|
| All games with a kick | 2,761 | **0.034 pp** | **0.047 pts** | +0.003 | **13** |
| Games with a 50+ yard attempt | 1,393 | 0.171 pp | 0.145 pts | +0.005 | 12 |

`p_make` falls a mean of 0.797 pp on field goals and rises 0.984 pp on every
extra point. Per row the correction is **0.0447 EPA** on a field goal and
**0.0100 EPA** on an extra point, against document 27 §14f's 0.0446 and 0.0100.

### Part B — what the exclusion does, on v1.3's arithmetic

| | 287 games with a blocked kick | All 2,761 games with a kick |
|---|---|---|
| Median \|ΔDTW\| | **0.983 pp** | **0.000 pp** |
| Mean \|ΔDTW\| | 8.202 pp | 0.963 pp |
| Max \|ΔDTW\| | 60.70 pp | 60.70 pp |
| Median \|Δ deserved margin\| | **2.682 pts** | 0.018 pts |
| Mean signed Δ deserved margin | −0.287 pts | −0.028 pts |
| DTW side flips | **22** | 26 |

Document 26 measured 1.167 pp, 2.688 pts and 22 flips on v1.2's arithmetic. The
median DTW movement **fell 16%** under the refit and the corrected read side,
while the deserved-margin movement and the flip count are essentially unchanged.

### How the predictions did

| Prediction | Outcome |
|---|---|
| §4: `p_make` falls beyond ~45 yd, rises on every extra point | **Right.** −0.80 pp mean on field goals, +0.98 pp on extra points |
| §4: median game moves less than 0.1 points | **Right**, 0.047 |
| §4: fewer than 25 side flips | **Right**, 13 |
| §4: first-order size ≈0.078 points on the mean game | **Wrong, and wrong in its derivation.** It read document 27 §14f's signed totals, which are in the kicking team's perspective, as if they were in the ledger's home-team perspective. The two components' errors largely cancel across teams; the mean game moves +0.003 points. The per-row means are the convention-free comparison and they agree to four decimals |
| §7e: ≈2.27 points of deserved margin on the median affected game | **Right and slightly low**, 2.68 |
| §7b: the removed EPA lands in `core` | **Wrong for six of 192.** Those also carry a penalty flag, so their EPA lands in `penalty`. The invariant that holds is the five-way partition |

### Defects this round found in its own instruments

| Defect | Status |
|---|---|
| The reconciliation's sign was inverted, producing a 5.354-point median gap | **Closed by correction**, recorded in document 31 §5 |
| Document 30 §7b's `core` assertion is false for six plays | **Closed by correction**, recorded in document 31 §5 |
| Document 27 §14c's league-curve table was priced through the defect §14f found | **Closed by disclosure.** Document 31 §4; §14c is annotated as superseded rather than edited |
