# 67 — Stadium elevation in the make-probability model, results

*Script: `research/81_fg_elevation.py`. Design, prior and gates fixed by document
66, committed at `e97a918` before this script existed; the null bound and power
table by `research/81a_fg_elevation_power.py` at `abf9beb`, before document 66.
Effect ladder: `research/81b_fg_elevation_effects.py`, which fits nothing and
reads the posterior. Raw report: `research/outputs/81_fg_elevation.json`.*

*Nothing in this document ships. The verdict is the maintainer's fork.*

---

## 1. Gate outcomes, stated first

| Gate | Rule | Result | |
|---|---|---|---|
| **E-1** sampler health | 0 divergences, r̂ < 1.01, ESS > 400 | 0 divergences, max r̂ 1.0057, min ESS bulk 1,013 / tail 1,536 | **PASS** |
| **E-2** elevation × distance calibration | worst standardized miss ≤ 94.5th pct of its own reference | 2.019 vs 2.729, 11 cells | **PASS** |
| **E-3** elevation resolvable | 89% lower bound on `beta_elev` above −0.00790 **and** above 0 | lower bound **+0.01649** | **PASS** |
| **E-4** distance calibration preserved | worst standardized miss ≤ 94.5th pct | 2.159 vs 2.575, 8 cells | **PASS** |
| **E-5** held-out log-loss must not worsen | elevation ≤ control, 5 game-grouped folds | 0.257903 vs 0.257949 | **PASS** |
| **E-6** without Denver | reported, no pass rule | `beta_elev` **+0.0955 [+0.0125, +0.1805]**, excludes zero | reported |
| **E-7** materiality | reported, no pass rule | **498 of 23,247 kicks** move ≥ 1 pp | reported |

**All five pass-rule gates pass.** Two of the three interesting facts in this
round are in the two gates that have no pass rule, and §4 and §5 are where the
reader should spend their time.

---

## 2. The coefficient

```
beta_elev = +0.0602  [+0.0165, +0.1043]  log-odds per 1,000 feet
```

Translated by `research/81b_fg_elevation_effects.py`, as a make-rate change
against a stadium at the population's mean elevation of 569 feet:

| Distance | League make % at 569 ft | **Denver** (5,280 ft) | **Mexico City** (7,280 ft) |
|---|---|---|---|
| 33 yd (the extra point) | 94.4% | **+1.24 pp** [+0.39, +2.03] | +1.67 pp [+0.54, +2.64] |
| **45 yd** | 79.9% | **+4.09 pp** [+1.24, +6.76] | **+5.56 pp** [+1.75, +8.99] |
| 50 yd | 71.5% | +5.34 pp [+1.58, +8.93] | +7.33 pp [+2.23, +12.00] |
| 55 yd | 61.2% | +6.42 pp [+1.84, +10.85] | +8.91 pp [+2.61, +14.86] |

The Denver-against-**sea-level** contrast a reader actually pictures is slightly
larger, because a sea-level stadium is priced *below* the centre: at 45 yards it
is **+4.65 pp** [+1.39, +7.73], and at 55 yards +7.23 pp [+2.06, +12.27].

**The effect grows with distance without an interaction term, and document 66 §3
predicted that in advance.** The logit link does the work: the same log-odds
shift is worth 1.24 pp on a 33-yard kick priced at 94% and 6.42 pp on a 55-yarder
priced at 61%. This is the pre-registered reason a linear term was chosen over a
distance interaction, and the outcome is consistent with it.

**Nothing else in the model moved.** Against document 27's published means:
`alpha` 1.909 vs 1.907, `beta` −0.1158 vs −0.1159, `gamma` 0.247 vs 0.249,
`delta_cubic` −0.081 vs −0.081, `sigma_kicker` 0.384 vs 0.385, `beta_wind`
−0.0221 vs −0.0224. Adding elevation did not buy its fit by disturbing anything
else, which is the cheapest evidence that the new column is not absorbing an
existing term.

---

## 3. The caveat that belongs next to the headline number

**The point estimate sits below the effect this design was powered to detect.**
Document 66 §6, committed before the fit, put the minimum detectable Denver gain
at 45 yards at **5 pp (power 0.907)**, with 4 pp reaching only **0.780** and 2 pp
**0.472**. The fitted estimate is **4.09 pp** — in the band where the design
misses a real effect of that size about one time in five.

That has a consequence for how the 4.09 pp should be read, and it is a
consequence of arithmetic rather than of doubt about the fit. When a design
detects an effect near its own resolution limit, the estimates that clear the
bar are the ones that happened to land high; the estimates that landed low did
not clear it. So **the true effect is more likely below 4.09 pp than above it**,
and the honest one-line summary is *"a Denver effect of a few percentage points
at 45 yards, resolvable but near the edge of what ten seasons can size."* How
much inflation this implies is measurable from the §6 power machinery — it is
the first suggested avenue in §7 and it has not been run.

The 89% interval already carries some of this: its lower bound is +1.24 pp.

---

## 4. Gate E-6 — it is elevation, not Denver

This is the round's most useful result, and it went the way document 66 §7 said
it probably would not.

Refitting with **all 680 Denver kicks removed**:

```
beta_elev = +0.0955  [+0.0125, +0.1805]   (interval EXCLUDES zero)
```

Document 66 §7 committed the reading in advance: *"a claim that this round
measured elevation rather than Denver requires the Denver-excluded interval to
exclude zero on its own."* **It does.** The 22,567 remaining kicks — with their
covariate topping out at Las Vegas's 451 kicks in a dome at 2,030 feet, plus 32
in Mexico City, 14 in Munich and 12 in São Paulo — still put the whole 89%
interval above zero.

Two honest qualifications:

- **The interval is 1.91× as wide**, which is exactly what §7 expected, and the
  point estimate is *larger* (+0.0955 against +0.0602), not smaller. A larger
  estimate on a thinner design is not evidence of a bigger effect; it is what a
  noisier estimate looks like. The two intervals overlap across almost their
  whole length.
- **Las Vegas is doing most of the work**, and Allegiant is a dome. Its 451 kicks
  are the only substantial block between 1,500 and 3,000 feet. The
  interpretation "thin air lengthens kicks" survives Denver's removal; the
  interpretation "Denver's stadium is peculiar" does not explain the E-6 result,
  which is the whole point of running it.

What this does **not** establish is that Empower Field has no site-specific
effect of its own on top of the altitude. Nothing in this design can separate
that, and document 66 §3 said so before the fit.

---

## 5. Gate E-5 — real in sample, worth almost nothing out of sample

The gate passes, and the margin is the finding:

| | mean held-out log-loss per kick |
|---|---|
| elevation arm | **0.257903** |
| control arm | **0.257949** |
| difference | **−0.000046** [−0.000198, **+0.000122**] |

Five folds grouped by `game_id`, both arms refitted inside each fold, 2,000-
resample paired bootstrap over the 2,761 games. The pass rule the maintainer set was a
**direction**, and the direction is right. But the 89% interval **straddles
zero**, and the improvement is 0.018% of the log-loss.

And on the 680 Denver kicks alone — the kicks the term exists for — **the
elevation arm is slightly worse**: 0.233229 against the control's 0.232900.

The two readings that survive this, both of which are defensible:

- **The term is right and prediction cannot see it.** 712 of 23,247 kicks are
  above 3,000 feet. A 4 pp shift on 3% of the data moves a mean log-loss by about
  the amount observed. Held-out log-loss over the whole population is a blunt
  instrument for a covariate this concentrated, and E-3's interval is the sharper
  one.
- **The term is fitting noise that does not generalize.** The Denver-only
  reversal is the evidence for this reading, and 680 kicks is enough that it is
  not nothing.

Document 66 §9 pre-registered what to do if E-3 passed and E-5 failed — not
adopted, and reported as exactly that. It did not pre-register this: E-5 passing
on a difference indistinguishable from zero. **The gate ladder as written says
adopt; the numbers say the ladder cannot tell.** That is the fork, and it is §7's
first question.

---

## 6. Gate E-7 — how much would actually move

Against `trace_fg_refit.nc`, the posterior the product reads today, on all 23,247
kicks:

- **498 kicks (2.14%) move by ≥ 1 percentage point** of make probability;
  **196 (0.84%) by ≥ 2 pp**. Largest single move **9.42 pp**.
- **174 of 2,761 games (6.3%)** hold at least one kick that moved ≥ 1 pp.
- The population mean shift is **−0.002 pp**. Adding elevation redistributes;
  it does not inflate.

| Band | Kicks | Mean shift | Moved ≥ 1 pp |
|---|---|---|---|
| ≥ 3,000 ft | 712 | **+1.72 pp** | **363** |
| 1,500–3,000 ft | 477 | +0.47 pp | 77 |
| 500–1,500 ft | 9,433 | +0.05 pp | 6 |
| < 500 ft | 12,625 | **−0.16 pp** | 52 |

The 52 low-elevation kicks that move are the mirror of the Denver ones: once
elevation enters, a sea-level kick is priced slightly *harder* than before, and
on the longest attempts that crosses a point. **Everything this change touches,
it touches in 174 games**, and 89 of those are the games played above 3,000 feet.

---

## 7. Gate E-2 in detail, and what did not fail

The linear log-odds form was not rejected: worst standardized miss 2.019 against
a 2.729 reference across 11 cells, so **the distance-shift fallback named in
document 66 §7 was not reached.**

The cell worth looking at anyway is `>=3000ft|40` — 100 attempts from 40–49
yards at altitude, observed 81.0% against a predicted 85.0%, a **−3.80 pp** miss
(standardized 1.07, comfortably inside). It is the one cell where the elevation
arm looks *optimistic* at altitude, it is the smallest cell in the table, and it
is the cell a distance-interaction form would have been built to fix. On 100
attempts it is not evidence of anything. It is recorded because it is the place a
future round would look first.

---

## 8. Defects added by this round

| Defect | Evidence | Status |
|---|---|---|
| `beta_elev` is not separable from an Empower-Field-specific effect | 680 of 712 kicks above 3,000 ft are Denver; document 66 §3 | **Open by construction.** Gate E-6 rules out *Denver-only*, not *Denver-also* |
| The estimate sits below the design's own resolution | §3; power 0.780 at 4 pp | **Open.** Quantifiable from the §6 machinery, not yet quantified |
| Held-out log-loss cannot adjudicate a covariate on 3% of the data | §5 | **Open.** A Denver-weighted or altitude-restricted scoring rule was not pre-registered and is not introduced now |
| The elevation table carries no measurement uncertainty | document 66 §2 | **Accepted, not open.** ±50 ft is 5% of one prior SD |

---

## 9. What a v1.4 round would have to own, if there is one

Unchanged from document 66 §9, restated with the numbers now attached: the
`beta_elev` read side in `FieldGoalModel`, the 0.5687 kft centring constant
travelling with the posterior, the document 30 §5a round-trip check between fit
and read side, the ledger and DTW impact on the 174 affected games, and a
re-render of every figure that prints a make probability. **The article is frozen
and stays frozen until the maintainer forks.**
