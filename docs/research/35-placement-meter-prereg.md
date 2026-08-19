# 35 — The placement meter, pre-registered

*Written 2026-08-19, **before any gated statistic on this round has been
computed**. Power calculation: `research/49_placement_power.py`, results in
`research/outputs/49_placement_power.json`. Committed to git before
`research/50_placement_meter.py` exists, so goalpost integrity is checkable by
commit archaeology.*

*This document executes step 1 of document 34 §10. Document 34 is the settled
design — its §2 decision table is the maintainer's and is not reopened here. What this
document adds is the part document 34 explicitly deferred: **thresholds, each
with its power measured first**, plus the input spec precise enough that
`research/50_placement_meter.py` chooses nothing.*

---

## 1. One-page story

The placement meter prices, in points, how much a team's play placement was
worth in one game — relative to indifferent placement of that same game's
production. It is descriptive. It books no ledger rows, so document 05's Gate A
is never in play, and it is displayed beside DTW% and never combined with it.

Six gates decide whether it ships. This document commits all six thresholds.
Three of them earned teeth and one did not, and the power table is what
decided:

| Gate | Question | Verdict on its own power |
|---|---|---|
| **M-1** identities | Do the books balance? | Identity check, no power needed |
| **M-2** calibration | Is the band honest? | **Teeth.** Coverage gate, false-alarm 0.000, power 0.92–0.98 at the miscalibrations that matter |
| **M-3** luck licence | Is the shipped score still luck? | **Teeth.** Power **0.900** at r = 0.12 |
| **M-4** skill preservation | Does the baseline leak skill? | **Teeth as a flag, with a stated ceiling.** Power 0.48 at a leak of 0.11, 0.80 only at ≈ 0.16 |
| **M-5** magnitude | How often does it matter? | Report, no threshold (Gate C convention) |
| **M-6** rematch | Does subtracting it lose information? | **Teeth.** Power to pass **0.973** — but the gate false-passes 32% of the time when the subtraction is pure harm |

### Three things this round learned before it fitted anything

1. **The exchangeability problem is not where document 34 expected it.** Doc 34
   §4 built the constraint ladder around the red zone, whose play-level EPA
   variance runs 1.19× the league. The **late-down cell is the real offender at
   1.99×**, with a mean 0.048 EPA per play below the league — worth −0.48
   points per team-game before any team does anything. No rung's variance
   matching addresses it; two rungs handle it only by accident (§5).
2. **Rungs 2 and 3 freeze the late-down cell entirely**, by arithmetic, not by
   measurement. §5 derives it. Their band answers a strictly narrower question
   than rung 1's, and that fact is on the record before any of them is run.
3. **The meter is big.** The home-minus-away placement differential has a
   standard deviation of **6.86 points** across 2,761 games. For scale, the
   whole v1.3 luck ledger moves the margin by a median of 2.37 points
   (document 33). The meter is not a rounding correction to the adjudication;
   it is a comparably sized second story about the same game, which is exactly
   why document 08 §6's ban on combining them is load-bearing.

---

## 2. The input stream, exactly

`research/50_placement_meter.py` reproduces this or it is wrong.

**Population.** The 2,761 games simulator v1.3 adjudicates
(`research/outputs/dtw_games_v13.parquet`), 2016–2025.

**Plays.** Document 08's S0–S2 filter, unchanged: `posteam` non-null,
`play_type` in {pass, run}, `epa` non-null, `down` non-null. **343,543 plays**
over **5,522 team-games**, every one of which has both teams present.

**Valuation.** Each play's EPA is `epa − luck_epa`, where `luck_epa` comes from
`research/outputs/dtw_ledger_v13.parquet` summed over that play's ledger rows,
and is **re-signed from the ledger's home perspective to the possessing team's**
before subtraction. This is the single place the two sign conventions meet.

- **5,541 plays** of the 343,543 carry a ledger row, and every one of them is a
  fumble. The other 24,211 ledger rows are field goals and extra points, which
  are kicks and therefore outside the filter already. The double-counting fix
  document 34 §5 asks for is real but small in extent and large in leverage:
  mean |repricing| is 2.07 EPA on the plays it touches.
- Re-pricing lowers league play-level EPA variance from document 08's 1.9158 to
  **1.8414**, which is the arithmetic of removing a ±4-EPA branch from 1.6% of
  plays.

**Cells.** Disjoint, in this order of application:

| Cell | Rule |
|---|---|
| red zone | `yardline_100 <= 20`, any down |
| late down | `down` in {3, 4} **and** `yardline_100 > 20` |
| everything else | the remainder |

**Score.** For one team-game, per cell,

```
cell_points = ( sum(epa_priced in cell) - n_cell * mean_all ) * points_per_epa
```

with `mean_all` the team's own luck-priced EPA per play in that game and
`points_per_epa = 0.8389`. Written this way rather than as
`(mean_cell − mean_all) * n_cell` so an empty cell is exactly 0 instead of
`0/0`. The **placement meter is the sum of the red-zone and late-down cells**;
the third cell is the negative of that sum. The game's headline is the
**differential**, home minus away.

---

## 3. Design parameters, measured before any threshold was set

League-pooled, no team identity survives any of them. This is document 08 §2's
category, and the disclosure rules there apply here (§9).

| Parameter | Value |
|---|---|
| Games / team-games / team-seasons | 2,761 / 5,522 / 320 |
| Scrimmage plays | 343,543 |
| Plays carrying a ledger row | 5,541 (all fumbles) |
| `points_per_epa` | 0.8389 |
| Play-level EPA mean / variance, all plays | +0.00068 / **1.8414** |
| … red zone | +0.01469 / **2.1845** |
| … late down outside the red zone | **−0.04814** / **3.6690** |
| … everything else | +0.01131 / 1.2494 |
| Variance ratio, red zone ÷ all | **1.1863** |
| Variance ratio, late down ÷ all | **1.9925** |
| League mean offset, red zone | +0.108 points per team-game |
| League mean offset, late down | **−0.477 points per team-game** |
| Plays per team-game | 62.2 |
| Red-zone plays per team-game | 9.20 |
| Late-down plays per team-game | 11.65 |
| Team-games with no red-zone play | **2.25%** |
| Team-games with no leverage play at all | 0.00% |
| Score dispersion, SD | **4.866 points** |
| … median \|score\| / q95 \|score\| / max \|score\| | 3.298 / 9.563 / 21.434 |
| Differential dispersion, SD | **6.862 points** |
| Three-cell identity, worst residual | 5.33 × 10⁻¹⁵ points |

**Why the score dispersion is measured here rather than in the results round.**
M-6's power check is *defined* by document 34 §6 as "at the meter's real
magnitude" and cannot run without it. It cannot move any threshold: M-2's, M-3's
and M-4's all come from simulated nulls. Disclosed in §9 rather than hidden.

---

## 4. Gate M-1 — the identities

No threshold is negotiable here; these are arithmetic.

| Check | Pass rule |
|---|---|
| Three cells sum to zero | \|red zone + late down + everything else\| ≤ 1 × 10⁻⁹ points, every team-game |
| Empty cell | a team-game with zero red-zone plays scores exactly 0.0 in that cell |
| Ledger reconciliation | for every game, the plays the meter re-prices are exactly the fumble rows of `dtw_ledger_v13.parquet` that survive the S0–S2 filter, and the summed repricing matches the ledger's own sum on those rows to 1 × 10⁻⁹ EPA |
| Differential | home score − away score, recomputed independently from the per-team scores, agrees to 1 × 10⁻⁹ points |

**M-1 failing stops the round.** Nothing below is readable.

---

## 5. The constraint ladder, and an arithmetic fact about two of its rungs

Document 34 §4's three rungs, least constrained first. The adoption rule in §6
reads this order.

1. **`raw`** — every play exchangeable across all three cells.
2. **`down_stratified`** — plays keep their down; only field-position
   membership moves.
3. **`down_stratified_var_matched`** — as 2, and the plays a draw assigns to the
   red-zone cell have their deviations from that down-stratum's mean stretched
   by `sqrt(var_rz / var_all)` = **1.0892**. The stretch touches the null only;
   the realized score is never rescaled, so M-1's identities are unaffected.

### The fact, derived before any rung was run

The score is the sum over the two leverage cells, so under any label
permutation only the leverage **union** matters, never the split between them:

```
score = points_per_epa * ( sum(epa over leverage plays) - k * mean_all ),
        k = n_red_zone + n_late_down
```

The leverage union is "red zone, any down" ∪ "late down, outside the red zone",
which **contains every third- and fourth-down play regardless of where the
red-zone labels land.** So a rung that holds each play's down fixed freezes the
entire late-down contribution as a constant, and the only quantity left random
is which *early-down* plays were red-zone plays.

**Rungs 2 and 3 are therefore a null for red-zone placement alone, not for the
meter as a whole.** Their bands are much narrower than rung 1's and centred near
the realized late-down contribution rather than near zero. This is derived, not
measured, and it is recorded here so that a narrow band from rung 2 is read as a
property of the rung and not as a finding about a game.

It also has a benign consequence, which the power table in §6 confirms: because
the late-down cell is the one whose second moment is badly non-exchangeable
(1.99× the league), freezing it is accidentally the right thing to do. Rung 1,
which shuffles it, is the rung the real structure breaks.

---

## 6. Gate M-2 — is the band honest?

### The statistic, and one place where document 34's language needed resolving

Document 34 §6 states M-2 as "PIT of realized score vs own-game null uniform
within a tolerance". PIT — probability integral transform — is the percentile of
the realized score inside its own game's permutation null, computed **mid-P**
because a permutation null is discrete and a one-sided rank would not be uniform
even under exact exchangeability.

Uniformity has to be reduced to a number, and the power calculation made the
choice for us rather than taste doing it:

- **Primary, with teeth: coverage of the 89% band.** The share of the 5,522
  team-games whose realized score lands inside its own reported band. It is the
  only number the band's copy actually claims, it is a binomial with a closed
  form, and its tolerance can be a materiality floor rather than a significance
  test.
- **Secondary, reported, no pass rule: the Kolmogorov–Smirnov distance of the
  PIT from uniform.** Measured, it turned out to be unusable as a gate: each
  rung's KS null under a *correct* truth has its own spread — 0.0198, 0.0305 and
  0.0399 at the 95th percentile — because a rung built from a narrower
  randomization produces a lumpier PIT in the middle even when its tails are
  exactly right. A common KS tolerance would fail rungs 2 and 3 for their
  construction; a per-rung tolerance would let a systematically wide rung grade
  itself (rung 3's own exchangeable-truth coverage is 91.90%, not 89%). Neither
  is a calibration test. KS is reported per rung against that rung's own
  exchangeable-truth null, as shape information only.

> **This is the one place where this document resolved language document 34 left
> open, rather than executing a settled decision.** The gate's shape — "is the
> band honest, judged against a tolerance whose false-alarm rate was measured
> first" — is unchanged, and both statistics are reported. It is flagged here so
> the maintainer can overrule it before `research/50_placement_meter.py` runs.

### The tolerance

> **The 89% band must cover between 87.0% and 91.0% of the 5,522 team-games.**

Two percentage points on the only number the copy claims. At a true coverage of
89% the binomial standard deviation is **0.421 pp**, so the tolerance is **4.7
standard deviations wide**: it is a materiality floor, and its false-alarm rate
is **0.000**, measured on the exchangeable-truth simulation and confirmed by the
closed form. A 95%-false-alarm tolerance would have been ±0.83 pp — a
significance gate at n = 5,522, which is the failure mode this project's
materiality floors exist to prevent.

### Power

Six simulated truths, each a synthetic league of 5,522 team-games built on the
**real** cell denominators and the **real** fat-tailed play distribution, with
only the cell-conditional moments manipulated. `rz_var_real` and `ld_var_real`
are the variance ratios of §3; `real_structure` carries both plus both mean
offsets, and is the truth production actually faces.

True coverage each rung's band achieves, and the probability a single league is
flagged:

| Truth | rung 1 `raw` | rung 2 `down_stratified` | rung 3 `…var_matched` |
|---|---|---|---|
| exchangeable *(false alarm)* | 88.73% → **0.000** | 88.98% → **0.000** | 91.90% → **0.993** |
| red-zone variance 1.10× | 88.69% → 0.000 | 88.10% → 0.006 | 91.10% → 0.602 |
| red-zone variance real (1.19×) | 88.36% → 0.001 | 87.24% → 0.296 | 90.40% → 0.064 |
| red-zone variance 1.30× | 88.35% → 0.001 | 86.43% → 0.892 | 89.68% → 0.001 |
| late-down variance real (1.99×) | 86.36% → **0.917** | 88.88% → 0.000 | 90.57% → 0.139 |
| **real structure** | 86.00% → **0.984** | 87.35% → 0.219 | 89.40% → 0.000 |

Read it as three separate findings, all available before any real data is
touched:

- **Rung 1 is correct under exchangeability and broken by the real structure.**
  Its band is ~3 pp too narrow when the late-down cell carries its real second
  moment, and the gate catches that 98% of the time.
- **Rung 3 is broken in the opposite direction under exchangeability** — 2.9 pp
  too *wide*, flagged 99% of the time — and lands at 89.40% under the real
  structure. It is a correction that is only correct when the thing it corrects
  for is present.
- **Rung 2 sits between them and is the least fragile**, passing under four of
  the six truths, but §5 is the price: it is not a null for the late-down half
  of the meter at all.

The simulated-league count is 40 for the exchangeable truth and 12 for each
graded truth, pooled to 221,000 and 66,000 PIT values respectively; the KS
resampler that reads those pools agrees with direct per-league simulation to
within 3% on all three rungs.

### The adoption rule, committed

> Run all three rungs and report all three. **Adopt the least-constrained rung
> whose coverage falls in [87.0%, 91.0%] on the real data**, ladder order as
> listed in §5. If no rung qualifies, **the meter ships without a band** and the
> score is displayed alone with the reason stated in copy.

Least-constrained rather than best-fitting, because every constraint narrows the
null toward the observed configuration and spends exchangeability licence; the
simplest calibrated null is the one that has claimed the least.

**One outcome forks to the maintainer rather than routing automatically.** If a rung
fails the coverage tolerance but its KS distance is within its own
exchangeable-truth null, or vice versa, the two readings disagree and that
disagreement is reported for a decision. Nothing about it is decided here.

---

## 7. Gates M-3 through M-6

### M-3 — is the shipped score still luck?

Document 34 §3 records why this gate exists: S2's luck verdict in document 08
was on the *success-rate* gap, and pricing in points forces the **EPA** gap on
late downs. The shipped score has to earn its own licence rather than borrow one.

**Statistic.** Document 08's split-half machinery, unchanged: 320 team-seasons,
8-game floor, 200 random within-season splits, mean correlation reported with
its 5th–95th percentile band. The half statistic is the **mean placement points
per game** over that half's games.

**Null.** Real team-games dealt at random into synthetic team-seasons of the
same sizes, 500 replicates. Team identity is destroyed; the real score
distribution, the real season lengths and the real split pattern all survive.

| | Value |
|---|---|
| Null mean | −0.0010 |
| Null SD | 0.0396 |
| **Gate M-3 threshold (null 95th pct)** | **r > 0.0671** |
| Null 99th pct | 0.0843 |

**Power**, 500 replicates per cell, simulated at the team-game level so the
200 correlated split draws are reproduced rather than assumed away:

| true r | 0.05 | 0.08 | 0.10 | **0.12** | 0.20 |
|---|---|---|---|---|---|
| `tau`, points per game | 0.380 | 0.488 | 0.552 | **0.612** | 0.828 |
| achieved mean r | +0.0517 | +0.0803 | +0.0981 | +0.1201 | +0.1974 |
| **power** | 0.368 | 0.636 | 0.776 | **0.900** | 1.000 |

> **Minimum detectable correlation at 80% power: r ≈ 0.105.** Power at the
> project's reference effect of r = 0.12 is **0.900**, so Gate S-3's condition
> is met and **M-3 is pass/fail rather than descriptive.**

**Verdicts.** `r > 0.0671` → placement is a team property, the word "luck" is
not licensed for this score, and the fork goes to the maintainer (document 34 §6).
`r ≤ 0.0671` → luck, at 90% power against an effect the size of the smallest
this project has ever called real. A one-SD team differs from average by 0.612
placement points per game at that reference — about 10 points across a season.

### M-4 — does the baseline leak skill?

**Statistic.** Pearson correlation across 320 team-seasons between the team-
season mean placement score and S0, its luck-priced offensive EPA per play.

**Null.** Placement scores redrawn independent of quality at each team-season's
own sampling SD, correlated with the **real** quality vector — so the null
carries the real quality distribution and only the relationship is removed.
4,000 replicates.

| | Value |
|---|---|
| Null median \|r\| | 0.0373 |
| **Gate M-4 bound (null 95th pct of \|r\|)** | **\|r\| ≤ 0.1065** |
| Null 99th pct | 0.1378 |

**Power to flag a true leak:**

| true leak | 0.05 | 0.10 | **0.11** | 0.15 | 0.20 | 0.30 |
|---|---|---|---|---|---|---|
| power | 0.148 | 0.428 | **0.483** | 0.763 | 0.945 | 1.000 |

> **M-4 is pass/fail, with its ceiling stated in the same breath.** At document
> 05 §7's ±0.11 detectability floor the design sees a leak only **48%** of the
> time; 80% power arrives at a leak of about **0.16**. A pass therefore licenses
> "no leak larger than roughly 0.16", never "no leak", and every report of this
> gate carries that sentence. This is document 05 §7's return-yardage lesson
> applied in advance rather than after the fact.

**Secondary, descriptive, no rule.** The dispersion read document 34 §7 asks
for: correlation between a team-season's *spread* of placement scores and its
quality, because good offenses take more red-zone snaps and `n_cell` scales the
score.

**Failure routing.** `|r| > 0.1065` → the baseline leaks, the design is broken,
back to design per document 34 §6. Not patched at the output.

### M-5 — magnitude

**Report, no threshold** (Gate C convention). Median and q95 |score|, share of
games outside their own band, correlation with DTW%, overlap with the 195
clear-flip games, per-season stability, and the per-cell decomposition. The
example games are document 33 §6's: `2018_05_GB_DET`, `2021_14_LV_KC`,
`2025_17_DET_MIN`.

### M-6 — does subtracting the meter lose information?

**Statistic.** Document 06's harness, untouched: 531 rematch pairs, mean
out-of-fold log loss over 10 folds at `random_seed = 20260817`, non-inferiority
margin **+0.010**. Incumbent is game 1's actual margin; challenger is game 1's
deserved margin minus the placement differential.

Combining the two meters is banned in the product and legitimate inside a
validation instrument — it is how DTW% itself was validated. That asymmetry is
document 34 §6's and is not reopened.

**Power**, 400 replicates per arm, at the meter's real differential SD of 6.862
points:

| Arm | What is true | Mean Δ log loss | Mean 95% upper | Pass rate |
|---|---|---|---|---|
| **Power** — placement is exchangeable noise, so subtracting it removes noise | challenger genuinely no worse | −0.00811 | +0.00257 | **0.973** |
| **False pass** — the subtraction is pure harm of the same magnitude | challenger genuinely worse | +0.00137 | +0.01149 | **0.318** |

> **M-6 is pass/fail: power to pass is 0.973, comfortably above 0.80.** But the
> gate's teeth are blunt and the number is committed here so a pass is read
> correctly: when the subtraction is pure harm at this magnitude, the gate still
> passes **32%** of the time. A pass is evidence, not proof; a **failure** is the
> strong signal, because a gate this permissive only fails when something real is
> wrong.

**Failure routing.** M-6 failing while M-3 and M-4 pass is a contradiction —
placement would contain skill that two better-powered gates missed — and per
document 34 §6 **the contradiction is the finding**, reported as such.

---

## 8. Order of operations, and failure routing

Committed before any of it runs, from document 34 §6 and §10.

```
M-1  ->  M-2  ->  M-3  ->  M-4  ->  M-5  ->  M-6
```

Each gate is reported to the maintainer before the next one starts. No gate is run twice
with a different setting; a re-run for a defect goes through amendment C-1, the
mechanism document 30 established.

| Failure | Routing |
|---|---|
| M-1 | Stop. Nothing is readable, nothing ships |
| M-2, no rung qualifies | The meter ships without a band, reason in copy |
| M-2, the two readings disagree | Reported to the maintainer for a decision. Not resolved here |
| M-3 | The word "luck" is not licensed for the shipped score. Fork to the maintainer |
| M-4 | The baseline is broken. Back to design |
| M-6 | Contradiction with M-3/M-4; the contradiction is the finding |

---

## 9. Known-defect register

| Defect | Evidence | Status |
|---|---|---|
| **The real M-3 split-half correlation was printed during a smoke test of the power machinery, before this document was committed.** A single value at a preliminary seed, `r = 0.050`, below the 0.0671 threshold committed in §7 | timing check of `split_half_r` while sizing the run | **Open, disclosed.** The threshold *rule* — the permutation null's 95th percentile — is fixed by process law inherited from document 05 §7 and is computed by simulation, so knowing the value could not move it, and the committed threshold is one the peeked value fails. Recording it is the only defence available; hiding it would be worse. Document 08 §7 is the precedent |
| The late-down cell prices an EPA gap where document 08's S2 verdict was on success rate | document 34 §3 | **Open, by design.** M-3 exists to re-license the shipped score itself |
| Rungs 2 and 3 are not a null for the late-down half of the meter | §5, derived | **Open, disclosed before any run.** Reported wherever a rung-2 or rung-3 band is displayed |
| The dominant exchangeability failure is the late-down cell (1.99×), which no rung's variance matching addresses | §3, §6 | **Open.** Document 34's ladder was built around the red zone. Rungs 2 and 3 handle it only by freezing it |
| M-2's power ran at 300 permutation draws; production uses 2,000 | §6 vs §10 | **Accepted.** Finer discreteness moves the mid-P PIT closer to uniform, so the measured false-alarm rate is an over-statement and the tolerance is conservative |
| M-2's coverage power uses a binomial closed form, treating the two team-games in one game as independent | §6 | **Accepted.** They are built from disjoint play sets. The simulated leagues carry the exact dependence and their coverage agrees |
| M-3's power simulation adds `tau` on top of the realized dispersion rather than inside it | `m3_power` | **Accepted, with the arithmetic.** At r = 0.12 total dispersion is inflated 0.8%. The achieved mean r is printed beside the target and lands at 0.1201 against 0.12 |
| M-4's "leak" is a latent coefficient; heteroskedastic team-season SEs attenuate the achieved correlation slightly | `m4_power` | **Accepted.** The direction under-states power, so the stated ceiling is conservative |
| M-6's power arm adds a synthetic contamination layer on top of the real placement luck already inside both margins | `m6_power` | **Accepted, and the reason the false-pass arm is reported beside it.** Neither arm alone is a verdict |
| Cell counts are endogenous to quality | document 34 §11 | **Open.** M-4's secondary read watches it |
| Special-teams placement is outside the meter | document 34 §5 | **Accepted, by design.** Stated wherever the meter is documented |
| No gated statistic in this document has run against data | whole document | **By construction.** That is what a pre-registration is |

---

## 10. Constants appendix

| Constant | Value | Where it lives |
|---|---|---|
| `RANDOM_SEED` | 20260819 | `research/49_placement_power.py`, `research/50_placement_meter.py` |
| `POINTS_PER_EPA` | 0.8389 | both scripts, from document 27 §13 |
| `RED_ZONE_YARDS` | 20 | both scripts |
| `LATE_DOWNS` | (3, 4) | both scripts |
| `N_BAND_DRAWS` (production) | 2,000 | `research/50_placement_meter.py` |
| `N_BAND_DRAWS_POWER` | 300 | `research/49_placement_power.py` |
| Band interval | 5.5 – 94.5 (89% equal-tailed) | both scripts |
| `N_SPLITS` / `MIN_GAMES` | 200 / 8 | document 08's, inherited |
| M-3 null / power replicates | 500 / 500 | `research/49_placement_power.py` |
| `REFERENCE_R` | 0.12 | document 02, via document 08 |
| Ladder | raw, down_stratified, down_stratified_var_matched | §5 |
| Rung-3 stretch factor | 1.0892 | §5, = sqrt(1.1863) |
| **M-1 identity tolerance** | **1 × 10⁻⁹ points** | §4 |
| **M-2 coverage tolerance** | **[87.0%, 91.0%]** | §6 |
| M-2 KS reference nulls (secondary) | 0.0198 / 0.0305 / 0.0399 | §6, per rung |
| **M-3 threshold** | **r > 0.0671** | §7 |
| **M-4 bound** | **\|r\| ≤ 0.1065** | §7 |
| **M-6 margin** | **+0.010 log loss, 95% upper** | §7, document 06's |
| M-6 power / false-pass rate | 0.973 / 0.318 | §7 |

Results are written back into this document as §11 by
`research/50_placement_meter.py`, which does not yet exist.
