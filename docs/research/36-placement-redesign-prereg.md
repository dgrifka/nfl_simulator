# 36 — The placement meter redesigned, pre-registered

*Written 2026-08-20, **before any gated statistic on this round has been
computed**. Power calculation: `research/51_placement_redesign_power.py`,
results in `research/outputs/51_redesign_power.json`. Committed to git before
`research/52_placement_redesign.py` exists, so goalpost integrity is checkable
by commit archaeology.*

*This document supersedes document 35's §§5–7 for the redesigned score. Document
34 remains the settled design; its §2 decision table is the maintainer's and is not
reopened. Document 35's §§1–4 input stream, cells, points scale and identities
are carried forward verbatim, and §11 is the record of why this round exists:
**gate M-4 failed at +0.5435 against a bound of 0.1065**, and document 34 §6
routes that back to design.*

---

## 0. Preamble — decisions carried in, with dates

Three decisions arrived settled and are recorded here rather than re-argued.

1. **Ladder rung 4 (`raw_var_matched`) is accepted on coverage** — the maintainer,
   2026-08-20. Coverage is the primary powered reading; the KS distance stays
   shape-only information per document 35 §6. This closes the one disagreement
   document 35 §11 handed back. §6 below shows the acceptance survives the
   redesign untouched, and why.
2. **Redesign avenue (a) is approved** — the maintainer, 2026-08-20. Baseline the count
   channel, not only the cell means. The precise construction was left to this
   document; §§2 and 5 propose it, and §5 is where the proposal departs from the
   literal wording of the avenue and says so.
3. **Fallback (b) is pre-approved but gated** — the maintainer, 2026-08-20. If this
   pre-registration cannot be powered at the usual standard, or the redesign
   fails its gates, the fallback (ship the red-zone half alone, drop the
   late-down half) is **not** executed silently: the round stops and reports.
   Avenues (c) and (d) stay parked.

> **The trigger in decision 3 has fired, and this document is where it fires.**
> §7 measures gate M-4's power on the redesigned score at **0.00–0.11 against
> every leak shape tried**, including a premium of 0.20 EPA per play confined to
> the top quality quartile. M-4 cannot be a powered gate on this construction,
> and §8 stops the round here rather than routing onward. Nothing gated has been
> computed. The fork is stated in §8 and belongs to the maintainer.

> **Resolved 2026-08-20: the maintainer chose (i)** — the full meter, M-4 demoted to
> descriptive, rung 3 left recorded rather than fixed. The stop did its job; the
> round then ran §8's order and reported after each gate. Results in §11. This
> block is left as written so the record shows the stop happened *before* the
> decision, not after it.

---

## 1. One-page story

The incumbent scored a leverage cell against the team's **own game-wide mean**.
That bar carries the league's structural profile — late downs run 0.048 EPA per
play below the league, worth −0.477 points per team-game — and the score is
`n_cell`-scaled, so every team-game's placement number was dragged down in
proportion to how often that team was in a situation its own quality had put it
in. Bad offences face far more third downs. The leak measured +0.5435.

The redesign centres each cell on **what a team of this quality produces there**,
fitted league-wide and leave-one-team-out, before the count multiplies anything.
The three-cell identity, the points scale, the luck-priced input stream, the
two-meters contract and the permutation band are all unchanged.

Five things this round established before it fitted anything gated.

1. **Baselining the mix cannot remove the leak; centring the profile can.** The
   leak is a product of two factors — a structural cell offset and a
   quality-correlated cell count. Replacing the realised count with the count
   quality predicts leaves the second factor exactly as quality-correlated as it
   was. Only the first factor can be zeroed, and centring zeroes it exactly.
   Derived in §5, which is why the committed construction centres the profile and
   the expected-mix reweight is a reported arm rather than the shipped score.
2. **The redesign does not touch the band.** With the three cell sizes held fixed
   — which every rung of the ladder does — the profile a draw subtracts is the
   same whichever plays land where. So the redesigned score and *every* null draw
   move by the same per-team-game constant, the PIT is a rank and ranks are
   shift-invariant, and M-2's coverage, its rung adoption and its power table
   carry forward exactly. Derived in §6 and checked to 1 × 10⁻¹⁴ points on all
   four rungs.
3. **The meter survives the correction at full size.** Score dispersion moves
   from 4.866 to **4.856** points and the differential SD from 6.862 to
   **6.851**. What the correction removes is the *level*: the league mean
   placement score goes from −0.4545 to **−0.0719** points per team-game. The
   fitted adjustment has an SD of 0.451 points against a score SD of 4.856 — the
   meter is 9% fitted correction and 91% realised placement.
4. **M-4's covariate shares its plays with the score, and that alone is worth
   +0.168.** The leverage cells carry more play-level variance than the game as a
   whole, so a team-game that scored well got there disproportionately through
   leverage plays. On 600 simulated leagues with *no leak at all*, the redesigned
   score correlates **+0.1681 ± 0.0112** with same-season quality. Document 35's
   M-4 null assumed the two shared nothing. §7 derives the coupling and replaces
   the reference.
5. **The correction removes any leak that is a function of quality, so M-4 has
   nothing left to detect.** Flag rates against a linear premium, a
   top-quartile step premium and a step of 0.20 EPA per play all sit at the
   false-alarm rate. This is not a defect of the simulation; it is what the
   construction is *for*, and it is why M-4 cannot carry teeth here.

| Gate | Question | Verdict on its own power |
|---|---|---|
| **M-1** identities | Do the books balance? | Identity check, no power needed. Two checks added (§4) |
| **M-2** calibration | Is the band honest? | **Carried forward by proof, not re-run** (§6). Rung 4 adopted, coverage 87.45% |
| **M-3** luck licence | Is the shipped score still luck? | **Teeth.** Power **0.916** at r = 0.12; cross-checked through the whole pipeline |
| **M-4** skill preservation | Does the baseline leak skill? | **Descriptive, with power measured at 0.00–0.11.** Gate S-3 pattern. §7, and the §8 stop |
| **M-5** magnitude | How often does it matter? | Report, no threshold (Gate C convention) |
| **M-6** rematch | Does subtracting it lose information? | **Teeth.** Power to pass **0.965**, false-pass **0.26** |

---

## 2. The construction, exactly

`research/52_placement_redesign.py` reproduces this or it is wrong.

**Everything document 35 §2 fixes is unchanged**: the 2,761 adjudicated games,
document 08's S0–S2 filter, 343,543 plays over 5,522 team-games, the luck-priced
valuation `epa − luck_epa` re-signed to the possessing team, the three disjoint
cells in their order of application, and `points_per_epa = 0.8389`.

**One new input: the team's quality, leave-one-game-out.**

```
s0_loo(t, g) = ( sum of epa_priced over t's other games this season )
             / ( count of those plays )
```

The game being scored never enters its own baseline. This is document 05 §5's
contamination defence applied at the input rather than bounded after the fact.

**The expected profile.** For each cell `c`, one two-parameter weighted least
squares fit across the 5,522 team-games,

```
mean_epa_in_cell_c  ~  a_c + b_c * s0_loo ,   weight = that team-game's n_c
```

estimated **leave-one-team-out**: the coefficients used for a franchise are fitted
without a single play that franchise ran, in any season. Write `mu_c` for the
fitted value.

Two things ride on the details, and both are deliberate.

* **The weight is the count the score multiplies that cell by.** That makes the
  fit's own orthogonality condition *be* the leak condition rather than merely
  resemble it: the normal equations set `sum over team-games of n_c * residual_c *
  s0_loo` to zero, and `n_c * residual_c` is exactly what the score sums.
* **The fold is the franchise, not the game.** A fit that included the team would
  make M-4 read its own arithmetic. Leave-one-team-out does not remove that
  entirely — §7 measures what is left — but it is the strongest fold available
  without spending the sample.

**The score.** For one team-game, per cell,

```
cell_points = ( sum(epa_priced in cell) - n_cell * mu_cell - n_cell * baseline ) * points_per_epa
baseline    = sum over cells of ( sum(epa_priced in cell) - n_cell * mu_cell ) / n_all
```

The **placement meter is the sum of the red-zone and late-down cells**; the third
cell is the negative of that sum. The game's headline is the **differential**,
home minus away. Written as a sum minus a count times a mean, exactly as
document 35 §2 was, so an empty cell is exactly `0.0` rather than `0/0` and the
three cells sum to zero by arithmetic rather than by decree.

**It reduces to the incumbent.** Set every `mu_c` to zero and this is document
35's score, character for character. The redesign is a strict generalisation of
what it replaces, which is the property rung 4 earned in document 35 §5 and the
same defence applies.

**The band is unchanged** — same ladder, same rung 4, same 2,000 draws, same 89%
equal-tailed interval. §6 proves it may be.

---

## 3. Design parameters, measured before any threshold was set

League-pooled; no team identity survives any of them. Document 08 §2's category,
and the disclosure rules there apply here (§9).

| Parameter | Value |
|---|---|
| Games / team-games / team-seasons | 2,761 / 5,522 / 320 |
| Scrimmage plays / plays carrying a ledger row | 343,543 / 5,541 |
| **Fitted profile — red zone** | intercept **+0.00757** EPA per play, slope on quality **0.764** |
| **Fitted profile — late down** | intercept **−0.04233**, slope **1.201** |
| **Fitted profile — everything else** | intercept **+0.01127**, slope **0.606** |
| Play-count-weighted mean slope | **0.7408** |
| Fitted profile SD, by cell | 0.0683 / 0.1073 / 0.0542 EPA per play |
| Profile shift `C`, mean / SD / range | **−0.383** / **0.451** / −2.141 … +0.917 points |
| Score mean / SD | **−0.0719** / **4.856** points *(incumbent: −0.4545 / 4.866)* |
| … median \|score\| / q95 / max | 3.263 / 9.591 / 21.560 *(incumbent: 3.298 / 9.563 / 21.434)* |
| Red-zone / late-down cell SD | 3.395 / 4.580 points |
| Differential SD | **6.851** points *(incumbent: 6.862)* |
| Three-cell identity, worst residual | 6.2 × 10⁻¹⁵ points |
| Team-games with no red-zone play | 124 (2.25%), every one scoring exactly 0.0 |

**The play-count-weighted mean slope is 0.7408, and it is a reliability, not a
finding.** Regressing a single game's EPA per play on the rest of that season's
gives the reliability of a 16-game quality estimate. It is quoted because it
bounds how much of the *interaction* channel the fit can reach: the fitted slopes
are attenuated by roughly a quarter relative to true quality. §9 records the
direction.

**Why dispersion is measured here rather than in the results round.** M-6's power
is *defined* by document 34 §6 as "at the meter's real magnitude" and cannot run
without it. It cannot move a threshold: M-3's and M-6's come from simulated
nulls, M-2's is carried forward, and M-4 has no threshold with teeth. Disclosed
in §9 rather than hidden. This is document 35 §3's precedent, followed.

---

## 4. Gate M-1 — the identities

Arithmetic, not negotiable. Document 35 §4's four checks are carried forward
**unchanged in meaning**, and two are added for what the redesign introduces.

| Check | Pass rule |
|---|---|
| Three cells sum to zero | \|red zone + late down + everything else\| ≤ 1 × 10⁻⁹ points, every team-game |
| Empty cell | a team-game with zero red-zone plays scores exactly 0.0 in that cell |
| Ledger reconciliation | the plays the meter re-prices are exactly the fumble rows of `dtw_ledger_v13.parquet` surviving the S0–S2 filter, summed repricing matching to 1 × 10⁻⁹ EPA |
| Differential | home minus away, recomputed independently, agrees to 1 × 10⁻⁹ points |
| **Reduction** *(new)* | with every `mu_c` set to zero the module reproduces document 35 §11's scores to 1 × 10⁻⁹ points, team-game by team-game |
| **No self-baselining** *(new)* | for a sample of ≥ 40 team-games, refitting with that team-game's **whole game** dropped from the league sample moves the score by ≤ 0.05 points |

The contamination check is already measured at **0.0035 points worst case** over
40 team-games (§3's run), against a 0.05 tolerance — one game is 1.8 × 10⁻⁴ of
the fit's rows, and the franchise is excluded outright, so the only surviving
path is the opponent's rows. The tolerance is set an order of magnitude above the
measured value so it tests the implementation rather than re-reporting it.

**M-1 failing stops the round.** Nothing below is readable.

---

## 5. The count channel: what can be baselined and what cannot

Redesign avenue (a) says: *score the leverage cells against the play mix the
team's quality predicts, not the mix it realised.* This section derives why the
committed construction centres the **profile** instead, and reports the literal
version as an arm.

### The arithmetic, before anything was run

Write `p_c` for the realised share of a team's plays in cell `c`, `q_c` for the
share its quality predicts, and `d_c` for the structural gap between that cell's
EPA per play and the team's own overall rate. Document 35 §11 found the leak is

```
score  ~  n * sum over leverage cells of  p_c * d_c
```

— a **product of two factors**, both quality-correlated. The count share `p_c`
correlates −0.787 with quality on late downs and +0.718 in the red zone; the
structural gap `d_c` is −0.048 EPA per play on late downs.

Replacing `p_c` by `q_c` replaces a quality-correlated factor with **a
deterministic function of quality**. The realised share is what quality predicts
plus noise; the expected share is what quality predicts. Baselining the mix
removes the noise and keeps the signal, which is the wrong half:

> **A leak that is the product of a structural offset and a quality-driven count
> cannot be removed by baselining the count. It can only be removed by zeroing
> the offset.**

Centring each cell on `mu_c` zeroes `d_c` by construction, whatever the counts
do, and the conditional expectation of every cell's contribution given quality is
zero as a consequence rather than as a hope.

> **This is the one place where this document departs from the literal wording of
> a decision the maintainer made, rather than executing it.** The avenue's *purpose* —
> baseline the count channel so M-4 can pass — is served, and served exactly; its
> *mechanism* — the predicted mix as a multiplier — is shown above not to serve
> it. Both constructions are run and both are reported. It is flagged here so
> the maintainer can overrule before `research/52_placement_redesign.py` runs.

### The expected-mix arm, and the arithmetic that disqualifies it as the score

The literal construction — reweight every play by `q_c / p_c`, so a cell is
priced at the count quality predicts — was built and measured. It keeps the
identity (8.9 × 10⁻¹⁵ points) and the empty-cell zero. What it does not keep is
proportion:

| | committed | expected-mix arm |
|---|---|---|
| Score SD | 4.856 | 5.332 |
| **Max \|score\|** | **21.56** | **34.30** |
| Red-zone reweight, p90 / p99 / max | — | **2.33 / 7.97 / 11.45** |

The reweight is unbounded as a cell empties. A team-game with one red-zone snap
has that snap priced as roughly nine, and the meter tells the reader that
placement was worth thirty points on the strength of a single play. That is
manufacturing information, and it is disqualifying for a number a product prints.
The arm is reported at every gate; it is not the shipped score.

### What the profile fit does and does not reach

The fitted slopes carry the *interaction* channel — that a good offence is
relatively better on late downs than the league profile predicts, not merely
better everywhere. The slopes differ sharply by cell (1.201 late down against
0.606 elsewhere), so the channel is real and large. The fit reaches it, but
attenuated by the reliability of a 16-game quality estimate, **0.7408** (§3).
Since M-4's covariate is *measured* quality and the fit's predictor is *measured*
quality, the attenuation does not itself leave a residual against the gate — it
leaves a residual against **true** quality, which nothing in this design
observes. §9 records it.

---

## 6. Gate M-2 — carried forward by proof, not re-run

### The derivation

Under any rung of the ladder the three cell sizes are held fixed. So for one draw
assigning `n_rz` plays to the red zone and `n_ld` to late downs,

```
sum over leverage of mu(assigned cell)  =  n_rz * mu_rz + n_ld * mu_ld
```

which does not depend on **which** plays were assigned — only on how many. The
baseline's profile term, `sum over cells of n_c * mu_c / n_all`, is fixed for the
same reason. Therefore

```
redesigned_score  =  incumbent_score - C ,
C  =  ( n_rz*mu_rz + n_ld*mu_ld  -  k * sum_c (n_c/n) * mu_c ) * points_per_epa
```

for **the realised score and every null draw alike**, with the same `C`. The PIT
is a rank of the realised score inside its own draws; a rank is invariant to a
common shift. So the PIT does not move, and with it neither does coverage, the
rung adoption, the KS reading, nor any number in document 35 §6 or §11.

**Rung 3's variance stretch does not break this**, because the stretch is applied
to deviations and the re-centring is applied after it; nor does rung 4's, for the
same reason.

### The check

Derived, then checked rather than asserted — 40 team-games, 2,000 draws each, all
four rungs, with both scores recomputed the slow way from a **shared** assignment
so the two sides are computed independently of each other and of the production
module.

| Rung | Worst \|(incumbent draw − C) − redesigned draw\| | Team-games whose PIT moved |
|---|---|---|
| 1 `raw` | 7.1 × 10⁻¹⁵ points | 0 of 40 |
| 4 `raw_var_matched` | 1.1 × 10⁻¹⁴ | 0 of 40 |
| 2 `down_stratified` | 7.1 × 10⁻¹⁵ | **3 of 40, worst gap 0.017** |
| 3 `down_stratified_var_matched` | 1.1 × 10⁻¹⁴ | 0 of 40 |

Rung 2's three movements are tie-breaking at machine precision: its null is the
most discrete of the four, exact ties between the realised score and a draw are
common, and a 10⁻¹⁵ difference turns a tie into a strict inequality inside the
mid-P. The band bounds are quantiles and shift with everything else, so coverage
is unaffected; the disclosure is in §9.

### What M-2 therefore says, unchanged

> **Rung 4 `raw_var_matched` is the adopted null, at 87.45% coverage of an 89%
> band** against a tolerance of [87.0%, 91.0%] — a pass by 0.45 pp against a
> binomial SD of 0.421 pp. Read as "not detectably miscalibrated at this
> tolerance", never as "exactly calibrated". Document 35 §6's adoption rule,
> §5's ladder order and §6's power table all stand; the maintainer's 2026-08-20 acceptance
> on coverage closes the KS disagreement.

`research/52_placement_redesign.py` re-runs all four rungs on the redesigned
score anyway and asserts each coverage reproduces document 35 §11's to within
0.1 pp. A carry-forward that is not checked in the ship is an assumption.

---

## 7. Gates M-3 through M-6

### M-3 — is the shipped score still luck?

**Statistic and null unchanged from document 35 §7**: document 08's split-half
machinery, 320 team-seasons, 8-game floor, 200 within-season splits; the null
deals real team-games at random into synthetic team-seasons of the same sizes,
500 replicates. Re-run because the score changed.

| | Value | *(document 35)* |
|---|---|---|
| Null mean / SD | −0.0002 / 0.0408 | −0.0010 / 0.0396 |
| **Gate M-3 threshold (null 95th pct)** | **r > 0.0636** | *0.0671* |
| Null 99th pct | 0.0879 | 0.0843 |

**Power**, 500 replicates per cell, simulated at the team-game level:

| true r | 0.05 | 0.08 | 0.10 | **0.12** | 0.20 |
|---|---|---|---|---|---|
| `tau`, points per game | 0.379 | 0.487 | 0.551 | **0.610** | 0.827 |
| achieved mean r | +0.0510 | +0.0795 | +0.0982 | +0.1176 | +0.1986 |
| **power** | 0.404 | 0.636 | 0.794 | **0.916** | 0.998 |

> **Minimum detectable correlation at 80% power: r ≈ 0.101.** Power at the
> project's reference effect of r = 0.12 is **0.916**, so Gate S-3's condition is
> met and **M-3 is pass/fail rather than descriptive.**

**Cross-checked through the whole construction, which is new.** The power above
injects a team offset on top of realised scores, so it cannot see whether the
profile fit would have absorbed the offset before the split-half ever read it.
Six hundred simulated leagues run through *fit, score, split-half* give a no-leak
split-half of **+0.0038 ± 0.0390** — agreeing with the permutation null's
−0.0002 ± 0.0408 — and a persistent, **quality-orthogonal** placement premium
raises it as it should:

| Premium, EPA per play on late downs | 0.02 | 0.05 | 0.10 |
|---|---|---|---|
| Split-half r | +0.0146 | +0.0399 | **+0.1192** |

The correction is targeted, not indiscriminate: a placement tendency that has
nothing to do with quality passes through it intact and M-3 sees it.

**Verdicts.** `r > 0.0636` → placement is a team property, the word "luck" is not
licensed for this score, and the fork goes to the maintainer. `r ≤ 0.0636` → luck, at 92%
power against an effect the size of the smallest this project has ever called
real.

### M-4 — does the baseline leak skill? **Descriptive, and here is the ceiling**

Two findings, both measured before any real correlation was computed.

**Finding one: the covariate shares its plays with the score, and that alone is
worth +0.168.** The score is a within-game contrast, `sum over leverage of
n_c * residual_c − (k/n) * sum over all cells of n_c * residual_c`. Its covariance
with the team-game's own EPA total is

```
sum over leverage of n_c * var_c   -   (k/n) * sum over all cells of n_c * var_c
```

and at document 35 §3's measured cell variances and mean counts that is
`62.84 − 0.335 × 114.50 = +24.5 EPA²` — **positive**, because the leverage cells
carry more play-level variance (2.18 and 3.67) than the game as a whole (1.84). A
team-game that scored well got there disproportionately through leverage plays,
so its placement score is positive; season quality is built from those same
games. Document 35 §7's null redrew placement independently of quality and
therefore carried none of this.

Measured on 600 simulated leagues with **no leak at all**:

| Score | vs same-season quality | vs other-seasons quality |
|---|---|---|
| Redesigned | **+0.1681 ± 0.0112** | **−0.0025 ± 0.0558** |
| Incumbent | +0.1868 ± 0.0544 | +0.0057 ± 0.0573 |

**Finding two: the construction removes any leak that is a function of quality,
so the gate has nothing left to detect.** Flag rates, against the no-leak
reference's own 95th percentile of \|r\| (0.1857 same-season, 0.1073
other-seasons):

| Injected leak | flag rate, same season | flag rate, other seasons |
|---|---|---|
| Late-down premium linear in quality, 0.05 EPA per play | 0.013 | 0.077 |
| … 0.10 | 0.010 | 0.107 |
| Premium given only to the top quality quartile, 0.05 | 0.033 | 0.043 |
| … 0.10 | 0.007 | 0.060 |
| … **0.20** | **0.000** | **0.060** |

A monotone step in quality is well approximated by a line, so the linear fit
absorbs almost all of it; only the nonlinear remainder survives, and it is small.
Every rate above sits at or barely over the 5% false alarm the bound is defined
to produce.

> **M-4 is pre-registered as descriptive, not pass/fail** (Gate S-3 pattern, the
> mechanism document 35 §7 used for the same gate at a milder ceiling). It
> verifies that the correction was implemented; it licenses nothing about leaks.
> **Its power against a quality-aligned leak of any magnitude tried is at the
> false-alarm rate.**

**Reported anyway, with both references.** The same-season correlation against
the pipeline reference `+0.1681 ± 0.0112`, and the other-seasons correlation
against its bound of **0.1073**. Document 35 §7's independent nulls are reported
beside them (bounds 0.1077 and 0.1104) so the change of reference is visible
rather than silent.

**And the change of covariate is not a rescue.** Read against other-seasons
quality on the real data, the **incumbent** score — the design this round
replaces — correlates **+0.2098**, against a bound of 0.1073. It fails the clean
covariate too. A covariate that still fails the design it was introduced after
is not a goalpost that was moved.

**Secondary, descriptive, no rule.** Document 34 §7's dispersion read:
correlation between a team-season's *spread* of placement scores and its quality.

### M-5 — magnitude

**Report, no threshold** (Gate C convention). Median and q95 \|score\|, share of
games outside their own band, correlation with DTW%, overlap with the 195
clear-flip games, per-season stability, the per-cell decomposition, and — new —
the distribution of the profile shift `C`, so a reader can see how much of the
meter is fitted correction. Example games are document 33 §6's:
`2018_05_GB_DET`, `2021_14_LV_KC`, `2025_17_DET_MIN`.

### M-6 — does subtracting the meter lose information?

**Statistic unchanged from document 35 §7**: document 06's harness, 531 rematch
pairs, mean out-of-fold log loss over 10 folds at `random_seed = 20260817`,
non-inferiority margin **+0.010**, incumbent is game 1's actual margin,
challenger is game 1's deserved margin minus the placement differential.

**Power**, 400 replicates per arm, at the redesigned differential SD of **6.851**
points:

| Arm | What is true | Mean Δ log loss | Mean 95% upper | Pass rate |
|---|---|---|---|---|
| **Power** — placement is exchangeable noise | challenger genuinely no worse | −0.00768 | +0.00294 | **0.965** |
| **False pass** — the subtraction is pure harm | challenger genuinely worse | +0.00211 | +0.01214 | **0.260** |

> **M-6 is pass/fail: power to pass is 0.965.** Its teeth are blunt and the number
> is committed here so a pass is read correctly — when the subtraction is pure
> harm at this magnitude the gate still passes **26%** of the time. A pass is
> evidence, not proof; a **failure** is the strong signal.

---

## 8. Order of operations, failure routing, and where this round stops

```
M-1  ->  M-2 (carry-forward assertion)  ->  M-3  ->  M-4  ->  M-5  ->  M-6
```

Each gate is reported to the maintainer before the next one starts. No gate is run twice
with a different setting; a re-run for a defect goes through amendment C-1, the
mechanism document 30 established.

| Failure | Routing |
|---|---|
| M-1 | Stop. Nothing is readable, nothing ships |
| M-2, a coverage fails to reproduce within 0.1 pp | The carry-forward proof is wrong; stop and re-run M-2 in full |
| M-3 | The word "luck" is not licensed for the shipped score. Fork to the maintainer |
| M-4 | No routing — it is descriptive. Reported with its ceiling in the same breath |
| M-6 | Contradiction with M-3; the contradiction is the finding |

### The stop

**§0's decision-3 trigger has fired.** The requirement the maintainer set for avenue (a)
was document 34 §6's M-4, **re-powered**. It cannot be re-powered: §7 measures its
power at the false-alarm rate against every leak shape tried. The round therefore
stops at this document and reports, exactly as decision 3 requires, rather than
running gates or executing the fallback.

The fork, stated so it can be decided rather than argued:

* **(i) Accept the redesign with M-4 demoted.** The meter ships gated on M-1,
  M-2 (carried forward), M-3 (power 0.916) and M-6 (power 0.965), with M-4
  reported descriptively and the ceiling printed beside it. The honest sentence
  the product would carry: *the placement meter is constructed so that a team's
  quality cannot appear in it, and that construction is not independently
  testable.* What is irreducibly lost is the ability to distinguish "the count
  artifact" from "good offences genuinely place better" — document 08's
  persistence verdict is the only evidence that the second is not real, and M-3
  keeps re-checking it.
* **(ii) Fallback (b) — ship the red-zone half alone.** The one option in which
  **M-4 stays a powered gate**: the red-zone cell needs no correction at all, so
  nothing is orthogonalised against quality and the gate retains document 35 §7's
  power (0.48 at a leak of 0.11, 0.80 at ≈ 0.16). Its incumbent correlation was
  +0.0670, inside the bound. The price is that the meter covers one of the two
  channels document 08 ruled luck, and document 35 §3's larger channel — the
  late-down cell, at 1.99× league variance — is dropped.
* **(iii) Something else.** Design C is parked (document 34 §9); a
  quality-orthogonal reformulation of the late-down cell has not been designed.

Nothing here is decided. **The recommendation, stated as one:** (i), because the
correction is derived rather than fitted-to-fit, the meter survives it at full
size, and M-3 — the gate that carries the actual licence for the word "luck" — is
better powered than it was. But (ii) is the only option where the skill defence
is *tested* rather than *constructed*, and that is a real difference which is
the maintainer's to weigh.

---

## 9. Known-defect register

| Defect | Evidence | Status |
|---|---|---|
| **M-4 has no power against a quality-aligned leak on this construction** | §7, flag rates 0.000–0.107 | **Open, disclosed, and the reason §8 stops the round.** The gate is pre-registered descriptive with the ceiling in the same breath |
| **The score and M-4's same-season covariate are built from the same plays**, worth +0.168 before any leak | §7, derived at +24.5 EPA² and measured on 600 no-leak leagues | **Open, disclosed.** Document 35 §7's null carried none of it; §7 replaces the reference rather than the statistic. Immaterial for the incumbent, which failed at 5× its bound |
| The fitted slopes are attenuated by the reliability of a 16-game quality estimate, 0.741 | §3, §5 | **Accepted.** The residual is against *true* quality, which nothing here observes; against the measured quality M-4 reads, the fit is orthogonal by construction. Direction: the correction under-reaches the interaction channel, so the meter keeps slightly more of it than intended |
| **Rung 3's implementation holds its baseline at the unstretched team-game mean while rung 4 stretches its baseline** — the inconsistency rung 4's discarded first draft had | §6's naive-vs-module check: mean relative SD gap **+2.71%** over 40 team-games, SD 2.74%, so 6.3 standard errors; every other rung within noise of zero | **Open, disclosed, not fixed.** Rung 3 is not the adopted null, its band is not displayed, and correcting it would require re-running M-2 under amendment C-1 for a rung nothing reads. Flagged for the maintainer |
| Rung 2's PIT moves on 3 of 40 team-games under the carry-forward, by at most 0.017 | §6 | **Accepted.** Tie-breaking at 10⁻¹⁵ on the most discrete null of the four. Band bounds shift with the score, so coverage is unaffected |
| The expected-mix arm — the literal reading of avenue (a) — is not the committed score | §5, max \|score\| 34.30 against 21.56; red-zone reweight up to 11.4× | **Open by choice, flagged for overrule.** Both constructions are run and reported at every gate |
| Rung 4 passes its own design truth by one binomial SD | document 35 §6, carried forward | **Open, disclosed.** A rung-4 pass licenses "not detectably miscalibrated", not "calibrated" |
| The late-down cell prices an EPA gap where document 08's S2 verdict was on success rate | document 34 §3 | **Open, by design.** M-3 exists to re-license the shipped score itself |
| Cell counts remain endogenous to quality; only the offset they multiply is centred | §5 | **Accepted, by design.** The count still scales the score's *dispersion*, which M-4's secondary read watches descriptively |
| Score dispersion and the differential SD were measured before the gates ran | §3 | **Accepted, disclosed.** M-6's power is defined at the meter's real magnitude and cannot run without it; no threshold in this document comes from real data. Document 35 §3's precedent |
| The full-pipeline simulation draws cell EPA sums as normal at document 35 §3's cell variances, not from the real fat-tailed play pool | §7 | **Accepted.** Those variances include between-team spread, so the simulated noise is if anything larger than the residual noise, and the shared-play coupling it reports is an over-statement rather than an under-statement |
| The pipeline simulation reproduces the incumbent's leak at +0.187, where the real data gave +0.5435 | §7 | **Accepted, and it bounds the reading.** The simulated truth carries only the structural-offset channel; the real leak is larger because the interaction channel is real too. The reference distribution is therefore conservative for the redesign |
| Special-teams placement is outside the meter | document 34 §5 | **Accepted, by design** |
| No gated statistic in this document has run against data | whole document | **By construction.** That is what a pre-registration is |

---

## 10. Constants appendix

| Constant | Value | Where it lives |
|---|---|---|
| `RANDOM_SEED` | 20260820 | `research/51_placement_redesign_power.py`, `research/52_placement_redesign.py` |
| `POINTS_PER_EPA` | 0.8389 | both scripts, from document 27 §13 |
| `RED_ZONE_YARDS` / `LATE_DOWNS` | 20 / (3, 4) | both scripts |
| Profile fit | WLS on team-game cell means, weight = `n_cell`, regressor `s0_loo`, **leave-one-team-out** | §2 |
| `N_BAND_DRAWS` (production) | 2,000 | `research/52_placement_redesign.py` |
| Band interval | 5.5 – 94.5 (89% equal-tailed) | §6 |
| Ladder, adoption order | raw, raw_var_matched, down_stratified, down_stratified_var_matched | document 35 §5 |
| Adopted rung | **`raw_var_matched`**, the maintainer 2026-08-20 | §0, §6 |
| **M-1 identity tolerance** | **1 × 10⁻⁹ points** | §4 |
| **M-1 no-self-baselining tolerance** | **0.05 points** | §4 |
| **M-2 carry-forward assertion** | each rung's coverage reproduces document 35 §11's within **0.1 pp** | §6 |
| **M-3 threshold** | **r > 0.0636** | §7 |
| M-3 power at r = 0.12 | 0.916 | §7 |
| **M-4** | **descriptive; no threshold.** References: pipeline +0.1681 ± 0.0112 (same season), bound 0.1073 (other seasons) | §7 |
| **M-6 margin** | **+0.010 log loss, 95% upper** | §7, document 06's |
| M-6 power / false-pass rate | 0.965 / 0.260 | §7 |
| `N_SPLITS` / `MIN_GAMES` | 200 / 8 | document 08's, inherited |
| M-3 null / power replicates | 500 / 500 | `research/51_placement_redesign_power.py` |
| Pipeline reference / power replicates | 600 / 300 | §7 |
| Carry-forward check | 40 team-games × 2,000 draws × 4 rungs | §6 |
| `REFERENCE_R` | 0.12 | document 02, via document 08 |

Results are written back into this document as §11 by
`research/52_placement_redesign.py`. *(Written 2026-08-20, after §8's fork was
decided. Everything above this line was committed before it existed —
`git log --follow` on this file is the check.)*

---

## 11. Results

*Run 2026-08-20 by `research/52_placement_redesign.py`, against the thresholds
§§4–7 committed above and nothing else. the maintainer chose §8's fork option **(i)** on
2026-08-20 — the full meter, M-4 demoted to descriptive, rung 3 left recorded
rather than fixed. Gates ran in §8's order and each was reported before the next
started.*

**Verdict in one line: M-1 PASS, M-2 PASS, M-3 PASS, M-4 descriptive (reported
with its ceiling), M-5 report, M-6 PASS.** Every gate with teeth passed. Two
findings that were not predicted are in §11f and both are disclosed rather than
smoothed: the redesign has driven the split-half correlation *negative*, and the
same-season M-4 read sits above its own no-leak reference.

### M-1 — the identities: **PASS**

| Check | Result |
|---|---|
| Three cells sum to zero | worst residual **6.2 × 10⁻¹⁵** points, tolerance 1 × 10⁻⁹ |
| Empty red-zone cell scores exactly 0.0 | 124 team-games (2.25%), all exactly zero |
| Empty late-down cell | none exist |
| Ledger reconciliation | the 5,541 re-priced plays are exactly the ledger rows surviving the S0–S2 filter; summed repricing gap **0.0** EPA |
| Differential recomputes | worst gap **0.0** points over 2,761 games |
| **Reduction** *(new)* | with every `mu_c` zero, **4.4 × 10⁻¹⁵** points worst gap against document 35 §11's scores, on all 5,522 team-games |
| **No self-baselining** *(new)* | **0.0035** points worst move over 40 team-games, tolerance 0.05 |
| Stream reproduces §2 | 343,543 plays / 2,761 games / 5,541 ledger-carrying plays / 5,522 team-games |

The reduction check is the one that matters most for what follows. The redesign
is a strict generalisation of document 35's score to fifteen decimal places, so
every defence that design earned is inherited rather than re-argued.

### M-2 — the band: **PASS**, and the carry-forward holds

| Rung | Coverage | Document 35 §11 | Gap | Fresh-seed arm |
|---|---|---|---|---|
| 1 `raw` | 82.27% | 82.27% | 0.001 pp | 82.52% |
| **4 `raw_var_matched`** *(adopted)* | **87.45%** | **87.45%** | **0.000 pp** | 87.70% |
| 2 `down_stratified` | 85.30% | 85.37% | 0.075 pp | 85.04% |
| 3 `down_stratified_var_matched` | 89.06% | 89.10% | 0.038 pp | 89.03% |

All four reproduce within the committed 0.1 pp. **The two non-zero gaps are
exactly where §6 predicted them** — only the down-stratified rungs move, and rung
2 moves most, because those are the most discrete nulls and a 10⁻¹⁵ difference
turns a band-edge tie into a strict inequality. §6 measured this on 40 team-games
(3 PITs moved on rung 2); at full size it is 4 team-games of 5,522 on rung 2 and
2 on rung 3.

> **A departure from the literal wording of §6, flagged rather than taken
> quietly.** The 0.1 pp assertion is only a test of the carry-forward against
> document 35's own draw stream: an independent seed moves coverage by roughly
> 0.25 pp of pure resampling noise, which is *wider than the tolerance*, so a
> document-36 seed would have tested the resampler instead. The primary arm
> therefore re-uses seed 20260819 and the document-36-seeded arm (20260820) is
> reported beside it. Rung 4 sits inside [87.0%, 91.0%] under both. Recorded in
> §11g.

Rung 4 stays adopted at 87.45% of an 89% band — a pass by 0.45 pp against a
binomial SD of 0.421 pp. Read as "not detectably miscalibrated at this
tolerance", never as "exactly calibrated".

### M-3 — the luck licence: **PASS**

> Split-half r = **−0.0986** across 320 team-seasons and 5,522 team-games, 200
> within-season splits, against a pre-registered threshold of r > 0.0636 for "not
> luck". Per-split 5th–95th: −0.1689 to −0.0320.

**The word "luck" is licensed for the shipped score**, at 0.916 power against
r = 0.12 — better powered than the incumbent's gate was.

It passes in the safe direction, and how far it passes is itself a finding. See
§11f.

### M-4 — skill preservation: **descriptive, no verdict**

| Reading | Redesign | Incumbent | Reference / bound |
|---|---|---|---|
| vs **same-season** quality | **+0.2191** | +0.5435 | pipeline no-leak +0.1681 ± 0.0112 (**+4.55 SD**); bound 0.1857 |
| vs **other-seasons** quality | **+0.0883** | **+0.2098** | bound 0.1073 — **inside** |
| Late-down cell vs quality | +0.1956 | +0.5258 | — |
| Red-zone cell vs quality | +0.0421 | +0.0670 | — |
| Expected-mix arm *(reported, not shipped)* | +0.2296 | — | other-seasons +0.0983; max \|score\| 34.30 pts |
| Secondary: score spread vs quality | −0.0811 | — | no rule |

Both committed cross-checks reproduce exactly: the incumbent reads **+0.2098**
against other-seasons quality, the number §7 committed, and **+0.5435** against
same-season quality, the number document 35 §11 failed at. Document 35 §7's
independent nulls (bounds 0.1077 and 0.1104) are reported so the change of
reference is visible rather than silent.

**On the clean covariate the correction worked**: +0.2098 down to +0.0883, inside
the bound. **On the same-season covariate it did not go all the way**: +0.2191
sits 4.6 SD above the no-leak pipeline reference. §9 already discloses why that
reference is conservative — the simulation carries only the structural-offset
channel and reproduced the incumbent's leak at +0.187 where the real data gave
+0.5435, a 2.9× understatement; the redesign's ratio is 1.3×. So the excess is
consistent with the disclosed defect and far smaller than the incumbent's, and
**M-4 has no power to tell those two stories apart.** That is the gate's ceiling,
not a result. Per §8 there is no routing: M-4 is reported with its ceiling in the
same breath, and it licenses nothing about leaks.

### M-5 — magnitude: report, no threshold

| Reading | Value |
|---|---|
| Score mean / SD | −0.0719 / **4.856** points |
| Median \|score\| / q95 / max | 3.263 / 9.591 / 21.560 |
| Differential SD / median \|·\| | 6.851 / 4.656 points over 2,761 games |
| Team-games outside their own rung-4 band | **12.55%** *(the complement of 87.45%)* |
| Profile shift `C` — mean / SD / range | −0.383 / **0.451** / −2.141 … +0.917 points |
| … as a share of the score's own SD | **9.3%** |
| corr(placement differential, DTW%) | **+0.3454** |
| 195 clear-flip games: mean \|differential\| | **4.94** points, against **5.52** elsewhere |
| Per-season mean / SD | −0.258 … +0.129 / 4.59 … 5.00 |

**The meter is 91% realised placement and 9% fitted correction.** The redesign
changed the level, not the magnitude.

**Placement is not what makes a game flip.** Clear-flip games carry *less*
placement differential than the rest (4.94 against 5.52 points), so the meter
tells a different story beside DTW% rather than a louder version of the same one
— which is what document 08 §6's two-meters contract wants.

Example games, document 33 §6's:

| Game | Differential | DTW% | Actual | Deserved |
|---|---|---|---|---|
| `2018_05_GB_DET` | +1.95 | 0.054 | +8 | −8.28 |
| `2021_14_LV_KC` | +2.46 | 1.000 | +39 | +27.93 |
| `2025_17_DET_MIN` | −9.61 | 0.548 | +13 | +0.70 |

### M-6 — the rematch: **PASS**

| Arm | Mean Δ log loss | 95% upper | vs margin +0.010 |
|---|---|---|---|
| Challenger = deserved − placement | −0.00354 | **+0.00813** | **passes** |
| Deserved margin alone *(reported beside)* | −0.00357 | +0.00218 | — |

531 pairs, 10 folds, seed 20260817. The two rows say how it passes: the point
estimate barely moves (−0.00354 against −0.00357) while the interval more than
triples in width. Subtracting the placement differential adds noise to the
predictor without adding information, which is what "placement is exchangeable
luck" predicts. Read with §7's committed rates — power to pass 0.965, false-pass
**0.26**. A pass is evidence, not proof.

### 11f. Two findings this round did not predict

**1. The redesign has driven persistence negative.** M-3's r is −0.0986, which is
2.4 null SDs *below* the null mean of −0.0002 (null SD 0.0408), not near it. A
post-hoc diagnostic on the same splits — run after the verdict was fixed, moving
no threshold — locates the mechanism:

| Quantity | Split-half r |
|---|---|
| Redesigned score | **−0.0986** |
| Incumbent score | +0.0447 *(document 35 §11: +0.0436)* |
| Profile shift `C` alone | **+0.9921** |

The correction is almost perfectly persistent, because it is a smooth function of
season-level quality. Subtracting it from a near-noise quantity removes the
persistent component and overshoots. Anti-persistence is not "luck" in the plain
sense: it says a team that placed well in one half of its season placed slightly
*badly* in the other. The gate is one-sided by pre-registration and this is on
its passing side, so the verdict stands — but the product copy should not say
"placement does not persist" when what was measured is "placement anti-persists".

A candidate cause worth a round of its own: `s0_loo` is shared across a season's
games, so a good game raises its team-mates' baselines and lowers their scores,
which is a mechanical negative dependence the full-pipeline simulation could not
carry (its per-game truth is constant within a season).

**2. The same-season M-4 read exceeds its no-leak reference** by 4.6 SD, detailed
above. It cannot be adjudicated by this gate at this construction.

Neither finding changes a gate's verdict. Both are §11g rows.

### 11g. Register additions from this round

| Defect | Evidence | Status |
|---|---|---|
| **The redesigned score anti-persists**: split-half r = −0.0986, 2.4 null SDs below the null mean, where the incumbent read +0.0447 | §11f, and the post-hoc diagnostic on the same splits | **Open, disclosed.** M-3 is one-sided and this is its passing side, so the verdict stands. Copy must not read it as "does not persist". Candidate mechanism — `s0_loo` shared within a season — is not measured |
| **The same-season M-4 read is +0.2191 against a no-leak reference of +0.1681 ± 0.0112** | §11d | **Open, disclosed.** Consistent with §9's disclosed conservatism of the pipeline reference (2.9× understatement on the incumbent, 1.3× here); M-4 has no power to separate that from a residual leak |
| **M-2's primary arm re-uses document 35's seed** rather than §10's, because an independent stream's 0.25 pp of resampling noise exceeds the 0.1 pp tolerance | §11b | **Open by choice, flagged.** The document-36-seeded arm is reported beside it; rung 4 is inside tolerance under both |
| M-6 passes with a 95% upper of +0.00813 against a margin of +0.010 | §11e | **Accepted, disclosed.** §7 committed the false-pass rate at 0.26 before the gate ran; a pass here is evidence, not proof |

Every row of §9 stands unchanged. Rung 3's baseline inconsistency remains
**recorded and unfixed** by the maintainer's 2026-08-20 decision: it is not the adopted
null, its band is not displayed, and a correction would re-run M-2 under
amendment C-1 for a rung nothing reads.

### 11h. What ships, and the sentence it carries

The full meter — red zone plus late downs — gated on M-1, M-2 (carried forward,
rung 4 at 87.45%), M-3 (r = −0.0986 against 0.0636, power 0.916) and M-6 (95%
upper +0.00813 against +0.010, power 0.965), with M-4 reported descriptively and
its ceiling printed beside it.

> *The placement meter is constructed so that a team's quality cannot appear in
> it, and that construction is not independently testable.*

What is irreducibly lost is the ability to distinguish "the count artifact" from
"good offences genuinely place better". Document 08's persistence verdict is the
only evidence that the second is not real, and M-3 keeps re-checking it — which,
after §11f, it now does from the other side of zero.
