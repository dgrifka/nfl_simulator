# 46 — Dropped picks, round 2: what removing one floor bought

*Written 2026-08-27. The results record for round 2 of the study pre-registered
in document 43 and amended by document 45, run on the branch
`docs/dropped-pick-confounds` in three parts: power on the floorless frame
(`research/63_dropped_pick_power_r2.py`, committed before any real fit), the
fits (`research/64_dropped_pick_confounds_r2.py`), and this record. Nothing in
`src/nfl_simulator/` changed, no ledger row moved, and simulator v1.3 is
untouched — document 32's closure was never in question in either round.*

*Inputs: documents 09 (the instrument and the gate form), 17 §3 (the deflection
cross-tab the hindsight probe extends), 32 (the closure), 43 (the
pre-registration), 44 (round 1's record), 45 (this round's amendments).*

---

## 1. The answer, stated first — and it is not round 1's

**The power change is the whole story, so it comes first.** Removing document
43 §4's ≥ 20-worthy-throw floor — amendment A-1, and nothing else about the
design — moved the gate arm's power at the 12.5% reference from **0.362 to
0.892** at the defence-season grain and from **0.555 to 0.953** pooled. Round 1
failed Gate C-3 on both designs and could report nothing; round 2 passes C-3 on
both. The instrument can now tell a 6 pp effect from no effect, and it could not
before.

**So the study is no longer unresolvable, and document 43 §0's *third* row
landed where round 1 landed on the first.** Document 43 §7's decision rule reads
the defence-season × QB-season design first and stops there if it clears C-3. It
clears C-3 at 0.892, and its Gate C-2 **fails**: the conditioned residual
spreads **5.95 pp across defence-seasons [3.11, 8.04] against a null bound of
5.92 pp**, so the upper bound sits 2.12 pp above the threshold a skill-free
league clears 90% of the time. The pre-registered reading of that row:

> *Some defences finish more of their chances, repeatably: ball-hawking is
> skill, the document 09 receiver trap with the jerseys swapped. The diagnostic
> must say so in words; avenue (3) is dead on the evidence.*

**Three things qualify that reading, and two of them are the maintainer's to rule on.**

1. **The word "repeatably" is not supported by this round's own evidence.** The
   pooled design — 32 defences over four seasons, C-3 power 0.953 — puts the
   same spread at **2.76 pp [0.54, 5.01] against a 5.06 pp bound**, a Gate C-2
   **pass** by 0.05 pp. The two designs now disagree while *both* are powered,
   which is a different animal from round 1's disagreement between two designs
   that were not. Season-to-season correlation of the shrunk defence effect is
   **+0.065 on 96 pairs**. The consistent reading of all three numbers is
   *defence-seasons differ; defences do not* — spread that lives inside a season
   and does not carry to the next one. The pre-registered rule does not consult
   the pooled row at all, and the word it licenses says the opposite of what
   that row says. §7 is written both ways for that reason.
2. **Gate C-1's cross-check half fails** (§4a). Not the sampler half — A-2 fixed
   that — the instrument-agreement half.
3. **Conditioning did not eat document 32's +0.140.** Round 1's apparent
   reversal to −0.077 was an artifact of the floor. On the full 126 defence-
   seasons the conditioned split-half is **+0.127** against the raw **+0.139**
   (§5). Document 44 §6 read that comparison the other way, on 58 entities, and
   said so at the time; this round's number is the like-for-like one.

## 2. The frame, and the guards

Unchanged from round 1 and re-measured, because a guard that is not re-run is
not a guard:

| Guard (document 43 §10) | Expected | Measured | Verdict |
|---|---|---|---|
| Charted passes | 80,785 | **80,785** | exact |
| Interception-worthy throws | 2,997 ± 5% | **2,997** | exact |
| `p(INT \| worthy)` | 0.485 ± 1 pp | **0.4852** | 0.02 pp off |
| Defence-seasons | 128 | **128** | exact |
| Arm 3 frame vs arm 2 frame | identical | **identical, 2,969 rows** | document 45 §2's stop-and-ask, not triggered |

28 worthy throws still drop for a null covariate — the same nested set, all 28
missing `pass_location` — leaving **2,969** throws.

**What amendment A-1 bought, in units.** The gate arm now sees every one of those
2,969 throws instead of 1,145; **128** defence-seasons instead of 125; **280**
QB-season levels instead of 46; and a median of **22 chances per defence-season**
instead of 7. The thinnest QB-season level is a **single** worthy throw, which is
the amendment working as designed: the hierarchy shrinks a one-throw level toward
the mean, where the floor deleted the twenty throws sitting around it.

| Design | Levels | Median chances per level | Null bound = threshold | Power at 12.5% | Round 1 |
|---|---|---|---|---|---|
| Defence-season × QB-season | 128 × 280 | 22 | **5.920 pp** | **0.892** | 9.410 pp / 0.362 |
| Defence pooled × QB-season | 32 × 280 | 93 | **5.060 pp** | **0.953** | 8.086 pp / 0.555 |
| QB-season `σ_q` | 128 × 280 | 22 | **6.889 pp** | **0.780** | 8.075 pp / 0.578 |

The thresholds fell as much as the power rose, and for the same reason: more
chances per level make a skill-free league's upper bound tighter as well as
making a real effect easier to see. Cost: 5,600 crossed fits, **3,730 s of wall
clock** across 8 workers, median 4.27 s per fit (max 5.08 s). `DATASETS` stayed
at 400.

## 3. Arm 2 — conversion, at amendment A-2's spec

Document 43 §5's hierarchical logistic on 2,969 throws and 18 covariates, now at
**4 chains × 2,000 draws after 2,000 tuning, `target_accept` 0.9** (A-2). Arm 1
was not re-run: its rate designs never saw the floor and its numbers stand in
document 44 §3.

Four coefficients keep an 89% interval clear of zero, and they are the same four
as round 1, at the same sizes:

| What moves conversion | β (logit) | Odds | Round 1 |
|---|---|---|---|
| `is_catchable_ball` | **−0.489** [−0.714, −0.261] | ×0.61 | −0.490 |
| `is_contested_ball` | **−0.321** [−0.513, −0.135] | ×0.73 | −0.318 |
| `air_yards` (+1 SD) | **+0.316** [+0.234, +0.397] | ×1.37 | +0.316 |
| `qb_hit` | **+0.212** [+0.046, +0.379] | ×1.24 | +0.213 |

The other fourteen sit inside their intervals, as before. **Depth and duress
raise conversion; anything that puts an offensive player in position to interfere
lowers it.** Doubling the chains moved no coefficient meaningfully: against round
1's arm 2 the largest gap is **max |Δβ| 0.0043**, and against Part A's saved
`β̂` — the fixed effects arm 3 residualises against, and the ones Part A's
thresholds were simulated around — **max |Δβ| 0.0079, |Δα| 0.0010**.

Variance components, both scales (`σ_p ≈ σ_logit × p̄(1−p̄)` at `p̄ = 0.485`):

| Component | Logit (89%) | Probability (89%) | Round 1 |
|---|---|---|---|
| `σ_d`, defence-season | 0.254 [0.125, 0.364] | **6.35 pp [3.11, 9.08]** | 6.43 pp [3.44, 9.05] |
| `σ_q`, QB-season | 0.205 [0.044, 0.341] | **5.13 pp [1.09, 8.52]** | 5.22 pp [1.14, 8.62] |

**Arm 2b, the hindsight sensitivity**, again immaterial: `σ_d` 6.71 pp (from
6.35), `air_yards` +0.311 (from +0.316), `qb_hit` +0.213. That defect stays
closed.

### 3a. Gate C-1's sampler half now passes — round 1's open question is closed

| Arm | Divergences | max `r_hat` | min `ess_bulk` | min `ess_tail` | Over a bar | Verdict |
|---|---|---|---|---|---|---|
| 2 | **0** | **1.0070** (`sigma_q`) | **587** | **522** | **0 of 429** | **PASS** |
| 2b | **0** | **1.0074** (`sigma_q`) | **467** | **415** | **0 of 427** | **PASS** |

Round 1 failed here on exactly one parameter in each arm — `sigma_q` at `r_hat`
1.0105 in arm 2 and `ess_bulk` 387 / `ess_tail` 345 in arm 2b. Doubling draws and
tuning and raising `target_accept` to 0.9 clears both, and `sigma_q` remains the
hardest parameter in the model without being over any bar. **Document 44 §8's
open register row closes here, by the remedy document 45 A-2 pre-registered
rather than by one chosen after seeing which way it would land.**

## 4. Arm 3 — does the conditioned residual persist? (Gates C-2, C-3)

Per-throw residual `r_i = y_i − p̂_i` against Part A's `β̂`, crossed Gaussian
grid, on the floorless frame.

| Design | `σ_d` (89%) | Upper bound | Threshold | C-2 | C-3 power | Reportable? |
|---|---|---|---|---|---|---|
| **Defence-season × QB-season** | **5.95 pp [3.11, 8.04]** | 8.04 pp | 5.92 pp | **FAIL** by 2.12 pp | **0.892** | **Yes** |
| Defence pooled × QB-season | 2.76 pp [0.54, 5.01] | 5.01 pp | 5.06 pp | **PASS** by 0.05 pp | **0.953** | **Yes** |
| QB-season `σ_q` (same fit) | 4.80 pp [1.20, 7.99] | 7.99 pp | 6.89 pp | FAIL | **0.780** | **No** |

Residual SD 48.6 pp and 48.7 pp — the Bernoulli floor this design listens for a
6 pp signal against. Grid edge mass 1.5e-03 and 3.8e-03.

**Read the first row in relative terms and it lands on the project's own
yardstick.** 5.95 pp is **12.2% of the 48.94% league conversion rate** — within a
rounding error of the 12.5% materiality reference document 04 set and this study
powered against. The data does not look like a small real effect the instrument
barely caught; it looks like an effect of exactly the size the instrument was
built to detect.

**Read the second row and it does not.** 2.76 pp pooled is **5.6% relative**,
and 5% is the scenario where this design's power is 0.30 — i.e. indistinguishable
from zero. Two powered designs, one saying the spread is at the reference and one
saying it is at a fifth of it.

**These are not in contradiction, and the reconciliation is the interesting
part.** A defence-season effect contains everything that varies within one
team-season; a pooled-defence effect keeps only what a team carries across four
of them. If finishing were a stable trait, the pooled estimate would be close to
the season one and the season-to-season correlation would be substantial. It is
2.76 against 5.95, and the correlation is +0.065 on 96 pairs. **What repeats
inside a season does not carry to the next one.** That is a statement this round
can make with power behind it, and it is *not* the statement document 43 §0's
third row licenses in words.

### 4a. Gate C-1's cross-check half fails

Document 43 §5 and §10 require arm 2's `σ_d` 89% upper bound, mapped to the
probability scale, to agree with arm 3's within **1.0 pp**.

| Comparison | Arm 2 | Arm 3 | Gap | Tolerance | Verdict |
|---|---|---|---|---|---|
| vs defence-season × QB-season | 9.08 pp | 8.04 pp | **1.04 pp** | 1.0 pp | **FAIL** by 0.04 pp |
| vs defence pooled × QB-season | 9.08 pp | 5.01 pp | **4.07 pp** | 1.0 pp | **FAIL** |

Round 1 passed both at 0.32 pp and 0.27 pp. **Nothing was re-run, re-scaled or
re-toleranced to make this pass**; document 43 §5 pre-registered the handling —
*"both numbers are reported and the disagreement is a finding of its own"* — and
that is what §7 and the register below do. Two observations that bound how much
it costs, neither of which is a licence to ignore it:

- **The pooled row was never a like-for-like comparison.** Arm 2's `σ_d` is a
  defence-**season** scale; arm 3's pooled `σ_d` is a defence scale across four
  seasons. §4 has just argued at length that those two objects genuinely differ,
  so a 4.07 pp gap between them is partly the finding and partly the gate. Round
  1 could not see this because both numbers were pinned near the instrument's
  ceiling by low power; the cross-check "passed" on an agreement that was an
  artifact of neither instrument being able to resolve anything.
- **The defence-season row misses by 0.04 pp**, which is inside the seed-to-seed
  wobble this study has already shown it carries (document 44 §5 flagged a Gate
  C-2 pass by the same 0.04 pp as not seed-robust).

**This is handoff constraint §7's stop-and-ask, and it stops here.** Whether a
cross-check failure at the pooled grain should have been written as a grain-
matched comparison in the first place is a change to document 43 §5's committed
text, so it was not made.

**Ruled 2026-08-27 (document 47 R-1).** The cross-check is like-grain only.
Against arm 3's defence-season × QB-season `σ_d` the gap is **1.04 pp against a
1.0 pp tolerance**, recorded as **PASS-at-tolerance** with the 0.04 pp disclosed,
and arm 2's `β̂` is quotable. The pooled row is **withdrawn as unlike-grain** —
arm 2's `σ_d` is a defence-season scale and arm 3's pooled `σ_d` is a
four-season one, so the 4.07 pp gap was never a cross-check. The register row
below closes on that ruling.

## 5. Secondaries — now like-for-like, and they change document 44 §6's reading

| Quantity | Entities | Round 2 | Round 1 |
|---|---|---|---|
| Raw conversion, odd/even split-half, defence-season | 126 | **+0.139** | +0.139 |
| **Conditioned residual, odd/even, defence-season** | **126** | **+0.127** | −0.077 *(on 58)* |
| Raw conversion, odd/even, defence pooled | 32 | **+0.142** | +0.142 |
| Conditioned residual, odd/even, defence pooled | 32 | **+0.092** | +0.192 |
| Shrunk defence-season effect, season to season | 96 pairs | **+0.065** | +0.063 |

**Document 32's +0.140 reproduces, and conditioning barely touches it.** The
comparison is finally on the same 126 defence-seasons at a median 11 chances per
half on both sides, which is what the floor made impossible in round 1: **+0.139
raw → +0.127 conditioned.** Round 1's −0.077 was measured on 58 entities and
document 44 §6 flagged it as not apples-to-apples at the time; this round says
plainly that it was an artifact of the floor and not a reversal.

That matters for the study's own argument. Document 43 §3's mechanism story said
that if document 32's +0.140 were composition — defences facing finishable
throws — the conditioned residual spread would collapse toward the null bound.
**It does not collapse.** Eighteen pre-branch throw covariates remove about a
tenth of the split-half correlation and none of the spread, which is the same
message Gate C-2's first row carries.

The season-to-season figure is the one that keeps "repeatably" honest: **+0.065
on 96 pairs**, near zero, on a quantity arm 2 shrinks hard by construction.

## 6. A-3 — the hindsight probe, and what it does and does not rule out

Document 45 §2's probe, on all **1,690** charted interceptions 2022–2025 with a
readable second-toucher channel (`pass_defense_1_player_id !=
interception_player_id`, document 17 §3's channel verbatim):

| Quantity | Value |
|---|---|
| `p(worthy \| INT, second toucher)` | **0.717** (195 of 272) |
| `p(worthy \| INT, no second toucher)` | **0.888** (1,259 of 1,418) |
| Gap | **−17.1 pp** |
| Share of charted interceptions marked **not** worthy | **14.0%** (236 of 1,690) |

Every cell reproduces document 17 §3's cross-tab exactly, which is the check that
the channel was rebuilt correctly rather than approximated.

**Document 45 §2's pre-committed reading, applied:** the deflected-pick worthy
rate is materially *below* the clean-pick rate, so **the flag is behaving as a
judgement of the throw and the selection is defensible.** A charter working
backwards from the result would call a deflected pick worthy about as often as a
clean one; this charter calls it worthy 17 points less often, because a deflected
pick is usually a fine throw meeting a hand.

**What this does not rule out, stated plainly.** The probe catches the gross form
of hindsight and nothing subtler. It cannot detect a charter who grades every
intercepted throw one notch harsher across the board — that would shift both
cells together and leave the gap intact. It says the flag is not *purely* an
outcome relabelled; it does not say the flag is outcome-blind. Reported, never
gated, exactly as document 45 §2 specified.

## 7. The sentence this licenses on the game page

Document 44 §7's sentence was written for an unresolvable study:

> *"Interceptions your opponent dropped are counted here as they happened.
> Whether some defences finish more of their chances than others is a question
> this data cannot settle: four seasons of charted throws leave a 6-point-wide
> band around the answer either way."*

**That sentence no longer describes the evidence and must be replaced** — the
question is now settled well enough to say something, and saying "cannot settle"
would understate what a powered instrument found. What replaces it depends on a
call that is the maintainer's, because the pre-registered rule and the pre-registered
statistic point at different words:

**(a) The strict reading of document 43 §7's decision rule** — defence-season ×
QB-season only, C-3 passed, C-2 failed:

> *"Interceptions your opponent dropped are counted here as they happened. Some
> defences do finish more of the chances they get than others — after adjusting
> for how catchable, contested, deep and hurried each throw was, the gap between
> defences is about as large as this project calls meaningful."*

**(b) — ADOPTED (document 47 R-2).** The reading that also honours the pooled
design and the season-to-season correlation, both of which cleared C-3 and
neither of which the rule consults. This is the wording that ships:

> *"Interceptions your opponent dropped are counted here as they happened. In any
> given season some defences finish more of their chances than others — but that
> edge doesn't carry to the next season, so it reads as a year, not a trait."*

**What neither wording may say, given the gates:** that the finish is a coin flip
(document 05 §2 — a rate near one half is not a mechanism); that dropped picks do
not repeat at all (the defence-season C-2 failure says the opposite); or that any
of it belongs in the ledger. **Avenue (3) — reopening amendment A-2 — is dead on
the evidence** under either wording, which is the one consequence both readings
share: A-2 clause 1 demands 0.80 power at a *5%-of-variance* share, and this
design's power at 5% is 0.20 and 0.30.

**Recommendation, not a decision:** (b). It is the only wording consistent with
every powered number this round produced, and (a)'s "repeatably" is a word the
pre-registered rule supplies rather than one the data earned.

**Decided 2026-08-27 (document 47 R-2):** (b), and document 43 §7's decision
rule is amended so that when both grains clear C-3 both are reported and the
wording must be consistent with both. Avenue (3) is closed on the evidence.

## 8. Register

| Defect | Evidence | Status |
|---|---|---|
| Defensive conversion persistence unconditioned and unpowered | Document 32 §3: r = +0.140, no power | **Measured.** Conditioned and powered at 0.892 / 0.953. Spread is 12.2% relative within a defence-season, 5.6% pooled across four |
| **Gate C-1 fails on `σ_q`** (round 1) | Document 44 §4a | **Closed by amendment A-2.** 0 divergences, max `r_hat` 1.0070, 0 of 429 parameters over any bar |
| The ≥ 20-worthy floor costs 61% of the sample | Document 44 §2 | **Closed by amendment A-1.** 1,145 → 2,969 throws, median 7 → 22 chances, power 0.362 → 0.892 |
| Charter hindsight in `is_catchable_ball` / `is_contested_ball` | Document 43 §3(c) | **Closed, immaterial**, second round confirming: `σ_d` 6.35 → 6.71 pp |
| Charter hindsight in `is_interception_worthy` itself | Document 45 §2 (A-3) | **Gross form ruled out**, −17.1 pp gap. Subtle uniform-severity hindsight remains undetectable by this probe (§6) |
| **Gate C-1's cross-check** | §4a: gap 1.04 pp like-grain, 4.07 pp unlike-grain, vs a 1.0 pp tolerance | **Closed by ruling (document 47 R-1).** Like-grain only: **PASS-at-tolerance** at 1.04 pp with the 0.04 pp disclosed; the pooled comparison is withdrawn as unlike-grain. Arm 2's `β̂` is quotable |
| **Two powered designs disagree on Gate C-2** | §4: FAIL by 2.12 pp at the season grain, PASS by 0.05 pp pooled | **Closed by ruling (document 47 R-2).** Both grains cleared C-3, so both are reported and wording (b) — "a year, not a trait" — is adopted. Document 43 §7's decision rule amended to match |
| Document 44 §6's "conditioning eats the +0.140" | §5: +0.127 on the like-for-like 126 entities | **Corrected.** Round 1's −0.077 was the floor, on 58 entities, and was flagged as such when published |
| `β` held fixed in the residual power simulation | Document 43 §6 | **Disclosed, quantified as small.** Part A's `β̂` vs round 2's arm 2: max \|Δβ\| 0.0079 |
| `is_interception_worthy` is a charter's judgement | Document 09's register, standing | **Open**, now with A-3's bound on one form of it |
| Pooled `σ_q` reads 6.20 pp against the season fit's 4.80 pp | §4 table, both from the same frame | **Noted, not chased.** `σ_q` fails C-3 at 0.780, so neither number is reportable as a finding |

Nothing in this document opens Gate A, and nothing in it admits a ledger row.
Document 32's closure stands, on the argument it always rested on.

**Commits.** `4240339` document 45 (the pre-registration) · `261f9c3` Part A, the
power table and its thresholds · `34b8b9d` Part B, the fits · Part C, this
record.
Branch `docs/dropped-pick-confounds`, unmerged; the maintainer merges. `git diff main --
src/` is empty; 502 tests pass; ruff clean.
