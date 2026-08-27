# 44 — What a dropped-pick count is made of, measured

*Written 2026-08-27. The results record for the study pre-registered in
document 43, run on the branch `docs/dropped-pick-confounds` — cut from the
pre-registration's own branch, because document 43 is not on `main` — in three
parts: power and thresholds (`research/61_dropped_pick_power.py`, committed
before any real fit), the fits (`research/62_dropped_pick_confounds.py`), and
this record. Nothing in `src/nfl_simulator/` changed, no ledger row moved, and
simulator v1.3 is untouched — document 32's closure was never in question here. The study
exists so the **reported diagnostic** document 32 §4 sanctioned can be worded
honestly, and so avenue (3) is argued from a powered number instead of from a
rough split-half correlation.*

*Inputs: documents 05 (the rule and the gates), 09 (the coin-flip round, whose
instrument and gate form this copies), 21 and 28 (why amendment A-2 stays shut),
32 (the closure and the numbers behind it), 43 (this study's pre-registration).*

---

## 1. The answer, stated first

**Document 43 §0's first row landed: unresolvable at this sample.** The
conditioned residual across defence-seasons has an 89% upper bound of
**5.80 pp mean, 9.37 pp at the top of the interval, against a null bound of
9.41 pp** — a Gate C-2 pass by four hundredths of a percentage point — but the
design's power at the 12.5% reference is **0.362**, so Gate C-3 fails and
*neither* outcome of C-2 may be reported as a finding. Pooled across the four
seasons the design has more chances per defence and still only reaches power
**0.555**, where the same statistic reads 9.32 pp against a tighter null bound
of 8.09 pp and C-2 *fails*. Two designs, opposite C-2 verdicts, both
uninterpretable — which is what an underpowered instrument looks like when you
let it speak twice.

So the consequence document 43 committed in advance applies as written:

> *Reported with its power table; the diagnostic says "not persistent **at this
> sample**"; avenue (3) stays closed on grounds of **unmeasurability**, not of a
> measured zero.*

Three things *were* measured, and they are the study's actual yield:

1. **The throw is where the spread is, and it is on both sides of the ball.**
   Interception-worthy rate spreads **0.94 pp across QB-seasons** and
   **0.98 pp across defence-seasons** — 26.6% and 26.4% of the league's 3.5%
   and 3.7% rates. The defence-season row is **the only design in the study
   that clears Gate C-3** (power 0.935).
2. **What decides whether a worthy throw is finished is mostly the throw.**
   Four covariates move conversion with an 89% interval clear of zero, and
   all four are properties of the ball in the air (§4).
3. **Conditioning eats document 32's +0.140.** The raw odd/even split-half
   reproduces at **+0.139**; the same split on the conditioned residual reads
   **−0.077** at the defence-season grain (§6).

And one gate failed on the way, which qualifies everything above:
**Gate C-1 fails on `sigma_q`** — see §4a. It is one parameter of 429, zero
divergences, and `sigma_d` (the study's gate statistic) is clean; the remedy is
the maintainer's call, not this document's.

## 2. Data, and the guards

Every charted pass 2022–2025, FTN joined to pbp on
`(nflverse_game_id, nflverse_play_id)`, filtered to `play_type == "pass"`. That
filter — rather than `pass_attempt == 1`, which adds 295 rows — is what
reproduces document 32's frame exactly:

| Guard (document 43 §10) | Expected | Measured | Verdict |
|---|---|---|---|
| Charted passes | 80,785 | **80,785** | exact |
| Interception-worthy throws | 2,997 ± 5% | **2,997** | exact |
| `p(INT \| worthy)` | 0.485 ± 1 pp | **0.4852** | 0.02 pp off |
| Defence-seasons | 128 | **128** | exact |

**28 worthy throws dropped for a null covariate**, and the 28 are one nested
set: all 28 lack `pass_location`, 27 of those also `air_yards`, 16 of those also
`down`. They are the throws nflverse could not place. Arm 2 therefore fits
**2,969** throws.

**What the gate arm actually sees, and why it is smaller.** Document 43 §4 sets
the residual question's QB-season unit at ≥ 20 worthy throws. A crossed design
gives every row a level on both factors, so a throw by a QB-season under the
floor has no level to belong to and leaves with it. The residual frame is
**1,145 throws, 125 defence-seasons (32 pooled defences), 46 QB-seasons**, at a
**median of 7 chances per defence-season** where document 32 §3 counted 22 on
the unfiltered rate. The floor buys conditioning at the price of 61% of the
worthy throws, and that trade — pre-registered before anyone knew which way it
would land — is the single arithmetic reason this study is unresolvable. The
125 is not a guard failure; the 128-defence-season guard is on the worthy frame,
which has 128.

## 3. Arm 1 — how much of a dropped-pick count is the quarterback's throwing

Exact beta-binomial grid (`research/_betabinom_grid.py`, unchanged), worthy
throws over charted passes, one fit per grain. Gate D-1 carries no pass rule:
skill is the expected answer here and the number's job is to tell the diagnostic
what it is describing.

| Grain | Entities | Median n | League rate | Population SD (89%) | Relative | C-3 power |
|---|---|---|---|---|---|---|
| QB-season (≥ 200 charted passes) | 148 | 484 | 3.527% | **0.939 pp** [0.767, 1.122] | 26.6% | 0.780 |
| Defence-season | 128 | 617 | 3.710% | **0.979 pp** [0.821, 1.148] | 26.4% | **0.935** |

Grid edge mass 1.0e-18 and 9.1e-25 — the posterior is dead long before the grid
boundary, so both fits are exact rather than truncated.

**The surprise is that the two rows are the same size.** Document 32 §3 read the
QB spread (split-half r = +0.284) as "the throw is skill", and it is: a
0.94 pp spread on a 3.5 pp league rate is enormous, more than double the 12.5%
the project calls real. But the *defence-season* worthy-rate spread is just as
large. Two readings fit and this study cannot separate them: defences differ in
how many interceptable throws they force (pressure, coverage, scheme), or
defences differ in the quarterbacks their schedule puts in front of them. Either
way, **a dropped-pick count is dominated by how many worthy throws happened at
all**, and that quantity is not close to coin-like on either side of the ball.

Note the honest wrinkle: at the QB grain, power at the 12.5% reference is
**0.780**, two hundredths under the project's 0.80 bar. Nothing turns on it —
D-1 has no pass rule — but the number is reported as it came out, not rounded
into compliance.

## 4. Arm 2 — what decides whether a worthy throw is finished

Document 43 §5's hierarchical logistic, verbatim: `logit p = α + Xβ + u_d + v_q`
with non-centred `u_d`, `v_q`, `HalfNormal(0.5)` scales, nutpie, 4 chains ×
1,000 draws after 1,000 tuning, seed 20260827, on 2,969 throws and 18
covariates. Covariates are pre-branch by rule — nothing recorded after the ball
reaches the defender (`is_drop`, `complete_pass`, `epa`, and the rest are
excluded and the exclusion list is in the script).

Four coefficients have an 89% interval clear of zero. In plain words:

| What moves conversion | β (logit) | Odds | Reading |
|---|---|---|---|
| `is_catchable_ball` | **−0.490** [−0.720, −0.259] | ×0.61 | a worthy throw the *receiver* could have caught is much less likely to be intercepted |
| `is_contested_ball` | **−0.318** [−0.502, −0.130] | ×0.73 | a body between defender and ball cuts the interception too |
| `air_yards` (+1 SD) | **+0.316** [+0.234, +0.398] | ×1.37 | the deeper the worthy throw, the likelier it is picked |
| `qb_hit` | **+0.213** [+0.042, +0.382] | ×1.24 | a hit quarterback's worthy throw converts more often |

The other fourteen — pass location, down, distance, field position, pre-snap win
probability, shotgun, play action, screen, out of pocket, pass rushers, and the
`air_yards` curvature — are all inside their intervals. The picture is simple
and it is about the ball: **depth and duress raise conversion, and anything that
puts an offensive player in position to interfere lowers it.**

Variance components, on both scales (probability scale via
`σ_p ≈ σ_logit × p̄(1−p̄)` at `p̄ = 0.485`):

| Component | Logit (89%) | Probability (89%) |
|---|---|---|
| `σ_d`, defence-season | 0.257 [0.138, 0.362] | 6.43 pp [3.44, 9.05] |
| `σ_q`, QB-season | 0.209 [0.046, 0.345] | 5.22 pp [1.14, 8.62] |

**Arm 2b, the hindsight sensitivity.** Document 43 §3(c) flagged that a charter
who saw the interception may grade `is_catchable_ball` and `is_contested_ball`
with hindsight. Refitting without them barely moves anything: `air_yards`
+0.309 (from +0.316), `qb_hit` +0.213 (unchanged), `σ_d` 6.67 pp (from 6.43),
`σ_q` 5.62 pp (from 5.22). **The study's conclusions do not rest on the two
hindsight-risk columns** — that defect can be closed as measured and immaterial.

**Reproducibility of the fixed effects.** Part A saved a posterior-mean `β̂`
before the thresholds were set; Part B re-fits the same model with the same
seed. Max |Δβ| **0.0044**, |Δα| **0.0040** — so the gate was judged at the
`p̂` its thresholds were built for.

### 4a. Gate C-1 failed, on one parameter

| Arm | Divergences | max `r_hat` | min `ess_bulk` | min `ess_tail` | Offender | Verdict |
|---|---|---|---|---|---|---|
| 2 | **0** | **1.0105** | 475 | 418 | `sigma_q` | **FAIL** (r_hat) |
| 2b | **0** | 1.0057 | **387** | **345** | `sigma_q` | **FAIL** (ESS) |

Checked over all 429 parameters, non-centred offsets included. In arm 2 exactly
**one** parameter is at or over the `r_hat` bar and none is under an ESS bar; in
arm 2b exactly one is under an ESS bar and none over `r_hat`. In both arms the
offender is `σ_q`, the QB-season scale — a variance parameter whose posterior
crowds the zero boundary (`0.046` at the bottom of its 89% interval), which is
where a hierarchical scale is hardest to sample even non-centred.

Three facts that bound how much this costs:

- **`σ_d` is clean.** The study's gate statistic and its cross-check ride on the
  defence-season component, not the QB one.
- **Zero divergences in both arms.** This is a mixing-length failure, not a
  geometry failure.
- **The instrument half of Gate C-1 passes.** Arm 2's `σ_d` 89% upper bound
  (9.05 pp) agrees with arm 3's crossed-grid bound to **0.32 pp** at the
  defence-season grain and **0.27 pp** pooled, against a pre-registered
  tolerance of 1.0 pp. Two different instruments, two different frames, the same
  answer.

The obvious remedy — more draws — is a change to document 43 §5's committed
inference spec, so it is **not applied here**. Handoff constraint 8 makes a
Gate C-1 failure a stop-and-ask, and this document is where it stops.

## 5. Arm 3 — does the conditioned residual persist? (Gates C-2, C-3)

Per-throw residual `r_i = y_i − p̂_i` against arm 2's fixed-effects prediction,
fitted with the crossed Gaussian grid (`research/_crossed_gaussian_grid.py`,
unchanged). The statistic is the 89% upper bound on `σ_d` on the probability
scale, in percentage points — the gate statistic and the power instrument being
the same object, per document 09's rule.

| Design | `σ_d` (89%) | Upper bound | Threshold | C-2 | C-3 power | Reportable? |
|---|---|---|---|---|---|---|
| Defence-season × QB-season | 5.80 pp [1.21, 9.37] | 9.37 pp | 9.41 pp | **PASS** by 0.04 pp | **0.362** | **No** |
| Defence pooled × QB-season | 5.09 pp [1.20, 9.32] | 9.32 pp | 8.09 pp | **FAIL** by 1.23 pp | **0.555** | **No** |
| QB-season `σ_q` (same fit) | 4.93 pp [0.88, 9.30] | 9.30 pp | 8.07 pp | **FAIL** | **0.578** | **No** |

Residual SD 48.5 pp in both fits — a Bernoulli outcome around a p̂ near one
half, which is the floor this design is trying to hear a 6 pp signal against.
Grid edge mass 2.9e-03 and 2.5e-03.

Document 43 §7's decision rule ran as written: defence-season × QB-season
first, defence-pooled only because the first failed C-3, **and both failed C-3,
so the study is unresolvable.** The lower bound of every interval is 1.2 pp
rather than zero, which is the log-scale grid's known floor (documented in
`_crossed_gaussian_grid.py`) — it is why these bounds are read against a
simulated null and never against zero.

The two C-2 verdicts disagreeing is worth naming rather than smoothing. Pooling
across seasons gives each defence ~4× the chances, which tightens the null bound
from 9.41 pp to 8.09 pp — but the observed bound barely moves (9.37 → 9.32),
so the same data reads as a pass against the loose threshold and a fail against
the tight one. Neither is a finding. It is the signature of a design that cannot
tell 0 pp from 6 pp, which is exactly what its 0.362 and 0.555 said in advance.

## 6. Secondaries, and the arm 2b sensitivity

Reported, never gated — document 43 §5's continuity check with document 32.

| Quantity | Entities | r |
|---|---|---|
| Raw conversion, odd/even split-half, defence-season | 126 | **+0.139** |
| Raw conversion, odd/even split-half, defence pooled | 32 | **+0.142** |
| Conditioned residual, odd/even, defence-season | 58 | **−0.077** |
| Conditioned residual, odd/even, defence pooled | 32 | **+0.192** |
| Shrunk defence-season effects (arm 2 `u_d`), season to season | 96 pairs | **+0.063** |

**Document 32's +0.140 reproduces.** +0.139 at ≥ 3 chances per half, +0.142
pooled. That is the number this study set out to condition, and it is real as
reported.

**Conditioning removes it at the defence-season grain and does not pool
cleanly.** The conditioned residual reads −0.077 on 58 defence-seasons — but
the comparison is not apples to apples, and the asymmetry is the point:
conditioning costs the ≥ 20-worthy floor, so 126 entities at a median 11
chances per half become 58 at a median 7. The pooled row moves the other way
(+0.192 on 32 defences), and both are far inside the noise a 58- or 32-point
correlation carries. **These numbers are consistent with composition explaining
document 32's +0.140, and equally consistent with it surviving.** That is the
same unresolvability §5 gated, arriving by a second route.

**Season-to-season persistence of the shrunk defence effect is +0.063 on 96
pairs** — near zero, and on a quantity that arm 2 shrinks hard by construction,
so it is the weakest of the three signals rather than the tiebreaker.

## 7. The sentence this licenses on the game page

One sentence, in the language the outcome supports and no stronger — the
deliverable document 43 §8 promised on every outcome:

> **"Interceptions your opponent dropped are counted here as they happened.
> Whether some defences finish more of their chances than others is a question
> this data cannot settle: four seasons of charted throws leave a 6-point-wide
> band around the answer either way."**

What the wording may **not** say, given the gates: that dropped picks do not
repeat (C-3 failed, so the pass is not a finding), that they do (C-2 failed
pooled, and that is not a finding either), or that the finish is a coin flip
(document 05 §2's distinction — a rate near one half is not a mechanism).
What it **may** say, on arm 1's evidence, is the thing that actually holds:
a dropped-pick count is mostly a count of interceptable throws, and that
quantity is skill on both sides of the ball.

## 8. Register

| Defect | Evidence | Status |
|---|---|---|
| Defensive conversion persistence unconditioned and unpowered | Document 32 §3: r = +0.140, no power | **Closed as unmeasurable.** Now conditioned and powered: C-3 = 0.362 / 0.555, both under 0.80. The number is not zero and not confirmed; the design cannot tell |
| Charter hindsight in `is_catchable_ball` / `is_contested_ball` | Document 43 §3(c), named risk | **Closed, immaterial.** Arm 2b moves `σ_d` 6.43 → 6.67 pp and `air_yards` +0.316 → +0.309 |
| `β` held fixed in the residual power simulation | Document 43 §6 | **Disclosed, quantified as small.** Part A's `β̂` and Part B's re-fit agree to max \|Δβ\| 0.0044 |
| **Gate C-1 fails on `σ_q`** | §4a: r_hat 1.0105 (arm 2), ess_tail 345 (arm 2b), 0 divergences, 1 of 429 parameters | **Open — the maintainer's call.** Remedy (longer chains) is a change to document 43 §5's committed spec |
| The ≥ 20-worthy floor costs 61% of the sample | §2: 2,969 → 1,145 throws, median 22 → 7 chances per defence-season | **Open, and it is the whole story.** Pre-registered, so not a violation; a future round wanting an answer must buy chances, not conditioning |
| `is_interception_worthy` is a charter's judgement | Document 09's register, standing | **Open.** Biases toward finding no spread; arm 1's 26% spreads are measured through that noise, not around it |
| Defence-seasons drop 128 → 125 in the residual frame | §2 | **Accepted.** A consequence of the pre-registered floor, not a data defect |

Nothing in this document opens Gate A, and nothing in it admits a ledger row.
Document 32's closure stands, on the argument it always rested on.
