# 43 — Dropped-interception confound study, pre-registered

*Written 2026-08-27 in a Fable 5 brainstorm, **before any fit**. This is the
change proposal and the gate for handoff §2 item 3, avenue (2): a measurement
of what a "dropped picks" count is actually made of. It changes nothing in
simulator v1.3. Document 32's closure stands: a dropped interception has no
branch point, Gate A is not reopened by any result below, and amendment C-1
admits no new rows. The study exists so that the **reported diagnostic**
document 32 §4 sanctioned can be worded honestly — "the opponent failed to
finish" versus "the opponent is bad at finishing" — and so that avenue (3),
reopening amendment A-2, is argued from a powered number rather than from the
rough +0.140 split-half document 32 §3 reported.*

*Inputs: documents 05 (rule and gates), 09 (the coin-flip round — the
instrument and the gate form this document copies), 21 (A-2's text), 28 (why
C-1 cannot admit a row), 32 (the closure and the numbers behind it).*

---

## 0. The question, and what each answer would mean

A team whose opponents dropped five would-be interceptions in a game had
something happen to it. Document 32 §3 measured that the *throw* is skill
(QB interception-worthy rate, split-half r = +0.284) and that the *finish*
may be too (defensive conversion, r = +0.140 on 22 chances per team-season).
Neither number was conditioned on anything, and neither carried a power
figure. Four questions follow, in the order they must be answered:

1. **How much of a dropped-pick count is the QB's own throwing?** — the
   spread of interception-worthy rate across QBs, and across defences.
2. **What decides whether a worthy throw is finished?** — conversion,
   p(INT | worthy), by covariates that exist before the defender's hands
   touch the ball.
3. **Does the conversion residual persist?** — after those covariates, is
   there a defence (or QB) component that repeats?
4. **Could this design see it if it were there?** — power, computed and
   committed before any threshold is filled in.

The readings are committed now, before a number exists:

| Outcome | Reading | Consequence |
|---|---|---|
| Gate C-3 fails (power < 0.80 at 12.5% rel) at every grain | Unresolvable at this sample, as document 09 read onside kicks | Reported with its power table; the diagnostic says "not persistent *at this sample*"; avenue (3) stays closed on grounds of **unmeasurability**, not of a measured zero |
| C-3 passes, C-2 passes (residual spread below the null bound) | The finish behaves like a rate, not a skill, once the throw is conditioned on | The diagnostic may describe dropped picks as opponent failure that does not repeat. Reopening A-2 is **the maintainer's decision only**; note A-2 clause 1 demands 0.80 power at a *5%-of-variance* share, a harder bar than C-3 |
| C-3 passes, C-2 fails (residual spread above the bound) | Some defences finish more of their chances, repeatably: ball-hawking is skill, the document 09 receiver trap with the jerseys swapped | The diagnostic must say so in words; avenue (3) is dead on the evidence |

No outcome touches the ledger. That is the settled part.

## 1. Tier declaration

**Model change** — a new hierarchical model, fitted to a new data slice (FTN
charting joined to pbp, 2022–2025, interception-worthy throws only). It feeds
a reported diagnostic, not the simulator, so §7's rollback is trivial; every
other section fills in full. The tier is "model change" rather than "data
change" because the generative story is new, not because anything in v1.3
moves.

## 2. DAG

Current (document 32 §3, implicit): `worthy → INT`, one league-wide rate.

Proposed:

```
QB skill ──────────► worthy throw ─────┐
defence pressure ──┘                   │
                                       ▼
   throw covariates X ─────────► p(INT | worthy, X)  ◄── defence finish u_d
   (air yards, location, contested,          ▲
    catchable, QB hit, rushers, out of       │
    pocket, play action, screen, down,       └────────── QB "droppability" v_q
    distance, field position, pre-snap WP)
                                       │
                                       ▼
                                     INT ──► EPA (not modelled here)
```

Two nodes are the study: `u_d` (defence-season finish effect) and `v_q`
(QB-season effect on how finishable that QB's worthy throws are). The
covariate arrow `X → p` is what document 32's unconditioned +0.140 lacked: a
defence that faces short, contested, hurried worthy throws will finish more of
them for reasons that are the *offence's* doing.

Nothing feeds back into `core` or any ledger component.

## 3. Mechanism story

- **Defect addressed.** Document 32 §3's defensive conversion figure is an
  odd/even split-half correlation at a median of 22 chances per team-season,
  described there as "a rough number, not a calibrated one". It is
  unconditioned, so it cannot separate "this defence finishes" from "this
  defence's opponents throw finishable balls". The register row in this
  document's §9 is the defect.
- **Why this change should move that number.** Conditioning on throw
  covariates removes the composition channel. If the +0.140 is composition,
  the conditioned residual spread collapses toward the null bound; if it is
  ball-hawking, it survives conditioning. Either way the 89% upper bound on
  `sd(u_d)` is a calibrated statement where the split-half r was not, and its
  power is measured rather than assumed.
- **What would make it fail.** (a) Power at 128 defence-seasons × ~22 chances
  may be too low to discriminate — document 09's onside row is the warning.
  Both grains (defence-season and defence pooled across four seasons) are
  pre-registered for that reason, and a C-3 failure is a legitimate outcome,
  reported as such. (b) FTN's `is_interception_worthy` is a charter's
  judgement, so "worthy" carries charter noise that looks like QB or defence
  effect; the QB-season and defence-season worthy-rate spreads in Gate D-1
  will show how much. (c) `is_catchable_ball` and `is_contested_ball` are
  charted about the throw, but a charter who has seen the interception may
  grade the throw with hindsight; §5 runs the conversion model with and
  without those two columns and reports both.

## 4. Data

- **Rows:** every charted pass 2022–2025 with `is_interception_worthy == True`,
  FTN joined to pbp on `(nflverse_game_id, nflverse_play_id)`. Document 32
  counted 2,997 worthy throws of 80,785 charted passes, 1,454 intercepted.
  **Stop and ask if the worthy count differs from 2,997 by more than 5%.**
- **Units:** defence-season `d` (expected 128), defence pooled `D` (32),
  QB-season `q` with ≥ 200 attempts for the worthy-rate question (document
  32's floor) and ≥ 20 worthy throws for the residual question.
- **Covariates `X`, all pre-branch:** `air_yards` (standardised, plus its
  square), `pass_location` (left / middle / right, middle as reference),
  `is_contested_ball`, `is_catchable_ball`, `qb_hit`, `n_pass_rushers`
  (standardised), `is_qb_out_of_pocket`, `is_play_action`, `is_screen_pass`,
  `down` (factor), `ydstogo` (standardised), `yardline_100` (standardised),
  `shotgun`, pre-snap `wp` (standardised). Rows with a null covariate are
  dropped and the count reported.
- **Excluded by rule:** anything recorded after the ball reaches the
  defender — `interception_player_*`, `is_drop`, `is_created_reception`,
  `complete_pass`, `epa`, `wpa`. `is_catchable_ball` and `is_contested_ball`
  are included in the main model and **removed in a sensitivity arm** (§5
  arm 2b) because of the hindsight risk named in §3(c).

## 5. Models, inference, compute

All PyMC + ArviZ (settled). The screen → confirm ladder follows the repo's
grid-then-NUTS convention (documents 05 §8, 09 §5 Gate C-1).

**Arm 1 — worthy rate, two grains.** Exact beta-binomial grid
(`research/_betabinom_grid.py`, unchanged) on worthy throws / attempts, once
with QB-season as the entity and once with defence-season. Statistic: 89%
upper bound on the population SD of the true entity rate. This is document
09's instrument verbatim. Descriptive: skill is *expected* here (document 32
§3); the number tells the diagnostic how much of a dropped-pick count is the
QB's throwing.

**Arm 2 — conversion by covariates.** PyMC hierarchical logistic:

```
y_i ~ Bernoulli(p_i),  logit p_i = α + X_i β + u_d[i] + v_q[i]
α ~ Normal(0, 1.5)           β_k ~ Normal(0, 1)   (covariates standardised)
u_d ~ Normal(0, σ_d)         v_q ~ Normal(0, σ_q)
σ_d, σ_q ~ HalfNormal(0.5)   non-centred
```

Sampled with nutpie (present in the environment), 4 chains × 1,000 draws
after 1,000 tuning. One fit, ~1 minute. Its job is (a) to estimate `β` for
`p̂_i = logit⁻¹(α̂ + X_i β̂)` (posterior-mean fixed effects only) and (b) to
cross-check arm 3. Arm 2b repeats it without `is_catchable_ball` and
`is_contested_ball`.

**Arm 3 — persistence of the conditioned residual (the gate arm).** Per-throw
residual `r_i = y_i − p̂_i`, fitted with the crossed Gaussian grid
(`research/_crossed_gaussian_grid.py`, `fit`, unchanged) with factors
defence-season and QB-season; a second run with defence pooled across seasons
and QB-season. Statistic: **the 89% upper bound on `σ_d`, expressed on the
probability scale in percentage points.** This is the gate statistic and the
power instrument — the same object, per document 09's rule.

Cross-check (Gate C-1 below): arm 2's `σ_d` posterior, mapped to the
probability scale by `σ_p ≈ σ_logit × p̄(1 − p̄)` at `p̄ = 0.485`, agrees with
arm 3's within **1.0 pp** on the 89% upper bound. If it does not, both numbers
are reported and the disagreement is a finding of its own.

**Secondary, for continuity with document 32:** the odd/even-week split-half r
of raw conversion (should reproduce ≈ +0.140) and of the conditioned residual,
plus season-to-season correlation of shrunk defence effects (32 teams × 3
adjacent pairs). Reported, never gated.

**Compute.** Power (§6): 400 simulated datasets × 4 non-null scenarios + the
null, for five entity designs (QB-season worthy, defence-season worthy,
defence-season residual, defence-pooled residual, QB-season residual), all
grid fits — under ten minutes on the laptop. Confirmatory: two nutpie fits,
~2 minutes. Total to a decision: **under an hour of compute.** No cluster, no
parallel arms, no isolation mechanism needed.

**Long-fit downtime plan.** Nothing runs long enough to need one; stated so
per the template.

## 6. Power, before thresholds

Runs first, per the process law documents 04 → 05 §7 → 09 §4 established.
Script: `research/61_dropped_pick_power.py`; results
`research/outputs/61_dropped_pick_power.json`; the table below is filled
from it and **committed before `research/62_dropped_pick_confounds.py`
fits anything real.**

Instrument, per entity design: simulate at the **real denominators** (the
observed chances per entity) under a known true population SD, fit the same
grid the gate uses, record the 89% upper bound. For the residual designs the
simulation is `y_i ~ Bernoulli(logit⁻¹(logit p̂_i + u_d))`, `u_d ~ N(0, τ)`,
with `β` held at arm 2's posterior mean (disclosed: this ignores `β`
uncertainty, so power is very slightly optimistic), residual recomputed
against the same fixed `p̂`, then the crossed grid fitted.

Scenarios: relative population SD of **5%, 12.5%, 25%, 50%** of the league
rate, 12.5% the reference (document 04's pooled-judgment-penalty yardstick,
used by 05 §7 and 09 §4). For conversion, 12.5% of 48.5% is **6.1 pp**;
`τ` on the logit scale is `SD_p / 0.2498`. For worthy rate, 12.5% of 3.7% is
0.46 pp.

Threshold = the **90th percentile of the null (τ = 0) upper-bound
distribution**, so a skill-free league clears it 90% of the time. Power =
fraction of scenario datasets whose upper bound exceeds the threshold.

| Entity design | Null bound (90th pct) = threshold | Power at 5% | **at 12.5%** | at 25% | at 50% | Resolvable? |
|---|---|---|---|---|---|---|
| Worthy rate, QB-season (≥ 200 att) | **0.551 pp** | 0.177 | **0.780** | 1.000 | 1.000 | **No** |
| Worthy rate, defence-season | **0.522 pp** | 0.230 | **0.935** | 1.000 | 1.000 | **Yes** |
| Residual, defence-season × QB-season | **9.410 pp** | 0.130 | **0.362** | 0.953 | 1.000 | **No** |
| Residual, defence pooled × QB-season | **8.086 pp** | 0.142 | **0.555** | 0.990 | 1.000 | **No** |
| Residual, QB-season (σ_q) | **8.075 pp** | 0.193 | **0.578** | 0.988 | 1.000 | **No** |

`RANDOM_SEED = 20260827`, `DATASETS = 400`, `MIN_POWER = 0.80`.

Filled 2026-08-27 from `research/61_dropped_pick_power.py` →
`research/outputs/61_dropped_pick_power.json`, **before**
`research/62_dropped_pick_confounds.py` existed as a fit. Nothing in §7 moved.

**The guards (constraint 8), as measured.** Charted passes 80,785 — document
32's figure exactly, on `play_type == "pass"`. Interception-worthy **2,997**
(0.00% off the expected count). Intercepted 1,454, so `p̄ = 0.4852` (0.02 pp
off 0.485). **128** defence-seasons. **28** worthy throws dropped for a null
covariate, and the 28 are one nested set: all 28 are missing `pass_location`,
27 of those also `air_yards`, 16 of those also `down`. They are the throws
nflverse could not place. That leaves **2,969** throws in arm 2's frame. All
guards ok; nothing needed asking.

**What the residual designs actually see.** The residual question's QB-season
unit is ≥ 20 worthy throws (§4), and a crossed design gives every row a level
on both factors, so throws by a QB-season below the floor have no level to
belong to and leave with it. The gate arm's frame is therefore **1,145 throws,
125 defence-seasons (32 pooled defences), 46 QB-seasons**, at a **median of 7
chances per defence-season** against the 22 document 32 §3 reported on the
unfiltered count. That is the whole reason the two residual rows above read
0.362 and 0.555 rather than something usable: the floor buys conditioning at
the price of two thirds of the sample, and it was pre-registered before anyone
knew which way that trade would land.

**One row deserves a sentence.** Worthy rate at the QB-season grain lands at
**0.780**, two hundredths under the 0.80 bar. Gate D-1 carries no pass rule, so
nothing turns on it — but the number is reported rather than rounded up, and it
means the QB-side spread is stated with an interval and a power figure that is
honestly just short of the project's own reference.

**Compute, against the §5 estimate.** §5 predicted "under ten minutes" for
power. The two crossed designs cost ~0.65 s and ~0.22 s per fit, so the 14
residual cells are ~5,600 grid fits: **565 s of wall clock across nine worker
processes**, about 40 minutes of CPU. The estimate was optimistic about the
crossed grid, not about the design; every dataset is seeded from its own index,
so serial and parallel runs return the same numbers.

## 7. Pre-registered gates

Committed before any result exists.

**Gate C-1 — sampler and instrument health.** Arm 2: zero divergences,
`r_hat < 1.01`, `ess_bulk > 400`, `ess_tail > 400`. Arms 1 and 3: the grid's
`self_check` passes. Cross-check: arm 2 and arm 3 agree on the 89% upper
bound of `σ_d` within 1.0 pp on the probability scale.

**Gate C-2 — is the conditioned residual spread negligible?** (one per
residual design). Statistic: 89% upper bound on `σ_d` in pp. **Pass:** below
that design's §6 threshold. Passing means the finish, conditioned on the
throw, spreads no more across defences than a coin-flip league would.
Failing means defences genuinely differ in finishing.

**Gate C-3 — is the result interpretable?** Power at 12.5% relative
≥ **0.80** for that design. If C-3 fails for a design, neither outcome of
C-2 on that design may be reported as a finding — only the power table.

**Gate D-1 — the worthy-rate spreads** (arm 1) are reported with their 89%
intervals and their C-3 power; no pass rule, because skill is the expected
answer and the number's use is descriptive.

**Decision rule.** The reading in §0's table, applied per residual design in
this order: defence-season × QB-season first; defence-pooled only if the
first fails C-3. If both fail C-3 the study is **unresolvable**, and says so.

## 8. Kill and rollback

Nothing ships into the simulator on any outcome, so there is no flag and no
version bump. On every outcome the deliverables are document 44 (results
record), the filled power table above, the results file, and one sentence of
diagnostic wording for the game page (avenue (1)) that the outcome licenses.
Arm 2's fitted `β` is kept in `research/outputs/` as the artifact avenue (1)
would use for an expected-conversion figure — it is **not** a bin table of
the A-2 clause 3 kind and must not be described as one.

## 9. Defect register (this study)

| Defect | Evidence | Status |
|---|---|---|
| Defensive conversion persistence unconditioned and unpowered | Document 32 §3: r = +0.140, 22 chances/team-season, no power | **Open — this document** |
| Charter hindsight in `is_catchable_ball` / `is_contested_ball` | Named risk, unmeasured | Arm 2b sensitivity |
| `β` held fixed in the residual power simulation | §6 | Disclosed; optimistic by a small, unquantified amount |

## 10. Constants

| Constant | Value | Where |
|---|---|---|
| `RANDOM_SEED` | 20260827 | `research/61_dropped_pick_power.py`, `research/62_dropped_pick_confounds.py` |
| `DATASETS` | 400 | `61` |
| `RELATIVE_SCENARIOS` | (0.05, 0.125, 0.25, 0.50) | `61` |
| `MIN_POWER` | 0.80 | this document §7 |
| `MIN_QB_ATTEMPTS` (worthy rate) | 200 | document 32 §3 |
| `MIN_QB_WORTHY` (residual) | 20 | this document §4 |
| League conversion `p̄` | 0.485 | document 32 §3; re-read from data, stop if it differs by > 1 pp |
| Cross-check tolerance | 1.0 pp | this document §5 |
| Expected worthy throws | 2,997 (stop if > 5% off) | document 32 §3 |
