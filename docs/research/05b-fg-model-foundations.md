# 05b — Field-goal make model: foundations

*Written 2026-08-17, **before the model was fit**. Power checks:
`research/07_fg_power.py`, results in `research/outputs/07_fg_power.json`.
Committed before `research/07_fg_model.py` produces any estimate.*

*Numbered 05b rather than 07 because document 07 is reserved for the rematch
validation results pre-registered in document 06. This is a component of the
neutralization principle in document 05, not a separate line of inquiry.*

*Stack verified at time of writing: PyMC 6.3.1, PyTensor 3.3.0, ArviZ 1.3.0,
nutpie 0.16.11.*

---

## 1. One-page story

### The question

Document 05's rule needs one number per luck event: `p(e)`, the probability of
the favourable branch **at the responsible entity's shrunk rate**. For fumbles
that number is a league class rate, because `w = 0.011` erases the entity. For
field goals it cannot be, because Phase 1 found real kicker skill — document 02
measured split-half `r = 0.145` on the `fg_luck` component and named the
mechanism outright: *"that is not luck; that is Justin Tucker."*

So the field goal component needs an actual model: **what was the probability
that this kicker made this kick from this distance?** Charging a kicker with bad
luck for missing a 55-yarder at the league-average rate would misprice both the
good kickers and the bad ones.

### How it answers, in one paragraph

A hierarchical logistic regression on 10,731 field-goal attempts, 2016–2025.
Make probability falls with distance on the log-odds scale, and each
kicker-season gets a random intercept drawn from a common Normal. The
population SD of those intercepts, `sigma_kicker`, is the parameter that decides
how much of a kicker's observed record is real — it is the FG component's
version of the `kappa` that document 04 estimated for every other component, and
it feeds the same `w = n/(n+kappa)` dial from document 05 §1.

### Five things to hold onto

1. `sigma_kicker` is this model's `kappa`. It sets how far a kicker's own record
   is trusted, and therefore how much luck the simulator books on each kick.
2. The reported quantity is a **make-rate gap at 45 yards**, not `sigma` itself,
   because a log-odds SD is not readable as football.
3. Distance enters **linearly on the log-odds scale**. That is a modelling
   choice with a documented fallback (§5), not a law of kicking.
4. **Weather is deferred**, per the handoff plan. This is the model's largest
   known defect and it biases in a stated direction (§7).
5. The grain is the **kicker-season**, not the kicker's career. A kicker's leg
   changes; a season is the unit the simulator adjudicates within.

### Statistic convention

Posterior means with 89% equal-tailed intervals, matching documents 03 and 05.

---

## 2. Data

- **Grain of a row**: one field-goal attempt.
- **Source**: `data/pbp/*.parquet`, 2016–2025.
- **Filter**: `play_type == "field_goal"` with a non-null `kick_distance` and a
  non-null `kicker_player_id`.
- **Population**: **10,731 attempts**, league make rate **84.66%**, **422
  kicker-seasons**, median **29 attempts** per kicker-season, distances 18–70
  yards.

### Facts that must be defensible by name

- **Blocked kicks are excluded.** *(Revised 2026-08-18 by document 27, approved
  by the maintainer and shipped in v1.3. The original bullet counted them as misses, on
  the grounds that `components.py` did; document 26 §2 ruled that
  `components.py` should not, and the consistency argument lost its referent.)*
  A block is not an observation of the quantity this model estimates — the ball
  never flew. 192 field goals and 110 extra points leave the training
  population, `p_make` rises 1.33 pp on average and by more the longer the kick,
  and `sigma_kicker` **rises** from 0.342 to 0.385, because blocks were noise
  diluting the measured spread between kickers. The refit clears every gate in
  §6 and §10 on the same cubic curve (document 27 §14).
- **Extra points are excluded.** They are a fixed-distance formality at a make
  rate near 94% and would dominate the sample while telling us nothing about the
  distance curve.
- **The grain is kicker-season, so a kicker appears up to ten times.** Those rows
  are treated as exchangeable draws from one population. A kicker whose true
  ability is stable across a decade therefore contributes ten independent-looking
  observations of the same thing, which slightly inflates the apparent population
  spread. Recorded in §7.
- **Distance is centred at 40 yards** so the intercept means "log-odds of a
  40-yarder" rather than of a 0-yard kick, which does not exist.

---

## 3. DAG

```
      alpha (log-odds at 40 yd)     beta (slope per yard)    sigma_kicker
              \                          |                      |
               \                         |                      v
                \                        |            kicker[k] ~ Normal(0, sigma_kicker)
                 \                       |                      |
                  v                      v                      v
                logit p(make) = alpha + beta * (distance - 40) + kicker[k]
                                         |
                                         v
                              made ~ Bernoulli(p)
```

**Where inference is cut.** Nowhere inside the model. But there *is* a cut
downstream, and it is the one that matters for the simulator: the posterior for
`(alpha, beta, sigma_kicker, kicker[·])` is handed to the simulator, which draws
from it rather than re-fitting. That is a genuine joint handoff, not a
point-estimate cut — document 05 §4's bootstrap layer 1 exists precisely so this
uncertainty survives into the DTW interval.

**Emergent behavior to watch.** A kicker-season with 29 attempts, most of them
inside 45 yards, carries very little information about the long-range part of
their curve. The model has no per-kicker slope, so it assumes every kicker's
curve has the same shape and differs only in level. If long-range ability is a
distinct skill, this model cannot see it, and it will attribute a specialist's
long-range success to the league slope plus a mildly elevated intercept.

---

## 4. Priors, site by site

| Site | Prior | Plain-language meaning |
|---|---|---|
| `alpha` | `Normal(2, 1.5)` | Log-odds of a 40-yard make. Centred near 88%, with 89% mass spanning roughly 45% to 99% |
| `beta` | `Normal(0, 0.2)` | Log-odds change per yard. Weakly negative-expected but not forced; 0.2 comfortably spans the plausible −0.05 to −0.15 |
| `sigma_kicker` | `HalfNormal(1)` | SD of kicker intercepts on the log-odds scale. 1.0 is generous — it allows a one-SD kicker to be ~13 points better at 45 yards, far more than anyone believes |
| `kicker[k]` | `Normal(0, sigma_kicker)`, non-centered | Each kicker-season's level relative to the league |
| `made` | `Bernoulli(p)` | What we observed |

### Why `beta` is not left flat

A flat prior on the slope lets the sampler explore curves where a 60-yarder is
more likely than a 20-yarder. Those are ruled out by the sport, not by the data,
and letting the sampler waste time on them is how a two-minute fit becomes a
twenty-minute one. `Normal(0, 0.2)` is agnostic about the value while being
informative about the scale.

### The identification story

`alpha` and `beta` are pinned by the pooled make rate as a function of distance —
10,731 attempts across a 52-yard range, so both are very well determined.
`sigma_kicker` is pinned by how much kicker-level make rates scatter *relative to
the binomial scatter their own attempt counts and distances imply*. That second
quantity is the entire signal, and §6 is where its strength gets measured rather
than assumed.

### Reported quantity

```
make_rate_gap_45 = sigmoid(logit(p_45) + sigma_kicker) − p_45
```

The make-rate difference at 45 yards between a one-SD-good kicker and an average
one. This is `sigma_kicker` restated in points, which is the only form in which
anyone can argue with it.

---

## 5. Inference plan

- **Engine**: NUTS via nutpie, auto-selected by PyMC 6. Smooth, continuous,
  moderate dimension (426 parameters) — exactly NUTS's case.
- **Configuration**: 4 chains, 1,000 tune, 1,000 draws, `target_accept = 0.9`.
- **Parameterization**: **non-centered on `kicker[k]`. This is a ruling, not a
  default.** Document 04's Gate 1 failure was a centered hierarchy funnelling,
  and step 3a's crossed model needed 3,000 draws for the same family of reason.
  With a median of 29 attempts per kicker-season and `sigma_kicker` expected to
  be small, the centered form is the known-bad geometry here.
- **Documented fallback if Gate FG-2 fails**: add a quadratic term in centred
  distance, refit, and report both. **Not** to widen the intervals until the
  gate passes. The quadratic is named now so that reaching for it later is
  execution rather than improvisation.
- **Compute cost**: one model, ~10,700 observations, 426 parameters. Expect
  under two minutes on this laptop. No cheaper screen is warranted — the confirm
  run *is* cheap.
- **Downtime plan**: nothing runs in parallel. The fit is short enough that
  arranging concurrency would cost more than it saves.

---

## 6. Pre-registered gates

Committed before any result exists. **Every threshold below carries a power
number**, per the process law from document 04.

### Gate FG-1 — sampler health

**Pass rule:** zero divergences, `r_hat < 1.01`, `ess_bulk > 400` and
`ess_tail > 400` on every parameter.

**On failure:** if divergences appear, the geometry is wrong and the fallback is
a reparameterization. If ESS is low with *zero* divergences — the signature step
3a hit — the fix is more draws, because that is slow mixing rather than bad
geometry. Raising `target_accept` to quiet a warning is forbidden by document 03
§5 and remains forbidden here.

*No power check: this is a diagnostic on the sampler, not an inference about
football.*

### Gate FG-2 — distance calibration

This is the gate that matters most for the simulator. A model that is right on
average and wrong at 55 yards will systematically mis-book luck on exactly the
kicks where the swing is largest.

**Statistic:** the **largest standardized miss** across 5-yard distance bins
holding at least 100 attempts, compared against its own posterior predictive
distribution.

**Pass rule:** the observed statistic is at or below the 94.5th percentile of
that reference distribution.

**Why a maximum rather than every bin.** The first version of this gate required
every bin to sit inside its own 89% interval. Its power check said it fails
**36% of the time on a correctly specified model** — that is multiplicity, since
eight bins at nominal 89% coverage pass together only 0.89⁸ ≈ 39% of the time.
Reducing to a single maximum and calibrating that maximum against its own
reference prices the multiplicity in exactly once. **The gate was fixed because
the power check caught it, before it was committed.**

**Power check:**

| Truth | Gate passes | Gate catches |
|---|---|---|
| Well-specified (linear is correct) | 0.960 | **0.040** *(false-alarm rate; nominal 0.055)* |
| Curve misses by 5 points at 55 yd | 0.680 | 0.320 |
| **Curve misses by 10 points at 55 yd** | 0.040 | **0.960** |
| Curve misses by 15 points at 55 yd | 0.000 | 1.000 |

The gate is calibrated and detects a 10-point misspecification almost always.
**It is explicitly weak against a 5-point miss (32%)**, and that limitation is
recorded here rather than discovered later: passing FG-2 rules out a large
distance misspecification, not a subtle one.

### Gate FG-3 — is kicker skill resolvable at all?

**Incumbent:** document 02's split-half `r = 0.145` on `fg_luck`, which says
kicker skill exists but does not size it.

**Statistic:** the 89% interval for `sigma_kicker`.

**Pass rule:** the 89% upper bound exceeds **0.2407** — the 90th percentile of
what this design produces when the truth is *exactly zero*.

Note the direction. This gate passing means "there is more kicker spread here
than a skill-free league would produce", which licenses the partial
neutralization document 05 §3 assigned to field goals. If it **fails**, the
honest reading is that 10,731 kicks cannot distinguish kicker skill from
binomial noise, and the FG row of document 05 §3 collapses to **full**
neutralization at the league distance curve — the same treatment fumbles get.
Either outcome is usable; that is what makes it worth pre-registering.

**Power check** (normal-normal reduction; see §7 for the approximation's status):

| True `sigma` | Make-rate gap at 45 yd | Mean 89% upper bound | **Power** |
|---|---|---|---|
| 0 (null) | 0 pp | 0.190 | *threshold set at 90th pct = 0.2407* |
| 0.10 | 1.5 pp | 0.210 | 0.225 |
| 0.20 | 2.9 pp | 0.279 | **0.767** |
| **0.30** | **4.2 pp** | 0.368 | **0.998** |
| 0.40 | 5.4 pp | 0.466 | 1.000 |

The public kicking literature puts real kicker spread near a 4–5 point gap, and
the design resolves that essentially always. It **cannot** resolve a 1.5-point
gap, so a failure of FG-3 means "no *large* kicker effect", never "no kicker
effect".

### Gate FG-4 — posterior predictive on the observable

**Pass rule:** the observed league make rate and the observed between-kicker
variance of make rates both fall within the central 89% of their posterior
predictive distributions.

The variance half is the one that matters, for the same reason document 03 §6
Gate 4 gave: a model that gets the mean right and the spread wrong is precisely
the model that would mislead about skill.

---

## 7. Known-defect register

| Defect | Evidence | Status |
|---|---|---|
| **Weather is absent** | Wind and temperature are in the data (`wind`, `temp`, `roof`) and deliberately unused, per the handoff | **Open, and the largest defect.** A windy 50-yarder is priced as a calm one, so the simulator overstates the kicker's bad luck outdoors in December and understates it in a dome. Direction is stated wherever the model is used |
| No per-kicker slope | The model assumes every kicker's curve has one shape | **Open.** Long-range specialists are absorbed into the intercept (§3) |
| Kicker-season rows are not independent across seasons | One kicker can supply ten rows of the same underlying ability | **Open.** Slightly inflates apparent population spread |
| FG-3's power check uses a normal-normal reduction | The full logistic hierarchy is too slow for hundreds of power fits | **Accepted, stated.** The reduction ignores the shrinkage the real hierarchy applies, so it *overstates* resolving power — the true power is at or below the table in §6 |
| FG-2 is weak against subtle misspecification | 32% power against a 5-point miss at 55 yd | **Accepted, stated in §6** |
| Long-distance bins are thin | 65+ yards holds 19 attempts across ten seasons (document 01) | **Open.** The model extrapolates there rather than borrowing a neighbour bin as `components.py` does; those kicks are rare enough not to move a game total |
| The game being adjudicated is inside the fit | Document 05 §5, the general contamination defect | **Open, bounded** at O(1/n); a kicker-season contributes ~2 of ~29 attempts |
| ~~Blocked kicks charged to the kicker~~ | 192 of 10,731 attempts | **CLOSED 2026-08-18** by the document 27 refit, shipped in v1.3. They are out of the training population and out of the ledger |
| **`w = 0.285` has no derivation in this repository** | Document 27 §14c could only recompute the refit's implied 0.336 under a stated assumption | **Open.** Any document quoting `w` for the field-goal row is quoting a number nothing regenerates |

---

## 8. Constants appendix

| Constant | Value | Where it lives |
|---|---|---|
| `RANDOM_SEED` | 20260817 | `research/07_fg_model.py` |
| chains / tune / draws | 4 / 1000 / 1000 | `research/07_fg_model.py` |
| `target_accept` | 0.9 | `research/07_fg_model.py` |
| `DISTANCE_CENTRE` | 40 yards | `research/07_fg_model.py`, `research/07_fg_power.py` |
| `alpha` prior | `Normal(2, 1.5)` | this document §4 |
| `beta` prior | `Normal(0, 0.2)` | this document §4 |
| `sigma_kicker` prior | `HalfNormal(1)` | this document §4 |
| **Gate FG-3 threshold** | **0.2407** | this document §6, from the null simulation |
| Gate FG-2 percentile | 94.5th | this document §6 |
| Minimum bin attempts | 100 | `research/07_fg_power.py` (`MIN_BIN_ATTEMPTS`) |
| Power datasets | 400 | `research/07_fg_power.py` (`DATASETS`) |
| Attempts / kicker-seasons | 10,731 / 422 | measured, §2 |

Results are written back into this document as §9.

---

## 9. Results

*Script: `research/07_fg_model.py`. Gates pre-registered in §6, committed at
`7e9e6d1` before any fit existed. Results in
`research/outputs/07_fg_model.json`.*

### Gate outcomes, stated first

| Gate | Linear (pre-registered) | Quadratic (fallback) |
|---|---|---|
| FG-1 sampler health | **PASS** | **PASS** |
| FG-2 distance calibration | **FAIL** — 2.828 vs 2.777 | **PASS** — 2.716 vs 2.755 |
| FG-3 kicker skill resolvable | **PASS** | **PASS** |
| FG-4 posterior predictive | **PASS** | **PASS** |

**The pre-registered linear form failed Gate FG-2, and the documented fallback
was applied exactly as named in §5** — add a quadratic term in centred distance,
refit, report both. The linear arm is retained in the record rather than
deleted; hiding a failed pre-registered arm would defeat the point of naming a
fallback in advance.

### Why the linear form failed, and it was not bad luck

The failure was narrow — 2.828 against a 2.777 threshold — but the bin table
shows it was structural rather than a marginal miss:

| Distance bin | Attempts | Observed | Linear predicted | Miss | Quadratic predicted | Miss |
|---|---|---|---|---|---|---|
| 20–24 | 1,024 | 98.8% | 97.8% | +1.0 pp | 98.6% | +0.2 pp |
| 25–29 | 1,400 | 96.9% | 96.5% | +0.5 pp | 97.2% | −0.2 pp |
| 30–34 | 1,497 | 95.1% | 94.2% | +0.9 pp | 94.5% | +0.6 pp |
| 35–39 | 1,563 | 89.8% | 90.6% | −0.7 pp | 90.2% | −0.3 pp |
| 40–44 | 1,569 | 83.2% | 85.1% | −1.9 pp | 83.9% | −0.7 pp |
| 45–49 | 1,599 | 74.2% | 77.1% | **−3.0 pp** | 75.8% | −1.7 pp |
| 50–54 | 1,392 | 71.0% | 67.6% | **+3.4 pp** | 67.6% | **+3.4 pp** |
| 55–59 | 540 | 60.0% | 57.5% | +2.5 pp | 60.3% | −0.3 pp |

The linear misses are **signed in a pattern**: over-predicting through the 40s,
under-predicting from 50 out. That is a curve the straight log-odds line cannot
bend to, and it is the exact failure mode Gate FG-2 was built to catch. The
quadratic fixes six of the eight bins.

**The 50–54 bin does not improve, and that is worth saying plainly.** It still
sits 3.4 points above prediction under both models, and it is now the single
worst bin by a wide margin — the gate passes on the strength of the other seven.
A plausible reading is selection: a coach who sends the offense out on 4th down
rather than attempting a 52-yarder does so more often in bad conditions, so the
52-yarders that *are* attempted are a favourable sample. That is a decision
process the model does not represent, and it is added to the defect register.

### The fitted curve

| Parameter | Mean | 89% interval |
|---|---|---|
| `alpha` (log-odds at 40 yd) | 1.898 | 1.83 – 1.97 |
| `beta` (per yard) | −0.1148 | −0.122 – −0.108 |
| `gamma` (quadratic / 100) | 0.130 | — |
| `sigma_kicker` | **0.360** | **0.273 – 0.442** |

| Distance | League make rate |
|---|---|
| 30 yd | 96.0% |
| 40 yd | 87.0% |
| 45 yd | 79.5% |
| 50 yd | 70.7% |
| 55 yd | 61.5% |

### Gate FG-3 — kicker skill is real and about the size the literature says

> `sigma_kicker` = **0.360**, 89% interval 0.273 – 0.442, against a
> pre-registered threshold of 0.2407. The interval's *lower* bound clears it.

In readable terms: a one-SD-good kicker makes **5.35 pp more** of their 45-yard
attempts than an average one (89% interval 4.17 – 6.45), against a league rate
of 79.5% there. That lands almost exactly on the `sigma = 0.30` row of §6's power
table, where power was 0.998 — so this is a result the design was built to
resolve, not one scraped off the edge of it.

**This is what licenses the partial treatment.** Document 05 §3 assigned field
goals partial neutralization on the strength of Phase 1's split-half `r = 0.145`;
this sizes it. The resulting shrinkage weight is **`w` = 0.285** at the median
kicker-season, ranging from 0.064 to 0.377 across the 10th–90th percentiles.

### Shrinkage in practice

| Kicker-season | Attempts | Observed | Shrunk effect |
|---|---|---|---|
| Best in sample | 39 | 97.4% | +0.507 log-odds |
| 26-for-26 season | 26 | 100% | +0.435 |
| Worst in sample | 15 | 60.0% | −0.492 |
| Worst full season | 31 | 70.9% | −0.489 |

A perfect 26-for-26 season is shrunk to a smaller effect than a 38-for-39
season, because 39 attempts is more evidence than 26 — the same partial-pooling
behaviour document 04 showed for fumbles, at a `w` an order of magnitude larger.

### Defects added by this round

| Defect | Evidence | Status |
|---|---|---|
| The 50–54 yard bin misses by +3.4 pp under both models | §9 bin table; unimproved by the quadratic | **Open.** Likely attempt-selection (coaches decline the kick in bad conditions), which the model does not represent |
| The adopted model is not the pre-registered one | Gate FG-2 failed on the linear form | **Closed** by the §5 fallback, which named the quadratic in advance. Both arms are reported |
| Gate FG-2 passed narrowly | 2.716 against 2.755 | **Open.** The quadratic is adequate, not comfortably right; a spline or a monotone fit is the Phase 3 option |
| `gamma` has no mechanism story | Added to fix a calibration failure, not from a theory of kicking | **Open.** It is a curvature correction, and should not be interpreted as anything more |

---

## 10. Change proposal — weather, and extra points

*Written 2026-08-17, **before `research/14_fg_weather_model.py` existed**. Power
calculation: `research/13_fg_weather_power.py`, results in
`research/outputs/13_fg_weather_power.json`. Follows the change-proposal
template: tier, DAG edit, mechanism, cost, gate, downtime, rollback.*

### 1. Tier declaration

**Model change.** Three new covariates and two new structural parameters enter
the linear predictor, and the observation set grows by 12,818 extra points. The
generative story changes, so every section of the template is filled.

### 2. DAG edit

The incumbent (§3) with the additions marked `NEW`:

```
   alpha      beta     gamma          sigma_kicker
     |          |        |                  |
     |          |        |         kicker[k] ~ Normal(0, sigma_kicker)
     |          |        |                  |
     |          |        |        +---------+---------+
     |          |        |        |                   |
     v          v        v        v                   v  (x lambda_xp)  NEW
   logit p = alpha + beta*c + gamma*c^2/100 + kicker[k] * (1 or lambda_xp)
             + roof[level]                                    NEW
             + beta_wind * (wind - 8.118) * has_weather       NEW
             + beta_temp * (temp - 58.156) * has_weather      NEW
             + delta_xp * is_extra_point                      NEW
                              |
                              v
                    made ~ Bernoulli(p)
```

**Nodes added:** `roof[dome|closed|open]` with outdoors as the reference,
`beta_wind`, `beta_temp`, `delta_xp`, `lambda_xp`.
**Arrows added:** stadium conditions and kick type now feed the make
probability. **Nothing is removed.**

**Where inference is cut.** Unchanged — the joint posterior is handed to the
simulator, which draws from it. The one new cut is that the **centring
constants** (8.118 mph, 58.156 °F) are properties of the fitted sample and are
passed to `FieldGoalModel` rather than stored in the trace, so a caller scoring
a future season must reuse the fit's own values.

**Emergent behaviour to watch.** `lambda_xp` scales the kicker effect on extra
points. At `lambda_xp = 1` a kicker's field-goal ability transfers one-for-one;
at 0 the two are unrelated. Sharing one effect *without* this scale would
**assert** perfect transfer, and asserting is what this parameter exists to
avoid. Per-attempt Fisher information is `p(1−p)`, so an extra point at 94.4%
carries 0.053 against a field goal's 0.128 — the 12,818 extra points contribute
roughly half what the 10,731 field goals do, rather than swamping them.

### 3. Mechanism story

**The defect this addresses is named in §7 as the model's largest**: *"Weather is
absent… a windy 50-yarder is priced as a calm one, so the simulator overstates
the kicker's bad luck outdoors in December and understates it in a dome."*
Document 05 §5 carries the same row.

**Why this change should move that number.** The simulator books
`(made − p) × swing` as luck. If `p` is the calm-day probability and the kick was
into a 20 mph wind, the miss is charged to the kicker as bad luck when it was
partly the conditions — an error whose *sign is systematic*, not noise: it runs
one way for every outdoor cold-weather team and the other way for every dome
team, across all ten seasons. Distance-adjusted, the raw data already shows
indoor kicking running about 2 pp above the league curve and 10–14 mph winds
about 4 pp below it.

**Extra points enter for a different reason**: document 09 §8 measured a 2.422 pp
population SD in kicker extra-point rates against a 1.840 pp null bound, so
kickers genuinely differ there, and document 09 §2 gave extra points a branch
point. They are a neutralizable component with an entity, and the entity is the
kicker already in this model.

**What would make it fail.** If `beta_wind`'s interval does not clear the null
bound, ten seasons cannot size a wind effect and the defect stays open with a
sharper statement than before. If the weather-cell calibration gate fails, a
*linear* wind term is the wrong shape — wind direction is not in the data at
all, and a 20 mph crosswind and a 20 mph tailwind are recorded identically,
which would show up as a poor fit in the high-wind cells.

### 4. Compute cost and inference plan

- **Arms:** one new fit. The control is the published Phase 2 posterior already
  on disk (`research/outputs/trace_fg_model.nc`), so no refit is needed for it.
- **Observations:** 23,549 (10,731 field goals + 12,818 extra points).
  **Parameters:** ~430.
- **Engine:** NUTS via nutpie, as §5. Same geometry, slightly larger.
- **Configuration:** 4 chains, 1,000 tune, 1,000 draws, `target_accept = 0.9` —
  unchanged, so a difference in diagnostics is attributable to the model rather
  than to the sampler settings.
- **Parameterization:** non-centered on `kicker[k]`, still a **ruling** for the
  reason §5 gave.
- **Wall clock:** expect under five minutes. **No cheaper screen is warranted —
  the confirm run is cheap**, and the power calculation (which is the expensive
  part at 2,000 logistic fits) has already run.
- **Efficiency levers considered:** the power calculation uses a plain logistic
  fit rather than the hierarchy, which is what made 2,000 fits affordable; its
  bias direction is stated in §6 below.

### 5. Sanitize rules — fixed before the fit

These are data-quality guards, not modelling choices, and they live in
`src/nfl_simulator/fg_model.sanitize_weather` so the fit and the simulator share
**one** implementation. A model trained on one definition of a windy day and
applied to another is a defect no gate would catch.

| Rule | Rows affected | Why |
|---|---|---|
| **Indoors, weather is nulled** | 4 of 3,200 dome/closed attempts | nflverse leaves temp and wind null on 3,196 of them; the four exceptions carry a stadium-ambient reading (46 °F, 2 mph) that describes the air *outside* the stadium. A wrong reading is worse than a missing one, because it is silently used |
| **Wind capped at 30 mph** | 17 of 7,327 outdoor attempts | The raw maximum is **71 mph**, on three attempts in one 2016 game. Bins above 25 mph hold 31 attempts in ten seasons — uncapped, three kicks would lever the league's wind coefficient |
| **Missing outdoor weather is not imputed** | 716 attempts (512 outdoors + all 204 "open") | Centred to the outdoor mean, which contributes exactly zero. The honest meaning is *no information about this kick's conditions, so use the outdoor baseline* — not *it was a calm 58-degree day* |
| **Both readings required together** | — | temp and wind are null together in every row of the source, so an all-or-nothing rule costs nothing and avoids centring one term while guessing the other |

This mirrors the `sanitize_temp` reasoning in the sibling baseball repo.

### 6. Priors, and the power behind the gate

| Site | Prior | Plain-language meaning |
|---|---|---|
| `alpha`, `beta`, `gamma`, `sigma_kicker` | **unchanged** from §4 | The incumbent's curve is not being re-argued |
| `roof[level]` | `Normal(0, 0.5)` | Level shift for a dome, a closed roof or an open retractable. 0.5 log-odds is about ±6 pp at an 85% base rate — generous |
| `beta_wind` | `Normal(0, 0.05)` | Log-odds per mph. At 0.05 a 15 mph wind moves 0.75 log-odds ≈ 9 pp, more than twice the effect the design can resolve |
| `beta_temp` | `Normal(0, 0.02)` | Log-odds per °F. A 40 °F swing moves 0.8 log-odds |
| `delta_xp` | `Normal(0, 1)` | How much easier or harder an extra point is than a field goal from the same distance |
| `lambda_xp` | `Normal(1, 0.5)` | How much of a kicker's field-goal ability transfers to extra points. Centred on full transfer; 0 and 2 are both within two SD |

**Power** (`research/13_fg_weather_power.py`, 400 datasets per scenario,
simulating from the published incumbent with `beta_wind` set to a known value
and refitting):

| True effect: make-rate drop at 45 yd, calm → 15 mph | `beta_wind` | **Power** |
|---|---|---|
| 1 pp | −0.00402 | 0.235 |
| 2 pp | −0.00791 | 0.388 |
| **4 pp** | **−0.01533** | **0.800** |
| 6 pp | −0.02236 | 0.948 |

> **Minimum detectable wind effect: a 4 pp make-rate drop at 45 yards between a
> calm day and a 15 mph wind.** The design cannot resolve a 2 pp effect, and that
> limitation is recorded here rather than discovered later.

The power fits use a plain logistic **without** kicker effects while simulating
data **with** them. The direction is the safe one: unmodelled kicker spread
inflates the residual, so the true power is at or **above** the table — the
opposite bias to Gate FG-3's, and stated for the same reason.

### 7. Pre-registered gates

**Gate W-1 — sampler health.** Zero divergences, `r_hat < 1.01`,
`ess_bulk > 400`, `ess_tail > 400`. Same rule and same fallbacks as Gate FG-1;
raising `target_accept` to quiet a warning remains forbidden.

**Gate W-2 — weather calibration.** The largest standardized miss across
**weather cells** — roof level crossed with 5 mph wind buckets, cells holding at
least 100 attempts — compared against its own posterior predictive
distribution. **Pass:** observed at or below the 94.5th percentile. Identical in
construction to Gate FG-2, which was itself fixed by a power check that caught
its multiplicity problem. **Documented fallback on failure:** replace the linear
wind term with 5 mph wind buckets and refit, reporting both arms. Named now so
reaching for it later is execution rather than improvisation.

**Gate W-3 — is the wind effect resolvable?** **Pass:** the 89% upper bound on
`beta_wind` is below **+0.00268**, the 10th percentile of what this design
produces when the truth is exactly zero. By construction a true-zero design
clears it 10% of the time — the same construction Gate FG-3 used. **A failure
means "no *large* wind effect", never "no wind effect"**, per the power table.

**Gate W-4 — the distance curve still works.** The Gate FG-2 statistic,
recomputed on distance bins. **Pass:** at or below the 94.5th percentile of its
own reference. Adding weather must not break what already passed.

**Gate W-5 — posterior predictive.** League make rate and between-kicker
variance of make rates both inside the central 89% of their posterior
predictive distributions, i.e. Gate FG-4 preserved.

**Gate W-6 — extra-point transfer. Reported, no pass rule.** `lambda_xp` and
`delta_xp` with 89% intervals. No threshold, because no prior estimate of either
exists and pre-registering one would be theatre — the convention document 03 §6
Gate 3 set. **Reporting rule, committed now:** a claim that extra-point ability
differs from field-goal ability requires `lambda_xp`'s 89% interval to **exclude
1**, stated as such.

**Gate W-7 — temperature. Reported, no pass rule.** `beta_temp` with its 89%
interval. **Reporting rule:** no claim about temperature unless its interval
clears a null bound built the same way Gate W-3's was.

**Gate W-8 — ledger impact. Reported, no pass rule.** How many ledger entries
moved, by how much, and in which direction by roof. Required by the Phase 3
plan's verification list, and it is the number that tells a reader whether any
of this mattered.

### 8. Long-fit downtime plan

Nothing runs in parallel. The fit is minutes, and the two expensive jobs of this
phase — the coin-flip power sweep and this one's power calculation — both
completed before this proposal was written. Stated explicitly, as required.

### 9. Kill and rollback

- **On Gate W-3 failure:** weather is **not adopted**. The Phase 2 posterior
  stays the simulator's input, `FieldGoalModel`'s weather parameters stay
  `None`, and §7's weather defect is restated with the power table attached. No
  code is reverted — the weather support is already merged and inert without a
  weather-bearing trace, which is exactly the default-off behaviour this section
  asks for.
- **On Gate W-2 failure:** apply the §7 bucketed-wind fallback, refit, report
  both arms.
- **On success:** `research/outputs/trace_fg_weather.nc` becomes the simulator's
  field-goal posterior, `model_metadata.json` records the change with a version
  bump to `simulator-v1.1`, and document 05 §3's treatment table gains the
  extra-point row. Downstream consumer: `research/09_simulator_demo.py`, which
  regenerates `dtw_games.parquet` and `dtw_ledger.parquet`.

### 10. Disclosure

Fixing the sanitize rules required looking at the raw weather distributions —
that is how the 71 mph reading and the 100%-null domes were found — and the same
pass exposed the distance-adjusted make rate by weather bucket. **The thresholds
above come from the null simulation and could not have been moved by those
numbers**, but the exposure is recorded here rather than left unsaid, as
document 08 §7 records the equivalent for the sequencing round.

---

## 11. Weather and extra points — results

*Script: `research/14_fg_weather_model.py`. Design, priors and gates fixed by §10
above, committed at `8e0b50e` before this script existed. Results in
`research/outputs/fg_weather_summary.json`.*

### Gate outcomes, stated first

| Gate | Quadratic (pre-registered) | Cubic (fallback) |
|---|---|---|
| **W-1** sampler health | **PASS** | **PASS** |
| **W-2** weather calibration | **PASS** — 1.970 vs 2.595 | **PASS** — 1.995 vs 2.739 |
| **W-3** wind resolvable | **PASS** | **PASS** |
| **W-4** distance calibration | **FAIL** — 2.863 vs 2.658 | **PASS** — 2.102 vs 2.674 |
| **W-5** posterior predictive | **PASS** | **PASS** |
| W-6 extra-point transfer | reported | reported |
| W-7 temperature | reported | reported |

**The pre-registered quadratic curve failed Gate W-4, and the fallback §9 named
in advance was applied** — *"a spline or a monotone fit is the Phase 3 option"* —
as a cubic term in centred distance. Both arms are kept in the record. **Adopted
arm: cubic.**

23,549 kicks (10,731 field goals, 12,818 extra points), 433 kicker-seasons.

### Why the quadratic failed W-4, and why it is not weather's fault

The temptation is to read this as "adding weather broke the distance curve". It
did not, and the per-bin table says so:

| Distance bin | Attempts | of which XP | Observed | Predicted | Miss | Standardized |
|---|---|---|---|---|---|---|
| 20–24 | 1,025 | 1 | 98.8% | 98.7% | +0.15 pp | 0.43 |
| 25–29 | 1,435 | 35 | 96.9% | 97.3% | −0.32 | 0.75 |
| 30–34 | 14,124 | 12,627 | 94.6% | 94.5% | +0.12 | 0.62 |
| 35–39 | 1,664 | 101 | 89.8% | 90.2% | −0.38 | 0.53 |
| 40–44 | 1,585 | 16 | 82.9% | 83.9% | −0.97 | 1.06 |
| 45–49 | 1,637 | 38 | 74.0% | 75.7% | −1.63 | 1.55 |
| **50–54** | **1,392** | **0** | **71.0%** | **67.4%** | **+3.57** | **2.86** |
| 55–59 | 540 | 0 | 60.0% | 60.3% | −0.33 | 0.16 |

The failure is **entirely the 50–54 yard bin**, which contains **zero extra
points** and no weather-driven reweighting. It is the same bin §9 already
flagged: *"The 50–54 bin does not improve, and that is worth saying plainly. It
still sits 3.4 points above prediction under both models."* It was **+3.4 pp
then and +3.57 pp now.**

What changed is the *bar*, not the miss. §9 recorded Gate FG-2 as passing
**narrowly** — 2.716 against 2.755 — and named the risk: *"the quadratic is
adequate, not comfortably right."* Adding 12,818 observations tightened the
posterior predictive reference from 2.755 to 2.658, and a defect that was
already on the register crossed the line.

**This is what a defect register is for.** The row was written in Phase 2, the
fallback was named in Phase 2, and Phase 3 executed it without improvising.

The field-goals-only variant of the statistic is identical (2.863 against a
2.649 reference), which settles the other candidate explanation: extra points
are not what broke it. §10's wording did not specify whether W-4 ran on all
kicks or on field goals alone; both are reported, they agree, and the ambiguity
is recorded as a defect rather than resolved after seeing which one passed.

The cubic fixes it — the 50–54 bin falls from +3.57 pp to +2.60 pp (standardized
2.86 → 2.10) at the cost of the 45–49 bin drifting from −1.63 to −2.14 pp. The
curve is redistributed rather than rescued, and the 50–54 selection hypothesis
from §9 remains the better explanation of what is left.

### The fitted model

*The **refit** column is the v1.3 posterior — the same model on the population
that excludes blocked kicks, adopted 2026-08-18 (document 27 §14c). The
incumbent column is preserved: v1.1's and v1.2's artifacts were built on it.*

| Parameter | Mean | 89% interval | **Refit mean** | **Refit 89%** |
|---|---|---|---|---|
| `alpha` (log-odds at 40 yd) | 1.747 | 1.70 – 1.80 | **1.907** | 1.822 – 1.993 |
| `beta` (per yard) | −0.108 | −0.117 – −0.099 | **−0.116** | −0.126 – −0.106 |
| `gamma` (quadratic / 100) | 0.204 | 0.13 – 0.27 | **0.249** | 0.172 – 0.326 |
| `delta_cubic` (cubic / 1000) | −0.068 | −0.11 – −0.026 | **−0.081** | −0.128 – −0.035 |
| **`sigma_kicker`** | **0.342** | **0.268 – 0.417** | **0.385** | 0.305 – 0.467 |
| `roof[dome]` | +0.285 | +0.176 – +0.393 | +0.246 | +0.127 – +0.366 |
| `roof[closed]` | +0.294 | +0.180 – +0.411 | +0.250 | +0.123 – +0.375 |
| `roof[open]` | +0.529 | +0.238 – +0.830 | +0.461 | +0.161 – +0.771 |
| **`beta_wind`** (per mph) | **−0.0213** | **−0.0305 – −0.0119** | **−0.0224** | −0.0321 – −0.0127 |
| `beta_temp` (per °F) | +0.00385 | +0.00122 – +0.00655 | +0.00341 | +0.00043 – +0.00641 |
| `delta_xp` | +0.167 | +0.050 – +0.284 | **+0.122** | **−0.008 – +0.252** |
| `lambda_xp` | 1.263 | 0.862 – 1.725 | 1.247 | 0.887 – 1.666 |

**Two rows moved enough to change a sentence elsewhere in this document.**
`sigma_kicker` rose 12.7%, which moves document 05 §3's shrinkage weight from
0.285 to about 0.336 — a kicker keeps a third of their own record rather than a
quarter. And `delta_xp`'s interval no longer excludes zero, which retires the
claim below.

**`sigma_kicker` barely moved** — 0.360 in Phase 2, 0.342 now. That is the check
that mattered most and it is easy to overlook: if adding weather had collapsed
the kicker spread, the previous model would have been attributing conditions to
kickers, and every Phase 2 kicker estimate would have been contaminated. It did
not. Kicker skill and kicking conditions are close to orthogonal, which is what
you would expect when kickers work in varied venues.

### Gate W-3 — wind is real, and about as large as the raw data suggested

> `beta_wind` = **−0.0213** per mph, 89% interval −0.0305 to −0.0119, against a
> pre-registered threshold of +0.00268. The interval's *upper* bound clears it by
> a wide margin.
>
> **At 45 yards, going from calm to a 15 mph wind costs 5.50 percentage points of
> make probability (89% interval 3.08 – 7.91).**

That lands above the 4 pp effect §10's power table put at 0.800 power, so this
is a result the design was built to resolve rather than one scraped off its
edge. The 5.5 pp figure is close to the ~5 pp gap the raw distance-adjusted bins
showed between calm and 10–14 mph, which is corroboration from a completely
different direction.

### Roof — indoor kicking is materially easier

| Roof | Make-rate change at 45 yd |
|---|---|
| Dome | **+4.53 pp** |
| Closed retractable | **+4.66 pp** |
| Open retractable | **+7.69 pp** |

Dome and closed are statistically indistinguishable from each other and both
exclude zero decisively — a roof is a roof. "Open" is larger but rests on 204
attempts and its interval is correspondingly wide (+0.238 to +0.830 log-odds);
it is also the one level where a selection story is plausible, since a
retractable roof is *opened* on pleasant days.

### Gate W-7 — temperature does something, and it is small

> `beta_temp` = **+0.00385** per °F, 89% interval +0.00122 to +0.00655. Its lower
> bound clears the null bound of −0.00084, so **the pre-registered reporting rule
> permits a claim.**
>
> **Across a 40 °F swing, +2.66 percentage points of make probability at 45
> yards.** Warmer is easier.

This was the term least expected to survive — the raw distance-adjusted temp
buckets showed no monotone trend at all. The reason it appears here and not
there is that the raw buckets confound temperature with wind and roof, and cold
games are windier. Conditioning on both isolates it.

### Gate W-6 — extra points, and what "folded in" turned out to mean

> `lambda_xp` = **1.263**, 89% interval **0.862 – 1.725**. **The interval
> contains 1**, so per the reporting rule committed in §10, **no claim is made
> that extra-point ability differs from field-goal ability.**

That is the useful answer: a kicker's field-goal ability transfers to extra
points, and the point estimate above 1 is not resolvable from noise. Sharing one
kicker effect is therefore supported rather than assumed — which is exactly what
`lambda_xp` was added to test.

> `delta_xp` = **+0.167** log-odds, 89% interval +0.050 to +0.284, **excluding
> zero.**

**Retired 2026-08-18.** On the population that excludes blocked kicks the
interval is −0.008 to +0.252 and **the claim does not survive** (document 27
§14c). Field goals lost 1.79% of their attempts to blocks and extra points only
0.86%, so part of what looked like an extra-point advantage was the two
populations' different block rates. What follows is the incumbent's reading,
kept as the record of what was claimed:

An extra point is genuinely easier than a field goal *from the same 33 yards* —
about **+0.9 pp** at that base rate. A plausible mechanism is that a PAT is a
routine, uncontested snap-hold-kick where a 33-yard field goal is a live
scrimmage play with a rush. The offset earns its place, and §10's pre-registered
check on it — *"the XP offset must be estimated, not assumed zero"* — resolves in
favour of keeping it.

### Gate W-8 — what actually changed in the ledgers

*Script: `research/15_simulator_v11.py`. v1's artifacts are left untouched, since
document 07's validation was run against them.*

| Quantity | v1 | v1.1 |
|---|---|---|
| Ledger entries | 16,645 | **29,463** |
| — fumble | 5,914 | 5,914 |
| — field goal | 10,731 | 10,731 |
| — **extra point** | **0** | **12,818** |

**Field goals repriced: 7,507 of 10,731** moved by more than half a point of
make probability. Mean absolute shift **2.27 pp**, maximum **23.07 pp**.

**And the direction is systematic, which was the entire argument for doing
this:**

| Roof | Attempts | Mean change in make probability |
|---|---|---|
| Open retractable | 204 | **+4.94 pp** |
| Closed retractable | 1,500 | **+2.85 pp** |
| Dome | 1,700 | **+2.79 pp** |
| Outdoors | 7,327 | **−0.22 pp** |

Indoor kicks are now priced as the easier kicks they are, so a missed dome kick
is charged *more* bad luck and an outdoor miss slightly less. Under v1 every
one of those 3,404 indoor attempts was mispriced in the same direction, for ten
seasons — precisely the systematic error §7 predicted.

Game-level effect:

| Quantity | Value |
|---|---|
| Mean absolute change in deserved margin | **0.355 points** (max 3.10) |
| Mean absolute change in DTW% | **0.0148** (max 0.403) |
| Games whose DTW winner flipped | **47 of 2,761 (1.70%)** |

### The re-validation

The treatment table changed, so document 06's Gate 1 had to be **re-earned
rather than inherited**. Same 531 pairs, same statistic, same +0.010
non-inferiority margin, same seed.

> **Mean paired Δ log loss = −0.00133, SE 0.00277, 95% CI
> [−0.00675, +0.00409]. PASS.**
>
> Document 07 measured −0.00159 (SE 0.00273) on v1.

Essentially unchanged, and still non-inferior. The weather adjustment and the
extra-point component together move 47 games' verdicts without degrading the
predictive content of a game by any amount this design would call meaningful.

### Defects added by this round

| Defect | Evidence | Status |
|---|---|---|
| **The 50–54 yard bin still misses by +2.6 pp** | Worst bin under all three curves now fitted | **Open.** The §9 attempt-selection hypothesis is untested; the cubic reduces the miss without explaining it |
| The cubic has no mechanism story | Added to fix a calibration failure, like `gamma` before it | **Open.** It is a curvature correction and must not be read as a theory of kicking |
| §10 did not say whether W-4 ran on all kicks or field goals only | Both computed; they agree | **Closed** by reporting both, but the wording was a gap |
| Wind direction is not in the data | A 20 mph crosswind and a 20 mph tailwind are recorded identically | **Open, and it caps how well any wind term can do.** It biases toward finding *less* wind effect, so −0.0213 is a floor |
| "Open" roof rests on 204 attempts | +7.69 pp with a wide interval | **Open.** Plausibly selection — retractable roofs open on pleasant days |
| Extra points are inside `core` in the Phase 1 decomposition | `components.py` buckets `extra_point` plays into `core`, while the simulator now books them as their own ledger rows | **Open, and self-consistent.** The simulator anchors on the actual margin rather than on the decomposition, so the ledger identity still holds exactly; but the two views of a game disagree about where XP luck lives |
| Kicker-season rows still not independent across seasons | Unchanged from §7 | **Open** |
