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

- **Blocked kicks count as misses.** A block is partly a protection failure
  rather than a kicking outcome, so charging it to the kicker is arguably wrong.
  At 192 of 10,731 attempts (1.8%) splitting it out would add a class without
  changing a conclusion. `components.py` already treats blocks this way, and
  consistency with the Phase 1 decomposition matters more here than the 1.8%.
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
| Blocked kicks charged to the kicker | 192 of 10,731 attempts | **Accepted** for consistency with `components.py` |
| Long-distance bins are thin | 65+ yards holds 19 attempts across ten seasons (document 01) | **Open.** The model extrapolates there rather than borrowing a neighbour bin as `components.py` does; those kicks are rare enough not to move a game total |
| The game being adjudicated is inside the fit | Document 05 §5, the general contamination defect | **Open, bounded** at O(1/n); a kicker-season contributes ~2 of ~29 attempts |

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
