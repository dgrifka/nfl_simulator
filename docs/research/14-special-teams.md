# 14 — Special teams: punting, the bounce, and returns

*Written 2026-08-17, **before `research/22_special_teams.py` existed**. Power
calculations and null bounds: `research/22_special_teams_power.py`, results in
`research/outputs/22_special_teams_power.json`. Committed to git before the
round produces a result, so goalpost integrity is checkable by commit
archaeology.*

*Inputs: documents 01–13, all settled. Process laws unchanged — pre-register
before fitting, power-check every threshold, **Gate A before Gate B**, relative
convergence tolerances, and characterize a new instrument before writing its
gate.*

---

## 1. One-page story

### The question

The project has adjudicated offense, defense, kicking and the sequencing of
production. **Special teams is the last untouched third of a football game**, and
a public product will be asked about it. Three components, in the order a punt
actually happens:

1. **The punt itself** — is punting a real, sized skill, and does weather move
   it the way it moves field goals?
2. **The bounce** — a punt that lands and rolls is a loose oblong ball nobody
   controls. That is structurally the fumble case, and the fumble case is the one
   component this project neutralizes in full.
3. **The return** — does return yardage persist, at the returner grain or the
   team grain?

### The answer to the Gate A question, stated first

**No component of this round earns a ledger row, and that is settled before any
model is fit.**

| Component | Gate A — is there a branch point? | Reaches Gate B? |
|---|---|---|
| **Punting** | **Fail.** A punt is a played-out sequence: a snap, a catch, a step, a strike, and eleven players covering. The ball's flight is *produced* by the punter the way a completed pass is produced by a quarterback — there is no moment where two branches are resolved by nobody | No |
| **The punt bounce** | **Pass.** A ball on the turf, oblong, bouncing; both teams converging. Structurally identical to the fumble case document 09 §2 admitted for onside kicks | **Yes** — and §4 shows it is unobservable |
| **Kick and punt returns** | **Fail.** Blocking and tackling — the same argument document 05 §3's return-yardage row already made, and document 08 §6 made for third-down conversions: *"there is no coin to replace with its expectation… there is a continuum of outcomes produced by football"* | No |

So the deliverables are **skill-or-noise verdicts as reported findings**, plus a
weather-aware characterization of punting, plus an honest determination on the
bounce. Nothing in document 05 §3's treatment table moves.

### Five things to hold onto

1. **Gate A does the deciding here, as it did in document 09**, where it
   disqualified four of five candidates before a model ran. That is the gate
   working, not the gate being lazy.
2. **The one component with a genuine branch point is the one the data cannot
   see.** §4 settles that from the play-by-play's own structure, before any fit,
   because the Phase 4 plan required the data limitation to be confronted head-on
   rather than discovered afterwards.
3. **Punting is far better measured than kicking.** A punter-season carries a
   median of **65 punts** against a kicker-season's 29, on 22,403 punts total.
   The design resolves a punter spread of 1.0 net yards essentially always.
4. **The kickoff-return question cannot be answered.** §6's power table says so
   in advance at every grain and every era — and it is not close.
5. **Weather enters through the same door as it did for field goals.**
   `fg_model.sanitize_weather` is called, not reimplemented, so a windy day means
   the same thing in both models.

### Statistic convention

Posterior means with 89% equal-tailed intervals, matching documents 03, 05, 05b,
08, 09 and 13. Split-half correlations are the mean over 200 random
within-season splits, matching documents 02, 08 and 11. Population spreads are
always read against a **simulated null bound**, never against zero — document
05 §8's closing defect.

---

## 2. Gate A, argued component by component

> **Is there a moment where the outcome is resolved by a mechanism outside either
> team's control, conditional on the state both teams created?**

### Punting — **FAIL**

A punt is a sequence of things people do. The snapper snaps, the punter catches
and steps and strikes, the coverage runs, the returner fields it, tacklers
tackle. Every yard of the result is produced by twenty-two players executing.

**The tempting counter-argument, and why it does not survive.** A punt is a ball
in flight, and document 05 §2 admitted field goals at *partial* neutralization on
exactly that basis — *"the kicker caused most of it, and then it drifts."* The
difference is what the flight resolves **into**. A field goal has two branches
and only two: it is good, or it is not, and a foot of drift decides which. A punt
has no branches at all — it lands somewhere on a continuum, and every point on
that continuum is a slightly different amount of field position. There is no
coin, and there is no `swing` value to book, because there is nothing to book it
*between*. That is document 08 §6's argument verbatim, applied to a kick instead
of a third down.

### The punt bounce — **PASS**

Once the ball hits the ground and nobody has caught it, it bounces. The ball is
oblong, the bounce is not controlled by anyone, and where it stops is a genuine
branch — the same structure that makes fumble recovery the one component this
project neutralizes in full, and the same structure document 09 §2 granted to
onside kicks.

The kick *itself* is skill and stays in `core`, exactly as choosing to attempt a
55-yard field goal stays in `core`. **Only the roll is the branch.**

### Kick and punt returns — **FAIL**

A played-out sequence of blocking and tackling. Document 05 §3 already carried
this row — *"a return is a played-out sequence, not a branch resolved by
nobody"* — on the strength of interception returns alone. This round extends the
*measurement* to kickoffs and punts; it does not reopen the mechanism.

---

## 3. Data

- **Punting**: `data/pbp/*.parquet`, 2016–2025. **22,403 punts**, **392
  punter-seasons**, **107 distinct punters**, median **65 punts** per
  punter-season. League mean net punt yards **41.10**, SD 10.35.
- **Returns**: 28,274 kickoffs and 22,519 punts over the same window.
- **Weather**: `roof`, `wind`, `temp`, passed through
  `nfl_simulator.fg_model.sanitize_weather` — indoors the reading is nulled, wind
  is capped at 30 mph, and missing outdoor weather is centred rather than
  imputed. **66.9%** of punts carry a usable reading.

### Facts that must be defensible by name

- **The response is net punt yards**, and the identity is exact:
  `net = kick_distance − return_yards`, or `spot − 20` on a touchback. The ball
  starts `spot` yards from the opponent's goal, travels `kick_distance`, and
  comes back `return_yards`.
- **Net yards, not gross distance, and that is a ruling.** Gross distance is
  closer to the punter's leg, but it is not monotone in *good*: from the
  opponent's 40 a 55-yard punt is a touchback and a 35-yard punt downed at the 5
  is excellent. Net yards is good in the same direction everywhere on the field.
  The cost is that the coverage unit and the returner are inside the response,
  and §7's secondary arm measures exactly how much of it they own.
- **The model conditions on the spot**, quadratically in centred yards to the
  opponent's goal, because punting from your own 20 and from the opponent's 40
  are different jobs. Centred at 65 yards, near the median punting spot.
- **Blocked punts are excluded.** A block is a protection failure. Document
  05b §2 made the *opposite* ruling for blocked field goals — charging them to
  the kicker — for consistency with `components.py`; nothing here needs that
  consistency, and a punter releases the ball later than a kicker does, so a
  block is even less his doing.
- **The likelihood is Student-t, and that is a ruling.** Net punt yards run from
  +79 down to **−57**, and the left tail is punts returned for a touchdown: a
  coverage failure, not a punting outcome. Under a Normal likelihood some thirty
  such plays would set the residual scale for all 22,403 punts and choose how far
  every punter is shrunk toward the mean. Estimating the degrees of freedom lets
  the data say how heavy the tail is instead.
- **The grain is the punter-season**, matching the kicker-season grain document
  05b fixed, and inheriting the same defect: one punter supplies up to ten rows
  of the same underlying ability, which slightly inflates the apparent spread.
- **Kickoff eras are never pooled.** The kickoff return rate ran 0.25 in 2023,
  **0.33 in 2024** (the dynamic kickoff) and **0.74 in 2025** (the touchback spot
  moved to the 35), and mean return yards jumped from 20.0 to 27.4. Two
  structural breaks in two years. Pooling them would measure the rulebook. Punt
  returns show no such break — the return rate sits between 0.68 and 0.71 in all
  ten seasons — so punts are pooled across the window.
- **Every split is within-season**, so a rule change can never straddle a split.

---

## 4. The punt bounce — an observability determination, made before any model

*The Phase 4 plan required this to be confronted head-on: **state explicitly
whether the branch is observable at all**, and if not, record the verdict like
document 09's onside row rather than fitting something anyway.*

**It is not observable, and the reason is structural rather than statistical.**

The play-by-play records **one spot per punt**. `kick_distance` is the distance
to where the ball was first touched or came to rest, which means:

| Punt outcome | What `kick_distance` contains |
|---|---|
| Fair catch, or returned | **Flight only** — the ball never touched the ground |
| Downed, or out of bounds | **Flight plus roll**, inseparably |
| Touchback | Flight plus roll, ending in the end zone |

There is no landing-spot column, no bounce indicator, and — checked directly —
**not one punt description in the sample mentions a bounce**. For the punts that
bounce, the only recorded number already has the roll baked into it; for the
punts that do not bounce, the roll is zero by construction. **The two quantities
the branch is defined by are never separately recorded on the same play.**

> **Verdict: unresolvable by construction.** Not "unresolvable at this sample
> size" — more seasons would not help, because the field does not exist. This is
> a stronger and cleaner denial than document 09's onside row, which at least had
> an estimable quantity behind an inadequate denominator.

**What is still reportable, and it is stated as a bound and not an estimate.**
Caught punts have zero roll by construction; downed and out-of-bounds punts carry
flight plus roll in the same number. Comparing their mean `kick_distance` within
matched 5-yard spot bins gives an **upper bound on the mean roll**. It is an
upper bound and not an estimate because the two groups differ in *intent* as well
as in roll — a punter aiming to pin the opponent kicks shorter and hangs it
less — and that confound cannot be removed from this data. The number is reported
in §9 with that sentence attached.

---

## 5. Punting — the model, and the power behind its gates

### The model

```
net_i ~ StudentT(nu, mu_i, sigma)

mu_i = alpha
     + beta_spot * (spot_i - 65)
     + beta_spot2 * (spot_i - 65)^2 / 100
     + roof[level_i]
     + beta_wind * (wind_i - 8.282) * has_weather_i
     + beta_temp * (temp_i - 57.713) * has_weather_i
     + punter[punter_season_i]

punter[k] ~ Normal(0, sigma_punter)      non-centered
```

**Non-centered on the punter effect is a ruling, not a default** — document 04's
Gate 1 failure was a centered hierarchy funnelling, and the same geometry applies
whenever a group-level scale is small relative to its per-level noise.

**Documented fallback, named now so reaching for it later is execution rather
than improvisation:** if Gate PU-2 fails, add a **cubic** term in the centred
spot, refit, and report both arms. This is the identical fallback ladder
documents 05b §5 and §10 walked for the distance curve, and both arms stay in
the record.

### Priors

| Site | Prior | Plain-language meaning |
|---|---|---|
| `alpha` | `Normal(41, 10)` | Mean net punt yards from the median spot |
| `beta_spot` | `Normal(0, 0.5)` | Net yards per yard of field position |
| `beta_spot2` | `Normal(0, 1)` | Curvature; the field is not linear near either end |
| `roof[level]` | `Normal(0, 3)` | Level shift indoors — 3 yards is generous |
| `beta_wind` | `Normal(0, 0.5)` | Net yards per mph. At 0.5 a 15 mph wind moves 7.5 yards, far more than anyone believes |
| `beta_temp` | `Normal(0, 0.2)` | Net yards per °F |
| `sigma_punter` | `HalfNormal(3)` | SD of punter-season effects, in net yards |
| `sigma` | `HalfNormal(10)` | Residual scale |
| `nu` | `Gamma(2, 0.1)` | Degrees of freedom; lets the data set the tail |

### Power — the wind term

400 simulated datasets per scenario at the real covariate design, fitted by
ordinary least squares. The shortcut's direction is stated: unmodelled punter
spread inflates the residual, so these intervals are **wider** than the
hierarchy's and the bound is conservative. Document 05b §10 took the same
shortcut and recorded the same direction.

| True effect | `beta_wind` | Net yards lost at 15 mph | **Power** |
|---|---|---|---|
| small | −0.05 | 0.8 | **0.973** |
| moderate | −0.10 | 1.5 | 1.000 |
| large | −0.20 | 3.0 | 1.000 |
| very large | −0.40 | 6.0 | 1.000 |

> **The wind term is resolvable down to well under a yard.** 22,403 punts is an
> order of magnitude more evidence than the field-goal model had for the same
> question, where the minimum detectable effect was a 4 pp make-rate drop.

### Power — punter skill

400 datasets per scenario, **one-way punter-season variance component on
covariate-residualized net yards**. The gated instrument is one-way rather than
crossed, and that is a compute ruling with a consequence worth stating: a crossed
punter × return-unit fit at 392 + 320 levels costs about thirty seconds, which
makes a 400-dataset null bound a three-hour job and a power curve impossible.
The one-way form is exact — with a single factor the Cholesky collapses to a
reciprocal — and runs in under a millisecond. **The crossed fit is reported once,
descriptively, on the real data; no threshold in this document belongs to it.**

| True `sigma_punter` (net yards) | **Power** |
|---|---|
| 0.25 | 0.000 |
| 0.50 | 0.015 |
| **1.00** | **0.998** |
| 1.50 | 1.000 |
| 2.50 | 1.000 |

> **Minimum detectable punter spread: about 0.9 net yards.** A failure of Gate
> PU-3 therefore means "no punter effect **larger than about a yard**", never "no
> punter effect" — the same qualification document 05b §6 attached to Gate FG-3.

### The null bounds

| Quantity | Null bound | Construction |
|---|---|---|
| `sigma_punter` | **0.5448 net yards** | 90th percentile of the 89% upper bound under a true zero |
| return-unit spread | 0.5335 net yards | same, for the secondary arm's reference |
| `beta_wind` | **+0.00614 yards per mph** | 10th percentile of the 89% upper bound under a true zero |

---

## 6. Returns — the power table, and what it forbids

Split-half correlation of yards per return, 200 within-season splits, entities
needing 8+ games. Thresholds are each cell's own permutation null 95th percentile
— real entity-games dealt at random into synthetic entity-seasons, destroying
identity while keeping every denominator.

| Cell | Entity-seasons | Null 95th pct | **Power at r = 0.12** | Resolvable? |
|---|---|---|---|---|
| Kickoff / returner / 2016–2023 | 186 | 0.0894 | **0.61** | **No** |
| Kickoff / team / 2016–2023 | 255 | 0.0767 | **0.75** | **No** |
| Kickoff / returner / 2024 | 17 | — | — | **No** — too few entities to run |
| Kickoff / team / 2024 | 31 | 0.1963 | **0.26** | **No** |
| Kickoff / returner / 2025 | 50 | 0.1732 | **0.27** | **No** |
| Kickoff / team / 2025 | 32 | 0.1920 | **0.30** | **No** |
| **Punt / returner / 2016–2025** | 341 | 0.0681 | **0.82** | **Yes** |
| **Punt / team / 2016–2025** | 320 | 0.0621 | **0.89** | **Yes** |

> **The kickoff-return question cannot be answered by this project, at any grain
> or in any era.** Not close: the best kickoff cell reaches 0.75 power against
> the reference effect, and the two post-rule-change eras sit near 0.27 because
> one season supplies about thirty team-seasons.

This is recorded **in advance**, which is the whole point. Document 05 §7's
return-yardage row is the worked example of what happens without it: a null
result gets read as evidence of absence when the design could never have shown
presence.

**Note what the era rule costs, honestly.** Splitting kickoffs into three eras is
what makes 2024 and 2025 unanswerable — pooled, they would carry 63
team-seasons. Pooling them anyway would measure the rulebook, and a measure of
the rulebook reported as a measure of returners is worse than no measure.

---

## 7. Pre-registered gates

### Gate PU-1 — sampler health

**Pass rule:** zero divergences, `r_hat < 1.01`, `ess_bulk > 400`,
`ess_tail > 400` on every parameter.

**Plus a convergence cross-check with a RELATIVE tolerance**, which is document
09 §9's corrective applied: the grid instrument's `sigma_punter` and the
hierarchy's must agree to within **5% of the posterior mean**. Document 09's Gate
C-1 failed two candidates on an *absolute* 0.01 pp tolerance borrowed from a
different scale, and the corrective it recorded was exactly this.

The two are **not the same estimator** — the grid runs on covariate-residualized
net yards under a Normal likelihood, the hierarchy on the raw response under a
Student-t — so this is a sanity band, not an identity check, and it is stated as
such before it runs.

**On failure:** divergences mean the geometry is wrong and the fix is
reparameterization; low ESS with *zero* divergences means slow mixing and the fix
is more draws. Raising `target_accept` to quiet a warning is forbidden by
document 03 §5 and remains forbidden.

### Gate PU-2 — spot calibration

**Statistic:** the **largest standardized miss** across 5-yard spot bins holding
at least 100 punts, compared against its own posterior predictive distribution.

**Pass rule:** the observed statistic is at or below the **94.5th percentile** of
that reference.

Identical in construction to document 05b's Gate FG-2, which was itself fixed by
a power check that caught its multiplicity problem: requiring every bin to sit
inside its own interval fails on a *correct* model about a third of the time,
because eight bins at nominal coverage pass together only 0.89⁸ of the time.
Reducing to a single maximum and calibrating that maximum against its own
reference prices the multiplicity in exactly once.

**Documented fallback:** the cubic spot term named in §5.

### Gate PU-3 — is punter skill resolvable?

**Statistic:** the 89% interval for `sigma_punter`.

**Pass rule:** the 89% **lower** bound exceeds **0.5448 net yards**, the null
bound from §5.

**On pass:** punting is a real, sized team skill and it stays in `core` — which
changes nothing, because Gate A already denied it a ledger row. The value is the
*size*, which nothing in this project has previously measured.
**On failure:** the honest reading is "no punter effect larger than about a
yard", per §5's power table, never "no punter effect".

### Gate PU-4 — is the wind effect resolvable?

**Statistic:** the 89% interval for `beta_wind`.

**Pass rule:** the 89% **upper** bound is below **+0.00614**, the null bound from
§5. Same construction as document 05b's Gate W-3, and the same qualification: a
failure means "no *large* wind effect", never "no wind effect".

### Gate PU-5 — posterior predictive

**Pass rule:** the observed league mean net punt yards **and** the observed
between-punter variance of mean net yards both fall within the central 89% of
their posterior predictive distributions.

The variance half is the one that matters, for the reason document 03 §6 Gate 4
gave: a model that gets the mean right and the spread wrong is precisely the
model that would mislead about skill.

### Gate PU-6 — temperature *(reported, no pass rule)*

`beta_temp` with its 89% interval. **Reporting rule:** no claim about temperature
unless its interval clears a null bound built the same way Gate PU-4's was. The
convention document 03 §6 Gate 3 set — no threshold for a quantity with no prior
estimate.

### Gate PU-7 — whose skill is net punting? *(reported, no pass rule)*

A crossed **punter-season × return-unit-season** fit on covariate-residualized
net yards, reported once with 89% intervals against each factor's own null bound
from §5. Net yards contains the coverage and the return by construction, so this
is the arm that says how much of it the punter actually owns.

**No pass rule**, following document 05 §7's convention for its own crossed
attribution round. **Reporting rule, committed now:** a claim that net punting
"belongs to" the punter or to the return unit requires that factor's interval to
clear its null bound **and** the other's to fail to. If both clear it, the number
is shared and is reported as shared.

### Gate RT-1 — does return yardage persist? *(one per cell)*

**Statistic:** the cell's split-half r, mean over 200 within-season splits.
**Threshold:** that cell's permutation-null 95th percentile from §6.

### Gate RT-2 — is the result interpretable? *(the honesty gate)*

**Pass rule:** power at r = 0.12 is at least **0.80**.

**Fails for every kickoff cell.** For those, neither outcome of Gate RT-1 may be
reported as a finding — they are reported as *unresolvable*, with §6's table
attached. Only the two punt-return cells carry a readable verdict.

### The decision rule, committed in advance

| Outcome | Verdict | What changes |
|---|---|---|
| Any Gate A failure | No ledger row, at any value of `w` | **Nothing.** Document 05 §3's table is unchanged |
| PU-2 fails | Apply the §5 cubic fallback, report both arms | The adopted arm is the one that passes |
| PU-3 passes | Punter skill is real and sized | A reported finding; punting stays in `core` |
| PU-3 fails | No punter effect larger than ~0.9 net yards | Reported with the power table |
| RT-1 passes where RT-2 passes | That return channel is skill | A reported finding; still no ledger row |
| RT-1 fails where RT-2 passes | That channel is noise, **evidence of absence** | A reported finding |
| RT-2 fails | **Unresolvable** | Reported with §6's power table. No verdict either way |
| The bounce | **Unresolvable by construction** (§4) | Recorded like document 09's onside row, with the aggregate upper bound attached |

---

## 8. Known-defect register

| Defect | Evidence | Status |
|---|---|---|
| **`research/22_special_teams.py` was committed alongside this document, not after it** | `git log --diff-filter=A` shows the results script and this pre-registration added in the same commit, `521694c` | **Open, disclosed.** The script had not been run — `research/outputs/22_special_teams.json` postdates the commit — but the file-add ordering is weaker than steps 1 and 2, where the results script landed in a later commit. An auditor checking goalpost integrity by commit archaeology should know that this round's evidence is "the output postdates the commit", not "the script postdates the commit" |
| **The bounce is unobservable** | §4 — one spot per punt, no landing-spot field, no bounce mention in any description | **Open and permanent.** More seasons cannot fix a field that does not exist |
| Net punt yards contain the coverage and the return | `net = kick_distance − return_yards` by construction | **Open, and sized by Gate PU-7** rather than assumed away |
| The roll bound is confounded with punt intent | A coffin-corner punt is aimed short and hung low | **Open, stated.** The number is reported as an upper bound only |
| Punter-season rows are not independent across seasons | One punter supplies up to ten rows | **Open.** Same defect document 05b §7 recorded for kickers; slightly inflates apparent spread |
| Wind direction is not in the data | A 20 mph crosswind and a 20 mph tailwind are recorded identically | **Open, and it caps how well any wind term can do.** It biases toward finding *less* effect, so the estimate is a floor. Document 05b §11 recorded the same |
| Kickoff returns are unanswerable in every era | §6, best power 0.75 | **Open, pre-registered.** The era rule is what costs it, and the alternative costs more |
| The gated punter instrument is one-way, not crossed | §5's compute ruling | **Accepted, stated.** The crossed fit is descriptive and no threshold belongs to it |
| Punts negated by penalty stay in the sample | 9.3% of punts carry a penalty flag | **Open.** nflverse records the play; separating nullified from enforced punts is not attempted here |
| The 2024 and 2025 kickoff eras hold one season each | 31 and 32 team-seasons | **Accepted, by design.** Pooling would measure the rulebook |

---

## 9. Constants appendix

| Constant | Value | Where it lives |
|---|---|---|
| `RANDOM_SEED` | 20260817 | `research/22_special_teams_power.py`, `research/22_special_teams.py` |
| chains / tune / draws | 4 / 1000 / 1000 | `research/22_special_teams.py` |
| `target_accept` | 0.9 | same |
| `SPOT_CENTRE` | 65 yards | both scripts |
| `wind_centre` / `temp_centre` | 8.282 mph / 57.713 °F | measured, §3 |
| `WIND_CAP_MPH` | 30.0 | `src/nfl_simulator/fg_model.py` |
| `DATASETS` per scenario | 400 | `research/22_special_teams_power.py` |
| `NULL_PERCENTILE` | 90 | same |
| **Gate PU-3 threshold** | **0.5448 net yards** | this document §5 |
| **Gate PU-4 threshold** | **+0.00614 yards per mph** | this document §5 |
| Gate PU-2 percentile | 94.5th | this document §7 |
| Gate PU-1 relative tolerance | 5% | this document §7 |
| **Gate RT-1 thresholds** | punt/returner 0.0681 · punt/team 0.0621 · kickoff/returner 0.0894 · kickoff/team 0.0767 (2016–2023) | this document §6 |
| Gate RT-2 threshold | power ≥ 0.80 at r = 0.12 | this document §7 |
| `MIN_GAMES` | 8 | both scripts |
| `N_SPLITS` | 200 | both scripts |
| Punts / punter-seasons / punters | 22,403 / 392 / 107 | measured, §3 |

Results are written back into this document as §10.

---

## 10. Results

*Script: `research/22_special_teams.py`. Gate A settled in §2, the bounce
determination in §4, and every threshold in §§5–7, all committed at `521694c`
before this script existed. Results in
`research/outputs/22_special_teams.json`.*

### Outcomes, stated first

| Component | Gate A | Outcome |
|---|---|---|
| **Punting** | Fail — no ledger row | **Punter skill is real and large. The model that measured it fails its own calibration gates.** |
| **The punt bounce** | Pass — a genuine branch | **Unresolvable by construction**, and even the aggregate bound is unusable |
| **Kick returns** | Fail — no ledger row | **Skill**, strongly — but in cells the design was pre-registered as underpowered |
| **Punt returns** | Fail — no ledger row | **Skill**, at both grains, on the two properly powered cells |

| Gate | Quadratic (pre-registered) | Cubic (fallback) |
|---|---|---|
| **PU-1** sampler health | **PASS** — 0 divergences, r̂ 1.0087, ESS 1,483 | **PASS** — 0 divergences, r̂ 1.0056, ESS 1,169 |
| **PU-2** spot calibration | **FAIL** — 9.058 vs 2.818 | **FAIL** — 6.952 vs 2.869 |
| **PU-3** punter skill resolvable | **PASS** | **PASS** |
| **PU-4** wind resolvable | **PASS** | **PASS** |
| **PU-5** posterior predictive | **FAIL** — mean tail 0.000 | **FAIL** — mean tail 0.000 |
| PU-6 temperature | reported | reported |
| PU-7 punter vs return unit | reported | reported |

### (a) Punting — a large skill, measured by a model that does not fit

> **`sigma_punter` = 1.27 net yards, 89% interval [1.14, 1.41], against a null
> bound of 0.545.** The interval's lower bound clears the bound by more than a
> factor of two.

In football: a one-SD-good punter is worth about **1.3 net yards per punt** over
an average one. Across the ~65 punts a punter-season sees that is **83 yards of
field position a year**, and the gap between a one-SD-good and a one-SD-bad
punter is twice that. **Punting is a bigger skill than anything the weather does
to it**, and this is the first time this project has sized it.

The estimate is stable across three specifications that disagree about almost
everything else:

| Estimator | `sigma_punter` |
|---|---|
| Hierarchy, quadratic spot, Student-t | 1.280 |
| Hierarchy, cubic spot, Student-t | 1.272 |
| Grid, OLS-residualized, Normal, crossed with the return unit | **1.302** |

The third differs from the second by **2.33%**, inside Gate PU-1's 5% relative
tolerance — which is document 09 §9's corrective applied, a tolerance stated
relative to the quantity rather than borrowed as an absolute number from another
scale.

#### Gate PU-2 and PU-5 fail, and the diagnosis is a one-yard bias

The calibration failure is large in standardized units and **small in yards**,
which is the whole story:

| Spot (yards to opponent goal) | Punts | Observed net | Predicted | Miss | Standardized |
|---|---|---|---|---|---|
| 45–49 | 1,586 | 34.88 | 35.49 | −0.61 | 2.48 |
| 50–54 | 1,893 | 38.05 | 38.81 | −0.76 | 3.36 |
| 60–64 | 2,451 | 42.88 | 43.07 | −0.19 | 0.98 |
| 65–69 | 3,042 | 44.08 | 44.20 | −0.13 | 0.71 |
| **70–74** | **3,190** | **43.44** | **44.67** | **−1.23** | **6.95** |
| **75–79** | **2,375** | **43.47** | **44.79** | **−1.32** | **6.49** |
| **80–84** | 1,718 | 43.32 | 44.55 | −1.23 | 5.18 |
| 85–89 | 1,150 | 43.14 | 44.25 | −1.11 | 3.85 |

**The model is about 1.2 net yards optimistic exactly where teams punt most —
backed up in their own territory.** With 22,403 punts the posterior predictive
mean of a 3,000-punt bin has a standard error near 0.18 yards, so a 1.2-yard bias
is seven standard errors. Gate PU-5's mean-tail probability of 0.000 is the same
fact seen from the league total: every posterior replicate produces a higher mean
net punt than the 41.10 actually observed.

The physical reading is that net punt yards **stop growing** once a team is
backed past its own 30 — hang time, coverage spacing and returner room bind
before leg strength does — and neither a quadratic nor a cubic in field position
can flatten that hard. The response is also **bounded above by the spot itself**,
since a punt cannot travel past the goal line without becoming a touchback, and
the model has no such ceiling.

**The pre-registered fallback was applied and did not rescue it.** §5 named the
cubic in advance precisely so reaching for it would be execution rather than
improvisation; it reduces the statistic from 9.06 to 6.95 against a 2.87
threshold, which is an improvement and not a pass. **No further form is tried
here.** Building a third curve after seeing which two failed, and re-running the
same gate, is the goalpost-moving document 08 §11 refused. The fix — a monotone
spline in field position with an explicit ceiling at the goal line — is named as
future work needing its own pre-registration.

**What may and may not be claimed, given that.** `sigma_punter` and `beta_wind`
are *variance and slope* parameters, and the failure is a *location* bias that
runs in the same direction across the whole high-volume region. Their stability
across three specifications — including one that shares neither the likelihood
nor the estimator nor the covariate treatment — is real evidence that they are
not artifacts of the misfit. But the model is **not adopted as a calibrated
description of punting**, and the per-spot fitted curve should not be quoted.

#### Gate PU-4 — wind moves a punt far less than it moves a field goal

> `beta_wind` = **−0.0204** net yards per mph, 89% interval −0.0411 to +0.0008,
> against a null bound of +0.0061. The upper bound clears it. **PASS.**
>
> **At 15 mph, a punt loses 0.31 net yards.**

That is a third of a yard, and §5's power table put a 0.8-yard effect at 97%
power — so **the true wind effect is smaller than the smallest scenario the
design was powered against.** Note also that the interval contains zero: the gate
is one-sided by construction and it passes narrowly.

Compare the field-goal model, where 15 mph costs **5.50 percentage points** of
make probability (document 05b §11). Two plausible reasons, neither tested here:
punters and coverage units adjust to conditions in a way a kicker aiming at fixed
uprights cannot, and **wind direction is not in the data at all**, so a tailwind
and a headwind are recorded identically — a defect that biases toward finding
less effect and which binds harder on a kick aimed downfield than on one aimed at
a target 45 yards away.

| Roof | Change in net punt yards |
|---|---|
| Dome | +0.56 [+0.30, +0.82] |
| Closed retractable | +0.90 [+0.63, +1.16] |
| Open retractable | +0.81 [+0.14, +1.49] |

Indoor punting is better by about three quarters of a yard — real, and an order
of magnitude smaller in football terms than the 4.5-point make-probability gap a
roof buys a kicker.

**Gate PU-6, temperature:** +0.038 net yards per °F, **+1.53 net yards across a
40 °F swing**. Warmer is better, and by more than the wind does — the same
direction the field-goal model found, and here the larger of the two weather
terms.

#### Gate PU-7 — net punting belongs to the punter

> Punter-season spread **1.302** [1.130, 1.325]; return-unit-season spread
> **0.895** [0.703, 0.965]. Null bounds 0.545 and 0.534.
>
> **Both clear their bounds, and `P(punter spread > return-unit spread) =
> 0.992`.**

Per the reporting rule committed in §7, that is not a claim that the gap
"belongs to" the punter — the rule required one factor to clear while the other
failed to, and both clear. The honest statement is that **both the punter and
the opposing return unit are real, sized contributors to net punt yards, and the
punter is the larger of the two with 99% posterior probability.**

This matters for the response choice §3 flagged. Net yards contains the return by
construction, and the worry was that it might be mostly the return. It is not:
the punter carries about 1.45 times the spread of the unit trying to bring it
back.

### (b) The punt bounce — unresolvable, and the bound does not rescue it

22,403 punts split as: 7,551 returned, 6,208 fair-caught, 2,900 downed, 2,272 out
of bounds, 1,533 touchbacks, 1,939 other. **Exactly one description in 22,519
punt rows mentions a bounce**, and there is no landing-spot column.

> **Verdict: unresolvable by construction**, as §4 determined before any model
> existed. The two quantities the branch is defined by — where the ball landed,
> and where it stopped — are never separately recorded on the same play.

§4 promised an aggregate upper bound on the mean roll and said it would be a
bound rather than an estimate because punt intent confounds it. **The confound
turns out to be larger than the effect, and it flips the sign with field
position**, which is a cleaner demonstration than the caveat sentence was:

| Spot | Caught punts (flight only) | Bounced punts (flight + roll) | Gap |
|---|---|---|---|
| 35–39 | 27.9 yd | 31.8 yd | **+3.9** |
| 40–44 | 31.9 | 34.8 | +2.9 |
| 45–49 | 36.0 | 38.5 | +2.5 |
| 50–54 | 40.1 | 40.9 | +0.8 |
| 55–59 | 44.0 | 43.1 | **−0.8** |
| 60–64 | 47.5 | 45.5 | −2.0 |
| 65–69 | 49.3 | 45.4 | −3.9 |

Punting from near midfield, a ball allowed to bounce travels **further** — which
is a roll. Punting from deep in your own territory it travels **shorter** —
which is a punter who has out-kicked nothing and a returner who has let a short
ball go. Volume-weighted the two cancel to **−2.08 yards**, a negative "upper
bound on a roll", which is arithmetic telling you the instrument is measuring
intent rather than physics.

**Recorded like document 09's onside row, and it is a stronger denial than that
one.** Onside kicks had an estimable quantity behind an inadequate denominator;
this has no quantity at all. More seasons cannot fix a field that does not exist.

### (c) Returns — skill, at both grains, and the kickoff era rule costs what it was said to cost

| Cell | Split-half r | Threshold | Power at r = 0.12 | Verdict |
|---|---|---|---|---|
| Kickoff / returner / 2016–2023 | **+0.273** | 0.0894 | 0.61 | **skill** (see below) |
| Kickoff / team / 2016–2023 | **+0.347** | 0.0767 | 0.75 | **skill** (see below) |
| Kickoff / returner / 2024 | — | — | — | too few entities to run |
| Kickoff / team / 2024 | +0.029 | 0.1963 | 0.26 | **unresolvable** |
| Kickoff / returner / 2025 | **+0.327** | 0.1732 | 0.27 | **skill** (see below) |
| Kickoff / team / 2025 | **+0.359** | 0.1920 | 0.30 | **skill** (see below) |
| **Punt / returner / 2016–2025** | **+0.162** | 0.0681 | 0.82 | **skill** |
| **Punt / team / 2016–2025** | **+0.151** | 0.0621 | 0.89 | **skill** |

**Return yardage persists, and it is not close.** Kickoff returns at the team
grain split at +0.347 — larger than any component document 02 measured, larger
than document 08's leverage-timing gap at +0.180, and roughly six standard
deviations of its own permutation null above the threshold.

That contradicts the one prior reading this project had. Document 05 §8 measured
**interception** return yards at r = −0.014 and concluded no measurable
persistence. Both are right: an interception return is a defensive back running
with a ball he was not expecting to have, on a play with no designed blocking. A
kickoff or punt return is a **scheduled play with a returner, a wall and a
scheme.** The earlier null was about a different football event, and document
05 §7's own power table said it could only have detected a 45%-relative effect
anyway.

**The honesty gate has a wording gap, and it is recorded rather than resolved
after the fact.** Gate RT-2 was written to stop an *underpowered null* being read
as evidence of absence — the failure mode document 05 §7's return-yardage row
demonstrated. Four kickoff cells detected an effect while sitting below the
0.80 power bar, and §7's decision-rule table did not name that case. The reading
applied here is that **a detection remains valid in an underpowered cell**,
because the threshold is the permutation null's 95th percentile and its
false-positive rate is therefore 5% by construction whatever the power is; low
power means the design would have missed a *smaller* effect, not that it
hallucinated this one. But the rule did not say so in advance, so it is a defect
below rather than a ruling.

**And the era rule cost exactly what §6 said it would.** 2024 is unanswerable at
the returner grain (17 entity-seasons clear the 8-game floor) and flat at the
team grain with 0.26 power. Pooling 2024 with 2025 would have produced 63
team-seasons and a readable number — measuring a rulebook that changed twice in
two years.

### What this changes

1. **Nothing in document 05 §3's treatment table.** Gate A denied all three
   components before a model ran, and no result here could breach it. That is
   now the fourth consecutive round where the mechanism gate, not the arithmetic,
   settled the ledger question.
2. **Punting is sized for the first time: 1.27 net yards of true spread between
   punter-seasons**, about 83 yards of field position a year — and the punter
   carries 1.45× the spread of the return unit opposing him.
3. **Weather barely touches punting.** 0.31 net yards at 15 mph against 5.50
   percentage points of make probability for a field goal. The one weather term
   that does something is temperature, at 1.53 net yards across 40 °F.
4. **Return skill is real and large at every grain the design can see**, which
   overturns the intuition that returns are where the season's blocks happened to
   fall — while leaving the interception-return null from document 05 §8 intact,
   because it was measuring a different play.
5. **The punting model is not adopted as a description**, and its calibration
   failure is on the record with the diagnosis and the named-but-unattempted fix.

### Defects added by this round

| Defect | Evidence | Status |
|---|---|---|
| **The punting model is ~1.2 net yards optimistic from deep own territory** | Gate PU-2, standardized misses of 5–7 in the four highest-volume bins; Gate PU-5 mean tail 0.000 | **Open, and fatal to the model as a calibrated description.** The named fallback did not fix it; the fix (monotone spline with a goal-line ceiling) needs its own pre-registration |
| **§7's decision rule did not cover PU-5 failing** | Only PU-2 had an on-failure rule | **New wording gap.** Reported rather than resolved after the fact, following document 05b §11's precedent |
| **§7's decision rule did not cover a detection in an underpowered cell** | Four kickoff cells passed RT-1 while failing RT-2 | **New wording gap.** The reading applied is stated in §10 and argued from the permutation null's calibrated false-positive rate |
| Net punt yards have a hard ceiling the model ignores | A punt cannot pass the goal line without a touchback | **Open.** Almost certainly part of the PU-2 failure |
| Gate PU-4 passes with an interval containing zero | −0.0204 [−0.0411, +0.0008] against a +0.0061 bound | **Open, stated.** The gate is one-sided by construction; the effect is real but smaller than any powered scenario |
| Wind direction is absent | A tailwind and a headwind are identical rows | **Open**, and it binds harder on punts than on field goals. Biases toward less effect, so 0.31 yards is a floor |
| Punter-season rows are not independent across seasons | 107 punters supply 392 punter-seasons | **Open.** Same defect document 05b §7 recorded for kickers |
| The 2024 kickoff era is unanswerable | 17 returner-seasons and 31 team-seasons | **Accepted, by design.** Pooling would measure the rulebook |
