# 27 — The make-probability model, refitted without blocked kicks, pre-registered

*Written 2026-08-18, **before `research/42_fg_refit.py` existed**. The threshold
re-derivation runs in `research/42a_fg_refit_power.py` and its results land in
`research/outputs/42a_fg_refit_power.json`. Committed to git before the fit
produces a number.*

*Inputs: document 05b (the model being refitted, its gates and its defect
register), document 05 §2 (Gate A, which is the argument for the population
change), document 26 §6 and §9 (the open defect this closes and the reason it
was carried rather than fixed), document 18 §4b (the two-population reporting
rule), document 19 (the ship-record template this would follow if approved).*

*Tier: **data change** under the change-proposal template — the generative story,
every prior and every sampler constant are unchanged, and only the population the
model is trained on narrows. It is nonetheless **held at the door**, because it
changes the numbers the product prints on every kick in every game.*

---

## 1. One-page story

### The question

Document 05b's field-goal model answers one question: *what was the probability
this kicker made this kick from this distance, in these conditions?* It was
trained on 10,731 field goals and 12,818 extra points, and **192 of those field
goals and 110 of those extra points were blocked**, entered as misses.

A blocked kick is not an observation of the quantity the model estimates. The
ball never flew. Counting it as a miss teaches the model that kicks are harder
than they are, and the model's output — `p_make` — is the expectation in every
field-goal and extra-point luck row the simulator prints.

> **Is the shipped make-probability model biased low, and by how much, and does
> the corrected model still pass the gates that admitted the incumbent?**

### The answer

**Unseen.** No arm of this study has been run. §3 predicts the direction
(`p_make` rises) and gives a first-order size (about 1.5 pp on a field goal,
0.8 pp on an extra point) from the empirical rates alone; the fitted answer is
not those numbers, because the model is hierarchical and a kicker whose blocked
kicks are removed also loses attempts from their own denominator.

### Five things to hold onto

1. **This is the same defect as document 26's, one layer up.** Document 26 found
   the *ledger* booking luck on a play Gate A denies. This document finds the
   *model* learning from that same play. They are independent corrections with
   independent arguments, and either can be adopted without the other.
2. **The gates are document 05b's own, unchanged.** Nothing here invents a new
   bar. The corrected model has to clear sampler health, weather calibration,
   wind resolvability, distance calibration, posterior predictive and kicker
   resolvability — the gates that admitted the incumbent in the first place.
   **There is no materiality floor on this change, and that is not an exemption
   granted here**: document 05b never imposed one. Its ship rule was, and
   remains, gate-based.
3. **The refit on its own makes document 26's violation worse, and this document
   says so before measuring it.** Raising `p_make` by ~1.5 pp raises the luck
   booked on a blocked kick, because a blocked kick is scored `realized = 0`
   against a make probability that just went up. §9c requires that number to be
   reported. A reader who adopts the refit alone should know it buys a better
   model and a slightly larger known error.
4. **Two thresholds are inherited from a null simulation run at a larger `n`,
   and are re-derived rather than reused.** §7b re-runs document 05b §10's null
   simulation on the non-blocked design matrix, seed unchanged, before the fit.
   Both the inherited and the re-derived values are reported.
5. **Nothing merges on this document.** It is a proposal, held at the door with
   document 26's candidate, and the maintainer approves or rejects it.

### Statistic convention

Posterior means with 89% equal-tailed intervals, as everywhere else in this
project.

---

## 2. The defect, and why it was carried rather than fixed

Document 05b §2 recorded the decision by name, and recorded it as uncomfortable:

> **Blocked kicks count as misses.** A block is partly a protection failure
> rather than a kicking outcome, so charging it to the kicker is arguably wrong.
> At 192 of 10,731 attempts (1.8%) splitting it out would add a class without
> changing a conclusion. `components.py` already treats blocks this way, and
> consistency with the Phase 1 decomposition matters more here than the 1.8%.

Two things about that paragraph have since changed.

**The consistency argument has lost its referent.** It says the model should
count blocks as misses *because `components.py` does*. Document 26 §2 ruled that
`components.py` should not: a blocked kick fails Gate A and the ledger must not
neutralize it. Whether or not document 26's correction is adopted, the reason
given in 05b §2 for the model's population no longer stands on its own — it
points at a decision that is now argued to be wrong.

**"Without changing a conclusion" was about `sigma_kicker`, not about `p_make`.**
05b §2 was defending the *skill* conclusion, and it was right: 1.8% of attempts
cannot move a population SD by anything that matters. But the simulator does not
consume `sigma_kicker`. It consumes `p_make`, once per kick, on 23,549 kicks, and
a systematic 1.5 pp shift on all of them is a different quantity from a
negligible shift in a variance parameter.

**Document 26 §6 named this fix and deferred it**, on the grounds that it is a
separate model change needing its own writeup. This is that writeup.

---

## 3. Mechanism story

### Which defect this addresses

Document 05b §7's register, the row added by this document's §12, and document
26 §6's first row: *the make-probability model still counts blocks as misses.*

### Why this specific change should move that number

The model's likelihood is `made ~ Bernoulli(p)`, where `p` is a function of
distance, roof, wind, temperature and the kicker. A blocked kick contributes a
`made = 0` observation at whatever distance it was attempted from. Removing 302
zeroes from 23,549 observations mechanically raises the fitted curve. The
empirical size of the shift is already known from document 26 §3b:

| Population | With blocks | Without blocks | Change |
|---|---|---|---|
| Field goals | 10,731 attempts at 84.66% | 10,539 at **86.20%** | **+1.54 pp** |
| Extra points | 12,818 attempts at 94.41% | 12,708 at **95.22%** | **+0.82 pp** |

**But the fitted shift will not equal the empirical shift, and the difference is
the interesting part.** Three reasons:

- **Blocks are not distributed uniformly over distance.** If blocked kicks are
  more common on long attempts, removing them steepens the distance curve; if
  they are more common on short ones, it flattens it. The bin table is reported
  in §9a and neither direction is predicted here.
- **The hierarchy re-shrinks.** A kicker-season loses attempts as well as
  misses, so its `w = n/(n + κ)` falls slightly and its own effect shrinks
  further toward the league. A kicker with two blocked kicks in a 25-attempt
  season moves more than the league does.
- **Extra points and field goals share `sigma_kicker` and `lambda_xp`.** The two
  populations lose blocks at different rates, so the transfer parameter can move
  even though nothing about transfer changed.

### What would make it fail

- **A calibration gate failing.** If removing blocks breaks Gate W-2 or W-4, the
  corrected population is telling us the fitted curve shape depended on the
  blocked kicks, which would be a much larger finding than the correction.
- **`sigma_kicker` collapsing.** If kicker spread falls materially, blocks were
  carrying kicker information — which would be an argument that a block is
  partly a kicking outcome after all, and would weaken document 26's Gate A
  ruling rather than support it. §9b reports it against the incumbent's 0.342.
- **Sampler health.** Nothing about the geometry changes, so a W-1 failure would
  point at something unrelated to this change and would have to be understood
  before the arm is read.

### The Gate A connection, stated so it is arguable

The population argument here is the same one document 26 §2 makes about the
ledger: **a blocked kick is not a kick in flight.** It is worth being explicit
about what that does and does not license.

- It licenses **removing blocked kicks from the training population**, because
  the model's job is to estimate the probability that a kick in flight goes
  through, and a blocked kick is not a draw from that process.
- It does **not** license removing them from the *game*. The blocked kick's EPA
  stays exactly where it is — in `core` under document 26's correction, or in
  the field-goal luck row under the incumbent. This document changes what the
  model learns from, not what the ledger books.

---

## 4. DAG edit

**None.** The tier is a data change, and the template skips this section for a
reason: the generative story is character-for-character the model in document
05b §10, which is the model in `research/14_fg_weather_model.py`.

```
alpha, beta, gamma, delta_cubic, roof[·], beta_wind, beta_temp,
delta_xp, lambda_xp, sigma_kicker
        |
        v
logit p = alpha + beta·d + gamma·d²/100 + delta_cubic·d³/1000
          + roof[level] + beta_wind·wind + beta_temp·temp
          + delta_xp·is_xp + kicker[k]·(1 + (lambda_xp − 1)·is_xp)
        |
        v
made ~ Bernoulli(p)          <-- the population of this node narrows
```

The edit is to the **last line only**, and it is a filter on rows rather than a
change to any arrow: `field_goal_result != "blocked"` and
`extra_point_result != "blocked"`.

---

## 5. Data

- **Grain of a row**: one kick attempt — a field goal or an extra point.
- **Source**: `data/pbp/*.parquet`, 2016–2025, through
  `research/14_fg_weather_model.py`'s `load_kicks`, unchanged except for the
  filter.
- **Sanitize rules**: unchanged, and still `src/nfl_simulator/fg_model.
  sanitize_weather`, so the fit and the simulator cannot drift apart.

| | Incumbent (v1.1/v1.2) | Refit | Change |
|---|---|---|---|
| Field goals | 10,731 | **10,539** | −192 |
| Extra points | 12,818 | **12,708** | −110 |
| Total kicks | 23,549 | **23,247** | −302 (1.28%) |
| Kicker-seasons | 433 | reported in §9 | ≤ 0 |
| Wind centre | 8.0261 mph | recomputed on the refit population | reported |
| Temp centre | 57.9699 °F | recomputed on the refit population | reported |

**Identification is not a gate here and this is why.** `field_goal_result` and
`extra_point_result` are charted categorical fields with exactly three levels
each, verified in this document's §13: 9,085 made / 1,454 missed / 192 blocked,
and 12,101 good / 607 failed / 110 blocked. Every row of the shipped population
carries one of them, no row is null, and no text parsing is involved. Document
26 §5c already passed this as its Gate P-2 and reconciled the counts with
document 23 §C1. Restating it as a gate would be theatre.

**The centres move and that is not cosmetic.** `wind_centre` and `temp_centre`
are properties of the fitted sample, and `FieldGoalModel.from_posterior` takes
them from `fg_weather_summary.json`. A refit that wrote a new posterior while a
consumer kept reading the old centres would be a silent misprice on every
outdoor kick. §11 names the file the refit writes and §9d checks the round-trip.

---

## 6. Inference plan and compute cost

Everything in this section is document 05b §10 §4's plan, unchanged, and it is
restated rather than referenced so that a difference would be visible.

| | Value |
|---|---|
| **Engine** | NUTS via nutpie, auto-selected by PyMC |
| **Configuration** | 4 chains, 1,000 tune, 1,000 draws, `target_accept = 0.9` |
| **Parameterization** | non-centered on `kicker[k]` — a **ruling**, per 05b §5 |
| **Arms** | **2**: the pre-registered quadratic curve, and the cubic fallback |
| **Control** | the published posterior already on disk (`trace_fg_weather.nc`). No refit of the incumbent is needed or permitted |
| **Observations / parameters** | 23,247 / ~430 |
| **Wall clock** | ~5 minutes per arm, ~10 minutes total |
| **Threshold re-derivation** | `research/42a_fg_refit_power.py`, 2,000 Newton-Raphson logistic fits, minutes |

**Why both arms are run rather than only the adopted cubic.** Document 05b §11
adopted the cubic *because the quadratic failed Gate W-4 on the full
population*. Whether it still fails on the corrected population is a fact about
the corrected population, and running only the cubic would assume the answer.
The arm ladder is therefore identical to 05b §11's: fit the quadratic, read
W-4, and reach for the cubic only if it fails — with both arms reported either
way.

**Efficiency levers considered.** A cheaper screen is not warranted: the confirm
run is ten minutes. The expensive part of the original round — the 2,000-fit
power calculation — is re-run because §7b requires it, and it uses the fast
Newton-Raphson logistic fitter rather than the hierarchy, exactly as document
05b §10 §6 did.

**Downtime plan.** Nothing runs in parallel with the fits. The gate amendment of
document 28 and the ledger audit of document 29 are written while the fits run;
both are text and neither reads this document's results. Stated explicitly, as
the template requires.

---

## 7. Pre-registered gates

### 7a. Which gates are unseen — stated first

| Gate | Known at writing? |
|---|---|
| **R-1** — sampler health | **No.** Unseen |
| **R-2** — weather calibration | **No.** Unseen |
| **R-3** — wind resolvable | **No.** Unseen. Threshold re-derived per §7b |
| **R-4** — distance calibration | **No.** Unseen, and the gate that decides which arm is adopted |
| **R-5** — posterior predictive | **No.** Unseen |
| **R-6** — kicker skill still resolvable | **No.** Unseen |
| **R-7** — temperature | **No.** Reported, no pass rule |
| **R-8** — extra-point transfer | **No.** Reported, no pass rule |
| **The §9 impact report** | **No.** Reported, no pass rule, and required |

**Nothing about the fitted answer is known at writing.** The empirical rates of
§3 are known — they are document 26 §3b's, published — and they are inputs to
the prediction, not to any threshold. Every threshold below either comes from
document 05b, committed in 2026-08-17, or from a null simulation that runs
before the fit and cannot see it.

### 7b. The two inherited thresholds, and how they are re-derived

Gates R-3 and R-7 read against bounds that document 05b §10 obtained by
simulating **400 datasets under a true zero at the incumbent's `n`** and taking a
percentile of what the design produces. Those bounds are properties of the
design, and the design just lost 1.28% of its rows.

**Committed rule:** `research/42a_fg_refit_power.py` re-runs
`research/13_fg_weather_power.py`'s null simulation with blocked kicks excluded
from the design matrix — same seed, same 400 datasets, same simulating
parameters — and the re-derived bounds are the thresholds R-3 and R-7 are read
against. **Both the inherited and the re-derived values are printed**, and the
verdict is reported under each.

| Threshold | Inherited value | Used at the gate |
|---|---|---|
| R-3, `beta_wind` 89% upper bound | **+0.0026759** | re-derived at the refit's `n` |
| R-7, `beta_temp` 89% lower bound | **−0.0008393** | re-derived at the refit's `n` |
| R-6, `sigma_kicker` 89% upper bound | **0.2407** | **inherited unchanged**, see below |

**R-6's threshold is deliberately not re-derived**, and the reason is the
direction of the error. 0.2407 is the 90th percentile of what a *field-goals
only, n = 10,731* design produces under a true zero (document 05b §6). The refit
design is larger than that — 23,247 kicks — so the true null bound is *lower*
than 0.2407 and the inherited threshold is **conservative**. Document 05b §11
already read the joint model against it for the same reason. Re-deriving it
would cost a second power run to make a passed gate easier to pass.

**Power is inherited, and the inheritance is defensible because the design
shrank by 1.28%.** Document 05b §10 §6's power table puts the minimum detectable
wind effect at a 4 pp make-rate drop at 45 yards (power 0.800). A 1.28% reduction
in `n` inflates a standard error by 0.6%, which moves that table's entries by
less than the rounding they are printed at. The table is restated in §13 and is
the power attached to R-3.

### 7c. Gate R-1 — sampler health

**Pass rule:** zero divergences, `r_hat < 1.01`, `ess_bulk > 400`,
`ess_tail > 400` on every parameter. Document 05b §6's Gate FG-1 and §10's W-1,
unchanged.

**On failure:** the geometry has not changed, so a failure points at something
other than this correction and must be understood before the arm is read.
Raising `target_accept` to quiet a warning remains forbidden (document 03 §5).

*No power check: this is a diagnostic on the sampler, not an inference about
football.*

### 7d. Gate R-2 — weather calibration

**Statistic:** the largest standardized miss across weather cells — roof level
crossed with 5 mph wind buckets, cells holding at least 100 attempts — against
its own posterior predictive distribution.

**Pass rule:** observed at or below the **94.5th percentile** of that reference.

Document 05b §10's Gate W-2, unchanged, including the self-calibrating
reference. No external threshold is involved, so nothing here is inherited from
a different `n`.

**On failure:** the bucketed-wind fallback named in 05b §10 §7. Named now so
reaching for it later is execution rather than improvisation.

### 7e. Gate R-3 — is the wind effect still resolvable?

**Statistic:** the 89% upper bound on `beta_wind`.

**Pass rule:** below the null bound re-derived in §7b (inherited value
+0.0026759).

**Power:** 0.800 against a 4 pp make-rate drop at 45 yards, calm → 15 mph;
0.388 against 2 pp. **A failure means "no large wind effect", never "no wind
effect."**

**On failure:** the refit is **not adopted**. A correction that costs the model
its weather terms is not a correction; the incumbent posterior stays the
simulator's input and the defect stays on the register with this result attached.

### 7f. Gate R-4 — distance calibration, and the arm ladder

**Statistic:** the largest standardized miss across 5-yard distance bins holding
at least 100 attempts, against its own posterior predictive distribution.
Computed on **all kicks** and on **field goals only**, both reported — document
05b §11 closed the ambiguity that way and this document keeps the resolution.

**Pass rule:** observed at or below the 94.5th percentile of its own reference,
read on the **all-kicks** statistic.

**The arm ladder, fixed now:**

1. Fit the **quadratic**. If R-4 passes, the quadratic is adopted and the cubic
   is not fitted. *That would be a finding in itself* — it would say the cubic
   term document 05b added was fixing a miscalibration that the blocked kicks
   were causing.
2. If R-4 fails on the quadratic, fit the **cubic**, which is the incumbent's
   adopted form. If R-4 passes there, the cubic is adopted.
3. If R-4 fails on both, the refit is **not adopted** and the failure is
   reported. **No third curve is fitted.** The polynomial ladder stops at the
   rung the incumbent occupies; climbing further to rescue this correction would
   be a curve chosen to fit a result.

**Power:** document 05b §6's table for this statistic's construction — 0.960
pass rate under a well-specified model, 0.960 detection of a 10-point
misspecification at 55 yards, and **explicitly weak (0.320) against a 5-point
miss**. That limitation is restated rather than rediscovered.

### 7g. Gate R-5 — posterior predictive

**Pass rule:** the observed league make rate and the observed between-kicker
variance of make rates both fall within the central 89% of their posterior
predictive distributions. Document 05b §6's FG-4 and §10's W-5, unchanged.

### 7h. Gate R-6 — is kicker skill still resolvable?

**Statistic:** the 89% interval for `sigma_kicker`.

**Pass rule:** the 89% upper bound exceeds **0.2407** (§7b, inherited and
conservative).

**Incumbent:** 0.342, 89% interval 0.268 – 0.417.

**On failure:** the field-goal row of document 05 §3 collapses from partial to
full neutralization at the league curve — the consequence document 05b §6 fixed
in advance for its Gate FG-3, and it is the same consequence here. That would be
a large finding and it would need its own round; the refit would not be adopted
on the strength of a gate failure.

**Power:** 0.998 at a true `sigma` of 0.30 (a 4.2 pp gap at 45 yards), 0.767 at
0.20. The incumbent sits at 0.342, well inside the resolvable range.

### 7i. Gates R-7 and R-8 — reported, no pass rule

**R-7, temperature.** `beta_temp` with its 89% interval. **Reporting rule:** no
claim about temperature unless the interval clears the null bound re-derived in
§7b. Incumbent: +0.00385 [+0.00122, +0.00655], clearing −0.00084.

**R-8, extra-point transfer.** `lambda_xp` and `delta_xp` with 89% intervals.
**Reporting rule, unchanged from document 05b §10 §7:** a claim that extra-point
ability differs from field-goal ability requires `lambda_xp`'s 89% interval to
**exclude 1**. Incumbent: 1.263 [0.862, 1.725] — contains 1, so no claim is
made. `delta_xp`: +0.167 [+0.050, +0.284], excluding zero.

**R-8 is worth watching for a specific reason.** Extra points lose blocks at
0.86% and field goals at 1.79%, so the two arms of the transfer parameter are
trimmed unevenly. If `lambda_xp` moves materially, part of what the incumbent
called transfer was the two populations' different block rates.

---

## 8. Gate summary and the decision rule, committed in advance

| R-1 | R-2 | R-3 | R-4 | R-5 | R-6 | **Action** |
|---|---|---|---|---|---|---|
| Pass | Pass | Pass | Pass on the quadratic | Pass | Pass | **Propose adoption** as the v1.3 make-probability posterior, on the quadratic curve, and report that the cubic is no longer needed. **Stop at the door and ask the maintainer** |
| Pass | Pass | Pass | Pass on the cubic | Pass | Pass | **Propose adoption** on the cubic, matching the incumbent's form. **Stop at the door and ask the maintainer** |
| Pass | **Fail** | — | — | — | — | Apply the bucketed-wind fallback (§7d), refit, report both arms |
| — | — | **Fail** | — | — | — | **Do not adopt.** The correction costs the model its weather terms |
| — | — | — | **Fail on both arms** | — | — | **Do not adopt.** No third curve is fitted |
| — | — | — | — | — | **Fail** | **Do not adopt**, and open a round on the field-goal row of document 05 §3 |
| **Fail** | — | — | — | — | — | **Do not read the arm.** Understand the sampler first |

**In every row of that table the action is a proposal or a non-adoption. There
is no row in which anything merges.** The refit changes shipped numbers on every
kick in ten seasons and it lands only with the maintainer's explicit approval.

---

## 9. The impact report — required, no pass rule

A gate report that says "the corrected model is well-calibrated" answers the
wrong question for a reader deciding whether to adopt it. **These four reports
are as mandatory as the gates**, and document 05b §10's Gate W-8 is their
precedent: *"it is the number that tells a reader whether any of this
mattered."*

### 9a. What moved in the model

- Every parameter, incumbent versus refit, with 89% intervals.
- The league make-rate curve at 30, 40, 45, 50 and 55 yards, both models.
- The per-distance-bin table, both models, so a reader can see whether the
  correction changed the *shape* of the curve or only its level.
- The number of kicker-seasons, and how many kicker-seasons lost attempts.

### 9b. What moved in `p_make`, kick by kick

- The distribution of `p_make(refit) − p_make(incumbent)` over all 23,247
  remaining kicks: mean, median, 89% range, maximum.
- The same, split by roof and by 5-yard distance bin, because a correction that
  is uniform is a different thing from one that is systematic — this is exactly
  the split that made document 05b §11's Gate W-8 readable.
- `sigma_kicker` against the incumbent's 0.342, and the largest per-kicker
  movement.

### 9c. What moved in the ledger, on **both** populations

Document 18 §4b's two-population rule is binding here, and this document fixes
the two populations in advance so neither can be chosen after the fact:

| Population | Why it is reported |
|---|---|
| **All 2,761 games containing a kick** | The refit touches every kick, so this is the population where the change actually happens. It is the **primary** number |
| **The 287 games containing a blocked kick** | Document 26's population. Reported so the interaction of §9d is legible, and **explicitly not** a gate |

For each: median and mean |Δ deserved margin|, median and mean |ΔDTW|, and DTW
side flips, against v1.2 with the fumble draws held to their own seeded
generator in both arms.

### 9d. The interaction with document 26's candidate, stated as an obligation

Three quantities are reported, because a reader deciding between adopting one
correction, the other, or both cannot read the choice without them:

1. **The luck the refit books on a blocked kick.** The incumbent books a mean
   3.361 EPA on a blocked field goal (document 26 §3a). Under the refit,
   `p_make` is higher, so this number **rises**. *The refit alone makes the known
   Gate A violation larger.*
2. **v1.2's median 89% DTW half-width on the 287 blocked-kick games, recomputed
   under the refit posterior** — document 26's Gate P-3 floor of 1.6250 pp, which
   §9 of that document said is "not stable under that refit". The refit's own
   number for it is reported here so that any re-measurement of that candidate
   reads a floor computed **before** anyone looks at the candidate's statistic.
3. **The wind and temperature centres the refit writes**, and a round-trip check
   that `FieldGoalModel.from_posterior` reproduces the fitted `p_make` to
   floating-point tolerance when handed them.

**Reporting the floor here, in the refit's own document, is deliberate and it is
this document's contribution to the blindness problem.** Document 26's candidate
is already measured and known. The floor it would face under a refit is not, and
computing it in a document whose gates cannot be affected by it — none of R-1 to
R-8 reads a DTW number — keeps that one quantity honest.

---

## 10. What adoption would mean, fixed now

- `research/42_fg_refit.py` writes `trace_fg_refit.nc` and
  `fg_refit_summary.json` alongside the incumbent's artifacts. **The incumbent's
  files are not overwritten**, exactly as v1.1's and v1.2's were preserved.
- `research/14_fg_weather_model.py` gains the population filter, so the fit
  script and the corrected posterior cannot drift apart. The filter is expressed
  as the same two mask narrowings document 26 §5h fixed for the ledger, and
  **never as a filter on the play frame** — that trap is document 26 §8's, and it
  applies to any code that touches these masks.
- Document 05b §2's "facts that must be defensible by name" loses the blocked-
  kicks bullet and gains its replacement; §7's register loses the blocked-kicks
  row; §11's fitted table gains a refit column.
- Document 05 §3's treatment table gains **no row**. The field-goal and
  extra-point rows keep their treatment and their `w`; only the model behind
  `p(e)` changes.
- The simulator artifacts are rebuilt as **v1.3**, on the document 19 template,
  and v1.2's artifacts are left untouched.
- **The rematch validation is re-run.** This is the one place this document
  departs from document 26 §5h, and the reason is document 05b §11's own
  precedent: the weather round moved 47 games' verdicts and re-earned document
  06's Gate 1 rather than inheriting it. A change to `p_make` on every kick is
  the same kind of change. Document 12's warning — the rematch test is nearly
  blind below roughly 20% damage — means a pass proves little, and it is
  therefore reported as a **check, not a gate**, and never as the sole evidence.

---

## 11. Kill and rollback

- **On any non-adoption row of §8:** no production module changes, the incumbent
  posterior stays the simulator's input, `trace_fg_refit.nc` stays in
  `research/outputs/` as the record, and §14 is the write-up. The branch keeps
  the code.
- **On a pass:** nothing merges until the maintainer says so. The ship record is a
  separate document on the document 19 template, and it carries the combined
  arithmetic if document 28's amendment is also approved.
- **Downstream consumers, named:** `research/31_simulator_v12.py` (artifact
  build), `research/40_blocked_pricing_power.py` and `research/41_blocked_
  pricing.py` (both load `trace_fg_weather.nc` and its centres),
  `src/nfl_simulator/fg_model.py` (the read side), and any future re-measurement
  of document 26's candidate.

---

## 12. Known-defect register

| Defect | Evidence | Status |
|---|---|---|
| **The refit alone enlarges document 26's Gate A violation** | `p_make` rises, and a blocked kick is scored against it | **Open by construction, disclosed, quantified in §9d.** It is an argument for adopting both corrections together, not an argument against this one |
| **Blocked kicks stay in `components.py`'s empirical class tables** | `fit_fg_baseline` and `fit_xp_baseline` are untouched by this document | **Open, deliberate.** The empirical `epa_made`/`epa_missed` branch means are a separate population question and they are document 26's, not this document's. Adopting this refit alone leaves the swing computed on a population that includes blocks |
| **Kicker-seasons that are entirely blocked kicks would vanish** | A kicker-season with all its attempts blocked drops out of the fit | **Open, expected to be empty.** §9a reports the count; if it is not zero, the affected kicker-seasons fall back to the league curve, which is `w = 0` and the documented behaviour |
| **R-6's threshold is inherited from a smaller design** | 0.2407 was derived at n = 10,731 field goals | **Accepted, conservative, stated in §7b** |
| Everything document 05b §7 and §11 already carries | The 50–54 yard bin, the missing wind direction, the non-independent kicker-seasons, the cubic's absent mechanism story | **Unchanged.** This correction neither fixes nor worsens any of them |

---

## 13. Constants appendix

| Constant | Value | Where it lives |
|---|---|---|
| `RANDOM_SEED` | 20260818 | `research/42_fg_refit.py`, `research/42a_fg_refit_power.py` |
| chains / tune / draws / `target_accept` | 4 / 1000 / 1000 / 0.9 | `research/42_fg_refit.py` |
| `DISTANCE_CENTRE` | 40.0 yards | `research/42_fg_refit.py` |
| `MIN_CELL_ATTEMPTS` | 100 | same |
| `WIND_BUCKET` | 5.0 mph | same |
| `WIND_CAP_MPH` | 30.0 | `src/nfl_simulator/fg_model.py` |
| Blocked field goals / extra points removed | **192 / 110** | document 26 §3, verified here |
| Refit population | **23,247** kicks = 10,539 FG + 12,708 XP | §5 |
| Incumbent population | 23,549 kicks = 10,731 FG + 12,818 XP | document 05b §11 |
| Charted result levels, field goal | made 9,085 / missed 1,454 / blocked 192 | §5 |
| Charted result levels, extra point | good 12,101 / failed 607 / blocked 110 | §5 |
| **R-3 threshold, inherited** | **+0.0026759** | `research/outputs/13_fg_weather_power.json` |
| **R-7 threshold, inherited** | **−0.0008393** | same |
| **R-6 threshold** | **0.2407**, inherited and conservative | document 05b §6 |
| R-3 power at a 4 pp / 2 pp wind drop | 0.800 / 0.388 | document 05b §10 §6 |
| R-4 power at a 10 pp / 5 pp curve miss | 0.960 / 0.320 | document 05b §6 |
| R-6 power at `sigma` 0.30 / 0.20 | 0.998 / 0.767 | document 05b §6 |
| Incumbent `sigma_kicker` | 0.342 [0.268, 0.417] | document 05b §11 |
| Incumbent `beta_wind` / `beta_temp` | −0.0213 / +0.00385 | same |
| Incumbent `lambda_xp` / `delta_xp` | 1.263 / +0.167 | same |
| Incumbent wind / temp centres | 8.0261 mph / 57.9699 °F | `research/outputs/fg_weather_summary.json` |
| Document 26's Gate P-3 floor, incumbent | 1.6250 pp on 287 games | document 26 §4 |
| `points_per_epa` | 0.8389 | `research/outputs/model_metadata_v12.json` |

Results are written back into this document as §14.

---

## 14. Results

> **Approved 2026-08-18 by the maintainer.** The refitted posterior is adopted as the
> simulator's field-goal model and ships as part of v1.3, on §10's terms with
> §14c's correction to `w` folded in. The read-side defect of §14f is approved
> for fixing in the same ship — it is pre-registered in document 30 and measured
> in isolation there, so v1.3's change decomposes. Document 31 is the ship
> record.


*Scripts: `research/42a_fg_refit_power.py` (thresholds), `research/42_fg_refit.py`
(the fits and gates, §9a and §9b), `research/42b_fg_refit_impact.py` (§9c and
§9d) and `research/42c_read_side_defect.py` (§14f, written after the round-trip
check failed). The gates were committed at `88ac49c` before any of them existed.
Results in `research/outputs/fg_refit_summary.json`,
`42a_fg_refit_power.json`, `42b_fg_refit_impact.json` and
`42c_read_side_defect.json`.*

### The verdict, stated first

> **The corrected model passes all six gates on the same cubic curve the
> incumbent adopted. `p_make` rises by a mean of 1.33 pp across 23,247 kicks and
> by more the longer the kick — 0.44 pp at 20 yards, 3.03 pp at 55. Kicker skill
> comes out *larger* rather than smaller, `sigma_kicker` rising from 0.342 to
> 0.385. And the round-trip check pre-registered in §9d as plumbing found
> something worse than the defect this round set out to fix: the shipped
> simulator has never read two of the fitted model's parameters, and prices a
> 55-yard field goal 6.8 percentage points too generously.**

| Gate | Statistic | Result | Verdict |
|---|---|---|---|
| **R-1** — sampler health | 0 divergences, max `r_hat` 1.0048, min `ess_bulk` 746 | — | **Pass** |
| **R-2** — weather calibration | worst standardized miss **2.145** vs 2.614, 9 cells | — | **Pass** |
| **R-3** — wind resolvable | `beta_wind` **−0.02241** [−0.03212, −0.01266] vs +0.00360 | Clears by a wide margin | **Pass** |
| **R-4** — distance calibration | **2.196** vs 2.721 on the cubic *(quadratic failed at 3.010 vs 2.859)* | — | **Pass on the cubic** |
| **R-5** — posterior predictive | make-rate tail p 0.500, between-kicker variance tail p 0.058 | — | **Pass** |
| **R-6** — kicker resolvable | `sigma_kicker` **0.385** [0.305, 0.467] vs 0.2407 | Lower bound clears | **Pass** |
| **R-7** — temperature | `beta_temp` **+0.00341** [+0.00043, +0.00641] vs −0.00071 | Clears the null bound | **Reported, claim permitted** |
| **R-8** — extra-point transfer | `lambda_xp` **1.247** [0.887, 1.667] — contains 1 | — | **Reported, no claim** |

Per the decision rule committed in §8 — *all gates pass → propose adoption, stop
at the door* — **nothing merges, and the maintainer decides.**

### 14a. The threshold re-derivation, and an honest note about it

`research/42a_fg_refit_power.py` reproduced document 05b's published null bounds
**exactly** on the full population, which is the check that it is running the
same instrument. On the refit population:

| Threshold | Inherited (n = 10,731) | Re-derived (n = 10,539) |
|---|---|---|
| R-3, `beta_wind` upper bound | +0.0026759 | **+0.0035955** |
| R-7, `beta_temp` lower bound | −0.0008393 | **−0.0007101** |

**The wind bound moved by 34% and §7b predicted 0.6%.** The prediction was about
the standard error, which does move by 0.6%; the *threshold* is a 10th percentile
of 400 simulated bounds, and a percentile of 400 draws carries Monte Carlo noise
far larger than the shift in `n`. **The §7b reasoning was right about the
quantity it named and wrong about the quantity it was applied to**, and it is
recorded here rather than quietly dropped. It changes nothing: `beta_wind`'s
89% upper bound is −0.0127, an order of magnitude below either threshold.

### 14b. The arm ladder — and the cubic was not the blocked kicks' fault

The pre-registered quadratic **failed R-4 again**, at 3.010 against a 2.859
reference, and it also failed R-5's between-kicker variance at a tail p of 0.046.
The cubic passed both. §7f named this as a finding either way, so here it is:

> **The cubic term document 05b added in Phase 3 was not a curvature correction
> for a population contaminated by blocked kicks.** Remove every block and the
> quadratic still cannot bend to the distance curve. The 50–54 yard bin remains
> the worst bin on the corrected population too, at +2.70 pp and 2.196
> standardized — down from +2.60 pp on the incumbent, essentially unchanged.
> Document 05b §9's attempt-selection hypothesis survives this round untouched.

### 14c. §9a — what moved in the model

| Parameter | Incumbent | Refit | Refit 89% interval |
|---|---|---|---|
| `alpha` (log-odds at 40 yd) | +1.7472 | **+1.9068** | +1.8215 – +1.9934 |
| `beta` (per yard) | −0.10804 | **−0.11587** | −0.12596 – −0.10592 |
| `gamma` (quadratic / 100) | +0.2038 | **+0.2489** | +0.1717 – +0.3257 |
| `delta_cubic` (cubic / 1000) | −0.0685 | **−0.0811** | −0.1280 – −0.0349 |
| **`sigma_kicker`** | **0.3420** | **0.3855** | 0.3045 – 0.4674 |
| `beta_wind` (per mph) | −0.02132 | −0.02241 | −0.03212 – −0.01266 |
| `beta_temp` (per °F) | +0.00385 | +0.00341 | +0.00043 – +0.00641 |
| `delta_xp` | +0.1669 | **+0.1222** | **−0.0080 – +0.2524** |
| `lambda_xp` | 1.2628 | 1.2472 | 0.8866 – 1.6658 |
| `roof[dome]` | +0.2846 | +0.2457 | +0.1274 – +0.3658 |
| `roof[closed]` | +0.2943 | +0.2501 | +0.1233 – +0.3748 |
| `roof[open]` | +0.5292 | +0.4605 | +0.1614 – +0.7711 |

**League make rate, average kicker, outdoors, no weather reading:**

> **Superseded 2026-08-18 by document 31 §4.** Every number in the table below
> was priced through `FieldGoalModel.league_make_probability`, which is the read
> side §14f found discarding `delta_cubic` — so both columns are a *quadratic*
> reading of a *cubic* posterior. `research/46_simulator_v13.py` reproduces this
> table exactly by stripping the cubic term, which is the check that the
> discrepancy is the defect and not a different fit. The corrected curve is
> steeper at range: **61.17% at 55 yards, not 67.39%**, against a corrected
> incumbent of 58.76%. The **change** column, which is what this section was
> arguing about, barely moves: +2.41 pp at 55 yards instead of +3.21 pp. The
> table is kept as printed because the pre-registration is the record.

| Distance | Incumbent | Refit | Change |
|---|---|---|---|
| 30 yd | 95.38% | 96.48% | **+1.09 pp** |
| 40 yd | 85.15% | 87.05% | **+1.91 pp** |
| 45 yd | 77.86% | 80.04% | **+2.19 pp** |
| 50 yd | 70.47% | 73.03% | **+2.55 pp** |
| 55 yd | 64.18% | 67.39% | **+3.21 pp** |

**Three things in that table are worth stating out loud.**

**1. Kicker skill got bigger, and §3 predicted the opposite direction was
possible.** `sigma_kicker` rose 12.7%, from 0.342 to 0.385, and the 89% interval
moved up with it. §3's failure condition was a *collapse* — that would have
argued blocks carry kicking information and would have weakened document 26's
Gate A ruling. The opposite happened: **blocked kicks were noise diluting the
measured spread between kickers, and removing them made kickers look more
different from each other.** This is also the first population change in the
project to *increase* a measured spread; documents 18 and 24 both widened a
population and watched the spread fall. Narrowing a population on a mechanism
argument is a different operation from widening it on one, and this is the
evidence.

**2. That moves document 05 §3's treatment table, and §10 of this document said
it would not.** §10 asserted the field-goal and extra-point rows "keep their
treatment and their `w`". The treatment stays partial, but `w` does not stay
0.285. Holding the median kicker-season's sampling variance fixed at the value
the incumbent's pair implies, `w = σ²/(σ² + s²)` gives:

> **`w` rises from 0.285 to about 0.336** at the median kicker-season — a kicker
> keeps a third of their own record rather than a quarter, and field goals are
> neutralized slightly less.

That is an arithmetic implication, not a re-derivation: **document 05b's
published 0.285 has no derivation anywhere in this repository**, so the exact
recomputation cannot be reproduced. It is registered as a defect below. The
§10 sentence is corrected here rather than edited, because the pre-registration
is the record.

**3. `delta_xp` no longer excludes zero.** Document 05b §11 claimed *"an extra
point is genuinely easier than a field goal from the same 33 yards — about
+0.9 pp"*, on an interval of +0.050 to +0.284. On the corrected population the
interval is **−0.008 to +0.252**. §7i predicted this exact mechanism in advance:
field goals lose 1.79% of their population to blocks and extra points only 0.86%,
so part of what looked like an extra-point advantage was the two populations'
different block rates. **The claim does not survive the correction**, and if the
refit is adopted, document 05b §11's wording has to change with it.

**Kicker-seasons: 433 → 432.** One dropped, and it is printed rather than
counted (document 20 §9): **`2025_00-0035042`**, a kicker-season whose only
charted attempts in the fitted population were blocked. Under the refit that
kicker falls back to the league curve, which is the documented `w = 0` behaviour
for an unknown entity. Mean absolute movement in a kicker effect is 0.055
log-odds; the largest is +0.344 (`2016_00-0025944`).

### 14d. §9b — what moved in `p_make`, kick by kick

Priced through `FieldGoalModel`, the path the product uses:

| Population | n | Mean shift | Median | 89% range | Max |
|---|---|---|---|---|---|
| All kicks | 23,247 | **+1.334 pp** | +1.119 pp | +0.448 – +2.943 | 10.33 pp |
| Field goals | 10,539 | +1.519 pp | +1.175 pp | — | — |
| Extra points | 12,708 | +1.182 pp | +1.095 pp | — | — |

**The shift is systematic in distance, and that is the answer to §3's open
question.**

| FG distance bin | n | Mean shift |
|---|---|---|
| 20–24 | 1,019 | +0.44 pp |
| 30–34 | 1,481 | +1.14 pp |
| 40–44 | 1,546 | +1.64 pp |
| 50–54 | 1,351 | +2.31 pp |
| 55–59 | 530 | +3.03 pp |
| 60–64 | 72 | +4.20 pp |

§3 asked whether blocks are distributed uniformly over distance and refused to
predict. They are not: **the correction grows monotonically with distance**,
which is what a population of blocks concentrated on longer, flatter-trajectory
kicks produces. By roof, the shift is +1.49 pp outdoors against +1.05 pp in a
dome and +0.98 pp under a closed roof — the same ordering, since long attempts
are not evenly spread across venues.

### 14e. §9c — what moved in the ledger, on both populations

Both arms share every seed, the v1.2 class tables and the fumble component; the
only difference is the field-goal posterior.

| | **All 2,761 games with a kick** *(primary)* | 287 games with a blocked kick |
|---|---|---|
| Median \|ΔDTW\| | **0.071 pp** | 0.298 pp |
| Mean \|ΔDTW\| | 0.463 pp | 0.835 pp |
| Max \|ΔDTW\| | 9.25 pp | 9.25 pp |
| Median \|Δ deserved margin\| | **0.081 pts** | 0.141 pts |
| Mean signed Δ deserved margin | +0.001 pts | −0.003 pts |
| DTW side flips | **18** | 3 |

**The refit is a small change to the product and a large change to the model,
and both halves of that sentence are true.** A 1.3 pp shift in `p_make` applies
to every kick in both directions of the ledger — a made kick books less good luck
and a missed kick books more bad luck — so the game-level effect largely cancels.
The mean signed change in deserved margin is +0.001 points across 2,761 games.
Eighteen games change hands, which is a third of what the weather round moved
(47, document 05b §11).

**These numbers were produced by a read side that does not read the model
correctly** — see §14f. Both arms carry the same defect, so the comparison is a
fair statement of *what the product would print*, and it is **not** a statement
of what the corrected model implies.

### 14f. §9d.3 — the round trip failed, and this is the round's largest finding

§9d asked for a plumbing check on the centring constants. The check failed on
field goals by up to **40.7 percentage points**, which is not a centring problem.
Two fitted parameters never reach the simulator:

| Discarded | What it does | Where it goes wrong |
|---|---|---|
| **`delta_cubic`** | The cubic distance term of the **adopted** Phase 3 curve | `FieldGoalModel._logit` computes `alpha + beta·d + gamma·d²/100` and stops. `from_posterior` never reads the variable. The simulator prices every kick on a **quadratic curve whose `gamma` was fitted jointly with a cubic term that is then dropped** |
| **`delta_xp`, `lambda_xp`** | The extra-point offset and the transfer of kicker ability to extra points | The read side has no extra-point terms at all. An extra point is priced on the plain field-goal curve at its 33 yards with the kicker effect at scale 1 |

**Sized on the shipped population** (`research/42c_read_side_defect.py`, 23,549
kicks, incumbent posterior, incumbent centres — this is v1.2 as it stands):

| FG distance bin | n | Shipped `p_make` | Fitted `p_make` | Error |
|---|---|---|---|---|
| 20–24 | 1,024 | 98.67% | 99.06% | −0.39 pp |
| 30–34 | 1,497 | 94.09% | 94.28% | −0.19 pp |
| 45–49 | 1,599 | 76.58% | 76.12% | +0.47 pp |
| **50–54** | **1,392** | **70.86%** | **68.38%** | **+2.48 pp** |
| **55–59** | **540** | **67.00%** | **60.21%** | **+6.80 pp** |
| 60–64 | 81 | 63.43% | 47.32% | **+16.11 pp** |
| 65–69 | 17 | 62.83% | 33.98% | **+28.85 pp** |

- **2,117 field goals are mispriced by more than a point** of make probability,
  and every one of the **12,818 extra points** is mispriced by −0.98 pp on
  average, in the same direction on all of them.
- In ledger terms: a mean **0.045 EPA** of misbooked luck per field goal against
  a mean |luck| of 0.939 EPA, and **0.010 EPA** per extra point against 0.108.
  Signed totals across ten seasons: **−385 EPA** on field goals and **+128 EPA**
  on extra points.

**Three things follow, and none of them is "fix it now".**

1. **This is not a Gate A violation and not this round's candidate.** It is a
   plumbing defect in `src/nfl_simulator/fg_model.py`, present identically in
   v1.1 and v1.2, and neither created nor worsened by the refit. Document 28's
   correctness gate does not govern it, and document 29's audit does not cover
   it — the audit sweeps for plays booked wrongly, not for parameters read
   wrongly.
2. **It is a correction candidate and it needs its own pre-registration**, per
   the process law this project has run on since document 04. The game-level
   consequence is unmeasured here on purpose: measuring it first and writing the
   round afterwards is the thing document 26 §9 warned about.
3. **If the refit is adopted, this should be fixed in the same ship or the refit
   is only half-consumed.** The refit's `delta_cubic` is *larger* than the
   incumbent's (−0.081 against −0.068), so shipping a corrected posterior into an
   uncorrected read side would carry slightly more of this error, not less.

**The check that found it was pre-registered as a formality.** §9d asked for it
in one sentence, as a round-trip on the centres. It is the strongest argument in
this document for writing down cheap checks whose expected result is "fine".

### 14g. §9d.1 and §9d.2 — the two obligations to document 26

**The refit alone enlarges the Gate A violation, exactly as §12 predicted.**

| Component | Population | Incumbent mean \|luck\| | Refit | Change |
|---|---|---|---|---|
| Field goal | **blocked (192)** | 3.360 EPA | **3.486 EPA** | **+0.125** |
| Field goal | not blocked (10,539) | 0.940 EPA | 0.898 EPA | −0.042 |
| Extra point | **blocked (110)** | 0.941 EPA | **0.961 EPA** | **+0.019** |
| Extra point | not blocked (12,708) | 0.108 EPA | 0.097 EPA | −0.011 |

A blocked field goal is scored `realized = 0` against a make probability that
just went up, so it books more luck; every other kick books slightly less,
because a higher `p_make` sits closer to the outcome that usually happens.
**Adopting the refit without document 26's correction buys a better model and a
2.82-point false credit that grows to 2.93 points.**

**Document 26's Gate P-3 floor, recomputed:**

| Posterior | Median 89% DTW half-width on the 287 games |
|---|---|
| Incumbent (v1.2) | **1.6250 pp** — reproduces document 26 §4 to four decimals |
| **Refit** | **1.4409 pp** |

The harness reproducing document 26's published floor exactly is the check that
this number is comparable to it. **The floor falls by 11.3%** under the refit,
which is the direction document 26 §9 expected and did not size.

**What this does *not* say.** Document 26's candidate measured 1.167 pp against
the old floor. Its statistic under the refit is **not measured here** and is
expected to *rise*, because the rows the correction removes are 3.7% larger. Both
numbers move and this document deliberately measures only the one that can be
measured without looking at the candidate. **Re-measuring the candidate is task 4
of this phase and it happens only if the maintainer approves both this refit and document
28's gate.**

### 14h. Defects added or discovered by this round

| Defect | Evidence | Status |
|---|---|---|
| **The simulator discards `delta_cubic`** | §14f: 55–59 yd field goals mispriced by +6.80 pp; 2,117 kicks off by more than a point | **Open, and the largest defect this round found.** Present in v1.1 and v1.2. Fixing it is a correction candidate with its own pre-registration |
| **The simulator discards `delta_xp` and `lambda_xp`** | §14f: every extra point mispriced by −0.98 pp on average | **Open.** Same round as the cubic; the two are one code path |
| **Document 05b's published `w = 0.285` has no derivation in the repository** | §14c: the recomputation could only be done as an arithmetic implication under a stated assumption | **Open.** Any document quoting `w` for the field-goal row is quoting a number nothing regenerates |
| **§7b's threshold-drift reasoning was applied to the wrong quantity** | §14a: the bound moved 34% where the argument predicted 0.6% | **Closed by disclosure.** The reasoning was about the standard error and the threshold is a percentile of 400 simulations. No verdict depends on it |
| **§10 asserted `w` would not move** | §14c: `sigma_kicker` rose 12.7% | **Closed by correction**, stated in §14c rather than edited into §10 |
| **The §9c impact numbers are computed through the defective read side** | §14f | **Accepted and stated.** Both arms share the defect, so the comparison is what the product would print |
| **Blocked kicks remain in `components.py`'s empirical swing tables** | Unchanged from §12 | **Open, deliberate.** That is document 26's correction, not this one |

### 14i. What the maintainer is being asked

**One decision, and one flagged item that is not part of it.**

> **Adopt the refitted make-probability posterior as the simulator's field-goal
> model?** It passes every gate document 05b ever imposed on this model, on the
> same curve form. It raises `p_make` by 1.33 pp on average and by more the
> longer the kick, moves the median game by 0.081 points, and flips 18 of 2,761
> verdicts. It also raises kicker skill by 12.7%, which moves document 05 §3's
> `w` from 0.285 to about 0.336, and it removes document 05b §11's claim that an
> extra point is easier than a field goal from the same distance.

- **If yes**, the ship is a v1.3 on the document 19 template, and §10's list of
  documents to edit applies with §14c's correction folded in.
- **If no**, the incumbent posterior stays, `research/outputs/trace_fg_refit.nc`
  stays as the record, and the blocked-kicks row stays on document 05b's defect
  register with these numbers attached.

**Separately, and not part of that decision:** the read-side defect of §14f is
larger than the one this round fixed and it is in shipped code today. It needs a
round of its own, and it should probably come before anything else in this
project.
