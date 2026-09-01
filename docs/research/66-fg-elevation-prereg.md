# 66 — Stadium elevation in the make-probability model, pre-registered

*Written 2026-08-31, **before `research/81_fg_elevation.py` existed**. The power
calculation and the null bound run in `research/81a_fg_elevation_power.py`,
committed at `abf9beb` with its results in
`research/outputs/81a_fg_elevation_power.json`; not one fit in that file touches
a real outcome. Committed to git before any fit produces a number.*

*Inputs: document 05b (the model being changed, its priors, its gates and its
defect register), document 05b §10–§11 (the weather round, whose gate
construction this one mirrors), document 27 (the refit that produced the
incumbent posterior the product reads), document 21 (Gate A), document 28 (the
correctness gate), document 30 §5a (the round-trip rule).*

*Tier: **model change** under the change-proposal template. A new covariate, a
new prior and a new arrow into the make probability — the generative story
gains a node, so every section of the template is filled.*

---

## 1. One-page story

### The question

Thin air is less drag, and less drag is a longer kick. Denver's field sits a
mile above sea level; Mexico City's sits 2,000 feet above Denver. The
make-probability model in document 05b conditions on distance, roof, wind,
temperature and the kicker, and on **nothing about the air the ball flies
through beyond the weather reading**. Every kick in the data is priced as if it
were kicked at 569 feet, the population mean.

> **Does stadium elevation belong in the make-probability model, and if it does,
> how much does it move a kick and how many kicks does it move?**

The question is asked now because it is the first thing a reader of the article
will ask. Denver is the most famous kicking environment in the sport, and a
model that prices a 53-yarder at Empower Field exactly like a 53-yarder at
MetLife owes the reader either a coefficient or a reason.

### The answer

**Unseen.** No arm of this study has been fitted. The power calculation in §6
has run — on simulated outcomes only — and it already constrains what the answer
can be: this design resolves a Denver effect of **5 percentage points** at 45
yards with power 0.91, and one of **2 pp** with power 0.47. If the true effect is
small, the honest outcome of this round is "too small for ten seasons to size",
not "zero".

### Five things to hold onto

1. **The covariate is nearly a Denver dummy, and that is the central limitation.**
   Of 23,247 kicks, **712 (3.06%) were kicked above 3,000 feet, and 680 of those
   are Denver**. Mexico City contributes 32, São Paulo 12, Munich 14. Whatever
   `beta_elev` measures, it is measured almost entirely on one stadium, and
   §3 states plainly what that does and does not license.
2. **Las Vegas is what keeps elevation from being collinear with roof.** Allegiant
   Stadium is a **dome at 2,030 feet**, 451 kicks. Without it, "high" and
   "outdoors" would be nearly the same column and the roof effects already in the
   model would soak up any altitude signal.
3. **Elevation is identified *within* a kicker, not just between kickers.**
   70.4% of the variance in kick elevation is within kicker-season: Denver's
   kicker also kicks eight road games, and 92 kicker-seasons contain at least one
   Denver kick. The hierarchy will still shrink some of the effect into the
   kicker term — §3 gives the direction — but it cannot absorb all of it.
4. **The gate is mirrored from Gate W-3, not invented here.** Wind's gate asked
   the 89% *upper* bound to clear a bound built from a true-zero simulation.
   Elevation's expected sign is positive, so the gate asks the 89% *lower* bound
   to clear the mirror-image bound — and, because the bound lands slightly below
   zero, **also to clear zero**, which is the binding clause.
5. **Nothing ships on this document.** If the gates pass, the refit is a
   v1.4-class change with its own round, its own round-trip check and its own
   ledger-impact report. This round's output is a coefficient, a materiality
   count and a verdict. The maintainer decides what happens to it.

### Statistic convention

Posterior means with 89% equal-tailed intervals, as everywhere else in this
project. Make-probability effects are quoted in percentage points at 45 yards,
the distance the weather round used, so the numbers are comparable across rounds.

---

## 2. DAG edit

The incumbent (document 05b §10 §2, as refitted by document 27) with the
addition marked `NEW`:

```
   alpha      beta     gamma    delta_cubic        sigma_kicker
     |          |        |          |                    |
     |          |        |          |       kicker[k] ~ Normal(0, sigma_kicker)
     |          |        |          |                    |
     v          v        v          v                    v  (x lambda_xp)
   logit p = alpha + beta*c + gamma*c^2/100 + delta_cubic*c^3/1000
             + kicker[k] * (1 or lambda_xp)
             + roof[level]
             + beta_wind * (wind - 8.0219) * has_weather
             + beta_temp * (temp - 57.9898) * has_weather
             + delta_xp * is_extra_point
             + beta_elev * (elev_kft - 0.5687)                    NEW
                              |
                              v
                    made ~ Bernoulli(p)

   where c = distance - 40, and elev_kft is a deterministic function of
   stadium_id via src/nfl_simulator/data/stadium_elevation.py           NEW
```

**Nodes added:** `beta_elev`, one scalar. **Arrows added:** `stadium_id →
elev_kft → logit p`. **Nothing is removed**, and no existing prior changes.

**Where inference is cut.** Unchanged — the joint posterior is handed to the
simulator, which draws from it. One new cut of the same kind the weather round
introduced: the **centring constant** 0.5687 kft is a property of the fitted
sample and would have to travel with the posterior, exactly as the wind and
temperature centres do.

`elev_kft` is **not** a latent node. It is a lookup, and a lookup with no
uncertainty attached: the table gives 5,280 feet for Denver and the model
believes it exactly. That is the right call at this resolution — the stadium
bowl is deeper than the disagreement between sources — but it is stated so that
nobody later reads a tight `beta_elev` interval as including measurement error
in the elevations themselves.

**Emergent behaviour to watch.** `beta_elev` and `kicker[k]` compete for the same
signal on eleven kicker-seasons that took more than 40% of their kicks in Denver.
Partial pooling shrinks the kicker term toward zero, which pushes the shared part
of the signal into `beta_elev`; the road games push back. The direction of the
residual bias is stated in §3 and a diagnostic for it is Gate E-6.

---

## 3. Mechanism story

**The defect this addresses is not yet in document 05b's register**, and this
proposal opens it: *the model conditions on the weather reading but not on the
air itself, so every kick is priced at the population's mean elevation of 569
feet — including the 680 kicked a mile up and the 32 kicked at 7,280 feet.*

**Why this change should move that number.** Drag on a football scales with air
density, and air density at 5,280 feet is about 83% of sea level. A kick that
would just fall short at sea level carries further in Denver — the effect is on
*distance*, and distance is the strongest term in this model. The simulator
books `(made − p) × swing` as luck on every field goal. If `p` is the
mean-elevation probability and the kick was in Denver, a make is credited as
partly lucky when it was partly the air, and a miss is charged as worse than it
was. The sign is systematic, not noise: it runs one way for every kick in Denver
across ten seasons, and the other way for the 12,625 kicks below 500 feet.

**What would make it fail.** Three named failure routes, each with a reading:

- **`beta_elev`'s interval does not clear the null bound and zero.** Ten seasons
  cannot size the effect. Per the §6 power table that means *"no effect as large
  as about 4 pp at Denver"*, never *"no effect"*, and the defect stays open with
  a sharper statement than it has now.
- **The elevation-cell calibration gate fails.** A *linear* term in elevation is
  the wrong shape. The physically motivated alternative is that altitude shifts
  the effective distance rather than the log-odds, and §7 names that fallback in
  advance.
- **The effect appears but lives entirely in Denver.** Gate E-6 refits with
  Denver removed. If the remaining 32 kicks in the covariate's upper reach cannot
  hold the coefficient up — and 32 kicks almost certainly cannot — then the
  honest report is that we measured *a Denver effect* and labelled it elevation.

**Why a linear logit term is the primary form, and not a distance interaction.**
The handoff asked for the choice to be stated. Two reasons, and one concession:

1. **A logit-linear term is already distance-dependent in the units that matter.**
   A constant log-odds shift moves a 20-yard kick priced at 99.4% by 0.06 pp and a
   50-yard kick priced at 68% by 2.4 pp — a factor of forty — because the logit
   link compresses at the ceiling. The claim "elevation should matter more on
   long kicks" is *already* what the simple form predicts. An explicit
   interaction is not needed to produce it.
2. **It is the house form.** Wind and temperature — the other two "air"
   covariates — enter as linear log-odds shifts. Adding a third one the same way
   means a difference in the fitted answer is attributable to the covariate
   rather than to a functional form chosen for it alone.
3. **The concession:** the physics acts on yards, not log-odds, and the two
   forms disagree in the tail. That disagreement is exactly what Gate E-2's
   elevation × distance cells are built to see, and the distance-shift form is
   pre-registered in §7 as the named fallback — reached by a gate failure, not by
   preference after seeing the numbers.

**Confounds not resolved by this design, stated before the fit:**

| Confound | Status |
|---|---|
| **Denver-the-stadium vs Denver-the-altitude** | **Not separable.** 680 of the 712 high-altitude kicks are at one site. Anything peculiar to Empower Field — its holder, its bowl, its wind pattern — loads onto `beta_elev`. Gate E-6 is the only check available and it is weak. |
| **Kicker shrinkage** | Partly handled. 70.4% of elevation variance is within kicker-season, so the terms are separately identified; residual shrinkage biases `beta_elev` **toward zero**, the conservative direction. |
| **Attempt selection at altitude** | Handled by conditioning. Denver's coaches attempt longer kicks *because* the ball flies, and distance is in the model. But the extra long attempts sit in the range where the cubic tail is least determined, so `beta_elev` could absorb curve misspecification. Gate E-4 (distance calibration preserved) is the check. |
| **Weather correlation** | Handled. Denver is cold and dry; temperature and wind are both already in the model. |

---

## 4. Compute cost and inference plan

- **Arms:** the elevation arm, plus a no-elevation control refitted **inside each
  cross-validation fold** (the full-data control already exists on disk as
  `trace_fg_refit.nc` and is reused for the materiality comparison, not refitted).
- **Observations:** 23,247 (10,539 field goals + 12,708 extra points) —
  document 27's population, blocked kicks excluded, because that is the posterior
  the product reads. **Parameters:** ~444.
- **Engine:** NUTS via nutpie, as document 05b §5. Unchanged geometry plus one
  well-scaled column.
- **Configuration:** 4 chains, 1,000 tune, 1,000 draws, `target_accept = 0.9` —
  document 05b's, unchanged, so a diagnostic difference is attributable to the
  model rather than to the sampler.
- **Parameterization:** non-centered on `kicker[k]`, still a **ruling** for
  document 05b §5's reason.
- **Fit budget:** 12 NUTS fits — 1 full elevation arm, 10 cross-validation fits
  (5 folds × 2 arms, Gate E-5), 1 Denver-excluded arm (Gate E-6). Document 05b §4
  measured this model at under five minutes a fit on essentially this `n`, so the
  round is budgeted at **about an hour of wall clock**.
- **Efficiency levers considered.** The expensive part — 2,400 logistic fits for
  the null bound and the power table — is a plain Newton-Raphson logistic rather
  than the hierarchy, which is what made it affordable (8 seconds, `research/81a`).
  Its bias direction is the safe one and is stated in §6. Five folds rather than
  ten is a deliberate halving of the cross-validation cost: the quantity being
  compared is a mean over 23,247 held-out kicks, and five folds already gives
  every kick a held-out prediction.
- **Parallel arms:** none. The fits run in sequence in one process, so there is
  no isolation mechanism to name and no cross-arm comparison rule to state.

---

## 5. The elevation table

`src/nfl_simulator/data/stadium_elevation.py`, committed before this document at
`c360e78`, with `tests/test_stadium_elevation.py`.

- 42 rows, one per `stadium_id` appearing in 2016–2025 play-by-play. The test
  suite fails if the cache ever contains an id the table does not.
- Site elevations in feet, rounded to the nearest 10, from public geographic
  references, with the city named in a comment per row. **No credential, no
  private source, no scraped table.**
- Rounding is deliberate. The covariate is in *thousands* of feet, so a 50-foot
  disagreement between sources is 5% of one prior standard deviation.
- `elevation_ft` **raises** on an unknown stadium. Sea level is a real value in
  this table — MetLife is 10 feet — so a silent default would be
  indistinguishable from a correct row.

The distribution the covariate actually has, since it is the whole design:

| Band | Kicks | Stadiums |
|---|---|---|
| ≥ 3,000 ft | 712 (3.06%) | Mexico City 32, **Denver 680** |
| 1,500–3,000 ft | 477 (2.05%) | São Paulo 12, **Las Vegas 451**, Munich 14 |
| 500–1,500 ft | 9,433 (40.6%) | Glendale, both Atlantas, Minneapolis, Orchard Park, Kansas City, Charlotte, Pittsburgh, Indianapolis, Green Bay, Chicago, Detroit, Cleveland, Arlington |
| < 500 ft | 12,625 (54.3%) | the remaining 22 |

Kick-weighted mean **0.5687 kft**, sd 0.9514 kft. 89 of 2,761 games were played
above 3,000 feet.

---

## 6. Prior, and the power behind the gate

| Site | Prior | Plain-language meaning |
|---|---|---|
| everything in document 05b §10 §6 | **unchanged** | The incumbent is not being re-argued |
| `beta_elev` | `Normal(0, 0.10)` | Log-odds per 1,000 feet. At one standard deviation, Denver's 5,280 feet moves the log-odds by 0.53 — about **+7.5 pp** at 45 yards, comparable to the `Normal(0, 0.5)` the roof levels get, and generous against a literature effect nobody puts above ~10 pp. Centred on zero: no sign is imposed, and a negative posterior would be a real result |

**Power** (`research/81a_fg_elevation_power.py`, seed 20260831, 400 datasets per
scenario, simulating from the document 27 incumbent's posterior means with
`beta_elev` set to a known value and refitting):

| True effect: make-rate gain at 45 yd, mean elevation → Denver | `beta_elev` | **Power** |
|---|---|---|
| 1 pp | +0.01347 | 0.228 |
| 2 pp | +0.02748 | 0.472 |
| 3 pp | +0.04212 | 0.640 |
| 4 pp | +0.05745 | 0.780 |
| **5 pp** | **+0.07357** | **0.907** |
| 6 pp | +0.09059 | 0.983 |

> **Minimum detectable elevation effect: a 5 pp make-rate gain at 45 yards
> between a mean-elevation stadium and Denver.** A 4 pp effect is a coin-flip
> better than nothing at 0.78; a 2 pp effect the design cannot resolve. That
> limitation is recorded here rather than discovered later.

The power fits use a plain logistic **without** kicker effects while simulating
data **with** them. The direction is the safe one: unmodelled kicker spread
inflates the residual, so true power is at or **above** the table. Document 05b
§6 recorded the same trade for wind, and said so.

Unlike document 05b §6's null, this one simulates with the incumbent's **roof
effects and extra-point offset switched on** rather than zeroed, because
elevation and roof are correlated in this design — Allegiant is a dome at 2,030
feet — and the faithful version of that collinearity is the one worth powering
against.

---

## 7. Pre-registered gates

**Gate E-1 — sampler health.** Zero divergences, `r_hat < 1.01`,
`ess_bulk > 400`, `ess_tail > 400`. Document 05b's Gate FG-1 rule and its
fallbacks; raising `target_accept` to quiet a warning remains forbidden.

**Gate E-2 — elevation calibration.** The largest standardized miss across
**elevation cells** — the four elevation bands of §5 crossed with 10-yard
distance bins, cells holding at least 100 attempts (11 cells) — compared against
its own posterior predictive distribution. **Pass:** observed at or below the
94.5th percentile. Identical in construction to Gates FG-2 and W-2.
**Documented fallback on failure:** replace the linear log-odds term with the
**distance-shift form** — `c = (distance − s · elev_kft) − 40` with
`s ~ HalfNormal(1.0)` yards of effective shortening per 1,000 feet — and refit,
reporting both arms. Named now so reaching for it later is execution rather than
improvisation.

**Gate E-3 — is the elevation effect resolvable?** **Pass:** the 89% lower bound
on `beta_elev` is above **both** −0.00790 — the 90th percentile of what this
design produces when the truth is exactly zero, the mirror of Gate W-3's
construction — **and zero**. The null bound alone lands at −0.60 pp of Denver
gain, i.e. slightly on the wrong side of zero, so **zero is the binding clause**
and the gate is strictly stricter than a 10% false-positive rule. A failure means
*"no elevation effect as large as about 4 pp at Denver"*, never *"no elevation
effect"*, per §6.

**Gate E-4 — the distance curve still works.** The Gate FG-2 statistic,
recomputed on 5-yard distance bins over the whole population. **Pass:** at or
below the 94.5th percentile of its own reference. Adding elevation must not break
what already passed.

**Gate E-5 — held-out log-loss must not worsen.** Five-fold cross-validation
**grouped by `game_id`**, so a held-out kick's own game never appears in
training. Both arms — with and without the elevation term — are refitted inside
each fold and scored by mean log-loss per held-out kick, using the posterior mean
probability. **Pass:** the elevation arm's held-out log-loss is **less than or
equal to** the no-elevation arm's. The difference is reported with a paired
bootstrap over the 2,761 games, 2,000 resamples, as an 89% interval; the interval
is **reported, not part of the pass rule**, because the pass rule the maintainer asked for
is a direction, not a significance test. Grouping by game rather than by kick is
the point: elevation is constant within a game, so a kick-level split would let a
Denver kick be predicted from another kick in the same Denver game.

**Gate E-6 — does the effect survive without Denver? Reported, no pass rule.**
Refit with all 680 Denver kicks removed and report `beta_elev` with its 89%
interval. The remaining upper reach of the covariate is 32 kicks in Mexico City,
12 in São Paulo, 14 in Munich and 451 in Las Vegas, so **this arm is expected to
be uninformative and its interval to be wide** — that is the point. **Reporting
rule, committed now:** a claim that this round measured *elevation* rather than
*Denver* requires the Denver-excluded interval to exclude zero on its own. If it
does not, the finding is reported as a Denver effect with an elevation
interpretation attached, and the distinction is stated in those words.

**Gate E-7 — materiality. Reported, no pass rule.** With the incumbent
`trace_fg_refit.nc` as the comparison, on all 23,247 kicks: how many kicks move
by **≥ 1 percentage point** of make probability, how many by ≥ 2 pp, the
distribution of the shift by elevation band, and the count of *games* holding at
least one kick that moved ≥ 1 pp. This is the number that tells a reader whether
any of it mattered.

---

## 8. Long-fit downtime plan

Nothing runs in parallel. The expensive job of this round — the null bound and
the power table — completed before this document was written, in 8 seconds. The
twelve NUTS fits run in sequence in one process for about an hour; the human wait
is filled by drafting document 67's results skeleton and the round's results
file, neither of which can contain a number until the fits land. **The article is frozen for the duration**, per the handoff.

---

## 9. Kill and rollback

- **On Gate E-3 failure:** elevation is **not adopted**. `trace_fg_refit.nc`
  stays the simulator's field-goal posterior, `src/nfl_simulator/fg_model.py` is
  untouched — it has no elevation parameter to leave inert, because this round
  does not add one — and §3's defect is opened in document 05b's register with
  the power table attached. The branch `feat/fg-elevation` keeps the elevation
  table, the power script and the fit script as the record.
- **On Gate E-2 failure:** apply the §7 distance-shift fallback, refit, report
  both arms.
- **On Gate E-5 failure with Gate E-3 passing:** the effect is resolvable in
  sample but does not generalize. **Not adopted**, and reported as exactly that —
  the two gates disagreeing is more informative than either alone.
- **On success:** nothing merges on this document either. A passing result opens
  a **v1.4-class round** with its own pre-registration, which would own: the
  `beta_elev` read side in `FieldGoalModel`, the elevation centring constant
  travelling with the posterior, the document 30 §5a round-trip check between the
  fit and the read side, the ledger and DTW impact reports, and the
  re-render of every figure that prints a make probability. **Downstream
  consumers, named:** `src/nfl_simulator/render.py` (which loads
  `trace_fg_refit.nc`), `research/79_render_all.py`, `research/80_writeup_
  figures.py`, and the frozen article draft on `docs/community-writeup`.

---

## 10. Disclosure

Building the elevation table required listing the 42 stadiums and their kick
counts, and the power calculation required the elevation distribution in §5 —
that is how the "680 of 712 high kicks are Denver" fact and the Las Vegas
dome-at-altitude fact were found. **No make rate, raw or adjusted, was computed
by elevation before this document was committed**, and the §7 thresholds come
from a simulation whose outcomes are generated under a known truth, so nothing in
them could have been moved by the answer. The exposure is recorded here rather
than left unsaid, as document 05b §10 §10 and document 08 §7 record theirs.

---

## 11. Constants appendix

| Constant | Value | Where it lives |
|---|---|---|
| `RANDOM_SEED` (power) | 20260831 | `research/81a_fg_elevation_power.py` |
| `RANDOM_SEED` (fit) | 20260831 | `research/81_fg_elevation.py` |
| `DATASETS` | 400 per scenario | `research/81a_fg_elevation_power.py` |
| chains / tune / draws / `target_accept` | 4 / 1000 / 1000 / 0.9 | inherited, document 05b §5 |
| `DISTANCE_CENTRE` | 40.0 yards | inherited, `research/14_fg_weather_model.py` |
| **elevation centre** | **0.5687 kft** | computed from the fitted sample, `research/81a` |
| `beta_elev` prior | `Normal(0, 0.10)` | §6 |
| **Gate E-3 null bound** | **−0.00790** | `research/outputs/81a_fg_elevation_power.json` |
| Gate E-3 binding clause | lower bound > 0 | §7 |
| Gate E-2 / E-4 percentile | 94.5th | inherited, document 05b §7 |
| `MIN_CELL_ATTEMPTS` | 100 | inherited, `research/14_fg_weather_model.py` |
| Gate E-5 folds | 5, grouped by `game_id` | §7 |
| Gate E-5 bootstrap | 2,000 resamples over 2,761 games | §7 |
| Gate E-7 materiality floor | 1 pp of make probability | §7 |
| Denver elevation | 5,280 ft | `src/nfl_simulator/data/stadium_elevation.py` |
| Mexico City elevation | 7,280 ft | same |
| Incumbent posterior | `trace_fg_refit.nc`, cubic arm | document 27 |
| Wind / temp centres | 8.0219 mph / 57.9898 °F | inherited, `fg_refit_summary.json` |

*Scripts: `research/81a_fg_elevation_power.py` (null bound and power, run before
this document), `research/81_fg_elevation.py` (the fit, written after it).
Results: document 67.*
