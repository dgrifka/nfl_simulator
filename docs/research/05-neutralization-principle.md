# 05 — The neutralization principle

*Written 2026-08-17, **before any Phase 2 model was fit**. This is the
simulator's foundations document: it states the one rule every component is
neutralized by, and fixes the two gates a component must pass before the rule is
applied to it. Committed before documents 06 and 07 exist.*

*Inputs: `docs/research/01`–`04`, all settled. Numbers quoted from document 04
are posterior means with 89% equal-tailed intervals.*

---

## 1. One-page story

### The question

Phase 1 ended with a classification — fumble recovery is luck, penalties are
skill, field goals and interceptions are both — and a warning that a binary
skill/luck switch is the wrong model for anything in the middle. Phase 2 has to
turn that into arithmetic that runs on a single game.

The temptation is three different mechanisms: flip fumbles, leave penalties
alone, and invent something bespoke for field goals. That would be three places
to be wrong and no way to compare them. This document says there is **one**
mechanism, and the differences between components are entirely differences in
one number that the Phase 1 models already estimated.

### The rule, in one line

> **Luck is the realized outcome minus its expectation at the entity's shrunk
> (posterior) rate.**

For a luck event `e` with a binary branch:

```
luck_epa(e) = (y(e) − p(e)) · swing(e)
```

where `y(e) ∈ {0,1}` is the branch that actually happened, `swing(e)` is the EPA
difference between the two branches, and `p(e)` is the probability of the
favourable branch **under the responsible entity's shrunk rate**. A game's
deserved EPA differential is then

```
neutralized_epa_diff = realized_epa_diff − Σ_e luck_epa(e)
```

Because each term is already measured as a deviation from an expectation, the
subtraction is exact and the ledger sums by construction. This is the same
identity `src/nfl_simulator/components.py` already uses for `fumble_luck` and
`fg_luck`; Phase 2 changes only where `p(e)` comes from.

### The one dial: where `p(e)` comes from

Every component's treatment is set by a single quantity — how far the entity's
own observed rate should be trusted over the league's. For the beta-binomial
hierarchies of document 04 that is a closed form:

```
p(e) = w · r̂_entity + (1 − w) · r̄_league        w = n / (n + κ)
```

`κ` is the concentration the Phase 1 models estimated and `n` is the entity's
opportunity count. The two familiar cases are the endpoints of this one
expression, not separate rules:

| Regime | What it means | What the rule does |
|---|---|---|
| `κ → ∞` (no entity skill) | every entity is the league mean | `w → 0`, `p(e) = r̄_league` — **full** neutralization |
| `κ → 0` (all skill) | the entity's observed rate is its true rate | `w → 1`, `p(e) = r̂_entity`, so `luck_epa → 0` — **no** neutralization |
| in between | some skill, some noise | **partial** neutralization, in exactly the measured proportion |

**This is the point of the document.** Full and partial neutralization are not
two policies requiring two justifications. They are one policy read at two
values of `w`, and `w` is not a choice — document 04 measured it.

### Five things to hold onto

1. One rule, one dial. `w = n/(n+κ)` is the whole per-component difference.
2. `w` is measured, not chosen. Moving it is a model change and needs its own
   pre-registration.
3. A component must pass the **branch-point gate** (§2) before `w` is even
   consulted. Arithmetic alone would neutralize penalties; the gate stops it.
4. The league rate is **class-specific** where the classes differ materially —
   fumble recovery ranges 40% to 76% by class, and a flat coin is wrong by up to
   26 points.
5. Uncertainty in the answer comes from two nested layers, the coin *and* the
   posterior for `p` itself. A deserve-to-win number reported without the second
   is overclaiming (document 04, closing item 4).

### Statistic convention

Posterior means with 89% equal-tailed intervals, matching document 03. The
simulator's headline number, **DTW%**, is a posterior probability that the team
deserved to win, defined in §4.

---

## 2. The gates a component must pass

Order matters. Gate A is qualitative and comes first; Gate B is the arithmetic
above and only runs on components that survive Gate A. **Gate C**, added by
amendment C-1 on 2026-08-18, governs the separate case of *correcting* a play a
shipped component already neutralizes and Gate A denies.

### Gate A — the branch-point gate

> **Is there a moment where the outcome is resolved by a mechanism outside
> either team's control, conditional on the state both teams created?**

A loose ball on the turf is such a moment: both teams caused the fumble, and
then an oblong object bounces. A kick in flight is a weaker one: the kicker
caused most of it, and then it drifts. A false start is **not** such a moment.
There is no post-hoc branch — a lineman moved, and the flag is the officials'
description of that, not a coin resolving afterward.

Failing Gate A means a component is not neutralized *at any value of `w`*.

**This gate is load-bearing, and the penalty row proves it.** Run the Gate B
arithmetic on penalties and you get `w ≈ 0.42`–`0.46` (§3), which would
neutralize almost half of every game's penalty EPA. That is a real number
correctly computed from a model that passed all its checks, and applying it
would be a serious error: it would credit a disciplined team's *good* penalty
game to luck. Persistence statistics cannot detect the absence of a branch
point. Only the mechanism story can, which is why it goes first.

### Gate B — the shrinkage gate

For components that pass Gate A, `w = n/(n+κ)` sets the degree of neutralization,
with no further discretion. `κ` comes from the component's fitted hierarchy and
`n` from the entity's opportunity count in the sample.

Two consequences worth stating before they surprise anyone:

- A component can pass Gate A and still be neutralized by almost nothing, if `w`
  is near 1. That is the correct outcome, not a failure to find luck.
- `w` is entity-specific, because `n` is. A kicker with 200 career attempts is
  trusted more than a rookie with 12, from the same `κ`. This falls out of the
  formula and requires no special case.

### Gate C — correcting a Gate A violation inside a shipped component

*Added 2026-08-18 by amendment C-1, proposed in document 28 §5 and accepted by
the maintainer. The text below is that section's, enacted verbatim.*

A **violation** is a play that a shipped component neutralizes and Gate A
denies. A correction to a violation is governed by this gate instead of by the
materiality floor. It is governed by **every other gate unchanged**.

A candidate qualifies for Gate C **only if all four of the following hold**,
each stated in a pre-registration committed before the correction is measured:

1. **It is a correction, not an addition.** Every play the candidate touches
   already carries a ledger row from a shipped component, and the candidate
   removes or re-prices rows without booking a row on any play that carries
   none today. **A candidate that adds a single new row is an omission and the
   materiality floor governs the whole of it**, including any corrective part.
2. **A violation memo.** It quotes the branch document 05 §2 admitted for the
   component, shows that the population contains plays whose outcome is
   resolved by something else, and **names what did resolve them** — a specific
   football act by a specific side. "This play is not really a coin" is not a
   memo.
3. **The memo argues the other side, and measures it where it is
   measurable.** The strongest case that the play *is* a branch is stated, and
   if it rests on a factual claim the data can settle, the claim is settled.
   Document 25 §2 is the worked example: the objection that the blocker and the
   recoverer are the same man was answered with 16 of 144.
4. **The correction is the one Gate A implies, not a free parameter.**
   Exclusion where the play has no branch; re-pricing to the correct branch
   where it has a different one. If more than one correction is Gate
   A-compatible, the pre-registration argues the choice on mechanism and
   measures both arms. **A correction whose form was chosen after seeing its
   size does not qualify.**

A qualifying candidate is then measured against:

- **Identification**, unchanged — the violating population is identifiable from
  charted fields, with the **rejected rows printed, not their count** (document
  20 §9).
- **The ledger-must-sum gate**, unchanged, including the exact row-count
  arithmetic and a check that removed luck lands in `core`.
- **The dial gate** (document 20 §5f), unchanged, wherever the correction
  assumes a `w` the data cannot read.
- **A materiality *report*, not a threshold.** Median and mean |ΔDTW|, median
  |Δ deserved margin| and verdict flips, on **both** populations document 18
  §4b requires: the games containing the violating play, and every game the
  component touches. Neither number is a pass rule. **Both are printed in the
  verdict, always.**
- **A reconciliation.** The per-event luck removed, times the number of events,
  read against the game-level movement — so that a correction whose size cannot
  be explained by its own arithmetic is visible as such.

**Verdict.** A candidate that qualifies under clauses 1–4 and clears
identification, ledger-sum and the dial gate is **correct, and is proposed for
adoption at whatever size it turns out to have**. Size never fails it. As with
every other round in this project, adoption requires the maintainer's explicit approval,
and the size report is what he approves against.

A candidate that qualifies and is **not** adopted is recorded in the
known-defect register as *a measured Gate A violation, knowingly retained*,
with both population numbers attached.

**Three things C-1 deliberately does not do** (document 28 §5): it sets no
threshold of any kind, it carries no sunset clause, and it does not touch Gate A
itself. Gate A's branch-point question is unchanged word for word; C-1 is a
procedure for acting on its answer.

---

## 3. Per-component treatment table

`w` is computed at the median opportunity count per entity in the fitted sample.

**Updated 2026-08-17 at the end of Phase 3.** Every row below is now
*neutralized*, *kept*, or *marked unresolvable with the power table that says
why*. **There are no pending rows left**, which was the closing condition of the
Phase 3 plan.

| Component | Gate A (branch point) | κ / σ | typical n | **w** | Treatment | Source |
|---|---|---|---|---|---|---|
| **Fumble recovery** | **Pass** — loose ball, nobody controls the bounce | 1,408 | 15 / team-season | **0.011** | **Full.** `p` = league rate for the fumble's *class* | 04 |
| **Field goal** | **Pass** — ball in flight, partly outside the kicker | `sigma` 0.342 | 29 / kicker-season | **0.285** | **Partial** vs that kicker's shrunk make probability at that distance, **now adjusted for roof, wind and temperature** | 05b §9, §11 |
| **Extra point** | **Pass** — ball in flight, same structure as a field goal | `sigma` 0.342 shared | 31 / kicker-season | **0.285** | **Partial** vs that kicker's shrunk extra-point probability. Population SD 2.42 pp against a 1.84 pp null bound, so kickers genuinely differ | 09 §8 |
| **Interception** | **Pass** — given an interception-worthy throw, whether it is caught is partly the defender's luck | 71.5 | 24 / team-season | **0.251** | **None in v1.** Step 3a could not attribute the spread to quarterbacks or defenses, so no entity can carry `w` | 04, step 3a |
| **Dropped pick** (interceptable throw, 2022+) | **Pass** — the ball is in the air and the defender's hands are partly outside the offence | `sigma_d` 0.258 | 22 / defence-season | — | **variant (A-3 pending a wording ruling).** A default-off component neutralizes each charted interception-worthy throw at the defence-season's posterior-sampled catch probability, **beside v1.3 and never in it**. All three of amendment A-3's own gates now pass — pricing (G-2, 0.94), materiality (G-3, 1.62 pp vs 0.56 pp) and self-fulfilment (G-1, bucket agreement 0.997 over 1,139 games, median \|ΔDTW\| between the in-sample and week-out arms 0.05 pp). Production keeps the in-sample read with that bound recorded. **Still not a treatment:** clause 3 makes defender and receiver drops one class, and the receiver mirror cleared G-4a-d in round 7 — so every computed gate now passes and what remains is the maintainer's ruling on document 52 §3's "near-random" wording (document 57 §1b) | 49, 50, 52, 53, 54, 55, 56, 57 |
| **Onside kick recovery** | **Pass** — a loose oblong ball in a scrum, structurally the fumble case | *not estimable* | **2 / team-season** | — | **None — unresolvable.** 599 kicks; power flat at **0.115** across a tenfold range of true spread. `w` cannot be measured, and document 05 §1 forbids choosing it | 09 §4, §8 |
| **Penalty (pre-snap)** | **Fail** — no post-hoc branch | 3,967 | 2,813 plays | (0.415) | **None** | 04, step 3b |
| **Penalty (judgment)** | **Fail** — officiating discretion measurably does not add noise (12.5% relative spread vs 14.0% pre-snap) | 3,243 | 2,813 plays | (0.465) | **None.** Subtype check closed: holding is *not* random | 04, step 3b |
| **Return yardage** | **Fail** — a return is a played-out sequence, not a branch resolved by nobody | — | — | — | **None.** No measurable persistence either (r = −0.014), on a test that could only have shown a large effect | step 3c |
| **Drops** (catchable ball, 2022+) | **Fail** — a receiver's hands, not a coin | `sigma` 0.63 pp | 437 / team-season | — | **variant (A-3 pending a wording ruling).** Gate A's answer is unchanged and document 09's skill finding stands: round 7 re-measured the team-season spread at **0.709 pp / 14.3% relative**, reproducing document 09 §8's 0.711 pp on the same 56,211 balls, and Gate C-2 **fails** on the conditioned residual too, so receiving corps genuinely differ. What A-3 opens is a second door beside Gate A: a default-off component neutralizes each charted catchable target at the **team-season's** posterior-sampled catch probability, beside v1.3 and never in it. All four of its gates pass — G-4a (C-3 0.877 at the charged grain), G-4b (V-1..V-8), G-4c (agreement 0.996 over nineteen folds), G-4d (2.32 pp against a 0.56 pp floor). The charged grain is the corps, not the receiver: C-3 power at the individual grain is 0.400 and carries no verdict | 09 §2, §8; 56, 57 |
| **Fourth-down conversion** | **Fail** — a played-out sequence | — | 21 / team-season | — | **None.** 7.0% relative spread | 09 §2, §8 |
| **Two-point conversion** | **Fail** — a played-out sequence | — | 4 / team-season | — | **None.** Also unresolvable (power 0.292) | 09 §2, §8 |
| **Sequencing — red-zone placement** | **Fail** — no branch point; a continuum of outcomes produced by football | — | 160 plays / team-season | — | **None.** It *is* luck (split-half r = −0.034 at 87% power) but it has no coin to replace. Reported separately, never as ledger rows | 08 §6, §9 |
| **Sequencing — late-down placement** | **Fail** — same | — | 241 plays / team-season | — | **None.** Also luck (r = +0.000 at 92% power) | 08 §6, §9 |
| **Sequencing — leverage timing** | **Fail** — same | — | 1,065 plays / team-season | — | **None.** And it is **skill**: r = +0.180, surviving a game-state control at +0.144 | 08 §9 |

Parenthesised `w` values are shown to make §2's argument concrete. They are not
used: those rows failed Gate A.

**Three kinds of "none" appear in that table, and they are not the same claim.**

1. **No branch point** (penalties, returns, drops, fourth down, two point, all
   three sequencing rows). Gate A rules them out at any value of `w`. Six of
   these were settled by mechanism before a model was fit.
2. **A branch point, but no estimable entity** (interceptions, onside kicks).
   The rule needs `w = n/(n+κ)` and the data cannot supply `κ`. Denied by
   default, with the power table attached.
3. **Nothing left to decide** — no row is in this state, which is the point.

**The field-goal row is the one place the dial is genuinely entity-specific.**
For every other component `w` is one number, because the opportunity counts are
similar across entities. Kickers are not: `w` runs from **0.064** at the 10th
percentile of kicker-seasons to **0.377** at the 90th. A kicker with a handful
of attempts is neutralized almost to the league curve; a full-season starter
keeps most of their own record. That falls straight out of `w = n/(n+κ)` with no
special case, which is the clearest demonstration that §1's rule is doing real
work rather than restating three policies in one notation.

### Fumble recovery — full, but class-specific

`w = 0.011` means a team-season's observed recovery rate carries about one
percent of the information about its true rate; the league does the rest. This
is why document 04 found Buffalo's 10-of-12 shrinking to 48.0%. In simulator
terms the entity term vanishes and `p(e) = r̄_league(class(e))`.

The class split is not optional. From document 01:

| Fumble class | n | League own-recovery rate |
|---|---|---|
| run, normal play | 1,149 | 40.3% |
| pass, normal play | 2,892 | 45.3% |
| kickoff | 182 | 46.2% |
| punt (muffed return) | 672 | 64.4% |
| run, aborted snap | 946 | 76.2% |
| pass, aborted snap | 68 | 100% |

A flat 50/50 would book a fake 26-point bad-luck charge against every offense
that recovered its own botched snap. `src/nfl_simulator/components.py` already
carries this table; the simulator consumes it unchanged.

### Interceptions — partial, and the entity is the open question

`w = 0.251` at the team-season grain: about three-quarters of a team's observed
interception-worthy-throw conversion rate is noise, one quarter is real. But
document 04's defect register flags that a team-season pools every quarterback
who played, so *whose* skill the 14.3% relative spread belongs to is unresolved.

Neutralizing at a team-level `w` when the skill actually belongs to a specific
quarterback would systematically misprice both. Step 3a settles this with
crossed quarterback and defense grouping factors, and this row is not final until
it does. Document 04 named this the highest-value open question in the project
and, at 19% of outcome variance, it is.

### Penalties — none, and the subtype caveat

Document 04 settled the pre-snap/judgment split: both are repeatable team
traits, and the officiating-noise hypothesis was not supported. Gate A rules the
whole component out independently.

One narrower hypothesis survives and is tested in step 3b: **offensive holding
specifically may be random even though pooled judgment calls persist**, on the
theory that holding occurs on most plays and the flag is a sampling of it. If
that lands, holding is the one penalty subtype with a genuine branch point, and
this row splits. Nothing else about the penalty treatment is in question.

---

## 4. From neutralized EPA to a deserve-to-win number

### Points, anchored on the actual result

Document 01 established that EPA differential and points margin are nearly the
same quantity (r² = 0.991, only 0.8% of margin variance invisible to EPA). The
simulator uses that to express its answer on the scoreboard's scale while
staying anchored to what actually happened:

```
deserved_margin = actual_margin − (Σ_e luck_epa(e)) · points_per_epa
```

`points_per_epa` is the slope of margin on EPA differential, fit on the full
sample. Anchoring on `actual_margin` rather than predicting a margin from
neutralized EPA matters: it means a game with no luck events returns its actual
result exactly, which is the smoke test in the handoff plan's verification list.

### DTW% and its interval — two nested layers

A point estimate of `deserved_margin` answers "what was the luck worth" but not
"who deserved to win", because the counterfactual is itself uncertain. The
simulator bootstraps, and the bootstrap has two layers:

1. **Draw the rate.** Sample `(μ, κ)` — and for field goals the kicker effects —
   from the fitted posterior, giving a draw of `p(e)` for each event. This is
   document 04's closing instruction: report uncertainty from the population
   posteriors, not from point estimates.
2. **Flip the coin.** Draw `y*(e) ~ Bernoulli(p(e))` for every luck event and
   recompute the margin with those branches in place of the realized ones.

```
DTW% = P(deserved_margin* > 0)
```

over the replicates, reported with an 89% interval on the margin distribution.
A game with no luck events collapses to a degenerate distribution at the actual
margin, and DTW% is 100% or 0% — correctly, since there was nothing to adjudicate.

Layer 1 is what stops the simulator from reporting a suspiciously tight interval
around a quantity estimated from 15 fumbles per team-season.

### The luck ledger

Every event that contributes a nonzero `luck_epa` is emitted as a ledger row —
play id, component, class, `y`, `p`, `swing`, `luck_epa`, and the team charged.
The ledger's `luck_epa` column must sum to the total adjustment applied to the
margin. That is an identity, not a tolerance, and the simulator asserts it.

---

## 5. Known-defect register

| Defect | Evidence | Status |
|---|---|---|
| The entity's rate is estimated from a sample containing the game being adjudicated | A kicker's game contributes ~2 of ~100+ career attempts; a team-season's game contributes ~1 of ~15 fumbles | **Open, bounded.** Contamination is O(1/n) and always shrinks the measured luck toward zero (self-fulfilling rates), so the bias is conservative. A leave-one-game-out refit was named for Phase 3 and **was not done** — Phase 3 spent its budget on the sequencing round, the coin-flip round, weather and interval coverage. Still open, still bounded, still conservative |
| `points_per_epa` is a single global slope | Document 01 found r² = 0.991, so the residual is small but real (≈0.8% of margin variance) | **Accepted.** The residual is scoring-environment conversion, not luck being adjudicated, so it is deliberately excluded from the DTW interval |
| Interception entity is unresolved | Team-season pools quarterbacks (document 04) | **Open, and step 3a could not resolve it** — both crossed factors straddle the design's null bound. The INT row of §3 is therefore *not neutralized* in v1. Needs more FTN seasons, not a better model |
| Fumble classes are hand-assigned | Our taxonomy from `play_type` × `aborted_play`, not the NFL's | **Open.** Class list is in §3 so it can be argued with |
| Gate A is a judgment, not a measurement | No statistic can detect the absence of a branch point | **Accepted, by design.** Stated in §2 so it is arguable rather than hidden |
| Weather is absent from the FG model | Deferred per the handoff plan | **CLOSED in Phase 3** (05b §10–11). Roof, wind and temperature are in the model; 7,507 of 10,731 field-goal ledger entries were repriced, and systematically by roof (+2.8 pp indoors, −0.2 pp outdoors) |
| Simultaneous luck events are treated as independent coins | Two fumbles in one game are drawn independently | **Accepted.** They are separate physical events; correlation would have to come through the shared `p` draw, which layer 1 already supplies |
| A shipped component can neutralize a play Gate A denies, and the materiality floor is the wrong instrument for fixing it | Document 26 §8: the blocked-kick correction is right and missed a floor that the incumbent's own error inflates | **Governed by Gate C since 2026-08-18** (§2, amendment C-1, document 28 §5). A correction to such a play is measured against a materiality *report* rather than a threshold; every other gate is unchanged |

---

## 6. Constants appendix

| Constant | Value | Where it lives |
|---|---|---|
| Fumble class recovery rates | 40.3% / 45.3% / 46.2% / 64.4% / 76.2% / 100% | `src/nfl_simulator/components.py` (`fit_fumble_baseline`) |
| Fumble `κ` | 1,408.3 | `research/outputs/03_bayesian_rates.json` |
| INT conversion `κ` | 71.5 | same |
| Penalty `κ` (pre-snap / judgment) | 3,967.1 / 3,243.1 | same |
| `points_per_epa` | fit at simulator build time | `src/nfl_simulator/simulator.py` |
| Bootstrap replicates | set in step 5, power-checked | `src/nfl_simulator/simulator.py` |

Verdicts for the pending rows landed in §3 as steps 3 and 4 completed, and Phase
3 closed the remainder. **The rule in §1 never changed; only the table did** —
which was the claim this document was written to make, and it survived two
phases of components being added to it.

### Constants added or revised by Phase 3

| Constant | Value | Where it lives |
|---|---|---|
| `sigma_kicker` (weather model) | 0.342 | `research/outputs/fg_weather_summary.json` |
| `beta_wind` | −0.0213 per mph | same — 5.50 pp lost at 45 yd, calm → 15 mph |
| `beta_temp` | +0.00385 per °F | same — +2.66 pp across a 40 °F swing |
| Roof effects at 45 yd | dome +4.53 pp · closed +4.66 pp · open +7.69 pp | same |
| `delta_xp` | +0.167 log-odds | an extra point is easier than a 33-yard field goal |
| `lambda_xp` | 1.263, interval contains 1 | kicking ability transfers to extra points |
| Extra-point league rate / swing | 94.41% / 1.020 EPA | `src/nfl_simulator/components.py` (`fit_xp_baseline`) |
| `WIND_CAP_MPH` | 30.0 | `src/nfl_simulator/fg_model.py` |
| **`DEFAULT_COIN_DRAWS`** | **800** | `src/nfl_simulator/simulator.py` — a calibration constant, not a performance knob (document 10 §8) |

---

## 7. Attribution round — pre-registered

*Added 2026-08-17, **before any attribution model was fit**. Power checks:
`research/06_attribution_power.py`, results in
`research/outputs/06_attribution_power.json`. Every threshold below carries a
power number, per the process law from document 04.*

### The instrument

Each power check simulates 400 datasets at the **real denominators** under a
known true population SD, fits the beta-binomial hierarchy, and records the 89%
upper bound the fit produces. That answers the question document 04's Gate 2
never asked: *can this many observations reach the bound the threshold demands,
even when the truth is exactly zero?*

Fits use an exact grid posterior over `(mu, log_kappa)` rather than NUTS —
the marginalized model has only two free parameters, so the posterior can be
evaluated directly. It reproduces the Phase 1 nutpie fumble fit to within
0.02 pp on the reported interval (`research/_betabinom_grid.py`, `self_check`),
which is what licenses using it for thousands of power fits.

### 3a — whose skill is the 14.3% interception spread?

**Model.** Hierarchical logistic on charted interception-worthy throws,
2022–2025, with **crossed** quarterback and defense random effects:

```
logit p(picked) = intercept + qb[passer] + defense[defteam]
qb[·]      ~ Normal(0, sigma_qb)
defense[·] ~ Normal(0, sigma_def)
```

Half-Normal(1) on both scales, on the log-odds scale. **Non-centered
parameterization is a ruling, not a default**: document 04's Gate 1 failure was
exactly a centered hierarchy funnelling, and with a median of 10 throws per
quarterback-season the per-entity data is thinner here than it was there.

**Reported quantity.** `sigma_qb` and `sigma_def`, each converted to a
population SD in rate units at the league mean, with 89% intervals.

**No pass/fail gate**, following the convention document 03 §6 Gate 3 set: these
are estimation, and pre-registering a threshold for a quantity with no prior
estimate is theatre. The **reporting rule** is pre-registered instead:

- Both scales are reported with 89% intervals whatever they say.
- A claim that the spread "belongs to" quarterbacks or to defenses requires that
  entity's interval to exclude the achievable-null bound **and** the other's to
  fail to, stated in percentage points.
- If both exclude it, the spread is shared and the interception row of §3 stays
  at the team grain.

**Power, and the honesty it forces:**

| Grain | Entities | Median n | Null 89% bound (90th pct) | Power at 10% relative | Power at 21% relative |
|---|---|---|---|---|---|
| Quarterback-season | 281 | 10 | 6.85 pp | 0.475 | 1.000 |
| Defense-season | 128 | 23 | 6.03 pp | 0.738 | 1.000 |

Document 04 measured the team-level spread at 6.6 pp (14.3% relative). Both
grains resolve an effect that size, but **neither can rule out a spread below
about 10% relative**. A null result at either grain is therefore not evidence of
absence, and must not be written as one.

### 3b — is offensive holding random?

**The hypothesis**, stated in §3 before the fit: offensive holding is random even
though pooled judgment calls persist, because holding occurs on most plays and
the flag is a sampling of it.

**Model.** The same beta-binomial hierarchy as document 03, on offensive-holding
counts per team-season over plays. 320 team-seasons, 916,700 plays, 6,597
holding calls, league rate 0.7196%. **False Start** is fit as a comparison arm
on the identical denominator, so the pre-snap benchmark is like-for-like.

**Pre-registered gate — this one has a real pass rule, because it is powered:**

> **Pass** (holding is effectively random): the 89% upper bound on the
> population SD of true offensive-holding rates is **below 0.0837 pp**
> (11.6% of the league rate).

**Power check:**

| Condition | Outcome |
|---|---|
| True SD = 0 | bound lands below 0.0837 pp **90%** of the time *(threshold is set as this percentile)* |
| True SD = 5% relative | correctly rejected 36.5% of the time |
| **True SD = 12.5% relative** (the pooled-judgment figure) | **correctly rejected 99.3%** of the time |
| True SD = 25% relative | correctly rejected 100% |

This is the one attribution question the data answers cleanly. 916,700 plays is
three orders of magnitude more evidence than the 4,898 fumbles that sank
document 04's Gate 2, and the threshold is derived from the null distribution
rather than from a football argument about what 0.0837 pp feels like.

**On pass:** offensive holding splits out of the judgment class and gains a
branch-point argument to be argued on its own merits in §2. **On failure:** the
hypothesis is dead, the penalty rows of §3 stand unchanged, and no penalty is
neutralized.

### 3c — does return yardage persist?

**Primary statistic.** Split-half correlation of mean interception return yards
per defense-season, 200 random within-season splits — the identical machinery
document 02 used, so the number is comparable to the ones already on record.
319 defense-seasons, 4,304 interceptions, median 13 per defense-season.

**Detectability floor.** At 319 team-seasons the 95% interval on a correlation
has a half-width of about **±0.11**. Any true correlation below that is
indistinguishable from zero here. Document 02's middle three components sat at
r = 0.12–0.16, i.e. **only just** above this floor.

**Secondary statistic.** Pick-six rate as a beta-binomial hierarchy — the binary
form of the same question, on a league rate of 8.90%.

**No pass/fail gate for 3c, and the power check is why:**

| True SD | Relative | Power to reject "no skill" |
|---|---|---|
| 1.0 pp | 11.2% | 0.142 |
| 2.0 pp | 22.5% | 0.330 |
| 4.0 pp | 45.0% | 0.930 |

Only a spread near **45% of the league rate** — roughly three times anything
document 04 measured for any component — is reliably detectable. Committing a
threshold here would repeat document 04's mistake with the arithmetic already in
hand to know better. Both statistics are reported descriptively, with this table
attached, and the return-yardage row of §3 will most likely resolve to *"not
resolvable with ten seasons"* rather than to a verdict.

### Constants added by this section

| Constant | Value | Where it lives |
|---|---|---|
| Power-check datasets per scenario | 400 | `research/06_attribution_power.py` (`DATASETS`) |
| **3b gate threshold** | **0.0837 pp** | this section, `research/06_attribution.py` |
| 3b power at 12.5% relative | 0.993 | power check |
| 3a null bound, QB / defense grain | 6.85 / 6.03 pp | power check |
| 3c correlation detectability floor | ±0.11 | this section |
| Reference relative spread | 12.5% | document 04, pooled judgment penalties |

---

## 8. Attribution round — results

*Script: `research/06_attribution.py`. Gates pre-registered in §7 above, committed
at `c1b454f` before any of these models existed. Results in
`research/outputs/06_attribution.json`.*

### Outcomes, stated first

| Question | Outcome |
|---|---|
| 3a — whose skill is the interception spread? | **Unresolved.** Quarterbacks and defenses carry statistically indistinguishable spreads and neither clears the design's null bound |
| 3b — is offensive holding random? | **Hypothesis rejected, decisively.** Holding is as repeatable as any other penalty class |
| 3c — does return yardage persist? | **No measurable persistence**, on a test that could only have detected a large effect |

### 3a — the interception spread is shared, and the design cannot split it

The crossed model needed 3,000 tune / 3,000 draws. At the standard 1,000/1,000
it returned `ess_bulk` 289 and `r_hat` 1.0138 with **zero divergences** — the
signature of slow mixing rather than bad geometry, because the two crossed scales
trade off along a ridge the chains cross slowly. More draws is the honest fix;
raising `target_accept` to quiet the warning is what document 03 §5 forbids. The
final fit passes Gate 1 at `r_hat` 1.0064, `ess_bulk` 959, zero divergences, and
the estimates moved by less than 0.15 pp from the failed attempt — which is
itself evidence the first failure was mixing speed, not a wrong answer.

| Factor | Log-odds SD | Rate-scale SD | 89% interval | Relative | Design's null bound |
|---|---|---|---|---|---|
| Quarterback-season | 0.250 [0.090, 0.375] | **6.11 pp** | 2.24 – 9.07 | 12.6% | 6.85 pp |
| Defense-season | 0.243 [0.118, 0.352] | **5.97 pp** | 2.94 – 8.54 | 12.3% | 6.03 pp |

`P(quarterback spread > defense spread) = 0.530` — a coin flip.

**Per the reporting rule pre-registered in §7, this is a null attribution.** The
rule required one factor's interval to clear the null bound while the other's
failed to. Neither clears it: both 89% intervals straddle the bound the design
would produce under a true zero. So the honest statement is not "the skill is
shared equally" — it is **"2,999 charted throws cannot tell these two apart."**

That was foreseeable and was foreseen: §7 recorded in advance that neither grain
could rule out a spread below about 10% relative, and both estimates land right
at 12%. The consequence for the simulator is concrete — **the interception row
stays at the team grain with `w = 0.251`.** Splitting to a quarterback-specific
rate would assert an attribution the data does not support.

### 3b — offensive holding is not random, and this test had the power to say so

| Penalty | Team-seasons | Calls | League rate | Population SD | 89% interval | Relative |
|---|---|---|---|---|---|---|
| **Offensive holding** | 320 | 6,597 | 0.7196% | **0.1110 pp** | 0.0883 – 0.1327 | **15.4%** |
| False start (benchmark) | 320 | 6,044 | 0.6593% | 0.1171 pp | 0.0968 – 0.1372 | 17.8% |

> **Gate 3b: FAIL.** The 89% upper bound is 0.1327 pp against a pre-registered
> threshold of 0.0837 pp. The interval's *lower* bound, 0.0883 pp, already sits
> above the threshold.

The hypothesis was that holding is random because it occurs on most plays and
the flag is a sampling of it. It is wrong, and unlike document 04's Gate 2 this
failure is informative rather than a data-volume artifact — the power check gave
this test **99.3% power** against a 12.5%-relative effect, on 916,700 plays.
Holding at 15.4% relative sits between pooled judgment calls (12.5%) and false
starts (17.8%). Teams differ in how often they are flagged for holding, and they
differ about as much as they differ in false starts.

**Nothing in the treatment table changes.** Penalties already failed Gate A, and
the one subtype that might have earned an exception did not.

Both fits agree with the exact grid posterior to within 0.0002 pp, which is an
independent check on convergence.

### 3c — return yardage does not persist, on a test that could barely have shown it

**Split-half correlation of mean interception return yards**, 314 defense-seasons
with 6+ interceptions, 200 random within-season splits:

> **r = −0.014, 5th–95th percentile [−0.088, +0.068]**

Flat against a detectability floor of ±0.11. Comparable numbers already on
record: fumble recovery +0.055, interception EPA +0.164 (document 02).

The **pick-six rate** model needs its power table to read correctly, and this is
the clearest example in the project of why:

| | Value |
|---|---|
| League pick-six rate | 8.90% |
| Population SD | 1.37 pp, 89% interval 0.46 – 2.59 |
| Upper bound a **true zero** typically produces | **2.91 pp** |

The interval excludes zero, and that looks like evidence until it is compared
with the right reference. The observed upper bound of 2.59 pp is *below* the
2.91 pp a genuinely skill-free league produces on this many interceptions. The
lower bound never reaches zero because `log_kappa ~ Normal(4, 2)` keeps `kappa`
finite, so "the interval excludes zero" is a property of the parameterization
here, not a finding. **There is no evidence of pick-six skill.**

**Verdict for the table:** return yardage is not neutralized in v1. It fails
Gate A — a return is a played-out sequence with blocking and tackling, not a
branch resolved by nobody — and there is no persistence signal that would push
back against that reading. Neutralizing it would also require an expected-return
model conditional on field position, which is a larger build than v1 warrants.

### Defects added by this round

| Defect | Evidence | Status |
|---|---|---|
| Interception attribution unresolved | Both crossed factors straddle the design's null bound | **Open.** Needs more FTN seasons, not a better model. Blocks any player-level neutralization |
| Crossed model mixes slowly | ess_bulk 289 at 1,000 draws, 959 at 3,000 | **Closed** by more draws; recorded because the fix was not the model |
| 3c cannot resolve realistic effects | Only a 45%-relative spread is detectable | **Accepted**, pre-registered in §7 |
| `population_sd` has a non-zero lower bound by construction | `log_kappa ~ Normal(4, 2)` keeps `kappa` finite | **Open.** Interpret against the simulated null bound, never against zero |
