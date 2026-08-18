# 16 — The overtime coin toss, pre-registered

*Written 2026-08-18, **before `research/26_overtime.py` existed**. Power and
impact calculation: `research/25_overtime_power.py`, results in
`research/outputs/25_overtime_power.json`, reproduced verbatim in §4. Committed
to git before the fit produces a number, so goalpost integrity is checkable by
commit archaeology.*

*Inputs: documents 05 (the one rule and the two gates), 09 (the shape of a
candidate round), 10 (interval coverage), 12 (the rematch test's measured
blindness), 15 (Phase 5 scouting). Process laws unchanged — **Gate A before
Gate B**, a power number attached to every threshold before it is committed, and
a relative rather than absolute convergence tolerance.*

---

## 1. One-page story

### The question

Every component this project neutralizes is a *physical* coin whose probability
had to be estimated: an oblong ball bouncing, a kick drifting. Phase 5 opens with
the one branch that is a **literal coin**, flipped by a referee, whose
probability needs no estimate at all — the overtime toss.

Document 15 measured the stakes: 155 overtime games 2016–2025, and the team with
the first overtime possession won **86 of 145 decided games (59.3%)**. That edge
is handed to one team by a coin before either team runs a play, and the shipped
simulator currently books all of it as deserved.

### How this document answers, in one paragraph

The toss passes Gate A more cleanly than anything the project has admitted, and
it does something no other component does: it makes Gate B **vacuous**. Document
05 §1's dial is `w = n/(n+κ)` — how far an entity's own record is trusted over
the league's — and a coin toss has no entity, so `p = 0.5` exactly, `w = 0` by
definition rather than by measurement. What has to be estimated instead is the
other half of the identity: `swing`, the difference in expected final margin
between receiving first and not. Because the toss is randomised, that estimate
is a **causal** one from a real experiment, the only such estimate in the
project.

### Five things to hold onto

1. **This is the project's only randomised experiment.** Every other `p` comes
   from an observational hierarchy that has to argue its way past confounding.
   Here the treatment — first possession — was assigned by a referee's coin, so
   the difference in mean margin *is* the causal effect, with no adjustment
   needed beyond balancing the 87/68 home split the toss happened to produce.
2. **Gate B does not run, and that is not a loophole.** Document 09 denied
   onside kicks because `κ` was unestimable and therefore `w` was uncomputable.
   Here `w` is not unestimable; it is **zero by definition**, because there is no
   entity whose skill could shift a coin. The two situations look similar and are
   opposite.
3. **The unit of the ledger is the hard part, not the statistics.** The ledger is
   per-play EPA and the toss is a game-level branch with no EPA. §5 converts the
   swing through `points_per_epa` so the existing identity and the existing
   bootstrap are untouched, and §6 registers the conversion as a defect.
4. **The floor is not a free parameter.** The does-it-change-anything gate is set
   at the incumbent simulator's *own* median 89% DTW interval half-width on these
   same games — **4.06 pp**. A change smaller than the uncertainty the product
   already prints is not a change worth shipping, and the number was chosen by
   the incumbent rather than by this document.
5. **The 2025 rule change cannot be tested and this is registered in advance.**
   With 16 games under the new rules, the design has power **0.243** to detect
   that the effect vanished entirely. Neither outcome of the era comparison may
   be reported as a finding.

### Statistic convention

Posterior means with 89% equal-tailed intervals, matching documents 03, 05, 05b,
08 and 09. Swings are in **points of final margin** throughout; the EPA-unit
figure the ledger stores is always the points figure divided by
`points_per_epa` = 0.8389.

---

## 2. Gate A — the branch-point memo

> **Is there a moment where the outcome is resolved by a mechanism outside either
> team's control, conditional on the state both teams created?**

### The toss — **PASS**

The state both teams created is *tied after sixty minutes*. A referee then flips
a physical coin. No player, coach or scheme touches the result; the visiting
captain calls a side, and the physics does the rest. There is no weaker reading
available: where the fumble row has to argue that an oblong ball's bounce is
uncontrolled, and the field-goal row has to concede that a kicker produces most
of a kick's flight, the toss is a randomising device introduced into the game on
purpose, precisely because the league wanted the choice made by nobody.

**This is the strongest Gate A pass in the project**, and it is worth saying
plainly because the rest of this document is about the difficulties that follow
from it, not about whether it holds.

### Three things next to the coin that are *not* the coin

Gate A is a scalpel, and its value comes from what it excludes. Three adjacent
things fail it and stay in `core`.

| Adjacent thing | Verdict | Why |
|---|---|---|
| **The election** — receive, kick, or defer | **Not the coin.** A coaching decision | The toss winner is handed an *option*; choosing well is football, exactly as choosing to attempt a 55-yard field goal is football (document 09 §2) |
| **The overtime plays** | **Not the coin.** Football | Blocking, tackling, play calling. The same line document 14 drew for punts and returns: a played-out sequence has no two branches to replace |
| **The overtime *rule*** | **Not luck at all** | Sudden-death structure is why first possession is valuable. Neutralising the toss removes the advantage the coin *handed to one team*; it does not relitigate the rulebook |

### What the toss makes vacuous: Gate B

Document 05 §1:

```
p(e) = w · r̂_entity + (1 − w) · r̄_league        w = n / (n + κ)
```

For every component so far, the interesting work was estimating `κ` and reading
`w` off it. For the toss there is nothing to estimate. A fair coin has no
entity — no team, no player, no season — whose record could carry information
about the next flip. So `r̄_league = 0.5`, the entity term is undefined rather
than small, and `w = 0` **by definition**.

Two consequences that must be stated before they are mistaken for shortcuts:

- **Document 09's onside denial does not transfer.** Onside kicks were denied
  because 599 kicks could not estimate `κ`, so `w` had no value to return. Here
  `w` has a value and it required no data. "Unestimable" and "known exactly" are
  opposite conditions that both end in a number this document did not fit.
- **The estimation burden moves to `swing`.** The identity has two unknowns and
  the toss simply relocates which one is hard. §4 powers the design for `swing`,
  not for `p`.

### Is the coin actually fair? — a check, not an assumption

`p = 0.5` is asserted from physics rather than measured, so it gets a check
rather than a gate: the share of overtime games in which the **home** team
received first should sit inside binomial noise of 50%. It is 87 of 155
(56.1%), which is 1.5 binomial standard errors — unremarkable, and handled by
the estimator in §5 rather than argued away. Per-team toss records are not
examined: at roughly five overtime games per team per decade the question is
unanswerable, and it does not need answering, because `p = 0.5` was never a
claim about teams.

---

## 3. Data

- **Grain of a row**: one overtime game.
- **Source**: `data/pbp/*.parquet`, 2016–2025, quarter 5.
- **Identification**: a game reaches overtime iff it has quarter-5 plays. The
  team with the first overtime possession is the first non-null `posteam` in
  quarter 5 — on an nflverse kickoff row `posteam` is the **receiving** team,
  verified by inspection of the play descriptions.
- **The margin is the overtime margin.** A game that reaches overtime is tied at
  the end of regulation, so the final margin `result` *is* the margin produced by
  the overtime period. No score reconstruction is needed.

### Denominators

| Quantity | Value |
|---|---|
| Overtime games 2016–2025 | **155** |
| …decided | 145 (10 ties) |
| …home team received first | 87 |
| …away team received first | 68 |
| Overtime games under the 2025 rules | 16 |
| Playoff overtime games | 10 |
| Regular-season overtime games per season | 10–23, median 15 |

### The outcome support

The final margin of an overtime game lives on nine values. The table below is the
distribution of **|margin|**, split evenly across each magnitude's two signs —
it is the *null* the power calculation draws from, and it deliberately carries no
information about which side of the coin won.

| Margin | −7 | −6 | −3 | −1 | 0 | +1 | +3 | +6 | +7 |
|---|---|---|---|---|---|---|---|---|---|
| Null weight | 0.0032 | 0.1677 | 0.2871 | 0.0097 | 0.0645 | 0.0097 | 0.2871 | 0.1677 | 0.0032 |

Mean |margin| across all 155 games is **3.80 points**, which fixes the largest
swing this support can express: 7.60 points.

### Facts that must be defensible by name

- **First possession is observed; the toss result is not.** nflverse carries no
  coin-toss field. Before 2025 the toss winner received in effectively every
  game, so the two coincide; under the 2025 rules, deferring is a live option and
  they may not. **Recorded as a defect** (§6) and it is the single largest threat
  to the component's meaning going forward.
- **Ties are kept.** Ten games ended 0. A tie is a real overtime outcome and
  dropping it would bias the swing upward by conditioning on someone winning.
- **Playoff and regular-season games are pooled**, across three different
  overtime rulebooks (15-minute sudden-death through 2016, 10-minute from 2017,
  both-teams-possess in playoffs from 2022 and in the regular season from 2025).
  §4 shows the era split is unresolvable, so pooling is not a preference — it is
  the only estimate the sample supports.
- **Document 15 already published the headline rate (59.3%).** This document was
  therefore written knowing the swing's approximate size. That is disclosed
  rather than hidden, and it is the reason §5's floor is defined as *the
  incumbent's own printed precision* rather than as a number chosen here: the
  4.06 pp figure comes from simulator v1.1's intervals, not from a judgment that
  could have been tuned to the expected answer.

---

## 4. The power and impact calculation

*Script: `research/25_overtime_power.py`; 2,000 simulated datasets per scenario
at the real denominators, fitted with the same Dirichlet posterior §5 registers.
Every posterior in this section is fitted to simulated data.*

### 4a. Can the design see the swing?

The estimator is the one §5 registers, run on data simulated at a known true
swing. A home-field nuisance of +0.274 points per group is carried through the
simulation so that the estimator's immunity to it is demonstrated rather than
assumed.

| True swing (points) | Power: 89% interval excludes zero | Mean estimate | SD of estimate |
|---|---|---|---|
| **0.0** | **0.104** *(false-alarm rate; nominal 0.11)* | +0.015 | 0.634 |
| 0.5 | 0.207 | +0.488 | 0.642 |
| 1.0 | 0.434 | +0.934 | 0.634 |
| 1.5 | 0.738 | +1.415 | 0.622 |
| 2.0 | **0.927** | +1.893 | 0.604 |
| 2.5 | 0.983 | +2.345 | 0.598 |
| 3.0 | 0.998 | +2.827 | 0.586 |

Two things to read off it. The false-alarm rate lands on its nominal value, so
the instrument is calibrated. And the estimates are unbiased to within a
hundredth of a point at every true swing despite the home-field nuisance, which
is the balancing in §5 doing its job.

### 4b. Would seeing it change anything?

The overtime branch is additive on top of the incumbent's bootstrap, so simulator
v1.1 was run once over all 155 overtime games and every candidate swing applied
to the same margin draws:

```
new_margin = margin − (received − replayed_coin) · swing
```

which is exactly what a ledger row for the toss would do — no approximation.

| Swing (points) | Median \|ΔDTW\| | Mean \|ΔDTW\| | Max \|ΔDTW\| | Games whose DTW side flips |
|---|---|---|---|---|
| 0.5 | 0.64 pp | 1.59 pp | 10.0 pp | 8 / 155 |
| 1.0 | 1.72 pp | 3.08 pp | 16.7 pp | 10 / 155 |
| 1.5 | 2.94 pp | 4.39 pp | 18.4 pp | 12 / 155 |
| 2.0 | 3.89 pp | 5.49 pp | 21.3 pp | 14 / 155 |
| **2.5** | **5.81 pp** | 7.02 pp | 23.0 pp | 16 / 155 |
| 3.0 | 7.91 pp | 8.96 pp | 31.4 pp | 19 / 155 |

**The incumbent's median 89% DTW interval half-width on these same games is
4.06 pp.** Overtime games sit near the decision boundary, so the product already
prints an interval eight times wider there than on a typical game (0.51 pp).

### 4c. The reference swing

Reading the two tables together fixes the reference effect size without a
judgment call:

> **The reference swing is the smallest simulated swing whose median |ΔDTW|
> exceeds the incumbent's own printed precision.** That is **2.5 points**, and
> the design's power at it is **0.983**.

This is the document-09 Gate C-3 test — *is the result interpretable?* — and it
passes with room to spare, which is exactly where onside kicks failed at 0.115.

### 4d. Can the 2025 rule change be tested? — **No.**

The alternative simulated here is as extreme as the question allows: the pre-2025
games carry the full swing and the 16 games under the new rules carry none.

| Pre-2025 swing (points) | Power to detect that the new rules removed it |
|---|---|
| 1.0 | 0.120 |
| 1.5 | 0.195 |
| **2.0** | **0.243** |
| 2.5 | 0.344 |
| 3.0 | 0.469 |

Even against total removal the design is blind. This is document 09's onside row
repeating in a different place, and it is pre-registered here for the same
reason: so that a difference between eras cannot be read as a finding after the
fact.

---

## 5. Pre-registered gates and the model

Committed before any result exists.

### 5a. The estimand

> `swing` = E[final margin | received the first overtime possession]
>          − E[final margin | did not]

Because the toss randomises the treatment, the difference in observed group means
is an unbiased estimate of this contrast, and by the symmetry of the two
perspectives it equals twice the mean margin taken from the receiving team's
view.

### 5b. The model

**Tier: model change** (a new component enters the ledger with its own
generative story).

**DAG edit.** One node is added upstream of the ledger and nothing existing is
rewired:

```
   coin (p = 0.5, known)  ──▶  first possession  ──▶  overtime margin
                                                            │
   [team quality, regulation play] ──────────────────────────┤
                                                            ▼
   realized margin ──▶  ledger  ──▶  deserved margin ──▶  DTW
                          ▲
        fumble / field goal / extra point rows (unchanged)
```

The arrow from *coin* to *first possession* has no parents — that is the whole
content of Gate A drawn as a graph.

**Likelihood.** For each of the two home/away groups, the nine-valued margin
support is modelled as a multinomial with a Dirichlet(α = 0.5) prior — the
Jeffreys prior for a multinomial, symmetric over the symmetrised support, so it
carries **zero prior opinion about which team the coin favours** (its prior mean
swing is exactly 0). The posterior is conjugate and therefore exact.

Why a category model rather than a normal mean: overtime margins are discrete
and strongly bimodal at ±3 and ±6. A normal likelihood would be a
misspecification with no compensating benefit, and there is no need for one —
nine categories with 155 observations is a small, exactly solvable problem.

**Home balancing.** `swing = μ_home-received + μ_away-received`, the sum of the
two group posterior means. Averaging the groups cancels home-field advantage,
which would otherwise leak in through the 87/68 split the coin happened to
produce.

**Inference.** Conjugate; no sampler, no step count, no convergence diagnostic to
run, because there is no approximation to converge. Posterior summaries come from
20,000 Dirichlet draws. **Document 09 §9's corrective is honoured differently
here:** rather than a NUTS-versus-grid tolerance, the cross-check is an
*independent estimator* — a nonparametric bootstrap of the same balanced
contrast — and the agreement gate is **relative**, not absolute (§5d).

**Compute cost.** Seconds. There is no long fit and therefore no downtime plan.

### 5c. Gate O-1 — is the swing real?

**Statistic:** the 89% equal-tailed posterior interval on `swing`.

**Pass rule:** the interval excludes zero.

**Power:** 0.927 at a 2.0-point swing, 0.983 at the 2.5-point reference (§4a).
False-alarm rate 0.104 against a nominal 0.11.

### 5d. Gate O-2 — is the estimate trustworthy?

**Pass rule, all three:**

1. The nonparametric bootstrap's 89% interval on the balanced contrast agrees
   with the posterior interval's endpoints to within **10% relative** — a
   relative tolerance, per document 09 §9's correction of the absolute one that
   failed two candidates for being unachievable.
2. Prior sensitivity: the posterior mean swing under α = 1.0 and under
   α = 0.01 (an approximate Bayesian bootstrap) both sit within **10% relative**
   of the α = 0.5 result.
3. The home-balanced estimate and the naive pooled estimate are both reported.
   Disagreement between them does not fail the gate; concealing it would.

### 5e. Gate O-3 — the does-it-change-anything floor

**Statistic:** the median |ΔDTW| the fitted swing produces across all 155
overtime games, computed exactly as §4b computed it.

**Pass rule:** **≥ 4.06 pp**, the incumbent simulator's own median 89% DTW
interval half-width on those same games.

**Rationale, stated before the result:** extra points passed every gate document
09 set and are worth 0.115 EPA of luck on the average attempt. This project does not
need a second component that is defensible and invisible. A change that is
smaller than the interval the product already prints cannot be seen by a reader
of the product, and shipping it would add machinery, a defect register entry and
a maintenance burden for nothing.

**The floor is close to the expected answer and that is disclosed.** §4b shows a
2.0-point swing landing at 3.89 pp and a 2.5-point swing at 5.81 pp; document 15's
59.3% implies a swing in that vicinity. The floor is not moved for it. If the
fitted swing lands between those rows, this gate is genuinely a coin flip of its
own, and the answer is whatever the arithmetic returns.

### 5f. Gate O-4 — the era question is not asked

Per §4d, the design has power 0.243 against total removal of the effect by the
2025 rules. **The 2016–2024 versus 2025 comparison is reported as a descriptive
with its power attached, and neither outcome may be treated as a finding or
change the treatment.** The successor is named in §6: revisit at 60 games under
the new rules, which is roughly four seasons away.

### 5g. The decision rule, committed in advance

| Gate O-1 | Gate O-3 | **Treatment** |
|---|---|---|
| Pass | Pass | **Neutralize.** A game-level `overtime_toss` component enters the ledger at `p = 0.5`, full neutralization, with the fitted swing |
| Pass | Fail | **None.** Reported as a measured, real, and immaterial effect — the extra-point lesson applied prospectively |
| Fail | — | **None — deny by default.** `w` is known but `swing` is not distinguishable from zero, so the rule has nothing to book |

### 5h. What shipping it would mean, fixed now

- **Component name** `overtime_toss`, `event_class` `"overtime coin toss"`,
  charged to the team that received first, `realized` = 1 if the home team
  received, `expected` = 0.5 exactly (a constant, not a posterior).
- **Swing in ledger units** = fitted swing in points ÷ `points_per_epa`, signed
  to the home perspective, so `deserved_margin = actual − Σ luck_epa · ppe`
  returns the points figure unchanged and the ledger keeps summing by
  construction.
- **One row per overtime game**, attached to the first quarter-5 play's
  `play_id`.
- **The swing's own uncertainty is not propagated**, matching every other
  component (fumble class swings and field-goal bin swings are fixed at their
  empirical means too). Because the toss's swing is estimated from 155 games
  rather than thousands of events, this shortcut is proportionally larger here,
  so a **sensitivity arm** reports DTW with the swing redrawn from its posterior
  on every posterior draw, alongside the shipped fixed-swing number.
- **Kill and rollback.** The component lands behind its own baseline object, as
  extra points did; failing a gate leaves the branch and the report as the record
  and changes nothing in `main`.

---

## 6. Known-defect register

| Defect | Evidence | Status |
|---|---|---|
| **First possession is a proxy for winning the toss** | nflverse has no coin-toss field. Before 2025 the winner received essentially always; the 2025 rules make deferring rational | **Open, and the most serious.** If deferring becomes common the component measures the wrong branch. Named as the trigger for revisiting |
| **The 2025 rule change is untestable** | Power 0.243 against total removal (§4d) | **Open.** Pre-registered as unanswerable; revisit at 60 new-rule games |
| **The ledger's EPA units are a conversion for this row** | The toss has no play-level EPA; the swing is measured in points and divided by `points_per_epa` | **Open.** `luck_epa` for this component is a converted quantity summed alongside measured ones |
| **The swing is a population average** | A team with a great offense plausibly gains more from receiving; 155 games cannot estimate heterogeneity | **Open.** Stated wherever the component is reported |
| **Three overtime rulebooks are pooled** | 2016 / 2017–2021 / 2022+ playoffs / 2025 regular season | **Accepted.** §4d shows no split is resolvable |
| **The support is the observed magnitudes, symmetrised** | A 2-point (safety) or 8-point overtime margin has never occurred in this window and carries no prior mass | **Open.** Would need a wider support if one occurs |
| **Swing uncertainty is not propagated into DTW** | Inherited from every existing component | **Open**, with a sensitivity arm quantifying it (§5h) |
| **The rematch test cannot validate this** | Document 12 measured it as nearly blind below ~20% damage; this component touches 155 of 2,761 games | **Accepted.** No validation via rematch is attempted or claimed |

---

## 7. Constants appendix

| Constant | Value | Where it lives |
|---|---|---|
| `RANDOM_SEED` | 20260818 | `research/25_overtime_power.py`, `research/26_overtime.py` |
| `N_DATASETS` per scenario | 2,000 | `research/25_overtime_power.py` |
| `DIRICHLET_ALPHA` | 0.5 (Jeffreys) | both scripts, §5b |
| `SWING_GRID` (points) | 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 | `research/25_overtime_power.py` |
| Reference swing | **2.5 points** | §4c, derived not chosen |
| Power at the reference | **0.983** | §4a |
| Gate O-3 floor | **4.06 pp** | §4b, the incumbent's own median half-width |
| Gate O-2 relative tolerance | 10% | §5d |
| `points_per_epa` | 0.8389 | `research/outputs/model_metadata_v11.json` |
| Simulator settings for the impact run | 200 posterior draws, 800 coin draws, seed 20260817 | matches `research/15_simulator_v11.py` |
| Overtime games / decided / ties | 155 / 145 / 10 | §3 |

Results are written back into this document as §8.

---

## 8. Results

*Script: `research/26_overtime.py`. Gate A settled in §2, thresholds fixed in
§4–§5, all committed at `3fbbc59` before this script existed. Results in
`research/outputs/26_overtime.json`, per-game impact in
`research/outputs/26_overtime_games.parquet`.*

### The verdict, stated first

> **The overtime coin toss is worth 2.05 points of final margin, the estimate is
> real and stable, and it does not reach the materiality floor. The component is
> not shipped.**

| Gate | Statistic | Result | Verdict |
|---|---|---|---|
| **A** | Is there a branch point? | A referee's coin | **Pass** (§2) |
| **O-1** | 89% interval on the swing excludes zero | +2.049 pts, **[+1.037, +3.065]** | **Pass** |
| **O-2** | Independent estimator and priors agree within 10% relative | worst gap 6.98% / 5.67% | **Pass** |
| **O-3** | Median \|ΔDTW\| ≥ 4.06 pp | **3.93 pp** | **FAIL by 0.13 pp** |

Per the decision rule committed in §5g — *pass O-1, fail O-3 → no treatment* —
**the toss is measured, reported, and left out of the ledger.** Nothing in
document 05 §3's treatment table moves.

### Gate O-1 — the swing is real

| Estimate | Swing (points) | 89% ETI |
|---|---|---|
| **Primary** — home-balanced Dirichlet(0.5) | **+2.049** | +1.037 – +3.065 |
| In EPA units (÷ 0.8389) | +2.443 | — |

Receiving the first overtime possession is worth about **two points of final
margin**, which is roughly two thirds of a field goal. The interval excludes zero
comfortably, and §4a puts the design's power at a 2.0-point swing at 0.927, so
this is a detection the instrument was built to make rather than one it stumbled
into.

### Gate O-2 — the estimate is trustworthy

| Check | Result | Tolerance | Verdict |
|---|---|---|---|
| Nonparametric bootstrap of the same contrast | +2.161, [+1.110, +3.202] | 10% relative on the endpoints | **6.98%** — pass |
| Prior α = 1.0 | +1.937 | 10% relative on the mean | 5.5% — pass |
| Prior α = 0.01 (≈ Bayesian bootstrap) | +2.164 | 10% relative on the mean | 5.7% — pass |
| Naive, unbalanced estimate *(reported, not gated)* | +2.176 | — | 6.2% above the balanced figure |

The naive figure is higher than the balanced one by 0.13 points, which is home
advantage leaking through the 87/68 split the coin produced — small, in the
direction §5b predicted, and removed by construction rather than by adjustment
after the fact.

### Gate O-3 — and it does not matter enough

Applying the fitted 2.049-point swing to all 155 overtime games in simulator
v1.1:

| Statistic | Value |
|---|---|
| **Median \|ΔDTW\|** | **3.93 pp** |
| Floor (incumbent's own median 89% half-width) | 4.06 pp |
| Mean \|ΔDTW\| | 5.56 pp |
| Max \|ΔDTW\| | 21.44 pp |
| Games whose DTW side flips | 14 / 155 (9.0%) |

**The failure is 0.13 pp, so the first question is whether it is noise.** It is
not. Redrawing the replayed coin eight independent times moves the median across
a range of **3.87 – 3.96 pp** — the whole spread sits below the floor, and the
gap to it is three times the spread. The gate is decided, not undecidable.

**What the failure does and does not say.** The median overtime game moves less
than the interval the product already prints on it. The *mean* game moves more
(5.56 pp), because the distribution is strongly right-skewed: most overtime games
are already far enough from the boundary that two points cannot reach it, while a
minority sitting on the boundary move by up to 21 pp. Fourteen games over ten
years change which team the simulator says deserved to win. That is a real
effect on a real minority of games, and the committed statistic was the median.

### The sensitivity arm — the shortcut this component could not afford

§5h required reporting DTW with the swing redrawn from its posterior on every
posterior draw, because the toss's swing rests on 155 games where a fumble class
rests on thousands of events.

| Median 89% DTW interval half-width, overtime games | Value |
|---|---|
| Incumbent (no overtime component) | 4.06 pp |
| With the component, swing fixed at its posterior mean | **3.69 pp** |
| With the component, swing redrawn per posterior draw | **5.00 pp** |

Read the second and third rows together. Shipping the component at a fixed swing
would have made the printed intervals **narrower** (3.69 pp) while the honest
accounting makes them **wider** (5.00 pp) than the incumbent's. The fixed-swing
convention every other component uses would have quietly bought a 1.3 pp
reduction in stated uncertainty that the data does not support. The point
estimate itself barely moves — median gap 0.54 pp — so this is a defect about
intervals, not about the number.

**This finding outlives the candidate.** It is the first quantification anywhere
in the project of what the unpropagated-swing shortcut costs, and it is recorded
against the register in §6.

### Reported, deciding nothing (§5f)

| Split | n | Swing | 89% ETI |
|---|---|---|---|
| 2016–2024 | 139 | +2.242 | +1.146 – +3.347 |
| **2025 (new rules)** | **16** | **+0.688** | **−1.756 – +3.069** |
| Regular season | 145 | +1.848 | +0.784 – +2.894 |
| Playoffs | 10 | +2.539 | −0.530 – +5.447 |

The 2025 row is the one a reader will want to interpret, and §4d pre-registered
that it cannot be: at 16 games the design has power **0.243** to detect that the
new rules removed the effect entirely, and the interval here is 4.8 points wide
against an effect of about 2. **A lower point estimate under the new rules is
exactly what an unchanged effect looks like a quarter of the time.** Nothing is
concluded from it, and the successor trigger stands at 60 new-rule games.

---

## 9. What this round changes, and what it teaches

### The ledger is unchanged

No component is added. `docs/research/05` §3's treatment table stands as it was
after Phase 3: fumble recovery in full, field goals and extra points partially,
nothing else.

### Three things worth carrying forward

1. **A gate that fails by a tenth of a point still fails, and the way to earn
   that verdict is to measure the noise rather than argue about it.** The eight
   replayed-coin replicates cost one line and turned an uncomfortable 0.13 pp
   into a settled question. Any future gate that lands this close should do the
   same before anyone writes a sentence about it.
2. **The floor did its job, and its job was unglamorous.** Document 09 shipped
   extra points — defensible, fully gated, and worth 0.115 EPA on the average
   attempt. This round produced a *larger* per-event effect on a *smaller* set of
   games and stopped it at the door. A materiality gate that never rejects
   anything is decoration.
3. **The median was the committed statistic, and a right-skewed impact
   distribution is where median and mean disagree.** The mean (5.56 pp) clears
   the floor and the median (3.93 pp) does not. This document does not relitigate
   the choice; it records that the next candidate's floor should state *which*
   summary it uses and why, because for a component that touches few games and
   moves a minority of them a lot, the two summaries answer different questions.

### The defect register gains a measured entry

The "swing uncertainty is not propagated" row was inherited from every existing
component and had never been sized. It now has a number: on this component, the
shortcut understates the median 89% DTW half-width by **1.31 pp** (3.69 against
5.00). That is a general finding about the simulator's interval convention, not
about overtime, and it belongs to whichever future round revisits document 10's
coverage work.

### What would reopen this

- **60 overtime games under the 2025 rules** (roughly four seasons), which would
  make the era question answerable and could move the swing in either direction.
- **A coin-toss field in the source data**, which would replace the
  first-possession proxy and let the *election* be separated from the *toss* —
  the largest open defect in §6.
- **A change to the materiality floor itself**, which would be a pre-registered
  amendment with its own justification, not a re-run of this document.
