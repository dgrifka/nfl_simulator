# 09 — The remaining coin-flip candidates, pre-registered

*Written 2026-08-17, **before any of these models was fit**. Power calculation:
`research/12_coinflips_power.py`, results in
`research/outputs/12_coinflips_power.json`. Committed to git before
`research/12_coinflips.py` produces a result.*

*Inputs: documents 01–08, all settled. The process laws hold: **Gate A before
Gate B**, and a power number attached to every threshold before it is
committed.*

---

## 1. One-page story

### The question

Phase 2 neutralized two things — fumble recovery and field goals — and Phase 3's
sequencing round (document 08) closed the largest untested channel inside the
core block. What remains is a short list of **specific, nameable events** that
look like coin flips and have never been checked:

- a **drop** — a catchable ball that hits the ground
- a **fourth-down conversion**, and a **two-point conversion**
- an **onside kick recovery**
- an **extra point**

This document decides each one, and after it there is no candidate left that a
public product could be asked about and this project could not answer.

### How it answers, in one paragraph

Two gates, in the order document 05 §2 fixed and for the reason it gave. **Gate
A asks whether there is a branch point at all** — a moment where the outcome is
resolved by a mechanism outside either team's control, conditional on the state
both teams created. It is a mechanism argument, not a statistic, and it comes
first because *no persistence statistic can detect the absence of a branch
point.* Only candidates that survive Gate A reach **Gate B**, which is the
beta-binomial hierarchy documents 03 and 04 already validated, run at the real
denominators with a threshold taken from a simulated null.

### Five things to hold onto

1. **Gate A does most of the work here, and it disqualifies three of the five
   candidates before any model is fit.** That is the gate functioning, not the
   gate being lazy — document 05 §3's penalty row exists to show what happens
   when arithmetic runs without it.
2. **Only two candidates have a real branch point: onside kicks and extra
   points.** One is a loose ball in a scrum, the other is a ball in flight —
   structurally the fumble case and the field-goal case respectively.
3. **Three candidates are unresolvable at this sample size**, and the power table
   says so in advance rather than the fit discovering it afterwards.
4. **The default on "unresolvable" is deny.** A component that cannot have its
   `w` estimated cannot be neutralized, because `w` is measured, not chosen
   (document 05 §1).
5. **The measurements are reported for the Gate-A failures anyway.** Document 05
   §3 prints a parenthesised `w` for penalties for the same reason: an argument
   you can see the numbers behind is one a reader can disagree with.

### Statistic convention

Posterior means with 89% equal-tailed intervals, matching documents 03, 05, 05b
and 08. Population SD is the standard deviation of the Beta distribution over
*true* entity rates, and it is always read against the **simulated null bound**
— never against zero, per document 05 §8's closing defect.

---

## 2. Gate A — the branch-point argument, candidate by candidate

> **Is there a moment where the outcome is resolved by a mechanism outside
> either team's control, conditional on the state both teams created?**

### Drops — **FAIL**

A drop is a receiver's hands failing on a ball FTN's charter judged catchable.
There is no post-hoc branch: the pass arrives, and a player either secures it or
does not. Whatever variation sits inside the "catchable" envelope — a ball an
inch high, a defender's hand arriving late — is either the quarterback's
placement or the defense's play, and both are football that the two teams
produced.

This is the same structure as a false start, which document 05 §2 uses as its
canonical Gate A failure: the charting *describes a player's action*, it does
not record a coin resolving afterwards.

**The tempting counter-argument, and why it does not survive.** Drop rate is
famously unstable year to year, and low persistence is exactly what a coin flip
looks like. But document 05 §2 is explicit: *"Persistence statistics cannot
detect the absence of a branch point. Only the mechanism story can, which is why
it goes first."* A statistic that cannot tell a coin from an inconsistent
receiver must not be allowed to decide which one it is looking at.

### Fourth-down conversion — **FAIL**

A played-out sequence of blocking, running and tackling. Document 08 §6 already
made this argument for third and fourth down together, when it ruled the
late-down sequencing measure out of the ledger: *"there is no coin to replace
with its expectation, and no `swing` value to book, because there are no two
branches — there is a continuum of outcomes produced by football."*

### Two-point conversion — **FAIL**

A scrimmage play from the two-yard line. Identical to the row above; the only
thing that distinguishes it is that it is scored differently.

### Onside kick recovery — **PASS**

A kicker deliberately drives the ball into the ground so that it bounces
unpredictably ten yards downfield, and both teams converge on it. The ball is
oblong, the bounce is not controlled by anyone, and whoever falls on it gets it.

**This is structurally the fumble case**, which is the reason fumble recovery is
the one component this project neutralizes in full. The kick itself is a skill
— placement, spin, how hard it is driven — and that skill stays in `core`,
exactly as *choosing* to attempt a 55-yard field goal stays in `core`. Only the
bounce is the branch.

### Extra point — **PASS**

A ball in flight from 33 yards. Identical in structure to a field goal, which
document 05 §3 admits at partial neutralization. The kick is mostly the kicker;
the drift is not.

### Gate A summary

| Candidate | Branch point | Reaches Gate B? |
|---|---|---|
| Drops (team grain) | **Fail** — the receiver's hands, not a coin | No |
| Drops (receiver grain) | **Fail** — same | No |
| Fourth-down conversion | **Fail** — a played-out sequence | No |
| Two-point conversion | **Fail** — a played-out sequence | No |
| **Onside kick recovery** | **Pass** — a loose oblong ball in a scrum | **Yes** |
| **Extra point** | **Pass** — a ball in flight | **Yes** |

---

## 3. Data

- **Grain of a row**: one entity-season — team-season, receiver-season or
  kicker-season depending on the candidate.
- **Sources**: `data/pbp/*.parquet` 2016–2025, and `data/ftn/*.parquet`
  2022–2025 for the drop columns.

| Candidate | Entities | Opportunities | League rate | Median n |
|---|---|---|---|---|
| Drops, team-season | 128 | 56,211 catchable balls | 4.947% | 437 |
| Drops, receiver-season | 897 | 47,151 catchable balls | 4.912% | 44 |
| Fourth-down conversion | 320 | 7,090 attempts | 52.85% | 21 |
| Two-point conversion | 308 | 1,302 attempts | 47.62% | 4 |
| Onside kick recovery | 237 | 599 kicks | 9.683% | 2 |
| Extra point, kicker-season | 427 | 12,818 attempts | 94.41% | 31 |

### Facts that must be defensible by name

- **The drop denominator is catchable balls, not targets.** A ball nobody could
  have caught is not a drop opportunity, and using targets would make a team
  whose quarterback throws badly look like a team that drops passes. This is the
  same move document 03 made for interceptions by conditioning on
  interception-worthy throws — and it inherits the same caveat: `is_catchable_ball`
  and `is_drop` are **human charting judgments** with no published inter-charter
  reliability, and charter noise biases the estimate **toward finding no skill**.
- **Receiver-seasons are filtered to 20+ catchable balls.** Below that the grain
  is a list of names, not a measurement. 897 of 1,934 receiver-seasons survive.
- **Onside kicks are identified by a text match on `desc`.** nflverse carries no
  onside flag in the play-by-play, so `desc.contains("onside")` is the only
  available identifier. It found 599 of 28,274 kickoffs (2.1%), and the resulting
  9.68% recovery rate is close to the publicly reported post-2018-rule-change
  figure — which is corroboration, not proof. A missed onside kick that the
  description phrases differently would be silently absent. **Recorded as a
  defect.**
- **Extra points are at a fixed 33 yards for 98.5% of attempts.** The 2015 rule
  change moved the snap back, and the handful at 28, 38 and 48 yards are
  penalty-adjusted. This is why document 05b §2 excluded them from the
  *distance-curve* fit and why they need a separate treatment rather than being
  poured into it.
- **Two-point attempts have a median of 4 per team-season.** That is thinner than
  the 15 fumbles that sank document 04's Gate 2, and §4 confirms what it implies.

---

## 4. The power calculation

*Ran first, on the same instrument document 05 §7 used: 400 simulated datasets
per scenario at the **real denominators**, fitted with the exact grid posterior
(`research/_betabinom_grid.py`), recording the 89% upper bound each fit
produces.*

Note the direction, which is the reverse of a skill hunt. A coin-flip candidate
is confirmed by showing its entity spread is **small**, so the gate has the form
*"the 89% upper bound is below X"*. Power is then the chance of correctly
**rejecting** that when a real effect exists — and **a candidate with low power
cannot be neutralized on the strength of a pass, because it would have passed
anyway.**

| Candidate | Null bound (90th pct) = threshold | Power at 5% rel | **at 12.5% rel** | at 25% rel | Resolvable? |
|---|---|---|---|---|---|
| Drops, team | **0.698 pp** | 0.233 | **0.865** | 1.000 | **Yes** |
| Drops, receiver | 1.081 pp | 0.098 | **0.365** | 0.985 | **No** |
| Fourth down | **4.977 pp** | 0.278 | **1.000** | 1.000 | **Yes** |
| Two point | 10.538 pp | 0.145 | **0.292** | 0.930 | **No** |
| **Onside recovery** | 9.317 pp | 0.147 | **0.115** | 0.158 | **No** |
| **Extra point, kicker** | **1.840 pp** | 1.000 | **1.000** | *impossible* | **Yes** |

12.5% relative is the reference throughout this project: it is document 04's
pooled-judgment-penalty figure, the yardstick document 05 §7 already adopted for
"an effect this project would call real".

### The onside row deserves a sentence of its own

Its power is **flat and near the false-alarm rate at every effect size** — 0.147,
0.115, 0.158, 0.282 across a tenfold range of true spread. That is not a weak
design; it is a design with no ability to discriminate whatsoever. With a median
of **two onside kicks per team-season**, the beta-binomial cannot separate a
league of identical teams from a league of wildly different ones.

This matters because onside is one of only two candidates that *passed* Gate A.
Having a genuine branch point is necessary and not sufficient: document 05 §1's
rule needs `w = n/(n+κ)`, and 599 kicks cannot estimate `κ`.

### Extra points: two scenarios are arithmetically impossible

At a 94.41% league rate the maximum possible population SD is
√(0.9441 × 0.0559) = 22.98 pp, so a "25% relative" spread of 23.60 pp cannot
exist. The rows are printed as impossible rather than silently dropped. Power at
the reference is 1.000, so this candidate is resolved either way it lands.

---

## 5. Pre-registered gates

Committed before any result exists.

### Gate C-1 — sampler health

**Pass rule:** zero divergences, `r_hat < 1.01`, `ess_bulk > 400`,
`ess_tail > 400` on every fit, plus agreement with the exact grid posterior to
within 0.01 pp. Document 05 §8 established the grid cross-check as an
independent convergence check, and it is cheap.

### Gate C-2 — is the entity spread negligible? *(one per candidate)*

**Statistic:** the 89% upper bound on the population SD of true entity rates.

**Pass rule:** below that candidate's threshold in §4 — the 90th percentile of
what this design produces when the truth is *exactly zero*. By construction a
skill-free league clears it 90% of the time.

**Passing means the entity spread is at most what a coin-flip league produces**,
i.e. `w ≈ 0` and, for a Gate-A survivor, full neutralization at the league rate.
**Failing means entities genuinely differ**, i.e. partial neutralization at the
entity's shrunk rate.

### Gate C-3 — is the result interpretable? *(the honesty gate)*

**Pass rule:** power at the 12.5% reference is at least **0.80**, per §4.

Fails for drops-at-receiver-grain (0.365), two-point (0.292) and **onside
(0.115)**. For those three, neither outcome of Gate C-2 may be reported as a
finding.

### The decision rule, committed in advance

| Candidate | Gate A | Gate C-3 | **Treatment in v1** |
|---|---|---|---|
| Drops, team | Fail | Pass | **None.** Measured and reported; no branch point |
| Drops, receiver | Fail | Fail | **None.** Reported with its power table |
| Fourth down | Fail | Pass | **None.** Measured and reported; no branch point |
| Two point | Fail | Fail | **None.** Reported with its power table |
| **Onside recovery** | **Pass** | **Fail** | **None — deny by default.** See below |
| **Extra point** | **Pass** | Pass | **Neutralize.** Full or partial per Gate C-2 |

### Why onside is denied despite passing Gate A

This is the one row where the two gates disagree, and the rule is committed here
rather than argued afterwards.

An onside kick has a real branch point — that is settled in §2 and it is not in
question. But document 05 §1's rule is `p(e) = w·r̂_entity + (1−w)·r̄_league`
with `w = n/(n+κ)`, and **`w` is measured, not chosen.** With 599 kicks and a
design whose power is flat at 0.115, `κ` is not estimable, so `w` is not
computable, so the rule has no value to return.

Setting `w = 0` anyway — neutralizing at the league's 9.68% — is *tempting and
defensible-sounding*, because the fumble precedent says loose-ball recoveries
carry no team skill (`w = 0.011` on 4,898 fumbles). It is still an assertion the
data in front of us does not support, and asserting it would be the same move
document 04's Gate 2 taught this project not to make.

**The honest position: onside kick recovery is a coin flip that this dataset
cannot size.** It is denied in v1, the reasoning is recorded here, and the
successor — a hierarchy that borrows strength from the fumble-recovery posterior
rather than estimating `κ` from onside kicks alone — is named as future work
needing its own pre-registration.

### Gate C-4 — what folding extra points into the kicker model requires

Extra points are the one candidate that changes the simulator, so their
downstream treatment is fixed now.

- The XP arm is fitted **jointly with the field-goal model** in step 4, sharing
  `sigma_kicker` and the per-kicker effects, with **its own intercept offset**.
  It is not poured into the distance curve: document 05b §2 excluded XPs because
  they would dominate the sample at a fixed distance, and that reasoning still
  holds. A shared kicker effect with a separate level is what "folded in" means.
- **Pre-registered check:** the XP offset must be estimated, not assumed zero.
  If an extra point priced as a 33-yard field goal were already right, the offset
  would be near zero and the separate level unnecessary — that is a real possible
  outcome and it is reported either way.
- The simulator gains an `extra_point` component with the same
  `(realized − expected) × swing` identity every other row uses, so the ledger
  keeps summing by construction.

---

## 6. Known-defect register

| Defect | Evidence | Status |
|---|---|---|
| **Onside kicks are identified by a text match** | nflverse has no onside flag; `desc.contains("onside")` finds 599 of 28,274 kickoffs | **Open.** A differently-phrased description would be silently missing |
| `is_drop` / `is_catchable_ball` are human judgments | No published inter-charter reliability, same as `is_interception_worthy` | **Open.** Biases toward finding no skill; stated wherever reported |
| Drops are a four-season measure | FTN starts in 2022, so 128 team-seasons against 320 elsewhere | **Accepted.** Same limitation document 03 recorded for Model B |
| Fourth-down conversion pools all distances | 4th-and-1 and 4th-and-8 share a denominator | **Open.** The row fails Gate A anyway, so no treatment depends on it |
| Two-point attempts are heavily selected | Teams go for two when the score chart says to, which correlates with being behind | **Open.** Fails Gate A and Gate C-3, so nothing rests on it |
| Receiver-season rows are not independent across seasons | One receiver supplies up to four rows | **Open.** Same defect document 05b §7 recorded for kicker-seasons |
| Gate A is a judgment, not a measurement | No statistic can detect the absence of a branch point | **Accepted, by design.** Stated in §2 so it is arguable rather than hidden |

---

## 7. Constants appendix

| Constant | Value | Where it lives |
|---|---|---|
| `RANDOM_SEED` | 20260817 | `research/12_coinflips_power.py`, `research/12_coinflips.py` |
| `DATASETS` per scenario | 400 | `research/12_coinflips_power.py` |
| `REFERENCE_RELATIVE` | 0.125 | this document §4, from document 04 |
| `MIN_POWER` (Gate C-3) | 0.80 | this document §5 |
| `MIN_RECEIVER_TARGETS` | 20 | `research/12_coinflips_power.py` |
| **Gate C-2 thresholds (pp)** | drops-team 0.698 · drops-receiver 1.081 · fourth-down 4.977 · two-point 10.538 · onside 9.317 · extra-point 1.840 | this document §4, from the null simulation |
| FTN seasons / pbp seasons | 2022–2025 / 2016–2025 | `src/nfl_simulator/ingest.py` |

Results are written back into this document as §8.

---

## 8. Results

*Script: `research/12_coinflips.py`. Gate A settled in §2, thresholds fixed in
§4, both committed at `f07cab2` before this script produced a result. Results in
`research/outputs/12_coinflips.json`.*

### Verdicts, stated first

| Candidate | Gate A | Population SD | 89% interval | Relative | Gate C-2 | Gate C-3 | **Treatment in v1** |
|---|---|---|---|---|---|---|---|
| Drops, team | Fail | 0.711 pp | 0.494 – 0.925 | **14.4%** | FAIL | Pass | **None** — no branch point |
| Drops, receiver | Fail | 1.018 pp | 0.616 – 1.362 | **20.7%** | FAIL | Fail | **None** — no branch point |
| Fourth down | Fail | 3.677 pp | 1.473 – 5.661 | 7.0% | FAIL | Pass | **None** — no branch point |
| Two point | Fail | 4.913 pp | 1.110 – 9.616 | 10.3% | Pass | **Fail** | **None** — no branch point |
| **Onside recovery** | **Pass** | 3.395 pp | 0.780 – 7.130 | 35.1% | Pass | **Fail** | **None** — deny by default |
| **Extra point** | **Pass** | **2.422 pp** | 1.717 – 3.087 | 2.6% | FAIL | Pass | **Neutralize partially** |

**One candidate changes the simulator: extra points.** Everything else is
denied, and four of the five denials were settled by Gate A before a model ran.

### Gate C-1 — sampler health, and a threshold that was set wrong

Every fit is clean on the sampler's own terms: **zero divergences everywhere**,
`r_hat` at most 1.0026, `ess_bulk` at least 1,377, and grid edge mass below
1.2 × 10⁻⁸ on all six.

But §5 also required agreement with the exact grid posterior **to within
0.01 pp**, and two candidates missed it:

| Candidate | NUTS | Grid | Difference | Relative | One MCSE |
|---|---|---|---|---|---|
| Fourth down | 3.6774 pp | 3.7084 pp | **0.0310 pp** | 0.84% | 0.035 pp |
| Onside recovery | 3.3950 pp | 3.3576 pp | **0.0374 pp** | 1.10% | 0.043 pp |
| *(drops, team)* | 0.7107 | 0.7086 | 0.0021 | 0.30% | 0.003 |
| *(extra point)* | 2.4221 | 2.4288 | 0.0067 | 0.28% | 0.007 |

> **Gate C-1 fails on two candidates, and the gate was the mistake.**

The last column is why. **Every difference — including the two failures — sits
inside a single Monte Carlo standard error of the posterior mean**, computed as
the posterior SD over √ESS. The chains and the grid are estimating the same
number; the gap is the sampling noise of 4,000 draws, not a disagreement about
the posterior.

The threshold was inherited from document 05 §8, where two penalty models agreed
"to within 0.0002 pp". Those were rates near 0.7% with population SDs near
0.11 pp. Applying the same **absolute** tolerance to a rate near 50% with a
population SD near 4 pp asks for agreement forty times tighter in relative
terms — which no finite number of draws would deliver.

**The threshold has not been moved, and the failure stands on the record.** This
is document 04's Gate 2 lesson repeating in miniature: a threshold set by
analogy to a previous result, without asking whether the new design could
achieve it. The corrective is the same and it is recorded in §9 — a convergence
tolerance must be **relative**, or stated per-candidate from that candidate's own
MCSE.

**No verdict in the table above depends on it.** Gate C-1 is a diagnostic on the
sampler, and its substantive components — divergences, `r_hat`, ESS, edge mass —
pass on all six fits.

### Drops are not random, and this is the round's genuine surprise

The folk claim is that drops are noise: drop rate is famously unstable year to
year, and every football-analytics writer has said so at some point.

**It is not what the data says.** At the team grain, the population SD of true
drop rates is **0.711 pp on a 4.947% league rate — 14.4% relative**, and the 89%
interval's *lower* bound of 0.494 pp already sits comfortably above the 0.698 pp
a skill-free league produces. This design had **86.5% power** at the 12.5%
reference, so it is a real detection rather than a bound scraped off the edge.

At the receiver grain the spread is larger still — **20.7% relative** — which is
the direction you would expect if hands belong to players rather than schemes,
though Gate C-3 fails there (power 0.365) so it is reported and not claimed.

In football terms: a one-SD-good receiving corps drops **4.2%** of catchable
balls where a one-SD-bad one drops **5.7%**. Over the ~440 catchable balls a
team-season sees, that is about **six extra drops a year.**

**And it changes nothing, because Gate A already ruled it out.** This is the
clearest vindication of the ordering document 05 §2 insisted on. Had Gate B run
first, a 14.4% relative spread — larger than the pooled judgment penalties, and
comparable to interceptions — would have looked like an obvious *skill* finding
and drops would have been left alone for the right reason by accident. But the
mirror case is what matters: had drops come back at 3% relative, the arithmetic
would have said "neutralize", and the simulator would have started crediting
teams for their receivers' hands. **The mechanism argument protects against
both errors; the statistic protects against neither.**

### Fourth down and two point — measured, and irrelevant to the ledger

Fourth-down conversion carries a **7.0% relative** spread, with the 89% upper
bound (5.66 pp) above the 4.98 pp null bound. Teams differ, modestly, in how
often they convert — unsurprising, since going for it on fourth down is a
decision as much as an execution, and the pooled measure mixes 4th-and-1 with
4th-and-8.

Two-point conversion "passes" Gate C-2 at 9.62 pp against a 10.54 pp threshold,
and **that pass means nothing**, which §5 committed to in advance. Gate C-3
fails at 0.292 power: with 1,302 attempts at a median of four per team-season,
this design would have passed a league with a real 12.5% spread 71% of the time.
It is the pre-registered "would have passed anyway" case, and it is reported as
uninterpretable rather than as evidence of randomness.

Neither row touches the simulator. Both failed Gate A in §2.

### Onside kicks — the honest denial

Onside recovery is the only row where a genuine branch point meets a design that
cannot size it, and it played out exactly as §5 predicted.

- The point estimate is **3.395 pp on a 9.68% rate — 35.1% relative**, the
  largest relative spread of any candidate.
- Its 89% interval, 0.780 – 7.130 pp, "passes" Gate C-2 against a 9.317 pp
  threshold.
- Its power at the reference is **0.115** — flat across a tenfold range of true
  effect sizes.

Read naively, that says "onside recovery is a coin flip, neutralize it". Read
correctly, it says nothing at all: a design that rejects the null 11.5% of the
time whatever the truth is cannot distinguish a 35% spread from a 0% one, and
the 35.1% point estimate is what 599 kicks at two per team-season produce out of
pure noise.

**Denied in v1**, per the rule committed in §5 before this ran. An onside kick
is a loose ball in a scrum and it belongs in the ledger on the mechanism; the
data simply cannot supply the `w` that document 05 §1's rule requires. The
successor — borrowing strength from the fumble-recovery posterior instead of
estimating `κ` from onside kicks alone — is named as future work with its own
pre-registration.

### Extra points — the one row that changes the simulator

> Population SD **2.422 pp**, 89% interval 1.717 – 3.087, against a 1.840 pp
> null bound. The interval's *lower* bound does not clear it, but the mean and
> upper bound do, and Gate C-2 is decided on the upper bound: **FAIL, so kickers
> genuinely differ.**

2.6% relative sounds small until it is read on a 94.41% base rate. A
one-SD-good kicker makes **96.8%** of extra points where a one-SD-bad one makes
**92.0%** — nearly a five-point gap. Over the ~31 attempts a kicker-season sees,
that is about **1.5 extra misses a year**, and a missed extra point decides real
games.

**Treatment: partial neutralization at the kicker's shrunk extra-point rate**,
which is the same treatment field goals get and for the same reason. Per Gate
C-4, the extra-point arm is fitted jointly with the field-goal model in step 4,
sharing `sigma_kicker` and the per-kicker effects with its own intercept offset,
and the simulator gains an `extra_point` component using the identical
`(realized − expected) × swing` identity.

### What this changes

1. **Document 05 §3 gains one neutralized component — extra points — and four
   explicit denials.** After this round there is no coin-flip candidate a public
   product could raise that this project has not answered.
2. **Drops are a skill finding, not a luck finding**, and the number is large
   enough to be interesting on its own terms. It is a candidate for a reported
   measure; it is not a candidate for the ledger.
3. **Onside kicks are the project's clearest "we cannot tell" row.** They are
   recorded as unresolvable rather than assumed.
4. **A convergence tolerance must be relative.** Recorded below.

### Defects added by this round

| Defect | Evidence | Status |
|---|---|---|
| **Gate C-1's grid tolerance was absolute, not relative** | 0.01 pp is 9% of a penalty-rate SD and 0.3% of a fourth-down SD; both "failures" are inside one MCSE | **New, and it is document 04's Gate 2 lesson again.** Corrective: state convergence tolerances relative to the quantity, or per-candidate from its own MCSE |
| Onside kicks cannot be sized | Power flat at ~0.12 across a tenfold range of true spread | **Open.** Needs a strength-borrowing prior, not more onside kicks |
| The drops result is four seasons of one charting vendor | 128 team-seasons, human judgments, no published reliability | **Open.** Biases toward finding *less* skill, so 14.4% is a floor |
| Fourth-down conversion pools all distances | 4th-and-1 with 4th-and-8 | **Open.** No treatment depends on it |
| Extra-point kicker-seasons are not independent across seasons | One kicker supplies up to ten rows | **Open.** Same defect document 05b §7 recorded; slightly inflates apparent spread |
