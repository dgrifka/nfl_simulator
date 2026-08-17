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
