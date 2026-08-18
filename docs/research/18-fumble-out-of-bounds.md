# 18 — Fumbles out of bounds, pre-registered

*Written 2026-08-18, **before `research/30_fumble_oob.py` existed**. Class
tables, entity-spread power and impact: `research/29_fumble_oob_power.py`,
results in `research/outputs/29_fumble_oob_power.json`, reproduced in §3–§5.
Committed to git before the Gate F-2 fit produces a number.*

*Inputs: documents 03/04 (the fumble hierarchy and its `w`), 05 (the one rule
and the two gates), 09 (the grid instrument and the reference effect size), 15
(Phase 5 scouting), 16 (the materiality floor), 17 (the identification gate).*

---

## 1. One-page story

### The question

The simulator's oldest and most confident component is fumble recovery,
neutralized **in full** because document 04 measured `w = 0.011` — a team's own
recovery record carries about one percent of the information about its true
rate. That component quietly conditions on something: it only looks at fumbles
that **somebody recovered**.

602 fumbles in ten seasons — 9.3% of them — never got recovered by anybody. They
skipped out of bounds, and by rule the fumbling team kept the ball. The simulator
currently books every one of those as **deserved**.

### The answer, stated first

**It is the same coin, one branch earlier, and the fix is to widen the branch
rather than add a component.** Instead of asking *who recovered the loose ball*,
the component asks **did the fumbling team end up with the ball** — and going out
of bounds is one of the two ways to keep it.

| What widening does | Effect |
|---|---|
| Population | 5,914 recovered fumbles → **6,505 fumbles** |
| League retention rate | 52.13% → **56.48%** |
| Swing (pass/live) | 4.088 → 4.268 EPA |
| Games where a new ledger row appears | **536** |
| Median \|ΔDTW\| on those games | **1.65 pp** against a 0.62 pp floor |
| Median \|Δ deserved margin\| on those games | **1.76 points** |
| Entity-spread resolution at the 12.5% reference | **1.000** (incumbent: 0.975) |

**This is the first Phase 5 candidate that would change the shipped numbers**,
and unlike candidates 1 and 2 it clears its materiality floor and its
identification requirement without difficulty. What remains genuinely unsettled
is Gate F-2 — whether the wider branch still shows negligible team spread, and
therefore whether *full* neutralization survives.

### Five things to hold onto

1. **This is a correction to a hidden conditioning, not a new component.** The
   incumbent's population was selected on the outcome of the very branch this
   document is about. Nothing in document 05's rule changes; the population it
   runs on does.
2. **Out of bounds is not a small branch — it is a cheap one.** A fumble that
   goes out of bounds is worth about +0.05 EPA to the fumbling team; a live
   fumble is worth about −3.0 EPA once its recovery coin is averaged. Rare
   events with a three-point swing are exactly what this project exists to
   adjudicate.
3. **The class structure survives and sharpens.** Aborted snaps go out of bounds
   **3.0%** of the time against 10–11% for everything else — the ball squirts
   backwards into a crowd of two, nowhere near a sideline. A flat out-of-bounds
   rate would have been wrong by a factor of three on the most common fumble
   class in the data.
4. **Widening buys resolution rather than costing it.** More opportunities per
   team-season (median 20 against 18) raise the power to detect a real team
   spread from 0.975 to 1.000 at the reference. The wider branch is the better
   measured one.
5. **The materiality question was answered during design and this is
   disclosed.** §5 says plainly which gate was unseen when it was written
   (F-2) and which was already known (F-3).

### Statistic convention

Posterior means with 89% equal-tailed intervals. Population SD is the standard
deviation of the Beta distribution over *true* entity rates, read against the
simulated null bound and never against zero, per document 05 §8.

---

## 2. Gate A — the branch-point memo

> **Is there a moment where the outcome is resolved by a mechanism outside either
> team's control, conditional on the state both teams created?**

### Whether a loose ball crosses the sideline — **PASS**

The state both teams created is: the ball is on the ground, near some particular
spot, with players converging. What happens next is an oblong object bouncing.
**This is not analogous to the fumble-recovery coin; it is the same bounce, one
instant earlier.** Document 05 §2 admitted the recovery branch on exactly this
reasoning, and there is no version of that argument which admits *who falls on
the ball* while refusing *whether the ball reaches the sideline first*.

The strongest argument against is worth stating: a ball fumbled at the numbers
cannot go out of bounds, and a ball fumbled at the sideline usually does, so the
branch is partly determined by *where* the fumble happened. That is true, and it
is a statement about the **state**, not about the branch. Where the fumble
happened is football that both teams produced, and it belongs in `core` — the
same way the field-goal component prices the *kick* and leaves the decision to
attempt a 55-yarder in `core`. §6 records the absence of field position from the
class definition as a defect, inherited from the incumbent rather than
introduced here.

### One thing next to the branch that is *not* the coin

| Adjacent thing | Verdict | Why |
|---|---|---|
| **Deliberately batting a loose ball out of bounds** | **Not the coin.** A defensive play | A player who swats a fumble through the sideline to deny the offense a recovery is making a play. The data cannot separate this from a ball that simply rolled out — **registered as a defect**, and the direction of the error is stated in §6 |

---

## 3. Data

- **Grain of a row**: one fumble.
- **Source**: `data/pbp/*.parquet` 2016–2025.
- **Population**: `fumble == 1` and `fumbled_1_team` is not null — **6,505
  fumbles** (two of 6,507 carry neither a recovery team nor an out-of-bounds
  flag and are dropped).
- **`retained`** is 1 when the fumbling team still has the ball afterwards:
  either it recovered, or the ball went out of bounds.

### The class table, incumbent beside widened

| Class | n (live) | p (live) | swing (live) | n (all) | **p (all)** | **P(out of bounds)** | swing (all) |
|---|---|---|---|---|---|---|---|
| pass/live | 2,892 | 0.4530 | 4.088 | 3,226 | **0.5096** | **10.54%** | 4.268 |
| run/live | 1,149 | 0.4030 | 5.007 | 1,273 | **0.4611** | **9.98%** | 4.963 |
| run/aborted | 946 | 0.7622 | 4.228 | 974 | 0.7690 | **2.98%** | 4.229 |
| punt/live | 672 | 0.6443 | 4.967 | 757 | **0.6843** | **11.36%** | 5.000 |
| kickoff/live | 182 | 0.4615 | 5.166 | 201 | **0.5124** | 9.45% | 5.140 |
| pass/aborted | 68 | 1.0000 | 4.280 | 68 | 1.0000 | 0.00% | 4.280 |
| field_goal/live | 3 | *pooled* | *pooled* | 4 | *pooled* | 25.00% | 7.590 |
| punt/aborted | 2 | *pooled* | *pooled* | 2 | *pooled* | 0.00% | 3.926 |

Out-of-bounds fumbles per season: 59, 60, 66, 56, 72, 59, 53, 72, 57, 48 — flat,
with no rule change or trend behind them.

### Facts that must be defensible by name

- **The 11 conflicted plays go to the recovery.** Eleven fumbles carry both an
  out-of-bounds flag and a named recovering team. A named recovering team is the
  more specific fact, so those are treated as live recoveries.
- **Aborted snaps are the reason a flat rate would be wrong.** 2.98% against
  10.54% is a factor of 3.5 on 974 plays, and the incumbent's class structure
  already exists to carry exactly this kind of difference.
- **The two thin classes get a class-specific swing under widening where the
  incumbent gave them a pooled one.** `field_goal/live` goes from 3 plays to 4,
  which is enough for both branch means to become non-null, so the pooling
  fallback stops firing. This affects **six plays in ten seasons** and is an
  artefact of mirroring the incumbent's code exactly rather than a design
  choice. Registered in §6.

---

## 4. Power and impact

### 4a. Is the entity spread on the wider branch resolvable?

The exact grid instrument of document 09 §4, run at the real denominators — 400
simulated datasets per scenario, recording the 89% upper bound on the population
SD.

| Branch | Entities | Opportunities | League rate | Median n | Null bound (90th pct) | Power at 5% rel | **at 12.5% rel** |
|---|---|---|---|---|---|---|---|
| **Retention, all fumbles** | 320 | 6,505 | 56.48% | 20 | **5.260 pp** | 0.278 | **1.000** |
| Recovery, live only *(incumbent)* | 320 | 5,914 | 52.13% | 18 | 5.474 pp | 0.223 | 0.975 |

Widening the branch **improves** the instrument. This is worth stating because
the natural worry runs the other way — that folding in a rarer outcome would
dilute the measurement — and the arithmetic says the extra 591 opportunities more
than pay for it.

### 4b. Does it change anything?

The shipped component and the widened one were run through the same bootstrap on
the same games, with the field-goal and extra-point draws generated from their
own seeded generators in both arms, so the difference is the fumble rows and
nothing else.

| Population | Games | Median \|ΔDTW\| | Mean \|ΔDTW\| | Max | Side flips | Median \|Δ deserved margin\| |
|---|---|---|---|---|---|---|
| **Games with an out-of-bounds fumble** | **536** | **1.65 pp** | 6.36 pp | — | **31** | **1.76 pts** |
| All games with any fumble | 2,497 | 0.10 pp | 2.07 pp | 41.0 pp | 48 | 0.27 pts |

The incumbent's median 89% DTW interval half-width is **0.62 pp** on the 536
games and 0.56 pp across all 2,497.

**The two rows measure different things and the difference is dilution, not
disagreement.** A new ledger row appears only in the 536 games that contain an
out-of-bounds fumble; the other 1,961 move only because the class rates were
refitted on a larger population. Reporting the all-games median as *the* impact
would understate the change by a factor of sixteen on the games where the change
actually happens, and reporting only the 536-game median would hide how rare
those games are. Both are reported, and §5 says which one the gate reads.

---

## 5. Pre-registered gates

### 5a. Which gate was unseen — stated first

Process honesty requires saying what was already known when this document was
written:

| Gate | Known at writing? |
|---|---|
| **F-1** — branch point | Settled by argument in §2 |
| **F-2** — is the entity spread negligible? | **No. Genuinely unseen.** §4a is a *power* calculation on simulated data; the widened branch has never been fitted to the real counts |
| **F-3** — materiality floor | **Yes.** §4b was computed during design, and the gate is stated below with its result known |

F-3 is recorded as a gate anyway, because a future round needs to know what bar
this component cleared. It is **not** the gate that decides the candidate. F-2
is.

### 5b. Gate F-1 — the branch point

Settled in §2: **pass**, on the same bounce the recovery coin already admits.

### 5c. Gate F-2 — is the entity spread negligible? *(the binding gate)*

**Statistic:** the 89% upper bound on the population SD of true team-season
retention rates on the widened branch, from the exact grid posterior.

**Pass rule:** below **5.260 pp** — the 90th percentile of what this design
produces when the truth is exactly zero (§4a). By construction a skill-free
league clears it 90% of the time.

**Power:** 1.000 at the 12.5% reference, so document 09's Gate C-3 honesty
requirement is met and either outcome is interpretable.

**What each outcome means:**

- **Pass** → `w ≈ 0`, the entity term vanishes, and the component stays at **full
  neutralization at the class league rate** — the incumbent treatment, applied to
  a wider population.
- **Fail** → teams genuinely differ at keeping fumbled balls, and the component
  moves to **partial** neutralization at the team's shrunk rate. That would be a
  larger change than this document is proposing and would need its own round; the
  committed action on a fail is **to ship nothing and open that round**.

### 5d. Gate F-3 — the materiality floor *(known, recorded)*

**Statistic:** median |ΔDTW| across the games where a new ledger row appears.

**Pass rule:** ≥ the incumbent's median 89% DTW interval half-width on those same
games, matching document 16 §5e.

**Result, known at writing:** 1.65 pp against 0.62 pp — **clears by 2.7×.**

### 5e. Gate F-4 — the ledger must still sum

**Pass rule:** for every game, `deserved_margin == actual_margin −
Σ luck_epa · points_per_epa` to floating-point tolerance, and every fumble in the
population appears exactly once in the ledger. Enforced as a test, not as a
report.

This gate exists because widening a population is precisely the kind of change
that silently double-counts: a fumble that is both flagged out of bounds and
credited to a recovering team would produce two rows if the mask were written
carelessly. §3's eleven conflicted plays are the ones that would do it.

### 5f. The decision rule, committed in advance

| F-1 | F-2 | F-3 | **Action** |
|---|---|---|---|
| Pass | Pass | Pass | **Ship**, as simulator v1.2: widen the fumble population, full neutralization at the widened class rates |
| Pass | **Fail** | Pass | **Ship nothing.** Open a partial-neutralization round with its own pre-registration |
| Pass | — | Fail | **Ship nothing.** Report as measured and immaterial |

### 5g. What shipping would mean, fixed now

- `live_fumble_mask` is replaced by a mask over **all** fumbles with an
  identified fumbling team; the recovery test becomes a retention test.
- `FumbleBaseline.table` gains a `p_out_of_bounds` column for reporting; `p_own`
  and `swing_value` are refitted on the wider population.
- **The component name, `event_class` and ledger schema do not change**, so
  document 05 §3's treatment table gains no row — the existing row's population
  widens.
- v1.1 artefacts are left untouched and v1.2 writes alongside, exactly as
  document 15's Phase 3 rebuild did, because document 07's rematch validation was
  run against v1.1.
- **the maintainer approves before any of this lands.** The round stops at the Gate F-2
  report.

---

## 6. Known-defect register

| Defect | Evidence | Status |
|---|---|---|
| **A deliberate bat-out-of-bounds is indistinguishable from a roll-out** | No field records intent | **Open.** Biases toward calling a defensive play luck; the population is 602 plays so the exposure is bounded |
| **Field position is not in the fumble class** | A fumble at the sideline goes out of bounds far more often than one at the numbers | **Open, inherited.** The incumbent's classes are play type × aborted only; adding field position would need its own round |
| **Six plays get a class-specific swing they did not have before** | `field_goal/live` (4 plays) and `punt/aborted` (2) cross the non-null threshold under widening | **Open.** Artefact of mirroring the incumbent's pooling code; the correct fix is to pool the swing on class size, which would change the incumbent too |
| **Eleven fumbles carry contradictory flags** | Both an out-of-bounds flag and a named recovering team | **Accepted.** Resolved to the recovery; the rule is stated in §3 |
| **Out of the end zone is not separated** | A fumble out of the *back* of the end zone is a touchback, not a retention | **Open.** Not checked in this round; would appear as mislabelled retentions among the 602 |
| **Widening changes a component the rematch validation was run against** | Document 07 validated v1.1 | **Accepted.** Document 12 measured the rematch test as nearly blind below ~20% damage, so re-running it would prove nothing either way; v1.1 artefacts are preserved instead |

---

## 7. Constants appendix

| Constant | Value | Where it lives |
|---|---|---|
| `RANDOM_SEED` | 20260818 | `research/29_fumble_oob_power.py`, `research/30_fumble_oob.py` |
| `DATASETS` per scenario | 400 | inherited from `research/12_coinflips_power.py` |
| `MIN_CLASS_SIZE` | 30 | matches `components.fit_fumble_baseline` |
| `REFERENCE_RELATIVE` | 0.125 | document 04, via document 09 §4 |
| `MIN_POWER` | 0.80 | document 09 §5 |
| **Gate F-2 threshold** | **5.260 pp** | §4a, from the null simulation |
| Gate F-3 floor | 0.62 pp | §4b, the incumbent's own median half-width |
| Fumbles / out of bounds | 6,505 / 602 | §3 |
| Widened league retention rate | 56.48% | §3 |
| `points_per_epa` | 0.8389 | `research/outputs/model_metadata_v11.json` |

Results are written back into this document as §8.

---

## 8. Results

*Script: `research/30_fumble_oob.py`. Gate F-2's threshold was committed at
`afae577` before this script existed. Results in
`research/outputs/30_fumble_oob.json`.*

### The verdict, stated first

> **All three gates pass. Widening the fumble branch is the first Phase 5 change
> that should ship — and it is held at the door pending approval, because it
> moves numbers the product already prints.**

| Gate | Statistic | Result | Verdict |
|---|---|---|---|
| **F-1** — branch point | The same bounce, one instant earlier | — | **Pass** (§2) |
| **F-2** — entity spread | 89% upper bound **4.222 pp** vs 5.260 pp | Below the null bound | **Pass** |
| **F-3** — materiality | 1.65 pp vs 0.62 pp on 536 games | Clears by 2.7× | **Pass** (known at writing) |

### Gate F-2 — full neutralization survives, and gets stronger

| Branch | Population SD | 89% ETI | Relative | κ | w at the median entity | Grid edge mass |
|---|---|---|---|---|---|---|
| **Retention, all fumbles (widened)** | **2.370 pp** | 0.777 – 4.222 | **4.2%** | 1,317 | **0.0150** | 1.7 × 10⁻⁷ |
| Recovery, live only (incumbent) | 2.672 pp | 0.862 – 4.816 | 5.1% | 1,083 | 0.0163 | 1.3 × 10⁻⁷ |

Two things worth reading carefully.

**The widened branch looks *more* like a coin, not less.** Its population SD is
lower, its κ is higher, and its `w` is smaller. The natural worry about folding
in a rarer outcome was that it would introduce team-to-team structure — that some
teams are better at batting loose balls through the sideline. The measurement
says the opposite: the extra 591 events make the league look flatter.

**`w = 0.015` is the whole justification for full neutralization**, and it holds.
A team-season's own record of keeping fumbled balls carries about one and a half
percent of the information about its true rate; the other 98.5% comes from the
league. Document 05 §1's dial reads essentially zero and the entity term drops
out. *(Document 04 published `w = 0.011` for the recovery branch on a
4,898-fumble population; the 0.0163 above is the same quantity recomputed on this
document's 5,914-fumble population with the grid posterior, so the two are close
rather than identical and neither contradicts the other.)*

Grid edge mass below 2 × 10⁻⁷ on both fits: the posterior has died out well
before the grid boundary, so the exact posterior is exact.

### What changes if it ships

| | v1.1 | v1.2 |
|---|---|---|
| Fumble population | 5,914 recovered | **6,505 fumbles** |
| League rate | 52.13% recovery | **56.48% retention** |
| pass/live `p` | 0.4530 | **0.5096** |
| run/aborted `p` | 0.7622 | 0.7690 |
| Games gaining a ledger row | — | **536** |
| Median \|Δ deserved margin\| on those games | — | **1.76 points** |
| Games whose DTW side flips | — | **31** |

Document 05 §3's treatment table gains **no new row**. Fumble recovery is still
the one component neutralized in full at the class league rate; the population it
runs on widens, and the row's name in the ledger does not change.

---

## 9. What this round changes, and what it teaches

### The ledger changes, once approved

This is the first Phase 5 candidate to reach the shipping gate. Candidates 1
(overtime toss, immaterial) and 2 (deflected interceptions, unidentifiable) are
closed with nothing shipped.

### Three things worth carrying forward

1. **The most valuable candidate in the round was a conditioning bug in an
   existing component, not a new component.** Documents 16 and 17 went looking
   for new coins and found one that does not matter and one that cannot be seen.
   This one was already inside the simulator, selecting its population on the
   outcome of the branch immediately upstream of the branch it neutralizes.
   **The next round should audit the other components' populations for the same
   shape before hunting new candidates.** Field goals condition on an attempt
   being made; extra points condition on a touchdown having been scored.
2. **Widening a branch can improve its measurement.** The intuition that adding a
   rarer outcome dilutes a rate is wrong when the rarer outcome adds
   opportunities to every entity: power at the reference went from 0.975 to
   1.000 and the measured spread went *down*.
3. **Two impact populations, both reported.** A component that fires on 602 plays
   moves the median game with any fumble by 0.10 pp and the median game with an
   out-of-bounds fumble by 1.65 pp. Reporting one number would have been
   misleading in whichever direction it was chosen. Document 16 §9's lesson —
   state which summary the floor reads and why — is now a two-population rule.

### What would reopen this

- **A Gate F-2 failure on future data**, which would move the component to
  partial neutralization and needs its own round.
- **Field position in the fumble class**, the largest open defect: a fumble at
  the sideline and a fumble at the numbers do not share an out-of-bounds rate,
  and pooling them is the same mistake a flat recovery rate would have been.
- **Separating fumbles out of the back of the end zone**, which are touchbacks
  rather than retentions and are currently mislabelled among the 602.
