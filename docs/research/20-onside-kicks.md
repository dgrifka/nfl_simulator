# 20 — Onside kicks at the league rate, pre-registered

*Written 2026-08-18, **before `research/33_onside.py` existed**. Identification
audit, class table and entity-spread power: `research/32_onside_power.py`,
results in `research/outputs/32_onside_power.json`, reproduced in §3–§4.
Committed to git before the binding gate produces a number.*

*Inputs: documents 05 (the one rule and the two gates), 09 (the branch-point
verdict and the denial this document reopens), 12 (the rematch test's
blindness), 15 (Phase 5 scouting), 16 (the materiality floor and the sensitivity
arm), 18 (the class-structure precedent), 19 (simulator v1.2, which this runs
against).*

---

## 1. One-page story

### The question

Document 09 looked at onside kicks, **passed** them on mechanism — a short kick
into a scrum, the same loose-ball physics the fumble component already
neutralizes — and then denied them anyway, for one reason: the per-team trust
dial `w` could not be estimated, and a component whose expectation you cannot
locate is a component you cannot book.

That denial had an unexamined alternative. **`w = 0` needs no per-team
estimate.** It is the treatment fumble recovery already uses: charge the luck
against the league rate for the event's class and give the entity term nothing.
Document 09 never pre-registered that variant. This document does.

### What is settled before the fit

| Question | Answer | Where |
|---|---|---|
| Is there a branch point? | Yes — settled in document 09, restated in §2 | §2 |
| Can an onside kick be identified? | **Yes, from the play text: 584 attempts, 2016–2025** | §3 |
| Is `own_kickoff_recovery` a second identification channel? | **No. It is the outcome channel, and document 15 misread it** | §3 |
| Is there class structure? | Yes — 9.0% when the kicking team must, 13.4% when it chooses | §3 |
| Could a per-team spread be resolved? | **No, and now with a number: power 0.10 at the 12.5% reference** | §4 |

### The answer, stated first — and what is still unknown

**Everything except whether it matters is settled.** The identification is
clean, the class structure is real, and document 09's denial is confirmed as
unresolvable rather than merely unattempted. What has not been computed is the
one thing the candidate lives or dies on:

> **Gate K-3, the materiality floor, is genuinely unseen at the time of writing.**
> 584 kicks land in 519 of 2,761 games, and whether replacing their coin moves
> those games by more than the interval the product already prints on them has
> not been calculated.

### Five things to hold onto

1. **Document 15's flag cross-check was wrong in kind, not in size.** It
   reported that `own_kickoff_recovery` adds four attempts the text misses. All
   four are deep kickoffs — 41, 43, 48 and 56 yards — whose *return* was muffed
   and fell to the kicking team. That is a loose ball and possibly a candidate
   of its own, but it is not an onside kick, and pooling it would price a
   different coin. §3 records the correction.
2. **The flag is the right channel for the outcome, and it is a good one.** It
   agrees with the play text on 581 of 584 attempts, and all three
   disagreements are recoveries **reversed on replay**, where the flag carries
   the final ruling and the text carries the on-field call.
3. **The class split is on game state, never on outcome.** A team that trails
   inside five minutes has no other option and the return team is lined up for
   it; a kick at any other moment is a surprise. 9.0% against 13.4%.
4. **`w = 0` is a choice here, where it was a measurement for fumbles.**
   Document 04 measured `w = 0.011` for fumble recovery. Nothing of the sort is
   available at a median of two kicks per team-season. §5f therefore makes the
   sensitivity to that dial a **gate**, not a footnote.
5. **This is the second-largest per-event swing in the project.** 4.15 EPA
   pooled, against 4.27 for a passing-play fumble. Rare events with a
   three-to-five point swing are what this project exists to adjudicate.

### Statistic convention

Posterior means with 89% equal-tailed intervals. Impact figures are in
percentage points of DTW; EPA converts at `points_per_epa` = 0.8389.

---

## 2. Gate A — the branch-point memo

> **Is there a moment where the outcome is resolved by a mechanism outside either
> team's control, conditional on the state both teams created?**

### Who comes up with the ball — **PASS**

The state both teams created is: a ball has been kicked ten to fifteen yards
into a crowd of players from both sides who are all trying to fall on it. What
happens next is an oblong ball bouncing among bodies. Document 09 passed this on
exactly the fumble-recovery reasoning and nothing here revisits it.

### Three things next to the branch that are *not* the coin

| Adjacent thing | Verdict | Why |
|---|---|---|
| **Deciding to kick onside** | **Not the coin.** A decision | Attempting one is a coaching call with a known price, exactly like attempting a 55-yard field goal. It belongs in `core`, and the class split (§3) is what keeps the *situation* out of the coin |
| **Where the kick is placed** | **Not cleanly separable, and this is the candidate's largest defect** | A well-executed onside kick — high hop, right speed, arriving as the coverage does — is a real skill. The data has no placement measure, so `w = 0` charges all of it to luck. §5f gates the sensitivity to this and §6 registers it |
| **The return team's alignment** | **Not the coin; it is state** | Whether the receiving team has its hands team on the field is a consequence of the game situation both teams produced. That is why the class is defined on score and clock |

The honest summary is that this branch is **less pure than a fumble bounce**.
Fumble recovery has no analogue of kick placement. The design's answer is not to
argue the impurity away but to price it: §5f requires the verdict to survive
handing a quarter and a half of the trust back to the kicking team.

---

## 3. Data and identification

- **Grain of a row**: one onside-kick attempt.
- **Source**: `data/pbp/*.parquet` 2016–2025, `play_type == "kickoff"`.
- **Population rule, fixed here**: the description matches `kicks onside`.
  **584 attempts.**
- **Outcome rule, fixed here**: `own_kickoff_recovery == 1`.
- **Perspective**: on a kickoff nflfastR puts `posteam` on the *receiving* team,
  so the kicking team is `defteam` and both `epa` and `score_differential` are
  signed from the receiver's side. Verified against the play text; an own-kickoff
  recovery carries `epa` = −2.95 for `posteam`.

### What was rejected, and why

| Rejected | n | Reason |
|---|---|---|
| `own_kickoff_recovery` fires, text says no onside kick | **4** | Deep kickoffs (41–56 yards) whose return was muffed. A different event with a different rate |
| Text contains "onside" but not "kicks onside" | **5** | Four are `(Onside Kick formation)` lines describing kicks that then travelled 13, 14, 43 and 54 yards; one is a nullified attempt followed by a 65-yard re-kick |

### The flag against the text

581 of 584 agree. The three disagreements are all recoveries **overturned on
replay**, named in the same description ("The Replay Official reviewed the loose
ball recovery ruling"). The flag carries the final ruling, so the flag is right
and the text is stale. This is recorded because it is the only place the two
channels conflict and the resolution should not be re-derived later.

### The class table

| Class | Definition | n | p(recover) | EPA recovered | EPA lost | **swing** |
|---|---|---|---|---|---|---|
| **expected** | kicking team trails **and** ≤ 300 seconds remain | 465 | **0.0903** | +2.386 | −1.299 | **3.685** |
| **surprise** | anything else | 119 | **0.1345** | +3.815 | −1.685 | **5.499** |
| *pooled, for reference* | — | 584 | 0.0993 | +2.780 | −1.374 | 4.155 |

Both classes clear the 30-play minimum the fumble component uses, so neither
borrows a pooled rate. The crude version of the same split — by quarter — runs
41.7%, 25.0%, 11.1%, 8.9% across quarters one to four, which is the same story
with a worse instrument.

### The rate over time

Per season: 9.0%, 22.4%, 8.8%, 12.9%, 5.6%, 15.5%, 6.8%, 4.5%, 5.4%, 9.4%. Two
rule changes sit inside that window — the 2018 restrictions on the kicking
team's run-up and the 2024 dynamic kickoff — and the series is noisy enough at
40–70 kicks a season that neither is visible above the noise. **The component
pools all ten seasons**, and §6 registers that as an open defect rather than
pretending the pooling is free.

---

## 4. Power

### 4a. The entity spread cannot be resolved — now with a number

The exact grid instrument of document 09 §4, run at the real denominators: 257
team-seasons, 584 kicks, median **2 kicks per team-season**.

| Branch | Entities | Opportunities | League rate | Median n | Null bound (90th pct) | Power at 5% rel | at 12.5% rel | at 50% rel |
|---|---|---|---|---|---|---|---|---|
| **Onside recovery** | 257 | 584 | 9.93% | **2** | **9.87 pp** | 0.092 | **0.102** | 0.318 |
| *Fumble retention (v1.2), for scale* | 320 | 6,505 | 56.48% | 20 | 5.26 pp | 0.278 | 1.000 | — |

**Document 09's denial is confirmed and quantified.** At the 12.5% reference the
test rejects the null a tenth of the time — barely above the 10% the design
produces by construction when the truth is exactly zero. Even a spread half the
size of the league rate is caught less than a third of the time. There is no
version of this data in which the dial can be read.

That is the whole justification for `w = 0` and also its whole weakness: the
same blindness that stops us estimating a spread stops us ruling one out.
§5f is the answer.

### 4b. What the materiality gate can see

519 of 2,761 games (18.8%) carry at least one attempt; mean 1.13 per such game,
maximum 2. The gate is a median over 519 games, so it is not a low-count
statistic — but no claim is made here about what it will show, because it has
not been computed.

---

## 5. Pre-registered gates

### 5a. Which gates are unseen — stated first

| Gate | Known at writing? |
|---|---|
| **K-1** — branch point | Settled by argument in §2 and in document 09 |
| **K-2** — identification | **Yes.** §3 was computed during design and the rule is fixed there |
| **K-3** — materiality floor | **No. Genuinely unseen, and it is the binding gate** |
| **K-4** — the ledger must still sum | Enforced as a test |
| **K-5** — sensitivity to the unmeasurable dial | **No. Genuinely unseen** |

### 5b. Gate K-1 — the branch point

Settled in §2: **pass**, on the same scrum document 09 already admitted.

### 5c. Gate K-2 — identification *(known, recorded)*

**Statistic:** agreement between the play text and `own_kickoff_recovery` on the
584-attempt population.

**Pass rule:** ≥ 95% agreement, with every disagreement explained by name.

**Result, known at writing:** 99.5% (581/584), all three disagreements being
replay reversals — **pass**.

### 5d. Gate K-3 — the materiality floor *(the binding gate)*

**Statistic:** median |ΔDTW| across the 519 games containing an onside kick,
comparing simulator v1.2 with and without the onside component, with the fumble,
field-goal and extra-point draws generated from their own seeded generators in
both arms so the difference is the onside rows and nothing else.

**Pass rule:** ≥ the v1.2 median 89% DTW interval half-width on those same
games. This is document 16 §5e's floor and document 18 §5d's floor, unchanged:
a component must move a game by more than the uncertainty already printed on it.

**Power:** the statistic is a median over 519 games, and document 16 measured
the redraw spread of exactly this statistic at ±0.05 pp on a 155-game
population. A failure within 0.1 pp of the floor will be re-drawn eight times
and reported with its spread, as document 16 §8 did, rather than called.

**What each outcome means:**

- **Pass** → the component ships as simulator v1.3, at the class league rate
  with `w = 0`, subject to K-5 and to the maintainer's approval.
- **Fail** → **ship nothing.** The measurement is reported as document 16's
  overtime toss was: real, sized, and below the bar.

### 5e. Gate K-4 — the ledger must still sum

**Pass rule:** for every game, `deserved_margin == actual_margin −
Σ luck_epa · points_per_epa` to floating-point tolerance, and every onside kick
in the population appears exactly once in the ledger. Enforced as a test, not a
report.

A kickoff can also carry a fumble on the return, so the concern is real: a play
that produced both an onside-kick row and a fumble row would double-count the
same loose ball. The population and the fumble population are checked for
overlap and the check is reported whatever it shows.

### 5f. Gate K-5 — the verdict must not depend on the dial we cannot read

**Statistic:** the K-3 median |ΔDTW|, recomputed with the expectation shrunk
toward the kicking team's own record,

    p = w · (team-season's own recovery rate) + (1 − w) · (class league rate)

at **w ∈ {0.00, 0.25, 0.50}**.

**Pass rule:** the K-3 verdict — pass or fail against its floor — is identical at
all three values of `w`.

**What each outcome means:**

- **Pass** → the treatment's conclusion does not rest on an unmeasurable
  quantity, and `w = 0` may be adopted as the simplest member of a family that
  all agree.
- **Fail** → **ship nothing**, and the register records the reason precisely:
  *the component's verdict is a function of a dial this data cannot read.* That
  is a different and more interesting failure than "it does not matter", and it
  would apply to any future candidate with two events per entity per season.

### 5g. The decision rule, committed in advance

| K-1 | K-2 | K-3 | K-5 | **Action** |
|---|---|---|---|---|
| Pass | Pass | Pass | Pass | **Ship** as simulator v1.3, pending the maintainer's approval |
| Pass | Pass | Pass | **Fail** | **Ship nothing.** Report as verdict-depends-on-an-unreadable-dial |
| Pass | Pass | **Fail** | — | **Ship nothing.** Report as measured and immaterial |
| Pass | **Fail** | — | — | **Ship nothing.** Report as unidentifiable |

### 5h. What shipping would mean, fixed now

- A new component `onside`, with `event_class` in `{expected, surprise}`,
  charged to the kicking team, `realized` = recovered, `swing` = the class swing
  signed to home perspective.
- Document 05 §3's treatment table gains **one row**: full neutralization at the
  class league rate, with the entry recording that `w = 0` is a *choice* and
  naming §4a as the reason.
- v1.2 artifacts are left untouched and v1.3 writes alongside, as v1.2 did.
- The rematch validation is **not** re-run, for document 18 §6's reason.
- **The maintainer approves before any of this lands.** The round stops at the report.

---

## 6. Known-defect register

| Defect | Evidence | Status |
|---|---|---|
| **Kick placement is a skill charged to luck** | No placement measure exists in any available dataset | **Open, and the largest.** §5f gates the sensitivity to it; a failure there is the honest outcome |
| **`w = 0` is assumed, not measured** | §4a: power 0.102 at the 12.5% reference | **Open.** Unresolvable at two kicks per team-season; the same blindness cuts both ways |
| **Ten seasons are pooled across two rule changes** | 2018 run-up restrictions, 2024 dynamic kickoff; season rates 4.5%–22.4% | **Open.** Splitting on era would leave 200-kick strata, and no era test is powered |
| **The class boundary is a step function** | 301 seconds is "surprise" and 300 is "expected" | **Accepted.** Any boundary is arbitrary; this one is stated in advance and never tuned |
| **Recovery is a scrum, not a bounce** | Multiple players from both sides converge deliberately | **Open.** Weaker than the fumble analogy on which Gate A rests, and stated as such in §2 |
| **Two attempts carry a null kick distance** | `kick_distance` is null on 2 of 584 | **Accepted.** Distance is not used by the component; it appears only in the §3 audit |

---

## 7. Constants appendix

| Constant | Value | Where it lives |
|---|---|---|
| `RANDOM_SEED` | 20260818 | `research/32_onside_power.py`, `research/33_onside.py` |
| Population rule | `desc` matches `kicks onside` | §3 |
| Outcome rule | `own_kickoff_recovery == 1` | §3 |
| Class rule | trailing **and** ≤ 300 seconds → `expected` | §3 |
| Attempts / recovered | 584 / 58 | §3 |
| League rate | 9.93% | §3 |
| Class rates | 0.0903 expected, 0.1345 surprise | §3 |
| Class swings | 3.685 / 5.499 EPA | §3 |
| Games touched | 519 of 2,761 | §4b |
| Entity-spread null bound | 9.87 pp | §4a |
| Power at the 12.5% reference | 0.102 | §4a |
| Sensitivity dial values | w ∈ {0.00, 0.25, 0.50} | §5f |
| `points_per_epa` | 0.8389 | `research/outputs/model_metadata_v12.json` |

Results are written back into this document as §8.

---

## 8. Results

*Script: `research/33_onside.py`. The gates were committed at `c5d4dfe` before
this script existed. Results in `research/outputs/33_onside.json`.*

### The verdict, stated first

> **The onside coin matters just enough to clear the bar when you assume it is
> pure luck, and stops clearing it the moment you hand back a quarter of the
> trust to the kicking team. The component is not shipped, and the reason is not
> that it is too small — it is that its verdict is a function of a dial this data
> cannot read.**

| Gate | Statistic | Result | Verdict |
|---|---|---|---|
| **K-1** — branch point | A short kick into a scrum | — | **Pass** (§2, document 09) |
| **K-2** — identification | 581/584 agreement, all 3 disagreements replay reversals | 99.5% vs 95% | **Pass** (known at writing) |
| **K-3** — materiality at `w = 0` | median \|ΔDTW\| **0.55 pp** vs a **0.44 pp** floor | Clears by 1.25× | **Pass** |
| **K-4** — the ledger must still sum | 0 plays shared with the fumble population, 0 duplicates | — | **Pass** |
| **K-5** — verdict independent of `w` | Passes at `w = 0`, fails at 0.25 and 0.50 | Not identical | **FAIL** |

Per the decision rule committed in §5g — *pass K-3, fail K-5 → ship nothing,
report as verdict-depends-on-an-unreadable-dial* — **the component is measured,
reported, and left out of the ledger.** Document 05 §3's treatment table does not
move.

### Gate K-3 — it clears the floor, by less than fumbles did

519 games carry an onside kick. Against simulator v1.2, with the fumble,
field-goal and extra-point draws held to their own seeded generators in both
arms:

| | `w = 0.00` | `w = 0.25` | `w = 0.50` |
|---|---|---|---|
| **Median \|ΔDTW\|** | **0.55 pp** | 0.42 pp | 0.32 pp |
| Mean \|ΔDTW\| | 1.73 pp | 1.49 pp | 1.26 pp |
| Max \|ΔDTW\| | 28.37 pp | 26.22 pp | 25.33 pp |
| Median \|Δ deserved margin\| | 0.281 pts | 0.211 pts | 0.141 pts |
| Games whose DTW side flips | 4 | 3 | 1 |
| **Against the 0.44 pp floor** | **Pass** | Fail | Fail |

For scale, the fumble widening cleared its floor by 2.7× on 536 games (document
18 §8) and the overtime toss missed its by 0.13 pp on 155 games (document 16
§8). This candidate sits between them and closer to the toss.

### The redraw, run because §5d required it

`w = 0.25` misses the floor by 0.02 pp, well inside the 0.1 pp band §5d said
would be re-drawn rather than called. Eight independent redraws of every coin:

| Arm | Median \|ΔDTW\| across 8 redraws | Passes |
|---|---|---|
| Floor (v1.2 half-width) | 0.44 – 0.50 pp | — |
| `w = 0.00` | **0.53 – 0.57 pp** | **8 / 8** |
| `w = 0.25` | 0.42 – 0.46 pp | **2 / 8** |
| `w = 0.50` | 0.31 – 0.34 pp | **0 / 8** |

**Both ends of the finding are stable.** The `w = 0` pass is not a lucky draw —
it survives every redraw. The `w = 0.25` failure is not a rounding artefact
either: it fails six times in eight, and it fails because the floor itself moves
up on some redraws, not because the statistic collapses. The gate is decided.

### Gate K-5 — the honest failure

The component's whole justification was that `w = 0` needs no per-team estimate.
That is true, and it turns out to be beside the point: **the answer to "does this
matter?" is different at `w = 0` than at `w = 0.25`,** and §4a established that
nothing in this data can say which is right. Power to detect a 12.5%-relative
spread is 0.102.

Two readings, and the design committed to the second in advance:

- *Charitable*: `w = 0` is the same choice fumble recovery makes, so make it and
  ship. But fumble recovery **measured** `w = 0.015` on 6,505 events. Here the
  choice would be an assumption doing the work of a measurement, on a branch
  (§2) where a kicker's placement skill is real and unmeasured.
- *Committed*: a component whose verdict flips with an unreadable dial is not
  ready, whatever the dial's true value happens to be. **Ship nothing.**

### What would reopen this

- **A placement measure.** Anything that separates a well-struck onside kick
  from a badly struck one would let the dial be estimated from execution rather
  than from a two-kick-per-season record, and would collapse the K-5 ambiguity.
- **More events.** The rate at which onside kicks are attempted is falling
  (69 in 2016, 53 in 2025), so this is not arriving on its own.
- **A change to the materiality floor.** The candidate fails on the interaction
  of two thresholds, and the floor is the incumbent's own interval width. If
  document 10's coverage work ever narrows that interval, this candidate's
  arithmetic changes without the football changing.

---

## 9. What this round changes, and what it teaches

### The ledger does not change

This is the third Phase 5 candidate to be measured and left out — after the
overtime toss (immaterial) and deflected interceptions (unidentifiable). One
candidate shipped: the fumble widening, which was a conditioning correction to an
existing component rather than a new one.

### Three things worth carrying forward

1. **Document 15's cross-check was read too generously, and the cost was
   nearly a wrong population.** "The flag adds only 4 plays the text misses" is
   true and irrelevant: all four are deep kickoffs with muffed returns. A
   cross-check that counts agreement without reading the disagreements is not a
   cross-check. **Every future identification memo should print the rejected
   rows, not their count.**
2. **A sensitivity arm can be the binding gate, and should be allowed to be.**
   Document 16 ran its sensitivity as a required report; here it was written as
   a gate with a committed on-failure action, and it is the only gate that
   failed. A component that passes materiality by 1.25× and has an unmeasurable
   dial was always going to fail in this direction — the value of pre-registering
   it is that the failure is a verdict rather than a judgment call made after
   seeing three numbers.
3. **"`w = 0` needs no estimate" is true and insufficient.** It removes the
   *estimation* problem and leaves the *identification* problem untouched. Any
   future candidate proposing full neutralization on an unmeasurable dial should
   expect to be judged on K-5 before K-3.
