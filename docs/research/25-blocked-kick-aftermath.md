# 25 — Blocked-kick aftermath, pre-registered

*Written 2026-08-18, **before `research/39_blocked_kicks.py` existed**.
Identification, the class table, the document 24 screen, entity power and the
materiality floor: `research/38_blocked_kick_power.py`, results in
`research/outputs/38_blocked_kick_power.json` and reproduced in §3–§4. Committed
to git before the fit produces a number.*

*Inputs: documents 05 (Gate A), 09 (the grid instrument), 14 (the punt-bounce
close, on identification), 20 §5f (the dial-sensitivity gate, applied here in
full), 23 §C2 (the scouting finding), 24 §9 (the screen, and the failure this
candidate is measured against).*

*Tier: **model change** — a new component with its own class table, charged to
the kicking team.*

---

## 1. One-page story

### The question

415 blocked kicks in ten seasons — 192 field goals, 110 extra points, 113 punts
— put a live ball on the turf, and **five of them reach the fumble component**.
The simulator books the fate of the other 410 as entirely deserved. Document 23
§C2 asked whether the ball's fate after a block is fumble-family physics that the
ledger is missing.

### The answer, stated first

**It is a real branch, it identifies cleanly, and both of the numbers that decide
materiality point the wrong way before the fit is run.** The candidate carries
less luck per event than the kickoff-muff candidate that just failed, and the
games it touches are already twice as uncertain.

| | Blocked kicks | Kickoff muffs (document 24, **failed**) |
|---|---|---|
| Events | 410 eligible | 269 |
| Games touched | **378** | 248 |
| Screen — `p(1 − p) × swing` per event | **0.3091 EPA** | 0.3948 EPA |
| v1.2 median 89% DTW half-width on those games | **1.4392 pp** | 0.7222 pp |
| Entity-spread power at the 12.5% reference | **0.177 — unresolvable** | 1.000 (component-wide) |

**Both axes are worse and the floor is twice as high.** The round proceeds
anyway, because document 24 §9's screen was written as a filter on *effort*, not
as a gate on *verdicts*, and a measured failure is a better record than an
argued one. What the screen buys is honesty about expectations: §5a records that
Gate B-4 is expected to fail, before it is run.

### Four things to hold onto

1. **The block and the aftermath are two different things and only one of them
   is a coin.** The block is a defensive play — protection lost, denied at Gate A
   the way a drop is. What the loose ball does next is the same scrum the fumble
   component already prices. §2 draws that line and §3c tests it with a number.
2. **The branch is "did the defense come up with it", not "who kept
   possession".** A kicking team that falls on its own blocked field goal behind
   the line to gain has still lost the ball on downs. What changes the game is
   whether a defender got the ball *in hand with room to run* — 36 of the 113
   blocked punts were returned for touchdowns.
3. **The aftermath is worth about 1.7 EPA, not 3.7.** The block itself costs the
   kicking team most of what it costs; the coin that follows is worth a fifth of
   a touchdown on average. That is the honest size of the thing and it is the
   reason the screen reads low.
4. **This component's dial cannot be measured at all.** 410 events across 220
   team-seasons is a median of two per team, and power to resolve a
   12.5%-relative spread is 0.177 — worse than the onside candidate's 0.102 was
   generous. Gate B-6 is document 20's K-5 applied in full, all three values
   gated, with no loosening of the kind document 24 §5g argued for.

### Statistic convention

Posterior means with 89% equal-tailed intervals; population SD read against the
simulated null bound, never against zero (document 05 §8).

---

## 2. Gate B-1 — the branch-point memo, and the line it has to draw

> **Is there a moment where the outcome is resolved by a mechanism outside either
> team's control, conditional on the state both teams created?**

### The block itself — **FAIL, and it stays in `core`**

A blocked kick is a defensive play. Somebody beat a block, timed a jump, got a
hand up. It is denied at Gate A on exactly the grounds document 05 §2 denies a
dropped pass: a play somebody made is not a branch nobody resolved. **No part of
the −3.7 EPA a blocked field goal costs is neutralized by this component.**

### The ball's fate afterwards — **PASS**

The state both teams created is: the kick is dead in the air, the ball is on the
ground behind or near the line of scrimmage, and eleven players from each side
are converging on it. That is the fumble scrum, and document 05 §2 admitted it.

**The hardest objection, and the measurement that answers it.** It is fair to
say the block and the recovery are one continuous defensive play — the man who
got his hand on the ball is standing right where it lands, so calling the
recovery a coin credits the defense's own momentum to luck.

**The data says otherwise: of the 144 defensive recoveries where both the
blocker and the recoverer are named in the play text, the blocker is the
recoverer 16 times — 11%.** Eighty-nine per cent of the time the ball is picked
up by somebody else, which is what a scrum looks like rather than what a
continuous play looks like. The objection is real, it is bounded at roughly one
recovery in nine, and §6 registers it at that size.

### Three things next to the branch that are *not* the coin

| Adjacent thing | Verdict | Why |
|---|---|---|
| **Beating the protection** | **Not the coin.** Stays in `core` | The block is the defensive play; the component never touches its EPA |
| **The return after a defensive recovery** | **Not the coin.** Stays in `core` | A return is a played-out sequence, and document 21's §9 decision confirms returns do not enter the ledger. The class swing prices *whether the defense got the ball*, and the return distribution is what makes that branch worth 1.7 EPA rather than 0.2 |
| **Where the blocked ball bounces** | **State, not branch** | A ball tipped straight down and a ball spinning ten yards back are different situations, and neither is in the class. §6 registers it, as documents 18 and 24 did for field position |

---

## 3. Data and identification

- **Grain of a row**: one blocked kick.
- **Source**: `data/pbp/*.parquet` 2016–2025.
- **Population rule, fixed here**: `field_goal_result == "blocked"` **or**
  `extra_point_result == "blocked"` **or** `punt_blocked == 1`. **415 plays in
  383 games.**
- **Charged team**: the kicking team, which is `posteam` on all three play types.
- **`retained` = 1** when the defense did **not** come up with the ball: it died
  where it lay, went out of bounds, or the kicking team fell on it.

### 3a. The exclusion rule, fixed here

**Five blocked kicks already carry a v1.2 fumble row and are excluded from this
population**, leaving **410 events in 378 games**. All five are printed in
`research/outputs/38_blocked_kick_power.json` and in the script's output rather
than counted; four are blocked field goals where the recovering defender then
fumbled, and one is a punter who recovered his own blocked punt and fumbled it
away. In every case the fumble row is a *second* loose ball on the same play, and
booking both would price the same possession twice. The fumble row wins, the
blocked-kick row is dropped, and the shipped fumble component does not move.

### 3b. The disposition rule, fixed here

The recovering team is read from `recovered by XXX-` in the text after the word
BLOCKED, case-insensitively. `recovered the blocked kick` with no team named is
the defensive-two-point phrasing and resolves to the defense, because on a try
only the defense may advance a blocked kick.

**Two plays in 415 mention a recovery the parser cannot attribute**, and both are
printed rather than counted:

| Class | EPA | Text |
|---|---|---|
| extra_point | −0.932 | `…extra point is Blocked (96-D.Barnett)… Ball smothered at BUF 36, no recovery after block` |
| extra_point | −0.932 | `…extra point is Blocked (91-D.Wise)… Kick went back through end zone, no recovery` |

Both say **no recovery** in as many words, so both resolve to `retained = 1`
under the rule as written. Identification is complete at 415 of 415.

### 3c. The class table

| Class | n | p(defense does not recover) | epa_own | epa_lost | **swing** | **screen `p(1−p)·swing`** |
|---|---|---|---|---|---|---|
| **field_goal** | 188 | 0.6436 | −3.064 | −4.791 | **1.726** | **0.3960** |
| **punt** | 112 | 0.3393 | −3.148 | −4.733 | **1.585** | **0.3553** |
| **extra_point** | 110 | 0.6818 | −0.927 | −1.450 | **0.523** | **0.1134** |
| *n-weighted* | 410 | 0.5707 | — | — | 1.709 | **0.3091** |

All three classes clear the 30-play minimum the fumble component uses, so none
borrows a pooled rate.

**Punts are the class where the branch actually swings.** A blocked punt is
recovered by the defense two times in three, and 36 of the 113 were returned for
touchdowns. Field goals run the other way — the defense gets it only a third of
the time, because a blocked field goal usually dies behind the line. Extra points
are the weakest class in the round: a 0.52 EPA swing on a play the kicking team
cannot score on.

### 3d. Blocked kicks per season

48, 49, 32, 39, 40, 36, 48, 33, 44, 46. Flat, with no rule change behind it —
unlike the kickoff-muff population, which more than doubled in 2025.

---

## 4. Power and the floor

### 4a. Can the entity spread be resolved? **No.**

| Branch | Entities | Opportunities | League rate | Median n | Null bound (90th pct) | Power at 12.5% rel |
|---|---|---|---|---|---|---|
| Blocked kicks, kicking team | 220 | 410 | 57.07% | **2** | 13.31 pp… | **0.177** |

*(Null bound: mean 13.07 pp, 90th percentile **17.31 pp**.)*

**0.177 against the 0.80 minimum document 09 §5 fixed.** At two blocked kicks per
team-season this design cannot tell a skill-free league from one where teams
differ by an eighth of the league rate. There is therefore **no Gate B-3 on the
entity spread in this round** — a gate nobody can pass or fail informatively is
theatre, and document 09 §5's honesty requirement forbids reporting one. What
takes its place is Gate B-6, which asks the only question the data can answer:
does the verdict depend on the number we cannot measure?

### 4b. The materiality floor *(Gate B-4's threshold)*

**378 games contain an eligible blocked kick.** Simulator v1.2's median 89% DTW
interval half-width on those games is **1.4392 pp**.

That is **twice** the floor the kickoff-muff candidate faced and more than three
times the onside candidate's. The reason is not mysterious and it is worth
writing down: a game containing a blocked kick is a game containing a kick, and
the field-goal and extra-point components are the widest luck rows the simulator
prints. **The candidate has to clear a bar its own subject matter raises.**

**No treatment arm was run**, so the B-4 statistic is unseen.

---

## 5. Pre-registered gates

### 5a. Which gates are unseen, and what is expected — stated first

| Gate | Known at writing? |
|---|---|
| **B-1** — the branch point, and the block/aftermath line | Settled by argument in §2, tested at 11% in §3c |
| **B-2** — identification | **Yes.** 415 of 415, §3b |
| **B-3** — entity spread | **Not run.** Unresolvable at power 0.177 (§4a) |
| **B-4** — materiality floor | **No. Genuinely unseen, and the binding gate** |
| **B-5** — the ledger must still sum | Enforced as a check |
| **B-6** — sensitivity to the unmeasurable dial | **No. Genuinely unseen** |

**Stated before the fit: B-4 is expected to fail.** The screen is 0.3091 EPA per
event against 0.3948 for a candidate that missed its floor by 1.8×, and this
floor is twice as high. If the arithmetic is roughly linear the median |ΔDTW|
lands near 0.3 pp against a 1.44 pp bar. **Writing that down in advance is the
point of the screen** — so that the result, whichever way it lands, cannot be
narrated afterwards as though it were a surprise. If B-4 passes despite this, the
prediction was wrong and the document says so.

### 5b. Gate B-1 — the branch point

Settled in §2: the block **fails** and stays in `core`; the aftermath **passes**,
on the same scrum the fumble component prices, with the continuous-play objection
bounded at 11%.

### 5c. Gate B-2 — identification *(known, recorded)*

**Statistic:** the share of the population whose recovering side resolves, with
every unattributable row printed by name.

**Pass rule:** ≥ 95% resolved, and every rejected or excluded row printed.

**Result, known at writing:** 415 of 415 resolved; the two unattributable rows
are printed in §3b and both say "no recovery" in the text; the five excluded
overlap plays are printed in the script output — **pass**.

### 5d. Gate B-3 — not run, and why

Deliberately absent. §4a measures the power to resolve the entity spread at
**0.177**, and document 09 §5 fixed 0.80 as the minimum at which a spread result
is interpretable. Reporting a pass here would mean "we could not see anything",
not "there is nothing". **The component is therefore proposed at `w = 0` as a
choice rather than a measurement, and Gate B-6 is what makes that choice
accountable.**

### 5e. Gate B-4 — the materiality floor *(the binding gate)*

**Statistic:** median |ΔDTW| across the **378 games containing an eligible
blocked kick**, comparing simulator v1.2 with and without the blocked-kick
component, with the fumble, field-goal and extra-point draws generated from their
own seeded generators in both arms so the difference is the blocked-kick rows and
nothing else.

**Pass rule:** ≥ **1.4392 pp**, v1.2's own median 89% DTW half-width on those
same games (§4b).

**Power:** the statistic is a median over 378 games; document 16 measured its
redraw spread at ±0.05 pp on 155 games. **A result within 0.1 pp of the floor is
re-drawn eight times and reported with its spread**, not called.

**What each outcome means:**

- **Pass** → subject to B-5 and B-6, the component ships as simulator v1.3.
- **Fail** → **ship nothing.** Reported as real, sized and below the bar.

### 5f. Gate B-5 — the ledger must still sum

**Pass rule:** for every game, `deserved_margin == actual_margin −
Σ luck_epa · points_per_epa` to floating-point tolerance, and **no play carries
both a blocked-kick row and a fumble row.** The five overlap plays of §3a are the
ones that would; the exclusion rule is what prevents it and the check is what
proves the rule fired.

A blocked field goal also carries a **field-goal** luck row today, priced as an
ordinary miss. That is not a double count — the two rows price different branches
of the same play, the way a punt carries both a core EPA and a fumble row — but
it is an **interaction**, because candidate 3 (document 23 §C1) proposes to
change how a blocked kick is priced inside the field-goal component. **This round
holds the field-goal and extra-point components exactly as shipped**, and any
future round that changes both must measure them together rather than in
sequence.

### 5g. Gate B-6 — sensitivity to the dial this data cannot read

**Statistic:** the B-4 median |ΔDTW|, recomputed with the expectation shrunk
toward the kicking team's own record,

    p = w · (team-season's own retention rate) + (1 − w) · (class league rate)

at **w ∈ {0.00, 0.25, 0.50}**.

**Pass rule:** the B-4 verdict is identical at **all three** values.

**This is document 20's K-5 unmodified, and the loosening document 24 §5g argued
for does not apply here.** That argument rested on the fumble component's `w`
being measured at power 1.000 on 6,756 events; this component has no measured `w`
anywhere, at power 0.177. When nothing is measured, everything is gated.

**What each outcome means:**

- **Pass** → the conclusion does not rest on an unmeasurable quantity, and `w = 0`
  is adopted as the simplest member of a family that agrees.
- **Fail** → **ship nothing**, recorded as *verdict-depends-on-an-unreadable-dial*.

### 5h. The decision rule, committed in advance

| B-1 | B-2 | B-4 | B-5 | B-6 | **Action** |
|---|---|---|---|---|---|
| Pass | Pass | Pass | Pass | Pass | **Ship** as simulator v1.3, pending the maintainer's approval, which this round does not have |
| Pass | Pass | Pass | Pass | **Fail** | **Ship nothing.** Verdict-depends-on-an-unreadable-dial |
| Pass | Pass | **Fail** | — | — | **Ship nothing.** Measured and immaterial |
| — | — | — | **Fail** | — | **Ship nothing.** A ledger that does not sum is a bug |

### 5i. What shipping would mean, fixed now

- A new component `blocked_kick`, `event_class` ∈ {`field_goal`, `extra_point`,
  `punt`}, charged to the kicking team, `realized` = the defense did not recover,
  `swing` = the class swing signed to home perspective.
- Document 05 §3's treatment table gains **one row**, recording that `w = 0` is a
  **choice** and naming §4a as the reason — the same wording document 20 §5h
  fixed for onside.
- v1.2 artifacts untouched; v1.3 writes alongside. The rematch validation is not
  re-run, for document 18 §6's reason.
- **The maintainer approves before any of this lands.**

### 5j. Kill and rollback

On failure the branch keeps the code, no production module changes, v1.2 stays
authoritative, and §8 is the record.

---

## 6. Known-defect register

| Defect | Evidence | Status |
|---|---|---|
| **The blocker sometimes recovers his own block** | 16 of 144 named defensive recoveries, **11%** | **Open, measured, bounded.** The strongest argument against Gate B-1 and the reason §2 answers it with a number |
| **`w = 0` is a choice, not a measurement** | Power 0.177 at the 12.5% reference, median 2 events per team-season | **Open, and the largest.** Gate B-6 exists for it; a failure there is the honest outcome |
| **Where the blocked ball bounces is not in the class** | A ball tipped straight down and one spinning back ten yards are different coins | **Open, inherited** from documents 18 and 24 |
| **The return is inside the swing** | `epa_lost` averages over returns including 36 blocked-punt touchdowns | **Accepted.** The branch prices *whether* the defense got it; how far they ran is `core`, and document 21 §9 keeps returns out of the ledger |
| **Five blocked kicks are dropped** | They already carry a v1.2 fumble row (§3a) | **Accepted**, all five printed; the alternative books one possession twice |
| **Blocked field goals are priced twice over, in two components** | The FG component prices the block as a miss; this one prices the aftermath | **Open interaction.** Not a double count, but candidate 3 changes the first and the two must be measured together (§5f) |
| **Extra points are a class with almost no branch value** | Swing 0.523 EPA on a play the kicking team cannot score on | **Open.** Kept in for completeness; it dilutes the n-weighted screen and a punt-only component would screen higher |

---

## 7. Constants appendix

| Constant | Value | Where it lives |
|---|---|---|
| `RANDOM_SEED` | 20260818 | `research/38_blocked_kick_power.py`, `research/39_blocked_kicks.py` |
| `DATASETS` per scenario | 400 | inherited from `research/12_coinflips_power.py` |
| `MIN_CLASS_SIZE` | 30 | matches `components.fit_fumble_baseline` |
| Population rule | blocked FG **or** blocked XP **or** `punt_blocked == 1` | §3 |
| Exclusion rule | play already carries a v1.2 fumble row | §3a |
| Blocked kicks / eligible | **415 / 410** | §3, §3a |
| Games / eligible games | 383 / **378** | §3, §4b |
| Class rates | 0.6436 FG, 0.3393 punt, 0.6818 XP | §3c |
| Class swings | 1.726 / 1.585 / 0.523 EPA | §3c |
| Screen, n-weighted | **0.3091 EPA per event** | §3c |
| Entity-spread power at the reference | **0.177 — no gate** | §4a |
| **Gate B-4 floor** | **1.4392 pp** on 378 games | §4b |
| Gate B-6 dial values | w ∈ {0.00, 0.25, 0.50}, all three gated | §5g |
| `points_per_epa` | 0.8389 | `research/outputs/model_metadata_v12.json` |

Results are written back into this document as §8.

---

## 8. Results

*Script: `research/39_blocked_kicks.py`. The gates were committed at `03a1a2b`
before this script existed. Results in `research/outputs/39_blocked_kicks.json`.*

### The verdict, stated first

> **The branch is real, it identifies at 415 of 415, the blocker is not the
> recoverer, and the ledger sums. It does not ship, and it misses by more than
> any candidate so far: 0.222 pp against a 1.439 pp floor, a factor of 6.5. The
> screen predicted the direction and understated the size.**

| Gate | Statistic | Result | Verdict |
|---|---|---|---|
| **B-1** — branch point | Block denied and left in `core`; aftermath admitted; blocker is the recoverer 11% | — | **Pass** (§2) |
| **B-2** — identification | 415/415 resolved, two "no recovery" rows and five overlap rows printed | 100% vs 95% | **Pass** (known at writing) |
| **B-3** — entity spread | Not run; power 0.177 | — | **Absent by design** (§5d) |
| **B-4** — materiality | median \|ΔDTW\| **0.222 pp** vs the **1.4392 pp** floor | Misses by 6.5× | **FAIL** |
| **B-5** — the ledger must still sum | 0 shared plays, 0 duplicates | — | **Pass** |
| **B-6** — verdict independent of `w` | Fails at 0.00, 0.25 and 0.50 alike | Identical | **Pass** |

Per the decision rule committed in §5h — *B-4 fail → ship nothing, report as
measured and immaterial* — **the component is measured, reported and left out of
the ledger.** No production code changes and simulator v1.2 stays authoritative.

### Gate B-4 — the numbers

378 games carry an eligible blocked kick. Against v1.2, with the fumble,
field-goal and extra-point draws held to their own seeded generators in both
arms:

| | `w = 0.00` | `w = 0.25` | `w = 0.50` |
|---|---|---|---|
| **Median \|ΔDTW\|** | **0.222 pp** | 0.187 pp | 0.160 pp |
| Mean \|ΔDTW\| | 1.765 pp | 1.495 pp | 1.232 pp |
| Max \|ΔDTW\| | 36.86 pp | 27.66 pp | 19.45 pp |
| Median \|Δ deserved margin\| | 0.519 pts | 0.389 pts | 0.259 pts |
| Games whose DTW side flips | 7 | 4 | 3 |
| **Against the 1.4392 pp floor** | **Fail** | **Fail** | **Fail** |

The run's own v1.2 median half-width came out at 1.3767 pp against the 1.4392 pp
committed in §4b — a 0.06 pp redraw difference, well inside the ±0.05 pp spread
document 16 measured and far too small to matter at a 6.5× miss. **The
pre-registered floor is the one the gate is read against**, and no redraw was
triggered because nothing landed within 0.1 pp of it.

### The structural reason it fails, which is the round's real finding

**This candidate moves the deserved margin by *more* than the kickoff-muff
candidate did — 0.519 points against 0.344 — and moves DTW by *half* as much.**
That is not a contradiction and it is worth understanding, because it generalises.

A blocked kick happens on a play that **already carries a large luck row**. The
field-goal component prices a blocked field goal as an ordinary miss, and that
row is one of the widest the simulator prints. So the games this component fires
in are games whose DTW distribution is already broad: v1.2's median 89% half-width
on them is 1.44 pp, twice the 0.72 pp on kickoff-muff games. The new row adds
half a point of deserved margin into a distribution that is already half a point
wide, and the median game barely notices.

**A component that fires on plays that already carry a big luck row is penalised
twice: the floor it must clear is raised by the incumbent row, and its own
marginal effect is diluted by the same row.** Nothing in documents 16, 18, 20 or
24 anticipated this, and it is the first structural rule the project has that
predicts a materiality failure from the *location* of a candidate rather than
from its size.

### Gate B-6 — the dial is not the problem

Identical verdict at all three values of `w`, so B-6 passes unmodified. As in
document 24, the candidate fails on size at every dial rather than flipping with
one, which is the cleaner of the two failures.

---

## 9. What this round changes, and what it teaches

### Nothing ships

Blocked-kick aftermath is closed. The 410 eligible blocked kicks stay outside the
ledger, and the reason is now a measured 0.222 pp rather than an open scouting
question.

### Four things worth carrying forward

1. **The screen works, and it under-predicted.** Document 24 §9 said to compute
   `p(1 − p) × swing` before writing a pre-registration, and §5a wrote the
   prediction down in advance: roughly 0.3 pp against a 1.44 pp bar. The answer
   was 0.222 pp. The screen got the direction and the verdict right and was
   optimistic on the magnitude by about a third, because it takes no account of
   the dilution effect above. **Screen and floor together, not the screen
   alone.**
2. **Where a candidate fires matters as much as how big it is.** The dilution
   finding above is the round's most transferable result: a candidate landing on
   plays that already carry a wide luck row faces a raised floor and a damped
   effect at the same time. Any future special-teams candidate on a kick play
   inherits this.
3. **A number beat an argument, cheaply.** The blocker-is-the-recoverer check
   turned the single hardest Gate A objection in the round into 16 of 144. It
   cost one regex and it is the kind of check that should be attached to every
   Gate A memo that has a "but isn't this really one continuous play" objection.
4. **Refusing to report a gate is sometimes the honest move.** Gate B-3 was
   designed and then deliberately not run, because at power 0.177 a pass would
   have read as "there is no team skill here" when it means "we cannot see". The
   register carries the reason instead of a misleading number.

### What would reopen this

- **Candidate 3's interaction.** If the field-goal and extra-point components
  ever stop pricing a blocked kick as an ordinary miss, the incumbent row on
  these plays narrows and both halves of the dilution effect weaken. **The two
  changes must then be measured together, in one arm**, never in sequence.
- **A punt-only component.** Extra points dilute the n-weighted screen from 0.375
  to 0.309 and contribute almost nothing. A punt-only version would screen higher
  — and would still have failed here by a factor of four, so this is a note for
  method rather than a live route back.
- **A lower floor**, the standing reopening condition from document 20 §8.
