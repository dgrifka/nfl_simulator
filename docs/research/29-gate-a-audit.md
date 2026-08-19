# 29 — A sweep of the ledger for Gate A violations

*Written 2026-08-18. Script: `research/43_gate_a_audit.py`, results in
`research/outputs/43_gate_a_audit.json`. **This document has no gate and no
threshold, and is therefore not pre-registered.** It reads no statistic against a
bar: it enumerates populations, prints rejected rows, and adjudicates each in
writing. Document 15 and document 23 are the precedent — a scouting document,
not a round.*

*Inputs: document 05 §2 (Gate A and the branches each component was admitted
on), document 18 §2 (the out-of-bounds argument), document 20 §9 (**print the
rejected rows, not their count**), document 25 §2 (the block, the aftermath, and
the ruling that a branch's price may contain football), document 26 §2 (the
violation this sweep was commissioned to look for more of), document 28 (the
gate that would govern anything found here — **written and committed before this
sweep ran**).*

---

## 1. One-page story

### The question

Document 26 found a Gate A violation inside a shipped component **by accident**,
while asking a pricing question. That is not a search strategy. This document is
the systematic version:

> **For every row type the v1.2 ledger can book, does any booked play's branch
> get resolved by something Gate A denies?**

### The answer, stated first

> **No new violation. Nineteen screens across all three row types: two flag the
> blocked kicks document 26 already found, sixteen return a clean bill, and one
> finds a curiosity that costs exactly zero EPA. The fumble component — the
> oldest, largest and most-revised component in the ledger, 6,505 rows — is
> clean on every screen.**

| Verdict | Screens | Which |
|---|---|---|
| **KNOWN** — already documented | **2** | FG-1 and XP-1, the 192 blocked field goals and 110 blocked extra points of document 26 |
| **FINDING** — new, and registered | **1** | FUM-2, a fumble class whose branch has only ever resolved one way |
| **CLEAN** | **16** | everything else |

### Four things to hold onto

1. **The blocked kicks are the only violation in the ledger.** That is a useful
   fact for the decision document 28 asks for: adopting the correctness gate
   does **not** open a queue of pending corrections. Its known caseload after
   this sweep is exactly one candidate.
2. **The one finding costs nothing, by arithmetic.** 68 aborted-snap passes form
   a class that has retained the ball 68 times out of 68, so its rate is 1.0 and
   every row books `(1 − 1) × swing = 0`. It is a class that should be pooled,
   not a violation to correct.
3. **One screen fired and reading its rows cleared it.** Two fumble rows sit on
   plays whose text contains "No Play". Counting them would have produced a
   finding; reading them showed the "No Play" belongs to a ruling that was
   *reversed on replay*, and the play that stands is a real fumble in both cases.
   Document 20 §9's rule earned its place inside this audit rather than being
   quoted by it (§4).
4. **Every violation and conditioning bug this project has found lives in the
   same place**: a charted categorical field with more levels than the branch it
   was collapsed into (§6).

---

## 2. Method

**What was swept.** Every ledger row v1.2 can emit, and there are three kinds:

| Component | Rows | Branch document 05 §2 admitted |
|---|---|---|
| `fumble` | **6,505** | a loose ball on the turf that nobody controls |
| `field_goal` | **10,731** | a kick in flight |
| `extra_point` | **12,818** | a kick in flight, same structure |

Nothing else books a row. Interceptions, penalties, returns, drops, fourth
downs, two-point tries and all three sequencing rows are `none` in document 05
§3's treatment table, and **a component that neutralizes nothing cannot
neutralize a denied play** (screen X-2).

**How each population was screened.** For each component, every charted field
that could indicate the branch was resolved by something other than the admitted
mechanism, enumerated exhaustively rather than sampled:

- the **outcome field's own levels** — every value of `field_goal_result`,
  `extra_point_result`, and the fumble disposition;
- whether the play **counted** (`aborted_play`, and "No Play" in the play
  description, which is the only marker of a negated play in this data);
- whether the branch has a **charged entity** (`kicker_player_id`);
- whether the branch's **price** contains a played-out sequence (return
  touchdowns on missed kicks, defensive two-point returns on failed tries);
- whether the branch is **degenerate** (a class rate of exactly 0 or 1);
- whether a **rule rather than a bounce** decides the disposition (out of
  bounds, touchbacks);
- whether the play carries **more than one branch** (second fumbles, plays
  carrying rows from two components).

**Both directions are printed.** Sixteen clean bills appear below with the same
weight as the findings. A sweep that reports only its hits is indistinguishable
from a sweep that only looked where it expected to find something, and the point
of commissioning this one was that the previous find was luck.

---

## 3. The sweep

### Field goal — 10,731 rows

Result levels: **9,085 made, 1,454 missed, 192 blocked.**

| Screen | Question | Flagged | Verdict |
|---|---|---|---|
| **FG-1** | Is the branch resolved by a defender rather than by the flight of the ball? | **192** | **KNOWN violation.** Document 26 §2. The ball never flew; the protection lost. Correction pending the maintainer's decision on documents 26 and 28 |
| **FG-2** | Does any booked field goal end before the kick, on a botched snap? | **0** | **Clean.** No field goal inside the shipped mask carries `aborted_play`. An aborted field-goal snap is charted as a run or a pass and is outside the component entirely — an omission if anything, not a violation |
| **FG-3** | Does the ledger book a coin on a kick that did not count? | **0** | **Clean.** Thirty-one kicks carry a penalty and every one is a dead-ball or post-play foul — unnecessary roughness (14), unsportsmanlike conduct (8), illegal block, taunting, face mask — enforced on the kickoff. The kick itself counted in all 31 |
| **FG-4** | Is any booked kick charged to nobody? | **0** | **Clean.** Every row names a kicker, so every row has an entity to carry `w` |
| **FG-5** | Does the swing on a missed field goal include a played-out sequence? | **0** | **Clean.** Zero missed field goals were returned for a touchdown in ten seasons |

### Extra point — 12,818 rows

Result levels: **12,101 good, 607 failed, 110 blocked.**

| Screen | Question | Flagged | Verdict |
|---|---|---|---|
| **XP-1** | Is the branch resolved by a defender? | **110** | **KNOWN violation.** The same defect as FG-1 |
| **XP-2** | Does any booked try end before the kick? | **0** | **Clean** |
| **XP-3** | Does the ledger book a coin on a try that did not count? | **0** | **Clean**, despite 190 tries carrying a penalty — defensive offside (66), unnecessary roughness (49), leverage (28), illegal formation (15). A live-ball foul on a try is enforced on the following kickoff or declined; the try is charted with a result either way |
| **XP-4** | Does the swing on a failed try include a defensive return? | **9** | **Clean, and printed because it is not empty.** Nine failed tries were returned by the defense for two points. That value sits inside the branch's mean EPA, which document 25 §2 ruled is *pricing* the branch rather than neutralizing a second one |

### Fumble — 6,505 rows of 6,507 fumbles

| Class | n | Own recovery | **Retained** | Fumbler recovers his own ball |
|---|---|---|---|---|
| pass, live | 3,226 | 40.6% | 51.1% | 17.3% |
| run, live | 1,273 | 36.4% | 46.3% | 18.1% |
| run, aborted snap | 974 | 74.0% | 76.9% | 38.5% |
| punt | 757 | 57.2% | 68.6% | 41.7% |
| kickoff | 201 | 41.8% | 51.2% | 13.4% |
| **pass, aborted snap** | **68** | **100%** | **100%** | 55.9% |
| field goal | 4 | 50.0% | 75.0% | 25.0% |
| punt, aborted | 2 | 100% | 100% | 50.0% |

| Screen | Question | Flagged | Verdict |
|---|---|---|---|
| **FUM-1** | Does the ledger book a coin on a fumble that did not happen? | **2** | **Clean — after reading them.** See §4 |
| **FUM-2** | Does any class have a branch that always resolves the same way? | **68** | **FINDING.** See §5 |
| **FUM-3** | Is a high self-recovery rate evidence that somebody made a play? | **1,547** | **Clean, adjudicated.** See §4 |
| **FUM-4** | Is a ball crossing the sideline resolved by a rule rather than a bounce? | **602** | **Clean.** Document 18 §2 argued these into the population deliberately. The rule converts the bounce into a retention; the bounce is still the branch, and excluding them was the conditioning bug v1.2 fixed |
| **FUM-5** | Is a touchback a branch or a rule? | **61** | **Clean.** Same structure as FUM-4 — the ball bounced into the end zone and the rule says who gets it. The bounce is the branch and the rule is the price |
| **FUM-6** | Does a play with two loose balls book one row or two? | **64** | **Clean.** One row each, priced on the first fumble. That *understates* the luck on those plays — an omission of 64 second bounces, governed by the materiality floor and far below it — and it books nothing false |
| **FUM-7** | Do laterals put a played-out sequence inside the fumble branch? | **819** | **Clean, and the flag is a red herring.** `lateral_recovery` fires on 819 rows of which **797 are aborted snaps** and only 20 mention a lateral in the play text. It is not the flag its name suggests |
| **FUM-8** | Does the fumble component book the aftermath of a blocked kick? | **4** | **Clean, and worth knowing.** Four blocked field goals also carry a fumble row, so **v1.2 already books four of document 25's 415 aftermath events**. Document 25 §2 admitted that branch, so the four rows are correct — and they are the four document 26's correction must not delete, which is why that correction narrows the kick masks and never the play frame |
| **FUM-9** | Does any fumble book a row without a resolved branch? | **2** | **Clean.** Two of 6,507 fumbles have neither a recovering team nor an out-of-bounds flag, and `_fumble_frame` drops both |

### Cross-cutting

| Screen | Question | Flagged | Verdict |
|---|---|---|---|
| **X-1** | Does any play get neutralized twice for the same event? | **4** | **Clean.** Four plays carry both a kick row and a fumble row and they price different things: whether the kick went through, and who came up with the loose ball afterwards. Two branches on one play is not double-counting |
| **X-2** | Can a non-neutralized component hide a violation? | — | **No, structurally.** Interceptions, penalties, returns, drops, fourth downs, two-point tries and the sequencing rows book no rows at all |

---

## 4. The two adjudications that took actual work

### FUM-1 — the screen fired and reading the rows cleared it

Two fumble rows sit on plays whose description contains **"No Play"**, which in
this data is the only marker of a play negated by penalty. A luck row on a play
that did not happen would be a real defect: the branch resolved nothing.

**Both rows are correct, and only reading them shows it.** In each case the
"No Play" belongs to a ruling that was **reversed on replay**, and the text that
follows describes the play that actually stands:

- **2017_14_DET_TB, play 837.** Ruled an incomplete pass with an unnecessary
  roughness penalty and no play. *"Detroit challenged the incomplete pass ruling,
  and the play was REVERSED"* — a 21-yard completion, stripped, recovered by
  Detroit. A real fumble, a real loose ball, `penalty` charted as 0.
- **2020_11_CIN_WAS, play 1114.** A Burrow scramble, fumbled, recovered by
  Washington, a safety, a holding penalty and "No Play" — then *"the play was
  REVERSED"* to a fumble recovered in the end zone for a touchback, with the
  holding **declined**.

**This is document 20 §9's rule doing its job inside an audit rather than being
cited by one.** That rule was written after document 15's cross-check counted
agreements without reading disagreements and nearly cost the project a wrong
population. Here the arithmetic ran the other way — counting the rejected rows
would have produced a spurious finding, and reading them produced a clean bill.

### FUM-3 — does a fumbler recovering his own ball break the branch?

1,547 of 6,505 booked fumbles are recovered by the player who fumbled, and the
rate is not uniform: 17–18% on live scrimmage plays, **38.5% on an aborted run
snap**, 41.7% on a punt, 55.9% on an aborted pass snap. It is fair to ask whether
"the quarterback fell on his own snap" is a coin nobody resolved or a play
somebody made — the same objection document 25 §2 answered for blocked kicks with
the 11% blocker-is-the-recoverer measurement.

**Adjudicated clean, and the reason is in the wording of Gate A itself:**

> Is there a moment where the outcome is resolved by a mechanism outside either
> team's control, **conditional on the state both teams created**?

The ball's location is part of the state both teams created. A quarterback
standing over his own aborted snap recovers it more often for the same reason a
punt muff is retained 64% of the time and a normal running fumble only 46%:
**proximity, not control.** A high branch probability is not evidence of an
absent branch — that is exactly the distinction document 05 §2 draws when it
admits a kick in flight, which the kicker also mostly determines.

And the ledger already prices it. The class rates exist precisely so that a
76.9% retention on an aborted snap is not charged against a team as though it
were a 46.3% coin — document 05 §3's warning that a flat 50/50 *"would book a
fake 26-point bad-luck charge against every offense that recovered its own
botched snap."*

**What would reopen it:** a measurement showing the fumbler recovers his own ball
more often than proximity explains — for instance, if self-recovery persisted as
a team-season skill after conditioning on class. That is a persistence question
with its own pre-registration, and this audit does not run it.

---

## 5. The one finding — a branch that has never had two outcomes

**FUM-2. The `pass / aborted snap` class has retained the ball on 68 of 68
plays.** Its fitted class rate is exactly 1.0, so every row in it books

```
luck = (retained − p) × swing = (1 − 1) × swing = 0
```

**It costs exactly zero EPA, in every game, by construction**, and no verdict
anywhere in the product depends on it. It is a finding rather than a clean bill
because a branch with one observed outcome is not a coin, and a class rate of
exactly 1.0 estimated from 68 plays is a statement the data cannot really make —
the 69th aborted pass snap that the defense recovers would move the rate to
98.6% and every one of those 69 rows would suddenly book luck.

**Registered, not acted on**, for three reasons:

1. It books nothing today, so there is nothing false in the ledger to correct.
2. The fix is not an exclusion but a **pooling rule** — shrink a thin class's
   rate toward the pooled rate rather than taking it at face value. Document 19
   §3 already registered that same defect from the other end: two classes
   totalling six plays carry branch means estimated from a handful of plays, and
   the fix named there is *"pool the swing on class size, which would change
   v1.1's numbers too and therefore needs its own round."* **This finding is more
   evidence for that round, not a round of its own.**
3. Under document 28's proposed gate it would **not** qualify as a correction:
   nothing false is booked, so there is no violation to correct.

The `punt / aborted` class, at 2 plays and 100%, is the same shape and the same
non-problem.

---

## 6. What the sweep teaches

### Every defect of this kind this project has found sits in the same place

| Round | The defect | The field |
|---|---|---|
| Document 18 | Fumbles out of bounds were excluded, because the population conditioned on a recovering team | **Disposition had three states** — recovered own, recovered opponent, out of bounds — and the branch was collapsed to two |
| Document 26 | Blocked kicks are priced as misses | **`field_goal_result` has three levels** — made, missed, blocked — and the branch is collapsed to two |
| This sweep, FUM-2 | A class rate of exactly 1.0 | A class thin enough that the branch has only shown one of its two states |

**Two of the three are the same mistake: a charted categorical field with more
levels than the branch it was collapsed into.** That is a concrete rule for
future components, and it is cheaper than an audit:

> **Before binarizing an outcome field, enumerate its levels and say in writing
> what happens to each one.** Document 05b §2 did exactly this for blocked kicks
> — *"blocked kicks count as misses"*, written down and defended — which is why
> document 26 could find it by reading rather than by measuring.

### What this sweep cannot see

Stated so nobody reads the clean bills as wider than they are:

- **Anything not in a charted field.** A ball recovered in the air rather than
  off the turf, a "fumble" that was really a muffed handoff nobody contested, a
  play where the whistle blew early — none of these are identifiable, and the
  sweep is blind to all of them.
- **Omissions.** A branch the ledger never books cannot show up in a sweep of
  the rows it does book. Kickoff muffs (document 24), blocked-kick aftermath
  (document 25) and the 64 second fumbles of FUM-6 are omissions, and they are
  governed by the materiality floor, not by document 28's gate.
- **Plumbing.** The largest defect Phase 7 found is not here: document 27 §14f
  shows the simulator discards three fitted parameters and prices a 55-yard field
  goal 6.8 percentage points too generously. **That is a play priced wrongly, not
  a play booked wrongly**, and no Gate A sweep would ever have found it. It needs
  its own round.
- **The branch's price.** FG-5, XP-4 and FUM-4/5 all touch the question of what
  belongs inside a branch's EPA, and all three defer to document 25 §2's ruling
  that a branch's value may contain football. If that ruling is ever revisited,
  those three screens change verdict together.

---

## 7. What this means for the decision in document 28

Document 28 §6's last row was the one that would worry a careful reader:
*"whatever document 29's audit finds — governed, and unknown at writing."* The
amendment text was fixed and committed before this script existed, so the answer
was genuinely unknown to it.

**The answer is: nothing new.** If the maintainer adopts the correctness gate:

- its **known caseload is one candidate**, document 26's blocked kicks, on 302
  rows in 287 games;
- **no other correction becomes shippable** as a side effect;
- documents 24 and 25 stay closed, because they are omissions;
- FUM-2 does not qualify, because it books nothing false.

If the maintainer rejects it, this sweep still stands as the record that the ledger
contains exactly one known violation and that somebody looked properly.

---

## 8. Register

| Item | Evidence | Status |
|---|---|---|
| **Blocked kicks are the ledger's only Gate A violation** | 19 screens, §3 | **Known, measured, and awaiting a decision** on documents 26, 27 and 28 |
| **`pass / aborted snap` has a degenerate class rate of 1.0** | §5, 68 of 68 | **Open, costs zero EPA.** Folds into the class-pooling round document 19 §3 already named |
| **64 plays carry a second fumble the ledger never books** | FUM-6 | **Open, an omission.** Materiality floor governs; far below it |
| **`lateral_recovery` does not mean what it is named** | FUM-7: 797 of 819 are aborted snaps | **Data note.** No consumer in this repository reads the column |
| **The sweep is blind to anything not charted** | §6 | **Accepted, by design.** Stated so the clean bills are not over-read |
| **Aborted field-goal snaps are outside the kicking components entirely** | FG-2 | **Open, an omission**, and a small one: charted as runs or passes, and their fumbles are already booked in the run/aborted class |
