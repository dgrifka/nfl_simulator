# 26 — Blocked kicks priced as misses, pre-registered

*Written 2026-08-18, **before `research/41_blocked_pricing.py` existed**. The
luck currently booked, the class tables, the touchdown audit and the materiality
floor: `research/40_blocked_pricing_power.py`, results in
`research/outputs/40_blocked_pricing_power.json` and reproduced in §3–§4.
Committed to git before the fit produces a number.*

*Inputs: documents 05 (Gate A and the one rule), 05b (the kicker model this
correction does **not** change), 09 §2 (the extra-point branch), 18/19 (the
widening template and the ship record), 23 §C1 (the finding), 24 §9 (the screen),
25 §2 (the ruling that a block fails Gate A) and 25 §8 (the dilution effect that
makes this floor high).*

*Tier: **model change** — it changes which plays two shipped components run on,
and therefore the numbers the product prints.*

---

## 1. One-page story

### The question

The field-goal and extra-point components condition on an attempt being made and
then treat **blocked** as a value of the outcome. A blocked kick is booked as a
miss, and the kicker's coin is charged for it. Document 23 §C1 sized the defect
as the **0.55 EPA** gap between a blocked kick and a true miss.

### The answer, stated first, and it is not the answer document 23 expected

**The defect is five times larger than §C1's framing, because the right fix is
not to re-price the block but to stop pricing it at all — and the simulator is
currently handing about 2.8 points of undeserved credit to a team whose
protection was beaten.**

| | Value |
|---|---|
| Blocked field goals carrying a field-goal luck row | **192** |
| Blocked extra points carrying an extra-point luck row | **110** |
| Games touched | **287** |
| Mean \|luck\| booked on a blocked **field goal** | **3.361 EPA** (2.82 points) |
| Mean \|luck\| booked on a blocked **extra point** | 0.941 EPA |
| Mean \|luck\| on a **non-blocked** field goal, for scale | 0.939 EPA |
| Document 23 §C1's sizing of the defect | 0.55 EPA |
| Gate P-3 floor | **1.6250 pp** — statistic unseen |

**A blocked field goal carries three and a half times the luck of an ordinary
field goal**, because `realized = 0` is charged against a modelled make
probability near 0.86 on a swing near four EPA. The component says the kicking
team was unlucky. The kicking team's protection was beaten.

### Four things to hold onto

1. **This is the only Phase 6 candidate that removes rows rather than adding
   them, and the only one that could plausibly ship.** Candidates 1 and 2 were
   closed for being too small. This one is large *because* the incumbent is
   wrong, not because a new coin was found.
2. **The correction is exclusion, and the argument is Gate A, not arithmetic.**
   Document 25 §2 already ruled that a block is a defensive play that fails Gate
   A. A component may not neutralize a play the gate denies. §2 below states why
   the alternative — a separate blocked class — is not merely worse but
   inadmissible.
3. **Document 23 §C1 measured the wrong quantity, and that is worth recording.**
   The 0.55 EPA figure is the difference between two *pricings* of a blocked
   kick. The quantity that matters is how much luck is booked at all, which is
   3.36 EPA on a field goal. A defect audit that asks "how wrong is the number"
   can miss "should there be a number".
4. **The floor is the highest any candidate has faced.** 1.6250 pp on the 287
   games, against 0.7222 pp for kickoff muffs, for the reason document 25 §8
   found: these games already carry the widest luck rows the simulator prints.
   The candidate has to beat a bar it raised itself.

### Statistic convention

Posterior means with 89% equal-tailed intervals, as everywhere else.

---

## 2. Gate P-1 — the branch-point memo, and why the fix is exclusion

> **Is there a moment where the outcome is resolved by a mechanism outside either
> team's control, conditional on the state both teams created?**

### A blocked kick — **FAIL. It is not a branch and it must not be neutralized**

Document 05 §2 admitted the field-goal component on a specific mechanism: **a
kick in flight**. The ball leaves the foot and whether it drifts inside the
upright is not something either side can influence afterwards. That is the coin.

A blocked kick never reaches that coin. Somebody beat a block, got a hand up and
ended the play before the branch opened. Document 25 §2 ruled this out in the
other direction — denying the block admission as a *new* component — and the
same ruling applies here with the sign flipped: **a play Gate A denies must not
be neutralized by a component that happens to already contain it.**

### Why a separate blocked class is inadmissible, not merely worse

The obvious alternative is to keep blocked kicks in the population and give them
their own class: `luck = (blocked − p_block) × swing_block`. It is inadmissible
for a reason that has nothing to do with its size:

- The quantity it neutralizes is **being blocked**, and `p_block` is a
  protection-quality rate. Neutralizing it would credit back to the kicking team
  the part of a block that its own protection caused — the exact error document
  05 §2 records for penalties, where a correct `w ≈ 0.42` would have neutralized
  half of every team's penalty EPA and the conclusion would have been wrong.
- It would require the component to name a charged entity for a play whose
  outcome is the defense's doing. The kicker is charged today; the protection is
  who failed; neither is a branch.

**Exclusion is therefore the only Gate-A-compatible correction**, and this
document does not treat the choice as a tuning parameter to be measured. It is
argued, and if the argument is wrong the candidate is wrong.

### What exclusion means for the play

The blocked kick's whole EPA stays in `core`, charged to the kicking team as
deserved. That is the right home for it: the protection lost, and losing your
protection is football.

---

## 3. Data

- **Grain of a row**: one kick attempt.
- **Source**: `data/pbp/*.parquet` 2016–2025.
- **Population removed**: `field_goal_result == "blocked"` inside
  `fg_attempt_mask` (**192** kicks, all of which carry a `kick_distance` and so
  are all inside the shipped population), and
  `extra_point_result == "blocked"` inside `xp_attempt_mask` (**110**).
- **Games touched**: **287**.

### 3a. What v1.2 books on these plays today

| Component | Blocked? | n | mean \|luck\| (EPA) | median \|luck\| (EPA) |
|---|---|---|---|---|
| field_goal | no | 10,539 | 0.939 | 0.539 |
| **field_goal** | **yes** | **192** | **3.361** | **3.482** |
| extra_point | no | 12,708 | 0.108 | 0.065 |
| **extra_point** | **yes** | **110** | **0.941** | **0.946** |

Blended across both components: **2.480 EPA of luck per blocked kick, 302 rows,
287 games.** For comparison, the two candidates this round already closed carried
0.395 EPA (kickoff muffs) and 0.309 EPA (blocked-kick aftermath) of luck per
event. **This one is six to eight times larger per event.**

### 3b. What exclusion does to the class tables

| | With blocks *(v1.2)* | Without blocks | Change |
|---|---|---|---|
| Field-goal attempts | 10,731 | **10,539** | −192 |
| Field-goal make rate | 0.8466 | **0.8620** | +1.54 pp |
| Field-goal `epa_missed` | −3.1782 | **−3.1139** | +0.064 EPA |
| Extra-point attempts | 12,818 | **12,708** | −110 |
| Extra-point good rate | 0.9441 | **0.9522** | +0.82 pp |
| Extra-point `epa_failed` | −0.9538 | **−0.9286** | +0.025 EPA |

**The swing every remaining kick carries shrinks by under 2%**, because blocked
kicks were only slightly worse than true misses. That is the honest content of
document 23 §C1's 0.55 EPA: it is the size of the *second-order* effect, on the
10,539 kicks that are not blocked. The first-order effect is the 302 rows that
disappear.

**The make-probability model is held fixed and this is a disclosed
inconsistency.** `FieldGoalModel` was fitted on a population that counts blocks
as misses, so after exclusion it understates `p_make` by roughly the rates above
— 1.54 pp on field goals, 0.82 pp on extra points. The bias exists in v1.2 too
and this correction neither creates nor worsens it; refitting the document 05b
model on non-blocked attempts is a **separate model change with its own
writeup**, named in §6 and in §9.

### 3c. The touchdown audit document 23 §C1 asked for

14,200 touchdowns, 12,818 extra-point attempts, 1,302 two-point attempts —
**80 touchdowns with neither charted.** Matching each touchdown against the tries
in the fifteen rows that follow it reproduces the same 80 and attributes them:

| Quarter | n | Explanation |
|---|---|---|
| **5 (overtime)** | **60** | A walk-off overtime touchdown ends the game and **no try is attempted, by rule.** Correctly absent |
| **4** | **20** | Fourth-quarter touchdowns with no try charted anywhere afterwards. **Genuinely missing from the data**, 0.16% of tries |

**Recorded, not acted on.** Twenty absent extra-point rows across ten seasons is
about two absent luck rows a season at a mean |luck| of 0.108 EPA. It is below
any threshold this project uses and it belongs in the register rather than in a
round.

---

## 4. The materiality floor *(Gate P-3's threshold)*

**287 games contain a blocked kick.** Simulator v1.2's median 89% DTW interval
half-width on them is **1.6250 pp** — the highest floor any candidate in this
project has faced, above blocked-kick aftermath's 1.4392 pp and more than double
the kickoff-muff candidate's 0.7222 pp.

Document 25 §8 explains why and the explanation applies with double force here:
these games carry the widest luck rows the simulator prints, **and the row this
candidate removes is itself one of the widest.** The floor is raised by the very
thing being corrected.

**No treatment arm was run.** The P-3 statistic does not exist yet.

### The screen, and what it predicts

Document 24 §9's screen does not apply unmodified — this candidate removes an
existing row rather than adding a new coin — so its analogue is the mean |luck|
removed per event, **2.480 EPA**. Calibrating on the two closed candidates on
comparable game populations: blocked-kick aftermath converted 0.309 EPA per event
into 0.222 pp of median |ΔDTW| on almost this exact game population, a ratio of
0.72 pp per EPA. **At that ratio this candidate lands near 1.8 pp against a
1.625 pp floor — a pass, but by well under 20%.**

That prediction is written down here, before the fit, precisely because it is
close. §5a records it as the expectation and §5d commits the redraw rule that a
near-miss requires.

---

## 5. Pre-registered gates

### 5a. Which gates are unseen, and what is expected — stated first

| Gate | Known at writing? |
|---|---|
| **P-1** — the branch point | Settled by argument in §2, following document 25 §2 |
| **P-2** — identification | **Yes.** The blocked flag is a charted field; 192 and 110 exactly |
| **P-3** — materiality floor | **No. Genuinely unseen, and the binding gate** |
| **P-4** — the ledger must still sum, and lose exactly the right rows | Enforced as a check |
| **P-5** — a dial gate | **Not applicable, and §5e says why** |

**Stated before the fit: P-3 is expected to pass, narrowly — near 1.8 pp against
1.625 pp.** A prediction inside 20% of a threshold is not a prediction, which is
why §5d's redraw rule matters more here than in any previous round.

### 5b. Gate P-1 — the branch point

Settled in §2: a blocked kick **fails** Gate A, so the component must not
neutralize it, and the correction is exclusion rather than re-pricing.

### 5c. Gate P-2 — identification *(known, recorded)*

**Statistic:** the count of blocked kicks inside each shipped attempt mask.

**Pass rule:** every blocked kick is identifiable from a charted field with no
text parsing, and the counts reconcile with document 23 §C1.

**Result, known at writing:** 192 blocked field goals — all carrying a
`kick_distance`, so none is silently outside `fg_attempt_mask` — and 110 blocked
extra points. Both match document 23 §C1 exactly. **Pass.**

### 5d. Gate P-3 — the materiality floor *(the binding gate)*

**Statistic:** median |ΔDTW| across the **287 games containing a blocked kick**,
comparing simulator v1.2 with the corrected components, with the fumble draws
generated from their own seeded generator in both arms so the difference is the
field-goal and extra-point rows and nothing else.

**Pass rule:** ≥ **1.6250 pp**, v1.2's own median 89% DTW half-width on those
same games (§4).

**Power and the redraw rule, which binds here.** The statistic is a median over
287 games and document 16 measured its redraw spread at ±0.05 pp on 155 games.
**Because §4 predicts a result within 20% of the floor, eight redraws are run
unconditionally in this round rather than only on a near-miss**, and the verdict
is the one that holds in at least six of eight. A gate this close is not decided
by one replay of the coin.

**The second population is reported beside the first.** Excluding blocked kicks
also shifts the swing on all 10,539 remaining field goals and 12,708 remaining
extra points, so **every game with a kick moves a little**. Document 18 §4b's
two-population rule applies: the floor is read on the 287 games, the all-games
median is reported next to it, and neither stands in for the other.

**What each outcome means:**

- **Pass** → the correction ships as simulator v1.3, subject to P-4 and to
  the maintainer's approval.
- **Fail** → **ship nothing in this round**, and record the uncomfortable
  result: *a component is knowingly neutralizing a play Gate A denies, and the
  correction is below the materiality floor.* §9 states in advance what that
  would mean, because it is a genuinely awkward verdict and it should not be
  improvised after the fact.

### 5e. Gate P-5 — why there is no dial gate

Document 20 §5f's gate applies to any candidate that **assumes** a trust dial the
data cannot read. This candidate assumes none:

- It removes rows. A removed row has no expectation to shrink.
- The field-goal and extra-point components' shrinkage is the document 05b
  kicker model's, measured at 0.285 for a median kicker-season, and this
  correction does not touch it.
- The only quantity it changes is which plays the components run on, which is a
  population question settled by Gate A rather than a dial.

Recording the absence explicitly, so that no future reader assumes the gate was
forgotten.

### 5f. Gate P-4 — the ledger must still sum

**Pass rule:** for every game, `deserved_margin == actual_margin −
Σ luck_epa · points_per_epa` to floating-point tolerance; the field-goal
component loses **exactly 192** rows and the extra-point component **exactly
110**; no other component's row count changes; and the blocked kicks' EPA is
present in `core`.

The last clause is the one worth checking rather than assuming: a correction that
removes a luck row must leave the play's EPA somewhere, and `core` is a residual.

### 5g. The decision rule, committed in advance

| P-1 | P-2 | P-3 | P-4 | **Action** |
|---|---|---|---|---|
| Pass | Pass | Pass | Pass | **Ship** as simulator v1.3 — **stop at the door and ask the maintainer.** This round is not pre-approved |
| Pass | Pass | **Fail** | — | **Ship nothing.** Record the defect as known, measured and below the floor, per §9 |
| — | — | — | **Fail** | **Ship nothing.** A ledger that does not sum is a bug |

### 5h. What shipping would mean, fixed now

- `fg_attempt_mask` gains `& (field_goal_result != "blocked")` and
  `xp_attempt_mask` gains the extra-point equivalent. Both masks are used by the
  baseline fitters and the event builders, so the class tables and the ledger
  change together and cannot drift apart.
- `fit_fg_baseline`'s docstring loses the sentence disclosing the simplification,
  because the simplification is gone.
- Document 05 §3's treatment table gains **no row** and loses none. Two existing
  rows have their populations narrowed, and the table's entry for each records
  that a blocked kick is `core`.
- v1.2 artifacts are left untouched and v1.3 writes alongside, as v1.2 did.
- The rematch validation is **not** re-run, for document 18 §6's reason.
- **the maintainer approves before any of this lands.**

### 5i. Kill and rollback

On failure the branch keeps the code, no production module changes, v1.2 stays
authoritative, and §8 is the record. On a pass the code still does not merge
until the maintainer says so, and the ship record is a separate document on the document
19 template.

---

## 6. Known-defect register

| Defect | Evidence | Status |
|---|---|---|
| **The make-probability model still counts blocks as misses** | Excluding them raises the empirical make rate by 1.54 pp (FG) and 0.82 pp (XP) | **Open, disclosed, inherited.** Present in v1.2 and unchanged by this correction. The fix is a refit of the document 05b model on non-blocked attempts and needs its own writeup |
| **20 fourth-quarter touchdowns have no try charted** | §3c | **Open, recorded, not acted on.** 0.16% of tries, two absent luck rows a season at 0.108 EPA each |
| **A blocked kick's EPA lands entirely in `core`** | Exclusion by construction | **Accepted.** It is the protection's failure and `core` is where football goes. Stated so no reader mistakes it for the EPA vanishing |
| **The 287 games are the same games document 25 measured** | Both candidates fire on blocked kicks | **Accepted.** Document 25 closed, so there is no interaction to measure; had it shipped, §5f of that document required the two to be measured in one arm |
| **`p_block` is never estimated** | The correction removes rather than re-prices | **Accepted, by design.** §2 argues estimating it would neutralize a protection-quality rate |

---

## 7. Constants appendix

| Constant | Value | Where it lives |
|---|---|---|
| `RANDOM_SEED` | 20260818 | `research/40_blocked_pricing_power.py`, `research/41_blocked_pricing.py` |
| Blocked field goals / extra points | **192 / 110** | §3 |
| Games touched | **287** | §3 |
| Mean \|luck\| per blocked FG / XP | **3.361 / 0.941 EPA** | §3a |
| Mean \|luck\| per blocked kick, blended | **2.480 EPA** | §3a |
| FG make rate, with / without blocks | 0.8466 / **0.8620** | §3b |
| FG `epa_missed`, with / without | −3.1782 / **−3.1139** | §3b |
| XP good rate, with / without | 0.9441 / **0.9522** | §3b |
| XP `epa_failed`, with / without | −0.9538 / **−0.9286** | §3b |
| **Gate P-3 floor** | **1.6250 pp** on 287 games | §4 |
| Redraws | **8, run unconditionally** | §5d |
| Touchdowns with no try charted | 80 = 60 overtime + **20 missing** | §3c |
| `points_per_epa` | 0.8389 | `research/outputs/model_metadata_v12.json` |

Results are written back into this document as §8.
