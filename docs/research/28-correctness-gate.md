# 28 — A correctness gate for Gate A violations, for the maintainer to accept or reject

*Written 2026-08-18. **This document enacts nothing.** It is a proposal with both
sides argued, a concrete amendment text, and an explicit list of what the
amendment would let into the ledger and what it would keep out — the document 21
template, because document 21 is the only other time this project has proposed
changing its own rules. No code changes, no fits, no thresholds. Document 05 §2
stands unchanged until the maintainer says otherwise.*

*Written **before `research/43_gate_a_audit.py` existed** and before its results
were known. That ordering is deliberate and §7c explains what it buys.*

*Inputs: document 05 §2 (Gate A, and the sentence this amendment turns on),
document 21 (the amendment that was rejected, and why), document 18 §4b (the
two-population reporting rule), document 20 §5f and §9 (the dial gate, and
rejected rows rather than counts), documents 24 and 25 (the two omissions the
materiality floor closed), document 26 §9 (the round that asked for this).*

---

## 1. The question, and the recommendation stated first

### The question

Document 26 measured a correction to a shipped component and closed it:

> The field-goal component books **3.361 EPA** of kicker luck on each of 192
> blocked field goals, and the extra-point component **0.941 EPA** on each of 110
> blocked extra points. A blocked kick is a play Gate A denies. Removing those
> rows moves the median affected game by **2.688 points** of deserved margin and
> flips **22** DTW verdicts — and it failed the pre-registered materiality floor
> at **1.167 pp against 1.625 pp**, in all eight redraws.

The rule as written said: ship nothing. Document 26 obeyed it, and then asked the
question this document answers:

> **Should a Gate A violation inside a shipped component be governed by the
> materiality floor at all?**

### The recommendation

> **No — and the reason is not that the floor is set too high. It is that the
> floor answers a question a correction never asks.** The materiality floor exists
> to decide whether a *new* ledger row is worth having: does it move a game by
> more than the uncertainty already printed on that game? That is a cost-benefit
> test on an addition, and it is a good one. Pointed at a correction it becomes
> *"keep the known error, because fixing it does not move the median game far
> enough"* — which is not a cost-benefit test, because there is no benefit being
> bought. The row already exists. The only question is whether it is true.

The proposed replacement is **Amendment C-1**, written out in §5. It governs one
narrow class of candidate — a correction to a play the ledger already books and
Gate A already denies — and for that class it replaces the materiality
*threshold* with a materiality *report*. Every other gate stays.

### Five things to hold onto

1. **This amendment removes rows. It never admits one.** Document 21's A-2 asked
   Gate A to let something *in* and was rejected for it. C-1 asks Gate A to take
   something *out* — the same gate, enforced in the direction it already points.
   §4 argues the case against anyway, because that symmetry is an argument and not
   a proof.
2. **The boundary is violations versus omissions, and it is load-bearing.** A
   kickoff muff the ledger never books (document 24) is an **omission**; the
   floor still governs it and document 24's verdict stands. A blocked kick the
   ledger books luck on (document 26) is a **violation**. §6 defends why the
   asymmetry is principled and not a convenience.
3. **The gate contains no threshold, and that is the mitigation for its
   non-blindness.** Its first test case is already measured and known (§7). A
   gate with no number in it cannot be tuned to a number.
4. **The circularity is real and its size is not measured.** A violation inflates
   the very floor its fix must clear, because the false row is part of the
   incumbent's interval width on exactly the games the fix touches. §3 states the
   argument and §4 states the honest limit of it.
5. **Nothing is enacted here.** Accept, reject, or accept-narrowed, exactly as
   document 21 offered.

---

## 2. What document 05 already says, read carefully

Gate A, in full, from document 05 §2:

> **Is there a moment where the outcome is resolved by a mechanism outside either
> team's control, conditional on the state both teams created?**
>
> [...] **Failing Gate A means a component is not neutralized *at any value of
> `w`*.**

The final sentence is the one this amendment turns on, and it is unconditional.
It does not say *a component that fails Gate A is not admitted*. It says such a
component **is not neutralized**, at any dial setting — and the dial's endpoints
include `w = 0`, `w = 1`, and every value in between. There is no reading of that
sentence under which a shipped component may keep neutralizing a play the gate
denies because the correction is small.

**So the situation document 26 documented is not a component awaiting a
cost-benefit test. It is a component operating outside document 05 §2 as
written.** The amendment proposed here does not grant a new permission. It writes
down the **procedure** for enforcing a prohibition the foundations document
already contains, and it says what evidence that procedure requires so that
"this is a violation" cannot be asserted without argument.

That framing matters for the strongest objection in §4 — that a correctness gate
is a backdoor. A backdoor lets something *in*. This gate has no inward direction
at all: **no candidate can enter the ledger through it.** The only actions it can
authorize are removing rows and re-pricing rows that already exist.

---

## 3. The case for the amendment

### 3a. The floor asks a question a correction does not pose

The materiality floor's justification, from document 24 §9 and document 25 §8, is
a complexity argument: a new component adds code, a class table, a population, a
new way for the ledger to be wrong, and a row on every affected game. It has to
earn that by moving the answer more than the answer's own uncertainty. Otherwise
it is *noise with extra steps*.

A correction buys none of that complexity. Document 26's correction **deletes**
302 ledger rows, **narrows** two masks by one clause each, and **removes** a
disclosed inconsistency from a docstring. It is not asking to be allowed to add
anything. Reading its size against a floor built for additions is a category
error, and the sentence the floor produces — *keep the known error* — is the
tell.

### 3b. The incumbent's error inflates the bar its own fix must clear

The floor is the incumbent's **own** median 89% DTW half-width on the affected
games. On document 26's population that is 1.6250 pp, the highest any candidate
in this project has faced, and part of why it is that high is that v1.2 books a
3.361 EPA luck row on every one of those games. A wide luck row makes a wide
interval; a wide interval makes a high floor.

**For an omission the floor is exogenous. For a violation it is endogenous.** A
kickoff muff the ledger never books contributes exactly zero width to the
incumbent's interval — it is not there. A blocked field goal the ledger *does*
book contributes the widest single row the field-goal component prints. So the
same rule, applied to the two cases, behaves differently in kind: for an omission
it is a bar; for a violation it is a bar the candidate is required to raise
before jumping it.

**The size of the circularity is not measured and this document does not claim
it.** §4's third objection states why it cannot be cleanly measured, and the
recommendation does not depend on it.

### 3c. The failure modes are not symmetric, and the floor treats them as if they were

| | Ledger prints | Cost of the error |
|---|---|---|
| **Omission** — a coin exists and is not booked | Nothing false. The game's luck is understated by a quantity the register names | Incompleteness, bounded by the size of the missing row |
| **Violation** — luck booked on a denied play | **A false number.** 2.82 points of "kicker bad luck" credited to a team whose protection was beaten | The product asserts something untrue about a specific game, in the direction of the team that lost the play |

A floor applied to both resolves both in favour of whatever is currently shipped.
That is a status-quo rule, and a status-quo rule is defensible for additions —
the incumbent has been validated, the candidate has not — but for a violation the
status quo *is* the error.

### 3d. Size is still reported, and it still decides something

The amendment does not make size irrelevant. It makes size a **report** rather
than a **threshold**, on both populations, per document 18 §4b. Document 26's own
result shows why both numbers are needed and why neither is a verdict:

| | 287 games with a blocked kick | All 2,761 games with a kick |
|---|---|---|
| Median \|ΔDTW\| | 1.167 pp | **0.000 pp** |
| Median \|Δ deserved margin\| | **2.688 pts** | 0.019 pts |
| DTW side flips | 22 | 25 |

The all-games median of exactly zero is the honest defence of a correction that
disturbs almost nothing; the 2.688 points is the honest statement of what it
does where it fires. **the maintainer reads both and approves or declines.** Under the
amendment the approval step is where size does its work, which is where a
judgment about churn belongs — not inside a pre-registered gate that cannot see
the whole product.

### 3e. It is the same gate that was defended by rejecting A-2

Document 21 §9 rejected the measured-zero door and recorded the reason: *Gate A
is what supplies the empirical expectation that the arithmetic requires; a
powered zero grants permission without supplying a `p`.* That decision made Gate
A more load-bearing, not less.

**A component neutralizing a play with no branch has the same defect A-2 would
have created, arrived at from the other side.** A blocked kick has no `p`: the
number the component uses, `p_make`, is the probability that a kick *in flight*
goes through, and no kick was in flight. The ledger row is
`(0 − 0.86) × 3.9 EPA` — an expectation borrowed from a branch that never opened.
Rejecting A-2 and declining C-1 would leave the project holding both that a
component may not enter without a branch, and that a component already inside may
keep pricing plays that have none.

---

## 4. The case against

Stated as strongly as it can be, because §7 concedes that the author of this
document already knows what the first candidate measured.

1. **"Violation" is a judgment, and judgments expand.** Gate A is explicitly a
   judgment call applied in degrees — document 05 §2 grades a loose ball as "such
   a moment" and a kick in flight as "a weaker one". A procedure that suspends a
   threshold whenever somebody writes a convincing memo is a procedure that will
   eventually suspend it for a memo that is merely convincing. **This is the
   strongest objection**, and §5's clauses 1–3 exist to answer it: the candidate
   must already be inside a shipped population, the memo must name what actually
   resolved the play, and the correction must be the one Gate A implies rather
   than a parameter chosen for its size.
2. **Churn has a cost the gate cannot see.** v1.2's artifacts are validated,
   reproduced and cited across a dozen documents. Every re-ship risks a bug, and
   document 26 §8 found one waiting: four blocked field goals also carry a fumble
   row, and a correction implemented by filtering the play frame would silently
   delete them. A floor at least forces a candidate to be worth that risk.
   **Answer, partial:** the approval step and the ledger-sum gate carry this, not
   the materiality threshold — but the objection is real and is not fully
   answered.
3. **The circularity argument proves less than it appears to.** The 1.6250 pp
   floor is inflated by *every* luck row in those games, not only the violating
   one, and the violating row's share of it is unmeasured. Measuring it cleanly is
   awkward, because removing the row is the treatment — the counterfactual "the
   floor as it would be if the incumbent were right" requires already having
   applied the fix. A reader who wants the circularity quantified will not get a
   clean number, and §3b is therefore a structural argument rather than an
   arithmetic one.
4. **The timing is bad and no process can fully repair it.** Document 26 §9 said
   the new gate "should be written before anybody looks at this candidate again —
   otherwise it is goalpost-moving with a delay". The candidate has already been
   looked at. Writing the gate one round later, pre-registered, with the
   candidate's numbers published and unchanged, is the best available version of
   this — but the counterfactual is unanswerable: had document 26 cleared its
   floor at 1.8 pp as its screen predicted, **this document would probably not
   exist.** §7 discloses that in the form document 18 §5a used.
5. **Two regimes mean every future candidate must be classified**, and a
   classification is one more thing to argue about. **Answer:** clause 1 makes
   the classification mechanical — if the candidate books a row on a play that
   carries none today, it is an omission — and §6 pre-classifies every candidate
   the project has on record.

---

## 5. The concrete amendment, if it is adopted

**Proposed text, not enacted.** It would land in document 05 §2 as a third
subsection after Gates A and B.

> ### Gate C — correcting a Gate A violation inside a shipped component
>
> A **violation** is a play that a shipped component neutralizes and Gate A
> denies. A correction to a violation is governed by this gate instead of by the
> materiality floor. It is governed by **every other gate unchanged**.
>
> A candidate qualifies for Gate C **only if all four of the following hold**,
> each stated in a pre-registration committed before the correction is measured:
>
> 1. **It is a correction, not an addition.** Every play the candidate touches
>    already carries a ledger row from a shipped component, and the candidate
>    removes or re-prices rows without booking a row on any play that carries
>    none today. **A candidate that adds a single new row is an omission and the
>    materiality floor governs the whole of it**, including any corrective part.
> 2. **A violation memo.** It quotes the branch document 05 §2 admitted for the
>    component, shows that the population contains plays whose outcome is
>    resolved by something else, and **names what did resolve them** — a specific
>    football act by a specific side. "This play is not really a coin" is not a
>    memo.
> 3. **The memo argues the other side, and measures it where it is
>    measurable.** The strongest case that the play *is* a branch is stated, and
>    if it rests on a factual claim the data can settle, the claim is settled.
>    Document 25 §2 is the worked example: the objection that the blocker and the
>    recoverer are the same man was answered with 16 of 144.
> 4. **The correction is the one Gate A implies, not a free parameter.**
>    Exclusion where the play has no branch; re-pricing to the correct branch
>    where it has a different one. If more than one correction is Gate
>    A-compatible, the pre-registration argues the choice on mechanism and
>    measures both arms. **A correction whose form was chosen after seeing its
>    size does not qualify.**
>
> A qualifying candidate is then measured against:
>
> - **Identification**, unchanged — the violating population is identifiable from
>   charted fields, with the **rejected rows printed, not their count** (document
>   20 §9).
> - **The ledger-must-sum gate**, unchanged, including the exact row-count
>   arithmetic and a check that removed luck lands in `core`.
> - **The dial gate** (document 20 §5f), unchanged, wherever the correction
>   assumes a `w` the data cannot read.
> - **A materiality *report*, not a threshold.** Median and mean |ΔDTW|, median
>   |Δ deserved margin| and verdict flips, on **both** populations document 18
>   §4b requires: the games containing the violating play, and every game the
>   component touches. Neither number is a pass rule. **Both are printed in the
>   verdict, always.**
> - **A reconciliation.** The per-event luck removed, times the number of events,
>   read against the game-level movement — so that a correction whose size cannot
>   be explained by its own arithmetic is visible as such.
>
> **Verdict.** A candidate that qualifies under clauses 1–4 and clears
> identification, ledger-sum and the dial gate is **correct, and is proposed for
> adoption at whatever size it turns out to have**. Size never fails it. As with
> every other round in this project, adoption requires the maintainer's explicit approval,
> and the size report is what he approves against.
>
> A candidate that qualifies and is **not** adopted is recorded in the
> known-defect register as *a measured Gate A violation, knowingly retained*,
> with both population numbers attached.

### What is deliberately not in the amendment

- **No threshold of any kind.** Not a floor, not a ceiling, not a minimum event
  count. §7 explains why the absence is the mitigation for this document's own
  non-blindness.
- **No sunset clause.** Document 21's A-2 needed one because a measured zero is a
  statement about ten seasons. A mechanism argument is not a statistic and does
  not expire; if it is wrong it was always wrong.
- **No change to Gate A itself.** The branch-point question is untouched,
  word for word. C-1 is a procedure for acting on its answer.

---

## 6. What the amendment would admit, and what it would keep out

This is the section to read before deciding. Document 21 §6's format, and its
lesson: an amendment's real effect is often on candidates other than the one that
prompted it.

| Candidate | Under C-1 | Why |
|---|---|---|
| **Blocked kicks priced as misses** (document 26) | **Qualifies** | Every affected play already carries a field-goal or extra-point row; the correction removes 302 rows and adds none; the memo exists in document 26 §2 and names the defensive act; the alternative correction (a blocked class) is argued inadmissible on mechanism rather than on size |
| **The make-probability refit** (document 27) | **Not governed by C-1, and does not need it** | Document 05b never imposed a materiality floor on the make model. Its gates are calibration and resolvability gates and the refit is measured against them unchanged. Recorded here so no reader assumes the two proposals stand or fall together |
| **Kickoff muffs** (document 24) | **Excluded — omission** | 232 of 248 affected games *gain* a row that does not exist today. Clause 1 fails on the first sentence. **Document 24's verdict stands: measured, immaterial, not shipped** |
| **Blocked-kick aftermath** (document 25) | **Excluded — omission** | A new component on 415 plays that carry no aftermath row today. **Document 25's verdict stands** |
| **Onside kicks** (document 20) | **Excluded twice over** | An addition, and it failed the *dial* gate rather than materiality. C-1 leaves the dial gate untouched, so the verdict is unchanged either way |
| **Deflected interceptions** (documents 17, 22) | **Excluded** | An addition, and it failed on identification. C-1 leaves identification untouched |
| **Interception returns, red-zone and late-down sequencing** (document 21) | **Excluded, and this is the important row** | These fail Gate A *themselves*. C-1 can only remove or re-price plays a component already neutralizes; it has no inward direction. **A-2 would have admitted them; C-1 cannot.** Document 21 §9's decision is untouched |
| **Penalties** (document 05 §3) | **Excluded** | Not neutralized at all, so there is nothing to correct. The row Gate A exists to protect stays protected |
| **Whatever document 29's audit finds** | **Governed, and unknown at writing** | The audit was written after this text was fixed and its results are not known here. §7c states why that ordering is worth something |

**The row that would worry a careful reader is the last one**, and it is the
reason §5's clauses are written as tightly as they are. A gate whose future
caseload is unknown must be defensible without knowing it. The test to apply:
*for any candidate the audit could possibly turn up, does C-1 authorize anything
other than removing or re-pricing a row that a shipped component books today?*
It does not — clause 1 is a hard boundary, not a presumption.

---

## 7. Disclosure — what was already known when this was written

Document 18 §5a's format, because process honesty here requires naming what the
author could see.

### 7a. Known

- **Document 26's candidate is fully measured.** 1.167 pp against a 1.625 pp
  floor, 0 of 8 redraws clearing, median 2.688 points of deserved margin, 22
  verdict flips, all-games median exactly 0.000 pp. Every one of those numbers is
  published in document 26 §8 and is quoted in this document.
- **This amendment was prompted by that failure.** Document 26 §9 recommended
  writing it, and this is that round. The counterfactual is unanswerable and is
  stated in §4's fourth objection: had the candidate cleared its floor, this
  document would probably not exist.

### 7b. The mitigation, and why it is stronger than the usual one

**The gate contains no threshold.** Not one number in §5's text was chosen, and
none can be tuned. The usual mitigation for a non-blind pre-registration — that
the threshold came from a null simulation and could not have been moved by the
observed statistic (document 05b §10 §10's wording) — is unavailable here,
because there is no threshold at all. What replaces it is a stronger property:

> **No value of document 26's statistic changes a single word of §5.** At 1.167
> pp, at 1.8 pp, or at 12 pp, the amendment text is identical, because size is
> not an input to it. A gate that cannot read the statistic cannot be fitted to
> it.

The second mitigation is the argument's provenance. §2 and §3 are derived from
document 05 §2's own sentence, from the additions-versus-corrections distinction
document 24 §9 and document 25 §8 already drew for other reasons, and from the
structure of the two failure modes. **None of them uses a number from document 26
§8**; §3d and §7a quote those numbers as illustration, and the argument survives
their deletion.

### 7c. What the ordering buys

This text was fixed before `research/43_gate_a_audit.py` existed and before
anybody knew what other violations the ledger contains. If the audit finds one,
it will be judged by a gate written without knowing it was there; if the audit
finds nothing, this amendment governs exactly one known candidate and its future
scope is whatever later rounds turn up. **Neither outcome could have shaped the
text**, and that is the one form of blindness genuinely available to a document
written after its first test case was measured.

---

## 8. Recommendation, and what would change it

**Recommended: adopt C-1 as written in §5.**

Not because document 26's candidate deserves to ship — that is a separate
decision, on separate evidence, and §6 does not prejudge it — but because the
rule as it stands produces a sentence the project should not be willing to
write: *the ledger will keep booking 2.82 points of luck on a play nobody
flipped a coin on, because removing it does not move the median game far
enough.*

**What would change the recommendation:**

- **A worked case where C-1 authorizes something unwanted.** §6 could not
  construct one, but §6 is written by the same person as §5. A candidate that
  qualifies under clauses 1–4 and is obviously wrong to ship would be decisive
  against.
- **A defensible way to de-circularize the floor.** If the incumbent's interval
  on the affected games could be computed with the violating row's contribution
  removed but the correction not applied, the floor would become an honest bar
  for corrections too, and C-1 would be unnecessary. §4's third objection says
  why that is awkward, not that it is impossible.
- **Evidence that churn costs more than the arithmetic suggests.** If a v1.3
  rebuild broke something a v1.2 artifact depended on, objection 2 would carry
  more weight than §3 gives it.

**What happens if it is rejected:** document 26's candidate is closed
permanently, the field-goal and extra-point components keep pricing blocked
kicks as misses, and the known-defect register carries the entry with both
population numbers attached. Document 29's audit still runs and still reports —
**an audit's findings are worth having whether or not anything may be done about
them** — and any violation it finds joins the same register under the same rule.

---

## 9. Status

**Not enacted.** No code, no threshold and no treatment-table row changes on the
strength of this document.

- **Accept C-1** → document 05 §2 gains §5's text as a third subsection, document
  05 §5's register gains a line pointing at it, and document 26's candidate
  becomes measurable under it — on the refit arithmetic of document 27, and only
  if that refit is also approved, because document 26 §9 recorded that its
  numbers are not stable under it.
- **Reject C-1** → this document stands as the record of why, document 26's
  verdict is final, and the argument does not need to be re-derived the next time
  a violation is found.
- **Accept C-1 narrowed** → the narrowing needs writing before it binds. The two
  obvious narrowings are a minimum event count (which reintroduces a threshold,
  and §5 says why there is none) and a clause requiring the violating component
  to have been shipped for at least one version (which would have no effect on
  any candidate on record).

## 10. Decision

**Status: accepted. Decided 2026-08-18 by the maintainer**, on the recommendation of §8.

**Amendment C-1 is adopted, unnarrowed.** §9's acceptance actions are enacted in
the same commit as this line:

- **Document 05 §2 gains §5's text as a third subsection**, *Gate C — correcting
  a Gate A violation inside a shipped component*, verbatim. The section heading
  changes from "the two gates" to "the gates".
- **Document 05 §5's register gains a line** recording that a shipped component
  can neutralize a play Gate A denies, and that Gate C — not the materiality
  floor — governs the correction.
- **Document 26's candidate becomes measurable under it.** Per §9, that
  re-measurement runs on document 27's refit arithmetic, which the maintainer approved on
  the same day, because document 26 §9 recorded that the candidate's numbers are
  not stable under the refit. It is task 3 of Phase 8 and lands in document 30.

What this settles, so it is not re-derived:

- **Size never fails a correction.** A candidate that qualifies under clauses
  1–4 and clears identification, ledger-sum and the dial gate is proposed for
  adoption at whatever size it turns out to have. The materiality report is
  mandatory, is printed on both populations, and is what the maintainer approves against.
- **The materiality floor is untouched for additions.** A candidate that books a
  row on any play carrying none today is an omission, and the floor governs the
  whole of it — clause 1, unchanged.
- **Gate A itself is unamended.** C-1 is a procedure for acting on Gate A's
  answer, not a change to the branch-point question. That is the second
  consecutive amendment round in which Gate A survived intact; document 21's A-2
  was rejected outright.

