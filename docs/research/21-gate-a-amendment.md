# 21 — A proposed amendment to Gate A, for the maintainer to accept or reject

*Written 2026-08-18. **This document enacts nothing.** It is a proposal with
both sides argued, a concrete amendment text, and an explicit list of what the
amendment would let into the ledger and what it would keep out. No code changes,
no fits, no thresholds. Document 05 §2 stands unchanged until the maintainer says
otherwise.*

*Inputs: documents 05 (Gate A and the penalty row that justifies it), 08 (the
sequencing zeros), 13 (the attribution failure), 14 (return skill, and the null
it left intact), 15 §C5 (the power table that prompted this).*

---

## 1. The question, and the recommendation stated first

### The question

Document 05 §2 admits a component to the ledger only if it passes Gate A:

> **Is there a moment where the outcome is resolved by a mechanism outside either
> team's control, conditional on the state both teams created?**

Interception returns fail that gate — a return is a played-out sequence with
tackling and blocking, not a branch resolved by nobody. But document 15 measured
something new: the *skill* dial on interception returns is a **powered zero**.
Split-half persistence is r = −0.014, and the instrument, simulated at the real
per-team sample sizes, would show r ≈ +0.24 if a mere 5% of the variance were
team skill.

So: **may a component with a powered measured-zero skill dial be neutralized
even though it has no clean coin moment?**

### The recommendation

> **Reject the amendment as stated, and reject it for a reason that is not about
> the evidence.** The evidence is good. The problem is that Gate A does not only
> grant *permission* to neutralize — it supplies the *expectation* that
> neutralizing requires. Every ledger row in this project reads
> `luck = (realized − expected) × swing`, and `expected` is always a branch
> probability measured from an empirical bin: p(recover), p(make). A component
> with no branch has no `p`. Its expectation would have to come from a fitted
> model of the very quantity being neutralized, and at that point the ledger
> stops being an accounting identity and becomes a **prediction error** — with
> the model's own errors booked as luck.

If the maintainer wants interception returns in the ledger anyway, §5 gives the narrowest
amendment that would do it honestly, and §6 shows what else walks through the
same door. The single most important fact in this document is in §6: **the
amendment as written would admit red-zone and late-down sequencing**, which
document 08 §6 deliberately kept out.

### Four things to hold onto

1. **The evidence really did change.** Document 05 §7 dismissed its own
   return-yardage null as underpowered; document 15's instrument says the
   opposite at the scale that matters. Anyone re-reading document 05's
   "on a test that could only have shown a large effect" should know it has been
   superseded.
2. **A zero between teams is not the same as no control.** It says teams do not
   *stably differ*, which is compatible with the play being decided entirely by
   skill exercised in the moment. Document 14 made exactly this point in the
   other direction: kickoff returns persist at +0.35 and interception returns do
   not, because one is a scheduled play with a wall and a scheme and the other is
   a defensive back running with a ball he did not expect to have.
3. **Gate A's own justification is symmetric, and it cuts against the
   amendment.** Document 05 §2 keeps penalties out because their fitted dial —
   `w ≈ 0.42` — would have neutralized half of every team's penalty EPA, and the
   number was correct and the conclusion would have been wrong. The lesson
   recorded there is that *persistence statistics cannot detect the presence or
   absence of a branch point.* A powered zero is still a persistence statistic.
4. **The stakes are small and that matters for the decision.** 4,304
   interception returns in ten seasons, mean 12.7 return yards, roughly 25
   return yards in a game that has one. This is not a change worth loosening a
   load-bearing gate for. If the amendment is worth making, it is worth making
   for a bigger candidate.

---

## 2. What is actually on the table

| | Value | Source |
|---|---|---|
| Interception-return split-half persistence | **r = −0.014**, 5th–95th pct [−0.088, +0.068] | document 05 §3c, 314 defense-seasons |
| What a 5% skill share would show | r = +0.238 [+0.140, +0.332] | document 15 §C5 |
| What a 10% skill share would show | r = +0.400 [+0.308, +0.476] | document 15 §C5 |
| What a true zero shows | r = −0.002 [−0.105, +0.101] | document 15 §C5 |
| Per-team sample size | median 13 returns per team-season, SD 17.6 yards | document 15 §C5 |
| Total events | 4,304 returns, mean 12.7 yards | document 15 §C5 |

The observed −0.014 sits inside the zero band and **below the 5% band's 5th
percentile**. On this instrument, at this sample size, a skill share of one
twentieth would very likely have been visible and was not.

**Two power statements about the same quantity disagree, and the disagreement is
resolvable.** Document 05 §7 said the return-yardage test could only have
detected a 45%-relative effect; document 15 says 5% of variance would have
shown. They are not measuring the same thing — §7's figure is a relative
population SD on a *rate*, §15's is a variance share of *yards* — and §15's is
the one that speaks to the question being asked here. Neither is wrong; the
older sentence is answering a question nobody is now asking.

---

## 3. The case for the amendment

1. **The project's stated purpose is to strip out what teams do not control.**
   If ten seasons say no team controls interception return yardage in any stable
   way, leaving it in `core` credits it to the defense that happened to get it.
2. **Full neutralization already rests on a measured near-zero dial elsewhere.**
   Fumble recovery is neutralized in full because `w = 0.015` (document 18 §8).
   The arithmetic justification is identical; only the mechanism story differs.
3. **Gate A is a judgment call applied in degrees, not a bright line.** Document
   05 §2 itself grades the branches: a loose ball is "such a moment", a kick in
   flight is "a weaker one", and both are admitted. A gate that already admits
   weaker cases is not obviously entitled to reject a case with better evidence.
4. **A powered zero is genuinely rare in this project.** Document 08 called its
   sequencing zeros "evidence of absence, not absence of evidence — the first
   time in the project that distinction has been available." Refusing to use
   information that expensive to obtain has a cost too.
5. **The amendment does not reopen the row Gate A was built to protect.**
   Penalties measure `w ≈ 0.42`, nowhere near zero, so no version of a
   measured-zero door lets them in. The worked example that justifies Gate A is
   untouched by the amendment. This is the strongest argument for.

---

## 4. The case against

1. **Gate A supplies the expectation, not just the permission.** This is the
   decisive objection and it is structural rather than statistical. Neutralizing
   needs a number to put in `expected`, and in every shipped component that
   number is a branch probability from an empirical bin. Interception return
   yardage has no branch and therefore no `p`; you would need
   `E[return yards | field position, defenders, blockers]`, which is a model.
   `components.py`'s own docstring says why that was avoided: *"the
   classification of skill vs luck should not depend on a model whose own errors
   could manufacture the answer."*
2. **"No stable team difference" and "outside anyone's control" are different
   claims.** A cornerback's cut is skill in the moment. It does not persist as a
   *team-season* property because the same cornerback does not get 13 identical
   opportunities. Document 14's contrast between kickoff returns (+0.35) and
   interception returns (−0.014) is the clean demonstration that persistence is
   measuring schedule and scheme, not agency.
3. **Document 05 §2's warning applies with the sign flipped.** It records that a
   correct persistence number pointed the wrong way once already. A powered zero
   is a better persistence number, not a different kind of evidence.
4. **There is no charged entity.** Every ledger row names a `charged_team`.
   Document 13 spent a whole round failing to attribute the interception spread
   to an entity — the design could not tell a quarterback from a head coach on
   5,522 team-games. A component whose defining claim is "nobody controls this"
   has to name somebody in the ledger anyway, and the honest answer is arbitrary.
5. **The gate would become a two-key gate, and the second key is easier to
   turn.** Gate A currently requires a mechanism argument that a person has to
   make in writing and defend. A measured-zero door can be opened by any
   component that happens to have a flat statistic and enough events. Over time
   that is the door that gets used.

---

## 5. The concrete amendment, if it is adopted

The narrowest version that would admit interception returns without admitting
everything. **Proposed text, not enacted:**

> ### Amendment A-2 — the measured-zero door
>
> A component that fails Gate A's branch-point test may nonetheless be
> neutralized if and only if **all six** of the following hold, each
> pre-registered before the component's fit:
>
> 1. **A powered zero.** The instrument, simulated at the component's own
>    observed per-entity sample sizes, detects a 5%-of-variance skill share with
>    probability ≥ 0.80, and the observed statistic falls inside the simulated
>    zero band.
> 2. **The zero holds at two grains.** It reproduces at both the team and the
>    individual-player grain, so that "no team differs" is not concealing "one
>    player differs".
> 3. **An expectation from pre-branch state only.** `expected` comes from an
>    empirical bin table keyed on state that exists *before* the event, never
>    from a fitted model of the outcome being neutralized. If no such table can
>    be built, the component does not qualify — full stop.
> 4. **A named charged entity**, justified in the same memo, and not chosen
>    because it is the only column available.
> 5. **The materiality floor**, unchanged: median |ΔDTW| on the affected games
>    at or above the incumbent's median 89% DTW half-width there.
> 6. **A sunset.** The component is re-tested every season against clauses 1–2
>    and reverts to `none` the first time the zero moves outside its band.
>
> A component admitted under A-2 is recorded in document 05 §3's treatment table
> with the treatment **`full (A-2)`**, so that no reader mistakes it for a
> component with a branch point.

Clause 3 is the load-bearing one and is written to make §4's first objection
binding rather than arguable. Clause 6 exists because a measured zero is a
statement about ten seasons, not a law.

---

## 6. What the amendment would admit, and what it would keep out

This is the section to read before deciding.

| Candidate | Under A-2 | Why |
|---|---|---|
| **Interception return yardage** | **Admitted only if clause 3 can be met** | The zero is powered (§2). Whether an empirical expected-return-yards bin table keyed on pre-return state exists has never been attempted; document 05 §3c flagged that requirement and stopped |
| **Red-zone finishing (document 08's S1)** | **Admitted** | r = −0.034 against a 0.0703 threshold, with 87% power at the reference. Document 08 §6 deliberately kept it out of the ledger and reports it separately |
| **Late-down conversion over baseline (S2)** | **Admitted** | r = +0.0005, 92% power. "It is difficult to construct a cleaner null result" — document 08 §9 |
| **Penalties** | **Excluded** | `w ≈ 0.42`. Not a zero, and not close to one. The row Gate A exists to protect stays protected |
| **Kickoff and punt return yardage** | **Excluded** | r = +0.35 and +0.16 (document 14). Measured skill, and large |
| **Deflected interceptions** | **Excluded** | Fails on identification, not on mechanism (document 17). A-2 does not touch identification |
| **Onside kicks** | **Unaffected** | Already passes Gate A on mechanism (documents 09, 20) |

**The sequencing rows are the real content of this amendment.** Interception
returns are worth about 25 yards in a game that has one. Red-zone and late-down
placement are worth substantially more and were excluded by an explicit
judgment, not by a lack of evidence — document 08 §6 kept them out because *where
production lands is not a branch anybody resolves*. A-2 as written overrules that
judgment as a side effect. If the maintainer wants A-2 for interception returns and not
for sequencing, the amendment needs a seventh clause naming the distinction, and
**no defensible version of that clause is obvious** — which is itself an argument
that the distinction being relied on is the branch point after all.

---

## 7. Recommendation, and what would change it

**Recommended: reject A-2.** Not because the interception-return zero is weak —
it is the best zero in the project — but because clause 3 is probably
unsatisfiable for it, and because the amendment's real effect is on the
sequencing rows rather than on the candidate that prompted it.

**What would change the recommendation:**

- **A working expectation table.** If an empirical bin table of expected
  interception-return yards keyed on pre-return state (field position, down
  distance, time) can be built and shown to be a description rather than a
  model, objection 1 falls and A-2 becomes worth having.
- **A seventh clause that distinguishes returns from sequencing** on some ground
  other than the branch point. If one exists, it is a better gate than Gate A
  and should replace it rather than amend it.
- **A larger candidate arriving with the same shape.** The cost-benefit here is
  driven by 25 yards a game. A powered zero on something worth a point a game
  would deserve a fresh look at the same argument.

**What happens if it is rejected:** nothing changes. Interception returns stay in
`core`, document 05 §3's treatment table is untouched, and the powered zero is
recorded in document 15 and here as a measured fact about the game rather than a
ledger row. The next round should note that this is now the **fifth consecutive
round in which the mechanism gate, not the arithmetic, settled the question**.

---

## 8. Status

**Not enacted.** No code, no threshold and no treatment-table row changes on the
strength of this document. It is a decision request:

- **Accept A-2** → document 05 §2 gains the amendment text of §5, and the
  sequencing rows of §6 must be explicitly ruled on in the same breath.
- **Reject A-2** → this document stands as the record of why, and the argument
  does not need to be re-derived the next time a powered zero appears.
- **Accept a narrowed A-2** → the seventh clause of §6 has to be written first,
  and it needs its own memo.
