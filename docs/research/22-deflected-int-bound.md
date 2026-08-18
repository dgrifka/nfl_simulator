# 22 — Deflected interceptions: the swat-down denominator as a bound

*Written 2026-08-18, **before `research/34_deflected_bound.py` existed**. This
is a pre-registered **sensitivity analysis with one escape hatch**, not a
component proposal. Counts in §2 were computed during design; the impact in §5
was not. Committed to git before any impact number exists.*

*Inputs: document 17 (which closed this candidate), 18 §5 (the materiality
floor), 20 (the sensitivity-arm-as-gate precedent), 19 (simulator v1.2).*

---

## 1. The question

Document 17 killed deflected interceptions on identification, not mechanism.
Gate A passed — a tipped ball tumbling with nobody's hands on it is the fumble
argument three feet higher — and the candidate died because the neutralization
identity needs `p(intercepted | deflected)`, and **the denominator is
invisible**: a deflection that falls harmlessly to the turf is recorded as a pass
defensed, indistinguishable from a cornerback swatting a ball down cleanly.

the maintainer's proposal is to stop treating the denominator as unavailable and start
treating it as **bounded**. Pass-defensed credits on non-interception passes are
observable: 18,777 of them, 2016–2025. If a fraction `f` of those were live tips
rather than deliberate swat-downs, then

    p(f) = 629 / (629 + f · 18,777)

and `f` is unidentified. The question this document asks is not *what is f* — it
cannot be answered — but:

> **Does the answer to "would this component matter?" depend on `f` at all?**

If the ledger impact is materially the same whether one in twenty of those
18,777 was a live tip or all of them, then the unidentified quantity is a
nuisance rather than an obstacle, and document 17's verdict should be revisited.
If not, the bound is too wide to act on and the candidate closes for a second
and better-stated reason.

**Expected outcome, written down in advance so it cannot be claimed afterwards:
the bound will be too wide, and the component will close.** §5c says exactly what
would have to be true for that expectation to be wrong.

---

## 2. The arithmetic, and the two arms

### The numbers (design, computed before any impact)

| Quantity | Value |
|---|---|
| Interceptions 2016–2025 | 4,304 |
| Deflected (`pass_defense_1_player_id != interception_player_id`) | **629** |
| Pass-defensed credits on non-interception passes | **18,777** (every one incomplete) |
| Naive `p` at `f = 1` | **3.24%** |
| `p` at `f = 0.05` | **40.1%** |
| Games carrying a deflected interception | **566** |
| Mean EPA, deflected interception (offense) | −4.712 |
| Mean EPA, pass-defensed incompletion (offense) | −0.891 |
| **Branch swing, defense's perspective** | **3.821 EPA** |

### Two arms, because only one of them is buildable

| Arm | Rows booked | Buildable? |
|---|---|---|
| **A — successes only** | the 629 deflected interceptions, `realized = 1`, `expected = p(f)` | **Yes.** This is the only implementation the data permits |
| **B — both branches** | the 629, plus `f · 18,777` pass-defensed incompletions at `realized = 0` | **No.** Which of the 18,777 are live tips is exactly the unidentified fact; the arm is simulated by drawing a random subset, repeated, to show the spread |

**Arm A is not the component.** A coin that only appears in the ledger when it
lands heads books one-sided luck by construction: every deflected interception
becomes good fortune for the defense and nothing ever offsets it. Arm B is what
document 05 §1's identity actually requires. Both are reported because the
difference between them is the point.

---

## 3. Method

Simulator v1.2 on the games each arm touches, with fumble, field-goal and
extra-point draws generated from their own seeded generators in both the with-
and without-component arms, so the difference is the deflection rows and nothing
else. Charged team is the intercepting defense; `swing` is the branch gap above,
signed to home perspective. `f` is swept over **{0.05, 0.10, 0.25, 0.50, 1.00}**.

Arm B's random subset is drawn **5 times per value of `f`** at fixed seeds, and
the spread across draws is reported beside the spread across `f`, so a reader can
see which of the two dominates.

---

## 4. Committed statistics and the floor

- **Impact statistic:** median |ΔDTW| across the games the arm touches.
- **Materiality floor:** the v1.2 median 89% DTW interval half-width on those
  same games — document 18 §5d's floor, unchanged.
- **Secondary, reported not gated:** mean and max |ΔDTW|, DTW side flips, median
  |Δ deserved margin|.

---

## 5. The escape hatch, pre-registered

### 5a. Gate D-1 — is the impact invariant in `f`?

**Statistic:** for each arm, the median |ΔDTW| at each of the five values of `f`.

**Pass rule — both conditions:**

1. **Same verdict everywhere.** The pass/fail against the materiality floor is
   identical at all five values of `f`.
2. **Narrow bound.** `max / min` of the median |ΔDTW| across the five values is
   **≤ 1.25**.

**Pass** → the identification failure is moot: whatever `f` is, the component
would matter (or would not) to the same degree. Document 17's verdict is
reopened and the candidate returns as a full pre-registration with `p` fixed at
the most conservative end of the range.

**Fail** → the candidate closes as **"bound too wide"**, the punt-bounce verdict
of document 14, with the arithmetic attached. No component, no ledger row, and
document 17's closure stands with a better-stated reason.

### 5b. Which arm decides

**Arm B decides.** Arm A is reported first because it is the only buildable
implementation, but a one-sided ledger is not the component, and passing D-1 on
Arm A alone would mean the identification problem had been avoided rather than
answered. If the two arms disagree, that disagreement is the finding.

### 5c. What would make the pre-registered expectation wrong

The expectation of failure rests on Arm B's row count scaling directly with `f`:
939 rows at `f = 0.05` against 19,406 at `f = 1.00`, a factor of twenty. The
expectation would be wrong if the per-event luck shrank at exactly the rate the
row count grew — `(0 − p(f))` falls as `f` rises — leaving the game-level
median unchanged. That is arithmetically possible and is the reason the gate is
worth running rather than asserting.

Arm A is expected to vary far less: its per-event luck is `(1 − p(f))`, which
runs only from 0.60 to 0.97 across the whole range of `f`, a factor of 1.6. **If
Arm A passes D-1 and Arm B fails it, the correct reading is that Arm A's
stability is an artefact of booking one branch**, not evidence that the
denominator does not matter — and §5b's precedence exists so that reading is
committed in advance.

---

## 6. Known defects, carried from document 17

| Defect | Status |
|---|---|
| A deliberate swat-down and a live tip are the same record | **Open, and the reason this document exists** |
| `pass_defense_1_player_id` is credited by a human scorer | **Open.** A deflection with no PD credit is invisible to both numerator and denominator |
| The persistence question is unanswerable at two deflected interceptions per team-season | **Open** (document 17 §1) |
| Deflected interceptions the receiver tipped are counted as defensive deflections | **Open.** A tipped-by-receiver interception has no PD credit to the deflector and is missed |

---

## 7. Constants appendix

| Constant | Value |
|---|---|
| `RANDOM_SEED` | 20260818 |
| Deflected interceptions | 629 |
| Pass-defensed non-interception passes | 18,777 |
| Branch swing | 3.821 EPA |
| `f` sweep | 0.05, 0.10, 0.25, 0.50, 1.00 |
| Arm B draws per `f` | 5 |
| Gate D-1 invariance bound | max/min ≤ 1.25 |
| `points_per_epa` | 0.8389 |

Results are written back into this document as §8.

---

## 8. Results

*Script: `research/34_deflected_bound.py`. The gate was committed at `6b00e4f`
before this script existed. Results in
`research/outputs/34_deflected_bound.json`.*

### The verdict, stated first

> **The bound is too wide. Sweeping the unidentified live-tip share across its
> full range moves the component's impact by a factor of four and flips its
> verdict against the materiality floor, so the identification failure is not a
> nuisance — it is the whole answer. The candidate stays closed, and document
> 17's verdict stands with a better-stated reason.**

| Arm | max/min across `f` | Same verdict at every `f`? | Gate D-1 |
|---|---|---|---|
| **A — successes only** (buildable, not the component) | **1.18×** | Yes, above the floor everywhere | **Pass** |
| **B — both branches** (the component the identity requires) | **4.07×** | **No** — above at `f ≤ 0.10`, below at `f ≥ 0.25` | **FAIL** |

Per §5b, **Arm B decides.** Per §5a's decision rule, a D-1 failure closes the
candidate as *bound too wide* — the punt-bounce verdict of document 14.

### The sweep

`p(f) = 629 / (629 + f · 18,777)`, branch swing 3.820 EPA from the defense's
perspective.

| `f` | `p` | Arm A: median \|ΔDTW\| | Arm B: rows | Arm B: games | Arm B: median \|ΔDTW\| | vs floor |
|---|---|---|---|---|---|---|
| 0.05 | 40.1% | 1.83 pp | 1,568 | 1,202 | **1.30 pp** [1.05–1.59] | above (0.78) |
| 0.10 | 25.1% | 2.15 pp | 2,507 | 1,632 | **0.96 pp** [0.81–1.15] | above (0.62) |
| 0.25 | 11.8% | 2.02 pp | 5,323 | 2,364 | **0.53 pp** [0.48–0.55] | **below** (0.69) |
| 0.50 | 6.3% | 1.97 pp | 10,017 | 2,676 | **0.35 pp** [0.33–0.37] | **below** (0.63) |
| 1.00 | 3.2% | 2.01 pp | 19,406 | 2,756 | **0.32 pp** [0.32–0.32] | **below** (0.63) |

Bracketed ranges are across the five random draws of which incompletions were
live tips. **The spread across draws is far smaller than the spread across
`f`** — at every value of `f`, the draws agree with each other to within about
0.2 pp while the values of `f` disagree by 1 pp. The uncertainty that matters is
the one that cannot be sampled away.

### Arm A's pass is the artefact §5c named in advance

Arm A varies by only 1.18× and clears the floor everywhere, which — read alone —
would have looked like a clean invariance result and reopened the candidate.
**§5b committed to Arm B deciding, and §5c wrote down why Arm A would look
stable, before either number existed.** The reason is now visible in the table:

- Arm A's per-event luck is `(1 − p)`, which runs only from 0.60 to 0.97 across
  the whole range of `f`. Its row count never changes — 629 interceptions,
  always.
- Arm B's row count runs from 1,568 to 19,406, a factor of twelve, and its
  per-event luck on the added rows is `(0 − p)`, which *falls* as `f` rises.
  Those two effects do not cancel; the median game gets steadily smaller
  adjustments as the coin is priced closer to a certainty.

Arm A is also one-sided by construction: every deflected interception becomes
good fortune for the defense and nothing ever offsets it, which is why it flips
41–64 games — more than the fumble widening — while booking a coin that only
appears in the ledger when it lands heads.

### What this closure is worth

Document 17 closed this candidate with "the denominator is invisible." That was
true and it was an assertion. This round turns it into arithmetic: **the range
the denominator could take moves the answer across the decision boundary**, and
the least generous assumption available (`f = 1`, every pass defensed was a live
tip) puts the component at half its own floor.

**What would reopen it:** anything that separates a swat-down from a live tip.
Charting that marked "ball deflected in the air" — the FTN feed does not carry
it, and document 17 §1 established that `is_interception_worthy` is answering a
different question — or player tracking. Neither is in this project's data
inventory, and both are named in `CLAUDE.md` under *Not available — do not go
looking*.

### One lesson, carried forward

**A sensitivity sweep needs its deciding arm named before the numbers arrive.**
The buildable implementation passed and the correct one failed, by a factor of
four, and the only thing that prevented the wrong reading was §5b having been
written first. Document 20's round produced the mirror-image lesson — a
sensitivity arm allowed to be the binding gate — and the two together are now
the round's clearest process result.
