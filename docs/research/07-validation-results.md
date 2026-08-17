# 07 — Rematch validation: results

*Script: `research/08_rematch.py`. Design, statistic and thresholds fixed by
`docs/research/06-rematch-validation.md`, committed at `70d6254` **before the
simulator produced a single deserve-to-win number**. Results in
`research/outputs/08_rematch.json`.*

---

## Gate outcomes, stated first

| Gate | Rule | Result |
|---|---|---|
| **3 — coefficient sanity** | `b1 > 0` in both arms, 95% CI excluding zero | **PASS** |
| **1 — non-inferiority** | 95% CI upper bound below +0.010 log loss | **PASS** — upper bound +0.0038 |
| **2 — direction** | reported, no pass rule | Favours deserve-to-win; interval contains zero, as pre-registered |

531 rematch pairs. Nothing below was chosen after seeing a number.

---

## Gate 3 — the harness works

| Arm | `b1` | 95% interval |
|---|---|---|
| Game-1 actual margin | +0.452 | +0.266 – +0.638 |
| Game-1 deserved margin | **+0.467** | +0.280 – +0.654 |

Both positive, both excluding zero: a team that won game 1 by more is more
likely to win game 2, so the pairing, the orientation and the sign conventions
are intact. This gate exists to catch an implementation bug, and it does not
fire.

Worth noting descriptively, since Gate 2's reporting rule permits it: the
deserved margin's coefficient is *slightly larger* than the actual margin's.
Neutralized game 1 carries marginally more signal about game 2 than the
scoreboard does. The difference is far inside the noise and is not a finding.

---

## Gate 1 — non-inferiority passes

> **Mean paired Δ log loss = −0.00159, SE 0.00273, 95% CI [−0.00695, +0.00377].**
> The upper bound sits below the pre-registered +0.010 margin. **PASS.**

Neutralizing luck does not degrade the information content of a game by any
amount this design would call meaningful. That is the claim the project actually
makes, and it is now tested rather than assumed.

Recall the burden of proof runs the opposite way to a superiority test: a wide,
uninformative interval would have **failed** this gate. It did not — the
interval is tight enough to rule out a degradation a quarter the size of the
Vegas-versus-raw-EPA gap Phase 1 measured.

### The design behaved exactly as simulated

This is the part that makes the result trustworthy, and it is only checkable
because the power calculation ran first:

| Quantity | Predicted by document 06 §3 | Observed |
|---|---|---|
| Mean Δ log loss at the realistic scenario | −0.00169 | **−0.00159** |
| SE of the statistic | 0.00269 | **0.00273** |

The pre-registered simulation assumed the neutralization removes about 6.4% of
margin variance — fumble luck plus field-goal luck, from document 01. The
observed statistic and its standard error land within a few percent of what that
assumption implied. The simulator is removing roughly the variance it was
designed to remove, and the estimator's noise is what the power calculation said
it would be.

---

## Gate 2 — direction, reported and not over-read

The point estimate favours deserve-to-win. The 95% interval contains zero.

**Document 06 §4 pre-registered that this would almost certainly happen and that
it must not be reported as evidence that neutralization does not work.** The
design's power against the realistic alternative is **7.2%**; the minimum
detectable effect is 52% of margin variance against the at most 25.1% available.
An interval containing zero is a property of 531 rematch pairs, not a property
of the simulator.

What can honestly be said: under the alternative, the point estimate favours
deserve-to-win 74% of the time, against 47% under the null. Observing a
favourable sign is therefore **weak corroboration**, and that is the whole of it.

---

## Secondary and exploratory arms

**Secondary — predict the game-2 margin** (pre-registered in document 06 §3):

> Mean Δ MSE = −0.321, SE 1.336, 95% CI [−2.940, +2.298].

Same direction, same inability to resolve it. The power table put this arm at
12.9%, roughly double the primary and still far from enough.

**Exploratory — DTW% as the predictor**, explicitly *not* pre-registered and
recorded as exploratory so it cannot be read as confirmatory later:

> Mean Δ log loss = −0.00271, SE 0.00576, 95% CI [−0.01399, +0.00857]. Passes
> non-inferiority.

DTW% is a probability compressed toward 0 and 1, so it discards margin
information the deserved margin keeps; its larger standard error reflects that.
It is reported for completeness, not as a better arm.

---

## What this validation establishes, and what it does not

Restating document 06 §5 against the numbers now in hand.

**Established.** Neutralizing luck per document 05's treatment table does not
meaningfully degrade a game's predictive content, on a test with a 0.008
false-alarm rate and 88.9% power against a neutralization that is 15% noise. The
harness is sound (Gate 3). The estimator behaves as its power calculation
predicted, in both centre and spread.

**Not established.**

- **That deserve-to-win forecasts better.** The design cannot show it, Phase 1
  (document 02) already found luck-stripping does not improve out-of-sample
  prediction, and nothing here contradicts that. The simulator remains a
  retrospective adjudication tool.
- **That the treatment table is optimal.** A pass is consistent with several
  different treatment tables; this design cannot separate them.
- **That the DTW credible interval has correct coverage.** Gate 1 scores a point
  estimate. Coverage is a separate question and is not answered here.
- **Generalization beyond division rematches.** 95.5% of these pairs flip the
  host, which is the division home-and-away pattern. Division games are
  systematically closer than average.

---

## Defect register

| Defect | Evidence | Status |
|---|---|---|
| Superiority is untestable at this sample size | MDE 52% of margin variance vs 25.1% available | **Accepted**, pre-registered in document 06 §3 |
| Rematch pairs skew to division games | 95.5% flip the host | **Open.** Limits generalization |
| Game 1 is inside the FG model's fitted sample | Document 05b §7's contamination defect | **Open, bounded** at O(1/n) |
| The secondary arm was run despite the primary passing | Both were pre-registered together | **Closed** — pre-registered as a pair, not selected after the fact |
| The exploratory DTW% arm is not pre-registered | Added after seeing the primary | **Open, labelled.** Reported as exploratory and excluded from every gate |

---

## Constants

| Constant | Value |
|---|---|
| Rematch pairs | 531 |
| CV folds | 10 |
| `RANDOM_SEED` | 20260817 |
| Non-inferiority margin | +0.010 log loss |
| Observed Δ log loss | −0.00159 (SE 0.00273) |
| Gate 1 verdict | **PASS** |
