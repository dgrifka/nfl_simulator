# 06 — Rematch validation, pre-registered

*Written 2026-08-17, **before the simulator produced a single deserve-to-win
number**. Power calculation: `research/08_rematch_power.py`, results in
`research/outputs/08_rematch_power.json`. Committed to git before document 07
exists, so goalpost integrity is checkable by commit archaeology.*

---

## 1. The question

Among same-season rematch pairs, does game 1's **deserve-to-win** predict who
wins game 2 better than game 1's **actual result** does?

The design is the cleanest natural experiment the sport offers. Two teams play,
and then — with roster, coaching and scheme almost unchanged — they play again.
If neutralizing luck recovers a truer picture of which team is better, the
neutralized game 1 should forecast game 2 better than the scoreboard did.

Both predictors enter **identical** logistic models, so nothing about model form
can decide the comparison:

```
logit P(team A wins game 2) = b0 + b1 · z(game-1 predictor) + b2 · A_hosts_game_2
```

Team A is game 1's host. The predictor is standardized in each arm, so the
comparison cannot be won on units. The home-field indicator is there because
rematches usually flip the host — 95.5% of pairs in this sample — so leaving it
out would let home-field masquerade as predictive skill.

---

## 2. Data

- **Grain of a row**: one rematch pair — the first two same-season meetings of
  two teams.
- **Source**: `research/outputs/game_components.parquet`, built from cached
  play-by-play 2016–2025.
- **Population**: 531 pairs. 514 teams met exactly twice; 17 met three times
  (division rivals who then met in the playoffs) and contribute **only their
  first two meetings**, because a second overlapping pair would share a game
  with its neighbour and would not be independent.
- **Exclusion**: pairs whose game 2 was a tie, which carries no winner to predict.
- **Orientation**: fixed by game 1's host, so game 2's home-perspective margin is
  flipped whenever A is the road team in the rematch.

### Measured design parameters

Estimated from realized margins only — no deserve-to-win quantity exists yet to
leak into them.

| Parameter | Value | What it is |
|---|---|---|
| Rematch pairs | **531** | usable sample size |
| SD of game margin | 13.73 points | |
| **Reliability** | **0.278** | correlation of game-1 and game-2 margin; the fraction of a single game's margin that is stable team strength |
| Home-field | 1.38 points | half the gap between A's home margin and A's rematch margin |
| A wins game 2 | 45.6% | A is usually the road team in the rematch |
| A hosts game 2 | 4.5% | playoff rematches at the same venue |

**Reliability of 0.278 is the number that governs everything below.** Roughly
72% of any single game's margin is not stable team strength. That is the noise
the design has to see a small effect through.

---

## 3. The power calculation

Document 04's closing lesson was that a threshold set from an effect-size
argument, with nobody asking whether the data *could* achieve it, is a gate that
fails for reasons unrelated to the hypothesis. So the power calculation runs
first, and it is the reason this document's gate is not the obvious one.

### Method

2,000 synthetic rematch datasets per scenario, at the measured design parameters:

```
delta        true strength difference, A minus B, in points
luck         the neutralizable part of a game's margin
residual     everything that is neither delta nor luck
margin_g1  = delta + luck + residual         (what the record says)
deserved_g1= delta + residual                (luck removed perfectly)
game 2     : A wins if delta + home-field + fresh noise > 0
```

The `deserved` arm is deliberately given the **best case** — luck removed
exactly, with no estimation error anywhere in the neutralization. A real
simulator does worse, so every power number below is an **upper bound**.

Scenarios are indexed by the share of margin variance the neutralization
removes, taken from document 01: fumble luck 3.7%, field-goal luck 2.7%,
interception 18.7%.

**Statistic:** mean out-of-fold log loss, deserved minus actual, paired at the
rematch pair. Ten-fold cross-validation, because the two arms are only
*structurally* identical — whichever predictor carries more signal also overfits
differently, so in-sample optimism does not cleanly cancel.

**Noise instrument:** two nulls, because one alone is arguable.

- *Independent null* — `deserved` is a different predictor of identical
  signal-to-noise. Type-I rate **0.022** against a nominal 0.025.
- *Correlated null* — the neutralization removes a proportional slice (no
  reliability gained) and adds estimation noise, leaving the two predictors
  highly correlated, as they will be in reality. Type-I rate **0.003**, i.e.
  conservative, as expected.

Setting the null to "remove zero luck" would make the two predictors numerically
identical and drive the type-I rate to a meaningless zero. That is why neither
null is defined that way.

### Result — the design cannot carry a superiority gate

| Scenario | Luck share removed | Mean Δ log loss | **Power** |
|---|---|---|---|
| Independent null | 0 | +0.0005 | 0.022 *(type-I)* |
| Correlated null | 0 | +0.0027 | 0.003 *(type-I)* |
| Fumble only | 3.7% | −0.0010 | **0.046** |
| **Fumble + FG — the realistic case** | **6.4%** | **−0.0017** | **0.072** |
| Fumble + FG + INT — the ceiling | 25.1% | −0.0087 | 0.297 |
| Implausibly large | 50% | −0.0265 | 0.789 |

> **Minimum detectable luck share at 80% power: 52% of margin variance.**
> The simulator can remove at most 25.1%, and realistically 6.4%.

The design is short of the power it would need by roughly a factor of eight. At
the realistic scenario a superiority test rejects **7.2%** of the time *when the
alternative is true and the neutralization is perfect*. A "deserve-to-win must
significantly beat the result" gate would therefore fail about 93% of the time
no matter how good the simulator is, and its failure would carry no information.

**That gate is not pre-registered, and the reason is recorded here in advance.**

### The secondary estimand does not rescue it

Predicting game 2's *margin* by OLS rather than its winner keeps the information
that dichotomising throws away — a 3-point win and a 30-point win are the same
event to a logistic model. It roughly doubles power and is still not enough.

| Scenario | Power, winner (log loss) | Power, margin (squared error) |
|---|---|---|
| Fumble only (3.7%) | 0.046 | 0.094 |
| Fumble + FG (6.4%) | 0.072 | **0.129** |
| Fumble + FG + INT (25.1%) | 0.297 | 0.501 |
| Implausibly large (50%) | 0.789 | 0.959 |

Type-I rate 0.022 against a nominal 0.025. It is reported as a **secondary**
estimand, not swapped in for the primary — the primary design is the one already
agreed, and changing the estimand after seeing which one looks better is exactly
the move pre-registration exists to prevent.

---

## 4. Pre-registered gates

Committed before any deserve-to-win number exists.

### Gate 1 — non-inferiority (the primary gate, and the one with power)

**Incumbent:** game 1's actual margin, in the identical logistic model.

**Statistic:** mean out-of-fold log loss, deserved minus actual, paired at the
rematch pair, 10-fold CV, folds assigned with `random_seed = 20260817`.

**Pass rule:** the **upper** bound of the 95% confidence interval on that
difference lies **below +0.010 log loss**.

Note the burden of proof runs opposite to a superiority test: a wide,
uninformative interval **fails** this gate, because it has not demonstrated the
absence of harm.

**Why +0.010:** Phase 1 measured the Vegas-versus-raw-EPA gap at 0.0398 log
loss, the largest predictive gap this project has resolved. The margin is
roughly a quarter of it — large enough that exceeding it would be a real
degradation, small enough to be worth checking.

**Power check (this is the part document 04 said must never be skipped):**

| Condition | Outcome |
|---|---|
| Healthy simulator (6.4% luck removed) | **false-alarm rate 0.008** |
| Neutralization 6.4% noise | caught 36.7% |
| Neutralization 15% noise | **caught 88.9%** |
| Neutralization 25% noise | caught 98.8% |
| Neutralization 40%+ noise | caught >99.8% |

The gate passes a healthy simulator 99.2% of the time and catches a materially
broken one — one whose luck estimate is 15% noise — 88.9% of the time. **This
gate is powered.** It is pre-registered on that basis and not on a football
argument about what 0.010 log loss feels like.

**On failure:** the neutralization is doing net harm to the information content
of a game. Report the failure, do not adjust the margin, and treat the
per-component treatment table in document 05 §3 as the suspect — most likely a
component was neutralized that should not have been.

### Gate 2 — direction, reported descriptively

**Statistic:** the sign of the same Δ log loss, plus its full 95% interval.

**No pass rule.** Pre-registering a threshold on a statistic with 7.2% power
would be theatre, which is the same reasoning document 03 §6 applied to its own
Gate 3. What is pre-registered is the **reporting rule**:

- The point estimate and its 95% interval are reported whatever they say.
- The interval will almost certainly contain zero. That is stated as a property
  of the **design**, established here in advance, and must not be reported as
  evidence that neutralization does not work.
- Under the alternative the point estimate favours deserve-to-win 74% of the
  time (81% on the margin estimand), against 47% under the null. A favourable
  sign is therefore weak corroboration and is to be described as such — never as
  confirmation.

### Gate 3 — the coefficient sanity check

**Statistic:** `b1`, the coefficient on the standardized game-1 predictor, in
both arms.

**Pass rule:** `b1 > 0` in both arms, with a 95% interval excluding zero.

This tests the harness, not the hypothesis. A team that won game 1 by more must
be more likely to win game 2; if that fails, the pairing, the orientation or the
sign convention is broken and no other number in document 07 can be read. It is
cheap, it is powered — reliability is 0.278 over 531 pairs — and it is the check
most likely to catch an implementation bug.

---

## 5. What this validation does *not* establish

Stated in advance so document 07 cannot quietly claim more.

- **It cannot show deserve-to-win is a better forecaster.** The design lacks the
  power, as computed above, and Phase 1 (document 02) already found that
  luck-stripping does not improve out-of-sample prediction. The simulator is a
  retrospective adjudication tool. This document tests that neutralization does
  no *harm*, which is a weaker and honest claim.
- **It does not validate the per-component treatment table.** A pass is
  consistent with several different treatment tables, because the design cannot
  resolve differences this small.
- **It does not validate the DTW credible interval.** Gate 1 scores a point
  estimate. Whether the interval has correct coverage is a separate question,
  answered by simulation in step 5, not here.
- **Rematch pairs are not a random sample of games.** They are mostly division
  games, which are systematically closer than average. Any result generalizes to
  division rematches first and to all games only by assumption.

---

## 6. Known-defect register

| Defect | Evidence | Status |
|---|---|---|
| Design is underpowered for superiority | MDE 52% of margin variance vs 25.1% available | **Accepted and pre-registered.** The reason Gate 1 is non-inferiority |
| Power figures are upper bounds | The simulated `deserved` arm has zero estimation error | **Accepted.** Stated wherever a power number appears |
| Simulation assumes Gaussian margins | Real margins are discrete, and lumpy at 3 and 7 | **Open.** Affects the type-I rate slightly; both nulls calibrated within 0.003 of nominal, so the effect is small |
| Rematch pairs skew to division games | 95.5% flip the host, which is the division home-and-away pattern | **Open.** Limits generalization, stated in §5 |
| Reliability estimated from the same 531 pairs the test runs on | The power calculation and the test share a sample | **Accepted.** Reliability uses only realized margins, which both arms share, so it cannot favour either |
| Three-meeting pairs discard a rematch | 17 pairs contribute one pair rather than two | **Accepted, by design.** Independence is worth more than 17 extra rows |

---

## 7. Constants appendix

| Constant | Value | Where it lives |
|---|---|---|
| `RANDOM_SEED` | 20260817 | `research/08_rematch_power.py`, `research/08_rematch.py` |
| Rematch pairs | 531 | measured, §2 |
| Reliability | 0.278 | measured, §2 |
| CV folds | 10 | `research/08_rematch_power.py` (`N_FOLDS`) |
| Power simulations per scenario | 2,000 | `research/08_rematch_power.py` (`N_SIMULATIONS`) |
| **Gate 1 non-inferiority margin** | **+0.010 log loss** | this document §4, `NONINFERIORITY_MARGIN` |
| Gate 1 false-alarm rate | 0.008 | power calculation, §4 |
| Gate 1 power at 15% noise | 0.889 | power calculation, §4 |
| MDE for superiority at 80% power | 52% of margin variance | power calculation, §3 |

Results are written to `docs/research/07-validation-results.md`.
