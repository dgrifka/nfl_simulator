# 10 — Does the DTW interval mean what it says?

*Written 2026-08-17, **before `research/17_coverage.py` existed**. Instrument
characterization: `research/16_coverage_power.py`, results in
`research/outputs/16_coverage_power.json`. Committed to git before the check
produces a result.*

---

## 1. The question

Document 07 closed by listing what its validation did *not* establish. This was
on the list:

> **"That the DTW credible interval has correct coverage.** Gate 1 scores a point
> estimate. Coverage is a separate question and is not answered here."

Every deserve-to-win number this project reports carries an 89% interval. If
that interval is not an 89% interval, the headline number is fine and the
honesty around it is a fiction — which is worse than reporting no interval at
all, because a stated interval invites a reader to trust its width.

**The question: when the simulator reports an 89% interval on DTW%, does that
interval contain the truth 89% of the time?**

## 2. What "truth" means here, and what this check does not cover

There is no observable truth for a single real game — DTW% is a counterfactual.
So truth has to be constructed, and the construction fixes the scope.

For a synthetic game with **known** per-event probabilities `p*`:

```
TRUE DTW%  =  P( actual_margin - Σ(y_e - y*_e)·swing_e·points_per_epa > 0 )
              with y*_e ~ Bernoulli(p*_e), over 20,000 coin draws
```

The simulator does not know `p*`. It sees a finite record — `n_e` observations
of the event class — and forms a posterior from it, exactly as
`_class_rate_draws` does. The check asks whether the interval built from that
posterior covers the DTW% you would have computed with perfect knowledge.

**Scope, stated so it cannot be overread.** This is a check on the **two-layer
bootstrap of document 05 §4** — whether layer 1 and layer 2 compose into a
calibrated interval, *given* per-event posteriors that are themselves calibrated.
It is **not** a check on whether the field-goal hierarchy's posterior is
calibrated; Gates FG-4 and W-5 covered that separately with posterior predictive
checks. Per-event posteriors here are Beta–Binomial by construction, which is
exactly right for the fumble path and an idealization of the field-goal path.

The realism that *is* preserved: swings, probabilities and events-per-game are
**sampled from the shipped v1.1 ledger**, so the check runs at the magnitudes the
simulator actually books rather than at invented ones.

## 3. The instrument, and why it needed characterizing first

A coverage check that cannot distinguish a healthy simulator from a broken one
is not a check. So the instrument was run on two arms before any threshold was
written:

- **healthy** — the shipped `bootstrap_margins`, called exactly as
  `simulate_game` calls it.
- **layer-1 disabled** — the same code with each event's posterior collapsed to
  its mean. This is the precise failure document 05 §4 built layer 1 to prevent:
  *"Layer 1 is what stops the simulator from reporting a suspiciously tight
  interval around a quantity estimated from 15 fumbles per team-season."*

| Arm | Coverage | Informative games only | Mean interval width |
|---|---|---|---|
| Healthy (two layers) | 0.9405 ± 0.0104 | 0.9731 | 0.0596 |
| Layer 1 disabled | 0.9235 ± 0.0116 | 0.9346 | 0.0433 |

**Separation: 1.7 pp, only 2.1 standard errors.** That is a weak instrument for
its stated purpose, and the reason is visible in the third column: *even with
layer 1 removed*, coverage stays above nominal. Something other than posterior
uncertainty is padding the interval.

Monte Carlo precision at 2,000 synthetic games is **0.53 pp**, so the check can
resolve a 3 pp miscalibration at roughly 6 standard errors. Precision is not the
problem; the estimand is doing something unexpected.

### The mechanism, characterized before the gate was written

`dtw_per_draw` is itself an average over a **finite** number of coin flips. Its
spread across posterior draws therefore mixes two things:

1. genuine uncertainty about `p` — which belongs in the interval, and
2. Monte Carlo noise from using only `n_coin_draws` flips — which does not.

If the second is inflating the width, raising the coin count must shrink it.

| Coin draws per posterior draw | **Mean interval width** |
|---|---|
| 25 | 0.0931 |
| **100 (shipped)** | **0.0632** |
| 400 | 0.0485 |
| 1,600 | 0.0473 |

**Unambiguous, and it converges.** Between 400 and 1,600 the width barely moves,
so 0.047 is the interval's genuine posterior width. At the shipped 100 draws it
is 0.063 — **about a third of the reported width is Monte Carlo noise rather
than uncertainty about anything.**

This is what turns a coverage number into an actionable finding, and it is why
the mechanism was characterized before the gate rather than after the failure.

## 4. Pre-registered gates

Committed before the check runs.

### Gate V-1 — is the interval calibrated?

**Statistic:** the fraction of synthetic games whose true DTW% falls inside the
reported 89% interval, at the **shipped settings** (200 posterior draws,
100 coin draws) over 4,000 synthetic games.

**Pass rule:** coverage within **3 percentage points of nominal**, i.e. in
**[0.86, 0.92]**.

**Where 3 points comes from — and note it is not from anything observed.**
Nominal is 0.89 by construction. Monte Carlo precision at 4,000 games is 0.37 pp,
so a 3 pp band is roughly 8 standard errors wide and the test resolves it
comfortably. Three points is also the most miscalibration that could be waved
through: an interval covering 92% of the time is not an 89% interval, and
labelling it one is the specific dishonesty this document exists to detect.

### Gate V-2 — which direction? *(reported, no pass rule)*

**Statistic:** the sign of `coverage − 0.89`.

The two failure directions are not equally serious and the asymmetry is
pre-registered here rather than argued afterwards:

- **Under-coverage is the serious failure.** The simulator would be
  *overclaiming* — reporting more confidence than it has, which is exactly what
  document 04's closing instruction ("report uncertainty from the population
  posteriors, not from point estimates") was meant to prevent.
- **Over-coverage is conservative.** The simulator would be understating its own
  confidence. The number would still be mislabelled and must be corrected, but
  no reader is misled into over-trusting it.

### Gate V-3 — coverage on informative games *(reported, no pass rule)*

**Statistic:** coverage restricted to games whose true DTW% is strictly between
0.001 and 0.999.

A game with no meaningful luck has a true DTW% of exactly 0 or 1 and a
degenerate interval, and it is covered trivially. Including those games inflates
the headline. Both are reported so the headline cannot be read as better than it
is.

### Gate V-4 — is any miscalibration attributable? *(reported, no pass rule)*

**Statistic:** mean interval width at 100 versus 1,600 coin draws, at otherwise
identical settings.

Its purpose is diagnostic. If Gate V-1 fails, this says whether the cause is a
**parameter** (Monte Carlo noise, fixable by raising a constant) or a
**design** (the two-layer construction is wrong). Those two failures have
completely different correctives and a coverage number alone cannot tell them
apart.

### The decision rule, committed in advance

| Outcome | Verdict | What changes |
|---|---|---|
| V-1 passes | The interval means what it says | Nothing |
| V-1 fails, **under**-coverage | **Serious.** The simulator overclaims | Intervals are not reported until fixed |
| V-1 fails, **over**-coverage, attributable to coin draws | The interval is conservative and mislabelled | Raise `n_coin_draws`, re-measure, and report the corrected coverage. Existing published numbers are not wrong, only wide |
| V-1 fails, **over**-coverage, not attributable | The two-layer construction is mis-specified | Report the failure; the interval's design is the suspect |

## 5. Disclosure

**The instrument run in §3 exposed the healthy arm's coverage before this
document was written.** That is the same exposure document 08 §7 records for the
sequencing round, and it is recorded here for the same reason.

Gate V-1's threshold is set from **the nominal 89% and the Monte Carlo
precision** — two quantities the observed coverage cannot move. A calibration
gate has a natural centre that has nothing to do with the data: the coverage it
claims. Characterizing the instrument first was also not optional, because §3's
two-arm run is what revealed the estimand was behaving strangely, and writing a
gate without knowing that would have produced an uninterpretable failure.

## 6. Known-defect register

| Defect | Evidence | Status |
|---|---|---|
| Per-event posteriors are Beta–Binomial by construction | Exact for fumbles and extra points; an idealization of the field-goal hierarchy | **Accepted, scoped in §2.** Gates FG-4 and W-5 cover the FG posterior separately |
| The instrument separates healthy from broken by only 2.1 SE | §3 | **Open.** The check is better at measuring calibration than at catching a specific sabotage, because Monte Carlo padding masks the difference |
| `points_per_epa` is fixed at 0.8389 in the synthetic games | The real simulator uses the same fitted constant | **Accepted.** It scales every margin identically and cannot affect coverage |
| Synthetic actual margins are Gaussian | Real margins are discrete and lumpy at 3 and 7 | **Open.** Same limitation document 06 §6 recorded; it affects how often a game is degenerate, which is why Gate V-3 splits them out |
| Truth uses 20,000 coin draws, not infinity | Monte Carlo error of ~0.35 pp on the truth itself | **Accepted.** An order of magnitude below the interval widths being tested |

## 7. Constants appendix

| Constant | Value | Where it lives |
|---|---|---|
| `RANDOM_SEED` | 20260817 | `research/16_coverage_power.py`, `research/17_coverage.py` |
| Synthetic games (check) | 4,000 | `research/17_coverage.py` |
| Synthetic games (instrument) | 2,000 per arm | `research/16_coverage_power.py` |
| Posterior draws / coin draws (shipped) | 200 / 100 | `research/15_simulator_v11.py` |
| Truth coin draws | 20,000 | `research/16_coverage_power.py` |
| Nominal coverage | 0.89 | document 03's convention |
| **Gate V-1 band** | **[0.86, 0.92]** | this document §4 |
| Informative-game bounds | 0.001 – 0.999 | this document §4 |
| `points_per_epa` | 0.8389 | `research/outputs/model_metadata_v11.json` |

Results are written back into this document as §8.
