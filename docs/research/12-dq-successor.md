# 12 — The DQW% successor, pre-registered

*Written 2026-08-17, **before `research/20_dq_successor.py` existed**. Power
calculation and instrument characterization: `research/20_dq_successor_power.py`,
results in `research/outputs/20_dq_successor_power.json`. Committed to git before
the measure produces a result, so goalpost integrity is checkable by commit
archaeology.*

*Inputs: documents 01–11, all settled. Process laws unchanged — pre-register
before fitting, power-check every threshold, Gate A before Gate B, relative
convergence tolerances, and characterize a new instrument before writing its
gate.*

---

## 1. One-page story

### The question

Document 08 §11's drive-outcome resampling failed its rematch gate because it
erased skill. Document 11 §10 built the fix that document named — a richer drive
summary — and found it **necessary but not sufficient**: spread retention rose
from 70.5% to 94.8% and residual persistence fell from r = +0.324 to +0.108, but
the residual still persisted, so the pre-registered decision rule refused to
pre-register a successor on it.

The diagnosis is what makes a successor possible. Valuing the same drives three
ways:

| Valuation | Residual split-half r | Null 95th pct | Reading |
|---|---|---|---|
| Points — all channels | +0.108 | 0.062 | persists |
| **Touchdown points only** | **+0.042** | 0.056 | **flat** |
| Field-goal points only | +0.229 | 0.066 | persists strongly |

**Reaching the end zone given a rich drive summary does not persist. Turning a
drive into three points does.** The persistence was the kicking channel — a
component this project already sized (`sigma_kicker` = 0.342), already
neutralizes inside DTW%, and whose presence inside the resampled quantity
document 08 §11's own defect register had flagged and filed under *"recorded for
the successor design."*

**The question this document asks: does a touchdown-only drive resampling,
conditioned on the rich summary, survive criteria strict enough to be worth
validating — and then pass the identical rematch gate?**

### The measure, in one paragraph

Drives that ended without a field-goal attempt — touchdown, punt, or turnover on
downs — have their points redrawn from the league's own distribution of drives
with a similar rich summary. Drives that attempted a field goal keep their
observed points, unchanged, because DTW% already prices those against the kicker
hierarchy with weather. The redrawn points replace the observed ones, the game's
margin moves accordingly, and **DQW%** is the share of replicates in which the
home team's margin is positive.

### Five things to hold onto

1. **The successor's defining move is a subtraction, not an addition.** Document
   08 §11 predicted the fix would be more features; document 11 §10 showed the
   binding problem was *which points were being redrawn*. Field goals leaving the
   resampled quantity is the change that matters.
2. **The rematch gate cannot carry this validation, and §5 proves it.** A
   purpose-built instrument shows the non-inferiority gate has **zero power**
   against a predictor that erases 10% of true team strength, and reaches 80%
   power only somewhere between 20% and 29%. It caught DQW% because DQW% erased
   29.4% — an enormous failure. **The sufficiency criteria are the real
   validation; the gate is a backstop against catastrophe.** That sentence is
   committed here, before the numbers, because it is the honest reading either
   way the measure lands.
3. **Sufficiency criteria are checked first and can stop the measure before it
   ever reaches the rematch test.** This is the Phase 4 plan's design and it is
   the right one: a measure that reaches a blind gate and passes has learned
   nothing.
4. **`corr(quality, adjustment)` is a weaker criterion than it looks.** Document
   11 §10 found it has a mechanical floor of √(1 − retention²), so most of the
   −0.784 that killed DQW% was an identity. SC-3 is therefore stated as an
   **excess over the floor**, against a sampling scale rather than a football
   argument.
5. **Nothing here touches DTW% or the ledger.** Gate A rules sequencing out of
   neutralization at any value of `w`. This is a second reported measure or it is
   nothing.

### Statistic convention

Split-half correlations are the mean over 200 random within-season splits with
the 5th–95th percentile, identical to documents 02, 08 and 11. The rematch
statistic is mean out-of-fold log loss, paired at the rematch pair, 10-fold CV,
`random_seed = 20260817` — byte-identical to documents 06, 07 and 08 §11.

---

## 2. The measure

### The universe

| Drive result | Drives | Treatment |
|---|---|---|
| Touchdown | 13,404 | **resampled** |
| Punt | 22,382 | **resampled** |
| Turnover on downs | 3,021 | **resampled** |
| Field goal | 9,070 | held at observed points |
| Missed field goal | 1,630 | held at observed points |
| Turnover, Safety, End of half/game | — | outside the universe entirely, unchanged from document 08 §10 |

**38,807 drives are resampled**, 78.4% of document 08's universe, across 5,520
team-games and 320 team-seasons. Mean 7.03 resampled drives per team-game.
League points per drive on this universe: 2.397, on the support {0, 6, 7, 8}.

### Facts that must be defensible by name

- **Field-goal drives are held, not resampled, and that is the whole design.**
  Three independent reasons, and each alone would be sufficient: the finishing
  residual's persistence lives there (document 11 §10, r = +0.229); DTW% already
  neutralizes field-goal luck against a kicker hierarchy with weather, so
  redrawing it here would double-count; and document 08 §11's defect register
  recorded both of those as open items for exactly this successor.
- **A drive's realized points are resampled, not a flat 6.** Keeping the
  {0, 6, 7, 8} support means extra-point and two-point outcomes ride along on
  their real frequencies, which a fitted distribution over points would smooth
  away. This is the same non-parametric choice document 08 §10 made and it is
  kept for comparability.
- **Conditioning is on the out-of-fold predicted value, binned.** The rich
  summary has seven features and a cell table cannot carry seven features. So the
  boosted conditional mean from document 11 is computed out of fold, drives are
  sorted into **20 equal-count bins** of that predicted value — roughly 1,940
  drives each, the same order as the depth-bin populations document 08's
  instrument used — and a drive draws its replacement uniformly from the observed
  points of league drives in its own bin.
- **The feature set is F3, not F4.** `net_yards` is entangled with the outcome
  (78.1% of touchdown drives have net yards within 5 of the starting distance to
  the goal) *and* its residual persisted **more** than F3's, not less. Using it
  would be conditioning partly on the answer.
- **No team identity enters any feature.** The Phase 4 plan's ruling: entity
  abilities may set the expectation for a coin flip, but must never revalue the
  football that was actually played. A team-quality covariate would leak a power
  ranking into an adjudication.
- **Depth keeps document 08's snap-time definition**, so a touchdown drive's
  conditioning variable is not the touchdown.

### The arithmetic

```
for each resampled drive d:
    bin(d)     = equal-count bin of E[points | rich summary of d], out of fold
    points*(d) ~ Uniform over observed points of league drives in bin(d)

DQ margin = actual_margin + sum_home(points* - points) - sum_away(points* - points)
DQW%      = P(DQ margin > 0) over replicates
```

The **DQ margin** uses the bin mean rather than a draw, so the point estimate
carries no Monte Carlo noise. **DQW%** uses the draws. Held drives contribute
identically to both terms and cancel exactly, which is what makes holding them a
subtraction rather than an assumption.

### The red-zone-trips-only variant

Named here in advance, as document 08 §10 named its own variant and for the same
reason: document 08 §9's null result is defined on `yardline_100 <= 20`, so
drives that reached the red zone are the part this resampling is *directly*
licensed on, and everything outside is an extension by assumption. The variant
restricts the resampling to drives that reached the red zone and holds every
other drive at its observed points. It is reported **whatever the primary arm
does**, so the licensed part stays separable and so it cannot be reached for
after a failure.

---

## 3. DAG

```
   rich drive summary  ---->  E[points | summary]  ---->  bin
   (start, depth, plays,           (out of fold)            |
    explosive, max gain,                                    v
    first downs, penalty aid)                    points* ~ league points in bin
                                                           |
   observed points of held drives  ------------------------+
   (field goals, and everything                            |
    outside the universe)                                  v
                                                     DQ margin, DQW%
```

**Where inference is cut.** The conditional mean is fitted once, out of fold, and
treated as fixed thereafter. That is a real cut and it is deliberate: the
alternative is a posterior over the conditional mean, which would widen DQW%'s
interval without changing its point estimate, and DQW% is reported as a share
rather than with a credible interval. Recorded in §8.

**Emergent behaviour to watch.** Bins near the top of the predicted-value range
are nearly all touchdowns, so a drive there is redrawn to a touchdown almost
surely and the resampling correctly declines to adjudicate it. That is a feature:
when the summary determines the outcome there is no finishing to redraw. It also
means the measure's total movement is concentrated in the middle bins, and a
vacuous measure would be one where *every* bin is degenerate — which is what
sufficiency criterion SC-4 exists to rule out.

---

## 4. The persistence instrument, and its power

*Sufficiency criterion SC-1's null and power, measured on the successor's own
universe and its own denominators rather than inherited from document 11.*

**Permutation null** — real team-games dealt at random into synthetic
team-seasons, destroying team identity while keeping every denominator, every
within-game correlation and the real residual distribution. 500 replicates, each
running the full 200-split protocol:

| Measure | Null mean r | Null SD | **95th pct** | 99th pct |
|---|---|---|---|---|
| touchdown-valuation residual | −0.0016 | 0.0414 | **0.0669** | 0.0886 |

**Power**, 500 replicates per cell, with the true team spread added at the
team-game level (never the half level — document 08 §5 records why):

| true r | 0.05 | 0.08 | **0.10** | **0.12** | 0.20 | 0.30 |
|---|---|---|---|---|---|---|
| power | 0.32 | 0.63 | **0.79** | **0.89** | 1.00 | 1.00 |

> **Minimum detectable split-half correlation at 80% power: r ≈ 0.10.** The
> r = 0.12 reference — the smallest effect this project has ever called real — is
> caught 89% of the time.

Out-of-fold R² of the conditional mean on this universe is **0.952**, residual SD
0.725 points.

---

## 5. The skill-erasure instrument — what the rematch gate can actually see

*This instrument did not exist before, and characterizing it before writing the
gate is document 10 §3's process law.*

Document 06 §4 measured the non-inferiority gate against **estimation noise** —
the neutralization removing a proportional slice and adding error on top — and
found 88.9% power against a predictor that is 15% noise. Nothing had ever
measured it against the failure mode that actually killed DQW%, which was not
noise but **the true between-team strength signal being shrunk**.

The generator erases strength and changes nothing else, so the catch rate is
attributable to erasure alone:

```
actual = delta + luck + residual
dq     = retention * delta + luck + residual
```

2,000 synthetic rematch datasets per row, at the measured design parameters (531
pairs, margin SD 13.73, reliability 0.278):

| Retention | Spread erased | Mean Δ log loss | **Power to catch** |
|---|---|---|---|
| 0.99 | 1.0% | +0.0004 | **0.000** |
| 0.97 | 3.0% | +0.0011 | **0.000** |
| 0.95 | 5.0% | +0.0019 | **0.000** |
| 0.90 | 10.0% | +0.0038 | **0.000** |
| 0.80 | 20.0% | +0.0074 | 0.632 |
| **0.706** | **29.4%** *(document 08's shipped instrument)* | +0.0108 | **0.980** |
| 0.60 | 40.0% | +0.0144 | 0.999 |

> **The non-inferiority gate is blind to skill erasure below 10%, and reaches
> 80% power only between 20% and 29%.**

This reframes Phase 3's result rather than undermining it. Document 08 §11's
failure was real and the gate caught it correctly — at 29.4% erasure the catch
rate is 98%. But **the gate caught it because the damage was enormous.** A
successor that quietly destroyed a tenth of the difference between NFL offenses
would pass Gate E-2 every single time.

**Two consequences, both committed here in advance.**

1. **The sufficiency criteria are the validation, and Gate E-2 is a backstop.**
   Passing E-2 is necessary for shipping because it was pre-registered as the
   standard in document 06 and re-used in document 08; it is nowhere near
   sufficient, and this document will not claim it is.
2. **SC-2's bound cannot be derived from the gate**, because the gate has no
   opinion below 20%. It has to come from a football-impact argument, which §6
   gives.

---

## 6. Sufficiency criteria — committed before the validation gate

*These run first. A measure that fails any of them never reaches Gate E-2, and
does not ship. This ordering is the Phase 4 plan's design and §5 is why it is
right.*

### SC-1 — the finishing residual must not persist *(binding)*

**Statistic:** split-half r of the per-drive residual `points − E[points |
summary]` on the resampled universe, mean over 200 splits.

**Pass rule:** **at or below 0.0669**, the permutation null's 95th percentile
from §4.

**Why this is the binding criterion.** Sampling noise does not persist across
the halves of a team's own season; erased skill does. It is the only criterion
that distinguishes "the residual is noise, so redrawing it destroys nothing"
from "the residual is a repeatable team property the summary cannot see" —
which is the exact failure document 11 §10 caught in the pooled valuation.

**Power:** 0.89 at r = 0.12, 0.79 at r = 0.10 (§4). **If power at the reference
falls below 0.80 the correct report is "unresolvable", not "the residual does
not persist"** — document 05 §7's return-yardage row is the worked example of
what happens without this clause.

### SC-2 — between-team spread must survive

**Statistic:** `SD(team-season mean adjusted offensive points per game) /
SD(team-season mean observed offensive points per game)`, where the adjustment is
the expected DQ adjustment. Stated at the **per-game** level rather than
per-drive, so it maps one-for-one onto §5's `retention` parameter, which is the
quantity the rematch gate is blind to.

**Pass rule:** **at or above 0.95.**

**Where 0.95 comes from — and note it is not derived from the gate, because §5
shows the gate has no opinion here.** Step 1 measured the between-team SD of
points per drive at 0.490. Over the ~11 offensive drives a team gets in a game,
a one-SD offense's scoring advantage is about 5.4 points per game, so erasing a
share `s` of the spread costs that offense `s × 5.4` points of its measured
advantage:

| Erasure | Cost to a one-SD offense | Reference |
|---|---|---|
| 3% | 0.16 points per game | |
| **5%** | **0.27 points per game** | the bound |
| 10% | 0.54 points per game | invisible to Gate E-2 |
| 29.4% | 1.59 points per game | document 08's failure |

The anchor is the smallest game-level effect this project has ever treated as
material: Phase 3's weather change moved the deserved margin by a **mean of
0.355 points** and that was judged large enough to justify an entire model
change and a re-validation. A criterion that admits erasure smaller than that is
admitting something the project has already decided is below its own threshold
of mattering. 5% clears it with room; 10% does not.

**Disclosure:** document 11 §10 measured 94.8% spread retention for the
points-valued F3 conditioning — just under this bound. That number was known
when this bound was written. The bound is derived above from the 0.355-point
anchor and would be 0.95 regardless, but a reader is entitled to know it sits
next to an already-observed value, and §7 records the exposure in full.

### SC-3 — the quality correlation must not exceed its mechanical floor

**Statistic:** `|corr(team-season points per drive, mean residual per drive)| −
√(1 − retention²)`, where the second term is the floor document 11 §10 derived:
a team-season's points per drive *contains* its own residual, so even a perfectly
specified model leaves a correlation of that size by arithmetic alone.

**Pass rule:** **at or below 0.0559**, which is `1/√320` — the scale of a
correlation that pure sampling produces at 320 team-seasons.

This is the Phase 4 plan's named criterion (*"the per-drive adjustment must be
~uncorrelated with offensive quality — the −0.784 failure diagnostic, now a
design requirement with a pre-set bound"*), corrected for the mechanical floor
document 11 §10 discovered. It is **deliberately a weak criterion**: most of
what the raw −0.784 described was an identity, and SC-1 is what actually binds.
It is retained because the plan named it and because a design that fails even
this is not worth validating.

### SC-4 — the measure must not be vacuous

**Statistic:** mean `|DQ margin − actual margin|` across all games, and the share
of games whose named winner differs from the actual result.

**Pass rule:** mean absolute adjustment **at least 1.0 point** *and* a different
winner in **at least 5%** of games.

**Why this criterion has to exist.** SC-1, SC-2 and SC-3 are all satisfied
perfectly by a measure that adjusts nothing at all, and so is Gate E-2. §3's
emergent-behaviour note is the concrete route to that: if every prediction bin is
degenerate, every drive is redrawn to itself and the measure is a very
well-validated identity function. The floors are set well below both existing
reference points — document 08's DQW% moved 6.79 points and flipped 21.9% of
games, DTW% moves 2.80 points and disagrees in 11.1% — so this rules out only a
measure that is doing nothing.

---

## 7. Disclosure

**Document 11 §10's exploratory channel split exposed the touchdown-valuation
residual's persistence (+0.042 against a 0.056 null) before this document was
written.** That is the finding this whole design rests on and it would be absurd
to pretend otherwise; the point of recording it is that a reader can weigh it.

What the exposure could and could not move:

- **Could not move SC-1's threshold.** 0.0669 is the permutation null's 95th
  percentile computed on this universe by simulation, and no observed value can
  shift it. It is also *different* from the 0.056 document 11 reported, because
  the universe (38,807 drives against 49,507) and the denominators differ — so
  the exposed number is not the number being tested.
- **Did move the design.** The decision to hold field-goal drives is a direct
  consequence of having seen the channel split. That is the intended use of an
  exploratory result — it generates a hypothesis, which is then pre-registered
  and tested on a stated statistic. What would be illegitimate is reporting the
  exploratory number *as* the confirmation, and §9's results section will report
  SC-1 on its own terms.
- **Could not move SC-2's bound**, which is derived in §6 from the 0.355-point
  Phase 3 anchor, nor **SC-3's**, which is `1/√n`, nor **SC-4's**, which is set
  below two published reference points.
- **§5's skill-erasure instrument was run before this document.** Its rows are
  simulated from the rematch design parameters and contain no information about
  the successor measure at all.

---

## 8. Pre-registered gates

*Reached only if all four sufficiency criteria pass.*

### Gate E-1 — the resampling is unbiased on the league

**Pass rule:** the mean resampled points per drive, pooled over the resampled
universe, is within **0.01 points** of the observed mean.

An identity check, not a hypothesis: resampling from the league's own conditional
distribution must reproduce the league's own mean. Failure means the binning or
the universe definition is wrong. Identical in construction and tolerance to
document 08 §10's Gate D-1.

### Gate E-2 — non-inferiority on the rematch harness

**Incumbent:** game 1's actual margin, in the identical logistic model document
06 §1 specifies.

**Statistic:** mean out-of-fold log loss, DQ margin minus actual margin, paired
at the rematch pair, 10-fold CV, `random_seed = 20260817`, on the same 531 pairs.

**Pass rule:** the **upper** bound of the 95% CI lies below **+0.010 log loss** —
the margin document 06 §4 pre-registered, with its measured false-alarm rate of
0.008 and 88.9% power against a predictor that is 15% estimation noise.

**Primary predictor is the DQ margin; DQW% is secondary**, in that order, because
document 07 measured that a probability compressed toward 0 and 1 discards margin
information and carries a larger standard error. Fixing the order now prevents
choosing the better-looking arm later. Both arms of the red-zone variant are
reported alongside.

**What this gate does and does not establish, committed in advance:** per §5 it
has **zero power** against erasure below 10% and 63% power at 20%. Passing it
means "not catastrophically broken", not "validated". A pass is reported with
§5's table attached, every time.

**No superiority gate.** Document 06 §3 computed the minimum detectable luck
share at 80% power as 52% of margin variance; nothing this project can build
reaches it. A superiority test would fail regardless of quality.

### Gate E-3 — DQW% and DTW% are distinct *(reported, no pass rule)*

**Statistic:** the correlation between DQW% and DTW% across all games, and the
count of games where they name different winners.

**No pass rule**, exactly as document 08 §10's Gate D-3. If they were to
correlate above ~0.95 the honest conclusion is that reporting both is redundant,
and that conclusion is admissible in advance.

### The decision rule, committed in advance

| Outcome | Verdict | What changes |
|---|---|---|
| Any sufficiency criterion fails | **Does not ship** | Report the failure and the criterion, like document 08 §11. The measure never reaches Gate E-2 |
| All SC pass, Gate E-1 fails | Implementation broken | Nothing is reported. Fix and re-run |
| All SC pass, Gate E-2 fails | **Does not ship** | Report the failure; treat the design as the suspect |
| All SC pass, Gate E-2 passes | **Ships as a second reported measure** | DQW% is reported alongside DTW%, never combined with it, always with §5's table attached |

**Their relationship, restated and unchanged from document 08 §10:** DTW% asks
*"once the coin flips are set to their expectations, who deserved to win?"* and
DQW% asks *"given the drives this team produced, how often does that production
win?"* They are **not to be multiplied, averaged, or combined into a single
index.** Doing so would count the same game twice through two different lenses,
and the per-component ledger that makes DTW% auditable has no equivalent for a
quantity built by averaging two bootstraps.

---

## 9. Known-defect register

| Defect | Evidence | Status |
|---|---|---|
| **Gate E-2 is blind below 10% skill erasure** | §5, 0.000 power at retention 0.90 | **Open, and pre-registered as such.** The reason the sufficiency criteria exist and the reason a pass is never reported alone |
| The conditional mean is a point estimate, not a posterior | §3's inference cut | **Accepted.** Widening it would widen DQW%'s spread without moving its centre, and DQW% is reported as a share |
| Whether a drive attempted a field goal is itself an outcome | The held/resampled split conditions on a decision the offense made | **Open, and the largest known weakness.** A team that kicks more field goals has more drives held; the direction is toward *less* adjustment for such teams |
| Fourth-down decisions sit inside the resampled quantity | A turnover on downs is in the universe, and going for it is a coach's choice | **Open.** Document 09 measured fourth-down conversion at 7.0% relative spread and Gate A denied it a ledger row; here it is inside a *reported measure*, not the ledger |
| Drives are resampled independently | Two drives in one game are drawn independently | **Accepted.** Same reasoning as document 05 §5's row on simultaneous luck events |
| The game being adjudicated is inside the league conditional table | ~7 drives of ~38,800 | **Open, bounded** at O(1/n) |
| The 2024 dynamic-kickoff change moves starting field position | `start_yardline_100` is a conditioning feature across a structural break | **Open.** Pooled across ten seasons; a season-aware conditioning is future work |
| Weakly outcome-entangled features remain in F3 | `first_downs`, `explosive_plays`, `max_gain` — document 11 §8 | **Open, stated.** F4 was excluded for the stronger version of the same problem |
| SC-2's bound sits near an already-observed value | Document 11 measured 94.8% for a related quantity | **Open, disclosed in §6 and §7** |

---

## 10. Constants appendix

| Constant | Value | Where it lives |
|---|---|---|
| `RANDOM_SEED` | 20260817 | `research/20_dq_successor_power.py`, `research/20_dq_successor.py` |
| `RESAMPLED_RESULTS` | Touchdown, Punt, Turnover on downs | both scripts |
| Feature set | F3 — depth, start, plays, explosive, max gain, first downs, penalty aid | `research/19_drive_anatomy_power.py` |
| `N_PREDICTION_BINS` | 20 | both scripts |
| `N_FOLDS` | 10 | both scripts |
| `N_SPLITS` | 200 | both scripts |
| Null / power replicates | 500 / 500 | `research/20_dq_successor_power.py` |
| Erasure-instrument datasets | 2,000 per row | same |
| **SC-1 threshold** | **0.0669** | this document §4, from the permutation null |
| **SC-2 threshold** | **0.95** | this document §6, from the 0.355-point Phase 3 anchor |
| **SC-3 threshold** | **0.0559** = 1/√320 | this document §6 |
| **SC-4 thresholds** | **1.0 point** and **5%** of games | this document §6 |
| Gate E-1 tolerance | 0.01 points per drive | this document §8 |
| **Gate E-2 margin** | **+0.010 log loss** | document 06 §4 |
| Rematch pairs / folds | 531 / 10 | document 06 |
| Resampled drives / team-games | 38,807 / 5,520 | measured, §2 |

Results are written back into this document as §11.
