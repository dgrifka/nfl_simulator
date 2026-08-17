# 11 — Drive anatomy: what is a drive, and what is left over?

*Written 2026-08-17, **before any split-half correlation on a finishing residual
was computed**. Design parameters, permutation null and power curve:
`research/19_drive_anatomy_power.py`, results in
`research/outputs/19_drive_anatomy_power.json`. Committed to git before
`research/19_drive_anatomy.py` produces a result, so goalpost integrity is
checkable by commit archaeology.*

*Inputs: documents 01–10, all settled. The process laws hold unchanged —
pre-register before fitting, power-check every threshold before committing it,
Gate A (branch point) before Gate B (arithmetic), convergence tolerances
relative rather than absolute (document 09's corrective), and characterize a new
instrument before writing its gate (document 10's precedent).*

---

## 1. One-page story

### The question

Document 08 §11 killed DQW%. The drive-outcome resampling failed its
pre-registered rematch gate, and the post-mortem was unambiguous:

> **corr(offensive points per drive, DQ adjustment per drive) = −0.784.**
> Between-team SD of points per drive: 0.490 observed → 0.346 after adjustment.
> Only **70.6%** of the real spread between offenses survives.

The instrument was **removing skill, not luck**. It conditioned each drive's
points on one number — how deep the drive got — and a drive that reached the 5
on a 60-yard touchdown pass and a drive that reached the 5 by grinding out three
yards a carry were treated as exchangeable. Explosive offenses were priced as
lucky ones.

§11 named the fix and refused to build it in the same breath: *"a drive summary
richer than depth — starting field position, plays, yards, and explosive-play
count — would plausibly pass. Building it after seeing this failure and
re-running the same gate would be goalpost-moving, so it is recorded as future
work requiring its own pre-registration and its own power calculation."*

**This document is the foundation that future work needs, and it asks two
questions, in order:**

1. **How much of a drive's scoring outcome does a rich summary actually
   explain**, against how much depth alone explains? If the answer is "barely
   more", the successor is dead before it is designed.
2. **What is left over — the finishing residual — and does it persist?** If the
   residual is a repeatable team property, then even a rich summary leaves skill
   in the thing a successor would resample, and the same failure recurs. If it
   does not persist, the resampling is licensed on a properly powered test.

### How it answers, in one paragraph

Every offensive drive gets a summary table built only from its **own plays** —
where it started, how deep it got, how many plays it took, how many were
explosive, how many first downs it earned, how much a defensive penalty helped.
Four **nested** feature sets are formed, each a strict superset of the last, and
each is turned into an out-of-fold conditional mean `E[points | features]`. The
leftover, `points − E[points | features]`, is the **finishing residual**. It is
summed per team-game, averaged per drive within each half of a team-season, and
correlated across halves — the same split-half machinery documents 02 and 08
used, so every number here is directly comparable to the ones already on record.

### Five things to hold onto

1. **No team identity enters any feature.** That is a ruling from the Phase 4
   plan, not an oversight: entity abilities may set the expectation for a coin
   flip — the kicker's shrunk rate — but they must never revalue the football
   that was actually played. A team-quality covariate here would leak a power
   ranking into an adjudication, which is the one thing this project's whole
   Gate-A discipline exists to prevent.
2. **The nesting is the design.** Reporting "a rich model explains 88% of drive
   points" alone would be uninterpretable. Reporting it against depth alone on
   the identical estimator, identical folds and identical data is the comparison
   that means something.
3. **Some features are entangled with the outcome, and they are named rather
   than hidden.** `net_yards` is very nearly `start_yardline_100` on any
   touchdown drive. It is put in its own top tier (F4) precisely so the size of
   that entanglement is *measurable* instead of assumed, and so a successor can
   decline to use it with the cost of declining on the record.
4. **This round is descriptive.** It changes nothing about DTW%, nothing about
   the ledger, and nothing in document 05's treatment table. Gate A rules
   sequencing out of neutralization at any value of `w`, and no result here can
   breach that. What it does is decide whether step 2 is worth pre-registering
   and, if so, on which feature set.
5. **The design is powered, and that was established before any threshold.**
   Minimum detectable split-half correlation is **r ≈ 0.10** at 80% power on
   every residual, and the r = 0.12 reference is caught 89–93% of the time. A
   null result here is therefore evidence of absence, which is a claim this
   project can only make where the power ran first.

### Statistic convention

Split-half correlations are the **mean over 200 random within-season splits**,
with the 5th–95th percentile across those splits, identical to documents 02 and
08. Variance-explained figures are **out-of-fold** R² on 10 folds. Where a
posterior appears it is a mean with an 89% equal-tailed interval, matching
documents 03, 05, 05b, 08 and 09.

---

## 2. Data

- **Grain of a row**: one offensive drive.
- **Source**: `data/pbp/*.parquet`, 2016–2025.
- **Population**: **49,507 drives** in the resampling universe, across **5,522
  team-games** and **320 team-seasons**. Mean 8.97 drives per team-game, mean
  **77.4 drives per half** of a team-season.
- **League scoring**: 2.4289 points per drive, variance 8.8324.
- **Universe**: the same five drive results document 08 §10 fixed — Touchdown,
  Field goal, Punt, Missed field goal, Turnover on downs. Drives ending in a
  turnover, a safety, or the clock are excluded, unchanged and for the reasons
  that document gave: fumble luck is already inside DTW%, interceptions are
  deliberately never neutralized in this project, a safety is not an
  offensive-points outcome, and a clock-terminated drive is a different kind of
  luck that is out of scope.
- **Minimum group size**: 8 games per team-season, so each half holds at least
  4. Documents 02 and 08 used the same floor.

### The summary table, feature by feature

| Feature | Definition | Clean of the outcome? |
|---|---|---|
| `start_yardline_100` | Distance to the goal at the drive's **first** scrimmage snap | **Yes** — fixed before the drive happens |
| `depth` | Minimum `yardline_100` **at the snap** of a run or a pass | **Yes**, by construction — see below |
| `scrimmage_plays` | Count of runs and passes on the drive | Mostly — a scoring drive ends when it scores |
| `explosive_plays` | Runs of 12+ yards plus passes of 16+ yards | Partly — the scoring play can itself be explosive |
| `max_gain` | Largest single-play gain | Partly, same reason |
| `first_downs` | nflverse `first_down` summed over the drive | Partly — a touchdown drive earns first downs on the way |
| `penalty_aid_yards` | Penalty yards flagged on the **defense** during the drive | Mostly |
| `net_yards` | Yards gained on scrimmage plays | **No** — ≈ `start_yardline_100` on every touchdown |

### Facts that must be defensible by name

- **`depth` keeps document 08's definition exactly.** The deepest yard line
  reached *at the snap*, never at the end of a play. A touchdown drive
  necessarily *ends* at `yardline_100 = 0`, so defining depth by where the drive
  finished would encode the outcome into the conditioning variable and make any
  resampling vacuous. Keeping the definition byte-identical is also what makes
  the F1 arm a fair statement of what document 08 actually shipped.
- **The explosive-play cut is a football convention, not a fitted threshold.**
  12+ yards on a run, 16+ on a pass. Choosing the cut after seeing which cut
  persists would be the goalpost-moving document 04's Gate 2 recorded, and it is
  the same discipline document 08 applied to the red-zone cut at 20 yards.
- **Out-of-fold prediction is not optional.** An in-sample residual from a
  flexible learner is shrunk toward zero by the fit itself, which would
  understate exactly the quantity being tested. Ten folds, assigned once with
  `random_seed = 20260817`, shared by every feature set.
- **The same estimator runs on every feature set.** A gradient-boosted regressor
  with fixed hyper-parameters, so the comparison between nested sets is a
  comparison of *information* and not of model families. A booster rather than a
  linear fit because the incumbent instrument was a nonparametric cell table:
  handing the richer sets a more flexible learner than the incumbent had would
  flatter them.
- **The split is by team-game, never by drive.** Two drives in the same game
  share a defense, a game script and a set of conditions. Splitting inside a
  game would break the independence the split-half estimator assumes and inflate
  every correlation on the page.
- **Penalty aid is measured over the whole drive, not just scrimmage plays**, so
  a defensive penalty on a punt that hands the offense the ball back is counted.

---

## 3. The nested feature sets

| Set | Features | What it represents |
|---|---|---|
| **F0** | 5-yard depth bins, nonparametric cell means | **Document 08's shipped instrument**, reimplemented out-of-fold |
| **F1** | `depth` | The same information, given to the shared learner |
| **F2** | + `start_yardline_100`, `scrimmage_plays` | "How far did the ball travel, and how long did it take" |
| **F3** | + `explosive_plays`, `max_gain`, `first_downs`, `penalty_aid_yards` | **The rich summary document 08 §11 named** |
| **F4** | + `net_yards` | The outcome-entangled ceiling, included to measure the entanglement |

F0 versus F1 is a control on the estimator itself: the same information through
two different machines. If they disagree materially, the boosted learner is
adding something the cell table could not, and every downstream comparison is
partly a model-class comparison rather than an information comparison.

---

## 4. The instrument, characterized before the gate was written

*Document 10 §3 established this step as a process law: a check that cannot
distinguish a healthy instrument from a broken one is not a check, so the
instrument is run on known arms before any threshold is committed.*

Two properties had to be known in advance.

**First, how much information each nested set carries.** Out-of-fold R² on drive
points, ten folds, shared estimator:

| Feature set | OOF R² | Residual SD (points) |
|---|---|---|
| F0 — depth cell means *(document 08's instrument)* | 0.5837 | 1.9176 |
| F1 — depth | 0.5837 | 1.9176 |
| F2 — + start, plays | 0.6406 | 1.7817 |
| **F3 — + explosive, max gain, first downs, penalty aid** | **0.8797** | **1.0309** |
| F4 — + net yards | 0.9021 | 0.9300 |

F0 and F1 agree to four decimal places, which settles the estimator control in
§3: the boosted learner extracts nothing from `depth` that the cell table did
not. Every difference below F1 is information, not machinery.

**Second, whether the split-half estimator can see anything at these
denominators.** The permutation null — real team-games dealt at random into
synthetic team-seasons, destroying team identity while keeping every
denominator, every within-game correlation and the real residual distribution —
centres on zero for all six measures, which is the first thing a null must do:

| Measure | Null mean r | Null SD | **95th pct** | 99th pct |
|---|---|---|---|---|
| points per drive *(control)* | +0.0030 | 0.0400 | **0.0706** | 0.0891 |
| residual, F0 cell means | +0.0020 | 0.0413 | **0.0705** | 0.0991 |
| residual, F1 depth | +0.0021 | 0.0414 | **0.0706** | 0.0970 |
| residual, F2 advance | +0.0025 | 0.0430 | **0.0674** | 0.0981 |
| residual, F3 production | +0.0018 | 0.0425 | **0.0695** | 0.0948 |
| residual, F4 yardage | +0.0020 | 0.0420 | **0.0642** | 0.1050 |

500 replicates, each running the full 200-split protocol.

---

## 5. The power calculation

500 replicates per cell. A true team-level spread `tau` is added at the
**team-game** level, never the half level — the executed statistic averages over
200 random splits of the *same* games, so its split draws are heavily
correlated, and redrawing noise per half would make them independent, shrink the
null spread by roughly √200 and hand the design power it does not have. Document
08 §5 records that exact mistake being caught in this project.

"Achieved mean r" is printed beside the nominal target so any calibration slip
is visible rather than assumed away.

| Measure | true r = 0.05 | 0.08 | **0.10** | **0.12** | 0.20 | 0.30 |
|---|---|---|---|---|---|---|
| residual, F0 cell means | 0.31 | 0.58 | **0.78** | **0.90** | 1.00 | 1.00 |
| residual, F1 depth | 0.31 | 0.55 | **0.77** | **0.89** | 1.00 | 1.00 |
| residual, F2 advance | 0.34 | 0.64 | **0.80** | **0.92** | 1.00 | 1.00 |
| residual, F3 production | 0.30 | 0.58 | **0.75** | **0.89** | 1.00 | 1.00 |
| residual, F4 yardage | 0.37 | 0.65 | **0.82** | **0.93** | 1.00 | 1.00 |

> **Minimum detectable split-half correlation at 80% power: r ≈ 0.10**, on every
> residual. The smallest effect this project has ever called real is document
> 02's r = 0.12, and every residual detects that with 89–93% power.

The positive control (points per drive) needs no power curve. It is known to
persist — document 08 measured offensive EPA per play at r = +0.601 on this same
machinery — and its role is to fail loudly if the harness is broken.

---

## 6. Pre-registered gates and the decision rule

Committed before any result exists. **This round is descriptive**, so nothing
below moves a ledger row, a treatment-table row, or a DTW% number. What these
gates decide is whether step 2 is worth pre-registering, and on which feature
set.

### Gate DA-1 — the harness works *(positive control)*

**Statistic:** split-half r of points per drive.

**Pass rule:** exceeds the permutation null's 99th percentile (0.0891) by a wide
margin, and lands in the neighbourhood of the r ≈ 0.5–0.6 range documents 02 and
08 measured for offensive quality.

If a quantity everyone already knows persists comes back flat, the drive table,
the grouping or the split logic is broken and **no other number in the results
section may be read.** This is the role document 06's Gate 3 and document 08's
Gate S-1 both played.

### Gate DA-1b — the replication control

**Statistic:** the share of between-team spread in points per drive that the F0
conditional mean retains, computed as
`SD(team-season mean predicted points per drive) / SD(team-season mean observed
points per drive)`.

**Pass rule:** within 5 percentage points of the **70.6%** document 08 §11
measured for the shipped depth-bin instrument.

This is the cheapest and most valuable check on the page. The drive table here
is a *reimplementation* — new features, a new estimator, out-of-fold rather than
in-sample — and if it cannot reproduce the number that killed DQW%, then the
richer sets are being compared against something other than the thing that
failed. **On failure, nothing below is readable and the reimplementation is the
suspect.**

### Gate DA-2 — does the finishing residual persist? *(one per feature set)*

**Statistic:** the residual's split-half r, mean over 200 splits.

**Threshold:** the permutation null's 95th percentile for that measure, from §4
— F0 0.0705, F1 0.0706, F2 0.0674, F3 0.0695, F4 0.0642.

**Reading, committed in advance:**

- **r above the threshold** → the residual is a **repeatable team property**.
  Skill survives that summary, and resampling the residual would erase it —
  which is precisely how DQW% failed. A successor built on that feature set
  would be expected to fail Gate D-2 again.
- **r below the threshold** → the residual **does not persist**, at a design
  that detects r = 0.12 with 89–93% power. Resampling it is licensed on
  evidence of absence rather than absence of evidence.

**No pass/fail is declared, because no simulator decision hangs on it.** These
are reported with their thresholds and their power attached, and they are the
input to step 2's design. The distinction matters: document 08 §6's Gate S-2 had
a decision rule because a verdict routed somewhere. This one routes into a
pre-registration that has not been written yet.

### Gate DA-3 — is a null interpretable? *(the honesty gate)*

**Pass rule:** power at the r = 0.12 reference is at least **0.80**, per §5.

All six measures clear this in advance. It is stated anyway, because if a future
re-run on a different subset drops power below 0.80 the correct report is
**"unresolvable"**, not "no finishing skill". Document 05 §7's return-yardage row
is the worked example of what happens without this gate.

### Gate DA-4 — how much between-team spread does each summary retain? *(reported, no pass rule)*

**Statistic:** the Gate DA-1b quantity, computed for every feature set, plus
`corr(team-season points per drive, mean residual per drive)` — the sign-flipped
twin of document 08 §11's −0.784.

**No threshold**, because this is the quantity step 2 will need to *set* a
threshold on, and pre-registering a bound for it here and then re-using it there
would be circular. It is reported descriptively and becomes step 2's design
input.

### The decision rule, committed in advance

| Outcome | What it means | What happens next |
|---|---|---|
| DA-1 or DA-1b fails | Harness or reimplementation broken | Nothing is reported. Fix and re-run |
| F3's residual persists (DA-2 above threshold) | A rich summary still leaves skill in the finishing | **Step 2 is not pre-registered on F3.** Report the finding; the successor's premise is refuted |
| F3's residual does not persist | Rich conditioning absorbs the quality that depth missed | Step 2 is pre-registered on the sets whose residual is flat, with sufficiency criteria set from DA-4's numbers |
| Only F4's residual is flat | The absorption needed the outcome-entangled feature | Report it as such. A successor leaning on `net_yards` is measuring the outcome, and that must be argued explicitly, not slipped in |

---

## 7. Disclosure

**§4's variance-explained table and the permutation null were computed before
this document was written.** That is the same exposure document 08 §7 and
document 10 §5 record, and it is recorded here for the same reason: recording it
is the only defence available, and hiding it would be worse.

What the exposure could and could not move:

- **Could not move the thresholds.** Gate DA-2's thresholds are the permutation
  null's 95th percentiles, computed by simulation from a null that destroys team
  identity. No R² figure can shift them.
- **Could not move Gate DA-1b.** Its target, 70.6%, is a number document 08 §11
  published in Phase 3, and the tolerance is stated as a round 5 points rather
  than fitted to anything.
- **Could have moved which feature sets exist.** It did not — the four sets are
  the ones document 08 §11 named ("starting field position, plays, yards, and
  explosive-play count"), plus the incumbent and plus the entangled ceiling. But
  this is the honest place to say that the *choice of sets* is the part of this
  design most exposed to having seen the R² column, and a reader who distrusts
  it should read F3 as the pre-named set and F4 as its explicitly-flagged
  entangled cousin.

Characterizing the instrument first was also not optional. §4's F0-versus-F1
agreement is what licenses reading every later comparison as information rather
than machinery, and a gate written without knowing that would have produced an
uninterpretable result either way.

---

## 8. Known-defect register

| Defect | Evidence | Status |
|---|---|---|
| **`net_yards` is entangled with the outcome** | ≈ `start_yardline_100` on every touchdown drive | **Open, by design.** Quarantined in F4 so its contribution is measurable; a successor using it must argue for it explicitly |
| `first_downs`, `explosive_plays` and `max_gain` are weakly entangled | A touchdown drive earns first downs on the way, and the scoring play can itself be explosive | **Open, stated.** Weaker than `net_yards` but not zero, and it is why F3's R² is an upper bound on genuinely pre-outcome information |
| Drives are summarized, never replayed | `CLAUDE.md` puts full play-level re-simulation out of scope | **Accepted, by design** |
| The 2024 dynamic-kickoff rule change moves drive start position | Starting field position is a structural break between 2023 and 2024 | **Open.** Every statistic here is within-season or league-pooled across ten seasons; a successor conditioning on `start_yardline_100` inherits it |
| Team-seasons pool coordinator changes | A team changes coordinators mid-season | **Open.** Same limitation documents 04 and 08 recorded |
| Drive results are nflverse's `fixed_drive_result` | Not our taxonomy | **Accepted.** Unchanged from document 08 so the arms stay comparable |
| The estimator is a boosted regressor with fixed hyper-parameters | No tuning was performed | **Accepted, deliberately.** Tuning per feature set would make the nesting a comparison of tuning effort |

---

## 9. Constants appendix

| Constant | Value | Where it lives |
|---|---|---|
| `RANDOM_SEED` | 20260817 | `research/19_drive_anatomy_power.py`, `research/19_drive_anatomy.py` |
| `N_SPLITS` | 200 | both scripts |
| Null / power replicates | 500 / 500 | `research/19_drive_anatomy_power.py` |
| `N_FOLDS` (out-of-fold) | 10 | both scripts |
| `MIN_GAMES` | 8 | both scripts |
| `EXPLOSIVE_RUN_YARDS` / `EXPLOSIVE_PASS_YARDS` | 12 / 16 | both scripts |
| `RED_ZONE_YARDS` | 20 | both scripts |
| `REFERENCE_R` | 0.12 | this document §5, from document 02 |
| **Gate DA-2 thresholds** | **F0 0.0705 · F1 0.0706 · F2 0.0674 · F3 0.0695 · F4 0.0642** | this document §4, from the permutation null |
| Gate DA-1 threshold | 0.0891 (null 99th pct, points) | this document §6 |
| Gate DA-1b target | 70.6% ± 5 pp | document 08 §11 |
| Gate DA-3 threshold | power ≥ 0.80 at r = 0.12 | this document §6 |
| Drives / team-games / team-seasons | 49,507 / 5,522 / 320 | measured, §2 |

Results are written back into this document as §10.

---

## 10. Results

*Script: `research/19_drive_anatomy.py`. Design, statistics and thresholds fixed
by §§1–9 above, committed at `a2b9376` before this script produced a result.
Results in `research/outputs/19_drive_anatomy.json`; the drive table itself is
persisted to `research/outputs/drive_anatomy.parquet`.*

### Gate outcomes, stated first

| Gate | Rule | Result |
|---|---|---|
| **DA-1 — positive control** | points per drive far above the null 99th pct (0.0891) | **PASS** — r = **+0.629** |
| **DA-1b — replication control** | F0 spread retention within 5 pp of 70.6% | **PASS** — **70.5%**, a gap of 0.08 pp |
| **DA-2 — does the residual persist?** | reported against the null 95th pct | **every feature set persists** — F1 +0.324, F3 **+0.108**, F4 +0.132 |
| **DA-3 — interpretability** | power ≥ 0.80 at r = 0.12 | **PASS** on all five (0.89–0.93) |
| **DA-4 — spread retention** | descriptive | 70.5% → **94.8%** from depth to the rich summary |

49,507 drives, 5,522 team-games, 320 team-seasons. Nothing below was chosen
after seeing a number.

### Gate DA-1b — the replication is exact, and that matters

| Quantity | Document 08 §11 | Here |
|---|---|---|
| Share of between-team spread retained | 70.6% | **70.5%** |
| corr(quality, adjustment) | −0.784 | **+0.784** *(sign-flipped twin — the residual is the negated adjustment)* |

A reimplementation with new features, a different estimator and out-of-fold
prediction rather than in-sample reproduces the number that killed DQW% to
within a tenth of a percentage point, and reproduces its headline diagnostic to
three decimal places. **Everything below is therefore a comparison against the
thing that actually failed**, not against a lookalike.

### The drive summary table

49,507 drives in the resampling universe:

| Result | Drives | Share |
|---|---|---|
| Punt | 22,382 | 45.2% |
| Touchdown | 13,404 | 27.1% |
| Field goal | 9,070 | 18.3% |
| Turnover on downs | 3,021 | 6.1% |
| Missed field goal | 1,630 | 3.3% |

| Feature | Mean | Median |
|---|---|---|
| `start_yardline_100` | 70.9 | 75 |
| `depth` | 39.2 | 39 |
| `scrimmage_plays` | 6.15 | 6 |
| `net_yards` | 34.4 | 29 |
| `max_gain` | 16.7 | 14 |
| `explosive_plays` | 0.70 | 0 |
| `first_downs` | 1.85 | 1 |
| `penalty_aid_yards` | 2.60 | 0 |

**The entanglement §2 named is real and it is large.** On the 13,404 touchdown
drives, **78.1%** have net yards within 5 of the starting distance to the goal —
because a touchdown drive by definition travels the whole field. `net_yards`
correlates with drive points at **+0.741** where `depth` correlates at −0.727.
That is why F4 is quarantined: a successor leaning on `net_yards` is partly
conditioning on the answer.

### Question 1 — how much does a rich summary actually explain?

| Feature set | OOF R² | Residual SD | **Spread retained** | corr(quality, residual) |
|---|---|---|---|---|
| F0 — depth cell means *(document 08's instrument)* | 0.581 | 1.92 | **70.5%** | +0.784 |
| F1 — depth | 0.584 | 1.92 | 70.3% | +0.786 |
| F2 — + start, plays | 0.641 | 1.78 | 76.4% | +0.737 |
| **F3 — + explosive, max gain, first downs, penalty aid** | **0.880** | **1.03** | **94.8%** | **+0.355** |
| F4 — + net yards | 0.902 | 0.93 | 96.8% | +0.266 |

**The answer to question 1 is emphatically yes.** Depth alone explains 58% of
the variance in a drive's points; the rich summary explains 88%, and it lifts
between-team spread retention from 70.5% to **94.8%**. Document 08's instrument
destroyed 29.5% of the real difference between NFL offenses; the rich summary
destroys **5.2%**.

F0 and F1 agree to three decimals, so none of that gain is the estimator.

**One caution on the correlation column, which turns out to be a weaker
diagnostic than it looks.** A team-season's points per drive *contains* its own
residual, so even a perfectly specified model leaves
`corr(quality, residual) = √(1 − retention²)` by pure arithmetic — 0.71 at F1's
retention and 0.32 at F3's. The observed values (0.786 and 0.355) sit barely
above those mechanical floors. **Most of what looks like a damning correlation is
an identity, not a defect**, and a successor's sufficiency criteria must not lean
on it. That is a design input for step 2 and it was not obvious in advance.

### Question 2 — does the finishing residual persist?

| Measure | Split-half r | 5th–95th pct | Threshold | Power at r = 0.12 | Reading |
|---|---|---|---|---|---|
| points per drive *(control)* | **+0.629** | +0.587 … +0.667 | — | — | harness works |
| residual, F0 cell means | **+0.318** | +0.262 … +0.371 | 0.0705 | 0.90 | persists |
| residual, F1 depth | **+0.324** | +0.269 … +0.377 | 0.0706 | 0.89 | persists |
| residual, F2 advance | **+0.246** | +0.188 … +0.300 | 0.0674 | 0.92 | persists |
| **residual, F3 production** | **+0.108** | +0.050 … +0.174 | 0.0695 | 0.89 | **persists** |
| residual, F4 yardage | **+0.132** | +0.065 … +0.199 | 0.0642 | 0.93 | persists |

**Every residual persists, including the rich one.** The rich summary cuts the
persistence by two-thirds — from +0.324 to +0.108 — but does not extinguish it.

Per the decision rule committed in §6 before this ran, that means: *"Step 2 is
not pre-registered on F3. Report the finding; the successor's premise is
refuted."* **That rule holds and it is what happens.** A resampling built on F3
would still be redrawing something repeatable, which is how DQW% failed.

Note also that F4 persists *more* than F3 despite explaining more variance.
Adding the outcome-entangled feature does not help; it moves the residual
somewhere the summary can see even less clearly.

### Why the residual persists — exploratory, and it changes what step 2 should be

**Labelled exploratory throughout because it was added after seeing the DA-2
result**, per the precedent document 07 set for its DTW% arm and document 08 §9
for its competitive-play control. It changes no gate. It changes what may
honestly be *said*, and what step 2 should be built on.

The obvious first suspect was that the persistence is nothing but offensive
quality the summary cannot see. That would reconcile DA-2 with document 08 §9's
finding that the red-zone **gap** does not persist (r = −0.034 at 87% power): a
gap subtracts the team's own overall efficiency, so a uniformly good offense
scores zero on it and positive on a residual.

**It is not that.** Regressing each half's residual rate on that half's own
offensive EPA per play, and correlating the regression residuals:

| Measure | Uncontrolled r | Controlled for offensive EPA/play | Exploratory null 95th pct |
|---|---|---|---|
| residual, F1 depth | +0.324 | **+0.112** | 0.054 |
| residual, F3 production | +0.108 | **+0.098** | 0.055 |
| residual, F4 yardage | +0.132 | **+0.134** | 0.072 |

F3's persistence barely moves. Whatever is repeating is not "being good at
football".

**So the same drives were valued three ways, and the answer is unambiguous:**

| Valuation of the drive | Residual split-half r | Exploratory null 95th pct | Reading |
|---|---|---|---|
| Points — all channels | **+0.108** | 0.062 | persists |
| **Touchdown points only** | **+0.042** | 0.056 | **flat** |
| **Field-goal points only** | **+0.229** | 0.066 | **persists strongly** |

> **Reaching the end zone, given a rich drive summary, does not persist. Turning
> a drive into three points does — and more than twice as strongly as the
> pooled measure.**

That reconciles everything on the page. Document 08 §9's S1 measured *offensive
EPA placement* and found no persistence; the touchdown row here measures the
same channel a different way and agrees. What the pooled residual was carrying
is the **kicking channel** — and kicker skill is not a mystery, it is a sized,
persistent team property this project already measured (`sigma_kicker` = 0.342,
document 05b §11) **and already neutralizes inside DTW%.**

Document 08 §11's own defect register saw this coming, twice:

> *"Field-goal make/miss sits inside the resampled quantity. A missed-FG drive
> can be redrawn to 3 points, and DTW% already neutralizes FG luck."*
>
> *"The red-zone-only arm still resamples FG outcomes. **Open. Recorded for the
> successor design.**"*

It was recorded as a second reason never to combine the two measures. It is
larger than that: **it is the dominant reason the finishing residual is not
resampleable, and removing it is the successor design.**

### What this changes

1. **The successor is not "a richer summary".** A rich summary was necessary —
   it lifted spread retention from 70.5% to 94.8% and cut residual persistence
   by two-thirds — and it is not sufficient. Building the measure document 08
   §11 sketched and re-running the rematch gate would have failed again, and
   this round is why that will not happen.
2. **A touchdown-valued drive resampling has a licensed premise**, on a
   properly powered test: r = +0.042 against a 0.056 null, at 89% power against
   the r = 0.12 reference. That is evidence of absence, not absence of evidence,
   and it is step 2's design.
3. **Field goals leave the resampled quantity entirely** and stay where they
   already are — priced by the kicker hierarchy inside DTW%, with weather.
   Resampling them in a second measure was double-counting a component the
   project had already solved.
4. **Nothing in document 05's treatment table moves.** No ledger row was at
   stake. Gate A rules sequencing out of neutralization at any value of `w`, and
   this round could not breach it.
5. **`corr(quality, adjustment)` is a weaker sufficiency criterion than it
   appears**, because most of it is the arithmetic identity above. Residual
   persistence is the sharp instrument, and step 2's sufficiency criteria are
   built on it.

### Defects added by this round

| Defect | Evidence | Status |
|---|---|---|
| **The pooled finishing residual persists at every summary tested** | F3 r = +0.108 against a 0.0695 threshold, 89% power | **Closed as diagnosed** — the persistence is the kicking channel, and the successor removes it. Not closed as *absent* |
| The touchdown-only null is closer to its threshold than S1's was | +0.042 against 0.056, where document 08's S1 sat at −0.034 against 0.0703 | **Open.** The margin is real but slimmer; step 2 re-tests it on its own denominators |
| `corr(quality, residual)` has a mechanical floor of √(1 − retention²) | 0.71 at F1, 0.32 at F3, against observed 0.786 and 0.355 | **New.** Any sufficiency criterion on this quantity must be stated net of the floor |
| The quality control uses offensive EPA per play from the same games | Control and outcome share a half's plays | **Open.** Attenuates the control, so the controlled column is if anything an *under*-statement of what survives |
| The exploratory channel split has 200-replicate nulls | Gated nulls used 500 | **Accepted.** Nothing is gated on them, and the two persisting rows clear their thresholds by 4× and 0.7× the null SD respectively |
| F4 persists more than F3 | +0.132 against +0.108, despite higher R² | **Open, unexplained.** Consistent with `net_yards` absorbing production that the residual then misses, but untested |
