# 08 — Sequencing luck, pre-registered

*Written 2026-08-17, **before any split-half correlation on this round was
recorded**. Power calculation: `research/10_sequencing_power.py`, results in
`research/outputs/10_sequencing_power.json`. Committed to git before
`research/10_sequencing.py` produces a result, so goalpost integrity is
checkable by commit archaeology.*

*Inputs: documents 01–07, all settled. The process laws they established carry
over unchanged — pre-register before fitting, power-check every threshold before
committing it, and Gate A (branch point) before Gate B (shrinkage arithmetic).*

---

## 1. One-page story

### The question

Phase 2 shipped a simulator that neutralizes fumble recovery and field goals —
together about 6.4% of margin variance. The other 70% sits in "core offense and
defense" (document 01), and Phase 1 treated that block as a single indivisible
thing called skill. It is not obviously indivisible.

Consider two teams that gain exactly the same yards, at exactly the same rate,
against exactly the same defenses. One of them happens to have its good plays
land on third down and inside the 20; the other has its good plays land on
first-and-10 at midfield and its bad plays land in the red zone. The first team
wins and the second loses. **The production was identical. Only its placement
differed.**

That placement is what this round calls **sequencing**. The question is whether
it is a team property — something coaches scheme and quarterbacks execute — or
whether it is where a season's noise happens to fall. It is the largest untested
channel left inside the 70% core block, and until it is tested the luck
accounting is not complete.

### How it answers, in one paragraph

Three measures, each a **gap**: the same team's efficiency in a high-leverage
subset minus its efficiency overall. A gap is the right shape because sequencing
is about *placement*, not amount — a team that is simply better everywhere shows
no gap at all, which is exactly the behaviour we want, because that team's
advantage is already counted as offense. Each gap is measured per team-season
and put through the split-half machinery of document 02: split a team's games in
two at random, compute the gap on each half, correlate the halves across all 320
team-seasons, repeat 200 times. A gap a team controls persists between its own
two halves. A gap that is where the season's noise fell does not.

### Five things to hold onto

1. **Sequencing is a gap, not a level.** Every measure subtracts the team's own
   overall efficiency, so being good at football cannot masquerade as being good
   at sequencing.
2. **This round is genuinely powered, and almost nothing else in this project
   has been.** The minimum detectable correlation is **r ≈ 0.10** at 80% power,
   against a smallest-effect-of-interest of 0.12 taken from document 02. Contrast
   document 04's Gate 2, which could not have passed under a true zero.
3. **The null comes from permuting real team-games**, not from an analytic
   formula. Team identity is destroyed; every within-game correlation, real
   denominator and fat play tail survives.
4. **The decision rule is committed in §6 before any result exists**, and it
   deliberately does *not* route a positive finding into the ledger. Sequencing
   has no branch point, so document 05's Gate A rules it out of neutralization
   at any value of `w`.
5. **A verdict of "luck" here does not make the simulator adjust anything.** It
   makes sequencing a *separate reported measure*, built on drive-outcome
   resampling. That distinction is the whole reason this document exists before
   the code does.

### Statistic convention

Split-half correlations are reported as the **mean over 200 random within-season
splits**, with the 5th–95th percentile across those splits — identical to
document 02, so every number here is directly comparable to the ones already on
record. Where a posterior appears it is a mean with an 89% equal-tailed
interval, matching documents 03, 05 and 05b.

---

## 2. Data

- **Grain of a row**: one team-game. A team-season is a group of those rows, and
  the split-half machinery splits within the group.
- **Source**: `data/pbp/*.parquet`, 2016–2025.
- **Population**: **320 team-seasons**, **5,522 team-games**, **343,543
  offensive scrimmage plays** and **451,190 plays carrying both an EPA and a WPA
  value**.
- **Filter, measures S0–S2**: `posteam` non-null, `play_type` in {pass, run},
  non-null `epa` and `down`. Kicks, punts and penalty-only rows are excluded:
  they are not "efficiency" in any sense a coach would accept, and a team that
  punts well would otherwise register as a team that sequences well.
- **Filter, measure S3**: every play the team had the ball for that carries both
  `epa` and `wpa`. S3 values the *whole* offensive record two ways, so
  restricting it to scrimmage plays would value only part of it.
- **Minimum group size**: 8 games, so each half holds at least 4. Document 02
  used the same floor.

### Facts that must be defensible by name

- **Pooling within a half, rather than averaging per game.** This is a
  deliberate deviation from document 02's primary machinery and the red-zone
  denominator is the reason: a team-game holds a *median of 9* red-zone plays
  and 2.0% of team-games hold none at all, so a per-game red-zone mean is either
  undefined or estimated off a handful of snaps. Pooling numerator and
  denominator across the half is the honest estimator — and it is the one
  document 02 itself switched to for the fumble *rate* test, for exactly this
  reason. The per-game-averaged variant is reported as a secondary so the
  numbers stay comparable to document 02's published table.
- **Red zone is defined as `yardline_100 <= 20`**, the football convention, not
  a fitted threshold. Choosing the cut after seeing which cut persists would be
  the same error document 04's Gate 2 recorded.
- **Late downs pool third and fourth together.** Fourth-down attempts are rare
  and are heavily selected by score and time, but separating them would leave a
  fourth-down-only measure with a median of about 15 attempts per team-season —
  a denominator this project has already learned not to trust. The pooled
  measure is the one with a real denominator; the fourth-down decision question
  is a *different* question and belongs to document 09.
- **`wpa` is missing on 7,157 of 484,254 rows (1.5%).** Those are clock rows and
  rows outside the win-probability model's support, the same MCAR-by-row-type
  pattern document 01 established for `epa`. They drop out of S3's denominator.
- **The EPA and WP models are both nflverse products and both have been revised
  across the ten-year window.** S3 is a *contrast between two nflverse models*,
  so it inherits both vintages. This is a real limitation and it is recorded in
  §7 rather than buried; all analysis is within-season, which contains it.

### Design parameters, measured before any threshold was set

Every one of these is league-pooled and carries no team information.

| Parameter | Value |
|---|---|
| `wpa_per_epa` slope | 0.023497 |
| WPA/EPA play-level correlation | 0.736 |
| Play-level EPA variance, all scrimmage | 1.916 |
| Play-level EPA variance, red zone | 2.250 |
| Success rate, all scrimmage / late down | 43.56% / 43.70% |
| Play-level variance of the WPA−EPA residual | 0.000775 |
| Red-zone plays per team-game (mean) | 9.43 |
| Late-down plays per team-game (mean) | 14.06 |
| Scrimmage plays per team-game (mean) | 62.4 |

---

## 3. The four measures

Each is computed on a **half** — the pooled plays of the team-games assigned to
that half of a team-season.

### S0 — overall offensive EPA per play *(positive control)*

```
S0 = sum(epa) / count(plays)
```

Not a sequencing measure. It is the harness check, and it exists for the same
reason document 06's Gate 3 existed: if a quantity we *already know* persists
comes back flat, the machinery is broken and nothing else on the page is
readable. Document 02 measured split-half `r = +0.519` on the core EPA
differential, so S0 must come back large.

### S1 — red-zone gap

```
S1 = mean(epa | yardline_100 <= 20) − mean(epa | all plays)
```

Positive means a team's production concentrated where it converts to points.
This is "red-zone efficiency over overall efficiency" in the plan's words, with
the subtraction doing the work: a uniformly good offense scores zero here.

### S2 — late-down gap

```
S2 = mean(success | down in {3,4}) − mean(success | all plays)
```

`success` is nflverse's EPA-positive indicator. Positive means a team's
production concentrated on the downs that extend drives. Note the league's
late-down success rate (43.70%) and overall rate (43.56%) are almost identical,
so this gap is centred near zero by construction rather than by luck.

### S3 — the WPA-minus-EPA gap

```
S3 = ( sum(wpa) − wpa_per_epa · sum(epa) ) / games
```

**Same plays, two valuations.** EPA values a play by the points it was worth;
WPA values the same play by the win probability it moved. The two agree at
`r = 0.736` play-level, and the league slope converting one into the other is
0.023497 win-probability points per EPA point. What is left over — the residual
— is *leverage*: the extent to which a team's production landed in moments where
it changed the outcome rather than in moments where it did not.

This is the purest of the three, because it does not require a subset to be
named in advance. A fourth-quarter touchdown in a tied game and a
fourth-quarter touchdown in a 30-point blowout are the same EPA and wildly
different WPA, and no red-zone or third-down definition captures that.

The residual is substantial: team-game WPA has an SD of 0.382 and the residual's
SD is 0.239, so **63% of the game-to-game spread in win-probability
contribution is not explained by EPA at the league slope.** Whether that 63% is
a team property is the single question this round most wants answered.

---

## 4. DAG

The generative story the power calculation simulates, and the one a verdict of
"skill" would assert.

```
   theta_team  ~  Normal(0, tau)          true sequencing tendency, per team-season
        |                                  (the SAME value in both halves)
        |
        +--------------------+
        |                    |
        v                    v
   half A statistic     half B statistic
   = theta + noise_A    = theta + noise_B

   noise_h ~ (sampling noise at half h's OWN real denominators)

   split-half r  ->  tau^2 / (tau^2 + sigma^2)
```

**Where inference is cut.** Nowhere inside the estimate — the split-half
correlation is a direct statistic, not a fitted model. But there is a cut
*between* this document and any use of its verdict: `wpa_per_epa` is estimated
once, league-wide, and treated as fixed thereafter. That slope is pinned by
451,190 plays, so its uncertainty is negligible next to the team-level question;
treating it as fixed is a rounding, not a modelling claim.

**Emergent behaviour to watch.** The gap measures are *differences of means over
nested samples* — red-zone plays are a subset of all plays. The two means
therefore covary, and the correct sampling variance is

```
Var(mean_sub − mean_all) = var_sub/n_sub + var_all/n_all − 2·var_sub/n_all
```

not the sum of two independent variances. Dropping the covariance term would
overstate the noise and hand the design more apparent power than it has. The
power script carries the term explicitly.

---

## 5. The power calculation

*Ran first. This section exists because document 04's Gate 2 did not have one.*

### Two nulls, because one alone is arguable

**Permutation null — the instrument the thresholds come from.** Real team-games
are dealt at random into synthetic team-seasons of the same sizes. Team identity
is destroyed; every within-game correlation, every real denominator and the real
fat-tailed play distribution survive. 500 replicates, each running the full
200-split protocol.

| Measure | Null mean r | Null SD | **95th pct** | 99th pct |
|---|---|---|---|---|
| S0 overall EPA | +0.0008 | 0.0403 | **0.0698** | 0.0917 |
| S1 red-zone gap | +0.0002 | 0.0419 | **0.0703** | 0.0943 |
| S2 late-down gap | +0.0005 | 0.0407 | **0.0689** | 0.0900 |
| S3 WPA−EPA gap | +0.0019 | 0.0404 | **0.0648** | 0.0920 |

All four nulls centre on zero, which is the first thing a null must do.

**Parametric null — the one that extends to a power curve.** A permutation
cannot produce a dataset with a *known* true spread, so the power column needs a
generative simulation. It runs at the **team-game** level, not the half level,
and that detail is load-bearing: the executed statistic averages over 200 random
splits of the *same* games, so its split draws are heavily correlated. A
simulation that redrew noise per half would make those draws independent, shrink
the null spread by roughly √200, and hand the design power it does not have.
The first draft of this script did exactly that and reported a null SD of 0.004
against the true 0.040 — a tenfold overstatement of power, caught by comparing
the two nulls.

| Measure | Permutation SD | Parametric SD | Ratio |
|---|---|---|---|
| S0 | 0.0403 | 0.0436 | 1.08 |
| S1 | 0.0419 | 0.0443 | 1.06 |
| S2 | 0.0407 | 0.0406 | 1.00 |
| S3 | 0.0404 | 0.0407 | 1.01 |

Every ratio is at or above 1, meaning the parametric arm carries *slightly more*
noise than reality. That direction is the safe one: the power column below is
**conservative**, an understatement rather than an overstatement.

### The generative story for `tau`

Sequencing in the literal sense — **the same total production, placed
differently.** Whatever the simulation adds to the high-leverage subset it
subtracts from the complement, so a team's overall efficiency is untouched and
only *where* the production landed moves. That is the only generative story
under which "sequencing is luck" is a meaningful hypothesis: if the total moved
too, the measure would be re-measuring offense, and S0 already does that.

### Power

500 replicates per cell. "Achieved mean r" is what the simulated `tau` actually
produced, printed beside the nominal target so any calibration slip is visible
rather than assumed away.

| Measure | true r = 0.05 | 0.08 | **0.10** | **0.12** | 0.20 | 0.30 |
|---|---|---|---|---|---|---|
| S0 overall EPA | 0.32 | 0.63 | **0.80** | **0.91** | 1.00 | 1.00 |
| S1 red-zone gap | 0.34 | 0.56 | **0.74** | **0.87** | 1.00 | 1.00 |
| S2 late-down gap | 0.32 | 0.62 | **0.78** | **0.92** | 1.00 | 1.00 |
| S3 WPA−EPA gap | 0.37 | 0.68 | **0.83** | **0.92** | 1.00 | 1.00 |

> **Minimum detectable correlation at 80% power: r ≈ 0.10**, on all four
> measures. The smallest effect this project has ever called real is document
> 02's r = 0.12, and every measure detects that with 87–92% power.

### What the detectable effects are, in football

`tau` restated in each measure's own units, at the r = 0.12 reference. A one-SD
team differs from average by this much.

| Measure | `tau` at r = 0.12 | In football |
|---|---|---|
| S1 red-zone gap | 0.0589 EPA per red-zone play | 0.56 EPA per game, **0.47 points per game** |
| S2 late-down gap | 0.0147 | **1.5 percentage points** of late-down success rate |
| S3 WPA−EPA gap | 0.0318 per game | **3.2 percentage points** of win probability per game |

Points are converted at the simulator's fitted `points_per_epa` = 0.8389.

**This is the first round in the project where the design resolves effects
smaller than anyone would care about.** A 0.47-points-per-game red-zone
sequencing skill is a real football effect and the design sees it 87% of the
time; effects half that size are still caught a third of the time. A null result
here will therefore mean something, which is exactly what document 04's Gate 2
failure taught us to check in advance.

---

## 6. Pre-registered gates and the decision rule

Committed before any result exists.

### Gate S-1 — the harness works *(positive control)*

**Statistic:** S0's split-half r.

**Pass rule:** exceeds the permutation null's 99th percentile (0.0917) by a wide
margin, and lands in the neighbourhood of document 02's `r = +0.519` for core
EPA differential.

This tests the machinery, not the hypothesis. Offensive EPA per play is known to
persist; if it comes back flat, the pooling, the grouping or the split logic is
broken and **no other number in the results document may be read.** It is the
cheapest check and the one most likely to catch an implementation bug — the same
role document 06's Gate 3 played.

### Gate S-2 — does the measure persist? *(one per sequencing measure)*

**Statistic:** the measure's split-half r, mean over 200 splits.

**Pass rule:** **r exceeds the permutation null's 95th percentile** for that
measure — 0.0703 for S1, 0.0689 for S2, 0.0648 for S3.

**Passing means the measure is a team property**, i.e. sequencing is skill for
that channel, and nothing about the simulator changes: skill is already counted
in `core`, and neutralizing it would be the error document 05 §2 built Gate A to
prevent.

**Failing means the measure did not persist**, and because §5 establishes that
this design detects r = 0.12 with 87–92% power, a failure is **evidence of
absence** rather than absence of evidence. That is a claim this project has not
been able to make before and it is only available because the power ran first.

### Gate S-3 — is a failure interpretable? *(the honesty gate)*

**Pass rule:** the measure's power at the reference effect (r = 0.12) is at
least 0.80, per §5's table.

All four measures clear this in advance. It is stated anyway, because if a
future re-run on a different subset drops power below 0.80, the correct report
is **"unresolvable"**, not "no sequencing skill". Document 05 §7's return-yardage
row is the worked example of what happens without this gate.

### The decision rule, committed in advance

| Outcome | Verdict | What changes |
|---|---|---|
| Gate S-1 fails | Harness broken | Nothing is reported. Fix and re-run. |
| S-2 passes for a measure | That sequencing channel is **skill** | **Nothing.** It already lives in `core`. No ledger row, no treatment-table change. |
| S-2 fails, S-3 passes | That channel is **luck** | Sequencing becomes a **separate reported measure** (below). **No ledger rows.** |
| S-2 fails, S-3 fails | **Unresolvable** | Reported with the power table that says why. No verdict. |

### Why a "luck" verdict does not produce ledger rows

This is the most important sentence in the document, and it is committed before
the result so it cannot be argued into afterwards.

**Sequencing has no branch point.** Document 05 §2's Gate A asks whether there
is a moment where the outcome is resolved by a mechanism outside either team's
control, conditional on the state both teams created. A loose ball on the turf
is such a moment. A third-down conversion is *not*: it is a played-out sequence
of blocking, route-running and tackling, with both teams exerting control
throughout. There is no coin to replace with its expectation, and no `swing`
value to book, because there are no two branches — there is a continuum of
outcomes produced by football.

Running document 05's Gate B arithmetic on a non-persistent sequencing measure
would produce a `w` near zero and neutralize almost all of it, which would erase
whatever real execution *is* in there. That is the same error the penalty row of
document 05 §3 exists to demonstrate. **Gate A stays intact, and this round
cannot breach it.**

So a luck verdict routes to a **second number reported alongside DTW%**, not
into it: a drive-level bootstrap answering *"how often does this set of drive
qualities produce a win?"* Its design is step 2 of this phase and is
pre-registered in §10 below, after these results are in, because its shape
depends on which channels came back as luck. Its relationship to DTW% is fixed
now:

> **DTW% and the sequencing measure are two complementary numbers about one
> game, and neither is a component of the other.** DTW% asks *"given the plays
> that happened, who deserved to win once the coin flips are set to their
> expectations?"* The sequencing measure asks *"given the drives this team
> produced, how often does that production win?"* The first holds sequencing
> fixed; the second holds the coin flips fixed. Multiplying or averaging them
> would double-count the game.

---

## 7. Known-defect register

| Defect | Evidence | Status |
|---|---|---|
| **The real statistics were printed during a smoke test of the power machinery, before this document was committed** | A timing/correctness check of `split_half_r` on real data ran before the thresholds were written | **Open, disclosed.** The threshold *rule* — the permutation null's 95th percentile — was fixed by the process law inherited from document 05 §7 and is computed by simulation, so knowing the observed values could not move it. Recording it is the only defence available; hiding it would be worse |
| S3 contrasts two nflverse models, both revised across the window | Document 01's EPA-vintage caveat, now doubled | **Open.** All analysis is within-season, which contains it |
| Late downs pool third and fourth | Fourth down alone would give ~15 attempts per team-season | **Accepted, by design (§2).** Fourth-down decision-making is document 09's question |
| The red-zone cut is a convention, not fitted | `yardline_100 <= 20` chosen from football usage | **Accepted, by design.** Fitting the cut would be goalpost-moving |
| Success rate is a coarse valuation | `success` is a binary EPA>0 indicator, so a 40-yard gain and a 1-yard gain on 3rd-and-1 are the same event | **Open.** S1 and S3 use continuous valuations, so the round is not relying on S2 alone |
| Half denominators vary across splits | A team-season with an odd game count splits 8/9 | **Accepted.** Both nulls carry the same imbalance, so it cannot favour either arm |
| Simulated success counts are Gaussian, not binomial | The parametric arm draws non-integer success sums | **Accepted.** At n ≥ 100 late-down plays per half the normal approximation is tight, and the permutation null — which uses real integers — agrees with it to within 0.2% |

---

## 8. Constants appendix

| Constant | Value | Where it lives |
|---|---|---|
| `RANDOM_SEED` | 20260817 | `research/10_sequencing_power.py`, `research/10_sequencing.py` |
| `N_SPLITS` | 200 | both scripts |
| Null / power replicates | 500 / 500 | `research/10_sequencing_power.py` |
| `RED_ZONE_YARDS` | 20 | both scripts |
| `LATE_DOWNS` | (3, 4) | both scripts |
| `MIN_GAMES` | 8 | both scripts |
| `REFERENCE_R` | 0.12 | this document §5, from document 02 |
| **Gate S-2 thresholds** | **S1 0.0703 / S2 0.0689 / S3 0.0648** | this document §6, from the permutation null |
| Gate S-1 threshold | 0.0917 (null 99th pct) | this document §6 |
| Gate S-3 threshold | power ≥ 0.80 at r = 0.12 | this document §6 |
| `wpa_per_epa` | 0.023497 | measured, §2 |
| `points_per_epa` | 0.8389 | `research/outputs/model_metadata.json` |
| Team-seasons / team-games | 320 / 5,522 | measured, §2 |

Results are written back into this document as §9.

---

## 9. Results

*Script: `research/10_sequencing.py`. Design, statistics and thresholds fixed by
§§1–8 above, committed at `9d6f11d` before this script produced a result.
Results in `research/outputs/10_sequencing.json`.*

### Gate outcomes, stated first

| Gate | Rule | Result |
|---|---|---|
| **S-1 — positive control** | S0 far above the null 99th pct (0.0917) | **PASS** — r = **+0.601** |
| **S-2 — S1 red-zone gap** | r > 0.0703 | **FAIL** — r = **−0.034** → **luck** |
| **S-2 — S2 late-down gap** | r > 0.0689 | **FAIL** — r = **+0.000** → **luck** |
| **S-2 — S3 WPA−EPA gap** | r > 0.0648 | **PASS** — r = **+0.180** → **skill** |
| **S-3 — interpretability** | power ≥ 0.80 at r = 0.12 | **PASS** on all three (0.87 / 0.92 / 0.92) |

320 team-seasons, 5,522 team-games, 343,543 scrimmage plays. Nothing below was
chosen after seeing a number.

### The full table

| Measure | Split-half r | 5th–95th pct | Gate S-2 threshold | Verdict |
|---|---|---|---|---|
| S0 overall offensive EPA/play *(control)* | **+0.601** | +0.560 … +0.641 | — | harness works |
| S1 red-zone gap | **−0.034** | −0.104 … +0.025 | 0.0703 | **luck** |
| S2 late-down gap | **+0.000** | −0.071 … +0.065 | 0.0689 | **luck** |
| S3 WPA−EPA gap | **+0.180** | +0.121 … +0.238 | 0.0648 | **skill** |

Document 02's own per-game-averaged estimator, run as the pre-registered
secondary so the numbers stay comparable to its published table, agrees on every
row: +0.597, −0.020, +0.013, +0.176.

### Gate S-1 — the harness works, and then some

Offensive EPA per play splits at **r = +0.601**, against document 02's +0.519 for
the core EPA *differential*. The gap between those two is expected and is worth
one sentence: document 02 measured a differential, which folds a team's defense
in alongside its offense and therefore carries two sources of noise. This measure
is offense-only, so it is the more reliable of the two. The harness is live.

### S1 and S2 — where production lands is not a team property

**Red-zone finishing does not persist. It is, if anything, mildly
anti-persistent** — a team that outperformed its own baseline inside the 20 in
one random half of its season was very slightly *worse* than its baseline in the
other half. −0.034 sits comfortably inside the null's body (null mean +0.000,
SD 0.042), so the negative sign is noise rather than a finding; what matters is
that it is nowhere near the 0.0703 the gate required.

**Late-down conversion over baseline is a dead flat zero.** +0.0005. It is
difficult to construct a cleaner null result.

These are the two claims most often made about "clutch" offenses, and this design
had **87% and 92% power** to detect an effect the size of the smallest one this
project has ever called real. So per §6's decision rule this is **evidence of
absence, not absence of evidence** — the first time in the project that
distinction has been available, and it is available only because the power
calculation ran first.

What it does *not* say: that red-zone offense is unimportant, or that teams do
not differ in red-zone EPA. They do — but they differ there in the same
proportion they differ everywhere else. The **gap** is what carries no signal.
A good red-zone offense is a good offense, measured in a small window.

### S3 — leverage timing is real, and it is the round's surprise

**r = +0.180**, with a 5th–95th band of +0.121 to +0.238 that never approaches
the 0.0648 threshold. This is larger than any of document 02's middle three
(interceptions +0.164, field goals +0.145, penalties +0.121) and it sits on a
channel nobody in this project had measured.

Restated in football: teams differ, repeatably, in how much win probability they
extract from a fixed amount of expected-points production. Two offenses can
generate identical EPA and one of them converts it into meaningfully more
winning, because its production lands in moments that decide games.

### What S3 is actually measuring — exploratory, not pre-registered

**Labelled exploratory because it was added after seeing the S-2 result**, per
the precedent document 07 set for its DTW% arm. It changes no verdict: §6's
decision rule already says an S-2 pass changes nothing about the simulator. It
changes only what may honestly be *said* about the number.

The obvious rival explanation is dull: win probability moves fast in a one-score
game and barely at all once a game is decided, so a team whose games stay close
converts EPA into WPA at a higher rate than the league slope **regardless of when
inside the game its production landed.** "Plays close games" is itself a stable
team property, so it would manufacture exactly this persistence with no timing
skill existing at all.

| Check | Result | Reading |
|---|---|---|
| corr(season S3, season S0 offensive quality) | **−0.008** | S3 is orthogonal to being good at offense. It is not a disguised quality measure |
| corr(season S3, mean \|score differential\|) | **−0.205** | The game-state confound is real but modest — closer games do give higher S3 |
| **S3 on competitive plays only** (\|score diff\| ≤ 8, 294,976 plays, 65% of the sample, slope refit to 0.03086) | **r = +0.144** | **It survives.** |

The third row is the decisive one. Restricting to one-score game states removes
the blowout-versus-nailbiter distinction by construction, and refitting the
slope on that subset removes the constant offset that would otherwise
manufacture a correlation. Persistence falls from +0.180 to **+0.144** — still
more than double the 0.0648 threshold, and still above document 02's penalty
figure.

**So roughly a fifth of S3's persistence was game state and four-fifths was not.**
Leverage timing is a real, repeatable team property inside competitive football.
That is a genuine finding and it is also, deliberately, a finding that changes
nothing the simulator does: it is skill, it already lives in `core`, and
document 05's Gate A would have kept it out of the ledger even if it had not.

### What this changes

1. **Nothing in document 05's treatment table.** No sequencing channel earns a
   ledger row, because none of them has a branch point. The rule was committed
   in §6 before the results existed and it holds in both directions.
2. **Red-zone and late-down finishing become a reported measure, not an
   adjustment.** They are luck, on a properly powered test, and §10 pre-registers
   the drive-outcome resampling that reports them.
3. **The 70% "core" block is no longer a single indivisible thing.** Two of its
   three tested sub-channels are noise. That does not shrink core's *variance
   share* — the production is still real — but it does mean a meaningful slice
   of what looks like offensive quality in a single game is placement rather than
   ability.
4. **S3 is a candidate for a future product surface**, and explicitly not for the
   simulator. A "this team converts production into wins" number is interesting
   on its own terms; folding it into DTW% would be double-counting.

### Defects added by this round

| Defect | Evidence | Status |
|---|---|---|
| S3's mechanism is unidentified | Persistence survives the game-state control, but *why* teams differ — coaching, quarterback, situational scheme — is untested | **Open.** A crossed QB × coach model is the natural next step and is out of Phase 3's scope |
| The competitive-play control is exploratory | Added after seeing the result | **Open, labelled.** Excluded from every gate |
| S1's point estimate is negative | −0.034, inside the null's body | **Accepted.** Reported as a null result, not as anti-persistence |
| Sequencing is tested only at the team-season grain | A team changes coordinators mid-season | **Open.** Same limitation document 04 recorded for interceptions |

---

## 10. The drive-outcome resampling — pre-registered

*Written after §9's results and **before `research/11_drive_bootstrap.py` exists**.
This is the "separate reported measure" §6 committed to in advance; its shape,
but not its existence, waited on which channels came back as luck.*

### What it is

S1 and S2 say that **where a team's production landed relative to the red zone
and relative to third down is noise.** So the measure holds fixed how far each
drive advanced the ball — which is production, and skill — and re-draws whether
that advance became points.

```
for each offensive drive:
    depth   = min(yardline_100 at the SNAP of any play in the drive)   # observed
    points* ~ League P(drive points | depth bin)                       # resampled

DQ margin = sum(home points*) - sum(away points*) + non-offensive scoring (observed)
DQW%      = P(DQ margin > 0) over replicates
```

**Depth is taken at the snap, never at the end of the play**, and that is the
single most important line in this section. A touchdown drive necessarily ends at
`yardline_100 = 0`, so defining depth by where the drive *finished* would encode
the outcome into the conditioning variable and make the resampling vacuous.
Taking the deepest snap makes a drive that scored from the 3 and a drive that was
stopped at the 3 exchangeable, which is exactly the comparison S1's null result
licenses.

### What is resampled and what is not

| Quantity | Treatment | Why |
|---|---|---|
| Depth reached by each drive | **Observed, never resampled** | It is the production. That is the skill S0 confirmed persists |
| Offensive points given depth | **Resampled** at the league conditional rate | S1 says finishing relative to baseline is noise |
| Non-offensive scoring (pick-sixes, return TDs, safeties) | **Observed** | Interceptions and returns are not neutralized anywhere in this project (document 05 §3) |
| Number of drives, drive start position | **Observed** | Both are consequences of the two teams' play, not of a branch |
| Play calling, drive continuation, clock | **Not modelled at all** | `CLAUDE.md` puts full play-level re-simulation out of scope, and this design respects that: no play is ever replayed |

### Gates

**Gate D-1 — the resampling is unbiased on the league.**
*Pass rule:* the mean resampled offensive points per drive, pooled over all
drives in the sample, is within 0.01 points of the observed mean. This is an
identity check, not a hypothesis: resampling from the league's own conditional
distribution must reproduce the league's own mean. Failure means the conditional
table or the depth definition is wrong.

**Gate D-2 — non-inferiority on the rematch harness.**
*Incumbent:* game 1's actual margin, in the identical logistic model document 06
§1 specifies.
*Statistic:* mean out-of-fold log loss, drive-quality margin minus actual margin,
paired at the rematch pair, 10-fold CV, `random_seed = 20260817`.
*Pass rule:* the **upper** bound of the 95% CI lies below **+0.010 log loss** —
the same margin document 06 §4 pre-registered, on the same 531 pairs, with the
same measured false-alarm rate of 0.008 and 88.9% power against a predictor that
is 15% noise.

The **drive-quality margin** is the primary predictor and **DQW%** the secondary,
in that order, because document 07 measured that a probability compressed toward
0 and 1 discards margin information and carries a larger standard error. Fixing
the order now prevents choosing the better-looking arm later.

**No superiority gate, and the reason is on record.** Document 06 §3 computed the
minimum detectable luck share at 80% power as **52% of margin variance**. This
resampling removes more variance than the fumble-plus-FG neutralization does, but
nothing this project can build reaches 52%. A superiority test would fail
regardless of quality and its failure would carry no information.

**Gate D-3 — DQW% and DTW% are distinct, reported descriptively.**
*Statistic:* the correlation between DQW% and DTW% across all games, and the
count of games where they disagree about the winner.
*No pass rule.* This is reported so a reader can see for themselves that the two
numbers are answering different questions. If they were to correlate above ~0.95
the honest conclusion would be that reporting both is redundant, and that
conclusion is admissible in advance.

### Their relationship, restated for the product

**Two complementary numbers about one game, and neither is a component of the
other.**

- **DTW%** — *"once the coin flips are set to their expectations, who deserved to
  win?"* Holds the sequencing fixed; re-draws the fumbles and the field goals.
- **DQW%** — *"given the drives this team produced, how often does that
  production win?"* Holds the coin flips fixed; re-draws the finishing.

They are not to be multiplied, averaged, or combined into a single index. Doing
so would count the same game twice through two different lenses, and the
per-component ledger — the thing that makes DTW% auditable — has no equivalent
for a quantity built by averaging two bootstraps.

### Known defects, stated before the build

| Defect | Evidence | Status |
|---|---|---|
| Conditioning on depth understates offenses that score from distance | A 45-yard touchdown has depth 45, where the league TD rate is low, so the drive is scored as fortunate | **Open, and the largest defect.** It conflates explosive-play ability with finishing luck. Direction is stated wherever the measure is reported |
| S1 licenses the resampling *inside* the red zone; outside it the extension is an assumption | S1 is defined on `yardline_100 <= 20` | **Open, stated.** Reported with a red-zone-drives-only variant so the licensed part is separable |
| Touchdowns are valued at the league average points-per-TD | Two-point conversions and missed extra points are folded into one constant | **Accepted.** Extra points are document 09's question, and a per-drive XP model would be a larger build than the measure warrants |
| Drives are resampled independently | Two drives in one game are drawn independently | **Accepted.** Same reasoning as document 05 §5's row on simultaneous luck events |
| The game being adjudicated is inside the league conditional table | ~10 drives of ~22,000 | **Open, bounded** at O(1/n), an order of magnitude smaller than the FG model's contamination |
