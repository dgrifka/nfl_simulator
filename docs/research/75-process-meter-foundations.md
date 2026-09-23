# 75 — The process meter: foundations

What the deserve-to-win process meter is, why each piece is there, and every
number anyone would have to defend. This is the foundations document for the
model of record, the arm named `foldn_take_sr` and shipped as library version
2.1.0. It merges three research documents written across rounds 6, 19 and 24 of
the process thread into one record of the model as it stands.

Every number below is pinned by a check that runs:
`tests/test_process_meter_stage0.py` reproduces the six reference percents and
the two losses through the library's own scoring seam;
`nfl_simulator.process_meter.fit` re-measures the weights, `margin_sd` and
`sigma_once` on every refit and refuses to write a file that misses any of them;
and `nfl_simulator.process_meter.weights.WEIGHTS_PIN` refuses to load one. Re-run
stage 0 after any edit here.

Minus signs are ASCII hyphens throughout. The research this document summarises
was run in the private orchestration repo's research record, rounds 1 to 24; the
numbers, the code and the checks are all here.

## 1. One-page story

**The question.** Given the plays each team ran in one game, how many points does
play like that usually score, and how often does the home side's play win? It is
a verdict on a game already played, not a prediction. The gap between actual and
likely points is what this project calls luck.

**How it answers.** Each team in each game gets two numbers from its own plays:

- **its success rate** — the share of its plays that gained expected points,
  which is nflverse's `success` flag, a play whose expected points added (EPA) is
  **above zero**. A takeaway counts as one more play for the team that took the
  ball, successful when the offense lost expected points on it;
- **its yards per play** — its own yards plus the yards it returned a takeaway
  for, divided by its own plays alone.

With N own plays, K own successes, T takeaways (T_s of them successful for the
taker), Y own yards and R credited return yards:

    own_success_rate_for = (K + T_s) / (N + T)
    own_ypp_for          = (Y + R) / N

One ordinary least squares fit (OLS, a straight-line regression) on 2016-2023
team-games turns the two into likely points, with weights -16.829 · +35.126 ·
+4.178 for the intercept, the success rate and yards per play. A team's likely
points allowed are the opponent's likely points scored. The **deserved margin**
is home likely points minus away likely points, and the meter is the share of
40,000 draws of that margin that a Normal at scale 5.4528 puts above zero.

**Five things to hold onto.**

1. **The takeaway is folded asymmetrically, and that is the whole of v2.1.** A
   takeaway is one more play in the success rate's denominator but not in the
   yards denominator: a team is not charged a play in its yards per play for
   intercepting the ball. The two are equal only on a team-game with no takeaway.
2. **The arm was adopted on its definition, not on a forecast gain.** Against the
   previous definition the next-game log loss did not move: -0.0007 with a 95%
   interval of [-0.0018, +0.0004], which does not clear zero. It describes the
   finished game markedly better, Brier 0.1503 to 0.1439 and log loss 0.4554 to
   0.4380, and §7 row 3 carries that gain as a stated caution rather than as
   evidence.
3. **Uncertainty comes from three places and no more**: the weights (a Normal
   from the OLS fit), each team's true success rate (a Beta on its counts) and
   each team's true yards per play (a Normal on its own plays). Every posterior
   is closed form, so there is no sampler to diagnose.
4. **The number of record is a draw average, not a formula.** Two constants make
   it reproducible: 40,000 draws and a per-game seed. A game scored alone reads
   exactly what it reads inside a week's batch.
5. **The meter is never held out.** It scores its own game. Brier and log loss
   are descriptive numbers about a verdict, and a verdict on a finished game is
   allowed to be miscalibrated, because luck is exactly what it strips.

**Statistic convention.** Forecast numbers are log loss of a season-to-date
rating on the next game, lower is better; a gap is arm minus reference from a
paired bootstrap (the same games re-drawn for both), 4,000 resamples, seed 0,
mean [95% bounds], negative is better. Meter numbers (Brier, log loss, shares,
calibration) are descriptive: the meter judges its own game. An 89% interval is
the 5.5th to 94.5th percentile.

## 2. Data

**Grain.** One row per team per regular-season game, from the offense's side.

**Source.** nflverse play-by-play through the nflreadpy cache
(`$NFL_SIM_DATA_DIR/pbp/pbp_<season>.parquet`), finals from `schedules.parquet`.
**No charting data of any kind.** The previous model read FTN charting for
dropped interceptions and receiver drops; the process meter reads none, which is
why its coverage does not depend on the season.

**Filters** (`nfl_simulator.process_meter.plays.load_plays`): `play_type` is
`pass` or `run`; `epa`, `posteam` and `defteam` are present; two-point tries out;
regular season only; relocated franchises remapped (SD to LAC, OAK to LV, STL to
LA). **All plays are kept**: there is no win-probability band, so garbage time
counts.

**The takeaway rows.** An interception or a lost fumble by the opponent, with the
taking team owning the row. Its EPA sign decides whether it is a success for the
taker: `epa < 0` for the offense is a success, and EPA exactly 0 or missing is a
failure — there is no tiebreak. The credited return yards are a pick's
`return_yards`, or a lost fumble's `fumble_recovery_1_yards` when the defense
recovered first and `fumble_recovery_2_yards` when it recovered second. A fumble
out of the end zone has no recovery and no yards; a negative return stays
negative; a pick the interceptor fumbles back carries the pick's return.

**Windows.** Fit on 2016-2023: 4,190 team-games over 2,095 games. The gate window
is the 543 decided 2024-2025 regular-season games (ties dropped) with both team
rows present. Population 2016-2026: 5,282 team-games, 3,664 of them with a
takeaway and 1,618 without, 6,574 takeaways, 99.4% with EPA below zero for the
offense.

**Facts to defend by name.**

- **No opponent columns.** A team's defense enters the model only as the
  opponent's offensive output. Opponent adjustment was measured and turned out to
  be shrinkage, so it is not carried.
- **Garbage time counts.** The win-probability band was measured and not carried.
- **Special teams never enter.** Field goals, punts, kickoffs and returns other
  than a takeaway's are outside the model entirely.
- **Penalties never enter.** A play wiped out by a flag is not in the frame.
- **Field position never enters.** Two teams with the same rate and the same
  yards per play get the same likely points wherever they started their drives.

## 3. DAG

```
one game's plays ─┬─> home (K, N, Y, T, T_s, R) ─┐
                  └─> away (K, N, Y, T, T_s, R) ─┤
                                                 │
2016-2023 team-games ─> [OLS on likely points] ──┤   beta ~ Normal(beta-hat, Sigma-hat)
                              ═══ cut ═══        │
                                                 v
                          p ~ Beta(K + T_s + 1, N - K + T - T_s + 1)      narrows by N + T
                          y ~ Normal(y-bar, s_y / sqrt(N))                narrows by N
                                                 │
        margin draw = b_rate (p_home - p_away) + b_ypp (y_home - y_away)
                                                 │
2016-2023 games ─> [OLS on the margin] ─> margin_sd ═ cut ═> sigma_once = 5.4528
                                                 │
                    p_home = mean over 40,000 draws of Phi(margin draw / sigma_once)
```

**Where inference is cut.** (1) The weights are fit once on 2016-2023 and never
see the game being scored. (2) A game's production never feeds back into the
weights. (3) The residual scale comes from a separate regression — of the actual
home margin on the two home-minus-away input differences — and enters as one
number. Cutting it twice is deliberate: it is what makes the verdict on a game
independent of the order games are scored in.

**Two sample sizes, not one.** This is the seam that Model v2.1 makes explicit.
The rate is taken over N + T plays and the yards mean over N, so the posterior
frame carries a fifth column, `n_y`, and the yards draw narrows by its square
root. The previous definition folded the difference into the yards SD instead,
multiplying it by sqrt((N + T) / N), which is the same arithmetic wearing a
disguise:

    own y_sd / sqrt(N)  ==  (own y_sd * sqrt((N + T) / N)) / sqrt(N + T)

A posterior with no `n_y` fills it with `n`, so anything fit before the column
existed draws exactly what it drew before. The identity is pinned to 1e-12 in
`tests/test_process_meter_record.py`.

**No feedback loops.**

## 4. Each stage, prior by prior

### 4a. The likely-points regression

| site | prior | value | plain-language meaning |
|---|---|---|---|
| intercept | flat (OLS) | -16.829 | points at zero success and zero yards; an extrapolation, meaningless alone |
| own success rate | flat (OLS) | +35.126 per unit | about 0.35 points per percentage point of success rate |
| own yards per play | flat (OLS) | +4.178 per yard | points per extra yard per play at a fixed success rate |
| residual | plug-in | R-squared 0.609 in sample | the coefficient covariance is built from it |

Full values: -16.829288585 · 35.126158543 · 4.177796953. Training window
2016-2023, 4,190 team-game rows over 2,095 games.

**Identification.** The two inputs move together but not in lockstep. The
intercept pins the points scale; nothing else does. The arm leans a little harder
on yards and a little less on the success rate than the previous definition did,
which is what a yards number that no longer shrinks when a team takes the ball
away should do.

**The all-zero pads.** Each own-side input gets an `_against` column of zeros, so
least squares gives it a coefficient of exactly 0 and the design stays intercept
plus two weights. This is a bookkeeping convention, not a model term.

### 4b. The weights posterior

`beta ~ Normal(beta-hat, sigma-squared (X'X)^-1)`, with sigma-squared carrying
n - 3 degrees of freedom. A flat prior, so the mean is the least-squares fit
exactly; `nfl_simulator.process_meter.fit` checks the two against each other on
every refit and stops at 5e-4 points.

### 4c. The margin regression (scale only)

The actual home margin on the two home-minus-away input differences. Only its
residual standard deviation leaves this stage: `margin_sd` = 8.875 (full
8.8753920693). The two teams' errors are therefore never assumed independent,
which matters: an independent-teams version read games close that were not,
because scoring surprises correlate across the two sides of one game.

### 4d. The production posteriors

| site | prior | narrows by | meaning |
|---|---|---|---|
| `p` | `Beta(K + T_s + 1, N - K + T - T_s + 1)` | N + T | the team's true within-game success rate |
| `y` | `Normal(y-bar, s_y / sqrt(n_y))` | N | its true within-game yards per play |

`s_y` is the standard deviation of yards gained over the team's own plays, with
no scale factor folded into it. A team-game with one play has no spread, so its
`s_y` is 0.

The Beta's mean is the smoothed (k + 1) / (n + 2), not the raw k / n, so the
draws are not centred on the plug-in margin. The offset is a modelling choice
rather than noise, and `tests/test_process_meter_score.py` subtracts it and
checks that what is left is Monte Carlo error.

### 4e. The residual scale, `sigma_once`

`margin_sd` was fit to realised rates, so it already holds their sampling noise;
the draws in 4d add that noise a second time. The scale of record takes it back
out, pooled over the gate window:

    sigma_once = sqrt(margin_sd^2 - median per-game margin-draw variance)
               = sqrt(8.875^2 - 49.0324485506)
               = 5.4528

**Convention 1 (the residual SD the scale is measured from).** `sigma_once` is
measured from the **published, three-decimal** `margin_sd` of 8.875, which gives
5.4528136269. The full 8.8753920693 would give 5.4534517357 instead, a 6.4e-4
move. The published value is the convention of record for two reasons: every
other constant gated alongside it was computed at 5.4528136269, and the rounded
value is the one the artifact carries, so an artifact's scale can be recomputed
from the artifact. `nfl_simulator.process_meter.fit` prints both readings on
every run and `tests/test_process_meter_record.py` pins both. Whether a future
version should measure from the full residual SD and re-gate the rest is open.

The pooled form is never floored. A per-game form exists and floors at 5.0
points; it was measured and not carried.

### 4f. The meter

`p_home` = the mean over 40,000 draws of `Phi(margin draw / 5.4528)`, where `Phi`
is the standard Normal curve. Display: a whole percent where the number is shown,
one decimal where it is recorded.

## 5. Inference

Engine: `numpy.linalg.lstsq` for both regressions and `numpy.random.Generator`
for the draws. No sampler, no Markov chain, nothing to diagnose for convergence.
These constants are **rulings, not defaults**:

| constant | ruling | evidence |
|---|---|---|
| `N_DRAWS_RECORD` = 40,000 | the draw count of record | the smallest of 4,000 / 16,000 / 40,000 meeting the pre-registered conditions; 4,000 and 16,000 fail seed stability. Doc 76 §3 |
| `SEED_OF_RECORD` = 0, SHA-256 with the game id | one generator per game | a game's percent then depends only on (seed, game id, draw count), so it reads the same alone and in a batch, to 0.0e+00 against a 1e-12 line. Doc 76 §2 |
| `SIGMA_ONCE_RECORD` = 5.4528 | the pooled scale, held as a constant | the scale question closed at the pooled form; the per-game form floored 151 of 543 games. Re-measured on every refit |
| `margin_sd` from a direct margin regression | the two teams' errors are never treated as independent | an independent-teams version put games near 53% that the home team won 54 times out of 54 |
| all plays, no win-probability band | the carried filter | the band was measured and moves the meter; it was not carried |
| own-side inputs only | the arm of record | opponent columns were measured and are not load-bearing |
| the weights as a committed artifact | the library never refits at score time | a refit at score time is a second model; `WEIGHTS_PIN` refuses anything but the fit that passed the gate |

## 6. Validation

- **The adoption rule.** A definition change is adopted when the next-game log
  loss of a season-to-date rating built on it does not get worse: the 95% upper
  bound on the gap must sit below zero for replacement, and an interval that
  straddles zero reads ADOPTABLE. This arm read ADOPTABLE at -0.0007
  [-0.0018, +0.0004], and it was adopted on the definition.
- **The same-game meter** (reported, never gated): Brier 0.1436 and log loss
  0.4376 on the 543, against 0.1503 and 0.4554 for the previous definition.
  Expected calibration error is measured and printed beside them.
- **Stage 0.** Every number of record reproduces through the library's own
  scoring seam: the six reference percents at four decimals, then the two losses.
  `tests/test_process_meter_stage0.py`.
- **The refit gate.** `nfl_simulator.process_meter.fit` re-measures the weights,
  the training row count, `margin_sd` and `sigma_once`, and refuses to write if
  any of them moved. A refit cannot ship the old scale on new weights.
- **Noise rules.** Paired bootstrap over games, 4,000 resamples, seed 0 of
  record, seeds 0-9 counted for a seed read.

**What the stack does NOT validate.**

- **The meter is never held out.** It scores the game it was built from.
- **There is no coverage check.** Nothing tests that an 89% interval holds the
  truth 89% of the time, because on a finished game there is no truth to hold.
- **Process against a settled scoreboard.** Once both ratings get the same
  pull toward the league mean, the process edge sits inside the noise. A forecast
  pass does not show the process rating beats a settled scoreboard.
- **Anything the model leaves out.** Nothing checks whether special teams, field
  position or penalties would improve the verdict; they are out by decision, and
  §7 says so.

## 7. Known-defect register

| # | defect | evidence | status |
|---|---|---|---|
| 1 | Identical rates from 45 plays and from 75 plays used to get the same number | the plug-in formula reads raw rates | **CLOSED.** The number of record is a draw average, and the draws narrow with the play count |
| 2 | The percent used to move across random seeds | median 1.15 points at 4,000 draws, one-decimal flips on 538 of 543 | **CLOSED.** Per-game seeding and 40,000 draws: median seed move 0.36 points, largest 0.91 |
| 3 | The description gain is large for a definition change | Brier -0.0064 [-0.0079, -0.0049] on the 543, larger than the previous definition's whole adoption gain | **STATED CAUTION.** It does not survive a forecast-only framing, and the forecast rule says so. Adopted on the definition, not on this number |
| 4 | The percent leans toward the scoreboard winner | 69.6% of games move toward the team that won | **ACCEPTED, reported.** The description gain leans the same way |
| 5 | `sigma_once` is measured from the published, rounded `margin_sd` | 8.875 gives 5.4528136269; the full 8.8753920693 gives 5.4534517357 | **CONVENTION, §4e.** Stated rather than silently carried; the fit prints both on every run |
| 6 | One residual scale for every game | `sigma_once` is one number; a 13-possession game and a 9-possession game get the same allowance | **CLOSED at the pooled form.** The per-game form floored 151 of 543 |
| 7 | The forecast reading turns on bounds thinner than the seed wobble | the halves of a previous split each cleared zero by less than the seed noise | **KNOWN.** The plain reading is carried rather than the bound |
| 8 | Garbage-time plays count | the win-probability band moves the meter | **CLOSED:** all plays is the carried filter |
| 9 | Special teams, field position and penalties are outside the model | by construction | **OPEN by decision.** Each was considered and left out; none is queued |

## 8. Constants appendix

| constant | value | where it lives |
|---|---|---|
| points weights | -16.829 · +35.126 · +4.178 (full -16.829288585 · 35.126158543 · 4.177796953) | `weights.json`, `WEIGHTS_PIN` |
| `margin_sd` | 8.875 (full 8.8753920693) | `weights.json`, `WEIGHTS_PIN` |
| `sigma_once` (pooled) | 5.4528 (full 5.4528136269; from the full `margin_sd`, 5.4534517357) | `record.SIGMA_ONCE_RECORD`, re-measured every refit |
| median per-game margin-draw variance | 49.0324485506 | measured over the 543, recorded in `weights.json` |
| noise-once floor | 5.0 | `posterior.NOISE_ONCE_FLOOR` (per-game form only) |
| draws per game | 40,000 | `record.N_DRAWS_RECORD` |
| seed of record | 0, SHA-256 with the game id | `record.SEED_OF_RECORD`, `record.game_seed` |
| training window | 2016-2023, 4,190 team-games over 2,095 games | `plays.TRAIN_SEASONS` |
| gate window | 2024-2025, 543 decided games | `fit.GATE_SEASONS`, `fit.EXPECTED_GATE_GAMES` |
| win-probability band (not carried) | 0.05, 0.95 | `plays.WP_BAND` |
| explosive-play lines (not in the model) | 20 pass, 12 run | `plays.EXPLOSIVE_PASS_YARDS`, `EXPLOSIVE_RUN_YARDS` |
| interval percentiles | 5.5, 94.5 | `posterior.INTERVAL` |
| readiness play floor | 25 | `readiness.MIN_SCRIMMAGE_PLAYS` |
| the six reference percents | NE_SEA 0.9121 · SF_LA 0.1707 · CIN_CLE 0.7488 · DAL_PHI 0.5363 · CHI_LV 0.8913 · CAR_TB 0.2835 | `tests/test_process_meter_stage0.py` |
| meter Brier / log loss (543) | 0.1436 / 0.4376 | `tests/test_process_meter_stage0.py` |

### The two names

One version, two spellings, and they move together or the artifact and the tag
disagree.

| what | value |
|---|---|
| arm | `foldn_take_sr` |
| library tag | `v2.1.0` |

The library's version number tracks the model label rather than the package's
own history. Version 2.0 was the two-week interim definition that preceded the
takeaway-fold fix and was never released as a library, so `v2.0.0` is never cut.
