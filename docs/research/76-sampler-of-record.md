# 76 — The sampler of record: the seed, the draw count, the scale

Why the published number is a draw average rather than a formula, and why the
three constants behind it are rulings. This document is the record of the change
that moved the number of record from a closed-form percent onto the draws, and of
the two constants that gate chose. Its companion is
[75 — The process meter: foundations](75-process-meter-foundations.md), which
this document numbers against.

Statistic convention is doc 75 §1's. The research it summarises was run in the
private orchestration repo's research record, rounds 1 to 24. The gate below was
written and committed **before** the arms were built, which is the point of
writing it down at all.

## 1. What changed, and why it was worth changing

Before this change the published percent was a formula: the plug-in likely-points
margin divided by the fit's residual standard deviation, run through the standard
Normal curve. That number has two defects with the same root, which is that it
reads each team's success rate and yards per play as if they were known exactly.

- **A team with 45 plays and a team with 75 plays at the same rates got the same
  number and the same certainty.** Nothing in the formula knows how much of a
  game the rates were measured over.
- **The formula has no way to say it is unsure.** Its only interval came from
  refitting the weights on resampled training games, which measures how well the
  weights are known — not how close the meter is to calling this game.

The sampler answers both, because it draws each team's true rate from a Beta on
that team's own counts and its true yards per play from a Normal that narrows
with that team's own plays. The draws narrow with the play count by construction.

The cost of the change is that a draw average has a random seed, and a number
that moves when you re-run it is not publishable. §2 and §3 are the two rulings
that fix that.

## 2. The seed: one generator per game

The obvious implementation runs one random-number generator down a batch of
games. That makes a game's draws depend on its row position and on which games sit
beside it, so the same game scored on the night it finished and scored again
inside a full week's batch reads differently. For a published verdict that is not
acceptable.

The ruling is **one generator per game**, seeded from a constant and a stable hash
of the game id:

    game_seed(game_id, seed) = the first 8 bytes of sha256(f"{seed}:{game_id}")

`hashlib.sha256` and never Python's built-in `hash()`, which is salted per process
and so gives a different answer in a different run. The seed offset folds into the
same digest, so seed offsets 0 through 9 give one game ten unrelated streams for a
seed read.

`SEED_OF_RECORD` is **0**.

The weights are drawn per game too, off the game's own generator, where a batch
implementation would draw one set for every game. Same expectation, no cross-game
correlation, and a game scored alone reads the same as a game scored in a batch.

**The invariance check.** Five gate games scored alone, together in a batch, and
in the reversed batch agree to **0.0e+00** — bit for bit, against a 1e-12 line.
`tests/test_process_meter_record.py` and `tests/test_process_meter_score.py` carry
the same check.

## 3. The draw count: the gate and what it returned

**Pre-registered rule.** `N_DRAWS_RECORD` is the smallest of 4,000, 16,000 and
40,000 that meets all three conditions below on the 543 gate games. If none does,
the seed ruling is adopted on its own and the draw count is a separate decision.

| condition | what it asks | line |
|---|---|---|
| **A, fidelity** | the sampler does not cost accuracy against the formula it replaces | Brier gap point estimate below +0.0005 and upper 95% bound below +0.0020; log-loss gap upper bound below +0.0050 |
| **B, seed stability** | the published number is the game, not the draw | over seed offsets 0-9, a game's largest minus smallest percent: median at or below 0.50 points, largest at or below 1.50 |
| **C, flag stability** | the count of flagged games does not depend on the seed | across seed offsets 0-9, largest minus smallest count at or below 3 games of 543 |

**What it returned: PASS at 40,000.**

| arm | draws | seeding | median seed move | largest | A | B | C | wall-clock |
|---|---|---|---|---|---|---|---|---|
| formula | — | none | 0.00 | 0.00 | control | control | control | — |
| batch, 4,000 | 4,000 | one generator per batch | 1.151 | 2.875 | control | control | control | — |
| per game, 4,000 | 4,000 | per game | 1.142 | 2.908 | PASS | **FAIL** | PASS | 2.4 s |
| per game, 16,000 | 16,000 | per game | 0.553 | 1.669 | PASS | **FAIL** | PASS | 8.7 s |
| per game, 40,000 | 40,000 | per game | 0.358 | 0.910 | PASS | PASS | PASS | 19.5 s |

Condition A is met at every count — at 40,000 the Brier gap against the formula is
-0.0001 [-0.0013, +0.0009] and the log-loss gap -0.0000 [-0.0036, +0.0034], so the
sampler costs nothing in accuracy. Condition C is met at every count; per-game
seeding alone already narrows the flagged-count range from 3 games to 2.

Condition B is what separates them, and 16,000 missed **both** halves rather than
the median by a hair. The median seed move falls as one over the square root of
the draw count almost exactly — 1.142 at 4,000 predicts 0.571 at 16,000 (measured
0.553) and 0.361 at 40,000 (measured 0.358) — which confirms that the wobble is
Monte Carlo noise, the randomness of drawing a finite sample, and not something
structural in the draws. 40,000 is the first count that earns the publication.

50,000 was considered and declined: another tenth of a point off the median move
is not a reason to reopen a pre-registered rule.

## 4. The scale

The draws give a deserved-margin distribution per game. Turning that into a share
needs a residual: the allowance for the scoreboard wandering away from the process
for reasons the model does not carry.

`margin_sd` is the wrong number to use directly. It is the residual SD of a
regression of the actual home margin on the two home-minus-away input
differences, and it was fit to **realised** rates — so it already contains the
sampling noise of those rates. The draws in doc 75 §4d add that same noise a
second time. Counting it once means taking it back out:

    sigma_once = sqrt(margin_sd^2 - median per-game margin-draw variance)

**On the arm of record this lands at 5.4528.** The previous definition's scale was
5.8741, measured the same way on the same 543 games; the arm's percents are
sharper by construction because its residual is smaller, which is why a version
change has to re-pin the scale rather than inherit it.

Two forms of the subtraction were measured. The **pooled** form takes the median
variance over the window and gives one number for every game; it is never floored
and it is the form of record. The **per-game** form subtracts each game's own draw
variance and floors at 5.0 points; it floored 151 of the 543, and on the games it
did not floor it landed on the pooled number anyway. The scale question closed at
the pooled form.

**The convention, stated.** `sigma_once` is measured from the **published,
three-decimal** `margin_sd` of 8.875, giving 5.4528136269. The full 8.8753920693
would give 5.4534517357 instead, a 6.4e-4 move. The published value is the
convention of record, for the reasons in doc 75 §4e; the fit prints both readings
on every run and the test suite pins both. This is a convention, not a defect, and
the reason it is written down is that an earlier scale was measured from the full
residual SD, so the convention changed between rounds without anyone saying so.

`nfl_simulator.process_meter.fit` re-measures the scale on the arm's own draws on
every refit and refuses to write the artifact if it has moved, so a refit cannot
ship the old constant on new weights.

## 5. The explicit yards sample size

The sampler reads one row per team-game and draws

    rate ~ Beta(k + 1, n - k + 1)      yards ~ Normal(y_mean, y_sd / sqrt(n_y))

The arm of record needs **two** sample sizes, because its two inputs are taken
over different plays: the rate draw narrows by N + T, the plays its folded rate is
taken over, and the yards draw by N, the plays its mean is taken over.

An earlier round could not change the draw code without breaking the pins that
depended on its stream, so it folded the second size into the first by multiplying
`y_sd` by sqrt((N + T) / N). The seam is explicit now: the posterior emits `n` =
N + T, `n_y` = N and the team's own `y_sd`, untouched. The two are the same
arithmetic,

    own y_sd / sqrt(N)  ==  (own y_sd * sqrt((N + T) / N)) / sqrt(N + T)

and `tests/test_process_meter_record.py` pins that identity to 1e-12 on a
synthetic frame. A posterior that does not carry `n_y` fills it with `n`, so
anything built before the column existed draws exactly what it drew before, and
`tests/test_process_meter_record.py` pins that too — against a second
implementation of the draw stream written out longhand, rather than against a
recorded hash. A byte hash of these draws would be a hash of the platform: the
multivariate Normal factorises its covariance through LAPACK, and macOS and CI's
Linux runner do not agree bit for bit. Two implementations on one machine pin the
draw order and the two sample sizes exactly, and travel.

## 6. What the sampler is measured on

On the 543 decided 2024-2025 regular-season games, scoring each game's own
outcome:

| number | value |
|---|---|
| Brier | 0.1436 |
| log loss | 0.4376 |
| expected calibration error | measured and reported, never gated |

The reliability read is "too timid" at both ends: the low bins win less often and
the high bins more often than the percent says. That shape is expected and is not
a defect to fix. The meter is a verdict on a finished game, and the thing it
strips out — luck — is exactly the thing that would make it calibrated against
results. A meter that scored perfectly against who actually won would be a meter
that had stopped removing luck.

Both losses are reproduced through the library's own scoring seam by
`tests/test_process_meter_stage0.py`, so the numbers a reader sees here are the
numbers the library computes rather than numbers that resemble them.

## 7. What this does not settle

- **Whether a future version should measure the scale from the full residual SD**
  and re-gate everything downstream. Open, §4.
- **Whether the pooled scale should ever become per-game.** Closed at the pooled
  form on the evidence above, not closed forever.
- **Whether 40,000 is right for a different arm.** The gate was run on this arm's
  draws. A definition change that widens the per-game draws would have to re-run
  it.
