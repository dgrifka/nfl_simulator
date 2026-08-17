# 02 — Skill vs luck

*Script: `research/02_skill_vs_luck.py`. 2,761 games, 320 team-seasons,
200 random split-half draws. Run 2026-08-17.*

## Headline

**Fumble recovery is the only component with no team skill in it.** Its
split-half correlation is +0.055, and the recovery rate itself — the number the
literature quotes — comes in at +0.051, both indistinguishable from zero. The
calibration case landed exactly where it had to, so the machinery can be trusted
on the components whose answer we did not already know.

Those answers are more interesting than expected. Field-goal results, penalties
and interceptions all carry **real, repeatable team skill**. None of them is a
clean coin flip. Only fumble recovery should be neutralized outright; the other
three need partial treatment or none.

And a negative result worth stating plainly: **stripping luck did not improve
out-of-sample prediction.** Not for any component, not in any combination. This
does not sink the project, but it sharpens what the project is for.

## Test 1 — split-half persistence

Within each team-season, games are randomly split into two halves, each
component is averaged per game within each half, and the two halves are
correlated across all 320 team-seasons. Repeated 200 times with different random
splits; the table reports the mean and the 5th–95th percentile across those
draws.

Splitting *within* a season holds roster, coaching and scheme constant. Anything
that still correlates is a stable property of the team. Anything that does not is
noise.

| Component | Split-half r | 5th–95th pct | Full-season reliability | Verdict |
|---|---|---|---|---|
| EPA differential (total) | **+0.524** | +0.475 … +0.570 | 0.687 | skill |
| core offense/defense | **+0.519** | +0.470 … +0.562 | 0.683 | skill |
| interception | +0.164 | +0.100 … +0.230 | 0.281 | mostly noise, real skill inside |
| field goal luck | +0.145 | +0.086 … +0.200 | 0.253 | mostly noise, real skill inside |
| penalty | +0.121 | +0.069 … +0.176 | 0.215 | mostly noise, real skill inside |
| **fumble recovery luck** | **+0.055** | −0.014 … +0.123 | 0.105 | **luck** |

Direct test of the recovery rate, pooling fumbles rather than averaging games
(299 team-seasons with at least 8 fumble games):

> **split-half r = +0.051, 90% interval [−0.008, +0.117]**

The interval contains zero. Combined with the league rate of 52.1% from document
01, this is the textbook result reproduced on our own pipeline. **The method
works.**

### Reading the middle three

The three components sitting at r ≈ 0.12–0.16 are the substantive finding, and
the temptation is to round them to zero. They are not zero — every one of them
has a 5th percentile comfortably above it across 200 splits.

What each one is measuring explains why:

- **Field goals.** The `fg_luck` term is `(made − p_make_for_this_distance) ×
  swing`. A kicker who is genuinely better than league average at 50 yards
  produces a positive value every week. That is not luck; that is Justin Tucker.
  The component conflates kicker skill with ball-drifts-inside-the-upright, and
  the persistence tells us the skill part is real.
- **Penalties.** Discipline is coachable and it shows. r = 0.121 is small but
  stable, which fits the intuition that some teams simply commit fewer false
  starts than others year after year.
- **Interceptions.** r = 0.164 means quarterbacks differ in how often they throw
  them — again, not a surprise. But it also means roughly 84% of the variation in
  a team's interception EPA within a season is *not* the team. That is a large
  noise fraction sitting on the single biggest non-core component (19% of outcome
  variance from document 01).

The honest conclusion for these three is **partial**, not binary. Splitting them
into their skill and noise parts is what document 03's Bayesian models are for.

## Test 2 — out-of-sample prediction

The design question here needed a decision. A logistic regression on *within-game*
EPA differential would score about 96% — but that number is meaningless, because
game EPA is computed from the same plays that produced the score, and it cannot be
compared with a pre-game number like the Vegas spread.

So the test asks the question that actually bears on the simulator: **does
removing a component from the historical record produce a better forward-looking
estimate of team strength?** Each team is rated by its mean EPA differential over
its previous 17 games (strictly prior — no leakage), the home-minus-away rating
difference feeds a logistic regression trained on 2016–2023, and it is scored on
2024–2025. The Vegas closing spread runs through the identical pipeline on the
identical games.

| Model | Log loss | Brier | AUC | Accuracy |
|---|---|---|---|---|
| **Vegas spread_line** | **0.5977** | 0.2060 | 0.738 | 68.2% |
| raw EPA differential | 0.6375 | 0.2233 | 0.687 | 64.7% |
| minus FG luck | 0.6389 | 0.2239 | 0.683 | 65.0% |
| minus fumble luck | 0.6401 | 0.2245 | 0.682 | 63.3% |
| minus fumble + FG + penalty | 0.6414 | 0.2253 | 0.678 | 63.8% |
| minus fumble + FG luck | 0.6418 | 0.2253 | 0.679 | 64.3% |
| minus fumble + FG + INT | 0.6444 | 0.2264 | 0.670 | 62.9% |

*2,053 training games, 569 test games.*

Paired bootstrap of the per-game log losses against the raw model, 2,000
resamples:

| Model | Δ log loss vs raw | 95% CI | P(better than raw) |
|---|---|---|---|
| minus FG luck | +0.0014 | −0.0023 … +0.0051 | 0.24 |
| minus fumble luck | +0.0026 | −0.0006 … +0.0062 | 0.06 |
| minus fumble + FG + penalty | +0.0039 | −0.0038 … +0.0120 | 0.17 |
| minus fumble + FG luck | +0.0043 | −0.0010 … +0.0097 | 0.06 |
| minus fumble + FG + INT | +0.0069 | −0.0034 … +0.0171 | 0.10 |
| **Vegas spread_line** | **−0.0398** | **−0.0592 … −0.0200** | **1.00** |

Every luck-stripped variant straddles zero — no significant difference from raw —
while leaning consistently, if slightly, toward *worse*. Vegas beats all of them
decisively, and that gap is the only one the data actually resolves.

Calibration is respectable for every model. The raw EPA model's five equal-count
bins predict 0.33 / 0.46 / 0.55 / 0.64 / 0.77 and realize 0.32 / 0.41 / 0.56 /
0.61 / 0.80. Vegas is tighter at the extremes, as you would expect from a market.

### Why stripping doesn't help, and why that is fine

Two mechanisms, both mundane:

1. **A 17-game average has already removed the luck.** Luck is zero-mean by
   construction, so averaging 17 games shrinks it by about a factor of four.
   There is little left for an explicit subtraction to remove.
2. **The subtraction throws away real signal along with the noise.** `fg_luck`
   and `penalty` are not pure noise — they carry r ≈ 0.12–0.15 of genuine team
   skill. Removing them deletes that skill from the rating, which is exactly why
   the stripped variants trend slightly worse rather than slightly better.

This is a real constraint on the project's claims, and it should be stated
up front rather than buried: **luck-neutralization is a retrospective
adjudication tool, not a forecasting improvement.** The simulator answers "given
what happened in *this* game, who deserved to win?" — a question about one game,
where a single fumble recovery genuinely swings the answer. It does not claim to
tell you who will win next week better than the market does.

## Bias notes carried forward

- **Game script confounding** (flagged in document 01) is unaddressed here and
  matters for the interception result. Teams that fall behind throw more, so
  interception counts are partly a *consequence* of losing rather than a cause.
  The split-half design does not separate these.
- **Survivorship at team-season level** is handled: splitting within team-season
  means playoff teams contribute more games to both halves, not extra
  team-seasons.
- **Kicker changes mid-season** would depress `fg_luck` persistence, so +0.145 is
  if anything a floor on kicker skill, not a ceiling.

## What this changes

1. **Fumble recovery: neutralize fully.** Confirmed luck, r ≈ 0.05 on two
   independent tests. This is the one component the simulator can flip with
   confidence — using class-specific coins, per document 01.
2. **Field goals, penalties, interceptions: neutralize partially or not at all.**
   All three carry real skill. A binary skill/luck switch is the wrong model for
   them; they need a shrinkage estimate that separates the team's true rate from
   the season's noise. That is document 03.
3. **Reframe the pitch.** The simulator is a retrospective fairness measure. The
   prediction test says so directly, and claiming otherwise would not survive
   contact with the numbers above.
