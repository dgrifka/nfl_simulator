# 01 — Descriptive EDA

*Script: `research/01_descriptive_eda.py`. Data: play-by-play 2016–2025,
484,254 plays, 2,761 games. Run 2026-08-17.*

## Headline

EPA differential is very nearly the scoreboard: the team with the higher game
EPA differential won 95.9% of decided games, and EPA differential correlates
with points margin at r = 0.996. Of the variance in that differential, **70%
sits in ordinary offense and defense, 19% in interceptions, and only 7% in the
two components that look like coin flips** — fumble recovery (3.8%) and
field-goal results (3.0%).

That 7% is the ceiling on what a deserve-to-win simulator can possibly move, if
fumble recovery and field goals turn out to be the only luck. Step 2 of this
research (`02-skill-vs-luck.md`) tests whether interceptions and penalties
belong on the luck side too, which would raise the ceiling substantially.

## Data audit

Ten seasons, no missing seasons, no duplicate games. Per season the pull holds
47k–50k plays across 267–285 games, and the counts of the events we care about
are stable year to year:

| Season | Plays | Games | Fumbles | INTs | FG att | Penalties |
|---|---|---|---|---|---|---|
| 2016 | 47,651 | 267 | 666 | 436 | 1,050 | 3,545 |
| 2020 | 47,705 | 269 | 608 | 416 | 1,015 | 2,986 |
| 2022 | 49,434 | 284 | 678 | 436 | 1,105 | 3,153 |
| 2025 | 48,771 | 285 | 576 | 406 | 1,140 | 3,559 |

Two structural notes the validator already encodes: 2022 has 284 games rather
than 285 because Bills–Bengals was abandoned and never replayed, and the game
count steps up in 2020 (playoff field 12 → 14 teams) and again in 2021
(16 → 17 game regular season).

**Missing data.** `epa` is null on roughly 540–570 rows per season. Every one of
them is a clock row — timeout, end of quarter, two-minute warning — not a play.
This is MCAR with respect to anything we model, because the missingness is a
property of the row type and not of any outcome. Ten games ended in ties and are
excluded from the win-rate tables.

## Question 1 — does EPA differential track winning?

| Split | n | P(higher-EPA team won) |
|---|---|---|
| Overall | 2,751 | **95.9%** |
| Home team led EPA | 1,405 | 99.9% |
| Away team led EPA | 1,346 | 91.8% |

By season the rate never leaves the 93.9%–97.5% band, so this is a stable
property of the sport rather than an artifact of one rule era.

The home/away asymmetry is the interesting part and it is not a bug. Home teams
win 55.0% of decided games. When the home team also outplays its opponent by EPA
it essentially never loses; when the *away* team outplays the home team it still
loses 8.2% of the time. That gap is home-field advantage showing up exactly where
you would expect it — in the games close enough for it to matter.

**The caveat that governs everything downstream.** This 95.9% is *not* a
prediction. Game EPA differential is measured from the same plays that produced
the score, so it contains the outcome. It is an accounting statement — "EPA
differential is a faithful summary of what happened on the field" — and that is
precisely what makes it the right currency for a deserve-to-win decomposition.
It is also why the predictive test in the next document may not use within-game
EPA as a feature.

## Question 2 — where does the variance live?

Each game's home-perspective EPA differential is partitioned exactly into five
components (see `src/nfl_simulator/components.py` for the construction). Shares
use the covariance decomposition, so they sum to 1.

| Component | SD per game (EPA) | Share of EPA-diff variance | Share of points-margin variance |
|---|---|---|---|
| core offense/defense | 14.13 | **70.1%** | 70.2% |
| interception | 6.58 | **19.1%** | 18.7% |
| penalty | 4.13 | 4.0% | 3.9% |
| fumble recovery luck | 3.07 | 3.8% | 3.7% |
| field goal luck | 3.06 | 3.0% | 2.7% |
| *unexplained by EPA* | — | — | 0.8% |

The two columns agree closely, which is another way of saying EPA differential
and points margin are nearly the same quantity (r² = 0.991). Only 0.8% of margin
variance is invisible to EPA.

**Interceptions are the surprise.** They carry nearly five times the variance of
fumble recovery. If interceptions turn out to contain a large luck component,
neutralizing them matters far more to a deserve-to-win estimate than fumbles do —
which reframes what Phase 2 should prioritize.

Note the construction is deliberately conservative for the two luck terms.
`fumble_luck` is *not* the EPA of fumble plays; it is only the swing attributable
to which way the ball bounced. The cost of fumbling in the first place stays in
`core`, where it belongs. Same for `fg_luck`: choosing to attempt a 55-yarder is
a decision that stays in `core`, and only make-versus-miss relative to the
distance-bin rate is counted as luck.

## The coin-flip baselines

### Fumble recovery — and a heterogeneity worth knowing about

Across 5,914 live fumbles (a loose ball recovered by an identified team), the
fumbling team got it back **52.1%** of the time. That reproduces the textbook
~50% and is the calibration case working as intended.

But the pooled number hides a genuinely heterogeneous mix:

| Class | n | Own-recovery rate |
|---|---|---|
| pass, normal play | 2,892 | 45.3% |
| run, normal play | 1,149 | 40.3% |
| run, aborted snap | 946 | **76.2%** |
| punt (muffed return) | 672 | 64.4% |
| kickoff | 182 | 46.2% |
| pass, aborted snap | 68 | 100% |

Aborted snaps are the big one. When a snap goes wrong the ball squirts backward
into a space occupied by the quarterback and nobody else, and the offense gets it
back three times in four. Pooling those with a running back's fumble in traffic
would be a modeling error in both directions: it would credit offenses with good
luck they didn't have on botched snaps, and it would understate the bad luck of
losing a normal fumble.

**This is a design constraint on Phase 2, not a curiosity.** The simulator must
not neutralize fumbles with a flat 50/50 coin. The coin is class-specific, and
the classes range from 40% to 76%.

Muffed punts at 64.4% run the other way from intuition and are worth a second
look later; the returning team is the one who fumbles on 97% of those, which the
attribution logic handles correctly.

### Field goals

Make rate falls monotonically with distance, as it must: 98.8% inside 20–24
yards, 83.2% at 40–44, 71.0% at 50–54, 40.7% at 60–64.

The 65+ yard bins hold only 19 attempts across ten seasons, so they borrow their
make rate from the nearest well-populated bin (60–64) rather than from the league
average. That distinction matters a lot: the league-wide make rate is 85%, driven
by thousands of chip shots, and applying it to a 65-yard heave would have booked
an enormous fake "bad luck" charge against every team that ever tried one.

## Bias assessment

Required gate before any modeling. Documented even where nothing was found.

- **Target population** — NFL regular-season and playoff games under roughly the
  current rule set. 2016 is the floor because EPA model vintage and rule changes
  make earlier seasons a different game.
- **Selection bias** — none identified. This is the full population of games, not
  a sample. Every game played in the window is present.
- **Survivorship bias** — not applicable at game level; games do not drop out.
  It *will* apply at team-season level in the persistence tests, where teams that
  play more games (playoff teams) contribute more observations. The split-half
  design in document 02 addresses this by splitting within team-season.
- **Measurement bias** — one real instance. `epa` comes from an nflverse model
  that has been revised over the ten-year window, so EPA values are not strictly
  comparable across seasons. All analysis here is either within-season or uses
  season as a grouping variable, which contains it. The FTN charting columns only
  exist from 2022, so anything built on interception-worthiness is a four-season
  analysis, not a ten-season one.
- **Confounding** — game script is the main one. Teams that fall behind throw
  more, which raises both interception counts and pass EPA variance. This is a
  genuine confound for the interception component and is flagged for document 02.
- **Temporal bias** — the 2020 season was played without normal crowds, which
  measurably shrank home-field advantage that year. It is one season in ten and
  is left in, but it is a known perturbation.
- **Conclusion** — no bias severe enough to block the analysis. Two are carried
  forward as explicit caveats: EPA model vintage across seasons, and game script
  confounding the interception component.

## What this changes

1. The simulator's fumble coin must be class-specific. A flat 50/50 is wrong by
   up to 26 percentage points on aborted snaps.
2. Interceptions deserve first-class attention. At 19% of variance they dominate
   the two components we already believed were luck.
3. The ceiling on deserve-to-win adjustments is 7% of outcome variance if only
   fumbles and field goals are luck. Whether that ceiling rises is the question
   document 02 answers.
