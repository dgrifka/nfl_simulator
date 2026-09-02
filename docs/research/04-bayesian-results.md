# 04 — Bayesian results

*Script: `research/03_bayesian_rates.py`. Gates pre-registered in
`docs/research/03-model-foundations.md`, committed at `b914fb6` before any fit
existed. PyMC 6.3.1 / ArviZ 1.3.0 / nutpie 0.16.11.*

## Gate outcomes, stated first

| Gate | Result | Detail |
|---|---|---|
| 1 — sampler health | **FAIL then PASS** | The pre-registered centered parameterization failed badly (r_hat 1.71, ESS 6). The fallback fixed it (r_hat ≤ 1.003, ESS ≥ 1,400, zero divergences on all five fits). |
| 2 — calibration case | **FAIL** | 89% upper bound on the population SD of true fumble-recovery rates is **4.38 pp**, above the pre-registered 4.00 pp threshold. |
| 4 — posterior predictive | **PASS** | All five fits. Between-team variance tail probabilities 0.25–0.56, all comfortably interior. |

**Gate 2 failed, and per the pre-registration that constrains everything below.**
The rule committed in advance was: if the interval cannot rule out a 4 pp
population SD, the calibration case is not confirmed and no result from the
other models may be reported as trustworthy. So every number in this document
carries that caveat, and the section "What Gate 2's failure actually means"
explains what it does and does not imply. The threshold has not been moved.

## Gate 1: the pre-registered model did not sample

The foundations doc predicted a funnel in `kappa` and pre-registered a fallback.
The prediction was right and the diagnosis was worse than expected.

| Fit | max r_hat | min ESS bulk | Divergences |
|---|---|---|---|
| Attempt 1 — centered `p_team` (pre-registered) | **1.708** | **6** | 0 |
| Attempt 2 — marginalized (fallback) | 1.001 | 2,228 | 0 |

An effective sample size of 6 from 4,000 draws means the chains never explored
the posterior. Note that zero divergences accompanied that failure — a reminder
that "no divergences" is not evidence of convergence on its own.

**The fix.** `Binomial(n, p)` with `p ~ Beta(a, b)` is exactly `BetaBinomial(n, a, b)`
with `p` integrated out. Fitting the marginalized form deletes the 320
funnel-inducing `p_team` parameters and leaves a clean two-parameter posterior.
The generative story is unchanged — this is the same model, algebraically.

Per-team rates are still available, recovered by conjugacy rather than sampled:
given `(mu, kappa)`, a team's exact posterior is
`Beta(mu·kappa + k, (1−mu)·kappa + n − k)`. Nothing is lost.

The foundations doc named a logit-normal non-centered hierarchy as the fallback.
Marginalization is the better fix for this specific geometry and was used
instead; the substituted fallback is recorded here rather than quietly swapped,
and the two attempts agree on the point estimate (2.40 pp vs 2.39 pp), which is
itself evidence the disagreement was in the *interval*, not the answer.

One reproducibility note: `random_seed` is fixed at 20260817, but repeated runs
of the same script have produced 0 or 1 divergences on the fumble model. nutpie's
threading is not fully seed-deterministic. It does not change any conclusion, but
it means "zero divergences" here is a near-certainty rather than a guarantee.

## Results

All five fits, marginalized parameterization. `population_sd` is the standard
deviation of the Beta distribution over *true* team rates — the direct answer to
"how much skill could hide here". The relative column divides by the league rate,
which is the only fair way to compare a 1.3% base rate against a 47% one.

| Model | Team-seasons | Opportunities | League rate | Population SD | 89% interval | Relative |
|---|---|---|---|---|---|---|
| Fumble recovery (ex. aborted) | 320 | 4,898 | 46.8% | **2.39 pp** | 0.75 – 4.38 | **5.1%** |
| Fumble recovery (incl. aborted) | 320 | 5,914 | 52.1% | 2.64 pp | 0.86 – 4.66 | 5.1% |
| INT-worthy throw → INT | 128 | 3,160 | 46.0% | **6.60 pp** | 4.11 – 8.81 | **14.3%** |
| Penalties, pre-snap | 320 | 916,700 | 1.31% | 0.183 pp | 0.156 – 0.211 | **14.0%** |
| Penalties, judgment | 320 | 916,700 | 1.97% | 0.247 pp | 0.215 – 0.280 | **12.5%** |

### Fumble recovery is a coin — a slightly bent one

League rate 46.8% once aborted snaps are excluded. Team spread is 5.1% of that
rate, roughly a third of what every other component shows, and the interval's
lower bound sits at 0.75 pp — nearly at zero.

The shrinkage this produces is total. Buffalo recovered 10 of 12 in 2024, an
observed rate of 83%; the model puts their true rate at **48.0%**. Minnesota
recovered 1 of 9 in 2018, an observed 11%; the model says **45.9%**. Across all
320 team-seasons the posterior estimates span 45.6% to 48.3% — a range of under
three points, against observed rates spanning 11% to 83%.

That is the practical answer Phase 2 needs: **there is no such thing as a good
fumble-recovery team.** Any season rate away from ~47% should be read as the
season's noise, not the team.

Including aborted snaps raises the league rate to 52.1% and nudges the spread up,
exactly as document 01 predicted it would. The exclusion was the right call.

### Interceptions: the surprise, with a caveat

Given a charted interception-worthy throw, teams differ substantially in whether
it actually becomes an interception. The population SD is 14.3% of the league
rate and the interval excludes zero by a wide margin — the lower bound is 4.11 pp
on a 46.0% base, so even the pessimistic end of the posterior is a real effect.

A one-SD-good team converts 39.4% of its interception-worthy throws into
interceptions where a one-SD-bad team converts 52.6%. Over a season of ~25
interception-worthy throws that is about three interceptions, which is a real
football difference.

**This runs against the intuition that interceptions are largely luck**, and
document 02's split-half correlation of +0.164 pointed the same way. Two caveats
travel with it, and both are pre-registered:

1. `is_interception_worthy` is a human charting judgment with no published
   inter-charter reliability. Charter noise inflates apparent randomness, which
   biases this estimate **toward finding less skill**. The true effect may be
   larger, not smaller.
2. Only 128 team-seasons (2022–2025, four seasons of FTN data) and a median of 21
   interception-worthy throws per team-season. The interval is wide for a reason.

Note also that a team-season pools every quarterback who played, so some of this
is roster churn rather than a stable team property.

### Penalties: the pre-snap/judgment hypothesis is not supported

The hypothesis was that judgment calls — holding, pass interference, roughing —
would show *less* team skill than pre-snap penalties, because an official's
discretion adds noise the team does not control.

They do not. Relative spread is **14.0% for pre-snap** and **12.5% for judgment
calls** — statistically distinguishable but practically the same, and pointing
mildly the *wrong* way for the hypothesis. Both intervals exclude zero decisively
(these models have 916,700 plays behind them, so the estimates are tight).

Penalty discipline is a genuine, repeatable team trait in both classes, and
officiating discretion does not measurably wash it out. **Penalties should not be
neutralized by the simulator.**

## What Gate 2's failure actually means

The gate asked for the 89% upper bound on fumble-recovery population SD to be
below 4 pp. It came in at 4.38 pp. That is a miss by 0.38 pp.

**It is not evidence that fumble recovery is a skill.** The posterior mean is
2.39 pp with a lower bound of 0.75 pp, the relative spread is a third of every
other component's, and document 02's two independent split-half tests both landed
within noise of zero. Every piece of evidence points the same way.

What the gate failure means is narrower and more useful: **ten seasons of NFL
fumbles are not enough data to prove a small effect is absent.** 4,898 live
fumbles across 320 team-seasons, a median of 15 per team-season. With denominators
that small, the binomial noise floor is high enough that a true 4 pp spread and a
true 0 pp spread produce nearly the same data.

The honest reading is: *fumble-recovery skill is at most small, plausibly zero,
and the available data cannot narrow it further.* That is enough to justify
neutralizing fumble recovery in the simulator — an at-most-4pp effect is far
smaller than the swing a single recovery causes in one game — but it is a weaker
statement than "proven to be pure luck", and the write-up should not overclaim.

**The gate itself was the mistake, and that is the lesson.** It was set from a
football-effect-size argument with no power calculation — nobody asked whether
4,898 fumbles could achieve a 4 pp upper bound even if the truth were exactly
zero. It could not. A pre-registered gate should include that check. This is
recorded as a new row in the defect register below, and the corrective for the
next model is a power calculation before the threshold is committed, not a looser
threshold.

## Posterior predictive checks

Gate 4 passed everywhere. The statistic is the between-team variance of observed
rates, chosen because a model that matches the mean but misses the spread is
exactly the model that would mislead about skill.

| Model | Observed variance | Tail probability |
|---|---|---|
| Fumble recovery | 0.01869 | 0.418 |
| Fumble recovery (incl. aborted) | — | 0.246 |
| INT conversion | — | 0.364 |
| Penalties, pre-snap | — | 0.559 |
| Penalties, judgment | — | 0.531 |

All interior, none near the 0.055/0.945 boundaries. The beta-binomial reproduces
the observed team-to-team spread in every case.

## Figures

`research/04_figures.py` and `research/05_arviz_diagnostics.py` write to
`research/outputs/` (gitignored — regenerate rather than committing them).

| Figure | What it shows |
|---|---|
| `fig4_population_sd.png` | Team spread per rate, relative to that rate's own league average |
| `fig5_shrinkage.png` | Observed season rate vs shrunk posterior, all 320 team-seasons |
| `fig7_forest_fumble.png` | ArviZ forest plot of the 14 most extreme team-seasons |
| `fig8_ppc_fumble.png` | ArviZ posterior predictive ECDF for per-team recovery counts |

The forest plot is the most persuasive of the four. Minnesota 2018 recovered 1 of
9 and Buffalo 2024 recovered 10 of 12, and their posterior intervals are almost
indistinguishable — both centered near 47%, both spanning roughly 41% to 53%.
Seven-tenths of a season's apparent difference between the best and worst
fumble-recovery teams simply is not there.

Note the per-team posteriors are reconstructed by conjugacy rather than sampled,
since the marginalized model never instantiates `p_team`. Given `(mu, kappa)`,
`Beta(mu·kappa + k, (1−mu)·kappa + n − k)` is the exact posterior the centered
model would have drawn from — the same distribution, without the funnel.

## Updated defect register

| Defect | Evidence | Status |
|---|---|---|
| Gate 2 threshold set without a power calculation | 4,898 fumbles cannot achieve a 4 pp upper bound even under a true zero | **New.** Corrective: power-check every future threshold before committing it |
| Centered beta-binomial hierarchy does not sample | ESS 6, r_hat 1.71 | **Closed** by marginalization |
| Opportunity counts treated as fixed | A team's fumble count is itself a team property | Accepted by design |
| FTN interception-worthiness is a human judgment | No inter-charter reliability published | Open; biases toward no-skill |
| Team-season pools multiple quarterbacks | Model B attributes to a team what may belong to one passer | Open; player-level model is Phase 2 |
| Game script confounds interceptions | Trailing teams throw more | Open; partially mitigated by conditioning on IW throws |
| nutpie is not fully seed-deterministic | 0 or 1 divergences across identical runs | Open; no effect on conclusions |
| Penalty classes are hand-assigned | Our taxonomy, not the NFL's | Open; class lists in `src/nfl_simulator/rates.py` |

## What this changes for Phase 2

1. **Neutralize fumble recovery.** Use class-specific coins (40% run, 45% pass,
   64% muffed punt, 76% botched snap), not a flat 50/50, and use the league rate
   rather than any team's own rate — the shrinkage says no team's rate is real.
2. **Do not neutralize penalties.** Both classes are repeatable team traits.
3. **Interceptions need a player-level model before any neutralization.** There is
   real skill in whether a bad throw gets picked, so a blanket flip would erase
   it. This is the single biggest open question, and at 19% of outcome variance it
   is also the highest-value one.
4. **Report the simulator's uncertainty from the population posteriors**, not from
   point estimates. The intervals here are wide, and a deserve-to-win number that
   hides that is overclaiming.
