# 03 — Model foundations: hierarchical rate models

*Written 2026-08-17, **before any model was fit**. The gates in §6 are
pre-registered: this file is committed to git before results exist, so
goalpost integrity is checkable by commit archaeology.*

*Stack verified at time of writing: PyMC 6.3.1, PyTensor 3.3.0, ArviZ 1.3.0,
nutpie 0.16.11.*

---

## 1. One-page story

### The question

Documents 01 and 02 gave each component a single number — a split-half
correlation — and that number answers "is there *any* skill here?" It does not
answer the question Phase 2 actually needs: **for a given team in a given
season, how much of its observed rate is real and how much is noise?**

A team that recovered 14 of 20 fumbles looks lucky, but how lucky? A quarterback
who threw 8 interceptions on 25 interception-worthy throws looks fortunate, but a
different quarterback with the same record over 60 throws is a different claim
entirely. Split-half correlation cannot distinguish those cases. A hierarchical
model can, because it estimates the **population spread** of true team rates and
shrinks each team toward the league mean in proportion to how little evidence
that team supplies.

### How it answers, in one paragraph

Each model is a beta-binomial hierarchy. A league-wide mean rate `mu` and a
concentration `kappa` define a Beta distribution over *true* team rates; each
team draws its true rate from that distribution; the team's observed successes
are Binomial draws on its own rate. Large `kappa` means the Beta is tight — every
team is essentially the league average, and there is no skill to find. Small
`kappa` means teams genuinely differ. The posterior for the implied population
standard deviation is therefore the direct answer to "how much skill could hide
here", and it comes with an interval rather than a point estimate.

### Five things to hold onto

1. `kappa` is the whole ballgame. It is the parameter that says whether teams
   differ at all.
2. The reported skill number is **the posterior for the population SD of true
   rates**, not `kappa` itself, because an SD in percentage points is readable
   and `kappa` is not.
3. Fumble recovery is the calibration model. Its population SD must come out
   near zero or the other two models cannot be believed.
4. Shrinkage is the deliverable, not a side effect. The shrunk per-team
   posteriors are what Phase 2's simulator will consume.
5. Every rate here is conditional on an *opportunity* denominator (fumbles,
   interception-worthy throws, plays). Rates without denominators are the thing
   this whole document exists to avoid.

### Statistic convention

Every number quoted in this document and its outputs is a **posterior mean with
an 89% equal-tailed interval** (ArviZ 1.x default, labelled `eti89_lb` /
`eti89_ub`). Where a probability is quoted it is a posterior probability, not a
p-value.

---

## 2. Data

### Model A — fumble recovery

- **Grain of a row**: one team-season. Two integers: live fumbles by that team,
  and how many that team recovered.
- **Source**: `data/pbp/*.parquet`, 2016–2025.
- **Filter**: `fumble == 1` with both `fumbled_1_team` and
  `fumble_recovery_1_team` non-null. Fumbles out of bounds are excluded — nobody
  recovers those, so there is no coin flip.
- **Uncomfortable fact that must be defensible**: aborted snaps are recovered by
  the offense 76% of the time versus ~42% for normal fumbles (document 01). The
  primary model **excludes aborted plays**, because pooling them would inject a
  fake between-team difference driven by how often a team's centre snaps badly.
  A secondary fit includes them to show what that pooling does.

### Model B — interception-worthy throw conversion

- **Grain of a row**: one team-season. Interception-worthy throws, and how many
  became actual interceptions.
- **Source**: `data/ftn/*.parquet` joined to pbp on
  `(nflverse_game_id, nflverse_play_id)`, **2022–2025 only**. FTN charting does
  not exist before 2022, so this is a four-season model, not a ten-season one.
- **Why this framing**: raw interception count confounds "throws bad passes"
  with "gets away with it". Conditioning on charted interception-worthiness
  isolates the second — which is the part that should be luck.
- **Uncomfortable fact**: `is_interception_worthy` is a human charting judgment,
  not a measurement. Charter inconsistency inflates apparent noise, which biases
  this model *toward* finding no skill. That direction of bias must be stated
  wherever the result is.

### Model C — penalty rates, pre-snap versus judgment

- **Grain of a row**: one team-season, one penalty class. Offensive plays run,
  and penalties of that class committed.
- **Source**: `data/pbp/*.parquet`, 2016–2025, `penalty == 1` with
  `penalty_team` equal to the team.
- **Split**: *pre-snap* penalties (false start, delay of game, illegal
  formation, illegal shift, encroachment, neutral zone infraction) versus
  *judgment* penalties (offensive/defensive holding, pass interference,
  unnecessary roughness, roughing the passer, face mask).
- **Why the split**: a false start is unambiguous and entirely the offense's
  doing. Defensive holding is an official's decision about a continuum. If those
  two behave differently, the officiating-noise hypothesis has support.

---

## 3. DAG

All three models share one shape.

```
        mu (league mean rate)      kappa (concentration)
              \                      /
               \                    /
                v                  v
          p_team[t]  ~  Beta(mu*kappa, (1-mu)*kappa)
                          |
                          |  n_opportunities[t]  (observed, fixed)
                          v
          successes[t]  ~  Binomial(n[t], p_team[t])
```

**Where inference is cut.** Nowhere inside a model — it is a single joint fit.
But there is a cut *between* documents: `n_opportunities` is treated as fixed and
known, when in reality how many fumbles a team suffers is itself a team property.
That is deliberate. This model asks only "given the fumbles you had, did you
recover an unusual share?" Making the opportunity count endogenous would answer a
different question and is out of scope.

**Emergent behavior to watch.** With `n` small (a team-season has ~10-20 live
fumbles), the posterior for any single team is dominated by the prior on
`p_team`, which is itself estimated. That is correct partial pooling, but it
means per-team forest plots will look nearly identical if `kappa` is large — and
that flatness is the *finding*, not a bug to fix.

---

## 4. Priors, site by site

| Site | Prior | Plain-language meaning |
|---|---|---|
| `mu` | `Beta(2, 2)` | League mean rate. Weak, symmetric, keeps mass off exactly 0 and 1. |
| `log_kappa` | `Normal(4, 2)` | Concentration on the log scale. Median `kappa` ≈ 55, 89% mass roughly 2 to 1,400 — spanning "teams differ wildly" to "teams are identical". |
| `p_team[t]` | `Beta(mu*kappa, (1-mu)*kappa)` | Each team's true rate, drawn from the league distribution. |
| `successes[t]` | `Binomial(n[t], p_team[t])` | What we observed. |

### Why `log_kappa` rather than `kappa`

The handoff plan's sketch used `kappa ~ HalfNormal(100)`. Sampling `kappa`
directly is the wrong geometry here: the likelihood is nearly flat in `kappa`
once `kappa` is large (all values above ~500 imply "no team differences" and look
alike to the data), so an unbounded positive prior lets the sampler wander into a
plateau and produce divergences. Working in `log_kappa` compresses that plateau
into a region the sampler can traverse. This is a **ruling, not a default**, and
the evidence is the geometry argument plus the divergence count reported in the
results.

`HalfNormal(100)` is also more informative than intended — it puts 68% of its
mass below `kappa = 100`, quietly asserting that teams *do* differ.
`Normal(4, 2)` on the log scale is closer to agnostic.

### The identification story

`mu` is pinned by the total successes across all teams. `kappa` is pinned by how
much the per-team observed rates scatter *relative to* the binomial scatter you
would expect from the observed denominators alone. That second quantity is the
entire signal, and it is weak when denominators are small — which is exactly why
the answer needs an interval, and why a large `kappa` posterior with a long right
tail is an expected and honest outcome rather than a failure.

### Reported quantity

```
population_sd = sqrt( mu * (1 - mu) / (kappa + 1) )
```

The standard deviation of the Beta distribution of true team rates, in
percentage points. This is what "how much skill could hide here" means
numerically.

---

## 5. Inference plan

- **Engine**: NUTS via **nutpie** (PyMC 6 auto-selects it when installed).
  Justified by geometry: these are small, smooth, continuous models with a known
  funnel risk in `kappa`, which is what NUTS handles well and what SVI would
  misreport by collapsing the `kappa` tail.
- **Configuration**: 4 chains, 1,000 tune, 1,000 draws, `target_accept = 0.9`.
  The raised `target_accept` is a pre-emptive concession to the `kappa` funnel.
- **Parameterization**: centered on `p_team`. Beta-binomial hierarchies do not
  admit the usual non-centered trick (there is no location-scale form for a
  Beta), and the data per team is not so thin that the centered form should fail.
  If divergences appear, the documented fallback is to reparameterize `p_team`
  via a logit-normal hierarchy with a non-centered offset — **not** to raise
  `target_accept` until the warnings stop.
- **Compute cost**: three primary models plus one secondary (fumbles including
  aborted snaps) = 4 arms. Each is ~320 rows and 2 hyperparameters; expected
  wall-clock is well under a minute per fit on this laptop, so total cost is
  minutes, not cluster-hours. No cheaper screen is warranted — the confirm run
  *is* cheap.
- **Downtime plan**: fits are short enough that nothing runs in parallel. Stated
  explicitly as required.

---

## 6. Pre-registered gates

Committed before any result exists.

### Gate 1 — sampler health (all four fits)

**Pass rule:** zero divergences, `r_hat < 1.01` on every parameter,
`ess_bulk > 400` and `ess_tail > 400`.
**On failure:** apply the documented reparameterization fallback from §5, refit,
and report both attempts. Do not tune `target_accept` upward to hide it.

### Gate 2 — the calibration case (Model A, fumbles, aborted excluded)

The incumbent claim is the public-literature result that fumble recovery has no
team skill, plus our own split-half r of +0.055 from document 02.

**Pass rule:** the 89% interval for `population_sd` of true recovery rates has an
upper bound **below 4 percentage points**.

Rationale for 4 points: a true SD of 4pp would mean a one-SD team recovers 46%
versus 54% for another — over a 16-fumble season that is a difference of about
1.3 fumbles, which is a real football effect and would be visible in split-half
correlation as r ≈ 0.15, comparable to what we measured for interceptions. If the
interval cannot rule that out, the model has not confirmed the calibration case
and **no result from Models B or C may be reported as trustworthy.**

**Noise instrument:** the posterior interval itself. There is no seed-pairing
concern; the same data and four chains are used throughout, with `random_seed`
fixed at 20260817.

### Gate 3 — Models B and C are interpretable only if Gate 2 passes

**Pass rule:** none of their own. These are estimation, not hypothesis tests, and
pre-registering a threshold for a quantity we have no prior estimate of would be
theatre. What *is* pre-registered is the reporting rule:

- The population SD posterior is reported with its 89% interval, always.
- Any claim that a component "has skill" requires the 89% interval for
  `population_sd` to exclude zero by a margin the text states in percentage
  points.
- Model B's result is reported with the charter-noise caveat from §2 attached,
  because that bias runs toward finding no skill.

### Gate 4 — posterior predictive

**Pass rule:** the observed distribution of per-team success counts falls inside
the posterior predictive envelope, and the observed *variance* across teams sits
within the central 89% of the posterior predictive variance. The variance check
is the one that matters: a model that gets the mean right and the spread wrong is
precisely a model that would mislead about skill.

---

## 7. Known-defect register

| Defect | Evidence | Status |
|---|---|---|
| Opportunity counts treated as fixed | A team's fumble count is itself a team property; conditioning on it discards that information | Accepted, by design (§3) |
| FTN interception-worthiness is a human judgment | No inter-charter reliability published | Open; biases Model B toward no-skill, stated wherever reported |
| Team-season is the unit, but quarterbacks change mid-season | Model B attributes to a team what may belong to one passer | Open; a player-level version is Phase 2 work |
| Game script confounds interceptions | Trailing teams throw more (document 01 bias assessment) | Open; conditioning on interception-worthy throws mitigates but does not remove it |
| EPA model vintage drifts across seasons | nflverse has revised its EPA model within 2016–2025 | Contained: these models use counts, not EPA, so they are immune |
| Penalty classes are hand-assigned | The pre-snap/judgment split is our taxonomy, not the NFL's | Open; the class list is in §2 so it can be argued with |

---

## 8. Constants appendix

| Constant | Value | Where it lives |
|---|---|---|
| `RANDOM_SEED` | 20260817 | `research/03_bayesian_rates.py` |
| chains / tune / draws | 4 / 1000 / 1000 | `research/03_bayesian_rates.py` |
| `target_accept` | 0.9 | `research/03_bayesian_rates.py` |
| `mu` prior | `Beta(2, 2)` | `research/03_bayesian_rates.py` |
| `log_kappa` prior | `Normal(4, 2)` | `research/03_bayesian_rates.py` |
| Gate 2 threshold | 4 percentage points | this document, §6 |
| FTN seasons | 2022–2025 | `src/nfl_simulator/ingest.py` (`FTN_SEASONS`) |
| pbp seasons | 2016–2025 | `src/nfl_simulator/ingest.py` (`PBP_SEASONS`) |

Results are written to `docs/research/04-bayesian-results.md`.
