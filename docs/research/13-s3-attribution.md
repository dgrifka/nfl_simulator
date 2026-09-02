# 13 — Leverage timing: does it follow the quarterback or the scheme?

*Written 2026-08-17, **before `research/21_s3_attribution.py` existed**. Null
bounds and power: `research/21_s3_attribution_power.py`, results in
`research/outputs/21_s3_attribution_power.json`. Committed to git before the
attribution produces a result, so goalpost integrity is checkable by commit
archaeology.*

*Inputs: documents 01–12, all settled. Process laws unchanged.*

---

## 1. One-page story

### The question

Document 08 §9 found the one sequencing channel that is **skill**. Two offenses
can generate identical expected points and one of them converts it into
meaningfully more winning, because its production lands in moments that decide
games. The measure — call it S3 — is the gap between a team's win-probability
contribution and its expected-points contribution, and it persists across the
halves of a team's own season at **r = +0.180**, surviving a control for playing
close games at **+0.144**. That is larger than any of document 02's middle three
components.

Its defect register left the obvious question open:

> *"S3's mechanism is unidentified. Persistence survives the game-state control,
> but **why** teams differ — coaching, quarterback, situational scheme — is
> untested. A crossed QB × coach model is the natural next step and is out of
> Phase 3's scope."*

**This document is that next step. Does clutch conversion follow the
quarterback, or does it follow the scheme?**

### How it answers, in one paragraph

Every team-game gets an S3 value, the quarterback who threw the most passes in
it, and the head coach who ran it. A crossed random-effects model splits the
variation in S3 into a quarterback component, a coach component and everything
else. The reported quantities are the two population standard deviations, each
in win-probability points per game, each with an 89% interval — and each read
against a bound built by simulating this exact design under a **true spread of
exactly zero**, which is the instrument document 05 §7 built for the interception
attribution round.

### Five things to hold onto

1. **The grain is the person, not the person-season, and that is the ruling this
   design turns on.** The Phase 4 plan named "QB-season × coach". A
   quarterback-season sits inside exactly one coach-season, so those two factors
   would be **nested rather than crossed** — a coach effect and a
   quarterback-season effect would be two names for the same partition, and no
   amount of data could separate them. Taking both factors at the person level,
   across seasons, is what makes the crossing real.
2. **Mobility is the identification, and it has to be checked rather than
   assumed.** A quarterback effect is separable from a coach effect only because
   quarterbacks change coaches and coaches change quarterbacks. §2 reports both
   rates before anything is fitted; with no mobility this design would be
   estimating a single confounded number twice.
3. **This is estimation, not a hypothesis test.** Document 03 §6's Gate 3
   convention holds: pre-registering a threshold for a quantity with no prior
   estimate is theatre. What is pre-registered is the **reporting rule**, in §6.
4. **The null bound comes first.** Both scales are read against what this design
   produces when the truth is exactly zero, never against zero itself — document
   05 §8's closing defect, which is a property of the parameterization rather
   than a finding.
5. **Nothing here can enter the ledger.** Document 08 §6 committed, before any
   sequencing result existed, that a sequencing channel has no branch point and
   is denied neutralization at any value of `w`. S3 came back as *skill*, which
   makes the point moot twice over: skill already lives in `core`. This round
   changes what may be *said* about the number, not what the simulator does with
   it.

### Statistic convention

Posterior means with 89% equal-tailed intervals, matching documents 03, 05, 05b,
08, 09 and 14. S3 is measured in **win-probability points per team-game**, so a
spread of 0.03 means a one-SD entity moves 3 percentage points of win
probability per game beyond what its expected-points production implied.

---

## 2. Data

- **Grain of a row**: one team-game.
- **Response**: `S3 = sum(wpa) − slope · sum(epa)` over every play the team had
  the ball for, exactly as document 08 §3 defines it. Positive means the team's
  production moved win probability more than its point value implies.
- **Source**: `data/pbp/*.parquet` 2016–2025 for the plays and the passers,
  `data/schedules.parquet` for the coaches.
- **Grouping factors**: the **primary passer** — the quarterback who threw the
  most passes for that team in that game — and the **head coach**, both taken at
  the person level across all ten seasons.

### Facts that must be defensible by name

- **A game started by a backup is attributed to the backup.** The unit is the
  team-game, not the team-season, so a quarterback gets the games he actually
  played. Attributing a game to the listed starter who did not take a snap would
  put the wrong name on the effect being estimated — and quarterback changes are
  precisely the variation that identifies a quarterback effect apart from a coach
  one.
- **Coaches are joined through the play-by-play's own team codes, not the
  schedule's.** `data/schedules.parquet` records relocated franchises under their
  **historical** codes — SD, STL, OAK — while the play-by-play uses current ones.
  Joining on team code directly silently drops 81 team-games, all of them the
  same three franchises. So the home/away role is read from the play-by-play,
  which is internally consistent with `posteam`, and only the coach names come
  from the schedule. **This was found by checking a join's null count**, and it
  is the kind of defect that would never announce itself.
- **`wpa_per_epa` is estimated once, league-wide, and treated as fixed.** It is
  pinned by roughly 450,000 plays, so its uncertainty is negligible next to the
  entity-level question; treating it as fixed is a rounding, not a modelling
  claim. Document 08 §4 made the same cut.
- **No minimum-games filter on either factor.** Partial pooling is what a
  hierarchy is for: a quarterback with two games contributes almost nothing after
  shrinkage, and filtering him out would change the design rather than the
  answer. The distributions of games per entity are reported instead.
- **The secondary arm restricts to one-score game states** (`|score
  differential| ≤ 8`) and **refits the slope inside the subset**, exactly as
  document 08 §9's exploratory control did and for the reason recorded there:
  leverage is uniformly higher in close games, so reusing the full-sample slope
  would leave a constant offset in every team's residual and manufacture a
  correlation from nothing.
- **S3 contrasts two nflverse models, both revised across the window.** Document
  08 §7 recorded this and it is inherited unchanged. All analysis is
  within-window and the entity effects are estimated across seasons, so an EPA
  vintage drift is absorbed into the residual rather than into a quarterback.

---

## 3. DAG

```
   sigma_qb                         sigma_coach
      |                                  |
      v                                  v
   qb[q] ~ Normal(0, sigma_qb)     coach[c] ~ Normal(0, sigma_coach)
      \                                  /
       \                                /
        v                              v
        S3[team-game] = mu + qb[q] + coach[c] + e
                                            e ~ Normal(0, sigma_e)
```

**Where inference is cut.** `wpa_per_epa` is estimated outside the model and
handed in as a constant (§2). Nothing else is cut — the two scales and the
residual are estimated jointly.

**Emergent behaviour to watch.** The two crossed scales trade off along a ridge:
variation that could be attributed to a quarterback could also be attributed to
the coach he played for, and only mobility breaks the tie. Document 05 §8
recorded exactly this geometry mixing slowly in the interception round — low
effective sample size with **zero divergences**, which is the signature of a slow
ridge rather than bad geometry. The fix there was more draws, and the same fix is
pre-registered here.

---

## 4. The instrument, characterized before the gate was written

*Document 10 §3's process law: characterize a new instrument on known arms before
committing a threshold to it.*

The achievable-null bound needs hundreds of fits, which NUTS cannot supply at
this size. So the same move document 05 §7 made for the beta-binomial is made
here for a crossed Gaussian: **an exact grid posterior**
(`research/_crossed_gaussian_grid.py`).

Writing `lambda = sigma² / sigma_e²` for each factor puts the marginal covariance
in the form `V = sigma_e² (I + Z Λ Z')`, so `sigma_e` profiles out of the
restricted likelihood in closed form and a **two-dimensional** surface over
`(lambda_qb, lambda_coach)` is all that remains. Every grid point costs one
Cholesky of a few-hundred-square matrix via the Woodbury identity, never an
n × n operation.

**The approximation, stated plainly:** the posterior is over the two variance
ratios with `sigma_e` held at its restricted-likelihood profile rather than
integrated over. That is a conditioning, not a marginalizing, and it makes the
intervals slightly narrower than a full three-parameter posterior would be.

**What licenses using it.** On a synthetic dataset sized to this design — 5,441
team-games, 167 quarterbacks, 93 coaches — the grid and a PyMC/NUTS fit agree on
the posterior means to well under one percent:

| Quantity | NUTS mean | Grid mean | Relative gap |
|---|---|---|---|
| `sigma_qb` | 0.03578 | 0.03574 | **0.11%** |
| `sigma_coach` | 0.02373 | 0.02391 | **0.75%** |
| `sigma_e` | 0.24021 | 0.24008 | **0.05%** |

The 89% *intervals* are narrower on the grid — 0.029–0.040 against NUTS's
0.028–0.043 for `sigma_qb` — which is the stated approximation showing up exactly
where it was predicted to. **That direction does not bias the achievable-null
comparison**, because the null bound and the observed fit are produced by the
same instrument. It would only matter if a grid interval were compared against a
NUTS one, which no gate in this document does.

The check is kept as runnable code (`_crossed_gaussian_grid.self_check`) rather
than a comment, so it can be re-run when the stack moves — the same role
`_betabinom_grid.self_check` plays for its own instrument.

---

## 5. The achievable-null bound and the power

*Computed before any threshold in §6 was committed. 400 datasets per scenario at
the **real design** — the real quarterbacks, the real coaches, the real number of
games each pairing played — under a known truth, each fitted, each contributing
the 89% bound it produced.*

### The design

Every number here is a property of who played for whom, and none of them carries
any information about the answer.

| Parameter | Value |
|---|---|
| Team-games | **5,522** |
| Quarterbacks (primary passers) | **167** |
| Head coaches | **95** |
| Median team-games per quarterback | 14 |
| Median team-games per coach | 45 |
| **Quarterbacks who played for more than one coach** | **59.3%** |
| **Coaches who had more than one quarterback** | **84.2%** |
| SD of S3 per team-game | 0.2390 |
| `wpa_per_epa` slope | 0.023497 |

**The last two mobility rows are the identification**, and they are healthy. Six
quarterbacks in ten played under more than one head coach, and five coaches in
six worked with more than one quarterback. Without that churn the two factors
would partition the data identically and no model could separate them.

### The achievable-null bounds

The 90th percentile of the 89% upper bound this design produces when the true
spread is **exactly zero**:

| Factor | Null bound |
|---|---|
| Quarterback | **0.02099** |
| Head coach | **0.02097** |

That the two are within 0.00002 of each other is itself informative: the design
is very nearly symmetric between the factors, so a difference in the fitted
spreads cannot be an artifact of one factor being better observed than the other.

### Power

400 datasets per cell. Power is the chance the factor's 89% **lower** bound
clears its own null bound — the claim rule §6 commits to.

| True spread | In football | Quarterback | Coach |
|---|---|---|---|
| 0.010 | 1.0 pp of win probability per game | 0.000 | 0.000 |
| 0.020 | 2.0 pp | 0.020 | 0.013 |
| **0.0318** | **3.2 pp — document 08 §5's own reference effect** | **0.632** | **0.667** |
| 0.050 | 5.0 pp | 1.000 | 1.000 |
| 0.080 | 8.0 pp | 1.000 | 1.000 |

> **Minimum detectable spread at 80% power: about 0.037 — roughly 3.7 percentage
> points of win probability per game.**

**And here is the uncomfortable part, stated before the fit rather than after
it.** Document 08 §5 computed that a sequencing spread producing its r = 0.12
reference correlation corresponds to `tau = 0.0318` in S3's units. **At that
effect size this design reaches only 63–67% power.**

So the honesty gate every other round in this project carries — "power at the
reference must be at least 0.80 for a null to be interpretable" — **fails here,
and it fails in advance.** The consequences are committed now:

- **A null on either factor is not evidence of absence.** If neither interval
  clears its bound, the only honest statement is *5,522 team-games cannot resolve
  an effect this size*, which is exactly what document 05 §8 had to say about the
  interception attribution. That is written into §6's decision rule.
- **A positive result remains readable.** The bound is the 90th percentile of a
  true-zero design, so its false-positive rate is 10% by construction whatever
  the power is. Low power means the design would miss a *smaller* effect, not
  that it invents this one. (Document 14 §10 had to make this same distinction
  after the fact for an underpowered return cell; making it in advance here is
  the improvement.)
- **The most likely outcome of this round is "unresolved"**, and saying so before
  the fit is what stops that outcome from being quietly reframed afterwards.

---

## 6. Pre-registered gates and reporting rules

### Gate A-1 — sampler health, with a relative tolerance

The grid is the reported instrument; a PyMC/NUTS fit on the same data is the
cross-check.

**Pass rule, two parts:**

1. **NUTS health:** zero divergences, `r_hat < 1.01`, `ess_bulk > 400`,
   `ess_tail > 400`. **3,000 tune / 3,000 draws rather than the usual 1,000**,
   because document 05 §8 recorded this exact geometry mixing slowly and
   pre-committing the larger budget is cheaper than discovering it again. Raising
   `target_accept` to quiet a warning remains forbidden by document 03 §5.
2. **Agreement:** the grid and NUTS posterior means agree to within **5% of the
   quantity**, on all three scales.

**The tolerance is relative, and that is document 09 §9's corrective applied
directly.** Gate C-1 there failed two candidates on an *absolute* 0.01 pp
tolerance borrowed from a penalty-rate model, where the same tolerance asked for
agreement forty times tighter in relative terms than any finite number of draws
could deliver. A convergence tolerance must be stated relative to the quantity,
or per-candidate from its own Monte Carlo standard error.

### Gate A-2 — the reporting rule *(no pass/fail)*

Following the convention document 03 §6 Gate 3 set and document 05 §7 reused:
these are estimation, and a threshold for a quantity with no prior estimate would
be theatre. **What is committed is how the result may be described:**

- **Both scales are reported with 89% intervals whatever they say**, alongside
  `P(quarterback spread > coach spread)`.
- A claim that leverage timing **"belongs to"** quarterbacks or to coaches
  requires that entity's 89% interval to **clear the achievable-null bound**
  *and* the other's to **fail to** — stated in win-probability points.
- **If both clear it**, the honest statement is that the effect is **shared**,
  and neither factor may be named as the mechanism.
- **If neither clears it**, the honest statement is that *this many team-games
  cannot tell these two apart* — not that the effect is absent. Document 05 §8's
  interception round is the worked example of that distinction, and it is
  pre-committed here because it is the most likely outcome.

### Gate A-3 — does the attribution survive the game-state control? *(secondary, reported)*

Document 08 §9 found that roughly a fifth of S3's persistence was playing close
games rather than timing. **Statistic:** the same two scales, refitted on
one-score plays only with the slope refit inside the subset.

**No pass rule.** The reporting rule is that the secondary arm is reported
beside the primary whatever it says, and that a claim surviving in the primary
but collapsing in the secondary is described as **game state, not timing**.

### The decision rule, committed in advance

| Outcome | Verdict | What changes |
|---|---|---|
| Gate A-1 fails on health | Sampler broken | Nothing is reported. Refit with more draws |
| Gate A-1 fails on agreement | The two instruments disagree | The grid's null bounds cannot be applied to a NUTS estimate; report both and treat the grid as the suspect |
| One factor clears its bound, the other does not | **The mechanism is named** | A reported finding. **No ledger row** — Gate A denied sequencing in document 08 §6 |
| Both clear their bounds | **Shared** | A reported finding, described as shared |
| Neither clears its bound | **Unresolved** | Reported with §5's power table. Not "no effect" |
| The primary claim collapses in Gate A-3 | **Game state, not timing** | The claim is withdrawn and described as such |

---

## 7. Disclosure

**Document 08 §9's S3 result — including the +0.180 persistence and the +0.144
competitive-play control — was known when this document was written.** It is the
reason the round exists, and it is a property of a *different statistic* (a
split-half correlation) than anything gated here (two population standard
deviations from a crossed model). Neither the null bounds nor the reporting rule
can be moved by it: the bounds come from simulating a true zero at this design,
and the rule is the convention documents 03 and 05 already fixed.

**The design parameters in §2 — the mobility rates, the entity counts — were
measured before this document was written**, because §1's identification claim is
not arguable without them. They carry no information about the answer: how often
quarterbacks change coaches says nothing about whether either matters.

**The instrument's self-check in §4 ran on synthetic data**, not on the real S3
response, so it exposed nothing about the result.

**And one blemish on the commit ordering, recorded rather than tidied away.**
`research/21_s3_attribution.py` — the script that produces §10 — was written
while the null-bound simulation was still running and was committed at `d32c928`,
**before** this document. It had not been run: `research/outputs/21_s3_attribution.json`
postdates this document's commit, and the null bounds it consumes did not exist
when it was written. But an auditor checking goalpost integrity the way this
project invites — `git log --diff-filter=A` — will see the results script land
first, and that is a weaker record than steps 1 and 2 produced. Rewriting the
history to make the ordering look right would be exactly the wrong instinct, so
it is written down instead.

---

## 8. Known-defect register

| Defect | Evidence | Status |
|---|---|---|
| **The design is underpowered at this project's own reference effect** | §5 — 63–67% power at `tau = 0.0318`, against the 0.80 every other round requires | **Open, and pre-registered.** A null here cannot be read as evidence of absence; §5 and §6 both say so before the fit. Fixing it needs more seasons, not a better model |
| **The grain deviates from the Phase 4 plan's wording** | The plan named "QB-season × coach"; a QB-season is nested inside a coach-season | **Open, disclosed in §1.** The nested form cannot separate the factors at all, so the deviation is the difference between an answerable and an unanswerable question |
| The grid conditions `sigma_e` on its profile | §4 | **Accepted, measured.** Intervals slightly narrow; the null bound uses the same instrument so the comparison is internally consistent |
| S3 contrasts two nflverse models, both revised | Document 08 §7 | **Open**, inherited unchanged |
| A head coach is not the play-caller | Many teams' offensive coordinator calls plays, and coordinators change without the head coach changing | **Open, and it is the largest limitation.** A "coach" effect here is a franchise-leadership effect, not a play-caller effect; nflverse carries no coordinator field |
| The primary passer is not the only passer | A game split evenly between two quarterbacks is attributed entirely to one | **Open.** Affects a small number of team-games and biases toward finding *less* quarterback effect |
| Quarterback and coach effects are estimated across seasons | A quarterback's ability changes; a coach's scheme changes | **Open.** The person-level grain is what buys identification, and the cost is that within-career change lands in the residual |
| Team identity is absent from the model | A quarterback and his coach share a roster, a defense and a schedule | **Open, and it cannot be fixed by adding a team factor** — team is very nearly the intersection of the two grouping factors, so adding it would compete with both for the same variation |

---

## 9. Constants appendix

| Constant | Value | Where it lives |
|---|---|---|
| `RANDOM_SEED` | 20260817 | `research/21_s3_attribution_power.py`, `research/21_s3_attribution.py` |
| `DATASETS` per scenario | 400 | `research/21_s3_attribution_power.py` |
| `NULL_PERCENTILE` | 90 | same |
| Grid points per axis | 41 | `research/_crossed_gaussian_grid.py` |
| Grid range, log10 λ | −5.0 to +0.5 | same |
| chains / tune / draws (NUTS) | 4 / 3000 / 3000 | `research/21_s3_attribution.py` |
| `target_accept` | 0.9 | same |
| **Achievable-null bound, quarterback** | **0.02099** | this document §5 |
| **Achievable-null bound, coach** | **0.02097** | this document §5 |
| Reference effect (`tau` at document 08's r = 0.12) | 0.0318 | document 08 §5 |
| Power at the reference | 0.632 / 0.667 | this document §5 |
| Minimum detectable spread at 80% power | ≈ 0.037 | this document §5 |
| Team-games / quarterbacks / coaches | 5,522 / 167 / 95 | measured, §5 |
| Gate A-1 relative tolerance | **5%** | this document §6 |
| Competitive-play cut | \|score differential\| ≤ 8 | both scripts, from document 08 §9 |
| pbp seasons | 2016–2025 | `src/nfl_simulator/ingest.py` |

Results are written back into this document as §10.

---

## 10. Results

*Script: `research/21_s3_attribution.py`. Design, instrument, null bounds and
reporting rules fixed by §§1–9 above, committed at `365f7d5` before this script
produced a result. Results in `research/outputs/21_s3_attribution.json`.*

### Outcome, stated first

| Gate | Rule | Result |
|---|---|---|
| **A-1 — sampler health** | 0 divergences, r̂ < 1.01, ESS > 400 | **PASS** — 0 divergences, r̂ 1.0016, ESS bulk 1,431, tail 1,318 |
| **A-1 — grid vs NUTS** | relative gap ≤ 5% on all three scales | **PASS** — 1.10% / 1.04% / 0.05% |
| **A-2 — the reporting rule** | one factor clears its bound, the other does not | **UNRESOLVED** — neither clears it |
| **A-3 — game-state control** | reported | Both spreads shrink; still not separable |

> **Verdict: 5,522 team-games cannot tell a quarterback from a head coach on
> leverage timing.** §5 pre-registered this as the most likely outcome, and it
> is.

### The estimates

| Factor | Spread | 89% interval | In football | Null bound | Clears it? |
|---|---|---|---|---|---|
| Quarterback | **0.0245** | 0.0130 – 0.0334 | **2.45 pp** of win probability per game | 0.0210 | **No** |
| Head coach | **0.0252** | 0.0129 – 0.0390 | **2.52 pp** per game | 0.0210 | **No** |
| *(residual)* | 0.2365 | 0.2359 – 0.2371 | — | — | — |

`P(quarterback spread > coach spread) = 0.408` — a coin flip, if anything
leaning to the coach.

**Read the "no" column carefully, because it is not saying what it looks like.**
Both point estimates sit **above** the achievable-null bound: 0.0245 and 0.0252
against 0.0210. What fails is the *lower* bound of each interval, which is what
§6's claim rule requires. So the data hint that both a quarterback effect and a
coach effect are real and of similar size — and cannot confirm either, nor
separate them.

That is a different sentence from "there is no effect", and §5 committed to the
distinction before the fit: at this design's 63–67% power against document 08's
own reference effect, a failure to clear is the design running out of resolution.

### Gate A-1 — the grid instrument is doing what §4 said it would

| Quantity | Grid | NUTS | Relative gap |
|---|---|---|---|
| `sigma_qb` | 0.02446 | 0.02420 | **1.10%** |
| `sigma_coach` | 0.02518 | 0.02545 | **1.04%** |
| `sigma_e` | 0.23646 | 0.23657 | **0.05%** |

Well inside the 5% relative tolerance, and on real data rather than the synthetic
check §4 reported. **This is document 09 §9's corrective working as intended.**
An absolute tolerance of the kind that failed two candidates there — 0.01 on a
quantity of 0.024 — would have demanded agreement to 0.04%, which no finite
number of draws delivers. Stated relative to the quantity, the same two
instruments agree comfortably.

Note also the sampler: **zero divergences and ESS above 1,300** at 3,000 draws.
§6 pre-committed the larger draw budget on document 05 §8's evidence that this
geometry mixes slowly, and it was the right call — the equivalent crossed model
there returned ESS 289 at 1,000 draws.

### Gate A-3 — the game-state control shrinks both, and settles nothing

Restricting to one-score game states (`|score differential| ≤ 8`) and refitting
the slope inside the subset:

| Factor | Full sample | Competitive plays only |
|---|---|---|
| Quarterback | 0.0245 | **0.0178** |
| Head coach | 0.0252 | **0.0161** |
| `P(QB > coach)` | 0.408 | **0.513** |

Both spreads fall by roughly a third, in the same proportion, and the two factors
become if anything *more* indistinguishable. That is consistent with document
08 §9's finding that about a fifth of S3's persistence was playing close games —
the control removes real variation from both factors alike, and removes no more
from one than the other.

**Nothing survives the control that did not survive it before**, because nothing
cleared the bound in the first place. The secondary arm is reported because §6
committed to reporting it whatever it said.

### What this changes

1. **Document 08 §9's defect stays open, with a sharper statement.** It said
   S3's mechanism was "untested". It is now **tested and unresolved**, and the
   reason is sample size rather than model choice — which is a more useful thing
   to know, because it says what would fix it.
2. **The quarterback and the coach carry effects of the same size, and the data
   cannot separate them.** If both are real, a one-SD entity of either kind moves
   about 2.5 percentage points of win probability per game beyond what its
   expected-points production implies. That is a real football effect and it is
   why the round was worth running even having failed.
3. **Nothing in the simulator changes**, and nothing could have. Document 08 §6
   committed before any sequencing result existed that a sequencing channel has
   no branch point and earns no ledger row at any value of `w`. S3 is skill; skill
   already lives in `core`.
4. **The instrument is now validated on real data** and is available for any
   future crossed-Gaussian attribution in this project, at about one second a fit.
5. **What would fix it is more seasons, not a better model** — the same closing
   sentence document 05 §8 had to write about the interception attribution. Two
   attribution rounds, two null results, both traceable to entity counts rather
   than to design.

### An observation worth recording, though it proves nothing

Document 05 §8 and this document are now the project's two crossed-attribution
rounds, and they failed the same way: interceptions split 12.6% (quarterback) vs
12.3% (defense) with `P = 0.530`; leverage timing splits 0.0245 (quarterback) vs
0.0252 (coach) with `P = 0.408`. Both times two plausible owners came back
**almost exactly equal** and neither cleared its bound.

The dull explanation is that both designs are underpowered in the same way and an
underpowered crossed model shrinks both factors toward their common mean, which
would manufacture this pattern from nothing. That explanation is sufficient, and
it is the one to prefer. It is recorded here only because two rounds is the point
at which a pattern becomes worth *watching* — not the point at which it becomes
evidence.

### Defects added by this round

| Defect | Evidence | Status |
|---|---|---|
| **The round is unresolved for want of entities, not for want of a model** | Both point estimates exceed the null bound; both lower bounds fall short | **Open.** Needs more seasons. Recorded against document 08 §9's open defect |
| Both point estimates exceed the null bound while both intervals fail to clear it | §10's estimate table | **Open, and it is the honest shape of the result.** Reported as "cannot confirm", never as "no effect" |
| The grid's 0.035% edge mass is not zero | `grid_edge_mass` = 3.5 × 10⁻⁴ on the primary arm, 1.2 × 10⁻³ on the secondary | **Open, and small.** Some posterior mass sits at the grid's lower boundary, which is the log scale never reaching a true zero — document 05 §8's defect, inherited |
| A head coach is not the play-caller | §8 | **Open, and the largest limitation.** A coordinator effect is the model this round would want and nflverse carries no coordinator field |
| Two attribution rounds have now returned near-identical spreads for both candidate owners | This section's closing observation | **Open, and deliberately under-interpreted.** The dull explanation is sufficient |
