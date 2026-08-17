# 05 — The neutralization principle

*Written 2026-08-17, **before any Phase 2 model was fit**. This is the
simulator's foundations document: it states the one rule every component is
neutralized by, and fixes the two gates a component must pass before the rule is
applied to it. Committed before documents 06 and 07 exist.*

*Inputs: `docs/research/01`–`04`, all settled. Numbers quoted from document 04
are posterior means with 89% equal-tailed intervals.*

---

## 1. One-page story

### The question

Phase 1 ended with a classification — fumble recovery is luck, penalties are
skill, field goals and interceptions are both — and a warning that a binary
skill/luck switch is the wrong model for anything in the middle. Phase 2 has to
turn that into arithmetic that runs on a single game.

The temptation is three different mechanisms: flip fumbles, leave penalties
alone, and invent something bespoke for field goals. That would be three places
to be wrong and no way to compare them. This document says there is **one**
mechanism, and the differences between components are entirely differences in
one number that the Phase 1 models already estimated.

### The rule, in one line

> **Luck is the realized outcome minus its expectation at the entity's shrunk
> (posterior) rate.**

For a luck event `e` with a binary branch:

```
luck_epa(e) = (y(e) − p(e)) · swing(e)
```

where `y(e) ∈ {0,1}` is the branch that actually happened, `swing(e)` is the EPA
difference between the two branches, and `p(e)` is the probability of the
favourable branch **under the responsible entity's shrunk rate**. A game's
deserved EPA differential is then

```
neutralized_epa_diff = realized_epa_diff − Σ_e luck_epa(e)
```

Because each term is already measured as a deviation from an expectation, the
subtraction is exact and the ledger sums by construction. This is the same
identity `src/nfl_simulator/components.py` already uses for `fumble_luck` and
`fg_luck`; Phase 2 changes only where `p(e)` comes from.

### The one dial: where `p(e)` comes from

Every component's treatment is set by a single quantity — how far the entity's
own observed rate should be trusted over the league's. For the beta-binomial
hierarchies of document 04 that is a closed form:

```
p(e) = w · r̂_entity + (1 − w) · r̄_league        w = n / (n + κ)
```

`κ` is the concentration the Phase 1 models estimated and `n` is the entity's
opportunity count. The two familiar cases are the endpoints of this one
expression, not separate rules:

| Regime | What it means | What the rule does |
|---|---|---|
| `κ → ∞` (no entity skill) | every entity is the league mean | `w → 0`, `p(e) = r̄_league` — **full** neutralization |
| `κ → 0` (all skill) | the entity's observed rate is its true rate | `w → 1`, `p(e) = r̂_entity`, so `luck_epa → 0` — **no** neutralization |
| in between | some skill, some noise | **partial** neutralization, in exactly the measured proportion |

**This is the point of the document.** Full and partial neutralization are not
two policies requiring two justifications. They are one policy read at two
values of `w`, and `w` is not a choice — document 04 measured it.

### Five things to hold onto

1. One rule, one dial. `w = n/(n+κ)` is the whole per-component difference.
2. `w` is measured, not chosen. Moving it is a model change and needs its own
   pre-registration.
3. A component must pass the **branch-point gate** (§2) before `w` is even
   consulted. Arithmetic alone would neutralize penalties; the gate stops it.
4. The league rate is **class-specific** where the classes differ materially —
   fumble recovery ranges 40% to 76% by class, and a flat coin is wrong by up to
   26 points.
5. Uncertainty in the answer comes from two nested layers, the coin *and* the
   posterior for `p` itself. A deserve-to-win number reported without the second
   is overclaiming (document 04, closing item 4).

### Statistic convention

Posterior means with 89% equal-tailed intervals, matching document 03. The
simulator's headline number, **DTW%**, is a posterior probability that the team
deserved to win, defined in §4.

---

## 2. The two gates a component must pass

Order matters. Gate A is qualitative and comes first; Gate B is the arithmetic
above and only runs on components that survive Gate A.

### Gate A — the branch-point gate

> **Is there a moment where the outcome is resolved by a mechanism outside
> either team's control, conditional on the state both teams created?**

A loose ball on the turf is such a moment: both teams caused the fumble, and
then an oblong object bounces. A kick in flight is a weaker one: the kicker
caused most of it, and then it drifts. A false start is **not** such a moment.
There is no post-hoc branch — a lineman moved, and the flag is the officials'
description of that, not a coin resolving afterward.

Failing Gate A means a component is not neutralized *at any value of `w`*.

**This gate is load-bearing, and the penalty row proves it.** Run the Gate B
arithmetic on penalties and you get `w ≈ 0.42`–`0.46` (§3), which would
neutralize almost half of every game's penalty EPA. That is a real number
correctly computed from a model that passed all its checks, and applying it
would be a serious error: it would credit a disciplined team's *good* penalty
game to luck. Persistence statistics cannot detect the absence of a branch
point. Only the mechanism story can, which is why it goes first.

### Gate B — the shrinkage gate

For components that pass Gate A, `w = n/(n+κ)` sets the degree of neutralization,
with no further discretion. `κ` comes from the component's fitted hierarchy and
`n` from the entity's opportunity count in the sample.

Two consequences worth stating before they surprise anyone:

- A component can pass Gate A and still be neutralized by almost nothing, if `w`
  is near 1. That is the correct outcome, not a failure to find luck.
- `w` is entity-specific, because `n` is. A kicker with 200 career attempts is
  trusted more than a rookie with 12, from the same `κ`. This falls out of the
  formula and requires no special case.

---

## 3. Per-component treatment table

`w` is computed at the median opportunity count per entity in the fitted sample.
Rows marked *pending* are resolved by the step-3 attribution round and step-4 FG
model, and this table is updated with their verdicts before the simulator is
built.

| Component | Gate A (branch point) | κ | typical n | **w** | Treatment | Source |
|---|---|---|---|---|---|---|
| **Fumble recovery** | **Pass** — loose ball, nobody controls the bounce | 1,408 | 15 / team-season | **0.011** | **Full.** `p` = league rate for the fumble's *class* | 04 |
| **Field goal** | **Pass** — ball in flight, partly outside the kicker | pending | ~30 / kicker-season | pending | **Partial** vs that kicker's shrunk make probability at that distance | step 4 |
| **Interception** | **Pass** — given an interception-worthy throw, whether it is caught is partly the defender's luck | 71.5 | 24 / team-season | **0.251** | **Partial**, pending re-attribution — see below | 04, step 3a |
| **Penalty (pre-snap)** | **Fail** — no post-hoc branch | 3,967 | 2,813 plays | (0.415) | **None** | 04, step 3b |
| **Penalty (judgment)** | **Fail** — officiating discretion measurably does not add noise (12.5% relative spread vs 14.0% pre-snap) | 3,243 | 2,813 plays | (0.465) | **None**, pending subtype check | 04, step 3b |
| **Return yardage** | pending | pending | pending | pending | pending | step 3c |

Parenthesised `w` values are shown to make §2's argument concrete. They are not
used: those rows failed Gate A.

### Fumble recovery — full, but class-specific

`w = 0.011` means a team-season's observed recovery rate carries about one
percent of the information about its true rate; the league does the rest. This
is why document 04 found Buffalo's 10-of-12 shrinking to 48.0%. In simulator
terms the entity term vanishes and `p(e) = r̄_league(class(e))`.

The class split is not optional. From document 01:

| Fumble class | n | League own-recovery rate |
|---|---|---|
| run, normal play | 1,149 | 40.3% |
| pass, normal play | 2,892 | 45.3% |
| kickoff | 182 | 46.2% |
| punt (muffed return) | 672 | 64.4% |
| run, aborted snap | 946 | 76.2% |
| pass, aborted snap | 68 | 100% |

A flat 50/50 would book a fake 26-point bad-luck charge against every offense
that recovered its own botched snap. `src/nfl_simulator/components.py` already
carries this table; the simulator consumes it unchanged.

### Interceptions — partial, and the entity is the open question

`w = 0.251` at the team-season grain: about three-quarters of a team's observed
interception-worthy-throw conversion rate is noise, one quarter is real. But
document 04's defect register flags that a team-season pools every quarterback
who played, so *whose* skill the 14.3% relative spread belongs to is unresolved.

Neutralizing at a team-level `w` when the skill actually belongs to a specific
quarterback would systematically misprice both. Step 3a settles this with
crossed quarterback and defense grouping factors, and this row is not final until
it does. Document 04 named this the highest-value open question in the project
and, at 19% of outcome variance, it is.

### Penalties — none, and the subtype caveat

Document 04 settled the pre-snap/judgment split: both are repeatable team
traits, and the officiating-noise hypothesis was not supported. Gate A rules the
whole component out independently.

One narrower hypothesis survives and is tested in step 3b: **offensive holding
specifically may be random even though pooled judgment calls persist**, on the
theory that holding occurs on most plays and the flag is a sampling of it. If
that lands, holding is the one penalty subtype with a genuine branch point, and
this row splits. Nothing else about the penalty treatment is in question.

---

## 4. From neutralized EPA to a deserve-to-win number

### Points, anchored on the actual result

Document 01 established that EPA differential and points margin are nearly the
same quantity (r² = 0.991, only 0.8% of margin variance invisible to EPA). The
simulator uses that to express its answer on the scoreboard's scale while
staying anchored to what actually happened:

```
deserved_margin = actual_margin − (Σ_e luck_epa(e)) · points_per_epa
```

`points_per_epa` is the slope of margin on EPA differential, fit on the full
sample. Anchoring on `actual_margin` rather than predicting a margin from
neutralized EPA matters: it means a game with no luck events returns its actual
result exactly, which is the smoke test in the handoff plan's verification list.

### DTW% and its interval — two nested layers

A point estimate of `deserved_margin` answers "what was the luck worth" but not
"who deserved to win", because the counterfactual is itself uncertain. The
simulator bootstraps, and the bootstrap has two layers:

1. **Draw the rate.** Sample `(μ, κ)` — and for field goals the kicker effects —
   from the fitted posterior, giving a draw of `p(e)` for each event. This is
   document 04's closing instruction: report uncertainty from the population
   posteriors, not from point estimates.
2. **Flip the coin.** Draw `y*(e) ~ Bernoulli(p(e))` for every luck event and
   recompute the margin with those branches in place of the realized ones.

```
DTW% = P(deserved_margin* > 0)
```

over the replicates, reported with an 89% interval on the margin distribution.
A game with no luck events collapses to a degenerate distribution at the actual
margin, and DTW% is 100% or 0% — correctly, since there was nothing to adjudicate.

Layer 1 is what stops the simulator from reporting a suspiciously tight interval
around a quantity estimated from 15 fumbles per team-season.

### The luck ledger

Every event that contributes a nonzero `luck_epa` is emitted as a ledger row —
play id, component, class, `y`, `p`, `swing`, `luck_epa`, and the team charged.
The ledger's `luck_epa` column must sum to the total adjustment applied to the
margin. That is an identity, not a tolerance, and the simulator asserts it.

---

## 5. Known-defect register

| Defect | Evidence | Status |
|---|---|---|
| The entity's rate is estimated from a sample containing the game being adjudicated | A kicker's game contributes ~2 of ~100+ career attempts; a team-season's game contributes ~1 of ~15 fumbles | **Open, bounded.** Contamination is O(1/n) and always shrinks the measured luck toward zero (self-fulfilling rates), so the bias is conservative. A leave-one-game-out refit is Phase 3 |
| `points_per_epa` is a single global slope | Document 01 found r² = 0.991, so the residual is small but real (≈0.8% of margin variance) | **Accepted.** The residual is scoring-environment conversion, not luck being adjudicated, so it is deliberately excluded from the DTW interval |
| Interception entity is unresolved | Team-season pools quarterbacks (document 04) | **Open.** Blocks the INT row of §3; step 3a resolves it |
| Fumble classes are hand-assigned | Our taxonomy from `play_type` × `aborted_play`, not the NFL's | **Open.** Class list is in §3 so it can be argued with |
| Gate A is a judgment, not a measurement | No statistic can detect the absence of a branch point | **Accepted, by design.** Stated in §2 so it is arguable rather than hidden |
| Weather is absent from the FG model | Deferred per the handoff plan | **Open.** A windy 50-yarder is priced as a calm one, overstating the kicker's bad luck |
| Simultaneous luck events are treated as independent coins | Two fumbles in one game are drawn independently | **Accepted.** They are separate physical events; correlation would have to come through the shared `p` draw, which layer 1 already supplies |

---

## 6. Constants appendix

| Constant | Value | Where it lives |
|---|---|---|
| Fumble class recovery rates | 40.3% / 45.3% / 46.2% / 64.4% / 76.2% / 100% | `src/nfl_simulator/components.py` (`fit_fumble_baseline`) |
| Fumble `κ` | 1,408.3 | `research/outputs/03_bayesian_rates.json` |
| INT conversion `κ` | 71.5 | same |
| Penalty `κ` (pre-snap / judgment) | 3,967.1 / 3,243.1 | same |
| `points_per_epa` | fit at simulator build time | `src/nfl_simulator/simulator.py` |
| Bootstrap replicates | set in step 5, power-checked | `src/nfl_simulator/simulator.py` |

Verdicts for the pending rows land in this document's §3 as step 3 and step 4
complete. The rule in §1 does not change; only the table does.
