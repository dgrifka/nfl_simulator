# 45 — Dropped-pick study, round 2: pooling redesign, pre-registered

*Written 2026-08-27 in a Fable 5 brainstorm after reading
`results-2026-08-27-exp1.md` and document 44, **before any fit**. This is an
amendment to document 43, not a replacement: the question, the DAG, the
covariates, the gate form and the readings in 43 §0 all stand. Three things
change, each because round 1 showed it had to, and each is written down here
so that goalpost integrity is checkable by commit archaeology. Document 32's
closure is untouched by any outcome.*

*Inputs: documents 43 (the pre-registration), 44 (round 1's record), 17 §3
(the deflection cross-tab this round's hindsight probe extends).*

---

## 1. What round 1 taught, in three sentences

The study was unresolvable because document 43 §4's "≥ 20 worthy throws per
QB-season" floor was written as a unit definition but acted as a row filter,
cutting 2,969 throws to 1,145 and the median defence-season from 22 chances
to 7 — power at the reference fell to 0.36 against the 0.80 bar. The floor
was never needed: a hierarchical model already pools thin levels toward the
mean, which is the whole point of the hierarchy. Separately, Gate C-1 failed
by a hair on one nuisance parameter (`σ_q`: r̂ 1.0105 vs 1.01; ESS 387/345 vs
400), which longer chains fix.

## 2. The amendments

**A-1 — no floor on the gate arm.** `MIN_QB_WORTHY` is removed. Every
QB-season with at least one worthy throw is its own level in arm 2 and in
arm 3's crossed grid; shrinkage does the pooling. Expected frame: 2,969
throws (the arm 2 frame of round 1), 128 defence-seasons at a median of 22
chances, roughly 250 QB-season levels. **Stop and ask if the arm 3 frame
differs from the arm 2 frame by a single row.** `MIN_QB_ATTEMPTS = 200` for
the arm 1 worthy-rate question is unchanged — that grain is a rate per
attempt and needs a denominator, and it resolved (power 0.78 / 0.94).

**A-2 — sampler spec.** Arms 2 and 2b: 4 chains × 2,000 draws after 2,000
tuning, `target_accept = 0.9`, nutpie. Gate C-1's bars are unchanged and
apply to every parameter, as before.

**A-3 — a hindsight probe for the "worthy" flag.** Round 1 tested charter
hindsight on `is_catchable_ball` / `is_contested_ball` (immaterial) but not
on the selection variable itself. If a charter marks a throw worthy *because*
it was intercepted, then selecting on "worthy" is conditioning on a
descendant of the outcome, which inflates conversion and biases every
coefficient. Probe, on all interceptions 2022–2025, using document 17's
second-toucher channel (`pass_defense_1_player_id != interception_player_id`):

| Quantity | Reading |
|---|---|
| `p(worthy \| INT, second toucher)` | A deflected pick is mostly a bounce; a charter without hindsight should call the *throw* worthy less often here |
| `p(worthy \| INT, no second toucher)` | The comparison group |
| Share of all INTs charted not-worthy | Context; document 17 §3 reported the flag misses 7 in 10 deflected picks |

**Pre-committed reading:** if the deflected-pick worthy rate is at or above
the clean-pick rate, hindsight is suspected and every conversion number in
this study carries that caveat in words; if it is materially below (document
17's 28.3% coverage predicts it will be), the flag is behaving as a judgement
of the throw and the selection is defensible. Reported, not gated — this
probe cannot prove absence of hindsight, only catch its gross form.

## 3. Everything that does not change

Data, join, guards (80,785 / 2,997 / 1,454, `p̄` 0.485 ± 1 pp, 128
defence-seasons), covariates and exclusions (43 §4), the models (43 §5, with
A-2's step counts), the power instrument and scenarios (43 §6, re-run on the
new frame), the gates C-1 / C-2 / C-3 / D-1 and their bars (43 §7), the
readings (43 §0), the decision rule (defence-season × QB-season first, pooled
only if it fails C-3), and §8's rollback: nothing ships into the simulator.
Seed `20260827`, `DATASETS = 400`.

Round 1's arm 1 numbers are not re-run; they stand in document 44 §3.

## 4. Power, before thresholds — round 2 table

Filled from `research/63_dropped_pick_power_r2.py` and **committed before
`research/64_dropped_pick_confounds_r2.py` fits anything real.** The
crossed grid's cost scales with level count; if a fit exceeds 5 s the script
reports it and `DATASETS` stays 400 regardless — wall clock is not a reason to
change an instrument.

| Entity design | Null bound (90th pct) = threshold | Power at 5% | **at 12.5%** | at 25% | at 50% | Resolvable? |
|---|---|---|---|---|---|---|
| Residual, defence-season × QB-season (no floor) | *to fill* | | | | | |
| Residual, defence pooled × QB-season (no floor) | *to fill* | | | | | |
| Residual, QB-season `σ_q` (no floor) | *to fill* | | | | | |

For comparison, round 1 (with the floor): 9.41 pp / 0.362; 8.09 pp / 0.555;
— / 0.578.

**Pre-committed expectation, so a surprise is recognisable as one:** with
~22 chances per defence-season instead of 7, binomial noise per unit drops
from ~19 pp to ~10.6 pp against a 6.1 pp reference effect; power at the
reference should land well above round 1's 0.36 but may still miss 0.80 at
the season grain. The pooled grain (32 defences × ~93 chances) is the one
most likely to resolve. If **both** fail C-3 again, the study is unresolvable
with four seasons of FTN and says so; there is no further redesign to try
short of more seasons.

## 5. What this round licenses, by outcome

The same three rows as 43 §0, with one addition: whichever row lands, arm 2's
`β̂` from a fit that passes C-1 becomes quotable, and the expected-conversion
diagnostic (the 2026-08-27 fork's avenue (1) — sum of per-throw `p̂` over the
interceptable throws a team's opponent threw, against actual picks, priced by
each play's own pick-versus-escape EPA difference, presented as the
*offence's* fortune) can be built in a later round. It stays a reported
diagnostic beside the red-zone and late-down gaps, never a ledger row.

## 6. Constants

| Constant | Value | Where |
|---|---|---|
| `MIN_QB_WORTHY` | **removed** (A-1) | `63`, `64` |
| `MIN_QB_ATTEMPTS` | 200 (unchanged, arm 1 only) | document 43 |
| `DRAWS` / `TUNE` / `CHAINS` / `TARGET_ACCEPT` | 2,000 / 2,000 / 4 / 0.9 (A-2) | `64` |
| Second-toucher channel | `pass_defense_1_player_id != interception_player_id` | document 17; `64` |
| Everything else | as document 43 §10 | |
