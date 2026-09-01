# 45 — Dropped-pick study, round 2: pooling redesign, pre-registered

*Written 2026-08-27 in a Fable 5 brainstorm after reading round 1's results
and document 44, and **before any fit** — both orderings checkable in
`git log --diff-filter=A`, which has document 44 landing before this document
and this document before `research/63_dropped_pick_power_r2.py`, the first of
round 2's fitting scripts. This is an amendment to document 43, not a
replacement: the question, the DAG, the covariates, the gate form and the
readings in 43 §0 all stand. Three things change, each because round 1 showed
it had to, and each is written down here so that goalpost integrity is
checkable by commit archaeology. Document 32's closure is untouched by any
outcome.*

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
| Residual, defence-season × QB-season (no floor) | **5.920 pp** | 0.200 | **0.892** | 1.000 | 1.000 | **Yes** |
| Residual, defence pooled × QB-season (no floor) | **5.060 pp** | 0.303 | **0.953** | 1.000 | 1.000 | **Yes** |
| Residual, QB-season `σ_q` (no floor) | **6.889 pp** | 0.200 | **0.780** | 1.000 | 1.000 | **No** |

For comparison, round 1 (with the floor): 9.41 pp / 0.362; 8.09 pp / 0.555;
— / 0.578.

Powers are the fraction of 400 simulated datasets whose upper bound cleared the
threshold, printed to three places as the script prints them. The first row's
0.892 is **357 of 400** exactly (0.8925); every other cell is exact at three
places.

Filled 2026-08-27 from `research/63_dropped_pick_power_r2.py` →
`research/outputs/63_dropped_pick_power_r2.json`, **before**
`research/64_dropped_pick_confounds_r2.py` existed as a fit. Nothing in
document 43 §7 moved; the gates and their bars are as committed.

**The frame these rows are powered on.** 2,969 throws — arm 2's frame exactly,
`arm3 rows == arm2 rows: True` — **128** defence-seasons (round 1's gate arm saw
125), **32** pooled defences, **280** QB-season levels. Chances per
defence-season: median **22**, min 5, max 43, against round 1's median of 7.
Worthy throws per QB-season: median **9**, min **1** — and that minimum is the
amendment working as intended, since a one-throw QB-season is now a level the
hierarchy shrinks rather than 20 throws deleted around it. Conversion in the
frame 0.4894, mean `p̂` 0.4937.

**Cost, measured.** 14 cells × 400 datasets = 5,600 crossed fits in **3,730 s of
wall clock** across a pool of 8 workers (about 6.6 hours of CPU). Per-fit cost,
as the mean within a cell: median **4.27 s**, min 2.57 s, max 5.08 s — the
128 × 280 designs sit just under the 5 s note and the pooled 32 × 280 design
just over half of it. `DATASETS` stayed 400, as this section requires.

**Both gate designs clear Gate C-3, and the expectation below was beaten in the
direction it hedged against.** The pre-committed text predicted the pooled grain
would be "the one most likely to resolve" and that the season grain "may still
miss 0.80". The season grain reached **0.892** — resolvable — so the study's
primary design, the one document 43 §7's decision rule reaches for first, is the
one that answers. The `σ_q` row is the only one that stays under the bar, at
0.780, and it is a nuisance parameter with no pass rule attached to it.

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

## 7. Outcome

Appended 2026-08-27, after the run. §§1–3 and §5–6 were not edited; §4 carries
only the Part A fill this section's own instruction required. Full record:
`docs/research/46-dropped-pick-round2.md`; the round's results file and
session log are not part of the public record.

| Gate | Statistic | Verdict |
|---|---|---|
| C-3, defence-season × QB-season | Power at 12.5% relative **0.892** (round 1: 0.362) | **PASS** — A-1 resolved the study |
| C-3, defence pooled × QB-season | Power at 12.5% relative **0.953** (round 1: 0.555) | **PASS** |
| C-3, QB-season `σ_q` | Power at 12.5% relative **0.780** (round 1: 0.578) | **FAIL** |
| C-1, arm 2 sampler | 0 divergences, max `r_hat` **1.0070** (`sigma_q`), min ess_bulk 587, min ess_tail 522, **0 of 429** parameters over a bar | **PASS** — A-2 closed round 1's failure |
| C-1, arm 2b sampler | 0 divergences, max `r_hat` 1.0074, min ess_bulk 467, min ess_tail 415, **0 of 427** over a bar | **PASS** |
| C-1, grid self-checks | Crossed edge mass 1.5e-03 and 3.8e-03 | **PASS** |
| C-1, arm 2 / arm 3 cross-check | `σ_d` 89% upper bound: arm 2 **9.08 pp** vs arm 3 **8.04 pp** (gap **1.04 pp**) and **5.01 pp** pooled (gap **4.07 pp**), tolerance 1.0 pp | **FAIL** on both — reported per document 43 §5, not worked around |
| C-2, defence-season × QB-season | 89% upper bound **8.04 pp** vs threshold 5.92 pp | **FAIL** by 2.12 pp — **reportable**, C-3 passed |
| C-2, defence pooled × QB-season | 89% upper bound **5.01 pp** vs threshold 5.06 pp | **PASS** by 0.05 pp — **reportable**, C-3 passed |
| C-2, QB-season `σ_q` | 89% upper bound **7.99 pp** vs threshold 6.89 pp | FAIL — not reportable, C-3 failed |
| D-1, worthy-rate spreads | Not re-run (§3); document 44 §3 stands | — |
| A-3, hindsight probe | `p(worthy \| INT, deflected)` **0.717** vs clean **0.888**, gap **−17.1 pp**; 14.0% of charted INTs not worthy | Reported — the flag is a judgement of the throw; selection defensible |

**§0's reading, as committed: the third row** — C-3 passes, C-2 fails on the
design the decision rule reads first. *"Some defences finish more of their
chances, repeatably: ball-hawking is skill... avenue (3) is dead on the
evidence."* Document 46 §7 carries the licensed wording, **and flags that the
word "repeatably" is contradicted by the pooled design (C-2 pass, C-3 0.953) and
by the +0.065 season-to-season correlation** — a fork the decision rule does not
resolve because it never consults the pooled row once the first design clears
C-3. Two wordings are offered there; the choice is the maintainer's.

**Two questions are open and both are the maintainer's:** the Gate C-1 cross-check
failure (document 46 §4a — the pooled comparison is grain-mismatched by
construction, and fixing that is a change to document 43 §5's committed text),
and which §7 wording the game page carries.

**Commits.** `4cfaa66` this amendment · `b645105` Part A, the power table and its
thresholds · `824cc72` Part B, the fits · Part C, the record, results file and
queue. Branch `docs/dropped-pick-confounds`, unmerged; the maintainer merges.
`git diff main -- src/` is empty; 502 tests pass.
