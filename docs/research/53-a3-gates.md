# 53 — Amendment A-3's gates: two answered, one blocked

*Written 2026-08-27. The results record for round 5 of the dropped-pick study,
pre-registered in document 52 (amendment A-3) and run on the branch
`feat/dropped-pick-variant` in four parts: ruling R-3 on V-8, gate G-1's
week-out refits (`research/69_dropped_pick_weekout.py`), gates G-2 and G-3
(`research/70_dropped_pick_sensitivity.py`), and this record.*

*Inputs: documents 52 (the amendment and its gates), 49 and 50 (round 4's
pre-registration and record), 43-48 (the study), 05 §3 and §7 (the treatment
table and the materiality floor), 28 (the consistency objection), 33 (the
element-wise lesson), 31 (v1.3's replay checks).*

---

## 1. The three gate lines, and what they license

```
G-1: BLOCKED — 7 of 18 week-out folds miss Gate C-1 (weeks 1, 3, 7, 9, 13, 14, 16).
     The agreement statistic is NOT computed.
G-2: flat-swing bucket moves 129 of 137 binned (0.94) -> PASS   [bar >= 0.50]
G-3: median |dDTW| affected 1.62 pp vs median half-width 0.56 pp -> PASS by 1.06 pp
```

**A-3 is not enacted, and the reason is not a failed gate.** Document 52 §5's two
substantive gates both pass, and comfortably: the variant's effect is the coin
rather than the goal-line cell (G-2), and it moves the verdict by more than the
incumbent's own uncertainty (G-3). What is missing is the *precondition* to G-1 —
seven of the eighteen week-out fits did not clear Gate C-1's sampler bars at the
pre-registered spec, and a held-out read is only as good as the fits it comes
from. G-1's statistic was therefore not computed rather than computed on fits
that had not converged.

So the treatment table stays where round 4 left it, one clause richer: the
component is a **variant**, its pricing and materiality are established, its
self-fulfilment bound is unmeasured, and clause 3's mirror is untouched. Document
52 §5's `G-4` language — *"the treatment table reads `variant (A-3 pending
mirror)`"* — describes the state after G-1 passes, which is not this state.

**v1.3 did not move.** The V-1 replay ran at the end of both scripts and reads
`0.00e+00` over 2,761 games on the deserved margin, the DTW% and both interval
bounds. Handoff constraint 1, satisfied twice.

| Gate | Statement | Bar | Result |
|---|---|---|---|
| **R-3** | ruling on V-8's 1.1 pp breach | the maintainer's, on the record | **Made.** Immaterial; the bound stands unamended (documents 49 §10, 50 §2) |
| **G-1** | self-fulfilment, clause 5 | agreement ≥ 0.90 **and** median \|ΔDTW\| < 1.0 pp | **BLOCKED** — 7 of 18 folds miss Gate C-1; statistic not computed |
| **G-2** | pricing sensitivity | ≥ 0.50 of the 137 binned bucket moves | **PASS** — 129 of 137, 0.94 |
| **G-3** | materiality, clause 4 | median move ≥ median half-width | **PASS** — 1.62 pp against 0.56 pp |
| **G-4** | the receiver mirror | its own gates, round 6 | **Not run** — round 6, and it now waits on G-1 too |
| **V-1** | v1.3 untouched | max \|Δ\| 0.00e+00 over 2,761 games | **PASS** twice, 0.00e+00 |

## 2. G-1 — eighteen fits, seven of which did not converge

### 2a. What was built

Document 52 §5's instruction is eighteen refits of the dropped-pick model, one
week of season masked out of each (all four seasons' week `w` together), reading
each game's `u_d` from the fit that excluded its week. `research/67`'s fit became
`fit(frame, seed)` with injectable level labels and a reusable summary builder;
`research/69` masks the rows and drives the folds. Three construction choices
were made where document 52 left room, each disclosed rather than absorbed:

1. **The covariate scale is the full frame's, not each fold's.** Handoff
   constraint 3 says only the row mask changes, and `research/61.design_matrix`
   already takes a `reference` frame for exactly this. Standardising each fold on
   its own rows would have put every fold's `beta` on its own scale and mixed a
   reparameterisation into a gate about the entity effect. The stored
   standardisation constants are the full frame's, so the read side centres a
   held-out throw on the scale its fold actually used.
2. **`u_d[k]` names the same defence-season in all eighteen fits.** Level codes
   are built against the full frame's level list, so a level whose rows were all
   held out shrinks to the prior instead of renumbering the vector. **No fold
   lost a defence-season entirely** — the "no rows" column below is 0 for all
   eighteen — so that safeguard never had to fire on the entity the gate is
   about. It did fire on QB-seasons, up to 19 of 280 in the week-18 fold, which
   is what a week of rested starters looks like.
3. **The swing table is the in-sample table, unchanged across folds.** G-1 holds
   out the *entity effect*; the bin prices are a pooled descriptive quantity and
   G-2 is the gate that interrogates them.

### 2b. The eighteen folds, verbatim

```
  week   rows  div  max r_hat           on  ess_bulk  ess_tail  sigma_d  C-1
     1  2,777    0     1.0122      sigma_d       564       500   0.2454  FAIL
     2  2,779    0     1.0060      sigma_q       797       784   0.2500  PASS
     3  2,786    0     1.0107      sigma_q       455       328   0.2355  FAIL
     4  2,788    0     1.0055      sigma_q       642       507   0.2843  PASS
     5  2,818    0     1.0055      sigma_q       880       952   0.2388  PASS
     6  2,821    0     1.0055      sigma_d       768       579   0.2534  PASS
     7  2,807    0     1.0146      sigma_d       733       688   0.2576  FAIL
     8  2,813    0     1.0094      sigma_q       715       558   0.2807  PASS
     9  2,841    0     1.0102      sigma_q       667       627   0.2587  FAIL
    10  2,817    0     1.0040        alpha       654       592   0.2604  PASS
    11  2,833    0     1.0041      sigma_q       727       675   0.2671  PASS
    12  2,825    0     1.0051    beta[x17]       580       484   0.2937  PASS
    13  2,834    0     1.0138      sigma_d       389       268   0.2050  FAIL
    14  2,833    0     1.0060        alpha       353       245   0.2205  FAIL
    15  2,798    0     1.0029      sigma_q       698       600   0.2547  PASS
    16  2,795    0     1.0055      sigma_q       448       359   0.2578  FAIL
    17  2,825    0     1.0053      sigma_d       682       600   0.2524  PASS
    18  2,830    0     1.0067      sigma_q       547       466   0.2808  PASS
  C-1 over the folds: FAIL — 7 of 18 folds miss the bars, weeks [1, 3, 7, 9, 13, 14, 16]
```

Gate C-1's bars are 0 divergences, r̂ < 1.01, `ess_bulk` > 400 and `ess_tail` >
400, over every parameter. Read the failures precisely, because their shape
decides what to do about them:

- **No fold diverged. Not one, out of eighteen.** Total divergences across all
  eighteen fits: **0**. This is not a geometry problem.
- **Five folds miss on r̂ alone**, at 1.0102–1.0146 against a 1.01 bar — a miss of
  0.0002 to 0.0046 — and in every case on a *variance component*, `sigma_d` or
  `sigma_q`.
- **Three folds miss on ESS-tail** (weeks 13, 14, 16 at 268, 245 and 359 against
  a 400 bar), weeks 13 and 14 on `ess_bulk` too (389, 353).
- The extremes over all eighteen folds: max r̂ **1.0146**, min `ess_bulk` **353**,
  min `ess_tail` **245**.

This is the failure mode amendment A-2 was written for. Round 1 of the study
failed C-1 on `sigma_q` and the answer was longer chains; A-2's 4 × 2,000 after
2,000 tuning fixed it, and the *default* fit sits at r̂ 1.0070 on `sigma_q` —
already within 0.003 of the bar. Take 128 to 192 rows out of 2,969 and the
variance components have marginally less to identify them with; seven of eighteen
folds crossing a line the default run was already leaning on is what that looks
like. It says nothing about the model and everything about how long the chains
are.

### 2c. What is therefore not in this document

Document 52 §5's G-1 statistic — bucket agreement between the in-sample and
week-out variants, and the median `|ΔDTW|` between them — **is not computed**, and
neither is the week-out variant's own bucket-move count, its median `|ΔDTW|`
against v1.3, or what it looks like on `2025_17_DET_MIN`, `2025_13_DEN_WAS` and
`2022_13_WAS_NYG`. Every one of those would have been read off a set of fits
containing seven that had not converged, and a self-fulfilment bound computed on
unconverged fits is worth less than no bound: it would carry a number the maintainer could
quote. `research/69` writes its JSON with `"g1_computed": false` and stops.

**All eighteen traces and summaries are on disk**
(`trace_dropped_pick_wk{1..18}.nc`, `dropped_pick_summary_wk{1..18}.json`),
gitignored and regenerable, so whichever way the fold spec is settled, the
eleven folds that passed need no refit unless the spec itself changes. The
eighteen fits cost **188 seconds** of wall clock in total — the 90-minute budget
handoff constraint 6 set was never in danger, which is the useful fact about the
cost of doing them again.

### 2d. One thing the folds said anyway, and it is not the gate

`sigma_d`'s posterior mean across the eighteen folds runs **0.205 to 0.294**, with
the default fit at 0.254. That spread is *not* a G-1 result and must not be read
as one — it mixes real fold-to-fold variation with the sampler noise the failures
above are made of. It is recorded because it is the first look anyone has had at
how stable the defence-season spread is under resampling, and because whatever
G-1 eventually reports will have to sit beside it.

## 3. G-2 — the coin, not the goal-line cell

Every throw repriced at document 47 §3's **pooled −3.55 EPA** instead of its own
bin, on the in-sample model, same seed, same games:

| Cell | binned | flat |
|---|---|---|
| 1–33 yd, down 1–2 | −5.04 | −3.55 |
| 1–33 yd, down 3–4 | −3.01 | −3.55 |
| 34–66 yd, down 1–2 | −3.82 | −3.55 |
| 34–66 yd, down 3–4 | −2.27 | −3.55 |
| 67–99 yd, down 1–2 | −4.03 | −3.55 |
| 67–99 yd, down 3–4 | −2.49 | −3.55 |

```
G-2: flat-swing bucket moves 129 of 137 binned (0.94) -> PASS   [bar >= 0.50]
```

**129 of 137, a share of 0.94 against a 0.50 bar.** Document 52 §5's claim — that
the variant's effect is the coin flip, not the bin table's most extreme cell —
holds with a wide margin. Flatten the pricing entirely and 94% as many games
still cross a verdict bucket.

**And the two move sets are not the same set.** Document 33's lesson applies
wherever a pre-registration quotes a ratio of counts, so the overlap was computed
rather than inferred: **116** games move under both pricings, **21** under the
binned swing only, **13** under the flat swing only. Element-wise agreement
between the two move sets is 0.77, not 0.94. The gate's statistic is the ratio,
by pre-registration, and 0.94 is the number it reports; the 116/21/13 split is
what that ratio does not say, and it is here so nobody later mistakes the one for
the other.

## 4. G-3 — the materiality floor

On the **1,033 affected games**, `|ΔDTW|` under the variant against v1.3's own 89%
half-width, `(high − low) / 2` per game:

```
G-3: median |dDTW| affected 1.62 pp vs median half-width 0.56 pp -> PASS by 1.06 pp
  for scale, the same two on the mean rather than the median: 6.96 pp against 1.91 pp;
  the move exceeds a game's own half-width in 86.8% of affected games
```

| Statistic | median move | median half-width | verdict |
|---|---|---|---|
| **the 1,033 affected games** (the gate) | **1.62 pp** | **0.56 pp** | **PASS** by 1.06 pp |
| the 137 bucket-move games (*reported only*) | 21.45 pp | 4.95 pp | clear by 16.50 pp |

The 137-mover line is **reported only and is not the gate**, exactly as document
52 §5 specifies. It is included because it is the population a reader's eye goes
to and it would be worse to leave the number to be guessed at.

**Document 52 §5's pre-committed note, quoted in full:**

> *Pre-committed note:* round 4's numbers put this close — 1.62 pp against a mean
> full width of 3.83 pp — and it may fail. If it fails, A-3 is **not enacted** for
> dropped picks on median grounds, the variant stays a labelled variant, and the
> tail (12% bucket moves) is reported beside the failure. The floor is not
> re-tuned for this component.

**It did not fail, and the reason the note expected it to is worth writing down.**
The note compared round 4's *median* move (1.62 pp) against a *mean full width*
(3.83 pp) — two different statistics of two different shapes. Document 05 §7's
floor, which document 52 §5 restates correctly one sentence earlier, is the median
move against the median **half**-width: 1.62 pp against 0.56 pp. The floor is
unchanged and un-retuned; the arithmetic was simply done on the gate's own terms
rather than the note's. For completeness the gate also clears on the mean pairing
(6.96 pp against 1.91 pp) and on 86.8% of individual affected games, so the pass
does not hang on the choice of central tendency.

## 5. What the maintainer decides next

**The decision this round produces is about G-1's fold spec, and nothing else.**
G-2 and G-3 are answered; A-3's enactment still waits on G-1 and then on clause
3's mirror. Three ways forward, none taken here:

1. **Lengthen the folds' chains and re-run G-1.** The A-2 precedent is exact:
   round 1's `sigma_q` failure was a chain-length failure and longer chains fixed
   it. Because handoff constraint 3 fixes the folds at A-2's spec, this is a
   *change to a pre-registered constant* and needs to be written as an amendment
   (or a ruling naming the new spec) before it is applied, not after. Cost: the
   eighteen fits took 188 seconds at 4 × 2,000; doubling to 4 × 4,000 after 3,000
   tuning is a handful of minutes, plus the audit passes. Fastest honest route to
   a G-1 number.
2. **Rule the misses immaterial, as R-3 ruled V-8's.** The case is arguable — 0
   divergences in eighteen fits, r̂ at worst 1.0146, and the reported-parameter
   ESS mostly comfortable — but it is weaker than R-3's was: three folds are
   genuinely thin in the tail (245–359 effective draws), and the parameters
   missing are `sigma_d` and `sigma_q`, the two the gate's own quantity is built
   from. If this route is taken, the ruling should say so out loud.
3. **Accept the blocked line and stop the round here.** G-2 and G-3 are on the
   record; A-3 stays unenacted; the variant stays a labelled variant. This costs
   nothing and delivers no self-fulfilment bound, which is the one thing clause 5
   requires.

Recommendation, offered and not taken: **(1)**. It is the only route that ends in
the number clause 5 asks for, the precedent for it already exists in A-2, and at
three minutes a pass the compute argument for the alternatives is thin. Whether
it is written as an amendment or as a ruling naming the spec is a
pre-registration question, and document 52's own §5 shows the shape.

Beyond that, unchanged: the receiver mirror (clause 3) is round 6 and stays
**Parked**, because document 52 §5's condition for queueing it was G-1 through
G-3 passing and G-1 has not.

## 6. Register

| Item | Status |
|---|---|
| V-8's 1.1 pp breach on 2022 NYG | **Closed** — ruling R-3, immaterial, bound unamended (documents 49 §10, 50 §2) |
| 7 of 18 week-out folds miss Gate C-1 | **Open — the maintainer's decision.** §5's three routes; G-1's statistic not computed |
| G-1's statistic, the week-out audit, and the three named games under the week-out read | **Not computed**, deliberately. §2c |
| Postseason games have no week-out fold | **Disclosed.** Document 52 §7 fixes 18 folds at weeks 1-18; the frame carries 147 throws in weeks 19-22, so ~51 games' weeks are inside every fold's training data. G-1 was scoped to weeks 1-18; including them would have pushed agreement *up*, so the exclusion is the conservative one |
| The folds' covariate scale is the full frame's | **Disclosed design choice**, §2a. Handoff constraint 3's "only the row mask changes", made literal |
| Up to 19 of 280 QB-seasons have no rows in a fold | **Disclosed.** Week 18. They shrink to the prior; `v_q` is fitted and never read (document 49 §2), so this does not touch `p_i` |
| No fold lost a defence-season entirely | **Checked, 0 of 128 in all eighteen folds** — the safeguard that mattered never had to fire |
| `sigma_d` runs 0.205–0.294 across the folds | **Reported, not a result.** §2d — it mixes real variation with the sampler noise the failures are made of |
| G-2's ratio is a ratio of counts | **Pre-registered as such;** the 116/21/13 element-wise split is reported beside it (§3) |
| Document 52 §5's G-3 note compared a median to a mean full width | **Disclosed, §4.** The gate was computed on its own text (median move vs median half-width); the floor is not re-tuned |
| `research/67` no longer exits on a V-8 breach | **Disclosed deviation.** Ruling R-3 makes the breach immaterial; the gate's text is unamended and its verdict still prints and stores as FAIL |
| Gate C-1 is enforced by `research/69` over all eighteen folds, not by `67.fit` per fold | **Disclosed deviation.** The bars are unchanged; the change is that all eighteen lines are recorded before the run stops, so the stop-and-ask says whether the problem is one fold or seven |
| Amendment A-3 enacted? | **No.** G-2 and G-3 pass; G-1 blocked; clause 3's mirror not run |

## 7. Commits

Branch `feat/dropped-pick-variant`, off `docs/dropped-pick-confounds`. **Unmerged;
the maintainer merges.**

| Part | Commit | What |
|---|---|---|
| A | `61e9717` | Ruling R-3 recorded in documents 50 §2 and 49 §10 |
| B | `57cc092` | `research/69_dropped_pick_weekout.py`; `67` and `68` refactored callable; 18 folds fitted and saved; G-1 blocked at C-1; 525 → 536 tests |
| C | `99a13e5` | `research/70_dropped_pick_sensitivity.py`; G-2 PASS 0.94, G-3 PASS 1.62 vs 0.56 pp; V-1 0.00e+00 |
| D | this commit | This document, document 52 §8, document 05 §3's row, and the queue |
