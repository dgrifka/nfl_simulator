# 17 — Deflected-pass interceptions, pre-registered

*Written 2026-08-18, **before `research/28_deflected_int.py` existed**. Channel
comparison, stakes bound and instrument power:
`research/27_deflected_int_power.py`, results in
`research/outputs/27_deflected_int_power.json`, reproduced in §3–§5. Committed
to git before any verdict is read off a fit.*

*Inputs: documents 05 (the one rule and the two gates), 09 (the shape of a
candidate round), 14 (a Gate-A pass the data could not see), 15 (Phase 5
scouting), 16 (the materiality floor). Process laws unchanged.*

---

## 1. One-page story

### The question

A pass leaves the quarterback's hand, a defensive lineman gets a fingertip on
it, and the ball hangs in the air until somebody falls under it. Document 15
counted the stakes at 236 of 1,690 interceptions and named FTN's
`is_interception_worthy == False` as the channel, with an instruction attached:
**the Gate A memo must decide whether that flag isolates the coin or merely
encloses it.**

### The answer, stated first

**Neither. The FTN flag barely tracks deflections at all** — and a better channel
exists that document 15 did not find. But the better channel still cannot be
neutralized, for a reason that has nothing to do with charting judgment.

| Finding | Number |
|---|---|
| The FTN channel that *is* a deflection | **32.6%** (77 of 236) |
| The deflections the FTN channel catches | **28.3%** (77 of 272) |
| Deflections identifiable from official scoring, 2016–2025 | **629 interceptions**, 48–79 a season |
| Impact if every one of them were pure luck | median 1.32 pp of DTW on 566 games, 62 side flips |
| Instrument power to say whether generating them is a skill | **blind** — a 10% skill share sits inside the 0% band |

### Why it still fails

The neutralization identity needs `p(e)` — the probability that a deflected ball
ends up intercepted. Computing it needs the deflections that **did not** become
interceptions, and those are invisible in every dataset this project has. A
deflection that falls harmlessly to the turf is recorded as a pass defensed,
identical to a cornerback swatting a ball down cleanly. **The numerator is now
observable and the denominator is not**, so the rule has no value to return.

This is document 14's punt-bounce row repeating: *the one component with a
genuine branch point is the one the data cannot see.*

### Five things to hold onto

1. **`is_interception_worthy` is a judgment about the throw, not a record of the
   ball being touched.** Two thirds of the flag's interceptions have no
   deflection at all, and it misses seven in ten of the deflections that do
   occur. It answers a different question well and this question badly.
2. **The right channel is official scoring, not charting.** On an interception,
   nflverse credits the pass defense to the interceptor unless somebody else got
   a hand on the ball first. `pass_defense_1_player_id != interception_player_id`
   is therefore a *record of a second toucher*, it covers all ten seasons rather
   than FTN's four, and it involves nobody's opinion.
3. **Gate A passes on mechanism and the candidate dies after it.** The
   post-deflection flight is a genuine branch. Failing on identification is a
   different verdict from failing on mechanism and the register says which.
4. **The loss is real, which is why the successor is named.** Under the most
   generous assumption available the component would move the median affected
   game by 1.32 pp against a 0.69 pp floor and flip 62 games. This is not a
   candidate dismissed because it would not have mattered.
5. **The persistence question cannot be answered either.** At a median of two
   deflected interceptions per team-season, the split-half instrument cannot
   separate a 10% skill share from zero.

### Statistic convention

Posterior means with 89% equal-tailed intervals where a posterior exists. Impact
figures are in percentage points of DTW; EPA figures are converted at
`points_per_epa` = 0.8389.

---

## 2. Gate A — the branch-point memo

> **Is there a moment where the outcome is resolved by a mechanism outside either
> team's control, conditional on the state both teams created?**

### The post-deflection flight — **PASS**

The state both teams created is: a quarterback threw a pass, and a defender got a
hand on it. What happens next is an oblong ball tumbling through the air with
nobody's hands on it, and eleven players converging on wherever it comes down.
**That is the fumble argument with the ball three feet higher.** Document 09 §2
admitted onside kicks on exactly this reasoning and document 14 admitted the punt
bounce on it.

### Three things next to the deflection that are *not* the coin

| Adjacent thing | Verdict | Why |
|---|---|---|
| **The deflection itself** | **Not the coin.** A defensive play | Getting a hand on a thrown ball is a rush lane won or a route jumped. It stays in `core`, exactly as the *kick* stays in `core` while the bounce is the branch |
| **The bad throw** | **Not the coin.** The quarterback's | Where the FTN flag is right is that a ball thrown into traffic is a mistake. That mistake is already priced by the play's EPA, and neutralizing it would be crediting a quarterback for his own error |
| **The decision to catch it rather than knock it down** | **Not the coin.** A defender's choice | A ballhawk who plays a tipped ball with two hands converts more of them. This is precisely what Gate B's `w` exists to measure — and §5 shows it cannot be measured here |

### What Gate A does *not* settle, and why this document is not over

Gate A is a question about mechanism, and mechanism is not the binding
constraint here. Documents 09 and 14 both recorded candidates that passed Gate A
and were denied afterwards — onside kicks because `κ` was unestimable, the punt
bounce because the event was unobservable. **This candidate is the second kind**,
and §4 is where it dies.

---

## 3. Data — and which channel identifies a deflection

- **Grain of a row**: one interception.
- **Source**: `data/pbp/*.parquet` 2016–2025; `data/ftn/*.parquet` 2022–2025 for
  the comparison only.
- **Deflection channel**: `pass_defense_1_player_id != interception_player_id`.

### The channel comparison

Cross-tabulated on the 1,690 interceptions from 2022–2025 where both channels
exist:

| | FTN: not interception-worthy | FTN: interception-worthy | Total |
|---|---|---|---|
| **Deflected** (second toucher) | **77** (−4.81 EPA) | **195** (−4.67 EPA) | **272** |
| Not deflected | 159 (−4.61 EPA) | 1,259 (−4.32 EPA) | 1,418 |
| Total | 236 | 1,454 | 1,690 |

Read the table twice.

- **Down the FTN column:** of the 236 interceptions document 15 proposed as the
  deflection channel, **159 involve no second toucher at all.** They are
  ordinary interceptions on throws a charter disliked — including the defensive
  plays Gate A explicitly refuses to neutralize.
- **Across the deflected row:** of the 272 interceptions where a second defender
  touched the ball, **195 were on throws FTN judged interception-worthy.** The
  flag misses seven in ten deflections, because a deflected ball is usually the
  consequence of a *fine* throw meeting a hand.

**So the memo's assigned question resolves cleanly: the FTN flag neither
isolates nor encloses the coin. It cross-cuts it.** Using it would have
neutralized 159 clean defensive plays while leaving 195 genuine deflections in
`core` — errors in both directions at once.

### The deflection channel's own facts

| Quantity | Value |
|---|---|
| Interceptions 2016–2025 | 4,304 |
| …with a second toucher credited | **629 (14.6%)** |
| Per season | 57, 61, 70, 48, 57, 64, 74, 79, 55, 64 |
| Mean EPA, deflected interceptions | −4.71 |
| Mean EPA, defended incompletions | −0.89 |
| Games containing at least one | 566 of 2,761 |

### Facts that must be defensible by name

- **The channel is an inference from a scoring convention, not a labelled
  field.** nflverse parses the parenthetical defender in
  `INTERCEPTED by A (B)` into `pass_defense_1_player_id`. Play descriptions
  confirm B is the toucher — the sampled plays are nose tackles and linebackers
  tipping at the line — but no documentation guarantees the convention never
  varies. **Recorded as a defect.**
- **Play descriptions are useless as a cross-check.** Across 4,304
  interceptions, "tipped" appears 12 times, "deflect" 5 times and "batted"
  never. The gamebook simply does not narrate deflections.
- **Deflections by the *offense* are invisible.** A ball off a receiver's hands
  into a defender's gets no pass-defense credit and is not in the 629. Document
  09 ruled drops a Gate A failure at the receiver's hands, but the *ball's
  flight afterwards* is the same coin as this one, and it is unmeasured.

---

## 4. The identification failure

The rule needs

```
luck_epa(e) = (y(e) − p(e)) · swing(e)
```

with `p(e) = P(the deflected ball is intercepted)`. That probability requires a
denominator: **deflected passes, whether or not they were intercepted.**

| Population | Observable? | Why |
|---|---|---|
| Deflected → intercepted | **Yes**, 629 | The second-toucher credit |
| Deflected → incomplete | **No** | Recorded as a pass defensed, indistinguishable from a clean swat |
| Deflected → completed | **No** | Pass defense is credited on **zero** completions in ten seasons |

The middle row is the whole problem. Among 64,203 defended incompletions there is
no field, flag or phrase separating *the defender tipped it and it floated* from
*the defender knocked it down*. The third row is worse: a tipped ball caught by
the offense leaves no trace at all.

**Three ways out were considered and rejected:**

1. **Use the FTN flag as the denominator's proxy.** §3 shows it cross-cuts the
   channel; a proxy that captures 28% of the target and is 33% pure cannot carry
   a probability.
2. **Assert `p` rather than estimate it.** This is the move document 09 §5
   refused for onside kicks, and there the project at least had the fumble
   posterior to borrow from. Here there is no analogous rate anywhere.
3. **Model interception probability directly.** Ruled out as a settled decision
   in `CLAUDE.md`: interception probability is not available in this data, and
   `is_interception_worthy` is the designated substitute — which §3 has just
   shown does not measure this.

### What is at stake — the bound

Because the denial rests on identification rather than materiality, the round
owes an answer to *how much is being left on the table*. The bound assumes
`p = 0` — every deflected interception was pure luck and deserved nothing —
which no real component could exceed.

| Quantity | Value |
|---|---|
| Branch swing | 3.820 EPA = **3.205 points** |
| Events / games touched | 629 / 566 |
| **Median \|ΔDTW\|** | **1.32 pp** |
| Mean \|ΔDTW\| | 9.58 pp |
| Floor: incumbent's median 89% DTW half-width on those games | **0.69 pp** |
| Games whose DTW side flips | 62 |

**The bound clears the materiality floor by roughly a factor of two.** Unlike the
overtime toss, this candidate is not immaterial — it is unmeasurable. That
distinction is the reason §6 names a successor rather than closing the question.

---

## 5. Pre-registered gates

Committed before `research/28_deflected_int.py` exists.

### Gate D-1 — the branch-point gate

Settled in §2: **pass**, on the post-deflection flight.

### Gate D-2 — the identification gate *(new, and the one that binds)*

> **Is the population that defines `p` observable — both the events where the
> branch fell one way and the events where it fell the other?**

**Pass rule:** both arms of the branch are identifiable in the data with a
channel that does not rest on a proxy of unknown purity.

**Result: FAIL**, on §4's middle and bottom rows.

This gate is written down because Phase 4 and Phase 5 have now produced three
candidates that passed Gate A and died on measurement (the punt bounce, onside
kicks, this one), and the project has been resolving each one ad hoc. Naming it
makes the next one a two-line determination.

### Gate D-3 — is generating deflected interceptions a defensive skill?

Reported as a finding in its own right, deciding nothing about the ledger.

**Statistic:** split-half correlation of deflected interceptions per opponent
dropback, across 320 team-seasons.

**Instrument power**, measured first at the real denominators (median **2**
deflected interceptions on **582** dropbacks faced, base rate 0.332%):

| True skill share of variance | Split-half r the test would show (median, 90% interval) |
|---|---|
| 0% | +0.002 [−0.095, +0.095] |
| 5% | +0.025 [−0.068, +0.125] |
| 10% | +0.051 [−0.040, +0.146] |
| 20% | +0.111 [+0.017, +0.201] |

**Pre-registered reading: the instrument is blind.** The 5% and 10% medians sit
inside the 0% band, and even a 20% skill share overlaps it. **Neither a positive
nor a null observed correlation may be reported as a finding** — the same ruling
document 09 §5 made for onside kicks and document 15 §C5 *declined* to make for
interception returns, where the equivalent table showed a 5% share would have
been detected. The contrast between those two tables is the point: a measured
zero is informative only when the instrument was capable of measuring it.

### The decision rule, committed in advance

| Gate D-1 | Gate D-2 | **Treatment in v1.1** |
|---|---|---|
| Pass | **Fail** | **None — deny on identification.** Bound reported, successor named, nothing in document 05 §3 moves |

---

## 6. Known-defect register

| Defect | Evidence | Status |
|---|---|---|
| **The deflection denominator is unobservable** | §4: defended incompletions do not distinguish a tip from a swat; pass defense is credited on zero completions | **Open, and fatal to this candidate.** The successor is a data source that labels ball contact — charting or tracking |
| **The channel is a scoring convention, not a labelled field** | `pass_defense_1_player_id != interception_player_id`, corroborated by play descriptions on sampled plays, documented nowhere | **Open.** A convention change would silently move the count |
| **Offensive deflections are invisible** | A ball off a receiver's hands earns no pass-defense credit | **Open.** The same coin, entirely unmeasured |
| **`is_interception_worthy` does not measure deflection** | §3's cross-tab: 32.6% pure, 28.3% coverage | **Closed by this document.** Recorded so no future round re-proposes it |
| **The persistence instrument is blind at this grain** | §5's power table | **Open.** Would need a pooled multi-season grain or a larger channel |
| **The stakes bound assumes a homogeneous swing** | One branch swing of 3.820 EPA for every deflected interception | **Accepted.** It is a bound, not a proposal, and heterogeneity would not change its order of magnitude |

---

## 7. Constants appendix

| Constant | Value | Where it lives |
|---|---|---|
| `RANDOM_SEED` | 20260818 | `research/27_deflected_int_power.py`, `research/28_deflected_int.py` |
| `N_SPLIT_HALF_SIMS` | 2,000 | `research/27_deflected_int_power.py` |
| `SKILL_SHARES` | 0%, 5%, 10%, 20% | §5, matching document 15 §C5 |
| Deflection channel | `pass_defense_1_player_id != interception_player_id` | §3 |
| Branch swing (bound) | 3.820 EPA / 3.205 points | §4 |
| Materiality floor on the touched games | 0.69 pp | §4, the incumbent's own median half-width |
| `points_per_epa` | 0.8389 | `research/outputs/model_metadata_v11.json` |
| Interceptions / deflected | 4,304 / 629 | §3 |

Results are written back into this document as §8.
