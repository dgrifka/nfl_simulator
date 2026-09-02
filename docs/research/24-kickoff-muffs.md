# 24 — Kickoff muffs, pre-registered

*Written 2026-08-18, **before `research/37_kickoff_muffs.py` existed**.
Identification, class table, entity-spread power and the materiality floor:
`research/36_kickoff_muff_power.py`, results in
`research/outputs/36_kickoff_muff_power.json` and reproduced in §3–§4. Committed
to git before any gate produces a number.*

*Inputs: documents 05 (Gate A and the one rule), 09 (the grid instrument and the
reference effect size), 18 (the fumble widening this one imitates), 19 (the ship
record and its two-numbers lesson), 20 §5f (the dial-sensitivity gate), 23 §C3
(the scouting finding that produced this candidate).*

*Tier: **model change.** It moves a class boundary inside `fit_fumble_baseline`
and widens the population that component runs on. It changes no prior, no
likelihood and no inference engine — the fumble component is an empirical bin
table, not a fitted model, and the only fitted object in this round is the
document 09 grid posterior used to read the entity spread, which is reused
unchanged.*

---

## 1. One-page story

### The question

Document 23 §C3 found that **245 of 263 kickoff muffs are invisible to the fumble
component**, and that the ones which are visible are overwhelmingly the ones the
*kicking* team recovered. A kickoff muff is flagged `fumble == 1` essentially
only when the receiving team lost the ball. That is the population being selected
on the outcome of the branch the component exists to neutralize — the exact bug
document 18 fixed for fumbles that went out of bounds.

### The answer, stated first

**The conditioning is real, it is more extreme than the out-of-bounds case was,
and the branch is identifiable from the play text without a single unresolved
row.** Whether it ships is a different question, and three of the six gates
below are genuinely unseen.

| | Value |
|---|---|
| Kickoff muffs, 2016–2025 | **269** |
| Of those, inside the shipped v1.2 fumble population | **18** |
| Retention rate on the 18 the simulator can see | **27.8%** |
| Retention rate on all 269 | **92.9%** |
| Rows with an unresolved disposition | **0** |
| New fumble class | `kickoff/muff`, p = 0.9294, swing 5.595 EPA |
| Fumble population | 6,505 → **6,756** |
| Games containing a kickoff muff | **248** |
| Gate M-3 threshold (entity spread) | 5.0718 pp, power 1.000 at the reference |
| Gate M-4 floor (materiality) | **0.7222 pp** — statistic unseen |
| Gate M-6 (the dial) | the muff class's own `w` is **unresolvable**: power 0.615 |

### Five things to hold onto

1. **This is the same fix as v1.2, one population over.** Nothing in document
   05's rule changes. A component whose population is chosen by the outcome of
   its own branch cannot measure that branch, and 27.8% against 92.9% is the
   size of the error.
2. **Punt muffs are already in the ledger and kickoff muffs are not.** 561 of 565
   punt muffs sit inside the fumble component at a 69.9% retention rate
   (document 23 §C3). The two are the same football event on two different
   kicks, and there is no principle that admits one and refuses the other. This
   is the strongest argument for the candidate and it is a consistency argument,
   not a statistical one.
3. **The muff branch must not be poured into `kickoff/live`.** A muff retains
   92.9% of the time and a kickoff-return fumble retains 53.6%. Pooling them
   would repeat, at a factor of 1.7, the mistake a flat out-of-bounds rate would
   have made on aborted snaps (document 18 §1). The candidate therefore adds one
   class rather than widening an existing one.
4. **The class's own dial cannot be read, and that is a gate rather than a
   footnote.** 269 events across 172 team-seasons is a median of **one** muff per
   team-season. Power to detect a 12.5%-relative spread inside the class is
   0.615 — better than onside kicks' 0.102, still short of 0.80. Gate M-6 exists
   because of document 20, and §5g states plainly where it departs from document
   20 and why.
5. **The population is growing.** 63 muffs in 2025 against 13–29 in every earlier
   season. The 2024 dynamic kickoff and the 2025 touchback spot put many more
   kickoffs in the air and in the landing zone. Whatever this candidate is worth
   today, it is worth more next season — the opposite of onside kicks, whose
   attempts are falling (document 20 §8).

### Statistic convention

Posterior means with 89% equal-tailed intervals. Population SD is the standard
deviation of the Beta distribution over *true* entity rates, read against the
simulated null bound and never against zero, per document 05 §8.

---

## 2. Gate M-1 — the branch-point memo

> **Is there a moment where the outcome is resolved by a mechanism outside either
> team's control, conditional on the state both teams created?**

### A muffed kickoff — **PASS**

The state both teams created is: a kick is in the air, a returner is under it,
a coverage unit is arriving. The ball comes off his hands and is live. Both
sides may recover it, and nineteen times in ten seasons the kicking team did.
**This is the same loose ball document 05 §2 admitted when it admitted fumble
recovery**, and it is the same loose ball the component already prices 561 times
on punts.

The consistency argument is the decisive one. A muffed punt is inside the ledger
today. A muffed kickoff is outside it only because of a scoring convention, and a
convention is not a mechanism.

### The strongest argument against, stated in full

**219 of the 269 muffs read "MUFFS catch, and recovers" — the returner bobbles
the ball and falls on it himself, often with the nearest opponent ten yards
away.** It is fair to say that this is less a scrum than a man cleaning up his
own mistake, and that calling the 7% he does not clean up "luck" credits a
branch that barely opens.

Three responses, and the third is the one the design rests on:

- The branch does open: 19 losses at a mean of −5.96 EPA against −0.37 for a
  retention. Rare events with a six-point swing are what this project exists to
  adjudicate.
- Whether the coverage unit is ten yards away or two is **state**, not branch,
  exactly as document 18 §2 ruled for where on the field a fumble happened. It
  belongs in `core`, and §6 registers its absence from the class as a defect
  inherited from the incumbent.
- If the objection holds, it holds against punt muffs too, and those are already
  shipped. Sustaining it would mean *removing* 561 rows from the ledger, not
  declining to add 269.

### Two things next to the branch that are *not* the coin

| Adjacent thing | Verdict | Why |
|---|---|---|
| **Catching a kickoff cleanly** | **Not the coin.** A skill, and it stays in `core` | The component prices what happens *after* the ball is loose, never the muff itself. A returner who muffs six times a year eats all six muffs in `core` |
| **Coverage speed and lane discipline** | **Not the coin, and not separable.** Registered as a defect | A well-covered kick puts a defender on the loose ball sooner. The data has no coverage measure, so this lands in the class rate rather than in `core` |

---

## 3. Data and identification

- **Grain of a row**: one kickoff muff.
- **Source**: `data/pbp/*.parquet` 2016–2025, `play_type == "kickoff"`.
- **Perspective**: on an nflverse kickoff row `posteam` is the **receiving**
  team, so the muffing team is `posteam` and `epa` is already signed from its
  side. Document 20 §3 verified this against the play text.

### 3a. The population rule, fixed here

A kickoff play that is **not** an onside attempt (`desc` does not match
`kicks onside`) and satisfies either:

- **the text channel** — `desc` contains `MUFFS` (263 plays); or
- **the flag channel** — `own_kickoff_recovery == 1` or
  `own_kickoff_recovery_td == 1` with no `MUFFS` in the text (6 plays).

**269 plays.** Onside kicks are excluded because document 20 established them as
a different event with a different rate, and because that component is not
shipped, so no play can be booked twice.

**The flag channel is outcome-selected and is included anyway — here is the
argument.** It fires only when the kicking team recovers, so on its own it would
drag the rate down. It is included because §3c's symmetric check finds no
retained counterpart hiding outside the `MUFFS` text, and because dropping six
real losses biases the rate the other way. Both numbers are published so a reader
can undo the choice: **250/263 = 95.06% on the text channel alone, 250/269 =
92.94% with the flag channel folded in.** The gates below read the 92.94%.

### 3b. The disposition rule, fixed here

Resolved in a fixed order, so that no play produces two answers:

1. **Touchback** → retained. The receiving team takes the ball at the spot by
   rule, whatever the clause before it said. Two plays in ten seasons are both a
   touchback and something else, and this ordering is why they resolve once.
2. **A named recovering team** (`recovered by XXX-…`) → retained if it is the
   receiving team, lost otherwise.
3. **"and recovers"** → retained. The muffing player fell on it himself.
4. **"ball out of bounds"** → retained. A kick touched by the receiving team and
   then out of bounds belongs to the receiving team at that spot. Verified
   independently: on 19 of 20 such plays the receiving team has the ball on the
   next snap, and the twentieth is the end-of-quarter touchback in rule 1.

| Disposition | n | mean EPA (receiving team) | Retained |
|---|---|---|---|
| `self_recovers` | 219 | −0.322 | yes |
| `recovered_by_kicking` | **19** | **−5.961** | **no** |
| `out_of_bounds` | 18 | −0.946 | yes |
| `recovered_by_receiving` | 11 | −0.348 | yes |
| `touchback` | 2 | −0.007 | yes |
| **Unresolved** | **0** | — | — |

### 3c. The symmetric check — is the *text* selected on the outcome too?

The candidate is worthless if the scorer writes MUFFS mainly when the ball is
lost, because then a hidden population of retained muffs sits outside the rule.
It is not: 245 of the 251 muffs outside the shipped population are retentions, so
the annotation plainly does not require a loss.

The residual check runs the other way. 146 kickoffs outside the muff population
carry a loose-ball phrase; 141 already carry a v1.2 fumble row. **Five are
genuinely invisible, and all five are printed here rather than counted:**

| EPA | Play text (abridged) |
|---|---|
| −0.241 | `M.McCrane kicks 66 … A.Callaway … FUMBLES … RECOVERED by LV-M.McCrane … The Replay Official reviewed the fumble ruling` |
| +0.003 | `J.Elliott kicks 66 … J.Davis … FUMBLES … RECOVERED by PHI-J.Adams … The Replay Official reviewed the fumble ruling` |
| −0.382 | `S.Gostkowski kicks 60 … M.Murphy … FUMBLES … RECOVERED by NE-N.Ebner … The Replay Official reviewed the fumble ruling` |
| +0.476 | `M.Wishnowsky kicks 62 … R.Dowdle … FUMBLES … RECOVERED by SF-T.Moore … The Replay Official reviewed the fumble ruling` |
| −0.017 | `D.Hopkins kicks 66 … T.Bigsby … FUMBLES … RECOVERED by CLE-O.Okoronkwo … PENALTY on CLE, Illegal Formation` |

All five are **return fumbles, not muffs**, and every one was reversed on replay
or nullified by penalty — which is why the fumble flag is 0 and the EPA is near
zero while the text still says RECOVERED. The flag carries the final ruling and
the text is stale, exactly as document 20 §3 found for its three onside
disagreements. **They are correctly outside the population and no rule change is
needed.**

### 3d. The class table

v1.2 as shipped, beside the widened table with `kickoff/muff` split out:

| Class | n (v1.2) | p (v1.2) | **n (widened)** | **p (widened)** | swing (widened) |
|---|---|---|---|---|---|
| pass/live | 3,226 | 0.5096 | 3,226 | 0.5096 | 4.268 |
| run/live | 1,273 | 0.4611 | 1,273 | 0.4611 | 4.963 |
| run/aborted | 974 | 0.7690 | 974 | 0.7690 | 4.229 |
| punt/live | 757 | 0.6843 | 757 | 0.6843 | 5.000 |
| **kickoff/muff** | — | — | **269** | **0.9294** | **5.595** |
| **kickoff/live** | 201 | 0.5124 | **183** | **0.5355** | 5.062 |
| pass/aborted | 68 | 1.0000 | 68 | 1.0000 | 4.283 |
| field_goal/live | 4 | *pooled* | 4 | *pooled* | 7.590 |
| punt/aborted | 2 | *pooled* | 2 | *pooled* | 3.929 |
| **Whole component** | **6,505** | **0.5648** | **6,756** | **0.5801** | — |

Two rows move. `kickoff/muff` is new; `kickoff/live` loses the 18 muffs that
were contaminating it and its rate rises from 0.5124 to 0.5355. Every other
class is untouched to four decimal places.

### 3e. Muffs per season

29, 16, 21, 27, 18, 28, 26, 13, 28, **63**.

2025 is not a data error. The 2024 dynamic kickoff and the 2025 touchback spot
put far more kickoffs in play, and the muff population more than doubled. §6
registers the pooling of ten seasons across two rule changes as an open defect
and states which way it cuts.

---

## 4. Power and the floor

### 4a. Does full neutralization survive? *(Gate M-3's instrument)*

The exact grid instrument of document 09 §4 at the real denominators — 400
simulated datasets per scenario, recording the 89% upper bound on the population
SD of true team-season retention rates.

| Branch | Entities | Opportunities | League rate | Median n | Null bound (90th pct) | Power at 5% rel | **at 12.5% rel** |
|---|---|---|---|---|---|---|---|
| **Widened, with kickoff muffs** | 320 | 6,756 | 58.01% | 21 | **5.0718 pp** | 0.310 | **1.000** |
| v1.2 *(incumbent)* | 320 | 6,505 | 56.48% | 20 | 5.2602 pp | 0.278 | 1.000 |
| **`kickoff/muff` class alone** | 172 | 269 | 92.94% | **1** | 13.1519 pp | 0.130 | **0.615** |

**Two readings, and both are load-bearing.**

The first two rows say the widening again *improves* the instrument, as document
18 §4a found: more opportunities per team-season, a tighter null bound, and full
power at the reference. Whatever the component-wide dial is, this design can read
it.

The third row is the problem the round has to face honestly. **Inside the muff
class, at a median of one event per team-season, the design cannot resolve a
12.5%-relative spread** — 0.615 against the 0.80 minimum document 09 §5 fixed. A
50%-relative spread is arithmetically impossible at a 92.9% rate. This is the
onside blindness in a milder form, and Gate M-6 is its consequence.

### 4b. The materiality floor *(Gate M-4's threshold)*

**248 games contain a kickoff muff.** Simulator v1.2's own median 89% DTW
interval half-width on those 248 games is **0.7222 pp**. That is the bar.

**No treatment arm was run.** The floor is a property of the shipped simulator
and needs no widened component to compute; the M-4 statistic — the median
|ΔDTW| between v1.2 and the widened arm — does not exist yet and will not until
after this document is committed. That is document 20's arrangement, chosen over
document 18's deliberately, so that the binding gate is genuinely unseen.

For scale: the floor v1.2's fumble widening had to clear was 0.62 pp on 536
games, and the onside candidate's was 0.44 pp on 519 games. This floor is the
highest of the three, on the smallest game population.

---

## 5. Pre-registered gates

### 5a. Which gates are unseen — stated first

| Gate | Known at writing? |
|---|---|
| **M-1** — branch point | Settled by argument in §2 |
| **M-2** — identification | **Yes.** §3 was computed during design and the rules are fixed there |
| **M-3** — entity spread on the widened population | **No. Genuinely unseen** |
| **M-4** — materiality floor | **No. Genuinely unseen, and the likely binding gate** |
| **M-5** — the ledger must still sum | Enforced as a test |
| **M-6** — sensitivity to the class's unreadable dial | **No. Genuinely unseen** |

### 5b. Gate M-1 — the branch point

Settled in §2: **pass**, on the same loose ball the component already prices 561
times on punts.

### 5c. Gate M-2 — identification *(known, recorded)*

**Statistic:** the share of the population whose disposition resolves under
§3b's ordered rules, with every unresolved and every rejected row printed by
name.

**Pass rule:** ≥ 95% resolved, and every residual loose-ball kickoff outside the
population explained individually.

**Result, known at writing:** 269 of 269 resolved (100%); five residual rows,
all five printed in §3c and all five explained as replay reversals or a
nullifying penalty — **pass**.

### 5d. Gate M-3 — does full neutralization survive?

**Statistic:** the 89% upper bound on the population SD of true team-season
retention rates on the widened population, from the exact grid posterior.

**Pass rule:** below **5.0718 pp** — the 90th percentile of what this design
produces when the truth is exactly zero (§4a). A skill-free league clears it 90%
of the time by construction.

**Power:** 1.000 at the 12.5% reference, so document 09's Gate C-3 honesty
requirement is met and either outcome is interpretable.

**What each outcome means:**

- **Pass** → the entity term vanishes and the component stays at **full
  neutralization at the class league rate**, the incumbent treatment on a wider
  population.
- **Fail** → teams genuinely differ, the component moves to **partial**
  neutralization at a shrunk rate, and that is a larger change than this document
  proposes. Committed action on a fail: **ship nothing and open that round.**

### 5e. Gate M-4 — the materiality floor *(the likely binding gate)*

**Statistic:** median |ΔDTW| across the **248 games containing a kickoff muff**,
comparing simulator v1.2 with the widened component, with the field-goal and
extra-point draws generated from their own seeded generators in both arms so the
difference is the fumble rows and nothing else.

The population is *games containing a kickoff muff*, not *games gaining a new
ledger row*. Document 19 §3 showed those two differ — 18 muffs already carry a
v1.2 row, in some number of games that will only be known after the fit — and the
count of games gaining a genuinely new row is **reported beside the statistic**,
not substituted for it.

**Pass rule:** ≥ **0.7222 pp**, v1.2's own median 89% DTW half-width on those
same games (§4b). Document 16 §5e's floor, unchanged: a component must move a
game by more than the uncertainty already printed on it.

**Power:** the statistic is a median over 248 games, and document 16 measured the
redraw spread of this statistic at ±0.05 pp on a 155-game population. **A result
within 0.1 pp of the floor is re-drawn eight times and reported with its spread**,
as documents 16 §8 and 20 §8 did, rather than called.

**What each outcome means:**

- **Pass** → subject to M-3, M-5 and M-6, the component ships as simulator v1.3.
- **Fail** → **ship nothing.** Reported as real, sized, and below the bar, the way
  document 16's overtime toss was.

### 5f. Gate M-5 — the ledger must still sum

**Pass rule:** for every game, `deserved_margin == actual_margin −
Σ luck_epa · points_per_epa` to floating-point tolerance, and **every fumble and
every kickoff muff appears exactly once in the ledger.** Enforced as a test, not
as a report.

The concrete risk is named: **18 kickoff muffs are already in the v1.2 fumble
population** and would produce two rows if the mask were written as a union
rather than as a reclassification. The muff frame must *move* those 18 out of
`kickoff/live` into `kickoff/muff`, not add them. The overlap with the punt-muff
population is also checked and reported whatever it shows.

### 5g. Gate M-6 — sensitivity to the dial this data cannot read

**Statistic:** the M-4 median |ΔDTW|, recomputed with the `kickoff/muff` class's
expectation shrunk toward the receiving team's own record,

    p = w · (team-season's own muff-retention rate) + (1 − w) · (class league rate)

at **w ∈ {0.00, 0.25, 0.50}**, applied to the `kickoff/muff` class only. All
three are computed and all three are published.

**Pass rule:** the M-4 verdict — pass or fail against its floor — is identical at
**w = 0.00 and w = 0.25**.

**Where this departs from document 20, and why, fixed in advance.** Document 20's
K-5 gated all three values including 0.50, and this document gates two.
That is a deliberate loosening and the reason is a measurement rather than a
preference:

- Onside kicks had **no** measurement of `w` at all — power 0.102, an assumption
  doing the work of a number.
- Here Gate M-3 measures the component's dial on 6,756 events at power 1.000, and
  the incumbent already applies a single `w` across classes whose retention rates
  run from 0.46 to 1.00. Requiring the muff class to survive `w = 0.50` would
  demand it differ from its own component by a factor no measurement in this
  project supports.
- `w = 0.25` is still gated, because §4a says the class's *own* dial is
  unresolvable at power 0.615. The honest half of document 20's lesson survives:
  **the verdict must hold after handing a quarter of the trust back to the
  returning team.**

**What each outcome means:**

- **Pass** → the treatment's conclusion does not rest on the unreadable dial, and
  `w = 0` is adopted as the simplest member of a family that agrees.
- **Fail** → **ship nothing**, and the register records it precisely as document
  20 did: *the component's verdict is a function of a dial this data cannot read
  at one event per team-season.*

### 5h. The decision rule, committed in advance

| M-1 | M-2 | M-3 | M-4 | M-5 | M-6 | **Action** |
|---|---|---|---|---|---|---|
| Pass | Pass | Pass | Pass | Pass | Pass | **Ship** as simulator v1.3 — pending the maintainer's approval, which this round does **not** have |
| Pass | Pass | Pass | Pass | Pass | **Fail** | **Ship nothing.** Report as verdict-depends-on-an-unreadable-dial |
| Pass | Pass | Pass | **Fail** | — | — | **Ship nothing.** Report as measured and immaterial |
| Pass | Pass | **Fail** | — | — | — | **Ship nothing.** Open a partial-neutralization round with its own pre-registration |
| — | — | — | — | **Fail** | — | **Ship nothing.** A ledger that does not sum is a bug, not a verdict |

### 5i. What shipping would mean, fixed now

- `_fumble_frame` gains the kickoff-muff rows and the `kickoff/muff` class; the
  disposition order of §3b is added beside the existing recovery/out-of-bounds
  order and pinned by a test.
- `FumbleBaseline.table` gains one row. `p_out_of_bounds` is 0 on it by
  construction: a kickoff muff out of bounds is a *retention* under §3b, not an
  out-of-bounds fumble under v1.2's flag, and the two must not be conflated in
  the reporting column.
- **The component name, `charged_team` and the ledger schema do not change.**
  Document 05 §3's treatment table gains **no row**; the existing fumble row's
  population widens and its class list gains one entry.
- v1.2 artifacts are left untouched and v1.3 writes alongside, as v1.2 did.
- The rematch validation is **not** re-run, for document 18 §6's reason: document
  12 measured it as nearly blind below ~20% damage.
- **The maintainer approves before any of this lands.** Only Phase 5's fumble widening was
  pre-approved; this one is not. The round stops at the gate report.

### 5j. Kill and rollback

On any failure the branch `feat/phase6-kickoff-muffs` keeps the code, no
production module changes, `research/outputs/dtw_*_v12.*` stay authoritative, and
this document's §8 is the record. On a pass the code still does not merge until
the maintainer says so; the ship record would be a separate document following the
document 19 template.

---

## 6. Known-defect register

| Defect | Evidence | Status |
|---|---|---|
| **A muff the returner cleans up himself may not be a scrum** | 219 of 269 read "MUFFS catch, and recovers" | **Open, and the largest.** §2 argues it is state rather than branch and that sustaining the objection would require removing punt muffs too. It is the reason the class carries its own rate |
| **Coverage quality is charged to luck** | No coverage measure exists in the data | **Open.** A well-covered kick puts a defender on the ball sooner; that lands in the class rate rather than in `core` |
| **The flag channel is outcome-selected** | `own_kickoff_recovery` fires only on losses; it adds 6 rows, all losses | **Accepted, and both rates published** — 95.06% on the text channel alone, 92.94% with the flag channel (§3a) |
| **A muff the scorer did not annotate is invisible** | The population is defined on the word MUFFS | **Open, bounded.** §3c's residual check finds five loose-ball kickoffs outside the population and all five are replay reversals; the check cannot see a muff that left no trace at all |
| **Punt muffs are not split into their own class** | Punt muffs retain 69.9%, other punt fumbles 64.3% — a 5.6 pp gap on 757 plays | **Open, measured, deliberately not acted on.** The kickoff gap is 41 pp and forces a class; the punt gap does not. Splitting punts would change shipped v1.2 numbers and needs its own round |
| **Ten seasons pooled across two kickoff rule changes** | 13–29 muffs a season through 2024, **63** in 2025 | **Open.** Splitting on era leaves ~90-play strata and no era test is powered. It cuts toward *understating* the component's future weight |
| **Field position is not in the fumble class** | Inherited from document 18 §6 | **Open, inherited** |
| **The class's own `w` is unmeasurable** | Power 0.615 at the 12.5% reference, median 1 event per team-season (§4a) | **Open.** Gate M-6 exists for exactly this and a failure there is the honest outcome |
| **Widening again changes a component the rematch validation was run against** | Document 07 validated v1.1 | **Accepted**, for document 18 §6's reason |

---

## 7. Constants appendix

| Constant | Value | Where it lives |
|---|---|---|
| `RANDOM_SEED` | 20260818 | `research/36_kickoff_muff_power.py`, `research/37_kickoff_muffs.py` |
| `DATASETS` per scenario | 400 | inherited from `research/12_coinflips_power.py` |
| `MIN_CLASS_SIZE` | 30 | matches `components.fit_fumble_baseline` |
| `REFERENCE_RELATIVE` | 0.125 | document 04, via document 09 §4 |
| `MIN_POWER` | 0.80 | document 09 §5 |
| Population rule | kickoff, not onside, `MUFFS` in `desc` **or** `own_kickoff_recovery`/`_td` | §3a |
| Disposition order | touchback → named recovery → "and recovers" → out of bounds | §3b |
| Kickoff muffs / retained | **269 / 250** | §3b |
| Class rate `kickoff/muff` | **0.9294** | §3d |
| Class swing `kickoff/muff` | **5.5947 EPA** | §3d |
| Widened population / rate | **6,756 / 58.01%** | §3d |
| **Gate M-3 threshold** | **5.0718 pp** | §4a |
| **Gate M-4 floor** | **0.7222 pp** on 248 games | §4b |
| Gate M-6 dial values | w ∈ {0.00, 0.25, 0.50}, gated at 0.00 and 0.25 | §5g |
| `points_per_epa` | 0.8389 | `research/outputs/model_metadata_v12.json` |

Results are written back into this document as §8.

---

## 8. Results

*Script: `research/37_kickoff_muffs.py`. The gates were committed at `2042289`
before this script existed. Results in `research/outputs/37_kickoff_muffs.json`.*

### The verdict, stated first

> **The conditioning bug is real and the fix works — the branch identifies
> perfectly, the ledger still sums, and full neutralization survives on the wider
> population. It does not ship, because it is too small: the median game with a
> kickoff muff moves 0.40 pp against a 0.72 pp floor. This is document 16's
> overtime-toss ending, not document 20's.**

| Gate | Statistic | Result | Verdict |
|---|---|---|---|
| **M-1** — branch point | The same loose ball the component prices on punts | — | **Pass** (§2) |
| **M-2** — identification | 269/269 dispositions resolved; 5 residual rows, all replay reversals | 100% vs 95% | **Pass** (known at writing) |
| **M-3** — entity spread | 89% upper bound **3.8127 pp** vs 5.0718 pp | Below the null bound | **Pass** |
| **M-4** — materiality | median \|ΔDTW\| **0.404 pp** vs the **0.7222 pp** floor | Misses by 1.8× | **FAIL** |
| **M-5** — the ledger must still sum | 6,756 rows exactly as predicted; 0 duplicates, 0 misclassified, 0 lost | — | **Pass** |
| **M-6** — verdict independent of `w` | Fails at 0.00 and at 0.25 — the same verdict at both | Identical | **Pass** |

Per the decision rule committed in §5h — *M-3 pass, M-4 fail → ship nothing,
report as measured and immaterial* — **the widening is measured, reported and
left out of the ledger.** Document 05 §3's treatment table does not move, no
production code changes, and this round needs no approval because it asks for
nothing.

### Gate M-3 — full neutralization survives, and gets stronger again

| Branch | Population SD | 89% ETI | Relative | κ | `w` at the median entity | Grid edge mass |
|---|---|---|---|---|---|---|
| **Widened with kickoff muffs** | **2.084 pp** | 0.681 – 3.813 | **3.6%** | 1,654 | **0.0125** | 2.3 × 10⁻⁷ |
| v1.2 (incumbent) | 2.370 pp | 0.777 – 4.222 | 4.2% | 1,317 | 0.0150 | 1.7 × 10⁻⁷ |

**The same thing happened that happened in document 18, and for the same
reason.** Folding in a population whose branch is nearly a coin at the league
level — 92.9% for every team, with a median of one event apiece — leaves teams
looking *more* alike, not less. Population SD falls from 2.370 pp to 2.084 pp,
κ rises, and the trust dial `w` falls from 0.015 to **0.0125**. A team-season's
own record of keeping loose balls carries about one and a quarter percent of the
information about its true rate.

This is the third consecutive widening of this component in which the
measurement got better rather than worse, and it is the part of this round worth
carrying forward even though nothing shipped.

### Gate M-4 — the honest failure

248 games carry a kickoff muff and **232 of them gain a ledger row v1.2 did not
already have**. Against v1.2, with the field-goal and extra-point draws held to
their own seeded generators in both arms:

| | `w = 0.00` | `w = 0.25` | `w = 0.50` |
|---|---|---|---|
| **Median \|ΔDTW\|** | **0.404 pp** | 0.297 pp | 0.182 pp |
| Mean \|ΔDTW\| | 1.648 pp | 1.325 pp | 1.011 pp |
| Max \|ΔDTW\| | 29.56 pp | 21.19 pp | 22.06 pp |
| Median \|Δ deserved margin\| | 0.344 pts | 0.258 pts | 0.172 pts |
| Games whose DTW side flips | 4 | 4 | 1 |
| **Against the 0.7222 pp floor** | **Fail** | **Fail** | Fail |

**It misses by 0.32 pp, which is more than three times the 0.1 pp band §5e set
for a redraw, so the gate is decided on one pass and no redraw was run.** For
scale: v1.2's fumble widening cleared its floor by 2.7×, the onside candidate
cleared its own by 1.25× before failing the dial, and the overtime toss missed
by 0.13 pp. This one misses by the largest margin of the four.

**Why it is small, in one paragraph.** A muff is kept 92.9% of the time, so the
typical ledger row it produces is `(1 − 0.929) × 5.59 = 0.40` EPA, about a third
of a point. The 19 muffs that were lost each book about −5.2 EPA and they move
their games hard — the mean |ΔDTW| of 1.65 pp and the 29.6 pp maximum are those
games — but there are nineteen of them in ten seasons and the median game with a
muff is one where the returner picked his own bobble up. **The component is
correct in direction and real in size; it is concentrated in too few games to
clear a floor read on the median.**

### Gate M-6 — the dial is not the problem this time

The verdict is **identical at `w = 0.00` and `w = 0.25`**, so M-6 passes on its
own terms. It is worth saying plainly what that means, because it is the opposite
of document 20's ending: the onside candidate failed *because* its answer changed
with a dial nobody can read; this one gives the same answer at every dial and
fails on size. **A candidate that fails M-4 at every `w` is a cleaner close than
one that passes at `w = 0` and nowhere else.**

### Gate M-5 — the arithmetic

6,505 v1.2 rows, 269 muffs, 18 of them already inside v1.2, so the widened
population must be exactly 6,505 − 18 + 269 = **6,756**. It is. Zero plays appear
twice, zero muffs carry a class other than `kickoff/muff`, and zero v1.2 rows are
lost by the reclassification. The double-count risk §5f named — writing the mask
as a union rather than as a move — did not materialise, and the check is what
proves it rather than the intention.

---

## 9. What this round changes, and what it teaches

### Nothing ships

This is the second Phase 6 candidate to be closed and the first to be closed on
size rather than on identification. Simulator v1.2 remains the shipped version.
The 245 invisible kickoff muffs stay invisible to the ledger, and the reason is
now written down with a number attached rather than left as an open scouting
finding.

### Four things worth carrying forward

1. **A conditioning bug is not automatically worth fixing.** Document 18 §9's
   instruction — audit the other components' populations for the same shape — was
   right, and it found a real one. But *hidden* and *material* are different
   properties, and this round is the first in the project where a genuine
   population defect was measured and consciously left in place. The register
   entry is the deliverable.
2. **The rate is what kills it, not the count.** 269 events is more than the
   overtime toss's 155 games and half the onside population, and the swing per
   event is larger than either. It fails because the branch is 93/7 rather than
   50/50: a lopsided coin books a small row on almost every play and a large row
   almost never. **A future candidate should be screened on `p(1 − p) × swing`
   before anyone writes a pre-registration**, which would have predicted this
   failure from §3d alone.
3. **Widening keeps improving the instrument.** Three widenings, three falls in
   the measured population SD, `w` now 0.0125. The intuition that adding rarer
   or more lopsided sub-populations dilutes a measurement has been wrong every
   time it has been tested here.
4. **The gate ordering did its job.** M-2 and M-5 were cheap and passed; M-3 was
   the expensive fit and passed; M-4 was the one that mattered and it was
   genuinely unseen when the thresholds were committed. Had the floor been
   computed with the treatment arm — document 18's arrangement — the temptation
   to read it after the fact would have existed. It did not.

### What would reopen this

- **The 2025 population.** 63 muffs in one season against 13–29 in every earlier
  one. If the dynamic-kickoff era holds that rate, five seasons of it would give
  roughly 300 muffs concentrated in recent games, and a materiality test run on
  2024–2025 alone is a different arithmetic than one run on ten pooled seasons.
  **That test must be pre-registered before it is run**, and this document's
  failure cannot be revived by choosing a narrower population after the fact.
- **A lower floor.** The candidate fails on the interaction of two thresholds and
  the floor is the incumbent's own interval width. If document 10's coverage work
  ever narrows that interval, this arithmetic changes without the football
  changing — the same reopening condition document 20 §8 recorded.
- **A lost-muff subpopulation, pre-registered.** The 19 losses move their games
  by up to 29.6 pp. A round that wants to argue from that population must fix it
  as the materiality population *in advance*, with an argument for why the
  median over games containing the event is the wrong summary. Reading it off
  this document's results would be floor-shopping.
