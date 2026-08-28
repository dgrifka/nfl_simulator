# 57 — The receiver-drop mirror: A-3 gate G-4

*Round 7 ran 2026-08-27 on `feat/dropped-pick-variant`, against the
pre-registration in document 56. **Unmerged.** This is the record: every gate
line, what the study found, what the component does, and the four-way audit
document 56 §3 asked for. Inputs: documents 09 (the prior drop study), 43/45
(study design), 47 §3 (EPA pricing) and R-1 (the like-grain cross-check), 49
(the component design), 52 (amendment A-3), 54 (the fold spec), 55 (G-1), 56
(this round's pre-registration).*

---

## 1. The gate lines, and where A-3 stands

| Gate | Bar | Result |
|---|---|---|
| **G-4a** — the study | C-3 ≥ 0.80 at the charged grain (A-3 clause 1) | **PASS.** 0.877 at team-season |
| **G-4b** — the component | V-1..V-8 | **PASS.** Every line below |
| **G-4c** — self-fulfilment | agreement ≥ 0.90, median \|ΔDTW\| < 1.0 pp between arms | **PASS.** 0.996 agreement, 0.04 pp |
| **G-4d** — materiality | document 52 §5's G-3 floor on `+rd` alone | **PASS.** 2.32 pp against 0.56 pp, clear by 1.75 pp |
| **G-5** — the combined audit | reported, never gated, never tuned | §5 |

**V-gates, in full.**

| | Statement | Result |
|---|---|---|
| **V-1** | v1.3 unchanged with both models off | **PASS**, max \|Δ deserved margin\| **0.00e+00** over 2,761 games, and 0.00e+00 on DTW% and both bounds. Printed first and last in both audit runs |
| **V-2** | ledger sum × `points_per_epa` = margin shift | **PASS**, **0.00e+00** on all three variant arms |
| **V-3** | a 2022+ game with no catchable target gives v1.3 | **PASS**, test |
| **V-4** | a pre-2022 game warns and returns v1.3 | **PASS**, test |
| **V-5** | one probability per posterior draw, in [0, 1] | **PASS**, test |
| **V-6** | Gate C-1's bars on the fit, every parameter | **PASS**, 0 divergences, max r̂ **1.0018** (`z_r[4]`), min ESS-bulk 4,064, min ESS-tail 4,329 over 276 parameters |
| **V-7** | a drop books negative `luck_epa` to the offence; a catch on a low-`p̂` ball books positive | **PASS**, four tests including the away-offence mirror |
| **V-8** | 89% catch probability on a median target inside **[0.85, 0.99]** for the five best and five worst | **PASS**, ten of ten lines, under **both** readings of "a median target" |
| *(extra)* | read side against the fit's own arithmetic | **PASS**, max \|read − fitted\| **0.00e+00** over 54,160 rows |

**Document 49 §10's open item is discharged.** It recorded that "a median
throw" was undefined and asked a future gate quoted on a reference row to write
the row down. Both readings are computed here, both pass, and **both design
rows are stored in `receiver_drop_summary.json`**.

### 1a. A-3's status, in document 56 §3's own words

Document 56 §3: *"A-3 is enacted for the class if G-4a–d pass **and** the maintainer
rules on §0's wording."*

**All four of G-4a, G-4b, G-4c and G-4d pass.** Every computed condition
amendment A-3 has ever set — G-1, G-2 and G-3 in rounds 5 and 6 for the
dropped pick, G-4a–d here for the receiver drop — is now met.

**A-3 is still not enacted, and this round cannot enact it.** The second half
of §3's enactment rule is not a computation: it requires the maintainer to rule on §0's
wording. Until he does, both directions stay labelled variants and document 05
§3 reads `variant (A-3 pending a wording ruling)` on both rows. §1b restates
the question with the arithmetic it now has.

### 1b. The wording question, restated for the maintainer — not resolved here

Document 52 §3's preamble says the class covers a finish that is
"near-random". A drop is a 1-in-20 event, not a coin. Document 56 §0
pre-committed the reading — *the gates govern, and "near-random" means the
entity's persistent skill explains a small share of the outcome* — **only if
the maintainer rules so**.

Round 7 supplies the arithmetic that reading needs, and it lands where §0
predicted. At the charged team-season grain, on arm 3's crossed fit, the
conditioned residual's entity spread is **0.647 pp** against a residual SD of
**21.836 pp**, so the receiving corps' persistent skill explains **0.088%** of
the variance in whether a catchable ball is caught — inside §0's pre-committed
0.1–0.2% band, and an order of magnitude below the dropped pick's 1.4%. (At the
individual receiver grain the same arithmetic gives 0.33%, still well under the
pick's — but that grain is unpowered and carries no verdict.)

**But the other half of §0's sentence now has a number too, and it cuts the
other way** (§3). Gate C-2 **fails** at the charged grain: receiving corps
genuinely differ, after conditioning. The two facts are both true and are not
in tension — a tiny share of a huge variance is still a real, measurable
difference between teams — but a reader deciding the wording should see both.

If the maintainer rules the other way, receiver drops fail the class on wording, A-3
cannot be enacted for picks either (clause 3), and **both stay labelled
variants**.

---

## 2. Power, and the grain the rule chose

`research/71`, 400 datasets per cell, seed 20260827. Guards exact: **56,211**
catchable targets, **2,781** drops (4.95%), **128** defence-seasons — the
pre-registered numbers to the row. Complete-case filtering leaves **54,160**
modelled targets over **1,931** receiver-seasons and **128** team-seasons.

The gate arm, on the conditioned residual:

| Grain | Null bound (threshold) | 5% | 12.5% | 25% | 50% | Resolvable |
|---|---|---|---|---|---|---|
| receiver-season | 1.009 pp | 0.15 | **0.40** | 1.00 | 1.00 | **No** |
| team-season | 0.634 pp | 0.19 | **0.88** | 1.00 | 1.00 | **Yes** |
| defence-season | 0.632 pp | 0.26 | **0.88** | 1.00 | 1.00 | **Yes** |

Gate D-1's power, on the raw drop-rate spreads (arm 1's own beta-binomial
instrument):

| Grain | Entities | Null bound | 5% | 12.5% | 25% | 50% | Resolvable |
|---|---|---|---|---|---|---|---|
| receiver-season (≥ 100 targets) | 67 | 1.540 pp | 0.13 | **0.20** | 0.65 | 1.00 | **No** |
| team-season | 128 | 0.714 pp | 0.20 | **0.87** | 1.00 | 1.00 | **Yes** |
| defence-season | 128 | 0.701 pp | 0.17 | **0.90** | 1.00 | 1.00 | **Yes** |

This reproduces document 09 §4's ordering on a different instrument — team
grain powered, receiver grain not — which document 56 §0's table anticipated at
0.87 and 0.37. The receiver grain is **thin, not absent**: 1,931
receiver-seasons share 54,160 targets, a median of **17 each**, against 422 per
team-season.

**So document 56 §1's clause-1 rule fires on its second branch: the component
charges the team-season — the receiving corps — and says so.** Two things
follow from the rule rather than from a choice made after seeing it. The
component's entity effect is the team-season effect, so arm 2 is fitted a
second time with a team-season term in place of the receiver-season one (arm
2t), because A-3 clause 2 requires posterior draws over the *charged* entity
and a receiver-season fit cannot supply them for a corps. And §0's `r_i`
notation is read as "the charged offensive entity", not specifically the
individual receiver.

### 2a. One implementation is not round 6's, and it is not an instrument change

The gate arm crosses 1,931 receiver-seasons with 128 defence-seasons, so
`research/_crossed_gaussian_grid.py` faces a 2,059-square Cholesky at each of
its 1,681 grid points. Measured on this machine: **183.8 s for one fit**,
against 1.6 s for the 256-level team-season design. Part A needs 400 datasets
in each of five cells at that grain — about **100 hours**. The dropped-pick
study never met this because its largest crossed design was 408 levels.

`research/_crossed_block_grid.py` evaluates the *same* profiled restricted
likelihood at the *same* grid points through a Schur complement on the
128-level block (both diagonal blocks of `Z'Z` are diagonal for two crossed
factors, so only the co-occurrence block is dense). One fit costs **0.17 s**,
and its `self_check` reproduces the original to **1.2e-15** on two designs
including one shaped like this study in miniature. That reproduction is printed
before any number derived from it, and the run stops if it fails.

---

## 3. What makes a catchable ball get dropped

Arm 2t, at document 54 F-1's spec (4 × 4,000 draws after 4,000 tuning,
`target_accept` 0.95). Gate C-1 **PASS** on all three arms — zero divergences
everywhere, worst r̂ **1.0022**.

In plain terms, on 54,160 charted catchable targets, a ball is **more** likely
to be dropped the further it travels in the air, and when the quarterback was
hit. It is **less** likely to be dropped on a screen, on play action, thrown to
either sideline rather than over the middle, nearer the opponent's goal line —
and, against the folk intuition, **less** likely when the ball is contested.

| Covariate | Effect on the odds of a drop | 89% excludes zero |
|---|---|---|
| `is_contested_ball` | ×0.75 | yes |
| `is_screen_pass` | ×0.79 | yes |
| `pass_location` right / left (vs middle) | ×0.83 / ×0.85 | yes |
| `is_play_action` | ×0.85 | yes |
| `air_yards` (+1 SD) | ×1.13 | yes |
| `qb_hit` | ×1.13 | yes |
| `yardline_100` (+1 SD, i.e. further out) | ×0.95 | yes |
| `ydstogo` (+1 SD) | ×0.96 | yes |
| down, shotgun, pre-snap `wp`, pass rushers, `air_yards²` | ×1.0 | no |

The contested coefficient is the interesting one and it is **why §4's hindsight
caveat exists**, not a discovery about hands: a contested ball that a defender
knocks away is very likely charted *not catchable* in the first place, so the
contested balls that survive into this frame are the ones the receiver could
cleanly have caught. Arm 2b, which drops the column entirely, moves the entity
spread by 0.03 pp (1.20 → 1.23 pp at the receiver grain) — the covariate is
doing very little work either way.

**The entity spreads.**

| Arm | Charged entity | σ(entity) | σ(defence-season) |
|---|---|---|---|
| 2 | receiver-season | 1.20 pp [0.83, 1.54] | 0.71 pp |
| 2b (no contested) | receiver-season | 1.23 pp [0.87, 1.56] | 0.72 pp |
| **2t (the fit the component reads)** | **team-season** | **0.63 pp [0.35, 0.88]** | 0.69 pp |

**Gate C-2, at the charged grain: FAIL — and reportable, because C-3 passes.**
The conditioned team-season spread's 89% upper bound is **0.87 pp** against a
**0.63 pp** threshold. Receiving corps genuinely differ at catching catchable
balls, after conditioning on air yards, contest, screen, play action, pressure
and field position. That is document 09's skill finding surviving the
conditioning, and document 52 §6 wrote it down in advance: *"Receiver drops
persist ~21% relative — more than defender finishing — so the mirror will
neutralize less per event and the two directions will be visibly asymmetric.
That is the honest outcome of clause 2, not a defect."*

At the receiver grain C-2 also fails (1.63 pp against 1.01 pp) but **C-3 fails
there too at 0.400**, so under document 43 §7 only the power table is
reportable at that grain and the verdict is not.

The defence-season SD, read off the receiver design's crossed fit: upper bound
1.02 pp against a 0.63 pp threshold, **C-2 FAIL, C-3 PASS at 0.880**. Defences
measurably differ at forcing drops too — and that effect is fitted and
deliberately never read (§4).

**Gate C-1's cross-check, like-grain only per document 47's ruling R-1:** arm 2
against arm 3 at team-season, gap **0.01 pp**; at receiver-season, **0.10 pp**.
Both far inside the 1.0 pp tolerance.

**Arm 1, Gate D-1 — the raw spreads, before any conditioning.** Descriptive by
rule, no pass condition, each quoted with the C-3 power from §2.

| Grain | Entities | League rate | Population SD | 89% | Relative | Power |
|---|---|---|---|---|---|---|
| receiver-season (≥ 100 targets) | 67 | 4.38% | 0.677 pp | [0.243, 1.198] | 15.5% | 0.198 — **unresolvable** |
| team-season | 128 | 4.95% | **0.709 pp** | [0.479, 0.933] | **14.3%** | 0.870 |
| defence-season | 128 | 4.95% | 0.749 pp | [0.527, 0.973] | 15.1% | 0.897 |

The team-season row reproduces document 09 §8's headline — 0.711 pp and 14.4%
relative — to within 0.002 pp on the same 56,211 balls, four rounds and one
different pipeline later. That is the strongest continuity check this round
has, and it was not planned as one.

**Secondaries (reported, never gated).** Odd/even split-half of the raw drop
rate at team-season r = **+0.062**, of the conditioned residual **+0.060** —
conditioning moves it almost not at all. At receiver-season, +0.035 raw and
+0.042 conditioned on 1,294 entities. Adjacent-season correlation of the shrunk
team effect r = **+0.079** on 96 pairs: like the defence's finishing, a
receiving corps' edge does not carry to the next season with any force.

---

## 4. The component, and its V-gates

`src/nfl_simulator/receiver_drops.py`, default off, TDD. 550 → **586 tests**.

- **`component = "receiver_drop"`, `charged_team = posteam`**, `actual = 1` if
  caught, `expected_draws` = P(catch) drawn from the arm-2t posterior over
  `α`, `β` and the team-season effect.
- **The defence-season effect is excluded on read** (document 56 §2). It is
  fitted so the offensive term is estimated free of schedule, and never paid:
  how well a defence covers is the defence's football and stays in `core`. It
  is loaded onto the model so a reader can report it, and a test pins that
  reading a row that names a defence-season does not change the probability.
- **The swing is per play, not per bin.** `|(air_epa + xyac_epa) −
  epa_incomplete|`, where the completion counterfactual is nflfastR's and
  `epa_incomplete` is the play's own realised EPA when it was dropped and the
  cell's dropped-branch mean when it was caught. **92.2%** of targets are
  priced this way; the six-cell bin table is the fallback for the rest.
- **The swing has one sign, on every play**, and the builder refuses otherwise:
  a catch must be worth more than an incompletion. Measured: median **1.37
  EPA**, mean 1.78, 89% [0.78, 3.93], min 0.01, **max 11.36**.

The bin table, document 47 §3's form on the drop branch — all six cells carry
their own difference, none falls back:

| Cell | caught | dropped | swing | E[EPA \| drop] |
|---|---|---|---|---|
| 1-33, early down | 8,589 | 469 | +1.24 | −0.66 |
| 1-33, late down | 3,854 | 223 | +2.58 | −1.50 |
| 34-66, early down | 17,101 | 831 | +1.29 | −0.75 |
| 34-66, late down | 6,316 | 383 | +2.69 | −1.65 |
| 67-99, early down | 13,368 | 671 | +1.23 | −0.73 |
| 67-99, late down | 4,040 | 197 | +2.18 | −1.29 |
| *pooled fallback* | | | **+1.61** | −0.95 |

**One clause of document 56 §2 was under-specified, and the reading is
disclosed rather than assumed.** §2 says "both `air_epa` and `xyac_epa` null →
bin table". In this data `air_epa` is null only where `xyac_epa` is too (2,038
rows), but `xyac_epa` is null on a further **2,359** rows where `air_epa` is
present. The completion counterfactual is a sum and half of it is not a value,
so the bin table is taken whenever **either** term is missing. That uses the
pre-registered fallback more often than the literal clause, never less, and it
never invents a zero for a quantity nflfastR declined to supply. Both counts
are reported.

**The hindsight substitutes (reported, never gated), and one of them fires.**
Document 45's probe does not transfer, because `p(catchable | incomplete,
is_drop)` is 1 by construction. Document 56 §1's two substitutes:

- Completions charted **not** catchable: **240 of 48,511 (0.49%)** — near zero,
  as it should be. Clean.
- Drop rate on contested vs uncontested catchable balls: **4.29% vs 5.04%, a
  ratio of 0.85×**. §1 pre-committed that contested should be *far higher*, and
  that if it is not, `is_catchable_ball` is being graded partly off the
  outcome. **It is not higher. The probe fires.** The likely mechanism is
  stated in §3: a contested ball knocked away is charted uncatchable, so the
  contested balls in this frame are the cleanly catchable ones. Every drop
  number in this round carries that caveat in words.

**`variant` now composes:** `"v1.3"`, `"v1.3+dp"`, `"v1.3+rd"`,
`"v1.3+dp+rd"` (= amendment A-3's `v2.0`). The receiver events are appended
after the dropped-pick events, so the shared random stream does not shift: a
test pins that `+dp`'s ledger rows are identical whether or not `+rd` is on,
and another pins that the `+dp+rd` ledger sums to the two ledgers added.

---

## 5. G-5 — the four-way audit, reported and not gated

Over the 1,139 charted games of 2022–2025, at v1.3's shipped settings.

| Arm | Affected games | Bucket moves | Median \|ΔDTW\| | 89% | Mean 89% interval width |
|---|---|---|---|---|---|
| `+dp` | 1,033 | **136** (11.94%) | 1.59 pp | [0.00, 27.32] | 0.0383 → 0.0514 |
| `+rd` | 1,138 | **162** (14.22%) | 2.32 pp | [0.00, 33.56] | 0.0387 → 0.0405 |
| `+dp+rd` | 1,138 | **200** (17.56%) | 3.85 pp | [0.00, 41.22] | 0.0387 → 0.0516 |

Coverage: **1,138 of 1,139** games carry at least one receiver-drop row
(99.9%), **54,336** events, median **48** per affected game, max 79. Where the
dropped pick has something to say about two thirds of game-teams, the receiver
drop has something to say about nearly every game.

**Element-wise, never by subtracting totals.** 76 games move under `+dp` only,
102 under `+rd` only, 60 under both. The combined arm moves 200, of which **32
neither alone moved**, and **70 that one alone moved the combined arm does
not**.

**§3's pre-committed expectation, tested.** It said the two directions should
be roughly independent per game, so the cancelling share should be near 50%,
and that a strong positive correlation would mean both are reading document
48's game effect and *is a finding to name*.

On the **758** games carrying both kinds of event, the two directions move DTW
**the same way 64.6%** of the time and opposite ways 35.4%, with **r = +0.197**
between the two shifts. That is a **mild positive dependence** — clearly not
the 50% independence §3 expected, and just as clearly not the two components
measuring one thing. Named as §3 asked, and read as: a game in which one
offence got unlucky on the ball in the air tends slightly to be a game in which
it got unlucky on the ball in its hands, which is what a shared game-level
component would produce at low strength.

The combined arm is also **not** the sum of its parts: max \|Δ(`+dp+rd`) −
(Δ`+dp` + Δ`+rd`)\| = **0.57 DTW**. Neutralizing both directions changes the
bootstrap's whole event set, so the DTW shifts do not add.

**The interval widening is the asymmetry document 52 §6 predicted, in a place
it did not name.** `+dp` widens the mean 89% DTW interval from 0.0383 to
0.0514; `+rd` barely widens it at all, 0.0387 to 0.0405. A team-season's catch
rate is estimated from ~437 catchable balls, so layer 1 has little uncertainty
to propagate; a defence-season's finishing rate rests on ~22 chances, so it has
a great deal.

**The three named games, four ways (DTW%, home):**

| Game | Actual | strict | `+dp` | `+rd` | `+dp+rd` | Bucket, strict → combined |
|---|---|---|---|---|---|---|
| `2025_17_DET_MIN` | +13 | 54.8 | 37.5 | 63.7 | 47.5 | too close → too close |
| `2025_13_DEN_WAS` | −1 | 14.5 | 29.3 | 32.6 | 40.2 | scoreboard holds → **too close** |
| `2022_13_WAS_NYG` | 0 | 80.7 | 95.4 | 88.9 | 97.9 | clear flip → clear flip |

`2025_17_DET_MIN` is the cleanest illustration of the cancelling: the two
directions pull it in *opposite* directions (37.5 and 63.7) and the combined
arm lands at 47.5, nearer the strict 54.8 than either alone.

**The five largest `+dp+rd` movers:**

| Game | Actual | Deserved, strict → combined | DTW% | Events |
|---|---|---|---|---|
| `2024_19_LAC_HOU` | +20 | +23.34 → −0.22 | 100.0 → 47.8 | 5 picks, 46 balls |
| `2022_07_GB_WAS` | +2 | +3.36 → −17.26 | 87.6 → 0.1 | 6 picks, 55 balls |
| `2025_14_HOU_KC` | −10 | −7.15 → +8.58 | 0.0 → 97.9 | 2 picks, 36 balls |
| `2023_21_DET_SF` | +3 | +3.44 → −10.27 | 82.5 → 1.8 | 3 picks, 49 balls |
| `2022_07_PIT_MIA` | +6 | +5.24 → −8.18 | 96.5 → 10.0 | 8 picks, 67 balls |

### 5a. The tail, checked rather than asserted

`+rd`'s median move is 2.32 pp but its **mean is 8.17 pp**, its 89% upper bound
33.56 pp, and its move exceeds a game's own 89% half-width in **87.9%** of
affected games. Document 49 §7's tripwire — *"if median \|ΔDTW\| exceeds 10 pp
the component is doing more than its skill share suggests and the sign
convention is the first suspect"* — does **not** fire on the median. The tail
was checked anyway, by re-simulating the largest mover end to end.

`2024_19_LAC_HOU` carries 46 catchable balls, of which **four** were dropped,
all by the Chargers. Two of those four priced at swings of **9.28** and **9.07
EPA** — balls whose completion counterfactual was a touchdown and whose
realised incompletion was a failed deep down. At a 95% catch probability each
books about **+8.7 EPA** to the home team, and those two plays are most of the
23.6-point move. The signs are right in both directions (a Chargers drop books
positive luck to Houston, a Chargers catch books slightly negative), and V-2
sums to 0.00e+00.

**So the tail is the component's own arithmetic, not a defect — but it is the
number to look at hardest.** Where `+dp` prices every event from a six-cell bin
table capped near 3.55 EPA, `+rd` prices each play from its own counterfactual,
which reaches 11.36 EPA. That is what document 56 §2 pre-registered, and
document 56 §0 predicted the shape — *"magnitude per game is therefore
comparable to picks, from many small credits and few large debits"* — but §0
expected comparable magnitude and the median move is **46% larger** than the
dropped pick's.

### 5b. G-4c — is the entity effect a game is priced at materially its own?

Nineteen refits at document 54 F-1's spec, one week of season held out of each
and the postseason held out together, every game read from the fit that never
saw its week. **114.9 minutes**, and Gate C-1 passes on **19 of 19** — zero
divergences anywhere, worst r̂ **1.0036** (the postseason fold, on `sigma_d`),
thinnest ESS-tail 1,904.

**G-4c: PASS.** Bucket agreement **0.996** (1,134 of 1,139) against a ≥ 0.90
bar; median \|ΔDTW\| between the arms **0.04 pp** against a < 1.0 pp bar. V-2
on every week-out game 0.00e+00.

Document 52 §5's consequence applies to this direction too: **production keeps
the in-sample read, with this bound recorded.**

The movement survives the check rather than being an artifact of it: the
in-sample arm moves 162 games across a bucket and the week-out arm **165** —
element-wise all 162 in both, none in-sample only, three week-out only. All
five disagreeing games sit within 2.9 pp of a bucket boundary and four of the
five move by under 1 pp:

| Game | Actual | DTW% in-sample → week-out | Move | Bucket |
|---|---|---|---|---|
| `2025_02_JAX_CIN` | +4 | 40.3 → 37.4 | 2.88 pp | too close → clear flip |
| `2023_06_NE_LV` | +4 | 40.5 → 39.5 | 0.93 pp | too close → clear flip |
| `2025_16_NE_BAL` | −4 | 60.6 → 59.9 | 0.73 pp | clear flip → too close |
| `2022_04_ARI_CAR` | −10 | 39.9 → 40.6 | 0.73 pp | scoreboard holds → too close |
| `2025_10_ATL_IND` | +6 | 60.3 → 59.9 | 0.38 pp | scoreboard holds → too close |

This is a tighter result than G-1's on the dropped pick (0.997 agreement,
0.05 pp) at a far larger event count, and for the reason §5 gives: a
team-season's catch rate rests on ~437 catchable balls, so removing one week's
worth barely moves it.

---

## 6. Register

| Item | Status |
|---|---|
| §0's wording question (is a 1-in-20 event "near-random"?) | **Open — the maintainer's.** The arithmetic is in §1b: 0.088% of per-target variance is persistent (0.647 pp against a 21.836 pp residual SD), against the pick's 1.4% |
| Gate C-2 fails at the charged grain | **Disclosed, not a blocker.** A-3 clause 1 asks for a *powered* measurement "reported whichever way it lands"; document 52 §6 predicted this direction would persist more |
| `is_catchable_ball` graded partly off the outcome | **Open.** The contested-ball substitute fires at 0.85× where §1 expected "far higher". Biases toward finding *less* skill in receivers, so the 0.63 pp spread is a floor |
| The receiver-season grain is unpowered (C-3 0.400) | **Accepted.** The component charges the corps, per the pre-committed clause-1 rule; the individual receiver's spread is reported with its power table and no verdict |
| Defence-season drop effect fitted and never read | **Disclosed**, as document 49 §2 disclosed `v_q`. A test pins the exclusion |
| The per-play swing's 11.36 EPA tail | **Disclosed** (§5a), checked end to end, sign convention independently verified |
| Document 56 §2's "both null" fallback clause read as "either null" | **Disclosed** (§4). 2,359 rows affected; the direction never invents a value |
| 2016–2021 coverage | None; `+rd` is a 2022+ object, like `+dp` |
| `research/_crossed_block_grid.py` replaces the grid's evaluation | **Licensed** (§2a) by a 1.2e-15 reproduction printed before any number that uses it |

---

## 7. Outcome

Round 7 ran 2026-08-27 on `feat/dropped-pick-variant`, off document 56.
**Unmerged.** 550 → **586 tests**, ruff clean, V-1 at 0.00e+00 at the end of
both audit runs.

| Part | Result | Commit |
|---|---|---|
| A — power, then thresholds | Guards exact. C-3 **0.40** receiver-season, **0.88** team-season, **0.88** defence-season. Clause-1 rule fires on its second branch. A third file was needed and is licensed in §2a | `048e614` |
| B — the study, arms 1–3 | C-1 PASS on all three arms. **G-4a PASS** (C-3 0.877 at the charged grain). C-2 **FAIL** at that grain and reportable. V-8 PASS. The hindsight substitute **fires** | `790cac0` |
| C — the component, TDD | `receiver_drops.py`, the switch, composable `variant`. V-2..V-5, V-7 as tests, plus the defence-exclusion pin and the swing-sign guard | `109799f` |
| D — G-4b, G-4d, G-5 | **G-4b PASS**, **G-4d PASS** (2.32 pp vs 0.56 pp). Handoff constraint 1 discharged: `+dp` alone reproduces document 55 at 136 / 1.59 pp exactly | `e88a42f` |
| D — G-4c | **PASS.** 19/19 folds clear C-1; agreement 0.996, 0.04 pp | `e774547` |
| E — the record | This document, document 52 §8, document 05 §3, the queue, `results-2026-08-27-exp7.md` | this commit |

**The round's deliverable is a decision for the maintainer, not a verdict.** Every
computed gate A-3 has ever set now passes. What stands between the amendment
and enactment is §1b's wording question, and it is his.
