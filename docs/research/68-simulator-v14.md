# 68 — Simulator v1.4: stadium elevation, shipped

*Written 2026-08-31, after the code landed. This is a **ship record**, not a
pre-registration: every gate this change had to clear was fixed in document 66
before a fit existed, and document 67 reported the results. What is new here is
the production implementation, the rebuilt artifacts, and the report on what
moved.*

*Scripts: `research/82_fg_v14_refit.py` (the refit and its four gates),
`research/83_simulator_v14.py` (the Strict build and its four),
`research/84_full_edition_v14.py` (the Full edition and document 64's headline
set recomputed). Artifacts: `research/outputs/trace_fg_v14.nc`,
`fg_v14_summary.json`, `dtw_games_v14.parquet`, `dtw_ledger_v14.parquet`,
`model_metadata_v14.json`, `full_summary_v14.parquet`, `83_ledger_delta.json`,
`84_full_edition_v14.json`.*

*Inputs: documents 66 (the pre-registration), 67 (the results the maintainer adopted),
27 (the refit this extends), 28 and 30 §5a (the correctness gate and the round
trip), 31 (v1.3, now superseded), 33 and 64 (the two corpus audits this
recomputes), 61 and 62 (the possession cap), 58 and 59 (ruling R-4's two
editions).*

---

## 1. The answer, stated first

> **The simulator now knows how high the stadium is.** One term joins the
> make-probability model — `beta_elev = +0.0602` log-odds per 1,000 feet, worth
> **+4.09 points of make probability on a 45-yard kick in Denver** against a
> stadium at the fitted mean elevation of 569 feet. Nothing else about the
> adjudication changes: the same components, the same population, the same
> seed, the same draws, and not one ledger row added or removed.

The change is real and it is small. Across 2,761 Strict games, **1,719 have a
deserved-to-win share that moves at all, 34 move by a point or more, and 22
move by more than the largest move a different random seed produces.** The
median moved game shifts **0.082 pp of DTW% and 0.022 points of deserved
margin**. Nine of the ten largest movers are games played in Denver.

| Check | Required | v1.4 production | |
|---|---|---|---|
| Sampler health | 0 divergences, r̂ < 1.01, ESS > 400 | 0, **1.0057**, **1,013 / 1,536** | passes |
| Fitted parameters reproduce document 67 §2 | all | `beta_elev` +0.06022 vs +0.0602 published | exact |
| Shipped posterior == studied posterior | draw for draw | **0.00e+00** across 9 parameters | exact |
| Round trip, read side vs fit | ≤ 1e-9 | **6.00e-15** overall, **4.33e-15** above 3,000 ft | passes |
| v1.3 replayed under v1.4's code | 0.00e+00, 2,761 games | **0.00e+00** on margin, DTW% and both bounds | exact |
| Ledger identity | 0 | **0.00e+00** | exact |
| Ledger rows | unchanged | 10,539 FG / 12,708 XP / 6,505 fumble, **change 0** | exact |
| Capped Full ledger sums | ≤ 1e-9 | **0.00e+00** | exact |
| Test suite | green | **961 pass**, 20 new | passes |

## 2. What changed in the code

`src/nfl_simulator/fg_model.py`, `simulator.py` and
`data/stadium_elevation.py`, under test-driven development; branch
`feat/fg-v14-elevation`.

| Before (v1.3) | After (v1.4) |
|---|---|
| `FieldGoalModel` carries eight fitted parameters | plus `beta_elev` and its `elevation_centre` |
| `_logit` sees distance, roof, wind, temperature, kicker, extra point | plus `_elevation_logit(stadium_id)` |
| `make_probability(kicker, distance, weather=, extra_point=)` | plus `stadium_id=` |
| `from_posterior` takes two centring constants | takes three, and **refuses** a trace with `beta_elev` and no elevation centre |
| The simulator hands the model roof, wind, temperature | plus `row.get("stadium_id")`, on field goals and extra points alike |
| `SIM_COLUMNS` — 33 play-by-play columns | 34, `stadium_id` added |
| The product reads `trace_fg_refit.nc` | reads **`trace_fg_v14.nc`** |

Four implementation facts worth stating in writing.

**a. The covariate is a lookup, not a column.** Elevation is not in nflverse
play-by-play. `stadium_elevation.py` is a hand-entered table, and v1.4 is the
first release in which the read side resolves a pricing input from something
that is not on the play row. The fit and the product call the same function,
so they cannot disagree about how high Denver is.

**b. Three silences, three different meanings.** This is where a version like
this one usually breaks, so each was written down before it was coded and each
is pinned by a test:

| Situation | Behaviour | Why |
|---|---|---|
| The posterior has no `beta_elev` | `stadium_id` is ignored entirely | v1.1, v1.2 and v1.3 stay reproducible on the wide frame. Absent means absent. |
| No `stadium_id` was passed | the kick is priced at the fitted centre | Document 05 §1's `w = 0` endpoint applied to the air. A Phase 2 replay frame has no such column. |
| The `stadium_id` is unknown | **raises**, naming the file to edit | Sea level is a real value in that table — MetLife is 10 feet — so a silent default would be indistinguishable from a correct row. |

**c. `elevation_centre` is required, not defaulted.** Wind and temperature
centres default to zero and no harm follows, because those terms reach only the
kicks that carry a reading. Every kick has an elevation. Loading a v1.4
posterior at a centre of zero would price the whole league as if it sat 569
feet lower than it does — one uniform, invisible shift, which is the exact
shape of the defect document 30 was written to stop. `from_posterior` now
raises rather than allowing it, and `load_model` carries the constant out of
`fg_v14_summary.json` so the pair cannot come apart.

**d. `stadium_id` is the first column in `SIM_COLUMNS` that prices.** Every
column added to that list since round 4 — `kicker_player_name`, `fixed_drive`,
`qtr`, `passer_player_name` — was presentation or grouping, and each was
argued inert. This one is not inert; it is inert *under a v1.3 posterior*,
which is a narrower claim and needed its own proof. Gate W-1 is that proof.

## 3. The coefficient, as the product prices it

`beta_elev = +0.0602` log-odds per 1,000 feet, 89% interval
**[+0.0165, +0.1043]**, centred at **0.5687 kft**. Read through
`FieldGoalModel` — the path the simulator books luck against, not the
posterior's own arithmetic — the league curve for an average kicker:

| Distance | At the fitted centre (569 ft) | Denver (5,280 ft) | Mexico City (7,280 ft) | Sea level (MetLife, 10 ft) |
|---|---|---|---|---|
| 33 yd (the extra point) | 94.62% | 95.86% (**+1.24 pp**) | 96.28% (+1.67 pp) | 94.44% |
| **45 yd** | 79.92% | **84.02% (+4.09 pp)** | 85.49% (**+5.56 pp**) | 79.38% |
| 50 yd | 71.46% | 76.79% (+5.34 pp) | 78.78% (+7.33 pp) | 70.77% |
| 55 yd | 61.21% | 67.63% (+6.42 pp) | 70.12% (+8.91 pp) | 60.41% |

**Every figure here reproduces document 67 §2 to the printed digit**, which is
the point of computing it twice: the study read the posterior directly and the
product reads it through the read side, and the two now agree by
demonstration rather than by assumption. The Denver-against-sea-level contrast
a reader actually pictures is **+4.64 pp at 45 yards** and +7.22 pp at 55.

**The caveat from document 67 §3 travels with these numbers and is not
weakened by shipping them.** The design was powered to resolve a 5 pp Denver
gain at 45 yards; the estimate is 4.09 pp, inside the band where a real effect
of that size is missed about one time in five. When a design detects an effect
near its own resolution limit, the estimates that clear the bar are the ones
that landed high — so **the true effect is more likely below 4.09 pp than
above it**. Adoption does not settle that, and document 67 §8 keeps it open.

## 4. What moved — the Strict edition, 2,761 games, 2016–2025

Document 33's audit, recomputed on both artifacts by `research/83`:

| Statistic | v1.3 | v1.4 | |
|---|---|---|---|
| Sign flips (deserved winner ≠ scoreboard) | 255 (9.24%) | **254 (9.20%)** | moved |
| Degenerate | 1,226 (44.40%) | **1,229 (44.51%)** | moved |
| Non-degenerate | 1,535 | **1,532** | moved |
| Clear flip | 197 | **197** | — |
| Too close to call | 186 | **182** | moved |
| The scoreboard holds | 2,378 | **2,382** | moved |
| Median \|deserved − actual\| | 2.366 pt | **2.376 pt** | moved |
| Games moving more than 3 pt | 1,081 | **1,077** | moved |
| Largest swing | 16.285 pt, `2018_05_GB_DET` | **16.258 pt**, same game | moved |
| Realized ties | 10 | 10 | — |

**The flip totals differ by 1 and the flip *sets* differ on 3 games.** Counted
element-wise, as document 33's defect register requires and as this repository
once got wrong:

| Game | Actual | Deserved v1.3 | Deserved v1.4 | |
|---|---|---|---|---|
| `2023_02_WAS_DEN` | −2 | +0.158 | **−0.557** | stops flipping |
| `2025_05_TEN_ARI` | −1 | +0.012 | **−0.021** | stops flipping |
| `2025_02_PHI_KC` | −3 | −0.039 | **+0.038** | starts flipping |

Two of the three cross zero by under a tenth of a point — they are games the
adjudication was already calling a coin flip. The third is a Denver game and
moves 0.71 points.

### 4a. How far anything actually moved

| Statistic | Value |
|---|---|
| Games whose DTW% moved at all | 1,719 of 2,761 |
| …by ≥ 0.5 pp | 155 |
| …by ≥ 1 pp | **34** |
| …by more than the seed floor's largest move (1.536 pp) | **22** |
| Median \|ΔDTW%\| on the games that moved | 0.082 pp |
| Largest \|ΔDTW%\| | **7.01 pp** (`2019_02_CHI_DEN`) |
| Median \|Δ deserved margin\| on the games that moved | 0.022 pt |
| Largest \|Δ deserved margin\| | **0.751 pt** (`2023_02_WAS_DEN`) |
| Mean *signed* Δ deserved margin | −0.001 pt |
| Monte Carlo floor (v1.4 at a second seed) | median 0.013 pp, **max 1.536 pp** |

**The 1,719 and the 34 are answers to different questions and both belong
here.** Almost every stadium sits off the fitted centre, so almost every game
with a kick in it moves by *something*; 1,719 is that count and it is not a
materiality claim. Document 67 §6 counted **174 games holding a kick that
moved by ≥ 1 pp of make probability**, and that is the pricing-side number.
The adjudication-side number is 34 games moving ≥ 1 pp of DTW%, of which 22
move further than re-drawing the coins on a different seed can move them.
**Everything this release changes, it changes in a few dozen games.**

The mean signed margin shift is −0.001 points. Elevation redistributes; it
does not inflate.

### 4b. The eight verdict buckets that moved

| Game | v1.3 | v1.4 | DTW% |
|---|---|---|---|
| `2016_15_NO_ARI` | scoreboard holds | too close to call | 39.67 → 40.13 |
| `2018_11_OAK_ARI` | too close to call | scoreboard holds | 40.13 → 39.60 |
| `2019_04_JAX_DEN` | too close to call | scoreboard holds | 41.60 → 39.93 |
| `2020_15_NYJ_LA` | too close to call | scoreboard holds | 40.69 → 39.92 |
| `2021_03_BAL_DET` | scoreboard holds | too close to call | 39.62 → 40.39 |
| `2023_16_NE_DEN` | too close to call | scoreboard holds | 41.85 → 39.90 |
| `2023_17_LA_NYG` | too close to call | scoreboard holds | 40.37 → 39.62 |
| `2024_07_LAC_ARI` | too close to call | scoreboard holds | 59.85 → 60.13 |

**Not one clear flip is created or destroyed.** All eight are games sitting on
the 0.40 or 0.60 edge of the too-close-to-call band, and every one crosses by
under 2 pp. They are boundary arithmetic, not changed verdicts, and the
article should not describe them as games v1.4 re-adjudicated.

### 4c. The ten games v1.4 moves furthest

| Game | Actual | Deserved v1.3 → v1.4 | DTW% v1.3 → v1.4 |
|---|---|---|---|
| `2019_02_CHI_DEN` | −2 | +0.671 → +0.183 | 55.50 → 48.50 (**−7.01 pp**) |
| `2023_02_WAS_DEN` | −2 | +0.158 → −0.557 | 53.61 → 47.66 (−5.95 pp) |
| `2016_01_CAR_DEN` | +1 | +1.167 → +0.771 | 71.38 → 67.71 (−3.67 pp) |
| `2023_01_LV_DEN` | −1 | +2.962 → +3.181 | 72.69 → 76.28 (+3.59 pp) |
| `2022_13_DEN_BAL` | +1 | +0.784 → +1.032 | 60.48 → 63.20 (+2.72 pp) |
| `2016_15_NE_DEN` | −13 | −1.040 → −1.321 | 37.40 → 34.99 (−2.40 pp) |
| `2025_21_NE_DEN` | −3 | −2.521 → −2.672 | 30.49 → 28.23 (−2.25 pp) |
| `2023_07_GB_DEN` | +2 | −2.331 → −2.246 | 15.97 → 13.84 (−2.13 pp) |
| `2025_07_NYG_DEN` | +1 | −3.019 → −3.012 | 10.17 → 8.14 (−2.03 pp) |
| `2023_10_ATL_ARI` | +2 | −0.710 → −0.575 | 43.25 → 45.22 (+1.97 pp) |

Nine of ten are at Empower Field; the tenth is Glendale, at 1,070 feet, and
the eleventh and twelfth (not shown) are Denver again. **This is the
concentration document 66 §3 named as the study's central limitation, arriving
in the product exactly as predicted.** v1.4 is, in its practical effect, a
Denver adjustment with an elevation justification — and document 67 §4's
Denver-excluded refit is the evidence that the justification is the right one,
not a post-hoc label.

## 5. What moved — the Full edition, 1,139 games, 2022–2025

| Statistic | Value |
|---|---|
| Games whose DTW% moved at all | 932 of 1,139 |
| Median \|ΔDTW%\| on them | 0.042 pp |
| Largest \|ΔDTW%\| | 4.31 pp (`2023_02_WAS_DEN`) |
| Median \|Δ deserved margin\| on them | 0.018 pt |
| Largest \|Δ deserved margin\| | 0.647 pt |
| Verdict buckets moved | **6 games** |

The six: `2022_05_IND_DEN`, `2024_13_TB_CAR`, `2025_07_PIT_CIN`,
`2025_09_JAX_LV`, `2025_14_CIN_BUF`, `2025_15_IND_SEA`.

Everything moves **less** in the Full edition than in Strict — 0.042 pp against
0.082, 4.31 pp against 7.01 — and the reason is structural rather than
coincidental. A Full game carries about sixty priced events to Strict's
twelve, so repricing four of them shifts a smaller share of the total. The
same change, seen through a wider ledger, matters less.

## 6. Document 64's headline set, recomputed

The article's number source, computed under v1.3, recomputed under both. **The
v1.3 column is this round's own recomputation from `full_summary.parquet`, not
a quotation of document 64** — which is how §6a's gap was found.

| Statistic | Document 64 | v1.3 rerun | **v1.4** | |
|---|---|---|---|---|
| Games | 1,139 | 1,139 | **1,139** | — |
| Sign flips | 168 (14.75%) | 168 | **168 (14.75%)** | — |
| DTW% below 0.5 for the realized winner | 167 | 167 | **167** | — |
| Degenerate | 310 (27.22%) | 310 | **309 (27.13%)** | moved |
| Non-degenerate | 829 | 829 | **830** | moved |
| Flips among the non-degenerate | 167 | 167 | **167** | — |
| Clear flip | 128 | *129* | **128** | see §6a |
| Too close to call | 95 | 95 | **97** | moved |
| The scoreboard holds | 916 | *915* | **914** | see §6a |
| Median \|deserved − actual\| | 3.43 pt | 3.425 | **3.436 pt** | moved |
| Games moving more than 3 pt | 631 | 631 | **631** | — |
| Largest swing | 19.05 pt | 19.05 | **19.03 pt** | moved |
| Largest-swing game | `2024_19_LAC_HOU` | same | **same** | — |

**What the article has to change, and what it does not.** The flip rate — the
headline of document 64 §1 and of the article's opening — **does not move**:
168 of 1,139, 14.75%, one game in seven. Neither does the degeneracy story's
shape, the largest swing's identity, nor the count of games moving more than a
field goal. What moves is small: one game leaves the degenerate set, two join
the too-close-to-call band, the median margin shift rounds from 3.43 to 3.44,
and the largest swing loses two hundredths of a point.

### 6a. A reproduction gap in document 64, and it is not v1.4's

This round could not reproduce document 64 §4's bucket counts from document
64's own artifact: **128/95/916 published against 129/95/915 recomputed**, a
one-game disagreement in the clear-flip bucket, present *before* v1.4 touches
anything.

The game is **`2022_13_WAS_NYG`, a drawn game** — Washington and the Giants
tied — with a deserved margin of +11.90 and a DTW% of 97.35%. Document 33's
convention books a realized tie in its own row, because a drawn game has no
realized winner to flip. `research/68`'s `bucket()` has no tie clause: with
`actual_margin == 0`, its test `(dtw > 0.5) == (actual > 0)` is
`True == False`, and the game is labelled a clear flip.

**Two functions in this repository label buckets differently on ties, and the
article quotes one of them.** That is a defect in the repository, opened here,
not a number v1.4 moved — and it is left open rather than fixed silently,
because choosing a side changes a published count and that is the maintainer's call.
Document 64's §2 already books ties in their own row, so the published
128/95/916 is the convention the article should keep; what needs the fix is
`bucket()`, and it needs it in a round that can re-check every figure the
function feeds.

## 7. The walk-through game, repriced

`2025_13_DEN_WAS` — **Denver at Washington**, played at Northwest Stadium's
**180 feet**, not at Denver's 5,280. It is the low-altitude side of this
change: 180 feet is below the fitted centre of 569, so every kick in it is now
priced slightly *harder*, and slightly more of each made kick is booked as
fortune.

### 7a. Document 64 §7's worked example does not move

The fumble at 6:43 of the second quarter — `p(e) = 0.5096`, `swing(e) = +4.2675
EPA`, `luck(e) = +2.0928 EPA`, +1.756 points — is **unchanged in v1.4**. It is
a fumble; nothing about it touches the kicking model. The article's single
most-quoted arithmetic example needs no edit.

### 7b. Document 64 §8a's twelve rows

The three fumble rows are unchanged. The nine kicking rows are not:

| Play | Component | Charged | `p` v1.3 | **`p` v1.4** | `luck` v1.4 (EPA) |
|---:|---|---|---:|---:|---:|
| 301 | field goal | DEN | 0.9470 | **0.9422** | −0.2121 |
| 1339 | field goal | DEN | 0.9473 | **0.9417** | −0.2138 |
| 3304 | field goal | WAS | 0.8817 | **0.8810** | +0.4741 |
| 4526 | field goal | WAS | 0.9460 | **0.9466** | +0.1959 |
| 1744 | extra point | WAS | 0.9446 | **0.9405** | +0.0592 |
| 2079 | extra point | DEN | 0.9545 | **0.9476** | −0.0521 |
| 2467 | extra point | WAS | 0.9440 | **0.9432** | +0.0565 |
| 2826 | extra point | DEN | 0.9539 | **0.9478** | −0.0519 |
| 4710 | extra point | DEN | 0.9544 | **0.9475** | −0.0522 |

The game's Strict total moves from **+2.7631 EPA (+2.318 pt)** to **+2.7106
EPA (+2.274 pt)**, and its deserved margin from −3.318 to **−3.274**. DTW% for
Washington moves from 0.1449 to **0.1497**.

### 7c. Document 64 §8b and §8c

| Component | Rows | v1.3 (EPA) | **v1.4 (EPA)** |
|---|---:|---:|---:|
| fumble | 3 | +2.5071 | +2.5071 |
| field goal | 4 | +0.2817 | **+0.2441** |
| extra point | 5 | −0.0258 | **−0.0405** |
| dropped pick | 2 | −1.1004 | −1.1004 |
| receiver drop | 60 | −1.1283 | −1.1283 |
| possession cap | 15 | −0.1175 | **−0.1132** |
| **total** | **89** | **+0.4169 (+0.350 pt)** | **+0.3687 (+0.309 pt)** |

The possession-cap row moves because the cap is bounded by its possession's
own largest event, and repricing a kick inside a possession can change which
event that is.

| | v1.3 Strict | **v1.4 Strict** | v1.3 Full | **v1.4 Full** |
|---|---:|---:|---:|---:|
| DTW% — Washington | 0.1449 | **0.1497** | 0.4058 | **0.4094** |
| DTW% — Denver | 0.8551 | **0.8503** | 0.5942 | **0.5906** |
| Deserved margin (home-signed) | −3.3181 | **−3.2741** | −1.3498 | **−1.3094** |

Document 64 §8c's story survives intact: Denver's deserved-win share falls from
85% to 59% when the hands-on-the-ball class is switched on, the verdict does
not flip, and 59% is inside the too-close-to-call band while 85% is not.

## 8. Every `v1.3` site in `src/` and `tests/`, decided

Handoff constraint 4. Forty sites; each read and decided rather than swept.

**Changed — sites naming what the product *is* or *reads*:**

| Site | Change |
|---|---|
| `render.py` GAMES_ARTIFACT / LEDGER_ARTIFACT / METADATA | → `dtw_games_v14.parquet`, `dtw_ledger_v14.parquet`, `model_metadata_v14.json` |
| `render.py` FULL_ARTIFACT | → `full_summary_v14.parquet` |
| `render.py` module docstring, the settings block, and five comments | v1.3 → v1.4, or → "Strict" |
| `plots.py` two `dtw_games_v13.parquet` references; the overtime share's version | → v1.4 |
| `simulator.py` the `variant` field's comment; the `edition` switch's docstring | → v1.4 |
| `simulator.py` two coverage warnings and four docstring lines | → "Strict", the name ruling R-4 gave the edition |
| `dropped_picks.py`, `receiver_drops.py` module docstrings | "beside the v1.3 ledger" → "beside the Strict ledger" |
| `fg_model.py` the list of posteriors that load through one path | + the v1.4 elevation posterior |

**Kept — sites stating history, where rewriting would make the record wrong:**

| Site | Why kept |
|---|---|
| `fg_model.py` "were not read here until v1.3" | a fact about when the cubic term reached the product |
| `components.py` "v1.2 neutralized these 192 kicks and v1.3 does not" | a fact about two past releases |
| `components.py` "books no `fg_luck` from v1.3 on" | still true, and dated correctly |
| `ledger.py` "the v1.3 artifacts were written before the column was named `actual`" | a fact about files on disk |
| `plots.py` "measured on simulator v1.1 against v1.3" | the impact run's actual arms |
| `render.py` "v1.3 never needed it", of `defteam` | a fact about a past frame |
| `simulator.py` two document 27 §14f citations | dated claims about v1.3's read-side fix |
| `tests/` local variables named `v13` in three files | identifiers meaning "the Strict arm", not version stamps |

Where a sentence meant *the edition without the hands-on-the-ball class* rather
than a version number, it now says **Strict**. That is the name ruling R-4
gave it, and it means the next release does not have to touch those lines
again.

## 9. Melbourne

The NFL plays at the Melbourne Cricket Ground in week 2 of 2026, at about 30
metres above sea level. `stadium_elevation.py` carries a forward-dated row for
it — **100 feet, under a provisional `stadium_id` of `MEL00`**, because
nflverse has not assigned that site an id and 2016–2025 play-by-play does not
contain one.

A guessed key is not a risk this table has to carry alone. A key that turns
out wrong is simply never matched, and `elevation_ft` then raises on the real
id and names the file to edit — which is the guard the maintainer asked for and which
`tests/test_stadium_elevation.py` now pins on the message text, not only on
the exception type. The elevation is the part worth entering early; the key is
a convenience that fails safe.

## 10. Defects

| Defect | Evidence | Status |
|---|---|---|
| `beta_elev` is not separable from an Empower-Field-specific effect | 680 of 712 kicks above 3,000 ft are Denver; nine of the ten largest movers are Denver games | **Open by construction**, inherited from document 67 §8. Gate E-6 rules out *Denver-only*, not *Denver-also* |
| The estimate sits below the design's own resolution | document 67 §3; power 0.780 at 4 pp | **Open.** Quantifiable from document 66 §6's machinery, still not quantified |
| Held-out log-loss cannot adjudicate a covariate on 3% of the data | document 67 §5 | **Open.** Shipping does not close it |
| The elevation table carries no measurement uncertainty | document 66 §2 | **Accepted, not open.** ±50 ft is 5% of one prior SD |
| `research/68`'s `bucket()` labels a drawn game as a clear flip | §6a; `2022_13_WAS_NYG` | **Open, and new.** Two bucket conventions coexist in this repository and the article quotes the other one |
| `MEL00` is a guessed `stadium_id` | §9 | **Open by design.** Fails safe; the real id raises |

## 11. What v1.4 does not do

- **It does not re-render a figure.** Every PNG in `research/outputs/` was
  drawn from v1.3 artifacts and none has been regenerated. `render.py` now
  points at the v1.4 files, so the next render pass produces v1.4 figures — and
  every figure that prints a make probability, a DTW% or a ledger row will
  move by the amounts in §4 and §5. Document 66 §9 named that re-render as part
  of a v1.4 round; it is the part not done here.
- **It does not touch the article.** Round 5 is the article's round. §6 is the
  list of numbers it must change and §6a is the one it must not.
- **It does not re-run the rematch check.** Document 27 §10 ran one for a
  change to `p_make` on every kick; v1.4 changes `p_make` on every kick again,
  and `research/47_rematch_v13.py` has not been re-run against v1.4. That is a
  named omission, not an oversight — document 12 measured the rematch test as
  nearly blind below ~20% damage, and v1.4's largest per-kick move is 9.42 pp.
- **It does not close the elevation thread.** §10's first three defects are
  live, and the verdict that adopted the term did not answer any of them.
