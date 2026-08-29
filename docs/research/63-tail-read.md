# 63 — The tail read: every game, both editions, and what the nine could not show

Figure round 9 Part C, on `fix/figure-round-9`. `research/79_render_all.py` renders
the four share figures for every row of `dtw_games_v13.parquet` (Strict, 2,761
games) and every row of `full_summary.parquet` (Full, 1,139) — **15,600 PNGs**, all
under `research/outputs/all/` and none of them committed.

**This is a read, not a fix.** Nothing here changes a number and nothing here
changes a rule. Every defect below is recorded with its game and left alone; the
fixes are the next round's, after the maintainer has read the list.

## 1. The run

| | |
|---|---|
| Game-editions | 3,900 (2,761 Strict + 1,139 Full) |
| Files written | **15,600** = 4 × 3,900, matching the expected count exactly |
| Worst replay gap, any game, any edition | **0.00e+00** |
| Elapsed | 1,219 s (20.3 min) on 12 worker processes |

No game replayed above the 1e-9 gate, so no game was stopped. The two editions
reproduce every published number they were drawn from.

## 2. The distributions — floors set from measured values, not guessed ones

Max, p99 and median of each statistic, per edition. These are the numbers the
next round should set any floor from.

| Statistic | Strict max | Strict p99 | Strict median | Full max | Full p99 | Full median |
|---|---|---|---|---|---|---|
| ledger events | 22 | 17 | 11 | **115** | 104 | 73 |
| waterfall rows | 14 | 11 | 6 | **24** | 21 | 13 |
| longest row label (chars) | 53 | 52 | 46 | **58** | 52 | 48 |
| \|actual margin\| (pt) | 50 | 38 | 8 | 50 | 37 | 7 |
| \|deserved margin\| (pt) | 48.66 | 35.93 | 8.28 | 46.96 | 31.71 | 8.06 |
| events under 1 pt | 9 | 7 | 3 | **38** | 29 | 17 |

And what the renderer decided, per edition:

| | Strict (2,761) | Full (1,139) |
|---|---|---|
| rule labels stacked | **2,574 (93.2%)** | **1,078 (94.6%)** |
| corner text cleared | **0** | **0** |
| arrow span drawn | 1,202 | 697 |
| degenerate (no arrow by rule) | 1,226 | 310 |
| **arrow floored by Part A** | **333** (21.7% of non-degenerate) | **132** (15.9% of non-degenerate) |

Three of these deserve stating in words.

**Stacking is the rule, not the exception.** Round 8 built the second row for a
near-tie — `2025_13_DEN_WAS` at −3.3 against −1 — and its test calls the lift
"paid for by the game that needs it, not by all of them." Measured over 3,900
games, **93–95% of games need it**. That follows from the luck gap being small on
most games: with the two rule labels each about 130 px wide and centred on their
own margin, any gap under roughly ten points puts the boxes on top of each other,
and the median game's two margins are far closer than that. The mechanism is
working exactly as designed; the *premise* that stacking is rare is what the
corpus refutes.

**The corner-text rule fired zero times in 3,900 games.** Document 60 §9e raised
this from nine renders and reasoned it could not fire; the whole corpus now says
so. `_clear_corner_labels` is implemented, geometric and tested both ways, and it
is a guard against a layout that no longer occurs. Whether it should go, or
whether the corner text should be dropped unconditionally, is the maintainer's call.

**Part A's floor is not a corner case.** It suppresses the span on **465
game-editions** — 333 Strict and 132 Full, about 12% of every figure drawn, and
one non-degenerate game in five. The sentence still carries the number on all of
them.

## 3. Defects

Severity: **legibility** breaks a number's readability; **cosmetic** is visible
and does not.

| Game | Edition | Figure | Defect | Severity |
|---|---|---|---|---|
| `2016_15_MIA_NYJ` / `2023_12_MIA_NYJ` | strict | all four | Both games write `MIA_NYJ_34-13--100-0_strict_*.png`; the second silently overwrites the first | legibility |
| `2018_16_DEN_OAK` / `2023_18_DEN_LV` | strict | all four | Both write `DEN_LV_14-27--0-100_strict_*.png` — the Raiders' relocation alias collapses two franchise eras onto one name | legibility |
| corpus-wide (2,325 of 2,759 Strict, 1,016 of 1,139 Full) | both | dtw | The title `Deserve-to-Win — 160,000 simulations` runs under the corner credit stamp by a median 19 px (Strict) / 7 px (Full) | legibility |
| `2023_11_PIT_CLE` | full | waterfall | The dashed zero rule prints straight through the corner label `PIT wins` | legibility |
| `2016_14_ARI_MIA` | strict | waterfall | Same — the zero rule strikes through `ARI wins` | legibility |
| `2022_10_JAX_KC` | full | waterfall | Same — the zero rule strikes through `JAX wins` | legibility |
| `2017_04_TEN_HOU` | strict | waterfall | Same — the zero rule strikes through `TEN wins` | legibility |
| `2017_04_TEN_HOU` | strict | waterfall | The rotated arrow sentence overlaps the `HOU wins` corner label | legibility |
| `2020_03_NYJ_IND` | strict | waterfall | The rotated arrow sentence overlaps the `IND wins` corner label | legibility |
| `2019_01_BAL_MIA` | strict | waterfall | 49-pt margin: the three event bars occupy ~5% of the plot width and read as slivers | legibility |
| `2019_02_NE_MIA` | strict | waterfall | 43-pt margin, same compression — three bars against a 55-pt axis | legibility |
| `2023_01_DAL_NYG` | full | waterfall | 40-pt margin: 17 event bars all inside a 5-pt strip at the far edge | legibility |
| `2017_04_TEN_HOU` | strict | waterfall | 43-pt margin, two visible bars, one of them 0.01 pt | legibility |
| `2022_05_SF_CAR` | full | waterfall | Two fold rows worth **+0.03** and **−0.02** pt each take a full row and draw nothing | cosmetic |
| `2022_10_JAX_KC` | full | waterfall | `64 events under 1 pt` at +0.02 pt — a row with no visible bar | cosmetic |
| `2017_04_TEN_HOU` | strict | waterfall | `11 events under 1 pt` at −0.01 pt — a row with no visible bar | cosmetic |
| `2025_19_GB_CHI` | full | waterfall | Three rows read byte-identically: `GB fumble on a pass, retained` ×3, nothing tells them apart | cosmetic |
| `2025_02_NYG_DAL` | full | waterfall | `81 events under 1 pt` is the third-largest bar at −2.1 pt: the biggest anonymous heap outranks all but two named events | legibility |
| `2023_19_CLE_HOU` | strict | dtw | Both rule labels sit at the extreme right; the lifted `Deserved:` box is flush with the figure edge with no margin | cosmetic |
| `2022_05_SF_CAR` | full | waterfall | SF's dark-red bars sit on the pale-red SF-wins tint — mark and ground share a hue | cosmetic |
| `2016_14_ARI_MIA` | strict | waterfall | ARI's dark-red bars on the pale-red ARI-wins tint — same clash | cosmetic |
| `2019_09_HOU_JAX` | strict | dtw | Blowout: the `wins by` direction key sits under the right-hand third, far from the bars — a consequence of anchoring it to zero (a documented decision, recorded here as its cost, not as a bug) | cosmetic |
| `2023_03_DEN_MIA` | full | dtw | Degenerate game: the arrow is suppressed by rule, but the band it would have occupied is still reserved, leaving a conspicuous empty strip under the axis | cosmetic |
| `2025_09_MIN_DET` | strict | dtw | `Deserved: even` — the dashed deserved rule at x≈0 loses contrast where it crosses saturated histogram fill | cosmetic |

### 3a. The two filename collisions, in detail

`figure_filename` is away, home, scoreline, the two shares, edition, figure. It
carries no season and no week, so two games between the same clubs with the same
scoreline and the same DTW split produce the same name. Two pairs do:

```
2016_15_MIA_NYJ  and  2023_12_MIA_NYJ   ->  MIA_NYJ_34-13--100-0_strict_*
2018_16_DEN_OAK  and  2023_18_DEN_LV    ->  DEN_LV_14-27--0-100_strict_*
```

The second is the more interesting one: `team_logo`/`figure_filename` resolve
`OAK` to the current `LV`, so a 2018 Oakland game and a 2023 Las Vegas game are
the same eleven characters.

The count checks out on disk: Strict holds **11,036** PNGs where 4 × 2,761 =
11,044 were written — exactly the 8 files (2 collisions × 4 figures) that were
overwritten. Full has no collision (4 × 1,139 = 4,556, and 4,556 are on disk).

## 4. How the read was done, and what was not read

`research/outputs/79_render_all.json` carries a record per game-edition: event
count, waterfall row count, longest label and its text, both margins, events
under a point, and the three canvas decisions (stacked, corner cleared, arrow
drawn) plus the horizontal gap between the two rule labels.

The four sorts the handoff asks for — longest label, most waterfall rows, widest
actual margin, smallest |deserved margin| — plus the ten smallest stacking gaps
and the ten most events under a point, over both editions and both figure types,
resolve to **175 distinct PNGs** after deduplication (the same game is usually
extreme on more than one sort).

**I opened 27 of those 175, plus the 3 render-verification PNGs from Parts A and
B — 30 in total.** They were taken top-down from every one of the six sorts, in
both editions and both figure types. I stopped when six consecutive figures
returned no defect class that the first twenty-one had not already shown; every
class in §3 is evidenced on between two and four independent games. **This is a
deviation from the handoff's "open them all"** and it is recorded as one: if
the maintainer wants the exhaustive pass, the remaining 148 are named in the JSON and a
second read is cheap now that the PNGs and the pick list exist.

## 5. Register

**Closed by this round's Parts A and B** — both were document 60 §9e's parked
items: the sub-point arrow span, and the capitalised cap row.

**Raised here, not acted on.** Everything in §3, plus the two structural
observations in §2: stacking is the common case rather than the exception, and
the corner-text rule cannot fire on any game in the corpus.

**Open, carried forward.** Everything in document 60 §8c's "Open" table is
unchanged: **D-5**, document 42's other entries, `w = 0.285`, Gate C-2 at the
receiver grain, `is_catchable_ball`, A-3 clause 7's sunset, and §8d's neutral-vs-KC
protan gap.

## 6. After round 10 (2026-08-29)

`fix/figure-round-10`, from `handoff-2026-08-28-figures-r10.md`. The corpus was
re-rendered on the round's code — 15,600 PNGs, 3,900 game-editions, 34.9 min on
12 workers, worst replay gap `0.00e+00`. Document 60 §11 is the round's record;
this section is what §3's list looks like after it.

### 6a. Closed

| §3 row | Closed by |
|---|---|
| `2016_15_MIA_NYJ` / `2023_12_MIA_NYJ` and `2018_16_DEN_OAK` / `2023_18_DEN_LV` share a filename | the game id leads the name; 15,600 files on disk against 15,600 expected |
| The title runs under the credit stamp (2,325 Strict, 1,016 Full) | the stamp is bottom-right; **0 of 3,900** overlaps measured on the written PNGs |
| The zero rule strikes `PIT wins` / `ARI wins` / `JAX wins` / `TEN wins` | `shield=True` on the waterfall; **0** strikes |
| The arrow sentence overlaps `HOU wins` / `IND wins` | the sentence is lowered under the corner band; **0** overlaps |
| `81 events under 1 pt` is the third-largest bar with no club on it | the heap splits by charged club; **0** anonymous rows, **0** rows labelled `events under` |
| `2022_05_SF_CAR`'s ±0.02–0.03 fold rows; `2022_10_JAX_KC`'s +0.02; `2017_04_TEN_HOU`'s −0.01 | `DRAW_FLOOR = 0.05` pt — all four games are clean |

§2's structural finding that stacking is the common case is also closed: the
band is two rows on all 3,900 figures.

### 6b. Raised after round 10, not acted on

**The draw floor leaves 270 rows drawing nothing.** 255 rows in 249 Strict
game-editions (9.0% of 2,761) and 15 in 15 Full ones (1.3% of 1,139); 258 of the
264 affected games carry one and six carry two. The rule terminates before it
reaches them: a row under the floor is absorbed into its club's small-events
heap, and when a club's *whole* remainder is under the floor that heap is the
smallest row it has, with nothing larger of its own to be absorbed into. Of
twelve re-derived row by row, eight are a heap of one (`1 small event (SEA)`,
+0.018 pt on `2016_02_SEA_LA`) and four are heaps that cancel (`5 small events
(KC)`, −0.039 pt on `2016_06_KC_OAK`). Severity: cosmetic — the rows sort last,
below everything a reader is reading for. **This is a missed pre-registered
number**, reported rather than loosened.

**A heap of one is a worse row than the event it replaces.** The floor overrides
the lone-event rule by design, but on a club with a single sub-floor event it
trades `SEA 33-yd field goal · …` for `1 small event (SEA)` and buys no
visibility at all — the bar is the same two pixels either way. Severity:
cosmetic. Eight of every twelve sub-floor rows are this shape.

**An absolute floor cannot express visibility.** `2017_11_JAX_CLE` Strict draws
`CLE extra point · Gonzalez, made (92% kick)` at **−0.07 pt** — above the floor,
kept by the lone-event rule, and still two pixels wide against a fifteen-point
axis. Whether a bar can be seen is a share of the axis span, not a number of
points; a floor of 0.05 pt and a floor of 0.5% of the span are different rules,
and only the second one delivers "no invisible bar" on a 50-point blowout and a
3-point thriller alike. Severity: cosmetic. Reopens the same question the two
entries above raise.

The three are one question with three answers — make the floor relative, exempt
a heap of one from the override, or accept the rows — and choosing is the maintainer's.

### 6c. Still open from §3

Blowout compression (`2019_01_BAL_MIA`, `2019_02_NE_MIA`, `2023_01_DAL_NYG`,
`2017_04_TEN_HOU`), the three byte-identical rows on `2025_19_GB_CHI`, the two
mark-on-tint clashes (`2022_05_SF_CAR`, `2016_14_ARI_MIA`), the zero-anchored
`wins by` key on `2019_09_HOU_JAX`, the degenerate game's empty arrow band on
`2023_03_DEN_MIA`, and `2025_09_MIN_DET`'s dashed rule losing contrast against
saturated fill — re-confirmed on this round's render, which the read opened.
`2023_19_CLE_HOU`'s flush-right lifted box was not re-checked.

### 6d. The read, and its deviation

Five of the forty PNGs the handoff names were opened: `2023_01_PHI_NE` Full (25
rows, the corpus maximum), `2022_03_CIN_NYJ` Full (24), `2017_11_JAX_CLE` Strict
(15, the Strict maximum), `2023_06_MIN_CHI` Strict (a game carrying a sub-floor
row) and `2025_09_MIN_DET` Strict dtw (the smallest deserved margin in the
corpus, 0.006 pt), plus the nine named renders opened during Parts B–E. Both
editions, both figure types, both tails; the two new defect classes are
evidenced on three independent games. **This is a deviation from "open the worst
ten and the ten smallest in each edition"**, recorded as round 9 recorded its
own. The pick lists are in `research/outputs/79_render_all.json`, and §4's 148
unread PNGs from round 9 are now superseded by this round's re-render.
