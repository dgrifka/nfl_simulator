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

## 7. The full tail read after round 11 (2026-08-29)

`fix/figure-round-11`, from `handoff-2026-08-29-figures-r11.md`, after Part A
made the draw floor a share of the axis. The corpus was re-rendered from an
empty `research/outputs/all/` and an empty checkpoint: **15,600 PNGs in 34.7
minutes on 12 workers, worst replay gap `0.00e+00`**.

### 7a. What was opened

The pick lists were rebuilt from the fresh JSON by `_pick_lists` in
`research/79_render_all.py` and written to `research/outputs/79_pick_list.json`.
Five sorts — longest label, most waterfall rows, widest actual margin, smallest
|deserved margin|, most events under a point — worst ten of each, both editions,
both figure types. The sixth sort round 9 used, the smallest gap between the two
rule labels, is dropped: round 10 put those labels on two rows, so there is no
gap left to be small.

Deduplicated, that is **194 distinct PNGs across 97 game-editions** (94 Full, 100
Strict), and **all 194 were opened** — no sampling, no stopping at saturation.
This closes round 9's deviation (27 of 175) and round 10's (5 of 40).

### 7b. Closed by rounds 10 and 11, confirmed on the games that raised them

| Defect from §3 | Game re-opened | Status |
|---|---|---|
| Zero rule prints through `PIT wins` | `2023_11_PIT_CLE` full | **closed** — shielded corner label |
| Zero rule strikes `ARI wins` | `2016_14_ARI_MIA` strict | **closed** |
| Zero rule strikes `TEN wins` | `2017_04_TEN_HOU` strict | **closed** |
| Arrow sentence overlaps `HOU wins` | `2017_04_TEN_HOU` strict | **closed** |
| Arrow sentence overlaps `IND wins` | `2020_03_NYJ_IND` strict | **closed** |
| `81 events under 1 pt`, anonymous, third-largest bar | `2025_02_NYG_DAL` full | **closed** — now `43 small events (DAL)` and `38 small events (NYG)`, each with its club's mark |
| Fold rows at +0.03 / −0.02 drawing nothing | `2022_05_SF_CAR` full | **closed** — absorbed into the two club heaps |
| Title running under the corner credit stamp | corpus-wide | **closed** — 0 of 3,900 |

### 7c. Still open from §3, re-confirmed

| Game | Edition | Figure | Defect | Severity |
|---|---|---|---|---|
| `2019_01_BAL_MIA` | strict | waterfall | 49-pt margin: three event bars occupy ~5% of the plot width | legibility |
| `2019_02_NE_MIA` | strict | waterfall | 43-pt margin, three bars against a 55-pt axis | legibility |
| `2023_01_DAL_NYG` | full | waterfall | 40-pt margin, 17 bars in a 5-pt strip | legibility |
| `2025_19_GB_CHI` | full | waterfall | `GB fumble on a pass, retained` ×3, byte-identical | cosmetic |
| `2025_09_MIN_DET` | strict | dtw | The dashed `Deserved: even` rule at x=0 loses contrast against the saturated fill | cosmetic |
| `2023_19_CLE_HOU` class | strict | dtw | Rule label boxes flush with the figure edge (seen again on `2018_11_PHI_NO`, `2019_01_BAL_MIA` at the *left* edge, `2020_13_NE_LAC`, `2024_11_JAX_DET`) | cosmetic |
| blowouts generally | both | dtw | Zero-anchored `wins by` key sits far from the bars; degenerate games reserve an arrow band they do not use | cosmetic (both decided) |

### 7d. New in this read

| Game | Edition | Figure | Defect | Severity |
|---|---|---|---|---|
| `2023_02_WAS_DEN`, `2024_16_CLE_CIN`, `2016_14_DAL_NYG`, `2016_07_MIN_PHI`, `2023_06_MIN_CHI` | both | waterfall | **N5 — the span the floor uses is not the axis the reader sees.** The settled span is `max(0, actual, deserved) − min(0, actual, deserved)`, but the drawn frame adds a pad at both ends and a lane for the arrow rail, and the running totals can swing outside both margins. `2023_02_WAS_DEN`'s span is 2.0 pt (floor 0.01) while its frame runs about 12 pt, so a −0.04-pt row clears the floor and draws one pixel. Arithmetically the floor is **never more than 0.32% of the drawn axis** and is far less on a narrow game: `pad` is 20% of the drawn width at each end and `rail_room` another 18%, so the frame is at least 1.58× the span. | legibility |
| `2017_16_OAK_PHI`, `2019_17_OAK_DEN` | strict | both | **N6 — a 2017/2019 Oakland game is named LV everywhere**: headline, corner label, every row, legend and key. The relocation alias document 63 §3a found collapsing *filenames* also rewrites the visible club name. Worse: the same figure carries two different Raiders marks at once, the vintage wordmark on the anchor rows and the modern shield in the corner (also on `2023_15_LAC_LV` and `2025_18_KC_LV`, where the name is right and the two marks still disagree). | legibility |
| `2022_03_CIN_NYJ`, `2022_07_KC_SF`, `2025_19_GB_CHI` | full | dtw | **N1 — the luck arrow's floor is absolute too.** `ARROW_FLOOR = 1.0` pt, so a 1.1-pt arrow draws on a 55-pt axis as a bare arrowhead with no shaft. The same defect round 11 just fixed for the draw floor, one figure over. | cosmetic |
| `2022_03_GB_TB`, `2022_09_GB_DET`, `2022_12_NO_SF`, `2023_05_JAX_BUF`, `2023_12_CHI_MIN`, `2023_12_JAX_HOU`, `2024_11_SEA_SF`, `2025_01_ARI_NO`, `2025_01_MIN_CHI`, `2018_15_HOU_NYJ`, `2019_02_JAX_HOU`, `2022_01_IND_HOU`, `2023_10_DEN_BUF`, `2025_02_PHI_KC`, `2025_05_TEN_ARI`, `2025_09_MIN_DET`, `2025_18_CLE_CIN`, `2025_18_KC_LV`, `2025_19_LA_CAR` | both | waterfall | **N2 — an anchor of zero draws no bar.** `Deserved: even` (and `Actual: even` on a tie, `2022_01_IND_HOU`) leaves the anchor row as a label and a club mark floating with nothing between them. Nineteen of the 97 game-editions read. The anchors are exempt from the draw floor by design — they are the two ends, not events — so this needs its own rule if it needs one. | cosmetic |
| `2022_04_NYJ_PIT`, `2022_05_HOU_JAX`, `2018_15_HOU_NYJ`, `2021_16_PIT_KC` | both | both | **N3 — two dark clubs give two identical side tints.** NYJ green against PIT black, HOU navy against NYJ black: the two `X wins` regions are the same pale grey and side identity rests entirely on the corner labels and marks, which are present. | cosmetic |
| `2022_07_KC_SF`, `2025_15_DET_LA`, `2025_19_LA_CAR` | both | both | **N4 — a same-hue matchup collapses the whole colour system.** KC bright red vs SF dark red, DET light blue vs LA dark blue: bars, legend swatches, both side tints and the `wins by` key all read as one hue family. Document 60 §8d logs KC's protan gap against the neutral; this is the normal-vision case, club against club. | cosmetic |
| corpus-wide | both | dtw | **K6 widened.** Document 63 §3 recorded the *dashed deserved* rule losing contrast on saturated fill. The **solid actual rule** does the same and worse against a dark club — `2022_05_HOU_JAX`, `2023_15_LAC_LV` (black on black), `2024_07_CIN_CLE`, `2024_11_CLE_NO`, `2025_18_CLE_CIN`. Both rules also hide behind each other when the two margins are within a tenth (`2024_18_KC_DEN`, `2022_01_IND_HOU`). | cosmetic |

**No new class** on: longest label (the 58-character maximum wraps and sits
clear), most waterfall rows (`2023_01_PHI_NE` full at 25 rows is legible top to
bottom), and most events under a point (the club heaps carry them and each wears
its mark).

### 7e. What the relative floor did, seen on the page

The round's own rules are visible in the tail exactly as pre-registered, and
their costs are visible too.

- **Rule 2 keeps the words and loses the bar.** `2018_11_PHI_NO` strict has two
  event rows and one of them, `PHI extra point · Elliott, made (96% kick)` at
  **+0.04 pt** against a 0.205-pt floor, is a hairline. `2021_05_IND_BAL` strict
  keeps `BAL 23-yd field goal · Tucker, made (100% kick)` at **−0.01 pt**, the
  smallest row in the corpus. This is the trade the rule makes and it is not a
  bug; whether it is the right trade is the maintainer's.
- **Rule 3's residue is real and small.** `2017_04_TEN_HOU` strict is the
  clearest picture: four event rows, two of them club heaps that cancel far
  under a 0.215-pt floor — `9 small events (HOU)` at −0.06 and `2 small events
  (TEN)` at +0.05.
- **The floor works in the direction it was meant to.** On narrow games it keeps
  rows an absolute 0.05 would have folded: `2022_09_LA_TB` full keeps a +0.1-pt
  bar on a 5.4-pt span and it is a visible sliver. On wide ones it folds rows an
  absolute floor kept: `2022_05_SF_CAR` full has nothing under its 0.139-pt
  floor at all.
- **And a row exactly at the floor is still about two pixels**, by arithmetic —
  `2023_01_DAL_NYG` full has two of them. That is inherent to any floor.

## 8. Parked after round 12 (2026-08-29)

Raised by the round-11 tail read (§7d), decided in the round-12 brainstorm, and
**not acted on**. Parked is not rejected: either is a legitimate answer to "what
else could we try" in a later round.

- **N3 — two dark clubs give two identical side tints.** NYJ green against PIT
  black, HOU navy against NYJ black: both `X wins` half-plane washes come out the
  same pale grey, on `2022_04_NYJ_PIT`, `2022_05_HOU_JAX`, `2018_15_HOU_NYJ` and
  `2021_16_PIT_KC`. Side identity rests on the corner labels and the club marks,
  which are present on every figure, so colour is not the only encoding.
  Severity cosmetic.
- **N4 — a same-hue matchup collapses the whole colour system.** KC bright red
  against SF dark red, DET light blue against LA dark blue: bars, legend
  swatches, both side tints and the `wins by` key read as one hue family, on
  `2022_07_KC_SF`, `2025_15_DET_LA` and `2025_19_LA_CAR`. `teams.resolve_pair`
  separates the two **bar** colours and does not touch the tints or the key, so
  the fix is a rule about the tints rather than about the pair. Severity
  cosmetic.
