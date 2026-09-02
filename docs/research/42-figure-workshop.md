# 42 — Figure workshop, round 2

**Date:** 2026-08-26
**Branch:** `feat/product-bootstrap-plot`
**Scripts:** `research/59_dtw_variants.py`, `research/60_matchup_colours.py`,
`research/58_brand_figures.py` (updated)
**Code:** `src/nfl_simulator/plots.py`, `render.py`, `teams.py`, `style.py`
**Round 1's record is document 41 and stands.**

**Nothing here is fitted and no number is new.** Every value on every figure is
read from `dtw_games_v13.parquet`, `dtw_ledger_v13.parquet`,
`26_overtime_games.parquet` and `model_metadata_v13.json`. The one quantity
recomputed is each game's bootstrap draws, which the shipped summary does not
keep, and `render.replay` checks the redraw against the published number before
a pixel is drawn: all five example games replay at **0.00e+00**.

---

## 1. The distribution — four variants, and why V4 ships

The maintainer's round-1 verdict: the figure "makes sense the more you read it" but is not
intuitive at first glance. That is a presentation problem, and the way to settle
a presentation problem is to render the candidates and look at them.
`research/59_dtw_variants.py` renders four cumulative variants for two games —
the worked clear flip and the game inside the band — eight PNGs, each adding
exactly one device to the one above it, so a preference between two adjacent
variants is a preference about that one device.

| | bins | callout | arrow | logo legend |
|---|---|---|---|---|
| V1 | 1 pt | ✓ | | |
| V2 | 3 pt | ✓ | | |
| V3 | 3 pt | ✓ | ✓ | |
| V4 | 3 pt | ✓ | ✓ | ✓ |

**V4 ships**, as `render.DTW_FIGURE`. Reading the eight:

- **The bin width is the biggest single change.** At one point the histogram is
  a picket fence — `59_2025_17_DET_MIN_V1.png` has eleven visible spikes with
  gaps between them, and a reader's first impression is that something is wrong
  with the rendering. At three points the same game is one clean mass on each
  side of zero and the 55/45 split is legible at a glance.
- **The spikes are honest and are not smoothed away.** The deserved margin is the
  actual margin minus a handful of fixed-size luck bars each switched on or off,
  so only certain values are reachable at all. Wider bins *pool* reachable
  values; a kernel density curve would draw margins the game cannot produce, and
  there is deliberately no such variant.
- **The callout is the sentence the figure exists to say.** "GB deserved to win
  95% of simulations", in that club's own colour, in the upper corner on that
  club's own side. A reader who takes nothing else off the plot takes this.
- **The arrow adds the thing the distribution alone cannot say**: not where the
  game landed, but how far luck moved it and toward whom.
- **The logo legend costs nothing and buys recognition.** Two coloured swatches
  ask a reader to hold a colour in their head; two club marks do not. The
  abbreviation stays beside each mark, so a club with no cached logo is still
  named and identity is never carried by an image alone.

Three further changes apply to every variant.

**"Re-flips" is gone.** It is the simulator's word, not a reader's. The heading
now reads `Deserve-to-Win — 160,000 simulations`, the y-axis label `% of
simulations`, the legend `GB wins`. `grep -n "re-flips" src/nfl_simulator/plots.py`
prints nothing. The two *committed caveats* keep their "re-flip" verbatim —
they are quotations from document 33 and are not this round's to reword (§6).

**The y axis carries per cent of the simulations, not a bare density with no
ticks.** A density's height depends on the bin width, so the same game drawn at
one point and at three would print two different y axes for one fact; a share of
the simulations is a number a reader can state out loud.

**`margin, DET perspective` becomes `final margin (DET − GB)`.** The subtraction
can be read straight off the axis instead of held in the head, and it is the same
label on the waterfall.

**The plot reserves headroom above its tallest bar.** The annotations are placed
by rule rather than by inspection, so the room they need is made rather than
hoped for. The first V1 printed its callout straight across the three tallest
bars; `ANNOTATED_HEADROOM = 1.62` (`PLAIN_HEADROOM = 1.18` when nothing is
annotated) fixed it, and a test asserts the callout's floor sits above the
tallest bar.

---

## 2. The luck ledger card, and its sign convention

Round 1 shipped the waterfall under the `luck_ledger` name. The maintainer needed help
reading it, which is a fair verdict on a chart type most people have not been
taught. A waterfall answers *how did the margin get from there to here* — an
article's question. The person scrolling past asks *who got the breaks, and on
what*.

So `luck_ledger` is now a portrait share card in the baseball simulator's own
shape: title, matchup, the final and the two shares, two rounded team boxes with
each club's net luck, then "Biggest luck swings" — top five on each team's own
plays, striped, in three columns (`Event` · `What happened` · `Points`). The
waterfall survives under the new `waterfall` suffix as the article figure, and
`render.SUFFIXES` grows to four.

### 2a. The sign convention, and the two quantities it distinguishes

Stated on the card itself, under the section heading:

> *Points are what the scoreboard gave the team beyond what the play deserved.*

So Green Bay missing a 41-yard field goal prints **−3.2** in Green Bay's table:
the scoreboard gave the Packers 3.2 points fewer than the kick deserved.

The card carries **two different quantities**, and this round found out the hard
way that they have to be named:

- **The headline is the game's net luck, signed toward that team.** Luck is
  zero-sum — there is one scoreboard, so a point it gave one team beyond what
  the play deserved is a point it took from the other — which is why the two
  headlines are one number with two signs, and why that number is exactly the
  gap between the actual margin and the deserved one. Verified on all five
  games: the pair sums to 0.00e+00 and matches |actual − deserved| to at worst
  3.55e-15.
- **A table lists only that team's own plays**, each row signed toward that team.

The two can have **opposite signs**, and on `DEN_WAS_27-26--86-14_luck_ledger.png`
they did: Denver's headline read −2.3 over six green rows summing to +2.3.
Denver's own plays were lucky; Washington's were luckier still. Both numbers are
right and they are not the same number. Each headline now carries a small
`NET LUCK` label — the baseball card's own "EST. PRODUCTION" device — and the
section subtitle reads "Top 5 on each team's own plays". **This is an open
presentation question, not a closed one** (§6, defect D-1).

Anything past the top five folds into one row, `and n more, ±x.x`, carrying the
exact sum of what it replaces — folding is not dropping. A row worth less than a
tenth of a point prints two decimals, because "+0.0" reads as a rounding failure
rather than as a small number.

The card **refuses a ledger that does not reconcile with its verdict**, exactly
as the waterfall does. A decomposition printed under a headline it does not
explain is worse than no decomposition at all.

### 2b. The waterfall's colours were backwards

The worse of the two round-1 problems, and not the chart type. A bar was
coloured by the direction *neutralising* it moves the margin, so Green Bay
missing a field goal was drawn in Green Bay's green — while being Detroit's
lucky break. Anyone who knows the game read the figure the wrong way round.

Bars now wear the colour of the team the break **helped**, and the legend says
`luck that helped DET`. On `LV_KC_9-48--0-100_waterfall.png` every bar is now
Kansas City red and the figure says in one glance what took a paragraph before.

Also on the waterfall: the title and subtitle are anchored at the figure's left
margin rather than at the axes (sentence-long row labels had pushed the axes a
third of the way across, and the title went with it, leaving a hole where the
title should be); each row gains its club's mark immediately left of its own
label; and the distribution's luck arrow runs down a right-hand rail with the
same sentence on it, so the two figures say the same thing about the same game.

---

## 3. The clash rule, made colour-vision-aware

Round 1's rule measured Euclidean distance in RGB. It is cheap and blind.

`2016_14_NYJ_SF` is what it was blind to. The Jets' `#003F2D` and the 49ers'
`#AA0000` are **0.42 apart in RGB** — comfortably "separate" — and **5.2 apart
in OKLab under protanopia**, under the 6 the `dataviz` skill calls a floor. The
figure shipped with two bars a colourblind reader sees as one, and nothing in
the pipeline said so.

`style.py` now carries that skill's own arithmetic, ported rather than
approximated because the simulation model is part of the calibration: Machado,
Oliveira & Fernandes (2009) at severity 1.0 in linear RGB, OKLab ΔE ×100, and the
floors — CVD floor 6 (target 8), normal-vision hard floor 15, WCAG contrast 3:1
against the surface.

`teams.resolve_pair` walks a ladder and names the rung that fired. The cheap RGB
rule stays as the first check; the four readings decide. A real club colour comes
before a synthetic tint, because a reader who knows the Buccaneers knows their
pewter and nobody knows a 45%-lightened red.

### 3a. Fallback counts — all 992 ordered matchups

`research/60_matchup_colours.py`, 32 clubs, home and away not symmetric:

| rung | matchups | example |
|---|---:|---|
| primaries, untouched | 738 | GB @ DET |
| away secondary | 131 | NYJ @ SF, TB @ ATL |
| home secondary | 69 | GB @ ARI, SF @ ATL |
| away primary lightened | 37 | GB @ DAL |
| away secondary lightened | 2 | TB @ GB, TB @ PHI |
| away primary darkened | 9 | SF @ KC, PHI @ LAR |
| away secondary darkened | 5 | PHI @ DAL, PHI @ SEA |
| home primary darkened | 1 | KC @ SF |
| **unresolved** | **0** | — |

Worst-case CVD separation across the league: **min 6.1, median 24.4**.
Normal-vision separation: **min 15.1** against the hard floor of 15.
`60_matchup_colours.png` is a 32 × 32 swatch grid of every decision.

**17 of the 992 land in the 6–8 warning band.** That band is legal *here* and
only here, because every figure in this product carries a legend, a direct label
or the club's own mark — colour is never the only thing telling two bars apart.
Counted rather than assumed.

### 3b. Three deviations, each forced by the sweep

**The handoff's four-rung ladder leaves 15 ordered matchups unresolved.**
Philadelphia's midnight green against seven opponents, Kansas City's red against
San Francisco's, Detroit's blue against the Chargers' and the Dolphins', and
three more. Every one fails the same way: the pair needs separating in
*lightness*, and on a cream surface every candidate light enough to separate is
too light to read against the background. Three darkening rungs were added and
take the count to zero. The baseball chart lightens only because it never ran a
contrast check.

**KC/SF was already broken, and the RGB rule could not see that either.** Two
reds **12.9 apart under normal vision** — below the hard floor of 15, which no
amount of labelling excuses. Round 1's test asserted the pair was left alone
(document 41 §8 called it "not a colour clash, 0.325 apart"); that reading was
correct about RGB and wrong about the reader. Both colours now move. San
Francisco at home against Kansas City is the single matchup in the league that
needs the home club's colour darkened, because neither club's secondary — the
Chiefs' gold, the 49ers' tan — reads on cream at all.

**New Orleans' `#D3BC8D` reads at 1.78:1 against the cream**, below the 3:1
floor, and cannot be fixed by moving anybody else. The incumbent colour is
therefore not gated on contrast — only the substitute is — because gating it
would make all 31 games the Saints host *unresolvable* rather than merely
low-contrast. The sweep reports it rather than passing it silently (§6, D-2).

**Also fixed:** the team-table fixture in `tests/test_teams.py` carried
`#FF7900` for Tampa Bay's secondary where nflreadpy 0.1.5 ships `#322F2B`. The
fixture's stated job is to mirror the real table; it did not.

---

## 4. Overtime: one line on a share image, a panel in an article

Round 1's review: the sidebar is overwhelming on a share image. Six paragraphs
of methodology beside a card is an article, not a post.

No share image carries it. `dtw`, `luck_ledger`, `card` and `waterfall` each
state document 16's refusal in one muted line — *"Went to overtime; the coin toss
is reported, not neutralized."* — and the panel moves to
`render_game(game_id, article=True)`, which writes one extra
`{...}_dtw_article.png` for an overtime game and nothing at all for a regulation
one. `ARTICLE_SUFFIX` is deliberately not a fifth member of `SUFFIXES`.

There was a second, quieter reason. The sidebar **grows the figure**, so an
overtime game and a regulation one came out at two different widths and a
timeline crops them differently. The share images are now the same size whatever
happened after the fourth quarter, and that is tested.

The distribution's overtime line is appended to the interval caveat *after* that
caveat has been wrapped, rather than drawn as a second object below it: the
caveat wraps to one line or two depending on the game, and an annotation at a
fixed offset lands on the second line half the time.

The card gains the one line its frozen layout permits, under the interval line
and in the same muted size.

---

## 5. Figure-rule review — every PNG, opened and read

Thirty-one images: 8 variants, 20 share, 2 article, 1 swatch grid. What the
reading found, and what was done about it.

| PNG | Note |
|---|---|
| `59_*_V1` (both games) | Callout printed across the three tallest bars. **Fixed** — annotation headroom, with a test. |
| `59_2025_17_DET_MIN_V1` | Eleven visible spikes; the picket-fence reading that decided the bin width. |
| `59_*_V3`, `V4` | Arrow head at the actual margin, pointing the way luck pushed, label agreeing. Reads correctly on both games. |
| `59_2025_17_DET_MIN_V4` | The callout asserts a 55% share on a game the pill calls "too close to call". Factually a statement about the simulations, not a verdict — flagged, not changed (D-3); closed in round 3, the callout now declines with the pill. |
| `GB_DET_*_luck_ledger` | First draft at 12 in drew the home accent bar through the away team's folded row; "vs" sat under both boxes. **Both fixed**, with tests. |
| `NYJ_SF_*_luck_ledger` | Three-event team left a hole down the middle. **Fixed** — sections flow and centre. |
| `DEN_WAS_*_luck_ledger` | Headline −2.3 over six green positives. **Fixed** — `NET LUCK` label and the reworded subtitle (§2a). |
| `LV_KC_*_luck_ledger` | Correct: six-event and nine-event tables, both folded. |
| `GB_DET_*_waterfall` | Title now at the figure margin; row marks legible; colours read forwards. |
| `LV_KC_*_waterfall` | Every bar Kansas City red — the single-direction game reads in one glance. |
| `DET_MIN_*_waterfall` | Value labels on sub-half-point bars sit close to their own bar edge. Cosmetic, not fixed (D-4). |
| `NYJ_SF_*_dtw`, `*_waterfall` | The round-1 protan defect is gone: 5.2 → **36.5**. |
| `DEN_WAS_*_dtw` | Document 37 §7a's close-margin lift still holds at −3.3 vs −1. |
| `*_card` (all five) | Unchanged but for the overtime line, as instructed. |
| `NYJ_SF_*_dtw_article`, `DEN_WAS_*_dtw_article` | Sidebar attached, footer still present, plot at its unwidened size. Trailing space under the plot where the panel is taller (D-5). |
| `60_matchup_colours.png` | Every cell two distinct halves. The New Orleans row was visibly pale — D-2, drawn rather than hidden; closed in round 3, and the row now reads as dark as the rest. |

---

## 6. Defect register

**D-1 — the card's headline and its columns are different quantities
(closed, round 3).** The headline was the game's net luck signed toward a team
while the table listed that team's own plays, so a reader who added Denver's
column got +2.3 against a −2.3 headline. Round 2 labelled the two quantities;
round 3 made them one. The headline is now `TeamLuck.own_points` — the sum of
the table under it, including the folded row, to 1e-9 — and the game's net luck
moved to the lane between the boxes as `Net luck: WAS +2.3`, where it is a fact
about the matchup rather than about a club. The maintainer's call was the third option:
neither a subtotal row nor a further label, but a different quantity in the
headline. See §7.

**D-2 — New Orleans' primary is below the contrast floor (closed, round 3).**
`#D3BC8D` at 1.78:1 on `PALETTE["bg"]`. Closed by a contrast floor in
`teams.readable_colours`, one level above the pair rule: a club whose primary
fails 3:1 on the surface wears its secondary everywhere, so the Saints are drawn
in `#101820` in every figure rather than repainted per matchup. The sweep count
went **31 matchups under 3:1 → 0**, `unresolved` stayed **0**, and the
`primaries` rung moved 738 → 732 as the Saints' new black met the league's other
blacks. `research/60_matchup_colours.py` now prints the count on every run
whether or not it fires; a missing line and a zero look the same on a console.

**D-3 — the callout states a share on a game the product refuses to call
(closed, round 3).** `MIN deserved to win 55% of simulations` sat beside a "too
close to call" pill: one figure, two verdicts. Inside the band the callout now
reads `MIN 55% · DET 45% — too close to call` in `PALETTE["text"]`, because the
sentence belongs to neither club. Outside the band nothing moved. A degenerate
game draws no callout at all — see §7 for the one caveat on that. The wording
was a settled decision and the maintainer settled it again.

**D-4 — waterfall value labels crowd very small bars (closed, round 4).** A bar
worth 0.2 points is narrower than its own label, so the label sat against the
bar's edge — and on `2025_17_DET_MIN` the `-0.3` and `-0.01` labels printed
through the dashed zero rule beside them, which reads as a number struck out.
Closed by `fix(waterfall): rename, shade sides, anchor bars, tip labels; close
D-4`, in three parts: every value label moved to its bar's **tip**, the end away
from the running total, so it lands over the half of the axis belonging to the
team the break helped; every value label gained the module's cream `_shielded`
backing, so no rule can strike one through; and a bar under `LEADER_FLOOR`
(0.5 pt) has its label pushed clear of the bar and joined back to it by a
leader. Pinned by a regression test on `2025_17_DET_MIN`'s own ten bars: every
label carries a surface, the two sub-half-point bars carry leaders, and no two
labels overlap. Document 51 §D is the round's record.

**D-5 — the article figure has trailing space under its plot (cosmetic).**
`attach_overtime_sidebar` spans the figure's full height by design (the
waterfall's height changes with its row count), so a seven-paragraph panel is
taller than a 4-inch distribution.

**D-6 — the most-likely scoreline can read as a tie on a game the product
calls (closed as moot, round 5).** Round 5 withdrew the team-points share image
— a margin swing is not a per-team points swing — and this defect was a defect
of that figure's second subtitle line. Nothing draws the most-likely scoreline
any more, so there is nothing left to fix or to disclose. The reasoning below is
kept because it is the measurement, not the verdict: the modal *joint* bin pair
agreed with the marginal modes on all five example games, so the tie was the
three-point bin meeting a one-point game rather than the estimator. Document 51
§R5-A is the round's record.

*Round 4's statement, for the record.* The share image's second subtitle line is
each team's modal three-point bin, and on `2025_13_DEN_WAS` both modes land in
the same 21-24 bin: `Most likely: DEN 23 - WAS 23` sits under a pill that says
the scoreboard holds and a callout that says DEN deserved to win 86% of
simulations. It is not the estimator's fault — the modal *joint* bin pair is the
same 23-23 on that game, and agrees with the marginal modes on all five example
games — it is what a three-point bin does to a one-point game. Disclosed rather
than fixed: naming a tie is honest about two distributions that genuinely
overlap, and narrowing the bin would comb the histogram. The maintainer's call.

**Open after round 4:** D-5 and D-6, both cosmetic-to-semantic. D-1, D-2, D-3
and D-4 are closed above.

**Open after round 5:** D-5 alone. D-6 is closed as moot above; D-1 through D-4
were already closed. Round 5 found and closed three new layout defects in the
same commit that introduced their cause, so none of them reached this register —
they are recorded in document 51 §R5-D. One thing is raised and not fixed: the
distribution now reads its axis unsigned with two direction labels while the
waterfall still reads `final margin (DET − GB)` with signed ticks. That is a
scope boundary, not a defect, and it is the maintainer's call whether the waterfall
follows.

**Closed in round 2:** the round-1 protan defect on `2016_14_NYJ_SF` (document
41 §6, §8) — resolved by §3, from ΔE 5.2 to 36.5. Round 1's KC/SF finding is
superseded: it is a clash, just not an RGB one.

**Standing:** the two committed caveats keep the word "re-flip" ("Every re-flip
lands the same way…"). They are verbatim quotations from document 33 carried
under the round's no-new-statistics constraint, so the figure's headings say
"simulations" while its footnotes say "re-flip". Rewording them is a decision,
not a fix.

---

## 7. Round 3 — the wording, 2026-08-27

Three decisions the maintainer made after reading this document, implemented on the same
branch with one commit each. Nothing was fitted, no statistic is new, and the
replay stayed at 0.00e+00 for all five games. 502 tests pass, 22 written first
and 3 replaced.

**V4 is confirmed** as the shipped distribution figure and the darkening rungs
in the clash ladder are accepted as implemented. Neither needed work.

**(a) The ledger card's headline is the luck on that team's own plays**
(`41be6fa`, closes D-1). Both boxes carry the same three lines — `LUCK ON OWN
PLAYS`, the headline, the event count — and the headline is the sum of the table
under it. The lane between the boxes carries `vs`, the game's net luck in the
favoured club's colour, and the two margins as one sentence about the scoreboard
winner: `DET won by 8, deserved to lose by 8.3`. `NET LUCK` and the
`Actual margin +8 → deserved −8.3` arrow are gone. The boxes narrowed from
3.10 in to 2.55 in to make the lane readable, and the sentence breaks at its
comma rather than wherever the measuring lands.

On `2025_13_DEN_WAS`, the card this defect was raised on: both headlines are now
green (+2.2 and +4.5) over green columns, and the lane reads `Net luck: WAS
+2.3`. Nothing on the card contradicts anything else on it.

**(b) The callout declines with the pill** (`f739cce`, closes D-3). Inside the
band it reads `MIN 55% · DET 45% — too close to call` in ink. Outside the band
the wording and the club colour are unchanged, and a test pins that.

One caveat worth the maintainer's eye: the handoff also specified that a degenerate game
draws no callout, which is new behaviour rather than preserved behaviour.
`LV_KC_9-48--0-100_dtw.png` has therefore lost `KC deserved to win 100% of
simulations`. Opened and read: the title, the pill, the subtitle and the
degeneracy caveat all still carry the 100%, and the luck arrow was already
suppressed on degenerate games for the same reason. Easily reversed.

**(c) A club under the contrast floor wears its secondary everywhere**
(`1296dc1`, closes D-2). The floor moved from the pair rule up to the club
lookup, where a club's identity is chosen rather than a pair's:
`teams.readable_colours` returns the secondary first when the primary is under
3:1 on the surface, and darkens the darker of the two until it reads if both
fail. New Orleans is drawn in `#101820` in every figure rather than repainted
per matchup, and every downstream caller inherits the floor without asking.

| sweep, 992 ordered matchups | before | after |
|---|---|---|
| colours under 3:1 on the cream | 31 (all NO) | **0** |
| `unresolved` | 0 | 0 |
| `primaries` rung | 738 | 732 |
| worst-case CVD, min / median | 6.1 / 24.4 | 6.1 / 25.0 |
| normal-vision separation, min | 15.1 | 15.1 |

The `primaries` rung lost six matchups because the Saints' new black now meets
the league's other blacks and has to be separated like any other pair.
`research/60_matchup_colours.py` prints the contrast count whether or not it
fires: a missing line and a zero look the same on a console, and only one of
them is a check.

**D-4 and D-5 stay disclosed**, unchanged, by decision.

---

## 8. What these rounds did not do

The logo, the `@[TBD]` handle, the merge to `main`, the community write-up, and
rendering all 2,761 games. All four remain out by decision.
