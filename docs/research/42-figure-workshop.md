# 42 — Figure workshop, round 2

**Date:** 2026-08-26
**Branch:** `feat/product-bootstrap-plot`
**Handoff:** `docs/research/handoff-2026-08-26-round2.md`
**Scripts:** `research/59_dtw_variants.py`, `research/60_matchup_colours.py`,
`research/58_brand_figures.py` (updated)
**Code:** `src/nfl_simulator/plots.py`, `render.py`, `teams.py`, `style.py`
**Log:** `docs/research/log-2026-08-26-round2.md`
**Round 1's record is document 41 and stands.**

**Nothing here is fitted and no number is new.** Every value on every figure is
read from `dtw_games_v13.parquet`, `dtw_ledger_v13.parquet`,
`26_overtime_games.parquet` and `model_metadata_v13.json`. The one quantity
recomputed is each game's bootstrap draws, which the shipped summary does not
keep, and `render.replay` checks the redraw against the published number before
a pixel is drawn: all five example games replay at **0.00e+00**.

---

## 1. The distribution — four variants, and why V4 ships

the maintainer's round-1 verdict: the figure "makes sense the more you read it" but is not
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

Round 1 shipped the waterfall under the `luck_ledger` name. the maintainer needed help
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
| `59_2025_17_DET_MIN_V4` | The callout asserts a 55% share on a game the pill calls "too close to call". Factually a statement about the simulations, not a verdict — flagged, not changed (D-3). |
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
| `60_matchup_colours.png` | Every cell two distinct halves. The New Orleans row is visibly pale — D-2, drawn rather than hidden. |

---

## 6. Defect register

**D-1 — the card's headline and its columns are different quantities (open).**
The headline is the game's net luck signed toward a team; the table lists that
team's own plays. They can have opposite signs, and a reader who adds Denver's
column gets +2.3 against a −2.3 headline. Both are labelled now (§2a) and no
partition of the rows can make a per-team column sum to a zero-sum net — the net
is "my gross luck minus yours" and spans both tables. Whether to add a per-team
subtotal, or to reword further, is the maintainer's call.

**D-2 — New Orleans' primary is below the contrast floor (open, disclosed).**
`#D3BC8D` at 1.78:1 on `PALETTE["bg"]`. Not fixable by the pair rule; fixing it
means repainting a club in every figure it appears in, which is out of this
round's scope. Reported by `research/60_matchup_colours.py` on every run.

**D-3 — the callout states a share on a game the product refuses to call
(open).** `MIN deserved to win 55% of simulations` sits beside a "too close to
call" pill. The sentence is true — it is a statement about the simulations, not a
verdict — and the pill is immediately above it. Flagged for the maintainer rather than
reworded, because the callout's wording is a settled decision.

**D-4 — waterfall value labels crowd very small bars (cosmetic).** A bar worth
0.2 points is narrower than its own label, so the label sits against the bar's
edge. Inherent to a waterfall with a wide dynamic range; the labels stay inside
the frame, which is the property that is tested.

**D-5 — the article figure has trailing space under its plot (cosmetic).**
`attach_overtime_sidebar` spans the figure's full height by design (the
waterfall's height changes with its row count), so a seven-paragraph panel is
taller than a 4-inch distribution.

**Closed this round:** the round-1 protan defect on `2016_14_NYJ_SF` (document
41 §6, §8) — resolved by §3, from ΔE 5.2 to 36.5. Round 1's KC/SF finding is
superseded: it is a clash, just not an RGB one.

**Standing:** the two committed caveats keep the word "re-flip" ("Every re-flip
lands the same way…"). They are verbatim quotations from document 33 carried
under the round's no-new-statistics constraint, so the figure's headings say
"simulations" while its footnotes say "re-flip". Rewording them is a decision,
not a fix.

---

## 7. What this round did not do

The logo, the `@[TBD]` handle, the merge to `main`, the community write-up, and
rendering all 2,761 games. All four remain out by decision.
