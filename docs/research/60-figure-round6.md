# 60 — Figure round 6: the two editions on the page

*2026-08-28, on `fix/figure-round-6` off `main` at `6f1f598`. Nothing here is
fitted and no gate is read. Ruling R-4 (document 58 §2) gave the simulator two
named adjudications and round 8 shipped both in the code; this round is the
first time either of them is named on an image. Every number the figures state
is read from a summary on disk and re-checked against it before a pixel is
drawn — this round adds the second summary that check needs.*

---

## 1. What a reader sees now that they did not before

Three things, and the third is the one that changes what the product is.

**Every image says which edition it is.** The top-right corner reads
`Full edition · Data: nflverse | @[TBD]` or `Strict edition · …`, and the
edition is in the filename as well — `LAC_HOU_12-32--52-48_full_card.png`. Two
editions of one game are two files that cannot overwrite each other.

**On a charted game, the other edition is one muted line at the foot of every
figure.** `Strict edition: HOU 100% · LAC 0% — deserved margin HOU by 23.3`
under a Full image; `Full edition: DET 53% · MIN 47% — deserved margin DET by
0.3` under a Strict one. On a game before 2022 the line reads `Strict edition
only — charting begins in 2022.` Nothing else on the image is ever the other
edition's number: the headline, the pill, the callout, the bar and the interval
are the rendered edition's alone.

**The Full edition is the headline on 2022 and later.** `render_game` with no
edition argument draws Full from the first charted season and Strict before it.

That third point is what makes the first two load-bearing. On
`2024_19_LAC_HOU` the two editions disagree completely: Strict makes it
Houston's game beyond any doubt — 100% and a deserved margin of +23.3 on a
20-point win — and Full, which prices the two dropped touchdowns near 9 EPA
apiece, makes it a coin flip at 48% and moves it from *scoreboard holds* to *too
close to call*. A reader who saw one of those images without knowing which
adjudication it was would have no way to reconcile it with the other.

---

## 2. Part A — a Full summary the figures can replay against

`dtw_games_v13.parquet` is the Strict summary and `render.replay` has always
checked the redrawn bootstrap against it before drawing. Full had no summary, so
a Full figure had nothing to be checked against.

`research/76_full_edition_summary.py` writes one. Every 2022–2025 game is
simulated at v1.3's exact settings with both hands-on-the-ball models switched
on, through `research/73`'s own `variant_pass`, and the pass is checked against
document 59 §4 before the file is written:

```
bucket moves 200 against 200 (tolerance ±5); median |ΔDTW| on the 1,138
affected games 3.85 pp against 3.85 (tolerance ±0.2) -> PASS
wrote research/outputs/full_summary.parquet  (1,139 rows)
```

Exact on both, not merely inside tolerance.

`render.Sources` now carries both summaries and `game_row(..., edition=)` reads
the one asked for, never falling back to the other. `replay(..., edition=)`
simulates the named edition and checks it against that edition's own row.
`check_edition` refuses an edition nobody named — `strict+dp` is callable in the
simulator and has no public name — and refuses Full on a pre-2022 game **before**
anything is loaded, because both variant builders warn and return empty there
and the render would otherwise come back with a Strict ledger under a Full stamp.

**Both editions replay at exactly 0.00e+00** on all seven games, in both
directions, with the wider frame loaded:

```
2018_05_GB_DET   strict 0.00e+00    2025_17_DET_MIN  full 0.00e+00 / strict 0.00e+00
2021_14_LV_KC    strict 0.00e+00    2025_13_DEN_WAS  full 0.00e+00 / strict 0.00e+00
2016_14_NYJ_SF   strict 0.00e+00    2022_13_WAS_NYG  full 0.00e+00
                                    2024_19_LAC_HOU  full 0.00e+00
```

### 2a. The edition is not the `variant`, and this is a decision

Of the 1,139 games in 2022–2025, only **1,033** come back with a ledger labelled
`"full"`. 105 are `"strict+rd"` and one is `"strict"`, because the charter called
no interceptable throw in them. `SimulationResult.edition` returns `None` for
those labels **by design** (document 58 §2: an audit arm has no public name).

So the render labels from **the edition asked for** — which components were
switched on — not from the ledger's `variant` string. A Full-edition game whose
charting held no interceptable throw is still a Full-edition figure with an
empty dropped-pick set; nothing was excluded from it. `variant` continues to
describe the ledger, which is what the audits read it for.

---

## 3. Part B — the edition on every image

`style.edition_stamp` puts the edition in front of the nflverse credit and
`finalize(..., edition=)` stamps it; an unnamed edition cannot be stamped.
`GameVerdict` carries its own `edition` and the other edition's whole verdict as
`counterpart`, and `edition_note()` builds the muted line from it — from the
same `headline()` the image's own title uses, so the two roundings cannot
disagree.

The counterpart is read from the other edition's published summary and **never
replayed**. The replay check exists for the distribution a figure draws; this
verdict is drawn by nothing and contributes a headline and a margin to a footer,
so re-simulating a second adjudication to print two numbers would double every
render for no guarantee the committed record does not already give.

A charted verdict built **without** its counterpart on hand prints nothing rather
than the pre-charting line, which would be false. Silence is the only safe
degradation there.

`footer_lines` puts the overtime note and the edition line in one order on all
four figures. Two layout constants moved to make room: the ledger card's table
band bottom (0.34 → 0.56 in) and the game card's footer block, which is now laid
out from a fixed top rather than at fixed heights — an overtime Full card
carries three lines where a regulation Strict one carries two.

---

## 4. Part C — the ledger under Full

A median Full ledger holds about **fifty** events; `2022_13_WAS_NYG` holds 77 and
`2024_19_LAC_HOU` 63. Fifty bars is a table with a dashed line down it.

**The grouping rule.** `group_rows(bars, threshold=1.0)`:

- every event worth **a point or more** keeps its own row;
- the remaining events of amendment A-3's two components fold **per component
  and per team** — `5 smaller receiver drops (HOU)`, `3 smaller dropped picks
  (WAS)` — because that is a fact about one club's afternoon and can wear that
  club's mark;
- everything else folds into the single un-teamed `n events under 1 pt` row the
  waterfall has always had, since a game carries a handful of Strict events
  rather than dozens;
  **Amended 2026-08-29 (round 10, §11).** That row is gone: the remainder
  splits by charged team into `n small events (LAC)` and `n small events
  (HOU)`, one row per club, each wearing that club's mark and carrying its own
  exact sum. And no row draws an empty bar — anything under `DRAW_FLOOR =
  0.05` pt, a single event or a fold, is absorbed into its club's row
  regardless of the lone-event rule below.
  **Amended again 2026-08-29 (round 11, §11a).** The floor is a share of the
  axis, `DRAW_FLOOR_SHARE = 0.005`, and it no longer overrides the lone-event
  rule: a club whose remainder is one event keeps that event's own words.
- every folded row carries the **exact sum** of what went into it, so the
  waterfall still reconciles its two ends to 1e-9;
- a lone small event is left as itself — `1 smaller receiver drops (GB)` is a
  worse row than the drop it hides;
- ordering is by |points|, biggest first, as before; the chronological reading
  is not grouped, because a folded row has no place on a timeline.

Result across the nine renders: **no waterfall exceeds 16 event rows**, from as
many as 77 events.

| Game | Edition | Events | Bars |
|---|---|---|---|
| `2018_05_GB_DET` | Strict | 15 | 9 |
| `2021_14_LV_KC` | Strict | 15 | 6 |
| `2016_14_NYJ_SF` | Strict | 8 | 4 |
| `2025_17_DET_MIN` | Full | 42 | 10 |
| `2025_17_DET_MIN` | Strict | 12 | 6 |
| `2025_13_DEN_WAS` | Full | 74 | 9 |
| `2025_13_DEN_WAS` | Strict | 12 | 4 |
| `2022_13_WAS_NYG` | Full | 77 | **16** |
| `2024_19_LAC_HOU` | Full | 63 | **16** |

**The wording.** The two components read as sentences now:
`LAC receiver drop · Dissly, dropped (95% catch)` and
`HOU dropped pick · thrown by Stroud, escaped (58% catch)`. Both branches are
quoted at the **catch** probability, which is the one number that carries across
the two components — a 95% catch that was dropped and a 58% catch that escaped
are the same kind of statement about how likely the ball was to be caught. The
dropped pick stores the probability the ball *escaped*, so it is turned round
here rather than printed as stored.

The names come from `passer_player_name` and `receiver_player_name`, added to the
loaded frame as **presentation only** the way `kicker_player_name` was in round
4. Nothing prices on either — the dropped pick was priced at the defence's shrunk
rate and the drop at the receiving corps' — and the replay is still 0.00e+00 with
both columns loaded. A play with no name on file keeps its bare label.

The card's "Top 5 on each team's own plays" is unchanged; its box headline still
sums all of that team's own-play luck including the grouped remainder, which is
now pinned by a test.

---

## 5. Part D — intervals, and the axis

**Intervals on the waterfall's row labels.** `88% kick, 83–92`;
`96% catch, 94–98`. Document 03's 5.5/94.5 convention, the same one the DTW
interval uses.

Reading them off required one change below the product layer. Every `LuckEvent`
carries `expected_draws` — the whole posterior on that branch — and
`LedgerEntry` keeps only its mean, which is what the arithmetic needs and all
that any artifact on disk holds. `SimulationResult` now returns the events too,
and `render.expected_intervals` takes the percentiles keyed by **play and
component**: four blocked field goals in the shipped population also book a
fumble row on their own play id, and a map keyed on the play alone would hand
one of those two rows the other's probability. The dropped pick's bounds are
mirrored onto the catch branch, the way the probability itself already is.

The waterfall asks for them; the ledger card keeps the short form, because a
spread on a share card is a third number competing with the two the card is
about.

What the intervals buy is visible on `2018_05_GB_DET`: five Green Bay misses
priced at 88%, 86%, 91%, 63% — and now the 56-yarder reads `63% kick, 51–74`
against the 38-yarder's `91% kick, 87–94`. The spread widens with the difficulty
of the kick, which is a fact the mean alone hid.

**The axis.** The waterfall's x axis is now the distribution's: unsigned ticks,
no `final margin (MIN − DET)` title, and the two club-coloured direction labels
under it — `← DET wins by` / `MIN wins by →` — drawn by round 5's own helper.
Two figures putting a margin on an x axis had two conventions for one quantity,
and one of them asked the reader to subtract.

**One collision that measurement found and the eye did not.** Three things now
live under that axis where round 5 had two. Measured with the renderer, the
colour key's box crossed a direction label's vertical band on **seven of the nine
renders** — they only looked separate because they happened to be far apart
horizontally. The three offsets are now named constants laid out as one stack,
and a test asserts none of the three overlap.

---

## 6. What was rendered

Nine renders of seven games — the headline edition for each, plus Strict for the
two 2025 games so the two can be put side by side. 36 share PNGs and 4 article
PNGs (the four overtime games), all under `research/outputs/`, none committed.

| Game | Edition | Files |
|---|---|---|
| `2018_05_GB_DET` | Strict | `GB_DET_23-31--95-5_strict_{dtw,luck_ledger,card,waterfall}.png` |
| `2021_14_LV_KC` | Strict | `LV_KC_9-48--0-100_strict_{dtw,luck_ledger,card,waterfall}.png` |
| `2016_14_NYJ_SF` | Strict | `NYJ_SF_23-17--36-64_strict_{dtw,luck_ledger,card,waterfall,dtw_article}.png` |
| `2025_17_DET_MIN` | **Full** | `DET_MIN_10-23--53-47_full_{dtw,luck_ledger,card,waterfall}.png` |
| `2025_17_DET_MIN` | Strict | `DET_MIN_10-23--45-55_strict_{dtw,luck_ledger,card,waterfall}.png` |
| `2025_13_DEN_WAS` | **Full** | `DEN_WAS_27-26--60-40_full_{dtw,luck_ledger,card,waterfall,dtw_article}.png` |
| `2025_13_DEN_WAS` | Strict | `DEN_WAS_27-26--86-14_strict_{dtw,luck_ledger,card,waterfall,dtw_article}.png` |
| `2022_13_WAS_NYG` | **Full** | `WAS_NYG_20-20--2-98_full_{dtw,luck_ledger,card,waterfall,dtw_article}.png` |
| `2024_19_LAC_HOU` | **Full** | `LAC_HOU_12-32--52-48_full_{dtw,luck_ledger,card,waterfall}.png` |

The two 2025 games are the comparison pair. `2025_17_DET_MIN` goes from MIN 55%
to DET 53% — the same verdict bucket, the other side of the coin — and
`2025_13_DEN_WAS` from DEN 86% (*scoreboard holds*) to DEN 60% (*too close to
call*).

**751 tests pass** (684 before this round), ruff clean.

---

## 7. Register

**Closed this round.**

| Item | Where |
|---|---|
| The two editions are not yet rendered (document 59 §5) | §3 above |
| Kick and catch probability intervals on the waterfall labels | §5 |
| Align the waterfall's axis with the distribution's | §5 |
| A variant line on the ledger card | §3 — it is on all four figures, not only the card |

**Open, carried forward.**

| Item | Status |
|---|---|
| **D-5** — the article figure's trailing space under a short plot | **Open**, unchanged, as document 42 §6 and document 51 left it. Cosmetic |
| Document 42's other register entries | **Unchanged** |
| `w = 0.285` has no derivation in this repository | **Open**, as document 31 §9 |
| Gate C-2 fails at the receiver component's charged grain | **Open and reported**, document 59 §5 |
| `is_catchable_ball` may be graded partly off the outcome | **Open caveat**, document 59 §3 |
| A-3 clause 7's sunset | **Open, scheduled** |

**Raised this round, not acted on** — both are in the hypothesis queue as Parked:

1. **"receiver drop … caught" reads as a contradiction.** The component is named
   for the bad branch, so a *caught* ball comes out as
   `MIN receiver drop · Jefferson, caught (95% catch)`. The wording is the
   handoff's, implemented as specified; naming the event for the *situation*
   rather than the outcome — `catchable ball · Jefferson, caught (95% catch)` —
   would read correctly on both branches. ~20 min, and it touches only
   `COMPONENT_NAMES`.
2. **The waterfall's anchor colour fails the module's own colour-vision
   validator on one matchup.** `anchor_colour` steps ink to the neutral using
   document 42 §3's RGB clash floor (0.20). On `2022_13_WAS_NYG` ink is 0.302
   from the Giants' navy and so keeps ink — but in OKLab it is **13.9** from that
   navy under normal vision, against `style.NORMAL_FLOOR = 15.0`, and
   `style.separated()` returns `False`. The anchors are 1.4× the height of every
   event bar and are labelled `Actual:` / `Deserved:`, so the reading never rests
   on the colour; the rule is nonetheless one the repository already owns a
   better version of. Switching `anchor_colour` to `separated()` is ~15 min and
   changes which matchups get ink, so it is the maintainer's call rather than a round's.

---

## 8. Round 7 — the maintainer's notes on these figures

*Appended 2026-08-28, on `fix/figure-round-7` off `main` at `27d29e8`. Handoff
`handoff-2026-08-28-figures-r7.md`, results `results-2026-08-28-exp11.md`, log
`log-2026-08-28-figures-r7.md`. Five settled changes from a Fable 5 chat after
the maintainer read the round-6 PNGs. No statistic moves: all nine renders replay at
`0.00e+00` before and after. 751 → **784 tests**, ruff clean.*

### 8a. What a reader sees now

**A row says whose luck it is and who did the thing, and those are not always
the same club.** Round 6 put the charged team's name at the front of every row,
which is right for a fumble and wrong for a dropped pick: the offence is charged
— it threw an interceptable ball and got away with it — but the hands that
dropped it were the defence's. The mark stays the charged club's and the
sentence becomes the actor's, so `2024_19_LAC_HOU` now reads

```
[HOU mark]  LAC dropped pick · thrown by Stroud (58% catch)     +2.5
[LAC mark]  LAC drop · Dissly (95% catch)                       −7.4
```

— Houston threw it, Los Angeles dropped it, and it is Houston's fortune. The
folded rows follow: `5 smaller HOU drops`, `3 smaller NYG dropped picks`.

**The branch is named once.** `receiver drop · Jefferson, caught (95% catch)`
said the outcome twice and contradicted itself doing it — §7's first parked
item. The noun now follows the branch: `interception` or `dropped pick`,
`catch` or `drop`. `MIN catch · Jefferson (95% catch)` on the Full `DET_MIN`
render is that fix visible. The card's What-happened column still states the
verb, and the two agree by construction because both read `actual`.

**The marks are a column.** Round 6 hung each mark off its own label's start;
the labels are right-aligned and vary in length by a factor of three, so they
came out on a diagonal. The labels are left-aligned on one x now with the marks
just outside it, and the y ticks are gone — a tick that no longer touches its
label is a stray dash.

**The two anchors are bold and the rows carry the short probability.** Weight,
not size. And `(88% kick)` on the waterfall as on the card: round 6's
`(88% kick, 83–92)` made one number read two ways across two figures of one
game. The interval survives behind `plot_luck_ledger(show_intervals=True)` for
an article figure with room to ask for it, and its tests survive with it.

**The card's cells are sentences.** `Drop · Dissly`, `Escaped (58% catch)`,
`Recovered by LAC`, `Retained`; `41-yd field goal · Crosby` keeps its digit.

### 8b. The anchor colour — §7's second parked item, with a correction

`anchor_colour` now asks `style.separated()` **as well as** document 42 §3's RGB
floor, and takes the ink only where both allow it.

The swap as §7 proposed it was implemented first and rendered. It does step
`2022_13_WAS_NYG` to the neutral, as predicted — and it also steps
`2016_14_NYJ_SF` and `2021_14_LV_KC` the *other* way, to the ink, because OKLab
reads `#000000` and the ink `#1A1A1A` as 21.8 apart for every reader while RGB
reads them 0.177 apart. On the `NYJ_SF` render that verdict is wrong in the only
way that matters: the two anchors and the three Jets bars came out one black and
the figure could not say which bar was a total, which is the defect §7 of
document 42 closed in round 2. `style.NORMAL_FLOOR = 15.0` is calibrated for
thin categorical marks; these are the two largest blocks on the figure.

Requiring both rules produces exactly the one affected render the handoff
predicted. The two catch different failures — RGB the pair a full-colour reader
loses at size, OKLab the pair a colourblind reader loses at any size.

| pair | RGB (floor 0.20) | normal (15.0) | protan (6.0) | deutan | tritan |
|---|---|---|---|---|---|
| ink vs WAS `#5A1414` | 0.253 | 13.9 | **5.2** | 11.6 | 16.2 |
| ink vs NYG `#0B2265` | 0.302 | **13.9** | 14.5 | 13.7 | 9.7 |
| ink vs NYJ `#000000` | **0.177** | 21.8 | 21.8 | 21.8 | 21.8 |
| neutral vs KC `#E31837` | 0.596 | 24.9 | **4.3** | 16.1 | 28.6 |

Both Washington clubs fail, not only the navy §7 cited. The last row is new and
is not this round's to fix — see 8d.

### 8c. Register — closed this round

| Item | Where |
|---|---|
| "receiver drop … caught" reads as a contradiction (§7, raised) | 8a — the noun follows the branch |
| `anchor_colour` fails the module's colour-vision validator (§7, raised) | 8b — both rules now |

**Open, carried forward.** Everything in §7's "Open" table is unchanged: **D-5**,
document 42's other entries, `w = 0.285`, Gate C-2 at the receiver grain,
`is_catchable_ball`, A-3 clause 7's sunset.

### 8d. Raised this round, not acted on

**The neutral the anchors fall back to is itself under the colour-vision floor
against one club in the set.** `#5E5B55` is 4.3 from Kansas City's `#E31837` for
a protan reader, against `style.CVD_FLOOR = 6.0`. This is the same class of
defect 8b just fixed at the other end of the function, on a render that has
always used the neutral, and it cannot be closed by choosing between the two
colours `anchor_colour` already owns — it needs a third, or a non-colour
encoding on the anchor bars. The anchors remain 1.4× the bar height and labelled
`Actual:` / `Deserved:`, so nothing rests on the colour. In the queue as Parked.

## 9. Round 8 — the annotation band, and the cap on the page

the maintainer's notes on `LAC_HOU_12-32--52-48_full_dtw.png`, executed on
`fix/figure-round-8` from handoff `handoff-2026-08-28-figures-r8.md`. Two
things: the nine renders move onto document 62's capped Full summary and the
waterfall learns to draw the cap, and the Deserve-to-Win figure's crowded strip
becomes one band. Presentation only — hard constraint 1 was again that no
statistic move, and both editions replay at **0.00e+00** on all nine renders
after every part.

### 9a. What a reader sees now

**The Deserve-to-Win figure has one band above the bars, and it holds two
things.** `Deserved: HOU by 0.9` sits centred over the dashed rule and
`Actual: HOU by 20` centred over the solid one, both bold, both above the top
spine. Before this round four things shared the strip inside the plot: the two
labels hung to the right of their own rules, the luck arrow's sentence ran
between them, and a callout restated the subtitle's `DTW:` line and the verdict
pill a third time. The callout is gone from the share image and kept on the
article figure, which has the sidebar's audience and the room for a sentence.

**A label to one side of its rule is read as belonging to whatever it sits
over.** That is why they are centred now rather than merely moved: on a lopsided
game the right-hung `Actual:` label started over the *deserved* rule, which is
the line it is not about.

**When the two margins are close the labels stack rather than overprint.**
`Deserved:` takes the upper row and `Actual:` holds the lower one, which is the
row a reader's eye meets first coming up off the plot. Of the nine renders only
`2025_13_DEN_WAS` stacks, and it stacks in both editions — 1.3 against 1 in
Full, 3.3 against 1 in Strict. The rule is geometric: two boxes that overlap
with `CORNER_CLEARANCE`'s padding, never a bucket or a margin size.

**The lift is paid for.** The second row is cut out of the band the header sits
above, so `draw_header` takes the same room back and the whole block — heading,
divider, subtitle, pill — moves up together. Without it the lifted box came to
within about 13 px of the subtitle on `2025_13_DEN_WAS`: no overlap, and exactly
the case §6's `CORNER_CLEARANCE` was written for. The header only makes way on a
game that stacks.

**The luck arrow moved under the axis.** It spans the same two margins with its
head still at the actual end, drawn between the tick labels and the direction
labels, with `luck moved the margin 19.1 points toward HOU` centred under the
span. It now sits directly over the ticks that number the distance it measures.
The two direction labels and the footnote drop by the arrow band's own height so
the room is made rather than borrowed — on every game, not only on one that has
an arrow, because furniture that moves between games is furniture a reader
cannot compare across them. A degenerate game still draws no arrow.

**The waterfall draws the possession cap.** A cap row reads
`WAS Possession cap · Q2 drive 6` — the component's name and the drive verbatim,
no player and no probability, because a cap is not a branch anybody flipped but
a possession's booked luck bounded by its own largest "what if". Small ones fold
per club as amendment A-3's two components do, `4 smaller possession caps (WAS)`,
and the club goes in parentheses rather than in front of the noun: nobody
*performs* a cap. No new colour — a cap row wears the colour of the side its clip
helped, by the same rule as every other bar, and the label is the encoding.

### 9b. A render-path defect the round found

`render.edition_handles` never passed `simulate_game` an `edition`, and document
61's cap is keyed on that **argument** rather than on the variant the ledger comes
out carrying — the audit arms deliberately reach it with `edition=None`. So every
Full render replayed *uncapped*. Nothing was visibly wrong while the summary on
disk was also uncapped; the moment document 62's capped summary was put in place,
`research/58`'s own replay gate stopped on `2025_17_DET_MIN` at 5.7e-03 of DTW.

This is the gate working. It is worth recording as a decision that the render
path keeps its **own** copy of the edition name rather than inferring one: the
edition is what the maintainer asked for, and §2a already settled that it is not the
`variant` the ledger happens to carry.

### 9c. What was rendered

Nine share sets and four article figures, all replaying at `0.00e+00`, on the
capped numbers. Every named Full figure is renamed by them.

| Game | Edition | DTW file | Waterfall rows | Cap rows drawn |
|---|---|---|---|---|
| `2018_05_GB_DET` | Strict | `GB_DET_23-31--95-5` | 9 | — |
| `2021_14_LV_KC` | Strict | `LV_KC_9-48--0-100` | 6 | — |
| `2016_14_NYJ_SF` | Strict | `NYJ_SF_23-17--36-64` | 4 | — |
| `2025_17_DET_MIN` | Full | `DET_MIN_10-23--53-47` | 10 | 0 bars of 8 |
| `2025_17_DET_MIN` | Strict | `DET_MIN_10-23--45-55` | 6 | — |
| `2025_13_DEN_WAS` | Full | `DEN_WAS_27-26--59-41` | 10 | 1 bar of 15 |
| `2025_13_DEN_WAS` | Strict | `DEN_WAS_27-26--86-14` | 4 | — |
| `2022_13_WAS_NYG` | Full | `WAS_NYG_20-20--3-97` | **17** | 1 bar, a fold of 4, of 14 |
| `2024_19_LAC_HOU` | Full | `LAC_HOU_12-32--45-55` | **18** | 2 bars of 13 |

"Cap rows drawn" is bars against ledger rows: most cap rows are worth under a
tenth of a point and fold into that figure's `events under 0.1 pt` row inside
`luck_bars`, before `group_rows` ever sees them. `2025_17_DET_MIN` books eight
and draws none for that reason.

The two bolded counts exceed §8's "no waterfall exceeds 16 event rows", by
exactly the cap rows added. Sixteen was a measured result from round 6 taken
before cap rows existed, not a bound the code enforces — the synthetic
fifty-drop frame in `tests/test_plots.py` still holds at 16 — and the fold was
not tightened to get back under it, because "a lone small event is left as
itself" is a rule this round had no licence to change.

### 9d. Register — closed this round

| Item | Where |
|---|---|
| Four annotations share the strip above the bars (round 8, raised) | 9a — one band, two labels |
| The callout repeats the subtitle and the pill (round 8, raised) | 9a — share image drops it |
| The luck arrow crosses both rule labels (round 8, raised) | 9a — under the axis |
| A cap row has no words on the waterfall (document 62 §5, implied) | 9a — `Possession cap · Q3 drive 7` |
| Full renders replay uncapped (found this round) | 9b — `edition_handles` names its edition |

**Open, carried forward.** Everything in §8c's "Open" table is unchanged: **D-5**,
document 42's other entries, `w = 0.285`, Gate C-2 at the receiver grain,
`is_catchable_ball`, A-3 clause 7's sunset, and §8d's neutral-vs-KC protan gap.

### 9e. Raised this round, not acted on

**The corner-text rule cannot fire.** The handoff asks that `LAC wins` / `HOU
wins` go logo-only "when a rule label's box would overlap the corner text". With
the rule labels now above the top spine and the corner texts eight points below
it, the two are on different rows and their padded boxes never meet: all
eighteen corner texts survive on the nine renders, measured. The rule is
implemented, geometric, and tested both ways — it is a guard against a layout
that no longer occurs, not dead code, but it is not doing the job the maintainer's sketch
showed it doing. If the intent was that the corner text should go *unconditionally*
now that the `wins by` line under the axis carries the same key, that is a
one-line change and the maintainer's call, not this round's.

**A very small gap draws a very small arrow.** `2025_13_DEN_WAS` Full moves
0.35 points, and its span renders **17.8 px** wide against a head that is most of
that — honest, and at print size it reads as a stray `>` glyph under the axis
rather than as a span. A minimum drawn length, or
suppressing the arrow under some floor and keeping only the sentence, would both
be new rules rather than layout fixes, so neither was invented here. Parked.

**`Possession cap` is the only row label that opens with a capital.** Every other
row reads `LAC drop · Dissly`, `HOU fumble on a run`; a cap reads
`LAC Possession cap · Q4 drive 26`. The capital is the handoff's own settled
wording and was not re-litigated, but it is visible in a column of otherwise
lower-case rows. Parked.

## 10. Round 9 — the arrow floor, the cap's case, and the 3,900-game read

the maintainer's notes after the round-8 renders, executed on `fix/figure-round-9` from
`handoff-2026-08-28-figures-r9.md`. Two one-line rule changes, both of them
§9e's parked items, and one measurement round that rendered every game in both
editions for the first time. Presentation only — hard constraint 1 was again
that no statistic move, and both editions replay at **0.00e+00** on all nine
renders after every part, and on all 3,900 game-editions in Part C.

### 10a. What a reader sees now

**A luck gap under a point keeps its sentence and loses its span.** §9e measured
`2025_13_DEN_WAS` Full at 0.35 points and 17.8 px of drawn span, and parked the
fix because a floor would be a new rule rather than a layout tweak. `ARROW_FLOOR
= 1.0` is that rule: under it `_draw_luck_arrow` returns `(None, label)` and only
the sentence is drawn, at the offset it always used. The floor is exclusive-below
and inclusive-at — a gap of exactly 1.0 still draws its span — so the rule has one
edge rather than a band of games that may or may not have an arrow.

Part C measured what that costs: the span is suppressed on **465 game-editions**,
333 Strict and 132 Full, which is one non-degenerate game in five and about 12%
of every figure the product draws. It is not a corner case, and the sentence
carries the number on every one of them.

**The cap row joins the column's case.** `COMPONENT_NAMES["possession_cap"]` was
the only component name with a capital, so `LAC Possession cap · Q4 drive 26` sat
beside `LAC drop · Dissly` and read as the start of a sentence. It is now
`possession cap`. The ledger card's cell is unaffected — it is sentence case and
`sentence_case` puts the capital back — which is the general point the change
records in a comment: **the case belongs to where a label is drawn, not to the
label.** `POSSESSION_CAP_PLURAL` was already lower case, so `2 smaller possession
caps (HOU)` needed nothing.

### 10b. What the corpus said that nine games could not

Document 63 is the record. Three findings bear on this document's own rules:

1. **Stacking is the common case.** §9a's second row was built for a near-tie;
   measured over 3,900 games it fires on **93.2% of Strict and 94.6% of Full**.
   The mechanism works; the premise that it is rare does not survive the corpus.
2. **The corner-text rule fired zero times in 3,900 games**, confirming §9e's
   reasoning from nine renders across the whole population.
3. **Two pairs of Strict games share a filename** and silently overwrite each
   other — `figure_filename` carries no season or week, and the Raiders'
   relocation alias puts a 2018 Oakland game and a 2023 Las Vegas game on the
   same eleven characters. Eight PNGs of 11,044 were lost to it.

### 10c. What was rendered

Nine share sets, re-rendered after each of Parts A and B, all replaying at
`0.00e+00`; then every game in both editions — **15,600 PNGs**, 4 × 3,900,
matching the expected count exactly, in 20.3 minutes on 12 workers, with a worst
replay gap of `0.00e+00` across the whole corpus.

### 10d. Register — closed this round

| Item | Where |
|---|---|
| A very small gap draws a very small arrow (round 8 §9e, parked) | 10a — `ARROW_FLOOR = 1.0` |
| `Possession cap` is the only row label opening with a capital (round 8 §9e, parked) | 10a — the column's case |

**Open, carried forward.** Everything in §8c's "Open" table is unchanged, plus
every entry in document 63 §3 and the two structural findings in 63 §2.

### 10e. Raised this round, not acted on

Document 63 §3 is the list, with a game id and a severity per row. The three
that reach past layout and into a rule are the filename collision (a lost file,
not a cosmetic), the title running under the credit stamp on 84–89% of
distribution figures, and the dashed zero rule printing through a waterfall's
corner label on a lopsided game — that last one because the waterfall passes
`shield=False` where the distribution passes `shield=True`, which §7's comment
justifies on the grounds that the waterfall has nothing crossing that band. It
has: its own zero rule.

## 11. Round 10 — the name, the corner, the band, the shield and the heap

the maintainer's five settled fixes after reading document 63, executed on
`fix/figure-round-10` from `handoff-2026-08-28-figures-r10.md`, with the
whole-corpus re-run as the verification against eight pre-registered numbers.
Presentation only — hard constraint 1 was again that no statistic move, and all
**3,900 game-editions replay at 0.00e+00** after every part.

### 11a. What a reader sees now

**No two games can share a file.** `figure_filename` opens with the game id —
`2018_05_GB_DET_23-31--95-5_strict_dtw.png` — and the rest of the name is
unchanged. The old name was the game id with its season and its week taken off,
which is why two Miami–Jets games seven seasons apart and a 2018 Oakland game
beside a 2023 Las Vegas one wrote the same eleven characters. Eight PNGs of
11,044 were being lost silently. A game id is unique by construction.

**The credit stamp is in the bottom-right corner, below the footer.** It was in
the top-right, painted onto the saved pixels after layout, and the title ran
under it on 2,325 of 2,759 Strict distribution figures and 1,016 of 1,139 Full
ones. The title cannot see a stamp that is added after it is placed, so the
corner is the only thing that could move. `stamp_box` is public — the strip
reservation and the corpus read both need to know where the stamp will land —
and `finalize` grows the canvas at the bottom when the figure reached into the
stamp's own columns, so the room is made rather than borrowed. Only the stamp's
columns are consulted: both cards put their footers at the **left** edge, and
growing every image to clear a footer the stamp is nowhere near would change two
fixed shapes for nothing.

**The band above the plot is two rows, on every figure.** Round 8 built the
second row for a near-tie and lifted only on a measured collision; §10b reported
what the corpus says about that premise — the labels collide on **93.2% of
Strict and 94.6% of Full**, because each box is about 130 px wide and centred on
its own margin. The exception was the rule. `Deserved:` now takes the upper row
and `Actual:` the lower one always, and the header gives the same room back
always. Furniture that moves between games is furniture a reader cannot compare
across them. `_lift_colliding_label` keeps the conditional behaviour for the
team-points figure, whose axis is a score rather than a margin and which the
corpus was not read on.

**The waterfall's corner labels wear the distribution's shield.** §7 justified
`shield=False` here on the grounds that the waterfall has nothing crossing that
band. It has: its own dashed zero rule, which printed straight through `PIT
wins`, `ARI wins`, `JAX wins` and `TEN wins` on four lopsided games. The rotated
arrow sentence, which runs the height of the rail and on a short waterfall is
most of the figure, is lowered until its top clears the corner band — measured
rather than reserved, because how far it reaches depends on the row count and on
how many digits the number takes.

**Every bar on the page belongs to a club.** The un-teamed remainder splits by
charged team: `30 small events (HOU)` under Houston's mark and `23 small events
(LAC)` under the Chargers', each carrying its own exact sum. `81 events under 1
pt` at −2.1 pt was the third-largest bar on `2025_02_NYG_DAL` and wore nothing
at all. One row per club and not two — a bar that is already a heap joins its
club's heap whatever it is worth, because `luck_bars` folds under a tenth of a
point and `group_rows` under a point, and on `2024_19_LAC_HOU` the larger piece
cleared the threshold and stood beside a second row of the same words.

**And a floor on what is worth a row.** `DRAW_FLOOR = 0.05` pt: any row, single
event or fold, worth less than that is absorbed into its club's heap regardless
of the lone-event rule. A row that draws nothing tells the reader nothing while
still costing them a row to read.

**Amendment, 2026-08-29 — round 11 replaces that floor with a relative one.**
`DRAW_FLOOR` is removed and `DRAW_FLOOR_SHARE = 0.005` takes its place: a row
has to be worth half a percent of the waterfall's axis span to be worth drawing.
The span is `max(0, actual, deserved) − min(0, actual, deserved)`, taken from the
verdict by `waterfall_span` and fixed before any fold, so the floor cannot depend
on what the floor did. That is 0.015 pt on a three-point game and 0.25 pt on a
fifty-point one, which is what §11b's own closing paragraph asked for: whether a
bar can be seen is a share of the axis, not an absolute number of points.

Two clauses come with it, and they are what stops the floor eating rows it
should not:

1. **A heap of one is the event.** If a club's heap would hold exactly one
   event, that event is kept under its own label whatever it is worth. The old
   rule's "regardless of the lone-event rule" is withdrawn — it produced `1
   small event (SEA)` at +0.018 pt, which is the same invisible bar with the
   event's words taken off it, and eight of §11b's twelve re-derived rows were
   exactly that.
2. **A club heap of two or more that still cancels to under the floor is kept as
   it is and counted.** There is nowhere left to fold it, so it stays, and round
   11 reports the residue rather than hiding it — see document 63 §7.

Ordering is unchanged, and reconciliation is unchanged: the fold moves rows,
never points, on any span.

### 11b. What the corpus said, including where the round fell short

15,600 PNGs — 4 × 3,900, matching exactly — in 34.9 minutes on 12 workers, worst
replay gap `0.00e+00`. Eight of the nine pre-registered numbers landed exactly:
files on disk, the replay gap, zero title/stamp overlaps, zero corner strikes,
zero arrow-sentence overlaps, zero rows still labelled `events under`, zero rows
with no club, and all 3,900 distribution figures on two rule rows. Waterfall
rows rose by exactly one at both maxima — 15 Strict and 25 Full against round
9's 14 and 24 — which is the second heap, as predicted.

**One missed. `DRAW_FLOOR` leaves 270 rows drawing nothing**, against a
pre-registered zero: 255 rows in 249 Strict game-editions (9.0%) and 15 in 15
Full ones (1.3%). The rule terminates before it can reach them. A row under the
floor is absorbed into its club's heap; when a club's *whole* sub-threshold
remainder is under the floor, that heap is the smallest row the club has and
there is nothing larger of its own to absorb it. Of twelve re-derived row by
row, eight are a heap of one — `1 small event (SEA)` at +0.018 pt, a row that
has traded the event's own words for a count and bought no visibility — and four
are heaps that cancel, like `5 small events (KC)` at −0.039 pt. Splitting the
remainder by club is what exposes most of them: the mixed heap was hiding them
by cancelling one club's slivers against the other's.

**And the floor cannot be the whole answer even where it fires.**
`2017_11_JAX_CLE` Strict draws `CLE extra point · Gonzalez, made (92% kick)` at
−0.07 pt — above the floor, kept by the lone-event rule, and still two pixels
wide against a fifteen-point axis. Whether a bar can be seen is a share of the
**axis span**, not an absolute number of points. A relative floor would say what
this one is trying to say; it is a different rule and it is the maintainer's, not this
round's.

### 11c. What was rendered

Nine share sets after each of Parts A–E, all replaying at `0.00e+00`; then every
game in both editions — 15,600 PNGs, 34.9 minutes, worst replay gap
`0.00e+00`. Five of the forty tail PNGs the handoff names were opened, a
deviation recorded in the round's log and in document 63 §5.

### 11d. Register — closed this round

| Item | Where |
|---|---|
| Two pairs of Strict games share a filename (doc 63 §3a) | 11a — the game id leads |
| The title runs under the credit stamp on 84–89% of figures (doc 63 §3) | 11a — bottom-right, with the strip reserved |
| The zero rule prints through a waterfall corner label (doc 63 §3, §10e) | 11a — `shield=True` |
| The rotated arrow sentence overlaps a corner label (doc 63 §3) | 11a — lowered under the band |
| The biggest anonymous heap outranks all but two named events (doc 63 §3) | 11a — one heap per club, with its mark |
| Fold rows worth ±0.02–0.03 pt draw nothing (doc 63 §3) | 11a — `DRAW_FLOOR`, as far as it reaches |
| Stacking is the common case, not the exception (doc 63 §2) | 11a — two rows always |

**Open, carried forward.** Everything in §8c's "Open" table is unchanged, plus
the entries in document 63 §3 this round did not touch — blowout compression,
the degenerate game's empty arrow band, the zero-anchored `wins by` key, the
`2025_09_MIN_DET` contrast, the three byte-identical rows on `2025_19_GB_CHI`,
and the two mark-on-tint clashes — and the three new entries in 63 §5.

### 11e. Raised this round, not acted on

The 270 sub-floor rows and the −0.07 pt lone event are one question with three
answers, and choosing between them is a rule change rather than a layout fix:
make the floor **relative** to the axis span, **exempt a heap of one** from the
override so it keeps the event's words, or **accept them** as a sub-pixel row at
the bottom of an order nobody reads that far down. Document 63 §5 carries all
three with their costs.

## 12. Round 11 — a floor the eye can see, and the whole tail read

`fix/figure-round-11`, from `handoff-2026-08-29-figures-r11.md`. Two things: the
draw floor became a share of the axis, and the tail pick list was opened in full
rather than sampled.

### 12a. The rule

`DRAW_FLOOR = 0.05` pt is removed; `DRAW_FLOOR_SHARE = 0.005` replaces it. §11a
carries the full amendment. In short: a row has to be worth half a percent of
the waterfall's axis span to be worth drawing, the span is
`max(0, actual, deserved) − min(0, actual, deserved)` taken from the verdict by
the new `waterfall_span` and fixed before any fold, a club heap holding exactly
one event keeps that event's own words whatever it is worth, and a club heap of
two or more that still cancels under the floor is kept and counted.

> **Amendment, 2026-08-29 (round 12, §13b).** The base is no longer the span. It
> is the **drawn frame** — the axis `set_xlim` ends at, which reaches every
> running total and adds a pad at both ends plus the arrow's rail lane, and is
> between 1.58× and 6× the span. `DRAW_FLOOR_SHARE` keeps its name and its
> value; `group_rows`' keyword is `frame` rather than `span`; `waterfall_frame`
> and `fold_to_frame` are the two new functions, and `waterfall_span` survives
> only as a reported statistic. Everything else in §11a stands unchanged —
> rules 2 and 3 in particular.

Nine tests, written first and watched fail. 864 → **873**, ruff clean.

### 12b. What the corpus said

15,600 PNGs — 4 × 3,900, matching exactly — in 34.7 minutes on 12 workers, worst
replay gap `0.00e+00`. **Every pre-registered zero landed**: no row labelled `1
small event`, no under-floor row outside the two classes rules 2 and 3 allow, no
title/stamp overlap, no corner strike, no arrow-sentence overlap, no row still
labelled `events under`, no row without a club, and all 3,900 distribution
figures on two rule rows. Waterfall rows at the maximum are unchanged, 15 Strict
and 25 Full.

**One reported number came in above expectation and is reported, not loosened.**
The handoff expected rows drawing under the floor "well under 270"; the count is
**392** (375 Strict, 17 Full, in 378 game-editions). It is not the same
measurement as round 10's 270. The floor is now higher than 0.05 pt on every
game whose axis spans more than 10 points — its median is 0.05 pt and its
maximum 0.25 pt — so it catches rows the absolute floor called visible on a
blowout and never should have. And 204 of the 392 are lone events that **rule 2
deliberately keeps**, which round 10's rule would have folded away; the other
188 are the rule-3 residue. Neither class is a rule failing to terminate. The
honest reading is that 392 is the first true count of invisible rows, not a
regression from 270.

### 12c. The tail, read in full

194 PNGs across 97 game-editions, every one opened — see document 63 §7. Eight
of document 63 §3's defects are confirmed closed on the games that raised them,
six are confirmed still open, and six classes are new. The one worth a round of
its own is **N5**: the span the floor is a share of is not the axis the reader
sees. The frame adds a pad at both ends and a lane for the arrow rail, so it is
never less than 1.58× the span, and the floor is therefore never more than 0.32%
of the drawn width — much less on a narrow game, where those pads have absolute
minimums. `2023_02_WAS_DEN`'s span is 2 pt and its frame is about 12.

The others: **N6**, a 2017 Oakland game named LV throughout and two different
Raiders marks on one figure; **N1**, the luck arrow's floor is still absolute, so
a 1.1-pt arrow on a 55-pt axis is a bare arrowhead; **N2**, an anchor of zero
draws no bar at all, on 19 of the 97; **N3** and **N4**, two dark clubs or two
shades of one hue collapsing the side tints, the legend and the key.

## 13. Round 12 — the club a game was played by, and two floors on the axis

`fix/figure-round-12`, from `handoff-2026-08-29-figures-r12.md`. The last figure
round before the write-up. Four of document 63 §7d's six new classes are closed
and two are parked; the corpus is the verification.

### 13a. N6 — the club a game was played by

The summary and ledger artifacts carry the **modern** abbreviation on every
game, so `2017_16_OAK_PHI` arrived as `LV` and said so on its headline, its
corner label, all seven row labels, its legend and its `wins by` key.

`teams.era_code(abbr, season)` reads a three-row `RELOCATIONS` table
**forwards** from the season — `LV` → `OAK` through 2019, `LAC` → `SD` through
2016, `LA` → `STL` through 2015 — so a 2021 Las Vegas game stays Las Vegas by
construction rather than by a special case. The season is threaded through
`team_name`, `team_colors`, `team_logo`, `resolve_pair` and `pair_colors`, and
applied at the two places club codes enter a figure: `plots.verdict_from_row`
and `render.prepare_rows`. No number moves; the season comes from the game id
the row already carries.

Two things the round measured rather than assumed. nflverse's
`team_logo_espn` column gives **OAK, SD and STL the same URL as their
successors**, so the drawn mark is unchanged in this window — the season now
decides the cache key (`data/logos/OAK.png`) and nothing else. And the "two
Raiders marks on one figure" half of N6 **does not reproduce**: one `logos` map
feeds the corner, the anchors and the row column, the round-11 render's two
marks are the same shield at two sizes, and the invariant is now locked by a
test and measured at **0** corpus-wide.

### 13b. N5 — the draw floor is a share of the drawn frame

§12a's amendment. The base moves from the span
`max(0, actual, deserved) − min(0, actual, deserved)` to the **frame the reader
sees**: the axis `set_xlim` ends at, which reaches every running total and adds
a pad at both ends plus the arrow's rail lane. Measured over the corpus the
frame is a median **1.66×** the span, never less than **1.58×**, and on the
narrowest game **69.6×**. `DRAW_FLOOR_SHARE` keeps its name and its value;
`group_rows`' keyword is `frame`; `waterfall_frame` and `frame_width` are new
and `waterfall_span` survives as a reported statistic.

**The fold runs to a fixed point, and that is a deviation from the settled
two-pass rule.** The rule was specified as: measure the frame from the unfolded
bars, then fold to it, on the premise that folding preserves the running totals.
It does not — two club heaps that cancel step out and back through an excursion
wider than the tail they replaced — and on 208 sampled game-editions one pass
left the floor measured on an axis more than 10% from the drawn one on **27** of
them, which would have missed §13d's pre-registered band by construction. So
`fold_to_frame` repeats until the frame stops moving: one pass on 279 of 450
sampled, two on 169, three on 2, never more. At the fixed point the floor and
the drawn axis are one number, and the corpus measures floor / axis at
**0.5000%** on all 3,900, median and maximum alike.

### 13c. N1 and N2 — the arrow's floor, and an anchor of zero

`ARROW_FLOOR = 1.0` pt is removed; `ARROW_FLOOR_SHARE = 0.03` is read against
the axis the caller has already pinned. **3,080** game-editions clear the old
absolute floor and **1,866** clear the new one: 1,256 lose a span that was a
stray glyph on a wide axis, 42 gain one that is a measurable distance on a
narrow one. The sentence is drawn either way and keeps the number.

An anchor the figure calls `even` draws **no bar by design** — there is no
distance to show, and a minimum-width bar would state a margin the game did not
have — and gains a 2-pt tick of the anchor colour at x = 0, the height of the
bar it stands in for. `ANCHOR_EVEN_EPS = 0.05` is read by both `anchor_label`
and `_draw_anchor`, so the words and the mark cannot disagree. **That constant
is Part E's finding, not Part D's rule as written**: the tick first fired on
`margin == 0.0`, which is 13 of the corpus's 29 `even` anchors, and the other 16
drew a two-pixel bar under a label saying there was nothing to see. The corpus
now measures **29 `even` anchors, 29 ticks, 0 without one**.

N3 (two dark clubs, identical side tints) and N4 (same-hue matchups) are parked
in document 63 §8 with no code.

### 13d. What the corpus said

15,600 PNGs — 4 × 3,900, matching exactly — in **73.8 minutes on 12 workers**,
worst replay gap **`0.00e+00`**. 873 → **919** tests, ruff clean.

**Sixteen of seventeen pre-registered checks landed.** 15,600 files on disk; 0
texts naming a club by a code wrong for its season, over the 79 game-editions
where a club had not yet moved, and 0 such codes in the verdict or the prepared
rows; 0 figures with two marks for one club and 0 marks from outside the logo
map; 0 waterfalls whose floor over the drawn axis falls outside 0.5% ± 0.05%; 0
drawn arrow spans under 3% of their axis; 0 `even` anchors without a tick; 0
title/stamp overlaps, 0 corner strikes, 0 arrow-sentence overlaps, 0 rows
labelled `1 small event` or `events under`, 0 anonymous rows, 3,900 of 3,900
distribution figures on two rule rows.

**One pre-registered zero missed, reported not loosened.** `rows_under_floor_other`
is **1**, not 0: `2022_02_IND_JAX` full's `8 smaller JAX drops` at −0.214 pt.
The frame–fold map has a **2-cycle** on that game. Fold at a 0.2138-pt floor and
20 rows survive, whose frame measures 44.03 pt; fold at that frame's 0.2201-pt
floor and 19 rows survive, whose frame measures 42.75 pt; and back. `fold_to_frame`
caps at eight passes and returns the last fold with **its own** frame, which is
why the floor-over-axis check still holds exactly — but that fold was made at a
floor a thousandth of a point below the one the returned frame implies, and this
row falls in the gap.

There is **no assignment that satisfies both checks on this game.** Taking the
larger frame's fold instead — 19 rows, no row under its floor — would leave the
floor at 0.2201 pt on an axis of 42.75, which is **0.515%**, outside the ±0.05
pp band. A one-line conservative tie-break (on non-convergence, fold at the
largest frame seen and report `frame_width` of that fold) would clear both
counts at the cost of folding slightly more than the drawn axis requires. It is
**not applied**: changing a rule after seeing its check fail is the move
constraint 5 exists to prevent, and the threshold is the maintainer's.

Severity is cosmetic either way. The row sorts last, carries its label and its
club's mark, and is 0.5% of the axis wide.

**Reported, not pre-registered to a number.** Rows drawing under the floor total
**717** (686 Strict, 31 Full) in 648 game-editions, against round 11's 392 — 244
rule-2 lone events, 472 rule-3 cancelling heaps, and the one above. The floor
rose and the two exemptions did not, so the count could only rise; it is the
residue those rules buy. Draw floor median 0.087 pt, maximum 0.395 pt. Waterfall
rows at the maximum unchanged, 15 Strict and 25 Full. 29 `even` anchors, 1,866
arrow spans drawn.

### 13e. The tail, read again

33 PNGs from the final corpus, all opened — see document 63 §9. Four of §7d's
classes are closed on the games that raised them, one class is withdrawn as a
misread, two are parked, and three cosmetic observations are new: the 2-cycle
game above, `2022_07_KC_SF` full's stub span at 3.01% of its axis, and the
zero-anchor tick reading as a segment of the dashed zero rule on one game.

## 14. The brand mark in the corner stamp (2026-08-30)

the maintainer supplied the NFL Simulator badge and asked for it on every figure, so
`finalize` now stamps `BRAND_LOGO` — a 47 KB RGBA PNG packaged at
`src/nfl_simulator/assets/logo.png` — unless it is handed `logo_path=False`.
The mark sits to the **left** of the credit line at 1.6 line heights, centred on
the credit's ink, and the text does not move for it: `apply_watermark` asks
`stamp_box` for the logo-free box and draws the text there either way, which
`test_the_mark_does_not_move_the_credit_line` locks by comparing every pixel
column from the credit's left edge rightward.
Left rather than the old stacked-above slot because §11's whole point was that
the stamp's rows are the ones no artist is laid out in; a mark above the credit
reaches back up into rows a title can occupy. `reserve_stamp_strip` now takes
the mark's columns into account, so ink reaching only the badge buys the strip
too. 929 tests pass; nine game-editions from document 63 §7d re-rendered and
three opened.

## 15. The stamp goes back to the top-right, and the waterfall's footer (2026-08-30)

Four decisions the maintainer made on 2026-08-30, after reading round 2's figures. None
of them is re-litigated here; this section records what was decided, what it
cost, and the one place the instruction and the code disagreed.

### 15a. The corner, reversed

**Decision.** The credit stamp goes back to the **top-right** — the corner the
sibling MLB simulator uses (`Simulator/visualizations.py`, `_apply_watermark`
at `position='top-right'`) — and the brand mark grows from **1.6 credit lines
to 2.5**. At 1.6 the badge read as a bullet in front of the credit rather than
as a mark, which is the whole reason it is on the image.

**What §11 measured, and why it no longer decides the corner.** Document 63
counted the title running under the top-right stamp on 2,325 of 2,759 Strict
distribution figures and 1,016 of 1,139 Full ones. That finding stands. What
round 10 concluded from it — that the top-right is a bad corner — does not:
the collision was never a fact about the corner, it was a fact about ordering.
The stamp is painted on the saved pixels after layout, so a title laid out
before the stamp exists cannot see it and move.

`reserve_stamp_strip` already knew how to answer that at the bottom edge, by
growing the canvas when the figure's ink reached the stamp's box. It is now
mirrored: it finds the **highest** inked row in the stamp's columns and, if
that row is inside the box plus its clearance, pastes the image lower on a
taller cream canvas. The room is made instead of the corner conceded.
`test_a_wide_title_does_not_run_under_the_stamp` is the check, measured over
the whole block — the mark's reserved rows as well as the credit's.

**The mark stacks above the credit, right-aligned with it.** §14 put it to the
left, and that only worked because the block was anchored to the bottom edge,
where a mark 1.6 lines tall could overhang the text's rows into empty pixels.
Anchored to the top edge a mark 2.5 lines tall has nowhere to overhang: at a
1.2% inset it would run off the canvas. So the arrangement becomes the MLB
simulator's stack, right-aligned rather than centred on the block so the pixel
nearest the corner is the mark itself.

**The credit still does not move for the mark.** `stamp_box` computes the mark's
height from the credit's own line height — a fact about the text, not about any
file — and hangs the credit a fixed mark-plus-gap below the top inset **whether
or not a mark is drawn**. `logo_path` decides only whether anything is painted
in the reserved rows above, and grows the returned box upward to cover them for
the strip reservation and the corpus read. §14's guarantee therefore survives
the move intact; only its test's geometry changes, from a column comparison to
a row one, because the badge shares the credit's columns now and has its own
rows.

### 15b. The waterfall's footer

Three changes, all to `plot_luck_ledger`'s footer block:

1. **It states `ha="left"` rather than inheriting it.** The round-3 handoff
   described this footer as centred. It was not: `Annotation` defaults to
   `ha="left"` and the block is anchored at `xy=(0, 0)` in axes fractions, so
   it already started at the axes' left edge. Nothing moved. The alignment is
   now spelled out so it is a decision on the record instead of a default
   nobody chose, and `test_the_waterfall_footer_is_left_aligned_on_the_axes_left_edge`
   holds it there.

2. **It gains a sentence about the small-events row.** Round 10 gave the
   remainder a club and a count — `46 small events (LAC)` — which says whose
   afternoon it was and still not what is inside it. `SMALL_EVENTS_FOOTER`
   says it once, in the footer rather than beside the bar, because it is true
   of both heaps and of every game that has one.
   `test_the_footer_explains_the_phrase_the_heap_row_actually_uses` pins the
   footer's wording to `_heap_label`'s, so a later rename cannot leave a gloss
   explaining a phrase no reader can find on the chart.

3. **It drops the overtime toss line.**

### 15c. Where the instruction and the code disagreed — the toss line

**The handoff's goal:** "the waterfall footer is left-aligned, has no overtime
line, explains small events". **The handoff's mechanism:** test that
`footer_lines` returns no overtime line on an overtime verdict.

Those are not the same change. `footer_lines` feeds **four** figures, not the
waterfall alone — `plot_bootstrap_distribution`, `plot_luck_ledger`,
`plot_luck_ledger_card` and `plot_game_card` — and the comment above
`OVERTIME_FOOTER` records the rule it exists to keep: document 16 measured the
overtime toss and refused it, so every figure that draws a ledger has to say
the ledger is one event short on purpose. Emptying `footer_lines` would strip
that caveat from the three **share images**, which is a disclosure change to
figures that travel on their own, not a layout change to the write-up's
waterfall.

**Enacted narrowly, and the widening is the maintainer's to order.** `footer_lines` takes
`overtime: bool = True`; the waterfall passes `overtime=False` and is the only
caller that does. The goal's sentence is met exactly. `test_render.py`'s
`test_every_share_figure_says_the_toss_is_reported_not_neutralized` still passes
unchanged on all three share figures, which is the other half of the record.

The reasoning for the narrow reading, stated so it can be overruled cheaply: a
waterfall shows the ledger's rows themselves, so a reader looking at every
priced event can see the toss is not among them; a share card shows a number
and no rows. If the maintainer wants the line gone from the share images too, it is one
argument's default flipped.

### 15d. The pick row's noun

`("dropped_pick", False)` reads **`interception (pick-able throw)`**. Every row
in the component is a throw FTN charted as interception-worthy; a row saying
only "interception" reads as though the ledger re-prices every interception in
the game, which document 05 §3 explicitly refuses to do. The parenthesis is the
population and the percentage after it is still the probability, so a shipped
row now reads `DEN interception (pick-able throw) · thrown by Mariota (52%
catch)`. No label-width test failed; the row was read on
`2025_13_DEN_WAS` full at 2,041 px and fits its column.

### 15e. Verification

935 tests pass (929 at `4491f2f`, plus six), ruff clean. Nine game-editions
re-rendered from document 63's list — `2018_05_GB_DET`, `2021_14_LV_KC`,
`2016_14_NYJ_SF` and `2017_11_JAX_CLE` Strict; `2025_17_DET_MIN`,
`2025_13_DEN_WAS`, `2022_13_WAS_NYG`, `2024_19_LAC_HOU` and `2023_01_PHI_NE`
Full — and three opened: the DEN–WAS waterfall (stamp clear of the title, new
footer, new noun), the LAC–HOU distribution (the canvas grown at the top, which
is `reserve_stamp_strip` doing the job §15a describes) and the PHI–NE card
(square on disk, stamp clear).

## §16 — write-up round 3 fix-ups (2026-08-30, Fable)

The mark scales with the image (`STAMP_LOGO_WIDTH_SHARE = 0.045`, credit-line
ratio as the floor); the waterfall footer anchors at the figure's left edge;
`reserve_stamp_strip` keeps a square image square by growing width with
height. All three in one commit with their tests.
