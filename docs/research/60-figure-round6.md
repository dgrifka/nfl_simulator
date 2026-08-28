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
