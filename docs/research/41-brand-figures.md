# 41 — Brand-matched per-game figures

**Date:** 2026-08-26
**Branch:** `feat/product-bootstrap-plot`
**Scripts:** `research/58_brand_figures.py`
**Code:** `src/nfl_simulator/style.py`, `teams.py`, `render.py`, changes to `plots.py`
**Nothing here is fitted.** Every number on every figure is read from
`dtw_games_v13.parquet`, `dtw_ledger_v13.parquet`, `26_overtime_games.parquet`
and `model_metadata_v13.json`. The only quantity recomputed is each game's
bootstrap draws, which the shipped summary does not keep; all five replay to
their committed summaries at **0.00e+00**.

---

## 1. What was built

Four pieces, one commit each.

**`style.py` — the house style.** The baseball simulator's `Simulator/style.py`
ported, not imported: `PALETTE`, `apply_base_style`, `rc_style`, `title_axes`,
`draw_title_block`, `finalize`, `lighten`, plus the PIL watermark that lived in
that repo's `visualizations.py`. The surface is warm cream `#FCFAF6`. The
watermark is stamped onto the **saved pixels** rather than drawn into the figure,
because `bbox_inches="tight"` crops by an amount that depends on how long the
longest tick label happened to be, and a corner in pixels is the same corner on
every PNG. `finalize` is the only way to write a figure in this product and it
always stamps `Data: nflverse | @[TBD]`.

**`teams.py` — colours and marks.** nflverse's 36-row team table cached to
`data/teams.parquet`; club logos cached to `data/logos/{abbr}.png`. Both
gitignored. Relocation aliases (OAK, SD, STL, LA) are first-class — 2016–2019
games are played by them, and a lookup that knew only the current 32 would draw
those games in the grey fallback with nothing to say why.

**"realized" → "actual"** across the ledger, the simulator's `LuckEvent`, the
figures and their tests (see §5).

**`render.py` + three figures.** `render_game(game_id, out_dir) -> list[Path]`
writes `dtw`, `luck_ledger` and `card` for any game, in the baseball filename
pattern.

---

## 2. The header grammar

Both wide figures wear the same block, drawn in the band above the axes:

```
Luck Ledger                                              [ clear flip ]
────────────────────────────────────────────────────────────────────────
Actual: GB 23 - DET 31   (10/07/2018)    DTW: GB 95% • DET 5%
Start at the actual margin. Each bar is one luck event re-priced at its …
```

- **Heading**, bold, in the heading family.
- **Divider rule**, the width of the plot, so the header reads as one block with
  the figure under it rather than as a caption floating above it.
- **Subtitle**, muted: the scoreboard, the date, both shares. A verdict built
  from the summary alone has no score and states the margin instead
  (`GameVerdict.score_line`); it never prints `None - None`.
- **Verdict pill**, right-aligned on the *subtitle* row, filled
  `PALETTE['bad']` for "clear flip", `text_muted` for "too close to call",
  `good` for "scoreboard holds". Never on the heading row — see §4a.
- **Caption** (waterfall only), wrapped to the room the pill leaves — §4b.

The block is drawn with offsets in **points off the top spine**, anchored to the
host axes, rather than in a separate strip axes as the baseball style does.
`attach_overtime_sidebar` widens the figure and rescales the host axes to hold
the plot at the inches it was drawn at; a second axes would stretch across the
growth and the header would drift off its own figure. Deviation from the
handoff's "from `draw_title_block`", and the reason. `draw_title_block` is still
used — the share card uses it.

---

## 3. The label rules

### 3a. Plain words

`plots.plain_label` says one ledger row the way it would be said out loud.

| Ledger row | Before | After |
|---|---|---|
| `field_goal` / `40-44 yd` / GB / branch 1 | `40-44 yd field goal — GB` | `GB 42-yd field goal, made` |
| `fumble` / `punt/live` / GB / branch 0 | `punt/live fumble — GB` | `GB fumble on a punt, recovered by DET` |
| `fumble` / `run/aborted` / DET / branch 1 | `run/aborted fumble — DET` | `DET fumble on an aborted run, retained` |
| `extra_point` / GB / branch 0 | `extra point — GB` | `GB extra point, missed` |

Three rules inside it:

- **The exact distance or nothing.** The ledger stores a five-yard class. The
  real yardage is joined from the play-by-play by `play_id`; where the play is
  not found the label falls back to the class (`GB 40-44 yd field goal`). It
  never prints the class midpoint as if it were the distance.
- **Retained is asymmetric.** A fumble the fumbling team recovered reads
  `retained`; "DET fumble, recovered by DET" says the same thing twice and reads
  as a mistake. A fumble it lost names who got it, because that is the fact a
  reader wants and the ledger does not record it — the opponent is supplied by
  `render.prepare_rows` from the game itself.
- **No branch, no outcome clause.** A row without its branch gets
  `DET fumble on a punt` and stops. Nothing is guessed.

### 3b. Recovering the branch

The v1.3 ledger artifact predates the `actual` column. `ledger.with_actual`
recovers it exactly from the identity the module is built on,

    luck_epa = (actual - expected) * swing

which determines `actual` given the other three, and `swing` is an EPA branch
value that is never zero. Re-simulating 2,761 games to rename a column would
have replaced a committed artifact rather than read it. A consequence worth
having: the label a figure prints beside a bar is derived from the same three
numbers as the bar, so the two cannot disagree.

---

## 4. The figure-rule review

*"All graphs must be clean, self-explanatory, helpful, and tell a story."* All
fifteen PNGs were rendered, opened and read. Seven defects were found and fixed;
each is recorded with what it was measured on.

### 4a. The verdict pill printed through the watermark and the heading

`LV_KC_9-48--0-100_dtw.png`. The pill was right-aligned on the heading row, which
is where `finalize` stamps the data credit — and the heading
"Deserve-to-Win — 160,000 luck re-flips" is wide enough that the pill also took
the last word of it. **Fix:** the pill moved to the subtitle row. Regression
tests: the pill is measured against the heading and the subtitle at the real
160,000-draw heading width in all three buckets, and a fourth test opens the
saved PNG and asserts nothing coloured is in the watermark's corner.

### 4b. The pill came down onto the "how to read" caption

`DET_MIN_10-23--45-55_luck_ledger.png`. **Fix:** the caption is wrapped to the
plot's width *minus the measured pill*, which puts it on two lines on a narrow
figure. Measured rather than assumed, because both the pill's text and the
figure's width vary per game.

### 4c. The away-team alpha destroyed the team's identity

`GB_DET_23-31--95-5_dtw.png`. The handoff specified alpha 0.78 home / 0.55 away,
which is the baseball run-distribution chart's rule. It is right there and wrong
here: that chart's two histograms **overlap**, and alpha is what stops one hiding
the other. These two never overlap — every bar is wholly one side's, since the
bins are aligned so none straddles zero. Drawn at 0.55, Green Bay's `#203731`
read as grey and Denver's `#002244` as slate. **Fix:** full-strength team colour
on both sides, no alpha. Deviation from the handoff, with the reason.

### 4d. A legend for a colour that was not on the figure

`LV_KC_9-48--0-100_dtw.png` is degenerate: KC wins every re-flip, so there is no
LV bar, and the legend keyed one anyway. **Fix:** the distribution now legends
only the sides that have bars, which is the rule the waterfall already had.

### 4e. The Jets' mark printed through the first bar

`NYJ_SF_23-17--36-64_luck_ledger.png`. A fixed `OffsetImage` zoom sizes by the
club's file, and those are not one shape: ESPN's Jets logo is a **4,096 px
square** holding a wide, short wordmark, where most clubs ship a 500 px shield.
**Fix, in two parts.** `teams.team_logo` now crops the knocked-out image to its
visible mark, so a wordmark is a wordmark-shaped array rather than a mostly-empty
square; and `plots.logo_zoom` fits it inside a **box** in inches, bounding both
dimensions, so every club's mark carries the same visual weight whatever its
aspect ratio.

### 4f. The Raiders' bars were the same colour as the totals

`LV_KC_9-48--0-100_luck_ledger.png`. The handoff asked for the two end bars in
`PALETTE['text']` ink. Las Vegas's primary is `#000000`, and the figure had no
way to say which bar was a total.

A threshold was tried first and abandoned, because the measured distances to ink
`#1A1A1A` are CHI 0.087, HOU 0.124, GB 0.147, LV 0.177 — there is no cut that
separates the clubs that need moving from Green Bay, which must stay green.

**Fix:** change the anchor instead of the teams. `PALETTE['anchor']` is
`#5E5B55`, chosen by measurement: it is at least **0.281** in RGB from every one
of the 32 club primaries (nearest is Minnesota's `#4F2683`), comfortably past the
0.20 the clash rule calls a collision. No club can ever collide with it and no
team's colour has to move for a figure's convenience. Deviation from the
handoff's "ink", with the arithmetic.

### 4g. The share card's lower third was empty

`GB_DET_23-31--95-5_card.png`. The stack ended at 15% of the card's height.
**Fix:** re-spaced so the block runs from the heading to a footer at 5%.

### Passed without change

- Document 37 §7a's rule-label collision fix survives the restyle. The boxed
  callouts are wider than the bare labels were, and `2025_13_DEN_WAS` — the game
  the fix was written for — still lifts "Deserved -3.3" clear of "Actual -1".
- The overtime sidebar still attaches to both wide figures without shrinking the
  plot, and the interval caveat still clears it.
- Degenerate games (`2021_14_LV_KC`) keep the single-sentence treatment on all
  three figures.

---

## 5. "realized" → "actual"

`LedgerEntry.realized` is now `LedgerEntry.actual`, the ledger schema column with
it, and the same word in `simulator.py`'s `LuckEvent`, the rule and end-bar
labels in `plots.py`, drivers 55 and 56's terminal output, and the three test
modules that assert on them.

`placement.py` is deliberately left alone. It carries `realized_score` seven
times, the handoff's own constraint 3 says not to touch the placement meter, and
its `realized_score` is a different concept — a score against its null
distribution, not the ledger's branch. `grep -rn -i realized src tests` therefore
prints those seven lines and `test_placement.py`'s two, and nothing else.

Documents 37–40 each carry a one-line note; their prose is left as written.

---

## 6. Colour, measured against the dataviz checks

The `dataviz` skill's validator was run on all five games' pairs against the
cream surface. A brand palette is not a chosen palette, so the results are
reported rather than acted on — except where a figure could actually be
misread.

| Game | Pair | Contrast vs surface | Normal-vision ΔE | Worst CVD ΔE |
|---|---|---|---|---|
| 2018_05_GB_DET | `#203731` / `#0076B6` | PASS | 26.0 | 25.0 (deutan) |
| 2021_14_LV_KC | `#E31837` / `#000000` | PASS | 62.7 | 46.4 (protan) |
| 2025_17_DET_MIN | `#4F2683` / `#0076B6` | PASS | 21.3 | 16.2 (deutan) |
| **2016_14_NYJ_SF** | `#003F2D` / `#AA0000` | PASS | 28.0 | **5.2 (protan)** |
| 2025_13_DEN_WAS | `#5A1414` / `#002244` | PASS | 17.4 | 9.6 (protan) |

Two checks fail on every pair and are not actionable: the lightness band and the
chroma floor. NFL primaries are deliberately dark and several are near-neutral
(Green Bay's chroma is 0.031, Denver's 0.075). Those checks exist to keep a
*chosen* palette legible; the whole point of this round is that the palette is
the club's, not ours.

**One result is actionable and is recorded as an open defect.** `2016_14_NYJ_SF`
puts the Jets' green against the 49ers' red at ΔE **5.2** for protanopia —
below even the 6–8 floor the skill allows with secondary encoding. The clash
rule is plain RGB distance, which does not model colour vision: the pair sits at
0.73 in RGB and sails through. Every figure does carry the same fact without
colour — the distribution encodes the winner by **position** about the zero rule,
the waterfall labels every bar in words and signs every value, and the card
prints both abbreviations at the ends of its bar — so no figure is unreadable.
It is still the wrong rule, and a CVD-aware clash rule is the first entry in §8.

**One deliberate divergence from the skill.** It says a single series needs no
legend box, and a degenerate game's distribution has one series. The legend is
kept: the fill's meaning ("KC wins") is the fact the figure exists to state, and
dropping it leaves the only colour on the figure unexplained.

---

## 7. The five games

Chosen, not typical, and each for a stated reason.

| Game | Verdict | Files |
|---|---|---|
| `2018_05_GB_DET` — the worked clear flip (docs 33, 37, 38) | GB 95% / DET 5%, clear flip | `GB_DET_23-31--95-5_{dtw,luck_ledger,card}.png` |
| `2021_14_LV_KC` — degenerate | KC 100% / LV 0%, scoreboard holds | `LV_KC_9-48--0-100_{dtw,luck_ledger,card}.png` |
| `2025_17_DET_MIN` — inside the band | MIN 55% / DET 45%, too close to call | `DET_MIN_10-23--45-55_{dtw,luck_ledger,card}.png` |
| `2016_14_NYJ_SF` — overtime, largest per-game toss move | SF 64% / NYJ 36%, clear flip | `NYJ_SF_23-17--36-64_{dtw,luck_ledger,card}.png` |
| `2025_13_DEN_WAS` — overtime under the 2025 rulebook | DEN 86% / WAS 14%, scoreboard holds | `DEN_WAS_27-26--86-14_{dtw,luck_ledger,card}.png` |

The filename is the baseball simulator's pattern verbatim:
`{away}_{home}_{away_score}-{home_score}--{away_dtw}-{home_dtw}_{suffix}.png`.
The two shares are rounded **once, together** — the home share is rounded and the
away share is 100 minus it — so the pair in the filename always sums to 100 and
always agrees with the headline on the figure.

---

## 8. Known-defect register

| Defect | Evidence | Status |
|---|---|---|
| The clash rule is plain RGB distance and does not model colour vision; `2016_14_NYJ_SF` is at protan ΔE 5.2 and passes it | §6 | **Open, disclosed.** Every figure carries the same fact by position or by label, so none is unreadable. A CVD-aware rule is the first next avenue |
| The brand handle is `@[TBD]` on all fifteen images | `style.BRAND_HANDLE` | **Deliberate.** the maintainer has not named the account; an invented handle would point readers at somebody else |
| No project logo — the watermark is the credit line alone | `style.apply_watermark(logo_path=None)` | **Deliberate, slot left open.** Adding one later does not move the text |
| `plot_game_card` has no automatic overflow handling; a long verdict or a long deserved line is placed, not fitted | `plots.plot_game_card` | **Accepted.** Both strings come from a fixed vocabulary (§6 of doc 33) and a fixed format; a game cannot produce a longer one |
| The waterfall's smallest value labels can touch the zero rule (`-0.01` on `2025_17_DET_MIN`) | §4, figure read | **Accepted.** The number stays legible; moving it would cost the alignment every other row has |
| Kick distances are joined from the cached play-by-play, so a machine without it renders the class instead of the yardage | `render.kick_distances` | **Accepted and degraded on purpose.** The class is true; the render does not stop |
| The five example games are chosen, and their colour pairs are five of 496 possible | §6, §7 | **Accepted.** No claim is made that the palette checks pass league-wide; §8's first row is the way to find out |

---

## 9. Verification

```
uv run pytest -q                                   396 passed (318 + 78 new, written first)
uv run ruff check . && ruff format --check .       clean
uv run python research/58_brand_figures.py         15 PNGs, all five replays 0.00e+00
uv run python research/54_bootstrap_figures.py     replays 0.00e+00 on all three
uv run python research/55_ledger_waterfall.py      2,761 games, max |residual| 7.11e-15
uv run python -c "…load_team_table().height"       36
git status --short data/ research/outputs/         (empty — both gitignored)
```

All fifteen figures were opened and read before this document was written.
