# 51 — Figure round 4: the maintainer's review notes, implemented

Round 4 of the product layer's figure work. Nothing here is fitted and no
published number moves: every DTW%, deserved margin, ledger row and interval is
the same number it was on `main` at `a0b6ec6`, and the five example games still
replay to their committed summaries at **0.00e+00**. What changed is what a
reader is asked to do — round 4's brief was that no figure should leave
arithmetic for the reader to finish.

Branch `fix/figure-round-4`, four commits, one per part, plus this record.
Handoff: `handoff-2026-08-27-figures-r4.md`. Log:
`log-2026-08-27-figures-r4.md`. Results: `results-2026-08-27-exp5.md`.

---

## A. The card's footer is the interval and nothing else

**Commit** `e891b18` — *fix(card): drop the coverage stamp from the share footer*

**Decision.** The share card's last line read
`89% interval on GB's share: 92–96% (measured coverage 91.5%).` It now reads
`89% interval on GB's share: 92–96%.`

**Why.** The card is the figure for somebody who will not open the other three.
Document 10's measured coverage is a methodological aside they cannot act on,
and beside a share it reads as a second, competing percentage — two numbers
about the same quantity with no way to tell which is the answer.

**What a fan sees differently.** One percentage where there were two.

**What did not move.** The coverage figure is not dropped from the product:
`GameVerdict.interval_note` still carries the full document-10 sentence on the
distribution, which is what `dtw_article` draws — the figure a reader reaches
only after asking for the methodology. The degenerate-game footer ("Every
re-flip lands the same way…") and the overtime line are untouched, and the
card's layout above the footer is frozen as it was.

---

## B. The share distribution is two teams' deserved points

**Commit** `43883c3` — *feat(dtw): team deserved-points histograms replace the
margin plot on the share image*

**Decision.** The `dtw` share image stops asking "by how much" and asks "what
would the scoreboard have said". Two overlapping histograms of each team's
deserved points on a points axis with no negative side; both actual scores as
dashed rules in their own clubs' colours, boxed and named; the most likely
scoreline stated in words as a second subtitle line. The margin histogram (V4,
document 42 §1) is not retired — it is what `dtw_article` draws, where the
overtime panel already lives. `SUFFIXES` is unchanged: same four filenames, one
of them now a different figure.

**The arithmetic, and why it cannot drift.** `simulator.team_point_draws` splits
the bootstrap by the team each luck event is charged to: a team's deserved
points are its actual points minus the luck booked on its own plays. The split
is of **one** replay, not a second one — `bootstrap_margins` and the split now
share `_replayed_adjustment`, so `home − away` is the published margin
distribution draw for draw rather than to a tolerance. Measured on the five
example games, the worst disagreement is **1.07e-14 points**, which is
floating-point noise and nothing else. `SimulationResult` gained two optional
draw arrays defaulted to `None`, so nothing that already read the dataclass
changed shape, and `simulate_game` now refuses a scoreboard that does not
subtract to the game's own margin. The figure refuses to draw a pair that does
not reconcile, exactly as the waterfall refuses a ledger that does not.

**Two deviations from the round's sketch, both forced.**

1. **The fills are 0.6 alpha as asked, and each histogram is also traced at full
   strength in its club's colour.** Green Bay's `#203731` at 0.6 alpha over the
   cream surface renders as grey — the trap `plot_bootstrap_distribution`
   already records in a comment, and the first render of this figure reproduced
   it exactly: a grey fill under a green mark in the legend is identity lost.
   The silhouette carries the colour; the dilution is confined to the interior,
   where its only job is to let the team behind show through.
2. **No luck arrow.** The arrow measures a distance along a margin axis, and
   this axis is a scoreboard. It stays on the article figure.

**What a fan sees differently.** A scoreline instead of a margin. On
`2018_05_GB_DET`: `Most likely: GB 44 – DET 35` against an actual `GB 23 – DET
31`, with both dashed actual rules sitting well left of Green Bay's fill. The
five games' most-likely lines are GB 44–DET 35, LV 17–KC 44, DET 17–MIN 20,
NYJ 23–SF 26, DEN 23–WAS 23.

**One thing to know about the numbers.** Per-team deserved points can run well
above a team's actual score on a luck-heavy game — Green Bay's four missed field
goals put its deserved points near 44 against an actual 23. That is the ledger's
own arithmetic stated per team rather than as a margin, and it is the first time
the product has said it out loud.

---

## C. A kick shows what it was expected to do

**Commit** `5dd2984` — *fix(ledger): kicks show their make probability and
kicker*

**Decision.** A field-goal or extra-point row's What-happened cell now reads
`missed (88% kick)` / `made (86% kick)`, with the percentage the row's own
`expected` — the shrunk make probability the luck was priced at. Kick rows also
name the kicker: `41-yd field goal · Crosby`. Fumble rows are untouched, and a
row without an `expected` or a kicker keeps the words it had.

**Why.** the maintainer's note: a 41-yard miss costs −3.2 and a 42-yard miss −3.1, and
nothing on the card said why. The two now read as two kicks of different
difficulty rather than as a rounding difference.

**What a fan sees differently.** On `2018_05_GB_DET`, Green Bay's three biggest
rows are 41-yd at 88%, 42-yd at 86% and 38-yd at 91% — and all three are the
same kicker, which the column now says.

**Plumbing.** `kicker_player_name` joins `research/44_read_side_fix.SIM_COLUMNS`
(that file is otherwise untouched) and `render.kicker_names` reads the surname
off the play-by-play the way `kick_distances` reads the yardage: quietly, and
never invented. Nothing prices on the name — the pricing still uses
`kicker_player_id` — and the five games replay at 0.00e+00 with the column
loaded. The waterfall's row labels are the same `plain_label`, so they say it
too.

---

## D. The waterfall reads without arithmetic

**Commit** `f1d4489` — *fix(waterfall): rename, shade sides, anchor bars, tip
labels; close D-4*

**Decision, in four parts.**

* **Renamed Luck Waterfall.** Two figures called "Luck Ledger" was one name too
  few. The portrait share card keeps the older name; its layout is frozen.
* **The zero line explains itself.** Each half of the axis wears its own club's
  colour at 0.06 alpha, with a corner label and mark: `GB wins` left of zero,
  `DET wins` right of it. A reader no longer has to unpack the axis label's
  subtraction before any bar means anything.
* **The two totals are unmistakable.** 1.4× the height of every event bar, and
  their row labels name a team instead of a sign: `Actual: DET by 8`,
  `Deserved: GB by 8.3`, and `Deserved: even` on a dead-level game.
* **Every value label sits at its bar's tip** — the end away from the running
  total — so a bar that helped Detroit carries its number on Detroit's half of
  the axis. Round 3 put every number at the running total, which put a
  DET-helping bar's value on GB's side and made the reader check the colour to
  learn whose it was.

**The one deviation, with the arithmetic.** The round asked for the two totals
in `PALETTE["text"]` (ink). Ink is **0.177** from the Jets' and the Raiders'
`#000000` and **0.147** from Green Bay's `#203731`, both inside document 42 §3's
**0.20** clash floor. Drawn unconditionally, `NYJ_SF`'s totals came out the same
colour as its luck bars — which is the defect document 42 §6 closed in round 2
and pinned with a test. So the ink is stepped by that same clash rule rather
than by taste: ink where ink is legible against both clubs, the previous neutral
`#5E5B55` where it is not. On the round's five games that is ink on **1 of 5**
(`2025_17_DET_MIN`) and the neutral on the other four, all of which field a
club whose primary is black or near-black. The totals never rest on colour
alone either way — the height and the `Actual:` / `Deserved:` labels are two
further channels, and both survive a greyscale print. **Unconditional ink is a
one-line change if the maintainer prefers it; this is his call, not the round's.**

**D-4 closed.** A bar under half a point is narrower than its own label, and on
`2025_17_DET_MIN` the `-0.3` and `-0.01` labels printed through the dashed zero
rule, which reads as a number struck out. Closed in three parts: labels moved to
the tip; every label given the module's cream `_shielded` backing, so no rule
can strike one through; and a bar under `LEADER_FLOOR` (0.5 pt) has its label
pushed clear of the bar and joined back by a leader. Pinned by a regression test
on `2025_17_DET_MIN`'s own ten bars — every label carries a surface, the two
sub-half-point bars carry leaders, and no two labels overlap.

**D-6 opened and disclosed.** See the register in document 42 §6.

---

## E. What is on disk

Twenty-two PNGs in `research/outputs/` (gitignored; the script is the artifact):
four share images for each of the five games, plus a `_dtw_article.png` for each
of the two overtime games.

| Game | Files |
|---|---|
| `2018_05_GB_DET` | `GB_DET_23-31--95-5_{dtw,luck_ledger,card,waterfall}.png` |
| `2021_14_LV_KC` | `LV_KC_9-48--0-100_{dtw,luck_ledger,card,waterfall}.png` |
| `2025_17_DET_MIN` | `DET_MIN_10-23--45-55_{dtw,luck_ledger,card,waterfall}.png` |
| `2016_14_NYJ_SF` | `NYJ_SF_23-17--36-64_{dtw,luck_ledger,card,waterfall,dtw_article}.png` |
| `2025_13_DEN_WAS` | `DEN_WAS_27-26--86-14_{dtw,luck_ledger,card,waterfall,dtw_article}.png` |

Regenerate with `uv run python research/58_brand_figures.py`.

---

## F. Verification

* `uv run python -m pytest -q` — **570 passed**, from 502 at the branch point.
* `uv run ruff check .` and `ruff format --check .` — clean.
* `research/58_brand_figures.py` — five games, replay `max |Δ vs committed|`
  **0.00e+00** on every one, before and after each part.
* Per-team reconciliation against the published margin draws — worst
  **1.07e-14 points** across the five games.
* Every one of the 22 PNGs opened and read.

---

## G. What this round did not do

The logo, the `@[TBD]` handle, the merge to `main`, the community write-up, and
rendering all 2,761 games. D-5 (the article figure's trailing space) and the new
D-6 stay open and disclosed. The anchor-colour step in §D is offered for
the maintainer's decision, not settled by the round.
