# 37 — The bootstrap distribution figure

*Written 2026-08-23. Product-layer round 2, avenue 1 of the queue in
`handoff-2026-08-23.md` §3. Production code:
`src/nfl_simulator/plots.py`, tests `tests/test_plots.py` (37 of them, written
first). Figure driver: `research/54_bootstrap_figures.py`, outputs
`research/outputs/54_bootstrap_<game_id>.png` and
`research/outputs/54_bootstrap_figures.json` — gitignored, as always; this
document is the record of the numbers.*

**Nothing is fitted in this round.** Every number below is read from
`dtw_games_v13.parquet`, and the three example games are re-simulated only so
the *draws* exist — the shipped parquet keeps the summary, not the 160,000
margins behind it.

---

## 1. What was built

A plotting function in the package, plus the verdict object it presents:

| Piece | What it owns |
|---|---|
| `GameVerdict` | degeneracy, the three-bucket label, the headline, the interval caveat |
| `plot_bootstrap_distribution` | the figure — density, realized rule, deserved rule, zero rule, legend |
| `verdict_from_row` | building one from a `dtw_games_v13.parquet` row plus its draws |

`GameVerdict` recomputes nothing. A presentation layer that re-derived DTW%
could drift from the number the research record carries, so it reads the
committed field and formats it.

## 2. The replay check

The draws are only usable if they belong to the published summary. The driver
re-simulates each example at v1.3's exact settings (seed 20260817, 200 posterior
draws, 800 coin draws, blocked kicks excluded, `points_per_epa` refit at 0.8389)
and compares four fields against `dtw_games_v13.parquet` before drawing anything:

| Game | max &#124;Δ vs committed&#124; across deserved margin, DTW%, both interval ends |
|---|---|
| `2018_05_GB_DET` | 0.00e+00 |
| `2021_14_LV_KC` | 0.00e+00 |
| `2025_17_DET_MIN` | 0.00e+00 |

Exact, not close. A mismatch is a `SystemExit`, not a reconciliation.

## 3. The three examples, as the product states them

| Game | Scoreboard | Deserved margin | Headline | Bucket | Degenerate? |
|---|---|---|---|---|---|
| `2018_05_GB_DET` | DET by 8 (home) | −8.28 | **GB 95% / DET 5%** | clear flip | no |
| `2021_14_LV_KC` | KC by 39 (home) | +27.93 | **KC 100% / LV 0%** | scoreboard holds | **yes** (DTW% = 1.000) |
| `2025_17_DET_MIN` | MIN by 13 (home) | +0.70 | **MIN 55% / DET 45%** | too close to call | no |

Note the second row: *degenerate* and *scoreboard holds* are different
statements. Degeneracy is about the interval collapsing; the bucket is about
whether the deserved winner differs from the scoreboard's. `2021_14_LV_KC` is
the handoff's degenerate example and its bucket is "scoreboard holds", which is
correct — Kansas City deserved the win, just by 28 rather than 39.

## 4. Presentation decisions, with reasons

**Bins are one point of margin wide, aligned to the integer grid.** A
neutralised margin is a sum of a handful of EPA swings times 0.8389, so the
distribution is genuinely lumpy: clusters roughly a field goal apart with
extra-point structure inside them. The first draft used matplotlib's default 48
bins, whose edges fell between those clusters and combed the histogram into
alternating spikes and gaps — real clustering rendered as an artifact. A
one-point grid puts the edges where a reader already thinks in margins, and zero
lands on an edge, so no bar straddles the line that decides the winner.

**A point mass is drawn as a note, not a density.** A game with no luck events
returns one margin draw. Histogramming a single value invents a shape. None of
the three examples is one, but 2,761 games contain some and the product cannot
render them as a spike.

**The interval is mirrored onto the team the headline names.** The stored
interval is on the *home* team's share. Quoting `2018_05_GB_DET` as "89%
interval 4–8%" beside a headline reading "GB 95%" attributes Detroit's bounds to
Green Bay's number. The figure states "the 89% interval on GB's share runs
92–96%".

**Colour and accessibility.** Two categorical hues on surface `#fcfcfb`,
`#2a78d6` (home) and `#eb6834` (away), validated: lightness band, chroma floor,
CVD separation (ΔE 24.7 protan, 32.7 tritan), normal-vision floor (ΔE 33.6) and
contrast all PASS. Both fills carry a legend naming the team, so identity is
never colour alone; every rule and label is ink rather than a series colour.
PNG for print — no hover layer, no dark mode, matching document 04's precedent.

## 5. Correction to the handoff's interval caveat

`handoff-2026-08-23.md` §1 states the caveat as *"the bootstrap's nominal 97%
interval is an 89% interval on non-degenerate games."* **That is backwards, and
the 97% is superseded.** The committed record says:

- The interval is **nominally 89%** (document 03's 5.5 / 94.5 convention;
  `ETI_LOW`, `ETI_HIGH` in `simulator.py`).
- 97% was the **pre-remediation** reading, at 100 coin draws — document 10's
  finding, not its conclusion. Document 10 diagnosed it to `n_coin_draws`,
  raised the constant to 800, and re-measured.
- At the shipped 800 coin draws, **coverage on informative games is 0.9152**
  against nominal 0.89 — mildly conservative, about two points wide. Document 10
  ships this as the claim: *"the two-layer bootstrap's 89% interval covers the
  truth about 91.5% of the time on games where there is something to
  adjudicate."*
- 44.4% of games are degenerate (1,226 of 2,761, document 33 §3) and their
  interval collapses to a point; those get a different sentence entirely, with
  no coverage figure in it.

The module quotes the corrected pair. Nothing shipped on the handoff's wording.

## 6. Reproduction of document 33's conventions

The band and degeneracy definitions were re-derived from
`dtw_games_v13.parquet` rather than trusted, and all four reproduce:

| Quantity | This round | Document 33 |
|---|---|---|
| DTW% inside 0.40–0.60 | 186 | 186 (§2a) |
| DTW% below 0.5 for the realized winner | 279 | 279 (§2) |
| Clear flips (DTW% definition, band applied) | 195 | 195 (§2a) |
| Degenerate games | 1,226 (44.40%) | 1,226 (44.40%) (§3) |

Inclusive and exclusive band edges give the same 186, so the edge convention is
free; the module uses inclusive.

## 7. Defects and open items

| Item | Status |
|---|---|
| Handoff §1's interval caveat is inverted and quotes a superseded figure | **Corrected here** (§5). Handoff file left as written — it is a dated record |
| A realized tie gets no bucket from document 33, which excluded all 10 from its flip counts | **Decided, disclosed.** The module labels a tie outside the band a *clear flip* — the scoreboard named nobody and the bootstrap does. No example game is a tie |
| The one-point bin width is a presentation choice, not derived | **Accepted and disclosed.** It is not fitted to any game; §4 gives the reason |
| Degeneracy uses document 10 Gate V-3's ε = 0.001 unchanged | **Carried forward**, not re-argued |
| The two rule labels overprint when the deserved and realized margins are close | **Fixed 2026-08-24** (addendum below) |

### 7a. Addendum, 2026-08-24 — the rule labels

Both rule labels hung inside the top of the plot, each to the right of its own
rule, so a game whose two margins are within a label's width of each other
printed one straight through the other. `2025_13_DEN_WAS` — deserved −3.3
against a realized −1 — was the case that exposed it; none of this document's
three examples is close enough to have shown it.

The labels are now **measured** rather than assumed apart: after both are
placed, `_lift_colliding_label` asks the renderer for their bounding boxes and,
only if they intersect, moves the **left-hand** one above the top spine, into
the empty band between the plot and its subtitle. Two choices are load-bearing.
Left-hand, because a label runs to the right of its own rule — lifting that one
also takes it off the other rule, which was otherwise striking it through.
Above the spine rather than a second row inside the plot, because the rules stop
at the spine while a second row lands the text on whatever bar is tallest at
that margin, which on a realized-margin rule is often the tallest bar in the
figure.

Four tests over margin gaps 0.0, 0.4, 1.0 and 2.3 assert the two boxes do not
intersect, and that neither lands on the subtitle; three more assert the boxes
stay clear on this document's three examples, which were never the defect and
are there so the fix cannot regress the common case. All three figures
regenerate unchanged, with the replay still at 0.00e+00.

2026-08-26: "realized" renamed "actual" in code and figures (the maintainer's wording).
Prose in this document is left as written.

## 8. Verification

`uv run pytest -q` — 252 passed (37 new). `uv run ruff check .` and
`ruff format --check .` clean. All three figures regenerate from a clean run of
`research/54_bootstrap_figures.py` with the replay check passing at 0.00e+00.
