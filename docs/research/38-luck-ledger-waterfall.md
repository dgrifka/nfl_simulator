# 38 — The luck-ledger waterfall

*Written 2026-08-23. Product-layer round 2, avenue 2 of the queue in
`handoff-2026-08-23.md` §3. Production code: `src/nfl_simulator/plots.py`,
tests `tests/test_plots.py` (22 new, written first — 59 in the file, 274 in the
suite). Figure driver: `research/55_ledger_waterfall.py`, outputs
`research/outputs/55_waterfall_<game_id>.png` and
`research/outputs/55_ledger_waterfall.json` — gitignored, as always; this
document is the record of the numbers.*

**Nothing is fitted or re-simulated in this round.** The waterfall is arithmetic
on two committed artifacts — `dtw_ledger_v13.parquet` (one row per neutralized
event) and `dtw_games_v13.parquet` (the summary) — at the slope recorded in
`model_metadata_v13.json` (`points_per_epa` = 0.8389495558). Document 37's
figure needed a replay because the shipped parquet does not keep the 160,000
margin draws; this one needs no replay, because the ledger it draws *is* the
committed artifact.

---

## 1. What was built

| Piece | What it owns |
|---|---|
| `LuckBar` | one signed step — its label, its points, how many events it stands for |
| `luck_bars` | ledger rows to bars: sign, order, folding of the slivers |
| `running_totals` | where each step begins and ends, walking from the realized margin |
| `plot_luck_ledger` | the figure — two anchors, one bar per row, connectors, legend |

A bar is `-luck_epa * points_per_epa`: the points that come off the home team's
margin when that event is neutralized. The minus is the simulator's own identity
(`simulator.py`), not a display convention —

    deserved_margin = actual_margin - total_luck_epa * points_per_epa

so luck that favoured the home team is *subtracted* from the home team's margin.

## 2. The reconciliation check

A waterfall that does not land on the deserved margin is a decomposition of
something else. The driver checks the identity on **all 2,761 games** before
drawing any of the three examples, and `plot_luck_ledger` checks it again for
the game in front of it, raising rather than drawing when the two disagree by
more than 1e-6.

| Check | Result |
|---|---|
| Games whose ledger sums to their published deserved margin | 2,761 of 2,761 |
| Max &#124;residual&#124; across all games | 7.11e-15 |

That is double-precision noise, not agreement-to-a-tolerance. The same check
inside the figure is what stops a ledger from another game, or from another
slope, being drawn under a headline it does not explain.

## 3. The three examples

| Game | Headline | Bucket | Events | Bars | Folded | Realized → deserved |
|---|---|---|---|---|---|---|
| `2018_05_GB_DET` | GB 95% / DET 5% | clear flip | 15 | 12 | 4 | +8 → −8.28 |
| `2021_14_LV_KC` | KC 100% / LV 0% | scoreboard holds | 15 | 9 | 7 | +39 → +27.93 |
| `2025_17_DET_MIN` | MIN 55% / DET 45% | too close to call | 12 | 10 | 3 | +13 → +0.70 |

Margins are stated from the home team's perspective, which is the axis the
figure draws. The three largest bars in each game:

| Game | Top three bars, in points |
|---|---|
| `2018_05_GB_DET` | −3.16 and −3.10 (40-44 yd field goals, GB), −3.03 (35-39 yd field goal, GB) |
| `2021_14_LV_KC` | −1.92 (run/live fumble, LV), −1.83 and −1.83 (pass/live fumbles, LV) |
| `2025_17_DET_MIN` | −2.73 and −2.73 (run/aborted fumbles, DET), −1.92 (run/live fumble, DET) |

Detroit's eight-point win is one game's worth of missed Green Bay kicks: three
field goals alone are worth 9.3 points of the 16.3-point swing.

**The bars in all three examples run mostly one way, and that is a property of
these games rather than of the sign convention.** League-wide the ledger's
29,752 rows split 51.5% / 48.5% between the two directions (mean `luck_epa`
0.0038 EPA). A game whose luck all points one way is exactly the kind of game
that gets picked as an example.

## 4. Presentation decisions, with reasons

**Biggest mover first, and it says so.** A waterfall looks sequential, and a
reader who takes the row order for the play order will read a story that is not
there. The default order is by size, because the figure is asked "what moved the
verdict", and the caption states that the bars are a sum whose order does not
change where it lands. `chronological=True` orders by play instead, for a
game-log reading; the endpoints are identical either way, which is the point.

**Slivers fold; they are never dropped.** An extra point is worth about 0.03
points of margin, and five of them are five blank rows. Events under 0.1 points
collapse into one row that names how many (`"4 events under 0.1 pt"`) and
carries their exact sum — so the waterfall still reconciles to the last decimal.
Dropping them instead would have broken the identity §2 checks. A lone sliver
stays as itself: "1 events under 0.1 pt" is a worse row than the event.

**Totals do not wear a team's colour.** The realized and deserved anchors are
grey; only the event bars carry the two team hues (`#eb6834` away, `#2a78d6`
home, the validated pair from document 37 §4). The legend names only the
directions the game actually contains — a key for a colour that appears nowhere
sends a reader hunting the figure for it.

**Titles, legend and caption are spaced in points, not axes fractions.** The
figure's height grows with the number of events, so a caption placed at a fixed
fraction of the height drifts further from the plot the more luck a game had.

**Room is reserved for the value labels.** The outermost bar ends at the
outermost x and its label sits beyond that; without padding the label ran off
the frame and landed on the row names. Caught by a test that measures the text
extents against the axes, not by eye.

## 5. Defects and open items

| Item | Status |
|---|---|
| The 0.1-point fold threshold is a presentation choice, not derived | **Accepted and disclosed.** It changes no arithmetic — the folded row carries the exact sum, and §2's reconciliation is checked after folding |
| Default bar order is by size, and a waterfall implies sequence | **Disclosed on the figure itself** (§4). `chronological=True` is available and unused by the driver |
| Two rows can be identical (`"40-44 yd field goal — GB"` twice) | **Left as is.** They are two different kicks; the ledger carries `play_id`, the label does not, and a reader counting rows is reading it correctly |
| The waterfall shows the point estimate of each event's luck, not its posterior | **Carried forward.** Each bar uses the ledger's `expected` at the posterior mean, which is what `ledger.py` publishes. Per-event uncertainty is document 37's distribution, taken jointly |

2026-08-26: "realized" renamed "actual" in code and figures (the maintainer's wording).
Prose in this document is left as written.

## 6. Verification

`uv run pytest -q` — 274 passed (22 new, written before the code, watched to
fail). `uv run ruff check .` and `ruff format --check .` clean. All three
figures regenerate from a clean run of `research/55_ledger_waterfall.py` with
the 2,761-game reconciliation passing at 7.11e-15.
