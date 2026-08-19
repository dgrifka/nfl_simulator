# 33 — Magnitude audit: does a 6.4% luck share ever change a verdict?

*Written 2026-08-19, product layer round 1, task 1. This is a **descriptive
audit, not a candidate round**: nothing is fitted, nothing is tested, no gate
is pre-registered. It reads the shipped v1.3 artifact
(`research/outputs/dtw_games_v13.parquet`, produced by
`research/46_simulator_v13.py`) and reports what the adjudication actually
says across 2,761 games, 2016–2025.*

*Script: `research/48_magnitude_audit.py`; it writes
`research/outputs/48_magnitude_audit.json`. That directory is gitignored, as
every round's outputs are — the script is the committed artifact and this
document is the committed record of the numbers. Regenerate with
`uv run python research/48_magnitude_audit.py`.*

*Inputs: documents 05 (the neutralization rule), 06 (the 6.4% / 25.1% variance
share), 10 (the degeneracy convention and the interval caveat), 31 (what v1.3
prices).*

---

## 1. The answer, stated first

**The deserved winner differs from the scoreboard winner in 255 of 2,761 games
— 9.24%, about one game in eleven, or roughly 25 games a season.** A 6.4%
share of margin variance is not the same thing as a 6.4% chance of mattering,
because luck is concentrated: 44.4% of games are degenerate (the verdict never
moves), and what is left carries enough weight to overturn one game in six of
the 1,535 non-degenerate remainder (255 of 1,535, 16.6%).

That 9.24% is the sign-of-deserved-margin definition. §2a reconciles it with
the DTW% definition and recommends reporting three buckets rather than one
number: **195 clear flips (7.06%), 186 too close to call (6.74%)**.

The margin itself moves more than the win/loss verdict does. The median game's
deserved margin sits 2.37 points from the realized one, and 39.2% of games
move by more than a field goal.

## 2. Winner flips

A **flip** is a sign disagreement between the deserved margin and the realized
margin. Ties are excluded from the count in both directions: the 10 realized
ties in the window have no realized winner, and no game has a deserved margin
of exactly zero.

| Statistic | Count | Share of 2,761 |
|---|---|---|
| Sign flips (deserved winner ≠ realized winner) | 255 | 9.24% |
| DTW% below 0.5 for the realized winner | 279 | 10.11% |
| DTW% inside 0.40–0.60 (genuinely undecided) | 186 | 6.74% |
| Realized ties handed a deserved winner | 10 | 100% of ties |

### 2a. Which definition — reconciled 2026-08-19

**Correction to this document's first draft.** It said the two definitions
"disagree by 24 games." That is the *net* difference (279 − 255), not the
disagreement count. They disagree on **56 games**: 239 games both call a flip,
16 are sign-flip-only, and 40 are DTW%-flip-only. Each definition catches
games the other misses, so the net cancels most of the disagreement and is the
wrong statistic to quote.

The 56 disagreements are not scattered. **Every one has a DTW% between 0.363
and 0.626**, 49 of the 56 sit inside 0.40–0.60, and their median |deserved
margin| is 0.28 points (max 1.59). The two definitions agree everywhere except
in the dead-even zone, which is exactly where a binary flip label carries no
information in the first place.

That reframes the choice. The problem is not which definition is right; it is
that a binary label is being asked to describe a coin flip. **Recommendation:
report three buckets, with "too close to call" at DTW% 0.40–0.60.**

| Bucket | Games | Share |
|---|---|---|
| Clear flip (DTW% definition) | 195 | 7.06% |
| Too close to call (DTW% 0.40–0.60) | 186 | 6.74% |
| Clear flip (sign definition, for comparison) | 194 | 7.03% |

With the band in place the residual disagreement is **7 games, down from 56** —
and all 7 have a deserved margin inside ±0.45 points, so they are dead-even
games that happen to fall just outside the band rather than genuine conflicts.
The definition choice stops mattering.

If a single number is required despite the above, use the DTW% version, because
DTW% is what the product displays and a headline that contradicts the on-screen
number is worse than a slightly different count.

The point-estimate flip asks whether the average luck-neutralized margin
changes sides; the DTW% flip asks whether the bootstrap majority does. A game
can move its margin across zero while the coins still favour the team that
won, and vice versa — §2a is why that distinction only bites near 0.5.

Flips are stable across seasons: 18 (2019) to 33 (2025) per year, on 267–285
games. No season is carrying the result.

## 3. Degeneracy — how often there is nothing to adjudicate

Document 10's Gate V-3 convention is used unchanged: a game is **degenerate**
when its DTW% falls outside the open interval (0.001, 0.999).

| Statistic | Count | Share |
|---|---|---|
| Degenerate (DTW% ≤ 0.001 or ≥ 0.999) | 1,226 | 44.40% |
| DTW% exactly 0 or 1 | 1,016 | 36.80% |
| Zero-width interval | 1,105 | 40.02% |
| Games with no luck events at all | 0 | 0.00% |

Every game in the window has luck events — mean 10.8 per game, median 11,
max 22. Degeneracy is therefore never "nothing happened"; it is "enough
happened, and none of it was close to decisive." That distinction matters for
the product: a degenerate game still has a luck ledger worth showing, which is
why avenue 2's waterfall is not wasted on it.

The 44.4% figure sits close to document 10's "~50% of games have a degenerate
DTW" estimate, which was computed on synthetic Gaussian margins. The real-data
number is 5.6 points lower. Document 10's own defect register anticipated this
— real margins are lumpy at 3 and 7, which changes how often a game is
degenerate — so the two figures are consistent rather than in conflict.

## 4. The DTW% distribution

| Bin | Games |
|---|---|
| [0.000, 0.001) | 497 |
| [0.001, 0.050) | 229 |
| [0.050, 0.200) | 225 |
| [0.200, 0.400) | 194 |
| [0.400, 0.600) | 186 |
| [0.600, 0.800) | 201 |
| [0.800, 0.950) | 242 |
| [0.950, 0.999) | 259 |
| [0.999, 1.000] | 728 |

Quantiles: q01 0.0000, q05 0.0000, q10 0.0000, q25 0.0345, q50 0.6505,
q75 0.9997, q90 1.0000, q95 1.0000, q99 1.0000.

The shape is a U with 1,225 games piled in the two end bins. The median of
0.6505 is a home-field artefact of an asymmetric bimodal distribution and
should not be quoted as "the typical game" — there is no typical game here.
Any product copy that summarises this distribution with a single centre
statistic is misleading; the histogram is the honest presentation.

Interval widths, quoted both ways per the document 10 caveat: **mean width
0.0369 across all games, 0.0663 across the 1,535 non-degenerate games.** Mean
DTW% on the non-degenerate games is 0.5151.

## 5. Margin movement

|realized − deserved|, in points:

| Statistic | Value |
|---|---|
| Mean | 2.853 |
| Median | 2.366 |
| q25 / q75 | 1.116 / 4.071 |
| q95 / q99 | 7.217 / 9.946 |
| Max | 16.28 |

| Threshold | Games over | Share |
|---|---|---|
| > 0.5 pt | 2,422 | 87.72% |
| > 1 pt | 2,132 | 77.22% |
| > 3 pt | 1,081 | 39.15% |
| > 7 pt | 153 | 5.54% |

The signed mean is −0.034 points, i.e. luck neutralization is very nearly
unbiased with respect to home field. That is a sanity read, not a gate — a
large signed mean would have indicated a systematic error in the ledger's
sign convention, and it does not appear.

## 6. Example games for avenues 1, 2 and 4

Chosen here, from the audit, so that the choice is reproducible rather than
picked by eye during the plotting work. All three candidate lists are in the
JSON the script writes.

**Luck-heavy, non-degenerate — `2018_05_GB_DET`.** Detroit won by 8; the
deserved margin is −8.28 to Green Bay, a 16.28-point swing, the largest in the
window. DTW% 0.054 with interval [0.036, 0.075], 15 luck events. It is
simultaneously the largest flip, so it exercises the waterfall and the
distribution plot at full stretch.

**Degenerate with luck events — `2021_14_LV_KC`.** Kansas City won by 39; the
deserved margin is 27.93, an 11.07-point movement that changes nothing. DTW%
exactly 1.0, zero-width interval, 15 luck events. This is the case the product
must handle without looking broken: a real ledger, a big shift, and a verdict
that does not move.

A useful third, if a mid-case is wanted: **`2025_17_DET_MIN`** — Detroit won by
13, deserved margin 0.696, DTW% 0.548. The scoreboard says comfortable, the
adjudication says coin flip, and the realized winner still holds a bare
majority of draws.

## 7. What this does and does not license

**Licensed:** "The luck-neutralized verdict clearly overturns the scoreboard in
about one game in fourteen (7.1%, 195 of 2,761 games, 2016–2025), and calls
another 6.7% too close to call." "About 44% of games are decided beyond luck's
reach." "The median game's margin moves 2.4 points when luck is neutralized."
The single-definition form — "differs in 9.2%, 255 games" — is licensed only if
the definition is named alongside it (§2a).

**Not licensed:** any claim that the flipped games were *wrongly decided* —
the flip is about the deserved margin, and the deserved margin is a
bookkeeping construct, not a claim about who would win a replay (that question
is out of scope, handoff §1). Nor any single-number summary of the DTW%
distribution (§4). Nor a coverage or interval claim quoted on the all-games
figure alone (document 10). Nor a bare flip percentage with no definition
attached — the two definitions differ on 56 games and a reader cannot tell
which is being quoted (§2a).

## 8. Known-defect register

| Defect | Evidence | Status |
|---|---|---|
| The first draft of §2 said the two flip definitions "disagree by 24 games" | §2a | **Corrected 2026-08-19.** 24 is the net difference; the disagreement count is 56. The erroneous figure was never quoted outside this document |
| Two flip definitions disagree on 56 games | §2a | **Closed by the three-bucket presentation.** All 56 sit at DTW% 0.363–0.626; a "too close to call" band at 0.40–0.60 leaves 7 residual disagreements |
| The 0.40–0.60 band is a presentation choice, not a fitted threshold | §2a | **Accepted and disclosed.** It is doc 33's own coin-flip band, chosen before this reconciliation and not tuned to minimise the residual |
| Degenerate share is 44.4% here vs "~50%" in document 10 | §3 | **Closed.** Document 10's figure is synthetic-Gaussian and its own register flagged margin lumpiness as the cause |
| The margin identity reproduces to 9.6e-4 points, not exactly | Cross-check in this round | **Accepted.** `points_per_epa` is stored rounded to 4 dp (0.8389); the residual is the rounding, not a bookkeeping error |
| Everything here is a within-window description | Whole document | **Accepted.** No out-of-sample claim is made and none should be read in |
