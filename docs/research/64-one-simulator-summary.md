# 64 — The one simulator: what the Full edition says across 1,139 games

*Written 2026-08-30, community write-up round 2. This is a **descriptive
summary, not a candidate round**: nothing is fitted, nothing is tested, no gate
is pre-registered. It reads the Full edition's summary
(`research/outputs/full_summary.parquet`, produced by
`research/76_full_edition_summary.py`) and reports what that adjudication says
across 1,139 games, 2022–2025 — the seasons FTN charting reaches, which are the
seasons the Full edition exists at all.*

*Script: `research/80_writeup_figures.py`, `--summary`. Regenerate with
`uv run python research/80_writeup_figures.py --summary`. Every number below is
in that command's output; nothing here was typed from memory.*

*Why this document exists: round 2 of the community write-up presents **one**
simulator rather than two editions, and that one is the Full edition. Document
33 is the same audit of the Strict edition over 2,761 games, and its section
headings are mirrored here so the two can be read side by side and cited by
analogy.*

*Inputs: documents 05 (the neutralization rule), 10 (the degeneracy convention
and the interval caveat), 33 (the same audit, Strict, 2016–2025), 58 and 59
(ruling R-4, the hands-on-the-ball class), 61 (the possession cap), 62 (the
capped Full edition's own numbers).*

---

## 1. The answer, stated first

**The deserved winner differs from the scoreboard winner in 168 of 1,139 games
— 14.75%, about one game in seven, or roughly 40 games a season.** Luck is
concentrated rather than spread evenly: 27.22% of games are degenerate (the
bootstrap never moves the verdict off the scoreboard), and on the 829 games
that are left, one in five flips (167 of 829, 20.14%).

That 14.75% is the sign-of-deserved-margin definition. §2a reconciles it with
the DTW% definition, and §4 reports the three buckets the product actually
labels: **128 clear flips (11.24%), 95 too close to call (8.34%), 916 where the
scoreboard holds (80.42%)**.

The margin moves further than the win/loss verdict does. The median game's
deserved margin sits **3.43 points** from the realized one, and **55.40% of
games move by more than a field goal**.

## 2. Winner flips

A **flip** is a sign disagreement between the deserved margin and the realized
margin. Ties are excluded in both directions, exactly as document 33 excludes
them: a drawn game has no realized winner to flip, and a deserved margin of
exactly zero names no deserved winner. There are 3 realized ties in the window
and no deserved tie.

| Statistic | Count | Share of 1,139 |
|---|---|---|
| Sign flips (deserved winner ≠ realized winner) | 168 | 14.75% |
| DTW% below 0.5 for the realized winner | 167 | 14.66% |
| DTW% inside 0.40–0.60 (genuinely undecided) | 95 | 8.34% |
| Realized ties handed a deserved winner | 3 | 100% of ties |

### 2a. Which definition

The two definitions **disagree on 3 games** — counted element-wise, by
comparing the two label sets game by game. It is never the difference of the
two totals: document 33's defect register records the round that quoted a net
difference as a disagreement count, and 168 − 167 = 1 is not the answer here
either.

With the "too close to call" band at DTW% 0.40–0.60 taking precedence, the
disagreement disappears from the reported buckets entirely: both definitions
identify the same **128 clear flips**. §4's three buckets are therefore
definition-independent, which is why the article quotes them rather than
either flip count on its own.

### 2b. What Fable's spot check said

The round-2 handoff carried an unverified spot check of **170 sign flips
(14.9%)**. The recomputation is **168 (14.75%)**. The gap is the tie rule: a
bare `(deserved > 0) != (actual > 0)` counts 170, because two of the three
drawn games have a positive deserved margin and are scored as flips by it. A
tie has no realized winner, so document 33 books those in their own row —
"realized ties handed a deserved winner" — and this document does the same.
The band count (95) and the degenerate count (310) both reproduce exactly.

## 3. Degeneracy — how often there is nothing to adjudicate

**Degenerate** means a DTW% outside the open interval (0.001, 0.999): the
bootstrap put essentially none of its mass on the other side, so no amount of
re-flipping the coins changes who deserved to win. The definition is document
10's gate V-3, quoted from `research/48_magnitude_audit.py` rather than
restated.

| Statistic | Count | Share of 1,139 |
|---|---|---|
| Degenerate (DTW% ≤ 0.001 or ≥ 0.999) | 310 | 27.22% |
| Non-degenerate — the games luck could still decide | 829 | 72.78% |
| Sign flips among the non-degenerate | 167 | 20.14% of 829 |

The Strict edition over 2016–2025 is degenerate in 44.4% of games (document 33
§3). The Full edition is degenerate in 27.22%, because the hands-on-the-ball
class prices dozens more events per game and a game with more coins in it is a
game whose distribution has more width.

## 4. The three buckets

The product labels a game with one of three verdicts, per document 33 §2a. The
band takes precedence over the flip: a game inside DTW% 0.40–0.60 is "too close
to call" whichever side of 0.5 it sits on.

| Bucket | Definition | Count | Share |
|---|---|---|---|
| Clear flip | DTW% on the losing side, outside the band | 128 | 11.24% |
| Too close to call | DTW% within 0.40–0.60 | 95 | 8.34% |
| The scoreboard holds | everything else | 916 | 80.42% |

The three partition the corpus: 128 + 95 + 916 = 1,139.

## 5. Margin movement

|deserved − actual|, in points, across all 1,139 games.

| Statistic | Value |
|---|---|
| Median | 3.43 pt |
| Games moving more than 3 pt | 631 (55.40%) |
| Largest swing | 19.05 pt — `2024_19_LAC_HOU`, actual +20 → deserved +0.95 |

The largest swing is the article's drops showcase: Houston won by 20 and the
Full edition puts the two sides within a point of each other.

## 6. What the ledger prices per game

| Statistic | Value |
|---|---|
| Luck events per game | median 60, mean 60.9 |
| Games with ≥ 1 dropped-pick chance | 90.69% |
| Games with ≥ 1 catchable ball | 99.91% |

**A naming caution for anyone quoting these two shares.** The parquet's
`n_dropped_picks` and `n_receiver_drops` count *priced chances*, not *dropped
balls*: every interception-worthy throw is a dropped-pick event and every
catchable target is a receiver-drop event, whether or not it was actually
dropped. That is why the median game has dozens of them. "90.69% of games have
at least one dropped-pick chance" is the sentence these numbers support; "90.69%
of games have a dropped interception in them" is not.

## 7. One fumble, term by term

The neutralization rule is `luck(e) = (y(e) − p(e)) · swing(e)`. Below is one
real event from the walk-through game, taken from the Full edition's own replay
— the same replay the article's figures are drawn from, checked against the
published summary before it is read.

**`2025_13_DEN_WAS`, second quarter, 6:43.** Marcus Mariota is sacked and
fumbles; Washington recovers its own ball.

| Term | Value | What it is |
|---|---|---|
| class | `pass/live`, charged to WAS | a live-ball fumble on a pass play |
| `y(e)` | 1 | what happened — the charged team recovered |
| `p(e)` | 0.5096 | what was expected — the class's shrunk own-recovery rate |
| `swing(e)` | +4.2675 EPA | how far apart the two branches are |
| `luck(e)` | **+2.0928 EPA** | (1 − 0.5096) × 4.2675 |
| in points | **+1.756 pt** | at 0.8389 points per EPA (document 01) |

Washington was credited with 2.09 EPA of a bounce that was a coin flip, and the
simulator hands it back. One event, 1.76 points of margin.

*A note on the choice: the round-2 handoff asked for a run-play fumble from
this game, and the game has none. All three of its fumbles are on a sack or a
kickoff. The event above is the closest one — a live-ball fumble priced at the
same `pass/live` class rate — and the article should describe it as a
sack-fumble rather than as a run.*

## 8. What this does and does not license

- **It licenses** every count above as a statement about the Full edition,
  2022–2025, and nothing about any other window. The 14.75% flip rate is not
  comparable to document 33's 9.24% — different editions, different seasons,
  and the Full edition prices a class Strict does not.
- **It does not license** calling the flip rate a measurement of how often the
  wrong team wins. It is how often *this* adjudication disagrees with the
  scoreboard, under this list of priced components, with the components refused
  for materiality (documents 16, 24, 25) and for want of a branch point
  (document 05 §3) still absent from it.
- **It does not license** a per-season or per-team split. Nothing here is
  conditioned on anything, and a 1,139-game corpus cut 32 ways is not a corpus
  any of these numbers were computed on.

## 9. Known-defect register

- **2026-08-30, the handoff's spot check.** 170 sign flips was carried into the
  round as an unverified number and is wrong by the tie rule; the audited count
  is 168. Recorded in §2b rather than silently corrected, because the spot check
  was the thing being checked.
- **No defect found in the parquet itself.** The 1,139-game count and the
  degenerate and band counts all reproduce, and `research/76`'s own gate against
  document 62 §4 is what stands behind the adjudication these numbers describe.
