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

### 7a. The fumble classes v1.3 actually prices with

**This table is the only source the article may cite for a class rate.** It is
read off the *same fitted baseline the product replays with* —
`render._simulation_context()["fumble_baseline"]`, built by
`components.fit_fumble_baseline` on 2016–2025 play-by-play — so every rate below
is a rate some ledger row was actually priced at, not a number from a study of a
narrower population.

| Class | Fumbles | `p` — the charged team keeps it | `swing` (EPA) | What it is |
|---|---:|---:|---:|---|
| `pass/live` | 3,226 | **50.96%** | 4.2675 | a fumble on a pass play, ball live |
| `run/live` | 1,273 | **46.11%** | 4.9631 | a fumble on a run, ball live |
| `run/aborted` | 974 | **76.90%** | 4.2295 | a botched snap on a run play |
| `punt/live` | 757 | **68.43%** | 5.0004 | a muffed or fumbled punt, ball live |
| `kickoff/live` | 201 | **51.24%** | 5.1405 | a fumble on a kickoff, ball live |
| `pass/aborted` | 68 | **100.00%** | 4.2805 | a botched snap on a pass play |
| `field_goal/live` | 4 | 56.48% (pooled) | 7.5900 | below the 30-fumble floor |
| `punt/aborted` | 2 | 56.48% (pooled) | 3.9258 | below the 30-fumble floor |
| **total** | **6,505** | | | |

Three things a reader has to know before quoting any of these.

1. **`p` is a *retention* rate, not a recovery rate.** Since v1.2 a ball that
   crosses the sideline counts as kept by the team that fumbled it
   (`components._fumble_frame`). Document 05 §3 published a narrower recovery
   rate on a narrower population, and its numbers — 40.3% on a run-play fumble,
   76.2% on an aborted snap — are **not** these numbers. They are a different
   statistic on a different set of plays. Round 2's article quotes this table.
2. **The last two classes carry no rate of their own.** Six plays across ten
   seasons is not a coin anyone can estimate, so `fit_fumble_baseline` gives
   them the pooled retention rate (56.48%) under its 30-fumble floor. They are
   listed for completeness; the article quotes only the six measured classes.
3. **`pass/aborted` is 100% because it has never gone the other way.** All 68
   botched snaps on a pass play across ten seasons were recovered by the
   offence. The class's own `p` of 1.0 correctly zeroes the luck on every one of
   them — `(y − p) = 0` — so the finite `swing` in the table is never multiplied
   by anything.

## 8. The walk-through game, row by row

`2025_13_DEN_WAS` — **Denver 27, Washington 26, in overtime**. Every number
below comes from two replays of that game, each checked against its own
published summary by `render.replay` before a single row was read, so this is
the adjudication the article's figures 5, 6 and 16 are drawn from.

**Signs.** `luck_epa` is already signed for the **home team, Washington**: a
positive row is EPA the game handed Washington, whichever team the event is
charged to. Points are that times 0.8389, and the column sums by construction to
the gap between the actual margin and the deserved one.

### 8a. Every v1.3 row

Twelve rows: three fumbles, four field goals, five extra points. That is the
whole of what v1.3 prices in this game.

| Play | Component | Class | Charged | `y` | `p` | `swing` | `luck` (EPA) | Points |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1438 | fumble | `pass/live` | WAS | 1 | 0.5096 | +4.2675 | **+2.0928** | +1.756 |
| 2094 | fumble | `kickoff/live` | WAS | 1 | 0.5124 | +5.1405 | **+2.5067** | +2.103 |
| 3404 | fumble | `pass/live` | DEN | 1 | 0.5097 | −4.2675 | **−2.0924** | −1.755 |
| 301 | field goal | 30–34 yd | DEN | 1 | 0.9470 | −3.6703 | −0.1946 | −0.163 |
| 1339 | field goal | 30–34 yd | DEN | 1 | 0.9473 | −3.6703 | −0.1933 | −0.162 |
| 3304 | field goal | 35–39 yd | WAS | 1 | 0.8817 | +3.9839 | +0.4713 | +0.395 |
| 4526 | field goal | 30–34 yd | WAS | 1 | 0.9460 | +3.6703 | +0.1983 | +0.166 |
| 1744 | extra point | extra point | WAS | 1 | 0.9446 | +0.9945 | +0.0551 | +0.046 |
| 2079 | extra point | extra point | DEN | 1 | 0.9545 | −0.9945 | −0.0453 | −0.038 |
| 2467 | extra point | extra point | WAS | 1 | 0.9440 | +0.9945 | +0.0557 | +0.047 |
| 2826 | extra point | extra point | DEN | 1 | 0.9539 | −0.9945 | −0.0459 | −0.038 |
| 4710 | extra point | extra point | DEN | 1 | 0.9544 | −0.9945 | −0.0454 | −0.038 |
| | **total** | | | | | | **+2.7631** | **+2.318** |

Every `y` in this game is 1: every fumble was recovered by the team that lost
it, and every kick was made. The two fumbles charged to Washington are what the
column is made of — 4.60 of its 2.76 EPA — and the article's worked example
(§7) is the first of them.

### 8b. What the Full edition adds

The Full edition prices amendment A-3's hands-on-the-ball class on top of those
twelve rows, and document 61's possession cap bounds each drive by its own
largest event.

| Component | Rows | `luck` (EPA) | Points |
|---|---:|---:|---:|
| fumble | 3 | +2.5071 | +2.103 |
| field goal | 4 | +0.2817 | +0.236 |
| extra point | 5 | −0.0258 | −0.022 |
| dropped pick | 2 | −1.1004 | −0.923 |
| receiver drop | 60 | −1.1283 | −0.947 |
| possession cap | 15 | −0.1175 | −0.099 |
| **total** | **89** | **+0.4169** | **+0.350** |

Sixty receiver-drop rows is ordinary rather than remarkable: every catchable
target is a priced chance, per the caution in §6, and the median game across the
corpus carries about sixty luck events in all. Individually they are small and
they point both ways. Their net — −0.95 points — together with the two dropped
picks is what turns a 2.32-point charge against Washington into a 0.35-point
one.

### 8c. With and without the hands-on-the-ball rows

| | Without A-3 (v1.3) | With A-3 (Full) |
|---|---:|---:|
| DTW% — Washington | 0.1449 | **0.4058** |
| DTW% — Denver | 0.8551 | **0.5942** |
| Deserved margin (home-signed) | −3.3181 | **−1.3498** |
| Actual margin (home-signed) | −1.0 | −1.0 |

Denver's deserved-win share falls from **86% to 59%**, and its deserved margin
from 3.3 points to 1.3. **The verdict did not flip** — Denver deserved to win
under both readings — but it stopped being a verdict: 59% is inside the
too-close-to-call band, and 86% is not. Figure 16 draws the two distributions on
one axis, and §8a plus §8b are the rows that moved between them.

## 9. What this does and does not license

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

## 10. Known-defect register

- **2026-08-30, the handoff's spot check.** 170 sign flips was carried into the
  round as an unverified number and is wrong by the tie rule; the audited count
  is 168. Recorded in §2b rather than silently corrected, because the spot check
  was the thing being checked.
- **2026-08-30, document 05 §3's class rates are a different statistic.** The
  round-2 draft carried "40.3% on a run-play fumble to 76.2% on an aborted snap"
  from document 05 §3 into the article. Those are recovery rates on the narrower
  pre-v1.2 population; the rates v1.3 prices with are **46.11%** and **76.90%**
  (§7a), because a ball crossing the sideline now counts as kept. Not a defect in
  either document — a defect in quoting one where the other applies.
- **2026-08-30, document 01's r² is rounded the wrong way.** Document 01 reports
  r² = 0.991 in prose; its own artifact stores r = 0.995763, whose square is
  0.99154 and rounds to **0.992**. Figure 13 prints 0.992 and checks the stored
  correlation rather than the printed square. Recorded rather than corrected in
  document 01, which is not this round's document.
- **No defect found in the parquet itself.** The 1,139-game count and the
  degenerate and band counts all reproduce, and `research/76`'s own gate against
  document 62 §4 is what stands behind the adjudication these numbers describe.

## 11. Round 3's new figures, and the numbers they publish

Five figures drawn by `research/80_writeup_figures.py` on 2026-08-30. Every
number below is `check`ed in that script against either an earlier document or
this section, so a redraw that drifted would fail the run rather than ship.

### 11a. Figure 17 — the fumble classes, drawn

`17_fumble_retention_bars.png` is §7a as a chart, and publishes no number of its
own. The six measured classes, their rates and their counts are `check`ed
against §7a's table; the pooled rate (56.48%) is drawn as a rule, and the two
sub-floor classes are **not** drawn, because a bar for each would be the pooled
rate under two labels a reader would take for measurements. Total 6,505 fumbles.

### 11b. Figure 18 — the kicker, and why it is not 5.35 pp

`18_kicker_prior_posterior.png`. The subject is chosen by a **rule**: the
largest posterior mean effect among `KICKER_SEASON` (2025) kicker-seasons with
at least `KICKER_FLOOR` (20) attempts, read from
`research/outputs/fg_kicker_effects.parquet`. It resolves to:

| | |
|---|---|
| Kicker-season | `2025_00-0034173` — **E. Pineiro**, San Francisco, 2025 |
| Attempts | **32**, of which **31** made (96.88%) |
| Posterior mean effect | **+0.4499** on the logit scale |
| League make rate at 45 yd | **79.88%** |
| Pineiro's make rate at 45 yd | **83.14%**, a gain of **+3.26 pp** |
| `sigma_kicker` (shipped refit) | **0.3855** |
| A one-SD kicker's gain at 45 yd | **+5.48 pp** |

**The last row is not document 05b §9's 5.35 pp, and the difference is not an
error in either place.** §9 reports the **Phase 2** posterior, whose
`sigma_kicker` is 0.360 and whose league rate at 45 yards is 79.5%. The model
the simulator actually replays with is the **v1.3 refit** — `trace_fg_refit.nc`,
loaded by `render._simulation_context()` — whose `sigma_kicker` is 0.385 (doc
05b §11's fitted-model table, refit column) and whose league rate at 45 yards is
79.88%. Pushing 0.385 through that curve gives 5.48 pp, 89% interval
[4.45, 6.52], against §9's 5.35 pp and [4.17, 6.45].

The figure draws the shipped model and prints the shipped model's number.
Drawing one model and captioning it with a retired model's arithmetic would be
a figure that disagrees with itself, and no reader could tell which half was
wrong. **The article must quote 5.48 pp when it is describing what the simulator
does**, and 5.35 pp only if it is describing Phase 2's gate — which it has no
reason to.

*This is a reporting reconciliation, not a defect in document 05b.* §9's number
was correct for the model §9 was about. It is added to the defect register below
only because it is a live mis-quotation risk.

### 11c. Figure 19 — Denver's 2024 defense-season

`19_denver_prior_posterior.png`, on `worthy_throw_frame`'s `2024|DEN`:

| | |
|---|---|
| Interception-worthy throws faced | **17** |
| Caught | **13** — an observed **76.5%** |
| The league surface, scored on those same 17 throws | **49.8%** |
| The model's posterior mean for Denver | **55.2%** |

The posterior mean reproduces the value figure 15 already draws a dot at. The
league figure and the two counts are published here for the first time. The
model moves **(55.2 − 49.8) / (76.5 − 49.8) = 20.2%** of the way from the league
to what happened, which is the figure's caption and the whole of its point:
seventeen throws is not enough evidence to move it further.

### 11d. Figure 20 — two near-coins, two statistics

`20_persistent_share.png` transcribes two published numbers and **they are not
the same statistic**:

| Component | Number | What it is | Source |
|---|---:|---|---|
| Dropped pick | **1.4%** | share of per-throw variance the defense's persistent skill explains | document 48, via document 52 §2 |
| Fumble recovery | **1.1%** | shrinkage weight `w` — how much of the information about its true rate a team-season's own record carries | document 05 §3 |

Both answer "how much of this is the team rather than the bounce?" and both
land near one percent, which is why the handoff paired them. Neither is the
other's number. The figure's axis labels name each statistic and its footer
says in full that they are different measurements; **the article must not
describe them as one quantity measured twice.**

The receiver drop's **0.088%** (document 57 §1b) is the same statistic as the
1.4% and would make a truer three-bar comparison. It is not drawn — the handoff
specified two bars — and is listed in the round's results file as an avenue.

### 11e. Figure 21 — the bootstrap, built up

`21_bootstrap_buildup.png`, on `2025_13_DEN_WAS` Full.
`SimulationResult.margin_draws` is `margins.ravel()` of the shipped
`(200, 800)` bootstrap, so reshaping it recovers the two layers exactly and no
panel is a re-simulation. Washington's deserved-win share by panel:

| Panel | Draws | WAS share |
|---|---|---:|
| 1 × 800 | one posterior draw, 800 coin flips | **0.385** |
| 10 × 800 | ten posterior draws | **0.412** |
| 200 × 800 | all 160,000 — the shipped histogram | **0.4058** |

The third reproduces §8c's 0.4058 by construction — it is the same array — and
the first two are published here. They are the figure's argument: one posterior
draw is off by two points of share, ten is within a point, and the shipped 200
is the number the product reports.

### 11f. Added to the defect register

- **2026-08-30, the kicker spread has two published values and only one is
  shipped.** Document 05b §9's **5.35 pp** is Phase 2 (`sigma_kicker` 0.360);
  the simulator prices with the refit at **0.385**, which is **5.48 pp** at 45
  yards. §11b reconciles them. **Accepted, not corrected** — §9 is correct about
  the model §9 describes, and document 05b is not this round's document. The
  risk is quotation, so the rule is written here: any public number about kicker
  spread comes from §11b unless the sentence is explicitly about Phase 2's gate.
