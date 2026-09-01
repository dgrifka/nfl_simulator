# 39 — The flip-band sweep: is 0.40–0.60 load-bearing?

*Written 2026-08-24. Product-layer round 2, item 3. Production code:
`src/nfl_simulator/plots.py`,
tests `tests/test_plots.py` (18 new, written first — 77 in the file, 292 in the
suite). Driver: `research/56_flip_band_sweep.py`, outputs
`research/outputs/56_flip_band_sweep.png` and `56_flip_band_sweep.json` —
gitignored, as always; this document is the record of the numbers.*

**Nothing is fitted and nothing is re-simulated.** Every number is a re-label of
the committed `dtw_games_v13.parquet` at sixteen candidate bands.

---

## 1. The answer, stated first

**The bucket counts are sensitive to the band and the definition problem is
not.** Moving the band one step in either direction from 0.40–0.60 moves about
15 games between "clear flip" and "too close to call" — 203 clear flips at the
shipped band, 210 one step narrower, 199 one step wider. Across the whole swept
range the counts move a lot: 289 clear flips with no band at all, 165 at
0.35–0.65.

What does **not** move is the thing the band was adopted for. Document 33 §2a
introduced it because the two available flip definitions disagreed on 56 games;
that residual falls monotonically as the band opens, and it is already down to 7
at 0.41–0.59, one step before the shipped choice.

**0.40–0.60 is visibly not fitted.** If it had been chosen to minimise the
residual disagreement, the sweep would have chosen 0.36–0.64, where the residual
is zero. The shipped band is a rounder, narrower, more conservative choice than
the optimum, which is what a presentation convention looks like and not what a
tuned threshold looks like.

## 2. Reproduction of document 33

The counts are computed a different way here — one call to
`plots.bucket_label` per game, the same function the product headline uses —
rather than by the audit's own arithmetic. Agreement is therefore a
reproduction, not a copy. The driver stops rather than drawing anything if any
row disagrees.

| Quantity | This round | Document 33 |
|---|---|---|
| Games in the window | 2,761 | 2,761 |
| Too close to call, shipped band | 186 | 186 (§2a) |
| Clear flips, shipped band, ties excluded | 195 | 195 (§2a) |
| DTW% flips with no band, ties excluded | 279 | 279 (§2) |
| Definition disagreements, no band | 56 | 56 (§2a) |
| Definition disagreements, shipped band | 7 | 7 (§2a) |

## 3. The sweep

The band is symmetric around 0.5 and opens in hundredths, from empty (a binary
flip label) to 0.35–0.65. "Ties out" is how many of the 10 realized ties fall
outside the band; "disagree" is how many games the sign-of-margin and DTW%
definitions still label differently once the band has taken the undecided games
out.

| Band | Clear flip | Too close | Scoreboard holds | Ties out | Disagree |
|---|---|---|---|---|---|
| 0.50 only | 289 | 0 | 2,472 | 10 | 56 |
| 0.49–0.51 | 284 | 11 | 2,466 | 10 | 50 |
| 0.48–0.52 | 281 | 21 | 2,459 | 10 | 48 |
| 0.47–0.53 | 273 | 37 | 2,451 | 10 | 39 |
| 0.46–0.54 | 264 | 52 | 2,445 | 10 | 34 |
| 0.45–0.55 | 255 | 76 | 2,430 | 10 | 27 |
| 0.44–0.56 | 238 | 101 | 2,422 | 9 | 18 |
| 0.43–0.57 | 229 | 118 | 2,414 | 9 | 16 |
| 0.42–0.58 | 218 | 146 | 2,397 | 8 | 10 |
| 0.41–0.59 | 210 | 169 | 2,382 | 8 | 7 |
| **0.40–0.60 (shipped)** | **203** | **186** | **2,372** | **8** | **7** |
| 0.39–0.61 | 199 | 200 | 2,362 | 8 | 4 |
| 0.38–0.62 | 193 | 218 | 2,350 | 8 | 3 |
| 0.37–0.63 | 184 | 243 | 2,334 | 8 | 2 |
| 0.36–0.64 | 175 | 263 | 2,323 | 7 | 0 |
| 0.35–0.65 | 165 | 286 | 2,310 | 7 | 0 |

Every row sums to 2,761: a game lands in exactly one bucket at every width, and
the sweep is a re-partition of the same league rather than three independent
counts.

## 4. The 203 / 195 gap — the product labels ties, the audit excluded them

The sweep's "clear flip" column reads **203** at the shipped band where document
33 §2a published **195**. The eight games are realized ties that fall outside the
band. Document 33 excluded all 10 ties from its flip counts because a tie has no
realized winner to flip; document 37 §7 decided that a product asked to render
one still has to say something, and labels a tie outside the band a clear flip —
the scoreboard named nobody and the bootstrap does.

Both readings are in every row (`ties_outside_band`), so neither count can be
quoted without the other being recoverable. Any public statement of a flip count
has to name which one it is, exactly as document 33 §7 requires of the two
definitions.

## 5. Presentation decisions, with reasons

**Three panels, each on its own scale.** "Scoreboard holds" runs near 2,400 while
the two buckets the band trades between run in the low hundreds. On one shared
axis the movement the figure exists to show is a flat line at the bottom of the
frame, and a second y scale is never the answer. Three panels on a shared x axis
keep all three legible and keep the axis honest.

**No series wears a colour.** A bucket is not an entity the way a team is, and
with one line to a panel the title already names it. The two team hues stay
reserved for the team fills in documents 37 and 38 — reusing them here would
teach a reader that blue means something other than the home team. Nothing in
this figure needs a legend or a palette validation, because nothing in it
encodes identity by colour.

**The value label goes to whichever side the line is leaving.** The shipped
band's own rule runs vertically through the point, so a label centred above it
comes out struck through. It sits beside the marker instead, above on a falling
series and below on a rising one, so the line never climbs through its own
number. Locked by a test rather than by eye.

**Counts have no negative side.** The y margin that keeps the lines off the frame
put a "−50 games" tick under the middle panel. The bottom is clamped at zero; the
top is left free.

## 6. What this does and does not license

**Licensed:** "The three-bucket counts move with the band — about 15 games a
step — and the band is a presentation convention, not a fitted threshold." "The
band was not chosen to minimise the disagreement between the two flip
definitions: a wider band drives that disagreement to zero and was not
adopted." "At the shipped band the label is 203 clear flips, 186 too close to
call, 2,372 scoreboard holds — 195 clear flips if realized ties are excluded as
document 33 excluded them."

**Not licensed:** any claim that a particular band is *correct*. The sweep prices
the choice; it does not adjudicate it, and nothing here was pre-registered as a
gate because nothing here is a test. Nor any flip count quoted without saying
whether ties are in it (§4) and which definition it uses (document 33 §7).

## 7. Known-defect register

| Defect | Evidence | Status |
|---|---|---|
| The sweep's clear-flip count is 203 where document 33 published 195 | §4 | **Explained, not a defect in either.** The 8 games are realized ties outside the band; both readings ship in every row |
| The band grid is hundredths, so a band between two steps is not shown | §3 | **Accepted.** The shipped 0.40–0.60 is on the grid, which is what the display has to place |
| The residual-disagreement column is computed in the driver, not in the package | `research/56_flip_band_sweep.py` | **Accepted and disclosed.** The product displays bucket counts; the disagreement is a record-keeping check, and putting it in the package would ship code nothing renders |
| Everything here is a within-window re-description of one committed artifact | Whole document | **Accepted.** No out-of-sample claim is made and none should be read in |

2026-08-26: "realized" renamed "actual" in code and figures (the maintainer's wording).
Prose in this document is left as written.

## 8. Verification

`uv run pytest -q` — 292 passed (18 new, written before the code and watched to
fail). `uv run ruff check .` and `ruff format --check .` clean. All six
reproduction checks in §2 pass on a clean run of
`research/56_flip_band_sweep.py`; the figure was rendered and read before this
document was written.
