# 34 — The placement meter: a per-game sequencing-luck design

*Written 2026-08-19, from a brainstorm session on sequencing luck. This is a
**design document, not a pre-registration**: no fit has run, no threshold is
committed here. The
implementation round commits its own pre-registration — thresholds with
power measured first — before any statistic is computed, per the standing
process laws. Every decision below was made explicitly by the maintainer in the
brainstorm; the "Decisions" table records which.*

*Inputs: documents 05 (neutralization principle, Gate A), 08 (sequencing
verdicts and the DQW% failure), 33 (magnitude audit and the flip-bucket
reconciliation), and the brainstorm brief.*

---

## 1. One-page story

Doc 08 established that where a team's production lands — red zone,
late downs — is luck: the gaps do not persist (r = −0.034 and +0.000 at
87–92% power). But that luck has no branch point, so it can never be a
ledger row, and the one attempt to report it per game (DQW%, doc 08
§10–11) failed its rematch gate because its baseline — depth reached —
destroyed 29.4% of the real spread between offenses.

This design reports placement luck per game without either mistake. The
**placement meter** is a descriptive, points-scaled score: how many
points a team's play placement was worth that game, relative to
indifferent placement of the same production. Its baseline is the team's
*own* overall efficiency in that same game, so a team that is uniformly
good scores zero by construction — the skill-destruction channel that
killed DQW% is closed at the definition, not patched afterwards. Its
uncertainty display is a within-game permutation band: shuffle which of
the team's realized plays landed in the leverage cells and read the
spread of the score.

**There is no counterfactual and the design says so.** No play is
replayed, no drive continued, no clock touched — only labels move. The
meter is a priced description of a realized luck draw, displayed beside
DTW%, never folded into it.

## 2. Decisions made in the brainstorm (the maintainer, 2026-08-19)

| Decision | Choice | Alternatives disposed |
|---|---|---|
| Meter scope | **Placement-only** — red-zone and late-down placement, the two channels doc 08 ruled luck | Full conversion residual (would mix S3 skill in) — not pursued |
| Meter units | **Points-scaled** — "placement was worth about +X points tonight," with a chance band | Percentile index; points-plus-percentile — not pursued |
| Design | **A+B**: own-game-baseline gap score + constrained permutation band | Design C (league-conditional red-zone pricing) **parked**; design D (drive-vs-play decomposition) **refused** (§9) |
| Flip label (doc 33 §2a) | **Three buckets at DTW% 0.40–0.60**: clear flip / too close to call / scoreboard holds, DTW% definition for the clear buckets | Band sweep queued as a robustness display; interval-based definition stays parked |
| Metric set | Confirmed as §8 | Avenues 1, 2, 4 unblocked |

## 3. The score

One team-game produces one number, **placement points**.

- **Cells.** Partition the team's offensive scrimmage plays (pass/run,
  non-null EPA and down — doc 08's S0–S2 filter) into three **disjoint**
  cells: red zone (`yardline_100 <= 20`, any down); late-down outside
  the red zone (down 3 or 4, `yardline_100 > 20`); everything else.
  Disjoint because the score must sum — doc 08's S1/S2 overlap was fine
  for two season-level tests, not for one number.
- **Per-cell score.**
  `(mean EPA in cell − mean EPA all plays) × n_cell × points_per_epa`,
  with `points_per_epa` = 0.8389. The meter is the **sum over the two
  leverage cells**. The third cell's score is exactly the negative of
  that sum — a built-in identity (all three cells sum to zero), the
  round-trip check in this design's currency.
- **Baseline.** The team's own overall EPA per play *in that game*. The
  skill defense: a uniformly good team scores zero by construction.
- **Per team, and a differential.** Each team gets an offensive score;
  the game's headline is the differential (home minus away) — margin
  scale, never addable to the margin, and the copy must say so.
- **Deviation from doc 08, named.** S2's luck verdict was on the
  *success-rate* gap; pricing in points forces the EPA gap on late
  downs. The shipped score therefore earns its own luck license (Gate
  M-3) rather than borrowing S2's.

## 4. The permutation band

The band answers: *how big a placement-points number does chance produce
with this same game's production?*

- **Mechanics.** Within one team-game, hold the multiset of luck-priced
  play EPAs and the three cell sizes fixed; randomly reassign plays to
  cells; recompute the score; ~2,000 draws. Display the 89%
  equal-tailed band in points, house convention: *"placement was worth
  +4.2 points (chance placement of this production spans −3.5 to
  +3.6)."*
- **Why shuffling is almost licensed.** EPA is state-adjusted (down,
  distance, field position), which is what makes cross-cell shuffling
  defensible at all. The residual problem is second-moment: red-zone
  play-level EPA variance is 2.250 vs 1.916 overall (doc 08 §2) — the
  state stretches outcomes, so a raw shuffle's red-zone band is
  slightly too narrow and too many games look unusually lucky.
- **The constraint ladder — the candidate set for the implementation
  pre-registration.** (1) raw shuffle; (2) down-stratified — plays keep
  their down, only field-position membership moves; (3) down-stratified
  plus a variance-matching correction on the red-zone cell. Too loose:
  mechanical state effects pollute the band. Too tight: the band
  collapses and every game looks extreme.
- **The calibration instrument that picks between them.** If a variant
  is well calibrated and placement is luck (doc 08), each game's
  realized score is one draw from its own game's null, so its
  percentile within the null should be uniform across all 2,761 games
  (a PIT check). The uniformity tolerance and the adoption rule are
  pre-registered **before** any variant runs; all variants are run and
  all are reported. Extreme-piling flags too loose; centre-piling flags
  too tight.
- **What the band is not.** Not a per-game hypothesis test and not an
  interval on an estimate — the realized score is exact. The band is
  the reference scale that stops ±2 points from being read as
  meaningful in a game where chance spans ±4.
- **Boundary statement.** Labels are permuted on realized plays;
  nothing is replayed. This sits on the safe side of the settled
  no-re-simulation line.

## 5. Inputs, double-counting, edges

- **Input stream.** Each play's EPA is taken as `epa − luck_epa` for
  any play carrying a v1.3 ledger row — the luck-priced stream. A
  red-zone fumble's coin swing lives only in the DTW ledger; the meter
  sees that play at its expectation. The double-counting fix is at the
  input, not patched at the output.
- **Two-meters contract.** DTW% re-draws the coins holding placement
  fixed; the placement meter re-shuffles placement holding the coins at
  expectation. Complementary by construction; never added, averaged, or
  combined (doc 08 §6, carried verbatim).
- **Empty and thin cells.** ~2% of team-games have zero red-zone
  plays: that cell contributes exactly 0 and the display says "no
  red-zone trips." No minimum-n floor elsewhere — a thin cell gets the
  wide band it deserves.
- **Degenerate games.** A blowout still gets a placement number; the
  meter is descriptive and independent of DTW degeneracy (doc 33 §3's
  reasoning for showing ledgers in degenerate games).
- **Scope.** Special-teams placement is outside the meter (outside the
  filter; never tested by doc 08). Stated wherever the meter is
  documented.

## 6. Validation gates and the power plan

Nothing here neutralizes anything, so Gate A is never in play — the
meter books no ledger rows. Every threshold below is committed by the
implementation round's pre-registration only after its power is
measured.

| Gate | Question | Pass rule (shape) | Power instrument |
|---|---|---|---|
| **M-1 identities** | Do the books balance? | Cells sum to zero exactly; empty cell = 0; luck-priced stream reconciles with the v1.3 ledger per game | None needed — identities |
| **M-2 calibration** | Is the band honest? | PIT of realized score vs own-game null uniform within a tolerance, across 2,761 games; candidate set §4, adoption rule fixed in advance | Simulate exchangeable-truth and graded miscalibration; set tolerance at a measured false-alarm rate |
| **M-3 luck license** | Is the shipped score still luck? | Doc 08 split-half machinery on the per-game score at team-season grain; r below a fresh permutation null's 95th pct | Power ≥ 0.80 at r = 0.12, measured first (Gate S-3 pattern) |
| **M-4 skill preservation** | Does the baseline leak skill? | corr(team-season mean score, S0 offensive quality) within a bound set from a simulated true-zero null (±0.11 detectability floor applies) | Simulated null; power measured before the bound is set. Secondary descriptive: dispersion vs quality (n_cell scales the score) |
| **M-5 magnitude report** | How often does it matter? | **Report, no threshold** (Gate C convention): median and q95 \|score\|, share of games outside their own band (~11% if M-2 holds), correlation with DTW%, overlap with the 195 flips, per-season stability | — |
| **M-6 rematch check** | Does subtracting it lose information? | Deserved margin minus placement differential vs deserved margin, doc 06 harness, 531 pairs, non-inferiority at +0.010 log loss — **only if powered** | Simulate pure-luck placement at the meter's real magnitude first; if the gate cannot pass under its own truth at 80% power, M-6 is pre-registered as descriptive, not pass/fail |

On M-6: combining the two meters is banned in the product but legitimate
inside a validation instrument — it is how DTW% itself was validated.
The DQW% red-zone arm partly failed on interval width, which is why the
power check runs before the gate is given teeth.

**Failure routing, stated in advance.** M-1 or M-2 fail → hold, nothing
ships. M-3 fail → the word "luck" is not licensed; the fork goes to
the maintainer. M-4 fail → the baseline is broken; back to design. M-6 fail when
powered → placement contains skill, contradicting M-3/M-4, and the
contradiction is the finding.

## 7. Mechanism story and honest admissions

- **Mechanism.** Doc 08: placement of state-adjusted production is
  exchangeable noise — teams do not control where their good plays
  land relative to the red zone and late downs. The meter prices the
  realized draw of that noise.
- **Counterfactual.** None, and none is claimed. The implied comparison
  — "the same plays, placed indifferently" — is doc 08 §5's generative
  story, not a physical branch.
- **Known risks, before the build.** (1) The EPA-gap late-down variant
  is not literally the tested S2 statistic — M-3 exists for this.
  (2) Cell-count endogeneity: good offenses take more red-zone snaps,
  so score dispersion may track quality even with a clean mean — M-4's
  secondary read watches it. (3) The constraint ladder may have no
  calibrated rung — then the band is unresolvable and the meter does
  not ship with one.

## 8. The confirmed product metric set (the maintainer, 2026-08-19)

Per game:

1. **DTW%** with 89% interval, plus the deserved margin (v1.3,
   unchanged).
2. **Verdict label** — three buckets: clear flip (DTW% definition, 195
   games in-window) / too close to call (DTW% 0.40–0.60, 186) /
   scoreboard holds. Band sweep 0.35–0.65 ships as a robustness
   display, validating — not gating — the choice.
3. **Placement meter** — placement points differential with 89% chance
   band; per-team cell breakdown available. "Luck" in copy only if M-3
   passes.
4. **Luck ledger waterfall** (avenue 2) and **bootstrap distribution
   plot** (avenue 1), unchanged from their approved briefs.
5. **Sidebar** — overtime-toss annotation only. The red-zone/late-down
   half of avenue 4 is absorbed into item 3.

Nothing on the page is ever summed across meters.

## 9. Disposed designs

- **Design C — league-conditional red-zone pricing** (realized points
  per red-zone trip minus league E[points | entry state], the
  "licensed part" of DQW%). **Parked**, the maintainer 2026-08-19: back burner
  for later investigation. Known scar tissue: the red-zone-only DQW%
  arm still degraded the rematch predictor (+0.0067 log loss), and the
  baseline inherits the depth-insufficiency defect.
- **Design D — drive-vs-play efficiency decomposition** (points per
  drive vs EPA per play; brief seed 3). **Refused with the arithmetic
  attached**: it carries no luck license — a full pre-registered
  persistence round would be needed before the product could say
  "luck" — and it measures the same channel as the placement meter
  with strictly worse conditioning (drive grain vs play grain, no
  own-game baseline). A second REJECT would cost a round and buy
  nothing the meter doesn't already cover.

## 10. What the implementation round must do, in order

1. Commit its pre-registration: constraint candidate set, PIT
   tolerance and adoption rule, M-3/M-4 thresholds — each with its
   power measured first.
2. Build the score and the band (TDD; this is production code).
3. Run M-1 → M-2 → M-3 → M-4 → M-5 (→ M-6 if powered), reporting after
   each gate before starting the next.
4. Only then touch presentation: the meter joins avenues 1 and 2 in
   the product round, with `dataviz` loaded before any chart code.

Example games carry over from doc 33 §6: `2018_05_GB_DET` (luck-heavy),
`2021_14_LV_KC` (degenerate), `2025_17_DET_MIN` (mid-case).

## 11. Known-defect register

| Defect | Evidence | Status |
|---|---|---|
| The late-down cell prices an EPA gap where S2's verdict was on success rate | §3 | **Open, by design.** M-3 re-licenses the shipped score itself |
| Cross-cell exchangeability is imperfect | Red-zone EPA variance 2.250 vs 1.916 (doc 08 §2) | **Open.** The constraint ladder + M-2 calibration exist for exactly this |
| Cell counts are endogenous to quality | Good offenses take more red-zone snaps; n_cell scales the score | **Open.** M-4 secondary read watches dispersion vs quality |
| The band could be read as an interval on an estimate | §4 | **Accepted, addressed in copy** — the score is exact; the band is a chance reference |
| M-6 may be unpowered at the meter's real magnitude | DQW% red-zone arm partly failed on interval width | **Anticipated.** Power check runs first; descriptive fallback pre-registered |
| No design in this document has run against data | Whole document | **By construction** — brainstorm output; the implementation round owns the pre-registration and the numbers |
