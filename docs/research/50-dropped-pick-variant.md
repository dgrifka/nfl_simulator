# 50 — The dropped-pick variant ledger: built, and stopped at V-8

*Written 2026-08-27. The results record for round 4 of the dropped-pick study,
pre-registered in document 49 and run on the branch `feat/dropped-pick-variant`
in four parts: the fit (`research/67_dropped_pick_model.py`), the component
(`src/nfl_simulator/dropped_picks.py` and `simulator.dropped_pick_events`), the
V-1 replay and the magnitude audit
(`research/68_dropped_pick_variant_audit.py`), and this record.*

*Inputs: documents 49 (the pre-registration and the gate), 43/45/46/47/48 (the
study), 32 (the closure the variant does not touch), 33 (the audit's form and
its element-wise lesson), 05 §3 (the treatment table), 31 (v1.3's replay
checks).*

---

## 1. The answer, stated first — v1.3 is untouched, and the fit failed its last gate

**Gate V-1, verbatim from the run:**

```
V-1 replay: 2,761 games, max |Δ deserved margin| 0.00e+00
  and on the rest of the summary: |Δ DTW%| 0.00e+00, |Δ interval| 0.00e+00  -> PASS
```

Every 2016–2025 game re-simulated with `dropped_pick_model=None` reproduces
`dtw_games_v13.parquet` exactly — not to a tolerance, to zero, on the deserved
margin, the DTW% and both interval bounds. The replay ran on the **wide**
play-by-play frame the variant itself needs, so the eight extra covariate
columns are proven inert rather than assumed to be. Document 49 §8's claim that
the variant needs no version bump is now a measured fact.

**And the fit did not clear V-8.** The 89% catch-probability interval on a median
throw for the worst defence-season in the sample, 2022 NYG, is **[0.289, 0.509]**
against a pre-registered bound of **[0.30, 0.70]** — 1.1 pp outside, on the lower
end, on **one of the ten** defence-seasons the gate prints. Document 49 §6 makes
that a stop-and-ask, and it is stopped: **document 49 §7's audit below is
provisional on the maintainer's ruling, and no ledger, figure or published number moves
either way.**

The component itself is built, tested and switched off. Everything in §3 except
V-8 passed.

## 2. The fit

`research/67_dropped_pick_model.py` fits document 43 §5's arm 2 at amendment
A-2's spec — 4 × 2,000 draws after 2,000 tuning, `target_accept` 0.9, nutpie,
seed `20260827` — **without** document 47 §2's game effect `w_g`, on the
floorless 2,969-row frame: 128 defence-seasons, 280 QB-seasons, 8,000 posterior
draws.

| Quantity | Value |
|---|---|
| `alpha` (posterior mean) | +0.0699 |
| `sigma_d` (defence-season) | 0.2544 on the logit scale ≈ 6.4 pp |
| `sigma_q` (QB-season) | 0.2053 on the logit scale ≈ 5.1 pp |
| Largest covariate effects | `is_catchable_ball` −0.489, `is_contested_ball` −0.321, `air_yards_z` +0.316, `qb_hit` +0.212 |

`sigma_d` sits where round 3 left it (5.83 pp with the game effect in, 6.35 pp
without), so removing `w_g` did not move the defence-season spread anywhere new.

**The QB-season term is fitted and never read.** Document 49 §2's reason, and
the read side honours it: `catch_probability` is
`logit⁻¹(α + Xβ + u_d)` with no `v_q` in it. A passer's own droppability belongs
to the offence, and paying it here would credit a quarterback for throwing
catchable interceptions.

**Gate V-6 — PASS.** 0 divergences; max r̂ **1.0070** on `sigma_q`; min ESS
**587** bulk / **522** tail; 429 parameters checked, 0 over the r̂ bar and 0
under an ESS bar. Amendment A-2's longer chains fixed round 1's `sigma_q`
failure and they hold here.

**Gate V-8 — FAIL, on one defence-season.** Document 49 §6 says "a median
throw" and does not define it, and the two readings of that phrase are not the
same row, so **both were computed and reported** rather than one being chosen:

- *column-wise median* — every design column at its own median. `air_yards_z_squared`
  then takes the median of the squared column (≈0.45), so the row describes no
  actual throw.
- *consistent median* — every covariate at its median, with the derived column
  derived from it, so `air_yards_z_squared` is the square of the median. This is
  a throw that could have happened.

| Reading | League p(catch) at `u_d` = 0 | 2022 NYG's 89% interval | Verdict |
|---|---|---|---|
| column-wise median | 0.506 | [0.289, 0.509] | FAIL |
| consistent median | 0.508 | [0.290, 0.511] | FAIL |

They breach identically, so the construction choice is **not** the cause and the
ambiguity is not the story. Nine of ten intervals are inside the bound under
both readings; the five best defence-seasons run 0.46–0.67 and the other four
worst run 0.313–0.535. Picking whichever reading passed would have been the
goalpost move documents 04 and 05 §7 wrote the power-first law to prevent, which
is why the script stops on a breach under *either*.

**What the breach probably is.** The bound was pre-registered as a *sanity*
bound — "a defence-season whose median-throw catch probability leaves this band
is a fitting artifact, not a defence". The fit is not behaving like an artifact:
the sampler is clean, `sigma_d` reproduces round 3, and 2022 NYG's point
estimate is 0.404, comfortably inside. What leaves the band is the **lower tail
of one 89% interval on the most extreme of 128 levels** — which is what a
posterior with `sigma_d` ≈ 6.4 pp and ~22 chances per level is expected to
produce. Read that way the bound was set on the point estimate's scale and
applied to an interval's, and 0.30 is roughly 1.6 `sigma_d` below the league
rate of 0.49. That reading is offered, not adopted: §5 leaves it to the maintainer.

**Ruled, 2026-08-27.** *R-3 (document 52 §5): the 2022 NYG breach of 1.1 pp on
one of ten lines is immaterial; the bound stands unamended.* the maintainer's ruling in
the document 52 brainstorm, taken after reading this document. The consequence
is recorded rather than assumed: §4's audit is **no longer provisional** — it is
round 4's result, read on the fit above — and §5's item 1 is closed. Nothing
else in §5 moves: amendment A-3 still needs its gates (document 52 §5, run in
round 5) and v1.3 is still the published adjudication.

## 3. The gate table, every row

| Gate | Statement | Bar | Result |
|---|---|---|---|
| **V-1** default-off | `dropped_pick_model=None` reproduces v1.3 | max \|Δ\| **0.00e+00** over 2,761 games | **PASS** — 0.00e+00 on margin, DTW% and both bounds |
| **V-2** round trip | variant ledger sum × `points_per_epa` = variant margin shift | ≤ 1e-9 per game | **PASS** — max residual 0.00e+00 over 1,139 games, and a unit test on a three-event game |
| **V-3** no-throw identity | a 2022+ game with zero worthy throws gives variant == v1.3 | exact, and a test | **PASS** — field for field, `variant == "v1.3"` |
| **V-4** pre-2022 | a pre-2022 game returns v1.3 with a warning, never an error | test | **PASS** — `UserWarning`, no events, identical ledger |
| **V-5** expected draws | each event's `expected_draws.mean()` equals the ledger's `expected`, in [0, 1] | test | **PASS** — and draw count equals `n_posterior_draws` |
| **V-6** sampler | Gate C-1's bars, every parameter | 0 divergences, r̂ < 1.01, ESS > 400 | **PASS** — 0, 1.0070, 587/522 over 429 parameters |
| **V-7** sign | a dropped pick books positive `luck_epa` to the offence in home perspective; a pick on a low-`p̂` throw books negative | two hand-built plays | **PASS** — plus a third test on the away-offence sign flip |
| **V-8** posterior spread | the ten printed intervals lie inside [0.30, 0.70] | sanity bound, stop-and-ask | **FAIL** — 2022 NYG [0.289, 0.509] under both readings; 9 of 10 inside |

**Two checks not in document 49 §6, run anyway.**

- **The read-side round trip.** Over the 2,969 fitted rows,
  `DroppedPickModel.catch_probability` reproduces the posterior's own arithmetic
  to **0.00e+00**. Document 31 §7 put this on every ship template after the same
  formality found a real field-goal defect (document 30); it finds nothing here.
- **The swing table against round 3's.** The `src/` builder document 49 §4 asked
  for reproduces `research/65`'s cell for cell, **max \|Δ\| 0.00e+00**, pooled
  fallback −3.55 EPA. All six cells carry their own difference; none falls back.

| Cell | 1–2 down | 3–4 down |
|---|---|---|
| 1–33 yd | −5.04 | −3.01 |
| 34–66 yd | −3.82 | −2.27 |
| 67–99 yd | −4.03 | −2.49 |

## 4. The audit — provisional, and it says something

Document 49 §7's bullets over all **1,139** scheduled 2022–2025 games, v1.3
against the variant, same seed, same frame, one model switched on. Descriptive,
never a gate, and **provisional while V-8 is unruled**.

**Coverage.** 1,033 of 1,139 games (**90.7%**) carry at least one dropped-pick
row; 2,997 events in total — every charted interception-worthy throw in the four
seasons, including the 28 whose covariates were priced at their reference level.
Median 3 events per affected game, max 12. That 90.7% is what document 48's
"68.9% of game-teams threw at least one interceptable pass" looks like when both
teams get a turn.

**Movement.**

| Population | median \|ΔDTW\| | 89% interval | median \|Δ margin\| | 89% interval |
|---|---|---|---|---|
| all 1,139 games | 0.81 pp | [0.00, 26.24] | 1.57 pt | [0.00, 4.86] |
| 1,033 affected | 1.62 pp | [0.00, 27.35] | 1.73 pt | [0.24, 4.99] |

Document 49 §7's pre-committed expectation was "median \|ΔDTW\| on affected
games lands in low single digits of a percentage point", with 10 pp as the
tripwire that would make the sign convention the first suspect. **1.62 pp: the
expectation held and the tripwire did not fire.**

**The tail is a different story, and it is the interesting part.** The 89% upper
bound is 27 pp, the maximum is 89.3 pp, and the largest margin move is 13.36
points. That is not a defect — it is arithmetic the component cannot avoid. Each
worthy throw is neutralized at roughly a coin flip, so a single escape books
about half its bin's swing, ≈ +2 EPA, and a game where one side escapes five and
the other is picked three times banks ≈ 16 EPA ≈ 13 points before anything else
happens. **A component priced at 50/50 on an event that happens three times a
game is loud by construction**, and document 49 §3's own arithmetic — a dropped
pick is ~99% fortune at the game grain — is what makes it so.

**Verdict flips, element-wise.** Document 33's lesson applied: the two label
sets are compared game by game, never by subtracting totals.

| Bucket (document 33 §2a) | v1.3 | variant |
|---|---|---|
| clear flip | 92 | 112 |
| too close to call | 80 | 104 |
| scoreboard holds | 967 | 923 |

**137 games (12.03%) change bucket.** The transitions, largest first: `scoreboard
holds → too close` 55, `too close → scoreboard holds` 26, `too close → clear
flip` 21, `scoreboard holds → clear flip` 17, `clear flip → too close` 16, `clear
flip → scoreboard holds` 2. The net movement is toward *less* certainty — 44
games leave "scoreboard holds" — which is what adding a component with real
posterior width should do.

And the trap document 33 §2a fell into, avoided out loud: on the sign
definition v1.3 flips 116 games and the variant flips 150, a net of **+34**, but
they **disagree on 68 games**. The net is not the disagreement set and never was.

**Interval widening.** On affected games the DTW 89% interval widens from a mean
of 0.0383 to 0.0511 (median 0.0112 → 0.0301), a mean widening of **+0.0128**,
wider in 52.0% of them. This is the effect the posterior draws exist to produce:
uncertainty about the defence's hands is carried into the verdict instead of
being ignored.

**The five largest movers.**

| Game | actual | deserved v1.3 → variant | DTW% v1.3 → variant | events |
|---|---|---|---|---|
| `2022_07_PIT_MIA` | +6 | +5.24 → −8.12 | 96.5 → 7.3 | 8 |
| `2022_14_BAL_PIT` | −2 | +1.79 → +11.06 | 72.2 → 98.8 | 6 |
| `2022_02_LAC_KC` | +3 | +2.81 → −5.90 | 94.4 → 10.6 | 8 |
| `2025_07_HOU_SEA` | +8 | +11.63 → +19.49 | 100.0 → 100.0 | 7 |
| `2022_02_TB_NO` | −10 | −11.03 → −3.28 | 0.9 → 29.3 | 8 |

`2022_07_PIT_MIA` is the shape to look at, because it is the component's
argument and its problem in one game. Miami threw five interceptable passes and
got away with all five; Pittsburgh threw three and was picked on all three.
Every one of the eight rows books fortune to Miami, ≈ +15.9 EPA, ≈ 13.4 points —
so a six-point Miami win becomes an eight-point deserved loss. Whether that is
an insight or an overreach is exactly what amendment A-3 has to decide, and it
is not decided here.

### 4a. The three named games, as worked examples

**`2025_17_DET_MIN` — the variant flips a game v1.3 called too close.**
Minnesota, at home, won by 13; v1.3 had already removed almost all of that,
putting the deserved margin at +0.70 with DTW% 54.8% [49.0, 59.9] — *too close
to call*. The variant finds two interceptable Detroit throws and **both were
intercepted**, so this game has no dropped picks at all: round 3's log flagged
exactly that. What the variant charges is the other direction — Minnesota caught
two balls it was 68.2% and 53.1% likely to catch, so the fortune booked is
modest, +0.79 and +1.89 EPA to Minnesota. Together that is 2.25 points off
Minnesota's deserved margin: −1.55, DTW% 37.7% [32.1, 43.4], a *clear flip* to
Detroit. A game can move a bucket on two throws that went exactly the way the
model expected.

**`2025_13_DEN_WAS` — the variant softens a verdict without changing it.**
Washington lost by 1. v1.3 says deserved −3.32, DTW% 14.5% [11.9, 17.1]; the
variant says −2.44 and 28.9% [23.8, 34.2]. Two throws, both intercepted, one
each way — Washington's from 1–33 yards at a 46.3% escape probability (−2.33
EPA to Washington) and Denver's on a late down at 51.8% (+1.29 EPA to
Washington in home perspective). The bucket does not move: *scoreboard holds*
both times. What moves is the interval, from 5.2 pp wide to 10.4 pp.

**`2022_13_WAS_NYG` — the variant hardens a flip.** The game was a **tie**.
v1.3 already calls it a *clear flip* at deserved +4.20 and DTW% 80.7% [76.9,
84.8]; the variant pushes it to +10.95 and 95.3% [92.8, 97.5]. Eight events,
seven of them Washington throws that **escaped** — at escape probabilities from
50.3% up to 80.6% — so roughly 8 EPA of good fortune is booked to Washington and
the Giants' deserved margin rises by the difference. This is the tail's
mechanism in a game where the scoreboard gave no answer at all: nothing unusual
happened on any single throw, and eight of them in one direction is worth nearly
seven points.

## 5. What these numbers do and do not license

**They license nothing yet.** Three things stand between the audit and any use
of it, and only the maintainer can move the first two.

1. **V-8's breach is unruled.** The audit above reads a fit whose last gate
   failed. §2 offers a reading — the bound was written on a point estimate's
   scale and applied to an interval's — but document 49 §6 called V-8 a
   stop-and-ask precisely so that reading is not made by the session that wants
   to keep going. Options are in §5a.
2. **Amendment A-3 is the maintainer's decision, unchanged.** Whether the hands-on-the-ball
   class becomes a *row* rather than a variant is document 49 §8's question and
   it needs its own document. Document 28's consistency argument — admitting this
   event admits receiver drops — is untouched by anything measured here, and so
   is the 2016–2021 coverage gap.
3. **v1.3 is the published adjudication and remains so.** No figure, artifact or
   public number changes in this round. Document 32's closure stands; the
   treatment table's new row (document 05 §3) says `variant`, not a treatment.

**What the numbers do establish, ruling or no ruling:** the component is
implementable inside the existing bootstrap with no change to
`bootstrap_margins`; the default-off switch is exact to 0.00e+00 over 2,761
games; the read side prices what the model fitted; and the variant is **loud** —
it moves 12% of games across a verdict bucket and moves the deserved margin by
13 points in the worst case, on a component whose persistent skill share
document 48 measured at ~1.4%. That last pairing is the finding of this round,
and it is the honest input to A-3.

### 5a. The fork on V-8

Neither of these is chosen here.

- **Rule the breach immaterial and read the audit as final.** The reasoning is
  in §2: the sampler is clean, `sigma_d` reproduces round 3, and one interval's
  lower tail on the most extreme of 128 levels is what a healthy posterior of
  this width produces. Cost: minutes. It would be recorded as a ruling on
  document 49 §6, in document 49 §10, with the 1.1 pp disclosed — the same shape
  as ruling R-1 on the cross-check tolerance.
- **Restate V-8 as a bound on the point estimate and re-derive the interval
  bound from `sigma_d`.** This is a *change to a pre-registered gate* and would
  need to be written as an amendment before it is applied, not after. Cost:
  under an hour, no refit — the trace and summary are on disk.

## 6. Register

| Item | Status |
|---|---|
| V-8 breached on 2022 NYG by 1.1 pp | **Closed — ruling R-3** (document 52 §5, recorded in §2): immaterial, the bound stands unamended. §4 is no longer provisional |
| "A median throw" was undefined in document 49 §6 | **Resolved by reporting both readings.** They agree on the verdict, so no choice was needed |
| Game effect (8.5 pp) treated as luck | Disclosed, excluded from `p_i` — document 49 §2 and §9, unchanged |
| In-sample `u_d` (the game's own ~3 of ~22 throws) | Accepted, v1.3's convention — document 49 §9, unchanged |
| 2016–2021 coverage | None; the variant is a 2022+ object — document 49 §9, unchanged |
| Receiver-drop symmetry (document 28) | Not built; the reason this is a variant — document 49 §9, unchanged |
| The variant is loud relative to its ~1.4% skill share | **New, and the round's finding.** §4's tail; the sign convention was checked and is not the cause (V-7, plus the round-trip and swing-table checks) |
| `build_swing_table` landed in Part A's commit, not Part B's | **Disclosed deviation.** Document 49 §4 requires the table to be recomputed in `src/` and the fit's summary carries it, so the builder had to exist before the fit ran. TDD order preserved: the tests were written and watched fail first |
| Part D was written with V-8 unruled | **Deliberate.** A round that stops is still a round that happened; leaving it undocumented would be the worse failure |

## 7. Commits

Branch `feat/dropped-pick-variant`, off `docs/dropped-pick-confounds`. **Unmerged;
the maintainer merges.**

| Part | Commit | What |
|---|---|---|
| A | `795f07b` | `research/67_dropped_pick_model.py`, the fit; V-6 PASS, V-8 FAIL; `build_swing_table` and its tests |
| B | `7ad17be` | `dropped_picks.py`, `simulator.dropped_pick_events`, the switch, `render._simulation_context`; 502 → 525 tests |
| C | `2cf395b` | `research/68_dropped_pick_variant_audit.py`; V-1 0.00e+00, V-2 0.00e+00, the read-side round trip, the audit |
| D | this commit | This document, document 05 §3's row, document 49 §10, and the queue |
