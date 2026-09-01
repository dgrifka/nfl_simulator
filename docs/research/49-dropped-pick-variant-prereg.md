# 49 — The dropped-pick variant ledger, pre-registered

*Written 2026-08-27 in a Fable 5 brainstorm, **before any code**. The maintainer's
decision after reading the dropped-pick docx and documents 43–48: build a
**labelled variant** of the adjudication that neutralizes interceptable
throws at the throwing defence's posterior-sampled catch probability, beside
— never instead of — the v1.3 ledger. This document is the change proposal
per the `bayesian-model-writeup` template and the correctness gate for the
build. It is not an amendment to Gate A: document 32's closure stands for
the official ledger, and the variant is recorded as such so no reader
mistakes it for a component with a branch point.*

*Inputs: documents 05 (§3 treatment table, §4 two-layer bootstrap), 09 (the
shrunk-rate treatment on a C-2 fail), 21/28 (why this is a variant and not a
row), 32, 43–48 (the study), 31 (v1.3's replay checks), 33 (the magnitude
audit whose form Part C copies).*

---

## 0. What is being built, in plain words

For every charted interception-worthy throw in a 2022+ game, the variant
asks: *given this throw and this defence-season, how likely was a pick?* —
drawing that probability from the round-2 model's posterior in every
replicate — and replaces what happened with a fair flip at that probability,
exactly as v1.3 does for a fumble. The offence is the charged entity; a
dropped pick is fortune to it only to the extent the defence could not have
been expected to catch it. Uncertainty about the defence's hands widens the
deserve-to-win distribution instead of being ignored.

**Why a variant and not a row.** The defender's hands are football. Document
28's consistency argument means admitting this event admits receiver drops
too. Coverage starts in 2022, so 2016–2021 games could never carry it. All
three stand; the variant exists so the maintainer can see the thing on real games
before deciding whether amendment A-3 (the hands-on-the-ball class) is
written. **The variant changes no v1.3 number.**

## 1. Tier declaration

**Model change.** A new component with its own generative story enters the
simulator behind a default-off switch. Every section fills.

## 2. DAG

v1.3, unchanged: `fumble → retention coin`, `FG/XP → make coin (kicker
shrunk)`, coins → bootstrap → margin distribution.

Variant adds one branch class:

```
throw covariates X_i ─┐
defence-season u_d ───┼──► p_i = logit⁻¹(α + X_i β + u_d)  ──► catch coin ──► EPA swing_i
(posterior draws)  ───┘          (draws over α, β, u_d)          (bin table, doc 47 §3)
```

Excluded on purpose: the QB-season term `v_q` (the offence's own
droppability is the offence's, so it stays in `core` — using it would
credit a QB for throwing catchable picks) and the game term `w_g` (document
48: 8.5 pp of unexplained within-game correlation with no team owner;
excluding it treats that variance as luck, and this is disclosed).

## 3. Mechanism story

- **Observation addressed.** Document 48: the defence-season finish spread is
  5.8 pp and does not carry across seasons; per throw, the defence's
  persistent skill explains ~1.4% of the variance in whether it is caught.
  A dropped pick is therefore ~99% fortune at the game grain by the
  simulator's own partial-neutralization arithmetic — but a *good* defence-
  season's shrunk rate is still the honest expectation for its throws, which
  is document 09's Gate C-2-fail treatment.
- **Why this mechanism.** It is the fumble template with a per-event
  probability instead of a class rate, and the FG template's posterior-draw
  plumbing (`LuckEvent.expected_draws`, `_resample`). Nothing new in the
  bootstrap; `bootstrap_margins` is untouched.
- **What would make it fail.** (a) The round-trip identity — ledger sum ≠
  margin shift — which the correctness gate pins. (b) Leakage: the model is
  fit on 2022–2025 including the adjudicated game; a defence-season's `u_d`
  contains this game's ~3 throws out of ~22. Accepted as the same convention
  every v1.3 baseline uses (league rates fit on all seasons), disclosed in
  §8. (c) Coverage: any 2022+ game with no FTN rows must produce a variant
  identical to v1.3, not an error.

## 4. Data

- Fit frame: document 45's floorless frame — 2,969 throws, 128 defence-
  seasons, 280 QB-seasons (2,997 worthy less 28 with a null covariate).
  Guards as document 43 §4.
- Adjudication frame per game: every charted worthy throw in that game,
  including the null-covariate ones, with nulls at the reference level and
  a flag (document 48 §6's rule).
- Swing per throw: document 47 §3's six-cell bin table (`yardline_100`
  thirds × down {1–2, 3–4}), recomputed in `src/` from the same data; all
  six cells cleared the 30-per-branch floor in round 3, every swing negative.

## 5. Model, inference, compute

**Fit** (`research/67_dropped_pick_model.py`, mirroring `42_fg_refit.py`):
document 43 §5's arm 2 verbatim at A-2's spec (4 × 2,000 / 2,000 / 0.9,
nutpie), **without** `w_g`. Saves `trace_dropped_pick.nc` +
`dropped_pick_summary.json` beside the FG trace, gitignored, regenerable.
One fit, ~2 minutes.

**Read side** (`src/nfl_simulator/dropped_picks.py`, new): `DroppedPickModel`
loaded from the trace, with `catch_probability(defence_season, X_row) ->
draws` = `logit⁻¹(α_s + X_row β_s + u_d[s])` across posterior draws `s`, and
a defence-season absent from the fit (impossible in 2022–2025, guarded)
falling back to `u_d = 0`. Standardisation constants and the reference
levels come from the summary JSON — **stored at fit time, never recomputed
on read** (round 3's surprise 4).

**Event builder** (`simulator.dropped_pick_events`): one `LuckEvent` per
worthy throw, `component="dropped_pick"`, `event_class` = the swing bin
label, `charged_team = posteam`, `actual = 1.0 if escaped else 0.0`,
`expected_draws = 1 − catch draws` (resampled to `n_posterior_draws`),
`swing = |swing_bin| × home_sign`. Signs follow the fumble builder exactly:
`actual` is the good branch for the charged team, `swing` its EPA value.

**Switch:** `simulate_game(..., dropped_pick_model=None)`; `None` (default)
is v1.3 byte-for-byte. `SimulationResult` gains `variant: str` —
`"v1.3"` or `"v1.3+dp"` — and nothing else changes shape.

**Compute:** the fit ~2 min; a full 2022–2025 pass (~1,139 games × two
runs) is v1.3's existing cost plus the event count, minutes.

**Long-fit downtime plan.** None needed; stated.

## 6. Pre-registered correctness gate (the ship gate)

Committed before code. All must pass; the audit in §7 is reported, not
gated.

| Gate | Statement | Bar |
|---|---|---|
| **V-1 default-off** | With `dropped_pick_model=None`, every 2016–2025 game reproduces v1.3 | max \|Δ deserved margin\| **0.00e+00** over 2,761 games (document 31's replay) |
| **V-2 round trip** | Variant ledger sum × `points_per_epa` equals the variant margin shift | ≤ 1e-9 per game |
| **V-3 no-throw identity** | A 2022+ game with zero worthy throws gives variant == v1.3 | exact, and the case is a test |
| **V-4 pre-2022** | A pre-2022 game asked for the variant returns v1.3 with `variant="v1.3"` and a warning, never an error | test |
| **V-5 expected draws** | Each event's `expected_draws.mean()` equals the ledger's `expected` and lies in [0, 1] | existing `LedgerEntry` check, plus a test on the builder |
| **V-6 sampler** | Gate C-1 bars on the fit, every parameter | 0 divergences, r̂ < 1.01, ESS > 400 |
| **V-7 sign** | A dropped pick books positive `luck_epa` to the offence in home perspective; a pick on a low-`p̂` throw books negative | two hand-built plays as tests |
| **V-8 posterior spread** | For the five worst and five best defence-seasons by `u_d`, the 89% interval of catch probability on a median throw is printed; each must lie inside [0.30, 0.70] | sanity bound, stop-and-ask if violated |

## 7. Magnitude audit (reported, not gated) — document 33's form

Over every 2022–2025 game, v1.3 versus variant:

- games with ≥ 1 dropped-pick event, and events per game (median, max);
- deserved-winner flips (element-wise on the label sets — document 33's
  lesson, never by subtracting totals): v1.3 → variant, by verdict bucket
  (clear flip / too close / scoreboard holds);
- median and 89% interval of `|ΔDTW|` and of `|Δ deserved margin|`, and the
  same restricted to games with ≥ 1 event;
- width of the DTW 89% interval, v1.3 vs variant, on affected games (the
  widening the posterior draws are expected to produce);
- the five largest `|Δ deserved margin|` games as ledger rows;
- `2025_17_DET_MIN`, `2025_13_DEN_WAS`, `2022_13_WAS_NYG` in full: both
  DTW%, both margins, the variant's dropped-pick rows.

**Pre-committed expectation:** ~69% of game-teams have ≥ 1 worthy throw
(document 48), so most games gain events; median `|ΔDTW|` on affected games
lands in low single digits of a percentage point; flips are a small
minority. If median `|ΔDTW|` exceeds 10 pp the component is doing more than
its 1.4% skill share suggests and the sign convention is the first suspect.

## 8. Kill and rollback

Default-off flag; the branch keeps the code; the audit is the record. No
version bump — v1.3 is unchanged by construction (V-1). The variant is
recorded in document 05 §3's treatment table as **`variant (no branch
point; A-3 pending)`**. Whether it becomes a row is the maintainer's decision after
reading §7, and would be written as amendment A-3 in its own document.
Nothing renders on the product figures in this round.

## 9. Register

| Item | Status |
|---|---|
| Game effect (8.5 pp) treated as luck | Disclosed, excluded from `p_i` |
| In-sample `u_d` (the game's own throws in its defence-season) | Accepted, v1.3 convention; ~3 of ~22 throws |
| 2016–2021 coverage | None; the variant is a 2022+ object |
| Receiver-drop symmetry (document 28) | Not built; the reason this is a variant |

---

## 10. Outcome

Round 4 ran 2026-08-27 on `feat/dropped-pick-variant`, off
`docs/dropped-pick-confounds`. Full record: document 50. **Unmerged.**

| Part | Result | Commit |
|---|---|---|
| A — the fit and its artifact | `trace_dropped_pick.nc` + `dropped_pick_summary.json` written. **V-6 PASS** (0 divergences, max r̂ 1.0070 on `sigma_q`, min ESS 587/522 over 429 parameters). **V-8 FAIL** — 2022 NYG's 89% interval [0.289, 0.509] against §6's [0.30, 0.70], 1.1 pp outside, 9 of 10 inside | `795f07b` |
| B — the component, TDD | `dropped_picks.py`, `simulator.dropped_pick_events`, the default-off switch, `SimulationResult.variant`. V-2, V-3, V-4, V-5, V-7 all PASS as tests; 502 → 525 | `7ad17be` |
| C — V-1 and the audit | **V-1 PASS exactly**: 2,761 games, max \|Δ deserved margin\| 0.00e+00, and 0.00e+00 on DTW% and both bounds. V-2 over 1,139 variant games: 0.00e+00. Read-side round trip 0.00e+00 over 2,969 rows. §7's audit written, **provisional on V-8's ruling** | `2cf395b` |
| D — the record | Document 50, this section, document 05 §3's row, and the queue | this commit |

**The round is stopped at §6's V-8 stop-and-ask, as pre-registered.** Document
50 §5a states the fork; neither option is taken by the session that found the
breach.

**Two things §6 left open, disclosed rather than resolved silently.**

1. **"A median throw" is undefined here.** Both readings — column-wise median of
   the design matrix, and every covariate at its median with the derived column
   derived from it — were computed. They breach identically, so the construction
   choice is not the cause and no choice was needed. A future gate quoted on a
   reference row should write the row down.
2. **§7's pre-committed expectation held; the tail was not pre-committed.**
   Median \|ΔDTW\| on affected games is 1.62 pp, inside "low single digits", and
   the 10 pp tripwire that would have made the sign convention the first suspect
   did not fire. The 89% upper bound is 27.4 pp and the largest margin move is
   13.36 points, which §7 did not anticipate in either direction. Document 50 §4
   shows it is the component's own arithmetic, and the sign convention was
   independently checked (V-7, the read-side round trip, the swing-table
   reproduction).

### Ruling R-3 on V-8, 2026-08-27

**R-3 (document 52 §5): the 2022 NYG breach of 1.1 pp on one of ten lines is
immaterial; the bound stands unamended.**

This is a ruling on §6's V-8 stop-and-ask, made by the maintainer in the document 52
brainstorm after reading document 50, in the same shape as ruling R-1 on the
cross-check tolerance: the gate's text is not edited, the breach is disclosed,
and the reason is on the record (document 50 §2 — a clean sampler, `sigma_d`
reproducing round 3, and one interval's lower tail on the most extreme of 128
levels). The first option in document 50 §5a is the one taken; the second — an
amendment restating V-8 on the point estimate — is not written. Round 4 is
unblocked and its audit is no longer provisional.

