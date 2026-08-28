# 56 — The receiver-drop mirror, pre-registered (A-3 gate G-4)

*Written 2026-08-27 in a Fable 5 brainstorm after document 55, **before any
fit**. Amendment A-3 clause 3 makes defender drops and receiver drops one
class; this is the study and component for the receiver side, compressed
into one round because rounds 2–4 already settled the shape. It reuses
document 43's study design, document 49's component design and document
52's gates by reference, and states only what differs.*

*Inputs: documents 09 (drops at team and receiver grain — the prior study),
43/45 (study design), 47 §3 (EPA pricing), 49 (component), 52 (A-3), 55
(G-1's bound).*

---

## 0. What differs from the dropped-pick side, stated first

| | Dropped picks | Receiver drops |
|---|---|---|
| Event | interception-worthy throw | catchable target (`is_catchable_ball`) |
| Outcome | intercepted / escaped | dropped (`is_drop`) / caught |
| League rate | 48.5% | **4.95%** (2,781 of 56,211, 2022–2025) |
| Charged entity | offence (via defence-season effect) | **offence, via receiver-season effect** `r_i` |
| Prior evidence | doc 32: r = +0.14 | doc 09: team spread **14.4% rel, powered (0.87)**; receiver grain 20.7% rel, **unpowered (0.37) under a 20-target floor** |
| Counterfactual EPA | six-cell bin table | **per play: `air_epa + xyac_epa`** (nflfastR's completion counterfactual); bin table only where null |

**A wording issue for the maintainer, flagged not resolved.** Document 52 §3's
preamble says the class covers a finish that is "near-random". A drop is a
1-in-20 event. The gates (clause 1, the powered spread; the skill share)
are what the amendment actually tests, and on those a drop qualifies more
easily than a pick — its persistent share of per-target variance is
~0.1–0.2%, against ~1.4% for picks. **Pre-committed reading:** the gates
govern; the preamble's "near-random" is read as "the entity's persistent
skill explains a small share of the outcome", and document 52 §3 gets a
one-line clarification to that effect **only if the maintainer rules so** after
this round. If he rules the other way, receiver drops fail the class on
wording, A-3 cannot be enacted for picks either (clause 3), and both stay
labelled variants.

**What a drop books.** With `p(catch) ≈ 0.95`, a drop is `(0.95 − 0) ×
swing` of bad fortune to the offence — roughly −1.5 to −2 EPA each — and
a catch is a small positive. Expect ~1.2 drops per team-game. Magnitude per
game is therefore comparable to picks, from many small credits and few
large debits.

## 1. Study (arms 1–3, as document 43 §5, floorless as document 45)

- **Frame:** every charted catchable target 2022–2025 (56,211; stop if
  ±2%), joined to pbp. Covariates, all pre-branch: `air_yards` (+ square),
  `pass_location`, `is_contested_ball`, `qb_hit`, `n_pass_rushers`,
  `is_qb_out_of_pocket`, `is_play_action`, `is_screen_pass`, `down`,
  `ydstogo`, `yardline_100`, `shotgun`, pre-snap `wp`. Excluded:
  `is_created_reception`, `complete_pass`, `epa`, `yac_*` realised values.
- **Model (arm 2):** `y_i ~ Bernoulli(p_i)`, `logit p_i = α + X_i β +
  r_s[i] + d_d[i]`, `r_s` receiver-season, `d_d` defence-season (the
  coverage's contribution to a drop is the defence's football and stays in
  `core`; it enters the model so `r_s` is not contaminated by schedule).
  Priors as document 43 §5; `σ_r, σ_d ~ HalfNormal(0.5)`; spec = document
  54 F-1 (4 × 4,000 / 4,000 / 0.95). No floor. Arm 2b drops
  `is_contested_ball`.
- **Gate arm (arm 3):** crossed grid on the residual, receiver-season ×
  defence-season; second grain **team-season** (offence) × defence-season.
- **Power first** (`research/71_receiver_drop_power.py`, document 43 §6's
  instrument, `DATASETS = 400`, seed `20260827`): null bound and power at
  5 / 12.5 / 25 / 50% relative of the **4.95%** rate (12.5% = 0.62 pp), at
  receiver-season, team-season and defence-season grains.

  **Computed 2026-08-27, before any verdict fit** (`research/71`, 400 datasets
  per cell, seed 20260827, guards exact at 56,211 / 2,781 / 128). The gate arm,
  on the conditioned residual:

  | Grain | Null bound (threshold) | 5% | 12.5% | 25% | 50% | Resolvable |
  |---|---|---|---|---|---|---|
  | receiver-season | 1.009 pp | 0.15 | **0.40** | 1.00 | 1.00 | **No** |
  | team-season | 0.634 pp | 0.19 | **0.88** | 1.00 | 1.00 | **Yes** |
  | defence-season | 0.632 pp | 0.26 | **0.88** | 1.00 | 1.00 | **Yes** |

  Gate D-1's power, on the raw drop-rate spreads (beta-binomial grid, arm 1's
  own instrument; the receiver row is the ≥ 100-target unit of this section):

  | Grain | Entities | Null bound | 5% | 12.5% | 25% | 50% | Resolvable |
  |---|---|---|---|---|---|---|---|
  | receiver-season | 67 | 1.540 pp | 0.13 | **0.20** | 0.65 | 1.00 | **No** |
  | team-season | 128 | 0.714 pp | 0.20 | **0.87** | 1.00 | 1.00 | **Yes** |
  | defence-season | 128 | 0.701 pp | 0.17 | **0.90** | 1.00 | 1.00 | **Yes** |

  This reproduces document 09 §4's ordering on a different instrument — team
  grain powered, receiver grain not — which §0's table anticipated at 0.87 and
  0.37. The receiver grain is thin rather than absent: 1,931 receiver-seasons
  share 54,160 modelled targets, a **median of 17 each**, against 422 per
  team-season.

  **The clause-1 grain rule below therefore fires on its second branch: only
  team-season clears C-3, so the component charges the team-season — the
  receiving corps — and says so.** Two consequences follow from the rule rather
  than from a choice made after seeing it: the component's entity effect
  `r[s]` in §2 is the **team-season** effect, so arm 2 is fitted a second time
  with a team-season term in place of the receiver-season one (clause 2 needs
  posterior draws over the charged entity, and a receiver-season fit cannot
  supply them for a corps); and §0's `r_i` notation is read as "the charged
  offensive entity", not specifically the individual receiver.
- **Gates C-1 / C-2 / C-3** as document 43 §7. **Clause-1 grain rule,
  pre-committed:** the component charges the **receiver-season** if that
  grain clears C-3; if only team-season clears, the component charges the
  team-season (the receiving corps) and says so; if neither, G-4 fails and
  A-3 is not enacted.
- **Hindsight probe:** `p(catchable | incomplete, is_drop)` is 1 by
  construction, so document 45's probe does not transfer. Substitute: the
  share of *completions* charted not-catchable (should be ≈ 0; report) and
  the drop rate on contested vs uncontested catchable balls (contested
  should be far higher; if not, `is_catchable_ball` is being graded off the
  outcome).

## 2. Component (as document 49, with these substitutions)

- `component = "receiver_drop"`, `charged_team = posteam`, `actual = 1.0 if
  caught else 0.0`, `expected_draws = catch draws` from
  `logit⁻¹(α_s + X_i β_s + r[s])` — **defence-season effect excluded on
  read** (it is the defence's play), disclosed as document 49 §2 disclosed
  `v_q`. `swing_i = |(air_epa_i + xyac_epa_i) − epa_incomplete_i|` where
  `epa_incomplete` is the play's realised EPA if it was a drop, else the
  bin-table incompletion mean for its cell; both `air_epa` and `xyac_epa`
  null → bin table (document 47 §3's form, six cells, min 30 per branch).
  Sign to home perspective as the fumble builder.
- Trace `trace_receiver_drop.nc` + summary; `ReceiverDropModel` in a new
  module `receiver_drops.py`, the same shape as `DroppedPickModel`; shared
  helpers may be factored out, `dropped_picks.py`'s behaviour unchanged.
- `simulate_game(..., receiver_drop_model=None)`; `variant` takes
  `"v1.3+dp"`, `"v1.3+rd"`, `"v1.3+dp+rd"` (= `"v2.0"`).
- **Correctness gate V-1..V-8 as document 49 §6**, with V-8's sanity bound
  on a median target at **[0.85, 0.99]** catch probability and V-7's two
  sign cases rewritten for a drop (a drop by the home offence books
  negative `luck_epa`; a catch on a low-`p̂` contested ball books positive).

## 3. Gates for this round

- **G-4a — the study:** C-3 ≥ 0.80 at the charged grain (clause 1).
- **G-4b — the component:** V-1..V-8.
- **G-4c — self-fulfilment:** document 54's 19 week-out folds on the
  receiver model; agreement ≥ 0.90 and median |ΔDTW| < 1.0 pp between arms
  (document 52 G-1's bars).
- **G-4d — materiality:** document 52 G-3's floor on the `+rd`-only variant.
- **G-5 — the combined audit (reported, not gated):** over 2022–2025,
  strict vs `+dp` vs `+rd` vs `+dp+rd`: bucket moves (element-wise, by
  bucket), median |ΔDTW| on affected games, interval widening; the share
  of games where the two directions move DTW the *same* way vs opposite
  ways; the three named games plus the five largest `+dp+rd` movers.
  **Pre-committed expectation:** the two directions are roughly
  independent per game, so `+dp+rd` moves more games than either alone
  and the share cancelling is near 50%; a strong positive correlation
  would mean both are picking up the same game-level thing (document 48's
  8.5 pp game effect) and is a finding to name.

**Enactment rule:** A-3 is enacted for the class if G-4a–d pass **and**
the maintainer rules on §0's wording. Then document 52 §4's two editions ship
(product rendering in a later figure round) and document 05 §3 carries
`partial (A-3)` on both rows.

## 4. Constants

| Constant | Value | Where |
|---|---|---|
| Catchable targets | 56,211 (±2%) | `71` |
| League drop rate | 0.0495 (±0.3 pp) | `71` |
| Relative scenarios / reference | (0.05, 0.125, 0.25, 0.50) / 0.125 | `71` |
| Sampler spec | doc 54 F-1 | `72` |
| Folds | doc 54: 19 | `74` |
| V-8 bound | [0.85, 0.99] on a median target | `72` |
| Swing source | `air_epa + xyac_epa` per play; bin fallback | `73` |
| Seeds | 20260827 fits (+week for folds), 20260817 simulator | as before |

---

## 5. Outcome

Round 7 ran 2026-08-27 on `feat/dropped-pick-variant`. Full record: **document
57**. **Unmerged.** 550 → 586 tests, ruff clean.

| Part | Result | Commit |
|---|---|---|
| A — power, then thresholds | Guards **exact** (56,211 / 2,781 / 128). §1's table filled above. **The clause-1 grain rule fired on its second branch**: C-3 is 0.400 at receiver-season and 0.877 at team-season, so the component charges the receiving corps. A third file was needed — §1's instrument costs 183.8 s per fit at the receiver grain, ~100 hours for this part; `research/_crossed_block_grid.py` computes the same posterior on the same grid in 0.17 s and reproduces it to 1.2e-15 | `048e614` |
| B — the study, arms 1–3 | Gate C-1 PASS on all three arms, 0 divergences, worst r̂ 1.0022. **G-4a PASS** (C-3 0.877). Gate **C-2 FAILS** at the charged grain, 0.87 pp against 0.63 pp, and is reportable. Like-grain cross-check 0.01 pp. **V-8 PASS**, both readings, both reference rows stored | `790cac0` |
| C — the component, TDD | `receiver_drops.py`, the default-off switch, composable `variant`. The defence-effect-excluded-on-read pin and the swing-sign guard are tests, as §2 and handoff constraint 3 asked | `109799f` |
| D — G-4b, G-4d, G-5 | **G-4b PASS**, **G-4d PASS** (2.32 pp vs 0.56 pp). V-1 0.00e+00 twice; read-side round trip 0.00e+00 over 54,160 rows | `e88a42f` |
| D — G-4c | **PASS.** 19/19 folds clear C-1; agreement 0.996, median \|ΔDTW\| 0.04 pp | `e774547` |
| E — the record | Document 57, this section, document 52 §8, document 05 §3's two rows, the queue, `results-2026-08-27-exp7.md` | this commit |

**§3's enactment rule is half satisfied.** G-4a–d all pass, so the computed
half is done; the other half is the maintainer's ruling on §0's wording, and no round
can supply it.

**§3's pre-committed expectation on G-5 was wrong in a nameable way.** It
expected the two directions to be roughly independent per game, with the
cancelling share near 50%. On the 758 games carrying both kinds of event they
move DTW the same way **64.6%** of the time, r = **+0.197** — a mild positive
dependence, which §3 said would be "a finding to name". Named in document 57
§5. It is far short of the two components reading one thing: `+dp+rd` moves 200
games where `+dp` moves 136 and `+rd` 162, and 70 games that one alone moves the
combined arm does not.

**Two things §0 and §2 left open, disclosed rather than resolved silently.**

1. **§0's magnitude expectation understated the receiver side.** §0 said
   "magnitude per game is therefore comparable to picks". The median `+rd` move
   is 2.32 pp against `+dp`'s 1.59 — **46% larger** — and the mean is 8.17 pp
   against 1.59. Document 57 §5a checks the tail end to end and finds it is the
   component's own arithmetic: a dropped touchdown prices at ~9 EPA, where the
   dropped-pick bin table caps near 3.55.
2. **§2's fallback clause is under-specified and was widened.** "Both `air_epa`
   and `xyac_epa` null → bin table" does not say what to do when only one is.
   The bin table is taken whenever either is missing — 2,038 rows both null,
   2,359 `xyac_epa` alone — which uses the pre-registered fallback more often
   and never invents a value.

**And one substitute fired.** §1's contested-ball probe expected contested balls
to be dropped "far higher" than uncontested ones. They are dropped **less**
often — 4.29% against 5.04%, 0.85× — which is what `is_catchable_ball` being
graded partly off the outcome would look like. Reported, never gated, and it
caveats every drop number in the round; it biases toward finding *less* receiver
skill, so the measured spread is a floor.
