# 47 — Dropped-pick study, round 3: rulings, game-clustering check, diagnostic

*Written 2026-08-27 in a Fable 5 brainstorm after reading
`results-2026-08-27-exp2.md` and document 46, **before any fit**. Two
rulings by the maintainer on round 2's stop-and-ask items, one robustness check on
the finding, and the reported diagnostic document 32 §4 sanctioned. Document
32's closure is untouched; nothing enters the ledger.*

*Inputs: documents 43, 45, 46 (the study so far), 32 §4 (where the
diagnostic lives), 05 §3 (reported separately, never as a ledger row).*

---

## 1. Rulings (the maintainer, 2026-08-27)

**R-1 — the cross-check compared unlike grains, and that was the author's
error.** Document 43 §5 asked arm 2's defence-*season* `σ_d` to agree within
1.0 pp with arm 3's `σ_d` at *both* grains, including the four-season pooled
one. Those are different quantities and document 46 §4 shows they genuinely
differ. Amended: the cross-check is **like-grain only** — arm 2's `σ_d`
against arm 3's defence-season × QB-season `σ_d`. At that grain round 2's
gap is 1.04 pp against 1.0 — at the line, within the study's known seed
wobble. **Ruling: C-1's cross-check half is recorded as PASS-at-tolerance
for round 2, with the 0.04 pp disclosed**, and arm 2's `β̂` is quotable.

**R-2 — wording (b).** Document 46 §7's option (b) replaces document 44 §7:

> *"Interceptions your opponent dropped are counted here as they happened.
> In any given season some defences finish more of their chances than others
> — but that edge doesn't carry to the next season, so it reads as a year,
> not a trait."*

Document 43 §7's decision rule ("season grain first, pooled only if it fails
C-3") is amended: **when both grains clear C-3, both are reported and the
wording must be consistent with both.** Avenue (3), reopening A-2, is closed
on the evidence (power at 5%-of-variance 0.20 / 0.30 against A-2's 0.80).

## 2. The robustness check — is "seasons differ" game clustering?

**Alternative hypothesis.** A defence-season's ~22 chances come from ~17
games. Throws inside one game share a quarterback, a weather, a game script
and a charter's mood. If conversion residuals are correlated within a game,
the defence-season spread of 5.95 pp can be inflated by *game* clustering
with no defensive contribution at all — which would produce exactly round
2's pattern (season-grain spread, no cross-season carry).

**Design.** Arm 2's PyMC model with one added node: `w_g ~ Normal(0, σ_g)`,
`g` = game (expected ~1,090 games with ≥ 1 worthy throw over 2022–2025),
non-centred, same spec as A-2 (4 × 2,000 / 2,000 / 0.9). Everything else
identical to round 2's arm 2. One fit. The crossed grid cannot take a third
factor, so this check runs in the confirmatory arm; it is disclosed that the
comparison below therefore mixes instruments.

**Statistic.** `σ_d` (defence-season) with the game effect, on the
probability scale, 89% interval; `σ_g` reported beside it.

**Pre-committed reading:**

| `σ_d` with game effect | Reading |
|---|---|
| 89% **upper** bound ≥ 5.92 pp (round 2's season-grain null bound) | Within-season spread survives clustering; R-2's wording stands |
| 89% upper bound < 5.92 pp and `σ_g` interval excludes zero | The within-season finding is game clustering, not the defence; wording reverts toward document 44 §7's and says so |
| Upper bound < 5.92 pp and `σ_g` includes zero | The finding was fragile to a nuisance term; reported as such, wording reverts |

Gate C-1's sampler bars apply to this fit as to any other.

## 3. The diagnostic — expected picks, priced

The forecasting statement document 32 §4 located in the product layer, made
computable. **Reported beside the red-zone and late-down gaps; never a
ledger row; never in the DTW distribution.**

**Per throw** `i` in a game, charted interception-worthy, thrown *by* the
offence `o` against defence `d`:

- `p̂_i = logit⁻¹(α̂ + X_i β̂)` from arm 2's posterior-mean fixed effects
  (round 2's fit, `research/outputs/64_*`), no random effects.
- `y_i` = 1 if intercepted.
- `swing_i` = the EPA cost of the pick branch relative to the escape branch,
  from a **bin table keyed on pre-throw state**: `yardline_100` in thirds ×
  `down` in {1–2, 3–4}, each cell the mean `epa` of picked worthy throws
  minus the mean `epa` of escaped worthy throws in that cell, 2022–2025.
  Cells with fewer than 30 throws on either branch fall back to the pooled
  difference (document 32 §3: −4.37 − (−0.83) ≈ −3.5). The six cells and
  their counts are printed.
- **Offence's fortune on the throw:** `f_i = (p̂_i − y_i) × |swing_i|`. A
  dropped pick is positive fortune to the offence; a pick on a low-`p̂` throw
  is negative.

**Per game, per team as offence:** `n_worthy`, `n_picked`,
`expected_picks = Σ p̂_i`, `fortune_epa = Σ f_i`. The game line is
`expected − actual` picks and `fortune_epa`, e.g.
*"GB threw 4 interceptable passes; DET picked 1; expected 2.1 — worth about
+3.8 EPA of good fortune to GB."*

**Presentation rule, committed:** the number is the **offence's** fortune,
because round 2 showed the finish does not carry across seasons. It is
never described as the defence's failure, skill, or luck.

**Deliverables:** `research/65_dropped_pick_diagnostic.py` writing
`research/outputs/65_dropped_pick_diagnostic.parquet` (one row per
game-team, all 2022–2025 games) and printing the two charted example
games from `research/58_brand_figures.py` (`2025_17_DET_MIN`,
`2025_13_DEN_WAS`; the other three predate FTN and print "not charted")
plus the three game-teams with the largest |`fortune_epa`| league-wide.
League distribution of `fortune_epa` per game-team: median, 89%
interval, share of games where |fortune| ≥ 3.5 EPA (one full pick). No
figure — rendering waits on the maintainer's figure notes; a queued item records it.

## 4. Power and gates

No new gate statistic is introduced; the clustering check reuses round 2's
committed null bound as its reference and is a robustness read, not a new
verdict. No power run. C-1 sampler bars on the new fit. Seed `20260827`.

## 5. Constants

| Constant | Value | Where |
|---|---|---|
| Cross-check grain | like-grain only (R-1) | this document; document 43 §5 amended |
| Reference bound for §2 | 5.920 pp | document 45 §4, row 1 |
| Bin table | `yardline_100` thirds × down {1–2, 3–4}; min 30 per branch; pooled fallback | `65` |
| Pooled swing fallback | escaped mean − picked mean, recomputed; expect ≈ 3.5 EPA | document 32 §3 |
| `β̂` source | `research/outputs/64_dropped_pick_confounds_r2.json` | `65` |
