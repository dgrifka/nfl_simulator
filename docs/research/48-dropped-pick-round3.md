# 48 — Dropped picks, round 3: the finding survives a nuisance term, and gets a number

*Written 2026-08-27. The results record for round 3 of the study pre-registered
in document 43, amended by document 45, and ruled on by document 47, run on the
branch `docs/dropped-pick-confounds` in four parts: the rulings (Part A), the
game-clustering robustness check (`research/66_dropped_pick_game_effect.py`),
the diagnostic (`research/65_dropped_pick_diagnostic.py`), and this record.
Nothing in `src/nfl_simulator/` changed, no ledger row moved, and simulator v1.3
is untouched. Nothing here enters the DTW distribution.*

*Inputs: documents 32 §3–§4 (the closure and where the diagnostic lives), 43
(the pre-registration), 45 (round 2's amendments), 46 (round 2's record), 47
(this round's rulings and design).*

---

## 1. The answer, stated first — row 1 landed, and R-2's wording stands

**Document 47 §2's first row landed.** With a game-level random effect added to
round 2's arm 2, the defence-season spread in conditioned conversion is

> **`σ_d` 5.83 pp, 89% interval [1.99, 8.88]** — and the upper bound **8.88 pp
> clears the 5.920 pp reference** by 2.96 pp.

The pre-committed reading of that row, applied without re-tolerancing: *the
within-season spread survives clustering; R-2's wording stands.* The sentence
adopted in document 46 §7 (b) — *"in any given season some defences finish more
of their chances than others — but that edge doesn't carry to the next season,
so it reads as a year, not a trait"* — ships unchanged. §4 gives its final text.

**The interesting part is that the game effect is real and large, and the
finding survives anyway.** `σ_g` is **8.53 pp [2.45, 13.22]**, an interval well
clear of zero and the *largest* of the three variance components in the model —
throws inside one game are more alike than throws across games, by more than
defence-seasons differ from each other. The alternative hypothesis document 47
§2 raised was not a straw man; it was true. It just does not explain away the
defence-season spread. Adding it moved `σ_d` from 6.35 pp to 5.83 pp, about half
a percentage point, and widened the interval downward (lower bound 3.11 → 1.99)
rather than collapsing the estimate.

**In plain words:** we asked whether "some defences finish more of their chances
within a season" was really just "some *games* are ball-hawking games — bad
weather, a rattled quarterback, a charter having a day". The answer is that
games really do differ, more than defence-seasons do, and after paying for that
the defence-season difference is still there.

## 2. The game-effect fit

Round 2's arm 2 with one node added and nothing else moved:

```
logit p_i = α + X_i β + u_d[i] + v_q[i] + w_g[i]
w_g ~ Normal(0, σ_g)        σ_g ~ HalfNormal(0.5)      non-centred
```

Same 2,969-row frame (asserted, not assumed), same 18 covariates, amendment
A-2's sampler spec (4 chains × 2,000 draws after 2,000 tuning, `target_accept`
0.9), seed `20260827`. **1,031 games** carry at least one worthy throw, median
3 chances per game. Document 47 §2 expected ~1,090; the gap is games in which no
throw was charted interception-worthy, and it is disclosed rather than chased.

**Gate C-1, sampler half: PASS.** Zero divergences, max `r_hat` **1.0074**
(`sigma_d`), min `ess_bulk` **771** and min `ess_tail` **671** (both `sigma_g`),
**0 of 1,461 parameters over any bar**. The added node did not cost geometry —
`sigma_g` is in fact better mixed than `sigma_q` was in round 2.

Variance components, both scales (`σ_p ≈ σ_logit × p̄(1−p̄)` at `p̄ = 0.485`):

| Component | Logit (89%) | Probability (89%) | Round 2 (no game effect) |
|---|---|---|---|
| `σ_d`, defence-season | 0.234 [0.079, 0.355] | **5.83 pp [1.99, 8.88]** | 6.35 pp [3.11, 9.08] |
| `σ_q`, QB-season | 0.186 [0.026, 0.330] | **4.64 pp [0.65, 8.25]** | 5.13 pp [1.09, 8.52] |
| `σ_g`, **game** | 0.341 [0.098, 0.529] | **8.53 pp [2.45, 13.22]** | — |

The four coefficients that kept an 89% interval clear of zero in round 2 are the
same four here, at the same sizes:

| What moves conversion | β (logit), with game effect | Round 2 |
|---|---|---|
| `is_catchable_ball` | **−0.518** [−0.761, −0.282] | −0.489 |
| `is_contested_ball` | **−0.318** [−0.509, −0.129] | −0.321 |
| `air_yards` (+1 SD) | **+0.325** [+0.239, +0.413] | +0.316 |
| `qb_hit` | **+0.215** [+0.044, +0.388] | +0.212 |

**max |Δβ| vs round 2 is 0.0293** (`is_catchable_ball`), **|Δα| 0.0131**. The
fixed effects are effectively unmoved: the game effect absorbs variance from the
random-effect side of the model, not from the covariates.

**One disclosure, carried from document 47 §2.** The crossed grid arm 3 uses
cannot take a third factor, so this check ran in the confirmatory arm. The
comparison therefore mixes instruments — `σ_d` here is arm 2's parameter read
against arm 3's season-grain null bound of 5.920 pp. That is the comparison
document 47 §2 committed to before any fit, and it is the one applied. A reader
who wants a like-instrument version of this check does not have one.

## 3. The diagnostic — expected picks, priced

Every 2022–2025 game-team now has an expected-picks line and a priced
`fortune_epa`, in `research/outputs/65_dropped_pick_diagnostic.parquet`
(**2,278 rows = 2 × 1,139 scheduled games**, verified). Teams that threw no
interceptable pass are zeros, not missing rows.

### 3a. The bin table

`swing` is the picked branch relative to the escaped branch, so every cell is
negative: an interception costs the offence EPA. All six cells cleared the
30-per-branch floor, so **none used the pooled fallback** (which computes to
**−3.55 EPA**, against document 32 §3's expected ≈ −3.5).

| `yardline_100` | down | n picked | n escaped | mean EPA picked | mean EPA escaped | swing |
|---|---|---|---|---|---|---|
| 1–33 | 1–2 | 194 | 218 | −5.47 | −0.43 | **−5.04** |
| 1–33 | 3–4 | 147 | 144 | −4.56 | −1.55 | **−3.01** |
| 34–66 | 1–2 | 395 | 410 | −4.38 | −0.56 | **−3.82** |
| 34–66 | 3–4 | 245 | 257 | −3.63 | −1.36 | **−2.27** |
| 67–99 | 1–2 | 312 | 331 | −4.53 | −0.50 | **−4.03** |
| 67–99 | 3–4 | 161 | 167 | −3.67 | −1.18 | **−2.49** |

The table earns its place rather than merely satisfying the pre-registration:
**a pick costs roughly twice as much on early down as on late down** (−5.04 vs
−3.01 in scoring range), because on 3rd or 4th down the escape branch is often a
punt or a turnover on downs anyway, so the interception takes away much less.
Pricing every pick at the pooled −3.55 would have overcharged late-down picks by
about a point of EPA and undercharged early-down ones in scoring range by 1.5.

### 3b. The example games

Both of document 47 §3's charted example games are present, and **both happen to
be games in which every interceptable throw was actually intercepted** — so
neither shows a dropped pick:

> **2025_17_DET_MIN** — DET threw 2 interceptable passes; MIN picked 2; expected
> 1.2 — worth about **−2.7 EPA** of bad fortune to DET.
> MIN threw no interceptable passes; no fortune either way.
>
> **2025_13_DEN_WAS** — DEN threw 1 interceptable pass; WAS picked 1; expected
> 0.5 — worth about **−1.3 EPA** of bad fortune to DEN.
> WAS threw 1 interceptable pass; DEN picked 1; expected 0.6 — worth about
> **−2.2 EPA** of bad fortune to WAS.

Document 47 §3's illustrative line ("MIN threw 3 interceptable passes; DET
picked 0; expected 1.4 — +5.1 EPA") was a made-up shape, not a claim about these
games; the real ones are checked against the play-by-play above. **The
consequence for the product is worth stating: the two games already charted for
the figure round are both bad-fortune-only examples.** A demonstration of the
diagnostic's headline case — a team that got away with several — needs a third
game, and §5 queues that.

The three largest good-fortune game-teams league-wide:

| Game | Offence | worthy | picked | expected | `fortune_epa` |
|---|---|---|---|---|---|
| 2022_13_WAS_NYG | WAS | 7 | 0 | 3.35 | **+11.72** |
| 2022_05_HOU_JAX | JAX | 7 | 1 | 3.98 | **+9.95** |
| 2022_03_BUF_MIA | BUF | 6 | 0 | 2.61 | **+9.52** |

and the largest bad-fortune ones are 2023_11_ARI_HOU (HOU, 3 of 3 picked,
−7.97), 2022_01_LV_LAC (LV, −7.13) and 2024_20_WAS_DET (DET, −7.08). **The
distribution is asymmetric by construction and that is not a bug:** a team can
drop at most as many picks as it threw interceptable passes, but the upside of
escaping all of them compounds — hence a +11.7 tail against a −8.0 one.

### 3c. The league distribution

| Quantity | Value |
|---|---|
| Median `fortune_epa` per game-team | **+0.00 EPA** |
| 89% interval | **[−3.32, +3.70]** |
| Share with \|`fortune_epa`\| ≥ 3.5 EPA (one full pick) | **10.8%** |
| Share with ≥ 1 interceptable throw | **68.9%** |
| Total expected picks vs actual, 2022–2025 | 1,479.8 vs **1,454** |

Roughly **one game-team in nine** was handed or denied a full pick's worth of
EPA by how the interceptable throws in that game happened to land. In about
three games in ten, the question does not arise at all.

### 3d. What this number is, and what it is not

*(Document 47 §3's presentation rule, applied.)*

`fortune_epa` is **the offence's fortune**. It answers one question: given how
catchable, contested, deep and hurried each interceptable throw was, how many of
them would a league-average defence have caught, and what did the difference
between that and what actually happened cost or save the team that threw them?
A positive number means a team got away with throws it usually does not get away
with. A negative number means it was picked on throws that usually survive.

It is **not** a measure of the defence — not its failure, not its skill, not its
luck. Round 2 established that a defence's finishing edge does not carry from one
season to the next, so there is no defensive trait for this number to be
evidence of, and describing a −8.0 as "the defence's great day" would be
attributing to a team something round 3 has just shown is mostly the game. It is
also **not a ledger row and not part of the deserve-to-win distribution**: it is
reported beside the red-zone and late-down gaps, as document 32 §4 located it,
and simulator v1.3 does not see it. Finally, it is a *point* estimate built on
posterior-mean fixed effects with no random effects and no uncertainty
propagated — it prices a game, it does not test a hypothesis about one.

## 4. The adopted game-page wording, final text

Document 46 §7 (b), adopted by document 47 R-2 and unchanged by this round's
robustness check:

> **"Interceptions your opponent dropped are counted here as they happened. In
> any given season some defences finish more of their chances than others — but
> that edge doesn't carry to the next season, so it reads as a year, not a
> trait."**

Round 3 adds no clause to it. What round 3 buys is the right to keep it: had
`σ_d`'s upper bound fallen below 5.920 pp, document 47 §2 would have forced this
sentence back toward document 44 §7's "cannot settle" version.

## 5. Register

| Defect | Evidence | Status |
|---|---|---|
| Is the within-season spread game clustering? | §2: `σ_d` 5.83 pp [1.99, 8.88], upper clears 5.920; `σ_g` 8.53 pp [2.45, 13.22] | **Closed, row 1.** Game clustering is real and larger than the defence-season term; the defence-season spread survives it |
| Gate C-1 on the game-effect fit | §2: 0 divergences, max `r_hat` 1.0074, 0 of 1,461 over a bar | **PASS** |
| Clustering check mixes instruments | §2, disclosed in document 47 §2 before any fit | **Open, by design.** Arm 2's `σ_d` read against arm 3's null bound; the crossed grid cannot take a third factor |
| Document 46 §2 characterises the 28 dropped rows as "all 28 missing `pass_location`" | §6: true, but 27 also miss `air_yards` and 16 miss `down` | **Corrected here.** Document 47 §3's imputation instruction was written on the incomplete version; §6 records what was done instead |
| Both charted example games have zero dropped picks | §3b | **Open.** The product cannot demonstrate the headline case from the two games already charted; queued |
| `fortune_epa` propagates no uncertainty | §3d: posterior-mean fixed effects, point estimate | **Disclosed, by design.** Document 47 §3 specified point `β̂`; a credible band per game-team is not costed |
| Defence finishing does not carry across seasons | Document 46 §4–§5, unchanged | **Standing.** The presentation rule in §3d rests on it |

## 6. Deviation from the pre-registration, disclosed

**One, and it is the pre-registration's factual premise, not its rule.**

Document 47 §3 and the round-3 handoff both say the 28 worthy throws the model
drops are "the 28 with null `pass_location`", and instruct that they be scored
"with `pass_location` set to the reference level". Document 46 §2 is where that
description comes from, and it is true as far as it goes — all 28 do miss
`pass_location`. It is not the whole truth: **on the same nested 28 rows,
`air_yards` is null on 27 and `down` on 16.** Setting only `pass_location`
leaves `p̂` undefined on 27 of the 28, which is how the defect was found.

**What was done instead**, and why it is the same rule rather than a new one:
every null covariate on those rows goes to *its* reference level. For a dummied
factor that is the omitted level (`pass_location` → middle, `down` → first
down). For a standardised covariate it is round 2's mean, which standardises to
exactly 0 and contributes nothing to the linear predictor — the same "no
information" the omitted dummy level encodes. `p_hat_imputed` flags the union,
and it is the same 28 rows either way.

Separately, the **bin** for those rows is assigned from the *unimputed* state: a
throw whose `down` was never recorded has an unknown pre-throw state, and
pricing it in the first-down cell would invent one, so it takes the pooled swing
— the mechanism document 47 §3 already provides for a cell it cannot read.

**Materiality:** 28 of 2,997 throws, under 1%, touching at most 28 of 2,278
game-teams, each at a `p̂` pinned to the league average. The 2,969 modelled rows
are untouched, which the pre-registered tripwire confirms: **mean `p̂` over them
is 0.4934 against round 2's 0.4937**, inside the ±0.002 tolerance document 47 §3
committed to.

**A second, smaller disclosure.** Round 2's JSON does not store the
standardisation constants as literals, which the handoff named as a stop-and-ask.
They are recoverable exactly rather than absent: `61`'s `design_matrix` takes a
`reference` frame and standardises against *its* mean and SD, and round 2's
2,969-row frame is rebuilt deterministically under guards that assert 80,785
charted passes, 2,997 worthy and 2,969 modelled rows before anything is fitted.
The diagnostic passes `reference=frame.model`, so the constants are round 2's by
construction, and the mean-`p̂` tripwire above is what proves it — it is exactly
the check that would fail if this frame's own constants had been used.

## 7. Commits

`e655060` document 47 (the pre-registration and the rulings) · Part A, the
rulings recorded in documents 43 and 46 · Part B,
`research/66_dropped_pick_game_effect.py` · Part C,
`research/65_dropped_pick_diagnostic.py` · Part D, this record.

Branch `docs/dropped-pick-confounds`, **unmerged; the maintainer merges.**
`git diff main -- src/` is empty; 502 tests pass; ruff clean.
