# 52 — Amendment A-3: the hands-on-the-ball class, pre-registered

*Written 2026-08-27 in a Fable 5 brainstorm after document 50, **before any
further fit**. the maintainer's decision: keep the dropped-pick variant and make it a
ledger row, on the terms below. This document is the amendment's text, the
gates it must clear, and what it costs. It is written the way document 21
wrote A-2 — so that a reader can see exactly what rule changed and why —
and unlike A-2 it is being enacted, conditionally, not merely proposed.*

*Inputs: documents 05 (the rule), 09 (receiver drops persist), 21 (A-2),
28 (C-1, and why consistency binds), 32, 43–50 (the dropped-pick study and
the variant's audit), 33 (materiality and the flip audit).*

---

## 1. What changes, stated plainly

**Before A-3:** the ledger removes what *nobody controlled* — a bounce, a
kick's drift. Gate A's test is mechanism.

**After A-3:** the ledger also removes outcomes that a player controlled
*unreliably* — where the finish is near a coin and the entity's persistent
skill explains a small, measured share of it — **priced at the entity's
shrunk, posterior-sampled rate**, so a genuinely skilled entity keeps its
edge. Gate A's mechanism test stays for the existing rows; A-3 opens a
second, narrower door beside it, with its own gates.

This is a redefinition of "deserve to win" toward *what would not have
persisted*, and the community write-up says so in its first paragraph.
Document 28's objection — that a measured zero is not a mechanism — is not
answered; it is overruled by the product's owner, on the record, with the
arithmetic attached (§6).

## 2. Why the maintainer is doing it, honestly

The variant changes the verdict bucket on 12.03% of 2022–2025 games
(document 50). Its persistent-skill share is ~1.4% per throw (document 48).
Both are true: a near-coin, three times a game, at ~2 EPA a flip, is a lot of
luck. the maintainer's judgement is that this is what most readers mean by luck, and
that a simulator which leaves it in is answering a narrower question than
the one fans ask. This document does not pretend the effect size is
evidence; it records that the *definition* was chosen, and that the size is
what the definition implies.

## 3. The amendment text

> ### Amendment A-3 — the hands-on-the-ball class
>
> A component whose outcome a player controls, but whose finish is
> near-random, may be neutralized **partially, at the entity's shrunk
> posterior rate**, if and only if all of the following hold, each
> pre-registered before the component's fit:
>
> 1. **A powered measurement of the entity spread**, at the grain the
>    component charges (Gate C-3 ≥ 0.80 at the 12.5% reference), with the
>    result reported whichever way it lands.
> 2. **The expectation is drawn, not fixed**: `p` for every event is a
>    vector of posterior draws over the entity effect and the covariate
>    effects, so uncertainty about the entity widens the distribution.
> 3. **Both directions of the same event class enter together or not at
>    all.** Defender drops and receiver drops are one class.
> 4. **The materiality floor** (document 05 §7): median |ΔDTW| on affected
>    games ≥ the incumbent's median 89% half-width on the same games.
> 5. **Self-fulfilment bounded**: the entity effect read for a game must be
>    shown, by a held-out check, not to be materially the game's own.
> 6. **Coverage declared**: a component with partial season coverage ships
>    as a separately labelled edition, never silently mixed with the strict
>    ledger.
> 7. **A sunset**: re-tested every season against clause 1; a class whose
>    entity spread becomes a trait (cross-season persistence clears the
>    same bound) reverts to `core`.
>
> A component admitted under A-3 is recorded in document 05 §3's treatment
> table as **`partial (A-3)`**.

## 4. The two editions (clause 6, made concrete)

| Edition | Seasons | Ledger | Product label |
|---|---|---|---|
| **Strict** (v1.3) | 2016–2025 | fumbles, FG, XP | `strict` on every image |
| **Hands-on-the-ball** (v2.0) | 2022–2025 | strict + dropped picks + receiver drops | `v2.0` on every image |

Every figure and every public number names its edition. A game before 2022
has only the strict edition and says so. Which edition is the *headline*
on a 2022+ game page is a product decision recorded in the figure round,
not here.

## 5. Gates for this round (round 5), committed now

**R-3 — ruling on V-8.** The 2022 NYG interval [0.289, 0.509] against a
[0.30, 0.70] sanity bound is recorded as **immaterial**: the bound was a
plausibility check, the breach is 1.1 pp on one of ten lines, and the model
is describing a bad defence-season as bad. Document 50 §2 carries the
reasoning; the bound is not amended.

**G-1 — self-fulfilment (clause 5).** Refit the dropped-pick model **18
times, leaving out one week-of-season at a time** (all four seasons' week
`w` out together), and read each game's `u_d` from the fit that excluded
its week. Re-run the audit with those draws. Statistic: element-wise
agreement of verdict bucket between the in-sample variant and the week-out
variant over 2022–2025, and median |ΔDTW| between them on affected games.
**Pass:** agreement ≥ 90% **and** median |ΔDTW| < 1.0 pp. On pass,
production keeps the in-sample read with this bound recorded. On fail,
production must use the week-out read (18 traces), and the cost is noted.

**G-2 — pricing sensitivity.** Re-run the audit with every throw priced at
the pooled swing (−3.55 EPA) instead of its bin. Statistic: the number of
games that change verdict bucket under the flat swing, as a share of the
137 that change under the binned swing. **Pass:** ≥ 50% — the effect is
the coin, not the goal-line cell. On fail, the swing table is re-derived
with a higher cell floor before any row ships, as its own pre-registered
step.

**G-3 — materiality (clause 4).** On the 1,033 affected games: median
|ΔDTW| under the variant versus the median of v1.3's 89% half-width on
those same games. **Pass:** median move ≥ median half-width. *Pre-committed
note:* round 4's numbers put this close — 1.62 pp against a mean full width
of 3.83 pp — and it may fail. If it fails, A-3 is **not enacted** for
dropped picks on median grounds, the variant stays a labelled variant, and
the tail (12% bucket moves) is reported beside the failure. The floor is
not re-tuned for this component.

**G-4 — the mirror (clause 3).** A-3 is enacted only when the receiver-drop
component (the parked mirror) has passed its own pre-registered gates in
round 6. Until then the treatment table reads `variant (A-3 pending
mirror)`.

## 6. What the amendment costs, so nobody discovers it later

- The word "luck" on the game page now includes things players did.
- Two editions of every 2022+ game; one edition before 2022.
- The kicker precedent is stretched: FG partial neutralization rests on a
  branch (the ball in flight); A-3's rests on low persistence alone.
- Receiver drops persist ~21% relative (document 09) — more than
  defender finishing — so the mirror will neutralize less per event and the
  two directions will be visibly asymmetric. That is the honest outcome of
  clause 2, not a defect.
- Anything else with a near-coin finish and a small skill share (contested
  catches, fourth-and-short stops) can now cite this door. The register in
  document 05 §3 must say, for each, why it does or does not qualify.

## 7. Constants

| Constant | Value | Where |
|---|---|---|
| Week-out folds | 18 (weeks 1–18, all seasons together) | `research/69_dropped_pick_weekout.py` |
| G-1 bars | agreement ≥ 0.90; median \|ΔDTW\| < 1.0 pp | this document |
| G-2 bar | ≥ 0.50 of binned bucket moves | this document; `research/70_dropped_pick_sensitivity.py` |
| G-3 floor | document 05 §7's, unchanged | this document |
| Flat swing | −3.55 EPA (document 47 §3's pooled fallback) | `70` |
| Seed | 20260827 (fits), 20260817 (simulator) | as before |

---

## 8. Outcome

Round 5 ran 2026-08-27 on `feat/dropped-pick-variant` (document 53); **round 6**
re-ran G-1 the same day at document 54's amended fold spec (document 55);
**round 7** ran clause 3's receiver mirror the same day, pre-registered in
document 56 and recorded in document 57. **Unmerged.**

**As of round 7: every computed gate this amendment has ever set now passes.**
G-1, G-2 and G-3 for the dropped pick; G-4a, G-4b, G-4c and G-4d for the
receiver drop. Clause 3 is satisfied — both directions of the class have been
built and gated, and they enter together or not at all.

**A-3 is still not enacted, and no round can enact it.** §5's G-4 clause and
document 56 §3 make enactment "G-4a–d pass **and** the maintainer rules on §0's
wording", and the second half is a decision rather than a computation: is a
1-in-20 event a "near-random" finish within the meaning of §3's preamble?
Document 57 §1b restates the question with the arithmetic it now has — the
receiving corps' persistent skill explains **0.088%** of the per-target
variance, against the dropped pick's 1.4%, while Gate C-2 nonetheless **fails**
at the charged grain, so corps do measurably differ. Both facts are on the
record and neither settles it.

The treatment table in document 05 §3 therefore reads **`variant (A-3 pending
a wording ruling)`** on both rows — one step on from `variant (A-3 pending
mirror)`, which described the state before round 7.

| Gate | Round 7 result | Commit |
|---|---|---|
| **G-4a** — the study (clause 1) | **PASS.** C-3 power **0.877** at the charged grain. The clause-1 grain rule fired on its second branch: power was 0.40 at receiver-season, so the component charges the **team-season**, the receiving corps. Gate C-2 **fails** at that grain (0.87 pp against a 0.63 pp threshold) and is reported, which is what clause 1's "reported whichever way it lands" provides for — and what §6 predicted | `790cac0` |
| **G-4b** — the component | **PASS.** V-1 0.00e+00 over 2,761 games, V-2 0.00e+00 on all three variant arms, V-6 (0 divergences, max r̂ 1.0018 over 276 parameters), V-8 ten of ten lines under both readings, read-side round trip 0.00e+00 over 54,160 rows. V-3, V-4, V-5, V-7 as tests | `109799f`, `e88a42f` |
| **G-4c** — self-fulfilment (clause 5) | **PASS.** Nineteen folds at document 54 F-1's spec, **all nineteen clearing Gate C-1 with zero divergences**; worst r̂ 1.0036, thinnest ESS-tail 1,904. Bucket agreement **0.996** (1,134/1,139) against ≥ 0.90; median \|ΔDTW\| between the arms **0.04 pp** against < 1.0 pp. §5's consequence applies to this direction too: **production keeps the in-sample read.** The week-out arm moves 165 games against the in-sample arm's 162, so the 14% is not an in-sample artifact | `e774547` |
| **G-4d** — materiality (clause 4) | **PASS.** On the 1,138 affected games, median \|ΔDTW\| **2.32 pp** against a median 89% half-width of **0.56 pp**, clear by 1.75 pp. The floor was not re-tuned | `e88a42f` |
| **G-5** — the combined audit | Reported, never gated. `+dp` 136 bucket moves, `+rd` 162, `+dp+rd` 200. §6's predicted asymmetry shows up in the interval widening: `+dp` widens the mean 89% DTW interval 0.0383 → 0.0514, `+rd` only 0.0387 → 0.0405, because a team-season's catch rate rests on ~437 balls and a defence-season's finishing on ~22 chances | `e88a42f` |
| **V-1** — v1.3 untouched | **PASS** again, 0.00e+00 over 2,761 games, at the start and end of both round-7 audit runs | round 7 |

*The round-6 table below is kept as it was written.*

---

**As of round 6: G-1, G-2 and G-3 all pass. A-3 is not yet enacted, and the only
thing it waits on is clause 3's receiver mirror (G-4, round 7).** The treatment
table in document 05 §3 therefore now reads `variant (A-3 pending mirror)` —
the wording §5's G-4 clause wrote for exactly this state.

| Gate | Round 6 result | Commit |
|---|---|---|
| **G-1** — self-fulfilment (clause 5) | **PASS.** Nineteen folds at document 54 F-1's spec (weeks 1-18 plus a postseason fold holding weeks 19-22 together), **all nineteen clearing Gate C-1 with zero divergences**; worst r̂ 1.0043, thinnest ESS-tail 1,602. Bucket agreement between the in-sample and week-out arms **0.997** (1,136 of 1,139 games) against a ≥ 0.90 bar; median \|ΔDTW\| between the arms **0.05 pp** against a < 1.0 pp bar. Document 52 §5's consequence applies: **production keeps the in-sample read, with this bound recorded.** The week-out arm moves 139 games across a bucket against the in-sample arm's 136, so the 12% is not an in-sample artifact | round 6, document 55 |
| **G-2**, **G-3** | Unchanged from round 5 — both PASS | `a4ce823` |
| **G-4** — the mirror (clause 3) | **Not run.** Round 7, and it is now the *only* thing A-3 waits on | — |
| **V-1** — v1.3 untouched | **PASS** again, 0.00e+00 over 2,761 games | round 6 |

The round-5 table below is kept as it was written.

---

Round 5 ran 2026-08-27 on `feat/dropped-pick-variant`. Full record: document 53.
**Unmerged.**

| Gate | Result | Commit |
|---|---|---|
| **R-3** — ruling on V-8 | **Made.** The 2022 NYG breach of 1.1 pp on one of ten lines is immaterial; the bound stands unamended. Recorded in documents 50 §2 and 49 §10 | `cf73e8b` |
| **G-1** — self-fulfilment (clause 5) | **BLOCKED.** Eighteen week-out folds fitted; **7 of 18 miss Gate C-1** at §7's spec (weeks 1, 3, 7, 9, 13, 14, 16) — 0 divergences in all eighteen, r̂ at worst 1.0146 on `sigma_d`/`sigma_q`, ESS-tail as low as 245. The agreement statistic is **not computed**, and neither is the week-out variant's audit. All eighteen traces are on disk; the fits cost 188 s | `2c3223a` |
| **G-2** — pricing sensitivity | **PASS.** Every throw at the pooled −3.55 EPA moves **129** games across a verdict bucket, **0.94** of the 137 the binned swing moves, against a ≥ 0.50 bar. Element-wise: 116 in both move sets, 21 binned-only, 13 flat-only | `a4ce823` |
| **G-3** — materiality (clause 4) | **PASS.** On the 1,033 affected games, median \|ΔDTW\| **1.62 pp** against a median 89% half-width of **0.56 pp**, clear by 1.06 pp; also clear on the means (6.96 vs 1.91 pp). §5's pre-committed note expected a possible fail by comparing a median move to a *mean full width*; the gate was computed on its own text | `a4ce823` |
| **G-4** — the mirror (clause 3) | **Not run.** Round 6, and it now waits on G-1 as well | — |
| **V-1** — v1.3 untouched | **PASS**, twice: 0.00e+00 over 2,761 games on the deserved margin, DTW% and both bounds, re-printed at the end of both audit runs | `2c3223a`, `a4ce823` |

**A-3 was not enacted at the end of round 5.** Both substantive gates passed;
what was missing was the precondition to G-1, so clause 5 had no bound and §4's
editions did not ship. The treatment table therefore did **not** read
`variant (A-3 pending mirror)` — that wording describes the state after G-1
passes. Document 53 §5 carried the three routes on the fold spec; none was taken
by the session that found the failure.

*Superseded by round 6, above: the maintainer took route (1), document 54 wrote the new
spec before it ran, and G-1 passed. §4's editions still do not ship, but the
reason is now clause 3's mirror rather than clause 5's missing bound.*

