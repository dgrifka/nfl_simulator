# 62 — The possession-level luck cap

*Round 9, 2026-08-28. Pre-registered in document 61 (§4 carries the
measurement, §8 the outcome); built on branch `feat/possession-cap`; audited by
`research/78_possession_cap_audit.py`. This document is the record: what
changed, what it cost, and the one thing it does not do.*

---

## 1. What changed, for a reader

Until now the Full edition priced every luck event against its own
counterfactual and added the rows up. Two events on one possession were added as
though the second would still have happened had the first gone the other way —
and in the bootstrap, where every coin is flipped independently, a possession
could book more luck than a possession can hold.

From round 9 the Full edition bounds each possession by the largest single "what
if" on it. If the biggest event on a drive had gone the other way, the drive's
outcome would have changed by at most that event's swing; the smaller events on
the same drive are contingent on it. Strict is untouched.

**The worked case, which did not turn out the way the proposal expected.**
Document 61 was raised from the `2024_19_LAC_HOU` waterfall: two
would-be-touchdown drops that looked like one possession booking about fourteen
points of luck. They are not one possession. They are

- `play_id` 3221 on `fixed_drive` **21**, swing −9.07, booking +8.60 EPA — a Los
  Angeles drop that was returned for a Houston score, which is why Houston's
  extra point sits on Los Angeles's drive; and
- `play_id` 3368 on `fixed_drive` **22**, swing −9.28, booking +8.80 EPA — the
  drop on Los Angeles's *next* possession.

Each drive holds one large swing that **is** its own `C_d`, so neither is
clipped, and the cap leaves both exactly where they were. What the cap does move
in that game is elsewhere: a Los Angeles possession carrying a dropped pick and
two drops (Q4 drive 26, −0.52 EPA), a ten-event Houston possession (Q2 drive 12,
−0.49), and eleven smaller ones. Houston's deserved margin goes from −0.22 to
+0.95 and its DTW% from 47.8 to 54.8 — a seven-point move that crosses the
coin-flip line and stays, correctly, inside "too close to call".

The cap is a **bound, not a drive simulation**. Document 08's decision stands:
the bars are a sum, not a sequence.

## 2. The measurement, before anything was built

`research/77_possession_cap_measure.py`, filled into document 61 §4. Three
things it established.

**There is something to bound almost everywhere.** 69,419 Full events sit on
21,327 possessions; **93.1% of them share a possession with another event**, and
16,548 possessions carry two or more — 77.6% of those that carry any. A median
event-carrying possession holds 3.25 events and the longest holds 15.

**The statistic the pre-registration named was nearly blind, and a second one
had to be added.** Document 61 §4 asks whether `Σ|luck_i| > C_d`, which is a
question about the point estimate. It bites on **0.9%** of possessions. But in
the bootstrap an event contributes either nothing or its **whole swing** —
`a_i = (actual_i − replayed_i) × swing_i` with both terms in {0, 1} — so a
possession's replicate sum ranges over subset sums of its swings, which are
several times larger than the expectation-weighted `luck_i` the ledger prints.
By that measure the clip engages on **69.3%** of possessions. On all three named
games the point-estimate give-back is 0.00 EPA, so M-3's stop-and-ask could not
have been decided on it at all. The pre-registered number is reported first and
unchanged; the quantity §5 actually books — `mean(clipped) − mean(unclipped)` —
was estimated beside it and named as an addition.

**M-3's stop condition did not trip.** All 200 of the uncapped Full edition's
bucket moves hold a possession the cap clips, so the exact upper bound is 200
and carries no information. The cap hands back a median 0.31 pt of deserved
margin on those games, a median 9.1% of the Full-minus-Strict shift; walking DTW
back by the same fraction, a linear proxy vanished **29 of the 200 (14.5%)**
against the one-third threshold. Part C's real number turned out to be 14.

**What it costs to leave Strict alone.** 4.8% of Strict possessions would take a
cap row, worth a median 0.029 EPA and at most 2.19 EPA (1.84 pt). Not applied —
P-1 — and now stated rather than assumed.

## 3. The cap, and its gates

Per bootstrap replicate, on each possession `d`:

```
A_d = clip( Σ_i a_i , −C_d , +C_d )      C_d = max_i |swing_i| on drive d
```

Applied inside `bootstrap_margins` via a keyword-only `drive_of`, on the Full
edition and nowhere else. Two implementation choices document 61 left open:

- **The clip is proportional across the possession's events**, not taken off a
  nominated one. That keeps the two teams' deserved-point split correct —
  `home − away` is still the margin the same replay produced — and makes "never
  grows a replicate, never flips its sign" true by construction.
- **A cap row is a ledger row.** `component = "possession_cap"`, `event_class` the
  possession (`"Q3 drive 7"`), `charged_team` the drive's offence, `play_id` the
  play that set `C_d`, and `luck_epa` the mean the clip removed. The round trip
  closes on it and a waterfall can draw it.

| Gate | Bar | Result |
|---|---|---|
| **P-1** Strict untouched | V-1 0.00e+00 over 2,761 games, on the widened frame | **PASS**, 0.00e+00 |
| **P-2** no-cap identity | the uncapped Full arm with `fixed_drive`/`qtr` against the same arm without them | **PASS**, 0.00e+00 over 1,139 games |
| **P-2** (b) | that arm still reproduces document 59 §4 | **PASS**, 200 moves / 3.85 pp exactly |
| **P-3** round trip | ledger with cap rows × slope = margin shift, ≤ 1e-9 | **PASS**, 0.00e+00 |
| **P-4** direction | the clip never grows a replicate or flips its sign | **PASS**, `tests/test_possession_cap.py` |
| **P-5** single-event drives | never capped | **PASS**, by construction and by test |
| **P-6** two-drops case | two 9-EPA drops on one possession book at most 9 | **PASS**, exactly 9 |
| **P-7** audit | reported beside document 59's | §4 below |

**P-4 has a limit worth stating.** "Every cap row reduces `|Σ drive luck|` and
never flips its sign" is exactly true **per replicate**, and that is what the
test pins. It is not a theorem at the level of the cap row, which is a mean over
replicates: on a possession carrying events of both signs, clipping the large
replicates can move that mean across zero. The per-replicate invariant is the
strong statement; the cap-row statement is pinned on a same-signed possession,
which is the case document 61's words describe.

## 4. The audit

`research/78_possession_cap_audit.py`, 2022-2025, 1,139 games, at v1.3's
settings. The three arms:

| arm | bucket moves | median \|ΔDTW\| on affected | mean \|ΔDTW\| | mean 89% width |
|---|---|---|---|---|
| Strict | — | — | — | 0.0387 |
| Full, uncapped (document 59) | 200 | 3.85 pp | 10.56 pp | 0.0516 |
| **Full, capped** | **190** | **3.39 pp** | **10.05 pp** | **0.0495** |

**The sentence that replaces document 59's.** Document 59 §4 says the Full
edition changes the verdict bucket on **200 of 1,139 games, 17.6%**. With the
possession cap it changes **190 of 1,139, 16.7%**, and the median move on the
1,138 games the class touches falls from 3.85 pp to **3.39 pp**.

Compared **element-wise**, never by subtracting totals: of the 200 games the
uncapped Full edition moves, **186 still move**, **14 no longer do** (7.0% of
them), and **4 move that did not before**. The net is ten; the disagreement is
eighteen.

**The width was the other half of the defect, and it moved too.** The mean 89%
interval on affected games narrows from 0.0516 to **0.0495**, a 4.0% reduction —
about a seventh of the way back from Full's width to Strict's 0.0387. That is
the over-dispersion document 61 §0 named: a distribution wider than any single
possession could have produced.

**What the cap actually booked.** 14,747 cap rows over the 1,139 games; every
game carries at least one, a median of 13 and at most 21. They book +46.6 EPA in
total, a **median of 0.012 EPA per row** and at most 5.67. That shape is the
finding: the cap touches nearly every possession and does almost nothing to
nearly all of them, then does something large a few dozen times.

**The three named games, Full before and after:**

| game | DTW% uncapped → capped | deserved margin | cap rows |
|---|---|---|---|
| `2025_17_DET_MIN` | 47.5 → 46.9 | −0.32 → −0.41 | 8, +0.10 EPA |
| `2022_13_WAS_NYG` | 97.9 → 97.4 | +13.90 → +11.90 | 14, +2.38 EPA |
| `2024_19_LAC_HOU` | 47.8 → 54.8 | −0.22 → +0.95 | 13, −1.39 EPA |

No bucket changes among the three. Washington at New York keeps its clear flip
with two points of the deserved margin handed back; the two "too close to call"
games stay too close to call.

**The largest single reduction** is `2023_11_PHI_KC`: deserved margin +12.53 →
+7.49, DTW% 99.2 → 95.0, and almost all of it is one cap row — Kansas City's Q4
drive 23, +5.67 EPA. A possession that had booked more than five extra points of
luck than its own biggest play was worth.

`full_summary.parquet` is regenerated from the capped arm, so every Full figure
now replays against these numbers.

## 5. Register

| Item | Status |
|---|---|
| The cap is a bound, not a drive simulation | By design; document 08 stands |
| Strict is not capped | P-1. M-4 measures the stakes: 4.8% of Strict possessions, at most 1.84 pt |
| **Cross-drive dependence** — a turnover ends one drive and starts another | **Out of scope, and it is what the LAC @ HOU case needs.** The two drops that raised document 61 are on consecutive possessions; no per-possession bound can reach them |
| A tighter, state-aware cap (best attainable drive EPA minus realised) | Parked; needs drive-level EP modelling this project has scoped out |
| P-4 at the cap-row level on a mixed-sign possession | Not a theorem; the per-replicate invariant is. §3 above |
| Document 61 §4's `Σ\|luck\| > C_d` statistic | Reported and kept, but nearly blind to what the cap does. §2 above |
| `possession_cap` rows in the figures | Figure round 7's, not round 9's. `plots.event_phrase` already renders an unfamiliar component without crashing |
