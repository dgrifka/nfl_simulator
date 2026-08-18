# 15 — Phase 5 candidate scouting: data-existence checks

*Written 2026-08-18. Scouting only — **no models were fit and no thresholds are
set here**. This document records step 1 of the five-step candidate ladder
(data existence + rough stakes) for the four Phase 5 candidates and the one
policy question. Pre-registration (steps 2–4) happens in a separate document
before any fit, per the process laws. Script:
`research/24_phase5_scouting.py`; results reproduced below verbatim.*

*Inputs: documents 05, 09, 14; cached pbp 2016–2025 (484,254 plays, 372
columns) and FTN charting 2022–2025.*

---

## Why these candidates

Phase 4 closed the research program on the components a reader could name. The
remaining frontier is events that contain a **hidden coin** — a moment of
physics or a literal random device — inside a play the project has otherwise
already classified. Four such candidates, plus one policy question, were
brainstormed on 2026-08-18. This round asks only: *is the data there, and are
the stakes non-trivial?*

## Results, candidate by candidate

### C1 — Deflected-pass interceptions. **Data exists; viable.**

- Play descriptions are useless for this: only 17 of 4,304 interceptions
  (0.4%) mention tipped/deflected/batted. The channel is FTN.
- FTN charting matches **100% of interceptions 2022–2025** (1,690/1,690) on
  `nflverse_game_id` + `nflverse_play_id` (play_id needs a dtype cast).
- **236 of 1,690 interceptions (14.0%) are on throws FTN judged NOT
  interception-worthy** — the candidate deflection/fluke channel. Mean EPA
  −4.67 per event (vs −4.37 for interception-worthy INTs).
- Caveat for the Gate A memo: "not interception-worthy" is a judgment of the
  *throw*. The bucket contains tips and freak bounces, but possibly also
  clean defensive plays FTN's charter still scored as good throws. The memo
  must decide whether the FTN flag isolates the coin or merely encloses it.

### C2 — Fumbles out of bounds. **Data exists; extends the existing component.**

- `fumble_out_of_bounds` is a real column. **602 of 6,507 fumbles (9.3%)**
  went out of bounds — retained by the fumbling team by rule. Stable at
  ~48–72 per season, no trend.
- Live fumbles are lost by the fumbling team 47.9% of the time. So
  conditional on a loose ball, "did it skip out of bounds?" is a branch worth
  roughly half a fumble-recovery swing, ~60 times a season.
- Note: `components.py` currently **excludes** OOB fumbles from the recovery
  coin (correctly — nobody recovers them). This candidate is the *prior*
  branch: whether the ball stayed in play at all. Same oblong-ball physics
  argument as recovery itself.

### C3 — The overtime coin toss. **Data exists; effect is real and unchanged by the 2025 rules.**

- 155 OT games 2016–2025, 10 ties. The team with first OT possession won
  **86/145 decided games = 59.3%**.
- By era: 2016–2024: 59.2% (77/130). **2025 under the new rules: 60.0%
  (9/15)** — no sign the rule change removed the edge, though one season
  cannot settle it. Regular season 58.5%, playoffs 70.0% (n = 10).
- The toss itself is the project's only literal coin. Winning it is worth
  roughly an 18-point swing in win probability (59.3 vs 40.7) before either
  team runs a play. ~15 games a season are decided downstream of it.
- Design note for pre-registration: the deserve-to-win ledger is EPA-based;
  an OT adjustment is a *game-level* branch, not a play-level EPA event. The
  neutralization design (replace OT outcome with its expectation at the
  end-of-regulation state) needs its own foundations section.

### C4 — Onside kicks at the league rate. **Data exists; the text-match defect is smaller than feared.**

- 589 of 28,238 kickoffs desc-match "onside"; kicking team recovered 9.8%.
- The `own_kickoff_recovery` flag fires on 62 kickoffs, only **4** of which
  the desc match misses — so document 09's open defect (text-match
  identification) has a cross-check and the miss rate is small.
- Document 09 denied onside kicks **only** because the trust dial `w` could
  not be estimated per team, not on mechanism (Gate A passed: loose ball in a
  scrum). The w = 0 (league-rate) treatment needs no per-team estimate — the
  same choice already made for fumble recovery. That variant was never
  pre-registered and is the candidate here.

### C5 — The measured-zero-skill side door (interception returns). **The zero is a powered zero.**

Document 14 recorded team interception-return persistence at r = −0.014.
The open question was whether that zero is informative or just an
underpowered test (Finding 3's warning). We measured the instrument, per the
Phase 4 toolset: simulate team-seasons at the *observed* per-team sample
sizes (median 13 INT returns per team-season, SD 17.6 yards) with a known
true skill share, and ask what split-half correlation the test would show.

| True skill share of variance | Split-half r the test would show (median, 90% interval) |
|---|---|
| 0% | −0.002 [−0.105, +0.101] |
| 5% | +0.238 [+0.140, +0.332] |
| 10% | +0.400 [+0.308, +0.476] |
| 20% | +0.595 [+0.517, +0.654] |

The observed −0.014 sits squarely in the 0% band and **below the 5% band's
5th percentile**. Unlike the rematch test, this instrument is *not* blind at
the scale that matters: a skill share as small as 5% would very likely have
been seen, and nothing was. The empirical prong of the side-door argument is
therefore stronger than expected. What remains is purely the policy prong:
admitting a component on a measured zero rather than a mechanism is an
**amendment to Gate A** and must be pre-registered as such, with the power
table above attached, before any component uses it. Stakes are modest: 4,304
INT returns in ten years, mean 12.7 return yards, ~25 return yards in a
typical game that has one.

## What this document does NOT do

No thresholds, no verdicts, no fits. All four candidates and the policy
amendment proceed (or not) through pre-registration first: Gate A memo →
power calculation → committed thresholds including a does-it-change-anything
floor → fit. The rematch test's measured blindness (document 12) applies to
any validation proposed for these.
