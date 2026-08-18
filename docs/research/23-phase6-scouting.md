# 23 — Phase 6 candidate scouting: data-existence checks

*Written 2026-08-18. Scouting only — **no models were fit and no thresholds are
set here**. Step 1 of the candidate ladder, in the shape of document 15. Script:
`research/35_phase6_scouting.py`; results in
`research/outputs/35_phase6_scouting.json` and reproduced below.*

*Inputs: document 18 §9 (which asked for the conditioning audit), documents 05,
09, 14, 19; cached pbp 2016–2025.*

---

## Why these four

Document 18 shipped the round's only change, and it was a **conditioning bug in
an existing component** rather than a new component: the fumble population was
selected on the outcome of the branch immediately upstream of the branch being
neutralized. §9 of that document left an instruction — *audit the other
components' populations for the same shape before hunting new candidates* — and
this round starts there.

## Results

### C1 — The conditioning audit. **One live finding, one clean bill.**

| Component | Conditions on | Branch immediately upstream | Does it hide a coin? |
|---|---|---|---|
| **Field goal** | an attempt was made | the decision to attempt | **No.** A decision, not a coin. It is `core` by construction (document 05 §2) |
| **Field goal** | `field_goal_result` ∈ {made, missed, blocked} | **whether the kick was blocked** | **Yes — and the block is inside the population, priced as a miss** |
| **Extra point** | a touchdown was scored | the touchdown itself | **No.** Not a branch; scoring is the game |
| **Extra point** | `extra_point_result` ∈ {good, failed, blocked} | **whether the kick was blocked** | **Yes, same shape, 110 kicks** |

| Result | n | mean EPA |
|---|---|---|
| Field goal made | 9,085 | +0.649 |
| Field goal missed | 1,454 | −3.114 |
| **Field goal blocked** | **192** | **−3.665** |
| Extra point good | 12,101 | +0.066 |
| Extra point failed | 607 | −0.929 |
| **Extra point blocked** | **110** | **−1.093** |

`components.fit_fg_baseline`'s docstring already discloses this — *"blocked kicks
count as misses here… rare enough that separating it would add a class without
changing any conclusion"* — so it is a known simplification rather than a
discovery. What the audit adds is the direction and the size: a blocked kick is
worth **0.55 EPA more against the kicking team than an ordinary miss**, and the
component charges the difference to the kicker's luck. **192 kicks, and the
conditioning runs the wrong way**: the kicker is charged luck for a play the
protection lost.

Also recorded: 14,200 touchdowns against 12,818 extra-point attempts and 1,302
two-point attempts, leaving **80 touchdowns with neither charted**. Small, and
worth a line in whatever round touches extra points next.

### C2 — Blocked-kick aftermath. **The loose ball is invisible.**

| Blocked | n | Carrying a fumble row | Mean EPA |
|---|---|---|---|
| Field goals | 192 | **4 (2%)** | −3.665 |
| Extra points | 110 | **0 (0%)** | −1.093 |
| Punts | 113 | **1 (1%)** | −4.187 |

**415 blocked kicks in ten seasons, and five of them reach the fumble
component.** A blocked kick puts a ball on the turf with players from both sides
converging on it — the same physics document 18 spent a round establishing — and
the scoring convention does not call it a fumble, so the simulator sees nothing.

The candidate has two parts that must not be confused:

- **The block itself is a defensive play**, denied at Gate A the way a drop is.
- **What happens to the ball afterwards** is fumble-family, and it is currently
  booked entirely as deserved.

Stakes are modest but the swing per event is large: a blocked punt recovered by
the kicking team versus by the defense is worth several points of field
position, and 113 blocked punts is comparable to the 119 surprise onside kicks
document 20 was willing to model.

### C3 — Muffed punts. **Confirmed for punts. Not true for kickoffs, and that is
the round's best candidate.**

838 plays describe a MUFFS. Split by play type and by whether they sit inside
the v1.2 fumble population:

| Play type | Inside the fumble population | Outside |
|---|---|---|
| **Punt** | **561** | 4 |
| **Kickoff** | **18** | **245** |
| no_play | — | 9 |
| Field goal | 1 | — |

**Punt muffs are inside, at 99.3% — the expectation is confirmed in writing.**

**Kickoff muffs are not: 245 of 263 are invisible to the component**, and **88%
of those say the muffing side recovered the ball.** The scoring convention flags
a kickoff muff as a fumble essentially only when the *kicking* team comes up with
it.

That is exactly the shape document 18 corrected. The population is selected on
the outcome of the branch being neutralized: keep the ball and it never enters
the population; lose it and it does. A component fitted on the 18 visible
kickoff muffs would measure a recovery rate near zero and would be measuring the
convention rather than the football.

**This is the strongest Phase 6 candidate and it should be pre-registered
first.** It is a widening of an existing component rather than a new one, exactly
like the out-of-bounds fix, and the same Gate F structure applies with the
population redefined on the play text rather than on `fumble == 1`.

### C4 — Replay and challenge. **Dies on the denominator, as expected.**

3,646 plays across 2,001 of 2,761 games carry `replay_or_challenge == 1` — 0.75%
of all plays. Of those, **1,945 were reversed** and 1,699 upheld (2 denied).

The identification check is one paragraph and it is negative. A reversal is a
branch of sorts — a call goes one way or the other after the play — but the
population is **the calls somebody chose to challenge or the booth chose to
review**, and the calls that were wrong and never looked at are invisible.
Worse, the choice to challenge is itself a coaching decision made with a
timeout at stake, so the population is selected on a decision *and* on the
expected outcome of the branch. There is no version of this data where the
denominator — plays where a review would have reversed the call — is observable.
**Closed, on the same grounds as deflected interceptions (document 17) and the
punt bounce (document 14).**

One thing it is genuinely useful for: reversals carry mean EPA −0.280 against
+0.008 for upheld calls, which is a sanity check that the flag means what it
says, and a reminder that any component touching a reviewed play must use the
post-reversal record.

## The ranking this round produces

| Candidate | Shape | Verdict from scouting |
|---|---|---|
| **Kickoff muffs** | Widening an existing component's population | **Proceed to pre-registration first** |
| **Blocked-kick aftermath** | New component, fumble-family | **Proceed, second** |
| **Blocked kicks priced as misses** | Correction inside an existing component | **Proceed, third.** Smallest, and the cheapest to fix |
| Replay and challenge | New component | **Closed on identification** |

## What this document does NOT do

No thresholds, no verdicts, no fits. Each surviving candidate proceeds through
its own pre-registration: Gate A memo → power → committed thresholds including a
materiality floor computed from the games it touches → fit. Documents 16, 18 and
20 are the templates, and document 20 §9's lesson applies to all three: **print
the rejected rows, not their count.**
