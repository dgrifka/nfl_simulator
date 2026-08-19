# 32 — Dropped interceptions and dropped passes, closed

*Written 2026-08-19. This is a **closure memo, not a candidate round**: no
gates are pre-registered and no model is fit, because the question was decided
by arguments this project has already committed to — document 09's drops row
and document 21/28's rejection of the no-branch side door. The numbers below
are context in the document 09 §2 tradition (report the measurement behind a
Gate A denial so a reader can disagree with the argument, not the arithmetic).
Computed from cached pbp and FTN charting, 2022–2025; the query is small enough
to live in this document's history rather than a numbered script.*

*Inputs: documents 05 (the rule and the gates), 09 (drops, receiver and team),
17 (deflected interceptions), 21 and 28 (what may and may not open Gate A).*

---

## 1. The two questions

Two adjustments a reader of any public "luck" analysis will eventually ask
about:

1. **Dropped interceptions** — interceptable throws a team's quarterback got
   away with because the opposing defender failed to secure the ball.
2. **Dropped passes** — catchable throws an offense lost because its own
   receiver failed to secure the ball.

Both name a real, countable event. Neither gets a ledger row, and the two
denials are the same denial.

## 2. The answer, stated first

> **A ball in a player's hands is football, not a coin.** Gate A asks whether
> the outcome is resolved by a mechanism outside either team's control,
> conditional on the state both teams created. A defender securing or failing
> to secure an interceptable throw is the defense playing; a receiver securing
> or failing to secure a catchable throw is the offense playing. Document 09
> settled the receiver case before any of these numbers existed, and the
> defender case is the same argument with the jerseys swapped. Neutralizing
> either would credit a team for a play it failed to make.

Document 09 added the empirical warning that makes the mechanism argument
bite: **drop rates persist.** Receivers differ by a relative spread of ~21%
(document 09 §4), so redrawing drops erases real bad-hands skill. The
defender-side measurement below points the same way.

## 3. The numbers behind the denial (reported, deciding nothing)

FTN charts `is_interception_worthy` on every pass, so — unlike the deflection
candidate document 17 lost to an invisible denominator — **identification here
is trivial.** 2022–2025, 80,785 charted passes:

| Quantity | Value |
|---|---|
| Interception-worthy throws | 2,997 (3.7% of passes) |
| … intercepted | 1,454 — **p(INT \| worthy) = 48.5%** |
| … escaped ("dropped picks") | 1,543 (77 were even completed) |
| Mean EPA, escaped vs picked | −0.83 vs −4.37 — a ~3.5 EPA swing |
| QB interception-worthy rate, split-half r (≥200 att) | **+0.284** — the throw is skill |
| Defensive conversion rate, split-half r (128 team-seasons) | **+0.140** |

Two readings worth pinning down:

- **The 48.5% is a coin-like *rate*, not a coin.** A league-wide probability
  near one half says nothing about mechanism — document 05 §2's whole point.
  Fumble recovery earned its row because nobody controls a bouncing ball, not
  because its rate is 50%.
- **The defensive conversion signal is positive.** +0.140 on a quick odd/even
  split at a median of 22 chances per team-season is a rough number, not a
  calibrated one — but its sign is the warning: some of what a "dropped picks"
  adjustment would redraw is ball-hawking skill, the same trap document 09
  measured for receivers. And even a measured zero would not open the gate:
  that side door was proposed as amendment A-2 and rejected (documents 21,
  28), and amendment C-1 admits corrections of violations only, never new
  rows.

## 4. Where the idea's real content lives

The regression intuition behind these adjustments is sound — a team whose
opponents dropped five would-be picks has been fortunate in a way that will
not repeat *as a schedule-of-opponents effect*, even though each drop was an
opponent's failure. That is a **forecasting** statement. The ledger answers a
different question — who deserved *this* game — and a dropped pick is part of
what both teams did in it. The sanctioned home for the forecasting statement
is the product layer's reported diagnostics, beside the red-zone and
late-down gaps (document 05 §3: *reported separately, never as ledger rows*).

## 5. Register

| Candidate | Verdict | Decided by |
|---|---|---|
| Dropped passes (own receivers) | **Denied** — no branch point | Document 09, standing |
| Dropped interceptions (opposing defenders) | **Denied** — no branch point | This document, by document 09's argument |

After this memo, both directions of the hands-on-the-ball family are closed in
writing, and the project's answer to the most common public "luck adjustment"
is on the record with its numbers attached.
