# 00 — Business context

*Written 2026-08-17, before touching the data. The EDA guide's first gate: if
you cannot name the decision, the analysis has no direction.*

## What decision does this inform?

**Which EPA components the simulator is allowed to neutralize.**

The deserve-to-win simulator works by replacing a luck event's realized Expected
Points Added with its expectation — a fumble on the ground becomes the average of
the recovered and not-recovered branches instead of the branch that happened.
That move is only defensible for components that really are coin flips. If we
neutralize something a team genuinely controls, we erase skill and the simulator
systematically flatters bad teams.

So Phase 1 produces a list: for each component of a game's scoring margin, is it
**skill** (persists for a team across games) or **luck** (does not)? Components
on the luck side get neutralized in Phase 2. Components on the skill side stay
exactly as they happened.

## Who is the audience?

The maintainer first — he has to believe the classification before building on it. Then a
general sports-analytics readership when the repo goes public. Both want the
number *and* the reason, so every claim here needs a figure or a table behind it.

## What changes if the answer is different?

A great deal, and that is what makes the question worth asking:

- **If fumble recovery shows real team skill**, the central premise breaks. The
  whole approach assumes fumble recovery is the cleanest coin flip in football.
- **If interceptions turn out to be mostly luck**, the simulator has to
  neutralize them too, which is a much bigger change to a game's margin than
  fumbles are — interceptions are far more common and carry more EPA each.
- **If penalties split** — pre-snap penalties (false start, delay of game) being
  skill while judgment calls (holding, pass interference) being closer to
  officiating noise — then the simulator needs a *partial* neutralization, not a
  single on/off switch per component.

## The calibration case

Fumble recovery is the known answer. League-wide recovery of a loose ball sits
near 50%, and team-level recovery rate has essentially no year-over-year or
split-half persistence. This is settled in the public football-analytics
literature.

We are not measuring it because it is unknown. We are measuring it because **it
tells us whether the method works.** If our split-half machinery says fumble
recovery is a skill, the machinery is broken and nothing else it reports can be
trusted. Every other component's answer is conditional on this one landing where
it should.

## Scope guard

Phase 1 writes no simulator code. It ends with a classification and the evidence
for it.
