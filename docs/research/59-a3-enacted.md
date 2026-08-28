# 59 — Amendment A-3 enacted: the Strict and Full editions, and two sensitivities

*Round 8, 2026-08-28, on `feat/dropped-pick-variant`. Document 58 recorded
the maintainer's ruling and planned this round; this is what the round did. Nothing here
is a gate. Every computed condition amendment A-3 ever set passed in rounds 5–7,
the ruling settled the one thing no round could compute, and the two checks below
are **reported** — they change what the write-up says, never what the component
does.*

---

## 1. What was enacted, in plain words

**The simulator now has two editions, and both are named.** Until today the
adjudication had one shipped form — v1.3, the fumble, field-goal and extra-point
ledger — and two default-off components that nobody could publish because the
rule admitting them was not enacted. Ruling R-4 enacted it. The two forms are:

| Edition | Seasons | What is in the ledger | `variant` string |
|---|---|---|---|
| **Strict** | 2016–2025 | fumbles, field goals, extra points | `"strict"` |
| **Full** | 2022–2025 | Strict **plus** dropped picks and receiver drops | `"full"` |

**Strict is v1.3, unchanged and unchangeable by this round.** The rename is a
string. Every audit run in this round replays the 2,761 shipped games at
**0.00e+00** on the deserved margin, DTW% and both interval bounds — before the
sensitivities and again after them.

**Full is the two hands-on-the-ball components together, and only together.**
Amendment A-3 clause 3 makes defender drops (a dropped interception) and
receiver drops one class: they enter together or not at all. The one-direction
arms `"strict+dp"` and `"strict+rd"` stay callable so audits can decompose the
class, but they have no public name and `SimulationResult.edition` returns
`None` for them, so nothing can render an arm the maintainer did not name.

**What the ruling actually changed.** Document 52 §3's preamble said the class
covers an outcome "whose finish is near-random". A drop is a 1-in-20 event, not
a coin, so the wording and the arithmetic were in tension and only the maintainer could
resolve it. R-4 rewrites the preamble to *"whose outcome the entity's persistent
skill explains only a small, measured share of"* and states the consequence
plainly: **the gates govern.** Under that reading a 1-in-20 drop whose receiving
corps explain 0.088% of the variance qualifies, as does a coin-flip finish at
1.4%. Gate C-2 failing at the charged grain — receiving corps genuinely differ —
is not a bar, because clause 2 already prices every event at the corps' own
shrunk posterior rate, so a good corps keeps its edge. None of the numbered
clauses moved; the door was renamed, not widened.

**Where it is recorded.** Document 52 §3 carries the new preamble with the
ruling quoted above it and §8 carries the enactment; document 05 §3's two rows
read `partial (A-3)`; document 31 gained a successor note saying that v1.3 is
now called Strict; `CLAUDE.md`'s project status names both editions.

---

## 2. S-1 — the capped swing, and the reading it was pre-committed to

**The question.** The receiver component prices each catchable target at what
catching it was worth on that play, and that swing has a long right tail:
median **1.37 EPA**, mean 1.78, maximum **11.36**. Round 7 found that `+rd`
alone changes the verdict bucket on **14.2%** of 2022–2025 games. Is that the
*event* — a 1-in-20 coin, several dozen times a game — or is it the *dropped
touchdowns*, a handful of enormous swings doing the work?

**The check.** Both arms were re-run twice with the swing magnitude capped: at
the **95th percentile** of the swing distribution the production read side
actually prices (**4.02 EPA**, capping 5.0% of targets) and at **5.04 EPA**, the
largest cell of the dropped-pick swing table, so the two directions of the class
meet a common ceiling (capping 1.6%). The uncapped arms were checked against
document 57 §5 first — 162 and 200 bucket moves, 2.32 pp and 3.85 pp — and
reproduce them exactly.

| Arm | Bucket moves | Median \|ΔDTW\| | Mean \|ΔDTW\| | 89% tail |
|---|---|---|---|---|
| `+rd` uncapped | **162** | 2.32 pp | 8.17 pp | [0.00, 33.56] pp |
| `+rd` p95 cap (4.02) | 157 | 1.94 pp | 7.52 pp | [0.00, 30.92] pp |
| `+rd` 5.04 cap | 156 | 2.11 pp | 7.84 pp | [0.00, 32.15] pp |
| `full` uncapped | **200** | 3.85 pp | 10.56 pp | [0.00, 41.22] pp |
| `full` p95 cap (4.02) | 192 | 3.27 pp | 9.94 pp | [0.00, 38.60] pp |
| `full` 5.04 cap | 198 | 3.65 pp | 10.23 pp | [0.00, 40.24] pp |

*1,138 affected games in every arm.*

**The reading, pre-committed in document 58 §3 before the run.** Bucket moves
under the 95th-percentile cap on the `+rd` arm are **157 of 162 — 96.9% of
uncapped**, against a bar of 80%. So:

> **The 14% is the event, not the dropped touchdowns, and the uncapped pricing
> ships.**

Cutting the top 5% of swings down to 4.02 EPA costs five games out of 162. That
is the answer to round 7's second surprise: the tail is real and it dominates
the *mean* move, but the verdict changes are carried by the ordinary 1-in-20
coin happening forty-odd times a game, not by the two 9-EPA dropped touchdowns
in `2024_19_LAC_HOU`. The `full` arm behaves the same way (192 of 200, 96.0%).

---

## 3. S-2 — the charting link, and what it caveats

**The question.** Round 7's hindsight probe fired the wrong way: among catchable
balls, contested ones are dropped **less** often than uncontested ones (4.29% vs
5.04%), where document 56 §1 pre-committed that contested should be far higher.
The proposed mechanism was that a contested ball knocked away gets charted
**uncatchable**, so the contested balls surviving into the drop frame are the
cleanly catchable ones. If that is right, `is_catchable_ball` partly encodes the
outcome and every drop-skill spread in document 57 is a floor.

**What the data says, on all 73,519 charted targets of 2022–2025.**

| Quantity | Contested | Uncontested |
|---|---|---|
| p(charted catchable) | **66.8%** (n = 9,817) | **74.4%** (n = 63,702) |
| among incompletions, share charted **un**catchable | **62.3%** (n = 5,172) | **81.1%** (n = 19,836) |

**The two halves disagree, and the pre-registered one does not fire.**
Marginally, contest makes a ball 7.6 pp less likely to be judged catchable
(ratio 0.90×) — the direction document 57 §4's mechanism predicts. But document
58 §3's actual trigger was the incompletion split, and it runs the other way:
conditional on the ball not being caught, contested balls are charted uncatchable
**less** often, by 18.8 pp.

These are not in contradiction. A contested throw is far likelier to be
incomplete at all, so conditioning on an incompletion is a different comparison:
the uncontested incompletions are dominated by overthrows and throwaways, which
are uncatchable by any charter's reading. The consequence for the record is
narrow and it is the honest one:

> Contest is associated with the charter's catchability judgement, but the
> pre-registered evidence that the judgement encodes the **outcome** is absent.
> Document 57's drop-skill spreads are reported with this as an **open caveat**,
> not as a measured floor.

That sentence is now on document 05 §3's `Drops` row. Nothing in the component
changes; the uncapped, uncorrected pricing ships either way, as document 58 §3
pre-committed.

---

## 4. The numbers a write-up may quote

Each with the document that owns it. Nothing here may be quoted from memory —
every line traces to a script run in this repository.

| Number | Value | Document |
|---|---|---|
| Editions and their seasons | Strict 2016–2025; Full 2022–2025 | 58 §2, this document §1 |
| Strict is v1.3 byte for byte | V-1 **0.00e+00** over 2,761 games | 49–59, every audit run |
| Games the Full edition moves across a verdict bucket | **200 of 1,139 (17.6%)** | 57 §5 |
| — the dropped pick alone | 136 (11.9%) | 55, 57 §5 |
| — the receiver drop alone | 162 (14.2%) | 57 §5 |
| Median \|ΔDTW\| on affected games, Full | **3.85 pp** | 57 §5 |
| Materiality floor it clears | median 89% half-width **0.56 pp** | 52 §5 G-3; 57 §5 G-4d |
| Persistent skill share — dropped pick | ~**1.4%** per throw | 48 |
| Persistent skill share — receiving corps | **0.088%** per target | 57 §1b |
| Receiver-drop swing, per play | median **1.37 EPA**, mean 1.78, max 11.36 | 57 §4 |
| S-1: bucket moves under the p95 cap | **96.9%** of uncapped (157 of 162) | this document §2 |
| S-2: p(catchable) contested vs uncontested | **66.8%** vs **74.4%** | this document §3 |
| Deserved winner differs from the scoreboard (Strict) | 255 of 2,761 games (9.24%) | 33 |

**Two things a write-up must say, not may say.** First, that A-3 redefines
"deserve to win" toward *what would not have persisted*, and that the word
"luck" on a Full-edition game page therefore includes things players did
(document 52 §1, §6). Second, S-2's caveat above, wherever a receiver drop-skill
number appears.

---

## 5. Register

**Defects and open items carried forward.**

| Item | Status |
|---|---|
| Document 05 §5, 05b §7 and document 31 §9's registers | **Unchanged.** Nothing in this round touches the Strict arithmetic |
| `w = 0.285` has no derivation in this repository | **Open**, as in document 31 §9 |
| Gate C-2 fails at the receiver component's charged grain | **Open and reported.** Receiving corps genuinely differ; R-4 ruled it is not a bar, because clause 2 prices at the corps' own shrunk rate | 
| `is_catchable_ball` may be graded partly off the outcome | **Open caveat** — S-2 §3 above. Not measured as a floor, not dismissed |
| Document 56 §2's fallback clause was read one step wider than written | **Disclosed**, document 57 §4; unchanged here |
| The two editions are not yet rendered | **Open by design.** Figure round 6 puts the edition label on every image and makes Full the headline on 2022+ games |
| A-3 clause 7's sunset | **Open, scheduled.** The class is re-tested against clause 1 every season; a spread that becomes a cross-season trait reverts the row to `core` |

**Nothing in this round was a stop-and-ask.** Every existing test was updated by
a string change alone; the uncapped arms reproduced document 57 §5 exactly (162
and 200 moves, 2.32 pp and 3.85 pp); the swing distribution reproduced document
57 §4's median, mean and maximum; S-2's target frame reproduced document 57 §4's
48,511 completions; and V-1 replayed at 0.00e+00 at the start and the end of
every run.

**What this round deliberately did not do:** render anything (figure round 6),
update the docx (rounds 4–8), draft the community write-up, or run the spelling
and publication scrub (`docs/pre-publication-checklist.md` §5 — `defence_season`
and similar identifiers are renamed there, not here, so the audit scripts keep
reproducing).
