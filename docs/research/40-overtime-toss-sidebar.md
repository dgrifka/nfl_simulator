# 40 — The overtime-toss sidebar: reported, not neutralized

*Written 2026-08-24. Product-layer round 2, item 4 of the queue in
`handoff-2026-08-23.md` §3. Production code: `src/nfl_simulator/plots.py`,
tests `tests/test_plots.py` (17 new, written first — 94 in the file, 309 in the
suite). Driver: `research/57_overtime_sidebar.py`, outputs
`research/outputs/57_overtime_<game_id>_{bootstrap,waterfall}.png` and
`57_overtime_sidebar.json` — gitignored, as always; this document is the record
of the numbers.*

**Nothing is fitted and nothing new is measured.** Every figure the panel prints
is a quotation from document 16, re-checked against that document's own impact
artifact before anything is drawn. The red-zone/late-down half of avenue 4 is
**not** here: it was absorbed into the parked placement meter (docs 34–36).

---

## 1. The answer, stated first

**The overtime coin toss is the one luck event the product has to talk about
without neutralizing, and now it does.** Document 16 measured the toss at 2.05
points of final margin (89% interval +1.04 to +3.07) and then refused to ship
it: gate O-3 asked whether neutralizing it moved the median overtime game by
more than the 4.06 pp interval the product already prints, and the answer was
3.93 pp. A reader looking at an overtime game's figure would otherwise have no
way to know the simulator saw the toss at all.

`attach_overtime_sidebar` puts document 16's finding beside either figure — the
bootstrap distribution or the luck-ledger waterfall — under the heading
**"Overtime — reported, not neutralized"**. A game that did not go to overtime
gets **no panel at all**, which is why documents 37 and 38's three examples
carry none.

## 2. What the panel is required to say

Three of its paragraphs are not editorial choices; they are obligations
document 16's defect register (§6) puts on anyone who reports this component.

| Obligation | Source | Where it lands |
|---|---|---|
| First possession is a **proxy** for winning the toss — nflverse has no coin-toss field | §6, "the most serious" open defect | Paragraph 2 |
| The swing is a **league average**; 155 games cannot estimate which offenses gain more from receiving | §6, status "stated wherever the component is reported" | Paragraph 4 |
| The 2025 rulebook **cannot be separated** from the earlier ones — power 0.243 against total removal, revisit at 60 new-rule games | §4d, pre-registered so no era reading can be made after the fact | Paragraph 6, on 2025-season games only |

The other three paragraphs name who received, state the swing with its interval
(never bare, as everywhere else in this module), and say the component was
**measured and refused** rather than overlooked — 3.93 pp against a 4.06 pp
floor, changing the deserved winner in 14 of 155 overtime games.

## 3. Reproduction of document 16 §8

The driver recomputes the impact run's headline numbers from
`26_overtime_games.parquet` and compares them to the strings the panel prints.
A disagreement is a `SystemExit`: a caveat quoting a figure its own artifact no
longer supports is worse than no caveat.

| Quantity | Recomputed | Panel prints | Document 16 |
|---|---|---|---|
| Overtime games, 2016–2025 | 155 | 155 | 155 (§3) |
| Median \|ΔDTW\| | 3.93 pp | 3.93 pp | 3.93 pp (§8, gate O-3) |
| Games whose deserved side flips | 14 of 155 | 14 of 155 | 14 / 155 (§8) |

The three fixed example games are also checked against the overtime list —
none of them went to overtime, so their sidebar-less figures in documents 37
and 38 are correct rather than incomplete.

## 4. The two examples

Both are **chosen, not typical**, and each was chosen to exercise something.

| Game | Headline | Bucket | Received first | ΔDTW (home share, v1.1) | Replay gap |
|---|---|---|---|---|---|
| `2016_14_NYJ_SF` | SF 64% / NYJ 36% | clear flip | SF (home) | **−21.4 pp** | 0.00e+00 |
| `2025_13_DEN_WAS` | DEN 86% / WAS 14% | scoreboard holds | DEN (away) | +13.6 pp | 0.00e+00 |

`2016_14_NYJ_SF` is the **largest per-game move in the whole window** — the game
where reporting rather than neutralizing the toss matters most, and one whose
deserved side document 16's component would have flipped. `2025_13_DEN_WAS` is a
**regular-season game under the 2025 rulebook**, which is the only case where
the panel has to add the era paragraph, and its favoured team is the **away**
side, which exercises the mirror described in §5.

Both games replay to the committed `dtw_games_v13.parquet` summary at exactly
0.00e+00 under v1.3's shipped settings (seed 20260817, 200 posterior draws, 800
coin draws), so the draws behind the distribution belong to the published
number.

## 5. Presentation decisions, with reasons

- **Silence when there was no overtime.** `toss=None` draws nothing and returns
  `None`. A panel explaining that this game had no overtime toss would put a
  caveat where there is no event, and would appear on 94% of games.
- **The figure grows to the right; the plot does not shrink.** A sidebar that
  squeezed the axes would re-scale a distribution in order to say something
  beside it, and two games annotated differently would then be drawn at two
  different scales. `attach_overtime_sidebar` widens the figure and re-pins the
  host axes to the inches it was drawn at.
- **The panel spans the figure's height, not the host axes' box.** The
  waterfall's height grows with its row count, so a note pinned to that box
  would start lower on a game with more luck events.
- **No colour.** A caveat is not an entity. The two team hues carry identity in
  these figures, and a third colour beside them would read as a third side; the
  panel is ink and muted ink, separated from the plot by a single hairline rule
  rather than a box.
- **The per-game move is mirrored onto the team the headline names.** The stored
  ΔDTW is on the *home* share. Printed beside a headline naming the away team it
  would hand one team's movement to the other — the same correction
  `interval_note` already makes for the interval.
- **The per-game move carries its simulator version.** Document 16's impact run
  is **v1.1**; the share above it is **v1.3**. Without the version the two
  numbers look subtractable, and they are not — the panel says the figure
  *sizes* the toss rather than correcting the share.
- **The interval caveat is kept to the plot's own column.** *(Revised
  2026-08-24 — §7a. It originally ran the full width of the widened figure, as a
  footnote to the figure rather than to the plot, and on a six-paragraph panel
  the two overprinted.)* It is a footnote to the plot, wrapped to the host axes'
  width, so widening the figure for a sidebar cannot widen the caveat with it.

## 6. What this does and does not license

It licenses saying, of a specific overtime game, that the toss was worth about
two points of margin to the team that received and that the simulator has
deliberately not removed it.

It does **not** license subtracting the per-game pp figure from the printed
share — different simulator version, and the swing is a league average rather
than this game's own. It does not license any reading of the 2025 rulebook,
which stays unanswerable until 60 new-rule overtime games exist. And it changes
nothing about the ledger: document 05 §3's treatment table stands exactly as
document 16 left it.

## 7. Known-defect register

| Defect | Evidence | Status |
|---|---|---|
| **The per-game ΔDTW is a v1.1 number beside a v1.3 share** | Document 16 §8's impact run predates v1.3 | **Open, disclosed in the panel itself.** Closing it means re-running the impact arm on v1.3, which is a measurement round and not a presentation one |
| **`received` is first possession, not the toss** | Inherited from document 16 §6 | **Open.** Named in the panel rather than silently carried |
| **The panel can overflow a short waterfall** | Six paragraphs need about 2.7 in; a two-event waterfall is 2.6 in tall | **Accepted.** `bbox_inches="tight"` includes the overflow rather than clipping it; the figure is merely bottom-heavy |
| **The example games are chosen for what they exercise** | §4 | **Disclosed.** Neither is a typical overtime game and neither is offered as one |
| **The interval caveat ran under the panel on a six-paragraph sidebar** | `57_overtime_2025_13_DEN_WAS_bootstrap.png` | **Fixed 2026-08-24** (addendum below) |

### 7a. Addendum, 2026-08-24 — the caveat and the panel

The caveat used matplotlib's `wrap=True`, which wraps to the **figure's** edge.
`attach_overtime_sidebar` widens the figure, so the caveat widened with it and
ran the full width of the widened figure — under the panel's last two
paragraphs on `2025_13_DEN_WAS`, whose 2025-season era paragraph makes six.
`2016_14_NYJ_SF`, one paragraph shorter, cleared it by luck rather than by
design.

The figure's width is not a boundary this module controls, so the caveat no
longer wraps to it. `_wrap_to_width` measures each candidate line against the
renderer and breaks it to the **host axes'** width, which `attach_overtime_sidebar`
already holds at the inches it was drawn at. The footnote is therefore laid out
before the figure ever grows, and cannot follow it. This reverses §5's original
bullet, which is marked as revised rather than quietly rewritten.

Two tests measure it: the six-paragraph panel and the five-paragraph one, each
asserting the caveat's bounding box does not intersect the union of the panel's
axes and its own text — the text is measured too, because a long panel overflows
the axes it sits in (the third row of §7's register). Both examples regenerate
with their replays still at 0.00e+00, and the caveat now sets in two lines
inside the plot's column on every figure in documents 37 and 40.

## 8. Verification

```
uv run pytest -q                      309 passed
uv run ruff check . && ruff format --check .   clean
uv run python research/57_overtime_sidebar.py  reproduction ok, both replays 0.00e+00
```

After the 2026-08-24 layout fix (§7a, and document 37 §7a): **318 passed** (nine
new, written first), ruff clean, all four drivers re-run — `54`/`57` replays
0.00e+00, `55` reconciliation 7.11e-15 across 2,761 games.
