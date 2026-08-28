# 58 — Amendment A-3 enacted: ruling R-4, editions, and two reported sensitivities

*Written 2026-08-28 in a Fable 5 brainstorm after document 57. Every computed
gate of amendment A-3 passed in rounds 5–7; this document records the maintainer's
ruling on the one open question, enacts the amendment, names the two
editions, and pre-registers two sensitivity checks that are **reported, not
gated** — the round cannot un-enact on their result, but the write-up must
carry whatever they show.*

## 1. Ruling R-4 (the maintainer, 2026-08-28)

**The gates govern.** Document 52 §3's preamble — "whose finish is
near-random" — is rewritten to: *"whose outcome the entity's persistent
skill explains only a small, measured share of."* Under that reading a
1-in-20 drop with a 0.088% persistent share qualifies, as does a coin-flip
finish with a 1.4% share. Gate C-2 failing at the charged grain (receiving
corps genuinely differ) is not a bar: clause 2 already prices each event at
the corps' shrunk posterior rate, so a good corps keeps its edge.

**A-3 is enacted for the hands-on-the-ball class, both directions.**
Document 05 §3's two rows read `partial (A-3)`. Document 52 §8 records the
enactment with this document's commit.

## 2. Editions and names

| Edition | Public name | Seasons | Ledger | `SimulationResult.variant` |
|---|---|---|---|---|
| Strict | **Strict** | 2016–2025 | fumbles, FG, XP | `"strict"` |
| Hands-on-the-ball | **Full** | 2022–2025 | Strict + dropped picks + receiver drops | `"full"` |

- **Full is the headline** on every 2022+ game; Strict is shown beside it.
  A pre-2022 game has Strict only and says so. Product rendering of the
  labels is figure round 6; this round changes the strings and the API.
- The `+dp`-only and `+rd`-only arms remain callable for audits but have no
  public name and never render.
- Both names are the maintainer's to change in the figure round; the strings above
  are what the code carries.

## 3. Two sensitivities, reported

**S-1 — capped swing on receiver drops.** Re-run the `+rd` and `full` audits
with each drop's swing winsorised at the 95th percentile of the swing
distribution (printed; expected ≈ 5 EPA) and again capped at the dropped-
pick table's largest cell (5.04 EPA). Report bucket moves, median and mean
|ΔDTW|, and the 89% tail, beside the uncapped run. **Pre-committed
reading:** if bucket moves under the 95th-percentile cap are ≥ 80% of
uncapped, the 14% is the event, not the dropped touchdowns, and the
uncapped pricing ships; if under 80%, the uncapped pricing still ships (it
is the play's own counterfactual and document 56 §2 pre-registered it) but
the write-up states that the tail is dropped touchdowns and the share of
movement they carry.

**S-2 — the contested/uncatchable charting link.** On all targets
2022–2025: `p(catchable | contested)` vs `p(catchable | uncontested)`, and
among *incompletions*, the share charted uncatchable by contest status.
Report, with one sentence on direction: if contested incompletions are
charted uncatchable far more often, the drop frame is conditioned on a
charter judgement that partly encodes the outcome, and every drop-skill
spread in document 57 is a floor. No change to the component follows in
this round; the caveat goes on the row in document 05 §3 and into the
write-up.

## 4. What ships in this round

- Code: `variant` strings `"strict"` / `"full"` (and the audit-only
  `"strict+dp"`, `"strict+rd"`); a `simulate_game(..., edition="full")`
  convenience that loads both models; `render._simulation_context` exposes
  both. No figure changes. All existing tests updated for the strings; V-1
  replay 0.00e+00.
- Docs: document 52 §3 preamble rewritten with R-4 cited; §8 enactment;
  document 05 §3 rows to `partial (A-3)` with the S-2 caveat if it fires;
  document 31's successor note pointing to this document; CLAUDE.md
  "Project status" paragraph gains one sentence on the two editions.
- The queue, the results file, the log.

## 5. Not in this round, deliberately

Product rendering of the editions (figure round 6); the docx update
(rounds 4–8); the community write-up; the spelling/publication scrub
(`docs/pre-publication-checklist.md` §5 — note `defence_season` and
similar identifiers are renamed then, not now, so the audit scripts keep
reproducing).

---

## 6. Outcome

**Round 8 ran 2026-08-28 on `feat/dropped-pick-variant`, and everything §4
listed shipped.** The record is **document 59**; the results file is
`results-2026-08-28-exp8.md`; the log is `log-2026-08-27-dropped-picks.md`
under "Round 8". Still **unmerged** — the maintainer merges.

- **The enactment.** Document 52 §3's preamble carries R-4's wording with the
  ruling quoted above it; §8 carries the enactment block and this round's
  commits. Document 05 §3's two rows read `partial (A-3)`. Document 31 gained a
  successor note; `CLAUDE.md` names both editions.
- **The code.** `"strict"`, `"full"`, `"strict+dp"`, `"strict+rd"`;
  `SimulationResult.edition` returns the public name and `None` for the two
  audit arms; `simulate_game(..., edition=...)` switches over the model handles;
  `render.edition_handles` and `_simulation_context["editions"]` expose both.
  586 → **594 tests**, ruff clean, **V-1 0.00e+00** over 2,761 games.
- **S-1 — the reading holds.** `+rd` bucket moves under the 95th-percentile cap
  (4.02 EPA) are **157 of 162, 96.9%** of uncapped, against the 80% bar. The 14%
  is the event, not the dropped touchdowns, and the uncapped pricing ships.
  `full` behaves the same: 192 of 200, 96.0%.
- **S-2 — the two halves disagree and the pre-registered trigger does not
  fire.** Marginally a contested target is charted catchable **66.8%** against
  **74.4%** uncontested, the direction the mechanism predicts; but §3's actual
  trigger, the incompletion split, runs the other way — contested incompletions
  are charted uncatchable **62.3%** against **81.1%**. Document 05 §3's `Drops`
  row therefore carries this as an **open caveat**, not as a measured floor.
  Nothing in the component changed, as §3 pre-committed.
