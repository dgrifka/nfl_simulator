# 54 — A-3 gate G-1: fold-spec amendment, pre-registered

*Written 2026-08-27 in a Fable 5 brainstorm after document 53, **before the
folds are re-run**. Round 5 left G-1 blocked: 7 of 18 week-out fits missed
Gate C-1 on r̂ (worst 1.0146) and tail ESS (worst 245), with zero
divergences in all 18 — the chain-length failure amendment A-2 was written
for. Document 52 §7 fixed the fold spec at A-2's step counts; changing it
is a change to a pre-registered constant and is written here first.*

## 1. Amendments

**F-1 — sampler spec for the folds and the default fit.** 4 chains ×
**4,000 draws after 4,000 tuning**, `target_accept = 0.95`, nutpie. Applies
to every fold and to the default (in-sample) fit, so the two arms of G-1
are compared at the same spec. Gate C-1's bars are unchanged and apply to
every parameter of every fit. **If any fold still misses C-1, G-1 is
blocked again and the round stops — no third spec is chosen mid-round.**

**F-2 — a postseason fold.** A nineteenth fold holds out weeks 19–22
together (147 worthy throws, ~51 games), so every 2022–2025 game has a fit
that excluded its week. Seed `20260827 + 19`.

**F-3 — the in-sample arm is re-fit at F-1's spec** and the round-4 audit
is re-run on it before G-1 is computed, so the agreement statistic compares
like with like. Its bucket-move count and median |ΔDTW| are reported
against round 4's (137; 1.62 pp); they are expected to reproduce within
sampler noise and any drift beyond ±5 games or ±0.2 pp is a surprise to
record, not a result to interpret.

## 2. Everything that does not change

G-1's statistic and bars (bucket agreement ≥ 0.90; median |ΔDTW| between
arms < 1.0 pp on affected games), G-2 and G-3's verdicts (PASS, document
53), R-3, the readings in document 52 §5, clause 3's mirror requirement
(G-4, round 7), V-1's replay line at the end of every run.

## 3. Pre-committed expectation

The default fit at A-2's spec sat at r̂ 1.0070 with 8,000 draws; doubling
to 16,000 draws should put every fold under 1.01 with tail ESS well over
400. Agreement is expected high — a defence-season loses at most one week's
~1–2 throws of ~22 per fold — so a G-1 *fail* would be the surprise, and
would mean the in-sample read is materially self-fulfilling on a handful of
high-leverage games. Either way the number is the deliverable.

## 4. Constants

| Constant | Old | New | Where |
|---|---|---|---|
| Draws / tune / `target_accept` | 2,000 / 2,000 / 0.9 | **4,000 / 4,000 / 0.95** | `research/67`, `69` |
| Folds | 18 (weeks 1–18) | **19** (+ weeks 19–22 together) | `69` |
| Fold seed | `20260827 + week` | unchanged; postseason fold uses `+ 19` | `69` |
