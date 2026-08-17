# nfl_simulator

An NFL **deserve-to-win** simulator: re-adjudicate a single game by neutralizing
luck rather than replaying plays.

## The idea

Most single-game outcomes mix two things: what a team *did* (skill, and the
choices that flow from it) and what *happened to them* (which way a loose ball
bounced, whether a 48-yard field goal drifted inside the upright). This repo
tries to separate them.

The approach is **luck-neutralized EPA accounting**, not play-splicing replay.
For each play whose outcome contains a coin flip, we replace the realized
Expected Points Added with its expectation:

- a fumble on the ground becomes the average of the recovered/not-recovered
  branches rather than the branch that actually happened,
- a field-goal attempt becomes make-probability-weighted points rather than
  3 or 0,
- and so on for the other components the EDA classifies as luck.

Re-bootstrapping those coin flips many times gives a distribution over margins —
a "deserve-to-win" probability for the game that actually got played.

Full play-level re-simulation (modeling play calling, drive continuation, clock
management) is explicitly **out of scope**.

## Status

**Phase 3 — the luck accounting is complete.** Phase 1 classified the
components; Phase 2 turned that classification into a working simulator under a
single rule; Phase 3 tested every remaining candidate and closed the two defects
a public product would have exposed.

**Every candidate is now neutralized, kept, or marked unresolvable with the
power table that says why. There are no pending rows left.**

```python
result = simulate_game(
    plays, fumble_baseline=..., fg_baseline=..., fg_model=..., points_per_epa=0.62
)
result.dtw_home  # 0.551  — deserve-to-win probability
result.dtw_interval  # (0.460, 0.631)
result.deserved_margin  # -0.37  (the team actually lost by 7)
result.ledger  # itemized: which fumble, which kick, how much
```

Working notes live in [`docs/research/`](docs/research/):

| Doc | What's in it |
|---|---|
| [00 — Business context](docs/research/00-business-context.md) | The decision this informs, and why fumble recovery is the calibration case |
| [01 — Descriptive EDA](docs/research/01-descriptive-eda.md) | Data audit, the EPA component decomposition, variance shares, bias assessment |
| [02 — Skill vs luck](docs/research/02-skill-vs-luck.md) | Split-half persistence per component; out-of-sample prediction test |
| [03 — Model foundations](docs/research/03-model-foundations.md) | The hierarchical models and their **pre-registered** gates, committed before any fit |
| [04 — Bayesian results](docs/research/04-bayesian-results.md) | Gate report (including two failures), population spreads, shrinkage |
| [05 — Neutralization principle](docs/research/05-neutralization-principle.md) | **The one rule**, the two gates, the per-component treatment table, and the attribution round |
| [05b — FG model foundations](docs/research/05b-fg-model-foundations.md) | The kicker-hierarchical make model, its pre-registered gates and its results |
| [06 — Rematch validation](docs/research/06-rematch-validation.md) | The validation design and its **power calculation**, committed before any result |
| [07 — Validation results](docs/research/07-validation-results.md) | Gate outcomes: non-inferiority passes |
| [08 — Sequencing luck](docs/research/08-sequencing-luck.md) | Is *where* production lands a team property? Plus the drive-outcome resampling that failed its gate |
| [09 — Coin-flip candidates](docs/research/09-coinflip-candidates.md) | Drops, fourth down, two-point, onside kicks, extra points — Gate A first, then the arithmetic |
| [10 — Interval coverage](docs/research/10-interval-coverage.md) | Does the 89% DTW interval mean 89%? It did not, and why |

### Headline findings

**Phase 1 — what is luck**

- The team with the higher game EPA differential wins **95.9%** of decided games.
  Of that differential's variance, **70%** is ordinary offense and defense and
  **19%** is interceptions; fumble recovery and field goals together are under 7%.
- **Fumble recovery is the only component with no detectable team skill.**
  Split-half correlation +0.055; the hierarchical model shrinks every team-season
  to within 45.6–48.3% regardless of whether it recovered 11% or 83% that year.
- The fumble coin is **class-specific**, not 50/50 — 40% on run plays, 76% on
  botched snaps. A flat coin would be wrong by up to 26 points.
- **Luck-stripping does not improve forward-looking prediction.** The simulator is
  a retrospective fairness measure, not a forecasting edge.

**Phase 2 — one rule, and what it does**

- **Luck is the realized outcome minus its expectation at the entity's shrunk
  posterior rate.** Full and partial neutralization are not two policies — they
  are the same expression read at two values of `w = n/(n+κ)`, and `w` is
  measured, not chosen.
- A component must pass a **branch-point gate** before that arithmetic runs.
  Penalties compute to `w ≈ 0.42–0.46`, which would neutralize half of every
  game's penalty EPA; only a mechanism story rules that out.
- **Offensive holding is not random.** The hypothesis that it is got a properly
  powered test (99.3% power on 916,700 plays) and failed decisively.
- **The interception spread cannot be attributed.** Quarterbacks (12.6%) and
  defenses (12.3%) come out indistinguishable, so interceptions stay untouched
  in v1 rather than being neutralized against a grain the data cannot support.
- **Kicker skill is real and sized**: a one-SD kicker makes **5.35 pp** more of
  their 45-yard attempts, giving field goals a genuinely entity-specific `w`
  from 0.064 to 0.377 depending on how much the kicker has kicked.
- Over ten seasons the simulator moves the mean margin by **2.80 points** and
  **disagrees with the actual result in 11.1% of games**.
- **Validation passes.** Neutralization does not degrade a game's predictive
  content (95% CI upper bound +0.0038 against a +0.010 margin) — and the
  observed statistic landed within a few percent of what the pre-registered
  power simulation predicted.

**Phase 3 — completing the accounting**

- **Where a team's production lands is luck; when it lands is skill.** Red-zone
  placement (r = −0.034) and third/fourth-down placement (r = +0.000) do not
  persist across the halves of a team's own season, on a test with **87–92%
  power** — so these are evidence of *absence*, not absence of evidence. But the
  gap between win-probability and expected-points value of the same plays does
  persist (**r = +0.180**), and survives a control for playing close games
  (+0.144). Teams differ, repeatably, in how much winning they extract from a
  fixed amount of production.
- **None of it becomes a ledger row.** Sequencing has no branch point, so
  Gate A rules it out at any value of `w` — a rule committed before the results
  existed, and it held in both directions.
- **Drops are not random.** Teams differ by **14.4% relative** on catchable-ball
  drop rate (86.5% power), against the folk claim. Also not a ledger row: a drop
  is a receiver's hands, not a coin.
- **Extra points join the ledger.** A branch point, and kickers genuinely differ
  (2.42 pp spread against a 1.84 pp null bound) — so they are partially
  neutralized at the kicker's shrunk rate, folded into the same kicker model.
- **Onside kicks are the honest denial.** A genuine branch point whose power is
  flat at **0.115** across a tenfold range of true spread. `w` cannot be
  measured, so it is not chosen.
- **Weather closed the FG model's largest defect.** A 15 mph wind costs **5.50
  pp** at 45 yards; a dome adds **4.53 pp**. 7,507 of 10,731 field-goal ledger
  entries were repriced — systematically, by roof. Kicker skill barely moved
  (σ 0.360 → 0.342), so the old model was not smuggling conditions into kickers.
- **The 89% interval was not an 89% interval.** It covered ~97% of informative
  games. The two-layer design was correct; the *coin-draw count* was too low, so
  a third of the reported width was Monte Carlo noise. Fixed by raising a
  constant — intervals are now 24.9% narrower and no verdict changed.

### Three process laws

Phase 1's Gate 2 failed because a threshold was set from a football-effect-size
argument with no power calculation behind it. Phase 2 treats two rules as
binding, and both changed real decisions:

1. **Pre-register before fitting.** Every gate doc lands in git before the
   script that fits its models — checkable with `git log --diff-filter=A`.
2. **Power-check every threshold before committing it.** This caught a
   field-goal calibration gate that would have failed 36% of the time on a
   correct model, and it converted the rematch validation from a superiority
   test the design could never pass into a non-inferiority test it can.
3. **Gate A before Gate B — mechanism before arithmetic.** No statistic can
   detect the *absence* of a branch point. In Phase 3 this disqualified six
   candidates before a model was fit, and the drops result shows why it matters
   in both directions: a 14.4% spread would have looked like an obvious skill
   finding, but had drops come back near zero the arithmetic would have said
   "neutralize" and the simulator would have started crediting teams for their
   receivers' hands.

Phase 3 produced **four pre-registered failures** — a drive-outcome resampling
that degraded the game's predictive content, a distance gate that a defect
register had already flagged, a convergence tolerance set by bad analogy, and an
interval that did not cover what it claimed. Each was reported rather than
tuned, and three of the four turned into a concrete fix.

## Data

Everything comes free via [`nflreadpy`](https://github.com/nflverse/nflreadpy):
play-by-play 2016–2025 and FTN charting 2022–2025. Pulls are cached to a
gitignored `data/` directory alongside a manifest recording seasons, pull date
and library version.

```bash
uv run python -m nfl_simulator.ingest          # cached; re-running is a no-op
uv run python -m nfl_simulator.ingest --force  # re-download
```

## Setup

```bash
uv venv
uv pip install -e '.[dev]'
uv run pytest
uv run ruff check .
```

## Layout

| Path | What's in it |
|---|---|
| `src/nfl_simulator/` | Importable package: ingest, validation, EPA decomposition, FG model, ledger, simulator |
| `research/` | Exploratory scripts — EDA, skill-vs-luck tests, Bayesian models, power calculations, validation |
| `docs/research/` | Written findings and working notes |
| `tests/` | pytest suite (network-free by default) |
| `data/` | Gitignored parquet cache + manifest |

## License

MIT
