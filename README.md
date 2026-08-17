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

**Phase 1 — scaffold + EDA.** No simulator code yet. Phase 1 answers the
prerequisite question empirically: *which EPA components are skill (they persist
for a team across games) and which are luck (they don't)?* Only the luck
components get neutralized, so this classification is the foundation everything
else sits on.

Working notes live in [`docs/research/`](docs/research/):

| Doc | What's in it |
|---|---|
| [00 — Business context](docs/research/00-business-context.md) | The decision this informs, and why fumble recovery is the calibration case |
| [01 — Descriptive EDA](docs/research/01-descriptive-eda.md) | Data audit, the EPA component decomposition, variance shares, bias assessment |
| [02 — Skill vs luck](docs/research/02-skill-vs-luck.md) | Split-half persistence per component; out-of-sample prediction test |
| [03 — Model foundations](docs/research/03-model-foundations.md) | The hierarchical models and their **pre-registered** gates, committed before any fit |
| [04 — Bayesian results](docs/research/04-bayesian-results.md) | Gate report (including two failures), population spreads, shrinkage |

### Headline findings

- The team with the higher game EPA differential wins **95.9%** of decided games.
  Of that differential's variance, **70%** is ordinary offense and defense and
  **19%** is interceptions; fumble recovery and field goals together are under 7%.
- **Fumble recovery is the only component with no detectable team skill.**
  Split-half correlation +0.055; the hierarchical model shrinks every team-season
  to within 45.6–48.3% regardless of whether it recovered 11% or 83% that year.
- The fumble coin is **class-specific**, not 50/50 — 40% on run plays, 76% on
  botched snaps. A flat coin would be wrong by up to 26 points.
- Field goals, penalties and interceptions all carry real, repeatable team skill,
  so none of them can simply be flipped.
- **Luck-stripping does not improve forward-looking prediction.** The simulator is
  a retrospective fairness measure, not a forecasting edge.

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
| `src/nfl_simulator/` | Importable package: ingest, validation, EPA decomposition |
| `research/` | Exploratory scripts — EDA, skill-vs-luck tests, Bayesian models |
| `docs/research/` | Written findings and working notes |
| `tests/` | pytest suite (network-free by default) |
| `data/` | Gitignored parquet cache + manifest |

## License

MIT
