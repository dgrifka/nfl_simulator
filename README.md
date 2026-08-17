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

Working notes live in [`docs/research/`](docs/research/).

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
