"""Phase 3, step 5 — power calculation for the DTW interval-coverage check.

Runs **before** `docs/research/10-interval-coverage.md` commits any threshold.

Document 07 closed by listing what its validation did *not* establish, and this
was on the list: *"That the DTW credible interval has correct coverage. Gate 1
scores a point estimate. Coverage is a separate question and is not answered
here."* This round answers it.

The question a power calculation has to settle first is not "how precise is a
proportion" — that is arithmetic — but **"would this check actually catch a
broken simulator?"** So the instrument runs two arms:

* **healthy** — the shipped two-layer bootstrap, exactly as `simulate_game`
  calls it.
* **layer-1 disabled** — the same code with each event's posterior collapsed to
  its mean, so only the coin flip varies. This is the specific failure document
  05 §4 built layer 1 to prevent: *"Layer 1 is what stops the simulator from
  reporting a suspiciously tight interval around a quantity estimated from 15
  fumbles per team-season."*

If the check cannot separate those two, it is not a check.

    uv run python research/16_coverage_power.py
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from nfl_simulator import paths
from nfl_simulator.simulator import ETI_HIGH, ETI_LOW, LuckEvent, bootstrap_margins

RANDOM_SEED = 20260817

N_GAMES = 2000
N_POSTERIOR_DRAWS = 200
N_COIN_DRAWS = 100

# Coin draws used to compute the *truth*. Large, because the true DTW% is a
# population quantity and any noise in it is noise in the thing being covered.
N_TRUTH_DRAWS = 20000

NOMINAL_COVERAGE = 0.89
POINTS_PER_EPA = 0.8389


def event_pool() -> pl.DataFrame:
    """Real ledger rows, so synthetic games have realistic swings and rates.

    Using the shipped ledger rather than invented numbers means the check is run
    at the swing magnitudes and probabilities the simulator actually books —
    a coverage result at implausible inputs would not transfer.
    """
    ledger = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / "dtw_ledger_v11.parquet")
    # `n` is the evidence count behind each event's rate. Fumble classes are
    # observed thousands of times; a kicker-season is observed ~30 times. These
    # are the two regimes the interval has to be calibrated across.
    return ledger.select(
        "component",
        pl.col("expected").alias("p_true"),
        pl.col("swing"),
    ).with_columns(
        pl.when(pl.col("component") == "fumble")
        .then(pl.lit(900.0))
        .when(pl.col("component") == "field_goal")
        .then(pl.lit(29.0))
        .otherwise(pl.lit(31.0))
        .alias("evidence_n")
    )


def events_per_game() -> np.ndarray:
    """Empirical distribution of luck events per game, from the shipped run."""
    games = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / "dtw_games_v11.parquet")
    return games["n_luck_events"].to_numpy()


def one_game(
    pool: pl.DataFrame,
    counts: np.ndarray,
    rng: np.random.Generator,
    *,
    disable_layer_1: bool,
    n_coin_draws: int = N_COIN_DRAWS,
) -> tuple[float, float, float, int] | None:
    """One synthetic game. Returns (truth, interval low, interval high, n events).

    The true DTW% is computed at the **true** per-event probabilities with a large
    number of coin draws. The estimated interval is computed from *posteriors*
    for those probabilities, formed from a finite observed record exactly as the
    simulator forms them — Jeffreys, which is what `_class_rate_draws` uses.
    """
    n_events = int(rng.choice(counts))
    if n_events == 0:
        return None

    rows = pool.sample(n_events, with_replacement=True, seed=int(rng.integers(1 << 31)))
    p_true = rows["p_true"].to_numpy()
    swing = rows["swing"].to_numpy() * rng.choice([-1.0, 1.0], size=n_events)
    evidence_n = rows["evidence_n"].to_numpy()

    realized = (rng.random(n_events) < p_true).astype(float)
    actual_margin = float(rng.normal(0.0, 13.7))

    # ---- truth: the DTW% you would report if you knew every p exactly -------
    replayed = (rng.random((N_TRUTH_DRAWS, n_events)) < p_true[None, :]).astype(float)
    truth_margins = (
        actual_margin
        - ((realized[None, :] - replayed) * swing[None, :]).sum(axis=1) * POINTS_PER_EPA
    )
    truth = float((truth_margins > 0).mean())

    # ---- estimate: what the simulator reports from a finite record ----------
    observed_successes = rng.binomial(evidence_n.astype(int), p_true)
    events = []
    for i in range(n_events):
        draws = rng.beta(
            observed_successes[i] + 0.5,
            evidence_n[i] - observed_successes[i] + 0.5,
            size=N_POSTERIOR_DRAWS,
        )
        if disable_layer_1:
            # The failure mode being tested: collapse the posterior to its mean,
            # so the reported interval carries only coin-flip noise.
            draws = np.full(N_POSTERIOR_DRAWS, draws.mean())
        events.append(
            LuckEvent(
                play_id=float(i),
                component="synthetic",
                event_class="synthetic",
                charged_team="HOM",
                actual=realized[i],
                expected_draws=draws,
                swing=float(swing[i]),
            )
        )

    _, dtw_per_draw = bootstrap_margins(events, actual_margin, POINTS_PER_EPA, n_coin_draws, rng)
    return (
        truth,
        float(np.percentile(dtw_per_draw, ETI_LOW)),
        float(np.percentile(dtw_per_draw, ETI_HIGH)),
        n_events,
    )


def run_arm(
    label: str,
    n_games: int,
    seed: int,
    *,
    disable_layer_1: bool,
    n_coin_draws: int = N_COIN_DRAWS,
) -> dict:
    pool = event_pool()
    counts = events_per_game()
    rng = np.random.default_rng(seed)

    results = []
    for _ in range(n_games):
        outcome = one_game(
            pool, counts, rng, disable_layer_1=disable_layer_1, n_coin_draws=n_coin_draws
        )
        if outcome is not None:
            results.append(outcome)

    truth = np.array([r[0] for r in results])
    low = np.array([r[1] for r in results])
    high = np.array([r[2] for r in results])
    n_events = np.array([r[3] for r in results])

    covered = (truth >= low) & (truth <= high)
    # A game whose true DTW% is 0 or 1 and whose interval is degenerate is
    # covered trivially. Reported separately so the headline is not inflated.
    informative = (truth > 0.001) & (truth < 0.999)

    coverage = float(covered.mean())
    se = float(np.sqrt(coverage * (1 - coverage) / len(covered)))
    report = {
        "label": label,
        "n_games": int(len(covered)),
        "coverage": coverage,
        "monte_carlo_se": se,
        "coverage_ci95": [coverage - 1.96 * se, coverage + 1.96 * se],
        "coverage_informative_only": float(covered[informative].mean()),
        "n_informative": int(informative.sum()),
        "mean_interval_width": float((high - low).mean()),
        "mean_events_per_game": float(n_events.mean()),
        "n_coin_draws": int(n_coin_draws),
    }
    print(
        f"  {label:24s} coverage {coverage:.4f} +/- {1.96 * se:.4f}  "
        f"(informative only {report['coverage_informative_only']:.4f} on "
        f"{report['n_informative']} games)  mean width {report['mean_interval_width']:.4f}"
    )
    return report


def main() -> None:
    paths.ensure_data_dirs()
    print(f"=== Interval-coverage instrument, {N_GAMES} synthetic games per arm ===")
    print(f"    nominal coverage {NOMINAL_COVERAGE:.0%}\n")

    healthy = run_arm("healthy (two layers)", N_GAMES, RANDOM_SEED, disable_layer_1=False)
    broken = run_arm("layer 1 disabled", N_GAMES, RANDOM_SEED + 1, disable_layer_1=True)

    separation = healthy["coverage"] - broken["coverage"]
    pooled_se = float(np.sqrt(healthy["monte_carlo_se"] ** 2 + broken["monte_carlo_se"] ** 2))
    print(
        f"\n  separation between arms: {separation:+.4f} "
        f"({separation / pooled_se:.1f} standard errors)"
    )
    print(
        f"  Monte Carlo SE at {N_GAMES} games: {healthy['monte_carlo_se']:.4f} "
        f"({healthy['monte_carlo_se'] * 100:.2f} pp)"
    )

    # A threshold has to sit between the two arms to be worth committing.
    midpoint = (healthy["coverage"] + broken["coverage"]) / 2
    print(
        f"\n  A pass threshold anywhere between {broken['coverage']:.3f} and "
        f"{healthy['coverage']:.3f} separates them.\n"
        f"  Midpoint {midpoint:.3f}."
    )

    # ---- mechanism: is the excess width Monte Carlo noise? ----------------
    #
    # `dtw_per_draw` is itself an average over a FINITE number of coin flips, so
    # its spread across posterior draws mixes two things: genuine uncertainty
    # about p (which belongs in the interval) and Monte Carlo noise from using
    # only n_coin_draws flips (which does not). If the over-coverage is the
    # second, raising the coin count must shrink it — and that makes the fix a
    # parameter rather than a redesign.
    print("\n=== Coverage as a function of coin draws per posterior draw ===")
    sweep = []
    for coins in (25, 100, 400, 1600):
        arm = run_arm(
            f"coin draws = {coins}",
            600,
            RANDOM_SEED + 20 + coins,
            disable_layer_1=False,
            n_coin_draws=coins,
        )
        sweep.append(arm)

    results = {
        "coin_draw_sweep": sweep,
        "n_games": N_GAMES,
        "n_posterior_draws": N_POSTERIOR_DRAWS,
        "n_coin_draws": N_COIN_DRAWS,
        "n_truth_draws": N_TRUTH_DRAWS,
        "nominal_coverage": NOMINAL_COVERAGE,
        "healthy_arm": healthy,
        "layer_1_disabled_arm": broken,
        "separation": separation,
        "separation_in_standard_errors": separation / pooled_se,
        "random_seed": RANDOM_SEED,
    }
    out = paths.RESEARCH_OUTPUT_DIR / "16_coverage_power.json"
    with out.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
