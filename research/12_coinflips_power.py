"""Phase 3, step 3 — power calculation for the coin-flip candidate round.

Runs **before** `docs/research/09-coinflip-candidates.md` commits any threshold,
per the process law document 04 established and document 05 §7 formalized.

Each candidate is a rate with an opportunity denominator, so each gets the same
instrument document 05 §7 used: simulate datasets at the **real denominators**
under a known true population SD, fit the exact grid posterior, and record the
89% upper bound the fit produces. That answers the question document 04's Gate 2
never asked — *can this many observations reach the bound the threshold demands,
even when the truth is exactly zero?*

Note the direction of the test, which is the reverse of a skill hunt. A
coin-flip candidate is confirmed by showing its team spread is **small**, so the
gate has the form "the 89% upper bound is below X". Power is then the chance of
correctly **rejecting** that when a real effect exists — and a candidate whose
power is low cannot be neutralized on the strength of a pass, because it would
have passed anyway.

    uv run python research/12_coinflips_power.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from _betabinom_grid import fit_grid, simulate_counts  # noqa: E402

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.ingest import (  # noqa: E402
    FTN_SEASONS,
    PBP_SEASONS,
    load_ftn,
    load_pbp,
)

RANDOM_SEED = 20260817
DATASETS = 400  # matches document 05 §7's convention

# Relative population SDs to test power against. 12.5% is the reference: it is
# document 04's pooled-judgment-penalty figure, the yardstick document 05 §7
# already used for "an effect this project would call real".
RELATIVE_SCENARIOS = (0.05, 0.125, 0.25, 0.50)
REFERENCE_RELATIVE = 0.125
MIN_POWER = 0.80

# A receiver-season needs enough catchable balls to say anything at all. Below
# this the grain is a list of names, not a measurement.
MIN_RECEIVER_TARGETS = 20

PBP_COLUMNS = [
    "game_id",
    "play_id",
    "season",
    "posteam",
    "play_type",
    "desc",
    "receiver_player_id",
    "kicker_player_id",
    "fourth_down_converted",
    "fourth_down_failed",
    "two_point_attempt",
    "two_point_conv_result",
    "extra_point_attempt",
    "extra_point_result",
    "kickoff_attempt",
    "own_kickoff_recovery",
]


# --------------------------------------------------------------------------
# candidate denominators
# --------------------------------------------------------------------------


def _counts(frame: pl.DataFrame, keys: list[str], success: pl.Expr) -> pl.DataFrame:
    return (
        frame.group_by(keys)
        .agg(pl.len().alias("n"), success.sum().cast(pl.Int64).alias("k"))
        .drop_nulls()
        .sort(keys)
    )


def candidate_counts() -> dict[str, pl.DataFrame]:
    """(n, k) per entity for every coin-flip candidate, at the real denominators."""
    pbp = load_pbp(PBP_SEASONS, columns=PBP_COLUMNS)
    ftn = load_ftn(FTN_SEASONS)

    # --- drops: FTN charts catchability, so the denominator is catchable balls,
    # not targets. A ball nobody could have caught is not a drop opportunity.
    charted = (
        ftn.select(
            pl.col("nflverse_game_id").alias("game_id"),
            pl.col("nflverse_play_id").cast(pl.Float64).alias("play_id"),
            "is_catchable_ball",
            "is_drop",
        )
        .filter(pl.col("is_catchable_ball"))
        .join(
            pbp.select("game_id", "play_id", "season", "posteam", "receiver_player_id"),
            on=["game_id", "play_id"],
            how="inner",
        )
    )

    fourth_down = pbp.filter(
        (pl.col("fourth_down_converted") == 1) | (pl.col("fourth_down_failed") == 1)
    )
    two_point = pbp.filter(
        (pl.col("two_point_attempt") == 1) & pl.col("two_point_conv_result").is_not_null()
    )
    # nflverse carries no onside flag, so the play description is the only
    # identifier. It is a text match and it is recorded as such in the doc.
    onside = pbp.filter(
        (pl.col("kickoff_attempt") == 1) & pl.col("desc").str.to_lowercase().str.contains("onside")
    )
    extra_point = pbp.filter(
        (pl.col("extra_point_attempt") == 1) & pl.col("extra_point_result").is_not_null()
    )

    receiver = _counts(
        charted.drop_nulls("receiver_player_id"),
        ["season", "receiver_player_id"],
        pl.col("is_drop"),
    ).filter(pl.col("n") >= MIN_RECEIVER_TARGETS)

    return {
        "drops_team": _counts(charted, ["season", "posteam"], pl.col("is_drop")),
        "drops_receiver": receiver,
        "fourth_down": _counts(
            fourth_down, ["season", "posteam"], pl.col("fourth_down_converted") == 1
        ),
        "two_point": _counts(
            two_point, ["season", "posteam"], pl.col("two_point_conv_result") == "success"
        ),
        "onside_recovery": _counts(
            onside, ["season", "posteam"], pl.col("own_kickoff_recovery") == 1
        ),
        "extra_point_kicker": _counts(
            extra_point.drop_nulls("kicker_player_id"),
            ["season", "kicker_player_id"],
            pl.col("extra_point_result") == "good",
        ),
    }


# --------------------------------------------------------------------------


def power_table(name: str, counts: pl.DataFrame, seed: int) -> dict:
    """Null bound distribution and the power to reject it at real effect sizes."""
    n = counts["n"].to_numpy().astype(float)
    k = counts["k"].to_numpy().astype(float)
    league_rate = float(k.sum() / n.sum())
    max_sd = float(np.sqrt(league_rate * (1.0 - league_rate)))

    print(
        f"\n--- {name}: {counts.height} entities, {int(n.sum()):,} opportunities, "
        f"league rate {league_rate:.4%}, median n {np.median(n):.0f}"
    )

    started = time.time()
    null_bounds = np.array(
        [
            fit_grid(n, simulate_counts(rng, n, league_rate, 0.0)).summary()["population_sd_eti89"][
                1
            ]
            for rng in (np.random.default_rng(seed + i) for i in range(DATASETS))
        ]
    )
    threshold = float(np.percentile(null_bounds, 90))
    print(
        f"    null 89% upper bound: mean {null_bounds.mean() * 100:.4f} pp, "
        f"90th pct {threshold * 100:.4f} pp  ({time.time() - started:.0f}s)"
    )

    rows = []
    for relative in RELATIVE_SCENARIOS:
        true_sd = relative * league_rate
        if true_sd >= max_sd:
            print(
                f"    {relative:.1%} relative = {true_sd * 100:.2f} pp is impossible at a "
                f"{league_rate:.1%} rate (max {max_sd * 100:.2f} pp) — skipped"
            )
            rows.append(
                {
                    "relative": relative,
                    "true_sd_pp": true_sd * 100,
                    "power": None,
                    "impossible": True,
                }
            )
            continue
        bounds = np.array(
            [
                fit_grid(n, simulate_counts(rng, n, league_rate, true_sd)).summary()[
                    "population_sd_eti89"
                ][1]
                for rng in (
                    np.random.default_rng(seed + 1000 + int(relative * 1000) * 97 + i)
                    for i in range(DATASETS)
                )
            ]
        )
        power = float((bounds >= threshold).mean())
        rows.append(
            {
                "relative": relative,
                "true_sd_pp": true_sd * 100,
                "mean_upper_bound_pp": float(bounds.mean()) * 100,
                "power": power,
                "impossible": False,
            }
        )
        print(
            f"    true SD {relative:5.1%} relative ({true_sd * 100:.3f} pp): "
            f"correctly rejected {power:.3f} of the time"
        )

    reference = next((row for row in rows if row["relative"] == REFERENCE_RELATIVE), None)
    resolvable = bool(
        reference is not None and reference["power"] is not None and reference["power"] >= MIN_POWER
    )
    print(
        f"    => power at the {REFERENCE_RELATIVE:.1%} reference: "
        f"{'RESOLVABLE' if resolvable else 'UNRESOLVABLE'}"
    )

    return {
        "name": name,
        "entities": int(counts.height),
        "opportunities": int(n.sum()),
        "successes": int(k.sum()),
        "league_rate": league_rate,
        "median_n": float(np.median(n)),
        "null_bound_mean_pp": float(null_bounds.mean()) * 100,
        "gate_threshold_pp": threshold * 100,
        "power": rows,
        "power_at_reference": reference["power"] if reference else None,
        "resolvable": resolvable,
    }


def main() -> None:
    paths.ensure_data_dirs()
    counts = candidate_counts()

    print("=== Coin-flip candidate denominators ===")
    for name, frame in counts.items():
        n = frame["n"].to_numpy()
        print(
            f"  {name:20s} {frame.height:5d} entities  {int(n.sum()):7,} opportunities  "
            f"median n {np.median(n):6.1f}  rate {frame['k'].sum() / n.sum():.4%}"
        )

    results = {
        name: power_table(name, frame, RANDOM_SEED + 13 * index)
        for index, (name, frame) in enumerate(counts.items())
    }

    print("\n=== Summary: which candidates can this data resolve? ===")
    table = pl.DataFrame(
        [
            {
                "candidate": name,
                "entities": report["entities"],
                "opportunities": report["opportunities"],
                "league_rate": report["league_rate"],
                "gate_threshold_pp": report["gate_threshold_pp"],
                "power_at_12.5pct": report["power_at_reference"],
                "resolvable": report["resolvable"],
            }
            for name, report in results.items()
        ]
    )
    with pl.Config(tbl_cols=-1, fmt_str_lengths=24, tbl_width_chars=200):
        print(table)

    out = paths.RESEARCH_OUTPUT_DIR / "12_coinflips_power.json"
    with out.open("w") as handle:
        json.dump(
            {
                "datasets_per_scenario": DATASETS,
                "reference_relative": REFERENCE_RELATIVE,
                "min_power": MIN_POWER,
                "min_receiver_targets": MIN_RECEIVER_TARGETS,
                "random_seed": RANDOM_SEED,
                "candidates": results,
            },
            handle,
            indent=2,
        )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
