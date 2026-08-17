"""Step 3 — power checks for the attribution round's thresholds.

Runs before `docs/research/05-neutralization-principle.md` §7 commits any
threshold, per the process law from document 04: *power-check every gate
threshold before committing it*.

For each attribution question it asks the question document 04's Gate 2 failed
to ask — **can this many observations produce the upper bound the threshold
demands, even when the truth is exactly zero?**

    uv run python research/06_attribution_power.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from _betabinom_grid import upper_bound_distribution  # noqa: E402

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.ingest import (  # noqa: E402
    ANALYSIS_COLUMNS,
    FTN_SEASONS,
    PBP_SEASONS,
    load_ftn,
    load_pbp,
)

RANDOM_SEED = 20260817
DATASETS = 400

# The comparison alternative throughout: the relative spread document 04 measured
# for pooled judgment penalties. An attribution question is worth asking only if
# the design can tell "no skill" apart from "as much skill as the pooled class".
REFERENCE_RELATIVE_SPREAD = 0.125


def check(
    label: str,
    n: np.ndarray,
    league_rate: float,
    alternatives: list[float],
    seed: int,
) -> dict:
    """Null and alternative distributions of the 89% upper bound."""
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(
        f"  entities {len(n)},  median n {np.median(n):.0f},  total n {n.sum():,.0f},  "
        f"league rate {league_rate:.4%}"
    )

    null_bounds = upper_bound_distribution(n, league_rate, 0.0, datasets=DATASETS, seed=seed)
    # A threshold has to sit above where the null lands, or it fails on data
    # volume rather than on the hypothesis — document 04's exact mistake.
    achievable = float(np.percentile(null_bounds, 90))
    print(
        f"\n  under TRUE SD = 0, the 89% upper bound lands at "
        f"{null_bounds.mean() * 100:.4f} pp on average, "
        f"90th pct {achievable * 100:.4f} pp"
    )
    print(f"  --> any threshold below {achievable * 100:.4f} pp fails >10% of the time at truth 0")

    rows = []
    for true_sd in alternatives:
        bounds = upper_bound_distribution(n, league_rate, true_sd, datasets=DATASETS, seed=seed + 1)
        # Power = the alternative is correctly NOT passed as "no skill".
        power = float((bounds > achievable).mean())
        rows.append(
            {
                "true_sd_pp": true_sd * 100,
                "true_relative_spread": true_sd / league_rate,
                "mean_upper_bound_pp": float(bounds.mean()) * 100,
                "power_to_reject_no_skill": power,
            }
        )
        print(
            f"  true SD {true_sd * 100:7.4f} pp ({true_sd / league_rate:5.1%} relative) "
            f"-> mean upper bound {bounds.mean() * 100:7.4f} pp,  power {power:.3f}"
        )

    return {
        "label": label,
        "entities": int(len(n)),
        "median_n": float(np.median(n)),
        "total_n": float(n.sum()),
        "league_rate": league_rate,
        "null_mean_upper_bound_pp": float(null_bounds.mean()) * 100,
        "achievable_threshold_pp": achievable * 100,
        "alternatives": rows,
    }


def main() -> None:
    paths.ensure_data_dirs()
    columns = [*ANALYSIS_COLUMNS, "passer_player_id", "touchdown", "return_yards"]
    pbp = load_pbp(PBP_SEASONS, columns=columns)
    ftn = load_ftn(FTN_SEASONS)

    results = {}

    # ---- 3b: offensive holding, the explicit hypothesis --------------------
    plays = pbp.filter(pl.col("posteam").is_not_null() & pl.col("defteam").is_not_null())
    on_field = (
        pl.concat(
            [
                plays.select(
                    pl.concat_str(
                        [pl.col("season").cast(pl.String), pl.col("posteam")], separator="_"
                    ).alias("team_season")
                ),
                plays.select(
                    pl.concat_str(
                        [pl.col("season").cast(pl.String), pl.col("defteam")], separator="_"
                    ).alias("team_season")
                ),
            ]
        )
        .group_by("team_season")
        .agg(pl.len().alias("n"))
    )
    holding = pbp.filter((pl.col("penalty") == 1) & (pl.col("penalty_type") == "Offensive Holding"))
    holding_rate = holding.height / on_field["n"].sum()
    results["offensive_holding"] = check(
        "3b — offensive holding (hypothesis: no team skill)",
        on_field["n"].to_numpy(),
        holding_rate,
        [
            holding_rate * 0.05,
            holding_rate * REFERENCE_RELATIVE_SPREAD,
            holding_rate * 0.25,
        ],
        RANDOM_SEED,
    )

    # ---- 3a: interception conversion, per-quarterback grain ---------------
    charted = (
        ftn.select(
            pl.col("nflverse_game_id").alias("game_id"),
            pl.col("nflverse_play_id").cast(pl.Float64).alias("play_id"),
            "is_interception_worthy",
        )
        .filter(pl.col("is_interception_worthy"))
        .join(pbp, on=["game_id", "play_id"], how="inner")
    )
    qb = (
        charted.drop_nulls("passer_player_id")
        .group_by(["season", "passer_player_id"])
        .agg(pl.len().alias("n"), (pl.col("interception") == 1).sum().alias("k"))
    )
    qb_rate = qb["k"].sum() / qb["n"].sum()
    results["int_conversion_by_qb"] = check(
        "3a — INT conversion, quarterback-season grain",
        qb["n"].to_numpy(),
        float(qb_rate),
        [0.02, 0.05, 0.10],
        RANDOM_SEED + 10,
    )

    defense = charted.group_by(["season", "defteam"]).agg(
        pl.len().alias("n"), (pl.col("interception") == 1).sum().alias("k")
    )
    results["int_conversion_by_defense"] = check(
        "3a — INT conversion, defense-season grain",
        defense["n"].to_numpy(),
        float(defense["k"].sum() / defense["n"].sum()),
        [0.02, 0.05, 0.10],
        RANDOM_SEED + 20,
    )

    # ---- 3c: pick-six rate, the binary form of return yardage -------------
    ints = pbp.filter(pl.col("interception") == 1)
    pick_six = (
        ints.group_by(["season", "defteam"])
        .agg(pl.len().alias("n"), (pl.col("touchdown") == 1).sum().alias("k"))
        .filter(pl.col("n") > 0)
    )
    results["pick_six_rate"] = check(
        "3c — pick-six rate, defense-season grain",
        pick_six["n"].to_numpy(),
        float(pick_six["k"].sum() / pick_six["n"].sum()),
        [0.01, 0.02, 0.04],
        RANDOM_SEED + 30,
    )

    out = paths.RESEARCH_OUTPUT_DIR / "06_attribution_power.json"
    with out.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
