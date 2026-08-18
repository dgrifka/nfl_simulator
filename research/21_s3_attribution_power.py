"""Phase 4, step 3 — achievable-null bounds for the S3 attribution round.

Document 08 §9 found the one sequencing channel that is **skill**: the gap
between a team's win-probability contribution and its expected-points
contribution persists at split-half ``r = +0.180``, and survives a control for
playing close games at ``+0.144``. Teams differ, repeatably, in how much winning
they extract from a fixed amount of production.

Its defect register recorded what was left open:

> *"S3's mechanism is unidentified. Persistence survives the game-state control,
> but **why** teams differ — coaching, quarterback, situational scheme — is
> untested. A crossed QB x coach model is the natural next step and is out of
> Phase 3's scope."*

This script measures the design parameters and computes the **achievable-null
bound** for that model, before `docs/research/13-s3-attribution.md` commits any
reporting rule. It is the instrument document 05 §7 built for the interception
attribution round, rebuilt for a Gaussian response:

    400 datasets simulated at the REAL design — the real quarterbacks, the real
    coaches, the real number of games each pairing played — under a true
    population SD of exactly zero, each fitted, each contributing the 89% upper
    bound it produced. The bound is the 90th percentile of those.

That answers the question document 04's Gate 2 never asked: *can this design
distinguish a quarterback effect from nothing at all?*

    uv run python research/21_s3_attribution_power.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_grid = import_module("_crossed_gaussian_grid")
_seq = import_module("10_sequencing_power")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.ingest import PBP_SEASONS, load_pbp  # noqa: E402

RANDOM_SEED = 20260817
DATASETS = 400

# Percentile of the null distribution that becomes the bound. Document 05 §7 used
# the 90th for the same construction, so a genuinely null design clears it 10% of
# the time by definition.
NULL_PERCENTILE = 90

# True population SDs to power against, in S3's own units: win-probability points
# per team-game. 0.0318 is the tau document 08 §5 computed for its r = 0.12
# reference effect — the smallest sequencing spread this project would call real.
POWER_SCENARIOS = (0.010, 0.020, 0.0318, 0.050, 0.080)

# One-score games. Document 08 §9's exploratory control used the same cut, and
# the slope is refit inside the subset for the reason recorded there: leverage is
# uniformly higher in close games, so reusing the full-sample slope would leave a
# constant offset in every team's residual.
COMPETITIVE_SCORE_DIFFERENTIAL = 8

COLUMNS = [
    *_seq.COLUMNS,
    "passer_player_id",
    "passer_player_name",
    "home_team",
    "away_team",
    "score_differential",
]


def primary_passer(pbp: pl.DataFrame) -> pl.DataFrame:
    """The quarterback who threw the most passes for a team in a game.

    A team-game, not a team-season, is the unit — so a game started by a backup
    is attributed to the backup. That is the point: attributing a game to the
    listed starter when he did not play would put the wrong name on the effect
    being estimated, and quarterback changes are exactly the variation that
    identifies a quarterback effect apart from a coach one.
    """
    return (
        pbp.filter(pl.col("passer_player_id").is_not_null() & pl.col("posteam").is_not_null())
        .group_by(["season", "posteam", "game_id", "passer_player_id", "passer_player_name"])
        .agg(pl.len().alias("attempts"))
        .sort(["season", "posteam", "game_id", "attempts"], descending=[False, False, False, True])
        .group_by(["season", "posteam", "game_id"], maintain_order=True)
        .first()
        .drop("attempts")
    )


def head_coach(pbp: pl.DataFrame) -> pl.DataFrame:
    """Head coach per team-game, joined through the play-by-play's own team codes.

    `data/schedules.parquet` carries the coaches but records relocated franchises
    under their **historical** codes — SD, STL, OAK — while the play-by-play uses
    current ones. Joining on team code directly silently drops 81 team-games. So
    the home/away role is read from the play-by-play, which is internally
    consistent with `posteam`, and only the coach names come from the schedule.
    """
    roles = (
        pbp.filter(pl.col("posteam").is_not_null())
        .group_by("game_id")
        .agg(pl.col("home_team").first(), pl.col("away_team").first())
    )
    schedule = pl.read_parquet(paths.SCHEDULE_PATH).select("game_id", "home_coach", "away_coach")
    joined = roles.join(schedule, on="game_id", how="inner")
    return pl.concat(
        [
            joined.select(
                "game_id", pl.col("home_team").alias("posteam"), pl.col("home_coach").alias("coach")
            ),
            joined.select(
                "game_id", pl.col("away_team").alias("posteam"), pl.col("away_coach").alias("coach")
            ),
        ]
    )


def s3_dataset(pbp: pl.DataFrame, *, competitive_only: bool = False) -> tuple[pl.DataFrame, float]:
    """One row per team-game: the S3 leverage-timing gap, its quarterback, its coach.

    ``S3 = sum(wpa) - slope * sum(epa)`` over every play the team had the ball
    for, exactly as document 08 §3 defines it. Positive means the team's
    production moved win probability more than its point value implies.
    """
    valued = pbp.filter(
        pl.col("posteam").is_not_null() & pl.col("epa").is_not_null() & pl.col("wpa").is_not_null()
    )
    if competitive_only:
        valued = valued.filter(pl.col("score_differential").abs() <= COMPETITIVE_SCORE_DIFFERENTIAL)

    epa = valued["epa"].to_numpy()
    wpa = valued["wpa"].to_numpy()
    slope = float(np.cov(epa, wpa)[0, 1] / np.var(epa, ddof=1))

    per_game = valued.group_by(["season", "posteam", "game_id"]).agg(
        pl.len().alias("plays"),
        pl.col("epa").sum().alias("epa_sum"),
        pl.col("wpa").sum().alias("wpa_sum"),
    )
    return (
        per_game.join(primary_passer(pbp), on=["season", "posteam", "game_id"], how="inner")
        .join(head_coach(pbp), on=["game_id", "posteam"], how="inner")
        .with_columns((pl.col("wpa_sum") - slope * pl.col("epa_sum")).alias("s3"))
        .sort(["season", "game_id", "posteam"])
    ), slope


def encode(frame: pl.DataFrame) -> tuple[list[np.ndarray], list[int], list[list[str]]]:
    """Integer level codes for the two crossed grouping factors."""
    codes, sizes, levels = [], [], []
    for column in ("passer_player_id", "coach"):
        names = sorted(frame[column].unique().to_list())
        lookup = {name: i for i, name in enumerate(names)}
        codes.append(np.array([lookup[value] for value in frame[column].to_list()]))
        sizes.append(len(names))
        levels.append(names)
    return codes, sizes, levels


def null_bound(
    design: _grid.CrossedDesign,
    codes: list[np.ndarray],
    sizes: list[int],
    residual_sd: float,
    datasets: int,
    seed: int,
    *,
    sigma_a: float = 0.0,
    sigma_b: float = 0.0,
) -> dict:
    """Upper bounds this design produces at a known truth. Zero by default."""
    rng = np.random.default_rng(seed)
    uppers_a, uppers_b, lowers_a, lowers_b = [], [], [], []
    for index in range(datasets):
        y = rng.normal(0.0, residual_sd, design.n)
        if sigma_a > 0:
            y = y + rng.normal(0.0, sigma_a, sizes[0])[codes[0]]
        if sigma_b > 0:
            y = y + rng.normal(0.0, sigma_b, sizes[1])[codes[1]]
        result = _grid.fit(design, _grid.project(codes, sizes, y), float(y @ y), float(y.sum()))
        uppers_a.append(result["sigma_a"]["eti89_ub"])
        uppers_b.append(result["sigma_b"]["eti89_ub"])
        lowers_a.append(result["sigma_a"]["eti89_lb"])
        lowers_b.append(result["sigma_b"]["eti89_lb"])
        if (index + 1) % 100 == 0:
            print(f"    dataset {index + 1}/{datasets}", flush=True)
    return {
        "upper_a": np.array(uppers_a),
        "upper_b": np.array(uppers_b),
        "lower_a": np.array(lowers_a),
        "lower_b": np.array(lowers_b),
    }


def main() -> None:
    paths.ensure_data_dirs()
    pbp = load_pbp(PBP_SEASONS, columns=COLUMNS)

    report = {}
    for label, competitive in (("primary", False), ("competitive-only", True)):
        frame, slope = s3_dataset(pbp, competitive_only=competitive)
        codes, sizes, levels = encode(frame)
        y = frame["s3"].to_numpy()

        mobility_qb = frame.group_by("passer_player_id").agg(
            pl.col("coach").n_unique().alias("coaches"), pl.len().alias("games")
        )
        mobility_coach = frame.group_by("coach").agg(
            pl.col("passer_player_id").n_unique().alias("quarterbacks"), pl.len().alias("games")
        )
        design_summary = {
            "team_games": int(frame.height),
            "quarterbacks": sizes[0],
            "coaches": sizes[1],
            "wpa_per_epa_slope": slope,
            "s3_sd": float(np.std(y, ddof=1)),
            "s3_mean": float(np.mean(y)),
            "median_games_per_qb": float(mobility_qb["games"].median()),
            "median_games_per_coach": float(mobility_coach["games"].median()),
            "share_qbs_with_multiple_coaches": float((mobility_qb["coaches"] > 1).mean()),
            "share_coaches_with_multiple_qbs": float((mobility_coach["quarterbacks"] > 1).mean()),
        }
        print(f"\n{'=' * 72}\nDESIGN — {label}\n{'=' * 72}")
        for key, value in design_summary.items():
            print(
                f"  {key:34s} {value:.4f}" if isinstance(value, float) else f"  {key:34s} {value}"
            )
        print(
            "\n  Mobility is the identification. A quarterback effect is separable from a\n"
            "  coach effect only because quarterbacks change coaches and coaches change\n"
            "  quarterbacks; with no mobility the two factors would be the same partition."
        )

        design, _, _, _ = _grid.prepare(codes, sizes, y)
        residual_sd = design_summary["s3_sd"]

        if label == "primary":
            print(f"\n=== Achievable-null bound ({DATASETS} datasets, true SD exactly zero) ===")
            null = null_bound(design, codes, sizes, residual_sd, DATASETS, RANDOM_SEED)
            bounds = {
                "quarterback": float(np.percentile(null["upper_a"], NULL_PERCENTILE)),
                "coach": float(np.percentile(null["upper_b"], NULL_PERCENTILE)),
            }
            print(
                f"  quarterback null 89% upper bound, {NULL_PERCENTILE}th pct: "
                f"{bounds['quarterback']:.5f}"
            )
            print(
                f"  coach       null 89% upper bound, {NULL_PERCENTILE}th pct: {bounds['coach']:.5f}"
            )
            print(
                f"  (null lower bounds: quarterback median {np.median(null['lower_a']):.5f}, "
                f"coach median {np.median(null['lower_b']):.5f})"
            )

            print(f"\n=== Power ({DATASETS} datasets per scenario) ===")
            print("    power = P(the factor's 89% LOWER bound exceeds its own null bound),")
            print("    which is the claim rule document 05 §7 pre-registered.")
            power_rows = []
            for sigma in POWER_SCENARIOS:
                for factor, kwargs in (
                    ("quarterback", {"sigma_a": sigma}),
                    ("coach", {"sigma_b": sigma}),
                ):
                    draws = null_bound(
                        design,
                        codes,
                        sizes,
                        residual_sd,
                        DATASETS,
                        RANDOM_SEED + int(sigma * 10000) + (0 if factor == "quarterback" else 1),
                        **kwargs,
                    )
                    key = "lower_a" if factor == "quarterback" else "lower_b"
                    power = float((draws[key] > bounds[factor]).mean())
                    power_rows.append({"factor": factor, "true_sigma": sigma, "power": power})
                    print(f"  {factor:12s} true sigma {sigma:.4f}  power {power:.3f}")

            report["null_bounds"] = bounds
            report["power"] = power_rows

        report[f"design_{label}"] = design_summary

    report["datasets"] = DATASETS
    report["null_percentile"] = NULL_PERCENTILE
    report["power_scenarios"] = list(POWER_SCENARIOS)
    report["random_seed"] = RANDOM_SEED
    out = paths.RESEARCH_OUTPUT_DIR / "21_s3_attribution_power.json"
    with out.open("w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
