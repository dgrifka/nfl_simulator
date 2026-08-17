"""Phase 4, step 4 — power and null bounds for the special-teams round.

Three components, each with its own Gate A argument and its own power check, run
**before** `docs/research/14-special-teams.md` commits any threshold:

**(a) Punting, weather-aware.** A hierarchical punter model of net punt yards
given the kick situation and the conditions, reusing
`fg_model.sanitize_weather` so the fit and any consumer share one definition of
a windy day. Punting from your own 20 and from the opponent's 40 are different
jobs, so the model conditions on the spot.

**(b) The punt bounce.** Whether the post-landing roll is observable at all is a
*data* question, not a statistical one, and it is settled in the results script
rather than powered here.

**(c) Kick and punt returns.** Persistence at returner-season and team-season
grains. Era-aware: the kickoff return rate went 0.25 (2023) -> 0.33 (2024) ->
0.74 (2025) as the dynamic-kickoff rule and then the touchback-spot change
landed, so kickoff eras are never pooled and every split is within-season.

    uv run python research/22_special_teams_power.py
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
from nfl_simulator.fg_model import sanitize_weather  # noqa: E402
from nfl_simulator.ingest import PBP_SEASONS, load_pbp  # noqa: E402

RANDOM_SEED = 20260817
DATASETS = 400
NULL_PERCENTILE = 90

N_SPLITS = 200
N_NULL_REPLICATES = 500
N_POWER_REPLICATES = 500
MIN_GAMES = 8
REFERENCE_R = 0.12

# The kickoff eras. 2024 brought the dynamic kickoff and 2025 moved the touchback
# spot to the 35; the return rate went 0.25 -> 0.33 -> 0.74 across the boundary.
# Pooling them would measure the rule change, not the returners.
KICKOFF_ERAS: dict[str, tuple[int, ...]] = {
    "2016-2023 (traditional)": tuple(range(2016, 2024)),
    "2024 (dynamic kickoff)": (2024,),
    "2025 (touchback to the 35)": (2025,),
}

SPOT_CENTRE = 65.0  # yards to the opponent goal, near the median punting spot
WIND_CENTRE_FALLBACK = 8.118  # document 05b §10's outdoor mean, reused if needed

COLUMNS = [
    "season",
    "week",
    "game_id",
    "posteam",
    "defteam",
    "punt_attempt",
    "kickoff_attempt",
    "punt_blocked",
    "kick_distance",
    "return_yards",
    "touchback",
    "yardline_100",
    "punter_player_id",
    "punt_returner_player_id",
    "kickoff_returner_player_id",
    "return_team",
    "roof",
    "wind",
    "temp",
]


# --------------------------------------------------------------------------
# (a) punting
# --------------------------------------------------------------------------


def punt_table(pbp: pl.DataFrame) -> pl.DataFrame:
    """One row per punt, with net yards, the kick situation and the conditions.

    **Net punt yards** is the field position the punting team gained::

        touchback      net = spot - 20     (the ball goes to the receiving 20)
        otherwise      net = kick_distance - return_yards

    That identity is exact: the ball starts ``spot`` from the opponent's goal,
    travels ``kick_distance``, and comes back ``return_yards``.

    Blocked punts are excluded rather than scored as terrible punts. A block is a
    protection failure, and document 05b §2 made the opposite ruling for blocked
    *field goals* — charging them to the kicker — for consistency with
    `components.py`. Nothing downstream here needs that consistency, and a block
    is even less the punter's doing than it is the kicker's, since the punter
    releases the ball later.
    """
    punts = pbp.filter(
        (pl.col("punt_attempt") == 1)
        & (pl.col("punt_blocked") != 1)
        & pl.col("kick_distance").is_not_null()
        & (pl.col("kick_distance") > 0)
        & pl.col("punter_player_id").is_not_null()
        & pl.col("yardline_100").is_not_null()
    ).with_columns(pl.col("return_yards").fill_null(0.0))

    weather = [
        sanitize_weather(roof, wind, temp)
        for roof, wind, temp in zip(
            punts["roof"].to_list(),
            punts["wind"].to_list(),
            punts["temp"].to_list(),
            strict=True,
        )
    ]
    return punts.with_columns(
        pl.when(pl.col("touchback") == 1)
        .then(pl.col("yardline_100") - 20)
        .otherwise(pl.col("kick_distance") - pl.col("return_yards"))
        .cast(pl.Float64)
        .alias("net"),
        pl.col("yardline_100").cast(pl.Float64).alias("spot"),
        pl.Series("clean_roof", [w.roof for w in weather]),
        pl.Series("clean_wind", [w.wind for w in weather], dtype=pl.Float64),
        pl.Series("clean_temp", [w.temp for w in weather], dtype=pl.Float64),
        pl.Series("has_weather", [w.has_weather for w in weather]),
        pl.concat_str(
            [pl.col("season").cast(pl.String), pl.col("punter_player_id")], separator="_"
        ).alias("punter_season"),
        pl.concat_str([pl.col("season").cast(pl.String), pl.col("defteam")], separator="_").alias(
            "return_unit_season"
        ),
    )


def punt_design_matrix(punts: pl.DataFrame, wind_centre: float, temp_centre: float) -> np.ndarray:
    """Fixed effects: quadratic in the centred spot, roof levels, wind, temperature."""
    spot = punts["spot"].to_numpy() - SPOT_CENTRE
    roof = punts["clean_roof"].to_list()
    has_weather = punts["has_weather"].to_numpy().astype(float)
    wind = np.nan_to_num(punts["clean_wind"].to_numpy(), nan=wind_centre)
    temp = np.nan_to_num(punts["clean_temp"].to_numpy(), nan=temp_centre)
    return np.column_stack(
        [
            np.ones(len(spot)),
            spot,
            spot**2 / 100.0,
            np.array([level == "dome" for level in roof], dtype=float),
            np.array([level == "closed" for level in roof], dtype=float),
            np.array([level == "open" for level in roof], dtype=float),
            (wind - wind_centre) * has_weather,
            (temp - temp_centre) * has_weather,
        ]
    )


def wind_null_bound(
    x: np.ndarray, residual_sd: float, datasets: int, seed: int, *, beta_wind: float = 0.0
) -> np.ndarray:
    """89% upper bounds on the wind coefficient this design produces, by simulation.

    Ordinary least squares rather than the hierarchy, which is what makes hundreds
    of fits affordable. The direction of that shortcut is stated: unmodelled
    punter spread inflates the residual, so the intervals here are **wider** than
    the hierarchy's and the bound is therefore conservative. Document 05b §10 took
    the same shortcut and recorded the same direction.
    """
    rng = np.random.default_rng(seed)
    wind_column = 6
    bounds = np.empty(datasets)
    xtx_inv = np.linalg.inv(x.T @ x)
    for index in range(datasets):
        y = x[:, wind_column] * beta_wind + rng.normal(0.0, residual_sd, len(x))
        beta = xtx_inv @ (x.T @ y)
        rss = float(((y - x @ beta) ** 2).sum())
        sigma2 = rss / (len(x) - x.shape[1])
        se = float(np.sqrt(sigma2 * xtx_inv[wind_column, wind_column]))
        bounds[index] = beta[wind_column] + 1.5982 * se  # 89% equal-tailed upper bound
    return bounds


# --------------------------------------------------------------------------
# (c) returns
# --------------------------------------------------------------------------


def return_table(pbp: pl.DataFrame, kind: str) -> pl.DataFrame:
    """One row per return, for kickoffs or punts."""
    attempt, returner = (
        ("kickoff_attempt", "kickoff_returner_player_id")
        if kind == "kickoff"
        else ("punt_attempt", "punt_returner_player_id")
    )
    return pbp.filter(
        (pl.col(attempt) == 1)
        & pl.col(returner).is_not_null()
        & pl.col("return_yards").is_not_null()
    ).with_columns(
        pl.col(returner).alias("returner"),
        pl.col("return_yards").cast(pl.Float64).alias("yards"),
    )


def entity_game_matrix(returns: pl.DataFrame, entity: str) -> tuple[np.ndarray, ...]:
    """Per entity-season, per game: return count and total return yards."""
    key = (
        pl.concat_str([pl.col("season").cast(pl.String), pl.col("returner")], separator="_")
        if entity == "returner"
        else pl.concat_str([pl.col("season").cast(pl.String), pl.col("return_team")], separator="_")
    )
    grouped = (
        returns.with_columns(key.alias("entity_season"))
        .group_by(["entity_season", "game_id"])
        .agg(pl.len().alias("returns"), pl.col("yards").sum().alias("yards"))
        .sort(["entity_season", "game_id"])
    )
    keys = grouped["entity_season"].to_numpy()
    boundaries = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1], True])
    spans = [
        (boundaries[i], boundaries[i + 1])
        for i in range(len(boundaries) - 1)
        if boundaries[i + 1] - boundaries[i] >= MIN_GAMES
    ]
    if not spans:
        return np.zeros((0, 2)), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    rows = np.concatenate([np.arange(lo, hi) for lo, hi in spans])
    matrix = np.column_stack(
        [grouped["returns"].to_numpy().astype(float), grouped["yards"].to_numpy().astype(float)]
    )[rows]
    sizes = np.array([hi - lo for lo, hi in spans])
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    return matrix, starts, sizes


def split_half_r(matrix: np.ndarray, mask: np.ndarray, starts: np.ndarray) -> float:
    """Mean split-half correlation of yards per return, over the split draws."""
    totals = np.add.reduceat(matrix, starts, axis=0)
    sums_a = _seq.half_sums(matrix, mask, starts)
    sums_b = totals[None, :, :] - sums_a
    with np.errstate(divide="ignore", invalid="ignore"):
        a = sums_a[:, :, 1] / sums_a[:, :, 0]
        b = sums_b[:, :, 1] / sums_b[:, :, 0]
    usable = np.isfinite(a).all(axis=0) & np.isfinite(b).all(axis=0)
    return float(_seq.correlate(a[:, usable], b[:, usable]).mean())


def return_null_and_power(
    matrix: np.ndarray, starts: np.ndarray, sizes: np.ndarray, per_return_sd: float, seed: int
) -> dict:
    """Permutation null threshold and a power curve for one return cell."""
    rng = np.random.default_rng(seed)
    mask = _seq.split_masks(starts, sizes, len(matrix), rng, N_SPLITS)

    null_draws = np.empty(N_NULL_REPLICATES)
    for replicate in range(N_NULL_REPLICATES):
        shuffled = matrix[rng.permutation(len(matrix))]
        null_draws[replicate] = split_half_r(shuffled, mask, starts)
    threshold = float(np.percentile(null_draws, 95))

    group_of_row = np.repeat(np.arange(len(sizes)), sizes)
    returns_per_half = float(
        np.mean([matrix[s : s + n, 0].sum() / 2.0 for s, n in zip(starts, sizes, strict=True)])
    )
    power_rows = []
    for target_r in (0.05, 0.10, REFERENCE_R, 0.20, 0.30):
        tau = float(np.sqrt(target_r * (per_return_sd**2 / returns_per_half) / (1.0 - target_r)))
        draws = np.empty(N_POWER_REPLICATES)
        simulated = matrix.copy()
        for replicate in range(N_POWER_REPLICATES):
            theta = (rng.standard_normal(len(sizes)) * tau)[group_of_row]
            counts = matrix[:, 0]
            simulated[:, 1] = rng.normal(counts * theta, per_return_sd * np.sqrt(counts))
            draws[replicate] = split_half_r(simulated, mask, starts)
        power_rows.append(
            {
                "target_true_r": target_r,
                "tau": tau,
                "mean_observed_r": float(draws.mean()),
                "power": float((draws > threshold).mean()),
            }
        )
    return {
        "n_entities": int(len(starts)),
        "n_entity_games": int(len(matrix)),
        "mean_returns_per_half": returns_per_half,
        "null_mean_r": float(null_draws.mean()),
        "null_sd_r": float(null_draws.std(ddof=1)),
        "null_p95": threshold,
        "power": power_rows,
    }


# --------------------------------------------------------------------------


def main() -> None:
    paths.ensure_data_dirs()
    pbp = load_pbp(PBP_SEASONS, columns=COLUMNS)
    report: dict = {"random_seed": RANDOM_SEED, "datasets": DATASETS}

    # ---------------------------------------------------------- (a) punting
    punts = punt_table(pbp)
    outdoor = punts.filter(pl.col("has_weather"))
    wind_centre = float(outdoor["clean_wind"].mean())
    temp_centre = float(outdoor["clean_temp"].mean())
    x = punt_design_matrix(punts, wind_centre, temp_centre)
    y = punts["net"].to_numpy()

    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    residual_sd = float(np.std(residual, ddof=1))

    punt_design = {
        "punts": int(punts.height),
        "punter_seasons": int(punts["punter_season"].n_unique()),
        "punters": int(punts["punter_player_id"].n_unique()),
        "return_unit_seasons": int(punts["return_unit_season"].n_unique()),
        "median_punts_per_punter_season": float(
            punts.group_by("punter_season").agg(pl.len().alias("n"))["n"].median()
        ),
        "mean_net_yards": float(y.mean()),
        "sd_net_yards": float(np.std(y, ddof=1)),
        "residual_sd_after_covariates": residual_sd,
        "wind_centre": wind_centre,
        "temp_centre": temp_centre,
        "share_with_weather": float(punts["has_weather"].mean()),
        "spot_centre": SPOT_CENTRE,
    }
    print(f"{'=' * 72}\n(a) PUNTING — design\n{'=' * 72}")
    for key, value in punt_design.items():
        print(f"  {key:34s} {value:.4f}" if isinstance(value, float) else f"  {key:34s} {value}")

    print(f"\n=== Wind null bound ({DATASETS} OLS datasets, true beta_wind = 0) ===")
    null_wind = wind_null_bound(x, residual_sd, DATASETS, RANDOM_SEED)
    wind_bound = float(np.percentile(null_wind, 10))
    print(f"  89% upper bounds under a true zero: 10th pct = {wind_bound:+.5f} yards per mph")
    wind_power = []
    for true_beta in (-0.05, -0.10, -0.20, -0.40):
        draws = wind_null_bound(x, residual_sd, DATASETS, RANDOM_SEED + 1, beta_wind=true_beta)
        power = float((draws < wind_bound).mean())
        wind_power.append({"true_beta_wind": true_beta, "power": power})
        print(
            f"  true beta_wind {true_beta:+.2f} yd/mph "
            f"({-true_beta * 15:.1f} yd lost at 15 mph)  power {power:.3f}"
        )

    print(f"\n=== Punter-skill null bound ({DATASETS} datasets, true SD exactly zero) ===")
    print("    one-way punter-season variance component, on covariate-residualized net yards")
    print(
        "    The GATED instrument is one-way, not crossed, and that is a compute ruling with\n"
        "    a stated consequence. A crossed punter x return-unit fit at 392 + 320 levels costs\n"
        "    about thirty seconds, which makes a 400-dataset null bound a three-hour job and a\n"
        "    power curve on top of it impossible. The one-way form is exact and runs in under a\n"
        "    millisecond. The crossed fit is reported ONCE, descriptively, on the real data — it\n"
        "    is not gated, and no threshold here belongs to it."
    )
    punter_names = sorted(punts["punter_season"].unique().to_list())
    punter_lookup = {name: i for i, name in enumerate(punter_names)}
    punter_codes = np.array([punter_lookup[name] for name in punts["punter_season"].to_list()])
    n_punters = len(punter_names)

    rng = np.random.default_rng(RANDOM_SEED)
    uppers = []
    for _ in range(DATASETS):
        simulated = rng.normal(0.0, residual_sd, len(punter_codes))
        result = _grid.fit_one_way(*_grid.one_way_statistics(punter_codes, n_punters, simulated))
        uppers.append(result["sigma_a"]["eti89_ub"])
    punter_bound = float(np.percentile(uppers, NULL_PERCENTILE))
    print(f"  punter-season null bound   {punter_bound:.4f} net yards")

    unit_names = sorted(punts["return_unit_season"].unique().to_list())
    unit_lookup = {name: i for i, name in enumerate(unit_names)}
    unit_codes = np.array([unit_lookup[name] for name in punts["return_unit_season"].to_list()])
    unit_rng = np.random.default_rng(RANDOM_SEED + 7)
    unit_uppers = []
    for _ in range(DATASETS):
        simulated = unit_rng.normal(0.0, residual_sd, len(unit_codes))
        result = _grid.fit_one_way(
            *_grid.one_way_statistics(unit_codes, len(unit_names), simulated)
        )
        unit_uppers.append(result["sigma_a"]["eti89_ub"])
    unit_bound = float(np.percentile(unit_uppers, NULL_PERCENTILE))
    print(f"  return-unit null bound     {unit_bound:.4f} net yards")

    punter_power = []
    for sigma in (0.25, 0.5, 1.0, 1.5, 2.5):
        power_rng = np.random.default_rng(RANDOM_SEED + int(sigma * 100))
        drawn = []
        for _ in range(DATASETS):
            simulated = (
                power_rng.normal(0.0, residual_sd, len(punter_codes))
                + power_rng.normal(0.0, sigma, n_punters)[punter_codes]
            )
            result = _grid.fit_one_way(
                *_grid.one_way_statistics(punter_codes, n_punters, simulated)
            )
            drawn.append(result["sigma_a"]["eti89_lb"])
        power = float((np.array(drawn) > punter_bound).mean())
        punter_power.append({"true_sigma_punter": sigma, "power": power})
        print(f"  true sigma_punter {sigma:.2f} net yards  power {power:.3f}")

    report["punting"] = {
        "design": punt_design,
        "wind_null_bound": wind_bound,
        "wind_power": wind_power,
        "punter_null_bound": punter_bound,
        "return_unit_null_bound": unit_bound,
        "punter_power": punter_power,
    }

    # ---------------------------------------------------------- (c) returns
    print(f"\n{'=' * 72}\n(c) RETURNS — null thresholds and power\n{'=' * 72}")
    return_cells = {}
    for kind in ("kickoff", "punt"):
        returns = return_table(pbp, kind)
        eras = (
            KICKOFF_ERAS
            if kind == "kickoff"
            else {"2016-2025 (no rule break)": tuple(range(2016, 2026))}
        )
        for era, seasons in eras.items():
            subset = returns.filter(pl.col("season").is_in(list(seasons)))
            per_return_sd = float(np.std(subset["yards"].to_numpy(), ddof=1))
            for entity in ("returner", "team"):
                matrix, starts, sizes = entity_game_matrix(subset, entity)
                label = f"{kind} / {entity} / {era}"
                if len(starts) < 30:
                    print(
                        f"  {label:56s} only {len(starts)} entity-seasons with "
                        f"{MIN_GAMES}+ games — not powered, skipped"
                    )
                    return_cells[label] = {
                        "n_entities": int(len(starts)),
                        "skipped": True,
                        "reason": f"fewer than 30 entity-seasons with {MIN_GAMES}+ games",
                    }
                    continue
                cell = return_null_and_power(
                    matrix, starts, sizes, per_return_sd, RANDOM_SEED + len(return_cells)
                )
                cell["per_return_sd"] = per_return_sd
                cell["n_returns"] = int(subset.height)
                return_cells[label] = cell
                power_at_reference = next(
                    row["power"] for row in cell["power"] if row["target_true_r"] == REFERENCE_R
                )
                print(
                    f"  {label:56s} {cell['n_entities']:4d} entity-seasons  "
                    f"null 95th pct {cell['null_p95']:.4f}  "
                    f"power at r={REFERENCE_R} {power_at_reference:.2f}"
                )

    report["returns"] = return_cells
    report["min_games"] = MIN_GAMES
    report["reference_r"] = REFERENCE_R
    report["kickoff_eras"] = {era: list(seasons) for era, seasons in KICKOFF_ERAS.items()}

    out = paths.RESEARCH_OUTPUT_DIR / "22_special_teams_power.json"
    with out.open("w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
