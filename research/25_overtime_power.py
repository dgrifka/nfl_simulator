"""Phase 5 candidate 1 — the overtime coin toss: power and impact calculation.

Runs **before** `docs/research/16-overtime-toss.md` fixes any threshold, per the
process law that a power number is attached to every gate before the gate is
committed. Nothing here fits the real data's swing: every posterior in this file
is fitted to a *simulated* dataset.

Three questions, in the order document 16 asks them:

1. **Can this design see the effect?** At 155 overtime games, what is the chance
   the 89% interval on the swing excludes zero, as a function of the true swing?
2. **Would seeing it change anything?** Applied to the incumbent simulator
   (v1.1), how far does a swing of a given size move DTW on the games it
   touches — measured against the width of the interval the product already
   prints on those same games?
3. **Can the 2025 rule change be tested?** With 16 games under the new rules,
   what is the chance of detecting that the effect went away entirely?

    uv run python research/25_overtime_power.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.components import (  # noqa: E402
    build_game_table,
    fit_fg_baseline,
    fit_fumble_baseline,
    fit_xp_baseline,
)
from nfl_simulator.fg_model import FieldGoalModel  # noqa: E402
from nfl_simulator.ingest import ANALYSIS_COLUMNS, PBP_SEASONS, load_pbp  # noqa: E402
from nfl_simulator.simulator import points_per_epa, simulate_game  # noqa: E402

RANDOM_SEED = 20260818
N_DATASETS = 2000
N_POSTERIOR_DRAWS = 2000
DIRICHLET_ALPHA = 0.5  # Jeffreys prior for a multinomial
ETI_LOW, ETI_HIGH = 5.5, 94.5

# Candidate true swings, in points of final margin. The swing is the difference
# in expected final margin between receiving the first overtime possession and
# not receiving it, so half of it is the adjustment a single game absorbs.
SWING_GRID = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)

# Simulator settings copied from research/15_simulator_v11.py so the impact
# numbers are measured on the shipped configuration, not a cheaper one.
SIM_POSTERIOR_DRAWS = 200
SIM_COIN_DRAWS = 800
SIM_SEED = 20260817

SIM_COLUMNS = [
    *ANALYSIS_COLUMNS,
    "kicker_player_id",
    "extra_point_attempt",
    "extra_point_result",
    "roof",
    "temp",
    "wind",
    # Not in ANALYSIS_COLUMNS: the overtime frame needs the period and the
    # regular-season/playoff split, and nothing before Phase 5 did.
    "qtr",
    "season_type",
]


# --------------------------------------------------------------------------
# the overtime game frame
# --------------------------------------------------------------------------


def ot_game_frame(pbp: pl.DataFrame) -> pl.DataFrame:
    """One row per overtime game, margin taken from the receiving team's view.

    A game that reaches overtime is tied at the end of regulation, so the final
    margin *is* the overtime margin. `posteam` on an nflverse kickoff row is the
    receiving team, so the first non-null `posteam` in quarter 5 is the team with
    the first overtime possession.
    """
    ot = pbp.filter(pl.col("qtr") == 5)
    frame = (
        ot.sort(["game_id", "play_id"])
        .group_by("game_id", maintain_order=True)
        .agg(
            pl.col("posteam").drop_nulls().first().alias("first_pos"),
            pl.col("home_team").first(),
            pl.col("away_team").first(),
            pl.col("result").first(),
            pl.col("season").first(),
            pl.col("season_type").first(),
            pl.col("week").first(),
        )
    )
    return frame.with_columns(
        home_received=pl.col("first_pos") == pl.col("home_team"),
        fp_margin=pl.when(pl.col("first_pos") == pl.col("home_team"))
        .then(pl.col("result"))
        .otherwise(-pl.col("result")),
    )


def support_and_weights(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Symmetrised outcome support and the magnitude distribution on it.

    The margin of an overtime game lives on a handful of values (0, ±3, ±6 …).
    The support is symmetrised so a prior that is flat over it carries no opinion
    about which team the coin favours; `weights` is the observed distribution of
    |margin|, spread evenly over each magnitude's two signs, and is the *null*
    from which simulated datasets are drawn.
    """
    magnitudes = np.abs(frame["fp_margin"].to_numpy())
    values = np.unique(magnitudes)
    support = np.unique(np.concatenate([values, -values])).astype(float)

    weights = np.zeros_like(support)
    for magnitude in values:
        share = float((magnitudes == magnitude).mean())
        if magnitude == 0:
            weights[support == 0.0] = share
        else:
            weights[support == magnitude] = share / 2
            weights[support == -magnitude] = share / 2
    return support, weights / weights.sum()


def tilted(support: np.ndarray, weights: np.ndarray, mean_shift: float) -> np.ndarray:
    """The null distribution tilted to have mean `mean_shift`.

    Mass moves from a magnitude's losing sign to its winning sign in proportion
    `delta`, leaving the distribution of |margin| untouched. That is the right
    shape for this effect: winning the toss changes *who* wins an overtime game,
    not whether it ends 3–0 or 6–0.
    """
    scale = float((weights * np.abs(support)).sum())
    delta = mean_shift / scale
    if abs(delta) > 1:
        raise ValueError(f"mean shift {mean_shift} is impossible on this support")
    return weights * (1 + delta * np.sign(support))


# --------------------------------------------------------------------------
# the estimator
# --------------------------------------------------------------------------


def swing_draws(
    counts_home: np.ndarray,
    counts_away: np.ndarray,
    support: np.ndarray,
    rng: np.random.Generator,
    n_draws: int = N_POSTERIOR_DRAWS,
) -> np.ndarray:
    """Posterior draws of the swing, home-balanced.

    Two Dirichlet posteriors — one for the games where the home team received,
    one for the games where the away team did — and the swing is their two means
    added. Averaging the groups cancels home-field advantage, which would
    otherwise leak into the estimate whenever the toss lands on the home team
    more often than half the time (it did: 87 of 155).
    """
    p_home = rng.dirichlet(counts_home + DIRICHLET_ALPHA, size=n_draws)
    p_away = rng.dirichlet(counts_away + DIRICHLET_ALPHA, size=n_draws)
    return p_home @ support + p_away @ support


def counts_on(values: np.ndarray, support: np.ndarray) -> np.ndarray:
    return np.array([(values == v).sum() for v in support], dtype=float)


def excludes_zero(draws: np.ndarray) -> bool:
    low, high = np.percentile(draws, [ETI_LOW, ETI_HIGH])
    return bool(low > 0 or high < 0)


# --------------------------------------------------------------------------
# question 1 — can the design see the swing?
# --------------------------------------------------------------------------


def detection_power(
    support: np.ndarray,
    weights: np.ndarray,
    n_home: int,
    n_away: int,
    home_effect: float,
    rng: np.random.Generator,
) -> dict:
    """Chance the 89% interval on the swing excludes zero, by true swing.

    `home_effect` is a nuisance shift applied in opposite directions to the two
    groups. The estimator is built to cancel it; carrying it here checks that
    claim rather than assuming it.
    """
    rows = []
    for swing in (0.0, *SWING_GRID):
        p_home = tilted(support, weights, swing / 2 + home_effect)
        p_away = tilted(support, weights, swing / 2 - home_effect)
        hits, estimates = 0, []
        for _ in range(N_DATASETS):
            counts_home = rng.multinomial(n_home, p_home).astype(float)
            counts_away = rng.multinomial(n_away, p_away).astype(float)
            draws = swing_draws(counts_home, counts_away, support, rng, n_draws=400)
            hits += excludes_zero(draws)
            estimates.append(float(draws.mean()))
        rows.append(
            {
                "true_swing": swing,
                "power": hits / N_DATASETS,
                "mean_estimate": float(np.mean(estimates)),
                "sd_estimate": float(np.std(estimates, ddof=1)),
            }
        )
    return {"rows": rows}


# --------------------------------------------------------------------------
# question 2 — would it change anything?
# --------------------------------------------------------------------------


def impact_on_incumbent(frame: pl.DataFrame, pbp: pl.DataFrame, rng: np.random.Generator) -> dict:
    """How far a swing of each size moves DTW on the games it touches.

    The overtime branch is additive on top of the incumbent's bootstrap, so the
    simulator runs once and every candidate swing is applied to the same margin
    draws:

        new_margin = margin - (received - replayed_coin) * swing

    with `replayed_coin` a fair flip. No approximation — this is exactly what a
    ledger row for the toss would do.
    """
    print("  fitting league baselines for the incumbent simulator ...")
    fumble_baseline = fit_fumble_baseline(pbp)
    fg_baseline = fit_fg_baseline(pbp)
    xp_baseline = fit_xp_baseline(pbp)
    games = build_game_table(pbp)
    slope = points_per_epa(games.drop_nulls("margin"))

    with (paths.RESEARCH_OUTPUT_DIR / "fg_weather_summary.json").open() as handle:
        centres = json.load(handle)["centres"]
    fg_model = FieldGoalModel.from_posterior(
        paths.RESEARCH_OUTPUT_DIR / "trace_fg_weather.nc",
        wind_centre=centres["wind"],
        temp_centre=centres["temp"],
    )

    ot_ids = set(frame["game_id"].to_list())
    received = dict(zip(frame["game_id"], frame["home_received"], strict=True))

    half_widths, deltas = [], {swing: [] for swing in SWING_GRID}
    flips = {swing: 0 for swing in SWING_GRID}
    simulated = 0
    for game_id, group in pbp.filter(pl.col("game_id").is_in(ot_ids)).group_by("game_id"):
        game_id = game_id[0] if isinstance(game_id, tuple) else game_id
        result = simulate_game(
            group,
            fumble_baseline=fumble_baseline,
            fg_baseline=fg_baseline,
            fg_model=fg_model,
            xp_baseline=xp_baseline,
            points_per_epa=slope,
            n_posterior_draws=SIM_POSTERIOR_DRAWS,
            n_coin_draws=SIM_COIN_DRAWS,
            seed=SIM_SEED,
        )
        margins = result.margin_draws
        dtw_old = float((margins > 0).mean())
        half_widths.append((result.dtw_interval[1] - result.dtw_interval[0]) / 2)
        y = 1.0 if received[game_id] else 0.0
        replayed = rng.integers(0, 2, size=margins.shape).astype(float)
        for swing in SWING_GRID:
            dtw_new = float((margins - (y - replayed) * swing > 0).mean())
            deltas[swing].append(dtw_new - dtw_old)
            flips[swing] += (dtw_old - 0.5) * (dtw_new - 0.5) < 0
        simulated += 1
        if simulated % 50 == 0:
            print(f"    {simulated} overtime games simulated")

    floor = float(np.median(half_widths))
    rows = [
        {
            "swing": swing,
            "median_abs_delta_dtw": float(np.median(np.abs(deltas[swing]))),
            "mean_abs_delta_dtw": float(np.mean(np.abs(deltas[swing]))),
            "max_abs_delta_dtw": float(np.max(np.abs(deltas[swing]))),
            "side_flips": int(flips[swing]),
            "side_flip_rate": flips[swing] / simulated,
        }
        for swing in SWING_GRID
    ]
    return {
        "games": simulated,
        "points_per_epa": slope,
        "incumbent_median_dtw_half_width": floor,
        "rows": rows,
    }


# --------------------------------------------------------------------------
# question 3 — can the 2025 rule change be tested?
# --------------------------------------------------------------------------


def era_power(
    support: np.ndarray,
    weights: np.ndarray,
    n_new: int,
    n_old: int,
    rng: np.random.Generator,
) -> dict:
    """Chance of detecting that the new rules removed the effect entirely.

    The alternative is as extreme as the question allows: the pre-2025 games
    carry the full swing and the 2025 games carry none. If the design cannot see
    *that*, it cannot see anything smaller either.
    """
    rows = []
    for swing in SWING_GRID:
        p_old = tilted(support, weights, swing / 2)
        p_null = weights
        hits = 0
        for _ in range(N_DATASETS):
            counts_old = rng.multinomial(n_old, p_old).astype(float)
            counts_new = rng.multinomial(n_new, p_null).astype(float)
            draws_old = rng.dirichlet(counts_old + DIRICHLET_ALPHA, size=400) @ support * 2
            draws_new = rng.dirichlet(counts_new + DIRICHLET_ALPHA, size=400) @ support * 2
            hits += excludes_zero(draws_new - draws_old)
        rows.append({"pre_2025_swing": swing, "power_to_detect_removal": hits / N_DATASETS})
    return {"n_new_rules": n_new, "n_old_rules": n_old, "rows": rows}


# --------------------------------------------------------------------------


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    pbp = load_pbp(PBP_SEASONS, columns=SIM_COLUMNS)
    frame = ot_game_frame(pbp)

    support, weights = support_and_weights(frame)
    n_home = int(frame["home_received"].sum())
    n_away = frame.height - n_home

    # Nuisance only. Half the difference between the two groups' observed means
    # is home-field advantage inside an overtime game; it is carried into the
    # simulation so the estimator's immunity to it is demonstrated, not assumed.
    means = (
        frame.group_by("home_received")
        .agg(pl.col("fp_margin").mean().alias("mean"))
        .sort("home_received")
    )
    away_mean, home_mean = means["mean"].to_list()
    home_effect = float((home_mean - away_mean) / 2)

    print(f"overtime games: {frame.height} ({n_home} home received, {n_away} away)")
    print(f"support: {support.tolist()}")
    print(f"magnitude weights: {np.round(weights, 4).tolist()}")
    print(f"home-field nuisance carried in simulation: {home_effect:+.4f} points")

    print("\n[1] detection power ...")
    detection = detection_power(support, weights, n_home, n_away, home_effect, rng)
    for row in detection["rows"]:
        print(
            f"    true swing {row['true_swing']:.1f} pts -> power {row['power']:.3f}"
            f"  (estimate {row['mean_estimate']:+.3f} ± {row['sd_estimate']:.3f})"
        )

    print("\n[2] impact on the incumbent simulator ...")
    impact = impact_on_incumbent(frame, pbp, rng)
    floor = impact["incumbent_median_dtw_half_width"]
    print(f"    incumbent median 89% DTW half-width on these games: {100 * floor:.2f} pp")
    for row in impact["rows"]:
        print(
            f"    swing {row['swing']:.1f} pts -> median |dDTW| {100 * row['median_abs_delta_dtw']:.2f} pp"
            f", side flips {row['side_flips']}/{impact['games']}"
        )

    print("\n[3] 2025 rule-change power ...")
    n_new = frame.filter(pl.col("season") == 2025).height
    era = era_power(support, weights, n_new, frame.height - n_new, rng)
    for row in era["rows"]:
        print(
            f"    pre-2025 swing {row['pre_2025_swing']:.1f} pts -> "
            f"power to detect removal {row['power_to_detect_removal']:.3f}"
        )

    # The reference swing: the smallest simulated swing that would move DTW by
    # more than the interval the product already prints. A design that cannot
    # detect this swing cannot detect anything that would matter.
    reference = next(
        (row["swing"] for row in impact["rows"] if row["median_abs_delta_dtw"] >= floor),
        None,
    )
    reference_power = next(
        (row["power"] for row in detection["rows"] if row["true_swing"] == reference), None
    )
    print(f"\nreference swing (smallest that clears the floor): {reference}")
    print(f"power at the reference: {reference_power}")

    payload = {
        "random_seed": RANDOM_SEED,
        "n_datasets": N_DATASETS,
        "dirichlet_alpha": DIRICHLET_ALPHA,
        "games": frame.height,
        "n_home_received": n_home,
        "n_away_received": n_away,
        "support": support.tolist(),
        "null_weights": weights.tolist(),
        "home_effect_nuisance": home_effect,
        "detection": detection,
        "impact": impact,
        "era": era,
        "reference_swing": reference,
        "reference_power": reference_power,
    }
    out = paths.RESEARCH_OUTPUT_DIR / "25_overtime_power.json"
    with out.open("w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
