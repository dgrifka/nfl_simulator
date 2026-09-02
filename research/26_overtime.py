"""Phase 5 candidate 1 — the overtime coin toss: the fit.

Runs the three gates `docs/research/16-overtime-toss.md` §5 committed at
`fea34ea`, before this file existed:

* **Gate O-1** — does the 89% interval on the swing exclude zero?
* **Gate O-2** — does an independent estimator agree to within 10% relative,
  and is the answer stable across priors?
* **Gate O-3** — does the fitted swing move DTW by more than the 4.06 pp
  interval half-width the incumbent already prints on these games?

Plus the two things §5f and §5h required be reported without being allowed to
decide anything: the era split with its power attached, and a sensitivity arm
that redraws the swing on every posterior draw.

    uv run python research/26_overtime.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_power = import_module("25_overtime_power")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.components import (  # noqa: E402
    build_game_table,
    fit_fg_baseline,
    fit_fumble_baseline,
    fit_xp_baseline,
)
from nfl_simulator.fg_model import FieldGoalModel  # noqa: E402
from nfl_simulator.ingest import PBP_SEASONS, load_pbp  # noqa: E402
from nfl_simulator.simulator import points_per_epa, simulate_game  # noqa: E402

RANDOM_SEED = _power.RANDOM_SEED
POSTERIOR_DRAWS = 20_000
BOOTSTRAP_DRAWS = 20_000
PRIOR_ALPHAS = (0.5, 1.0, 0.01)
RELATIVE_TOLERANCE = 0.10
GATE_O3_FLOOR = 0.0406  # docs/research/16 §5e — the incumbent's own half-width
N_REPLAY_REPLICATES = 8
ETI_LOW, ETI_HIGH = _power.ETI_LOW, _power.ETI_HIGH


def eti(draws: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(draws, [ETI_LOW, ETI_HIGH])
    return float(low), float(high)


def balanced_swing_draws(
    frame: pl.DataFrame,
    support: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
    n_draws: int = POSTERIOR_DRAWS,
) -> np.ndarray:
    """Posterior draws of the swing, home-balanced (document 16 §5b)."""
    home = frame.filter(pl.col("home_received"))["fp_margin"].to_numpy()
    away = frame.filter(~pl.col("home_received"))["fp_margin"].to_numpy()
    counts_home = _power.counts_on(home, support) + alpha
    counts_away = _power.counts_on(away, support) + alpha
    return (
        rng.dirichlet(counts_home, size=n_draws) @ support
        + rng.dirichlet(counts_away, size=n_draws) @ support
    )


def naive_swing_draws(
    frame: pl.DataFrame, support: np.ndarray, alpha: float, rng: np.random.Generator
) -> np.ndarray:
    """The unbalanced estimate — one Dirichlet over all games, doubled."""
    counts = _power.counts_on(frame["fp_margin"].to_numpy(), support) + alpha
    return 2 * (rng.dirichlet(counts, size=POSTERIOR_DRAWS) @ support)


def bootstrap_swing(frame: pl.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Independent cross-check: a nonparametric bootstrap of the same contrast.

    Document 09 §9's corrective asked for a *relative* agreement tolerance rather
    than an absolute one inherited by analogy. It also asked the cross-check to be
    a genuinely different instrument; a second sampler on a conjugate posterior
    would only measure Monte Carlo noise, so the check here resamples games.
    """
    home = frame.filter(pl.col("home_received"))["fp_margin"].to_numpy()
    away = frame.filter(~pl.col("home_received"))["fp_margin"].to_numpy()
    draws = np.empty(BOOTSTRAP_DRAWS)
    for i in range(BOOTSTRAP_DRAWS):
        draws[i] = rng.choice(home, size=home.size).mean() + rng.choice(away, size=away.size).mean()
    return draws


def impact(
    frame: pl.DataFrame,
    pbp: pl.DataFrame,
    swing_mean: float,
    swing_posterior: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Gate O-3, plus the swing-uncertainty sensitivity arm of §5h."""
    print("  fitting league baselines ...")
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

    rows = []
    for game_id, group in pbp.filter(pl.col("game_id").is_in(ot_ids)).group_by("game_id"):
        game_id = game_id[0] if isinstance(game_id, tuple) else game_id
        result = simulate_game(
            group,
            fumble_baseline=fumble_baseline,
            fg_baseline=fg_baseline,
            fg_model=fg_model,
            xp_baseline=xp_baseline,
            points_per_epa=slope,
            n_posterior_draws=_power.SIM_POSTERIOR_DRAWS,
            n_coin_draws=_power.SIM_COIN_DRAWS,
            seed=_power.SIM_SEED,
        )
        margins = result.margin_draws.reshape(_power.SIM_POSTERIOR_DRAWS, _power.SIM_COIN_DRAWS)
        y = 1.0 if received[game_id] else 0.0
        replayed = rng.integers(0, 2, size=margins.shape).astype(float)
        branch = y - replayed

        fixed = margins - branch * swing_mean
        # Sensitivity: one swing draw per posterior draw, so layer 1 carries the
        # swing's own uncertainty as well as the probability's.
        drawn = swing_posterior[rng.integers(0, swing_posterior.size, size=margins.shape[0])]
        varying = margins - branch * drawn[:, None]

        old_per_draw = (margins > 0).mean(axis=1)
        fixed_per_draw = (fixed > 0).mean(axis=1)
        varying_per_draw = (varying > 0).mean(axis=1)

        # Gate O-3 lands within a tenth of a point of its floor, so the record
        # has to say whether that gap is real or is the replayed coin's own
        # Monte Carlo noise. Extra realisations are nearly free once the game is
        # simulated: only the coin is redrawn, never the margins.
        replicate_dtw = [
            float(
                ((margins - (y - rng.integers(0, 2, size=margins.shape)) * swing_mean) > 0).mean()
            )
            for _ in range(N_REPLAY_REPLICATES)
        ]

        rows.append(
            {
                "game_id": game_id,
                "home_received": bool(y),
                "actual_margin": result.actual_margin,
                "dtw_old": float(old_per_draw.mean()),
                "dtw_new": float(fixed_per_draw.mean()),
                "dtw_new_varying": float(varying_per_draw.mean()),
                **{f"dtw_replicate_{i}": v for i, v in enumerate(replicate_dtw)},
                "half_width_old": (result.dtw_interval[1] - result.dtw_interval[0]) / 2,
                "half_width_new": (np.diff(np.percentile(fixed_per_draw, [ETI_LOW, ETI_HIGH]))[0])
                / 2,
                "half_width_varying": (
                    np.diff(np.percentile(varying_per_draw, [ETI_LOW, ETI_HIGH]))[0]
                )
                / 2,
            }
        )
        if len(rows) % 50 == 0:
            print(f"    {len(rows)} overtime games simulated")

    table = pl.DataFrame(rows).with_columns(
        delta=(pl.col("dtw_new") - pl.col("dtw_old")),
        flipped=((pl.col("dtw_old") - 0.5) * (pl.col("dtw_new") - 0.5)) < 0,
    )
    replicate_medians = [
        float((table[f"dtw_replicate_{i}"] - table["dtw_old"]).abs().median())
        for i in range(N_REPLAY_REPLICATES)
    ]
    return {
        "points_per_epa": slope,
        "games": table.height,
        "median_abs_delta_dtw": float(table["delta"].abs().median()),
        "replicate_median_abs_delta_dtw": replicate_medians,
        "replicate_spread": [min(replicate_medians), max(replicate_medians)],
        "mean_abs_delta_dtw": float(table["delta"].abs().mean()),
        "max_abs_delta_dtw": float(table["delta"].abs().max()),
        "side_flips": int(table["flipped"].sum()),
        "median_half_width_old": float(table["half_width_old"].median()),
        "median_half_width_new": float(table["half_width_new"].median()),
        "median_half_width_varying": float(table["half_width_varying"].median()),
        "median_abs_dtw_sensitivity_gap": float(
            (table["dtw_new"] - table["dtw_new_varying"]).abs().median()
        ),
        "table": table.to_dicts(),
    }


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    pbp = load_pbp(PBP_SEASONS, columns=_power.SIM_COLUMNS)
    frame = _power.ot_game_frame(pbp)
    support, _ = _power.support_and_weights(frame)

    print(f"overtime games: {frame.height}")

    # ---- Gate O-1 --------------------------------------------------------
    posterior = balanced_swing_draws(frame, support, 0.5, rng)
    swing_mean = float(posterior.mean())
    swing_eti = eti(posterior)
    o1 = swing_eti[0] > 0 or swing_eti[1] < 0
    print(
        f"\n[O-1] swing = {swing_mean:+.4f} pts, 89% ETI [{swing_eti[0]:+.4f}, {swing_eti[1]:+.4f}]"
    )
    print(f"      excludes zero: {o1}")

    # ---- Gate O-2 --------------------------------------------------------
    boot = bootstrap_swing(frame, rng)
    boot_eti = eti(boot)
    endpoint_gap = max(
        abs(boot_eti[0] - swing_eti[0]) / abs(swing_eti[0]),
        abs(boot_eti[1] - swing_eti[1]) / abs(swing_eti[1]),
    )
    priors = {}
    for alpha in PRIOR_ALPHAS:
        draws = balanced_swing_draws(frame, support, alpha, rng)
        priors[alpha] = {"mean": float(draws.mean()), "eti": eti(draws)}
    prior_gap = max(
        abs(priors[a]["mean"] - priors[0.5]["mean"]) / abs(priors[0.5]["mean"])
        for a in PRIOR_ALPHAS
    )
    naive = naive_swing_draws(frame, support, 0.5, rng)
    o2 = endpoint_gap <= RELATIVE_TOLERANCE and prior_gap <= RELATIVE_TOLERANCE
    print(
        f"\n[O-2] bootstrap swing = {boot.mean():+.4f}, 89% ETI "
        f"[{boot_eti[0]:+.4f}, {boot_eti[1]:+.4f}]; worst endpoint gap {endpoint_gap:.2%}"
    )
    for alpha in PRIOR_ALPHAS:
        print(f"      alpha={alpha:<5}: {priors[alpha]['mean']:+.4f}")
    print(f"      worst prior gap {prior_gap:.2%}; naive (unbalanced) swing {naive.mean():+.4f}")
    print(f"      pass: {o2}")

    # ---- Gate O-3 --------------------------------------------------------
    print("\n[O-3] impact on the incumbent simulator ...")
    imp = impact(frame, pbp, swing_mean, posterior, rng)
    o3 = imp["median_abs_delta_dtw"] >= GATE_O3_FLOOR
    print(
        f"      median |dDTW| {100 * imp['median_abs_delta_dtw']:.2f} pp "
        f"vs floor {100 * GATE_O3_FLOOR:.2f} pp -> pass: {o3}"
    )
    spread = imp["replicate_spread"]
    print(
        f"      across {N_REPLAY_REPLICATES} independent replayed-coin realisations the median "
        f"ranges {100 * spread[0]:.2f}-{100 * spread[1]:.2f} pp"
    )
    print(
        f"      mean {100 * imp['mean_abs_delta_dtw']:.2f} pp, max "
        f"{100 * imp['max_abs_delta_dtw']:.2f} pp, side flips {imp['side_flips']}/{imp['games']}"
    )
    print(
        f"      median 89% half-width: {100 * imp['median_half_width_old']:.2f} pp -> "
        f"{100 * imp['median_half_width_new']:.2f} pp (fixed swing), "
        f"{100 * imp['median_half_width_varying']:.2f} pp (swing redrawn)"
    )

    # ---- reported, not deciding (§5f) ------------------------------------
    eras = {}
    for label, expr in [
        ("2016-2024", pl.col("season") < 2025),
        ("2025", pl.col("season") == 2025),
        ("regular season", pl.col("season_type") == "REG"),
        ("playoffs", pl.col("season_type") != "REG"),
    ]:
        subset = frame.filter(expr)
        draws = balanced_swing_draws(frame.filter(expr), support, 0.5, rng)
        eras[label] = {"n": subset.height, "mean": float(draws.mean()), "eti": eti(draws)}
        print(
            f"\n[reported] {label}: n={subset.height}, swing {draws.mean():+.4f} "
            f"[{eras[label]['eti'][0]:+.4f}, {eras[label]['eti'][1]:+.4f}]"
        )
    print("      power to detect the 2025 rules removing the effect entirely: 0.243 (doc 16 §4d)")
    print("      -> neither outcome of this comparison is a finding")

    payload = {
        "random_seed": RANDOM_SEED,
        "posterior_draws": POSTERIOR_DRAWS,
        "games": frame.height,
        "support": support.tolist(),
        "swing_points": swing_mean,
        "swing_eti": swing_eti,
        "swing_epa": swing_mean / imp["points_per_epa"],
        "gate_o1_pass": bool(o1),
        "bootstrap": {"mean": float(boot.mean()), "eti": boot_eti, "endpoint_gap": endpoint_gap},
        "priors": {str(a): priors[a] for a in PRIOR_ALPHAS},
        "prior_gap": prior_gap,
        "naive_swing": float(naive.mean()),
        "gate_o2_pass": bool(o2),
        "gate_o3_floor": GATE_O3_FLOOR,
        "gate_o3_pass": bool(o3),
        "impact": {k: v for k, v in imp.items() if k != "table"},
        "eras": eras,
    }
    out = paths.RESEARCH_OUTPUT_DIR / "26_overtime.json"
    with out.open("w") as handle:
        json.dump(payload, handle, indent=2)
    pl.DataFrame(imp["table"]).write_parquet(
        paths.RESEARCH_OUTPUT_DIR / "26_overtime_games.parquet"
    )
    print(f"\nwrote {out}")

    verdict = "NEUTRALIZE" if (o1 and o3) else "NO TREATMENT"
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
