"""Phase 7, task 1 — refit the make-probability model without blocked kicks.

Runs the gates `docs/research/27-make-probability-refit.md` §7 committed at
`88ac49c`, before this file existed:

* **R-1** sampler health, **R-2** weather calibration, **R-3** wind resolvable,
  **R-4** distance calibration (which also picks the arm), **R-5** posterior
  predictive, **R-6** kicker skill still resolvable, **R-7** temperature and
  **R-8** extra-point transfer.
* §9a and §9b of the same document — the mandatory impact report on the model
  and on `p_make` kick by kick. The ledger and DTW halves, §9c and §9d, are
  `research/42b_fg_refit_impact.py`, which reads the posterior this script
  writes.

The model, the priors, the sampler settings and the gate machinery are document
05b's, reused from `research/14_fg_weather_model.py` rather than copied, so the
incumbent and the refit cannot drift apart. The only difference between the two
fits is one filter on the training population.

    uv run python research/42_fg_refit.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import arviz as az
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_weather = import_module("14_fg_weather_model")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.fg_model import FieldGoalModel, sanitize_weather  # noqa: E402

RANDOM_SEED = 20260818  # document 27 §13
TRACE_PREFIX = "trace_fg_refit"
GATE_R6_THRESHOLD = 0.2407  # inherited and conservative, document 27 §7b
REPORT_DISTANCES = (30.0, 40.0, 45.0, 50.0, 55.0)


# --------------------------------------------------------------------------
# the read side, as the simulator uses it
# --------------------------------------------------------------------------


def read_side_probabilities(model: FieldGoalModel, kicks: pl.DataFrame) -> np.ndarray:
    """Mean `p_make` per kick through `FieldGoalModel`, the path the product prints.

    Deliberately not `14_fg_weather_model.make_probabilities`, which is the
    *fitted* model's own arithmetic. The number a reader cares about is the one
    the simulator books luck against, and document 27 §9d asks for exactly this
    comparison — which is how the discrepancy reported in §14 was found.
    """
    out = np.empty(kicks.height)
    for i, row in enumerate(kicks.iter_rows(named=True)):
        weather = sanitize_weather(row["roof"], row["wind"], row["temp"])
        out[i] = model.make_probability(
            row["kicker_season"], float(row["distance"]), weather=weather
        ).mean()
    return out


def eti89(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 5.5)), float(np.percentile(values, 94.5))]


# --------------------------------------------------------------------------
# document 27 §9a — what moved in the model
# --------------------------------------------------------------------------


def parameter_comparison(refit_idata, incumbent_idata) -> dict:
    """Every shared parameter, incumbent against refit, with 89% intervals."""
    names = [
        "alpha",
        "beta",
        "gamma",
        "delta_cubic",
        "sigma_kicker",
        "beta_wind",
        "beta_temp",
        "delta_xp",
        "lambda_xp",
    ]
    rows = []
    for name in names:
        row = {"parameter": name}
        for label, idata in (("incumbent", incumbent_idata), ("refit", refit_idata)):
            posterior = idata["posterior"]
            if name not in posterior:
                row[f"{label}_mean"] = None
                row[f"{label}_eti89"] = None
                continue
            draws = posterior[name].values.ravel()
            row[f"{label}_mean"] = float(draws.mean())
            row[f"{label}_eti89"] = eti89(draws)
        rows.append(row)

    for level in _weather.ROOF_LEVELS:
        row = {"parameter": f"roof[{level}]"}
        for label, idata in (("incumbent", incumbent_idata), ("refit", refit_idata)):
            posterior = idata["posterior"]
            i = list(posterior["roof_level"].values).index(level)
            draws = posterior["roof"].values.reshape(-1, len(_weather.ROOF_LEVELS))[:, i]
            row[f"{label}_mean"] = float(draws.mean())
            row[f"{label}_eti89"] = eti89(draws)
        rows.append(row)

    print("\n[9a] parameters, incumbent -> refit")
    for row in rows:
        if row["refit_mean"] is None or row["incumbent_mean"] is None:
            print(f"  {row['parameter']:16s} {'(absent from one arm)':>28s}")
            continue
        print(
            f"  {row['parameter']:16s} {row['incumbent_mean']:+9.5f} -> "
            f"{row['refit_mean']:+9.5f}   "
            f"refit 89% [{row['refit_eti89'][0]:+.5f}, {row['refit_eti89'][1]:+.5f}]"
        )
    return {"rows": rows}


def league_curves(refit: FieldGoalModel, incumbent: FieldGoalModel) -> dict:
    """The league make-rate curve at five distances, both models, no kicker, no weather."""
    rows = []
    print("\n[9a] league make rate for an average kicker, outdoors, no weather reading")
    for distance in REPORT_DISTANCES:
        before = float(incumbent.league_make_probability(distance).mean())
        after = float(refit.league_make_probability(distance).mean())
        rows.append(
            {
                "distance": distance,
                "incumbent": before,
                "refit": after,
                "change_pp": (after - before) * 100,
            }
        )
        print(
            f"  {distance:.0f} yd: {before * 100:5.2f}% -> {after * 100:5.2f}%  "
            f"({(after - before) * 100:+.2f} pp)"
        )
    return {"rows": rows}


# --------------------------------------------------------------------------
# document 27 §9b — what moved in p_make, kick by kick
# --------------------------------------------------------------------------


def p_make_shift(kicks: pl.DataFrame, refit: FieldGoalModel, incumbent: FieldGoalModel) -> dict:
    print(f"\n[9b] pricing {kicks.height:,} kicks through both posteriors ...")
    before = read_side_probabilities(incumbent, kicks)
    after = read_side_probabilities(refit, kicks)
    delta = after - before

    frame = kicks.with_columns(
        pl.Series("p_incumbent", before),
        pl.Series("p_refit", after),
        pl.Series("delta_pp", delta * 100),
        ((pl.col("distance") // 5) * 5).cast(pl.Int32).alias("distance_bin"),
    )

    def summarise(values: np.ndarray) -> dict:
        return {
            "n": int(len(values)),
            "mean_pp": float(values.mean()),
            "median_pp": float(np.median(values)),
            "eti89_pp": eti89(values),
            "max_abs_pp": float(np.abs(values).max()),
        }

    overall = summarise(delta * 100)
    print(
        f"  all kicks: mean {overall['mean_pp']:+.3f} pp, median {overall['median_pp']:+.3f} pp, "
        f"89% [{overall['eti89_pp'][0]:+.3f}, {overall['eti89_pp'][1]:+.3f}], "
        f"max |shift| {overall['max_abs_pp']:.3f} pp"
    )

    by_kind = {}
    for label, mask in (("field_goal", frame["is_xp"] == 0), ("extra_point", frame["is_xp"] == 1)):
        values = frame.filter(mask)["delta_pp"].to_numpy()
        by_kind[label] = summarise(values)
        print(
            f"  {label:12s} ({by_kind[label]['n']:,}): mean {by_kind[label]['mean_pp']:+.3f} pp, "
            f"median {by_kind[label]['median_pp']:+.3f} pp"
        )

    print("\n  by roof:")
    by_roof = (
        frame.group_by("roof")
        .agg(pl.len().alias("n"), pl.col("delta_pp").mean().alias("mean_shift_pp"))
        .sort("n", descending=True)
    )
    print(by_roof)

    print("\n  by distance bin, field goals only:")
    by_bin = (
        frame.filter(pl.col("is_xp") == 0)
        .group_by("distance_bin")
        .agg(
            pl.len().alias("n"),
            pl.col("delta_pp").mean().alias("mean_shift_pp"),
            pl.col("p_incumbent").mean().alias("p_incumbent"),
            pl.col("p_refit").mean().alias("p_refit"),
        )
        .sort("distance_bin")
    )
    with pl.Config(tbl_rows=20):
        print(by_bin)

    return {
        "overall": overall,
        "by_kind": by_kind,
        "by_roof": by_roof.to_dicts(),
        "by_distance_bin_field_goals": by_bin.to_dicts(),
    }


def kicker_movement(refit: FieldGoalModel, incumbent: FieldGoalModel) -> dict:
    """How far individual kicker-seasons moved, and whether any vanished."""
    shared = sorted(set(refit.kicker_effects) & set(incumbent.kicker_effects))
    dropped = sorted(set(incumbent.kicker_effects) - set(refit.kicker_effects))
    added = sorted(set(refit.kicker_effects) - set(incumbent.kicker_effects))
    moves = np.array(
        [float(refit.kicker_effects[k].mean() - incumbent.kicker_effects[k].mean()) for k in shared]
    )
    order = np.argsort(-np.abs(moves))[:5]
    largest = [{"kicker_season": shared[i], "log_odds_shift": float(moves[i])} for i in order]
    print(
        f"\n[9a] kicker-seasons: {len(incumbent.kicker_effects)} incumbent, "
        f"{len(refit.kicker_effects)} refit, {len(dropped)} dropped, {len(added)} added"
    )
    if dropped:
        # Rejected rows, not counts — document 20 §9.
        print(f"  dropped: {dropped}")
    print(
        f"  mean |shift| in kicker effect {np.abs(moves).mean():.4f} log-odds, "
        f"largest {largest[0]['log_odds_shift']:+.4f} ({largest[0]['kicker_season']})"
    )
    return {
        "n_incumbent": len(incumbent.kicker_effects),
        "n_refit": len(refit.kicker_effects),
        "dropped": dropped,
        "added": added,
        "mean_abs_shift_log_odds": float(np.abs(moves).mean()),
        "largest_shifts": largest,
    }


# --------------------------------------------------------------------------


def main() -> None:
    paths.ensure_data_dirs()

    kicks = _weather.load_kicks(exclude_blocked=True)
    centres = {
        "wind": float(kicks.filter(pl.col("has_weather"))["wind"].mean()),
        "temp": float(kicks.filter(pl.col("has_weather"))["temp"].mean()),
    }
    with (paths.RESEARCH_OUTPUT_DIR / "42a_fg_refit_power.json").open() as handle:
        power = json.load(handle)
    thresholds = {"wind": power["gate_r3_threshold"], "temp": power["gate_r7_threshold"]}

    kicker_levels = sorted(kicks["kicker_season"].unique().to_list())
    lookup = {level: i for i, level in enumerate(kicker_levels)}
    kicker_idx = np.array([lookup[v] for v in kicks["kicker_season"].to_list()])

    n_fg = int(kicks.height - kicks["is_xp"].sum())
    print(
        f"{kicks.height:,} kicks ({n_fg:,} field goals, {int(kicks['is_xp'].sum()):,} extra "
        f"points), {len(kicker_levels)} kicker-seasons — blocked kicks excluded"
    )
    print(f"centres: wind {centres['wind']:.4f} mph, temp {centres['temp']:.4f} F")
    print(
        f"thresholds re-derived at the refit's n: wind {thresholds['wind']:+.7f}, "
        f"temp {thresholds['temp']:+.7f}"
    )

    results = {
        "n_kicks": int(kicks.height),
        "n_field_goals": n_fg,
        "n_extra_points": int(kicks["is_xp"].sum()),
        "n_kicker_seasons": len(kicker_levels),
        "centres": centres,
        "thresholds": thresholds,
        "random_seed": RANDOM_SEED,
        "gate_r6_threshold": GATE_R6_THRESHOLD,
    }

    # ---- the arm ladder, fixed in document 27 §7f -------------------------
    idata, quadratic = _weather.fit_arm(
        "ARM 1 — pre-registered quadratic distance curve, blocked kicks excluded",
        kicks,
        kicker_levels,
        kicker_idx,
        centres,
        thresholds,
        cubic=False,
        seed=RANDOM_SEED,
        trace_prefix=TRACE_PREFIX,
    )
    results["arm1_quadratic"] = quadratic
    adopted, adopted_report = "quadratic", quadratic

    if not quadratic["gate_w4_distance_calibration"]["pass"]:
        print(
            "\nGate R-4 failed on the quadratic. Fitting the cubic — the incumbent's own\n"
            "adopted form, and the last rung of the ladder document 27 §7f fixed. If R-4\n"
            "fails there too, the refit is not adopted and no third curve is fitted."
        )
        idata, cubic_report = _weather.fit_arm(
            "ARM 2 — cubic distance curve, blocked kicks excluded",
            kicks,
            kicker_levels,
            kicker_idx,
            centres,
            thresholds,
            cubic=True,
            seed=RANDOM_SEED,
            trace_prefix=TRACE_PREFIX,
        )
        results["arm2_cubic"] = cubic_report
        if cubic_report["gate_w4_distance_calibration"]["pass"]:
            adopted, adopted_report = "cubic", cubic_report
        else:
            adopted, adopted_report = "none", cubic_report

    results["adopted_arm"] = adopted
    trace_path = paths.RESEARCH_OUTPUT_DIR / f"{TRACE_PREFIX}.nc"
    idata.to_netcdf(trace_path)
    print(f"\nAdopted arm: {adopted}")

    # ---- gate R-6, which fit_arm reports but does not gate ----------------
    sigma_upper = adopted_report["sigma_kicker_eti89"][1]
    r6 = {
        "sigma_kicker_mean": adopted_report["sigma_kicker_mean"],
        "sigma_kicker_eti89": adopted_report["sigma_kicker_eti89"],
        "threshold": GATE_R6_THRESHOLD,
        "pass": bool(sigma_upper > GATE_R6_THRESHOLD),
    }
    print(
        f"\nGate R-6 (kicker skill resolvable): {'PASS' if r6['pass'] else 'FAIL'} — "
        f"sigma_kicker {r6['sigma_kicker_mean']:.3f} "
        f"[{r6['sigma_kicker_eti89'][0]:.3f}, {r6['sigma_kicker_eti89'][1]:.3f}] "
        f"against {GATE_R6_THRESHOLD}"
    )
    results["gate_r6_kicker_resolvable"] = r6

    # ---- the impact report, document 27 §9a and §9b -----------------------
    with (paths.RESEARCH_OUTPUT_DIR / "fg_weather_summary.json").open() as handle:
        incumbent_summary = json.load(handle)
    incumbent_centres = incumbent_summary["centres"]
    incumbent_model = FieldGoalModel.from_posterior(
        paths.RESEARCH_OUTPUT_DIR / "trace_fg_weather.nc",
        wind_centre=incumbent_centres["wind"],
        temp_centre=incumbent_centres["temp"],
    )
    refit_model = FieldGoalModel.from_posterior(
        trace_path, wind_centre=centres["wind"], temp_centre=centres["temp"]
    )
    incumbent_idata = az.from_netcdf(paths.RESEARCH_OUTPUT_DIR / "trace_fg_weather.nc")

    results["impact_9a_parameters"] = parameter_comparison(idata, incumbent_idata)
    results["impact_9a_league_curve"] = league_curves(refit_model, incumbent_model)
    results["impact_9a_kickers"] = kicker_movement(refit_model, incumbent_model)
    results["impact_9b_p_make_shift"] = p_make_shift(kicks, refit_model, incumbent_model)

    # ---- gate summary ------------------------------------------------------
    gates = {
        "R-1 sampler health": adopted_report["gate_w1_sampler_health"]["pass"],
        "R-2 weather calibration": adopted_report["gate_w2_weather_calibration"]["pass"],
        "R-3 wind resolvable": adopted_report["gate_w3_wind_resolvable"]["pass"],
        "R-4 distance calibration": adopted_report["gate_w4_distance_calibration"]["pass"],
        "R-5 posterior predictive": adopted_report["gate_w5_posterior_predictive"]["pass"],
        "R-6 kicker resolvable": r6["pass"],
    }
    print("\n=== Gate summary, adopted arm ===")
    for name, passed in gates.items():
        print(f"  {name:28s} {'PASS' if passed else 'FAIL'}")
    results["gate_summary"] = gates
    results["all_gates_pass"] = bool(all(gates.values()) and adopted != "none")

    out = paths.RESEARCH_OUTPUT_DIR / "fg_refit_summary.json"
    with out.open("w") as handle:
        json.dump(results, handle, indent=2, default=str)
    print(f"\nwrote {out}")
    print(
        "\nVERDICT: "
        + (
            "ALL GATES PASS — propose adoption, stop at the door and ask the maintainer"
            if results["all_gates_pass"]
            else "DO NOT ADOPT — a pre-registered gate failed"
        )
    )
    print("Next: research/42b_fg_refit_impact.py for document 27 §9c and §9d.")


if __name__ == "__main__":
    main()
