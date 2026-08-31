"""v1.4, part 1 — refit the shipped make-probability model with the elevation term.

Document 67 reported the elevation study and the maintainer adopted it. This is the
refit that turns document 66's candidate arm into the posterior the product
reads: same population, same priors, same sampler settings, one column more.

**Nothing here is a new measurement.** Document 66 fixed the covariate, the
prior and the gates before a fit existed, and document 67 reported them. This
script re-runs the adopted arm under its own name and checks four things
before it will write a shipped artifact:

    V14-1  sampler health — document 05b's Gate FG-1 rule, unchanged:
           0 divergences, r_hat < 1.01, ess_bulk and ess_tail > 400.
    V14-2  the fitted parameters reproduce document 67 §2's published table.
           A mismatch is a stop, not a reconciliation (the rule document 46
           set for v1.3).
    V14-3  the posterior is the *studied* posterior. Same data, same model,
           same seed as `research/81_fg_elevation.py`, so `trace_fg_v14.nc`
           must equal `trace_fg_elevation.nc` draw for draw. This is what lets
           document 67's gates be cited for the shipped model rather than for
           a sibling of it.
    V14-4  the round trip, document 30 §5a: every kick in the fitted
           population, priced through `FieldGoalModel` — the path the product
           books luck against — must reproduce the fit's own arithmetic to
           1e-9. The elevation term is the first one resolved from a lookup
           rather than from a column, so this is the gate that would catch a
           read side consulting a different table than the fit did.

    uv run python research/82_fg_v14_refit.py

Writes:
    research/outputs/trace_fg_v14.nc        the shipped posterior
    research/outputs/fg_v14_summary.json    its centring constants and gates
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
_power = import_module("81a_fg_elevation_power")
_elevation = import_module("81_fg_elevation")
_read_side = import_module("44_read_side_fix")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.fg_model import FieldGoalModel, sanitize_weather  # noqa: E402

RANDOM_SEED = _elevation.RANDOM_SEED  # 20260831, document 66 §11
TRACE_NAME = "trace_fg_v14.nc"
SUMMARY_NAME = "fg_v14_summary.json"
STUDY_TRACE = "trace_fg_elevation.nc"
ROUND_TRIP_TOLERANCE = 1e-9  # document 30 §5a

# Document 67 §2's published means, and document 27 §14c's for the terms the
# elevation round did not move. Hard-coded rather than read from a trace: a
# mismatch here means the shipped posterior is not the one the record describes.
PUBLISHED = {
    "alpha": 1.909,
    "beta": -0.1158,
    "gamma": 0.247,
    "delta_cubic": -0.081,
    "sigma_kicker": 0.384,
    "beta_wind": -0.0221,
    "beta_elev": 0.0602,
}
PUBLISHED_TOLERANCE = {"beta_wind": 5e-4, "beta_elev": 5e-4}
DEFAULT_TOLERANCE = 5e-3

# Document 67 §2's effect ladder, the numbers the article quotes.
REPORT_DISTANCES = (33.0, 45.0, 50.0, 55.0)


def gate_v14_1(idata) -> dict:
    """Sampler health, document 05b's Gate FG-1 rule, reused from the study."""
    return _elevation.gate_e1(idata, "v1.4 shipped arm")


def gate_v14_2(posterior) -> dict:
    """Every published parameter, against the record that published it."""
    print("\nGate V14-2 (the fitted parameters are document 67 §2's)")
    rows, disagreements = [], []
    for name, published in PUBLISHED.items():
        got = float(posterior[name].values.ravel().mean())
        tolerance = PUBLISHED_TOLERANCE.get(name, DEFAULT_TOLERANCE)
        agrees = abs(got - published) <= tolerance
        rows.append(
            {
                "parameter": name,
                "published": published,
                "fitted": got,
                "tolerance": tolerance,
                "agrees": agrees,
            }
        )
        if not agrees:
            disagreements.append(name)
        print(
            f"  {name:14s} published {published:+9.5f}  fitted {got:+9.5f}  "
            f"(±{tolerance:g})  {'ok' if agrees else 'MISMATCH'}"
        )
    report = {"rows": rows, "pass": not disagreements}
    if disagreements:
        raise SystemExit(
            f"the refit does not reproduce document 67 §2 on {disagreements}. Stop and report."
        )
    return report


def gate_v14_3(idata) -> dict:
    """The shipped posterior is the studied one, draw for draw.

    Same data, same model, same seed. If this ever disagrees, the gates in
    document 67 were run on a different object than the product reads, and
    every number in that document would have to be recomputed here.
    """
    study_path = paths.RESEARCH_OUTPUT_DIR / STUDY_TRACE
    if not study_path.exists():
        raise SystemExit(
            f"{STUDY_TRACE} is missing — run `uv run python research/81_fg_elevation.py` "
            "so the shipped posterior can be checked against the studied one."
        )
    study = az.from_netcdf(study_path)["posterior"]
    shipped = idata["posterior"]
    gaps = {}
    for name in (*PUBLISHED, "delta_xp", "lambda_xp"):
        gaps[name] = float(np.abs(shipped[name].values.ravel() - study[name].values.ravel()).max())
    worst = max(gaps.values())
    report = {"max_abs_draw_difference": worst, "by_parameter": gaps, "pass": worst == 0.0}
    print(
        f"\nGate V14-3 (shipped posterior == studied posterior): "
        f"{'PASS' if report['pass'] else 'FAIL'} — "
        f"largest per-draw difference {worst:.2e} across {len(gaps)} parameters"
    )
    if not report["pass"]:
        raise SystemExit(
            "the refit did not reproduce research/81's trace. Document 67's gates "
            "would then describe a different posterior than the one being shipped."
        )
    return report


def gate_v14_4(kicks: pl.DataFrame, centres: dict, elev_centre: float) -> dict:
    """The round trip, document 30 §5a, with the elevation term in it.

    Deliberately not `44_read_side_fix.round_trip`, which prices through
    `14_fg_weather_model.make_probabilities` and knows nothing about elevation.
    The fitted side here is `research/81`'s own arithmetic — the model that was
    actually sampled — and the read side is `FieldGoalModel`, the path the
    simulator books luck against.
    """
    model, loaded_centres = _read_side.load_model(TRACE_NAME, SUMMARY_NAME)
    idata = az.from_netcdf(paths.RESEARCH_OUTPUT_DIR / TRACE_NAME)
    _levels, kicker_idx = _elevation.kicker_index(kicks)
    fitted = _elevation.make_probabilities(idata, kicks, kicker_idx, centres, elev_centre).mean(
        axis=0
    )

    read_side = np.empty(kicks.height)
    for i, row in enumerate(kicks.iter_rows(named=True)):
        read_side[i] = model.make_probability(
            row["kicker_season"],
            float(row["distance"]),
            weather=sanitize_weather(row["roof"], row["wind"], row["temp"]),
            extra_point=bool(row["is_xp"]),
            stadium_id=row["stadium_id"],
        ).mean()

    is_xp = kicks["is_xp"].to_numpy().astype(bool)
    high = kicks["elev_kft"].to_numpy() >= 3.0
    delta = np.abs(read_side - fitted)
    report = {
        "n_kicks": int(kicks.height),
        "max_abs_diff_field_goals": float(delta[~is_xp].max()),
        "max_abs_diff_extra_points": float(delta[is_xp].max()),
        "max_abs_diff_above_3000ft": float(delta[high].max()),
        "n_above_3000ft": int(high.sum()),
        "tolerance": ROUND_TRIP_TOLERANCE,
        "elevation_centre_loaded": loaded_centres.get("elevation"),
    }
    report["pass"] = bool(delta.max() <= ROUND_TRIP_TOLERANCE)
    print(
        f"\nGate V14-4 (round trip, document 30 §5a): "
        f"{'PASS' if report['pass'] else 'FAIL'} — {report['n_kicks']:,} kicks, "
        f"max |read − fitted| FG {report['max_abs_diff_field_goals']:.2e}, "
        f"XP {report['max_abs_diff_extra_points']:.2e}, "
        f"on the {report['n_above_3000ft']} kicks above 3,000 ft "
        f"{report['max_abs_diff_above_3000ft']:.2e}"
    )
    if not report["pass"]:
        raise SystemExit("the round trip failed — the product would not price what it fitted.")
    return report


def effect_ladder(model: FieldGoalModel) -> dict:
    """Document 67 §2's table, recomputed through the shipped read side.

    The study computed it from the posterior directly; this computes it the way
    the product will, so the article can quote one number rather than two that
    ought to agree.
    """
    rows = []
    print("\nThe elevation effect, as the shipped read side prices it:")
    print("    dist | mean-elevation |    Denver |  Mexico City | sea level")
    for distance in REPORT_DISTANCES:
        centre = model.league_make_probability(distance).mean()
        denver = model.league_make_probability(distance, stadium_id="DEN00").mean()
        mexico = model.league_make_probability(distance, stadium_id="MEX00").mean()
        sea = model.league_make_probability(distance, stadium_id="NYC01").mean()
        rows.append(
            {
                "distance": distance,
                "p_at_fitted_centre": float(centre),
                "p_denver": float(denver),
                "p_mexico_city": float(mexico),
                "p_sea_level": float(sea),
                "denver_gain_pp": float(denver - centre) * 100,
                "mexico_city_gain_pp": float(mexico - centre) * 100,
                "denver_over_sea_level_pp": float(denver - sea) * 100,
            }
        )
        print(
            f"    {distance:4.0f} | {centre * 100:13.2f}% | {denver * 100:8.2f}% | "
            f"{mexico * 100:11.2f}% | {sea * 100:8.2f}%   "
            f"(Denver {(denver - centre) * 100:+.2f} pp vs centre, "
            f"{(denver - sea) * 100:+.2f} pp vs sea level)"
        )
    return {"rows": rows}


def main() -> None:
    paths.ensure_data_dirs()
    kicks = _power.load_elevation_kicks()
    elev_centre = _power.elevation_centre(kicks)
    centres = {"wind": _power.WIND_CENTRE, "temp": _power.TEMP_CENTRE}
    levels, kicker_idx = _elevation.kicker_index(kicks)

    n_fg = int(kicks.height - kicks["is_xp"].sum())
    print(
        f"{kicks.height:,} kicks ({n_fg:,} field goals, {int(kicks['is_xp'].sum()):,} extra "
        f"points), {len(levels)} kicker-seasons, {kicks['game_id'].n_unique():,} games "
        "— document 27's population, blocked kicks excluded"
    )
    print(
        f"centres: wind {centres['wind']:.4f} mph, temp {centres['temp']:.4f} F, "
        f"elevation {elev_centre:.4f} kft"
    )

    print(f"\n{'#' * 72}\n### The v1.4 shipped arm — v1.3's model plus beta_elev\n{'#' * 72}")
    model = _elevation.build_model(kicks, levels, kicker_idx, centres, elev_centre, elevation=True)
    idata = _elevation.sample(model, RANDOM_SEED)
    trace_path = paths.RESEARCH_OUTPUT_DIR / TRACE_NAME
    idata.to_netcdf(trace_path)
    print(f"wrote {trace_path}")

    results = {
        "version": "simulator-v1.4",
        "random_seed": RANDOM_SEED,
        "n_kicks": int(kicks.height),
        "n_field_goals": n_fg,
        "n_extra_points": int(kicks["is_xp"].sum()),
        "n_kicker_seasons": len(levels),
        "n_games": int(kicks["game_id"].n_unique()),
        # The three centring constants travel together, and `load_model` reads
        # them from here. Document 66 §2 said the elevation centre would have to
        # travel with the posterior exactly as wind's and temperature's do.
        "centres": {**centres, "elevation": elev_centre},
        "trace": TRACE_NAME,
        "prior_beta_elev_sd": _elevation.BETA_ELEV_PRIOR_SD,
        "source_documents": ["66 (pre-registration)", "67 (results)", "68 (the v1.4 record)"],
    }

    names = [
        "alpha", "beta", "gamma", "delta_cubic", "sigma_kicker",
        "roof", "beta_wind", "beta_temp", "delta_xp", "lambda_xp", "beta_elev",
    ]  # fmt: skip
    summary = az.summary(idata, var_names=names)
    print(summary[["mean", "sd", "eti89_lb", "eti89_ub", "ess_bulk", "r_hat"]])
    results["posterior_summary"] = summary.reset_index().to_dict(orient="records")

    posterior = idata["posterior"]
    results["gate_v14_1_sampler_health"] = gate_v14_1(idata)
    results["gate_v14_2_reproduces_document_67"] = gate_v14_2(posterior)
    results["gate_v14_3_is_the_studied_posterior"] = gate_v14_3(idata)

    beta_elev = posterior["beta_elev"].values.ravel()
    results["beta_elev"] = {
        "mean": float(beta_elev.mean()),
        "eti89": _weather.eti89(beta_elev),
        "units": "log-odds per 1,000 feet",
    }

    out = paths.RESEARCH_OUTPUT_DIR / SUMMARY_NAME
    with out.open("w") as handle:
        json.dump(results, handle, indent=2, default=str)

    # The round trip needs the summary on disk, because it loads the model the
    # way the product does — through `load_model`, off the centres this file
    # just wrote — rather than through a shortcut only this script would take.
    results["gate_v14_4_round_trip"] = gate_v14_4(kicks, centres, elev_centre)
    shipped, _ = _read_side.load_model(TRACE_NAME, SUMMARY_NAME)
    results["effect_ladder"] = effect_ladder(shipped)

    gates = {
        "V14-1 sampler health": results["gate_v14_1_sampler_health"]["pass"],
        "V14-2 reproduces document 67": results["gate_v14_2_reproduces_document_67"]["pass"],
        "V14-3 is the studied posterior": results["gate_v14_3_is_the_studied_posterior"]["pass"],
        "V14-4 round trip": results["gate_v14_4_round_trip"]["pass"],
    }
    results["gate_summary"] = gates
    results["all_gates_pass"] = bool(all(gates.values()))
    print(f"\n{'=' * 72}")
    for name, passed in gates.items():
        print(f"  {name:32s} {'PASS' if passed else 'FAIL'}")
    print("=" * 72)

    with out.open("w") as handle:
        json.dump(results, handle, indent=2, default=str)
    print(f"\nwrote {out}")
    print("Next: research/83_simulator_v14.py rebuilds both editions on this posterior.")


if __name__ == "__main__":
    main()
