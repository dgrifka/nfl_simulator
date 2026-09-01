"""Phase 8, task 1 — the read-side fix, measured in isolation.

Runs the gates `docs/research/30-v13-corrections.md` §5 committed at `b9e44c4`,
before this file existed:

* **S-1** the round trip — priced through `FieldGoalModel`, every kick in the
  fitted population must reproduce the fit script's own `make_probabilities` to
  1e-9, on both posteriors and both kick kinds. This is the check document 27
  §9d asked for as a formality and that found the defect.
* **S-2** backward compatibility — the corrected code, handed a posterior with
  none of the three parameters, must reproduce `dtw_games_v12.parquet` exactly.
* **S-3** the ledger must still sum, and the fix must add and remove no rows.
* **§5d** the impact report, on both populations, with the reconciliation.

Both arms use the **incumbent** posterior and the **v1.2 kicking population**, so
what is measured here is the read-side fix alone. The refit is document 27's and
the blocked-kick exclusion is `research/45_blocked_exclusion_c1.py`.

The shipped read side is reproduced by loading the fitted posterior and then
dropping the three parameters it never read, which is exactly what the shipped
code did with them.

    uv run python research/44_read_side_fix.py
"""

from __future__ import annotations

import dataclasses
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
from nfl_simulator.components import (  # noqa: E402
    build_game_table,
    fg_attempt_mask,
    fit_fg_baseline,
    fit_fumble_baseline,
    fit_xp_baseline,
    xp_attempt_mask,
)

# `SIM_COLUMNS` and `load_model` moved into the package in Round E so the
# installed wheel can read them; they are re-exported here under the names six
# research scripts already import, so the correction has one home.
from nfl_simulator.fg_model import FieldGoalModel, sanitize_weather  # noqa: E402
from nfl_simulator.fg_model import load_fitted_model as load_model  # noqa: E402,F401
from nfl_simulator.ingest import (  # noqa: E402
    ANALYSIS_COLUMNS,  # noqa: F401
    PBP_SEASONS,
    SIM_COLUMNS,  # noqa: F401
    load_pbp,
)
from nfl_simulator.simulator import points_per_epa, simulate_game  # noqa: E402

RANDOM_SEED = 20260817  # v1.2's build seed, so the arms differ by the fix alone
POSTERIOR_DRAWS = 200
COIN_DRAWS = 800
ROUND_TRIP_TOLERANCE = 1e-9  # document 30 §5a
LONG_ATTEMPT_YARDS = 50.0  # where document 27 §14f localizes the error

# Document 27 §14f, on the shipped population and the incumbent posterior: the
# mean absolute luck the pricing error misbooks, per row. Stated as a mean
# rather than a signed total because the ledger signs luck to the home team
# and §14f signed it to the kicking team, so the totals are not comparable.
DOCUMENT_27_MEAN_ABS_LUCK_ERROR = {"field_goal": 0.0446, "extra_point": 0.0100}


def shipped_read_side(model: FieldGoalModel) -> FieldGoalModel:
    """The v1.1/v1.2 read side: the same posterior, minus the terms it never read."""
    return dataclasses.replace(model, delta_cubic=None, delta_xp=None, lambda_xp=None)


# --------------------------------------------------------------------------
# Gate S-1 — the round trip
# --------------------------------------------------------------------------


def round_trip(label: str, trace: str, summary: str, *, exclude_blocked: bool) -> dict:
    """Does the production read side reproduce the fitted model's own `p_make`?"""
    model, centres = load_model(trace, summary)
    kicks = _weather.load_kicks(exclude_blocked=exclude_blocked)
    kicker_levels = sorted(kicks["kicker_season"].unique().to_list())
    lookup = {level: i for i, level in enumerate(kicker_levels)}
    kicker_idx = np.array([lookup[v] for v in kicks["kicker_season"].to_list()])

    idata = az.from_netcdf(paths.RESEARCH_OUTPUT_DIR / trace)
    fitted = _weather.make_probabilities(idata, kicks, kicker_idx, centres).mean(axis=0)

    read_side = np.empty(kicks.height)
    for i, row in enumerate(kicks.iter_rows(named=True)):
        read_side[i] = model.make_probability(
            row["kicker_season"],
            float(row["distance"]),
            weather=sanitize_weather(row["roof"], row["wind"], row["temp"]),
            extra_point=bool(row["is_xp"]),
        ).mean()

    is_xp = kicks["is_xp"].to_numpy().astype(bool)
    delta = np.abs(read_side - fitted)
    report = {
        "population": label,
        "n_kicks": int(kicks.height),
        "max_abs_diff_field_goals": float(delta[~is_xp].max()),
        "max_abs_diff_extra_points": float(delta[is_xp].max()),
        "tolerance": ROUND_TRIP_TOLERANCE,
    }
    report["pass"] = bool(delta.max() <= ROUND_TRIP_TOLERANCE)
    print(
        f"  {label:34s} n={report['n_kicks']:,}  max |read − fitted|: "
        f"FG {report['max_abs_diff_field_goals']:.2e}, "
        f"XP {report['max_abs_diff_extra_points']:.2e}  "
        f"-> {'PASS' if report['pass'] else 'FAIL'}"
    )
    return report


# --------------------------------------------------------------------------
# the two arms
# --------------------------------------------------------------------------


def simulate_arm(pbp, margins, baselines, model, slope) -> tuple[pl.DataFrame, pl.DataFrame]:
    rows, ledgers = [], []
    for game_id, group in pbp.group_by("game_id"):
        game_id = game_id[0] if isinstance(game_id, tuple) else game_id
        if margins.get(game_id) is None:
            continue
        result = simulate_game(
            group,
            fumble_baseline=baselines["fumble"],
            fg_baseline=baselines["fg"],
            xp_baseline=baselines["xp"],
            fg_model=model,
            points_per_epa=slope,
            n_posterior_draws=POSTERIOR_DRAWS,
            n_coin_draws=COIN_DRAWS,
            seed=RANDOM_SEED,
            # The v1.2 population on both arms: this study isolates the read-side
            # fix, and the blocked-kick exclusion is a separate correction.
            include_blocked=True,
        )
        rows.append(
            {
                "game_id": result.game_id,
                "actual_margin": result.actual_margin,
                "deserved_margin": result.deserved_margin,
                "dtw_home": result.dtw_home,
                "total_luck_epa": result.total_luck_epa,
                "n_luck_events": len(result.ledger),
            }
        )
        frame = result.ledger.to_frame()
        if frame.height:
            ledgers.append(frame.with_columns(pl.lit(result.game_id).alias("game_id")))
    return pl.DataFrame(rows), pl.concat(ledgers)


def summarise(joined: pl.DataFrame) -> dict:
    delta_dtw = (joined["dtw_fixed"] - joined["dtw_shipped"]).abs()
    delta_margin = joined["deserved_fixed"] - joined["deserved_shipped"]
    return {
        "games": joined.height,
        "median_abs_delta_dtw_pp": float(delta_dtw.median()) * 100,
        "mean_abs_delta_dtw_pp": float(delta_dtw.mean()) * 100,
        "max_abs_delta_dtw_pp": float(delta_dtw.max()) * 100,
        "median_abs_delta_deserved_margin": float(delta_margin.abs().median()),
        "mean_signed_delta_deserved_margin": float(delta_margin.mean()),
        "max_abs_delta_deserved_margin": float(delta_margin.abs().max()),
        "side_flips": joined.filter(
            ((pl.col("dtw_fixed") - 0.5) * (pl.col("dtw_shipped") - 0.5)) < 0
        ).height,
    }


def main() -> None:
    paths.ensure_data_dirs()

    print(f"{'=' * 72}\n[S-1] the round trip — read side against the fitted model\n{'=' * 72}")
    s1 = [
        round_trip(
            "incumbent (trace_fg_weather.nc)",
            "trace_fg_weather.nc",
            "fg_weather_summary.json",
            exclude_blocked=False,
        ),
        round_trip(
            "refit (trace_fg_refit.nc)",
            "trace_fg_refit.nc",
            "fg_refit_summary.json",
            exclude_blocked=True,
        ),
    ]
    if not all(report["pass"] for report in s1):
        raise SystemExit(
            "Gate S-1 FAILED — the production read side still disagrees with the fit. "
            "Stop and report; do not reconcile."
        )

    pbp = load_pbp(PBP_SEASONS, columns=SIM_COLUMNS)
    fixed, _ = load_model("trace_fg_weather.nc", "fg_weather_summary.json")
    shipped = shipped_read_side(fixed)

    baselines = {
        "fumble": fit_fumble_baseline(pbp),
        "fg": fit_fg_baseline(pbp, include_blocked=True),
        "xp": fit_xp_baseline(pbp, include_blocked=True),
    }
    games = build_game_table(pbp)
    slope = points_per_epa(games.drop_nulls("margin"))
    margins = dict(zip(games["game_id"], games["margin"], strict=True))
    print(f"\npoints per EPA: {slope:.4f}")

    print("\nsimulating the shipped arm (incumbent posterior, v1.2 read side) ...")
    shipped_games, shipped_ledger = simulate_arm(pbp, margins, baselines, shipped, slope)
    print("simulating the fixed arm (incumbent posterior, corrected read side) ...")
    fixed_games, fixed_ledger = simulate_arm(pbp, margins, baselines, fixed, slope)

    # ---- Gate S-2 --------------------------------------------------------
    print(f"\n{'=' * 72}\n[S-2] does the corrected code still reproduce v1.2?\n{'=' * 72}")
    v12 = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / "dtw_games_v12.parquet")
    check = v12.select("game_id", "deserved_margin", "dtw_home").join(
        shipped_games.select(
            "game_id",
            pl.col("deserved_margin").alias("replayed_margin"),
            pl.col("dtw_home").alias("replayed_dtw"),
        ),
        on="game_id",
        how="inner",
    )
    margin_gap = float((check["deserved_margin"] - check["replayed_margin"]).abs().max())
    dtw_gap = float((check["dtw_home"] - check["replayed_dtw"]).abs().max())
    s2 = {
        "games_compared": check.height,
        "games_in_v12": v12.height,
        "max_abs_deserved_margin_gap": margin_gap,
        "max_abs_dtw_gap": dtw_gap,
        "pass": bool(check.height == v12.height and margin_gap < 1e-9 and dtw_gap < 1e-9),
    }
    print(
        f"  {check.height:,} of {v12.height:,} games replayed; max |Δ deserved margin| "
        f"{margin_gap:.2e}, max |ΔDTW| {dtw_gap:.2e} -> {'PASS' if s2['pass'] else 'FAIL'}"
    )
    if not s2["pass"]:
        raise SystemExit(
            "Gate S-2 FAILED — the corrected code does not replay v1.2. Stop and report."
        )

    # ---- Gate S-3 --------------------------------------------------------
    print(f"\n{'=' * 72}\n[S-3] the ledger must still sum\n{'=' * 72}")
    counts = {
        arm: dict(
            frame.group_by("component").agg(pl.len().alias("n")).sort("component").iter_rows()
        )
        for arm, frame in (("shipped", shipped_ledger), ("fixed", fixed_ledger))
    }
    fg_rows = pbp.filter(fg_attempt_mask(include_blocked=True)).height
    xp_rows = pbp.filter(xp_attempt_mask(include_blocked=True)).height
    identity = float(
        (
            fixed_games["deserved_margin"]
            - (fixed_games["actual_margin"] - fixed_games["total_luck_epa"] * slope)
        )
        .abs()
        .max()
    )
    s3 = {
        "rows_by_component": counts,
        "field_goal_attempts_in_frame": fg_rows,
        "extra_point_attempts_in_frame": xp_rows,
        "max_abs_identity_residual": identity,
        "pass": bool(counts["shipped"] == counts["fixed"] and identity < 1e-9),
    }
    for arm, table in counts.items():
        print(f"  {arm:8s} {table}")
    print(f"  frame carries {fg_rows:,} field-goal and {xp_rows:,} extra-point attempts")
    print(
        f"  max |deserved − (actual − luck × slope)| = {identity:.2e} "
        f"-> {'PASS' if s3['pass'] else 'FAIL'}"
    )
    if not s3["pass"]:
        raise SystemExit("Gate S-3 FAILED — the fix moved rows. Stop and report.")

    # ---- the impact report, document 30 §5d ------------------------------
    print(f"\n{'=' * 72}\n[§5d] what the fix does to the product\n{'=' * 72}")
    joined = shipped_games.select(
        "game_id",
        pl.col("deserved_margin").alias("deserved_shipped"),
        pl.col("dtw_home").alias("dtw_shipped"),
    ).join(
        fixed_games.select(
            "game_id",
            pl.col("deserved_margin").alias("deserved_fixed"),
            pl.col("dtw_home").alias("dtw_fixed"),
        ),
        on="game_id",
    )

    kick_games = set(pbp.filter(fg_attempt_mask(True) | xp_attempt_mask(True))["game_id"].to_list())
    long_games = set(
        pbp.filter(fg_attempt_mask(True) & (pl.col("kick_distance") >= LONG_ATTEMPT_YARDS))[
            "game_id"
        ].to_list()
    )
    populations = {
        "all_games_with_a_kick": joined.filter(pl.col("game_id").is_in(kick_games)),
        "games_with_a_50_yard_attempt": joined.filter(pl.col("game_id").is_in(long_games)),
    }
    impact = {}
    for label, frame in populations.items():
        impact[label] = summarise(frame)
        stats = impact[label]
        print(
            f"  {label:32s} ({stats['games']:,}): median |ΔDTW| "
            f"{stats['median_abs_delta_dtw_pp']:.3f} pp, mean "
            f"{stats['mean_abs_delta_dtw_pp']:.3f} pp, max "
            f"{stats['max_abs_delta_dtw_pp']:.2f} pp; median |Δ deserved margin| "
            f"{stats['median_abs_delta_deserved_margin']:.3f} pts (mean signed "
            f"{stats['mean_signed_delta_deserved_margin']:+.4f}); flips {stats['side_flips']}"
        )

    # ---- the reconciliation ---------------------------------------------
    print(
        f"\n{'=' * 72}\n[§5d] reconciliation — per-row arithmetic against game movement\n{'=' * 72}"
    )
    rows = shipped_ledger.select(
        "game_id",
        "play_id",
        "component",
        pl.col("expected").alias("p_shipped"),
        pl.col("luck_epa").alias("luck_shipped"),
        "swing",
    ).join(
        fixed_ledger.select(
            "game_id",
            "play_id",
            "component",
            pl.col("expected").alias("p_fixed"),
            pl.col("luck_epa").alias("luck_fixed"),
        ),
        on=["game_id", "play_id", "component"],
        how="inner",
    )
    per_component = {}
    for component in ("field_goal", "extra_point"):
        frame = rows.filter(pl.col("component") == component)
        d_luck = (frame["luck_fixed"] - frame["luck_shipped"]).to_numpy()
        d_p = (frame["p_fixed"] - frame["p_shipped"]).to_numpy()
        per_component[component] = {
            "rows": frame.height,
            "mean_abs_delta_p_pp": float(np.abs(d_p).mean()) * 100,
            "mean_delta_p_pp": float(d_p.mean()) * 100,
            "mean_abs_delta_luck_epa": float(np.abs(d_luck).mean()),
            "max_abs_delta_luck_epa": float(np.abs(d_luck).max()),
            "signed_total_delta_luck_epa_home_perspective": float(d_luck.sum()),
            "document_27_14f_mean_abs_luck_error_epa": DOCUMENT_27_MEAN_ABS_LUCK_ERROR[component],
        }
        stats = per_component[component]
        print(
            f"  {component:12s} {stats['rows']:,} rows: mean Δp "
            f"{stats['mean_delta_p_pp']:+.3f} pp (|Δp| {stats['mean_abs_delta_p_pp']:.3f}), "
            f"mean |Δ luck| {stats['mean_abs_delta_luck_epa']:.4f} EPA "
            f"against document 27 §14f's "
            f"{DOCUMENT_27_MEAN_ABS_LUCK_ERROR[component]:.4f}"
        )

    total_change = float(rows["luck_fixed"].sum() - rows["luck_shipped"].sum())
    implied_points = -total_change * slope / max(len(kick_games), 1)
    measured_points = impact["all_games_with_a_kick"]["mean_signed_delta_deserved_margin"]
    reconciliation = {
        "per_component": per_component,
        "total_signed_luck_change_epa_home_perspective": total_change,
        "implied_mean_delta_deserved_margin": implied_points,
        "measured_mean_delta_deserved_margin": measured_points,
        "agreement": abs(implied_points - measured_points),
    }
    print(
        f"\n  Ledger luck is signed to the home team, so the two components' errors — which\n"
        f"  document 27 §14f reported in the kicking team's perspective as −385 and +128 EPA —\n"
        f"  largely cancel across teams: the home-perspective total moves {total_change:+.1f} EPA.\n"
        f"  That implies {implied_points:+.4f} points on the mean game; measured "
        f"{measured_points:+.4f}, gap {abs(implied_points - measured_points):.6f}.\n"
        f"  The per-row means above are the convention-free comparison, and they are what\n"
        f"  document 27 §14f's 0.0446 and 0.0100 EPA should be read against."
    )

    payload = {
        "gate_s1_round_trip": s1,
        "gate_s2_v12_reproduction": s2,
        "gate_s3_ledger_sums": s3,
        "impact": impact,
        "reconciliation": reconciliation,
        "points_per_epa": slope,
        "random_seed": RANDOM_SEED,
    }
    out = paths.RESEARCH_OUTPUT_DIR / "44_read_side_fix.json"
    with out.open("w") as handle:
        json.dump(payload, handle, indent=2, default=float)
    print(f"\nwrote {out}")
    print("\nVERDICT: gates S-1, S-2 and S-3 all PASS — the fix is correct and its size is above.")


if __name__ == "__main__":
    main()
