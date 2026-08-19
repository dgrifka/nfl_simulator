"""Phase 8, task 4 — simulator v1.3: three approved corrections, rebuilt and decomposed.

Three verdicts reach the simulator, all approved by the maintainer on 2026-08-18:

1. **The make-probability refit** (document 27) — the field-goal model is fitted
   on the population that excludes blocked kicks, and the simulator reads
   `trace_fg_refit.nc`.
2. **The read-side fix** (document 30 Part A) — `delta_cubic`, `delta_xp` and
   `lambda_xp` are applied, so the product prices what the model fitted.
3. **The blocked-kick exclusion** (document 30 Part B, under Gate C) — 302 kicks
   Gate A denies leave the ledger.

v1.1's and v1.2's artifacts are left untouched. v1.3 writes alongside, as every
version before it did:

    research/outputs/dtw_games_v13.parquet
    research/outputs/dtw_ledger_v13.parquet
    research/outputs/model_metadata_v13.json
    research/outputs/46_ledger_delta.json

**Six arms are simulated and five of them are not shipped.** Four isolate one
correction each so the release decomposes (document 30 §8); the sixth is v1.3 at
a different seed, because removing rows shifts the shared random stream and the
Monte Carlo floor has to be measured rather than assumed.

    uv run python research/46_simulator_v13.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_read_side = import_module("44_read_side_fix")
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
from nfl_simulator.fg_model import FieldGoalModel  # noqa: E402
from nfl_simulator.ingest import PBP_SEASONS, load_pbp  # noqa: E402
from nfl_simulator.simulator import points_per_epa, simulate_game  # noqa: E402

RANDOM_SEED = 20260817  # v1.1's and v1.2's, so the arms differ by the change alone
NOISE_SEED = 20260819  # the reshuffle arm; never shipped
POSTERIOR_DRAWS = 200
COIN_DRAWS = 800

SIM_COLUMNS = _read_side.SIM_COLUMNS

# Deterministic numbers published before this script existed. A mismatch here is
# a stop, not a reconciliation.
EXPECTED = {
    "fg_rows_v12": 10731,  # document 05b §11
    "xp_rows_v12": 12818,
    "fg_rows_v13": 10539,  # documents 26 §8, 30 §12
    "xp_rows_v13": 12708,
    "fumble_rows": 6505,  # document 19 §1
    "kicks_in_the_refit_population": 23247,  # document 27 §13
    "kicker_seasons_refit": 432,  # document 27 §14c
    "points_per_epa": 0.8389,  # document 27 §13
}
# Document 27 §14c's fitted table, which the shipped posterior must reproduce.
EXPECTED_PARAMETERS = {
    "alpha": 1.9068,
    "beta": -0.11587,
    "gamma": 0.2489,
    "delta_cubic": -0.0811,
    "delta_xp": 0.1222,
    "lambda_xp": 1.2472,
}
# Document 27 §14c's published league curve for the refit. **These numbers were
# printed through the defective read side** — `league_curves` in
# `research/42_fg_refit.py` calls `FieldGoalModel.league_make_probability`, which
# at the time dropped `delta_cubic`. They are therefore a quadratic reading of a
# cubic posterior, and v1.3 supersedes them. The check below verifies both
# halves of that claim: the corrected read side must equal the posterior's own
# arithmetic, and the read side with the cubic stripped must still reproduce the
# published table — which is what proves the discrepancy is the defect and not a
# different posterior.
DOCUMENT_27_QUADRATIC_CURVE = {
    30.0: 0.9648,
    40.0: 0.8705,
    45.0: 0.8004,
    50.0: 0.7303,
    55.0: 0.6739,
}


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=paths.REPO_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_models() -> tuple[FieldGoalModel, FieldGoalModel]:
    """The incumbent and refit posteriors, both read through the corrected side."""
    incumbent, _ = _read_side.load_model("trace_fg_weather.nc", "fg_weather_summary.json")
    refit, _ = _read_side.load_model("trace_fg_refit.nc", "fg_refit_summary.json")
    return incumbent, refit


def reproduction_checks(pbp: pl.DataFrame, refit: FieldGoalModel, slope: float) -> dict:
    """Every deterministic number this ship claims, checked against its source."""
    print(
        f"\n{'=' * 72}\nREPRODUCTION CHECK — deterministic numbers, against their sources\n{'=' * 72}"
    )
    observed = {
        "fg_rows_v12": pbp.filter(fg_attempt_mask(include_blocked=True)).height,
        "xp_rows_v12": pbp.filter(xp_attempt_mask(include_blocked=True)).height,
        "fg_rows_v13": pbp.filter(fg_attempt_mask()).height,
        "xp_rows_v13": pbp.filter(xp_attempt_mask()).height,
        "fumble_rows": None,  # filled by the caller from the shipped ledger
        "kicks_in_the_refit_population": _weather.load_kicks(exclude_blocked=True).height,
        "kicker_seasons_refit": len(refit.kicker_effects),
        "points_per_epa": round(slope, 4),
    }
    return observed


def report_reproduction(observed: dict) -> dict:
    disagreements = []
    for key, expected in EXPECTED.items():
        got = observed[key]
        agrees = abs(got - expected) <= (5e-5 if isinstance(expected, float) else 0)
        print(f"  {key:<32} {got!s:<10} expected {expected}  {'ok' if agrees else 'MISMATCH'}")
        if not agrees:
            disagreements.append(key)
    if disagreements:
        raise SystemExit(
            f"v1.3 disagrees with its own sources on {disagreements}. "
            "Stop and report rather than reconcile."
        )
    return {"observed": observed, "expected": EXPECTED, "pass": True}


def parameter_checks(refit: FieldGoalModel) -> dict:
    """The shipped posterior is the one document 27 §14c reported."""
    print("\n  fitted parameters, as loaded by the production read side:")
    got = {
        "alpha": float(refit.alpha.mean()),
        "beta": float(refit.beta.mean()),
        "gamma": float(refit.gamma.mean()),
        "delta_cubic": float(refit.delta_cubic.mean()),
        "delta_xp": float(refit.delta_xp.mean()),
        "lambda_xp": float(refit.lambda_xp.mean()),
    }
    disagreements = []
    for name, expected in EXPECTED_PARAMETERS.items():
        agrees = abs(got[name] - expected) <= 5e-4
        print(
            f"    {name:<12} {got[name]:+.5f}  document 27 §14c {expected:+.4f}  "
            f"{'ok' if agrees else 'MISMATCH'}"
        )
        if not agrees:
            disagreements.append(name)

    # The posterior's own arithmetic, written out here rather than called, so the
    # read side is checked against an independent statement of the fitted curve.
    def fitted_curve(distance: float) -> float:
        centred = distance - refit.distance_centre
        eta = (
            refit.alpha
            + refit.beta * centred
            + refit.gamma * centred**2 / 100.0
            + refit.delta_cubic * centred**3 / 1000.0
        )
        return float((1.0 / (1.0 + np.exp(-eta))).mean())

    stripped = _read_side.shipped_read_side(refit)
    print("\n  league make rate, average kicker, outdoors, no reading:")
    print("    dist |  fitted |  v1.3 read side | v1.2 read side | document 27 §14c")
    curve = {}
    for distance, published in DOCUMENT_27_QUADRATIC_CURVE.items():
        fitted = fitted_curve(distance)
        shipped = float(refit.league_make_probability(distance).mean())
        quadratic = float(stripped.league_make_probability(distance).mean())
        curve[distance] = {
            "fitted": fitted,
            "v13_read_side": shipped,
            "v12_read_side": quadratic,
            "document_27_published": published,
        }
        agrees = abs(shipped - fitted) <= 1e-9 and abs(quadratic - published) <= 5e-4
        print(
            f"    {distance:4.0f} | {fitted * 100:6.2f}% | {shipped * 100:14.2f}% | "
            f"{quadratic * 100:13.2f}% | {published * 100:6.2f}%  "
            f"{'ok' if agrees else 'MISMATCH'}"
        )
        if not agrees:
            disagreements.append(f"curve@{distance:.0f}")
    print(
        "\n    Document 27 §14c's curve reproduces exactly when the cubic term is stripped, so\n"
        "    the published table was a quadratic reading of a cubic posterior — the same defect\n"
        "    §14f found, applied to §14c's own numbers. v1.3 supersedes it: a 55-yarder is\n"
        f"    {curve[55.0]['fitted'] * 100:.2f}%, not {DOCUMENT_27_QUADRATIC_CURVE[55.0] * 100:.2f}%."
    )
    if disagreements:
        raise SystemExit(f"the shipped posterior is not document 27's on {disagreements}.")
    return {
        "parameters": got,
        "league_curve": curve,
        "document_27_14c_curve_was_quadratic": True,
        "pass": True,
    }


def round_trip_check() -> list[dict]:
    """Gate S-1, now a permanent item on the ship template (document 31 §7)."""
    print(f"\n{'=' * 72}\nROUND TRIP — the read side against the fit it ships\n{'=' * 72}")
    reports = [
        _read_side.round_trip(
            "shipped posterior (refit)",
            "trace_fg_refit.nc",
            "fg_refit_summary.json",
            exclude_blocked=True,
        ),
        _read_side.round_trip(
            "incumbent (v1.1 / v1.2 replay)",
            "trace_fg_weather.nc",
            "fg_weather_summary.json",
            exclude_blocked=False,
        ),
    ]
    if not all(report["pass"] for report in reports):
        raise SystemExit("the round trip failed — the product would not price what it fitted.")
    return reports


# --------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------


def simulate_arm(pbp, margins, baselines, model, slope, *, include_blocked, seed=RANDOM_SEED):
    rows, ledgers = [], []
    for game_id, group in pbp.group_by("game_id"):
        game_id = game_id[0] if isinstance(game_id, tuple) else game_id
        if margins.get(game_id) is None:
            continue
        result = simulate_game(
            group,
            fumble_baseline=baselines["fumble"],
            fg_baseline=baselines["fg_included"] if include_blocked else baselines["fg_excluded"],
            xp_baseline=baselines["xp_included"] if include_blocked else baselines["xp_excluded"],
            fg_model=model,
            points_per_epa=slope,
            n_posterior_draws=POSTERIOR_DRAWS,
            n_coin_draws=COIN_DRAWS,
            seed=seed,
            include_blocked=include_blocked,
        )
        rows.append(
            {
                "game_id": result.game_id,
                "actual_margin": result.actual_margin,
                "deserved_margin": result.deserved_margin,
                "dtw_home": result.dtw_home,
                "dtw_low": result.dtw_interval[0],
                "dtw_high": result.dtw_interval[1],
                "total_luck_epa": result.total_luck_epa,
                "n_luck_events": len(result.ledger),
            }
        )
        frame = result.ledger.to_frame()
        if frame.height:
            ledgers.append(frame.with_columns(pl.lit(result.game_id).alias("game_id")))
    return pl.DataFrame(rows), pl.concat(ledgers)


def compare(reference: pl.DataFrame, arm: pl.DataFrame, label: str) -> dict:
    joined = reference.select(
        "game_id",
        pl.col("deserved_margin").alias("deserved_ref"),
        pl.col("dtw_home").alias("dtw_ref"),
    ).join(
        arm.select("game_id", "deserved_margin", "dtw_home"),
        on="game_id",
    )
    margin_delta = joined["deserved_margin"] - joined["deserved_ref"]
    dtw_delta = (joined["dtw_home"] - joined["dtw_ref"]).abs()
    stats = {
        "arm": label,
        "games": joined.height,
        "median_abs_delta_dtw_pp": float(dtw_delta.median()) * 100,
        "mean_abs_delta_dtw_pp": float(dtw_delta.mean()) * 100,
        "max_abs_delta_dtw_pp": float(dtw_delta.max()) * 100,
        "median_abs_delta_deserved_margin": float(margin_delta.abs().median()),
        "mean_abs_delta_deserved_margin": float(margin_delta.abs().mean()),
        "mean_signed_delta_deserved_margin": float(margin_delta.mean()),
        "max_abs_delta_deserved_margin": float(margin_delta.abs().max()),
        "side_flips": joined.filter(
            ((pl.col("dtw_home") - 0.5) * (pl.col("dtw_ref") - 0.5)) < 0
        ).height,
    }
    print(
        f"  {label:<34} median |ΔDTW| {stats['median_abs_delta_dtw_pp']:6.3f} pp, "
        f"mean {stats['mean_abs_delta_dtw_pp']:6.3f} pp | median |Δ margin| "
        f"{stats['median_abs_delta_deserved_margin']:5.3f} pts, mean "
        f"{stats['mean_abs_delta_deserved_margin']:5.3f} | flips {stats['side_flips']:>3}"
    )
    return stats


def main() -> None:
    paths.ensure_data_dirs()
    pbp = load_pbp(PBP_SEASONS, columns=SIM_COLUMNS)
    incumbent, refit = load_models()
    shipped_read_side = _read_side.shipped_read_side

    baselines = {
        "fumble": fit_fumble_baseline(pbp),
        "fg_included": fit_fg_baseline(pbp, include_blocked=True),
        "fg_excluded": fit_fg_baseline(pbp),
        "xp_included": fit_xp_baseline(pbp, include_blocked=True),
        "xp_excluded": fit_xp_baseline(pbp),
    }
    games = build_game_table(pbp)
    slope = points_per_epa(games.drop_nulls("margin"))
    margins = dict(zip(games["game_id"], games["margin"], strict=True))

    observed = reproduction_checks(pbp, refit, slope)
    parameters = parameter_checks(refit)
    round_trips = round_trip_check()

    # ---- the six arms ----------------------------------------------------
    print(f"\n{'=' * 72}\nSIMULATING — six arms, five of them never shipped\n{'=' * 72}")
    arms = {}
    for label, model, include_blocked, seed in (
        ("v1.2 (reference)", shipped_read_side(incumbent), True, RANDOM_SEED),
        ("A — read-side fix alone", incumbent, True, RANDOM_SEED),
        ("B — refit alone", shipped_read_side(refit), True, RANDOM_SEED),
        ("C — exclusion alone", shipped_read_side(incumbent), False, RANDOM_SEED),
        ("v1.3 (combined)", refit, False, RANDOM_SEED),
        ("v1.3 reshuffle", refit, False, NOISE_SEED),
    ):
        print(f"  {label} ...")
        arms[label] = simulate_arm(
            pbp, margins, baselines, model, slope, include_blocked=include_blocked, seed=seed
        )

    table, ledger = arms["v1.3 (combined)"]
    observed["fumble_rows"] = ledger.filter(pl.col("component") == "fumble").height
    reproduction = report_reproduction(observed)

    # ---- v1.2 must still replay exactly ----------------------------------
    v12 = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / "dtw_games_v12.parquet")
    replay = v12.select("game_id", "deserved_margin", "dtw_home").join(
        arms["v1.2 (reference)"][0].select(
            "game_id",
            pl.col("deserved_margin").alias("replayed_margin"),
            pl.col("dtw_home").alias("replayed_dtw"),
        ),
        on="game_id",
    )
    replay_gap = float((replay["deserved_margin"] - replay["replayed_margin"]).abs().max())
    print(
        f"\n  v1.2 replayed under v1.3's code: {replay.height:,} games, "
        f"max |Δ deserved margin| {replay_gap:.2e}"
    )
    if replay_gap > 1e-9 or replay.height != v12.height:
        raise SystemExit("v1.3's code no longer reproduces v1.2. Stop and report.")

    # ---- the decomposition ------------------------------------------------
    print(f"\n{'=' * 72}\nDECOMPOSITION — each correction alone, against v1.2\n{'=' * 72}")
    reference = arms["v1.2 (reference)"][0]
    decomposition = [
        compare(reference, arms[label][0], label)
        for label in (
            "A — read-side fix alone",
            "B — refit alone",
            "C — exclusion alone",
            "v1.3 (combined)",
        )
    ]
    floor = compare(table, arms["v1.3 reshuffle"][0], "Monte Carlo floor (v1.3, other seed)")
    print(
        "\n  The combined row is not the sum of the three, and document 30 §8 said so before\n"
        "  the numbers existed: DTW is a bounded non-linear function of luck, the refit\n"
        "  raises the p_make the exclusion then removes, and dropping rows reshuffles the\n"
        "  shared random stream. The last row is that reshuffle, measured rather than assumed."
    )

    # ---- ledger row counts -------------------------------------------------
    v12_ledger = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / "dtw_ledger_v12.parquet")
    counts = (
        ledger.group_by("component")
        .agg(pl.len().alias("v13"))
        .join(
            v12_ledger.group_by("component").agg(pl.len().alias("v12")),
            on="component",
            how="full",
            coalesce=True,
        )
        # Cast first: both counts are u32, and 10,539 − 10,731 underflows to a
        # nine-digit positive number rather than −192.
        .with_columns((pl.col("v13").cast(pl.Int64) - pl.col("v12").cast(pl.Int64)).alias("change"))
        .sort("component")
    )
    print(f"\n{'=' * 72}\nLEDGER ROWS\n{'=' * 72}")
    print(counts)
    identity = float(
        (table["deserved_margin"] - (table["actual_margin"] - table["total_luck_epa"] * slope))
        .abs()
        .max()
    )
    print(f"  max |deserved − (actual − luck × slope)| = {identity:.2e}")
    if identity > 1e-9:
        raise SystemExit("the v1.3 ledger does not sum. Stop and report.")

    # ---- write the artifacts ----------------------------------------------
    table = table.join(
        games.select("game_id", "season", "week", "home_team", "away_team"), on="game_id"
    )
    table.write_parquet(paths.RESEARCH_OUTPUT_DIR / "dtw_games_v13.parquet")
    ledger.write_parquet(paths.RESEARCH_OUTPUT_DIR / "dtw_ledger_v13.parquet")

    with (paths.RESEARCH_OUTPUT_DIR / "fg_refit_summary.json").open() as handle:
        refit_summary = json.load(handle)

    delta = {
        "reproduction_check": reproduction,
        "parameter_check": parameters,
        "round_trip": round_trips,
        "v12_replay_max_abs_margin_gap": replay_gap,
        "decomposition": decomposition,
        "monte_carlo_floor": floor,
        "ledger_rows": counts.to_dicts(),
        "ledger_identity_residual": identity,
    }

    metadata = {
        "version": "simulator-v1.3",
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "seasons": list(PBP_SEASONS),
        "games_simulated": table.height,
        "random_seed": RANDOM_SEED,
        "posterior_draws": POSTERIOR_DRAWS,
        "coin_draws": COIN_DRAWS,
        "points_per_epa": slope,
        "field_goal_posterior": "research/outputs/trace_fg_refit.nc",
        "field_goal_centres": refit_summary["centres"],
        "changes_from_v12": [
            "the field-goal model is refitted on the population that excludes blocked "
            "kicks, and the simulator reads trace_fg_refit.nc (document 27)",
            "the read side applies delta_cubic, delta_xp and lambda_xp, which were fitted "
            "in Phase 3 and never reached the product (document 30 Part A)",
            "blocked kicks leave the field-goal and extra-point components under Gate C "
            "(documents 26 and 30 Part B)",
        ],
        "component_treatment": {
            "fumble_retention": {
                "treatment": "full",
                "expectation": "league retention rate for the fumble's class",
                "population": "all fumbles with a resolved disposition",
                "w": 0.015,
            },
            "field_goal": {
                "treatment": "partial",
                "expectation": "kicker's shrunk make probability at that distance, "
                "adjusted for roof, wind and temperature",
                "population": "field-goal attempts that were not blocked",
                "w_median": 0.336,
                "w_note": "implied by the refit's sigma_kicker under the incumbent's sampling "
                "variance (document 27 §14c). The published 0.285 has no derivation in this "
                "repository, which is an open defect",
                "model": "docs/research/05b §11 refit column, cubic arm with weather",
                "blocked_kicks": "core — Gate A denies them (document 05 §2 Gate C)",
            },
            "extra_point": {
                "treatment": "partial",
                "expectation": "kicker's shrunk extra-point probability, through the fitted "
                "delta_xp offset and lambda_xp transfer",
                "population": "extra-point attempts that were not blocked",
                "model": "docs/research/09 §8, folded into the kicker model",
                "league_rate": baselines["xp_excluded"].p_make,
                "swing_epa": baselines["xp_excluded"].swing_value,
                "blocked_kicks": "core — Gate A denies them",
            },
            "interception": {
                "treatment": "none",
                "reason": "step 3a could not attribute the spread to an entity",
            },
            "penalty": {"treatment": "none", "reason": "fails the branch-point gate (05 §2)"},
            "return_yardage": {"treatment": "none", "reason": "fails the branch-point gate"},
            "onside_recovery": {
                "treatment": "none",
                "reason": "passes the branch-point gate but 599 kicks cannot size the spread",
            },
            "overtime_toss": {
                "treatment": "none",
                "reason": "measured at +2.05 points but below the materiality floor (16 §8)",
            },
            "deflected_interception": {
                "treatment": "none",
                "reason": "the denominator is invisible in every available dataset (17 §1)",
            },
            "kickoff_muff": {
                "treatment": "none",
                "reason": "245 of 263 muffs are invisible in the charting (24)",
            },
            "sequencing": {
                "treatment": "none",
                "reason": "no branch point; reported separately, never as ledger rows (08 §6)",
            },
        },
        "rematch_revalidation": {
            "run": True,
            "reason": "document 27 §10: a change to p_make on every kick is the kind of change "
            "document 05b §11's weather round re-earned document 06's Gate 1 for. Reported as a "
            "check, not a gate, because document 12 measured the rematch test as nearly blind "
            "below ~20% damage.",
            "results": "research/outputs/47_rematch_v13.json",
        },
        "delta_from_v12": delta,
    }
    out = paths.RESEARCH_OUTPUT_DIR / "model_metadata_v13.json"
    with out.open("w") as handle:
        json.dump(metadata, handle, indent=2, default=str)
    with (paths.RESEARCH_OUTPUT_DIR / "46_ledger_delta.json").open("w") as handle:
        json.dump(delta, handle, indent=2, default=str)
    print(f"\nwrote {out}")
    print("Next: research/47_rematch_v13.py for the rematch check.")


if __name__ == "__main__":
    main()
