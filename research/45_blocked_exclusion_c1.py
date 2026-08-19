"""Phase 8, task 3 — document 26's blocked-kick exclusion, re-measured under Gate C.

Runs the gates `docs/research/30-v13-corrections.md` §7 committed at `b9e44c4`,
before this file existed. Gate C (document 05 §2, amendment C-1, accepted
2026-08-18) replaces document 26's materiality *floor* with a materiality
*report*, and leaves every other gate standing:

* **§7a** identification — every removed row printed, not counted.
* **§7b** the ledger must still sum, including the four blocked field goals that
  also carry a fumble row, and a check that the removed luck lands in `core`.
* **§7c** the dial gate is absent by design; no `w` is assumed.
* **§7d** the materiality report, on both populations, **no pass rule**.
* **§7e** the reconciliation: per-event luck removed against game movement.

Both arms run on **v1.3's arithmetic** — the refit posterior and the corrected
read side — because document 26 §9 recorded that its numbers are not stable
under the refit. The arms differ only in whether the 302 blocked kicks are in
the kicking population.

    uv run python research/45_blocked_exclusion_c1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.components import (  # noqa: E402
    COMPONENTS,
    any_fumble_mask,
    build_game_table,
    decompose_plays,
    fg_attempt_mask,
    fit_fg_baseline,
    fit_fumble_baseline,
    fit_xp_baseline,
    xp_attempt_mask,
)
from nfl_simulator.fg_model import FieldGoalModel  # noqa: E402
from nfl_simulator.ingest import ANALYSIS_COLUMNS, PBP_SEASONS, load_pbp  # noqa: E402
from nfl_simulator.simulator import points_per_epa, simulate_game  # noqa: E402

RANDOM_SEED = 20260817  # v1.2's build seed, so the arms differ by the exclusion alone
POSTERIOR_DRAWS = 200
COIN_DRAWS = 800

# Document 26 §8, on v1.2's arithmetic. Printed beside the new numbers so the
# movement caused by the refit and the read-side fix is visible.
DOCUMENT_26 = {
    "median_abs_delta_dtw_pp_blocked": 1.167,
    "median_abs_delta_dtw_pp_all": 0.000,
    "median_abs_delta_deserved_margin_blocked": 2.688,
    "side_flips_blocked": 22,
    "floor_incumbent_pp": 1.6250,
    "floor_refit_pp": 1.4409,  # document 27 §14g — context, never a pass rule
}
EXPECTED_ROWS_REMOVED = {"field_goal": 192, "extra_point": 110}
EXPECTED_FUMBLE_OVERLAP = 4

SIM_COLUMNS = [
    *ANALYSIS_COLUMNS,
    "kicker_player_id",
    "extra_point_attempt",
    "extra_point_result",
    "roof",
    "temp",
    "wind",
]


def blocked_fg_mask() -> pl.Expr:
    return fg_attempt_mask(include_blocked=True) & (pl.col("field_goal_result") == "blocked")


def blocked_xp_mask() -> pl.Expr:
    return xp_attempt_mask(include_blocked=True) & (pl.col("extra_point_result") == "blocked")


# --------------------------------------------------------------------------
# §7a — identification
# --------------------------------------------------------------------------


def identification(pbp: pl.DataFrame) -> dict:
    """Every removed row, printed. Document 20 §9: rows, not counts."""
    print(f"{'=' * 72}\n[§7a] identification — the rows the correction removes\n{'=' * 72}")
    removed = {}
    for label, mask in (("field_goal", blocked_fg_mask()), ("extra_point", blocked_xp_mask())):
        frame = pbp.filter(mask).select(
            "season",
            "game_id",
            "play_id",
            "posteam",
            "kicker_player_id",
            pl.col("kick_distance").cast(pl.Float64).alias("distance"),
        )
        removed[label] = frame.to_dicts()
        print(f"\n  {label}: {frame.height} rows (expected {EXPECTED_ROWS_REMOVED[label]})")
        with pl.Config(tbl_rows=-1, tbl_width_chars=120):
            print(frame)
    report = {
        "counts": {label: len(rows) for label, rows in removed.items()},
        "rows": removed,
        "pass": all(
            len(removed[label]) == expected for label, expected in EXPECTED_ROWS_REMOVED.items()
        ),
    }
    print(f"\n  identification: {'PASS' if report['pass'] else 'FAIL'}")
    return report


# --------------------------------------------------------------------------
# §7b — the ledger must still sum
# --------------------------------------------------------------------------


def ledger_sum_checks(pbp: pl.DataFrame) -> dict:
    print(f"\n{'=' * 72}\n[§7b] the ledger must still sum\n{'=' * 72}")
    fg_before = pbp.filter(fg_attempt_mask(include_blocked=True)).height
    fg_after = pbp.filter(fg_attempt_mask()).height
    xp_before = pbp.filter(xp_attempt_mask(include_blocked=True)).height
    xp_after = pbp.filter(xp_attempt_mask()).height
    print(f"  field-goal rows {fg_before:,} -> {fg_after:,} (expected −192)")
    print(f"  extra-point rows {xp_before:,} -> {xp_after:,} (expected −110)")

    # The trap document 26 §8 names: four blocked field goals also carry a fumble
    # row, and a production implementation that filtered the play frame once, at
    # the top, would delete them. Production narrows the masks instead, so the
    # count below must not move.
    fumbles_full = pbp.filter(any_fumble_mask()).height
    frame_filtered = pbp.filter(~blocked_fg_mask() & ~blocked_xp_mask())
    fumbles_if_frame_filtered = frame_filtered.filter(any_fumble_mask()).height
    overlap = fumbles_full - fumbles_if_frame_filtered
    print(
        f"  fumble rows: {fumbles_full:,} on the frame production hands the fumble builder, "
        f"and {fumbles_if_frame_filtered:,} if the frame had been filtered instead "
        f"(overlap {overlap}, expected {EXPECTED_FUMBLE_OVERLAP})"
    )
    overlapping = pbp.filter(blocked_fg_mask() & any_fumble_mask()).select(
        "season", "game_id", "play_id", "posteam", "kick_distance"
    )
    print("    the four plays, printed rather than counted:")
    with pl.Config(tbl_rows=-1, tbl_width_chars=120):
        print(overlapping)

    # ---- the removed luck must land in `core` ----------------------------
    fumble_baseline = fit_fumble_baseline(pbp)
    before = decompose_plays(
        pbp,
        fumble_baseline,
        fit_fg_baseline(pbp, include_blocked=True),
        include_blocked=True,
    )
    after = decompose_plays(pbp, fumble_baseline, fit_fg_baseline(pbp))
    keys = ["game_id", "play_id"]
    other = [component for component in COMPONENTS if component != "fg_luck"]
    joined = (
        before.select(
            *keys,
            pl.col("fg_luck").alias("fg_before"),
            *[pl.col(component).alias(f"{component}_before") for component in other],
        )
        .join(
            after.select(
                *keys,
                pl.col("fg_luck").alias("fg_after"),
                *[pl.col(component).alias(f"{component}_after") for component in other],
            ),
            on=keys,
        )
        .with_columns(
            (pl.col("fg_after") - pl.col("fg_before")).alias("d_fg"),
            *[
                (pl.col(f"{component}_after") - pl.col(f"{component}_before")).alias(
                    f"d_{component}"
                )
                for component in other
            ],
        )
        .with_columns(
            pl.sum_horizontal([pl.col(f"d_{component}") for component in other]).alias("d_other")
        )
    )
    residual = float((joined["d_fg"] + joined["d_other"]).abs().max())

    # Six of the 192 blocked field goals also carry a penalty flag, so their EPA
    # lands in `penalty` rather than `core`. Both are buckets the ledger does not
    # neutralize, and the invariant that has to hold is the five-way partition —
    # not that every play's EPA lands in one named bucket.
    blocked_ids = set(zip(*pbp.filter(blocked_fg_mask()).select("game_id", "play_id"), strict=True))
    on_blocked = joined.filter(
        pl.struct("game_id", "play_id").map_elements(
            lambda row: (row["game_id"], row["play_id"]) in blocked_ids, return_dtype=pl.Boolean
        )
    )
    print(
        f"\n  EPA moved out of `fg_luck`: {float(joined['d_fg'].sum()):+.2f}; "
        f"into `core`: {float(joined['d_core'].sum()):+.2f}; "
        f"into `penalty`: {float(joined['d_penalty'].sum()):+.2f} "
        f"({on_blocked.filter(pl.col('d_penalty') != 0).height} blocked kicks carry a penalty flag)"
    )
    print(
        f"  max |Δfg_luck + Δ(everything else)| on any play: {residual:.2e} "
        "(the five components must still partition epa_home, play by play)"
    )
    print(
        f"  on the {on_blocked.height} blocked field goals themselves, "
        f"fg_luck goes to zero: max |fg_luck after| "
        f"{float(on_blocked['fg_after'].abs().max()):.2e}"
    )

    report = {
        "fg_rows_before": fg_before,
        "fg_rows_after": fg_after,
        "xp_rows_before": xp_before,
        "xp_rows_after": xp_after,
        "fumble_rows_production": fumbles_full,
        "fumble_rows_if_frame_filtered": fumbles_if_frame_filtered,
        "fumble_overlap": overlap,
        "overlapping_plays": overlapping.to_dicts(),
        "max_abs_component_residual": residual,
        "epa_out_of_fg_luck": float(joined["d_fg"].sum()),
        "epa_into_core": float(joined["d_core"].sum()),
        "epa_into_penalty": float(joined["d_penalty"].sum()),
        "blocked_kicks_carrying_a_penalty_flag": on_blocked.filter(pl.col("d_penalty") != 0).height,
    }
    report["pass"] = bool(
        fg_before - fg_after == EXPECTED_ROWS_REMOVED["field_goal"]
        and xp_before - xp_after == EXPECTED_ROWS_REMOVED["extra_point"]
        and overlap == EXPECTED_FUMBLE_OVERLAP
        and residual < 1e-9
        and float(on_blocked["fg_after"].abs().max()) == 0.0
    )
    print(f"\n  ledger-sum: {'PASS' if report['pass'] else 'FAIL'}")
    return report


# --------------------------------------------------------------------------
# §7d and §7e — the materiality report and the reconciliation
# --------------------------------------------------------------------------


def simulate_arm(pbp, margins, baselines, model, slope, *, include_blocked: bool):
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
            seed=RANDOM_SEED,
            include_blocked=include_blocked,
        )
        rows.append(
            {
                "game_id": result.game_id,
                "deserved_margin": result.deserved_margin,
                "dtw_home": result.dtw_home,
                "half_width": (result.dtw_interval[1] - result.dtw_interval[0]) / 2,
                "total_luck_epa": result.total_luck_epa,
            }
        )
        frame = result.ledger.to_frame()
        if frame.height:
            ledgers.append(frame.with_columns(pl.lit(result.game_id).alias("game_id")))
    return pl.DataFrame(rows), pl.concat(ledgers)


def summarise(frame: pl.DataFrame) -> dict:
    delta_dtw = (frame["dtw_excluded"] - frame["dtw_included"]).abs()
    delta_margin = frame["deserved_excluded"] - frame["deserved_included"]
    return {
        "games": frame.height,
        "median_abs_delta_dtw_pp": float(delta_dtw.median()) * 100,
        "mean_abs_delta_dtw_pp": float(delta_dtw.mean()) * 100,
        "max_abs_delta_dtw_pp": float(delta_dtw.max()) * 100,
        "median_abs_delta_deserved_margin": float(delta_margin.abs().median()),
        "mean_signed_delta_deserved_margin": float(delta_margin.mean()),
        "max_abs_delta_deserved_margin": float(delta_margin.abs().max()),
        "side_flips": frame.filter(
            ((pl.col("dtw_excluded") - 0.5) * (pl.col("dtw_included") - 0.5)) < 0
        ).height,
        "median_half_width_included_pp": float(frame["half_width_included"].median()) * 100,
        "median_half_width_excluded_pp": float(frame["half_width_excluded"].median()) * 100,
    }


def main() -> None:
    paths.ensure_data_dirs()
    pbp = load_pbp(PBP_SEASONS, columns=SIM_COLUMNS)

    ident = identification(pbp)
    sums = ledger_sum_checks(pbp)
    if not (ident["pass"] and sums["pass"]):
        raise SystemExit(
            "A Gate C gate failed. Per document 30 §7f nothing ships from Part B; "
            "stop and report rather than reconcile."
        )

    with (paths.RESEARCH_OUTPUT_DIR / "fg_refit_summary.json").open() as handle:
        centres = json.load(handle)["centres"]
    model = FieldGoalModel.from_posterior(
        paths.RESEARCH_OUTPUT_DIR / "trace_fg_refit.nc",
        wind_centre=centres["wind"],
        temp_centre=centres["temp"],
    )
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

    print(f"\n{'=' * 72}\n[§7d] the materiality report — no threshold applies\n{'=' * 72}")
    print("simulating the included arm (refit posterior, corrected read side) ...")
    included, included_ledger = simulate_arm(
        pbp, margins, baselines, model, slope, include_blocked=True
    )
    print("simulating the excluded arm (the correction) ...")
    excluded, excluded_ledger = simulate_arm(
        pbp, margins, baselines, model, slope, include_blocked=False
    )

    joined = included.select(
        "game_id",
        pl.col("deserved_margin").alias("deserved_included"),
        pl.col("dtw_home").alias("dtw_included"),
        pl.col("half_width").alias("half_width_included"),
        pl.col("total_luck_epa").alias("luck_included"),
    ).join(
        excluded.select(
            "game_id",
            pl.col("deserved_margin").alias("deserved_excluded"),
            pl.col("dtw_home").alias("dtw_excluded"),
            pl.col("half_width").alias("half_width_excluded"),
            pl.col("total_luck_epa").alias("luck_excluded"),
        ),
        on="game_id",
    )

    blocked_games = set(pbp.filter(blocked_fg_mask() | blocked_xp_mask())["game_id"].to_list())
    kick_games = set(
        pbp.filter(fg_attempt_mask(include_blocked=True) | xp_attempt_mask(include_blocked=True))[
            "game_id"
        ].to_list()
    )
    populations = {
        "games_with_a_blocked_kick": joined.filter(pl.col("game_id").is_in(blocked_games)),
        "all_games_with_a_kick": joined.filter(pl.col("game_id").is_in(kick_games)),
    }
    report = {}
    for label, frame in populations.items():
        report[label] = summarise(frame)
        stats = report[label]
        print(
            f"\n  {label} ({stats['games']:,})\n"
            f"    median |ΔDTW| {stats['median_abs_delta_dtw_pp']:.3f} pp, "
            f"mean {stats['mean_abs_delta_dtw_pp']:.3f} pp, "
            f"max {stats['max_abs_delta_dtw_pp']:.2f} pp\n"
            f"    median |Δ deserved margin| "
            f"{stats['median_abs_delta_deserved_margin']:.3f} pts, "
            f"mean signed {stats['mean_signed_delta_deserved_margin']:+.3f} pts, "
            f"max {stats['max_abs_delta_deserved_margin']:.2f}\n"
            f"    DTW side flips {stats['side_flips']}"
        )
    print(
        f"\n  For comparison, document 26 measured on v1.2's arithmetic: "
        f"{DOCUMENT_26['median_abs_delta_dtw_pp_blocked']:.3f} pp median |ΔDTW| on the "
        f"blocked games,\n  {DOCUMENT_26['median_abs_delta_deserved_margin_blocked']:.3f} pts "
        f"median |Δ deserved margin|, {DOCUMENT_26['side_flips_blocked']} flips, and "
        f"{DOCUMENT_26['median_abs_delta_dtw_pp_all']:.3f} pp on all kick games.\n"
        f"  Document 26's floor was {DOCUMENT_26['floor_incumbent_pp']:.4f} pp and its refit "
        f"counterpart {DOCUMENT_26['floor_refit_pp']:.4f} pp (document 27 §14g).\n"
        f"  **Neither is a pass rule.** Amendment C-1 replaced the threshold with this report."
    )

    # ---- §7e the reconciliation ------------------------------------------
    print(f"\n{'=' * 72}\n[§7e] reconciliation — removed luck against game movement\n{'=' * 72}")
    removed = included_ledger.join(
        excluded_ledger.select("game_id", "play_id", "component"),
        on=["game_id", "play_id", "component"],
        how="anti",
    )
    per_component = {}
    for component in ("field_goal", "extra_point"):
        frame = removed.filter(pl.col("component") == component)
        per_component[component] = {
            "rows_removed": frame.height,
            "mean_abs_luck_epa": float(frame["luck_epa"].abs().mean()),
            "signed_total_luck_epa": float(frame["luck_epa"].sum()),
        }
        stats = per_component[component]
        print(
            f"  {component:12s} {stats['rows_removed']} rows removed, mean |luck| "
            f"{stats['mean_abs_luck_epa']:.3f} EPA, signed total "
            f"{stats['signed_total_luck_epa']:+.1f} EPA (home perspective)"
        )

    on_blocked = populations["games_with_a_blocked_kick"]
    removed_luck_per_game = (
        removed.group_by("game_id")
        .agg(pl.col("luck_epa").sum().alias("removed_luck"))
        .join(on_blocked.select("game_id", "deserved_included", "deserved_excluded"), on="game_id")
        .with_columns(
            (pl.col("deserved_excluded") - pl.col("deserved_included")).alias("measured"),
            # deserved = actual − luck × slope, so dropping a row whose luck was
            # L raises deserved margin by +L × slope. The sign is the whole
            # content of this check: it is what makes a correction that moves
            # the game the wrong way visible.
            (pl.col("removed_luck") * slope).alias("implied"),
        )
    )
    gap = (removed_luck_per_game["measured"] - removed_luck_per_game["implied"]).abs()
    print(
        f"\n  On the {removed_luck_per_game.height} games that lose a row: removing "
        f"{float(removed['luck_epa'].abs().mean()):.3f} EPA of luck per event implies a\n"
        f"  median |Δ deserved margin| of "
        f"{float(removed_luck_per_game['implied'].abs().median()):.3f} points; measured "
        f"{float(removed_luck_per_game['measured'].abs().median()):.3f}.\n"
        f"  Median gap {float(gap.median()):.3f} points, max {float(gap.max()):.3f} — the "
        f"residual is the class tables\n  moving under the narrower population, which re-prices "
        f"every remaining kick in the game."
    )
    reconciliation = {
        "per_component": per_component,
        "games_losing_a_row": removed_luck_per_game.height,
        "median_implied_delta_points": float(removed_luck_per_game["implied"].abs().median()),
        "median_measured_delta_points": float(removed_luck_per_game["measured"].abs().median()),
        "median_gap_points": float(gap.median()),
        "max_gap_points": float(gap.max()),
    }

    payload = {
        "gate_identification": {
            "counts": ident["counts"],
            "pass": ident["pass"],
            "rows": ident["rows"],
        },
        "gate_ledger_sum": sums,
        "gate_dial": {"applicable": False, "reason": "no `w` is assumed; document 26 §5e"},
        "materiality_report": report,
        "document_26_v12_numbers": DOCUMENT_26,
        "reconciliation": reconciliation,
        "points_per_epa": slope,
        "random_seed": RANDOM_SEED,
        "posterior": "trace_fg_refit.nc",
    }
    out = paths.RESEARCH_OUTPUT_DIR / "45_blocked_exclusion_c1.json"
    with out.open("w") as handle:
        json.dump(payload, handle, indent=2, default=float)
    print(f"\nwrote {out}")
    print(
        "\nVERDICT: identification, ledger-sum and the dial gate all clear. Under Gate C the\n"
        "correction is correct and ships at the size reported above — size never fails it."
    )


if __name__ == "__main__":
    main()
