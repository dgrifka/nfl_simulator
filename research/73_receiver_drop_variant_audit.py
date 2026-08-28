"""Part D of round 7 — gates G-4b and G-4d, and G-5's four-way audit.

Document 56 §3. Four arms are simulated over 2022-2025 at v1.3's settings —
**strict**, **`+dp`**, **`+rd`** and **`+dp+rd`** — and the round's remaining
gates are read off them:

    V-1   v1.3 is untouched, 0.00e+00 over 2,761 games. First, and again last.
    V-2   the variant ledger sums to the margin it moved, on every arm.
    V-6   Gate C-1's sampler bars on the fit, from `research/72`'s summary.
    V-8   the catch-probability bound on a median target, likewise.
    G-4b  V-1..V-8 together.
    G-4d  document 52 §5's G-3 materiality floor, on the `+rd`-only arm.
    G-5   reported, never gated, never tuned: the four-way comparison.

**Handoff constraint 1, discharged rather than asserted.** The `+dp` arm here is
run through `research/68`'s own `variant_pass`, which round 7 extended with an
additive `receiver_drop_model=None` argument. That extension is only safe if the
`+dp` arm still reproduces document 55's numbers when run alone, so this script
checks it against 136 bucket moves and 1.59 pp median \\|ΔDTW\\| before any G-5
number is read, and stops if it does not.

    uv run python research/73_receiver_drop_variant_audit.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_audit = import_module("68_dropped_pick_variant_audit")
_receiver = import_module("71_receiver_drop_power")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.dropped_picks import DroppedPickModel  # noqa: E402
from nfl_simulator.ingest import FTN_SEASONS  # noqa: E402
from nfl_simulator.receiver_drops import ReceiverDropModel  # noqa: E402

TRACE_NAME = "trace_receiver_drop.nc"
SUMMARY_NAME = "receiver_drop_summary.json"
OUTPUT_NAME = "73_receiver_drop_variant_audit.json"

# Document 55 / document 54 F-3's in-sample arm, the numbers the `+dp` arm must
# still produce now that `68.simulate_all` takes a second model.
DOC55_BUCKET_MOVES = 136
DOC55_MEDIAN_ABS_DELTA_DTW_PP = 1.59
DOC55_MOVE_TOLERANCE = 5
DOC55_DTW_TOLERANCE_PP = 0.2

NAMED_GAMES = _audit.NAMED_GAMES
ETI_LOW, ETI_HIGH = _audit.ETI_LOW, _audit.ETI_HIGH


def read_side_round_trip(model: ReceiverDropModel) -> dict:
    """The production read side against the posterior's own arithmetic.

    Document 30 §5a made this a gate rather than a hope on the field-goal model
    after the formality found a real defect, and `research/68` repeated it for
    the dropped pick. Same formality, same shape, on the fit round 7 adds.
    """
    import arviz as az

    frame = _receiver.build_catchable_frame(verbose=False)
    posterior = az.from_netcdf(paths.RESEARCH_OUTPUT_DIR / TRACE_NAME)["posterior"]
    alpha = posterior["alpha"].values.ravel()
    beta = posterior["beta"].values.reshape(len(alpha), frame.design_matrix.shape[1])
    effects = posterior["r_s"]
    levels = [str(level) for level in effects.coords["entity_season"].values]
    draws = effects.values.reshape(len(alpha), len(levels))
    lookup = {level: index for index, level in enumerate(levels)}

    labels = frame.model.select(
        pl.concat_str([pl.col("season").cast(pl.String), pl.col("posteam")], separator="|").alias(
            "label"
        )
    )["label"].to_list()

    fitted = np.empty(frame.model.height)
    read = np.empty(frame.model.height)
    for index, row in enumerate(frame.model.iter_rows(named=True)):
        eta = alpha + beta @ frame.design_matrix[index] + draws[:, lookup[labels[index]]]
        fitted[index] = (1.0 - 1.0 / (1.0 + np.exp(-eta))).mean()
        read[index] = model.catch_probability(labels[index], row).mean()

    delta = float(np.abs(read - fitted).max())
    report = {
        "rows": int(frame.model.height),
        "max_abs_diff": delta,
        "tolerance": _audit.ROUND_TRIP_TOLERANCE,
        "pass": bool(delta <= _audit.ROUND_TRIP_TOLERANCE),
    }
    print(
        f"\nROUND TRIP — read side against the fit, {report['rows']:,} rows: "
        f"max |read − fitted| {delta:.2e}  -> {'PASS' if report['pass'] else 'FAIL'}"
    )
    if not report["pass"]:
        raise SystemExit(
            "the read side does not price what the model fitted. Stop and report — "
            "this is document 30's defect in a new place."
        )
    return report


def bucket_moves(v13: pl.DataFrame, arm: pl.DataFrame) -> tuple[set[str], np.ndarray, np.ndarray]:
    """The set of games whose verdict bucket moves, and the per-game deltas."""
    joined = (
        v13.select(
            "game_id",
            "actual_margin",
            pl.col("dtw_home").alias("dtw_v13"),
            pl.col("dtw_low").alias("low_v13"),
            pl.col("dtw_high").alias("high_v13"),
            pl.col("deserved_margin").alias("margin_v13"),
        )
        .join(
            arm.select(
                "game_id",
                pl.col("dtw_home").alias("dtw_arm"),
                pl.col("dtw_low").alias("low_arm"),
                pl.col("dtw_high").alias("high_arm"),
                pl.col("deserved_margin").alias("margin_arm"),
            ),
            on="game_id",
        )
        .sort("game_id")
    )
    moved = set()
    for row in joined.iter_rows(named=True):
        before = _audit.bucket(row["dtw_v13"], row["actual_margin"])
        after = _audit.bucket(row["dtw_arm"], row["actual_margin"])
        if before != after:
            moved.add(row["game_id"])
    delta_dtw = (joined["dtw_arm"] - joined["dtw_v13"]).to_numpy()
    half_width = (joined["high_v13"] - joined["low_v13"]).to_numpy() / 2.0
    return moved, delta_dtw, half_width


def doc55_reproduction(v13: pl.DataFrame, dp: pl.DataFrame) -> dict:
    """Handoff constraint 1: the `+dp` arm run alone is still round 6's arm."""
    moved, delta_dtw, _ = bucket_moves(v13, dp)
    affected = dp.sort("game_id")["n_dropped_pick_events"].to_numpy() > 0
    median_pp = float(np.median(np.abs(delta_dtw[affected]))) * 100
    move_gap = abs(len(moved) - DOC55_BUCKET_MOVES)
    dtw_gap = abs(median_pp - DOC55_MEDIAN_ABS_DELTA_DTW_PP)
    report = {
        "statistic": "the `+dp` arm alone against document 55's in-sample arm",
        "bucket_moves": len(moved),
        "bucket_moves_doc55": DOC55_BUCKET_MOVES,
        "median_abs_delta_dtw_pp": median_pp,
        "median_abs_delta_dtw_pp_doc55": DOC55_MEDIAN_ABS_DELTA_DTW_PP,
        "affected_games": int(affected.sum()),
        "pass": bool(move_gap <= DOC55_MOVE_TOLERANCE and dtw_gap <= DOC55_DTW_TOLERANCE_PP),
    }
    print(
        f"\n{'=' * 72}\nHANDOFF CONSTRAINT 1 — `+dp` alone still reproduces document 55\n{'=' * 72}"
    )
    print(
        f"  bucket moves {len(moved)} against {DOC55_BUCKET_MOVES} (tolerance "
        f"±{DOC55_MOVE_TOLERANCE}); median |ΔDTW| on affected games {median_pp:.2f} pp "
        f"against {DOC55_MEDIAN_ABS_DELTA_DTW_PP:.2f} (tolerance "
        f"±{DOC55_DTW_TOLERANCE_PP:.1f}) -> {'PASS' if report['pass'] else 'FAIL'}"
    )
    if not report["pass"]:
        raise SystemExit(
            "the `+dp` arm no longer reproduces document 55. Round 7's additive "
            "argument to `68.simulate_all` changed a round-6 number; stop and report."
        )
    return report


def gate_g4d(v13: pl.DataFrame, rd: pl.DataFrame) -> dict:
    """G-4d — document 52 §5's G-3 materiality floor, on the `+rd`-only arm.

    The bar and the statistic are G-3's, unchanged and not re-tuned: the median
    \\|ΔDTW\\| the component moves on the games it touches, against the median of
    v1.3's own 89% half-width on those same games. A component that moves games
    by less than the interval already admits it does not have is not material,
    whatever its tail looks like.
    """
    moved, delta_dtw, half_width = bucket_moves(v13, rd)
    affected = rd.sort("game_id")["n_receiver_drop_events"].to_numpy() > 0
    move = np.abs(delta_dtw[affected]) * 100
    half = half_width[affected] * 100
    median_move, median_half = float(np.median(move)), float(np.median(half))
    report = {
        "statistic": (
            "median |ΔDTW| on affected games against the median of v1.3's 89% "
            "half-width, (high − low) / 2 per game (document 52 §5's G-3)"
        ),
        "affected_games": int(affected.sum()),
        "bucket_moves": len(moved),
        "median_abs_delta_dtw_pp": median_move,
        "median_half_width_pp": median_half,
        "difference_pp": median_move - median_half,
        "mean_abs_delta_dtw_pp": float(move.mean()),
        "mean_half_width_pp": float(half.mean()),
        "share_of_games_move_exceeds_own_half_width": float((move >= half).mean()),
        "pass": bool(median_move >= median_half),
    }
    print(f"\n{'=' * 72}\nGATE G-4d — the materiality floor on `+rd` alone\n{'=' * 72}")
    print(
        f"G-4d: median |ΔDTW| affected {median_move:.2f} pp vs median half-width "
        f"{median_half:.2f} pp -> {'PASS' if report['pass'] else 'FAIL'} by "
        f"{abs(report['difference_pp']):.2f} pp   [n = {report['affected_games']:,}]"
    )
    print(
        f"  for scale, the same two on the mean: {report['mean_abs_delta_dtw_pp']:.2f} pp "
        f"against {report['mean_half_width_pp']:.2f} pp; the move exceeds a game's own "
        f"half-width in {report['share_of_games_move_exceeds_own_half_width']:.1%} of them"
    )
    print(f"  and {len(moved)} games change verdict bucket under `+rd` alone")
    return report


def gate_g5(
    v13: pl.DataFrame,
    arms: dict[str, pl.DataFrame],
    ledgers: dict[str, pl.DataFrame],
) -> dict:
    """G-5 — the four-way comparison. Reported, never gated, never tuned.

    Document 56 §3's pre-committed expectation: the two directions are roughly
    independent per game, so `+dp+rd` moves more games than either alone and the
    share cancelling is near 50%. A strong positive correlation would mean both
    are picking up the same game-level thing (document 48's 8.5 pp game effect),
    and §3 says that is a finding to name.
    """
    print(f"\n{'=' * 72}\nG-5 — strict vs +dp vs +rd vs +dp+rd (reported, not gated)\n{'=' * 72}")
    per_arm = {}
    deltas = {}
    for label, arm in arms.items():
        moved, delta_dtw, half_width = bucket_moves(v13, arm)
        deltas[label] = delta_dtw
        events = arm.sort("game_id")
        touched = (
            events["n_dropped_pick_events"].to_numpy() + events["n_receiver_drop_events"].to_numpy()
        ) > 0
        move = np.abs(delta_dtw[touched]) * 100
        joined_widths = arm.sort("game_id")
        width_arm = (joined_widths["dtw_high"] - joined_widths["dtw_low"]).to_numpy()
        width_v13 = half_width * 2.0
        per_arm[label] = {
            "games": int(arm.height),
            "affected_games": int(touched.sum()),
            "bucket_moves": len(moved),
            "share_bucket_moved": len(moved) / arm.height,
            "median_abs_delta_dtw_pp_affected": float(np.median(move)),
            "eti89_abs_delta_dtw_pp_affected": [
                float(v) for v in np.percentile(move, [ETI_LOW, ETI_HIGH])
            ],
            "max_abs_delta_dtw_pp_affected": float(move.max()),
            "mean_interval_width_v13": float(width_v13[touched].mean()),
            "mean_interval_width_arm": float(width_arm[touched].mean()),
            "mean_widening": float((width_arm - width_v13)[touched].mean()),
            "moved_set": sorted(moved),
        }
        entry = per_arm[label]
        print(
            f"  {label:10s} affected {entry['affected_games']:5,d}  bucket moves "
            f"{entry['bucket_moves']:4d} ({entry['share_bucket_moved']:.2%})  median "
            f"|ΔDTW| {entry['median_abs_delta_dtw_pp_affected']:5.2f} pp  89% "
            f"[{entry['eti89_abs_delta_dtw_pp_affected'][0]:.2f}, "
            f"{entry['eti89_abs_delta_dtw_pp_affected'][1]:.2f}]  interval width "
            f"{entry['mean_interval_width_v13']:.4f} -> "
            f"{entry['mean_interval_width_arm']:.4f}"
        )

    # Element-wise, never by subtracting totals (document 33's lesson).
    dp_set = set(per_arm["+dp"]["moved_set"])
    rd_set = set(per_arm["+rd"]["moved_set"])
    both_set = set(per_arm["+dp+rd"]["moved_set"])
    overlap = {
        "moved_under_dp_only": len(dp_set - rd_set),
        "moved_under_rd_only": len(rd_set - dp_set),
        "moved_under_both_directions": len(dp_set & rd_set),
        "moved_under_the_combined_arm": len(both_set),
        "combined_moves_neither_alone_moved": len(both_set - dp_set - rd_set),
        "either_alone_moved_but_combined_did_not": len((dp_set | rd_set) - both_set),
    }
    print(
        f"\n  element-wise: {overlap['moved_under_dp_only']} games move under +dp only, "
        f"{overlap['moved_under_rd_only']} under +rd only, "
        f"{overlap['moved_under_both_directions']} under both; the combined arm moves "
        f"{overlap['moved_under_the_combined_arm']}, of which "
        f"{overlap['combined_moves_neither_alone_moved']} neither alone moved, and "
        f"{overlap['either_alone_moved_but_combined_did_not']} that one alone moved the "
        f"combined arm does not"
    )

    # The pre-committed statistic: do the two directions pull the same way?
    dp_delta, rd_delta = deltas["+dp"], deltas["+rd"]
    both = (np.abs(dp_delta) > 0) & (np.abs(rd_delta) > 0)
    same_way = (np.sign(dp_delta[both]) == np.sign(rd_delta[both])).mean()
    correlation = float(np.corrcoef(dp_delta[both], rd_delta[both])[0, 1])
    combined_vs_sum = float(
        np.abs(deltas["+dp+rd"][both] - (dp_delta[both] + rd_delta[both])).max()
    )
    directions = {
        "games_with_both_kinds_of_event": int(both.sum()),
        "share_moving_dtw_the_same_way": float(same_way),
        "share_moving_dtw_opposite_ways": float(1.0 - same_way),
        "pearson_r_of_the_two_dtw_shifts": correlation,
        "max_abs_combined_minus_sum_of_parts": combined_vs_sum,
        "pre_committed_expectation": (
            "roughly independent per game, so the cancelling share is near 50%; a "
            "strong positive correlation would mean both are reading document 48's "
            "game effect and is a finding to name"
        ),
    }
    print(
        f"\n  on the {directions['games_with_both_kinds_of_event']:,} games carrying both "
        f"kinds of event: the two directions move DTW the same way in "
        f"{same_way:.1%} and opposite ways in {1 - same_way:.1%}; "
        f"r = {correlation:+.3f} between the two shifts"
    )
    print(
        f"  and the combined arm is not the sum of the parts: max |Δ(+dp+rd) − "
        f"(Δ+dp + Δ+rd)| = {combined_vs_sum:.4f} DTW"
    )

    named = {}
    print("\n  the three named games, four ways:")
    for game_id in NAMED_GAMES:
        row_v13 = v13.filter(pl.col("game_id") == game_id)
        if not row_v13.height:
            named[game_id] = None
            print(f"    {game_id}: not in the simulated population")
            continue
        base = row_v13.to_dicts()[0]
        entry = {"actual_margin": base["actual_margin"], "strict": base}
        line = (
            f"    {game_id}  actual {base['actual_margin']:+3.0f}  DTW% strict "
            f"{base['dtw_home'] * 100:5.1f}"
        )
        for label, arm in arms.items():
            got = arm.filter(pl.col("game_id") == game_id)
            if not got.height:
                continue
            got = got.to_dicts()[0]
            entry[label] = got
            line += f"  {label} {got['dtw_home'] * 100:5.1f}"
        entry["bucket_strict"] = _audit.bucket(base["dtw_home"], base["actual_margin"])
        entry["bucket_combined"] = _audit.bucket(entry["+dp+rd"]["dtw_home"], base["actual_margin"])
        named[game_id] = entry
        print(line + f"   ({entry['bucket_strict']} -> {entry['bucket_combined']})")

    movers = largest_movers(v13, arms["+dp+rd"], ledgers["+dp+rd"])

    return {
        "reported_not_gated": True,
        "per_arm": per_arm,
        "move_set_overlap": overlap,
        "directions": directions,
        "named_games": named,
        "largest_combined_movers": movers,
    }


def largest_movers(v13: pl.DataFrame, arm: pl.DataFrame, ledger: pl.DataFrame) -> list[dict]:
    joined = (
        v13.select(
            "game_id",
            "actual_margin",
            pl.col("deserved_margin").alias("margin_v13"),
            pl.col("dtw_home").alias("dtw_v13"),
        )
        .join(
            arm.select(
                "game_id",
                pl.col("deserved_margin").alias("margin_arm"),
                pl.col("dtw_home").alias("dtw_arm"),
                "n_dropped_pick_events",
                "n_receiver_drop_events",
            ),
            on="game_id",
        )
        .with_columns((pl.col("margin_arm") - pl.col("margin_v13")).abs().alias("abs_shift"))
        .sort("abs_shift", descending=True)
        .head(5)
    )
    print("\n  the five largest `+dp+rd` movers:")
    movers = []
    for row in joined.iter_rows(named=True):
        print(
            f"    {row['game_id']}  actual {row['actual_margin']:+.0f}  deserved "
            f"{row['margin_v13']:+.2f} -> {row['margin_arm']:+.2f}  DTW% "
            f"{row['dtw_v13'] * 100:.1f} -> {row['dtw_arm'] * 100:.1f}  "
            f"({row['n_dropped_pick_events']} picks, "
            f"{row['n_receiver_drop_events']} catchable balls)"
        )
        rows = (
            ledger.filter(
                (pl.col("game_id") == row["game_id"])
                & (pl.col("component").is_in(["dropped_pick", "receiver_drop"]))
            )
            .sort(["component", "play_id"])
            .to_dicts()
        )
        movers.append({**row, "variant_rows": rows[:40]})
    return movers


def main() -> None:
    ctx = _audit.load_context()
    slope = ctx.slope

    # ---- V-1, first and unconditional -------------------------------------
    v13_table, _v13_ledger, v1 = _audit.v13_pass(ctx)

    # ---- the two models and the read-side round trip ------------------------
    dropped_pick_model = DroppedPickModel.from_posterior(
        paths.RESEARCH_OUTPUT_DIR / _audit.TRACE_NAME,
        paths.RESEARCH_OUTPUT_DIR / _audit.SUMMARY_NAME,
    )
    receiver_model = ReceiverDropModel.from_posterior(
        paths.RESEARCH_OUTPUT_DIR / TRACE_NAME, paths.RESEARCH_OUTPUT_DIR / SUMMARY_NAME
    )
    summary = json.loads((paths.RESEARCH_OUTPUT_DIR / SUMMARY_NAME).read_text())
    round_trip = read_side_round_trip(receiver_model)

    v6 = summary["gate_v6_sampler"]
    v8 = summary["gate_v8_posterior_spread"]
    print(f"\n{'=' * 72}\nV-6 and V-8, from `research/72`'s fit\n{'=' * 72}")
    print(
        f"  V-6: divergences {v6['divergences']}, max r_hat {v6['max_r_hat']:.4f} "
        f"({v6['max_r_hat_parameter']}), min ess_bulk {v6['min_ess_bulk']:.0f}, "
        f"min ess_tail {v6['min_ess_tail']:.0f} over {v6['parameters_checked']} "
        f"parameters -> {'PASS' if v6['pass'] else 'FAIL'}"
    )
    print(
        f"  V-8: catch probability on a median target inside "
        f"[{v8['bound'][0]:.2f}, {v8['bound'][1]:.2f}] on every one of the ten lines, "
        f"under both readings -> {'PASS' if v8['pass'] else 'FAIL'}"
    )

    # ---- the three variant arms ---------------------------------------------
    dp_table, dp_ledger = _audit.variant_pass(ctx, dropped_pick_model, label="+dp")
    rd_table, rd_ledger = _audit.variant_pass(
        ctx, None, receiver_drop_model=receiver_model, label="+rd"
    )
    both_table, both_ledger = _audit.variant_pass(
        ctx, dropped_pick_model, receiver_drop_model=receiver_model, label="+dp+rd"
    )
    charted = v13_table.filter(pl.col("game_id").is_in(dp_table["game_id"].to_list())).sort(
        "game_id"
    )

    # ---- handoff constraint 1 ------------------------------------------------
    reproduction = doc55_reproduction(charted, dp_table)

    # ---- V-2 on every arm ----------------------------------------------------
    identities = {
        label: _audit.round_trip_identity(table, slope)
        for label, table in (("+dp", dp_table), ("+rd", rd_table), ("+dp+rd", both_table))
    }
    print(f"\n{'=' * 72}\nV-2 — the ledger sums, on every arm\n{'=' * 72}")
    for label, value in identities.items():
        print(f"  {label:8s} max |deserved − (actual − luck × slope)| = {value:.2e}")
    if max(identities.values()) > 1e-9:
        raise SystemExit("a variant ledger does not sum. Stop and report.")

    # ---- coverage ------------------------------------------------------------
    events = rd_table["n_receiver_drop_events"].to_numpy()
    affected = events > 0
    coverage = {
        "games": int(rd_table.height),
        "games_with_an_event": int(affected.sum()),
        "share_with_an_event": float(affected.mean()),
        "events_total": int(events.sum()),
        "events_per_game_median_affected": float(np.median(events[affected])),
        "events_per_game_max": int(events.max()),
        "games_labelled_rd": int((rd_table["variant"] == "v1.3+rd").sum()),
        "games_labelled_dp_rd": int((both_table["variant"] == "v1.3+dp+rd").sum()),
    }
    print(f"\n{'=' * 72}\nCOVERAGE — where the receiver variant has anything to say\n{'=' * 72}")
    print(
        f"  {coverage['games_with_an_event']:,} of {coverage['games']:,} games carry at "
        f"least one receiver-drop row ({coverage['share_with_an_event']:.1%}); "
        f"{coverage['events_total']:,} events, per affected game median "
        f"{coverage['events_per_game_median_affected']:.0f}, max "
        f"{coverage['events_per_game_max']}"
    )

    # ---- the gates -----------------------------------------------------------
    g4d = gate_g4d(charted, rd_table)
    g5 = gate_g5(
        charted,
        {"+dp": dp_table, "+rd": rd_table, "+dp+rd": both_table},
        {"+dp": dp_ledger, "+rd": rd_ledger, "+dp+rd": both_ledger},
    )

    g4b = {
        "gates": {
            "V-1": v1["pass"],
            "V-2": max(identities.values()) <= 1e-9,
            "V-3..V-5, V-7": "pinned in tests/test_receiver_drops.py",
            "V-6": v6["pass"],
            "V-8": v8["pass"],
            "read_side_round_trip": round_trip["pass"],
        },
        "pass": bool(
            v1["pass"]
            and max(identities.values()) <= 1e-9
            and v6["pass"]
            and v8["pass"]
            and round_trip["pass"]
        ),
    }
    print(f"\n{'=' * 72}\nGATE G-4b — V-1..V-8 together\n{'=' * 72}")
    print(f"G-4b: {'PASS' if g4b['pass'] else 'FAIL'}  {g4b['gates']}")

    results = {
        "document": "56 — the receiver-drop mirror (A-3 gate G-4)",
        "part": "D — gates G-4b, G-4d and the G-5 audit",
        "settings": {
            "random_seed": _audit.RANDOM_SEED,
            "posterior_draws": _audit.POSTERIOR_DRAWS,
            "coin_draws": _audit.COIN_DRAWS,
            "points_per_epa": slope,
            "seasons": list(FTN_SEASONS),
        },
        "charged_grain": summary["charged_grain"],
        "gate_v1_default_off": v1,
        "gate_v2_round_trip_max_residual": identities,
        "gate_v6_sampler": v6,
        "gate_v8_posterior_spread": v8,
        "read_side_round_trip": round_trip,
        "handoff_constraint_1_doc55_reproduction": reproduction,
        "coverage": coverage,
        "gate_g4b": g4b,
        "gate_g4d_materiality": g4d,
        "gate_g5_four_way": g5,
    }
    out = paths.RESEARCH_OUTPUT_DIR / OUTPUT_NAME
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {out}")

    # Handoff constraint 1: the V-1 line is printed again at the end of the run.
    print(f"\n{'=' * 72}\nV-1, RE-PRINTED AT THE END OF THE RUN\n{'=' * 72}")
    print(
        f"V-1 replay: {v1['games_matched']:,} games, max |Δ deserved margin| "
        f"{v1['max_abs_gaps']['deserved_margin']:.2e}  -> "
        f"{'PASS' if v1['pass'] else 'FAIL'}"
    )
    if not v1["pass"]:
        raise SystemExit("v1.3 moved. Stop and report.")
    print("Next: research/74_receiver_drop_weekout.py for G-4c.")


if __name__ == "__main__":
    main()
