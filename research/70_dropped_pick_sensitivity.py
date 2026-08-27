"""Round 5, Part C — gates G-2 (pricing sensitivity) and G-3 (materiality).

Both read the **in-sample** fit round 4 wrote, so neither depends on document 52
§5's week-out folds. That independence is why this script runs even when G-1 is
blocked: two of the three gate lines are answerable without it.

**G-2 — pricing sensitivity.** Re-run round 4's audit with every throw priced at
document 47 §3's *pooled* swing (−3.55 EPA) instead of its own bin, and count the
games that change verdict bucket. Document 52 §5's statistic is that count as a
share of the **137** that change under the binned swing, and the bar is
**>= 0.50** — the claim being tested is that the variant's effect is the coin, not
the goal-line cell. On a fail the swing table is re-derived with a higher cell
floor before any row ships, as its own pre-registered step.

**G-3 — materiality (amendment A-3 clause 4, document 05 §7's floor).** On the
1,033 affected games, the median `|ΔDTW|` the variant moves against the median of
v1.3's own 89% half-width on those same games. The bar is **median move >=
median half-width**: a component that moves the verdict by less than the
incumbent's own uncertainty is not saying anything the incumbent did not already
admit. Document 52 §5 pre-commits that this may fail and that the floor is **not
re-tuned** for this component; the 137 bucket-move games are reported beside it
and are explicitly *not* the gate.

Document 33's element-wise lesson is honoured where the pre-registration leaves
room for it: G-2's ratio is a ratio of counts because that is what document 52
§5 says, so the *overlap* between the two move sets — which games move under
both pricings, and which are new — is computed and printed beside it rather than
inferred from the ratio.

    uv run python research/70_dropped_pick_sensitivity.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_audit = import_module("68_dropped_pick_variant_audit")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.dropped_picks import DroppedPickModel, SwingTable  # noqa: E402

# Document 52 §5 and §7, verbatim. No re-tolerancing: a fail is reported as a
# fail, and G-3 in particular is not to be rescued by changing the floor, the
# game set or the statistic.
FLAT_SWING = -3.55  # document 47 §3's pooled fallback
G2_BINNED_BUCKET_MOVES = 137  # round 4's denominator, document 50 §4
G2_MIN_SHARE = 0.50
FLAT_SWING_TOLERANCE = 0.005  # the table's own pooled value must be this swing

# Handoff §2's guards on the in-sample arm.
ROUND4_AFFECTED = 1033
ROUND4_MEDIAN_ABS_DELTA_DTW_PP = 1.62
GUARD_TOLERANCE_PP = 0.01

OUTPUT_NAME = "70_dropped_pick_sensitivity.json"


def flat_swing_table(table: SwingTable) -> SwingTable:
    """Every cell at the pooled swing — document 52 §5's G-2 arm.

    The counts are carried over unchanged and relabelled, so the artifact still
    shows how many throws each cell held: the point of the arm is that the *price*
    is flat, not that the bins vanished.
    """
    if abs(table.pooled - FLAT_SWING) > FLAT_SWING_TOLERANCE:
        raise SystemExit(
            f"the fitted table's pooled swing is {table.pooled:+.2f} EPA, not "
            f"document 47 §3's {FLAT_SWING:+.2f} — stop and ask."
        )
    counts = {
        key: {**entry, "source": "flat (G-2)", "binned_swing": table.cells[key]}
        for key, entry in table.counts.items()
    }
    return SwingTable(
        cells={key: table.pooled for key in table.cells},
        counts=counts,
        pooled=table.pooled,
    )


def bucket_movers(v13: pl.DataFrame, variant: pl.DataFrame) -> tuple[set[str], dict]:
    """The set of games whose verdict bucket moves, and the counts beside it."""
    joined = v13.select(
        "game_id",
        "actual_margin",
        pl.col("dtw_home").alias("dtw_v13"),
    ).join(
        variant.select("game_id", pl.col("dtw_home").alias("dtw_var")),
        on="game_id",
    )
    moved = set()
    for row in joined.iter_rows(named=True):
        before = _audit.bucket(row["dtw_v13"], row["actual_margin"])
        after = _audit.bucket(row["dtw_var"], row["actual_margin"])
        if before != after:
            moved.add(row["game_id"])
    return moved, {"games": int(joined.height), "n_moved": len(moved)}


def gate_g2(binned_moved: set[str], flat_moved: set[str]) -> dict:
    """Document 52 §5's G-2, verbatim, with the overlap reported beside it."""
    share = len(flat_moved) / G2_BINNED_BUCKET_MOVES
    passes = share >= G2_MIN_SHARE

    # Document 33's lesson: two same-sized sets are not the same set. The ratio
    # is the pre-registered statistic; this is what it does not say.
    both = binned_moved & flat_moved
    report = {
        "statistic": (
            "games changing verdict bucket under the flat swing, as a share of "
            "the 137 that change under the binned swing (document 52 §5)"
        ),
        "flat_swing_epa": FLAT_SWING,
        "n_moved_flat": len(flat_moved),
        "n_moved_binned_this_run": len(binned_moved),
        "denominator": G2_BINNED_BUCKET_MOVES,
        "share": share,
        "bar": G2_MIN_SHARE,
        "pass": bool(passes),
        "reported_not_gated": {
            "n_in_both_move_sets": len(both),
            "n_binned_only": len(binned_moved - flat_moved),
            "n_flat_only": len(flat_moved - binned_moved),
            "element_wise_agreement": float(
                len(both) / len(binned_moved | flat_moved) if (binned_moved | flat_moved) else 1.0
            ),
        },
    }

    print(f"\n{'=' * 72}\nGATE G-2 — the coin, or the goal-line cell?\n{'=' * 72}")
    print(
        f"G-2: flat-swing bucket moves {len(flat_moved)} of "
        f"{G2_BINNED_BUCKET_MOVES} binned ({share:.2f}) -> "
        f"{'PASS' if passes else 'FAIL'}   [bar >= {G2_MIN_SHARE:.2f}]"
    )
    overlap = report["reported_not_gated"]
    print(
        f"  reported, not gated — the two move sets are not the same set: "
        f"{overlap['n_in_both_move_sets']} games move under both pricings, "
        f"{overlap['n_binned_only']} under the binned only, "
        f"{overlap['n_flat_only']} under the flat only"
    )
    if not passes:
        print(
            "  G-2 FAILS. Document 52 §5's consequence: the swing table is "
            "re-derived with a higher cell floor before any row ships, as its "
            "own pre-registered step."
        )
    return report


def gate_g3(v13: pl.DataFrame, variant: pl.DataFrame, binned_moved: set[str]) -> dict:
    """Document 52 §5's G-3 — the materiality floor, on the affected games."""
    joined = v13.select(
        "game_id",
        pl.col("dtw_home").alias("dtw_v13"),
        pl.col("dtw_low").alias("low_v13"),
        pl.col("dtw_high").alias("high_v13"),
    ).join(
        variant.select(
            "game_id",
            pl.col("dtw_home").alias("dtw_var"),
            "n_dropped_pick_events",
        ),
        on="game_id",
    )

    def measure(frame: pl.DataFrame, label: str) -> dict:
        move = (frame["dtw_var"] - frame["dtw_v13"]).abs().to_numpy() * 100
        half = (frame["high_v13"] - frame["low_v13"]).to_numpy() / 2.0 * 100
        median_move = float(np.median(move))
        median_half = float(np.median(half))
        return {
            "population": label,
            "n": int(frame.height),
            "median_abs_delta_dtw_pp": median_move,
            "median_half_width_pp": median_half,
            "difference_pp": median_move - median_half,
            "mean_abs_delta_dtw_pp": float(move.mean()),
            "mean_half_width_pp": float(half.mean()),
            "share_of_games_move_exceeds_own_half_width": float((move >= half).mean()),
            "pass": bool(median_move >= median_half),
        }

    affected = joined.filter(pl.col("n_dropped_pick_events") > 0)
    gated = measure(affected, "the 1,033 affected games")
    movers = measure(joined.filter(pl.col("game_id").is_in(list(binned_moved))), "the 137 movers")

    print(f"\n{'=' * 72}\nGATE G-3 — the materiality floor (document 05 §7)\n{'=' * 72}")
    print(
        f"G-3: median |dDTW| affected {gated['median_abs_delta_dtw_pp']:.2f} pp vs "
        f"median half-width {gated['median_half_width_pp']:.2f} pp -> "
        f"{'PASS' if gated['pass'] else 'FAIL'} by "
        f"{abs(gated['difference_pp']):.2f} pp   [n = {gated['n']:,}]"
    )
    print(
        f"  for scale, the same two on the mean rather than the median: "
        f"{gated['mean_abs_delta_dtw_pp']:.2f} pp against "
        f"{gated['mean_half_width_pp']:.2f} pp; the move exceeds a game's own "
        f"half-width in {gated['share_of_games_move_exceeds_own_half_width']:.1%} "
        f"of affected games"
    )
    print(
        f"  REPORTED ONLY, not the gate — on the {gated_movers_n(movers)} bucket-move "
        f"games: median |dDTW| {movers['median_abs_delta_dtw_pp']:.2f} pp vs median "
        f"half-width {movers['median_half_width_pp']:.2f} pp"
    )
    print(
        '  document 52 §5, pre-committed: "If it fails, A-3 is not enacted for '
        "dropped picks on\n  median grounds, the variant stays a labelled variant, "
        "and the tail (12% bucket moves) is\n  reported beside the failure. The "
        'floor is not re-tuned for this component."'
    )
    return {
        "statistic": (
            "median |ΔDTW| on affected games against the median of v1.3's 89% "
            "half-width, (high − low) / 2 per game (document 52 §5)"
        ),
        "gated": gated,
        "reported_only_bucket_movers": movers,
        "pass": gated["pass"],
    }


def gated_movers_n(movers: dict) -> int:
    return int(movers["n"])


def guards(movement: dict) -> dict:
    """Handoff §2's in-sample numbers, before any gate is read off this arm."""
    checks = {
        "affected_games": (movement["affected_games"]["n"], ROUND4_AFFECTED, 0),
        "median_abs_delta_dtw_pp": (
            movement["affected_games"]["median_abs_delta_dtw_pp"],
            ROUND4_MEDIAN_ABS_DELTA_DTW_PP,
            GUARD_TOLERANCE_PP,
        ),
    }
    print(f"\n{'=' * 72}\nROUND 4's GUARDS — the binned in-sample arm, reproduced\n{'=' * 72}")
    report, failures = {}, []
    for name, (got, want, tolerance) in checks.items():
        ok = abs(float(got) - float(want)) <= tolerance
        report[name] = {"round4": want, "reproduced": got, "ok": bool(ok)}
        print(f"  {name:26s} round 4 {want:>8}   now {got:>8}   {'ok' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(name)
    report["pass"] = not failures
    if failures:
        raise SystemExit(f"handoff §2's guards do not reproduce: {failures} — stop and ask.")
    return report


# --------------------------------------------------------------------------


def main() -> None:
    print("=== Round 5 Part C — gates G-2 and G-3 (document 52 §5) ===")
    ctx = _audit.load_context()

    v13_table, _, v1 = _audit.v13_pass(ctx)

    model = DroppedPickModel.from_posterior(
        paths.RESEARCH_OUTPUT_DIR / _audit.TRACE_NAME,
        paths.RESEARCH_OUTPUT_DIR / _audit.SUMMARY_NAME,
    )
    flat_model = dataclasses.replace(model, swing_table=flat_swing_table(model.swing_table))

    print(f"\n{'=' * 72}\nTHE TWO PRICINGS\n{'=' * 72}")
    print(f"  {'cell':16s} {'binned':>8s} {'flat':>8s}")
    for key, value in model.swing_table.cells.items():
        print(f"  {key:16s} {value:+8.2f} {flat_model.swing_table.cells[key]:+8.2f}")

    binned_table, binned_ledger = _audit.variant_pass(ctx, model, label="binned variant")
    flat_table, _ = _audit.variant_pass(ctx, flat_model, label="flat-swing variant")
    charted = v13_table.filter(pl.col("game_id").is_in(binned_table["game_id"].to_list()))

    binned_moves = _audit.flips(charted, binned_table)
    binned_movement = _audit.movement(charted, binned_table)
    guard_report = guards(binned_movement)

    binned_moved, binned_counts = bucket_movers(charted, binned_table)
    flat_moved, flat_counts = bucket_movers(charted, flat_table)
    if binned_counts["n_moved"] != binned_moves["n_bucket_moved"]:
        raise SystemExit("the two bucket-move counts disagree — stop and ask.")

    g2 = gate_g2(binned_moved, flat_moved)
    g3 = gate_g3(charted, binned_table, binned_moved)

    identity = _audit.round_trip_identity(flat_table, ctx.slope)
    print(
        f"\n  V-2 on every flat-swing game: max |deserved − (actual − luck × slope)| = {identity:.2e}"
    )
    if identity > 1e-9:
        raise SystemExit("the flat-swing variant ledger does not sum. Stop and report.")

    results = {
        "gates": "document 52 §5 G-2 and G-3",
        "settings": {
            "random_seed": _audit.RANDOM_SEED,
            "posterior_draws": _audit.POSTERIOR_DRAWS,
            "coin_draws": _audit.COIN_DRAWS,
            "points_per_epa": ctx.slope,
            "flat_swing_epa": FLAT_SWING,
        },
        "gate_v1_default_off": v1,
        "round4_guards": guard_report,
        "swing_tables": {
            "binned": model.swing_table.to_dict(),
            "flat": flat_model.swing_table.to_dict(),
        },
        "binned_arm": {"flips": binned_moves, "movement": binned_movement},
        "flat_arm": flat_counts,
        "gate_g2": g2,
        "gate_g3": g3,
        "gate_v2_round_trip_max_residual_flat_arm": identity,
        "bucket_move_game_ids": {
            "binned": sorted(binned_moved),
            "flat": sorted(flat_moved),
        },
    }
    out = paths.RESEARCH_OUTPUT_DIR / OUTPUT_NAME
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {out}")
    del binned_ledger

    # Handoff constraint 1: the V-1 line closes every audit run.
    print(f"\n{'=' * 72}\nV-1, RE-PRINTED AT THE END OF THE RUN (handoff constraint 1)\n{'=' * 72}")
    print(
        f"V-1 replay: {v1['games_matched']:,} games, max |Δ deserved margin| "
        f"{v1['max_abs_gaps']['deserved_margin']:.2e}  -> "
        f"{'PASS' if v1['pass'] else 'FAIL'}"
    )
    if not v1["pass"]:
        raise SystemExit("v1.3 moved. Stop and report.")
    print(f"\nG-2: {'PASS' if g2['pass'] else 'FAIL'}   G-3: {'PASS' if g3['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
