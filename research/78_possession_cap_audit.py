"""Round 9, part C — the possession cap audited, and the Full summary regenerated.

Document 61 §6's gates, in the order that makes them readable.

    P-1  Strict untouched. V-1 against `dtw_games_v13.parquet`, 0.00e+00 over
         2,761 games, on the **wider** frame round 9 loads — `fixed_drive` and
         `qtr` are proven inert rather than assumed to be.
    P-2  The no-cap identity. The uncapped Full arm on the wide frame against
         the same arm on the narrow one: 0.00e+00 on every game. Then that arm
         against document 59 §4's published audit, which is the number the cap
         is measured *from*.
    P-3  The round trip, with cap rows in the ledger: ≤ 1e-9 on every game.
    P-7  The audit. Strict, Full uncapped and Full capped, three ways, beside
         document 59's 200 bucket moves / 3.85 pp / 0.0516 mean interval width.

P-4, P-5 and P-6 are pinned in `tests/test_possession_cap.py`, where a synthetic
possession can be built to order; they are properties of the clip, not results a
population happens to produce.

The capped arm is then written to `full_summary.parquet`, replacing the uncapped
one figure round 6 wrote. `research/76`'s reproduction line is printed for both
arms — the uncapped one is gated, the capped one is reported, because a change
there is what the cap *is*.

    uv run python research/78_possession_cap_audit.py

Writes `research/outputs/full_summary.parquet` and
`research/outputs/78_possession_cap_audit.json`. Neither is committed —
`research/outputs/` is gitignored, this script is the artifact, and document 62
is the record.
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
_receiver_audit = import_module("73_receiver_drop_variant_audit")
_full = import_module("76_full_edition_summary")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.ingest import FTN_SEASONS  # noqa: E402
from nfl_simulator.render import FULL_ARTIFACT  # noqa: E402
from nfl_simulator.simulator import POSSESSION_CAP_COMPONENT  # noqa: E402

OUTPUT_NAME = "78_possession_cap_audit.json"

# The columns round 9 adds to `44_read_side_fix.SIM_COLUMNS`. P-2 drops them
# again to prove the Full arm does not read them either.
ROUND_9_COLUMNS = ("fixed_drive", "qtr")

# Document 59 §4 / document 60, the three numbers the capped arm is set beside.
DOC59_BUCKET_MOVES = 200
DOC59_MEDIAN_ABS_DELTA_DTW_PP = 3.85
DOC59_MEAN_INTERVAL_WIDTH = 0.0516

NAMED_GAMES = ("2025_17_DET_MIN", "2022_13_WAS_NYG", "2024_19_LAC_HOU")

ROUND_TRIP_TOLERANCE = 1e-9


def gate_p2_columns_are_inert(ctx) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """P-2, first half: the uncapped Full arm does not read round 9's columns.

    V-1 says so for Strict over ten seasons. This says so for the arm the cap is
    actually applied to, which is the one whose numbers document 62 will move.
    """
    wide, wide_ledger = _full.full_pass(ctx, label="full (uncapped, wide frame)")
    narrow_ctx = dataclasses.replace(ctx, pbp=ctx.pbp.drop(ROUND_9_COLUMNS))
    narrow, _ = _full.full_pass(narrow_ctx, label="full (uncapped, narrow frame)")

    joined = wide.select(
        "game_id",
        pl.col("deserved_margin").alias("wide_margin"),
        pl.col("dtw_home").alias("wide_dtw"),
        pl.col("dtw_low").alias("wide_low"),
        pl.col("dtw_high").alias("wide_high"),
    ).join(narrow, on="game_id", how="inner")
    gaps = {
        "deserved_margin": float((joined["deserved_margin"] - joined["wide_margin"]).abs().max()),
        "dtw_home": float((joined["dtw_home"] - joined["wide_dtw"]).abs().max()),
        "dtw_low": float((joined["dtw_low"] - joined["wide_low"]).abs().max()),
        "dtw_high": float((joined["dtw_high"] - joined["wide_high"]).abs().max()),
    }
    report = {
        "statistic": (
            "the uncapped Full arm with `fixed_drive` and `qtr` loaded against the "
            "same arm without them"
        ),
        "games_matched": int(joined.height),
        "max_abs_gaps": gaps,
        "tolerance": 0.0,
        "pass": bool(joined.height == wide.height and max(gaps.values()) <= 0.0),
    }
    print(f"\n{'=' * 76}\nP-2 (a) — round 9's columns are inert on the Full arm\n{'=' * 76}")
    print(
        f"  {joined.height:,} games, max |Δ deserved margin| {gaps['deserved_margin']:.2e}, "
        f"|Δ DTW%| {gaps['dtw_home']:.2e}, |Δ interval| "
        f"{max(gaps['dtw_low'], gaps['dtw_high']):.2e}  -> "
        f"{'PASS' if report['pass'] else 'FAIL'}"
    )
    if not report["pass"]:
        raise SystemExit(
            "widening the frame moved the Full arm. Round 9 added a column something "
            "prices on; stop and report."
        )
    return wide, wide_ledger, report


def gate_p3_round_trip(capped: pl.DataFrame, slope: float) -> dict:
    """P-3 — the ledger, cap rows included, still sums to the margin it moved."""
    residual = _audit.round_trip_identity(capped, slope)
    report = {
        "statistic": "max |deserved − (actual − total_luck_epa × slope)| with cap rows in",
        "max_abs_residual": residual,
        "tolerance": ROUND_TRIP_TOLERANCE,
        "pass": bool(residual <= ROUND_TRIP_TOLERANCE),
    }
    print(f"\n{'=' * 76}\nP-3 — the round trip, with the cap rows inside it\n{'=' * 76}")
    print(
        f"  max residual {residual:.2e} against {ROUND_TRIP_TOLERANCE:.0e}  -> "
        f"{'PASS' if report['pass'] else 'FAIL'}"
    )
    if not report["pass"]:
        raise SystemExit("the capped ledger does not sum to the margin it moved. Stop and report.")
    return report


def arm_numbers(strict: pl.DataFrame, arm: pl.DataFrame) -> dict:
    """The three statistics document 59 published, on one arm."""
    moved, delta_dtw, half_width = _receiver_audit.bucket_moves(strict, arm)
    ordered = arm.sort("game_id")
    touched = (
        ordered["n_dropped_pick_events"].to_numpy() + ordered["n_receiver_drop_events"].to_numpy()
    ) > 0
    move = np.abs(delta_dtw[touched]) * 100
    width_arm = (ordered["dtw_high"] - ordered["dtw_low"]).to_numpy()
    return {
        "games": int(arm.height),
        "affected_games": int(touched.sum()),
        "bucket_moves": len(moved),
        "moved_set": sorted(moved),
        "median_abs_delta_dtw_pp_affected": float(np.median(move)),
        "mean_abs_delta_dtw_pp_affected": float(move.mean()),
        "max_abs_delta_dtw_pp_affected": float(move.max()),
        "mean_interval_width": float(width_arm[touched].mean()),
        "mean_interval_width_strict": float(half_width[touched].mean() * 2.0),
    }


def gate_p7(strict: pl.DataFrame, uncapped: pl.DataFrame, capped: pl.DataFrame) -> dict:
    """P-7 — the audit, reported and never tuned.

    Three arms against Strict, and the pair that matters set side by side: the
    Full edition as document 59 published it, and the Full edition with document
    61's possession cap on. The move sets are compared **element-wise** — document
    33's lesson, learned here by reporting a 24-game net as a 56-game
    disagreement — so "how many moves the cap takes away" is a count of games,
    never a difference of totals.
    """
    arms = {
        "full (uncapped)": arm_numbers(strict, uncapped),
        "full (capped)": arm_numbers(strict, capped),
    }
    print(f"\n{'=' * 76}\nP-7 — the Full edition, with the cap and without\n{'=' * 76}")
    print(
        f"  {'arm':18s} {'moves':>6}  {'median |ΔDTW|':>14}  {'mean |ΔDTW|':>12}  "
        f"{'mean 89% width':>15}"
    )
    for label, entry in arms.items():
        print(
            f"  {label:18s} {entry['bucket_moves']:6d}  "
            f"{entry['median_abs_delta_dtw_pp_affected']:11.2f} pp  "
            f"{entry['mean_abs_delta_dtw_pp_affected']:9.2f} pp  "
            f"{entry['mean_interval_width']:15.4f}"
        )
    print(
        f"  {'document 59':18s} {DOC59_BUCKET_MOVES:6d}  "
        f"{DOC59_MEDIAN_ABS_DELTA_DTW_PP:11.2f} pp  {'—':>12}  "
        f"{DOC59_MEAN_INTERVAL_WIDTH:15.4f}"
    )
    print(
        f"  {'strict':18s} {'—':>6}  {'—':>14}  {'—':>12}  "
        f"{arms['full (capped)']['mean_interval_width_strict']:15.4f}"
    )

    before = set(arms["full (uncapped)"]["moved_set"])
    after = set(arms["full (capped)"]["moved_set"])
    overlap = {
        "moved_before_and_after": len(before & after),
        "moves_the_cap_takes_away": len(before - after),
        "moves_the_cap_creates": len(after - before),
        "share_of_uncapped_moves_the_cap_takes_away": len(before - after) / max(len(before), 1),
    }
    print(
        f"\n  element-wise: of the {len(before)} games the uncapped Full edition moves, "
        f"{overlap['moved_before_and_after']} still move with the cap on, "
        f"{overlap['moves_the_cap_takes_away']} no longer do "
        f"({overlap['share_of_uncapped_moves_the_cap_takes_away']:.1%}), and "
        f"{overlap['moves_the_cap_creates']} move that did not before"
    )
    narrowing = (
        arms["full (uncapped)"]["mean_interval_width"]
        - arms["full (capped)"]["mean_interval_width"]
    )
    print(
        f"  the cap narrows the mean 89% interval on affected games by {narrowing:.4f} "
        f"({narrowing / arms['full (uncapped)']['mean_interval_width']:.1%}) — document 61 §0's "
        f"second defect, the width no possession could have produced"
    )
    return {
        "reported_not_gated": True,
        "arms": arms,
        "document_59": {
            "bucket_moves": DOC59_BUCKET_MOVES,
            "median_abs_delta_dtw_pp": DOC59_MEDIAN_ABS_DELTA_DTW_PP,
            "mean_interval_width": DOC59_MEAN_INTERVAL_WIDTH,
        },
        "move_set_overlap": overlap,
        "mean_interval_narrowing": narrowing,
    }


def named_games(
    strict: pl.DataFrame, uncapped: pl.DataFrame, capped: pl.DataFrame, ledger: pl.DataFrame
) -> dict:
    """Document 61 §4's three games, Full before and after the cap."""
    print(f"\n{'=' * 76}\nTHE THREE NAMED GAMES, FULL BEFORE AND AFTER\n{'=' * 76}")
    out = {}
    for game_id in NAMED_GAMES:
        base = strict.filter(pl.col("game_id") == game_id).to_dicts()
        if not base:
            print(f"  {game_id}: not in the Strict summary")
            out[game_id] = None
            continue
        base = base[0]
        before = uncapped.filter(pl.col("game_id") == game_id).to_dicts()[0]
        after = capped.filter(pl.col("game_id") == game_id).to_dicts()[0]
        rows = (
            ledger.filter(
                (pl.col("game_id") == game_id) & (pl.col("component") == POSSESSION_CAP_COMPONENT)
            )
            .sort("play_id")
            .to_dicts()
        )
        entry = {
            "actual_margin": base["actual_margin"],
            "strict": {"dtw_home": base["dtw_home"], "deserved_margin": base["deserved_margin"]},
            "full_uncapped": {
                "dtw_home": before["dtw_home"],
                "deserved_margin": before["deserved_margin"],
                "interval": [before["dtw_low"], before["dtw_high"]],
            },
            "full_capped": {
                "dtw_home": after["dtw_home"],
                "deserved_margin": after["deserved_margin"],
                "interval": [after["dtw_low"], after["dtw_high"]],
                "n_cap_rows": after["n_possession_cap_rows"],
                "cap_epa": after["possession_cap_epa"],
            },
            "bucket_uncapped": _audit.bucket(before["dtw_home"], base["actual_margin"]),
            "bucket_capped": _audit.bucket(after["dtw_home"], base["actual_margin"]),
            "cap_rows": rows,
        }
        out[game_id] = entry
        print(
            f"\n  {game_id}  actual {base['actual_margin']:+3.0f}   Strict DTW% "
            f"{base['dtw_home'] * 100:5.1f}"
        )
        print(
            f"    Full uncapped  DTW% {before['dtw_home'] * 100:5.1f}  margin "
            f"{before['deserved_margin']:+6.2f}  89% width "
            f"{before['dtw_high'] - before['dtw_low']:.4f}  ({entry['bucket_uncapped']})"
        )
        print(
            f"    Full capped    DTW% {after['dtw_home'] * 100:5.1f}  margin "
            f"{after['deserved_margin']:+6.2f}  89% width "
            f"{after['dtw_high'] - after['dtw_low']:.4f}  ({entry['bucket_capped']})"
        )
        print(
            f"    {after['n_possession_cap_rows']} cap rows worth "
            f"{after['possession_cap_epa']:+.2f} EPA"
        )
        for row in sorted(rows, key=lambda item: -abs(item["luck_epa"]))[:4]:
            print(
                f"      {row['event_class']:<16} ({row['charged_team']})  "
                f"{row['luck_epa']:+.2f} EPA"
            )
    return out


def largest_reductions(uncapped: pl.DataFrame, capped: pl.DataFrame, ledger: pl.DataFrame) -> list:
    """The ten games the cap moves furthest, as the cap rows that moved them."""
    joined = (
        uncapped.select(
            "game_id",
            "actual_margin",
            pl.col("deserved_margin").alias("margin_uncapped"),
            pl.col("dtw_home").alias("dtw_uncapped"),
        )
        .join(
            capped.select(
                "game_id",
                pl.col("deserved_margin").alias("margin_capped"),
                pl.col("dtw_home").alias("dtw_capped"),
                "n_possession_cap_rows",
                "possession_cap_epa",
            ),
            on="game_id",
        )
        .with_columns((pl.col("margin_capped") - pl.col("margin_uncapped")).alias("shift"))
        .with_columns(pl.col("shift").abs().alias("abs_shift"))
        .sort("abs_shift", descending=True)
        .head(10)
    )
    print(f"\n{'=' * 76}\nTHE TEN LARGEST REDUCTIONS\n{'=' * 76}")
    out = []
    for row in joined.iter_rows(named=True):
        print(
            f"  {row['game_id']}  actual {row['actual_margin']:+3.0f}  deserved "
            f"{row['margin_uncapped']:+6.2f} -> {row['margin_capped']:+6.2f} "
            f"({row['shift']:+5.2f})  DTW% {row['dtw_uncapped'] * 100:5.1f} -> "
            f"{row['dtw_capped'] * 100:5.1f}   {row['n_possession_cap_rows']} cap rows"
        )
        rows = (
            ledger.filter(
                (pl.col("game_id") == row["game_id"])
                & (pl.col("component") == POSSESSION_CAP_COMPONENT)
            )
            .sort(pl.col("luck_epa").abs(), descending=True)
            .head(5)
            .to_dicts()
        )
        for cap in rows:
            print(
                f"      {cap['event_class']:<16} ({cap['charged_team']})  "
                f"{cap['luck_epa']:+.2f} EPA"
            )
        out.append({**row, "cap_rows": rows})
    return out


def main() -> None:
    ctx = _audit.load_context()
    slope = ctx.slope
    print(f"points_per_epa = {slope:.10f}  (read from the game table, not refitted)")

    # ---- P-1, first and unconditional ---------------------------------------
    print(f"\n{'=' * 76}\nP-1 — Strict untouched, on the frame round 9 widened\n{'=' * 76}")
    _strict_all, _strict_ledger, v1 = _audit.v13_pass(ctx)

    # ---- P-2 ------------------------------------------------------------------
    uncapped, uncapped_ledger, p2_columns = gate_p2_columns_are_inert(ctx)
    strict = ctx.shipped.filter(pl.col("game_id").is_in(uncapped["game_id"].to_list())).sort(
        "game_id"
    )
    print(f"\n{'=' * 76}\nP-2 (b) — the uncapped arm still is document 59's\n{'=' * 76}")
    p2_doc59 = _full.reproduction(strict, uncapped)

    # ---- the capped arm --------------------------------------------------------
    capped, capped_ledger = _full.full_pass(ctx, edition="full", label="full (capped)")
    caps = capped_ledger.filter(pl.col("component") == POSSESSION_CAP_COMPONENT)
    print(f"\n{'=' * 76}\nWHAT THE CAP BOOKED\n{'=' * 76}")
    print(
        f"  {caps.height:,} cap rows over {capped.height:,} games — "
        f"{int((capped['n_possession_cap_rows'] > 0).sum()):,} games carry at least one, "
        f"median {float(capped['n_possession_cap_rows'].median()):.0f} per game, max "
        f"{int(capped['n_possession_cap_rows'].max())}"
    )
    print(
        f"  they book {float(caps['luck_epa'].sum()):+,.1f} EPA in total; per row median "
        f"{float(caps['luck_epa'].abs().median()):.3f} EPA, max "
        f"{float(caps['luck_epa'].abs().max()):.2f} EPA"
    )
    p3 = gate_p3_round_trip(capped, slope)
    print(f"\n{'=' * 76}\nTHE CAPPED ARM AGAINST DOCUMENT 59 (reported, not gated)\n{'=' * 76}")
    capped_reproduction = _full.reproduction(strict, capped, enforce=False)

    p7 = gate_p7(strict, uncapped, capped)
    named = named_games(strict, uncapped, capped, capped_ledger)
    movers = largest_reductions(uncapped, capped, capped_ledger)

    # ---- the artifact ----------------------------------------------------------
    summary = _full.to_summary(
        capped, strict.with_columns(pl.col("actual_margin").alias("actual_margin_strict"))
    )
    out = paths.RESEARCH_OUTPUT_DIR / FULL_ARTIFACT
    summary.write_parquet(out)
    print(f"\nwrote {out}  ({summary.height:,} rows) — the Full edition, capped")

    results = {
        "document": "61 — the possession-level luck cap",
        "part": "C — the audit, and the Full summary regenerated with the cap",
        "settings": {
            "random_seed": _audit.RANDOM_SEED,
            "posterior_draws": _audit.POSTERIOR_DRAWS,
            "coin_draws": _audit.COIN_DRAWS,
            "points_per_epa": slope,
            "seasons": list(FTN_SEASONS),
        },
        "gate_p1_v1_replay": v1,
        "gate_p2_columns_inert": p2_columns,
        "gate_p2_doc59_reproduction_uncapped": p2_doc59,
        "gate_p3_round_trip": p3,
        "capped_against_doc59": capped_reproduction,
        "cap_rows": {
            "rows": int(caps.height),
            "games_with_a_cap_row": int((capped["n_possession_cap_rows"] > 0).sum()),
            "total_epa": float(caps["luck_epa"].sum()),
            "median_abs_epa": float(caps["luck_epa"].abs().median()),
            "max_abs_epa": float(caps["luck_epa"].abs().max()),
        },
        "gate_p7_audit": p7,
        "named_games": named,
        "largest_reductions": movers,
        "artifact": FULL_ARTIFACT,
    }
    (paths.RESEARCH_OUTPUT_DIR / OUTPUT_NAME).write_text(
        json.dumps(results, indent=2, default=float)
    )
    print(f"wrote {paths.RESEARCH_OUTPUT_DIR / OUTPUT_NAME}")

    print(f"\n{'=' * 76}\nP-1, RE-PRINTED AT THE END OF THE RUN\n{'=' * 76}")
    print(
        f"V-1 replay: {v1['games_matched']:,} games, max |Δ deserved margin| "
        f"{v1['max_abs_gaps']['deserved_margin']:.2e}  -> "
        f"{'PASS' if v1['pass'] else 'FAIL'}"
    )
    if not v1["pass"]:
        raise SystemExit("v1.3 moved. Stop and report.")
    print("Next: document 62 records what moved, and figure round 7 draws the cap bar.")


if __name__ == "__main__":
    main()
