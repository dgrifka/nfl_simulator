"""v1.4, part 3 — the Full edition rebuilt, and document 64's headline set recomputed.

`research/83_simulator_v14.py` rebuilt Strict on the elevation posterior. This
does the same for the Full edition — amendment A-3's hands-on-the-ball class
with document 61's possession cap, over the 2022-2025 seasons FTN charting
reaches — and then recomputes every number document 64 published, so the
article's round 5 can cite one source per claim rather than two that disagree.

The comparison here is **v1.4 Full against v1.3 Full**, not Full against Strict.
Document 64 already reports the second; what a reader of this round needs is
which of its numbers the elevation term moved and by how much.

Gates:

    F-1  the Strict v1.4 arm replays `dtw_games_v14.parquet` at 0.00e+00, so
         this run and `research/83` are adjudicating the same corpus.
    F-2  the capped ledger sums (document 61's P-3), cap rows included.
    F-3  the Full arm's own round trip: every game's deserved margin is its
         actual margin less its booked luck.

Reported, not gated: the v1.3-to-v1.4 comparison, document 64's headline set,
and the walk-through game — which is `2025_13_DEN_WAS`, a Denver game whose
four field goals and five extra points are all repriced by this change.

    uv run python research/84_full_edition_v14.py

Writes `research/outputs/full_summary_v14.parquet` and
`research/outputs/84_full_edition_v14.json`. Neither is committed —
`research/outputs/` is gitignored, this script is the artifact, and document 68
is the record.
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
_receiver_audit = import_module("73_receiver_drop_variant_audit")
_full = import_module("76_full_edition_summary")
_cap = import_module("78_possession_cap_audit")
_refit = import_module("82_fg_v14_refit")
_v14 = import_module("83_simulator_v14")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.ingest import FTN_SEASONS  # noqa: E402

V14_FULL_ARTIFACT = "full_summary_v14.parquet"
V13_FULL_ARTIFACT = "full_summary.parquet"
OUTPUT_NAME = "84_full_edition_v14.json"

# Document 64 §8's walk-through game — and, as it happens, a game at 5,280 feet.
WALKTHROUGH = "2025_13_DEN_WAS"

# Document 64's published headline set, quoted so the run says out loud which
# of them moved rather than leaving the reader to diff two documents.
DOC64 = {
    "games": 1139,
    "sign_flips": 168,
    "sign_flip_share": 0.1475,
    "degenerate": 310,
    "degenerate_share": 0.2722,
    "non_degenerate": 829,
    "flips_among_non_degenerate": 167,
    "clear_flip": 128,
    "too_close_to_call": 95,
    "scoreboard_holds": 916,
    "median_abs_margin_shift": 3.43,
    "games_moving_more_than_3pt": 631,
    "largest_swing": 19.05,
    "largest_swing_game": "2024_19_LAC_HOU",
}


def headline(summary: pl.DataFrame, label: str) -> dict:
    """Document 64 §§1-5, recomputed. Element-wise everywhere, never by netting."""
    actual = summary["actual_margin"].to_numpy()
    deserved = summary["deserved_margin"].to_numpy()
    dtw = summary["dtw_home"].to_numpy()
    live = actual != 0.0
    flips = live & ((deserved > 0) != (actual > 0)) & (deserved != 0.0)
    degenerate = (dtw <= 0.001) | (dtw >= 0.999)
    band = (dtw >= 0.40) & (dtw <= 0.60)
    buckets = np.array(
        [_audit.bucket(float(d), float(a)) for d, a in zip(dtw, actual, strict=True)]
    )
    shift = np.abs(deserved - actual)
    dtw_flip = live & ((dtw < 0.5) != (actual < 0))
    return {
        "edition": label,
        "games": int(summary.height),
        "sign_flips": int(flips.sum()),
        "sign_flip_share": float(flips.sum() / summary.height),
        "dtw_below_half_for_the_realized_winner": int(dtw_flip.sum()),
        "definitions_disagree_on": int((flips != dtw_flip).sum()),
        "realized_ties": int((~live).sum()),
        "degenerate": int(degenerate.sum()),
        "degenerate_share": float(degenerate.sum() / summary.height),
        "non_degenerate": int((~degenerate).sum()),
        "flips_among_non_degenerate": int((flips & ~degenerate).sum()),
        "clear_flip": int((buckets == "clear flip").sum()),
        "too_close_to_call": int(band.sum()),
        "scoreboard_holds": int((buckets == "scoreboard holds").sum()),
        "median_abs_margin_shift": float(np.median(shift)),
        "games_moving_more_than_3pt": int((shift > 3.0).sum()),
        "largest_swing": float(shift.max()),
        "largest_swing_game": str(summary["game_id"].to_numpy()[int(np.argmax(shift))]),
        "luck_events_median": float(np.median(summary["n_events"].to_numpy())),
        "luck_events_mean": float(summary["n_events"].to_numpy().mean()),
    }


def against_doc64(v14: dict, v13: dict) -> list[dict]:
    """Every published number, three ways: document 64, this rerun of v1.3, v1.4."""
    print(f"\n{'=' * 82}\nDOCUMENT 64'S HEADLINE SET — published, v1.3 rerun, v1.4\n{'=' * 82}")
    print(f"  {'statistic':<34} {'doc 64':>10} {'v1.3':>10} {'v1.4':>10}   moved?")
    rows = []
    for key, published in DOC64.items():
        got13, got14 = v13.get(key), v14.get(key)
        if isinstance(published, str):
            moved = got14 != got13
            line = f"  {key:<34} {published:>10} {str(got13):>10} {str(got14):>10}"
        else:
            moved = not np.isclose(float(got14), float(got13), rtol=0, atol=5e-4)
            line = f"  {key:<34} {published:>10.4g} {float(got13):>10.4g} {float(got14):>10.4g}"
        rows.append(
            {
                "statistic": key,
                "document_64": published,
                "v13_rerun": got13,
                "v14": got14,
                "moved": bool(moved),
            }
        )
        print(f"{line}   {'MOVED' if moved else '-'}")
    print(
        "\n  The v1.3 column is this run's own recomputation of document 64, not a quotation\n"
        "  of it. Where it differs from the published column, the published column is what\n"
        "  the article cited and the difference is a reproduction gap to explain — not\n"
        "  something v1.4 did."
    )
    return rows


def edition_delta(v13: pl.DataFrame, v14: pl.DataFrame) -> dict:
    """What the elevation term moved inside the Full edition, game by game."""
    joined = v13.select(
        "game_id",
        "actual_margin",
        pl.col("deserved_margin").alias("margin_v13"),
        pl.col("dtw_home").alias("dtw_v13"),
    ).join(
        v14.select(
            "game_id",
            pl.col("deserved_margin").alias("margin_v14"),
            pl.col("dtw_home").alias("dtw_v14"),
        ),
        on="game_id",
    )
    actual = joined["actual_margin"].to_numpy()
    d_dtw = (joined["dtw_v14"] - joined["dtw_v13"]).to_numpy()
    d_margin = (joined["margin_v14"] - joined["margin_v13"]).to_numpy()
    bucket13 = np.array(
        [
            _audit.bucket(float(d), float(a))
            for d, a in zip(joined["dtw_v13"].to_numpy(), actual, strict=True)
        ]
    )
    bucket14 = np.array(
        [
            _audit.bucket(float(d), float(a))
            for d, a in zip(joined["dtw_v14"].to_numpy(), actual, strict=True)
        ]
    )
    moved = bucket13 != bucket14
    touched = np.abs(d_dtw) > 1e-12
    report = {
        "games": int(joined.height),
        "games_touched": int(touched.sum()),
        "bucket_moves": int(moved.sum()),
        "bucket_move_games": sorted(joined["game_id"].to_numpy()[moved].tolist()),
        "median_abs_delta_dtw_pp_on_touched": float(np.median(np.abs(d_dtw[touched]))) * 100
        if touched.any()
        else 0.0,
        "max_abs_delta_dtw_pp": float(np.abs(d_dtw).max()) * 100,
        "median_abs_delta_margin_on_touched": float(np.median(np.abs(d_margin[touched])))
        if touched.any()
        else 0.0,
        "max_abs_delta_margin": float(np.abs(d_margin).max()),
    }
    print(f"\n{'=' * 82}\nWHAT v1.4 MOVED INSIDE THE FULL EDITION\n{'=' * 82}")
    print(
        f"  {report['games_touched']:,} of {joined.height:,} games moved at all; "
        f"median |ΔDTW| on them {report['median_abs_delta_dtw_pp_on_touched']:.3f} pp, "
        f"largest {report['max_abs_delta_dtw_pp']:.2f} pp"
    )
    print(
        f"  median |Δ deserved margin| on them "
        f"{report['median_abs_delta_margin_on_touched']:.3f} pt, largest "
        f"{report['max_abs_delta_margin']:.3f} pt"
    )
    print(
        f"  verdict bucket moved in {report['bucket_moves']} games: "
        f"{report['bucket_move_games'] or 'none'}"
    )

    top = (
        joined.with_columns(
            (pl.col("dtw_v14") - pl.col("dtw_v13")).alias("delta_dtw"),
            (pl.col("margin_v14") - pl.col("margin_v13")).alias("delta_margin"),
        )
        .sort(pl.col("delta_dtw").abs(), descending=True)
        .head(10)
    )
    print("\n  The ten Full-edition games v1.4 moves furthest:")
    with pl.Config(tbl_rows=12, tbl_width_chars=200):
        print(
            top.select(
                "game_id",
                "actual_margin",
                pl.col("margin_v13").round(3),
                pl.col("margin_v14").round(3),
                (pl.col("dtw_v13") * 100).round(2).alias("DTW%_v13"),
                (pl.col("dtw_v14") * 100).round(2).alias("DTW%_v14"),
                (pl.col("delta_dtw") * 100).round(2).alias("ΔDTW_pp"),
            )
        )
    report["largest_movers"] = top.to_dicts()
    return report


def walkthrough(strict_ledger: pl.DataFrame, full_ledger: pl.DataFrame, slope: float) -> dict:
    """Document 64 §§7-8, recomputed — and this game is played at 5,280 feet.

    Every kicking row in it is repriced by v1.4, which makes it the single most
    load-bearing recomputation of this round for the article: §7's worked
    example is a fumble and does not move, but §8a's twelve rows and §8b's
    component totals both do.
    """
    print(f"\n{'=' * 82}\nTHE WALK-THROUGH GAME, {WALKTHROUGH}, REPRICED\n{'=' * 82}")
    out = {}
    for label, ledger in (("strict_v14", strict_ledger), ("full_v14", full_ledger)):
        rows = ledger.filter(pl.col("game_id") == WALKTHROUGH).sort("play_id")
        if not rows.height:
            print(f"  {label}: no rows")
            out[label] = None
            continue
        by_component = (
            rows.group_by("component")
            .agg(
                pl.len().alias("rows"),
                pl.col("luck_epa").sum().alias("luck_epa"),
            )
            .with_columns((pl.col("luck_epa") * slope).alias("points"))
            .sort("component")
        )
        print(
            f"\n  {label} — {rows.height} rows, "
            f"{float(rows['luck_epa'].sum()):+.4f} EPA "
            f"({float(rows['luck_epa'].sum()) * slope:+.3f} pt)"
        )
        with pl.Config(tbl_rows=20, tbl_width_chars=160):
            print(by_component)
        out[label] = {
            "rows": int(rows.height),
            "total_luck_epa": float(rows["luck_epa"].sum()),
            "total_points": float(rows["luck_epa"].sum()) * slope,
            "by_component": by_component.to_dicts(),
        }
    kicks = strict_ledger.filter(
        (pl.col("game_id") == WALKTHROUGH)
        & pl.col("component").is_in(["field_goal", "extra_point"])
    ).sort("play_id")
    print(
        f"\n  Every kicking row in {WALKTHROUGH}, as v1.4 prices it "
        "(document 64 §8a's table, repriced):"
    )
    with pl.Config(tbl_rows=20, tbl_width_chars=200):
        print(
            kicks.select(
                "play_id",
                "component",
                "event_class",
                "charged_team",
                pl.col("actual").alias("y"),
                pl.col("expected").round(4).alias("p"),
                pl.col("swing").round(4),
                pl.col("luck_epa").round(4),
                (pl.col("luck_epa") * slope).round(3).alias("points"),
            )
        )
    out["kicking_rows_v14"] = kicks.to_dicts()
    return out


def main() -> None:
    ctx = _audit.load_context(
        fg_trace=_refit.TRACE_NAME,
        fg_summary=_refit.SUMMARY_NAME,
        shipped_artifact=_v14.V14_GAMES,
    )
    slope = ctx.slope
    print(f"points_per_epa = {slope:.10f}  (read from the game table, not refitted)")
    print(
        f"field-goal posterior: {_refit.TRACE_NAME}, beta_elev "
        f"{ctx.fg_model.beta_elev.mean():+.5f} per 1,000 ft"
    )

    # ---- F-1, first and unconditional -------------------------------------
    print(f"\n{'=' * 82}\nF-1 — the Strict v1.4 arm against dtw_games_v14.parquet\n{'=' * 82}")
    strict_all, strict_ledger, f1 = _audit.v13_pass(ctx)
    if not f1["pass"]:
        raise SystemExit(
            "the Strict arm here is not research/83's. The two runs would be adjudicating "
            "different corpora. Stop and report."
        )

    # ---- the capped Full arm ----------------------------------------------
    capped, capped_ledger = _full.full_pass(ctx, edition="full", label="full v1.4 (capped)")
    f2 = _cap.gate_p3_round_trip(capped, slope)

    strict = ctx.shipped.filter(pl.col("game_id").is_in(capped["game_id"].to_list())).sort(
        "game_id"
    )
    summary = _full.to_summary(
        capped, strict.with_columns(pl.col("actual_margin").alias("actual_margin_strict"))
    )

    v13_full = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / V13_FULL_ARTIFACT)
    print(f"\n  v1.3 Full on file: {v13_full.height:,} games; v1.4 Full: {summary.height:,} games")
    delta = edition_delta(v13_full, summary)

    v14_head = headline(summary, "Full v1.4")
    v13_head = headline(v13_full, "Full v1.3")
    comparison = against_doc64(v14_head, v13_head)

    walk = walkthrough(strict_ledger, capped_ledger, slope)

    out = paths.RESEARCH_OUTPUT_DIR / V14_FULL_ARTIFACT
    summary.write_parquet(out)
    print(f"\nwrote {out}  ({summary.height:,} rows) — the Full edition at v1.4, capped")

    results = {
        "document": "68 — simulator v1.4",
        "part": "C — the Full edition, and document 64's headline set recomputed",
        "settings": {
            "random_seed": _audit.RANDOM_SEED,
            "posterior_draws": _audit.POSTERIOR_DRAWS,
            "coin_draws": _audit.COIN_DRAWS,
            "points_per_epa": slope,
            "seasons": list(FTN_SEASONS),
            "field_goal_posterior": _refit.TRACE_NAME,
        },
        "gate_f1_strict_v14_replay": f1,
        "gate_f2_capped_round_trip": f2,
        "headline_v14": v14_head,
        "headline_v13": v13_head,
        "against_document_64": comparison,
        "edition_delta_v13_to_v14": delta,
        "walkthrough_game": {"game_id": WALKTHROUGH, **walk},
        "artifact": V14_FULL_ARTIFACT,
        "strict_games": int(strict_all.height),
    }
    (paths.RESEARCH_OUTPUT_DIR / OUTPUT_NAME).write_text(
        json.dumps(results, indent=2, default=float)
    )
    print(f"wrote {paths.RESEARCH_OUTPUT_DIR / OUTPUT_NAME}")
    print("\nNext: document 68 records what v1.4 prices, and what of document 64 moved.")


if __name__ == "__main__":
    main()
