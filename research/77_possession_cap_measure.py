"""Round 9, part A — the possession-level cap, measured before it is built.

Document 61 §4. The defect the cap addresses is stated in document 61 §0: the
ledger prices every event from the game state *at that play*, then sums, so two
events on one possession are added as though the second would still have
happened had the first gone the other way. Document 61 §3 proposes the bound

    C_d = max_i |swing_i|   on drive d

and §4 says measure before building. Nothing here changes the simulator. The
Full edition is run exactly as `research/73` runs its `+dp+rd` arm, the ledger
rows are joined to `fixed_drive`, and the four M-items are read off that join:

    M-1  how much of the Full ledger even shares a drive
    M-2  how often `C_d` would bite, and by how much (EPA and points)
    M-3  the 200 Full bucket-move games — how many could lose their move
    M-4  the same M-2 on Strict, so the cost of leaving v1.3 alone is stated

V-1 rides along at the top, because `research/68`'s v1.3 pass is what supplies
the Strict ledger M-4 needs and the gate is free once the pass has run.

    uv run python research/77_possession_cap_measure.py

Writes `research/outputs/77_possession_cap_measure.json`. `research/outputs/`
is gitignored — this script is the committed artifact and document 61 §4 is the
committed record of the numbers.
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

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.ingest import FTN_SEASONS, PBP_SEASONS, load_pbp  # noqa: E402

OUTPUT_NAME = "77_possession_cap_measure.json"
FULL_LEDGER_CACHE = "77_full_ledger_drives.parquet"
STRICT_LEDGER_CACHE = "77_strict_ledger_drives.parquet"

# Document 61 §4's three, and they are the ones figure round 6 renders.
NAMED_GAMES = ("2025_17_DET_MIN", "2022_13_WAS_NYG", "2024_19_LAC_HOU")

# Document 61 §4's pre-committed threshold, quoted so the run can say out loud
# whether it tripped: if more than a third of the 200 bucket moves would vanish,
# document 59's headline was substantially double counting and the maintainer sees the
# measurement before anything is built.
DOC59_BUCKET_MOVES = 200
STOP_AND_ASK_SHARE = 1.0 / 3.0

DRIVE_COLUMNS = ["game_id", "play_id", "fixed_drive", "fixed_drive_result", "qtr", "posteam"]


# --------------------------------------------------------------------------
# the join
# --------------------------------------------------------------------------


def drive_frame(seasons: tuple[int, ...]) -> pl.DataFrame:
    """`play_id` to the drive it belongs to, for every play in `seasons`.

    Loaded separately rather than through `44_read_side_fix.SIM_COLUMNS`: part A
    measures, and widening the frame every simulation reads is a change to the
    thing being measured. Part B adds `fixed_drive` to `SIM_COLUMNS` and proves
    V-1 still lands on 0.00e+00 with it there.
    """
    return load_pbp(seasons, columns=DRIVE_COLUMNS)


def attach_drives(ledger: pl.DataFrame, drives: pl.DataFrame) -> pl.DataFrame:
    """Every ledger row with the drive its play sat on.

    Document 61 §2: events with no drive are their own group. They should not
    exist — every play nflverse charts carries a `fixed_drive` — so the guard
    keys them on their own `play_id` and the run reports how many there were
    rather than dropping them.
    """
    joined = ledger.join(drives, on=["game_id", "play_id"], how="left")
    if joined.height != ledger.height:
        raise SystemExit(
            f"the drive join changed the row count, {ledger.height:,} -> {joined.height:,}. "
            "A (game_id, play_id) pair is not unique in the play-by-play; stop and report."
        )
    return joined.with_columns(
        pl.when(pl.col("fixed_drive").is_null())
        .then(pl.concat_str([pl.lit("no-drive:"), pl.col("play_id").cast(pl.String)]))
        .otherwise(pl.col("fixed_drive").cast(pl.Int64).cast(pl.String))
        .alias("drive_key")
    )


def drive_table(ledger: pl.DataFrame) -> pl.DataFrame:
    """One row per (game, drive) that carries at least one luck event.

    The three quantities document 61 turns on, per drive:

    * ``sum_abs_luck`` — ``Σ|luck_i|``, the statistic document 61 §4 M-2 names.
    * ``signed_luck`` — ``Σ luck_i``, which is what the clip actually acts on at
      the point estimate, so it is what the reduction is computed from.
    * ``cap`` — ``C_d = max_i |swing_i|``.

    and one more the pre-registration did not name but the bootstrap makes
    exact. Each event's per-replicate contribution is
    ``a_i = (actual_i − replayed_i) × swing_i`` with ``replayed_i ∈ {0, 1}`` and
    ``actual_i ∈ {0, 1}``, so ``a_i`` is either zero or ``s_i = (2·actual_i − 1)
    · swing_i`` — one fixed signed value per event. The largest ``|Σ_i a_i|`` any
    replicate can reach is therefore the larger of the drive's positive and
    negative ``s_i`` totals, and the cap bites in at least one replicate exactly
    when that exceeds ``C_d``. ``bites_ever`` is that, and it is the honest
    answer to "does this drive ever get clipped".
    """
    signed = ((2.0 * pl.col("actual") - 1.0) * pl.col("swing")).alias("replicate_swing")
    grouped = (
        ledger.with_columns(signed)
        .group_by("game_id", "drive_key")
        .agg(
            pl.len().alias("n_events"),
            pl.col("luck_epa").abs().sum().alias("sum_abs_luck"),
            pl.col("luck_epa").sum().alias("signed_luck"),
            pl.col("swing").abs().max().alias("cap"),
            pl.col("replicate_swing")
            .filter(pl.col("replicate_swing") > 0)
            .sum()
            .alias("pos_swing"),
            pl.col("replicate_swing")
            .filter(pl.col("replicate_swing") < 0)
            .sum()
            .alias("neg_swing"),
            pl.col("fixed_drive").min().alias("fixed_drive"),
            pl.col("qtr").min().alias("qtr"),
            # The drive's offence is the team possessing on its *first* play, by
            # play order rather than by whatever order the group-by happened to
            # hand back. It is not always the only `posteam` on the drive: a
            # would-be touchdown dropped and returned for a score puts the other
            # team's extra point on the same `fixed_drive`, which is exactly what
            # 2024_19_LAC_HOU's drive 21 does.
            pl.col("posteam").sort_by("play_id").first().alias("posteam"),
            pl.col("posteam").n_unique().alias("n_posteams"),
            pl.col("component").sort().alias("components"),
        )
    )
    # Sorted, because `group_by` hands rows back in whatever order the threads
    # finished in and `expected_clip` assigns its random draws by row position.
    # Without this the Monte Carlo estimate is not reproducible run to run.
    return (
        grouped.sort("game_id", "drive_key")
        .with_columns(
            pl.max_horizontal(pl.col("pos_swing"), -pl.col("neg_swing")).alias("max_attainable"),
            pl.col("signed_luck").clip(-pl.col("cap"), pl.col("cap")).alias("clipped_luck"),
        )
        .with_columns(
            (pl.col("sum_abs_luck") > pl.col("cap")).alias("bites_prereg"),
            (pl.col("signed_luck").abs() > pl.col("cap")).alias("bites_point_estimate"),
            (pl.col("max_attainable") > pl.col("cap")).alias("bites_ever"),
            (pl.col("signed_luck") - pl.col("clipped_luck")).alias("reduction_epa"),
        )
    )


# --------------------------------------------------------------------------
# what the bootstrap will actually clip
# --------------------------------------------------------------------------

# The estimate below is a Monte Carlo over layer 2 alone, at v1.3's coin count
# rounded up for stability. Seeded, so the number is reproducible.
CLIP_DRAWS = 2000
CLIP_CHUNK = 200
CLIP_SEED = 20260828


def expected_clip(ledger: pl.DataFrame, drives: pl.DataFrame) -> pl.DataFrame:
    """Per drive, the clipped amount the *bootstrap* will average — not the point estimate.

    Document 61 §4's M-2 statistic is ``Σ|luck_i| > C_d``, a point-estimate
    question, and this run found it nearly blind: the cap bites on 0.9% of Full
    drives that way while clipping in at least one replicate on 69.3% of them.
    The two disagree because they are not the same quantity. In the bootstrap a
    single event contributes either nothing or its **whole swing** —
    ``a_i = (actual_i − replayed_i) × swing_i`` with both terms in {0, 1} — so a
    drive's replicate sum ranges over subset sums of ``s_i = (2·actual_i − 1) ·
    swing_i``, which are far larger than the expectation-weighted
    ``luck_i = (actual_i − p_i) × swing_i`` the ledger prints.

    What document 61 §5 actually books as a cap row is
    ``mean(clipped) − mean(unclipped)``, a mean over replicates. That is what
    this estimates, and it is the honest answer to M-2's "by how much".

    Event ``i`` contributes ``s_i`` with probability ``q_i = |actual_i − p_i|``
    and zero otherwise, and ``q_i · s_i = luck_i`` exactly — so the unclipped
    mean this reproduces is the ledger's own row sum, and only the clip is new.
    Layer 1 is held at the posterior mean rather than redrawn; this is a
    measurement of the clip's size, not a replacement for the bootstrap part C
    runs.

    The row order is pinned to a total order before the draws are assigned.
    Polars hands a `group_by` back in whatever order its threads finished in,
    and this function gives each event its own column of uniforms — so without
    the sort the same input would produce a different estimate every run, which
    is not a number worth writing into document 61.
    """
    ordered = ledger.join(
        drives.select("game_id", "drive_key", "cap").with_row_index("drive_index"),
        on=["game_id", "drive_key"],
        how="left",
    ).sort("drive_index", "play_id", "component", "event_class", "charged_team")
    drive_index = ordered["drive_index"].to_numpy()
    q = np.abs(ordered["actual"].to_numpy() - ordered["expected"].to_numpy())
    s = (2.0 * ordered["actual"].to_numpy() - 1.0) * ordered["swing"].to_numpy()
    cap = drives["cap"].to_numpy()

    starts = np.flatnonzero(np.r_[True, drive_index[1:] != drive_index[:-1]])
    rng = np.random.default_rng(CLIP_SEED)
    signed = np.zeros(len(cap))
    absolute = np.zeros(len(cap))
    clipped_any = np.zeros(len(cap), dtype=bool)
    for start in range(0, CLIP_DRAWS, CLIP_CHUNK):
        chunk = min(CLIP_CHUNK, CLIP_DRAWS - start)
        contribution = (rng.random((chunk, len(q))) < q) * s
        totals = np.add.reduceat(contribution, starts, axis=1)
        excess = totals - np.clip(totals, -cap, cap)
        signed += excess.sum(axis=0)
        absolute += np.abs(excess).sum(axis=0)
        clipped_any |= (excess != 0.0).any(axis=0)
    return drives.with_columns(
        # The cap row document 61 §5 defines: mean(clipped) − mean(unclipped).
        pl.Series("cap_row_epa", -signed / CLIP_DRAWS),
        pl.Series("expected_abs_clip_epa", absolute / CLIP_DRAWS),
        pl.Series("clipped_in_the_draw", clipped_any),
    )


# --------------------------------------------------------------------------
# M-1 … M-4
# --------------------------------------------------------------------------


def m1(drives: pl.DataFrame, ledger: pl.DataFrame, all_drives: int) -> dict:
    """How much of the Full ledger shares a possession with something else."""
    counts = drives["n_events"].to_numpy()
    shared_events = int(counts[counts >= 2].sum())
    histogram = {
        str(int(row["n_events"])): int(row["count"])
        for row in drives["n_events"].value_counts().sort("n_events").to_dicts()
    }
    report = {
        "events": int(ledger.height),
        "events_sharing_a_drive": shared_events,
        "share_of_events_sharing_a_drive": shared_events / int(ledger.height),
        "drives_with_an_event": int(drives.height),
        "drives_with_two_or_more_events": int((counts >= 2).sum()),
        "share_of_event_drives_with_two_or_more": float((counts >= 2).mean()),
        "drives_in_the_population": all_drives,
        "share_of_all_drives_with_two_or_more": float((counts >= 2).sum()) / all_drives,
        "events_per_event_drive_mean": float(counts.mean()),
        "events_per_event_drive_max": int(counts.max()),
        "events_per_drive_histogram": histogram,
        "events_with_no_drive": int(ledger["fixed_drive"].is_null().sum()),
    }
    print(f"\n{'=' * 76}\nM-1 — how much of the Full ledger shares a possession\n{'=' * 76}")
    print(
        f"  {report['events']:,} Full events sit on {report['drives_with_an_event']:,} "
        f"drives; {report['events_sharing_a_drive']:,} of them "
        f"({report['share_of_events_sharing_a_drive']:.1%}) share their drive with at "
        f"least one other event"
    )
    print(
        f"  {report['drives_with_two_or_more_events']:,} drives carry two or more events "
        f"— {report['share_of_event_drives_with_two_or_more']:.1%} of the drives that "
        f"carry any, {report['share_of_all_drives_with_two_or_more']:.1%} of all "
        f"{all_drives:,} drives played"
    )
    print(
        f"  events per event-carrying drive: mean {report['events_per_event_drive_mean']:.2f}, "
        f"max {report['events_per_event_drive_max']}"
    )
    print(f"  distribution: {histogram}")
    if report["events_with_no_drive"]:
        print(f"  GUARD: {report['events_with_no_drive']:,} events carry no `fixed_drive`")
    return report


def m2(drives: pl.DataFrame, slope: float, *, label: str, heading: str) -> dict:
    """How often the cap bites, and by how much. M-2 on Full, M-4 on Strict."""
    prereg = drives["bites_prereg"].to_numpy()
    point = drives["bites_point_estimate"].to_numpy()
    ever = drives["bites_ever"].to_numpy()
    reduction = np.abs(drives["reduction_epa"].to_numpy())
    bitten = reduction[point]
    cap_rows = drives["cap_row_epa"].to_numpy()
    clipped = drives["clipped_in_the_draw"].to_numpy()
    report = {
        "arm": label,
        "drives_with_an_event": int(drives.height),
        "statistic_prereg": "document 61 §4 M-2's own statistic, Σ|luck_i| > C_d",
        "drives_bitten_prereg": int(prereg.sum()),
        "share_bitten_prereg": float(prereg.mean()),
        "statistic_point_estimate": "the clip at the point estimate, |Σ luck_i| > C_d",
        "drives_bitten_point_estimate": int(point.sum()),
        "share_bitten_point_estimate": float(point.mean()),
        "statistic_ever": (
            "exact for the bootstrap: max attainable |Σ a_i| over replicates > C_d, so "
            "the drive is clipped in at least one replicate"
        ),
        "drives_bitten_ever": int(ever.sum()),
        "share_bitten_ever": float(ever.mean()),
        "reduction_epa_total": float(bitten.sum()),
        "reduction_points_total": float(bitten.sum() * slope),
        "reduction_epa_median_on_bitten": float(np.median(bitten)) if bitten.size else 0.0,
        "reduction_epa_mean_on_bitten": float(bitten.mean()) if bitten.size else 0.0,
        "reduction_epa_max": float(bitten.max()) if bitten.size else 0.0,
        "reduction_points_max": float(bitten.max() * slope) if bitten.size else 0.0,
        "statistic_cap_row": (
            "document 61 §5's cap row, mean(clipped) − mean(unclipped) over replicates, "
            f"estimated at {CLIP_DRAWS:,} layer-2 draws with layer 1 held at the "
            "posterior mean"
        ),
        "drives_with_a_cap_row": int(clipped.sum()),
        "share_with_a_cap_row": float(clipped.mean()),
        "cap_row_epa_total": float(cap_rows[clipped].sum()),
        "cap_row_points_total": float(cap_rows[clipped].sum() * slope),
        "abs_cap_row_epa_median_on_capped": float(np.median(np.abs(cap_rows[clipped])))
        if clipped.any()
        else 0.0,
        "abs_cap_row_epa_mean_on_capped": float(np.abs(cap_rows[clipped]).mean())
        if clipped.any()
        else 0.0,
        "abs_cap_row_epa_max": float(np.abs(cap_rows).max()) if cap_rows.size else 0.0,
        "abs_cap_row_points_max": float(np.abs(cap_rows).max() * slope) if cap_rows.size else 0.0,
    }
    print(f"\n{'=' * 76}\n{heading}\n{'=' * 76}")
    print(
        f"  as document 61 §4 writes it (Σ|luck| > C_d): {report['drives_bitten_prereg']:,} of "
        f"{report['drives_with_an_event']:,} event-carrying drives "
        f"({report['share_bitten_prereg']:.1%})"
    )
    print(
        f"  as the clip acts at the point estimate (|Σ luck| > C_d): "
        f"{report['drives_bitten_point_estimate']:,} ({report['share_bitten_point_estimate']:.1%})"
    )
    print(
        f"  as the bootstrap will actually see it (clipped in ≥ 1 replicate): "
        f"{report['drives_bitten_ever']:,} ({report['share_bitten_ever']:.1%})"
    )
    if bitten.size:
        print(
            f"  and by how much, on the drives the point estimate clips: median "
            f"{report['reduction_epa_median_on_bitten']:.2f} EPA "
            f"({report['reduction_epa_median_on_bitten'] * slope:.2f} pt), mean "
            f"{report['reduction_epa_mean_on_bitten']:.2f} EPA, max "
            f"{report['reduction_epa_max']:.2f} EPA "
            f"({report['reduction_points_max']:.2f} pt); "
            f"{report['reduction_epa_total']:,.0f} EPA "
            f"({report['reduction_points_total']:,.0f} pt) over the whole population"
        )
    else:
        print("  no drive is clipped at the point estimate")
    print(
        "  and what the bootstrap will actually book — document 61 §5's cap row, "
        "mean(clipped) − mean(unclipped):"
    )
    if clipped.any():
        print(
            f"    {report['drives_with_a_cap_row']:,} drives take a cap row "
            f"({report['share_with_a_cap_row']:.1%}); |cap row| median "
            f"{report['abs_cap_row_epa_median_on_capped']:.3f} EPA "
            f"({report['abs_cap_row_epa_median_on_capped'] * slope:.3f} pt), mean "
            f"{report['abs_cap_row_epa_mean_on_capped']:.3f} EPA, max "
            f"{report['abs_cap_row_epa_max']:.2f} EPA "
            f"({report['abs_cap_row_points_max']:.2f} pt)"
        )
        print(
            f"    over the whole population the cap moves "
            f"{report['cap_row_epa_total']:+,.0f} EPA "
            f"({report['cap_row_points_total']:+,.0f} pt) of booked luck"
        )
    else:
        print("    no drive takes a cap row")
    return report


def m3(
    strict: pl.DataFrame,
    full: pl.DataFrame,
    drives: pl.DataFrame,
    slope: float,
) -> dict:
    """The 200 Full bucket-move games, against the cap.

    Two numbers document 61 §4 asks for — how many of the moved games hold a
    drive the cap bites, and the size of the point-estimate reduction on them —
    and one the handoff's stop-and-ask needs: how many of the 200 moves would
    *vanish*.

    The third cannot be computed exactly here, because a verdict bucket is a
    function of DTW% and DTW% comes out of the bootstrap this round has not yet
    changed. Two things are reported instead, and neither is dressed up as the
    answer part C will produce:

    * an **upper bound**, which is exact: a move can only vanish on a game the
      cap touches at all, so the count of moved games holding a bitten drive
      bounds it from above;
    * a **linear proxy**, stated as such: the cap gives back ``R_g × slope``
      points of the ``margin_full − margin_strict`` shift, so DTW is walked back
      by that same fraction of ``DTW_full − DTW_strict`` and the game re-bucketed.
      It assumes the DTW shift is locally linear in the margin shift and that the
      cap only ever undoes Full-direction luck, neither of which is exactly true.
    """
    moved, _delta, _half = _receiver_audit.bucket_moves(strict, full)
    per_game = (
        drives.group_by("game_id")
        .agg(
            pl.col("clipped_in_the_draw").any().alias("bitten"),
            pl.col("bites_ever").any().alias("bitten_ever"),
            pl.col("cap_row_epa").sum().alias("cap_row_epa"),
        )
        .with_columns((-pl.col("cap_row_epa") * slope).alias("reduction_points"))
    )
    joined = (
        strict.select(
            "game_id",
            "actual_margin",
            pl.col("deserved_margin").alias("margin_strict"),
            pl.col("dtw_home").alias("dtw_strict"),
        )
        .join(
            full.select(
                "game_id",
                pl.col("deserved_margin").alias("margin_full"),
                pl.col("dtw_home").alias("dtw_full"),
            ),
            on="game_id",
        )
        .join(per_game, on="game_id", how="left")
        .with_columns(
            pl.col("bitten").fill_null(False),
            pl.col("bitten_ever").fill_null(False),
            pl.col("cap_row_epa").fill_null(0.0),
            pl.col("reduction_points").fill_null(0.0),
        )
    )

    rows = joined.filter(pl.col("game_id").is_in(list(moved))).sort("game_id")
    bitten = rows["bitten"].to_numpy()
    bitten_ever = rows["bitten_ever"].to_numpy()
    give_back = np.abs(rows["reduction_points"].to_numpy())

    # The linear proxy, on the moved games only.
    shift = (rows["margin_full"] - rows["margin_strict"]).to_numpy()
    fraction = np.where(
        np.abs(shift) > 1e-12, np.abs(rows["reduction_points"].to_numpy() / shift), 0.0
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    dtw_proxy = rows["dtw_strict"].to_numpy() + (1.0 - fraction) * (
        rows["dtw_full"].to_numpy() - rows["dtw_strict"].to_numpy()
    )
    actual_margin = rows["actual_margin"].to_numpy()
    still_moved = [
        _audit.bucket(float(strict_dtw), float(margin))
        != _audit.bucket(float(proxy_dtw), float(margin))
        for strict_dtw, proxy_dtw, margin in zip(
            rows["dtw_strict"].to_numpy(), dtw_proxy, actual_margin, strict=True
        )
    ]
    vanished = int(len(still_moved) - sum(still_moved))

    report = {
        "bucket_moves": len(moved),
        "bucket_moves_doc59": DOC59_BUCKET_MOVES,
        "moved_games_with_a_cap_row": int(bitten.sum()),
        "moved_games_with_a_bitten_drive_ever": int(bitten_ever.sum()),
        "share_of_moved_games_touched": float(bitten_ever.mean()),
        "give_back_points_median_on_touched": float(np.median(give_back[bitten]))
        if bitten.any()
        else 0.0,
        "give_back_points_mean_on_touched": float(give_back[bitten].mean())
        if bitten.any()
        else 0.0,
        "give_back_points_max": float(give_back.max()) if give_back.size else 0.0,
        "median_fraction_of_the_full_shift_returned": float(np.median(fraction)),
        "upper_bound_on_moves_that_could_vanish": int(bitten_ever.sum()),
        "linear_proxy_moves_that_would_vanish": vanished,
        "linear_proxy_share": vanished / len(moved),
        "stop_and_ask_threshold_share": STOP_AND_ASK_SHARE,
        "stop_and_ask_tripped": bool(vanished / len(moved) > STOP_AND_ASK_SHARE),
        "proxy_caveat": (
            "the vanish count is a linear proxy on the point estimate, not a bootstrap "
            "result; part C recomputes it with the cap actually in the bootstrap"
        ),
    }
    print(f"\n{'=' * 76}\nM-3 — the {len(moved)} Full bucket moves against the cap\n{'=' * 76}")
    print(
        f"  {report['moved_games_with_a_bitten_drive_ever']:,} of the {len(moved)} moved games "
        f"hold a drive the cap clips in at least one replicate "
        f"({report['share_of_moved_games_touched']:.1%}); "
        f"{report['moved_games_with_a_cap_row']:,} take at least one cap row in the draw"
    )
    print(
        f"  the cap gives back (document 61 §5's cap rows, summed per game), on those "
        f"games: median "
        f"{report['give_back_points_median_on_touched']:.2f} pt, mean "
        f"{report['give_back_points_mean_on_touched']:.2f} pt, max "
        f"{report['give_back_points_max']:.2f} pt of deserved margin"
    )
    print(
        f"  that is a median {report['median_fraction_of_the_full_shift_returned']:.1%} of the "
        f"Full-minus-Strict margin shift handed back"
    )
    print(
        f"  UPPER BOUND on moves that could vanish: "
        f"{report['upper_bound_on_moves_that_could_vanish']} of {len(moved)} "
        f"({report['upper_bound_on_moves_that_could_vanish'] / len(moved):.1%}) — exact, a "
        f"move cannot vanish on a game the cap never touches"
    )
    print(
        f"  LINEAR PROXY for how many actually vanish: {vanished} of {len(moved)} "
        f"({report['linear_proxy_share']:.1%}) against the "
        f"{STOP_AND_ASK_SHARE:.1%} stop-and-ask threshold -> "
        f"{'STOP AND ASK' if report['stop_and_ask_tripped'] else 'proceed to part B'}"
    )
    return report


def named_game_drives(ledger: pl.DataFrame, drives: pl.DataFrame, slope: float) -> dict:
    """Document 61 §4's three games, drive by drive."""
    print(f"\n{'=' * 76}\nTHE THREE NAMED GAMES, DRIVE BY DRIVE\n{'=' * 76}")
    out = {}
    for game_id in NAMED_GAMES:
        rows = drives.filter(pl.col("game_id") == game_id).sort("fixed_drive", "drive_key")
        if not rows.height:
            print(f"\n  {game_id}: no Full luck events")
            out[game_id] = None
            continue
        print(f"\n  {game_id}")
        print(
            f"    {'drive':>6}  {'qtr':>3}  {'off':>3}  {'n':>2}  {'Σ luck':>8}  "
            f"{'Σ|luck|':>8}  {'C_d':>7}  {'cap row':>8}  components"
        )
        entries = []
        for row in rows.iter_rows(named=True):
            events = (
                ledger.filter(
                    (pl.col("game_id") == game_id) & (pl.col("drive_key") == row["drive_key"])
                )
                .sort("play_id")
                .select("play_id", "component", "event_class", "charged_team", "swing", "luck_epa")
                .to_dicts()
            )
            entries.append({**row, "events": events})
            flag = "  <- cap row" if row["clipped_in_the_draw"] else ""
            print(
                f"    {int(row['fixed_drive']) if row['fixed_drive'] is not None else '--':>6}  "
                f"{int(row['qtr']) if row['qtr'] is not None else '-':>3}  "
                f"{str(row['posteam']):>3}  {row['n_events']:>2}  {row['signed_luck']:>8.2f}  "
                f"{row['sum_abs_luck']:>8.2f}  {row['cap']:>7.2f}  "
                f"{row['cap_row_epa']:>8.3f}  "
                f"{','.join(sorted(set(row['components'])))}{flag}"
            )
        total = float(rows["cap_row_epa"].sum())
        print(
            f"    total cap rows: {total:+.2f} EPA ({-total * slope:+.2f} pt of deserved "
            f"margin handed back)"
        )
        out[game_id] = {
            "cap_row_epa": total,
            "reduction_points": -total * slope,
            "drives": entries,
        }
    return out


# --------------------------------------------------------------------------


def main() -> None:
    ctx = _audit.load_context()
    slope = ctx.slope
    print(f"points_per_epa = {slope:.10f}  (read from the game table, not refitted)")

    # V-1 first, and the Strict ledger M-4 reads comes out of the same pass.
    strict_table, strict_ledger, v1 = _audit.v13_pass(ctx)

    full_table, full_ledger = _full.full_pass(ctx)
    strict_charted = ctx.shipped.filter(
        pl.col("game_id").is_in(full_table["game_id"].to_list())
    ).sort("game_id")

    charted_drives = drive_frame(FTN_SEASONS)
    all_drives = drive_frame(PBP_SEASONS)

    full_rows = attach_drives(full_ledger, charted_drives)
    strict_rows = attach_drives(strict_ledger, all_drives)
    # Cached so the measurement can be re-read — and its reproducibility checked —
    # without re-simulating 3,900 games. `research/outputs/` is gitignored.
    full_rows.write_parquet(paths.RESEARCH_OUTPUT_DIR / FULL_LEDGER_CACHE)
    strict_rows.write_parquet(paths.RESEARCH_OUTPUT_DIR / STRICT_LEDGER_CACHE)
    full_drives = expected_clip(full_rows, drive_table(full_rows))
    strict_drives = expected_clip(strict_rows, drive_table(strict_rows))

    played = int(
        charted_drives.filter(pl.col("game_id").is_in(full_table["game_id"].to_list()))
        .select(pl.struct("game_id", "fixed_drive").n_unique())
        .item()
    )

    report_m1 = m1(full_drives, full_rows, played)
    report_m2 = m2(
        full_drives,
        slope,
        label="full",
        heading="M-2 — how often C_d would bite on the Full edition, and by how much",
    )
    report_m3 = m3(strict_charted, full_table, full_drives, slope)
    report_m4 = m2(
        strict_drives,
        slope,
        label="strict",
        heading="M-4 — the same on Strict, reported (the cap is not applied there)",
    )
    named = named_game_drives(full_rows, full_drives, slope)

    results = {
        "document": "61 — the possession-level luck cap",
        "part": "A — measurement, before the cap is built",
        "settings": {
            "random_seed": _audit.RANDOM_SEED,
            "posterior_draws": _audit.POSTERIOR_DRAWS,
            "coin_draws": _audit.COIN_DRAWS,
            "points_per_epa": slope,
            "full_seasons": list(FTN_SEASONS),
            "strict_seasons": list(PBP_SEASONS),
        },
        "gate_v1_default_off": v1,
        "m1_sharing": report_m1,
        "m2_full": report_m2,
        "m3_bucket_moves": report_m3,
        "m4_strict": report_m4,
        "named_games": named,
    }
    out = paths.RESEARCH_OUTPUT_DIR / OUTPUT_NAME
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {out}")

    print(f"\n{'=' * 76}\nV-1, RE-PRINTED AT THE END OF THE RUN\n{'=' * 76}")
    print(
        f"V-1 replay: {v1['games_matched']:,} games, max |Δ deserved margin| "
        f"{v1['max_abs_gaps']['deserved_margin']:.2e}  -> "
        f"{'PASS' if v1['pass'] else 'FAIL'}"
    )
    if not v1["pass"]:
        raise SystemExit("v1.3 moved. Stop and report.")
    print(
        "Next: part B builds the cap, but only after document 61 §4 is filled in and "
        "M-3 has been read."
    )
    if report_m3["stop_and_ask_tripped"]:
        print(
            "\nSTOP-AND-ASK TRIPPED: more than a third of the 200 bucket moves would "
            "vanish under the linear proxy. Report to the maintainer before building."
        )


if __name__ == "__main__":
    main()
