"""Figure round 6, part A — the Full edition's summary, so the figures can replay.

Ruling R-4 named two editions (document 58 §2) and round 8 shipped both in the
simulator, but only one of them has a summary on disk. `dtw_games_v13.parquet`
is Strict, and `render.replay` checks every redrawn distribution against it
before a pixel is drawn. A Full figure needs the same guarantee against its own
numbers, and there was nothing to check it against.

This writes that file. Every 2022-2025 game — the seasons FTN charting reaches —
is simulated at v1.3's exact settings with **both** hands-on-the-ball models
switched on, through `research/73`'s own machinery, and the pass is checked
against document 59's published audit before it is written:

    200 games change verdict bucket against Strict, and the median |ΔDTW| on the
    1,138 affected games is 3.85 pp.

Those are document 57 §5's numbers, re-quoted in document 59 §4. This is a
reproduction, not a new measurement: nothing here is fitted, no gate is read,
and a pass that misses either number by more than sampler noise is a stop.

Strict is re-checked too, and first: the shipped summary is joined to this pass's
games and the two editions' rows are compared, so the run says out loud that the
Full numbers are a second adjudication of the same games rather than a
replacement for the first.

    uv run python research/76_full_edition_summary.py

Writes `research/outputs/full_summary.parquet` and
`research/outputs/76_full_edition_summary.json`. Neither is committed —
`research/outputs/` is gitignored, this script is the artifact, and document 60
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

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.dropped_picks import DroppedPickModel  # noqa: E402
from nfl_simulator.ingest import FTN_SEASONS  # noqa: E402
from nfl_simulator.receiver_drops import ReceiverDropModel  # noqa: E402
from nfl_simulator.render import FULL_ARTIFACT  # noqa: E402

RESULTS = "76_full_edition_summary.json"

# Document 59 §4, which quotes document 57 §5. The tolerances are the ones
# `research/73` used for the same kind of reproduction check: a handful of games
# either way is the sampler, a different number is a different adjudication.
DOC59_BUCKET_MOVES = 200
DOC59_MEDIAN_ABS_DELTA_DTW_PP = 3.85
DOC59_AFFECTED_GAMES = 1138
MOVE_TOLERANCE = 5
DTW_TOLERANCE_PP = 0.2

# The seven games figure round 6 renders, so the run prints their two editions
# side by side and the maintainer can read the change before opening a PNG.
EXAMPLES = (
    "2018_05_GB_DET",
    "2021_14_LV_KC",
    "2016_14_NYJ_SF",
    "2025_17_DET_MIN",
    "2025_13_DEN_WAS",
    "2022_13_WAS_NYG",
    "2024_19_LAC_HOU",
)

SUMMARY_COLUMNS = (
    "game_id",
    "deserved_margin",
    "dtw_home",
    "dtw_low",
    "dtw_high",
    "n_events",
    "n_dropped_picks",
    "n_receiver_drops",
    "edition",
)


def full_pass(ctx) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The Full edition over 2022-2025, at v1.3's settings.

    `73`'s `+dp+rd` arm under its public name. The models are the shipped fits;
    the arm is `variant_pass`, unchanged, so this pass and round 7's audit are
    the same computation and can be compared line for line.
    """
    dropped_pick_model = DroppedPickModel.from_posterior(
        paths.RESEARCH_OUTPUT_DIR / _audit.TRACE_NAME,
        paths.RESEARCH_OUTPUT_DIR / _audit.SUMMARY_NAME,
    )
    receiver_drop_model = ReceiverDropModel.from_posterior(
        paths.RESEARCH_OUTPUT_DIR / _receiver_audit.TRACE_NAME,
        paths.RESEARCH_OUTPUT_DIR / _receiver_audit.SUMMARY_NAME,
    )
    return _audit.variant_pass(
        ctx, dropped_pick_model, receiver_drop_model=receiver_drop_model, label="full"
    )


def reproduction(strict: pl.DataFrame, full: pl.DataFrame) -> dict:
    """Document 59 §4's two numbers, recomputed from this pass.

    The bucket sets are compared **element-wise** — document 33's lesson, which
    this repository learned by reporting a 24-game net as a 56-game
    disagreement — and the median is taken over the games the class actually
    touches, which is the population document 57 §5 measured it on.
    """
    moved, delta_dtw, _half = _receiver_audit.bucket_moves(strict, full)
    ordered = full.sort("game_id")
    touched = (
        ordered["n_dropped_pick_events"].to_numpy() + ordered["n_receiver_drop_events"].to_numpy()
    ) > 0
    median_pp = float(np.median(np.abs(delta_dtw[touched]))) * 100
    move_gap = abs(len(moved) - DOC59_BUCKET_MOVES)
    dtw_gap = abs(median_pp - DOC59_MEDIAN_ABS_DELTA_DTW_PP)
    report = {
        "statistic": "the Full edition against Strict, as document 59 §4 publishes it",
        "games": int(full.height),
        "affected_games": int(touched.sum()),
        "affected_games_doc59": DOC59_AFFECTED_GAMES,
        "bucket_moves": len(moved),
        "bucket_moves_doc59": DOC59_BUCKET_MOVES,
        "median_abs_delta_dtw_pp": median_pp,
        "median_abs_delta_dtw_pp_doc59": DOC59_MEDIAN_ABS_DELTA_DTW_PP,
        "pass": bool(move_gap <= MOVE_TOLERANCE and dtw_gap <= DTW_TOLERANCE_PP),
    }
    print(f"\n{'=' * 76}\nREPRODUCTION — this pass against document 59 §4\n{'=' * 76}")
    print(
        f"  bucket moves {len(moved)} against {DOC59_BUCKET_MOVES} (tolerance "
        f"±{MOVE_TOLERANCE}); median |ΔDTW| on the {int(touched.sum()):,} affected games "
        f"{median_pp:.2f} pp against {DOC59_MEDIAN_ABS_DELTA_DTW_PP:.2f} (tolerance "
        f"±{DTW_TOLERANCE_PP:.1f}) -> {'PASS' if report['pass'] else 'FAIL'}"
    )
    if not report["pass"]:
        raise SystemExit(
            "the Full pass does not reproduce document 59 §4. Stop rather than write a "
            "summary the published record does not agree with."
        )
    return report


def to_summary(full: pl.DataFrame, strict: pl.DataFrame) -> pl.DataFrame:
    """The arm's table as the artifact `render.Sources` reads.

    Every *number* in the row is this pass's own. The two team abbreviations are
    joined from the Strict summary because they are a fact of the schedule
    rather than of an adjudication — `plots.verdict_from_row` reads them off the
    row it is handed, and a Full row without them could not be drawn at all. The
    join also re-states the actual margin, and the two are checked against each
    other rather than one silently winning: a mismatch means the two summaries
    are not describing the same game.
    """
    joined = full.join(
        strict.select("game_id", "home_team", "away_team", "actual_margin_strict"),
        on="game_id",
        how="inner",
    )
    if joined.height != full.height:
        raise SystemExit(
            f"{full.height - joined.height} Full games are not in the Strict summary. "
            "The two editions must adjudicate the same games."
        )
    gap = float((joined["actual_margin"] - joined["actual_margin_strict"]).abs().max())
    if gap > 0.0:
        raise SystemExit(
            f"the two editions disagree about an actual margin by {gap}. Stop and report."
        )
    return joined.select(
        "game_id",
        "home_team",
        "away_team",
        "actual_margin",
        "deserved_margin",
        "dtw_home",
        "dtw_low",
        "dtw_high",
        pl.col("n_luck_events").alias("n_events"),
        pl.col("n_dropped_pick_events").alias("n_dropped_picks"),
        pl.col("n_receiver_drop_events").alias("n_receiver_drops"),
        pl.lit("full").alias("edition"),
    ).sort("game_id")


def examples(shipped: pl.DataFrame, summary: pl.DataFrame) -> list[dict]:
    """The seven round-6 games, in both editions where both exist.

    ``shipped`` is the whole Strict summary, 2016-2025, not the charted slice:
    three of the seven predate FTN charting and the run has to say so in the
    words the figures will use, rather than reporting them missing."""
    print(f"\n{'=' * 76}\nTHE SEVEN GAMES FIGURE ROUND 6 RENDERS\n{'=' * 76}")
    rows = []
    for game_id in EXAMPLES:
        strict_row = shipped.filter(pl.col("game_id") == game_id).to_dicts()
        full_row = summary.filter(pl.col("game_id") == game_id).to_dicts()
        if not strict_row:
            print(f"  {game_id:<18} not in the Strict summary")
            continue
        base = strict_row[0]
        entry = {
            "game_id": game_id,
            "actual_margin": base["actual_margin"],
            "strict": {
                "deserved_margin": base["deserved_margin"],
                "dtw_home": base["dtw_home"],
                "bucket": _audit.bucket(base["dtw_home"], base["actual_margin"]),
            },
            "full": None,
        }
        line = (
            f"  {game_id:<18} actual {base['actual_margin']:+3.0f}   Strict DTW% "
            f"{base['dtw_home'] * 100:5.1f}  margin {base['deserved_margin']:+6.2f}"
        )
        if full_row:
            got = full_row[0]
            entry["full"] = {
                "deserved_margin": got["deserved_margin"],
                "dtw_home": got["dtw_home"],
                "bucket": _audit.bucket(got["dtw_home"], base["actual_margin"]),
                "n_events": got["n_events"],
                "n_dropped_picks": got["n_dropped_picks"],
                "n_receiver_drops": got["n_receiver_drops"],
            }
            line += (
                f"   Full DTW% {got['dtw_home'] * 100:5.1f}  margin "
                f"{got['deserved_margin']:+6.2f}  ({got['n_dropped_picks']} picks, "
                f"{got['n_receiver_drops']} catchable balls)"
            )
        else:
            line += "   Strict edition only — charting begins in 2022."
        print(line)
        rows.append(entry)
    return rows


def main() -> None:
    ctx = _audit.load_context()
    print(f"points_per_epa = {ctx.slope:.10f}  (read from the game table, not refitted)")

    full, _ledger = full_pass(ctx)
    strict = ctx.shipped.filter(pl.col("game_id").is_in(full["game_id"].to_list())).sort("game_id")
    print(
        f"\n  Strict on file: {ctx.shipped.height:,} games 2016-2025, of which "
        f"{strict.height:,} are in {FTN_SEASONS[0]}-{FTN_SEASONS[-1]} and have a Full edition"
    )

    report = reproduction(strict, full)
    summary = to_summary(
        full, strict.with_columns(pl.col("actual_margin").alias("actual_margin_strict"))
    )

    labels = {
        str(row["variant"]): int(row["count"]) for row in full["variant"].value_counts().to_dicts()
    }
    print(f"\n{'=' * 76}\nWHAT THE FULL LEDGERS ACTUALLY HOLD\n{'=' * 76}")
    for variant, count in sorted(labels.items(), key=lambda pair: -pair[1]):
        print(f"  {count:5,d} games' ledgers are labelled {variant!r}")
    print(
        "  The edition is what was switched on, not what the charting happened to find:\n"
        "  a Full-edition game whose charter called no interceptable throw carries a\n"
        "  `strict+rd` ledger and is still a Full-edition figure."
    )

    example_rows = examples(ctx.shipped, summary)

    out = paths.RESEARCH_OUTPUT_DIR / FULL_ARTIFACT
    summary.write_parquet(out)
    print(f"\nwrote {out}  ({summary.height:,} rows, columns {list(summary.columns)})")

    with (paths.RESEARCH_OUTPUT_DIR / RESULTS).open("w") as handle:
        json.dump(
            {
                "document": "60 — figure round 6, part A",
                "settings": {
                    "random_seed": _audit.RANDOM_SEED,
                    "posterior_draws": _audit.POSTERIOR_DRAWS,
                    "coin_draws": _audit.COIN_DRAWS,
                    "points_per_epa": ctx.slope,
                    "seasons": list(FTN_SEASONS),
                },
                "artifact": FULL_ARTIFACT,
                "columns": list(SUMMARY_COLUMNS),
                "rows": int(summary.height),
                "reproduction_against_doc59": report,
                "variant_labels": labels,
                "examples": example_rows,
            },
            handle,
            indent=2,
            default=float,
        )
    print("Next: research/58_brand_figures.py renders both editions.")


if __name__ == "__main__":
    main()
