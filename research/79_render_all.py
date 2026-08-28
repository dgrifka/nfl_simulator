"""Product layer, figure round 9 Part C — render every game and read the tails.

Nine chosen games cannot show what 3,900 games do. This renders the four share
figures for every row of `dtw_games_v13.parquet` (Strict, 2,761 games) and every
row of `full_summary.parquet` (Full, 1,139), measures the handful of statistics a
layout defect shows up in, and writes them to one JSON so the next round can set
its floors from measured values rather than guessed ones.

    uv run python research/79_render_all.py

**This is a read, not a fix.** Nothing here changes a number and nothing here
changes a rule. Every game passes `render.replay`'s gate before a pixel is drawn
— a redraw that lands more than 1e-9 from its published row is a stop, exactly
as driver 58 treats it — and what comes out is a list of games worth looking at.
Document 63 is the record of what the look found.

Writes ~15,600 PNGs to ``research/outputs/all/<edition>/`` plus
``research/outputs/79_render_all.json``. None of it is committed;
``research/outputs/`` is gitignored and this script is the artifact.

Parallel across processes because it is 3,900 replays and 15,600 figures: one
core does it in about two hours, and the work is per-game independent. The
replay gate is what makes that safe to say — a worker that drew a different
bootstrap than the parent would fail its own game's gate rather than pass a
different adjudication back.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.plots import (  # noqa: E402
    ARROW_FLOOR,
    group_rows,
    luck_bars,
    verdict_from_row,
)
from nfl_simulator.render import (  # noqa: E402
    DTW_FIGURE,
    SUFFIXES,
    counterpart_verdict,
    kick_distances,
    kicker_names,
    load_sources,
    passer_names,
    prepare_rows,
    receiver_names,
    render_game,
    replay,
)
from nfl_simulator.teams import pair_colors, team_logo  # noqa: E402

RESULTS = "79_render_all.json"

# One JSON object per finished game-edition, appended as it lands. The pass is
# half an hour of work whose only product until the very end was a number in
# memory, and a first run was killed at 89% and lost every measurement it had
# made. The checkpoint makes the run resumable: a re-run reads what is here,
# skips those game-editions, and renders the rest.
CHECKPOINT = "79_render_all.jsonl"

# The gate driver 58 applies per game, quoted rather than loosened: a figure is
# only allowed to be drawn from a redraw that lands on the published row.
REPLAY_TOLERANCE = 1e-9

# One worker per core less two, so the machine stays usable while it runs and
# the parent has room to collect. Each worker loads the artifacts once —
# `load_sources` is cached per process — and then costs about 1.7 s a game.
WORKERS = max(1, (os.cpu_count() or 4) - 2)


def _dtw_layout(verdict, colours, logos) -> dict:
    """Draw the distribution once more and measure what a defect shows up in.

    Three of the round's statistics are decided by the renderer against a live
    canvas rather than by arithmetic on the row — whether the two rule labels
    stacked, whether the arrow drew its span, whether a corner label had its
    words taken off. There is no way to read them off a written PNG and
    ``render_game`` does not hand its figure back, so the figure is drawn a
    second time here, with the same keyword arguments the share image uses so
    that what is measured is what was written.
    """
    from nfl_simulator.plots import plot_bootstrap_distribution

    fig, ax = plot_bootstrap_distribution(
        verdict, colors=colours, logos=logos, coverage=False, **DTW_FIGURE
    )
    fig.canvas.draw()

    # Matched on the gid `_rule` stamps, not on the words: the subtitle opens
    # `Actual: LAC 12 - HOU 32` and a text match picks it up as a third rule.
    rules = [t for t in ax.texts if t.get_gid() == "rule-label"]
    stacked, gap_px = False, None
    if len(rules) == 2:
        boxes = [t.get_window_extent() for t in rules]
        # `bool(...)`, not the bare comparison: the bbox coordinates are numpy
        # floats, so `>` returns a `numpy.bool_`, which `json.dumps` refuses.
        stacked = bool(abs(boxes[0].y0 - boxes[1].y0) > 1.0)
        # How close the two boxes came horizontally. On a game that did not
        # stack this is the room the rule still had; on one that did it is
        # what the second row bought.
        gap_px = float(max(boxes[0].x0, boxes[1].x0) - min(boxes[0].x1, boxes[1].x1))

    wanted = {f"{verdict.home_team} wins", f"{verdict.away_team} wins"}
    corners_left = sum(1 for t in ax.texts if t.get_text() in wanted)
    spans = [t for t in ax.texts if isinstance(getattr(t, "arrow_patch", None), FancyArrowPatch)]

    plt.close(fig)
    return {
        "rules_stacked": stacked,
        "rule_gap_px": gap_px,
        "corner_cleared": bool(corners_left < len(wanted)),
        "arrow_drawn": bool(spans),
    }


def measure(task: tuple[str, str, str]) -> dict:
    """One game in one edition: gate it, render it, and measure it."""
    game_id, edition, out_dir = task
    sources = load_sources()
    row = sources.game_row(game_id, edition=edition)
    schedule = sources.schedule_row(game_id)

    result, gaps = replay(game_id, row, schedule, edition=edition)
    worst = max(gaps.values())
    if worst > REPLAY_TOLERANCE:
        # Not raised here: the parent stops the run, so one bad game reports
        # itself rather than killing the pool mid-write.
        return {"game_id": game_id, "edition": edition, "replay_worst": worst, "stop": True}

    verdict = verdict_from_row(
        row,
        result.margin_draws,
        schedule,
        edition=edition,
        counterpart=counterpart_verdict(sources, game_id, edition, schedule),
    )
    ledger = (
        result.ledger.to_frame()
        if edition == "full"
        else sources.ledger.filter(pl.col("game_id") == game_id).drop("game_id")
    )
    rows = prepare_rows(
        ledger,
        verdict,
        kick_distances(game_id),
        kicker_names(game_id),
        passer_names(game_id),
        receiver_names(game_id),
    )
    bars = luck_bars(rows, points_per_epa=sources.slope)
    grouped = group_rows(bars)
    colours = pair_colors(verdict.home_team, verdict.away_team)
    logos = {team: team_logo(team) for team in (verdict.home_team, verdict.away_team)}

    layout = _dtw_layout(verdict, colours, logos)
    written = render_game(game_id, Path(out_dir), edition=edition)
    plt.close("all")

    labels = [bar.label for bar in grouped]
    longest = max(labels, key=len) if labels else ""
    return {
        "game_id": game_id,
        "edition": edition,
        "stop": False,
        "replay_worst": worst,
        "n_files": len(written),
        # --- the six statistics document 63 summarises -----------------
        "n_events": len(rows),
        # The waterfall draws every grouped bar plus its two anchor rows.
        "n_waterfall_rows": len(grouped) + 2,
        "longest_label": len(longest),
        "actual_margin": abs(float(verdict.actual_margin)),
        "deserved_margin": abs(float(verdict.deserved_margin)),
        "events_under_a_point": sum(1 for bar in bars if abs(bar.points) < 1.0),
        # --- and what the canvas decided -------------------------------
        **layout,
        "longest_label_text": longest,
        "is_degenerate": bool(verdict.is_degenerate),
        "bucket": verdict.bucket,
    }


def _summarise(records: list[dict], edition: str) -> dict:
    """Max, p99 and median of each statistic, for one edition."""
    rows = [r for r in records if r["edition"] == edition]
    frame = pl.DataFrame(rows)
    stats = (
        "n_events",
        "n_waterfall_rows",
        "longest_label",
        "actual_margin",
        "deserved_margin",
        "events_under_a_point",
    )
    return {
        "n_games": len(rows),
        **{
            name: {
                "max": float(frame[name].max()),
                "p99": float(frame[name].quantile(0.99)),
                "median": float(frame[name].median()),
            }
            for name in stats
        },
        "n_stacked": int(frame["rules_stacked"].sum()),
        "n_corner_cleared": int(frame["corner_cleared"].sum()),
        "n_arrow_drawn": int(frame["arrow_drawn"].sum()),
        "n_degenerate": int(frame["is_degenerate"].sum()),
    }


def _load_checkpoint(path: Path) -> list[dict]:
    """Every record a previous run finished, or an empty list."""
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    paths.ensure_data_dirs()
    sources = load_sources()

    root = paths.RESEARCH_OUTPUT_DIR / "all"
    tasks = []
    for edition, frame in (("strict", sources.games), ("full", sources.full)):
        out_dir = root / edition
        out_dir.mkdir(parents=True, exist_ok=True)
        tasks.extend((gid, edition, str(out_dir)) for gid in frame["game_id"].to_list())
    total = len(tasks)

    checkpoint = paths.RESEARCH_OUTPUT_DIR / CHECKPOINT
    records = _load_checkpoint(checkpoint)
    already = {(record["game_id"], record["edition"]) for record in records}
    tasks = [task for task in tasks if (task[0], task[1]) not in already]

    print(
        f"{'=' * 76}\nRENDER ALL — {total} game-editions on {WORKERS} workers\n{'=' * 76}",
        flush=True,
    )
    for edition, frame in (("strict", sources.games), ("full", sources.full)):
        print(f"  {edition:<7} {frame.height:>5} games", flush=True)
    if already:
        print(f"  resuming: {len(already)} done, {len(tasks)} to render", flush=True)

    start = time.time()
    with (
        ProcessPoolExecutor(max_workers=WORKERS) as pool,
        checkpoint.open("a") as sink,
    ):
        for done, record in enumerate(pool.map(measure, tasks, chunksize=8), start=1):
            if record.get("stop"):
                raise SystemExit(
                    f"{record['game_id']} ({record['edition']}) replayed "
                    f"{record['replay_worst']:.2e} from its published row, over "
                    f"{REPLAY_TOLERANCE:.0e}. Stop and ask before drawing anything."
                )
            records.append(record)
            # Flushed per record: the run is long enough that a kill halfway
            # through must not cost the half it had already measured.
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            if done % 100 == 0 or done == len(tasks):
                rate = done / (time.time() - start)
                print(
                    f"  {done:>5}/{len(tasks)}  {rate:5.1f} game/s  "
                    f"eta {(len(tasks) - done) / rate / 60:5.1f} min",
                    flush=True,
                )

    elapsed = time.time() - start
    written = sum(record["n_files"] for record in records)
    expected = total * len(SUFFIXES)
    print(f"\n{'=' * 76}\nwritten: {written} files\n{'=' * 76}", flush=True)
    print(
        f"  expected {expected} ({len(SUFFIXES)} x {total})   "
        f"{'ok' if written == expected else 'CHECK'}",
        flush=True,
    )
    print(f"  elapsed  {elapsed:.0f} s ({elapsed / 60:.1f} min), this pass", flush=True)
    print(f"  worst replay gap across every game: {max(r['replay_worst'] for r in records):.2e}")

    summary = {edition: _summarise(records, edition) for edition in ("strict", "full")}
    print(f"\n{'=' * 76}\nDISTRIBUTIONS\n{'=' * 76}")
    for edition, block in summary.items():
        print(f"\n  {edition} ({block['n_games']} games)")
        for name, values in block.items():
            if isinstance(values, dict):
                print(
                    f"    {name:<22} max {values['max']:>8.2f}  "
                    f"p99 {values['p99']:>8.2f}  median {values['median']:>8.2f}"
                )
            elif name != "n_games":
                print(f"    {name:<22} {values}")

    with (paths.RESEARCH_OUTPUT_DIR / RESULTS).open("w") as handle:
        json.dump(
            {
                "workers": WORKERS,
                "elapsed_seconds": elapsed,
                "files_written": written,
                "replay_tolerance": REPLAY_TOLERANCE,
                "arrow_floor": ARROW_FLOOR,
                "summary": summary,
                "games": records,
            },
            handle,
            indent=2,
        )
    print(f"\n  {paths.RESEARCH_OUTPUT_DIR / RESULTS}")


if __name__ == "__main__":
    main()
