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
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.plots import (  # noqa: E402
    ARROW_FLOOR,
    DRAW_FLOOR_SHARE,
    group_rows,
    luck_bars,
    plot_luck_ledger,
    verdict_from_row,
    waterfall_span,
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
from nfl_simulator.style import edition_stamp, stamp_box  # noqa: E402
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

# Anything darker than this inside the stamp's box is somebody else's ink. The
# stamp is painted at 140 grey on a 249 cream and the title is #1A1A1A at 26, so
# a threshold between the two counts the overlap without having to map a title's
# figure coordinates through `bbox_inches="tight"`'s crop.
FOREIGN_INK = 120

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
    stacked, gap_px, rule_rows = False, None, len(rules)
    if len(rules) == 2:
        boxes = [t.get_window_extent() for t in rules]
        # Round 10 Part C: how many rows the band actually put them on, which is
        # two on every figure now rather than on the 93-95% that collided.
        rule_rows = len({round(box.y0, 1) for box in boxes})
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
        "rule_rows": int(rule_rows),
        "corner_cleared": bool(corners_left < len(wanted)),
        "arrow_drawn": bool(spans),
    }


def _waterfall_layout(verdict, rows, slope, colours, logos) -> dict:
    """Draw the waterfall once more and measure what round 10 Part D fixed.

    The same reason :func:`_dtw_layout` exists: whether the dashed zero rule
    crosses a corner label, and whether the rotated arrow sentence reaches one,
    are decided against a live canvas and cannot be read off a written PNG.

    A **strike** is the rule crossing a corner label that is not shielded — the
    defect document 63 recorded on four games. A shielded label the rule passes
    behind is not a strike: the cream box is opaque and above it, which is the
    settled fix. An **overlap** is the rotated sentence reaching a corner label,
    which `_lower_under_corners` is supposed to make impossible.
    """
    fig, ax = plot_luck_ledger(verdict, rows, points_per_epa=slope, colors=colours, logos=logos)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    wanted = {f"{verdict.home_team} wins", f"{verdict.away_team} wins"}
    corners = [t for t in ax.texts if t.get_text() in wanted]
    rule_x = ax.transData.transform((0.0, 0.0))[0]
    strikes = 0
    for corner in corners:
        box = corner.get_window_extent(renderer)
        if not (box.x0 <= rule_x <= box.x1):
            continue
        shield = corner.get_bbox_patch()
        if shield is None or shield.get_facecolor()[3] < 1.0:
            strikes += 1

    overlaps = 0
    sentences = [t for t in ax.texts if t.get_text().startswith("luck moved")]
    for sentence in sentences:
        span = sentence.get_window_extent(renderer)
        overlaps += sum(1 for c in corners if span.overlaps(c.get_window_extent(renderer)))

    plt.close(fig)
    return {"corner_strikes": strikes, "sentence_overlaps": overlaps}


def _stamp_overlap_px(path: Path, edition: str) -> int:
    """Foreign ink inside the credit stamp's box on a written PNG.

    Document 63 measured the title running under the stamp on 84-89% of
    distribution figures by a median 19 px. This is the same measurement taken
    on the shipped image rather than on a re-drawn figure.
    """
    from PIL import Image

    pixels = np.asarray(Image.open(path).convert("RGB"), dtype=float).mean(axis=2)
    left, top, right, bottom = stamp_box((pixels.shape[1], pixels.shape[0]), edition_stamp(edition))
    return int((pixels[top:bottom, left:right] < FOREIGN_INK).sum())


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
    span = waterfall_span(verdict)
    floor = span * DRAW_FLOOR_SHARE
    grouped = group_rows(bars, span=span)
    colours = pair_colors(verdict.home_team, verdict.away_team)
    logos = {team: team_logo(team) for team in (verdict.home_team, verdict.away_team)}

    layout = _dtw_layout(verdict, colours, logos)
    layout |= _waterfall_layout(verdict, rows, sources.slope, colours, logos)
    written = render_game(game_id, Path(out_dir), edition=edition)
    plt.close("all")

    dtw_png = next((path for path in written if path.name.endswith("_dtw.png")), None)
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
        # --- round 10's five, one per part -----------------------------
        "stamp_overlap_px": _stamp_overlap_px(dtw_png, edition) if dtw_png else None,
        "rows_named_events_under": sum(1 for bar in grouped if "events under" in bar.label),
        "anonymous_rows": sum(1 for bar in grouped if bar.team is None),
        # --- round 11's four, and the axis they are measured against ------
        "waterfall_span": span,
        "draw_floor": floor,
        "rows_named_one_small_event": sum(
            1 for bar in grouped if bar.label.startswith("1 small event")
        ),
        **_under_floor(grouped, floor),
        # --- and what the canvas decided -------------------------------
        **layout,
        "longest_label_text": longest,
        "is_degenerate": bool(verdict.is_degenerate),
        "bucket": verdict.bucket,
    }


def _under_floor(grouped, floor: float) -> dict:
    """The rows that still draw nothing, sorted into what the round 11 rules allow.

    Two classes are allowed and one is not. A **lone event** is kept whatever it
    is worth, because `1 small event (SEA)` is the same invisible bar with the
    event's words taken off it — rule 2. A **club heap of two or more** that
    cancels to under the floor is kept because there is nowhere left to fold it
    — rule 3, and this is the residue the round reports rather than hides.

    **Anything else** is a bug rather than a residue: a component fold under the
    floor should have been absorbed into its club's heap before it ever reached
    a row, so the third count is pre-registered at zero.
    """
    under = [bar for bar in grouped if abs(bar.points) < floor]
    heaps = [
        bar for bar in under if bar.n_events > 1 and bar.component is None and bar.play_id is None
    ]
    lone = [bar for bar in under if bar.n_events == 1]
    return {
        "rows_under_draw_floor": len(under),
        "rows_under_floor_lone_event": len(lone),
        "rows_under_floor_cancelled_heap": len(heaps),
        "rows_under_floor_other": len(under) - len(lone) - len(heaps),
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
        # --- round 10's pre-registered checks, per edition --------------
        "n_two_rule_rows": int((frame["rule_rows"] == 2).sum()),
        "n_stamp_overlaps": int((frame["stamp_overlap_px"] > 0).sum()),
        "stamp_overlap_px_total": int(frame["stamp_overlap_px"].sum()),
        "n_corner_strikes": int(frame["corner_strikes"].sum()),
        "n_sentence_overlaps": int(frame["sentence_overlaps"].sum()),
        "n_rows_under_draw_floor": int(frame["rows_under_draw_floor"].sum()),
        "n_games_with_a_row_under_the_floor": int((frame["rows_under_draw_floor"] > 0).sum()),
        "n_rows_under_floor_lone_event": int(frame["rows_under_floor_lone_event"].sum()),
        "n_rows_under_floor_cancelled_heap": int(frame["rows_under_floor_cancelled_heap"].sum()),
        "n_rows_under_floor_other": int(frame["rows_under_floor_other"].sum()),
        "n_rows_named_one_small_event": int(frame["rows_named_one_small_event"].sum()),
        "draw_floor_median": float(frame["draw_floor"].median()),
        "draw_floor_max": float(frame["draw_floor"].max()),
        "n_rows_named_events_under": int(frame["rows_named_events_under"].sum()),
        "n_anonymous_rows": int(frame["anonymous_rows"].sum()),
    }


# The sorts document 63 §4 reads the tail by, and the direction each extreme
# lies in. The fifth of round 9's six — the smallest gap between the two rule
# labels — is dropped: round 10 put the labels on two rows, so there is no gap
# left to be small.
PICK_SORTS = (
    ("longest_label", True),
    ("n_waterfall_rows", True),
    ("actual_margin", True),
    ("deserved_margin", False),
    ("events_under_a_point", True),
)
PICKS_PER_SORT = 10
PICK_FIGURES = ("dtw", "waterfall")
PICK_LIST = "79_pick_list.json"


def _pick_lists(records: list[dict], root: Path) -> list[dict]:
    """The PNGs the tail read opens: the worst ten of each sort, both editions.

    One game is usually extreme on more than one sort, so the lists overlap
    heavily and the deduplicated result is much smaller than five sorts times
    ten times two editions times two figures. Every sort a PNG was picked by is
    kept on its entry, because "why is this one here" is the first question the
    read asks of a figure that looks fine.
    """
    picked: dict[tuple[str, str, str], dict] = {}
    for edition in ("strict", "full"):
        rows = [r for r in records if r["edition"] == edition]
        for name, biggest_first in PICK_SORTS:
            worst = sorted(rows, key=lambda r: r[name], reverse=biggest_first)
            for record in worst[:PICKS_PER_SORT]:
                for figure in PICK_FIGURES:
                    key = (record["game_id"], edition, figure)
                    entry = picked.setdefault(
                        key,
                        {
                            "game_id": record["game_id"],
                            "edition": edition,
                            "figure": figure,
                            "sorts": [],
                            "path": _figure_path(root, record["game_id"], edition, figure),
                        },
                    )
                    entry["sorts"].append(f"{name}={record[name]}")
    return sorted(picked.values(), key=lambda e: (e["edition"], e["game_id"], e["figure"]))


def _figure_path(root: Path, game_id: str, edition: str, figure: str) -> str | None:
    """Where this game-edition's figure landed, or None if it is not on disk.

    The filename carries the scoreboard and the two DTW shares, so it cannot be
    spelled from the game id alone — it is matched rather than constructed.
    """
    matches = sorted((root / edition).glob(f"{game_id}_*_{edition}_{figure}.png"))
    return str(matches[0]) if matches else None


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

    on_disk = sum(1 for _ in root.rglob("*.png"))
    print(f"  files on disk: {on_disk} (expected {expected})", flush=True)

    summary = {edition: _summarise(records, edition) for edition in ("strict", "full")}
    checks = {
        "files_on_disk": (on_disk, expected),
        "worst_replay_gap": (max(r["replay_worst"] for r in records), 0.0),
        "title_stamp_overlaps": (sum(s["n_stamp_overlaps"] for s in summary.values()), 0),
        "corner_strikes": (sum(s["n_corner_strikes"] for s in summary.values()), 0),
        "sentence_overlaps": (sum(s["n_sentence_overlaps"] for s in summary.values()), 0),
        "rows_named_one_small_event": (
            sum(s["n_rows_named_one_small_event"] for s in summary.values()),
            0,
        ),
        "rows_under_floor_other": (
            sum(s["n_rows_under_floor_other"] for s in summary.values()),
            0,
        ),
        "rows_named_events_under": (
            sum(s["n_rows_named_events_under"] for s in summary.values()),
            0,
        ),
        "anonymous_rows": (sum(s["n_anonymous_rows"] for s in summary.values()), 0),
        "figures_on_two_rule_rows": (
            sum(s["n_two_rule_rows"] for s in summary.values()),
            total,
        ),
    }
    print(f"\n{'=' * 76}\nROUND 11 — PRE-REGISTERED CHECKS\n{'=' * 76}")
    for name, (got, want) in checks.items():
        verdict_word = "ok" if got == want else "MISS"
        print(f"  {name:<28} {got!s:>12}  expected {want!s:<12} {verdict_word}", flush=True)

    # Reported, not pre-registered to a number: the residue rules 2 and 3 allow,
    # and the widest waterfall the corpus draws.
    print(f"\n{'=' * 76}\nROUND 11 — REPORTED\n{'=' * 76}")
    for name in (
        "n_rows_under_draw_floor",
        "n_rows_under_floor_lone_event",
        "n_rows_under_floor_cancelled_heap",
        "n_rows_under_floor_other",
        "n_games_with_a_row_under_the_floor",
    ):
        by_edition = "  ".join(f"{e} {summary[e][name]}" for e in ("strict", "full"))
        print(f"  {name:<38} {sum(summary[e][name] for e in summary):>6}   ({by_edition})")
    print(
        "  waterfall rows, max                    "
        f"{max(summary[e]['n_waterfall_rows']['max'] for e in summary):>6.0f}   "
        + "  ".join(f"{e} {summary[e]['n_waterfall_rows']['max']:.0f}" for e in ("strict", "full"))
    )
    for edition in ("strict", "full"):
        print(
            f"  draw floor, {edition:<7} median "
            f"{summary[edition]['draw_floor_median']:.3f} pt   "
            f"max {summary[edition]['draw_floor_max']:.3f} pt"
        )

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
                "draw_floor_share": DRAW_FLOOR_SHARE,
                "files_on_disk": on_disk,
                "checks": {
                    name: {"got": got, "want": want} for name, (got, want) in checks.items()
                },
                "summary": summary,
                "games": records,
            },
            handle,
            indent=2,
        )
    picks = _pick_lists(records, root)
    missing = [pick for pick in picks if pick["path"] is None]
    print(f"\n{'=' * 76}\nTAIL READ — THE PICK LIST\n{'=' * 76}")
    print(f"  {len(picks)} distinct PNGs after deduplication, {len(missing)} not on disk")
    with (paths.RESEARCH_OUTPUT_DIR / PICK_LIST).open("w") as handle:
        json.dump({"n_picks": len(picks), "picks": picks}, handle, indent=2)
    print(f"  {paths.RESEARCH_OUTPUT_DIR / PICK_LIST}")

    print(f"\n  {paths.RESEARCH_OUTPUT_DIR / RESULTS}")


if __name__ == "__main__":
    main()
