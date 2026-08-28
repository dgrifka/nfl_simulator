"""Product layer, round 3 — the brand-matched per-game PNGs.

Nothing is fitted and no number is new. Every figure this writes is a
presentation of `dtw_games_v13.parquet`, `dtw_ledger_v13.parquet` and
`26_overtime_games.parquet`, in the house style shared with the baseball
simulator: cream surface, one title grammar, the nflverse credit stamped on
every image.

The one thing recomputed is each game's bootstrap draws — the shipped summary
keeps the summary, not the 160,000 draws — and `render.replay` checks the
redraw against the published number before anything is drawn. A disagreement
is a `SystemExit`.

    uv run python research/58_brand_figures.py

Writes twenty share PNGs to ``research/outputs/`` — four per game — plus one
``_dtw_article.png`` for each of the two overtime games, plus
``research/outputs/58_brand_figures.json``. None of it is committed —
``research/outputs/`` is gitignored, this script is the artifact, and document
41 is the record.

The seven games are the round's, chosen and not typical. The first three
predate FTN charting and have only a Strict edition; the last four have both,
and are rendered in the headline edition — Full — with the two 2025 games
rendered a second time in Strict so the two can be compared side by side.

* ``2018_05_GB_DET`` — the worked clear flip (documents 33, 37, 38).
* ``2021_14_LV_KC`` — degenerate: every re-flip lands the same way.
* ``2016_14_NYJ_SF`` — overtime, the largest per-game toss move in the window.
* ``2025_17_DET_MIN`` — inside the band, "too close to call".
* ``2025_13_DEN_WAS`` — overtime under the 2025 rulebook, and the game whose
  two rule labels collided before document 37 §7a's fix.
* ``2022_13_WAS_NYG`` — a tie the Full edition hardens: eight dropped picks,
  seven of them Washington throws that escaped (document 50).
* ``2024_19_LAC_HOU`` — two dropped touchdowns near 9 EPA apiece, and the
  largest edition disagreement of the seven.
"""

from __future__ import annotations

import json

from nfl_simulator import paths
from nfl_simulator.plots import luck_bars, plain_label, verdict_from_row
from nfl_simulator.render import (
    ARTICLE_SUFFIX,
    SUFFIXES,
    counterpart_verdict,
    figure_filename,
    kick_distances,
    kicker_names,
    load_sources,
    prepare_rows,
    render_game,
    replay,
)
from nfl_simulator.teams import load_team_table, pair_colors, team_logo

RESULTS = "58_brand_figures.json"

# Each game and the editions it is rendered in, headline edition first. The two
# 2025 games carry a second, Strict render so the maintainer can put the two side by
# side; the other five are rendered once, in whichever edition is their
# headline.
EXAMPLES = (
    ("2018_05_GB_DET", ("strict",)),
    ("2021_14_LV_KC", ("strict",)),
    ("2016_14_NYJ_SF", ("strict",)),
    ("2025_17_DET_MIN", ("full", "strict")),
    ("2025_13_DEN_WAS", ("full", "strict")),
    ("2022_13_WAS_NYG", ("full",)),
    ("2024_19_LAC_HOU", ("full",)),
)

# nflverse's 32 clubs plus the four relocation aliases. A different number means
# the table changed shape and the colour lookup should be re-read before it is
# trusted, not after fifteen figures have been drawn in the grey fallback.
EXPECTED_TEAM_ROWS = 36


def main() -> None:
    paths.ensure_data_dirs()

    table = load_team_table()
    print(f"{'=' * 76}\nTEAM TABLE\n{'=' * 76}")
    print(f"  {table.height} rows (expected {EXPECTED_TEAM_ROWS})")
    if table.height != EXPECTED_TEAM_ROWS:
        raise SystemExit(
            f"nflverse's team table has {table.height} rows, not {EXPECTED_TEAM_ROWS}. "
            "Stop and re-read it before drawing anything."
        )

    sources = load_sources()
    print(f"  points_per_epa = {sources.slope:.10f}  (read from metadata, not refitted)")

    print(
        f"\n{'=' * 76}\nREPLAY — every redrawn distribution must belong to its summary\n{'=' * 76}"
    )
    records = []
    for game_id, editions in EXAMPLES:
        for edition in editions:
            schedule = sources.schedule_row(game_id)
            row = sources.game_row(game_id, edition=edition)
            result, gaps = replay(game_id, row, schedule, edition=edition)
            draws = result.margin_draws
            worst = max(gaps.values())
            print(
                f"  {game_id:<18} {edition:<7} max |Δ vs committed| {worst:.2e}  "
                f"{'ok' if worst == 0 else 'check'}"
            )

            verdict = verdict_from_row(
                row,
                draws,
                schedule,
                edition=edition,
                counterpart=counterpart_verdict(sources, game_id, edition, schedule),
            )
            ledger = (
                result.ledger.to_frame()
                if edition == "full"
                else sources.ledger.filter(sources.ledger["game_id"] == game_id).drop("game_id")
            )
            rows = prepare_rows(ledger, verdict, kick_distances(game_id), kicker_names(game_id))
            bars = luck_bars(rows, points_per_epa=sources.slope)
            home_colour, away_colour = pair_colors(verdict.home_team, verdict.away_team)
            records.append(
                {
                    "game_id": game_id,
                    "edition": edition,
                    "verdict": verdict,
                    "bars": bars,
                    "colours": (home_colour, away_colour),
                    "logos": {
                        team: team_logo(team) is not None
                        for team in (verdict.home_team, verdict.away_team)
                    },
                    "replay_worst": worst,
                    "n_events": len(rows),
                    "labels": [plain_label(r) for r in rows],
                }
            )

    print(f"\n{'=' * 76}\nFIGURES\n{'=' * 76}")
    written = []
    for record in records:
        verdict = record["verdict"]
        paths_out = render_game(record["game_id"], article=True, edition=record["edition"])
        written.extend(paths_out)
        home, away = record["colours"]
        marks = ", ".join(f"{team} {'✓' if ok else '—'}" for team, ok in record["logos"].items())
        print(
            f"\n  {record['game_id']:<18} {verdict.edition_name:<7} "
            f"{verdict.headline():<22} {verdict.bucket}"
        )
        print(f"    {verdict.subtitle_line()}")
        print(f"    {verdict.edition_note() or 'no other edition'}")
        print(f"    colours {home} / {away}   logos {marks}   overtime {verdict.went_to_overtime}")
        for path in paths_out:
            print(f"    {path.name}")
        for bar in record["bars"][:3]:
            print(f"      {bar.points:+6.2f}  {bar.label}")

    share = [path for path in written if ARTICLE_SUFFIX not in path.name]
    article = [path for path in written if ARTICLE_SUFFIX in path.name]
    print(f"\n{'=' * 76}\n{len(share)} share PNGs + {len(article)} article PNGs\n{'=' * 76}")
    for path in written:
        print(f"  {path}")

    with (paths.RESEARCH_OUTPUT_DIR / RESULTS).open("w") as handle:
        json.dump(
            {
                "points_per_epa": sources.slope,
                "team_table_rows": table.height,
                "games": [
                    {
                        "game_id": record["game_id"],
                        "edition": record["edition"],
                        "edition_note": record["verdict"].edition_note(),
                        "headline": record["verdict"].headline(),
                        "bucket": record["verdict"].bucket,
                        "subtitle": record["verdict"].subtitle_line(),
                        "deserved_line": record["verdict"].deserved_line(),
                        "overtime": record["verdict"].went_to_overtime,
                        "colours": list(record["colours"]),
                        "logos": record["logos"],
                        "replay_worst": record["replay_worst"],
                        "n_events": record["n_events"],
                        "n_bars": len(record["bars"]),
                        "labels": record["labels"],
                        "files": [
                            figure_filename(record["verdict"], suffix)
                            for suffix in (
                                *SUFFIXES,
                                *((ARTICLE_SUFFIX,) if record["verdict"].went_to_overtime else ()),
                            )
                        ],
                    }
                    for record in records
                ],
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
