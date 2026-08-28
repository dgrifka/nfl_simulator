"""Product layer, round 4 — four readings of the deserve-to-win distribution.

Round 1's review said the distribution figure "makes sense the more you read
it" and is not intuitive at first glance. Nothing about the *numbers* is in
question: every value here is read from `dtw_games_v13.parquet`, and the draws
are `render.replay`'s, checked against the shipped summary before a pixel is
drawn. What is in question is how much scaffolding a reader needs, and that is
a question a figure answers by being looked at rather than by being argued
about.

So the four variants are cumulative, and each adds exactly one device:

* ``V1`` — one-point bins, plus the deserve-to-win callout.
* ``V2`` — three-point bins. The margin is a sum of a handful of fixed-size
  luck bars switched on or off, so only certain values are reachable and the
  histogram is genuinely spiky. Wider bins pool the reachable values without
  smoothing between them; a kernel density curve would draw margins the game
  cannot produce, which is why there is no such variant here.
* ``V3`` — the luck arrow, spanning the two rules and naming what luck moved.
* ``V4`` — the clubs' marks in place of the two coloured swatches.

    uv run python research/59_dtw_variants.py

Writes eight PNGs to ``research/outputs/``. None of it is committed —
``research/outputs/`` is gitignored, this script is the artifact, and document
42 is the record.

The two games are the ones whose readings differ most: `2018_05_GB_DET` is the
worked clear flip, and `2025_17_DET_MIN` sits inside the "too close to call"
band, where the two sides of zero are nearly the same size.
"""

from __future__ import annotations

from nfl_simulator import paths
from nfl_simulator.plots import plot_bootstrap_distribution, verdict_from_row
from nfl_simulator.render import load_sources, replay
from nfl_simulator.style import finalize
from nfl_simulator.teams import pair_colors, team_logo

GAMES = ("2018_05_GB_DET", "2025_17_DET_MIN")

# Cumulative on purpose: each row turns on one more device than the row above
# it, so a preference between two adjacent variants is a preference about that
# one device and not about a bundle of them.
VARIANTS = {
    "V1": {"bin_width": 1.0, "callout": True},
    "V2": {"bin_width": 3.0, "callout": True},
    "V3": {"bin_width": 3.0, "callout": True, "arrow": True},
    "V4": {"bin_width": 3.0, "callout": True, "arrow": True},
}


def main() -> None:
    paths.ensure_data_dirs()
    sources = load_sources()

    written = []
    for game_id in GAMES:
        row = sources.game_row(game_id)
        draws, gaps = replay(game_id, row)
        worst = max(gaps.values())
        verdict = verdict_from_row(row, draws, sources.schedule_row(game_id))
        colours = pair_colors(verdict.home_team, verdict.away_team)
        logos = {team: team_logo(team) for team in (verdict.home_team, verdict.away_team)}

        print(f"\n{'=' * 76}\n{game_id}  {verdict.headline()}  {verdict.bucket}\n{'=' * 76}")
        print(f"  replay max |Δ vs committed| {worst:.2e}")
        print(f"  {verdict.subtitle_line()}")
        print(f"  colours {colours[0]} / {colours[1]}")

        for name, options in VARIANTS.items():
            fig, _ax = plot_bootstrap_distribution(verdict, colors=colours, logos=logos, **options)
            path = finalize(fig, paths.RESEARCH_OUTPUT_DIR / f"59_{game_id}_{name}.png")
            written.append(path)
            print(f"  {name}  {options}")

    print(f"\n{'=' * 76}\n{len(written)} PNGs\n{'=' * 76}")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
