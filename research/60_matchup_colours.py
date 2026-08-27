"""Product layer, round 4 — every matchup in the league, checked for every reader.

Round 1's clash rule measured Euclidean distance in RGB. That rule is cheap and
blind: it kept the Jets' ``#003F2D`` beside the 49ers' ``#AA0000`` because the
two are 0.42 apart — and 5.2 apart in OKLab for a reader with protanopia, well
under the 6 the `dataviz` skill calls a floor. `2016_14_NYJ_SF` shipped with two
bars a colourblind reader sees as one, and nothing in the pipeline said so.

This runs `teams.resolve_pair` over all 32 x 31 = 992 **ordered** matchups —
ordered because home and away are not symmetric: the rule tries the away team's
colours before the home team's, so a pair can resolve one way at one venue and
another way at the other. For each it records the two colours drawn, which rung
of the ladder chose them, and the worst of the four readings.

    uv run python research/60_matchup_colours.py

Writes ``research/outputs/60_matchup_colours.parquet`` and a 32 x 32 swatch grid
PNG. ``research/outputs/`` is gitignored — this script is the artifact and
document 42 is the record.

``unresolved`` must be zero. If it is not, the ladder is short a rung and the
pairs are printed rather than drawn.
"""

from __future__ import annotations

import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import polars as pl

from nfl_simulator import paths
from nfl_simulator.style import (
    CVD_FLOOR,
    CVD_KINDS,
    CVD_TARGET,
    NORMAL_FLOOR,
    PALETTE,
    contrast_ratio,
    finalize,
    heading_font,
    rc_style,
    separations,
)
from nfl_simulator.teams import FALLBACKS, load_team_table, resolve_pair

RESULTS = "60_matchup_colours.parquet"
GRID = "60_matchup_colours.png"

# The four relocation aliases play the same clubs under other names, and a
# 36 x 35 grid would carry each of them twice. The sweep is over the league as
# it stands; `resolve_pair` treats an alias like any other abbreviation.
ALIASES = ("OAK", "SD", "STL", "LA")


def main() -> None:
    paths.ensure_data_dirs()
    table = load_team_table()
    clubs = sorted(row["team_abbr"] for row in table.to_dicts() if row["team_abbr"] not in ALIASES)
    print(f"{'=' * 76}\nMATCHUP COLOURS\n{'=' * 76}")
    print(f"  {len(clubs)} clubs, {len(clubs) * (len(clubs) - 1)} ordered matchups")

    records = []
    for home_team, away_team in itertools.permutations(clubs, 2):
        home, away, rule = resolve_pair(home_team, away_team)
        readings = separations(home, away)
        records.append(
            {
                "home_team": home_team,
                "away_team": away_team,
                "home_colour": home,
                "away_colour": away,
                "fallback": rule,
                "normal": readings["normal"],
                **{kind: readings[kind] for kind in CVD_KINDS},
                "worst_cvd": min(readings[kind] for kind in CVD_KINDS),
                "home_contrast": contrast_ratio(home, PALETTE["bg"]),
                "away_contrast": contrast_ratio(away, PALETTE["bg"]),
            }
        )

    frame = pl.DataFrame(records)
    frame.write_parquet(paths.RESEARCH_OUTPUT_DIR / RESULTS)

    counts = {rule: int((frame["fallback"] == rule).sum()) for rule in FALLBACKS}
    print("\n  " + "   ".join(f"{rule}: {count}" for rule, count in counts.items() if count))
    print(f"   unresolved: {counts['unresolved']}")

    if counts["unresolved"]:
        stuck = frame.filter(pl.col("fallback") == "unresolved")
        print("\n  Matchups the ladder could not separate:")
        for row in stuck.to_dicts():
            print(
                f"    {row['away_team']} @ {row['home_team']}  {row['home_colour']} / "
                f"{row['away_colour']}  worst CVD {row['worst_cvd']:.1f}  "
                f"normal {row['normal']:.1f}"
            )
        raise SystemExit("unresolved matchups — the ladder is short a rung.")

    # The 6-8 band is legal here and only here: every figure in this product
    # carries a legend, a direct label or the club's own mark, so colour is never
    # the only thing telling two bars apart. Counting the band is how that stays
    # a measured claim rather than an assumption.
    warn = frame.filter(pl.col("worst_cvd") < CVD_TARGET)
    print(
        f"\n  worst-case CVD separation: min {frame['worst_cvd'].min():.1f}, "
        f"median {frame['worst_cvd'].median():.1f}  "
        f"(floor {CVD_FLOOR:g}, target {CVD_TARGET:g})"
    )
    print(f"  in the {CVD_FLOOR:g}-{CVD_TARGET:g} band, legal on secondary encoding: {warn.height}")
    print(
        f"  normal-vision separation: min {frame['normal'].min():.1f} (hard floor {NORMAL_FLOOR:g})"
    )
    low = frame.filter(pl.col("home_contrast") < 3.0)
    if low.height:
        clubs_low = sorted({row["home_team"] for row in low.to_dicts()})
        print(
            f"  home colours under 3:1 on the cream: {', '.join(clubs_low)} "
            f"({low.height} matchups) — the incumbent is not gated, see `resolve_pair`"
        )

    print("\n  Fallbacks that fired, by matchup:")
    for rule in FALLBACKS:
        rows = frame.filter(pl.col("fallback") == rule)
        if rule == "primaries" or not rows.height:
            continue
        shown = ", ".join(f"{r['away_team']}@{r['home_team']}" for r in rows.to_dicts()[:12])
        more = "" if rows.height <= 12 else f", and {rows.height - 12} more"
        print(f"    {rule:26} {rows.height:3}  {shown}{more}")

    swatch_grid(frame, clubs)


def swatch_grid(frame: pl.DataFrame, clubs) -> None:
    """A 32 x 32 grid: each cell the two colours that matchup is drawn in.

    Rows are the home team, columns the away team. The diagonal is blank — a club
    does not play itself — and a cell split on the diagonal shows the pair as the
    figures will show it, which is the only way to check 992 decisions by eye.
    """
    index = {club: position for position, club in enumerate(clubs)}
    with mpl.rc_context(rc_style()):
        fig, ax = plt.subplots(figsize=(13.0, 13.4))
        for row in frame.to_dicts():
            x, y = index[row["away_team"]], index[row["home_team"]]
            # Lower-left triangle the home colour, upper-right the away — the
            # same split a bar chart would show, at one cell per matchup.
            ax.fill([x, x + 0.92, x], [y, y, y + 0.92], color=row["home_colour"], linewidth=0)
            ax.fill(
                [x + 0.92, x + 0.92, x],
                [y, y + 0.92, y + 0.92],
                color=row["away_colour"],
                linewidth=0,
            )
        ax.set_xlim(-0.4, len(clubs) + 0.4)
        ax.set_ylim(len(clubs) + 0.4, -0.4)
        ax.set_xticks([position + 0.46 for position in range(len(clubs))])
        ax.set_yticks([position + 0.46 for position in range(len(clubs))])
        ax.set_xticklabels(clubs, fontsize=7, rotation=90)
        ax.set_yticklabels(clubs, fontsize=7)
        ax.tick_params(length=0)
        ax.set_xlabel("away team", fontsize=9, color=PALETTE["text_muted"])
        ax.set_ylabel("home team", fontsize=9, color=PALETTE["text_muted"])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.text(
            0.0,
            1.055,
            "Every matchup, separated for every reader",
            fontsize=17,
            fontweight="bold",
            color=PALETTE["text"],
            va="bottom",
            fontfamily=heading_font(),
            transform=ax.transAxes,
        )
        fig.text(
            0.0,
            1.020,
            f"{frame.height} ordered matchups, none unresolved — lower-left is the home "
            "colour, upper-right the away",
            fontsize=9,
            color=PALETTE["text_muted"],
            va="bottom",
            transform=ax.transAxes,
        )
        print(f"\n  {finalize(fig, paths.RESEARCH_OUTPUT_DIR / GRID)}")


if __name__ == "__main__":
    main()
