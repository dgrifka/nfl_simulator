"""Team identity — the colours and logos a per-game figure is drawn in.

nflverse ships a 36-row team table: the 32 current clubs plus the relocation
aliases (OAK, SD, STL, LA) that 2016–2019 games are actually played by. All 36
are treated the same here; a lookup that only knew the current names would draw
a 2018 Raiders game in the grey fallback and give no reason for it.

Two rules are the module's whole job.

**Colour follows the entity, never its position.** A team wears its own primary
in every figure it appears in, home or away. The one exception is the clash
rule below, and it repaints only the away team — so a club's colour is stable
across the figures where it is at home.

**A pair has to be separable before it is pretty, and separable for
everybody.** Some primaries are identical (ATL and TB both ship ``#A71930``; LV,
OAK and PIT are all black), some are a few percent apart, and some are far apart
in RGB and identical to a colourblind reader — the Jets' green against the 49ers'
red is 0.42 in RGB and 5.2 in OKLab under protanopia. :func:`resolve_pair` runs
the `dataviz` skill's four readings, keeps the cheap RGB rule as its first check,
and substitutes club colours before synthetic tints.

Colour is never the only encoding regardless: every figure that uses these also
carries a legend, direct labels, or the club's logo. That is what makes the
skill's 6-8 warning band legal here rather than merely tolerated.

Everything downloaded here lands under ``data/`` and is gitignored. The logos
are ESPN's and the clubs' — cached for rendering, never redistributed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from nfl_simulator import paths
from nfl_simulator.style import (
    CLASH_DARKEN,
    CLASH_DISTANCE,
    CLASH_LIGHTEN,
    colour_distance,
    darken,
    lighten,
    reads_on,
    separated,
)

# The baseball style's grey pair, for an abbreviation the table does not carry.
# A figure that cannot colour a team is still worth drawing; a crash is not.
FALLBACK_COLORS = ("#333333", "#666666")

# `CLASH_DISTANCE`, `CLASH_LIGHTEN` and `colour_distance` are re-exported from
# `style`: the same rule separates a team from the waterfall's ink anchors, and
# a figure module must not import this one to get at it.
__all__ = [
    "CLASH_DISTANCE",
    "CLASH_LIGHTEN",
    "FALLBACK_COLORS",
    "FALLBACKS",
    "colour_distance",
    "load_team_table",
    "pair_colors",
    "resolve_pair",
    "team_colors",
    "team_logo",
    "team_name",
]

# The order the clash rule tries things in, and the name each attempt is
# reported under. A real club colour before a synthetic tint: a reader who knows
# the Buccaneers knows their pewter, and nobody knows a 45%-lightened red.
FALLBACKS = (
    "primaries",
    "away secondary",
    "home secondary",
    "away lightened",
    "away secondary lightened",
    "away darkened",
    "away secondary darkened",
    "home darkened",
    "unresolved",
)

# A logo pixel this close to white is the surround the club's PNG ships with,
# not part of the mark. Left in, it prints as a white postage stamp on cream.
NEAR_WHITE = 240

TABLE_COLUMNS = ("team_abbr", "team_name", "team_nick", "team_color", "team_color2")


def _pull_teams() -> pl.DataFrame:
    """The one network call in this module, isolated so tests can replace it."""
    import nflreadpy as nfl

    return nfl.load_teams()


def load_team_table(*, path: Path | None = None, refresh: bool = False) -> pl.DataFrame:
    """The nflverse team table, cached to ``data/teams.parquet`` after one pull.

    Cached rather than memoised: the figures are rendered by scripts that run
    once, and a parquet on disk survives between them. 36 rows is small enough
    that re-reading it per lookup costs nothing worth caching in process.
    """
    path = paths.TEAMS_PATH if path is None else Path(path)
    if refresh or not path.exists():
        table = _pull_teams()
        path.parent.mkdir(parents=True, exist_ok=True)
        table.write_parquet(path)
        return table
    return pl.read_parquet(path)


def _row(team_abbr: str) -> dict | None:
    table = load_team_table()
    matches = table.filter(pl.col("team_abbr") == team_abbr)
    return matches.to_dicts()[0] if matches.height else None


def team_colors(team_abbr: str) -> tuple[str, str]:
    """``(primary, secondary)`` for a team, or the grey fallback for an unknown one."""
    row = _row(team_abbr)
    if row is None:
        return FALLBACK_COLORS
    primary = row.get("team_color") or FALLBACK_COLORS[0]
    secondary = row.get("team_color2") or FALLBACK_COLORS[1]
    return primary, secondary


def team_name(team_abbr: str) -> str:
    """The club's full name, falling back to the abbreviation itself."""
    row = _row(team_abbr)
    return (row or {}).get("team_name") or team_abbr


def resolve_pair(home_team: str, away_team: str) -> tuple[str, str, str]:
    """The two colours to draw a game in, and the name of the rule that chose them.

    **The RGB rule is blind and this is what it was blind to.** It kept the Jets'
    ``#003F2D`` beside the 49ers' ``#AA0000`` because the two are 0.42 apart in
    RGB — and for a reader with protanopia they are 5.2 apart in OKLab, well
    under the 6 the `dataviz` skill calls a floor. `2016_14_NYJ_SF` shipped in
    round 1 with two bars a colourblind reader sees as one. The RGB rule stays as
    the first, cheap check; the four readings in :func:`style.separations` are the
    ones that decide.

    Candidates are tried in a fixed order, and the first that separates *and*
    reads against the cream is taken. A real club colour comes before a synthetic
    tint, because a reader who knows the Buccaneers knows their pewter and nobody
    knows a 45%-lightened red:

    1. the two primaries, untouched
    2. the away team's secondary
    3. the home team's secondary, the away primary restored
    4. the away primary lightened toward white
    5. the away secondary lightened
    6. the away primary darkened toward black
    7. the away secondary darkened
    8. the home primary darkened

    Step 8 exists for exactly one matchup in the league: San Francisco at home
    against Kansas City. Two reds 12.9 apart under normal vision, and neither
    club's secondary — the Chiefs' gold, the 49ers' tan — reads on cream at all.
    Nothing the away side can wear separates them; darkening the 49ers' own red
    to ``#5e0000`` does, at 30.1. Last in the order because the home team's
    colour is the one a reader is least expecting to move.

    Steps 6 and 7 are this round's addition, and they are why ``unresolved`` is
    zero. The four the round was specified with leave 15 of the 992 ordered
    matchups with nowhere to go — Philadelphia's midnight green against seven
    opponents, Kansas City's red against San Francisco's, and three more. Every
    one of them fails the same way: the pair needs separating in *lightness*, and
    on a cream surface every candidate light enough to separate is too light to
    read on the background. Darkening separates and gains contrast at once. The
    baseball chart lightens only because it never ran a contrast check.

    Step 3 is the one place a home team's colour moves, and it is deliberate:
    round 1's rule only ever repainted the away side, and that left pairs like
    Kansas City's red against San Francisco's — 12.9 apart under **normal**
    vision — with nowhere to go that was still a club's own colour.

    Only the colour being *substituted* is checked for contrast against the
    surface. New Orleans' ``#D3BC8D`` reads at 1.78:1 on the cream and cannot be
    fixed by moving anybody else, so gating the incumbent would make every game
    the Saints host unresolvable rather than merely low-contrast.
    """
    home, home_second = team_colors(home_team)
    away, away_second = team_colors(away_team)

    candidates = (
        (home, away, None),
        (home, away_second, away_second),
        (home_second, away, home_second),
        (home, lighten(away, CLASH_LIGHTEN), lighten(away, CLASH_LIGHTEN)),
        (home, lighten(away_second, CLASH_LIGHTEN), lighten(away_second, CLASH_LIGHTEN)),
        (home, darken(away, CLASH_DARKEN), darken(away, CLASH_DARKEN)),
        (home, darken(away_second, CLASH_DARKEN), darken(away_second, CLASH_DARKEN)),
        (darken(home, CLASH_DARKEN), away, darken(home, CLASH_DARKEN)),
    )
    for (first, second, moved), name in zip(candidates, FALLBACKS, strict=False):
        # The cheap check first, exactly as before: two colours a quarter of the
        # RGB cube apart are usually fine, and the four simulations cost more.
        if name == "primaries" and colour_distance(first, second) < CLASH_DISTANCE:
            continue
        if separated(first, second) and (moved is None or reads_on(moved)):
            return first, second, name
    # A figure is still worth drawing. `research/60_matchup_colours.py` counts
    # how many matchups get this far, and the answer is meant to stay zero.
    return home, darken(away, CLASH_DARKEN), "unresolved"


def pair_colors(home_team: str, away_team: str) -> tuple[str, str]:
    """The two colours to draw a game in. See :func:`resolve_pair` for the rule."""
    home, away, _rule = resolve_pair(home_team, away_team)
    return home, away


def _knock_out_near_white(image) -> np.ndarray:
    """Make the logo's white surround transparent and return the array."""
    pixels = np.array(image.convert("RGBA"))
    surround = (
        (pixels[:, :, 0] > NEAR_WHITE)
        & (pixels[:, :, 1] > NEAR_WHITE)
        & (pixels[:, :, 2] > NEAR_WHITE)
    )
    pixels[surround, 3] = 0
    return pixels


def _crop_to_mark(pixels: np.ndarray) -> np.ndarray:
    """Trim the transparent border so the array is the mark and nothing else.

    Clubs' files are not all drawn to the same margins. ESPN's Jets logo is a
    4,096 px **square** holding a wide, short wordmark with most of the canvas
    empty; scaled to fit a box by its canvas, the visible mark comes out four
    times smaller than a shield drawn edge to edge in the same box. Cropping
    first means every club's mark is fitted on what a reader can actually see.

    A fully transparent image is returned untouched — a zero-sized crop is worse
    than the original.
    """
    visible = pixels[:, :, 3] > 0
    if not visible.any():
        return pixels
    rows, columns = np.where(visible.any(axis=1))[0], np.where(visible.any(axis=0))[0]
    return pixels[rows[0] : rows[-1] + 1, columns[0] : columns[-1] + 1]


def team_logo(team_abbr: str, *, cache_dir: Path | None = None) -> np.ndarray | None:
    """The club's mark as an RGBA array, downloaded once and cached thereafter.

    Returns ``None`` for a team the table does not carry, or when the download
    fails: a figure without a logo is a figure, and stopping a render because a
    CDN was slow would be the worse outcome. The caller draws the rest.
    """
    from PIL import Image

    cache_dir = paths.LOGO_DIR if cache_dir is None else Path(cache_dir)
    cached = cache_dir / f"{team_abbr}.png"

    if not cached.exists():
        row = _row(team_abbr)
        url = (row or {}).get("team_logo_espn")
        if not url:
            return None
        try:
            import urllib.request

            cache_dir.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=15) as response:
                cached.write_bytes(response.read())
        except Exception as error:  # pragma: no cover - network path
            print(f"Warning: could not fetch the {team_abbr} logo: {error}")
            return None

    try:
        with Image.open(cached) as image:
            return _crop_to_mark(_knock_out_near_white(image))
    except Exception as error:  # pragma: no cover - a corrupt cache entry
        print(f"Warning: could not read the cached {team_abbr} logo: {error}")
        return None
