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

**A pair has to be separable before it is pretty.** Some primaries are
identical (ATL and TB both ship ``#A71930``; LV, OAK and PIT are all black) and
some are a few percent apart. Drawn as two bars they are one bar. When the RGB
distance falls under :data:`CLASH_DISTANCE` the away team's colour is lightened
toward white — it stays recognisably that team's hue, and the two fills stop
being one shape. This is the baseball run-distribution chart's rule, kept
identical so the two projects separate a clash the same way.

Colour is never the only encoding regardless: every figure that uses these also
carries a legend, direct labels, or the club's logo.

Everything downloaded here lands under ``data/`` and is gitignored. The logos
are ESPN's and the clubs' — cached for rendering, never redistributed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from nfl_simulator import paths
from nfl_simulator.style import CLASH_DISTANCE, CLASH_LIGHTEN, colour_distance, lighten

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
    "colour_distance",
    "load_team_table",
    "pair_colors",
    "team_colors",
    "team_logo",
    "team_name",
]

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


def pair_colors(home_team: str, away_team: str) -> tuple[str, str]:
    """The two primaries to draw a game in, separated if they clash.

    Only the **away** colour ever moves. Repainting whichever of the two happened
    to be darker would mean a club's colour changed depending on who it played,
    and a reader who knows the team would read the figure wrong.

    Nothing here has to dodge the figures' own marks. That was tried, and the
    threshold it needed sat between Green Bay at 0.147 and the Raiders at 0.177
    — too fine to be a rule. `style.PALETTE["anchor"]` is instead chosen to be
    0.281 from the nearest primary in the league, so no club ever collides with
    it and no team's colour has to move for a figure's convenience.
    """
    home = team_colors(home_team)[0]
    away = team_colors(away_team)[0]
    if colour_distance(home, away) < CLASH_DISTANCE:
        away = lighten(away, CLASH_LIGHTEN)
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
