"""Team colours and logos — the lookup, the clash rule, and the fallbacks.

No test here touches the network. The fixture frame is shaped exactly like
``nflreadpy.load_teams()`` and its hex values are copied from that table
(verified 2026-08-26, nflreadpy 0.1.5), so the assertions below document the
real values while the real pull is checked once, in the driver.

The alias rows matter more than they look. 2016–2019 games carry OAK, SD and
STL, and a lookup that only knew the current 32 would draw those games with the
grey fallback and no error to say why.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from matplotlib.colors import to_rgb
from PIL import Image

from nfl_simulator import teams as teams_module
from nfl_simulator.teams import (
    CLASH_DISTANCE,
    FALLBACK_COLORS,
    colour_distance,
    pair_colors,
    team_colors,
    team_logo,
)

# Copied from nflreadpy.load_teams(). OAK is one of the four relocation aliases
# and points at the Raiders' current ESPN logo, exactly as nflverse ships it.
ROWS = [
    ("GB", "Green Bay Packers", "Packers", "#203731", "#FFB612", "gb"),
    ("DET", "Detroit Lions", "Lions", "#0076B6", "#B0B7BC", "det"),
    ("KC", "Kansas City Chiefs", "Chiefs", "#E31837", "#FFB612", "kc"),
    ("SF", "San Francisco 49ers", "49ers", "#AA0000", "#B3995D", "sf"),
    ("ATL", "Atlanta Falcons", "Falcons", "#A71930", "#000000", "atl"),
    ("TB", "Tampa Bay Buccaneers", "Buccaneers", "#A71930", "#FF7900", "tb"),
    ("OAK", "Oakland Raiders", "Raiders", "#000000", "#A5ACAF", "lv"),
]


@pytest.fixture
def table(monkeypatch) -> pl.DataFrame:
    frame = pl.DataFrame(
        {
            "team_abbr": [row[0] for row in ROWS],
            "team_name": [row[1] for row in ROWS],
            "team_nick": [row[2] for row in ROWS],
            "team_color": [row[3] for row in ROWS],
            "team_color2": [row[4] for row in ROWS],
            "team_logo_espn": [
                f"https://a.espncdn.com/i/teamlogos/nfl/500/{row[5]}.png" for row in ROWS
            ],
        }
    )
    monkeypatch.setattr(teams_module, "load_team_table", lambda **_: frame)
    return frame


def synthetic_logo(path, *, near_white=(250, 250, 250), mark=(20, 40, 200)) -> None:
    """A 4x4 PNG: three rows of near-white surround, one row of ink."""
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    pixels[:, :] = near_white
    pixels[2, :] = mark
    Image.fromarray(pixels, mode="RGB").save(path)


# --------------------------------------------------------------------------
# colours
# --------------------------------------------------------------------------


def test_team_colors_returns_the_primary_and_secondary_nflverse_carries(table):
    assert team_colors("DET") == ("#0076B6", "#B0B7BC")


def test_an_unknown_abbreviation_falls_back_to_grey_rather_than_raising(table):
    """A figure for a game we cannot colour is still a figure worth drawing."""
    assert team_colors("XYZ") == FALLBACK_COLORS


def test_a_relocation_alias_resolves_like_any_other_team(table):
    """2016-2019 games are played by OAK, SD and STL, not by their successors."""
    assert team_colors("OAK") == ("#000000", "#A5ACAF")


# --------------------------------------------------------------------------
# the clash rule
# --------------------------------------------------------------------------


def test_two_distinguishable_primaries_are_left_alone(table):
    assert pair_colors("GB", "DET") == ("#203731", "#0076B6")


def test_a_pair_that_merely_shares_a_family_is_still_left_alone(table):
    """KC's red and SF's are 0.32 apart, which the rule does not treat as a clash."""
    home, away = pair_colors("KC", "SF")
    assert (home, away) == ("#E31837", "#AA0000")
    assert colour_distance(home, away) >= CLASH_DISTANCE


def test_a_genuine_clash_lightens_the_away_team_until_the_two_separate(table):
    """ATL and TB ship the identical primary #A71930 — undrawable as two bars."""
    home, away = pair_colors("ATL", "TB")
    assert home == "#A71930"
    assert away != "#A71930"
    assert colour_distance(home, away) >= CLASH_DISTANCE
    # Lightened toward white, so it stays recognisably the same team's red.
    assert all(
        channel >= original for channel, original in zip(to_rgb(away), to_rgb(home), strict=True)
    )


def test_the_clash_rule_never_repaints_the_home_team(table):
    """Colour follows the entity: the home team wears its own primary either way."""
    assert pair_colors("ATL", "TB")[0] == pair_colors("ATL", "GB")[0]


# --------------------------------------------------------------------------
# logos
# --------------------------------------------------------------------------


def test_a_cached_logo_is_read_without_touching_the_network(table, tmp_path):
    synthetic_logo(tmp_path / "OAK.png")
    logo = team_logo("OAK", cache_dir=tmp_path)
    assert logo is not None
    assert logo.shape == (4, 4, 4)


def test_the_near_white_surround_is_knocked_out_so_a_logo_sits_on_cream(table, tmp_path):
    """An un-knocked logo prints as a white postage stamp on the cream surface."""
    synthetic_logo(tmp_path / "OAK.png")
    logo = team_logo("OAK", cache_dir=tmp_path)
    alpha = logo[:, :, 3]
    assert (alpha[[0, 1, 3], :] == 0).all(), "near-white surround should be transparent"
    assert (alpha[2, :] == 255).all(), "the mark itself should be opaque"


def test_an_unknown_team_has_no_logo_and_says_so_quietly(table, tmp_path):
    assert team_logo("XYZ", cache_dir=tmp_path) is None
