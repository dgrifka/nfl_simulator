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
from PIL import Image

from nfl_simulator import teams as teams_module
from nfl_simulator.style import (
    CONTRAST_MIN,
    CVD_FLOOR,
    CVD_KINDS,
    NORMAL_FLOOR,
    PALETTE,
    contrast_ratio,
    relative_luminance,
    separated,
    separations,
)
from nfl_simulator.teams import (
    CLASH_DISTANCE,
    FALLBACK_COLORS,
    FALLBACKS,
    RELOCATIONS,
    colour_distance,
    era_code,
    pair_colors,
    resolve_pair,
    team_colors,
    team_logo,
    team_name,
)

# Copied from nflreadpy.load_teams(). OAK is one of the four relocation aliases
# and points at the Raiders' current ESPN logo, exactly as nflverse ships it.
ROWS = [
    ("GB", "Green Bay Packers", "Packers", "#203731", "#FFB612", "gb"),
    ("DET", "Detroit Lions", "Lions", "#0076B6", "#B0B7BC", "det"),
    ("KC", "Kansas City Chiefs", "Chiefs", "#E31837", "#FFB612", "kc"),
    ("SF", "San Francisco 49ers", "49ers", "#AA0000", "#B3995D", "sf"),
    ("ATL", "Atlanta Falcons", "Falcons", "#A71930", "#000000", "atl"),
    ("TB", "Tampa Bay Buccaneers", "Buccaneers", "#A71930", "#322F2B", "tb"),
    ("OAK", "Oakland Raiders", "Raiders", "#000000", "#A5ACAF", "lv"),
    ("LV", "Las Vegas Raiders", "Raiders", "#000000", "#A5ACAF", "lv"),
    ("SD", "San Diego Chargers", "Chargers", "#007BC7", "#ffc20e", "lac"),
    ("LAC", "Los Angeles Chargers", "Chargers", "#007BC7", "#ffc20e", "lac"),
    ("NYJ", "New York Jets", "Jets", "#003F2D", "#000000", "nyj"),
    ("NO", "New Orleans Saints", "Saints", "#D3BC8D", "#101820", "no"),
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


def test_a_club_whose_primary_cannot_be_read_on_the_cream_wears_its_secondary(table):
    """New Orleans' old gold is 1.78:1 on the surface. Every figure the Saints
    appeared in drew them in a colour a reader has to hunt for."""
    row = table.filter(pl.col("team_abbr") == "NO").to_dicts()[0]
    primary, secondary = row["team_color"], row["team_color2"]
    assert contrast_ratio(primary, PALETTE["bg"]) < CONTRAST_MIN, "the fixture's premise"
    assert team_colors("NO") == (secondary, primary)
    assert contrast_ratio(team_colors("NO")[0], PALETTE["bg"]) >= CONTRAST_MIN


def test_a_club_whose_primary_reads_on_the_cream_keeps_it(table):
    """The floor moves one club. It must not quietly reorder the other 31."""
    assert team_colors("GB") == ("#203731", "#FFB612")
    assert team_colors("KC") == ("#E31837", "#FFB612")


def test_a_club_whose_two_colours_both_fail_is_darkened_until_it_reads(monkeypatch):
    """Nobody in the league is this club, and the rule still has to answer for
    it: swapping to a second unreadable colour is not a fix."""
    frame = pl.DataFrame(
        {
            "team_abbr": ["CRM"],
            "team_name": ["Cream City"],
            "team_nick": ["Cream"],
            "team_color": ["#F2E9D8"],
            "team_color2": ["#FFFFFF"],
            "team_logo_espn": ["https://example.invalid/crm.png"],
        }
    )
    monkeypatch.setattr(teams_module, "load_team_table", lambda **_: frame)
    drawn, _second = team_colors("CRM")
    assert contrast_ratio(drawn, PALETTE["bg"]) >= CONTRAST_MIN
    assert relative_luminance(drawn) < relative_luminance("#F2E9D8"), "darkened, not swapped"


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


def test_a_pair_the_rgb_rule_passes_can_still_be_two_reds_nobody_can_read(table):
    """Kansas City's red and San Francisco's are 0.32 apart in RGB, which round
    1's rule called separate, and 12.9 apart in OKLab under *normal* vision —
    under the 15 the dataviz skill calls a hard floor. Both move."""
    assert colour_distance("#E31837", "#AA0000") >= CLASH_DISTANCE, "the RGB rule sees nothing"
    home, away = pair_colors("KC", "SF")
    assert (home, away) != ("#E31837", "#AA0000")
    assert separations(home, away)["normal"] >= NORMAL_FLOOR


def test_a_genuine_clash_reaches_for_a_club_colour_before_a_tint(table):
    """ATL and TB ship the identical primary #A71930 — undrawable as two bars.

    Round 1 lightened the away red by 45%. Tampa Bay's own pewter separates and
    is a colour somebody actually recognises, so it goes first now."""
    home, away = pair_colors("ATL", "TB")
    assert home == "#A71930"
    assert away == "#322F2B", "the Buccaneers' own secondary, not a tinted red"
    assert resolve_pair("ATL", "TB")[2] == "away secondary"


def test_the_home_team_moves_only_after_everything_the_away_team_can_wear(table):
    """Round 1's rule only ever repainted the away side, and that left pairs with
    nowhere to go that was still a club's own colour. The home colour may now
    move — but only to the home club's *own* secondary, and only once the away
    side has been tried first."""
    assert resolve_pair("ATL", "TB")[2] == "away secondary"
    home, _away = pair_colors("ATL", "GB")
    assert home in team_colors("ATL"), "if it moves, it moves to that club's own colour"


def test_the_fallback_named_is_the_one_that_actually_fired(table):
    for home_team, away_team in (("DET", "GB"), ("ATL", "TB"), ("SF", "NYJ")):
        home, away, rule = resolve_pair(home_team, away_team)
        assert (home, away) == pair_colors(home_team, away_team)
        assert rule in FALLBACKS
        assert rule != "unresolved"


# --------------------------------------------------------------------------
# logos
# --------------------------------------------------------------------------


def test_a_cached_logo_is_read_without_touching_the_network(table, tmp_path):
    synthetic_logo(tmp_path / "OAK.png")
    logo = team_logo("OAK", cache_dir=tmp_path)
    assert logo is not None
    assert logo.ndim == 3 and logo.shape[2] == 4, "RGBA, ready for OffsetImage"


def test_the_near_white_surround_is_knocked_out_so_a_logo_sits_on_cream(table, tmp_path):
    """An un-knocked logo prints as a white postage stamp on the cream surface.

    The fixture is one ink row in three near-white ones, so a knocked-out and
    cropped logo is exactly that row: nothing opaque is lost, nothing white is
    kept."""
    synthetic_logo(tmp_path / "OAK.png")
    logo = team_logo("OAK", cache_dir=tmp_path)
    assert logo.shape[:2] == (1, 4)
    assert (logo[:, :, 3] == 255).all(), "what survives the crop is the mark, fully opaque"


def test_an_unknown_team_has_no_logo_and_says_so_quietly(table, tmp_path):
    assert team_logo("XYZ", cache_dir=tmp_path) is None


def transparent_bordered_logo(path) -> None:
    """A 8x8 PNG whose mark is a 6x2 band with near-white padding all round.

    ESPN's Jets file is this shape at 4096 px: a square canvas holding a wide,
    short wordmark. Fitted by its canvas the mark comes out unreadably small.
    """
    pixels = np.full((8, 8, 3), 250, dtype=np.uint8)
    pixels[3:5, 1:7] = (20, 40, 200)
    Image.fromarray(pixels, mode="RGB").save(path)


def test_a_logo_is_cropped_to_the_mark_rather_than_to_its_canvas(table, tmp_path):
    transparent_bordered_logo(tmp_path / "OAK.png")
    logo = team_logo("OAK", cache_dir=tmp_path)
    assert logo.shape[:2] == (2, 6), "the padding should be gone, the mark kept whole"
    assert (logo[:, :, 3] == 255).all()


def test_cropping_leaves_a_mark_that_fills_its_canvas_alone(table, tmp_path):
    """A logo with no padding is returned at its own size, not shrunk."""
    synthetic_logo(tmp_path / "OAK.png", near_white=(10, 10, 10))
    assert team_logo("OAK", cache_dir=tmp_path).shape[:2] == (4, 4)


def test_a_logo_that_is_entirely_padding_is_returned_rather_than_cropped_away(table, tmp_path):
    """Never return a zero-sized image: an empty crop is worse than the original."""
    Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8), mode="RGB").save(tmp_path / "OAK.png")
    logo = team_logo("OAK", cache_dir=tmp_path)
    assert logo.shape[:2] == (4, 4)


# --------------------------------------------------------------------------
# staying clear of the figures' own marks
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "primary, team",
    [("#0B162A", "CHI"), ("#000000", "LV"), ("#203731", "GB"), ("#03202F", "HOU")],
    ids=["CHI", "LV", "GB", "HOU"],
)
def test_the_anchor_neutral_is_clear_of_every_club_a_total_could_be_confused_with(primary, team):
    """The waterfall's two totals must never be a colour a club also wears.

    These four are the league's darkest primaries, and against #1A1A1A ink they
    sit at 0.087, 0.177, 0.147 and 0.124 — no threshold separates the ones that
    would need moving from the ones that would not. The anchor neutral is chosen
    so the question never arises."""
    assert colour_distance(primary, PALETTE["anchor"]) >= CLASH_DISTANCE


# --------------------------------------------------------------------------
# colour-vision separation — figure workshop round 2, Part C
# --------------------------------------------------------------------------


def worst_cvd(first: str, second: str) -> float:
    return min(separations(first, second)[kind] for kind in CVD_KINDS)


def test_two_colours_a_protan_reader_cannot_tell_apart_are_separated(table):
    """`2016_14_NYJ_SF`: Jets green against 49ers red is protan ΔE 5.2. The RGB
    rule cannot see it — the two are 0.42 apart — so the figure shipped with two
    bars a colourblind reader reads as one."""
    assert worst_cvd("#003F2D", "#AA0000") < CVD_FLOOR, "the defect this rule exists for"
    home, away = pair_colors("SF", "NYJ")
    assert worst_cvd(home, away) >= CVD_FLOOR


def test_a_separated_pair_is_still_separated_for_a_reader_with_normal_vision(table):
    home, away = pair_colors("SF", "NYJ")
    assert separations(home, away)["normal"] >= NORMAL_FLOOR


def test_a_separated_pair_still_reads_against_the_cream_surface(table):
    home, away = pair_colors("SF", "NYJ")
    assert contrast_ratio(away, PALETTE["bg"]) >= 3.0


def test_the_identical_primaries_still_resolve(table):
    """ATL and TB both ship #A71930, which no simulation can pull apart."""
    home, away = pair_colors("ATL", "TB")
    assert home != away
    assert worst_cvd(home, away) >= CVD_FLOOR
    assert separations(home, away)["normal"] >= NORMAL_FLOOR


def test_a_pair_that_already_separates_is_left_alone(table):
    """Green Bay's green against Detroit's blue needs no help under any vision."""
    assert pair_colors("DET", "GB") == ("#0076B6", "#203731")


def test_the_rule_is_deterministic(table):
    assert pair_colors("SF", "NYJ") == pair_colors("SF", "NYJ")
    assert pair_colors("ATL", "TB") == pair_colors("ATL", "TB")


def test_the_separations_are_the_dataviz_validator_s_four_readings(table):
    readings = separations("#0076B6", "#203731")
    assert set(readings) == {"normal", *CVD_KINDS}
    assert all(value > 0 for value in readings.values())


def test_a_colour_is_perfectly_separable_from_nothing_but_itself(table):
    assert all(value == pytest.approx(0.0) for value in separations("#AA0000", "#AA0000").values())


def test_a_pair_that_can_only_be_separated_in_lightness_is_darkened_not_lightened(table):
    """On a cream surface, lightening a colour walks it into the background.

    San Francisco's red beside Kansas City's is the league's hardest matchup:
    neither club's secondary reads on the cream, and every candidate light enough
    to separate the two reds is too light to see. Darkening does both at once."""
    home, away, rule = resolve_pair("SF", "KC")
    assert rule.endswith("darkened")
    assert separated(home, away)
    assert contrast_ratio(home, PALETTE["bg"]) >= 3.0
    assert contrast_ratio(away, PALETTE["bg"]) >= 3.0


# --------------------------------------------------------------------------
# round 12 Part A: the club a game was played by
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("abbr", "season", "expected"),
    [
        # The Raiders left Oakland after 2019.
        ("LV", 2016, "OAK"),
        ("LV", 2019, "OAK"),
        ("LV", 2020, "LV"),
        ("LV", 2025, "LV"),
        # The Chargers left San Diego after 2016.
        ("LAC", 2016, "SD"),
        ("LAC", 2017, "LAC"),
        # The Rams left St. Louis after 2015, which is before this corpus starts.
        ("LA", 2015, "STL"),
        ("LA", 2016, "LA"),
        # A club that never moved, and an era code handed back to itself.
        ("GB", 2016, "GB"),
        ("OAK", 2017, "OAK"),
        ("SD", 2016, "SD"),
    ],
)
def test_the_era_code_is_the_abbreviation_the_season_was_played_under(abbr, season, expected):
    """Document 63 §7d N6. The play-by-play carries the era code and the summary
    artifacts carry the modern one, so a 2017 Oakland game arrives as `LV` and
    every surface of its figure says so. The season decides, never the code."""
    assert era_code(abbr, season) == expected


def test_a_season_nobody_supplied_leaves_the_abbreviation_alone():
    """A caller without a season is not guessing one. Handing back the modern
    code is the only answer that cannot be wrong about a year."""
    assert era_code("LV") == "LV"
    assert era_code("LV", None) == "LV"


def test_the_relocation_table_is_keyed_on_the_modern_code_and_a_last_season():
    """Constraint 6: era names come from the season. The table is read forwards
    — `(modern code, last season under the old one) -> old code` — so no lookup
    can turn a 2021 Las Vegas game into an Oakland one."""
    assert RELOCATIONS == (("LV", 2019, "OAK"), ("LAC", 2016, "SD"), ("LA", 2015, "STL"))


def test_a_2017_raiders_game_is_named_for_oakland(table):
    """`2017_16_OAK_PHI` was captioned `Las Vegas Raiders` on every surface."""
    assert team_name("LV", 2017) == "Oakland Raiders"


def test_a_2021_raiders_game_is_still_named_for_las_vegas(table):
    """The fix must not rewrite the seasons the club really was in Las Vegas."""
    assert team_name("LV", 2021) == "Las Vegas Raiders"


def test_a_name_asked_for_without_a_season_is_the_modern_one(table):
    assert team_name("LV") == "Las Vegas Raiders"
    assert team_name("OAK") == "Oakland Raiders"


def test_a_2016_chargers_game_is_named_for_san_diego(table):
    assert team_name("LAC", 2016) == "San Diego Chargers"
    assert team_name("LAC", 2017) == "Los Angeles Chargers"


def test_a_pre_relocation_club_wears_the_colours_its_era_row_carries(table):
    """Not a number, so not gated — but it must read off the era row rather
    than the modern one, or a club whose palette changed would be drawn wrong."""
    assert team_colors("LV", season=2017) == team_colors("OAK")
    assert resolve_pair("PHI", "LV", season=2017)[:2] == resolve_pair("PHI", "OAK")[:2]


def test_the_era_mark_is_the_one_the_table_carries_for_the_era_code(table, tmp_path):
    """`team_logo` caches per abbreviation, so a season-aware lookup has to cache
    under the era code — otherwise a 2017 Oakland render would read Las Vegas's
    cache entry and the two clubs could never differ."""
    synthetic_logo(tmp_path / "OAK.png")
    assert team_logo("LV", season=2017, cache_dir=tmp_path) is not None
    assert not (tmp_path / "LV.png").exists(), "a 2017 game must not touch the modern cache entry"


def test_a_modern_season_still_reads_the_modern_mark(table, tmp_path):
    synthetic_logo(tmp_path / "LV.png")
    assert team_logo("LV", season=2021, cache_dir=tmp_path) is not None
    assert not (tmp_path / "OAK.png").exists()
