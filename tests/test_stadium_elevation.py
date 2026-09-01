"""The hand-entered stadium elevation table.

The table is data entry, so the tests are the guard against a typo that no gate
would catch: a wrong elevation on a stadium with 700 kicks would move a
coefficient and nothing would complain. The fast tests pin the two rows the
whole covariate rests on and the shape of every other row; the slow test is the
completeness check against the real play-by-play cache.
"""

from __future__ import annotations

import pytest

from nfl_simulator.data.stadium_elevation import (
    STADIUM_ELEVATION_FT,
    elevation_ft,
    elevation_kft,
)


def test_denver_is_a_mile_up():
    assert elevation_ft("DEN00") > 5_000


def test_mexico_city_is_the_highest_row():
    assert elevation_ft("MEX00") > 7_000
    assert elevation_ft("MEX00") == max(STADIUM_ELEVATION_FT.values())


def test_kft_is_feet_over_a_thousand():
    assert elevation_kft("DEN00") == pytest.approx(elevation_ft("DEN00") / 1000.0)
    assert elevation_kft("DEN00") == pytest.approx(5.28)


def test_every_elevation_is_plausible():
    for stadium_id, feet in STADIUM_ELEVATION_FT.items():
        assert isinstance(feet, float), stadium_id
        assert -100.0 <= feet <= 8_000.0, stadium_id


def test_ids_are_the_nflverse_shape():
    for stadium_id in STADIUM_ELEVATION_FT:
        assert len(stadium_id) == 5, stadium_id
        assert stadium_id[:3].isalpha() and stadium_id[:3].isupper(), stadium_id
        assert stadium_id[3:].isdigit(), stadium_id


def test_unknown_stadium_raises_rather_than_guessing():
    """A missing stadium must never fall back to sea level.

    Zero is a real elevation — MetLife's is 10 feet — so a silent default would
    be indistinguishable from a correct row and would price a kick at the wrong
    altitude with no warning.
    """
    with pytest.raises(KeyError, match="ZZZ99"):
        elevation_ft("ZZZ99")


def test_only_denver_and_mexico_city_clear_three_thousand_feet():
    """The design fact the pre-registration rests on, pinned as a test.

    If a future row pushed a third stadium above 3,000 feet the covariate would
    stop being near-binary and document 66's power table would no longer
    describe the design it was computed for.
    """
    high = {k for k, v in STADIUM_ELEVATION_FT.items() if v >= 3_000}
    assert high == {"DEN00", "MEX00"}


@pytest.mark.slow
def test_every_stadium_in_the_pbp_resolves():
    """Completeness against the real cache — the check a typo in an id survives.

    The cache is gitignored and regenerable, so a fresh clone has no `data/`
    yet. Skip rather than fail there: the check is a data-entry guard for the
    machine that holds the pulls, not a claim the repo carries them.
    """
    import polars as pl

    from nfl_simulator import paths
    from nfl_simulator.ingest import PBP_SEASONS, load_pbp

    uncached = [s for s in PBP_SEASONS if not paths.pbp_path(s).exists()]
    if uncached:
        pytest.skip(
            f"play-by-play cache absent for {uncached[0]}-{uncached[-1]} — run "
            "`uv run python -m nfl_simulator.ingest` to enable this check"
        )

    pbp = load_pbp(PBP_SEASONS, columns=["stadium_id"])
    seen = set(pbp["stadium_id"].unique().drop_nulls().to_list())
    assert seen, "no stadium ids in the cache"
    missing = seen - set(STADIUM_ELEVATION_FT)
    assert not missing, f"stadium ids with no elevation: {sorted(missing)}"
    assert pbp["stadium_id"].null_count() == 0
    del pl


# --------------------------------------------------------------------------
# the 2026 season's new site, and the guard that makes a missing row loud
# --------------------------------------------------------------------------


def test_the_melbourne_cricket_ground_has_a_row():
    """The NFL plays at the MCG in week 2 of 2026, and it is not in 2016-2025 data.

    The row is entered ahead of the season rather than after the first kick is
    mispriced. Its `stadium_id` is provisional — nflverse has not assigned one
    — which is exactly why the guard below matters more than the row does.
    """
    assert elevation_ft("MEL00") == pytest.approx(100.0, abs=60.0)


def test_the_unknown_stadium_message_names_the_file_to_edit():
    """Handoff constraint 5: the error has to say what to do about itself.

    A `KeyError` reading `'AUS01'` tells a reader a stadium is missing. This one
    tells them which file to add it to, which is the difference between a
    five-minute fix and a bisect.
    """
    with pytest.raises(KeyError, match=r"stadium_elevation\.py"):
        elevation_ft("AUS01")
    with pytest.raises(KeyError, match=r"stadium_elevation\.py"):
        elevation_kft("AUS01")
