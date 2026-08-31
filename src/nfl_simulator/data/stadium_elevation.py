"""Stadium elevation, in feet above sea level, keyed by nflverse ``stadium_id``.

Elevation is **not** in nflverse play-by-play, so this table is hand-entered.
It exists because thinner air lengthens a kick, and the field-goal model
(`docs/research/05b-fg-model-foundations.md`) prices every kick as if the air
were the same everywhere — Denver's mile of altitude included.

**Every row is a site elevation for the stadium's location, rounded to the
nearest 10 feet, from public geographic references** (USGS/GNIS elevations for
the US sites, national mapping equivalents abroad, as reported by the standard
public gazetteers). Rounding to 10 feet is deliberate: the covariate this table
feeds is expressed in *thousands* of feet, so a 50-foot disagreement between
sources moves a coefficient by 5% of one prior standard deviation, and pretending
to foot-level precision on a stadium bowl that is itself 100 feet deep would be
false accuracy. The two rows that carry the covariate — Denver and Mexico City —
are the two the tests pin, and neither is in dispute at that resolution.

The keys are exactly the 42 ``stadium_id`` values that appear in 2016–2025
play-by-play. `tests/test_stadium_elevation.py` checks that completeness against
the cache, so a stadium added by a future season fails the suite rather than
silently falling back to a default.
"""

from __future__ import annotations

# Feet above sea level. One row per stadium, comment = city the elevation is of.
STADIUM_ELEVATION_FT: dict[str, float] = {
    # --- the two rows the covariate rests on -------------------------------
    "DEN00": 5280.0,  # Denver, CO — Empower Field at Mile High; the stadium is
    #                   named for the elevation and is the only US site above
    #                   3,000 ft. 700 kicks, 2016-2025.
    "MEX00": 7280.0,  # Mexico City — Estadio Azteca, ~2,220 m. The highest site
    #                   in the data and the only one above Denver. 32 kicks.
    # --- the mid band, 1,500-3,000 ft --------------------------------------
    "SAO00": 2530.0,  # Sao Paulo, Brazil — Neo Quimica Arena, ~770 m. 12 kicks.
    "VEG00": 2030.0,  # Las Vegas, NV — Allegiant Stadium. A *domed* stadium at
    #                   altitude, which is what lets roof and elevation separate.
    "GER00": 1690.0,  # Munich, Germany — Allianz Arena, ~515 m. 14 kicks.
    # --- 500-1,100 ft -------------------------------------------------------
    "PHO00": 1070.0,  # Glendale, AZ — State Farm Stadium.
    "ATL97": 1050.0,  # Atlanta, GA — Mercedes-Benz Stadium.
    "ATL00": 1050.0,  # Atlanta, GA — Georgia Dome (2016 only), same site.
    "MIN01": 830.0,  # Minneapolis, MN — U.S. Bank Stadium.
    "BUF00": 820.0,  # Orchard Park, NY — Highmark/New Era Field.
    "KAN00": 750.0,  # Kansas City, MO — Arrowhead Stadium.
    "CAR00": 730.0,  # Charlotte, NC — Bank of America Stadium.
    "PIT00": 730.0,  # Pittsburgh, PA — Acrisure/Heinz Field.
    "IND00": 715.0,  # Indianapolis, IN — Lucas Oil Stadium.
    "GNB00": 640.0,  # Green Bay, WI — Lambeau Field.
    "CHI98": 600.0,  # Chicago, IL — Soldier Field.
    "DET00": 600.0,  # Detroit, MI — Ford Field.
    "CLE00": 570.0,  # Cleveland, OH — Huntington Bank/FirstEnergy Stadium.
    "DAL00": 550.0,  # Arlington, TX — AT&T Stadium.
    "CIN00": 490.0,  # Cincinnati, OH — Paycor/Paul Brown Stadium.
    "NAS00": 440.0,  # Nashville, TN — Nissan Stadium.
    "FRA00": 370.0,  # Frankfurt, Germany — Deutsche Bank Park, ~112 m.
    # --- under 300 ft -------------------------------------------------------
    "BOS00": 290.0,  # Foxborough, MA — Gillette Stadium.
    "LAX99": 180.0,  # Los Angeles, CA — LA Memorial Coliseum.
    "WAS00": 180.0,  # Landover, MD — FedExField / Northwest Stadium.
    "LON00": 165.0,  # London, England — Wembley Stadium, ~50 m.
    "LAX01": 125.0,  # Inglewood, CA — SoFi Stadium.
    "LON02": 100.0,  # London, England — Tottenham Hotspur Stadium, ~30 m.
    "SDG00": 90.0,  # San Diego, CA — Qualcomm Stadium (2016 only).
    "LAX97": 65.0,  # Carson, CA — StubHub Center (2017-2019).
    "HOU00": 50.0,  # Houston, TX — NRG Stadium.
    "LON01": 40.0,  # London, England — Twickenham Stadium, ~12 m.
    "BAL00": 30.0,  # Baltimore, MD — M&T Bank Stadium.
    "SFO01": 30.0,  # Santa Clara, CA — Levi's Stadium.
    "TAM00": 25.0,  # Tampa, FL — Raymond James Stadium.
    "OAK00": 20.0,  # Oakland, CA — Oakland Coliseum (2016-2019).
    "PHI00": 20.0,  # Philadelphia, PA — Lincoln Financial Field.
    "SEA00": 20.0,  # Seattle, WA — Lumen/CenturyLink Field.
    "JAX00": 15.0,  # Jacksonville, FL — EverBank Stadium.
    "MIA00": 10.0,  # Miami Gardens, FL — Hard Rock Stadium.
    "NOR00": 10.0,  # New Orleans, LA — Caesars/Mercedes-Benz Superdome.
    "NYC01": 10.0,  # East Rutherford, NJ — MetLife Stadium.
}


def elevation_ft(stadium_id: str) -> float:
    """Feet above sea level for one nflverse ``stadium_id``.

    Raises ``KeyError`` on an unknown stadium rather than defaulting. Sea level
    is a real elevation in this table — MetLife is 10 feet — so a silent default
    would be indistinguishable from a correct row.
    """
    try:
        return STADIUM_ELEVATION_FT[stadium_id]
    except KeyError:
        raise KeyError(
            f"no elevation for stadium_id {stadium_id!r}; add it to "
            "src/nfl_simulator/data/stadium_elevation.py with its source"
        ) from None


def elevation_kft(stadium_id: str) -> float:
    """Elevation in thousands of feet — the unit the model's covariate uses.

    Thousands, not feet, so ``beta_elev`` lands on the same order of magnitude
    as the model's other log-odds coefficients and its prior means something a
    reader can check by hand.
    """
    return elevation_ft(stadium_id) / 1000.0
