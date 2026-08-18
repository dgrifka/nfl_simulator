"""Phase 6 candidate 1 — kickoff muffs: identification, power and the floor.

Runs **before** `docs/research/24-kickoff-muffs.md` fixes any threshold, and it
deliberately stops short of the treatment arm: the materiality *floor* is a
property of shipped v1.2 and is computed here, but the materiality *statistic*
is left for the fit script so the binding gate is genuinely unseen when the
thresholds are committed. That is document 20's arrangement rather than document
18's.

Document 23 C3 found that 245 of 263 kickoff muffs are invisible to the fumble
component, and that the muffs which are visible are overwhelmingly the ones the
kicking team recovered. That is the conditioning bug v1.2 fixed for out-of-bounds
fumbles, one population over. Four questions:

1. **Identification.** Can the disposition of a kickoff muff be read off the play
   text reliably enough to define a population on it? Every rejected row is
   printed, not counted (document 20 §9).
2. **What does the widened class table look like** — and does the muff branch
   belong in `kickoff/live` or in a class of its own?
3. **Is the entity spread on the widened population still negligible**, i.e. does
   full neutralization survive? Exact grid instrument of document 09 §4 at the
   real denominators.
4. **What is the materiality floor** the fit will have to clear, on the games a
   kickoff muff touches?

    uv run python research/36_kickoff_muff_power.py
"""

from __future__ import annotations

import json
import re
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_coinflips = import_module("12_coinflips_power")
_power = import_module("25_overtime_power")
_oob = import_module("29_fumble_oob_power")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.components import (  # noqa: E402
    build_game_table,
    fit_fg_baseline,
    fit_fumble_baseline,
    fit_xp_baseline,
)
from nfl_simulator.fg_model import FieldGoalModel  # noqa: E402
from nfl_simulator.ingest import PBP_SEASONS, load_pbp  # noqa: E402
from nfl_simulator.simulator import (  # noqa: E402
    bootstrap_margins,
    extra_point_events,
    field_goal_events,
    fumble_events,
    points_per_epa,
)

RANDOM_SEED = 20260818
MIN_CLASS_SIZE = 30
SIM_POSTERIOR_DRAWS = _power.SIM_POSTERIOR_DRAWS
SIM_COIN_DRAWS = _power.SIM_COIN_DRAWS

MUFF_COLUMNS = sorted(
    {
        *_power.SIM_COLUMNS,
        "desc",
        "own_kickoff_recovery",
        "own_kickoff_recovery_td",
    }
)

# "recovered by KC-80-J.Chesson" / "RECOVERED by ARI-36-B.Baker" — the team
# abbreviation is the only part the disposition needs.
RECOVERED_BY = re.compile(r"recovered by ([A-Z]{2,3})[- ]", re.IGNORECASE)


# --------------------------------------------------------------------------
# identification
# --------------------------------------------------------------------------


def kickoff_muff_mask() -> pl.Expr:
    """Kickoff plays where the receiving team put a hand on a live ball.

    Two channels, deliberately unioned:

    * the play text says **MUFFS** — 263 plays, and the channel that carries
      both branches;
    * `own_kickoff_recovery` or `own_kickoff_recovery_td` fires on a kickoff the
      text does not call a muff — 6 plays, every one of them a loss.

    The second channel is outcome-selected by construction and adding it alone
    would bias the rate downward. It is included because §1 checks the symmetric
    question and finds no retained counterpart hiding outside the MUFFS text,
    and because leaving six real losses out would bias the rate the other way.
    Onside kicks are excluded: document 20's population is a different event with
    a different rate, and its component is not shipped.
    """
    onside = pl.col("desc").str.contains("kicks onside")
    return (
        (pl.col("play_type") == "kickoff")
        & ~onside
        & (
            pl.col("desc").str.contains("MUFFS")
            | (pl.col("own_kickoff_recovery") == 1).fill_null(False)
            | (pl.col("own_kickoff_recovery_td") == 1).fill_null(False)
        )
    )


def _disposition(desc: str, posteam: str) -> str:
    """Who ended up with the ball, read off the play text in a fixed order.

    Order matters and is fixed here rather than discovered later:

    1. **Touchback** — the receiving team takes the ball at the spot by rule,
       whatever the sentence before it said. One play in ten seasons is both a
       named recovery and a touchback, and one is both out of bounds in the end
       zone and a touchback.
    2. **A named recovering team** — the most specific fact available.
    3. **"and recovers"** — the muffing player fell on it himself.
    4. **"ball out of bounds"** — touched by the receiving team and out, which
       by rule leaves the ball with the receiving team at that spot.
    """
    tail = desc[desc.find("MUFFS") :] if "MUFFS" in desc else desc
    if "ouchback" in tail:
        return "touchback"
    match = RECOVERED_BY.search(tail)
    if match:
        return "recovered_by_receiving" if match.group(1) == posteam else "recovered_by_kicking"
    if "and recovers" in tail:
        return "self_recovers"
    if "out of bounds" in tail:
        return "out_of_bounds"
    return "unresolved"


RETAINING = {"touchback", "recovered_by_receiving", "self_recovers", "out_of_bounds"}


def muff_frame(pbp: pl.DataFrame) -> pl.DataFrame:
    """One row per kickoff muff, from the receiving team's point of view.

    On an nflverse kickoff row `posteam` is the *receiving* team, so the muffing
    team is `posteam` and `epa` is already signed from its side (document 20 §3
    verified this against the play text).
    """
    frame = pbp.filter(kickoff_muff_mask())
    dispositions = [
        _disposition(row["desc"], row["posteam"]) for row in frame.iter_rows(named=True)
    ]
    return frame.with_columns(
        pl.Series("disposition", dispositions),
        pl.col("posteam").alias("muffing_team"),
        pl.col("epa").alias("epa_fumbler"),
    ).with_columns(
        pl.col("disposition").is_in(list(RETAINING)).cast(pl.Int8).alias("retained"),
    )


def identification(pbp: pl.DataFrame) -> dict:
    """Can the branch be read off the text? Every rejected row printed by name."""
    print("[1] identification")
    frame = muff_frame(pbp)
    counts = (
        frame.group_by("disposition")
        .agg(pl.len().alias("n"), pl.col("epa").mean().round(3).alias("mean_epa"))
        .sort("n", descending=True)
    )
    print(f"  population: {frame.height} kickoff muffs, 2016-2025")
    print(counts)

    unresolved = frame.filter(pl.col("disposition") == "unresolved")
    print(f"  unresolved dispositions: {unresolved.height}")
    for row in unresolved.iter_rows(named=True):
        print(f"    REJECT {row['season']} {row['game_id']}: {row['desc'][:150]}")

    # The symmetric question. If the scorer only writes MUFFS when the ball is
    # lost, there is a hidden population of retained muffs described some other
    # way, and the whole candidate is measuring the convention again.
    kickoffs = pbp.filter((pl.col("play_type") == "kickoff") & ~kickoff_muff_mask())
    loose_phrase = (
        pl.col("desc").str.contains("RECOVERED by")
        | pl.col("desc").str.contains("recovered by")
        | pl.col("desc").str.contains("touched by")
    )
    residual = kickoffs.filter(loose_phrase & ~pl.col("desc").str.contains("kicks onside"))
    visible = residual.filter(_oob.any_fumble_mask())
    invisible = residual.filter(~_oob.any_fumble_mask())
    print(
        f"\n  kickoffs outside the muff population showing a loose-ball phrase: "
        f"{residual.height}, of which {visible.height} already carry a v1.2 fumble row"
    )
    print(f"  genuinely invisible loose-ball kickoffs: {invisible.height}")
    for row in invisible.iter_rows(named=True):
        print(
            f"    REJECT fumble={row['fumble']} f1={row['fumbled_1_team']} "
            f"epa={row['epa']:+.3f}: {row['desc'][:170]}"
        )

    # Overlap with the shipped v1.2 fumble population: these plays already carry
    # a ledger row and must not gain a second one.
    inside = frame.filter(_oob.any_fumble_mask())
    print(f"\n  already inside the v1.2 fumble population: {inside.height}")
    print(inside.group_by("disposition").agg(pl.len().alias("n")).sort("n", descending=True))
    outside = frame.filter(~_oob.any_fumble_mask())
    print(
        f"  outside it: {outside.height}, of which retained "
        f"{int(outside['retained'].sum())} ({outside['retained'].mean():.1%})"
    )

    return {
        "population": frame.height,
        "dispositions": counts.to_dicts(),
        "unresolved": unresolved.height,
        "residual_loose_ball_kickoffs": residual.height,
        "residual_already_visible": visible.height,
        "residual_invisible": invisible.height,
        "inside_v12": inside.height,
        "inside_v12_retained": int(inside["retained"].sum()),
        "outside_v12": outside.height,
        "outside_v12_retained": int(outside["retained"].sum()),
        "retention_rate_all": float(frame["retained"].mean()),
        "retention_rate_visible_only": float(inside["retained"].mean()),
        "by_season": frame.group_by("season")
        .agg(pl.len().alias("n"), pl.col("retained").sum().alias("k"))
        .sort("season")
        .to_dicts(),
    }


# --------------------------------------------------------------------------
# the widened population and its class table
# --------------------------------------------------------------------------


def widened_frame(pbp: pl.DataFrame) -> pl.DataFrame:
    """v1.2's fumble population plus the kickoff muffs it cannot see.

    The muff rows are appended with `fumble_class` fixed to `kickoff/muff`, and
    the 18 muffs already inside v1.2 are *moved* into that class rather than
    duplicated — the join key is the play, and the muff frame wins.
    """
    v12 = _oob.widened_frame(pbp)
    muffs = muff_frame(pbp)
    muff_keys = set(zip(muffs["game_id"].to_list(), muffs["play_id"].to_list(), strict=True))

    kept = v12.filter(
        ~pl.struct(["game_id", "play_id"]).map_elements(
            lambda row: (row["game_id"], row["play_id"]) in muff_keys, return_dtype=pl.Boolean
        )
    )
    muff_rows = muffs.select(
        pl.col("game_id"),
        pl.col("play_id"),
        pl.col("season"),
        pl.col("home_team"),
        pl.col("posteam"),
        pl.col("muffing_team").alias("fumbled_1_team"),
        pl.col("epa_fumbler"),
        pl.col("retained"),
        pl.lit(False).alias("out_of_bounds"),
        pl.lit("kickoff/muff").alias("fumble_class"),
    )
    return pl.concat([kept.select(muff_rows.columns), muff_rows], how="vertical_relaxed")


def class_table(pbp: pl.DataFrame) -> dict:
    """v1.2's class table beside the widened one."""
    print("\n[2] class table")
    incumbent = _oob.fit_widened_baseline(_oob.widened_frame(pbp))
    widened = _oob.fit_widened_baseline(widened_frame(pbp))
    print("  v1.2 (shipped):")
    print(incumbent)
    print("  widened, with kickoff/muff split out:")
    print(widened)
    return {
        "incumbent": incumbent.to_dicts(),
        "widened": widened.to_dicts(),
    }


# --------------------------------------------------------------------------
# entity spread
# --------------------------------------------------------------------------


def entity_power(pbp: pl.DataFrame) -> dict:
    """Does full neutralization survive the widening? Same instrument as §29.

    Two grains are run and both are reported:

    * the **whole fumble component**, which is what `w` governs and what
      document 18 measured;
    * the **kickoff/muff class alone**, which is the new population and which is
      expected to be badly underpowered at roughly one event per team-season.
      Reporting the second is the point: it is the number Gate F-5 exists for.
    """
    print("\n[3] entity spread")
    reports = {}
    widened = widened_frame(pbp)
    for name, frame in [
        ("fumble retention, widened with kickoff muffs", widened),
        ("fumble retention, v1.2 (incumbent)", _oob.widened_frame(pbp)),
        ("kickoff/muff class alone", widened.filter(pl.col("fumble_class") == "kickoff/muff")),
    ]:
        counts = (
            frame.group_by(["season", "fumbled_1_team"])
            .agg(pl.len().alias("n"), pl.col("retained").sum().cast(pl.Int64).alias("k"))
            .drop_nulls()
            .sort(["season", "fumbled_1_team"])
        )
        reports[name] = _coinflips.power_table(name, counts, RANDOM_SEED)
    return reports


# --------------------------------------------------------------------------
# the materiality floor
# --------------------------------------------------------------------------


def materiality_floor(pbp: pl.DataFrame) -> dict:
    """v1.2's own median 89% DTW half-width on the games a kickoff muff touches.

    This is the bar the fit will have to clear. It is a property of the shipped
    simulator alone — no treatment arm is run here, so Gate F-3's statistic
    stays unseen until the pre-registration is committed.
    """
    print("\n[4] materiality floor")
    fumble_baseline = fit_fumble_baseline(pbp)
    fg_baseline = fit_fg_baseline(pbp)
    xp_baseline = fit_xp_baseline(pbp)
    games = build_game_table(pbp)
    slope = points_per_epa(games.drop_nulls("margin"))
    with (paths.RESEARCH_OUTPUT_DIR / "fg_weather_summary.json").open() as handle:
        centres = json.load(handle)["centres"]
    fg_model = FieldGoalModel.from_posterior(
        paths.RESEARCH_OUTPUT_DIR / "trace_fg_weather.nc",
        wind_centre=centres["wind"],
        temp_centre=centres["temp"],
    )

    muff_games = sorted(set(muff_frame(pbp)["game_id"].to_list()))
    margins = dict(zip(games["game_id"], games["margin"], strict=True))
    half_widths = []
    for game_id, group in pbp.filter(pl.col("game_id").is_in(muff_games)).group_by("game_id"):
        game_id = game_id[0] if isinstance(game_id, tuple) else game_id
        actual = margins.get(game_id)
        if actual is None:
            continue
        rng_fg = np.random.default_rng(RANDOM_SEED + 1)
        rng_xp = np.random.default_rng(RANDOM_SEED + 2)
        rng_coin = np.random.default_rng(RANDOM_SEED + 3)
        rng_fum = np.random.default_rng(RANDOM_SEED)
        events = [
            *fumble_events(group, fumble_baseline, SIM_POSTERIOR_DRAWS, rng_fum),
            *field_goal_events(group, fg_baseline, fg_model, SIM_POSTERIOR_DRAWS, rng_fg),
            *extra_point_events(group, xp_baseline, fg_model, SIM_POSTERIOR_DRAWS, rng_xp),
        ]
        if not events:
            continue
        _, per_draw = bootstrap_margins(events, actual, slope, SIM_COIN_DRAWS, rng_coin)
        low, high = np.percentile(per_draw, [5.5, 94.5])
        half_widths.append(float((high - low) / 2))

    floor = float(np.median(half_widths))
    print(f"  games touched by a kickoff muff: {len(muff_games)}")
    print(f"  v1.2 median 89% DTW half-width on them: {floor * 100:.4f} pp")
    return {
        "muff_games": len(muff_games),
        "games_scored": len(half_widths),
        "median_half_width": floor,
        "points_per_epa": slope,
    }


def main() -> None:
    pbp = load_pbp(PBP_SEASONS, columns=MUFF_COLUMNS)
    report = {
        "identification": identification(pbp),
        "class_table": class_table(pbp),
        "entity_power": entity_power(pbp),
        "materiality_floor": materiality_floor(pbp),
    }
    out = paths.RESEARCH_OUTPUT_DIR / "36_kickoff_muff_power.json"
    with out.open("w") as handle:
        json.dump(report, handle, indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
