"""Phase 5 candidate 3 — fumbles out of bounds: power and impact.

Runs **before** `docs/research/18-fumble-out-of-bounds.md` fixes any threshold.

The incumbent fumble component conditions on the ball being recovered, which
silently books "it skipped out of bounds" as a deserved outcome. This candidate
widens the branch from *who recovered it* to **did the fumbling team end up with
the ball**, with out of bounds as one way of keeping it. Three questions:

1. **What does widening the branch do to the class table** — the rates, the
   branch EPA means, and the swing each class carries?
2. **Is the entity spread on the wider branch still negligible**, i.e. does full
   neutralization survive? Measured with the exact grid instrument of document
   09 §4, at the real denominators.
3. **Does it change anything?** The shipped component and the widened one are
   run through the same bootstrap on the same games, with the field-goal and
   extra-point draws held identical so the difference is the fumble change and
   nothing else.

    uv run python research/29_fumble_oob_power.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_coinflips = import_module("12_coinflips_power")
_power = import_module("25_overtime_power")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.components import (  # noqa: E402
    _fumble_frame,
    build_game_table,
    fit_fg_baseline,
    fit_fumble_baseline,
    fit_xp_baseline,
    live_fumble_mask,
)
from nfl_simulator.fg_model import FieldGoalModel  # noqa: E402
from nfl_simulator.ingest import PBP_SEASONS, load_pbp  # noqa: E402
from nfl_simulator.simulator import (  # noqa: E402
    LuckEvent,
    _class_rate_draws,
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

# `fumble_out_of_bounds` is already in ANALYSIS_COLUMNS; nothing extra is needed.
FUMBLE_COLUMNS = list(_power.SIM_COLUMNS)


# --------------------------------------------------------------------------
# the widened branch
# --------------------------------------------------------------------------


def any_fumble_mask() -> pl.Expr:
    """Every fumble with an identified fumbling team, recovered or not.

    The incumbent `live_fumble_mask` additionally requires a recovery team,
    which drops the out-of-bounds fumbles. This is the same population plus
    those.
    """
    return (pl.col("fumble") == 1) & pl.col("fumbled_1_team").is_not_null()


def widened_frame(pbp: pl.DataFrame) -> pl.DataFrame:
    """All fumbles, annotated with `retained` instead of `recovered_own`.

    `retained` is 1 when the fumbling team still has the ball afterwards, which
    happens either by recovering it or by the ball crossing the sideline. The 11
    plays carrying both an out-of-bounds flag and a recovery team are treated as
    recoveries: a named recovering team is the more specific fact.
    """
    frame = pbp.filter(any_fumble_mask()).with_columns(
        pl.when(pl.col("fumbled_1_team") == pl.col("posteam"))
        .then(pl.col("epa"))
        .otherwise(-pl.col("epa"))
        .alias("epa_fumbler"),
        (pl.col("aborted_play") == 1).fill_null(False).alias("is_aborted"),
        pl.col("fumble_recovery_1_team").is_not_null().alias("was_recovered"),
        (pl.col("fumble_out_of_bounds") == 1).fill_null(False).alias("out_of_bounds"),
    )
    return frame.with_columns(
        pl.when(pl.col("was_recovered"))
        .then((pl.col("fumble_recovery_1_team") == pl.col("fumbled_1_team")).cast(pl.Int8))
        .when(pl.col("out_of_bounds"))
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(None)
        .alias("retained"),
        pl.concat_str(
            [
                pl.col("play_type").fill_null("other"),
                pl.when(pl.col("is_aborted")).then(pl.lit("aborted")).otherwise(pl.lit("live")),
            ],
            separator="/",
        ).alias("fumble_class"),
    ).drop_nulls("retained")


def fit_widened_baseline(frame: pl.DataFrame) -> pl.DataFrame:
    """Per-class retention probability and branch EPA means, widened population.

    Mirrors `fit_fumble_baseline` exactly, including the thin-class pooling, so
    the only difference between the two tables is the population they are fitted
    on.
    """
    table = (
        frame.group_by("fumble_class")
        .agg(
            pl.len().alias("n"),
            pl.col("retained").mean().alias("p_own"),
            pl.col("out_of_bounds").mean().alias("p_out_of_bounds"),
            pl.col("epa_fumbler").filter(pl.col("retained") == 1).mean().alias("epa_own"),
            pl.col("epa_fumbler").filter(pl.col("retained") == 0).mean().alias("epa_lost"),
        )
        .sort("n", descending=True)
    )
    pooled_p = frame["retained"].mean()
    pooled_own = frame.filter(pl.col("retained") == 1)["epa_fumbler"].mean()
    pooled_lost = frame.filter(pl.col("retained") == 0)["epa_fumbler"].mean()
    return table.with_columns(
        pl.when(pl.col("n") >= MIN_CLASS_SIZE)
        .then(pl.col("p_own"))
        .otherwise(pooled_p)
        .alias("p_own"),
        pl.col("epa_own").fill_nan(None).fill_null(pooled_own),
        pl.col("epa_lost").fill_nan(None).fill_null(pooled_lost),
    ).with_columns(swing_value=pl.col("epa_own") - pl.col("epa_lost"))


def widened_fumble_events(
    plays: pl.DataFrame, table: pl.DataFrame, n_draws: int, rng: np.random.Generator
) -> list[LuckEvent]:
    """The widened component's ledger rows, built like `simulator.fumble_events`."""
    fumbles = widened_frame(plays).join(
        table.select("fumble_class", "n", "p_own", "swing_value"), on="fumble_class", how="left"
    )
    events = []
    for row in fumbles.iter_rows(named=True):
        if row["p_own"] is None or row["swing_value"] is None:
            continue
        home_sign = 1.0 if row["fumbled_1_team"] == row["home_team"] else -1.0
        events.append(
            LuckEvent(
                play_id=float(row["play_id"]),
                component="fumble",
                event_class=row["fumble_class"],
                charged_team=row["fumbled_1_team"],
                realized=float(row["retained"]),
                expected_draws=_class_rate_draws(
                    float(row["n"]), float(row["p_own"]), n_draws, rng
                ),
                swing=float(row["swing_value"]) * home_sign,
            )
        )
    return events


# --------------------------------------------------------------------------


def class_comparison(pbp: pl.DataFrame) -> dict:
    """The incumbent class table beside the widened one."""
    incumbent = fit_fumble_baseline(pbp.filter(live_fumble_mask())).table
    widened = fit_widened_baseline(widened_frame(pbp))
    merged = (
        incumbent.select(
            "fumble_class",
            pl.col("n").alias("n_live"),
            pl.col("p_own").alias("p_live"),
            pl.col("swing_value").alias("swing_live"),
        )
        .join(
            widened.select(
                "fumble_class",
                pl.col("n").alias("n_all"),
                pl.col("p_own").alias("p_all"),
                pl.col("p_out_of_bounds"),
                pl.col("swing_value").alias("swing_all"),
            ),
            on="fumble_class",
            how="full",
            coalesce=True,
        )
        .sort("n_all", descending=True)
    )
    return {"rows": merged.to_dicts()}


def entity_power(pbp: pl.DataFrame) -> dict:
    """Is the entity spread on the widened branch still negligible?

    Same instrument, same reference and same 0.80 minimum power as document 09
    §4-§5, run at the widened denominators. The comparison arm is the incumbent
    population, so the question 'did widening cost or buy resolution?' has an
    answer rather than an assumption.
    """
    reports = {}
    for name, frame, success in [
        (
            "fumble retention, all fumbles",
            widened_frame(pbp),
            pl.col("retained"),
        ),
        (
            "fumble recovery, live only (incumbent)",
            _fumble_frame(pbp.filter(live_fumble_mask())),
            pl.col("retained"),
        ),
    ]:
        counts = (
            frame.group_by(["season", "fumbled_1_team"])
            .agg(pl.len().alias("n"), success.sum().cast(pl.Int64).alias("k"))
            .drop_nulls()
            .sort(["season", "fumbled_1_team"])
        )
        reports[name] = _coinflips.power_table(name, counts, RANDOM_SEED)
    return reports


def impact(pbp: pl.DataFrame) -> dict:
    """Shipped component against the widened one, everything else held fixed.

    Field-goal and extra-point draws are generated from their own seeded
    generators in both arms, so the two bootstraps differ only in the fumble
    rows. A whole-pipeline rerun would have mixed the change with Monte Carlo
    noise from components that did not change.
    """
    print("  fitting baselines ...")
    fumble_baseline = fit_fumble_baseline(pbp.filter(live_fumble_mask()))
    widened = fit_widened_baseline(widened_frame(pbp))
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

    touched = set(pbp.filter(any_fumble_mask())["game_id"].to_list())
    # The games where a *new* ledger row appears, as opposed to the games where
    # only the refitted class rates move. The materiality floor is read on the
    # first population and the dilution across the second is reported beside it.
    oob_games = set(
        pbp.filter(any_fumble_mask() & (pl.col("fumble_out_of_bounds") == 1))["game_id"].to_list()
    )
    margins = dict(zip(games["game_id"], games["margin"], strict=True))

    rows = []
    for game_id, group in pbp.filter(pl.col("game_id").is_in(touched)).group_by("game_id"):
        game_id = game_id[0] if isinstance(game_id, tuple) else game_id
        actual = margins.get(game_id)
        if actual is None:
            continue

        def arm(
            fumble_rows: list[LuckEvent], plays: pl.DataFrame, actual_margin: float
        ) -> tuple[float, float, float]:
            rng_fg = np.random.default_rng(RANDOM_SEED + 1)
            rng_xp = np.random.default_rng(RANDOM_SEED + 2)
            rng_coin = np.random.default_rng(RANDOM_SEED + 3)
            events = [
                *fumble_rows,
                *field_goal_events(plays, fg_baseline, fg_model, SIM_POSTERIOR_DRAWS, rng_fg),
                *extra_point_events(plays, xp_baseline, fg_model, SIM_POSTERIOR_DRAWS, rng_xp),
            ]
            if not events:
                return (1.0 if actual_margin > 0 else 0.0), actual_margin, 0.0, 0.0
            drawn, per_draw = bootstrap_margins(
                events, actual_margin, slope, SIM_COIN_DRAWS, rng_coin
            )
            luck = sum(event.to_entry().luck_epa for event in events)
            low, high = np.percentile(per_draw, [5.5, 94.5])
            return (
                float((drawn > 0).mean()),
                float(actual_margin - luck * slope),
                float(luck),
                float((high - low) / 2),
            )

        rng_old = np.random.default_rng(RANDOM_SEED)
        rng_new = np.random.default_rng(RANDOM_SEED)
        dtw_old, deserved_old, luck_old, half_width_old = arm(
            fumble_events(
                group.filter(live_fumble_mask()), fumble_baseline, SIM_POSTERIOR_DRAWS, rng_old
            ),
            group,
            actual,
        )
        dtw_new, deserved_new, luck_new, _ = arm(
            widened_fumble_events(group, widened, SIM_POSTERIOR_DRAWS, rng_new), group, actual
        )
        rows.append(
            {
                "game_id": game_id,
                "dtw_old": dtw_old,
                "dtw_new": dtw_new,
                "deserved_old": deserved_old,
                "deserved_new": deserved_new,
                "luck_old": luck_old,
                "luck_new": luck_new,
                "half_width_old": half_width_old,
                "has_oob": game_id in oob_games,
            }
        )
        if len(rows) % 400 == 0:
            print(f"    {len(rows)} games")

    table = pl.DataFrame(rows).with_columns(
        delta=pl.col("dtw_new") - pl.col("dtw_old"),
        flipped=((pl.col("dtw_old") - 0.5) * (pl.col("dtw_new") - 0.5)) < 0,
    )
    oob = table.filter("has_oob")
    return {
        "games_touched": table.height,
        "median_abs_delta_dtw": float(table["delta"].abs().median()),
        "mean_abs_delta_dtw": float(table["delta"].abs().mean()),
        "max_abs_delta_dtw": float(table["delta"].abs().max()),
        "side_flips": int(table["flipped"].sum()),
        "median_abs_delta_deserved_margin": float(
            (table["deserved_new"] - table["deserved_old"]).abs().median()
        ),
        "oob_games": oob.height,
        "oob_median_abs_delta_dtw": float(oob["delta"].abs().median()),
        "oob_mean_abs_delta_dtw": float(oob["delta"].abs().mean()),
        "oob_side_flips": int(oob["flipped"].sum()),
        "oob_median_abs_delta_deserved_margin": float(
            (oob["deserved_new"] - oob["deserved_old"]).abs().median()
        ),
        "oob_floor_median_half_width": float(oob["half_width_old"].median()),
        "all_floor_median_half_width": float(table["half_width_old"].median()),
        "points_per_epa": slope,
    }


def main() -> None:
    pbp = load_pbp(PBP_SEASONS, columns=FUMBLE_COLUMNS)
    frame = widened_frame(pbp)
    print(
        f"fumbles with an identified fumbling team: {frame.height:,}; "
        f"out of bounds: {int(frame['out_of_bounds'].sum()):,}"
    )

    print("\n[1] class table, incumbent beside widened")
    classes = class_comparison(pbp)
    for row in classes["rows"]:
        print(
            f"    {row['fumble_class']:<18} live n={row['n_live']} p={row['p_live']} "
            f"swing={row['swing_live']}"
        )
        print(
            f"    {'':<18} all  n={row['n_all']} p={row['p_all']} "
            f"p_oob={row['p_out_of_bounds']} swing={row['swing_all']}"
        )

    print("\n[2] entity spread on the widened branch")
    power = entity_power(pbp)

    print("\n[3] impact, everything but the fumble rows held fixed")
    imp = impact(pbp)
    print(
        f"    {imp['games_touched']:,} games touched; median |dDTW| "
        f"{100 * imp['median_abs_delta_dtw']:.2f} pp, mean {100 * imp['mean_abs_delta_dtw']:.2f} pp, "
        f"max {100 * imp['max_abs_delta_dtw']:.2f} pp"
    )
    print(
        f"    side flips {imp['side_flips']}; median |d deserved margin| "
        f"{imp['median_abs_delta_deserved_margin']:.4f} points"
    )
    print(
        f"    on the {imp['oob_games']:,} games carrying an out-of-bounds fumble: median |dDTW| "
        f"{100 * imp['oob_median_abs_delta_dtw']:.2f} pp, mean "
        f"{100 * imp['oob_mean_abs_delta_dtw']:.2f} pp, side flips {imp['oob_side_flips']}"
    )
    print(
        f"    incumbent median 89% DTW half-width: "
        f"{100 * imp['oob_floor_median_half_width']:.2f} pp on those games, "
        f"{100 * imp['all_floor_median_half_width']:.2f} pp across all touched games"
    )

    payload = {
        "random_seed": RANDOM_SEED,
        "fumbles": frame.height,
        "out_of_bounds": int(frame["out_of_bounds"].sum()),
        "classes": classes,
        "entity_power": power,
        "impact": imp,
    }
    out = paths.RESEARCH_OUTPUT_DIR / "29_fumble_oob_power.json"
    with out.open("w") as handle:
        json.dump(payload, handle, indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
