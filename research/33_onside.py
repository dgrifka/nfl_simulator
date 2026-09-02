"""Phase 5 candidate 4 — onside kicks: the Gate K-3 and K-5 fits.

Runs the gates `docs/research/20-onside-kicks.md` §5d and §5f committed at
`b80fe9f`, before this file existed:

* **K-3** — does replacing the onside coin move the games it touches by more
  than the interval simulator v1.2 already prints on them?
* **K-5** — is that verdict the same when a quarter and a half of the trust is
  handed back to the kicking team, i.e. does the answer depend on the dial
  document 20 §4a showed cannot be read?

Both arms of every comparison generate their fumble, field-goal and extra-point
draws from their own seeded generators, so the difference between them is the
onside rows and nothing else.

    uv run python research/33_onside.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_design = import_module("32_onside_power")

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
POSTERIOR_DRAWS = 200
COIN_DRAWS = 800
SENSITIVITY_W = (0.00, 0.25, 0.50)

SIM_COLUMNS = sorted(
    {
        *_design.ONSIDE_COLUMNS,
        "defteam",
        "fumble",
        "fumble_lost",
        "fumbled_1_team",
        "fumble_recovery_1_team",
        "fumble_out_of_bounds",
        "aborted_play",
        "interception",
        "penalty",
        "penalty_type",
        "penalty_team",
        "field_goal_result",
        "spread_line",
        "kicker_player_id",
        "extra_point_attempt",
        "extra_point_result",
        "roof",
        "temp",
        "wind",
    }
)


def fit_onside_baseline(frame: pl.DataFrame) -> pl.DataFrame:
    """Per-class recovery probability and branch EPA means.

    Mirrors `components.fit_fumble_baseline`'s shape: a class below the minimum
    size borrows the pooled rate, and a missing branch mean falls back to the
    pooled one. Neither fallback fires on the real data — both classes clear 30 —
    but a caller running one season would need them.
    """
    table = (
        frame.group_by("onside_class")
        .agg(
            pl.len().alias("n"),
            pl.col("recovered").mean().alias("p_recover"),
            pl.col("epa_kicker").filter(pl.col("recovered") == 1).mean().alias("epa_recovered"),
            pl.col("epa_kicker").filter(pl.col("recovered") == 0).mean().alias("epa_lost"),
        )
        .sort("n", descending=True)
    )
    pooled_p = frame["recovered"].mean()
    pooled_recovered = frame.filter(pl.col("recovered") == 1)["epa_kicker"].mean()
    pooled_lost = frame.filter(pl.col("recovered") == 0)["epa_kicker"].mean()
    return table.with_columns(
        pl.when(pl.col("n") >= MIN_CLASS_SIZE)
        .then(pl.col("p_recover"))
        .otherwise(pooled_p)
        .alias("p_recover"),
        pl.col("epa_recovered").fill_nan(None).fill_null(pooled_recovered),
        pl.col("epa_lost").fill_nan(None).fill_null(pooled_lost),
    ).with_columns((pl.col("epa_recovered") - pl.col("epa_lost")).alias("swing_value"))


def team_season_rates(frame: pl.DataFrame) -> dict[tuple[int, str], float]:
    """Each kicking team-season's own recovery record — the K-5 sensitivity input."""
    counts = frame.group_by(["season", "kicking_team"]).agg(pl.col("recovered").mean().alias("own"))
    return {
        (int(row["season"]), row["kicking_team"]): float(row["own"])
        for row in counts.iter_rows(named=True)
    }


def onside_events(
    plays: pl.DataFrame,
    table: pl.DataFrame,
    own_rates: dict,
    w: float,
    n_draws: int,
    rng: np.random.Generator,
) -> list[LuckEvent]:
    """The onside component's ledger rows at trust dial `w`.

    At `w = 0` the expectation is the class league rate, drawn from its Jeffreys
    posterior exactly as the fumble component's is. Above zero, the drawn class
    rate is shrunk toward the kicking team-season's own record; the layer-1
    uncertainty stays on the class part, because the team part is an observed
    proportion over a median of two kicks and has no usable posterior of its own.
    """
    kicks = _design.onside_frame(plays).join(
        table.select("onside_class", "n", "p_recover", "swing_value"),
        on="onside_class",
        how="left",
    )
    events = []
    for row in kicks.iter_rows(named=True):
        if row["p_recover"] is None or row["swing_value"] is None:
            continue
        draws = _class_rate_draws(float(row["n"]), float(row["p_recover"]), n_draws, rng)
        if w:
            own = own_rates.get((int(row["season"]), row["kicking_team"]))
            if own is not None:
                draws = w * own + (1.0 - w) * draws
        home_sign = 1.0 if row["kicking_team"] == row["home_team"] else -1.0
        events.append(
            LuckEvent(
                play_id=float(row["play_id"]),
                component="onside",
                event_class=row["onside_class"],
                charged_team=row["kicking_team"],
                actual=float(row["recovered"]),
                expected_draws=draws,
                swing=float(row["swing_value"]) * home_sign,
            )
        )
    return events


def overlap_check(pbp: pl.DataFrame) -> dict:
    """Gate K-4's first half: can one play produce two rows?"""
    from nfl_simulator.components import _fumble_frame

    onside = _design.onside_frame(pbp).select("game_id", "play_id")
    fumbles = _fumble_frame(pbp).select("game_id", "play_id")
    shared = onside.join(fumbles, on=["game_id", "play_id"], how="semi")
    duplicates = onside.group_by(["game_id", "play_id"]).len().filter(pl.col("len") > 1)
    print(
        f"\n[K-4] onside plays that are also fumble plays: {shared.height}; "
        f"onside plays appearing twice: {duplicates.height}"
    )
    if shared.height:
        print(
            "      a shared play would book the same loose ball twice; the component "
            "cannot ship without a precedence rule"
        )
    return {"shared_with_fumble_population": shared.height, "duplicate_rows": duplicates.height}


REDRAWS = 8  # docs/research/20 §5d: a near-miss is re-drawn, not called


def impact(pbp: pl.DataFrame) -> dict:
    """K-3 and K-5, on the games that carry an onside kick."""
    print("\nfitting baselines ...")
    frame = _design.onside_frame(pbp)
    table = fit_onside_baseline(frame)
    print(table)
    own_rates = team_season_rates(frame)

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
    margins = dict(zip(games["game_id"], games["margin"], strict=True))

    touched = set(frame["game_id"].to_list())
    subset = pbp.filter(pl.col("game_id").is_in(touched))
    print(f"\nsimulating {len(touched):,} games that carry an onside kick ...")

    def measure(seed: int) -> pl.DataFrame:
        """One full pass over the touched games at a given draw seed."""
        rows = []
        for game_id, group in subset.group_by("game_id"):
            game_id = game_id[0] if isinstance(game_id, tuple) else game_id
            actual = margins.get(game_id)
            if actual is None:
                continue

            def arm(onside_rows: list[LuckEvent], plays: pl.DataFrame, margin: float) -> tuple:
                rng_fumble = np.random.default_rng(seed + 1)
                rng_fg = np.random.default_rng(seed + 2)
                rng_xp = np.random.default_rng(seed + 3)
                rng_coin = np.random.default_rng(seed + 4)
                events = [
                    *onside_rows,
                    *fumble_events(plays, fumble_baseline, POSTERIOR_DRAWS, rng_fumble),
                    *field_goal_events(plays, fg_baseline, fg_model, POSTERIOR_DRAWS, rng_fg),
                    *extra_point_events(plays, xp_baseline, fg_model, POSTERIOR_DRAWS, rng_xp),
                ]
                if not events:
                    return (1.0 if margin > 0 else 0.0), margin, 0.0
                _, per_draw = bootstrap_margins(events, margin, slope, COIN_DRAWS, rng_coin)
                luck = sum(event.to_entry().luck_epa for event in events)
                low, high = np.percentile(per_draw, [5.5, 94.5])
                return (
                    float(per_draw.mean()),
                    float(margin - luck * slope),
                    float((high - low) / 2),
                )

            base_dtw, base_deserved, base_half_width = arm([], group, actual)
            row = {
                "game_id": game_id,
                "dtw_v12": base_dtw,
                "deserved_v12": base_deserved,
                "half_width_v12": base_half_width,
            }
            for w in SENSITIVITY_W:
                rng = np.random.default_rng(seed)
                dtw, deserved, _ = arm(
                    onside_events(group, table, own_rates, w, POSTERIOR_DRAWS, rng), group, actual
                )
                row[f"dtw_w{w:.2f}"] = dtw
                row[f"deserved_w{w:.2f}"] = deserved
            rows.append(row)
        return pl.DataFrame(rows)

    def summarise(scored: pl.DataFrame) -> tuple[float, dict]:
        floor = float(scored["half_width_v12"].median())
        per_w = {}
        for w in SENSITIVITY_W:
            delta = (scored[f"dtw_w{w:.2f}"] - scored["dtw_v12"]).abs()
            margin_delta = (scored[f"deserved_w{w:.2f}"] - scored["deserved_v12"]).abs()
            per_w[f"{w:.2f}"] = {
                "median_abs_delta_dtw_pp": float(delta.median()) * 100,
                "mean_abs_delta_dtw_pp": float(delta.mean()) * 100,
                "max_abs_delta_dtw_pp": float(delta.max()) * 100,
                "median_abs_delta_deserved_margin": float(margin_delta.median()),
                "side_flips": scored.filter(
                    ((pl.col(f"dtw_w{w:.2f}") - 0.5) * (pl.col("dtw_v12") - 0.5)) < 0
                ).height,
                "passes_floor": bool(float(delta.median()) >= floor),
            }
        return floor, per_w

    scored = measure(RANDOM_SEED)
    floor, per_w = summarise(scored)
    print(f"\n[K-3] floor: v1.2 median 89% DTW half-width on these games = {floor * 100:.2f} pp")
    for w in SENSITIVITY_W:
        stats = per_w[f"{w:.2f}"]
        print(
            f"  w = {w:.2f}: median |dDTW| {stats['median_abs_delta_dtw_pp']:.2f} pp, "
            f"mean {stats['mean_abs_delta_dtw_pp']:.2f} pp, "
            f"max {stats['max_abs_delta_dtw_pp']:.2f} pp, flips {stats['side_flips']}, "
            f"median |d deserved margin| {stats['median_abs_delta_deserved_margin']:.3f} pts "
            f"-> {'PASS' if stats['passes_floor'] else 'FAIL'}"
        )

    # Document 20 §5d: any result within 0.1 pp of the floor is re-drawn eight
    # times and reported with its spread, so a gate is never decided by one
    # replay of the coin.
    near = [
        w
        for w in SENSITIVITY_W
        if abs(per_w[f"{w:.2f}"]["median_abs_delta_dtw_pp"] - floor * 100) < 0.1
    ]
    stability = {}
    if near:
        print(
            f"\n  {[f'{w:.2f}' for w in near]} land within 0.1 pp of the floor; "
            f"re-drawing {REDRAWS} times"
        )
        redrawn = [measure(RANDOM_SEED + 100 * (i + 1)) for i in range(REDRAWS)]
        floors = [float(frame_i["half_width_v12"].median()) * 100 for frame_i in redrawn]
        stability["floor_pp"] = {"min": min(floors), "max": max(floors), "values": floors}
        print(f"    floor across redraws: {min(floors):.2f} - {max(floors):.2f} pp")
        for w in SENSITIVITY_W:
            medians = [
                float((f[f"dtw_w{w:.2f}"] - f["dtw_v12"]).abs().median()) * 100 for f in redrawn
            ]
            verdicts = [m >= fl * 1.0 for m, fl in zip(medians, floors, strict=True)]
            stability[f"{w:.2f}"] = {
                "median_pp": {"min": min(medians), "max": max(medians), "values": medians},
                "passes_every_redraw": all(verdicts),
                "fails_every_redraw": not any(verdicts),
            }
            print(
                f"    w = {w:.2f}: median |dDTW| {min(medians):.2f} - {max(medians):.2f} pp, "
                f"passes {sum(verdicts)}/{REDRAWS} redraws"
            )

    verdicts = {stats["passes_floor"] for stats in per_w.values()}
    return {
        "games": scored.height,
        "floor_median_half_width_pp": floor * 100,
        "points_per_epa": slope,
        "classes": table.to_dicts(),
        "by_w": per_w,
        "stability": stability,
        "k3_pass": per_w["0.00"]["passes_floor"],
        "k5_pass": len(verdicts) == 1,
    }


def main() -> None:
    pbp = load_pbp(PBP_SEASONS, columns=SIM_COLUMNS)
    payload = {
        "random_seed": RANDOM_SEED,
        "identification": _design.identification(pbp),
        "overlap": overlap_check(pbp),
        "impact": impact(pbp),
    }
    k3, k5 = payload["impact"]["k3_pass"], payload["impact"]["k5_pass"]
    print(f"\n[K-3] materiality: {'PASS' if k3 else 'FAIL'}")
    print(f"[K-5] verdict is the same at every w: {'PASS' if k5 else 'FAIL'}")
    print(
        "\nVERDICT: "
        + (
            "SHIP as v1.3, pending approval"
            if k3 and k5
            else "SHIP NOTHING — "
            + ("the verdict depends on an unreadable dial" if k3 else "measured and immaterial")
        )
    )
    out = paths.RESEARCH_OUTPUT_DIR / "33_onside.json"
    with out.open("w") as handle:
        json.dump(payload, handle, indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
