"""Phase 6 candidate 2 — blocked-kick aftermath: the Gate B-4 / B-5 / B-6 fit.

Runs the gates `docs/research/25-blocked-kick-aftermath.md` §5 committed at
`0427bd1`, before this file existed:

* **B-4** — does a blocked-kick component move the 378 games that carry one by
  more than v1.2's own 1.4392 pp median half-width on them?
* **B-5** — does the ledger still sum, with no play carrying both a blocked-kick
  row and a fumble row?
* **B-6** — is the B-4 verdict identical at `w` = 0.00, 0.25 and 0.50?

There is deliberately no entity-spread gate: §4a measured the power to resolve
it at 0.177.

    uv run python research/39_blocked_kicks.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_design = import_module("38_blocked_kick_power")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.components import (  # noqa: E402
    _fumble_frame,
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

RANDOM_SEED = _design.RANDOM_SEED
POSTERIOR_DRAWS = _design.SIM_POSTERIOR_DRAWS
COIN_DRAWS = _design.SIM_COIN_DRAWS
REDRAWS = 8  # docs/research/25 §5e

GATE_B4_FLOOR_PP = 1.4392  # §4b
SENSITIVITY_W = (0.00, 0.25, 0.50)  # §5g, all three gated


def team_season_rates(frame: pl.DataFrame) -> dict[tuple[int, str], float]:
    counts = frame.group_by(["season", "kicking_team"]).agg(pl.col("retained").mean().alias("own"))
    return {
        (int(row["season"]), row["kicking_team"]): float(row["own"])
        for row in counts.iter_rows(named=True)
    }


def blocked_kick_events(
    plays: pl.DataFrame,
    table: pl.DataFrame,
    own_rates: dict,
    w: float,
    n_draws: int,
    rng: np.random.Generator,
) -> list[LuckEvent]:
    """The blocked-kick component's ledger rows at trust dial `w`."""
    kicks = _design.eligible_frame(plays).join(
        table.select("kick_class", "n", "p_retain", "swing_value"), on="kick_class", how="left"
    )
    events = []
    for row in kicks.iter_rows(named=True):
        if row["p_retain"] is None or row["swing_value"] is None:
            continue
        draws = _class_rate_draws(float(row["n"]), float(row["p_retain"]), n_draws, rng)
        if w:
            own = own_rates.get((int(row["season"]), row["kicking_team"]))
            if own is not None:
                draws = w * own + (1.0 - w) * draws
        home_sign = 1.0 if row["kicking_team"] == row["home_team"] else -1.0
        events.append(
            LuckEvent(
                play_id=float(row["play_id"]),
                component="blocked_kick",
                event_class=row["kick_class"],
                charged_team=row["kicking_team"],
                actual=float(row["retained"]),
                expected_draws=draws,
                swing=float(row["swing_value"]) * home_sign,
            )
        )
    return events


def ledger_checks(pbp: pl.DataFrame) -> dict:
    """Gate B-5's first half."""
    eligible = _design.eligible_frame(pbp).select("game_id", "play_id")
    fumbles = _fumble_frame(pbp).select("game_id", "play_id")
    shared = eligible.join(fumbles, on=["game_id", "play_id"], how="semi")
    duplicates = eligible.group_by(["game_id", "play_id"]).len().filter(pl.col("len") > 1)
    print(
        f"\n[B-5] eligible blocked kicks that are also fumble plays: {shared.height}; "
        f"appearing twice: {duplicates.height}"
    )
    for row in shared.iter_rows(named=True):
        print(f"    SHARED {row['game_id']} play {row['play_id']}")
    return {
        "eligible": eligible.height,
        "shared_with_fumble_population": shared.height,
        "duplicate_rows": duplicates.height,
        "pass": bool(shared.height == 0 and duplicates.height == 0),
    }


def impact(pbp: pl.DataFrame) -> dict:
    """B-4 and B-6, on the games that carry an eligible blocked kick."""
    print("\nfitting baselines ...")
    frame = _design.eligible_frame(pbp)
    table = _design.class_table(pbp)
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
    touched = sorted(set(frame["game_id"].to_list()))
    subset = pbp.filter(pl.col("game_id").is_in(touched))
    print(f"\nsimulating {len(touched):,} games that carry a blocked kick ...")

    def measure(seed: int) -> pl.DataFrame:
        rows = []
        for game_id, group in subset.group_by("game_id"):
            game_id = game_id[0] if isinstance(game_id, tuple) else game_id
            actual = margins.get(game_id)
            if actual is None:
                continue

            def arm(extra: list[LuckEvent], plays: pl.DataFrame, margin: float) -> tuple:
                rng_fumble = np.random.default_rng(seed + 1)
                rng_fg = np.random.default_rng(seed + 2)
                rng_xp = np.random.default_rng(seed + 3)
                rng_coin = np.random.default_rng(seed + 4)
                events = [
                    *extra,
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
                    blocked_kick_events(group, table, own_rates, w, POSTERIOR_DRAWS, rng),
                    group,
                    actual,
                )
                row[f"dtw_w{w:.2f}"] = dtw
                row[f"deserved_w{w:.2f}"] = deserved
            rows.append(row)
        return pl.DataFrame(rows)

    def summarise(scored: pl.DataFrame) -> dict:
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
                "passes_floor": bool(float(delta.median()) * 100 >= GATE_B4_FLOOR_PP),
            }
        return per_w

    scored = measure(RANDOM_SEED)
    per_w = summarise(scored)
    observed_floor = float(scored["half_width_v12"].median()) * 100
    print(
        f"\n[B-4] pre-registered floor {GATE_B4_FLOOR_PP:.4f} pp; "
        f"this run's v1.2 median half-width {observed_floor:.4f} pp"
    )
    for w in SENSITIVITY_W:
        stats = per_w[f"{w:.2f}"]
        print(
            f"  w = {w:.2f}: median |dDTW| {stats['median_abs_delta_dtw_pp']:.3f} pp, "
            f"mean {stats['mean_abs_delta_dtw_pp']:.3f} pp, "
            f"max {stats['max_abs_delta_dtw_pp']:.2f} pp, flips {stats['side_flips']}, "
            f"median |d deserved margin| {stats['median_abs_delta_deserved_margin']:.3f} pts "
            f"-> {'PASS' if stats['passes_floor'] else 'FAIL'}"
        )

    near = [
        w
        for w in SENSITIVITY_W
        if abs(per_w[f"{w:.2f}"]["median_abs_delta_dtw_pp"] - GATE_B4_FLOOR_PP) < 0.1
    ]
    stability = {}
    if near:
        print(f"\n  {[f'{w:.2f}' for w in near]} within 0.1 pp of the floor; re-drawing {REDRAWS}x")
        redrawn = [measure(RANDOM_SEED + 100 * (i + 1)) for i in range(REDRAWS)]
        for w in SENSITIVITY_W:
            medians = [
                float((f[f"dtw_w{w:.2f}"] - f["dtw_v12"]).abs().median()) * 100 for f in redrawn
            ]
            passes = sum(m >= GATE_B4_FLOOR_PP for m in medians)
            stability[f"{w:.2f}"] = {"min": min(medians), "max": max(medians), "passes": passes}
            print(
                f"    w = {w:.2f}: {min(medians):.3f} - {max(medians):.3f} pp, "
                f"passes {passes}/{REDRAWS}"
            )

    verdicts = {per_w[f"{w:.2f}"]["passes_floor"] for w in SENSITIVITY_W}
    return {
        "games": scored.height,
        "pre_registered_floor_pp": GATE_B4_FLOOR_PP,
        "observed_floor_pp": observed_floor,
        "points_per_epa": slope,
        "classes": table.to_dicts(),
        "by_w": per_w,
        "stability": stability,
        "b4_pass": per_w["0.00"]["passes_floor"],
        "b6_pass": len(verdicts) == 1,
    }


def main() -> None:
    pbp = load_pbp(PBP_SEASONS, columns=_design.BLOCK_COLUMNS)
    checks = ledger_checks(pbp)
    result = impact(pbp)
    payload = {"gate_b5": checks, "impact": result}
    out = paths.RESEARCH_OUTPUT_DIR / "39_blocked_kicks.json"
    with out.open("w") as handle:
        json.dump(payload, handle, indent=2, default=float)
    print(f"\nwrote {out}")

    verdict = "SHIP as v1.3, pending the maintainer's approval"
    if not checks["pass"]:
        verdict = "SHIP NOTHING — the ledger does not sum"
    elif not result["b4_pass"]:
        verdict = "SHIP NOTHING — measured and immaterial"
    elif not result["b6_pass"]:
        verdict = "SHIP NOTHING — verdict depends on an unreadable dial"
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
