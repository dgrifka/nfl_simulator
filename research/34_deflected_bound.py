"""Phase 5 candidate 2b — deflected interceptions as a bound, not a component.

Runs Gate D-1 of `docs/research/22-deflected-int-bound.md` §5, committed at
`6b00e4f` before this file existed: does the ledger impact of neutralizing
deflected interceptions depend on `f`, the unidentified fraction of
pass-defensed incompletions that were live tips?

Two arms, per §2:

* **A — successes only.** The 629 deflected interceptions at `expected = p(f)`.
  The only implementation the data permits, and not the component.
* **B — both branches.** Arm A plus `f · 18,777` pass-defensed incompletions at
  `realized = 0`, drawn at random five times per `f` because which ones are live
  tips is exactly the unidentified fact. **This arm decides.**

    uv run python research/34_deflected_bound.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.components import (  # noqa: E402
    build_game_table,
    fit_fg_baseline,
    fit_fumble_baseline,
    fit_xp_baseline,
)
from nfl_simulator.fg_model import FieldGoalModel  # noqa: E402
from nfl_simulator.ingest import ANALYSIS_COLUMNS, PBP_SEASONS, load_pbp  # noqa: E402
from nfl_simulator.simulator import (  # noqa: E402
    LuckEvent,
    bootstrap_margins,
    extra_point_events,
    field_goal_events,
    fumble_events,
    points_per_epa,
)

RANDOM_SEED = 20260818
POSTERIOR_DRAWS = 200
COIN_DRAWS = 800
F_SWEEP = (0.05, 0.10, 0.25, 0.50, 1.00)
ARM_B_DRAWS = 5
INVARIANCE_BOUND = 1.25  # docs/research/22 §5a

SIM_COLUMNS = sorted(
    {
        *ANALYSIS_COLUMNS,
        "interception_player_id",
        "pass_defense_1_player_id",
        "kicker_player_id",
        "extra_point_attempt",
        "extra_point_result",
        "roof",
        "temp",
        "wind",
    }
)

DEFLECTED = (
    (pl.col("interception") == 1)
    & pl.col("pass_defense_1_player_id").is_not_null()
    & (pl.col("pass_defense_1_player_id") != pl.col("interception_player_id"))
)
PD_INCOMPLETION = (pl.col("interception") != 1) & pl.col("pass_defense_1_player_id").is_not_null()


def p_of_f(n_deflected: int, n_pd: int, f: float) -> float:
    """p(intercepted | live tip) at an assumed live-tip fraction `f`."""
    return n_deflected / (n_deflected + f * n_pd)


def deflection_events(
    plays: pl.DataFrame, p: float, swing: float, chosen: set[tuple[str, float]] | None
) -> list[LuckEvent]:
    """Ledger rows for one game.

    `chosen` is Arm B's drawn subset of pass-defensed incompletions, or None for
    Arm A. `expected` is a constant rather than a vector of posterior draws:
    `p(f)` is a stated assumption about an unidentified quantity, not an
    estimate with a posterior, and dressing it in draws would imply a precision
    that does not exist.
    """
    events = []
    for row in plays.filter(DEFLECTED | PD_INCOMPLETION).iter_rows(named=True):
        intercepted = row["interception"] == 1
        if not intercepted and (
            chosen is None or (row["game_id"], float(row["play_id"])) not in chosen
        ):
            continue
        home_sign = 1.0 if row["defteam"] == row["home_team"] else -1.0
        events.append(
            LuckEvent(
                play_id=float(row["play_id"]),
                component="deflection",
                event_class="deflected pass",
                charged_team=row["defteam"],
                realized=1.0 if intercepted else 0.0,
                expected_draws=np.full(POSTERIOR_DRAWS, p),
                swing=swing * home_sign,
            )
        )
    return events


def main() -> None:
    pbp = load_pbp(PBP_SEASONS, columns=SIM_COLUMNS)
    deflected = pbp.filter(DEFLECTED)
    pd_incompletions = pbp.filter(PD_INCOMPLETION)
    swing = float(-deflected["epa"].mean()) - float(-pd_incompletions["epa"].mean())
    print(
        f"deflected interceptions {deflected.height:,}; pass-defensed incompletions "
        f"{pd_incompletions.height:,}; swing {swing:.4f} EPA (defense's perspective)"
    )
    for f in F_SWEEP:
        print(f"  f = {f:.2f} -> p = {p_of_f(deflected.height, pd_incompletions.height, f):.4f}")

    print("\nfitting baselines ...")
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

    # Every game either arm can touch, simulated once without the component.
    candidates = pbp.filter(DEFLECTED | PD_INCOMPLETION)
    touched = set(candidates["game_id"].to_list())
    subset = pbp.filter(pl.col("game_id").is_in(touched))
    groups = {}
    for game_id, group in subset.group_by("game_id"):
        game_id = game_id[0] if isinstance(game_id, tuple) else game_id
        if margins.get(game_id) is not None:
            groups[game_id] = group
    deflected_games = set(deflected["game_id"].to_list())
    print(f"\n{len(groups):,} games carry a deflection candidate; {len(deflected_games):,} carry")
    print("a deflected interception")

    def arm(game_id: str, rows: list[LuckEvent]) -> tuple[float, float, float]:
        plays = groups[game_id]
        actual = float(margins[game_id])
        rng_fumble = np.random.default_rng(RANDOM_SEED + 1)
        rng_fg = np.random.default_rng(RANDOM_SEED + 2)
        rng_xp = np.random.default_rng(RANDOM_SEED + 3)
        rng_coin = np.random.default_rng(RANDOM_SEED + 4)
        events = [
            *rows,
            *fumble_events(plays, fumble_baseline, POSTERIOR_DRAWS, rng_fumble),
            *field_goal_events(plays, fg_baseline, fg_model, POSTERIOR_DRAWS, rng_fg),
            *extra_point_events(plays, xp_baseline, fg_model, POSTERIOR_DRAWS, rng_xp),
        ]
        if not events:
            return (1.0 if actual > 0 else 0.0), actual, 0.0
        _, per_draw = bootstrap_margins(events, actual, slope, COIN_DRAWS, rng_coin)
        luck = sum(event.to_entry().luck_epa for event in events)
        low, high = np.percentile(per_draw, [5.5, 94.5])
        return float(per_draw.mean()), float(actual - luck * slope), float((high - low) / 2)

    print("\nsimulating the v1.2 baseline arm ...")
    base = {game_id: arm(game_id, []) for game_id in groups}  # (dtw, deserved margin, half-width)

    def summarise(with_component: dict, population: set[str]) -> dict:
        rows = [
            {
                "game_id": game_id,
                "delta": abs(with_component[game_id][0] - base[game_id][0]),
                "margin_delta": abs(with_component[game_id][1] - base[game_id][1]),
                "half_width": base[game_id][2],
                "flipped": (with_component[game_id][0] - 0.5) * (base[game_id][0] - 0.5) < 0,
            }
            for game_id in population
            if game_id in with_component
        ]
        table = pl.DataFrame(rows)
        floor = float(table["half_width"].median())
        median = float(table["delta"].median())
        return {
            "games": table.height,
            "median_abs_delta_dtw_pp": median * 100,
            "mean_abs_delta_dtw_pp": float(table["delta"].mean()) * 100,
            "max_abs_delta_dtw_pp": float(table["delta"].max()) * 100,
            "median_abs_delta_deserved_margin": float(table["margin_delta"].median()),
            "side_flips": int(table["flipped"].sum()),
            "floor_pp": floor * 100,
            "passes_floor": bool(median >= floor),
        }

    pd_keys = [
        (row["game_id"], float(row["play_id"]))
        for row in pd_incompletions.select("game_id", "play_id").iter_rows(named=True)
    ]

    report = {"arm_a": {}, "arm_b": {}, "swing": swing, "points_per_epa": slope}

    print("\n[Arm A] successes only — the buildable implementation")
    for f in F_SWEEP:
        p = p_of_f(deflected.height, pd_incompletions.height, f)
        scored = {
            game_id: arm(game_id, deflection_events(groups[game_id], p, swing, None))
            for game_id in deflected_games
            if game_id in groups
        }
        stats = summarise(scored, deflected_games) | {"p": p, "rows": deflected.height}
        report["arm_a"][f"{f:.2f}"] = stats
        print(
            f"  f = {f:.2f} (p = {p:.4f}): median |dDTW| {stats['median_abs_delta_dtw_pp']:.2f} pp "
            f"vs floor {stats['floor_pp']:.2f} pp, flips {stats['side_flips']} "
            f"-> {'above' if stats['passes_floor'] else 'below'}"
        )

    print("\n[Arm B] both branches — the component the identity requires")
    for f in F_SWEEP:
        p = p_of_f(deflected.height, pd_incompletions.height, f)
        draws = []
        for draw in range(ARM_B_DRAWS):
            rng = np.random.default_rng(RANDOM_SEED + 1000 * draw + int(f * 100))
            take = int(round(f * len(pd_keys)))
            chosen = {pd_keys[i] for i in rng.choice(len(pd_keys), size=take, replace=False)}
            population = deflected_games | {key[0] for key in chosen}
            scored = {
                game_id: arm(game_id, deflection_events(groups[game_id], p, swing, chosen))
                for game_id in population
                if game_id in groups
            }
            draws.append(summarise(scored, population) | {"p": p, "rows": deflected.height + take})
        medians = [d["median_abs_delta_dtw_pp"] for d in draws]
        report["arm_b"][f"{f:.2f}"] = {
            "p": p,
            "rows": draws[0]["rows"],
            "games": draws[0]["games"],
            "floor_pp": draws[0]["floor_pp"],
            "median_abs_delta_dtw_pp": float(np.median(medians)),
            "median_across_draws": {"min": min(medians), "max": max(medians)},
            "mean_abs_delta_dtw_pp": float(np.mean([d["mean_abs_delta_dtw_pp"] for d in draws])),
            "side_flips": int(np.median([d["side_flips"] for d in draws])),
            "passes_floor": bool(np.median(medians) >= draws[0]["floor_pp"]),
        }
        stats = report["arm_b"][f"{f:.2f}"]
        print(
            f"  f = {f:.2f} (p = {p:.4f}, {stats['rows']:,} rows, {stats['games']:,} games): "
            f"median |dDTW| {stats['median_abs_delta_dtw_pp']:.2f} pp "
            f"[{min(medians):.2f}-{max(medians):.2f} across draws] vs floor "
            f"{stats['floor_pp']:.2f} pp -> {'above' if stats['passes_floor'] else 'below'}"
        )

    def gate(arm_report: dict) -> dict:
        medians = [v["median_abs_delta_dtw_pp"] for v in arm_report.values()]
        verdicts = {v["passes_floor"] for v in arm_report.values()}
        ratio = max(medians) / min(medians)
        return {
            "ratio": ratio,
            "same_verdict": len(verdicts) == 1,
            "passes": bool(len(verdicts) == 1 and ratio <= INVARIANCE_BOUND),
        }

    report["gate_d1_arm_a"] = gate(report["arm_a"])
    report["gate_d1_arm_b"] = gate(report["arm_b"])
    print(
        f"\n[D-1] Arm A: max/min = {report['gate_d1_arm_a']['ratio']:.2f}x, same verdict "
        f"{report['gate_d1_arm_a']['same_verdict']} -> "
        f"{'PASS' if report['gate_d1_arm_a']['passes'] else 'FAIL'}"
    )
    print(
        f"[D-1] Arm B: max/min = {report['gate_d1_arm_b']['ratio']:.2f}x, same verdict "
        f"{report['gate_d1_arm_b']['same_verdict']} -> "
        f"{'PASS' if report['gate_d1_arm_b']['passes'] else 'FAIL'}  (this arm decides)"
    )
    print(
        "\nVERDICT: "
        + (
            "the bound is narrow enough to act on; reopen document 17"
            if report["gate_d1_arm_b"]["passes"]
            else "BOUND TOO WIDE — the candidate stays closed"
        )
    )

    out = paths.RESEARCH_OUTPUT_DIR / "34_deflected_bound.json"
    with out.open("w") as handle:
        json.dump(report, handle, indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
