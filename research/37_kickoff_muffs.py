"""Phase 6 candidate 1 — kickoff muffs: the Gate M-3 / M-4 / M-5 / M-6 fit.

Runs the gates `docs/research/24-kickoff-muffs.md` §5 committed at `b5ec6d4`,
before this file existed:

* **M-3** — is the entity spread on the widened population below the 5.0718 pp
  null bound, so that full neutralization survives?
* **M-4** — does the widened component move the 248 games that carry a kickoff
  muff by more than v1.2's own 0.7222 pp median half-width on them?
* **M-5** — does every fumble and every muff appear exactly once, and does the
  ledger still sum?
* **M-6** — is the M-4 verdict the same at `w = 0.00` and `w = 0.25`?

    uv run python research/37_kickoff_muffs.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from _betabinom_grid import fit_grid  # noqa: E402

_design = import_module("36_kickoff_muff_power")
_oob = import_module("29_fumble_oob_power")

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
REDRAWS = 8  # docs/research/24 §5e: a near-miss is re-drawn, not called

GATE_M3_THRESHOLD_PP = 5.0718  # §4a, the null 90th percentile
GATE_M4_FLOOR_PP = 0.7222  # §4b, v1.2's own median half-width on the 248 games
SENSITIVITY_W = (0.00, 0.25, 0.50)  # §5g; gated at 0.00 and 0.25
GATED_W = (0.00, 0.25)
MUFF_CLASS = "kickoff/muff"


# --------------------------------------------------------------------------
# M-3
# --------------------------------------------------------------------------


def fit(name: str, counts: pl.DataFrame) -> dict:
    n = counts["n"].to_numpy().astype(float)
    k = counts["k"].to_numpy().astype(float)
    posterior = fit_grid(n, k)
    summary = posterior.summary()
    low, high = summary["population_sd_eti89"]
    rate = float(k.sum() / n.sum())
    median_n = float(counts["n"].median())
    report = {
        "name": name,
        "entities": counts.height,
        "opportunities": int(n.sum()),
        "league_rate": rate,
        "population_sd_pp": float(summary["population_sd_mean"]) * 100,
        "population_sd_eti89_pp": [low * 100, high * 100],
        "relative": float(summary["population_sd_mean"]) / rate,
        "kappa_mean": float(summary["kappa_mean"]),
        "grid_edge_mass": posterior.edge_mass(),
        "w_median_entity": median_n / (median_n + float(summary["kappa_mean"])),
    }
    print(
        f"  {name}: SD {report['population_sd_pp']:.4f} pp "
        f"[{low * 100:.4f}, {high * 100:.4f}], relative {report['relative']:.1%}, "
        f"kappa {report['kappa_mean']:.1f}, w(median n) {report['w_median_entity']:.4f}, "
        f"grid edge mass {report['grid_edge_mass']:.2e}"
    )
    return report


def counts_for(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.group_by(["season", "fumbled_1_team"])
        .agg(pl.len().alias("n"), pl.col("retained").sum().cast(pl.Int64).alias("k"))
        .drop_nulls()
        .sort(["season", "fumbled_1_team"])
    )


# --------------------------------------------------------------------------
# the widened component
# --------------------------------------------------------------------------


def team_season_rates(frame: pl.DataFrame) -> dict[tuple[int, str], float]:
    """Each receiving team-season's own muff-retention record — M-6's input."""
    counts = frame.group_by(["season", "muffing_team"]).agg(pl.col("retained").mean().alias("own"))
    return {
        (int(row["season"]), row["muffing_team"]): float(row["own"])
        for row in counts.iter_rows(named=True)
    }


def widened_fumble_events(
    plays: pl.DataFrame,
    table: pl.DataFrame,
    own_rates: dict,
    w: float,
    n_draws: int,
    rng: np.random.Generator,
) -> list[LuckEvent]:
    """The widened component's ledger rows at trust dial `w`.

    `w` moves the `kickoff/muff` class only. Every other class keeps the class
    league rate it has today, because this round is not proposing to change how
    they are neutralized — only which population the component sees.
    """
    fumbles = _design.widened_frame(plays).join(
        table.select("fumble_class", "n", "p_own", "swing_value"),
        on="fumble_class",
        how="left",
    )
    events = []
    for row in fumbles.iter_rows(named=True):
        if row["p_own"] is None or row["swing_value"] is None:
            continue
        draws = _class_rate_draws(float(row["n"]), float(row["p_own"]), n_draws, rng)
        if w and row["fumble_class"] == MUFF_CLASS:
            own = own_rates.get((int(row["season"]), row["fumbled_1_team"]))
            if own is not None:
                draws = w * own + (1.0 - w) * draws
        home_sign = 1.0 if row["fumbled_1_team"] == row["home_team"] else -1.0
        events.append(
            LuckEvent(
                play_id=float(row["play_id"]),
                component="fumble",
                event_class=row["fumble_class"],
                charged_team=row["fumbled_1_team"],
                realized=float(row["retained"]),
                expected_draws=draws,
                swing=float(row["swing_value"]) * home_sign,
            )
        )
    return events


# --------------------------------------------------------------------------
# M-5
# --------------------------------------------------------------------------


def ledger_checks(pbp: pl.DataFrame) -> dict:
    """Gate M-5's first half: can one play produce two rows, or vanish?"""
    v12 = _fumble_frame(pbp)
    muffs = _design.muff_frame(pbp)
    widened = _design.widened_frame(pbp)

    duplicates = widened.group_by(["game_id", "play_id"]).len().filter(pl.col("len") > 1)
    muff_keys = muffs.select("game_id", "play_id")
    muff_rows_in_widened = widened.join(muff_keys, on=["game_id", "play_id"], how="semi")
    misclassed = muff_rows_in_widened.filter(pl.col("fumble_class") != MUFF_CLASS)
    # Every v1.2 row must survive, either where it was or moved into the muff class.
    lost = v12.select("game_id", "play_id").join(
        widened.select("game_id", "play_id"), on=["game_id", "play_id"], how="anti"
    )
    expected = (
        v12.height
        - muff_rows_in_widened.join(
            v12.select("game_id", "play_id"), on=["game_id", "play_id"], how="semi"
        ).height
        + muffs.height
    )

    print("\n[M-5] population arithmetic")
    print(
        f"  v1.2 rows {v12.height:,}; muffs {muffs.height}; widened {widened.height:,} "
        f"(expected {expected:,})"
    )
    print(f"  plays appearing twice: {duplicates.height}")
    print(f"  muff plays not carrying the {MUFF_CLASS} class: {misclassed.height}")
    print(f"  v1.2 plays lost by the widening: {lost.height}")
    for row in lost.head(20).iter_rows(named=True):
        print(f"    LOST {row['game_id']} play {row['play_id']}")

    passed = (
        duplicates.height == 0
        and misclassed.height == 0
        and lost.height == 0
        and widened.height == expected
    )
    return {
        "v12_rows": v12.height,
        "muff_rows": muffs.height,
        "widened_rows": widened.height,
        "expected_rows": int(expected),
        "duplicate_plays": duplicates.height,
        "misclassified_muffs": misclassed.height,
        "v12_plays_lost": lost.height,
        "pass": bool(passed),
    }


# --------------------------------------------------------------------------
# M-4 and M-6
# --------------------------------------------------------------------------


def impact(pbp: pl.DataFrame) -> dict:
    """M-4 and M-6, on the games that carry a kickoff muff."""
    print("\nfitting baselines ...")
    muffs = _design.muff_frame(pbp)
    widened_table = _oob.fit_widened_baseline(_design.widened_frame(pbp))
    print(widened_table)
    own_rates = team_season_rates(muffs)

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

    touched = sorted(set(muffs["game_id"].to_list()))
    subset = pbp.filter(pl.col("game_id").is_in(touched))
    # The games that gain a row the v1.2 ledger did not already carry. Document
    # 19 §3: this is not the same population as "games containing the event",
    # and both are reported rather than one standing in for the other.
    gained = len(
        {
            g
            for g in muffs.join(
                _fumble_frame(pbp).select("game_id", "play_id"),
                on=["game_id", "play_id"],
                how="anti",
            )["game_id"].to_list()
        }
    )
    print(f"\nsimulating {len(touched):,} games that carry a kickoff muff ...")
    print(f"  of which {gained} gain a ledger row v1.2 did not already have")

    def measure(seed: int) -> pl.DataFrame:
        rows = []
        for game_id, group in subset.group_by("game_id"):
            game_id = game_id[0] if isinstance(game_id, tuple) else game_id
            actual = margins.get(game_id)
            if actual is None:
                continue

            def arm(fumble_rows: list[LuckEvent], plays: pl.DataFrame, margin: float) -> tuple:
                rng_fg = np.random.default_rng(seed + 2)
                rng_xp = np.random.default_rng(seed + 3)
                rng_coin = np.random.default_rng(seed + 4)
                events = [
                    *fumble_rows,
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

            rng_base = np.random.default_rng(seed + 1)
            base_dtw, base_deserved, base_half_width = arm(
                fumble_events(group, fumble_baseline, POSTERIOR_DRAWS, rng_base), group, actual
            )
            row = {
                "game_id": game_id,
                "dtw_v12": base_dtw,
                "deserved_v12": base_deserved,
                "half_width_v12": base_half_width,
            }
            for w in SENSITIVITY_W:
                rng = np.random.default_rng(seed + 1)
                dtw, deserved, _ = arm(
                    widened_fumble_events(group, widened_table, own_rates, w, POSTERIOR_DRAWS, rng),
                    group,
                    actual,
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
                "passes_floor": bool(float(delta.median()) * 100 >= GATE_M4_FLOOR_PP),
            }
        return floor, per_w

    scored = measure(RANDOM_SEED)
    floor, per_w = summarise(scored)
    print(
        f"\n[M-4] pre-registered floor {GATE_M4_FLOOR_PP:.4f} pp; "
        f"this run's v1.2 median half-width {floor * 100:.4f} pp"
    )
    for w in SENSITIVITY_W:
        stats = per_w[f"{w:.2f}"]
        gated = " (gated)" if w in GATED_W else " (reported, not gated)"
        print(
            f"  w = {w:.2f}: median |dDTW| {stats['median_abs_delta_dtw_pp']:.3f} pp, "
            f"mean {stats['mean_abs_delta_dtw_pp']:.3f} pp, "
            f"max {stats['max_abs_delta_dtw_pp']:.2f} pp, flips {stats['side_flips']}, "
            f"median |d deserved margin| {stats['median_abs_delta_deserved_margin']:.3f} pts "
            f"-> {'PASS' if stats['passes_floor'] else 'FAIL'}{gated}"
        )

    near = [
        w
        for w in SENSITIVITY_W
        if abs(per_w[f"{w:.2f}"]["median_abs_delta_dtw_pp"] - GATE_M4_FLOOR_PP) < 0.1
    ]
    stability = {}
    if near:
        print(
            f"\n  {[f'{w:.2f}' for w in near]} land within 0.1 pp of the floor; "
            f"re-drawing {REDRAWS} times"
        )
        redrawn = [measure(RANDOM_SEED + 100 * (i + 1)) for i in range(REDRAWS)]
        floors = [float(f["half_width_v12"].median()) * 100 for f in redrawn]
        stability["floor_pp"] = {"min": min(floors), "max": max(floors), "values": floors}
        print(f"    v1.2 half-width across redraws: {min(floors):.3f} - {max(floors):.3f} pp")
        for w in SENSITIVITY_W:
            medians = [
                float((f[f"dtw_w{w:.2f}"] - f["dtw_v12"]).abs().median()) * 100 for f in redrawn
            ]
            verdicts = [m >= GATE_M4_FLOOR_PP for m in medians]
            stability[f"{w:.2f}"] = {
                "median_pp": {"min": min(medians), "max": max(medians), "values": medians},
                "passes": sum(verdicts),
                "of": REDRAWS,
            }
            print(
                f"    w = {w:.2f}: median |dDTW| {min(medians):.3f} - {max(medians):.3f} pp, "
                f"passes {sum(verdicts)}/{REDRAWS} redraws"
            )

    gated_verdicts = {per_w[f"{w:.2f}"]["passes_floor"] for w in GATED_W}
    return {
        "games": scored.height,
        "games_gaining_a_new_row": gained,
        "pre_registered_floor_pp": GATE_M4_FLOOR_PP,
        "observed_floor_pp": floor * 100,
        "points_per_epa": slope,
        "classes": widened_table.to_dicts(),
        "by_w": per_w,
        "stability": stability,
        "m4_pass": per_w["0.00"]["passes_floor"],
        "m6_pass": len(gated_verdicts) == 1,
    }


def main() -> None:
    pbp = load_pbp(PBP_SEASONS, columns=_design.MUFF_COLUMNS)

    print("[M-3] entity spread on the widened population")
    widened = fit(
        "retention, widened with kickoff muffs",
        counts_for(_design.widened_frame(pbp)),
    )
    incumbent = fit("retention, v1.2 (incumbent)", counts_for(_oob.widened_frame(pbp)))
    upper = widened["population_sd_eti89_pp"][1]
    m3_pass = upper < GATE_M3_THRESHOLD_PP
    print(
        f"  89% upper bound {upper:.4f} pp vs threshold {GATE_M3_THRESHOLD_PP:.4f} pp "
        f"-> {'PASS' if m3_pass else 'FAIL'}"
    )

    checks = ledger_checks(pbp)
    result = impact(pbp)

    payload = {
        "gate_m3": {
            "threshold_pp": GATE_M3_THRESHOLD_PP,
            "pass": bool(m3_pass),
            "widened": widened,
            "incumbent": incumbent,
        },
        "gate_m5": checks,
        "impact": result,
    }
    out = paths.RESEARCH_OUTPUT_DIR / "37_kickoff_muffs.json"
    with out.open("w") as handle:
        json.dump(payload, handle, indent=2, default=float)
    print(f"\nwrote {out}")

    verdict = "SHIP as v1.3, pending the maintainer's approval"
    if not m3_pass:
        verdict = "SHIP NOTHING — open a partial-neutralization round"
    elif not checks["pass"]:
        verdict = "SHIP NOTHING — the ledger does not sum"
    elif not result["m4_pass"]:
        verdict = "SHIP NOTHING — measured and immaterial"
    elif not result["m6_pass"]:
        verdict = "SHIP NOTHING — verdict depends on an unreadable dial"
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
