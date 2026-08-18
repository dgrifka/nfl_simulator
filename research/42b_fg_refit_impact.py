"""Phase 7, task 1 — document 27 §9c and §9d, the ledger half of the impact report.

`research/42_fg_refit.py` fits the corrected model and reports what moved inside
it. This script reads the posterior that script wrote and reports what moves in
the product:

* **§9c** — deserved margin and DTW on **both** populations document 27 fixed in
  advance: all games containing a kick (the primary number, because the refit
  touches every kick) and the 287 games containing a blocked kick.
* **§9d** — the three obligations. What the refit does to the luck booked on a
  blocked kick; what it does to document 26's Gate P-3 floor; and whether
  `FieldGoalModel.from_posterior` reproduces the fitted `p_make`.

Neither population is a gate. Document 27 §9 makes both a reporting requirement.

The two arms differ **only** in the field-goal posterior. The class tables, the
fumble component and every seed are identical, because `components.py` is
untouched by this correction — the blocked kicks are still in the empirical
swing tables, which is document 26's change and not this one.

    uv run python research/42b_fg_refit_impact.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_design = import_module("40_blocked_pricing_power")
_weather = import_module("14_fg_weather_model")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.components import (  # noqa: E402
    build_game_table,
    fg_attempt_mask,
    fit_fg_baseline,
    fit_fumble_baseline,
    fit_xp_baseline,
    xp_attempt_mask,
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
POSTERIOR_DRAWS = _design.SIM_POSTERIOR_DRAWS
COIN_DRAWS = _design.SIM_COIN_DRAWS
INCUMBENT_FLOOR_PP = 1.6250  # document 26 §4, on the incumbent posterior


def load_models() -> tuple[FieldGoalModel, FieldGoalModel, dict]:
    with (paths.RESEARCH_OUTPUT_DIR / "fg_weather_summary.json").open() as handle:
        incumbent_centres = json.load(handle)["centres"]
    with (paths.RESEARCH_OUTPUT_DIR / "fg_refit_summary.json").open() as handle:
        refit_summary = json.load(handle)
    incumbent = FieldGoalModel.from_posterior(
        paths.RESEARCH_OUTPUT_DIR / "trace_fg_weather.nc",
        wind_centre=incumbent_centres["wind"],
        temp_centre=incumbent_centres["temp"],
    )
    refit = FieldGoalModel.from_posterior(
        paths.RESEARCH_OUTPUT_DIR / "trace_fg_refit.nc",
        wind_centre=refit_summary["centres"]["wind"],
        temp_centre=refit_summary["centres"]["temp"],
    )
    return incumbent, refit, refit_summary


# --------------------------------------------------------------------------
# §9d.3 — the round trip
# --------------------------------------------------------------------------


def round_trip(refit: FieldGoalModel, refit_summary: dict) -> dict:
    """Does the simulator's read side reproduce the fitted model's own `p_make`?

    Document 27 §9d asked for this as a plumbing check on the centring
    constants. It is reported as a plumbing check **and** as the discovery it
    turned into: `FieldGoalModel` has no extra-point terms at all, so the two
    paths cannot agree on an extra point however the centres are passed.
    """
    print("\n[9d.3] read-side round trip against the fitted model")
    kicks = _weather.load_kicks(exclude_blocked=True)
    kicker_levels = sorted(kicks["kicker_season"].unique().to_list())
    lookup = {level: i for i, level in enumerate(kicker_levels)}
    kicker_idx = np.array([lookup[v] for v in kicks["kicker_season"].to_list()])

    import arviz as az

    idata = az.from_netcdf(paths.RESEARCH_OUTPUT_DIR / "trace_fg_refit.nc")
    fitted = _weather.make_probabilities(idata, kicks, kicker_idx, refit_summary["centres"]).mean(
        axis=0
    )

    read_side = np.empty(kicks.height)
    from nfl_simulator.fg_model import sanitize_weather

    for i, row in enumerate(kicks.iter_rows(named=True)):
        read_side[i] = refit.make_probability(
            row["kicker_season"],
            float(row["distance"]),
            weather=sanitize_weather(row["roof"], row["wind"], row["temp"]),
        ).mean()

    is_xp = kicks["is_xp"].to_numpy().astype(bool)
    delta = (read_side - fitted) * 100
    report = {
        "field_goals_max_abs_diff_pp": float(np.abs(delta[~is_xp]).max()),
        "extra_points_mean_diff_pp": float(delta[is_xp].mean()),
        "extra_points_max_abs_diff_pp": float(np.abs(delta[is_xp]).max()),
        "field_goals_agree": bool(np.abs(delta[~is_xp]).max() < 1e-6),
    }
    print(
        f"  field goals: max |read-side − fitted| = "
        f"{report['field_goals_max_abs_diff_pp']:.2e} pp -> "
        f"{'agree' if report['field_goals_agree'] else 'DISAGREE'}"
    )
    print(
        f"  extra points: mean {report['extra_points_mean_diff_pp']:+.3f} pp, "
        f"max |diff| {report['extra_points_max_abs_diff_pp']:.3f} pp"
    )
    print(
        "    -> the read side has no `delta_xp` and no `lambda_xp`. It prices an extra\n"
        "       point on the plain field-goal curve at 33 yards with the kicker's effect\n"
        "       at scale 1. Both parameters are fitted and both are discarded downstream.\n"
        "       This is a defect of the shipped code, not of the refit: it is present in\n"
        "       v1.1 and v1.2 identically, and document 27 §14 registers it."
    )
    return report


# --------------------------------------------------------------------------
# §9d.1 — what the refit does to the luck booked on a blocked kick
# --------------------------------------------------------------------------


def blocked_kick_luck(pbp: pl.DataFrame, models: dict, baselines: dict) -> dict:
    print("\n[9d.1] luck booked on a blocked kick, incumbent against refit")
    blocked_fg = set(
        zip(*pbp.filter(_design.blocked_fg_mask()).select("game_id", "play_id"), strict=True)
    )
    blocked_xp = set(
        zip(*pbp.filter(_design.blocked_xp_mask()).select("game_id", "play_id"), strict=True)
    )

    rows = []
    for label, model in models.items():
        rng = np.random.default_rng(RANDOM_SEED)
        for game_id, group in pbp.group_by("game_id"):
            game_id = game_id[0] if isinstance(game_id, tuple) else game_id
            for event in field_goal_events(group, baselines["fg"], model, POSTERIOR_DRAWS, rng):
                rows.append(
                    {
                        "arm": label,
                        "component": "field_goal",
                        "luck": event.to_entry().luck_epa,
                        "blocked": (game_id, event.play_id) in blocked_fg,
                    }
                )
            for event in extra_point_events(group, baselines["xp"], model, POSTERIOR_DRAWS, rng):
                rows.append(
                    {
                        "arm": label,
                        "component": "extra_point",
                        "luck": event.to_entry().luck_epa,
                        "blocked": (game_id, event.play_id) in blocked_xp,
                    }
                )
    frame = pl.DataFrame(rows)
    summary = (
        frame.group_by(["component", "blocked", "arm"])
        .agg(pl.len().alias("n"), pl.col("luck").abs().mean().alias("mean_abs_luck_epa"))
        .sort(["component", "blocked", "arm"])
    )
    with pl.Config(tbl_rows=20):
        print(summary)
    return {"rows": summary.to_dicts()}


# --------------------------------------------------------------------------
# §9c — deserved margin and DTW, on both populations
# --------------------------------------------------------------------------


def impact(pbp: pl.DataFrame, models: dict, baselines: dict) -> dict:
    games = build_game_table(pbp)
    slope = points_per_epa(games.drop_nulls("margin"))
    margins = dict(zip(games["game_id"], games["margin"], strict=True))

    blocked_games = set(
        pbp.filter(_design.blocked_fg_mask() | _design.blocked_xp_mask())["game_id"].to_list()
    )
    kick_games = sorted(set(pbp.filter(fg_attempt_mask() | xp_attempt_mask())["game_id"].to_list()))
    print(
        f"\n[9c] {len(kick_games):,} games with a kick, of which {len(blocked_games)} "
        f"carry a blocked kick; points_per_epa {slope:.4f}"
    )

    subset = pbp.filter(pl.col("game_id").is_in(kick_games))
    rows = []
    for game_id, group in subset.group_by("game_id"):
        game_id = game_id[0] if isinstance(game_id, tuple) else game_id
        actual = margins.get(game_id)
        if actual is None:
            continue

        def arm(model, margin=None, plays=group) -> tuple:
            # One seeded generator per component in both arms, so the only
            # difference between them is the field-goal posterior.
            rng_fumble = np.random.default_rng(RANDOM_SEED + 1)
            rng_fg = np.random.default_rng(RANDOM_SEED + 2)
            rng_xp = np.random.default_rng(RANDOM_SEED + 3)
            rng_coin = np.random.default_rng(RANDOM_SEED + 4)
            events = [
                *fumble_events(plays, baselines["fumble"], POSTERIOR_DRAWS, rng_fumble),
                *field_goal_events(plays, baselines["fg"], model, POSTERIOR_DRAWS, rng_fg),
                *extra_point_events(plays, baselines["xp"], model, POSTERIOR_DRAWS, rng_xp),
            ]
            if not events:
                return (1.0 if margin > 0 else 0.0), margin, 0.0
            _, per_draw = bootstrap_margins(events, margin, slope, COIN_DRAWS, rng_coin)
            luck = sum(event.to_entry().luck_epa for event in events)
            low, high = np.percentile(per_draw, [5.5, 94.5])
            return float(per_draw.mean()), float(margin - luck * slope), float((high - low) / 2)

        dtw_in, deserved_in, half_in = arm(models["incumbent"], actual)
        dtw_re, deserved_re, half_re = arm(models["refit"], actual)
        rows.append(
            {
                "game_id": game_id,
                "dtw_incumbent": dtw_in,
                "dtw_refit": dtw_re,
                "deserved_incumbent": deserved_in,
                "deserved_refit": deserved_re,
                "half_width_incumbent": half_in,
                "half_width_refit": half_re,
                "blocked": game_id in blocked_games,
            }
        )
    scored = pl.DataFrame(rows)

    def summarise(frame: pl.DataFrame) -> dict:
        delta = (frame["dtw_refit"] - frame["dtw_incumbent"]).abs()
        margin_delta = frame["deserved_refit"] - frame["deserved_incumbent"]
        return {
            "games": frame.height,
            "median_abs_delta_dtw_pp": float(delta.median()) * 100,
            "mean_abs_delta_dtw_pp": float(delta.mean()) * 100,
            "max_abs_delta_dtw_pp": float(delta.max()) * 100,
            "median_abs_delta_deserved_margin": float(margin_delta.abs().median()),
            "mean_signed_delta_deserved_margin": float(margin_delta.mean()),
            "max_abs_delta_deserved_margin": float(margin_delta.abs().max()),
            "side_flips": frame.filter(
                ((pl.col("dtw_refit") - 0.5) * (pl.col("dtw_incumbent") - 0.5)) < 0
            ).height,
            "median_half_width_incumbent_pp": float(frame["half_width_incumbent"].median()) * 100,
            "median_half_width_refit_pp": float(frame["half_width_refit"].median()) * 100,
        }

    on_all = summarise(scored)
    on_blocked = summarise(scored.filter("blocked"))
    for label, stats in (
        ("all games with a kick (PRIMARY)", on_all),
        ("games with a blocked kick", on_blocked),
    ):
        print(
            f"  {label} ({stats['games']:,}): median |dDTW| "
            f"{stats['median_abs_delta_dtw_pp']:.3f} pp, mean "
            f"{stats['mean_abs_delta_dtw_pp']:.3f} pp, max "
            f"{stats['max_abs_delta_dtw_pp']:.2f} pp; median |d deserved margin| "
            f"{stats['median_abs_delta_deserved_margin']:.3f} pts (mean signed "
            f"{stats['mean_signed_delta_deserved_margin']:+.3f}); flips {stats['side_flips']}"
        )

    # ---- §9d.2 — document 26's floor, recomputed under the refit ----------
    print("\n[9d.2] document 26's Gate P-3 floor, on the 287 blocked-kick games")
    print(
        f"  incumbent posterior : {on_blocked['median_half_width_incumbent_pp']:.4f} pp "
        f"(document 26 §4 published {INCUMBENT_FLOOR_PP:.4f} pp)"
    )
    print(f"  refit posterior     : {on_blocked['median_half_width_refit_pp']:.4f} pp")
    print(
        "    -> the second number is the floor a re-measurement of document 26's candidate\n"
        "       would face if this refit is adopted. It is computed here, in the refit's own\n"
        "       document, before that candidate is re-measured."
    )

    return {
        "points_per_epa": slope,
        "on_all_kick_games": on_all,
        "on_blocked_games": on_blocked,
        "published_incumbent_floor_pp": INCUMBENT_FLOOR_PP,
    }


def main() -> None:
    paths.ensure_data_dirs()
    pbp = load_pbp(PBP_SEASONS, columns=_design.PRICING_COLUMNS)
    incumbent, refit, refit_summary = load_models()
    models = {"incumbent": incumbent, "refit": refit}

    print("fitting v1.2 class tables (unchanged by this correction) ...")
    baselines = {
        "fumble": fit_fumble_baseline(pbp),
        "fg": fit_fg_baseline(pbp),
        "xp": fit_xp_baseline(pbp),
    }

    payload = {
        "round_trip": round_trip(refit, refit_summary),
        "blocked_kick_luck": blocked_kick_luck(pbp, models, baselines),
        "impact": impact(pbp, models, baselines),
    }
    out = paths.RESEARCH_OUTPUT_DIR / "42b_fg_refit_impact.json"
    with out.open("w") as handle:
        json.dump(payload, handle, indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
