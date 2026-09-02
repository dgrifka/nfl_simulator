"""Phase 6 candidate 3 — blocked kicks priced as misses: the Gate P-3 / P-4 fit.

Runs the gates `docs/research/26-blocked-kick-pricing.md` §5 committed at
`e33edc6`, before this file existed:

* **P-3** — does excluding blocked kicks from the field-goal and extra-point
  components move the 287 games that carry one by more than v1.2's own 1.6250 pp
  median half-width on them? Eight redraws are run **unconditionally**, because
  §4 predicted a result inside 20% of the threshold.
* **P-4** — does the ledger still sum, and does it lose exactly 192 field-goal
  rows and 110 extra-point rows and nothing else?

The all-games median is computed in the same pass and reported beside the gate
statistic, per document 18 §4b's two-population rule.

    uv run python research/41_blocked_pricing.py
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

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.components import (  # noqa: E402
    build_game_table,
    fg_attempt_mask,
    fit_fg_baseline,
    fit_fumble_baseline,
    fit_xp_baseline,
    xp_attempt_mask,
)
from nfl_simulator.ingest import PBP_SEASONS, load_pbp  # noqa: E402
from nfl_simulator.simulator import (  # noqa: E402
    bootstrap_margins,
    extra_point_events,
    field_goal_events,
    fumble_events,
    points_per_epa,
)

RANDOM_SEED = _design.RANDOM_SEED
POSTERIOR_DRAWS = _design.SIM_POSTERIOR_DRAWS
COIN_DRAWS = _design.SIM_COIN_DRAWS
REDRAWS = 8  # docs/research/26 §5d, run unconditionally
GATE_P3_FLOOR_PP = 1.6250  # §4
REDRAW_MAJORITY = 6  # §5d: the verdict is the one holding in at least 6 of 8


def unblocked(frame: pl.DataFrame) -> pl.DataFrame:
    """The corrected kicking population: every attempt whose kick was not blocked.

    Filtering the play frame is exactly equivalent to narrowing
    `fg_attempt_mask` and `xp_attempt_mask`, because both the baseline fitters
    and the event builders apply those masks to the frame they are handed. It is
    used here rather than editing production code, because nothing merges until
    the maintainer approves.
    """
    return frame.filter(~_design.blocked_fg_mask() & ~_design.blocked_xp_mask())


def ledger_checks(pbp: pl.DataFrame) -> dict:
    """Gate P-4's row arithmetic."""
    fg_before = pbp.filter(fg_attempt_mask()).height
    fg_after = unblocked(pbp).filter(fg_attempt_mask()).height
    xp_before = pbp.filter(xp_attempt_mask()).height
    xp_after = unblocked(pbp).filter(xp_attempt_mask()).height
    fumble_mask = (pl.col("fumble") == 1) & pl.col("fumbled_1_team").is_not_null()
    fumbles_full = pbp.filter(fumble_mask).height
    fumbles_if_frame_filtered = unblocked(pbp).filter(fumble_mask).height
    print("\n[P-4] row arithmetic")
    print(f"  field-goal rows {fg_before:,} -> {fg_after:,} (expected -192)")
    print(f"  extra-point rows {xp_before:,} -> {xp_after:,} (expected -110)")
    print(
        f"  fumble rows: {fumbles_full:,} on the full frame, and {fumbles_if_frame_filtered:,} "
        "if the correction were implemented by filtering the play frame instead of narrowing "
        "the two kick masks"
    )
    print(
        "    -> four blocked field goals also carry a fumble row, so a frame-level filter would "
        "silently delete them. Both arms of this study hand the fumble builder the unfiltered "
        "frame, and a production implementation must narrow the masks, not the frame."
    )
    passed = (
        fg_before - fg_after == 192
        and xp_before - xp_after == 110
        and fumbles_full - fumbles_if_frame_filtered == 4
    )
    return {
        "fg_rows_before": fg_before,
        "fg_rows_after": fg_after,
        "xp_rows_before": xp_before,
        "xp_rows_after": xp_after,
        "fumble_rows_full_frame": fumbles_full,
        "fumble_rows_if_frame_filtered": fumbles_if_frame_filtered,
        "pass": bool(passed),
    }


def impact(pbp: pl.DataFrame) -> dict:
    print("\nfitting baselines ...")
    fumble_baseline = fit_fumble_baseline(pbp)
    fg_v12 = fit_fg_baseline(pbp)
    xp_v12 = fit_xp_baseline(pbp)
    fg_fixed = fit_fg_baseline(unblocked(pbp))
    xp_fixed = fit_xp_baseline(unblocked(pbp))
    fg_model = _design._model(pbp)
    games = build_game_table(pbp)
    slope = points_per_epa(games.drop_nulls("margin"))
    margins = dict(zip(games["game_id"], games["margin"], strict=True))

    blocked_games = set(
        pbp.filter(_design.blocked_fg_mask() | _design.blocked_xp_mask())["game_id"].to_list()
    )
    kick_games = sorted(set(pbp.filter(fg_attempt_mask() | xp_attempt_mask())["game_id"].to_list()))
    print(f"  games with any kick: {len(kick_games):,}; of which blocked: {len(blocked_games)}")

    def score(game_ids: list[str], seed: int) -> pl.DataFrame:
        subset = pbp.filter(pl.col("game_id").is_in(game_ids))
        rows = []
        for game_id, group in subset.group_by("game_id"):
            game_id = game_id[0] if isinstance(game_id, tuple) else game_id
            actual = margins.get(game_id)
            if actual is None:
                continue
            clean = unblocked(group)

            def arm(kick_frame, fg_baseline, xp_baseline, margin, plays=group) -> tuple:
                rng_fumble = np.random.default_rng(seed + 1)
                rng_fg = np.random.default_rng(seed + 2)
                rng_xp = np.random.default_rng(seed + 3)
                rng_coin = np.random.default_rng(seed + 4)
                events = [
                    *fumble_events(plays, fumble_baseline, POSTERIOR_DRAWS, rng_fumble),
                    *field_goal_events(kick_frame, fg_baseline, fg_model, POSTERIOR_DRAWS, rng_fg),
                    *extra_point_events(kick_frame, xp_baseline, fg_model, POSTERIOR_DRAWS, rng_xp),
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

            dtw_v12, deserved_v12, half_width = arm(group, fg_v12, xp_v12, actual)
            dtw_fix, deserved_fix, _ = arm(clean, fg_fixed, xp_fixed, actual)
            rows.append(
                {
                    "game_id": game_id,
                    "dtw_v12": dtw_v12,
                    "dtw_fixed": dtw_fix,
                    "deserved_v12": deserved_v12,
                    "deserved_fixed": deserved_fix,
                    "half_width_v12": half_width,
                    "blocked": game_id in blocked_games,
                }
            )
        return pl.DataFrame(rows)

    print(f"\nsimulating {len(kick_games):,} games with a kick (both arms) ...")
    scored = score(kick_games, RANDOM_SEED)

    def summarise(frame: pl.DataFrame) -> dict:
        delta = (frame["dtw_fixed"] - frame["dtw_v12"]).abs()
        margin_delta = frame["deserved_fixed"] - frame["deserved_v12"]
        return {
            "games": frame.height,
            "median_abs_delta_dtw_pp": float(delta.median()) * 100,
            "mean_abs_delta_dtw_pp": float(delta.mean()) * 100,
            "max_abs_delta_dtw_pp": float(delta.max()) * 100,
            "median_abs_delta_deserved_margin": float(margin_delta.abs().median()),
            "mean_signed_delta_deserved_margin": float(margin_delta.mean()),
            "side_flips": frame.filter(
                ((pl.col("dtw_fixed") - 0.5) * (pl.col("dtw_v12") - 0.5)) < 0
            ).height,
            "median_half_width_pp": float(frame["half_width_v12"].median()) * 100,
        }

    on_blocked = summarise(scored.filter("blocked"))
    on_all = summarise(scored)
    passed = on_blocked["median_abs_delta_dtw_pp"] >= GATE_P3_FLOOR_PP
    print(f"\n[P-3] pre-registered floor {GATE_P3_FLOOR_PP:.4f} pp")
    print(
        f"  games with a blocked kick ({on_blocked['games']}): median |dDTW| "
        f"{on_blocked['median_abs_delta_dtw_pp']:.3f} pp, mean "
        f"{on_blocked['mean_abs_delta_dtw_pp']:.3f} pp, max "
        f"{on_blocked['max_abs_delta_dtw_pp']:.2f} pp, flips {on_blocked['side_flips']}, "
        f"median |d deserved margin| {on_blocked['median_abs_delta_deserved_margin']:.3f} pts "
        f"-> {'PASS' if passed else 'FAIL'}"
    )
    print(
        f"  all games with a kick ({on_all['games']:,}): median |dDTW| "
        f"{on_all['median_abs_delta_dtw_pp']:.3f} pp, median |d deserved margin| "
        f"{on_all['median_abs_delta_deserved_margin']:.3f} pts, flips {on_all['side_flips']}"
    )

    print(f"\n  running {REDRAWS} redraws on the {on_blocked['games']} blocked-kick games ...")
    redraw_medians, redraw_floors = [], []
    blocked_ids = sorted(blocked_games)
    for i in range(REDRAWS):
        frame = score(blocked_ids, RANDOM_SEED + 100 * (i + 1))
        stats = summarise(frame)
        redraw_medians.append(stats["median_abs_delta_dtw_pp"])
        redraw_floors.append(stats["median_half_width_pp"])
        print(
            f"    redraw {i + 1}: median |dDTW| {stats['median_abs_delta_dtw_pp']:.3f} pp, "
            f"v1.2 half-width {stats['median_half_width_pp']:.3f} pp"
        )
    votes = sum(m >= GATE_P3_FLOOR_PP for m in redraw_medians)
    verdict = votes >= REDRAW_MAJORITY
    print(
        f"  redraw verdict: {votes}/{REDRAWS} clear the pre-registered floor "
        f"-> {'PASS' if verdict else 'FAIL'} (majority rule, {REDRAW_MAJORITY} of {REDRAWS})"
    )

    return {
        "floor_pp": GATE_P3_FLOOR_PP,
        "on_blocked_games": on_blocked,
        "on_all_kick_games": on_all,
        "base_seed_pass": bool(passed),
        "redraw_medians_pp": redraw_medians,
        "redraw_floors_pp": redraw_floors,
        "redraw_votes": votes,
        "p3_pass": bool(verdict),
        "points_per_epa": slope,
    }


def main() -> None:
    pbp = load_pbp(PBP_SEASONS, columns=_design.PRICING_COLUMNS)
    checks = ledger_checks(pbp)
    result = impact(pbp)
    payload = {"gate_p4": checks, "gate_p3": result}
    out = paths.RESEARCH_OUTPUT_DIR / "41_blocked_pricing.json"
    with out.open("w") as handle:
        json.dump(payload, handle, indent=2, default=float)
    print(f"\nwrote {out}")

    if not checks["pass"]:
        verdict = "SHIP NOTHING — the ledger does not sum"
    elif result["p3_pass"]:
        verdict = "PASSES ALL GATES — stop at the door and ask the maintainer"
    else:
        verdict = "SHIP NOTHING — measured and below the floor"
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
