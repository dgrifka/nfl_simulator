"""Phase 6 candidate 3 — blocked kicks priced as misses: identification and floor.

Runs **before** `docs/research/26-blocked-kick-pricing.md` fixes any threshold.
The materiality *floor* is a property of shipped v1.2 and is computed here; the
materiality *statistic* is not.

Document 23 §C1 found that the field-goal and extra-point components price a
blocked kick as an ordinary miss and charge the difference to the kicker's coin.
Four questions:

1. **How much luck does the simulator currently book on a blocked kick?**
   Document 23 sized the defect as the 0.55 EPA gap between a block and a miss.
   That is the gap between two ways of *pricing* the play; the quantity that
   matters is how much luck is booked at all.
2. **What does removing them do to the class tables** — the empirical branch
   means, and therefore the swing every other kick carries?
3. **The materiality floor** on the games a blocked kick touches.
4. **The 80 touchdowns with neither an extra point nor a two-point attempt
   charted**, which document 23 §C1 asked the next round touching extra points
   to record.

    uv run python research/40_blocked_pricing_power.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_power = import_module("25_overtime_power")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.components import (  # noqa: E402
    _fg_frame,
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
SIM_POSTERIOR_DRAWS = _power.SIM_POSTERIOR_DRAWS
SIM_COIN_DRAWS = _power.SIM_COIN_DRAWS

PRICING_COLUMNS = sorted({*_power.SIM_COLUMNS, "desc", "touchdown", "two_point_attempt"})


def blocked_fg_mask() -> pl.Expr:
    return fg_attempt_mask() & (pl.col("field_goal_result") == "blocked")


def blocked_xp_mask() -> pl.Expr:
    return xp_attempt_mask() & (pl.col("extra_point_result") == "blocked")


def _model(pbp: pl.DataFrame) -> FieldGoalModel:
    with (paths.RESEARCH_OUTPUT_DIR / "fg_weather_summary.json").open() as handle:
        centres = json.load(handle)["centres"]
    return FieldGoalModel.from_posterior(
        paths.RESEARCH_OUTPUT_DIR / "trace_fg_weather.nc",
        wind_centre=centres["wind"],
        temp_centre=centres["temp"],
    )


def booked_luck(pbp: pl.DataFrame) -> dict:
    """How much luck v1.2 books on a blocked kick, per play and in total."""
    print("[1] the luck the simulator currently books on a blocked kick")
    fg_baseline = fit_fg_baseline(pbp)
    xp_baseline = fit_xp_baseline(pbp)
    fg_model = _model(pbp)
    rng = np.random.default_rng(RANDOM_SEED)

    blocked_fg = set(zip(*pbp.filter(blocked_fg_mask()).select("game_id", "play_id"), strict=True))
    blocked_xp = set(zip(*pbp.filter(blocked_xp_mask()).select("game_id", "play_id"), strict=True))

    rows = []
    for game_id, group in pbp.group_by("game_id"):
        game_id = game_id[0] if isinstance(game_id, tuple) else game_id
        for event in field_goal_events(group, fg_baseline, fg_model, SIM_POSTERIOR_DRAWS, rng):
            rows.append(
                {
                    "game_id": game_id,
                    "component": "field_goal",
                    "luck": event.to_entry().luck_epa,
                    "blocked": (game_id, event.play_id) in blocked_fg,
                }
            )
        for event in extra_point_events(group, xp_baseline, fg_model, SIM_POSTERIOR_DRAWS, rng):
            rows.append(
                {
                    "game_id": game_id,
                    "component": "extra_point",
                    "luck": event.to_entry().luck_epa,
                    "blocked": (game_id, event.play_id) in blocked_xp,
                }
            )
    frame = pl.DataFrame(rows)
    summary = (
        frame.group_by(["component", "blocked"])
        .agg(
            pl.len().alias("n"),
            pl.col("luck").abs().mean().alias("mean_abs_luck_epa"),
            pl.col("luck").abs().median().alias("median_abs_luck_epa"),
        )
        .sort(["component", "blocked"])
    )
    print(summary)
    blocked = frame.filter("blocked")
    print(
        f"  blocked kicks carrying a luck row: {blocked.height} in "
        f"{blocked['game_id'].n_unique()} games; mean |luck| "
        f"{blocked['luck'].abs().mean():.3f} EPA"
    )
    return {
        "rows": summary.to_dicts(),
        "blocked_rows": blocked.height,
        "blocked_games": blocked["game_id"].n_unique(),
        "mean_abs_luck_epa": float(blocked["luck"].abs().mean()),
    }


def class_tables(pbp: pl.DataFrame) -> dict:
    """What removing blocked kicks does to the empirical branch means."""
    print("\n[2] the class tables, with and without blocked kicks")
    fg = _fg_frame(pbp.filter(fg_attempt_mask()))
    fg_all = {
        "n": fg.height,
        "p_make": float(fg["made"].mean()),
        "epa_missed": float(fg.filter(pl.col("made") == 0)["epa"].mean()),
    }
    fg_clean = fg.filter(pl.col("field_goal_result") != "blocked")
    fg_wo = {
        "n": fg_clean.height,
        "p_make": float(fg_clean["made"].mean()),
        "epa_missed": float(fg_clean.filter(pl.col("made") == 0)["epa"].mean()),
    }
    xp = pbp.filter(xp_attempt_mask())
    xp_all = {
        "n": xp.height,
        "p_good": float((xp["extra_point_result"] == "good").mean()),
        "epa_failed": float(xp.filter(pl.col("extra_point_result") != "good")["epa"].mean()),
    }
    xp_clean = xp.filter(pl.col("extra_point_result") != "blocked")
    xp_wo = {
        "n": xp_clean.height,
        "p_good": float((xp_clean["extra_point_result"] == "good").mean()),
        "epa_failed": float(xp_clean.filter(pl.col("extra_point_result") != "good")["epa"].mean()),
    }
    for name, a, b in [("field goal", fg_all, fg_wo), ("extra point", xp_all, xp_wo)]:
        print(f"  {name}: with blocks {a}")
        print(f"  {name}: without    {b}")
    return {
        "field_goal": {"with": fg_all, "without": fg_wo},
        "extra_point": {"with": xp_all, "without": xp_wo},
    }


def touchdown_audit(pbp: pl.DataFrame) -> dict:
    """Document 23 §C1's leftover: touchdowns with no try charted."""
    print("\n[3] touchdowns with neither an extra point nor a two-point attempt")
    tds = pbp.filter((pl.col("touchdown") == 1).fill_null(False))
    xp = pbp.filter(xp_attempt_mask())
    two = pbp.filter((pl.col("two_point_attempt") == 1).fill_null(False))
    gap = tds.height - xp.height - two.height
    print(
        f"  touchdowns {tds.height:,}; extra-point attempts {xp.height:,}; "
        f"two-point attempts {two.height:,}; unaccounted {gap}"
    )
    # Match each touchdown to a try in the rows that follow it, so the gap is
    # attributed rather than merely counted. An overtime walk-off needs no try
    # by rule; a fourth-quarter touchdown with time left does.
    indexed = pbp.with_row_index("_row")
    tries = indexed.filter(
        xp_attempt_mask() | (pl.col("two_point_attempt") == 1).fill_null(False)
    ).select("game_id", "_row")
    by_game: dict[str, list[int]] = {}
    for game_id, row_index in tries.iter_rows():
        by_game.setdefault(game_id, []).append(int(row_index))
    unmatched = [
        row
        for row in indexed.filter((pl.col("touchdown") == 1).fill_null(False)).iter_rows(named=True)
        if not any(
            row["_row"] < candidate <= row["_row"] + 15
            for candidate in by_game.get(row["game_id"], [])
        )
    ]
    by_qtr = (
        pl.DataFrame({"qtr": [int(row["qtr"]) for row in unmatched]})
        .group_by("qtr")
        .agg(pl.len().alias("n"))
        .sort("qtr")
    )
    print(f"  touchdowns with no try charted within 15 rows: {len(unmatched)}")
    print(by_qtr)
    return {
        "touchdowns": tds.height,
        "extra_point_attempts": xp.height,
        "two_point_attempts": two.height,
        "unaccounted": gap,
        "unmatched": len(unmatched),
        "unmatched_by_quarter": by_qtr.to_dicts(),
    }


def materiality_floor(pbp: pl.DataFrame) -> dict:
    """v1.2's median 89% DTW half-width on the games a blocked kick touches."""
    print("\n[4] materiality floor")
    fumble_baseline = fit_fumble_baseline(pbp)
    fg_baseline = fit_fg_baseline(pbp)
    xp_baseline = fit_xp_baseline(pbp)
    fg_model = _model(pbp)
    games = build_game_table(pbp)
    slope = points_per_epa(games.drop_nulls("margin"))
    margins = dict(zip(games["game_id"], games["margin"], strict=True))
    touched = sorted(set(pbp.filter(blocked_fg_mask() | blocked_xp_mask())["game_id"].to_list()))
    half_widths = []
    for game_id, group in pbp.filter(pl.col("game_id").is_in(touched)).group_by("game_id"):
        game_id = game_id[0] if isinstance(game_id, tuple) else game_id
        actual = margins.get(game_id)
        if actual is None:
            continue
        events = [
            *fumble_events(
                group, fumble_baseline, SIM_POSTERIOR_DRAWS, np.random.default_rng(RANDOM_SEED + 1)
            ),
            *field_goal_events(
                group,
                fg_baseline,
                fg_model,
                SIM_POSTERIOR_DRAWS,
                np.random.default_rng(RANDOM_SEED + 2),
            ),
            *extra_point_events(
                group,
                xp_baseline,
                fg_model,
                SIM_POSTERIOR_DRAWS,
                np.random.default_rng(RANDOM_SEED + 3),
            ),
        ]
        if not events:
            continue
        _, per_draw = bootstrap_margins(
            events, actual, slope, SIM_COIN_DRAWS, np.random.default_rng(RANDOM_SEED + 4)
        )
        low, high = np.percentile(per_draw, [5.5, 94.5])
        half_widths.append(float((high - low) / 2))
    floor = float(np.median(half_widths))
    print(f"  games touched by a blocked kick: {len(touched)}")
    print(f"  v1.2 median 89% DTW half-width on them: {floor * 100:.4f} pp")
    return {
        "games": len(touched),
        "median_half_width": floor,
        "points_per_epa": slope,
    }


def main() -> None:
    pbp = load_pbp(PBP_SEASONS, columns=PRICING_COLUMNS)
    report = {
        "booked_luck": booked_luck(pbp),
        "class_tables": class_tables(pbp),
        "touchdown_audit": touchdown_audit(pbp),
        "materiality_floor": materiality_floor(pbp),
    }
    out = paths.RESEARCH_OUTPUT_DIR / "40_blocked_pricing_power.json"
    with out.open("w") as handle:
        json.dump(report, handle, indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
