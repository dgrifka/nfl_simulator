"""Phase 7, task 1 — the defect document 27 §9d's round-trip check found.

Document 27 §9d asked, as a plumbing check, whether
`FieldGoalModel.from_posterior` reproduces the fitted model's own `p_make` when
handed the centres the fit wrote. It does not, and the reason is not the
centres:

* **`delta_cubic` is discarded.** The adopted Phase 3 curve is cubic in centred
  distance. `FieldGoalModel._logit` computes `alpha + beta·d + gamma·d²/100` and
  stops, and `from_posterior` never reads `delta_cubic`. So the simulator prices
  every kick with a **quadratic** curve whose `gamma` was fitted *jointly with a
  cubic term that is then dropped*.
* **`delta_xp` and `lambda_xp` are discarded.** The read side has no extra-point
  terms at all, so an extra point is priced on the plain field-goal curve at 33
  yards with the kicker's effect at scale 1.

Both are present in v1.1 and v1.2 identically. Neither is a Gate A violation —
they are plumbing — and neither is created or worsened by the refit. This script
sizes them on the shipped population, descriptively.

**It measures pricing and booked luck only.** The game-level consequence is a
correction candidate with its own pre-registration, per the project's process
law, and this document does not pre-empt it.

    uv run python research/42c_read_side_defect.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import arviz as az
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_weather = import_module("14_fg_weather_model")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.components import (  # noqa: E402
    fit_fg_baseline,
    fit_xp_baseline,
)
from nfl_simulator.fg_model import FieldGoalModel, sanitize_weather  # noqa: E402
from nfl_simulator.ingest import PBP_SEASONS, load_pbp  # noqa: E402

COLUMNS = _weather.COLUMNS


def read_side(model: FieldGoalModel, kicks: pl.DataFrame) -> np.ndarray:
    out = np.empty(kicks.height)
    for i, row in enumerate(kicks.iter_rows(named=True)):
        out[i] = model.make_probability(
            row["kicker_season"],
            float(row["distance"]),
            weather=sanitize_weather(row["roof"], row["wind"], row["temp"]),
        ).mean()
    return out


def main() -> None:
    paths.ensure_data_dirs()

    # The shipped population, priced by the shipped posterior — blocked kicks
    # included, because that is what v1.2 does.
    kicks = _weather.load_kicks()
    with (paths.RESEARCH_OUTPUT_DIR / "fg_weather_summary.json").open() as handle:
        centres = json.load(handle)["centres"]

    idata = az.from_netcdf(paths.RESEARCH_OUTPUT_DIR / "trace_fg_weather.nc")
    kicker_levels = sorted(kicks["kicker_season"].unique().to_list())
    lookup = {level: i for i, level in enumerate(kicker_levels)}
    kicker_idx = np.array([lookup[v] for v in kicks["kicker_season"].to_list()])

    fitted = _weather.make_probabilities(idata, kicks, kicker_idx, centres).mean(axis=0)
    model = FieldGoalModel.from_posterior(
        paths.RESEARCH_OUTPUT_DIR / "trace_fg_weather.nc",
        wind_centre=centres["wind"],
        temp_centre=centres["temp"],
    )
    shipped = read_side(model, kicks)

    frame = kicks.with_columns(
        pl.Series("p_shipped", shipped),
        pl.Series("p_fitted", fitted),
        pl.Series("error_pp", (shipped - fitted) * 100),
        ((pl.col("distance") // 5) * 5).cast(pl.Int32).alias("distance_bin"),
    )

    print(f"{kicks.height:,} kicks, priced two ways from the same posterior\n")
    print("[1] pricing error, shipped read side minus the fitted model")
    for label, mask in (
        ("field goals", pl.col("is_xp") == 0),
        ("extra points", pl.col("is_xp") == 1),
    ):
        values = frame.filter(mask)["error_pp"].to_numpy()
        print(
            f"  {label:13s} n={len(values):,}  mean {values.mean():+.3f} pp  "
            f"median {np.median(values):+.3f} pp  "
            f"max |error| {np.abs(values).max():.2f} pp  "
            f"|error| > 1 pp on {int((np.abs(values) > 1).sum()):,} kicks"
        )

    print("\n[2] field goals, by distance bin — the cubic term's signature")
    by_bin = (
        frame.filter(pl.col("is_xp") == 0)
        .group_by("distance_bin")
        .agg(
            pl.len().alias("n"),
            pl.col("p_shipped").mean().alias("p_shipped"),
            pl.col("p_fitted").mean().alias("p_fitted"),
            pl.col("error_pp").mean().alias("mean_error_pp"),
        )
        .sort("distance_bin")
    )
    with pl.Config(tbl_rows=20):
        print(by_bin)

    # ---- what it costs the ledger ----------------------------------------
    print("\n[3] the luck rows that pricing error produces")
    pbp = load_pbp(PBP_SEASONS, columns=sorted({*COLUMNS, "play_id", "game_id", "epa"}))
    fg_baseline = fit_fg_baseline(pbp)
    xp_baseline = fit_xp_baseline(pbp)
    swing = dict(
        zip(
            fg_baseline.table["fg_bin"].to_list(),
            fg_baseline.table["swing_value"].to_list(),
            strict=True,
        )
    )

    swings = np.array(
        [
            xp_baseline.swing_value
            if row["is_xp"]
            else (swing.get(int(row["distance"] // 5 * 5)) or 0.0)
            for row in frame.iter_rows(named=True)
        ]
    )
    # luck = (realized − p) · swing, so the luck error is −(p_shipped − p_fitted)·swing.
    luck_error = -(shipped - fitted) * swings
    frame = frame.with_columns(pl.Series("luck_error_epa", luck_error))

    rows = []
    for label, mask in (
        ("field_goal", pl.col("is_xp") == 0),
        ("extra_point", pl.col("is_xp") == 1),
    ):
        values = frame.filter(mask)["luck_error_epa"].to_numpy()
        rows.append(
            {
                "component": label,
                "n": int(len(values)),
                "mean_abs_luck_error_epa": float(np.abs(values).mean()),
                "max_abs_luck_error_epa": float(np.abs(values).max()),
                "total_signed_luck_error_epa": float(values.sum()),
            }
        )
        print(
            f"  {label:12s} mean |luck error| {rows[-1]['mean_abs_luck_error_epa']:.4f} EPA, "
            f"max {rows[-1]['max_abs_luck_error_epa']:.3f} EPA, "
            f"signed total {rows[-1]['total_signed_luck_error_epa']:+.1f} EPA"
        )
    print(
        "\n  For scale: v1.2 books a mean |luck| of 0.939 EPA on a non-blocked field goal\n"
        "  and 0.108 EPA on an extra point (document 26 §3a)."
    )

    payload = {
        "n_kicks": int(kicks.height),
        "pricing_error_by_bin_field_goals": by_bin.to_dicts(),
        "luck_error": rows,
        "field_goal_mean_error_pp": float(frame.filter(pl.col("is_xp") == 0)["error_pp"].mean()),
        "extra_point_mean_error_pp": float(frame.filter(pl.col("is_xp") == 1)["error_pp"].mean()),
    }
    out = paths.RESEARCH_OUTPUT_DIR / "42c_read_side_defect.json"
    with out.open("w") as handle:
        json.dump(payload, handle, indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
