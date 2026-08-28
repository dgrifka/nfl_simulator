"""Part B of round 7 — the receiver-drop study, and the fit the component reads.

Document 56's arms 1-3, at document 54 F-1's sampler spec, with the thresholds
Part A committed in `research/outputs/71_receiver_drop_power.json` before this
file fitted anything. Three things come out of it:

* **the study** — what makes a catchable ball get dropped, and whether the
  entity spread in the *conditioned* residual is negligible (Gates C-2 / C-3);
* **`trace_receiver_drop.nc` + `receiver_drop_summary.json`** — the posterior
  the read side loads, at the grain document 56 §1's clause-1 rule selected;
* **V-6 and V-8** on that fit, stored in the summary so Part D reports them
  without a refit, exactly as `research/67` stores them for the dropped pick.

**The grain, and why there are three conversion fits rather than two.** Part A
measured C-3 power at the 12.5% reference as 0.40 at the receiver-season grain
and 0.88 at the team-season grain. Document 56 §1's pre-committed clause-1 rule
says: receiver-season if it clears C-3, else team-season, else G-4 fails. It
clears only at team-season, so **the component charges the team-season — the
receiving corps** — and clause 2 of amendment A-3 requires the expectation to be
drawn over the *charged* entity's posterior. A receiver-season fit cannot supply
draws for a corps, so arm 2 is fitted a second time with a team-season term in
its place (`arm 2t`), and that is the fit the component reads. Arm 2 at the
receiver grain is still fitted and reported, because document 56 §1 asks for it
and because its C-3 failure is part of the record — under document 43 §7, a
design that fails C-3 may report its power table and not its verdict.

    uv run python research/72_receiver_drop_confounds.py

Nothing in `src/nfl_simulator/` changes on any number below.
"""

from __future__ import annotations

import json
import sys
import time
from importlib import import_module
from pathlib import Path

import arviz as az
import numpy as np
import polars as pl
import pymc as pm

sys.path.insert(0, str(Path(__file__).parent))

_power = import_module("61_dropped_pick_power")
_confounds = import_module("62_dropped_pick_confounds")
_receiver = import_module("71_receiver_drop_power")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.ingest import FTN_SEASONS, load_pbp  # noqa: E402

RANDOM_SEED = _receiver.RANDOM_SEED
LOGIT_SLOPE = _receiver.LOGIT_SLOPE
CROSS_CHECK_TOLERANCE_PP = _confounds.CROSS_CHECK_TOLERANCE_PP

# Document 54 amendment F-1, the spec every fit in this programme now uses.
DRAWS = 4000
TUNE = 4000
CHAINS = 4
TARGET_ACCEPT = 0.95

POWER_NAME = "71_receiver_drop_power.json"
TRACE_NAME = "trace_receiver_drop.nc"
SUMMARY_NAME = "receiver_drop_summary.json"
RECEIVER_GRAIN_TRACE = "trace_receiver_drop_receiver_grain.nc"
OUTPUT_NAME = "72_receiver_drop_confounds.json"

# Document 56 §2, V-8's sanity bound — the mirror of document 49 §6's, rewritten
# for an event that happens one time in twenty rather than one in two.
V8_LOW, V8_HIGH = 0.85, 0.99
V8_N_EACH = 5

ETI_LOW, ETI_HIGH = 5.5, 94.5

REFERENCE_LEVELS = {"pass_location": "middle", "down": 1.0}

# Document 56 §1's arm-2b column. `is_catchable_ball` is the selection here and
# never a covariate, so the hindsight-column arm drops the one that remains.
HINDSIGHT_COLUMNS = ("is_contested_ball",)

# Document 47 §3's form, applied to the drop branch: six cells, 30 per branch.
MIN_PER_BRANCH = 30
YARDLINE_BINS: tuple[tuple[int, int, str], ...] = (
    (1, 33, "1-33"),
    (34, 66, "34-66"),
    (67, 99, "67-99"),
)
DOWN_BINS: tuple[tuple[tuple[float, ...], str], ...] = (
    ((1.0, 2.0), "1-2"),
    ((3.0, 4.0), "3-4"),
)


def _labels(frame: pl.DataFrame, keys: list[str]) -> list[str]:
    """The level labels in `_power._codes` order, so a code indexes this list."""
    return (
        frame.select(pl.concat_str(keys, separator="|").alias("label"))["label"]
        .unique(maintain_order=True)
        .to_list()
    )


# --------------------------------------------------------------------------
# the EPA columns, joined after the design matrix rather than into it
# --------------------------------------------------------------------------


def catchable_with_epa(frame) -> pl.DataFrame:
    """Every charted catchable target with `epa`, `air_epa` and `xyac_epa` joined.

    All three are post-branch quantities and document 56 §1 excludes them from
    `X` by rule, so they are joined here rather than added to the modelling
    column list — which would change the frame arm 2 is fitted on. This is
    `research/67.worthy_with_epa`'s shape, with the two counterfactual columns
    document 56 §2 prices the swing from.
    """
    epa = load_pbp(FTN_SEASONS, columns=["game_id", "play_id", "epa", "air_epa", "xyac_epa"])
    joined = frame.catchable.join(epa, on=["game_id", "play_id"], how="left")
    if joined.height != frame.catchable.height:
        raise SystemExit(
            f"the epa join changed the row count: {joined.height:,} against "
            f"{frame.catchable.height:,} — stop and ask."
        )
    return joined


def with_bins(frame: pl.DataFrame) -> pl.DataFrame:
    """Add the ``swing_cell`` key column — Polars-native, so no per-row Python."""
    yard = pl.when(pl.col("yardline_100").is_null()).then(None)
    for low, high, label in YARDLINE_BINS:
        yard = yard.when((pl.col("yardline_100") >= low) & (pl.col("yardline_100") <= high)).then(
            pl.lit(label)
        )
    yard = yard.otherwise(None)

    down = pl.when(pl.col("down").is_null()).then(None)
    for downs, label in DOWN_BINS:
        down = down.when(pl.col("down").cast(pl.Float64).is_in(downs)).then(pl.lit(label))
    down = down.otherwise(None)

    return frame.with_columns(
        pl.when(yard.is_null() | down.is_null())
        .then(None)
        .otherwise(pl.concat_str([yard, down], separator="|"))
        .alias("swing_cell")
    )


def build_drop_swing_table(catchable: pl.DataFrame) -> dict:
    """Document 47 §3's bin table, mirrored onto the drop branch.

    Each cell holds the mean realised EPA of **caught** catchable targets and of
    **dropped** ones, and their difference. Two things read it (document 56 §2):
    the difference is the whole swing when a play's completion counterfactual is
    unreadable, and the dropped-branch mean is ``epa_incomplete`` for a play that
    *was* caught, where no realised incompletion exists to price.

    The swing here is **positive** by construction — a catch is worth more than
    an incompletion — which is the opposite sign from the dropped-pick table and
    is the sign guard handoff constraint 6 names.
    """
    caught = catchable.filter(~pl.col("is_drop"))
    dropped = catchable.filter(pl.col("is_drop"))
    if not caught.height or not dropped.height:
        raise SystemExit("cannot price a swing table with an empty branch")
    pooled_caught = float(caught["epa"].mean())
    pooled_dropped = float(dropped["epa"].mean())
    pooled = pooled_caught - pooled_dropped

    binned = with_bins(catchable)
    cells: dict[str, float] = {}
    incompletion: dict[str, float] = {}
    counts: dict[str, dict] = {}
    for _low, _high, yard in YARDLINE_BINS:
        for _downs, down_label in DOWN_BINS:
            key = f"{yard}|{down_label}"
            cell = binned.filter(pl.col("swing_cell") == key)
            cell_caught = cell.filter(~pl.col("is_drop"))
            cell_dropped = cell.filter(pl.col("is_drop"))
            n_caught, n_dropped = cell_caught.height, cell_dropped.height
            thin = n_caught < MIN_PER_BRANCH or n_dropped < MIN_PER_BRANCH

            mean_caught = float(cell_caught["epa"].mean()) if n_caught else None
            mean_dropped = float(cell_dropped["epa"].mean()) if n_dropped else None
            cells[key] = pooled if thin else mean_caught - mean_dropped
            incompletion[key] = pooled_dropped if thin else mean_dropped
            counts[key] = {
                "n_caught": n_caught,
                "n_dropped": n_dropped,
                "mean_epa_caught": mean_caught,
                "mean_epa_dropped": mean_dropped,
                "source": "pooled" if thin else "cell",
            }

    table = {
        "cells": cells,
        "incompletion_mean": incompletion,
        "counts": counts,
        "pooled": pooled,
        "pooled_incompletion_mean": pooled_dropped,
        "pooled_completion_mean": pooled_caught,
    }
    print("\n=== the swing table — document 47 §3's form on the drop branch ===")
    print(
        f"  {'cell':16s} {'n_caught':>9s} {'n_drop':>7s} {'swing':>8s} {'E[epa|drop]':>12s}  source"
    )
    for key in cells:
        entry = counts[key]
        print(
            f"  {key:16s} {entry['n_caught']:9d} {entry['n_dropped']:7d} "
            f"{cells[key]:+8.2f} {incompletion[key]:+12.2f}  {entry['source']}"
        )
    print(
        f"  pooled fallback {pooled:+.2f} EPA (caught {pooled_caught:+.2f} − dropped "
        f"{pooled_dropped:+.2f})"
    )
    negative = [key for key, value in cells.items() if value <= 0]
    if negative or pooled <= 0:
        raise SystemExit(
            f"a swing cell is not positive ({negative or 'pooled'}) — a catch would be "
            "worth no more than an incompletion. Handoff constraint 6: stop and ask."
        )
    table["all_cells_positive"] = True
    return table


def per_play_swing(catchable: pl.DataFrame, table: dict) -> dict:
    """Document 56 §2's per-play swing, and the counts of every fallback it took.

    ``swing = |(air_epa + xyac_epa) − epa_incomplete|`` with ``epa_incomplete``
    the realised EPA of a play that was dropped and the cell's dropped-branch
    mean for one that was caught.

    **One clause of document 56 §2 is under-specified and the reading is
    disclosed rather than assumed.** It says "both ``air_epa`` and ``xyac_epa``
    null -> bin table". In this data ``air_epa`` is null only where ``xyac_epa``
    is too (2,038 rows), but ``xyac_epa`` is null on a further 2,359 rows where
    ``air_epa`` is present. The completion counterfactual is a *sum* and half of
    it is not a value, so the bin table is taken whenever **either** term is
    missing. That uses the pre-registered fallback more often than the literal
    clause, never less, and it never invents a zero for a quantity nflfastR
    declined to supply. Both counts are reported separately below.
    """
    binned = with_bins(catchable)
    rows = binned.select("swing_cell", "is_drop", "epa", "air_epa", "xyac_epa").to_dict(
        as_series=False
    )

    swings = np.empty(binned.height)
    reasons = {"per_play": 0, "bin_both_null": 0, "bin_xyac_only_null": 0, "bin_no_cell": 0}
    for index in range(binned.height):
        key = rows["swing_cell"][index]
        air, xyac = rows["air_epa"][index], rows["xyac_epa"][index]
        if air is None and xyac is None:
            reasons["bin_both_null"] += 1
            swings[index] = table["cells"].get(key, table["pooled"]) if key else table["pooled"]
            continue
        if air is None or xyac is None:
            reasons["bin_xyac_only_null"] += 1
            swings[index] = table["cells"].get(key, table["pooled"]) if key else table["pooled"]
            continue
        if key is None:
            # No readable pre-throw state, so no cell mean to stand in for the
            # incompletion the play did not have. The pooled swing is document
            # 47 §3's own answer for a cell it cannot read.
            reasons["bin_no_cell"] += 1
            swings[index] = table["pooled"]
            continue
        reasons["per_play"] += 1
        completion = float(air) + float(xyac)
        incomplete = (
            float(rows["epa"][index]) if rows["is_drop"][index] else table["incompletion_mean"][key]
        )
        swings[index] = abs(completion - incomplete)

    report = {
        "rows": int(binned.height),
        "fallback_counts": reasons,
        "share_priced_per_play": reasons["per_play"] / binned.height,
        "median_swing_epa": float(np.median(swings)),
        "mean_swing_epa": float(swings.mean()),
        "eti89_swing_epa": [float(v) for v in np.percentile(swings, [ETI_LOW, ETI_HIGH])],
        "min_swing_epa": float(swings.min()),
        "max_swing_epa": float(swings.max()),
        "all_positive": bool((swings > 0).all()),
    }
    print("\n=== the per-play swing (document 56 §2) ===")
    print(
        f"  {reasons['per_play']:,} of {binned.height:,} priced per play "
        f"({report['share_priced_per_play']:.1%}); bin fallback: "
        f"{reasons['bin_both_null']:,} both terms null, "
        f"{reasons['bin_xyac_only_null']:,} xyac_epa alone null, "
        f"{reasons['bin_no_cell']:,} no readable cell"
    )
    print(
        f"  swing: median {report['median_swing_epa']:.2f} EPA, mean "
        f"{report['mean_swing_epa']:.2f}, 89% "
        f"[{report['eti89_swing_epa'][0]:.2f}, {report['eti89_swing_epa'][1]:.2f}], "
        f"min {report['min_swing_epa']:.2f}, max {report['max_swing_epa']:.2f}"
    )
    if not report["all_positive"]:
        raise SystemExit(
            "a play's swing is not positive — a catch must be worth more than an "
            "incompletion. Handoff constraint 6: stop and ask."
        )
    return report


# --------------------------------------------------------------------------
# arm 2 — the conversion model, at F-1's spec
# --------------------------------------------------------------------------


def fit_conversion(
    label: str,
    matrix: np.ndarray,
    names: tuple[str, ...],
    outcome: np.ndarray,
    entity_codes: np.ndarray,
    n_entity: int,
    defence_codes: np.ndarray,
    n_defence: int,
    *,
    entity_name: str,
) -> tuple[dict, object]:
    """Document 56 §1's arm-2 model at document 54 F-1's step counts.

    ``entity_codes`` is the offensive entity the arm charges — receiver-season
    for arm 2 and 2b, team-season for arm 2t. The model object is
    `71.build_drop_model`, imported rather than copied, so the three arms differ
    only in which offensive grouping they are given.
    """
    print(
        f"\n  arm {label}: {matrix.shape[0]:,} targets, {matrix.shape[1]} covariates, "
        f"{n_entity:,} {entity_name} levels x {n_defence} defence-seasons, "
        f"{CHAINS} x {DRAWS:,} draws after {TUNE:,} tuning, target_accept {TARGET_ACCEPT}"
    )
    started = time.time()
    model = _receiver.build_drop_model(
        matrix, outcome, entity_codes, n_entity, defence_codes, n_defence
    )
    with model:
        idata = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            random_seed=RANDOM_SEED,
            progressbar=False,
            nuts_sampler="nutpie",
            nuts={"target_accept": TARGET_ACCEPT},
        )
    elapsed = time.time() - started

    health = _confounds.sampler_health(idata, ["alpha", "beta", "sigma_r", "sigma_d", "z_r", "z_d"])
    posterior = idata["posterior"]
    beta_summary = az.summary(idata, var_names=["beta"])

    coefficients = []
    for index, name in enumerate(names):
        row = beta_summary.iloc[index]
        coefficients.append(
            {
                "name": name,
                "mean": float(row["mean"]),
                "eti89": [float(row["eti89_lb"]), float(row["eti89_ub"])],
                "excludes_zero": bool(row["eti89_lb"] > 0 or row["eti89_ub"] < 0),
                "odds_ratio": float(np.exp(row["mean"])),
            }
        )

    print("    beta (logit scale, on P(drop); * = 89% interval excludes zero)")
    for row in sorted(coefficients, key=lambda item: -abs(item["mean"])):
        marker = "*" if row["excludes_zero"] else " "
        print(
            f"      {marker} {row['name']:22s} {row['mean']:+.3f} "
            f"[{row['eti89'][0]:+.3f}, {row['eti89'][1]:+.3f}]  odds x{row['odds_ratio']:.2f}"
        )

    variances = {}
    for parameter, entity in (("sigma_r", entity_name), ("sigma_d", "defence-season")):
        draws = posterior[parameter].values.ravel()
        quantiles = np.quantile(draws, [0.055, 0.945])
        variances[parameter] = {
            "entity": entity,
            "logit_mean": float(draws.mean()),
            "logit_eti89": [float(quantiles[0]), float(quantiles[1])],
            "pp_mean": float(draws.mean()) * LOGIT_SLOPE * 100,
            "pp_eti89": [float(q) * LOGIT_SLOPE * 100 for q in quantiles],
        }
        entry = variances[parameter]
        print(
            f"    {parameter} ({entity}): logit {entry['logit_mean']:.3f} "
            f"[{entry['logit_eti89'][0]:.3f}, {entry['logit_eti89'][1]:.3f}]  "
            f"= {entry['pp_mean']:.2f} pp [{entry['pp_eti89'][0]:.2f}, "
            f"{entry['pp_eti89'][1]:.2f}] on the probability scale"
        )
    print(f"    fitted in {elapsed:.0f} s")

    report = {
        "label": label,
        "offensive_entity": entity_name,
        "targets": int(matrix.shape[0]),
        "covariates": list(names),
        "draws": DRAWS,
        "tune": TUNE,
        "chains": CHAINS,
        "target_accept": TARGET_ACCEPT,
        "wall_clock_seconds": elapsed,
        "sampler": health,
        "beta": coefficients,
        "alpha_mean": float(posterior["alpha"].values.mean()),
        "variance_components": variances,
    }
    return report, idata


def name_the_levels(idata, entity_levels: list[str], defence_levels: list[str]):
    """Give `r_s` and `d_d` their level names, so the read side can look one up."""
    posterior = (
        idata["posterior"]
        .to_dataset()
        .rename({"receiver": "entity_season", "defence": "defence_season"})
        .assign_coords(entity_season=entity_levels, defence_season=defence_levels)
    )
    idata["posterior"] = posterior
    return idata


def median_target(matrix: np.ndarray, names: tuple[str, ...], *, consistent: bool) -> np.ndarray:
    """The design row a V-8 interval is quoted on, under either reading.

    Document 49 §10 recorded that "a median throw" was undefined there and asked
    a future gate quoted on a reference row to write the row down. Both readings
    are computed and **both are stored in the summary JSON**, which is that ask
    discharged.
    """
    row = np.median(matrix, axis=0)
    if not consistent:
        return row
    row = row.copy()
    row[list(names).index("air_yards_z_squared")] = row[list(names).index("air_yards_z")] ** 2
    return row


def v8_report(idata, matrix, names, entity_levels: list[str], entity_name: str) -> dict:
    """V-8 — the five best and five worst charged entities, as catch probabilities.

    The model's ``p`` is the probability of a **drop**, so the catch probability
    the bound is quoted on is its complement — which is also what the component
    books as ``expected``.
    """
    posterior = idata["posterior"]
    alpha = posterior["alpha"].values.ravel()
    beta = posterior["beta"].values.reshape(-1, matrix.shape[1])
    effects = posterior["r_s"].values.reshape(-1, len(entity_levels))
    order = np.argsort(effects.mean(axis=0))

    print(f"\n=== V-8 — catch probability on a median target, 89% interval ({entity_name}) ===")
    print(f"  bound: every interval must lie inside [{V8_LOW:.2f}, {V8_HIGH:.2f}]")

    readings, breaches = {}, {}
    for reading, consistent in (("column_wise_median", False), ("consistent_median", True)):
        row = median_target(matrix, names, consistent=consistent)
        base = alpha + beta @ row
        print(
            f"\n  reading: {reading}  (league p(catch) at the entity effect 0: "
            f"{(1.0 - 1.0 / (1.0 + np.exp(-base))).mean():.3f})"
        )
        lines, breached = [], []
        # `order` is ascending in the drop effect, so the *best* hands come first.
        for label, indices in (
            ("best", order[:V8_N_EACH]),
            ("worst", order[-V8_N_EACH:][::-1]),
        ):
            for index in indices:
                draws = 1.0 - 1.0 / (1.0 + np.exp(-(base + effects[:, index])))
                low, high = (float(v) for v in np.percentile(draws, [ETI_LOW, ETI_HIGH]))
                inside = low >= V8_LOW and high <= V8_HIGH
                lines.append(
                    {
                        "rank": label,
                        "entity_season": entity_levels[index],
                        "drop_effect_mean": float(effects[:, index].mean()),
                        "p_catch_mean": float(draws.mean()),
                        "eti89": [low, high],
                        "inside_bound": bool(inside),
                    }
                )
                if not inside:
                    breached.append(entity_levels[index])
                print(
                    f"    {label:5s}  {entity_levels[index]:12s}  drop effect "
                    f"{effects[:, index].mean():+.3f}  p(catch) {draws.mean():.3f}  "
                    f"89% [{low:.3f}, {high:.3f}]  {'ok' if inside else 'OUT OF BOUND'}"
                )
        readings[reading] = {
            "median_target_design_row": [float(v) for v in row],
            "lines": lines,
            "breaches": breached,
            "pass": not breached,
        }
        breaches[reading] = breached
        print(
            f"    {reading}: {'PASS' if not breached else 'FAIL'} "
            f"({len(lines)} lines, {len(breached)} out of bound)"
        )

    report = {
        "bound": [V8_LOW, V8_HIGH],
        "n_each": V8_N_EACH,
        "quoted_on": entity_name,
        "readings": readings,
        "pass": not any(breaches.values()),
    }
    print(f"\n  V-8: {'PASS' if report['pass'] else 'FAIL'}")
    return report


# --------------------------------------------------------------------------
# the hindsight substitutes (document 56 §1, reported never gated)
# --------------------------------------------------------------------------


def hindsight_substitutes(charted: pl.DataFrame, catchable: pl.DataFrame) -> dict:
    """Document 56 §1's two substitutes for document 45's probe.

    ``p(catchable | incomplete, is_drop)`` is 1 by construction on this data, so
    the dropped-pick probe does not transfer. These two do the same job from the
    other side: a charter grading the **ball** rather than the outcome should
    call almost every completion catchable, and should call a contested ball
    dropped far more often than an uncontested one.
    """
    completions = charted.filter(pl.col("complete_pass") == 1)
    not_catchable = int((~completions["is_catchable_ball"]).sum())
    share_not_catchable = not_catchable / completions.height

    contested = catchable.filter(pl.col("is_contested_ball"))
    uncontested = catchable.filter(~pl.col("is_contested_ball"))
    rate_contested = float(contested["is_drop"].mean())
    rate_uncontested = float(uncontested["is_drop"].mean())
    ratio = rate_contested / rate_uncontested

    report = {
        "completions": int(completions.height),
        "completions_charted_not_catchable": not_catchable,
        "share_completions_not_catchable": share_not_catchable,
        "contested": {"n": int(contested.height), "drop_rate": rate_contested},
        "uncontested": {"n": int(uncontested.height), "drop_rate": rate_uncontested},
        "contested_over_uncontested": ratio,
        "hindsight_suspected": bool(ratio <= 1.0),
        "reading": (
            "contested balls are dropped no more often than uncontested ones — "
            "`is_catchable_ball` may be graded off the outcome, and every drop number "
            "in this study carries that caveat in words"
            if ratio <= 1.0
            else "contested balls are dropped materially more often, which is what a "
            "judgement of the ball rather than of the result looks like"
        ),
    }
    print("\n=== the hindsight substitutes (reported, never gated) ===")
    print(
        f"  completions charted NOT catchable: {not_catchable:,} of "
        f"{completions.height:,} ({share_not_catchable:.4%}) — should be ~0"
    )
    print(
        f"  drop rate contested {rate_contested:.4f} ({contested.height:,} balls) vs "
        f"uncontested {rate_uncontested:.4f} ({uncontested.height:,}) — "
        f"ratio {ratio:.2f}x"
    )
    print(f"  -> {report['reading']}")
    return report


# --------------------------------------------------------------------------


def main() -> None:
    paths.ensure_data_dirs()
    print("=== Round 7 Part B — the receiver-drop study (document 56 §1) ===")

    power_path = paths.RESEARCH_OUTPUT_DIR / POWER_NAME
    if not power_path.exists():
        raise SystemExit(f"{power_path} is missing — Part A must run and commit before Part B")
    power = json.loads(power_path.read_text())["designs"]

    frame = _receiver.build_catchable_frame()
    catchable = catchable_with_epa(frame)

    # --- the swing table and the per-play swing ------------------------------
    table = build_drop_swing_table(catchable)
    swing = per_play_swing(catchable, table)

    # --- arm 1, Gate D-1 -----------------------------------------------------
    print("\n=== arm 1 — the raw drop-rate spreads, three grains (Gate D-1) ===")
    receiver_counts = _receiver.rate_counts(
        frame.charted.drop_nulls("receiver_player_id").filter(pl.col("is_catchable_ball")),
        ["season", "receiver_player_id"],
        minimum=_receiver.MIN_RECEIVER_TARGETS,
    )
    arm1 = {
        "receiver_season": _confounds.worthy_rate_spread(
            f"drop rate, receiver-season (>= {_receiver.MIN_RECEIVER_TARGETS} targets)",
            receiver_counts,
            power["drop_rate_receiver_season"],
        ),
        "team_season": _confounds.worthy_rate_spread(
            "drop rate, team-season",
            _receiver.rate_counts(frame.catchable, ["season", "posteam"]),
            power["drop_rate_team_season"],
        ),
        "defence_season": _confounds.worthy_rate_spread(
            "drop rate, defence-season",
            _receiver.rate_counts(frame.catchable, ["season", "defteam"]),
            power["drop_rate_defence_season"],
        ),
    }

    # --- arms 2, 2b and 2t ---------------------------------------------------
    receiver_levels = _labels(frame.model, ["season", "receiver_player_id"])
    team_levels = _labels(frame.model, ["season", "posteam"])
    defence_levels = _labels(frame.model, ["season", "defteam"])

    print("\n=== arm 2 — conversion by covariates, receiver-season grain (Gate C-1) ===")
    arm2, arm2_idata = fit_conversion(
        "2",
        frame.design_matrix,
        frame.feature_names,
        frame.outcome,
        frame.receiver_season_codes,
        frame.n_receiver_seasons,
        frame.defence_season_codes,
        frame.n_defence_seasons,
        entity_name="receiver-season",
    )

    keep = [
        index for index, name in enumerate(frame.feature_names) if name not in HINDSIGHT_COLUMNS
    ]
    print("\n=== arm 2b — the same, without `is_contested_ball` ===")
    arm2b, _ = fit_conversion(
        "2b (no is_contested_ball)",
        frame.design_matrix[:, keep],
        tuple(frame.feature_names[index] for index in keep),
        frame.outcome,
        frame.receiver_season_codes,
        frame.n_receiver_seasons,
        frame.defence_season_codes,
        frame.n_defence_seasons,
        entity_name="receiver-season",
    )

    print(
        "\n=== arm 2t — the charged grain, team-season (document 56 §1's clause-1 rule) ===\n"
        "  Part A: C-3 power 0.40 at receiver-season, 0.88 at team-season. The rule's\n"
        "  second branch fires, so this is the fit the component reads."
    )
    arm2t, arm2t_idata = fit_conversion(
        "2t (team-season, the charged grain)",
        frame.design_matrix,
        frame.feature_names,
        frame.outcome,
        frame.team_season_codes,
        frame.n_team_seasons,
        frame.defence_season_codes,
        frame.n_defence_seasons,
        entity_name="team-season",
    )

    for name, arm in (("2", arm2), ("2b", arm2b), ("2t", arm2t)):
        print(
            f"  Gate C-1 (arm {name}): divergences {arm['sampler']['divergences']}, "
            f"max r_hat {arm['sampler']['max_r_hat']:.4f}, "
            f"min ess_bulk {arm['sampler']['min_ess_bulk']:.0f}, "
            f"min ess_tail {arm['sampler']['min_ess_tail']:.0f} -> "
            f"{'PASS' if arm['sampler']['pass'] else 'FAIL'}"
        )
    if not all(arm["sampler"]["pass"] for arm in (arm2, arm2b, arm2t)):
        raise SystemExit(
            "an arm missed Gate C-1 at document 54 F-1's spec. Handoff constraint 6: "
            "stop and ask — no second spec is chosen mid-round."
        )

    # --- V-8 on the fit the component reads ----------------------------------
    arm2t_idata = name_the_levels(arm2t_idata, team_levels, defence_levels)
    arm2_idata = name_the_levels(arm2_idata, receiver_levels, defence_levels)
    v8 = v8_report(
        arm2t_idata, frame.design_matrix, frame.feature_names, team_levels, "team-season"
    )

    # --- arm 3, the gate arm -------------------------------------------------
    print("\n=== arm 3 — persistence of the conditioned residual (Gates C-2, C-3) ===")
    beta_hat = json.loads((paths.RESEARCH_OUTPUT_DIR / "71_beta_hat.json").read_text())
    eta = _receiver.linear_predictor(beta_hat, frame.model, frame.model)
    p_hat = 1.0 / (1.0 + np.exp(-eta))
    residual = frame.outcome - p_hat

    refit_beta = np.array([row["mean"] for row in arm2["beta"]])
    beta_gap = float(np.abs(refit_beta - np.asarray(beta_hat["beta"])).max())
    alpha_gap = abs(arm2["alpha_mean"] - beta_hat["alpha"])
    print(
        f"  arm 2 vs Part A's saved fixed effects: max |d beta| {beta_gap:.4f}, "
        f"|d alpha| {alpha_gap:.4f}"
    )

    receiver_codes, n_receiver = _power._codes(frame.model, ["season", "receiver_player_id"])
    team_codes, n_team = _power._codes(frame.model, ["season", "posteam"])
    defence_codes, n_defence = _power._codes(frame.model, ["season", "defteam"])

    arm3 = {}
    for key, name, codes, size, power_key, arm2_upper in (
        (
            "receiver_season_x_defence_season",
            "receiver-season x defence-season",
            receiver_codes,
            n_receiver,
            "residual_receiver_season_x_defence_season",
            arm2["variance_components"]["sigma_r"]["pp_eti89"][1],
        ),
        (
            "team_season_x_defence_season",
            "team-season x defence-season",
            team_codes,
            n_team,
            "residual_team_season_x_defence_season",
            arm2t["variance_components"]["sigma_r"]["pp_eti89"][1],
        ),
    ):
        arm3[key] = residual_persistence_blocked(
            name, residual, codes, size, defence_codes, n_defence, power[power_key], arm2_upper
        )

    # The defence-season SD, read off the receiver design's crossed fit — the
    # third grain Part A powered, and the one the read side deliberately drops.
    defence_power = power["residual_defence_season_sigma_d"]
    defence_upper = arm3["receiver_season_x_defence_season"]["sigma_b_eti89_pp"][1]
    arm3["defence_season_sigma_d"] = {
        "name": "defence-season sigma_d (read from the receiver-season x defence-season fit)",
        "upper_bound_pp": defence_upper,
        "gate_threshold_pp": defence_power["gate_threshold_pp"],
        "gate_c2_pass": bool(defence_upper < defence_power["gate_threshold_pp"]),
        "power_at_reference": defence_power["power_at_reference"],
        "gate_c3_pass": bool(defence_power["resolvable"]),
        "reportable_as_finding": bool(defence_power["resolvable"]),
    }
    entry = arm3["defence_season_sigma_d"]
    print(
        f"\n  defence-season sigma_d: upper bound {defence_upper:.2f} pp vs threshold "
        f"{entry['gate_threshold_pp']:.2f} pp -> "
        f"{'PASS' if entry['gate_c2_pass'] else 'FAIL'}; power "
        f"{entry['power_at_reference']:.3f} -> {'PASS' if entry['gate_c3_pass'] else 'FAIL'}"
    )

    # --- secondaries ---------------------------------------------------------
    print("\n=== secondaries (reported, never gated) ===")
    conditioned = frame.model.with_columns(pl.Series("residual", residual))
    raw = frame.catchable.with_columns(pl.col("is_drop").cast(pl.Float64))
    secondaries = {
        "raw_drop_split_half_team_season": _confounds.split_half(
            raw, "is_drop", ["season", "posteam"]
        ),
        "conditioned_residual_split_half_team_season": _confounds.split_half(
            conditioned, "residual", ["season", "posteam"]
        ),
        "raw_drop_split_half_receiver_season": _confounds.split_half(
            raw.drop_nulls("receiver_player_id"), "is_drop", ["season", "receiver_player_id"]
        ),
        "conditioned_residual_split_half_receiver_season": _confounds.split_half(
            conditioned, "residual", ["season", "receiver_player_id"]
        ),
    }
    for name, entry in secondaries.items():
        value = "n/a" if entry["r"] is None else f"{entry['r']:+.3f}"
        print(f"  {name}: r = {value} on {entry['entities']} entities")

    effect_means = arm2t_idata["posterior"]["r_s"].values.mean(axis=(0, 1))
    effects = pl.DataFrame(
        {
            "season": [int(label.split("|")[0]) for label in team_levels],
            "defteam": [label.split("|")[1] for label in team_levels],
            "effect": effect_means[: len(team_levels)],
        }
    )
    secondaries["shrunk_team_effect_season_to_season"] = _confounds.season_to_season(effects)
    entry = secondaries["shrunk_team_effect_season_to_season"]
    value = "n/a" if entry["r"] is None else f"{entry['r']:+.3f}"
    print(f"  shrunk_team_effect_season_to_season: r = {value} on {entry['pairs']} pairs")

    # --- the hindsight substitutes -------------------------------------------
    probe = hindsight_substitutes(frame.charted, frame.catchable)

    # --- the artifacts the read side loads -----------------------------------
    trace_path = paths.RESEARCH_OUTPUT_DIR / TRACE_NAME
    arm2t_idata.to_netcdf(trace_path)
    arm2_idata.to_netcdf(paths.RESEARCH_OUTPUT_DIR / RECEIVER_GRAIN_TRACE)

    standardisation = {
        column: {
            "mean": float(frame.model[column].cast(pl.Float64).mean()),
            "sd": float(frame.model[column].cast(pl.Float64).std()),
        }
        for column in _receiver.STANDARDISED
    }
    beta_means = arm2t_idata["posterior"]["beta"].values.mean(axis=(0, 1))
    summary = {
        "document": "56 — the receiver-drop mirror (A-3 gate G-4)",
        "fitted_by": "research/72_receiver_drop_confounds.py",
        "model": (
            "document 56 §1 arm 2 with the charged grain substituted per §1's clause-1 "
            "rule (logit p(drop) = alpha + X beta + t_s + d_d), document 54 F-1's spec"
        ),
        "charged_grain": "team-season",
        "charged_grain_reason": (
            "Part A: C-3 power 0.40 at receiver-season against a 0.80 bar, 0.88 at "
            "team-season. Document 56 §1's clause-1 rule selects team-season."
        ),
        "read_side_note": (
            "p_i excludes the defence-season effect by design (document 56 §2): the "
            "coverage's contribution to a drop is the defence's football and stays in "
            "`core`. It is fitted so the offensive term is estimated free of schedule, "
            "and never read. The modelled p is P(drop); the component books P(catch)."
        ),
        "fit_seed": RANDOM_SEED,
        "draws": DRAWS,
        "tune": TUNE,
        "chains": CHAINS,
        "target_accept": TARGET_ACCEPT,
        "n_posterior_draws": int(arm2t_idata["posterior"]["alpha"].values.size),
        "rows": int(frame.model.height),
        "catchable_rows": int(frame.catchable.height),
        "n_entity_seasons": int(frame.n_team_seasons),
        "n_defence_seasons": int(frame.n_defence_seasons),
        "guards": frame.guards,
        "covariate_order": list(frame.feature_names),
        "standardisation": standardisation,
        "standardisation_from_rows": int(frame.model.height),
        "reference_levels": REFERENCE_LEVELS,
        "standardised_covariates": list(_receiver.STANDARDISED),
        "entity_season_levels": team_levels,
        "defence_season_levels": defence_levels,
        "swing_table": table,
        "per_play_swing": swing,
        "beta_means": {
            name: float(value) for name, value in zip(frame.feature_names, beta_means, strict=True)
        },
        "alpha_mean": float(arm2t_idata["posterior"]["alpha"].values.mean()),
        "sigma_entity_mean": float(arm2t_idata["posterior"]["sigma_r"].values.mean()),
        "sigma_defence_mean": float(arm2t_idata["posterior"]["sigma_d"].values.mean()),
        "gate_v6_sampler": arm2t["sampler"],
        "gate_v8_posterior_spread": v8,
    }
    summary_path = paths.RESEARCH_OUTPUT_DIR / SUMMARY_NAME
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {trace_path}")
    print(f"wrote {summary_path}")

    # --- the grain decision, as one line -------------------------------------
    charged = "team-season"
    print(
        f"\nCLAUSE-1 GRAIN DECISION: receiver-season C-3 "
        f"{power['residual_receiver_season_x_defence_season']['power_at_reference']:.2f} "
        f"(fails the 0.80 bar), team-season C-3 "
        f"{power['residual_team_season_x_defence_season']['power_at_reference']:.2f} "
        f"(clears it) -> the component charges the {charged.upper()}, the receiving corps."
    )

    results = {
        "document": "56 — the receiver-drop mirror",
        "part": "B — the study, arms 1-3",
        "random_seed": RANDOM_SEED,
        "sampler_spec": {
            "draws": DRAWS,
            "tune": TUNE,
            "chains": CHAINS,
            "target_accept": TARGET_ACCEPT,
        },
        "cross_check_tolerance_pp": CROSS_CHECK_TOLERANCE_PP,
        "guards": frame.guards,
        "swing_table": table,
        "per_play_swing": swing,
        "arm1_drop_rate_spreads": arm1,
        "arm2_receiver_season": arm2,
        "arm2b_no_contested": arm2b,
        "arm2t_team_season_charged": arm2t,
        "arm2_vs_part_a_beta": {"max_abs_beta_gap": beta_gap, "alpha_gap": alpha_gap},
        "arm3_residual_persistence": arm3,
        "clause_1_grain_decision": charged,
        "gate_v8_posterior_spread": v8,
        "secondaries": secondaries,
        "hindsight_substitutes": probe,
    }
    out = paths.RESEARCH_OUTPUT_DIR / OUTPUT_NAME
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out}")
    print("Next: the component (Part C), then research/73 for the gates and the audit.")


def residual_persistence_blocked(
    name: str,
    residual: np.ndarray,
    code_a: np.ndarray,
    size_a: int,
    code_b: np.ndarray,
    size_b: int,
    power: dict,
    arm2_upper_pp: float,
) -> dict:
    """`62.residual_persistence`, evaluated on the blocked grid.

    Identical statistic, identical gates, identical printing — only the module
    that evaluates the grid differs, and `research/_crossed_block_grid.self_check`
    is what licenses that (`research/71` prints it before any number).

    The cross-check is **like-grain only**, which is document 47's ruling R-1:
    arm 2's `sigma` at a grain against arm 3's `sigma_a` at *that same* grain.
    """
    _block = import_module("_crossed_block_grid")
    fitted = _block.fit_from_codes(code_a, code_b, size_a, size_b, residual)

    upper_pp = fitted["sigma_a"]["eti89_ub"] * 100
    threshold_pp = power["gate_threshold_pp"]
    c2_pass = upper_pp < threshold_pp
    c3_pass = power["resolvable"]
    gap = abs(upper_pp - arm2_upper_pp)

    report = {
        "name": name,
        "rows": int(len(residual)),
        "levels_a": int(size_a),
        "levels_b": int(size_b),
        "sigma_a_mean_pp": fitted["sigma_a"]["mean"] * 100,
        "sigma_a_eti89_pp": [
            fitted["sigma_a"]["eti89_lb"] * 100,
            fitted["sigma_a"]["eti89_ub"] * 100,
        ],
        "sigma_b_mean_pp": fitted["sigma_b"]["mean"] * 100,
        "sigma_b_eti89_pp": [
            fitted["sigma_b"]["eti89_lb"] * 100,
            fitted["sigma_b"]["eti89_ub"] * 100,
        ],
        "sigma_e_mean_pp": fitted["sigma_e"]["mean"] * 100,
        "edge_mass": fitted["edge_mass"],
        "gate_threshold_pp": threshold_pp,
        "gate_c2_pass": bool(c2_pass),
        "power_at_reference": power["power_at_reference"],
        "gate_c3_pass": bool(c3_pass),
        "reportable_as_finding": bool(c3_pass),
        "arm2_upper_bound_pp": arm2_upper_pp,
        "cross_check_gap_pp": gap,
        "cross_check_pass": bool(gap <= CROSS_CHECK_TOLERANCE_PP),
    }

    print(f"\n  {name}: {report['rows']:,} rows, {size_a:,} x {size_b} levels")
    print(
        f"    sigma_a {report['sigma_a_mean_pp']:.2f} pp "
        f"[{report['sigma_a_eti89_pp'][0]:.2f}, {report['sigma_a_eti89_pp'][1]:.2f}] 89%; "
        f"sigma_b {report['sigma_b_mean_pp']:.2f} pp "
        f"[{report['sigma_b_eti89_pp'][0]:.2f}, {report['sigma_b_eti89_pp'][1]:.2f}]; "
        f"residual SD {report['sigma_e_mean_pp']:.2f} pp; edge mass {report['edge_mass']:.1e}"
    )
    print(
        f"    Gate C-2, {name}: upper bound {upper_pp:.2f} pp vs threshold "
        f"{threshold_pp:.2f} pp -> {'PASS' if c2_pass else 'FAIL'}"
    )
    print(
        f"    Gate C-3: power at 12.5% = {report['power_at_reference']:.3f} -> "
        f"{'PASS' if c3_pass else 'FAIL'}"
        + ("" if c3_pass else " -> C-2 not reportable as a finding")
    )
    print(
        f"    Gate C-1 cross-check vs arm 2 at the same grain ({arm2_upper_pp:.2f} pp): "
        f"gap {gap:.2f} pp -> {'PASS' if report['cross_check_pass'] else 'FAIL'} "
        f"(tolerance {CROSS_CHECK_TOLERANCE_PP:.1f} pp, like-grain only per document 47 R-1)"
    )
    return report


if __name__ == "__main__":
    main()
