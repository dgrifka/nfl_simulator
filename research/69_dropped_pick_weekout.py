"""Round 6 — gate G-1: is a game's `u_d` the game's own?

Amendment A-3's clause 5 (document 52 §3) requires that the entity effect the
variant reads for a game be shown, *by a held-out check*, not to be materially
the game's own. Document 52 §5 makes that concrete: refit the dropped-pick model
leaving one week of season out at a time (all four seasons' week `w` together),
read each game's `u_d` from the fit that excluded its week, and re-run round 4's
audit on those draws.

**The bars, verbatim from document 52 §5.** Element-wise bucket agreement
between the in-sample variant and the week-out variant over 2022-2025 must be
**>= 0.90**, and the median `|ΔDTW|` between them on affected games must be
**< 1.0 pp**. Both, or the gate fails. On a pass, production keeps the in-sample
read with this bound recorded; on a fail, production must use the week-out
traces.

**What document 54 changed, and why this file is not round 5's.** Round 5 fitted
eighteen folds at amendment A-2's sampler spec and seven of them missed Gate
C-1 — on the variance components, with zero divergences in all eighteen. That is
a chain-length failure, so document 54 amends three things:

* **F-1** — every fit, folds *and* the default, samples 4 chains x 4,000 draws
  after 4,000 tuning at `target_accept` 0.95. The two arms of G-1 are compared
  at one spec, which is why the default fit is re-run rather than reused.
* **F-2** — a nineteenth fold holds out weeks 19-22 **together** (147 worthy
  throws), so every 2022-2025 game has a fit that excluded its week and G-1 is
  measured on all 1,139 games instead of the 1,091 weeks 1-18 covered.
* **F-3** — the in-sample arm is re-fitted at F-1's spec and round 4's audit
  re-run on it *before* G-1 is computed, against round 4's 137 bucket moves and
  1.62 pp. Drift beyond ±5 games or ±0.2 pp is a surprise to record, not a
  result to interpret, and stops the run.

Three things about how the folds are built, each of which could have been done a
worse way silently.

* **Only the row mask changes.** The nineteen fits are the same model at the
  same sampler spec (`67.fit`), and the covariate scale is the *full* frame's,
  passed through `61.design_matrix`'s ``reference`` argument, so a fold differs
  from the default run only in which rows the likelihood sees. A standardisation
  refitted per fold would have made every fold's `beta` live on its own scale
  and mixed a reparameterisation into the gate's statistic.
* **`u_d` keeps one meaning across all nineteen fits.** The level codes are
  built against the full frame's defence-season list, so level *k* is the same
  defence-season in every fold and a level whose only rows were held out simply
  shrinks to the prior — which is the honest answer for it, and is counted
  below.
* **The swing table is the in-sample table, unchanged across folds.** G-1 holds
  out the *entity effect*; the bin prices are a pooled descriptive quantity and
  G-2 is the gate that interrogates them.

Run `research/67_dropped_pick_model.py` first: this script reads the default
trace and refuses to start unless it was fitted at F-1's spec.

    uv run python research/69_dropped_pick_weekout.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_power = import_module("61_dropped_pick_power")
_fit = import_module("67_dropped_pick_model")
_audit = import_module("68_dropped_pick_variant_audit")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.dropped_picks import DroppedPickModel  # noqa: E402

# Document 52 §5 and document 54 F-2, verbatim. Not re-tolerated for any reason.
REGULAR_WEEKS = tuple(range(1, 19))
POSTSEASON_FOLD = "post"
POSTSEASON_WEEKS = (19, 20, 21, 22)
FOLDS: tuple = (*REGULAR_WEEKS, POSTSEASON_FOLD)
WEEKS_BY_FOLD: dict = {
    **{week: (week,) for week in REGULAR_WEEKS},
    POSTSEASON_FOLD: POSTSEASON_WEEKS,
}

FOLD_SEED_BASE = 20260827  # seed per fold = FOLD_SEED_BASE + week
POSTSEASON_SEED_OFFSET = 19  # document 54 F-2, verbatim

G1_MIN_AGREEMENT = 0.90
G1_MAX_MEDIAN_ABS_DELTA_DTW_PP = 1.0

# Handoff constraint 6 of round 5, carried forward: over this, stop and ask
# rather than report and continue. Nineteen fits at F-1's spec are perhaps four
# times round 5's 188 seconds, so this is a tripwire and not a schedule.
FOLD_WALL_CLOCK_BUDGET_S = 90 * 60

# Document 54 F-3. The in-sample arm is re-fitted at the new spec, so round 4's
# audit numbers are expected to reproduce *within sampler noise*, not exactly:
# the counts that do not depend on the fit stay exact, and the two that do get
# F-3's tolerances.
ROUND4_GAMES = 1139
ROUND4_AFFECTED = 1033
ROUND4_BUCKET_MOVES = 137
ROUND4_MEDIAN_ABS_DELTA_DTW_PP = 1.62
F3_MOVE_TOLERANCE = 5  # games
F3_MEDIAN_TOLERANCE_PP = 0.2

# Round 4's interval widths on the affected games. Document 54 F-3 names the
# bucket-move count and the median |ΔDTW| and nothing else, so these are printed
# beside their round-4 values and are **reported, not gated**.
ROUND4_MEAN_WIDTH_V13 = 0.0383
ROUND4_MEAN_WIDTH_VARIANT = 0.0511

# Document 54 F-2's count, as the frame carries it.
EXPECTED_POSTSEASON_ROWS = 147

N_LARGEST_DISAGREEMENTS = 10

TRACE_PATTERN = "trace_dropped_pick_wk{fold}.nc"
SUMMARY_PATTERN = "dropped_pick_summary_wk{fold}.json"

OUTPUT_NAME = "69_dropped_pick_weekout.json"


# --------------------------------------------------------------------------
# the folds themselves
# --------------------------------------------------------------------------


def weeks_of(fold) -> tuple[int, ...]:
    """The weeks a fold holds out. One, except the postseason fold's four."""
    return WEEKS_BY_FOLD[fold]


def seed_of(fold) -> int:
    """Document 52 §7's per-fold seed, with document 54 F-2's `+ 19` postseason."""
    offset = POSTSEASON_SEED_OFFSET if fold == POSTSEASON_FOLD else fold
    return FOLD_SEED_BASE + offset


def fold_label(fold) -> str:
    return "post" if fold == POSTSEASON_FOLD else f"{fold}"


def check_the_fold_list(frame: pl.DataFrame) -> dict:
    """Every game's week belongs to exactly one fold, and the folds cover them all.

    Document 54 F-2 exists because round 5's eighteen folds left the postseason
    inside every fold's training data. A partition that quietly overlapped, or
    quietly missed a week, would put that failure back without saying so.
    """
    seen: dict[int, list] = {}
    for fold in FOLDS:
        for week in weeks_of(fold):
            seen.setdefault(week, []).append(fold)
    overlapping = {week: folds for week, folds in seen.items() if len(folds) > 1}
    weeks_in_frame = sorted(int(week) for week in frame["week"].unique().to_list())
    uncovered = [week for week in weeks_in_frame if week not in seen]
    report = {
        "n_folds": len(FOLDS),
        "weeks_covered": sorted(seen),
        "weeks_in_frame": weeks_in_frame,
        "weeks_covered_twice": {str(k): [fold_label(f) for f in v] for k, v in overlapping.items()},
        "weeks_in_frame_with_no_fold": uncovered,
        "pass": not overlapping and not uncovered,
    }
    print(f"\n{'=' * 72}\nTHE FOLD LIST — document 54 F-2\n{'=' * 72}")
    print(
        f"  {report['n_folds']} folds covering weeks {weeks_in_frame[0]}-{weeks_in_frame[-1]}; "
        f"the postseason fold holds out weeks {list(POSTSEASON_WEEKS)} together"
    )
    print(f"  every week in exactly one fold: {'yes' if report['pass'] else 'NO'}")
    if not report["pass"]:
        raise SystemExit(
            f"the fold list does not partition the weeks: covered twice "
            f"{report['weeks_covered_twice']}, uncovered {uncovered}. Stop and ask."
        )
    return report


def codes_against(frame: pl.DataFrame, keys: list[str], levels: list[str]) -> np.ndarray:
    """Level codes for `frame`'s rows against a *fixed* level list.

    `61._codes` derives the levels from the frame it is given, which is right for
    the frame it was written for and wrong for a fold: a subset would renumber
    the levels and `u_d[k]` would name a different defence-season in every fit.
    """
    lookup = {label: index for index, label in enumerate(levels)}
    labels = frame.select(pl.concat_str(keys, separator="|").alias("label"))["label"].to_list()
    missing = sorted({label for label in labels if label not in lookup})
    if missing:
        raise SystemExit(f"levels absent from the full frame's list: {missing[:5]} — stop and ask.")
    return np.array([lookup[label] for label in labels], dtype=int)


def masked_frame(full, fold, defence_levels: list[str], qb_levels: list[str]):
    """The full frame with one fold's weeks removed, everything else held.

    `n_defence_seasons` and `n_qb_seasons` stay at the full frame's counts on
    purpose: a defence-season whose only worthy throws were in the held-out weeks
    still exists in the model, with no rows of its own, and shrinks to `u_d = 0`.
    That is the correct read for it — no evidence, no entity term, document 05
    §1's `w = 0` endpoint — and the count of such levels is reported per fold.
    """
    weeks = list(weeks_of(fold))
    model = full.model.filter(~pl.col("week").is_in(weeks))
    held_out = full.model.height - model.height
    if not held_out:
        raise SystemExit(f"fold {fold_label(fold)} removed no rows — stop and ask.")
    if fold == POSTSEASON_FOLD and held_out != EXPECTED_POSTSEASON_ROWS:
        raise SystemExit(
            f"the postseason fold holds out {held_out} rows against document 54 "
            f"F-2's {EXPECTED_POSTSEASON_ROWS} — stop and ask."
        )

    matrix, names = _power.design_matrix(model, reference=full.model)
    if tuple(names) != tuple(full.feature_names):
        raise SystemExit("the fold's design columns are not the full frame's — stop and ask.")

    masked = replace(
        full,
        model=model,
        design_matrix=matrix,
        outcome=model["interception"].cast(pl.Float64).to_numpy(),
        defence_season_codes=codes_against(model, ["season", "defteam"], defence_levels),
        qb_season_codes=codes_against(model, ["season", "passer_player_id"], qb_levels),
    )
    empty_defence = len(defence_levels) - len(set(masked.defence_season_codes.tolist()))
    empty_qb = len(qb_levels) - len(set(masked.qb_season_codes.tolist()))
    return masked, {
        "fold": fold_label(fold),
        "weeks_held_out": list(weeks),
        "rows_fitted": int(model.height),
        "rows_held_out": int(held_out),
        "defence_seasons_with_no_rows": int(empty_defence),
        "qb_seasons_with_no_rows": int(empty_qb),
    }


# --------------------------------------------------------------------------
# the nineteen fits
# --------------------------------------------------------------------------


def fit_the_folds(full, table) -> tuple[dict, list[dict]]:
    """One fit per fold, each printing Gate C-1, each saved beside the default run."""
    defence_levels = _fit._labels(full.model, ["season", "defteam"])
    qb_levels = _fit._labels(full.model, ["season", "passer_player_id"])

    models, folds = {}, []
    started = time.time()
    for fold in FOLDS:
        masked, mask_report = masked_frame(full, fold, defence_levels, qb_levels)
        seed = seed_of(fold)
        print(
            f"\n{'-' * 72}\nFOLD {fold_label(fold)} held out (weeks "
            f"{mask_report['weeks_held_out']}) — {mask_report['rows_held_out']} rows out, "
            f"{mask_report['rows_fitted']:,} fitted, seed {seed}\n{'-' * 72}"
        )
        fold_started = time.time()
        # The gate is enforced after the loop, not inside it: an unhealthy fold
        # is a stop-and-ask, and an ask is only useful if it says whether the
        # problem is one fold or nineteen. The bars are `62.sampler_health`'s,
        # unchanged, and `gate_c1_over_the_folds` below refuses to compute G-1
        # unless every fold clears them.
        idata, health = _fit.fit(
            masked, seed, label=f"fold {fold_label(fold)} held out", stop_on_c1=False
        )
        idata, _, _ = _fit.name_the_levels(
            idata, masked, defence_levels=defence_levels, qb_levels=qb_levels
        )

        name = fold_label(fold)
        trace_path = paths.RESEARCH_OUTPUT_DIR / TRACE_PATTERN.format(fold=name)
        summary_path = paths.RESEARCH_OUTPUT_DIR / SUMMARY_PATTERN.format(fold=name)
        idata.to_netcdf(trace_path)
        summary = _fit.build_summary(
            idata,
            masked,
            table,
            defence_levels=defence_levels,
            qb_levels=qb_levels,
            seed=seed,
            scale_frame=full.model,
            extra={
                "fitted_by": "research/69_dropped_pick_weekout.py",
                "gate": "document 52 §5 G-1 at document 54's fold spec, week-out fold",
                "fold": name,
                "weeks_held_out": list(weeks_of(fold)),
                "mask": mask_report,
                "gate_c1_sampler": health,
            },
        )
        summary_path.write_text(json.dumps(summary, indent=2, default=str))

        models[fold] = DroppedPickModel.from_posterior(trace_path, summary_path)
        elapsed = time.time() - fold_started
        folds.append(
            {
                **mask_report,
                "seed": seed,
                "sigma_d_mean": summary["sigma_d_mean"],
                "sigma_q_mean": summary["sigma_q_mean"],
                "alpha_mean": summary["alpha_mean"],
                "c1_pass": bool(health["pass"]),
                "divergences": health.get("divergences"),
                "max_r_hat": health.get("max_r_hat"),
                "max_r_hat_parameter": health.get("max_r_hat_parameter"),
                "min_ess_bulk": health.get("min_ess_bulk"),
                "min_ess_tail": health.get("min_ess_tail"),
                "wall_clock_s": elapsed,
            }
        )
        print(
            f"  fold {fold_label(fold):>4s}: sigma_d {summary['sigma_d_mean']:.4f}, "
            f"alpha {summary['alpha_mean']:+.4f}, {elapsed:.0f} s"
        )

        spent = time.time() - started
        if spent > FOLD_WALL_CLOCK_BUDGET_S:
            raise SystemExit(
                f"the folds have spent {spent / 60:.1f} minutes, over the 90-minute "
                f"budget, with {len(FOLDS) - len(folds)} to go. Stop and ask — "
                "reporting and continuing is the wrong call here."
            )

    print(f"\n  {len(FOLDS)} folds in {(time.time() - started) / 60:.1f} minutes")
    return models, folds


def gate_c1_over_the_folds(folds: list[dict]) -> dict:
    """Gate C-1 on all nineteen folds at once — the precondition G-1 rests on.

    Document 52 §5's G-1 is a statement about a set of fits, so it may only be
    read off fits that sampled. A fold that missed C-1 is not a fold whose `u_d`
    can be compared with anything, and document 54 F-1 says what to do about it:
    stop and report, before the statistic exists. **No third spec is chosen
    mid-round.**
    """
    print(f"\n{'=' * 72}\nGATE C-1 OVER THE {len(folds)} FOLDS\n{'=' * 72}")
    print(
        f"  {'fold':>4s} {'rows':>6s} {'div':>4s} {'max r_hat':>10s} {'on':>12s} "
        f"{'ess_bulk':>9s} {'ess_tail':>9s} {'sigma_d':>8s}  C-1"
    )
    failures = []
    for fold in folds:
        if not fold["c1_pass"]:
            failures.append(fold["fold"])
        print(
            f"  {fold['fold']:>4s} {fold['rows_fitted']:6,d} {fold['divergences']:4d} "
            f"{fold['max_r_hat']:10.4f} {str(fold['max_r_hat_parameter']):>12s} "
            f"{fold['min_ess_bulk']:9.0f} {fold['min_ess_tail']:9.0f} "
            f"{fold['sigma_d_mean']:8.4f}  {'PASS' if fold['c1_pass'] else 'FAIL'}"
        )
    report = {
        "bars": "0 divergences, max r_hat < 1.01, min ess_bulk > 400, min ess_tail > 400",
        "spec": "document 54 F-1: 4 x 4,000 draws after 4,000 tuning, target_accept 0.95",
        "folds_checked": len(folds),
        "folds_failed": failures,
        "n_failed": len(failures),
        "divergences_total": int(sum(fold["divergences"] for fold in folds)),
        "max_r_hat_over_folds": max(fold["max_r_hat"] for fold in folds),
        "min_ess_bulk_over_folds": min(fold["min_ess_bulk"] for fold in folds),
        "min_ess_tail_over_folds": min(fold["min_ess_tail"] for fold in folds),
        "sigma_d_range": [
            min(fold["sigma_d_mean"] for fold in folds),
            max(fold["sigma_d_mean"] for fold in folds),
        ],
        "pass": not failures,
    }
    print(
        f"  C-1 over the folds: {'PASS' if report['pass'] else 'FAIL'} — "
        f"{report['n_failed']} of {report['folds_checked']} folds miss the bars"
        + (f", folds {failures}" if failures else "")
    )
    print(
        f"  extremes: max r_hat {report['max_r_hat_over_folds']:.4f}, min ess_bulk "
        f"{report['min_ess_bulk_over_folds']:.0f}, min ess_tail "
        f"{report['min_ess_tail_over_folds']:.0f}, {report['divergences_total']} divergences"
    )
    return report


# --------------------------------------------------------------------------
# G-1
# --------------------------------------------------------------------------


def gate_g1(in_sample: pl.DataFrame, week_out: pl.DataFrame) -> dict:
    """Document 52 §5's G-1, on every 2022-2025 game the folds now cover."""
    joined = (
        in_sample.select(
            "game_id",
            "actual_margin",
            pl.col("dtw_home").alias("dtw_in"),
            pl.col("deserved_margin").alias("margin_in"),
            pl.col("n_dropped_pick_events").alias("events_in"),
        )
        .join(
            week_out.select(
                "game_id",
                pl.col("dtw_home").alias("dtw_out"),
                pl.col("deserved_margin").alias("margin_out"),
                pl.col("n_dropped_pick_events").alias("events_out"),
            ),
            on="game_id",
            how="inner",
        )
        .sort("game_id")
    )
    if joined.height != in_sample.height or joined.height != week_out.height:
        raise SystemExit(
            f"the two arms do not cover the same games: in-sample {in_sample.height:,}, "
            f"week-out {week_out.height:,}, joined {joined.height:,} — stop and ask."
        )

    # The two arms build their events from the same charted throws, so an event
    # count that disagreed would mean the arms are not the same population and
    # the statistic below would be measuring the wrong thing.
    if (joined["events_in"] != joined["events_out"]).any():
        raise SystemExit("the two arms disagree on the event count — stop and ask.")

    rows = joined.to_dicts()
    bucket_in = [_audit.bucket(row["dtw_in"], row["actual_margin"]) for row in rows]
    bucket_out = [_audit.bucket(row["dtw_out"], row["actual_margin"]) for row in rows]
    agree = [a == b for a, b in zip(bucket_in, bucket_out, strict=True)]

    # The disagreements are named, not just counted. G-1's statistic *is* the
    # disagreement set, and a record that reports "three games" without saying
    # which three cannot be checked by the next reader — document 33's lesson,
    # one step further on.
    transitions: dict[str, int] = {}
    disagreeing = []
    for row, a, b in zip(rows, bucket_in, bucket_out, strict=True):
        if a != b:
            transitions[f"{a} -> {b}"] = transitions.get(f"{a} -> {b}", 0) + 1
            disagreeing.append(
                {
                    "game_id": row["game_id"],
                    "actual_margin": row["actual_margin"],
                    "dtw_in": row["dtw_in"],
                    "dtw_out": row["dtw_out"],
                    "abs_delta_dtw_pp": abs(row["dtw_out"] - row["dtw_in"]) * 100,
                    "margin_in": row["margin_in"],
                    "margin_out": row["margin_out"],
                    "events": row["events_in"],
                    "bucket_in_sample": a,
                    "bucket_week_out": b,
                }
            )
    disagreeing.sort(key=lambda entry: -entry["abs_delta_dtw_pp"])

    affected = joined["events_in"].to_numpy() > 0
    d_dtw = (joined["dtw_out"] - joined["dtw_in"]).abs().to_numpy() * 100
    d_margin = (joined["margin_out"] - joined["margin_in"]).abs().to_numpy()

    agreement = float(np.mean(agree))
    median_pp = float(np.median(d_dtw[affected]))
    passes = agreement >= G1_MIN_AGREEMENT and median_pp < G1_MAX_MEDIAN_ABS_DELTA_DTW_PP

    report = {
        "population": (
            "every 2022-2025 game, weeks 1-18 from their own week's fold and the "
            "postseason from the postseason fold (document 54 F-2)"
        ),
        "games_compared": int(joined.height),
        "postseason_games_not_covered": 0,
        "affected_games": int(affected.sum()),
        "bucket_agreement": agreement,
        "n_bucket_agree": int(sum(agree)),
        "n_bucket_disagreements": int(len(agree) - sum(agree)),
        "transitions_in_sample_to_week_out": transitions,
        "disagreeing_games": disagreeing,
        "median_abs_delta_dtw_pp_affected": median_pp,
        "eti89_abs_delta_dtw_pp_affected": [
            float(v) for v in np.percentile(d_dtw[affected], [_audit.ETI_LOW, _audit.ETI_HIGH])
        ],
        "max_abs_delta_dtw_pp_affected": float(d_dtw[affected].max()),
        "median_abs_delta_margin_affected": float(np.median(d_margin[affected])),
        "max_abs_delta_margin_affected": float(d_margin[affected].max()),
        "bars": {
            "min_agreement": G1_MIN_AGREEMENT,
            "max_median_abs_delta_dtw_pp": G1_MAX_MEDIAN_ABS_DELTA_DTW_PP,
        },
        "pass": bool(passes),
    }

    print(f"\n{'=' * 72}\nGATE G-1 — in-sample against week-out\n{'=' * 72}")
    print(
        f"  {report['games_compared']:,} games, every one of them read from a fit "
        f"that never saw its week"
    )
    for name, count in sorted(transitions.items(), key=lambda item: -item[1]):
        print(f"    {name:45s} {count:5d}")
    for entry in disagreeing:
        print(
            f"    {entry['game_id']:16s} actual {entry['actual_margin']:+3.0f}  DTW% "
            f"{entry['dtw_in'] * 100:5.1f} -> {entry['dtw_out'] * 100:5.1f} "
            f"({entry['abs_delta_dtw_pp']:4.2f} pp)  {entry['bucket_in_sample']} -> "
            f"{entry['bucket_week_out']}"
        )
    print(
        f"G-1: bucket agreement {agreement:.3f} "
        f"({sum(agree):,}/{len(agree):,}); median |dDTW| between arms "
        f"{median_pp:.2f} pp -> {'PASS' if passes else 'FAIL'}"
    )
    print(
        f"  bars: agreement >= {G1_MIN_AGREEMENT:.2f} "
        f"({'met' if agreement >= G1_MIN_AGREEMENT else 'NOT met'}); "
        f"median |dDTW| < {G1_MAX_MEDIAN_ABS_DELTA_DTW_PP:.1f} pp "
        f"({'met' if median_pp < G1_MAX_MEDIAN_ABS_DELTA_DTW_PP else 'NOT met'})"
    )
    if not passes:
        print(
            "  G-1 FAILS. Document 52 §5's consequence: production must use the "
            "week-out traces, and the cost is noted."
        )
    return report


def survival(label: str, v13: pl.DataFrame, variant: pl.DataFrame) -> dict:
    """Does the 12% survive? — bucket moves and median |ΔDTW| against v1.3."""
    moves = _audit.flips(v13, variant)
    move = _audit.movement(v13, variant)
    report = {
        "arm": label,
        "games": moves["games"],
        "n_bucket_moved": moves["n_bucket_moved"],
        "share_bucket_moved": moves["share_bucket_moved"],
        "median_abs_delta_dtw_pp_affected": move["affected_games"]["median_abs_delta_dtw_pp"],
        "median_abs_delta_margin_affected": move["affected_games"]["median_abs_delta_margin"],
        "affected_games": move["affected_games"]["n"],
        "interval_width_affected": move["interval_width_affected"],
        "bucket_counts_variant": moves["bucket_counts_variant"],
        "transitions": moves["transitions"],
    }
    print(
        f"\n{label} vs v1.3: {report['n_bucket_moved']} bucket moves "
        f"({report['share_bucket_moved']:.1%}); median |dDTW| affected "
        f"{report['median_abs_delta_dtw_pp_affected']:.2f} pp"
    )
    return report


def f3_reproduction(in_sample_full: dict) -> dict:
    """Document 54 F-3 — round 4's audit, re-run on the arm re-fitted at F-1's spec.

    Two of these are exact because they do not depend on the fit at all: the
    game count and the count of games carrying a dropped-pick event come from the
    charted data. The two that do depend on the fit get F-3's tolerances, ±5
    games and ±0.2 pp, and a drift past them stops the run as a surprise to
    record rather than a result to interpret.
    """
    checks = {
        "games": (in_sample_full["games"], ROUND4_GAMES, 0, "exact — data, not the fit"),
        "affected_games": (
            in_sample_full["affected_games"],
            ROUND4_AFFECTED,
            0,
            "exact — data, not the fit",
        ),
        "bucket_moves": (
            in_sample_full["n_bucket_moved"],
            ROUND4_BUCKET_MOVES,
            F3_MOVE_TOLERANCE,
            "document 54 F-3: +/- 5 games",
        ),
        "median_abs_delta_dtw_pp": (
            in_sample_full["median_abs_delta_dtw_pp_affected"],
            ROUND4_MEDIAN_ABS_DELTA_DTW_PP,
            F3_MEDIAN_TOLERANCE_PP,
            "document 54 F-3: +/- 0.2 pp",
        ),
    }
    print(f"\n{'=' * 72}\nDOCUMENT 54 F-3 — round 4's audit at F-1's spec\n{'=' * 72}")
    report, failures = {}, []
    for name, (got, want, tolerance, why) in checks.items():
        ok = abs(float(got) - float(want)) <= tolerance
        report[name] = {
            "round4": want,
            "reproduced": got,
            "tolerance": tolerance,
            "rule": why,
            "ok": bool(ok),
        }
        print(
            f"  {name:26s} round 4 {want:>8}   now {got:>8}   {'ok' if ok else 'DRIFTED'}   ({why})"
        )
        if not ok:
            failures.append(name)

    widths = in_sample_full["interval_width_affected"]
    report["interval_width_affected_reported_only"] = {
        "mean_v13": {"round4": ROUND4_MEAN_WIDTH_V13, "now": widths["mean_v13"]},
        "mean_variant": {"round4": ROUND4_MEAN_WIDTH_VARIANT, "now": widths["mean_variant"]},
    }
    print(
        f"  {'mean interval width':26s} round 4 v1.3 {ROUND4_MEAN_WIDTH_V13:.4f} / variant "
        f"{ROUND4_MEAN_WIDTH_VARIANT:.4f}   now {widths['mean_v13']:.4f} / "
        f"{widths['mean_variant']:.4f}   (reported only — F-3 names two statistics)"
    )

    report["pass"] = not failures
    print(f"  F-3: {'PASS' if report['pass'] else 'DRIFTED'}")
    if failures:
        raise SystemExit(
            f"document 54 F-3's reproduction drifted past its tolerance on {failures}. "
            "Stop and report — this is a surprise to record, not a result to interpret."
        )
    return report


def named_games_under_both_arms(
    v13: pl.DataFrame,
    in_sample: pl.DataFrame,
    week_out: pl.DataFrame,
    ledger_in: pl.DataFrame,
    ledger_out: pl.DataFrame,
) -> dict:
    """Document 52 §5's three named games, v1.3 and both variant arms side by side."""
    print(f"\n{'=' * 72}\nTHE THREE NAMED GAMES, UNDER BOTH ARMS\n{'=' * 72}")
    report = {}
    for game_id in _audit.NAMED_GAMES:
        frames = {
            "v13": v13.filter(pl.col("game_id") == game_id),
            "in_sample": in_sample.filter(pl.col("game_id") == game_id),
            "week_out": week_out.filter(pl.col("game_id") == game_id),
        }
        if any(not frame.height for frame in frames.values()):
            print(f"  {game_id}: not in every arm's population")
            report[game_id] = None
            continue
        rows = {name: frame.to_dicts()[0] for name, frame in frames.items()}
        actual = rows["v13"]["actual_margin"]
        print(f"  {game_id}  actual margin {actual:+.0f}")
        for name, label in (
            ("v13", "v1.3    "),
            ("in_sample", "in-sample"),
            ("week_out", "week-out "),
        ):
            row = rows[name]
            print(
                f"    {label} deserved {row['deserved_margin']:+.2f}  DTW% "
                f"{row['dtw_home'] * 100:5.1f}  89% [{row['dtw_low'] * 100:5.1f}, "
                f"{row['dtw_high'] * 100:5.1f}]  bucket: {_audit.bucket(row['dtw_home'], actual)}"
            )
        print("    dropped-pick rows, in-sample arm:")
        _audit.print_ledger_rows(_audit.ledger_rows_for(ledger_in, game_id))
        print("    dropped-pick rows, week-out arm:")
        _audit.print_ledger_rows(_audit.ledger_rows_for(ledger_out, game_id))
        report[game_id] = {
            **rows,
            "dropped_pick_rows_in_sample": _audit.ledger_rows_for(ledger_in, game_id),
            "dropped_pick_rows_week_out": _audit.ledger_rows_for(ledger_out, game_id),
        }
    return report


def largest_disagreements(
    in_sample: pl.DataFrame,
    week_out: pl.DataFrame,
    ledger_in: pl.DataFrame,
    ledger_out: pl.DataFrame,
    n: int = N_LARGEST_DISAGREEMENTS,
) -> list[dict]:
    """The games where holding a week out moves DTW% the most, with their rows.

    This is the tail G-1's median cannot see. A gate can pass on the median and
    still be wrong about the handful of games a reader would notice, so the
    largest disagreements are printed whichever way the gate lands.
    """
    joined = (
        in_sample.select(
            "game_id",
            "actual_margin",
            pl.col("deserved_margin").alias("margin_in"),
            pl.col("dtw_home").alias("dtw_in"),
            pl.col("n_dropped_pick_events").alias("events"),
        )
        .join(
            week_out.select(
                "game_id",
                pl.col("deserved_margin").alias("margin_out"),
                pl.col("dtw_home").alias("dtw_out"),
            ),
            on="game_id",
        )
        .with_columns(((pl.col("dtw_out") - pl.col("dtw_in")).abs() * 100).alias("abs_dtw_pp"))
        .sort("abs_dtw_pp", descending=True)
        .head(n)
    )
    print(f"\n{'=' * 72}\nTHE {n} LARGEST DISAGREEMENTS BETWEEN THE ARMS\n{'=' * 72}")
    out = []
    for row in joined.iter_rows(named=True):
        bucket_in = _audit.bucket(row["dtw_in"], row["actual_margin"])
        bucket_out = _audit.bucket(row["dtw_out"], row["actual_margin"])
        print(
            f"  {row['game_id']}  actual {row['actual_margin']:+.0f}  DTW% "
            f"{row['dtw_in'] * 100:5.1f} -> {row['dtw_out'] * 100:5.1f} "
            f"({row['abs_dtw_pp']:5.2f} pp)  deserved {row['margin_in']:+.2f} -> "
            f"{row['margin_out']:+.2f}  bucket {bucket_in} -> {bucket_out}  "
            f"({row['events']} events)"
        )
        print("    in-sample arm:")
        _audit.print_ledger_rows(_audit.ledger_rows_for(ledger_in, row["game_id"]))
        print("    week-out arm:")
        _audit.print_ledger_rows(_audit.ledger_rows_for(ledger_out, row["game_id"]))
        out.append(
            {
                **row,
                "bucket_in_sample": bucket_in,
                "bucket_week_out": bucket_out,
                "bucket_moved": bucket_in != bucket_out,
                "dropped_pick_rows_in_sample": _audit.ledger_rows_for(ledger_in, row["game_id"]),
                "dropped_pick_rows_week_out": _audit.ledger_rows_for(ledger_out, row["game_id"]),
            }
        )
    return out


def check_the_default_fit_is_at_f1s_spec() -> dict:
    """The in-sample arm must be the re-fit one, or the two arms differ by spec.

    Document 54 F-1's whole point is that G-1 compares like with like. Reading a
    trace fitted at A-2's spec against nineteen folds fitted at F-1's would put a
    sampler-spec difference inside the gate's statistic, so this refuses rather
    than warns.
    """
    summary_path = paths.RESEARCH_OUTPUT_DIR / _audit.SUMMARY_NAME
    if not summary_path.exists():
        raise SystemExit(
            f"{summary_path} is missing — run research/67_dropped_pick_model.py first."
        )
    summary = json.loads(summary_path.read_text())
    got = (summary.get("draws"), summary.get("tune"), summary.get("target_accept"))
    want = (_fit.DRAWS, _fit.TUNE, _fit.TARGET_ACCEPT)
    print(f"\n{'=' * 72}\nTHE IN-SAMPLE ARM's SPEC — document 54 F-1\n{'=' * 72}")
    print(
        f"  default trace fitted at {got[0]:,} draws / {got[1]:,} tuning / "
        f"target_accept {got[2]}; F-1 asks for {want[0]:,} / {want[1]:,} / {want[2]}  "
        f"-> {'ok' if got == want else 'MISMATCH'}"
    )
    if got != want:
        raise SystemExit(
            "the default trace was not fitted at document 54 F-1's spec. Re-run "
            "research/67_dropped_pick_model.py before this script — G-1 compares "
            "two arms and they must share a spec."
        )
    return {"draws": got[0], "tune": got[1], "target_accept": got[2], "pass": True}


# --------------------------------------------------------------------------


def main() -> None:
    print("=== Round 6 — gate G-1 at document 54's fold spec ===")

    full = _power.build_worthy_frame()
    if full.model.height != _fit.EXPECTED_MODEL_ROWS:
        raise SystemExit(f"the fit frame is {full.model.height:,} rows — stop and ask.")
    table, _ = _fit.swing_table_check(_fit.worthy_with_epa(full))

    spec = check_the_default_fit_is_at_f1s_spec()
    fold_list = check_the_fold_list(full.model)

    ctx = _audit.load_context()

    # ---- the in-sample arm, and F-3, before nineteen fits are spent ---------
    v13_table, _, v1 = _audit.v13_pass(ctx)
    in_sample_model = DroppedPickModel.from_posterior(
        paths.RESEARCH_OUTPUT_DIR / _audit.TRACE_NAME,
        paths.RESEARCH_OUTPUT_DIR / _audit.SUMMARY_NAME,
    )
    in_sample_table, in_sample_ledger = _audit.variant_pass(
        ctx, in_sample_model, label="in-sample variant (re-fit at F-1's spec)"
    )
    charted = v13_table.filter(pl.col("game_id").is_in(in_sample_table["game_id"].to_list()))
    in_sample_all = survival("in-sample variant (all games)", charted, in_sample_table)
    f3 = f3_reproduction(in_sample_all)

    # ---- the nineteen folds ------------------------------------------------
    models, folds = fit_the_folds(full, table)
    c1 = gate_c1_over_the_folds(folds)
    if not c1["pass"]:
        out = paths.RESEARCH_OUTPUT_DIR / OUTPUT_NAME
        out.write_text(
            json.dumps(
                {
                    "gate": "document 52 §5 G-1 (amendment A-3 clause 5)",
                    "spec": c1["spec"],
                    "status": "BLOCKED — the folds' Gate C-1 precondition failed again",
                    "folds_list": [fold_label(fold) for fold in FOLDS],
                    "fold_seed_base": FOLD_SEED_BASE,
                    "gate_c1_over_the_folds": c1,
                    "folds": folds,
                    "g1_computed": False,
                },
                indent=2,
                default=float,
            )
        )
        print(f"\nwrote {out}")
        raise SystemExit(
            f"{c1['n_failed']} of {len(folds)} folds miss Gate C-1 at document 54 "
            f"F-1's spec (folds {c1['folds_failed']}). F-1: stop and report — no "
            "third spec is chosen mid-round. G-1's statistic is NOT computed."
        )

    # ---- the week-out arm and G-1 -----------------------------------------
    week_out_table, week_out_ledger = _audit.variant_pass(
        ctx, None, models_by_week=models, weeks_by_fold=WEEKS_BY_FOLD, label="week-out variant"
    )

    g1 = gate_g1(in_sample_table, week_out_table)

    print(f"\n{'=' * 72}\nDOES THE 12% SURVIVE? — the two arms side by side\n{'=' * 72}")
    week_out_all = survival("week-out variant (all games)", charted, week_out_table)

    print(f"\n{'=' * 72}\nTHE WEEK-OUT VARIANT's OWN AUDIT\n{'=' * 72}")
    week_out_audit = _audit.audit(charted, week_out_table, week_out_ledger, named=False)

    named = named_games_under_both_arms(
        charted, in_sample_table, week_out_table, in_sample_ledger, week_out_ledger
    )
    disagreements = largest_disagreements(
        in_sample_table, week_out_table, in_sample_ledger, week_out_ledger
    )

    identity = _audit.round_trip_identity(week_out_table, ctx.slope)
    print(
        f"\n  V-2 on every week-out game: max |deserved − (actual − luck × slope)| = {identity:.2e}"
    )
    if identity > 1e-9:
        raise SystemExit("the week-out variant ledger does not sum. Stop and report.")

    results = {
        "gate": "document 52 §5 G-1 (amendment A-3 clause 5), at document 54's fold spec",
        "folds_list": [fold_label(fold) for fold in FOLDS],
        "weeks_by_fold": {fold_label(k): list(v) for k, v in WEEKS_BY_FOLD.items()},
        "fold_seed_base": FOLD_SEED_BASE,
        "postseason_seed_offset": POSTSEASON_SEED_OFFSET,
        "settings": {
            "sampler_spec": spec,
            "random_seed": _audit.RANDOM_SEED,
            "posterior_draws": _audit.POSTERIOR_DRAWS,
            "coin_draws": _audit.COIN_DRAWS,
            "points_per_epa": ctx.slope,
            "standardisation": "the full frame's, via 61.design_matrix's reference argument",
            "swing_table": "the in-sample table, unchanged across folds",
        },
        "fold_list_partition": fold_list,
        "gate_v1_default_off": v1,
        "document_54_f3": f3,
        "gate_c1_over_the_folds": c1,
        "folds": folds,
        "gate_g1": g1,
        "g1_computed": True,
        "survival": {
            "in_sample_all_games": in_sample_all,
            "week_out_all_games": week_out_all,
        },
        "week_out_audit": week_out_audit,
        "named_games_under_both_arms": named,
        "largest_disagreements": disagreements,
        "gate_v2_round_trip_max_residual": identity,
    }
    out = paths.RESEARCH_OUTPUT_DIR / OUTPUT_NAME
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {out}")

    # Handoff constraint 1: the V-1 line is printed again at the end of every
    # audit run, and it must read 0.00e+00.
    print(f"\n{'=' * 72}\nV-1, RE-PRINTED AT THE END OF THE RUN (handoff constraint 1)\n{'=' * 72}")
    print(
        f"V-1 replay: {v1['games_matched']:,} games, max |Δ deserved margin| "
        f"{v1['max_abs_gaps']['deserved_margin']:.2e}  -> "
        f"{'PASS' if v1['pass'] else 'FAIL'}"
    )
    if not v1["pass"]:
        raise SystemExit("v1.3 moved. Stop and report.")
    print(
        f"\nG-1: {'PASS' if g1['pass'] else 'FAIL'} — bucket agreement "
        f"{g1['bucket_agreement']:.3f} ({g1['n_bucket_agree']:,}/{g1['games_compared']:,}); "
        f"median |dDTW| between arms {g1['median_abs_delta_dtw_pp_affected']:.2f} pp"
    )


if __name__ == "__main__":
    main()
