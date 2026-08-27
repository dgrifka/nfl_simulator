"""Round 5, Part B — gate G-1: is a game's `u_d` the game's own?

Amendment A-3's clause 5 (document 52 §3) requires that the entity effect the
variant reads for a game be shown, *by a held-out check*, not to be materially
the game's own. Document 52 §5 makes that concrete: refit the dropped-pick model
**eighteen times, leaving out one week of season at a time** (all four seasons'
week `w` together), read each game's `u_d` from the fit that excluded its week,
and re-run round 4's audit on those draws.

**The bars, verbatim from document 52 §5 and §7.** Element-wise bucket agreement
between the in-sample variant and the week-out variant over 2022-2025 must be
**>= 0.90**, and the median `|ΔDTW|` between them on affected games must be
**< 1.0 pp**. Both, or the gate fails. On a pass, production keeps the in-sample
read with this bound recorded; on a fail, production must use the eighteen
week-out traces.

Three things about how this is built, each of which could have been done a worse
way silently.

* **Only the row mask changes.** Handoff constraint 3. The eighteen fits are the
  same model at the same sampler spec (`67.fit`), and the covariate scale is the
  *full* frame's, passed through `61.design_matrix`'s ``reference`` argument, so a
  fold differs from the default run only in which rows the likelihood sees. A
  standardisation refitted per fold would have made every fold's `beta` live on
  its own scale and mixed a reparameterisation into the gate's statistic.
* **`u_d` keeps one meaning across all eighteen fits.** The level codes are built
  against the full frame's defence-season list, so level *k* is the same
  defence-season in every fold and a level whose only rows were held out simply
  shrinks to the prior — which is the honest answer for it, and is counted below.
* **The postseason is not covered by the pre-registration, and is excluded rather
  than fudged.** Document 52 §7 fixes eighteen folds at weeks 1-18; the fit frame
  also carries 147 throws in weeks 19-22, and those games' weeks are inside every
  fold's training data. Including them would push the agreement statistic *up*
  (they agree by construction), so G-1 is measured on weeks 1-18 and the
  postseason count is reported beside it.

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

# Document 52 §5 and §7, verbatim. Not re-tolerated for any reason.
WEEKS = tuple(range(1, 19))
FOLD_SEED_BASE = 20260827  # seed per fold = FOLD_SEED_BASE + week
G1_MIN_AGREEMENT = 0.90
G1_MAX_MEDIAN_ABS_DELTA_DTW_PP = 1.0

# Handoff constraint 6: over this, stop and ask rather than report and continue.
FOLD_WALL_CLOCK_BUDGET_S = 90 * 60

# Round 4's guards (handoff §2). If the in-sample arm does not reproduce these,
# nothing downstream is comparable with document 50 and the run stops.
ROUND4_GAMES = 1139
ROUND4_AFFECTED = 1033
ROUND4_BUCKET_MOVES = 137
ROUND4_MEDIAN_ABS_DELTA_DTW_PP = 1.62
ROUND4_MEAN_WIDTH_V13 = 0.0383
ROUND4_MEAN_WIDTH_VARIANT = 0.0511
GUARD_TOLERANCE_PP = 0.01
GUARD_TOLERANCE_WIDTH = 1e-4

TRACE_PATTERN = "trace_dropped_pick_wk{week}.nc"
SUMMARY_PATTERN = "dropped_pick_summary_wk{week}.json"

OUTPUT_NAME = "69_dropped_pick_weekout.json"


# --------------------------------------------------------------------------
# the masked frame
# --------------------------------------------------------------------------


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


def masked_frame(full, week: int, defence_levels: list[str], qb_levels: list[str]):
    """The full frame with one week of season removed, everything else held.

    `n_defence_seasons` and `n_qb_seasons` stay at the full frame's counts on
    purpose: a defence-season whose only worthy throws were in the held-out week
    still exists in the model, with no rows of its own, and shrinks to `u_d = 0`.
    That is the correct read for it — no evidence, no entity term, document 05
    §1's `w = 0` endpoint — and the count of such levels is reported per fold.
    """
    model = full.model.filter(pl.col("week") != week)
    held_out = full.model.height - model.height
    if not held_out:
        raise SystemExit(f"week {week} removed no rows — stop and ask.")

    matrix, names = _power.design_matrix(model, reference=full.model)
    if tuple(names) != tuple(full.feature_names):
        raise SystemExit("the fold's design columns are not the full frame's — stop and ask.")

    fold = replace(
        full,
        model=model,
        design_matrix=matrix,
        outcome=model["interception"].cast(pl.Float64).to_numpy(),
        defence_season_codes=codes_against(model, ["season", "defteam"], defence_levels),
        qb_season_codes=codes_against(model, ["season", "passer_player_id"], qb_levels),
    )
    empty_defence = len(defence_levels) - len(set(fold.defence_season_codes.tolist()))
    empty_qb = len(qb_levels) - len(set(fold.qb_season_codes.tolist()))
    return fold, {
        "week": week,
        "rows_fitted": int(model.height),
        "rows_held_out": int(held_out),
        "defence_seasons_with_no_rows": int(empty_defence),
        "qb_seasons_with_no_rows": int(empty_qb),
    }


# --------------------------------------------------------------------------
# the eighteen fits
# --------------------------------------------------------------------------


def fit_the_folds(full, table) -> tuple[dict, list[dict]]:
    """One fit per week, each printing Gate C-1, each saved beside the default run."""
    defence_levels = _fit._labels(full.model, ["season", "defteam"])
    qb_levels = _fit._labels(full.model, ["season", "passer_player_id"])

    models, folds = {}, []
    started = time.time()
    for week in WEEKS:
        fold, mask_report = masked_frame(full, week, defence_levels, qb_levels)
        seed = FOLD_SEED_BASE + week
        print(
            f"\n{'-' * 72}\nFOLD week {week:2d} held out — {mask_report['rows_held_out']} rows out, "
            f"{mask_report['rows_fitted']:,} fitted, seed {seed}\n{'-' * 72}"
        )
        fold_started = time.time()
        # The gate is enforced after the loop, not inside it: handoff constraint
        # 6 makes an unhealthy fold a stop-and-ask, and an ask is only useful if
        # it says whether the problem is one fold or eighteen. The bars are
        # `62.sampler_health`'s, unchanged, and `gate_c1_over_the_folds` below
        # refuses to compute G-1 unless every fold clears them.
        idata, health = _fit.fit(fold, seed, label=f"week {week} held out", stop_on_c1=False)
        idata, _, _ = _fit.name_the_levels(
            idata, fold, defence_levels=defence_levels, qb_levels=qb_levels
        )

        trace_path = paths.RESEARCH_OUTPUT_DIR / TRACE_PATTERN.format(week=week)
        summary_path = paths.RESEARCH_OUTPUT_DIR / SUMMARY_PATTERN.format(week=week)
        idata.to_netcdf(trace_path)
        summary = _fit.build_summary(
            idata,
            fold,
            table,
            defence_levels=defence_levels,
            qb_levels=qb_levels,
            seed=seed,
            scale_frame=full.model,
            extra={
                "fitted_by": "research/69_dropped_pick_weekout.py",
                "gate": "document 52 §5 G-1, week-out fold",
                "week_held_out": week,
                "mask": mask_report,
                "gate_c1_sampler": health,
            },
        )
        summary_path.write_text(json.dumps(summary, indent=2, default=str))

        models[week] = DroppedPickModel.from_posterior(trace_path, summary_path)
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
            f"  fold week {week:2d}: sigma_d {summary['sigma_d_mean']:.4f}, "
            f"alpha {summary['alpha_mean']:+.4f}, {elapsed:.0f} s"
        )

        spent = time.time() - started
        if spent > FOLD_WALL_CLOCK_BUDGET_S:
            raise SystemExit(
                f"the folds have spent {spent / 60:.1f} minutes, over handoff "
                f"constraint 6's 90-minute budget, with {len(WEEKS) - week} to go. "
                "Stop and ask — reporting and continuing is the wrong call here."
            )

    print(f"\n  eighteen folds in {(time.time() - started) / 60:.1f} minutes")
    return models, folds


def gate_c1_over_the_folds(folds: list[dict]) -> dict:
    """Gate C-1 on all eighteen folds at once — the precondition G-1 rests on.

    Document 52 §5's G-1 is a statement about eighteen fits, so it may only be
    read off eighteen fits that sampled. A fold that missed C-1 is not a fold
    whose `u_d` can be compared with anything, and handoff constraint 6 says what
    to do about it: stop and ask, before the statistic exists.
    """
    print(f"\n{'=' * 72}\nGATE C-1 OVER THE EIGHTEEN FOLDS\n{'=' * 72}")
    print(
        f"  {'week':>4s} {'rows':>6s} {'div':>4s} {'max r_hat':>10s} {'on':>12s} "
        f"{'ess_bulk':>9s} {'ess_tail':>9s} {'sigma_d':>8s}  C-1"
    )
    failures = []
    for fold in folds:
        if not fold["c1_pass"]:
            failures.append(fold["week"])
        print(
            f"  {fold['week']:4d} {fold['rows_fitted']:6,d} {fold['divergences']:4d} "
            f"{fold['max_r_hat']:10.4f} {str(fold['max_r_hat_parameter']):>12s} "
            f"{fold['min_ess_bulk']:9.0f} {fold['min_ess_tail']:9.0f} "
            f"{fold['sigma_d_mean']:8.4f}  {'PASS' if fold['c1_pass'] else 'FAIL'}"
        )
    report = {
        "bars": "0 divergences, max r_hat < 1.01, min ess_bulk > 400, min ess_tail > 400",
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
        + (f", weeks {failures}" if failures else "")
    )
    return report


# --------------------------------------------------------------------------
# G-1
# --------------------------------------------------------------------------


def gate_g1(
    in_sample: pl.DataFrame,
    week_out: pl.DataFrame,
    *,
    postseason_games: int,
) -> dict:
    """Document 52 §5's G-1, on the games weeks 1-18 covers."""
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

    # The two arms build their events from the same charted throws, so an event
    # count that disagreed would mean the arms are not the same population and
    # the statistic below would be measuring the wrong thing.
    if (joined["events_in"] != joined["events_out"]).any():
        raise SystemExit("the two arms disagree on the event count — stop and ask.")

    rows = joined.to_dicts()
    bucket_in = [_audit.bucket(row["dtw_in"], row["actual_margin"]) for row in rows]
    bucket_out = [_audit.bucket(row["dtw_out"], row["actual_margin"]) for row in rows]
    agree = [a == b for a, b in zip(bucket_in, bucket_out, strict=True)]

    transitions: dict[str, int] = {}
    for a, b in zip(bucket_in, bucket_out, strict=True):
        if a != b:
            transitions[f"{a} -> {b}"] = transitions.get(f"{a} -> {b}", 0) + 1

    affected = joined["events_in"].to_numpy() > 0
    d_dtw = (joined["dtw_out"] - joined["dtw_in"]).abs().to_numpy() * 100
    d_margin = (joined["margin_out"] - joined["margin_in"]).abs().to_numpy()

    agreement = float(np.mean(agree))
    median_pp = float(np.median(d_dtw[affected]))
    passes = agreement >= G1_MIN_AGREEMENT and median_pp < G1_MAX_MEDIAN_ABS_DELTA_DTW_PP

    report = {
        "population": "2022-2025 games in weeks 1-18, the weeks the folds cover",
        "games_compared": int(joined.height),
        "postseason_games_not_covered": int(postseason_games),
        "affected_games": int(affected.sum()),
        "bucket_agreement": agreement,
        "n_bucket_disagreements": int(len(agree) - sum(agree)),
        "transitions_in_sample_to_week_out": transitions,
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
        f"  {report['games_compared']:,} games in weeks 1-18 "
        f"({report['postseason_games_not_covered']} postseason games have no "
        f"week-out read and are excluded)"
    )
    for name, count in sorted(transitions.items(), key=lambda item: -item[1]):
        print(f"    {name:45s} {count:5d}")
    print(
        f"G-1: bucket agreement {agreement:.3f} "
        f"({sum(agree):,}/{len(agree):,}); median |dDTW| in-sample vs week-out "
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
            "eighteen week-out traces, and the cost is noted."
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


def round4_guards(in_sample_full: dict) -> dict:
    """Handoff §2's numbers, reproduced before anything is read off the new arm."""
    checks = {
        "games": (in_sample_full["games"], ROUND4_GAMES, 0),
        "affected_games": (in_sample_full["affected_games"], ROUND4_AFFECTED, 0),
        "bucket_moves": (in_sample_full["n_bucket_moved"], ROUND4_BUCKET_MOVES, 0),
        "median_abs_delta_dtw_pp": (
            in_sample_full["median_abs_delta_dtw_pp_affected"],
            ROUND4_MEDIAN_ABS_DELTA_DTW_PP,
            GUARD_TOLERANCE_PP,
        ),
        "mean_width_v13": (
            in_sample_full["interval_width_affected"]["mean_v13"],
            ROUND4_MEAN_WIDTH_V13,
            GUARD_TOLERANCE_WIDTH,
        ),
        "mean_width_variant": (
            in_sample_full["interval_width_affected"]["mean_variant"],
            ROUND4_MEAN_WIDTH_VARIANT,
            GUARD_TOLERANCE_WIDTH,
        ),
    }
    print(f"\n{'=' * 72}\nROUND 4's GUARDS — the in-sample arm, reproduced\n{'=' * 72}")
    report, failures = {}, []
    for name, (got, want, tolerance) in checks.items():
        ok = abs(float(got) - float(want)) <= tolerance
        report[name] = {"round4": want, "reproduced": got, "tolerance": tolerance, "ok": bool(ok)}
        print(f"  {name:26s} round 4 {want:>8}   now {got:>8}   {'ok' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(name)
    report["pass"] = not failures
    if failures:
        raise SystemExit(
            f"handoff §2's guards do not reproduce: {failures}. Handoff constraint 6: stop and ask."
        )
    return report


# --------------------------------------------------------------------------


def main() -> None:
    print("=== Round 5 Part B — gate G-1, the week-out self-fulfilment check ===")

    full = _power.build_worthy_frame()
    if full.model.height != _fit.EXPECTED_MODEL_ROWS:
        raise SystemExit(f"the fit frame is {full.model.height:,} rows — stop and ask.")
    table, _ = _fit.swing_table_check(_fit.worthy_with_epa(full))

    ctx = _audit.load_context()

    models, folds = fit_the_folds(full, table)
    c1 = gate_c1_over_the_folds(folds)
    if not c1["pass"]:
        out = paths.RESEARCH_OUTPUT_DIR / OUTPUT_NAME
        out.write_text(
            json.dumps(
                {
                    "gate": "document 52 §5 G-1 (amendment A-3 clause 5)",
                    "status": "BLOCKED — the folds' Gate C-1 precondition failed",
                    "weeks": list(WEEKS),
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
            f"{c1['n_failed']} of {len(folds)} week-out folds miss Gate C-1 "
            f"(weeks {c1['folds_failed']}). Handoff constraint 6: stop and ask. "
            "G-1's statistic is NOT computed — a held-out read is only as good as "
            "the fits it comes from. The eighteen traces and summaries are on "
            "disk, so a ruling or a longer-chain amendment costs no refit of the "
            "folds that passed."
        )

    # ---- the three arms ---------------------------------------------------
    v13_table, _, v1 = _audit.v13_pass(ctx)
    in_sample_model = DroppedPickModel.from_posterior(
        paths.RESEARCH_OUTPUT_DIR / _audit.TRACE_NAME,
        paths.RESEARCH_OUTPUT_DIR / _audit.SUMMARY_NAME,
    )
    in_sample_table, in_sample_ledger = _audit.variant_pass(
        ctx, in_sample_model, label="in-sample variant"
    )
    week_out_table, week_out_ledger = _audit.variant_pass(
        ctx, None, models_by_week=models, label="week-out variant"
    )

    charted = v13_table.filter(pl.col("game_id").is_in(in_sample_table["game_id"].to_list()))
    covered = week_out_table["game_id"].to_list()
    postseason = in_sample_table.height - len(covered)

    # Round 4's guards on the full 1,139-game in-sample arm, before G-1 is read.
    in_sample_all = survival("in-sample variant (all 1,139 games)", charted, in_sample_table)
    guards = round4_guards(in_sample_all)

    v13_covered = charted.filter(pl.col("game_id").is_in(covered))
    in_sample_covered = in_sample_table.filter(pl.col("game_id").is_in(covered))

    g1 = gate_g1(in_sample_covered, week_out_table, postseason_games=postseason)

    print(
        f"\n{'=' * 72}\nDOES THE 12% SURVIVE? — weeks 1-18, the two arms side by side\n{'=' * 72}"
    )
    in_sample_weeks = survival("in-sample variant (weeks 1-18)", v13_covered, in_sample_covered)
    week_out_weeks = survival("week-out variant (weeks 1-18)", v13_covered, week_out_table)

    print(f"\n{'=' * 72}\nTHE WEEK-OUT VARIANT's OWN AUDIT\n{'=' * 72}")
    week_out_audit = _audit.audit(v13_covered, week_out_table, week_out_ledger)

    identity = _audit.round_trip_identity(week_out_table, ctx.slope)
    print(
        f"\n  V-2 on every week-out game: max |deserved − (actual − luck × slope)| = {identity:.2e}"
    )
    if identity > 1e-9:
        raise SystemExit("the week-out variant ledger does not sum. Stop and report.")

    results = {
        "gate": "document 52 §5 G-1 (amendment A-3 clause 5)",
        "weeks": list(WEEKS),
        "fold_seed_base": FOLD_SEED_BASE,
        "settings": {
            "random_seed": _audit.RANDOM_SEED,
            "posterior_draws": _audit.POSTERIOR_DRAWS,
            "coin_draws": _audit.COIN_DRAWS,
            "points_per_epa": ctx.slope,
            "standardisation": "the full frame's, via 61.design_matrix's reference argument",
            "swing_table": "the in-sample table, unchanged across folds",
        },
        "gate_v1_default_off": v1,
        "round4_guards": guards,
        "gate_c1_over_the_folds": c1,
        "folds": folds,
        "gate_g1": g1,
        "survival": {
            "in_sample_all_games": in_sample_all,
            "in_sample_weeks_1_18": in_sample_weeks,
            "week_out_weeks_1_18": week_out_weeks,
        },
        "week_out_audit": week_out_audit,
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
        f"\nG-1: {'PASS' if g1['pass'] else 'FAIL'}. "
        "Next: research/70_dropped_pick_sensitivity.py for G-2 and G-3."
    )


if __name__ == "__main__":
    main()
