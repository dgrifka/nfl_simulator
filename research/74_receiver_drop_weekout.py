"""Part D of round 7, second half — gate G-4c, the receiver model's week-out check.

Document 56 §3's G-4c is document 52 §5's G-1 applied to the other direction of
the class: refit the model **nineteen times**, holding out one week of season at
a time and the postseason together (document 54 F-2), read every game's charged
entity effect from the fit that never saw its week, and re-run the audit with
those draws. It answers the same question G-1 answered for the dropped pick —
*is the entity effect a game is priced at materially the game's own?* — with the
same bars, which are not re-tuned here for a component that might need them to
be:

    agreement >= 0.90 of verdict buckets, and median |ΔDTW| < 1.0 pp between arms.

Almost everything is `research/69`'s, imported rather than copied: the fold list
and its partition guard, the seed rule, the level-coding-against-the-full-frame
rule that stops `r_s[k]` naming a different team-season in each fit, and the
budget tripwire. What differs is the frame, the model and the grain — this one
charges the team-season, per document 56 §1's clause-1 rule.

    uv run python research/74_receiver_drop_weekout.py
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
_audit = import_module("68_dropped_pick_variant_audit")
_folds = import_module("69_dropped_pick_weekout")
_receiver = import_module("71_receiver_drop_power")
_study = import_module("72_receiver_drop_confounds")
_gates = import_module("73_receiver_drop_variant_audit")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.receiver_drops import ReceiverDropModel  # noqa: E402

# Document 54 F-2's fold list and seed rule, imported so the two components can
# never drift onto different folds.
FOLDS = _folds.FOLDS
WEEKS_BY_FOLD = _folds.WEEKS_BY_FOLD
POSTSEASON_FOLD = _folds.POSTSEASON_FOLD
POSTSEASON_WEEKS = _folds.POSTSEASON_WEEKS
FOLD_SEED_BASE = _folds.FOLD_SEED_BASE
POSTSEASON_SEED_OFFSET = _folds.POSTSEASON_SEED_OFFSET

# Document 52 §5's G-1 bars, which G-4c inherits unchanged.
G4C_MIN_AGREEMENT = _folds.G1_MIN_AGREEMENT
G4C_MAX_MEDIAN_ABS_DELTA_DTW_PP = _folds.G1_MAX_MEDIAN_ABS_DELTA_DTW_PP

# Round 5's handoff constraint 6, carried forward. The receiver frame is
# eighteen times the dropped-pick frame's rows, so this budget is set from the
# measured cost of `research/72`'s three fits rather than from round 6's, and it
# is a tripwire rather than a schedule: over it, stop and ask.
FOLD_WALL_CLOCK_BUDGET_S = 6 * 60 * 60

TRACE_PATTERN = "trace_receiver_drop_wk{fold}.nc"
SUMMARY_PATTERN = "receiver_drop_summary_wk{fold}.json"
OUTPUT_NAME = "74_receiver_drop_weekout.json"

N_LARGEST_DISAGREEMENTS = 10


def fold_label(fold) -> str:
    return _folds.fold_label(fold)


def seed_of(fold) -> int:
    return _folds.seed_of(fold)


def masked_frame(full, fold, entity_levels: list[str], defence_levels: list[str]):
    """The full frame with one fold's weeks removed, everything else held.

    The level counts stay at the full frame's on purpose: a team-season whose
    only catchable targets were in the held-out weeks still exists in the model,
    with no rows of its own, and shrinks to zero — document 05 §1's ``w = 0``
    endpoint, and the correct read for it. In practice no team-season can vanish
    at this grain, which is itself a difference from the dropped-pick folds and
    is reported per fold.

    The design matrix is built against the **full** frame's mean and SD, because
    a fold masks rows and not the covariate scale (round 3's fourth surprise, in
    the place `research/69` first met it).
    """
    weeks = list(WEEKS_BY_FOLD[fold])
    model = full.model.filter(~pl.col("week").is_in(weeks))
    held_out = full.model.height - model.height
    if not held_out:
        raise SystemExit(f"fold {fold_label(fold)} removed no rows — stop and ask.")

    matrix, names = _receiver.design_matrix(model, reference=full.model)
    if tuple(names) != tuple(full.feature_names):
        raise SystemExit("the fold's design columns are not the full frame's — stop and ask.")

    masked = replace(
        full,
        model=model,
        design_matrix=matrix,
        outcome=model["is_drop"].cast(pl.Float64).to_numpy(),
        # Recoded too, though this arm never reads it: a stale array of the full
        # frame's length sitting on a masked frame is a trap for the next reader.
        receiver_season_codes=_power._codes(model, ["season", "receiver_player_id"])[0],
        team_season_codes=_folds.codes_against(model, ["season", "posteam"], entity_levels),
        defence_season_codes=_folds.codes_against(model, ["season", "defteam"], defence_levels),
    )
    empty_entity = len(entity_levels) - len(set(masked.team_season_codes.tolist()))
    empty_defence = len(defence_levels) - len(set(masked.defence_season_codes.tolist()))
    return masked, {
        "fold": fold_label(fold),
        "weeks_held_out": list(weeks),
        "rows_fitted": int(model.height),
        "rows_held_out": int(held_out),
        "entity_seasons_with_no_rows": int(empty_entity),
        "defence_seasons_with_no_rows": int(empty_defence),
    }


def fit_the_folds(full, table: dict) -> tuple[dict, list[dict]]:
    """One fit per fold, each printing Gate C-1, each saved beside the default run."""
    entity_levels = _study._labels(full.model, ["season", "posteam"])
    defence_levels = _study._labels(full.model, ["season", "defteam"])
    standardisation = {
        column: {
            "mean": float(full.model[column].cast(pl.Float64).mean()),
            "sd": float(full.model[column].cast(pl.Float64).std()),
        }
        for column in _receiver.STANDARDISED
    }

    models, folds = {}, []
    started = time.time()
    for fold in FOLDS:
        masked, mask_report = masked_frame(full, fold, entity_levels, defence_levels)
        seed = seed_of(fold)
        print(
            f"\n{'-' * 72}\nFOLD {fold_label(fold)} held out (weeks "
            f"{mask_report['weeks_held_out']}) — {mask_report['rows_held_out']:,} rows out, "
            f"{mask_report['rows_fitted']:,} fitted, seed {seed}\n{'-' * 72}",
            flush=True,
        )
        fold_started = time.time()
        # The gate is enforced after the loop, not inside it: an unhealthy fold
        # is a stop-and-ask, and an ask is only useful if it says whether the
        # problem is one fold or nineteen.
        report, idata = _study.fit_conversion(
            f"fold {fold_label(fold)}",
            masked.design_matrix,
            masked.feature_names,
            masked.outcome,
            masked.team_season_codes,
            masked.n_team_seasons,
            masked.defence_season_codes,
            masked.n_defence_seasons,
            entity_name="team-season",
        )
        idata = _study.name_the_levels(idata, entity_levels, defence_levels)

        name = fold_label(fold)
        trace_path = paths.RESEARCH_OUTPUT_DIR / TRACE_PATTERN.format(fold=name)
        summary_path = paths.RESEARCH_OUTPUT_DIR / SUMMARY_PATTERN.format(fold=name)
        idata.to_netcdf(trace_path)
        summary_path.write_text(
            json.dumps(
                {
                    "document": "56 — the receiver-drop mirror",
                    "fitted_by": "research/74_receiver_drop_weekout.py",
                    "gate": "document 56 §3 G-4c, at document 54's fold spec",
                    "charged_grain": "team-season",
                    "fold": name,
                    "weeks_held_out": list(WEEKS_BY_FOLD[fold]),
                    "mask": mask_report,
                    "fit_seed": seed,
                    "draws": _study.DRAWS,
                    "tune": _study.TUNE,
                    "chains": _study.CHAINS,
                    "target_accept": _study.TARGET_ACCEPT,
                    "covariate_order": list(full.feature_names),
                    # The full frame's scale, not the fold's — see `masked_frame`.
                    "standardisation": standardisation,
                    "standardisation_from_rows": int(full.model.height),
                    "reference_levels": _study.REFERENCE_LEVELS,
                    "entity_season_levels": entity_levels,
                    "defence_season_levels": defence_levels,
                    "swing_table": table,
                    "gate_c1_sampler": report["sampler"],
                },
                indent=2,
                default=str,
            )
        )

        models[fold] = ReceiverDropModel.from_posterior(trace_path, summary_path)
        elapsed = time.time() - fold_started
        health = report["sampler"]
        folds.append(
            {
                **mask_report,
                "seed": seed,
                "sigma_entity_mean": report["variance_components"]["sigma_r"]["logit_mean"],
                "sigma_defence_mean": report["variance_components"]["sigma_d"]["logit_mean"],
                "alpha_mean": report["alpha_mean"],
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
            f"  fold {fold_label(fold):>4s}: sigma_entity "
            f"{report['variance_components']['sigma_r']['logit_mean']:.4f}, alpha "
            f"{report['alpha_mean']:+.4f}, {elapsed:.0f} s",
            flush=True,
        )

        spent = time.time() - started
        if spent > FOLD_WALL_CLOCK_BUDGET_S:
            raise SystemExit(
                f"the folds have spent {spent / 60:.1f} minutes, over the "
                f"{FOLD_WALL_CLOCK_BUDGET_S / 60:.0f}-minute budget, with "
                f"{len(FOLDS) - len(folds)} to go. Stop and ask — reporting and "
                "continuing is the wrong call here."
            )

    print(f"\n  {len(FOLDS)} folds in {(time.time() - started) / 60:.1f} minutes")
    return models, folds


def gate_c1_over_the_folds(folds: list[dict]) -> dict:
    """Gate C-1 on all nineteen folds at once — the precondition G-4c rests on.

    A fold that missed C-1 is not a fold whose entity effect can be compared with
    anything, and document 54 F-1 says what to do about it: stop and report,
    before the statistic exists. **No second spec is chosen mid-round.**
    """
    print(f"\n{'=' * 72}\nGATE C-1 OVER THE {len(folds)} FOLDS\n{'=' * 72}")
    print(
        f"  {'fold':>4s} {'rows':>7s} {'div':>4s} {'max r_hat':>10s} {'on':>14s} "
        f"{'ess_bulk':>9s} {'ess_tail':>9s} {'sigma_ent':>10s}  C-1"
    )
    failures = []
    for fold in folds:
        if not fold["c1_pass"]:
            failures.append(fold["fold"])
        print(
            f"  {fold['fold']:>4s} {fold['rows_fitted']:7,d} {fold['divergences']:4d} "
            f"{fold['max_r_hat']:10.4f} {str(fold['max_r_hat_parameter']):>14s} "
            f"{fold['min_ess_bulk']:9.0f} {fold['min_ess_tail']:9.0f} "
            f"{fold['sigma_entity_mean']:10.4f}  {'PASS' if fold['c1_pass'] else 'FAIL'}"
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
        "sigma_entity_range": [
            min(fold["sigma_entity_mean"] for fold in folds),
            max(fold["sigma_entity_mean"] for fold in folds),
        ],
        "total_wall_clock_s": sum(fold["wall_clock_s"] for fold in folds),
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


def gate_g4c(in_sample: pl.DataFrame, week_out: pl.DataFrame) -> dict:
    """Document 56 §3's G-4c, on every 2022-2025 game the folds cover."""
    joined = (
        in_sample.select(
            "game_id",
            "actual_margin",
            pl.col("dtw_home").alias("dtw_in"),
            pl.col("deserved_margin").alias("margin_in"),
            pl.col("n_receiver_drop_events").alias("events_in"),
        )
        .join(
            week_out.select(
                "game_id",
                pl.col("dtw_home").alias("dtw_out"),
                pl.col("deserved_margin").alias("margin_out"),
                pl.col("n_receiver_drop_events").alias("events_out"),
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
    if (joined["events_in"] != joined["events_out"]).any():
        raise SystemExit("the two arms disagree on the event count — stop and ask.")

    rows = joined.to_dicts()
    bucket_in = [_audit.bucket(row["dtw_in"], row["actual_margin"]) for row in rows]
    bucket_out = [_audit.bucket(row["dtw_out"], row["actual_margin"]) for row in rows]
    agree = [a == b for a, b in zip(bucket_in, bucket_out, strict=True)]

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
    passes = agreement >= G4C_MIN_AGREEMENT and median_pp < G4C_MAX_MEDIAN_ABS_DELTA_DTW_PP

    report = {
        "population": (
            "every 2022-2025 game, weeks 1-18 from their own week's fold and the "
            "postseason from the postseason fold (document 54 F-2)"
        ),
        "games_compared": int(joined.height),
        "affected_games": int(affected.sum()),
        "bucket_agreement": agreement,
        "n_bucket_agree": int(sum(agree)),
        "n_bucket_disagreements": int(len(agree) - sum(agree)),
        "transitions_in_sample_to_week_out": transitions,
        "disagreeing_games": disagreeing[:N_LARGEST_DISAGREEMENTS],
        "median_abs_delta_dtw_pp_affected": median_pp,
        "eti89_abs_delta_dtw_pp_affected": [
            float(v) for v in np.percentile(d_dtw[affected], [_audit.ETI_LOW, _audit.ETI_HIGH])
        ],
        "max_abs_delta_dtw_pp_affected": float(d_dtw[affected].max()),
        "median_abs_delta_margin_affected": float(np.median(d_margin[affected])),
        "max_abs_delta_margin_affected": float(d_margin[affected].max()),
        "bars": {
            "min_agreement": G4C_MIN_AGREEMENT,
            "max_median_abs_delta_dtw_pp": G4C_MAX_MEDIAN_ABS_DELTA_DTW_PP,
        },
        "pass": bool(passes),
    }

    print(f"\n{'=' * 72}\nGATE G-4c — in-sample against week-out\n{'=' * 72}")
    print(
        f"  {report['games_compared']:,} games, every one of them read from a fit "
        f"that never saw its week"
    )
    for name, count in sorted(transitions.items(), key=lambda item: -item[1]):
        print(f"    {name:45s} {count:5d}")
    for entry in report["disagreeing_games"]:
        print(
            f"    {entry['game_id']:16s} actual {entry['actual_margin']:+3.0f}  DTW% "
            f"{entry['dtw_in'] * 100:5.1f} -> {entry['dtw_out'] * 100:5.1f} "
            f"({entry['abs_delta_dtw_pp']:4.2f} pp)  {entry['bucket_in_sample']} -> "
            f"{entry['bucket_week_out']}"
        )
    print(
        f"G-4c: bucket agreement {agreement:.3f} ({sum(agree):,}/{len(agree):,}); "
        f"median |dDTW| between arms {median_pp:.2f} pp -> "
        f"{'PASS' if passes else 'FAIL'}"
    )
    print(
        f"  bars: agreement >= {G4C_MIN_AGREEMENT:.2f} "
        f"({'met' if agreement >= G4C_MIN_AGREEMENT else 'NOT met'}); "
        f"median |dDTW| < {G4C_MAX_MEDIAN_ABS_DELTA_DTW_PP:.1f} pp "
        f"({'met' if median_pp < G4C_MAX_MEDIAN_ABS_DELTA_DTW_PP else 'NOT met'})"
    )
    if not passes:
        print(
            "  G-4c FAILS. Document 52 §5's consequence applies to this direction too: "
            "production must use the week-out traces, and the cost is noted."
        )
    return report


def main() -> None:
    print("=== Round 7 Part D — gate G-4c, the receiver model's week-out check ===")

    full = _receiver.build_catchable_frame()
    fold_list = _folds.check_the_fold_list(full.model)
    summary = json.loads((paths.RESEARCH_OUTPUT_DIR / _gates.SUMMARY_NAME).read_text())
    table = summary["swing_table"]

    got = (summary.get("draws"), summary.get("tune"), summary.get("target_accept"))
    want = (_study.DRAWS, _study.TUNE, _study.TARGET_ACCEPT)
    if got != want:
        raise SystemExit(
            f"the default trace was fitted at {got} against F-1's {want}. G-4c compares "
            "two arms and they must share a spec — re-run research/72 first."
        )

    ctx = _audit.load_context()

    # ---- the in-sample arm, before nineteen fits are spent -------------------
    v13_table, _, v1 = _audit.v13_pass(ctx)
    in_sample_model = ReceiverDropModel.from_posterior(
        paths.RESEARCH_OUTPUT_DIR / _gates.TRACE_NAME,
        paths.RESEARCH_OUTPUT_DIR / _gates.SUMMARY_NAME,
    )
    in_sample_table, _ = _audit.variant_pass(
        ctx, None, receiver_drop_model=in_sample_model, label="in-sample +rd"
    )
    charted = v13_table.filter(pl.col("game_id").is_in(in_sample_table["game_id"].to_list()))

    # ---- the nineteen folds -------------------------------------------------
    models, folds = fit_the_folds(full, table)
    c1 = gate_c1_over_the_folds(folds)
    if not c1["pass"]:
        out = paths.RESEARCH_OUTPUT_DIR / OUTPUT_NAME
        out.write_text(
            json.dumps(
                {
                    "gate": "document 56 §3 G-4c (amendment A-3 clause 5, receiver side)",
                    "spec": c1["spec"],
                    "status": "BLOCKED — the folds' Gate C-1 precondition failed",
                    "gate_c1_over_the_folds": c1,
                    "folds": folds,
                    "g4c_computed": False,
                },
                indent=2,
                default=float,
            )
        )
        print(f"\nwrote {out}")
        raise SystemExit(
            f"{c1['n_failed']} of {len(folds)} folds miss Gate C-1 at document 54 "
            f"F-1's spec (folds {c1['folds_failed']}). Stop and report — no second "
            "spec is chosen mid-round. G-4c's statistic is NOT computed."
        )

    # ---- the week-out arm and G-4c ------------------------------------------
    week_out_table = _weekout_pass(ctx, models)

    g4c = gate_g4c(in_sample_table, week_out_table)

    identity = _audit.round_trip_identity(week_out_table, ctx.slope)
    print(
        f"\n  V-2 on every week-out game: max |deserved − (actual − luck × slope)| = {identity:.2e}"
    )
    if identity > 1e-9:
        raise SystemExit("the week-out variant ledger does not sum. Stop and report.")

    in_sample_moves, _, _ = _gates.bucket_moves(charted, in_sample_table)
    week_out_moves, _, _ = _gates.bucket_moves(charted, week_out_table)
    survival = {
        "in_sample_bucket_moves": len(in_sample_moves),
        "week_out_bucket_moves": len(week_out_moves),
        "moves_in_both": len(in_sample_moves & week_out_moves),
        "in_sample_only": len(in_sample_moves - week_out_moves),
        "week_out_only": len(week_out_moves - in_sample_moves),
    }
    print(
        f"\n  does the movement survive? in-sample {survival['in_sample_bucket_moves']} "
        f"bucket moves, week-out {survival['week_out_bucket_moves']}; element-wise "
        f"{survival['moves_in_both']} in both, {survival['in_sample_only']} in-sample "
        f"only, {survival['week_out_only']} week-out only"
    )

    results = {
        "gate": "document 56 §3 G-4c (amendment A-3 clause 5, receiver side)",
        "charged_grain": "team-season",
        "folds_list": [fold_label(fold) for fold in FOLDS],
        "weeks_by_fold": {fold_label(k): list(v) for k, v in WEEKS_BY_FOLD.items()},
        "fold_seed_base": FOLD_SEED_BASE,
        "postseason_seed_offset": POSTSEASON_SEED_OFFSET,
        "settings": {
            "sampler_spec": {"draws": got[0], "tune": got[1], "target_accept": got[2]},
            "random_seed": _audit.RANDOM_SEED,
            "posterior_draws": _audit.POSTERIOR_DRAWS,
            "coin_draws": _audit.COIN_DRAWS,
            "points_per_epa": ctx.slope,
            "standardisation": "the full frame's, via 71.design_matrix's reference argument",
            "swing_table": "the in-sample table, unchanged across folds",
        },
        "fold_list_partition": fold_list,
        "gate_v1_default_off": v1,
        "gate_c1_over_the_folds": c1,
        "folds": folds,
        "gate_g4c": g4c,
        "g4c_computed": True,
        "survival": survival,
        "gate_v2_round_trip_max_residual": identity,
    }
    out = paths.RESEARCH_OUTPUT_DIR / OUTPUT_NAME
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {out}")

    print(f"\n{'=' * 72}\nV-1, RE-PRINTED AT THE END OF THE RUN\n{'=' * 72}")
    print(
        f"V-1 replay: {v1['games_matched']:,} games, max |Δ deserved margin| "
        f"{v1['max_abs_gaps']['deserved_margin']:.2e}  -> "
        f"{'PASS' if v1['pass'] else 'FAIL'}"
    )
    if not v1["pass"]:
        raise SystemExit("v1.3 moved. Stop and report.")
    print(
        f"\nG-4c: {'PASS' if g4c['pass'] else 'FAIL'} — bucket agreement "
        f"{g4c['bucket_agreement']:.3f} "
        f"({g4c['n_bucket_agree']:,}/{g4c['games_compared']:,}); median |dDTW| between "
        f"arms {g4c['median_abs_delta_dtw_pp_affected']:.2f} pp"
    )


def _weekout_pass(ctx, models: dict) -> pl.DataFrame:
    """Each game scored by the fit that never saw its week.

    `68.variant_pass`'s ``models_by_week`` branch drives the dropped-pick model,
    so the receiver arm needs its own loop rather than a fourth keyword bolted
    onto a function three rounds depend on.
    """
    tables = []
    for fold, fold_model in models.items():
        weeks = list(WEEKS_BY_FOLD[fold])
        rows = ctx.pbp.filter(
            (pl.col("season").is_in(_audit.FTN_SEASONS)) & (pl.col("week").is_in(weeks))
        )
        if not rows.height:
            continue
        table, _ = _audit.simulate_all(
            rows,
            ctx.margins,
            ctx.baselines,
            ctx.fg_model,
            ctx.slope,
            receiver_drop_model=fold_model,
            ftn_by_game=ctx.ftn_by_game,
        )
        tables.append(table)
    table = pl.concat(tables)
    if table["game_id"].n_unique() != table.height:
        raise SystemExit(
            "a game was scored by more than one fold — the folds overlap. Stop and ask."
        )
    print(
        f"\n  week-out +rd: {table.height:,} games over "
        f"{_audit.FTN_SEASONS[0]}-{_audit.FTN_SEASONS[-1]}, each scored by the fit that "
        f"excluded its week"
    )
    return table


if __name__ == "__main__":
    main()
