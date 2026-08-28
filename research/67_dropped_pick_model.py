"""Part A of round 4 — the fit the dropped-pick variant reads.

The component pre-registered in
`docs/research/49-dropped-pick-variant-prereg.md`. This script fits it once and
writes the two artifacts the read side loads:

    research/outputs/trace_dropped_pick.nc
    research/outputs/dropped_pick_summary.json

The model is document 43 §5's arm 2, at amendment **F-1**'s sampler spec
(4 x 4,000 draws after 4,000 tuning, `target_accept` 0.95, nutpie; document 54
raised it from A-2's 2,000/2,000/0.9 so the week-out folds converge) — imported
from `research/61_dropped_pick_power.build_conversion_model`, not copied — and
**without** document 47 §2's game effect `w_g`. Document 49 §2 states why: the
8.5 pp of unexplained within-game correlation has no team owner, so excluding it
treats that variance as luck, and that is disclosed rather than hidden.

Two things this file is careful about.

* **The QB-season term is fitted and not read.** `v_q` stays in the model so the
  defence's effect is estimated free of it, but document 49 §2 keeps it out of
  `p_i`: a quarterback's own droppability belongs to the offence, and paying it
  here would credit a passer for throwing catchable interceptions.
* **Everything the read side needs is stored, never recomputed.** Round 3's
  surprise 4 was a standardisation recomputed at read time on a different frame.
  The summary JSON therefore carries the covariate order, the mean and SD each
  standardised covariate was centred on, the reference levels the dummies were
  coded against, and the swing table — all of them properties of *this* fit's
  2,969-row sample.

Gates printed here: **V-6** (Gate C-1's sampler bars, every parameter) and
**V-8** (the 89% catch-probability interval on a median throw for the five best
and five worst defence-seasons, each of which must lie inside [0.30, 0.70]).

    uv run python research/67_dropped_pick_model.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl
import pymc as pm

sys.path.insert(0, str(Path(__file__).parent))

_power = import_module("61_dropped_pick_power")
_confounds = import_module("62_dropped_pick_confounds")
_diagnostic = import_module("65_dropped_pick_diagnostic")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.dropped_picks import build_swing_table  # noqa: E402
from nfl_simulator.ingest import FTN_SEASONS, load_pbp  # noqa: E402

# Document 54's amendment F-1, verbatim. Round 5 fitted at amendment A-2's
# 4 x 2,000 after 2,000 tuning, `target_accept` 0.9, and seven of the eighteen
# week-out folds missed Gate C-1 on the variance components with zero
# divergences in all eighteen — the chain-length failure A-2 itself was written
# for. F-1 doubles the chains and raises `target_accept` for **every** fit,
# folds and the default alike, so G-1's two arms are compared at one spec.
DRAWS = 4000
TUNE = 4000
CHAINS = 4
TARGET_ACCEPT = 0.95
RANDOM_SEED = _power.RANDOM_SEED  # 20260827, the study's fit seed

# The spec rounds 1-5 fitted at, kept so the round-4 reproduction tripwire below
# can say whether it still applies. It does not under F-1, and saying so is
# better than deleting the check and hoping a reader remembers why.
A2_SPEC = (2000, 2000, 0.9)

TRACE_NAME = "trace_dropped_pick.nc"
SUMMARY_NAME = "dropped_pick_summary.json"

# Stop-and-ask guards (handoff constraint 8 and document 49 §4).
EXPECTED_MODEL_ROWS = 2969
EXPECTED_WORTHY = 2997
EXPECTED_DEFENCE_SEASONS = 128

# Document 49 §6, V-8's sanity bound. A defence-season whose median-throw catch
# probability leaves this band is not a defence, it is a fitting artifact.
V8_LOW, V8_HIGH = 0.30, 0.70
V8_N_EACH = 5

# Document 05's interval convention, carried through every document since 03.
ETI_LOW, ETI_HIGH = 5.5, 94.5

# Document 47 §3's pooled swing, as round 3 measured it. A tripwire on the
# `src/` builder, not a target.
ROUND3_POOLED_SWING = -3.55
POOLED_SWING_TOLERANCE = 0.10

REFERENCE_LEVELS = {"pass_location": "middle", "down": 1.0}

# Round 4's posterior mean of `sigma_d` at this seed on this frame, to full
# precision from `dropped_pick_summary.json`. The refactor round 5 needed — `fit`
# taking a frame and a seed so the week-out folds can reuse it — must not move
# the default run by a floating-point hair, and this is the tripwire that says
# so. Round 4's document 50 quotes it to four places as 0.2544.
ROUND4_SIGMA_D_MEAN = 0.25437862651398274
ROUND4_REPRODUCTION_TOLERANCE = 1e-6


def _labels(frame: pl.DataFrame, keys: list[str]) -> list[str]:
    """The level labels in `_power._codes` order, so a code indexes this list."""
    return (
        frame.select(pl.concat_str(keys, separator="|").alias("label"))["label"]
        .unique(maintain_order=True)
        .to_list()
    )


def worthy_with_epa(frame) -> pl.DataFrame:
    """Every charted worthy throw 2022-2025 with `epa` joined on, as `65` does.

    `epa` is a post-branch quantity and document 43 §4 excludes it from `X` by
    rule, so it is joined here rather than added to the modelling column list —
    which would change the frame round 2 was fitted on.
    """
    epa = load_pbp(FTN_SEASONS, columns=["game_id", "play_id", "epa"])
    joined = frame.worthy.join(epa, on=["game_id", "play_id"], how="left")
    if joined.height != EXPECTED_WORTHY:
        raise SystemExit(
            f"the epa join changed the row count: {joined.height:,} against "
            f"{EXPECTED_WORTHY:,} — stop and ask."
        )
    return joined


def swing_table_check(worthy: pl.DataFrame) -> dict:
    """The `src/` builder against `research/65`'s, on the same rows.

    Document 49 §4 says the bin table is "recomputed in `src/` from the same
    data". This is the check that makes that a fact rather than an intention: if
    the production builder and round 3's research one disagree anywhere, the
    variant would price a throw at a swing document 48 never reported.
    """
    table = build_swing_table(worthy)
    research_table, research_pooled = _diagnostic.swing_table(_diagnostic.bin_labels(worthy))

    gaps, source_disagreements = [], []
    for row in research_table.iter_rows(named=True):
        key = f"{row['yard_bin']}|{row['down_bin']}"
        gaps.append(abs(table.cells[key] - row["swing"]))
        if table.counts[key]["source"] != row["source"]:
            source_disagreements.append(key)
    report = {
        "max_abs_cell_gap": float(max(gaps)),
        "source_disagreements": source_disagreements,
        "pooled_src": table.pooled,
        "pooled_research": float(research_pooled),
        "pooled_gap": abs(table.pooled - float(research_pooled)),
        "round3_pooled": ROUND3_POOLED_SWING,
        "all_cells_negative": bool(all(value < 0 for value in table.cells.values())),
        "all_cells_own": bool(all(entry["source"] == "cell" for entry in table.counts.values())),
    }
    report["pass"] = bool(
        report["max_abs_cell_gap"] <= 1e-12
        and not source_disagreements
        and report["pooled_gap"] <= 1e-12
        and report["all_cells_negative"]
        and abs(table.pooled - ROUND3_POOLED_SWING) <= POOLED_SWING_TOLERANCE
    )

    print("\n=== swing table — `src/` against round 3's research builder ===")
    print(f"  {'cell':16s} {'n_pick':>7s} {'n_esc':>7s} {'swing':>8s}  source   |src − 65|")
    for row in research_table.iter_rows(named=True):
        key = f"{row['yard_bin']}|{row['down_bin']}"
        counts = table.counts[key]
        print(
            f"  {key:16s} {counts['n_picked']:7d} {counts['n_escaped']:7d} "
            f"{table.cells[key]:+8.2f}  {counts['source']:7s}  "
            f"{abs(table.cells[key] - row['swing']):.2e}"
        )
    print(
        f"  pooled fallback {table.pooled:+.2f} EPA (round 3: {ROUND3_POOLED_SWING:+.2f})  "
        f"-> {'PASS' if report['pass'] else 'FAIL'}"
    )
    if not report["pass"]:
        raise SystemExit(
            "the production swing table is not round 3's. Document 49 §4's "
            "'recomputed in src/ from the same data' is violated; stop and report."
        )
    return table, report


def fit(
    frame,
    seed: int = RANDOM_SEED,
    *,
    draws: int = DRAWS,
    tune: int = TUNE,
    chains: int = CHAINS,
    label: str = "the full frame",
    stop_on_c1: bool = True,
) -> tuple[object, dict]:
    """Arm 2 at F-1's spec, without `w_g`, on whatever rows `frame` carries.

    Round 6 (document 52 §5's gate G-1, at document 54's fold spec) needs
    nineteen of these — the same model at the same spec, one week-of-season
    masked out of each and the postseason as a nineteenth — so the fit takes a
    frame and a seed rather than reading module state. Only the row mask changes:
    `draws`, `tune`, `chains` and `target_accept` keep F-1's values for every
    fold *and* for the default fit, and the caller passes a fold seed.

    Returns the trace and the sampler-health summary — Gate C-1's bars over every
    parameter, which is V-6 on the default run and the per-fold gate G-1 prints.

    ``stop_on_c1=False`` hands the gate to the caller and is not a way around it.
    A caller fitting eighteen folds needs all eighteen C-1 lines before it stops,
    and needs the traces on disk so a ruling costs no refit — the same reasoning
    round 4 recorded for writing this script's artifacts before enforcing V-8.
    The bars are unchanged and the caller must enforce them.
    """
    print(
        f"\n=== the fit — document 43 §5 arm 2, F-1's spec, no game effect ===\n"
        f"  {label}: {frame.model.height:,} throws, "
        f"{frame.design_matrix.shape[1]} covariates, "
        f"{frame.n_defence_seasons} defence-seasons, {frame.n_qb_seasons} QB-seasons\n"
        f"  {chains} x {draws} draws after {tune} tuning, target_accept "
        f"{TARGET_ACCEPT}, seed {seed}"
    )
    model = _power.build_conversion_model(
        frame.design_matrix,
        frame.outcome,
        frame.defence_season_codes,
        frame.n_defence_seasons,
        frame.qb_season_codes,
        frame.n_qb_seasons,
    )
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=seed,
            progressbar=False,
            nuts_sampler="nutpie",
            nuts={"target_accept": TARGET_ACCEPT},
        )

    print(f"\n=== Gate C-1's sampler bars, every parameter ({label}) ===")
    health = _confounds.sampler_health(idata, ["alpha", "beta", "sigma_d", "sigma_q", "z_d", "z_q"])
    print(f"  C-1: {'PASS' if health['pass'] else 'FAIL'}")
    if not health["pass"] and stop_on_c1:
        raise SystemExit(
            f"the sampler did not clear Gate C-1 on {label}. On the default run "
            "that is V-6; on a week-out fold it is round 5's handoff constraint "
            "6. Either way: stop and ask."
        )
    return idata, health


def name_the_levels(
    idata,
    frame,
    *,
    defence_levels: list[str] | None = None,
    qb_levels: list[str] | None = None,
) -> tuple[object, list[str], list[str]]:
    """Give `u_d` and `v_q` their level names, the way the FG trace names kickers.

    `61.build_conversion_model` dimensions them by integer position because Part
    A of round 2 never needed the labels. The read side does — it looks a
    defence-season up by name — so the coordinates are attached here, in the
    same `_codes` order the codes were built in.

    A week-out fold passes its level lists in: its own rows are a subset, so
    reading the labels off the masked frame would name level *k* after whichever
    defence-season happens to sit at position *k* in the subset. Round 5's folds
    are coded against the full frame's levels precisely so `u_d` keeps one
    meaning across all eighteen fits.
    """
    defence_levels = defence_levels or _labels(frame.model, ["season", "defteam"])
    qb_levels = qb_levels or _labels(frame.model, ["season", "passer_player_id"])
    if len(defence_levels) != frame.n_defence_seasons or len(qb_levels) != frame.n_qb_seasons:
        raise SystemExit("level labels do not line up with the fitted level counts.")

    posterior = (
        idata["posterior"]
        .to_dataset()
        .rename({"defence": "defence_season", "qb": "qb_season"})
        .assign_coords(defence_season=defence_levels, qb_season=qb_levels)
    )
    idata["posterior"] = posterior
    return idata, defence_levels, qb_levels


def median_throw(frame, *, consistent: bool) -> np.ndarray:
    """The design row a V-8 interval is quoted on — under either reading.

    **Document 49 §6 says "a median throw" and does not define it, and the two
    readings disagree.** Both are computed and reported; neither is chosen here.

    * ``consistent=False`` — the column-wise median of the fitted design matrix.
      Every column takes its own median independently, which is the literal
      reading of "the median of X" and the one a reader would write first.
      ``air_yards_z_squared`` then takes the median of the squared column, which
      is roughly 0.45 rather than 0: no actual throw has that pair of values, so
      the row describes no play.
    * ``consistent=True`` — the median of each *covariate*, with the derived
      column derived from it: ``air_yards_z_squared`` is the square of the
      median standardised ``air_yards``. This is a throw that could have
      happened, which is what "a median throw" names in English.
    """
    row = np.median(frame.design_matrix, axis=0)
    if not consistent:
        return row
    names = list(frame.feature_names)
    row = row.copy()
    row[names.index("air_yards_z_squared")] = row[names.index("air_yards_z")] ** 2
    return row


def v8_report(idata, frame, defence_levels: list[str]) -> dict:
    """V-8 — the five best and five worst defence-seasons, as catch probabilities.

    Reported under both readings of "a median throw" (see :func:`median_throw`),
    because the bound's verdict is not the same under the two and the choice was
    never pre-registered. The caller stops on a breach under *either*: picking
    the reading that passes would be the goalpost move documents 04 and 05 §7
    wrote the power-first law to prevent.
    """
    posterior = idata["posterior"]
    alpha = posterior["alpha"].values.ravel()
    beta = posterior["beta"].values.reshape(-1, frame.design_matrix.shape[1])
    u_d = posterior["u_d"].values.reshape(-1, len(defence_levels))
    order = np.argsort(u_d.mean(axis=0))

    print("\n=== V-8 — catch probability on a median throw, 89% interval ===")
    print(f"  bound: every interval must lie inside [{V8_LOW:.2f}, {V8_HIGH:.2f}]")

    readings, breaches = {}, {}
    for reading, consistent in (("column_wise_median", False), ("consistent_median", True)):
        row = median_throw(frame, consistent=consistent)
        base = alpha + beta @ row
        print(
            f"\n  reading: {reading}  "
            f"(league p(catch) at u_d = 0: {(1.0 / (1.0 + np.exp(-base))).mean():.3f})"
        )
        lines, breached = [], []
        for label, indices in (
            ("worst", order[:V8_N_EACH]),
            ("best", order[-V8_N_EACH:][::-1]),
        ):
            for index in indices:
                draws = 1.0 / (1.0 + np.exp(-(base + u_d[:, index])))
                low, high = (float(v) for v in np.percentile(draws, [ETI_LOW, ETI_HIGH]))
                inside = low >= V8_LOW and high <= V8_HIGH
                lines.append(
                    {
                        "rank": label,
                        "defence_season": defence_levels[index],
                        "u_d_mean": float(u_d[:, index].mean()),
                        "p_catch_mean": float(draws.mean()),
                        "eti89": [low, high],
                        "inside_bound": bool(inside),
                    }
                )
                if not inside:
                    breached.append(defence_levels[index])
                print(
                    f"    {label:5s}  {defence_levels[index]:12s}  "
                    f"u_d {u_d[:, index].mean():+.3f}  p(catch) {draws.mean():.3f}  "
                    f"89% [{low:.3f}, {high:.3f}]  {'ok' if inside else 'OUT OF BOUND'}"
                )
        readings[reading] = {
            "median_throw_design_row": [float(v) for v in row],
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
        "median_throw_undefined_in_document_49": True,
        "readings": readings,
        "pass": not any(breaches.values()),
    }
    print(f"\n  V-8: {'PASS' if report['pass'] else 'FAIL'}")
    return report


def build_summary(
    idata,
    frame,
    table,
    *,
    defence_levels: list[str],
    qb_levels: list[str],
    seed: int,
    scale_frame=None,
    extra: dict | None = None,
) -> dict:
    """The read side's constants, as `DroppedPickModel.from_posterior` needs them.

    Lifted out of :func:`main` in round 5 so a week-out fold writes the same
    object the default run does. Two arguments earn their keep:

    * ``scale_frame`` — the rows the standardisation constants describe. A fold's
      design matrix is built against the *full* frame's mean and SD (document 52
      §5's G-1 masks rows, not the covariate scale), so the constants stored with
      a fold must be the full frame's or the read side would centre a throw on a
      scale the fit never used. Round 3's fourth surprise, in a new place.
    * ``extra`` — the gate reports, which differ between the default run (V-6 and
      V-8) and a fold (C-1 alone).
    """
    posterior = idata["posterior"]
    scale_frame = frame.model if scale_frame is None else scale_frame
    standardisation = {
        column: {
            "mean": float(scale_frame[column].cast(pl.Float64).mean()),
            "sd": float(scale_frame[column].cast(pl.Float64).std()),
        }
        for column in _power.STANDARDISED
    }
    beta_means = posterior["beta"].values.mean(axis=(0, 1))
    summary = {
        "document": "49 — the dropped-pick variant ledger",
        "fitted_by": "research/67_dropped_pick_model.py",
        "model": (
            "document 43 §5 arm 2 (logit p = alpha + X beta + u_d + v_q), amendment "
            "document 54 F-1's sampler spec, no game effect w_g (document 49 §2)"
        ),
        "read_side_note": (
            "p_i excludes v_q by design (document 49 §2): the QB-season term is "
            "fitted so u_d is estimated free of it, and never read."
        ),
        "fit_seed": seed,
        "draws": DRAWS,
        "tune": TUNE,
        "chains": CHAINS,
        "target_accept": TARGET_ACCEPT,
        "n_posterior_draws": int(posterior["alpha"].values.size),
        "rows": int(frame.model.height),
        "worthy_rows": int(frame.worthy.height),
        "n_defence_seasons": int(frame.n_defence_seasons),
        "n_qb_seasons": int(frame.n_qb_seasons),
        "guards": frame.guards,
        "covariate_order": list(frame.feature_names),
        "standardisation": standardisation,
        "standardisation_from_rows": int(scale_frame.height),
        "reference_levels": REFERENCE_LEVELS,
        "standardised_covariates": list(_power.STANDARDISED),
        "defence_season_levels": defence_levels,
        "qb_season_levels": qb_levels,
        "swing_table": table.to_dict(),
        "beta_means": {
            name: float(value) for name, value in zip(frame.feature_names, beta_means, strict=True)
        },
        "alpha_mean": float(posterior["alpha"].values.mean()),
        "sigma_d_mean": float(posterior["sigma_d"].values.mean()),
        "sigma_q_mean": float(posterior["sigma_q"].values.mean()),
    }
    summary.update(extra or {})
    return summary


def main() -> None:
    paths.ensure_data_dirs()
    print("=== Round 4 Part A — the dropped-pick fit (document 49 §5) ===")

    frame = _power.build_worthy_frame()
    if frame.model.height != EXPECTED_MODEL_ROWS:
        raise SystemExit(
            f"the fit frame is {frame.model.height:,} rows against "
            f"{EXPECTED_MODEL_ROWS:,} — handoff constraint 8: stop and ask."
        )
    if frame.n_defence_seasons != EXPECTED_DEFENCE_SEASONS:
        raise SystemExit(
            f"{frame.n_defence_seasons} defence-seasons against "
            f"{EXPECTED_DEFENCE_SEASONS} — stop and ask."
        )

    table, table_report = swing_table_check(worthy_with_epa(frame))

    idata, health = fit(frame, RANDOM_SEED, label="the full 2,969-row frame")
    idata, defence_levels, qb_levels = name_the_levels(idata, frame)
    v8 = v8_report(idata, frame, defence_levels)

    # Round 5's refactor tripwire, and what document 54 does to it. The check was
    # written to prove that making `fit` take a frame and a seed moved no number
    # at A-2's spec. Amendment F-1 changes the spec on purpose, so the tripwire
    # is no longer a statement about the refactor — a different sampler spec is
    # *supposed* to move `sigma_d` by sampler noise. It is therefore enforced
    # only at A-2's spec and reported otherwise, and F-3's audit reproduction
    # (research/69) is the check that the re-fit arm is still round 4's arm.
    sigma_d = float(idata["posterior"]["sigma_d"].values.mean())
    gap = abs(sigma_d - ROUND4_SIGMA_D_MEAN)
    at_a2_spec = (DRAWS, TUNE, TARGET_ACCEPT) == A2_SPEC
    print("\n=== round-4 reproduction — the refactor changed no number ===")
    print(
        f"  sigma_d posterior mean {sigma_d:.12f} against round 4's "
        f"{ROUND4_SIGMA_D_MEAN:.12f}\n"
        f"  |gap| {gap:.2e}, tolerance {ROUND4_REPRODUCTION_TOLERANCE:.0e}  -> "
        + (
            f"{'PASS' if gap <= ROUND4_REPRODUCTION_TOLERANCE else 'FAIL'}"
            if at_a2_spec
            else "REPORTED ONLY"
        )
    )
    if not at_a2_spec:
        print(
            f"  the spec is document 54 F-1's ({CHAINS} x {DRAWS:,} after {TUNE:,} "
            f"tuning, target_accept {TARGET_ACCEPT}), not A-2's "
            f"({A2_SPEC[0]:,}/{A2_SPEC[1]:,}/{A2_SPEC[2]}), so this gap is sampler\n"
            "  noise between two specs and is not a gate. Document 54 F-3's audit "
            "reproduction, in research/69, is the check that applies."
        )
    elif gap > ROUND4_REPRODUCTION_TOLERANCE:
        raise SystemExit(
            "the default run no longer reproduces round 4's fit. Round 5's Part B "
            "refactor was supposed to change nothing here; stop and report."
        )

    trace_path = paths.RESEARCH_OUTPUT_DIR / TRACE_NAME
    idata.to_netcdf(trace_path)

    summary = build_summary(
        idata,
        frame,
        table,
        defence_levels=defence_levels,
        qb_levels=qb_levels,
        seed=RANDOM_SEED,
        extra={
            "swing_table_check": table_report,
            "gate_v6_sampler": health,
            "gate_v8_posterior_spread": v8,
        },
    )
    summary_path = paths.RESEARCH_OUTPUT_DIR / SUMMARY_NAME
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(f"\nwrote {trace_path}")
    print(f"wrote {summary_path}")
    print(
        f"\n  alpha {summary['alpha_mean']:+.4f}, sigma_d {summary['sigma_d_mean']:.4f}, "
        f"sigma_q {summary['sigma_q_mean']:.4f}, "
        f"{summary['n_posterior_draws']:,} posterior draws"
    )

    # The artifacts are written before the gate is reported on purpose. They are
    # gitignored and regenerable, and a stop-and-ask that costs a refit to
    # resolve is a stop-and-ask nobody re-reads: the summary carries the breach,
    # so a ruling needs no refit.
    #
    # **Ruling R-3 (document 52 §5), 2026-08-27.** Round 4 stopped here, because
    # V-8's breach was unruled. It is ruled now: the 2022 NYG breach of 1.1 pp on
    # one of ten lines is immaterial and the bound stands unamended. The gate's
    # text is not edited and its verdict is not re-tolerated — V-8 still reads
    # FAIL, here and in the summary JSON — so what changes is only that the
    # script no longer exits on a breach the owner has ruled on.
    if not v8["pass"]:
        breached = {
            reading: entry["breaches"]
            for reading, entry in v8["readings"].items()
            if entry["breaches"]
        }
        print(
            f"\n  V-8 FAIL stands on the record: {breached}. Ruling R-3 "
            "(document 52 §5) declares it immaterial — document 50 §2 carries the\n"
            "  reasoning and document 49 §10 the ruling. The bound is unamended; "
            "this is not a re-tolerancing."
        )
    print("Next: research/68_dropped_pick_variant_audit.py for V-1 and document 49 §7.")


if __name__ == "__main__":
    main()
