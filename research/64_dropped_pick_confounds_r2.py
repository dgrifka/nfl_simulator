"""Part B of round 2 — the fits, on the floorless frame, at A-2's sampler spec.

Round 2 of the study pre-registered in
`docs/research/43-dropped-pick-confounds-prereg.md` and amended by
`docs/research/45-dropped-pick-pooling-prereg.md`. Three amendments, and nothing
else moves:

    A-1  `MIN_QB_WORTHY` is removed. Arm 3's frame is arm 2's frame — every
         QB-season with at least one worthy throw is a level, and the hierarchy
         does the pooling the floor was doing by deletion.
    A-2  Arms 2 and 2b sample 4 chains x 2,000 draws after 2,000 tuning at
         `target_accept = 0.9`. Round 1's Gate C-1 failed on one nuisance
         parameter (`sigma_q`: r_hat 1.0105, ESS 387/345); longer chains are the
         remedy, and they are an amendment because they change document 43 §5's
         committed inference spec.
    A-3  A hindsight probe on the selection variable itself. If a charter marks
         a throw interception-worthy *because* it was intercepted, selecting on
         "worthy" conditions on a descendant of the outcome. Reported, never
         gated.

Arm 1 is **not** re-run: its two rate designs never saw the floor, its sampler is
a grid rather than NUTS, and its numbers stand in document 44 §3 (document 45 §3).

Thresholds come from `research/outputs/63_dropped_pick_power_r2.json`, which was
computed and committed before this file fitted anything real.

    uv run python research/64_dropped_pick_confounds_r2.py

Nothing in `src/nfl_simulator/` changes on any outcome.
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import arviz as az
import numpy as np
import polars as pl
import pymc as pm

sys.path.insert(0, str(Path(__file__).parent))

_power = import_module("61_dropped_pick_power")
_confounds = import_module("62_dropped_pick_confounds")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.ingest import FTN_SEASONS, load_ftn, load_pbp  # noqa: E402

RANDOM_SEED = _power.RANDOM_SEED
LOGIT_SLOPE = _power.LOGIT_SLOPE
CROSS_CHECK_TOLERANCE_PP = _confounds.CROSS_CHECK_TOLERANCE_PP
HINDSIGHT_COLUMNS = _confounds.HINDSIGHT_COLUMNS

# Document 45 §6 / amendment A-2. Round 1 ran 1,000 / 1,000 / 4 with the sampler
# default target_accept.
DRAWS = 2000
TUNE = 2000
CHAINS = 4
TARGET_ACCEPT = 0.9

# Document 17 §3's deflection channel, verbatim: a second defender touched the
# ball before the interceptor did.
SECOND_TOUCHER = "pass_defense_1_player_id"


# --------------------------------------------------------------------------
# arm 2 — conversion by covariates, at A-2's spec (Gate C-1)
# --------------------------------------------------------------------------


def fit_conversion(
    label: str,
    matrix: np.ndarray,
    names: tuple[str, ...],
    outcome: np.ndarray,
    defence_codes: np.ndarray,
    n_defence: int,
    qb_codes: np.ndarray,
    n_qb: int,
) -> tuple[dict, object]:
    """Document 43 §5's arm-2 model at amendment A-2's step counts.

    The model itself is `61`'s `build_conversion_model` and the Gate C-1 report is
    `62`'s `sampler_health`, both unchanged — only the sampler arguments differ
    from round 1, which is precisely what A-2 amends and why this function exists
    separately rather than reusing `62.fit_conversion` with patched constants.
    """
    print(
        f"\n  arm {label}: {matrix.shape[0]:,} throws, {matrix.shape[1]} covariates, "
        f"{CHAINS} x {DRAWS} draws after {TUNE} tuning, target_accept {TARGET_ACCEPT}"
    )
    model = _power.build_conversion_model(matrix, outcome, defence_codes, n_defence, qb_codes, n_qb)
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

    health = _confounds.sampler_health(idata, ["alpha", "beta", "sigma_d", "sigma_q", "z_d", "z_q"])
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

    print("    beta (logit scale, standardised covariates; * = 89% interval excludes zero)")
    for row in sorted(coefficients, key=lambda item: -abs(item["mean"])):
        marker = "*" if row["excludes_zero"] else " "
        print(
            f"      {marker} {row['name']:22s} {row['mean']:+.3f} "
            f"[{row['eti89'][0]:+.3f}, {row['eti89'][1]:+.3f}]  "
            f"odds x{row['odds_ratio']:.2f}"
        )

    variances = {}
    for parameter, entity in (("sigma_d", "defence-season"), ("sigma_q", "QB-season")):
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

    report = {
        "label": label,
        "throws": int(matrix.shape[0]),
        "covariates": list(names),
        "draws": DRAWS,
        "tune": TUNE,
        "chains": CHAINS,
        "target_accept": TARGET_ACCEPT,
        "sampler": health,
        "beta": coefficients,
        "alpha_mean": float(posterior["alpha"].values.mean()),
        "variance_components": variances,
    }
    return report, idata


# --------------------------------------------------------------------------
# A-3 — the hindsight probe on `is_interception_worthy`
# --------------------------------------------------------------------------


def hindsight_probe() -> dict:
    """Document 45 §2's probe: is "worthy" a judgement of the throw, or of the outcome?

    Every interception 2022-2025 joined to FTN, split by document 17 §3's
    second-toucher channel. A deflected pick is mostly a bounce, so a charter
    grading the *throw* without hindsight should call it worthy **less** often
    than a clean pick; a charter working backwards from the result would call
    both worthy at much the same rate.

    This frame is loaded separately from `load_charted_passes` on purpose.
    ``interception_player_id`` is on document 43 §4's post-branch exclusion list
    and must never reach the conversion model's design matrix — but the probe is
    *about* what happens after the ball reaches the defender, so it needs its own
    frame rather than an exception carved into the model's.
    """
    pbp = load_pbp(
        FTN_SEASONS,
        columns=[
            "game_id",
            "play_id",
            "season",
            "play_type",
            "interception",
            SECOND_TOUCHER,
            "interception_player_id",
        ],
    )
    ftn = load_ftn(FTN_SEASONS)
    charted = ftn.select(
        pl.col("nflverse_game_id").alias("game_id"),
        pl.col("nflverse_play_id").cast(pl.Float64).alias("play_id"),
        pl.col("is_interception_worthy"),
    ).join(pbp, on=["game_id", "play_id"], how="inner")

    picks = charted.filter((pl.col("play_type") == "pass") & (pl.col("interception") == 1))
    # A second toucher is a *different* defender credited with a pass defence on
    # the play the interceptor finished, so the channel needs an interceptor id to
    # compare against. Measured: every one of the charted interceptions carries
    # one, so this filter drops nothing — it is kept because a silent null would
    # otherwise be counted as "clean" rather than as unreadable.
    readable = picks.filter(pl.col("interception_player_id").is_not_null())
    deflected = (
        pl.col(SECOND_TOUCHER).is_not_null()
        & pl.col("interception_player_id").is_not_null()
        & (pl.col(SECOND_TOUCHER) != pl.col("interception_player_id"))
    )
    tagged = readable.with_columns(deflected.alias("deflected"))

    def _rate(subset: pl.DataFrame) -> dict:
        worthy = int(subset["is_interception_worthy"].sum())
        return {
            "interceptions": int(subset.height),
            "worthy": worthy,
            "p_worthy": worthy / subset.height if subset.height else None,
        }

    report = {
        "interceptions_charted": int(picks.height),
        "interceptions_with_a_readable_channel": int(tagged.height),
        "deflected": _rate(tagged.filter(pl.col("deflected"))),
        "clean": _rate(tagged.filter(~pl.col("deflected"))),
        "share_charted_not_worthy": float(1.0 - tagged["is_interception_worthy"].mean()),
    }
    gap_pp = (report["deflected"]["p_worthy"] - report["clean"]["p_worthy"]) * 100
    report["gap_pp"] = gap_pp
    # Document 45 §2's pre-committed reading, applied without a choice left in it.
    report["hindsight_suspected"] = bool(gap_pp >= 0.0)
    report["reading"] = (
        "worthy rate at or above the clean rate on deflected picks — hindsight suspected; "
        "every conversion number in this study carries that caveat in words"
        if report["hindsight_suspected"]
        else "worthy rate materially below the clean rate on deflected picks — the flag is "
        "behaving as a judgement of the throw and the selection is defensible"
    )

    print(
        f"\n  {report['interceptions_charted']:,} charted interceptions 2022-2025, "
        f"{report['interceptions_with_a_readable_channel']:,} with a readable second-toucher "
        f"channel"
    )
    for key in ("deflected", "clean"):
        entry = report[key]
        print(
            f"    p(worthy | INT, {key:9s}) = {entry['p_worthy']:.4f}  "
            f"({entry['worthy']:,} of {entry['interceptions']:,})"
        )
    print(
        f"    share of charted interceptions marked NOT worthy: {report['share_charted_not_worthy']:.4f}"
    )
    print(
        f"  Hindsight probe: p(worthy | INT, deflected) {report['deflected']['p_worthy']:.2f} "
        f"vs clean {report['clean']['p_worthy']:.2f} (gap {gap_pp:+.1f} pp) -> {report['reading']}"
    )
    return report


# --------------------------------------------------------------------------


def main() -> None:
    paths.ensure_data_dirs()
    power_path = paths.RESEARCH_OUTPUT_DIR / "63_dropped_pick_power_r2.json"
    if not power_path.exists():
        raise SystemExit(f"{power_path} is missing — Part A must run and commit before Part B")
    power = json.loads(power_path.read_text())["designs"]

    frame = _power.build_worthy_frame()

    # --- arm 2 and 2b, at A-2's spec -----------------------------------------
    print("\n=== arm 2 — conversion by covariates, amendment A-2 (Gate C-1) ===")
    arm2, idata = fit_conversion(
        "2",
        frame.design_matrix,
        frame.feature_names,
        frame.outcome,
        frame.defence_season_codes,
        frame.n_defence_seasons,
        frame.qb_season_codes,
        frame.n_qb_seasons,
    )

    keep = [
        index for index, name in enumerate(frame.feature_names) if name not in HINDSIGHT_COLUMNS
    ]
    arm2b, _ = fit_conversion(
        "2b (no is_catchable_ball / is_contested_ball)",
        frame.design_matrix[:, keep],
        tuple(frame.feature_names[index] for index in keep),
        frame.outcome,
        frame.defence_season_codes,
        frame.n_defence_seasons,
        frame.qb_season_codes,
        frame.n_qb_seasons,
    )

    print(
        f"\n  Gate C-1 (arm 2): divergences {arm2['sampler']['divergences']}, "
        f"max r_hat {arm2['sampler']['max_r_hat']:.4f}, "
        f"min ess_bulk {arm2['sampler']['min_ess_bulk']:.0f}, "
        f"min ess_tail {arm2['sampler']['min_ess_tail']:.0f} -> "
        f"{'PASS' if arm2['sampler']['pass'] else 'FAIL'}"
    )
    print(
        f"  Gate C-1 (arm 2b): divergences {arm2b['sampler']['divergences']}, "
        f"max r_hat {arm2b['sampler']['max_r_hat']:.4f}, "
        f"min ess_bulk {arm2b['sampler']['min_ess_bulk']:.0f}, "
        f"min ess_tail {arm2b['sampler']['min_ess_tail']:.0f} -> "
        f"{'PASS' if arm2b['sampler']['pass'] else 'FAIL'}"
    )

    # --- the fixed effects the gate is judged at ------------------------------
    # Arm 3 residualises against Part A's saved beta_hat, not against this fit,
    # because Part A's thresholds were simulated around that p_hat. The two
    # agreements below say how far apart the objects are.
    saved = json.loads((paths.RESEARCH_OUTPUT_DIR / "61_beta_hat.json").read_text())
    refit_beta = np.array([row["mean"] for row in arm2["beta"]])
    beta_gap_part_a = float(np.abs(refit_beta - np.asarray(saved["beta"])).max())
    alpha_gap_part_a = abs(arm2["alpha_mean"] - saved["alpha"])

    round1_path = paths.RESEARCH_OUTPUT_DIR / "62_dropped_pick_confounds.json"
    round1 = json.loads(round1_path.read_text()) if round1_path.exists() else None
    if round1 is not None:
        round1_beta = np.array([row["mean"] for row in round1["arm2_conversion"]["beta"]])
        beta_gap_round1 = float(np.abs(refit_beta - round1_beta).max())
        alpha_gap_round1 = abs(arm2["alpha_mean"] - round1["arm2_conversion"]["alpha_mean"])
    else:
        beta_gap_round1 = alpha_gap_round1 = None

    print(
        f"\n  arm 2 vs Part A's saved fixed effects: max |d beta| {beta_gap_part_a:.4f}, "
        f"|d alpha| {alpha_gap_part_a:.4f}"
    )
    if beta_gap_round1 is not None:
        print(
            f"  arm 2 vs round 1's arm 2 (1,000/1,000 draws): max |d beta| "
            f"{beta_gap_round1:.4f}, |d alpha| {alpha_gap_round1:.4f}"
        )

    # --- arm 3, on the floorless frame ---------------------------------------
    print("\n=== arm 3 — persistence of the conditioned residual (Gates C-2, C-3) ===")
    residual_rows = frame.model  # A-1: no floor, so arm 3's frame is arm 2's
    if residual_rows.height != frame.model.height:
        raise SystemExit("arm 3's frame differs from arm 2's — document 45 §2's stop-and-ask.")
    print(f"  arm3 rows == arm2 rows: True ({residual_rows.height:,} throws)")

    eta = _power.linear_predictor(saved, residual_rows, frame.model)
    p_hat = 1.0 / (1.0 + np.exp(-eta))
    outcome = residual_rows["interception"].cast(pl.Float64).to_numpy()
    residual = outcome - p_hat

    defence_season_codes, n_defence_season = _power._codes(residual_rows, ["season", "defteam"])
    defence_pooled_codes, n_defence_pooled = _power._codes(residual_rows, ["defteam"])
    qb_codes, n_qb = _power._codes(residual_rows, ["season", "passer_player_id"])

    arm2_upper_pp = arm2["variance_components"]["sigma_d"]["pp_eti89"][1]
    arm3 = {
        "defence_season_x_qb_season": _confounds.residual_persistence(
            "defence-season x QB-season",
            residual,
            defence_season_codes,
            n_defence_season,
            qb_codes,
            n_qb,
            power["residual_defence_season_x_qb_season"],
            arm2_upper_pp,
        ),
        "defence_pooled_x_qb_season": _confounds.residual_persistence(
            "defence pooled x QB-season",
            residual,
            defence_pooled_codes,
            n_defence_pooled,
            qb_codes,
            n_qb,
            power["residual_defence_pooled_x_qb_season"],
            arm2_upper_pp,
        ),
    }
    qb_power = power["residual_qb_season_sigma_q"]
    qb_upper_pp = arm3["defence_season_x_qb_season"]["sigma_q_eti89_pp"][1]
    arm3["qb_season_sigma_q"] = {
        "name": "QB-season sigma_q (read from the defence-season x QB-season fit)",
        "upper_bound_pp": qb_upper_pp,
        "gate_threshold_pp": qb_power["gate_threshold_pp"],
        "gate_c2_pass": bool(qb_upper_pp < qb_power["gate_threshold_pp"]),
        "power_at_reference": qb_power["power_at_reference"],
        "gate_c3_pass": bool(qb_power["resolvable"]),
        "reportable_as_finding": bool(qb_power["resolvable"]),
    }
    entry = arm3["qb_season_sigma_q"]
    print(
        f"\n  QB-season sigma_q: upper bound {qb_upper_pp:.2f} pp vs threshold "
        f"{entry['gate_threshold_pp']:.2f} pp -> {'PASS' if entry['gate_c2_pass'] else 'FAIL'}; "
        f"power {entry['power_at_reference']:.3f} -> "
        f"{'PASS' if entry['gate_c3_pass'] else 'FAIL'}"
    )

    # --- secondaries, now on the full 128 defence-seasons ---------------------
    print("\n=== secondaries (reported, never gated) ===")
    conditioned = residual_rows.with_columns(pl.Series("residual", residual))
    secondaries = {
        "raw_conversion_split_half_defence_season": _confounds.split_half(
            frame.worthy.with_columns(pl.col("interception").cast(pl.Float64)),
            "interception",
            ["season", "defteam"],
        ),
        "conditioned_residual_split_half_defence_season": _confounds.split_half(
            conditioned, "residual", ["season", "defteam"]
        ),
        "raw_conversion_split_half_defence_pooled": _confounds.split_half(
            frame.worthy.with_columns(pl.col("interception").cast(pl.Float64)),
            "interception",
            ["defteam"],
        ),
        "conditioned_residual_split_half_defence_pooled": _confounds.split_half(
            conditioned, "residual", ["defteam"]
        ),
    }
    for name, entry in secondaries.items():
        value = "n/a" if entry["r"] is None else f"{entry['r']:+.3f}"
        print(f"  {name}: r = {value} on {entry['entities']} entities")

    labels = (
        frame.model.select("season", "defteam")
        .with_columns(pl.concat_str(["season", "defteam"], separator="|").alias("label"))["label"]
        .unique(maintain_order=True)
        .to_list()
    )
    effect_means = idata["posterior"]["u_d"].values.mean(axis=(0, 1))
    effects = pl.DataFrame(
        {
            "season": [int(label.split("|")[0]) for label in labels],
            "defteam": [label.split("|")[1] for label in labels],
            "effect": effect_means[: len(labels)],
        }
    )
    secondaries["shrunk_defence_effect_season_to_season"] = _confounds.season_to_season(effects)
    entry = secondaries["shrunk_defence_effect_season_to_season"]
    value = "n/a" if entry["r"] is None else f"{entry['r']:+.3f}"
    print(f"  shrunk_defence_effect_season_to_season: r = {value} on {entry['pairs']} pairs")

    # --- A-3 ------------------------------------------------------------------
    print("\n=== A-3 — hindsight probe on `is_interception_worthy` (reported, never gated) ===")
    probe = hindsight_probe()

    out = paths.RESEARCH_OUTPUT_DIR / "64_dropped_pick_confounds_r2.json"
    out.write_text(
        json.dumps(
            {
                "amendments": ["A-1 no floor", "A-2 sampler spec", "A-3 hindsight probe"],
                "random_seed": RANDOM_SEED,
                "draws": DRAWS,
                "tune": TUNE,
                "chains": CHAINS,
                "target_accept": TARGET_ACCEPT,
                "cross_check_tolerance_pp": CROSS_CHECK_TOLERANCE_PP,
                "guards": frame.guards,
                "arm3_rows_equal_arm2_rows": True,
                "residual_frame": {
                    "rows": int(residual_rows.height),
                    "defence_seasons": int(n_defence_season),
                    "defences_pooled": int(n_defence_pooled),
                    "qb_seasons": int(n_qb),
                    "conversion_rate": float(outcome.mean()),
                    "mean_p_hat": float(p_hat.mean()),
                },
                "arm2_conversion": arm2,
                "arm2b_no_hindsight_columns": arm2b,
                "arm2_vs_part_a_beta": {
                    "max_abs_beta_gap": beta_gap_part_a,
                    "alpha_gap": alpha_gap_part_a,
                },
                "arm2_vs_round1_beta": {
                    "max_abs_beta_gap": beta_gap_round1,
                    "alpha_gap": alpha_gap_round1,
                },
                "arm3_residual_persistence": arm3,
                "secondaries": secondaries,
                "hindsight_probe": probe,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
