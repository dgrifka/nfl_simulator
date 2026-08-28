"""Round 8, Part C — amendment A-3's two reported sensitivities, S-1 and S-2.

Document 58 §3. Both are **reported, not gated**: A-3 is enacted by ruling R-4
and nothing in the component changes on what this script prints. What changes is
what the write-up has to say beside the numbers, which is why the readings are
pre-committed here rather than chosen after the arithmetic lands.

**S-1 — capped swing on receiver drops.** The receiver component prices each
catchable target at what catching it was worth, and that per-play swing has a
long right tail: document 57 §4 measured a median of 1.37 EPA against a maximum
of 11.36. The question is whether the `+rd` arm's 14% of games changing verdict
bucket is *the event* — a 1-in-20 coin, three dozen times a game — or *the
dropped touchdowns*, a handful of enormous swings doing the work. So the `+rd`
and `full` arms are re-run twice more with the swing magnitude capped: once at
the **95th percentile** of the swing distribution the production read side
actually prices (printed, not assumed), and once at **5.04 EPA**, the largest
cell in the dropped-pick swing table, so the two directions of the class are
priced against a common ceiling.

    Pre-committed reading (document 58 §3, before the run): if bucket moves
    under the 95th-percentile cap are >= 80% of uncapped **on the `+rd` arm**,
    the 14% is the event and the uncapped pricing ships. If under 80%, the
    uncapped pricing still ships -- it is the play's own counterfactual and
    document 56 §2 pre-registered it -- but the write-up must state that the
    tail is dropped touchdowns and the share of movement they carry.

**S-2 — the contested/uncatchable charting link.** Document 57 §4's hindsight
probe fired: drops are *less* frequent on contested catchable balls than on
uncontested ones (4.29% vs 5.04%), the opposite of what document 56 §1
pre-committed. The stated mechanism is that a contested ball knocked away gets
charted **uncatchable**, so the contested balls inside the drop frame are the
cleanly catchable ones. S-2 measures that directly on all 2022-2025 targets:
`p(catchable | contested)` against `p(catchable | uncontested)`, and among
*incompletions*, the share charted uncatchable by contest status. If contested
incompletions are charted uncatchable far more often, the drop frame is
conditioned on a charter judgement that partly encodes the outcome, and every
drop-skill spread in document 57 is a **floor**. No change to the component
follows; the caveat goes on document 05 §3's row and into the write-up.

**V-1 is first and last, as in every round that touches the variant.** Strict
(v1.3, renamed by ruling R-4) must replay at 0.00e+00 over 2,761 games before
any sensitivity number is read, and again after.

    uv run python research/75_a3_sensitivities.py
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_audit = import_module("68_dropped_pick_variant_audit")
_mirror = import_module("73_receiver_drop_variant_audit")
_receiver = import_module("71_receiver_drop_power")
_confounds = import_module("72_receiver_drop_confounds")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.dropped_picks import DroppedPickModel  # noqa: E402
from nfl_simulator.receiver_drops import ReceiverDropModel  # noqa: E402

OUTPUT_NAME = "75_a3_sensitivities.json"

# Document 58 §3, fixed before the run and not re-tolerancing anything: S-1's
# reading is a share of bucket moves and its bar is 0.80. The second cap is the
# dropped-pick swing table's largest cell, so the two directions of the
# hands-on-the-ball class meet a common ceiling.
S1_PERCENTILE = 95.0
S1_DP_LARGEST_CELL = 5.04
S1_READING_BAR = 0.80

# Document 57 §5's uncapped arms, the numbers this script has to reproduce
# before any capped arm is read. Sampler noise only — the seed and the draw
# counts are v1.3's, so an arm that misses these is a code change, not noise.
DOC57_RD_BUCKET_MOVES = 162
DOC57_FULL_BUCKET_MOVES = 200
DOC57_RD_MEDIAN_PP = 2.32
DOC57_FULL_MEDIAN_PP = 3.85
MOVE_TOLERANCE = 5
DTW_TOLERANCE_PP = 0.2

# Document 57 §4's per-play swing summary, so the frame the percentile is taken
# over is checked against the record before the cap is set from it.
DOC57_SWING_MEDIAN = 1.37
DOC57_SWING_MEAN = 1.78
DOC57_SWING_MAX = 11.36
SWING_TOLERANCE_EPA = 0.02

# Document 57 §4's completion count, so S-2's target frame is checked to be the
# frame the hindsight probe was measured on before its shares are read.
DOC57_COMPLETIONS = 48_511

ETI_LOW, ETI_HIGH = _audit.ETI_LOW, _audit.ETI_HIGH


# --------------------------------------------------------------------------
# S-1 — the capped-swing arms
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CappedSwingModel:
    """A `ReceiverDropModel` whose per-play swing magnitude is capped.

    Wrapping rather than editing the swing table, because the swing S-1 caps is
    the **per-play** one — `|(air_epa + xyac_epa) − epa_incomplete|`, computed
    from the play's own completion counterfactual — and the table only holds the
    six-cell fallback. `simulator.receiver_drop_events` asks a model for exactly
    two things, a catch probability and a swing, so a delegate that answers both
    is the whole change and the production path is untouched.

    The cap is on the magnitude and the sign is preserved, mirroring
    `receiver_drop_events`'s own `abs()`: capping a signed value would silently
    flip who the luck belonged to.
    """

    base: ReceiverDropModel
    cap: float

    def catch_probability(self, entity_season, row):
        return self.base.catch_probability(entity_season, row)

    def swing_for_play(self, row) -> float:
        swing = self.base.swing_for_play(row)
        return math.copysign(min(abs(swing), self.cap), swing)


def per_play_swings(model: ReceiverDropModel) -> np.ndarray:
    """Every swing the production read side prices, over 2022-2025.

    Read through `model.swing_for_play` rather than re-derived, so the
    percentile S-1 caps at is a percentile of the distribution the component
    actually uses — including both of document 57 §4's fallback branches. The
    frame is `research/72`'s, the one document 57 §4's median, mean and maximum
    were printed from, so the guard below can check this against the record.
    """
    frame = _receiver.build_catchable_frame(verbose=False)
    with_epa = _confounds.catchable_with_epa(frame)
    return np.array(
        [abs(model.swing_for_play(row)) for row in with_epa.iter_rows(named=True)],
        dtype=float,
    )


def swing_distribution(swings: np.ndarray) -> dict:
    report = {
        "targets": int(swings.size),
        "median_epa": float(np.median(swings)),
        "mean_epa": float(swings.mean()),
        "eti89_epa": [float(v) for v in np.percentile(swings, [ETI_LOW, ETI_HIGH])],
        "p95_epa": float(np.percentile(swings, S1_PERCENTILE)),
        "max_epa": float(swings.max()),
        "share_above_p95_cap": float((swings > np.percentile(swings, S1_PERCENTILE)).mean()),
        "share_above_dp_cap": float((swings > S1_DP_LARGEST_CELL).mean()),
    }
    print(f"\n{'=' * 72}\nS-1 — the swing distribution the caps bite into\n{'=' * 72}")
    print(
        f"  {report['targets']:,} catchable targets: median {report['median_epa']:.2f} EPA, "
        f"mean {report['mean_epa']:.2f}, 89% [{report['eti89_epa'][0]:.2f}, "
        f"{report['eti89_epa'][1]:.2f}], max {report['max_epa']:.2f}"
    )
    print(
        f"  95th percentile {report['p95_epa']:.2f} EPA — it caps "
        f"{report['share_above_p95_cap']:.1%} of targets; the dropped-pick table's "
        f"largest cell {S1_DP_LARGEST_CELL:.2f} EPA caps {report['share_above_dp_cap']:.1%}"
    )
    off_by = {
        "median": abs(report["median_epa"] - DOC57_SWING_MEDIAN),
        "mean": abs(report["mean_epa"] - DOC57_SWING_MEAN),
        "max": abs(report["max_epa"] - DOC57_SWING_MAX),
    }
    report["reproduces_doc57_swing_summary"] = bool(max(off_by.values()) <= SWING_TOLERANCE_EPA)
    print(
        f"  against document 57 §4's median {DOC57_SWING_MEDIAN:.2f} / mean "
        f"{DOC57_SWING_MEAN:.2f} / max {DOC57_SWING_MAX:.2f} -> "
        f"{'PASS' if report['reproduces_doc57_swing_summary'] else 'FAIL'}"
    )
    if not report["reproduces_doc57_swing_summary"]:
        raise SystemExit(
            "the swing distribution is not document 57 §4's. The cap would be a "
            "percentile of the wrong frame; stop and report."
        )
    return report


def arm_summary(charted: pl.DataFrame, arm: pl.DataFrame, label: str) -> dict:
    """Bucket moves, median and mean |ΔDTW|, and the 89% tail, on affected games."""
    moved, delta_dtw, half_width = _mirror.bucket_moves(charted, arm)
    affected = arm.sort("game_id")["n_receiver_drop_events"].to_numpy() > 0
    move = np.abs(delta_dtw[affected]) * 100
    return {
        "label": label,
        "affected_games": int(affected.sum()),
        "bucket_moves": len(moved),
        "bucket_move_game_ids": sorted(moved),
        "median_abs_delta_dtw_pp": float(np.median(move)),
        "mean_abs_delta_dtw_pp": float(move.mean()),
        "eti89_abs_delta_dtw_pp": [float(v) for v in np.percentile(move, [ETI_LOW, ETI_HIGH])],
        "median_half_width_pp": float(np.median(half_width[affected]) * 100),
    }


def reproduction_guard(uncapped: dict[str, dict]) -> dict:
    """The uncapped arms must still be document 57 §5's, or nothing below means anything."""
    checks = {}
    for label, expected_moves, expected_pp in (
        ("+rd", DOC57_RD_BUCKET_MOVES, DOC57_RD_MEDIAN_PP),
        ("full", DOC57_FULL_BUCKET_MOVES, DOC57_FULL_MEDIAN_PP),
    ):
        arm = uncapped[label]
        move_gap = abs(arm["bucket_moves"] - expected_moves)
        dtw_gap = abs(arm["median_abs_delta_dtw_pp"] - expected_pp)
        checks[label] = {
            "bucket_moves": arm["bucket_moves"],
            "bucket_moves_doc57": expected_moves,
            "median_abs_delta_dtw_pp": arm["median_abs_delta_dtw_pp"],
            "median_abs_delta_dtw_pp_doc57": expected_pp,
            "pass": bool(move_gap <= MOVE_TOLERANCE and dtw_gap <= DTW_TOLERANCE_PP),
        }
    print(f"\n{'=' * 72}\nGUARD — the uncapped arms are still document 57 §5's\n{'=' * 72}")
    for label, check in checks.items():
        print(
            f"  {label:6s} {check['bucket_moves']} moves against "
            f"{check['bucket_moves_doc57']} (±{MOVE_TOLERANCE}); median |ΔDTW| "
            f"{check['median_abs_delta_dtw_pp']:.2f} pp against "
            f"{check['median_abs_delta_dtw_pp_doc57']:.2f} (±{DTW_TOLERANCE_PP:.1f}) -> "
            f"{'PASS' if check['pass'] else 'FAIL'}"
        )
    if not all(check["pass"] for check in checks.values()):
        raise SystemExit(
            "an uncapped arm no longer reproduces document 57 §5. The rename or the "
            "cap wrapper moved a round-7 number; stop and report."
        )
    return checks


def s1_report(arms: dict[str, dict], distribution: dict) -> dict:
    """S-1's table, and the one line document 58 §3 pre-committed to reading off it."""
    print(f"\n{'=' * 72}\nS-1 — capped swing, reported and not gated\n{'=' * 72}")
    print(f"  {'arm':22s} {'moves':>6s} {'median':>8s} {'mean':>7s} {'89% tail':>16s}  {'n':>6s}")
    for key in (
        "+rd uncapped",
        f"+rd p{S1_PERCENTILE:.0f} cap",
        f"+rd {S1_DP_LARGEST_CELL:.2f} cap",
        "full uncapped",
        f"full p{S1_PERCENTILE:.0f} cap",
        f"full {S1_DP_LARGEST_CELL:.2f} cap",
    ):
        arm = arms[key]
        low, high = arm["eti89_abs_delta_dtw_pp"]
        print(
            f"  {key:22s} {arm['bucket_moves']:6d} "
            f"{arm['median_abs_delta_dtw_pp']:7.2f}pp {arm['mean_abs_delta_dtw_pp']:6.2f}pp "
            f"  [{low:5.2f}, {high:6.2f}]pp {arm['affected_games']:6d}"
        )

    shares = {}
    for direction in ("+rd", "full"):
        uncapped = arms[f"{direction} uncapped"]["bucket_moves"]
        for cap_label, cap_key in (
            ("p95", f"{direction} p{S1_PERCENTILE:.0f} cap"),
            ("dp_cell", f"{direction} {S1_DP_LARGEST_CELL:.2f} cap"),
        ):
            shares[f"{direction}/{cap_label}"] = arms[cap_key]["bucket_moves"] / uncapped

    share = shares["+rd/p95"]
    holds = bool(share >= S1_READING_BAR)
    reading = (
        "the 14% is the event, not the dropped touchdowns, and the uncapped pricing ships"
        if holds
        else (
            "the uncapped pricing still ships — it is the play's own counterfactual "
            "and document 56 §2 pre-registered it — but the write-up must state that "
            "the tail is dropped touchdowns and the share of movement they carry"
        )
    )
    print(
        f"\n  READING (pre-committed, document 58 §3): `+rd` bucket moves under the "
        f"{S1_PERCENTILE:.0f}th-percentile cap are {share:.1%} of uncapped, "
        f"{'at or above' if holds else 'below'} the {S1_READING_BAR:.0%} bar — {reading}."
    )
    return {
        "statistic": (
            "bucket moves, median and mean |ΔDTW| and its 89% tail on affected games, "
            "under each swing cap, beside uncapped"
        ),
        "gated": False,
        "caps_epa": {"p95": distribution["p95_epa"], "dp_largest_cell": S1_DP_LARGEST_CELL},
        "swing_distribution": distribution,
        "arms": arms,
        "bucket_move_share_of_uncapped": shares,
        "reading_bar": S1_READING_BAR,
        "reading_share": share,
        "reading_holds": holds,
        "reading": reading,
    }


# --------------------------------------------------------------------------
# S-2 — the contested/uncatchable charting link
# --------------------------------------------------------------------------


def s2_report() -> dict:
    """Document 58 §3's two conditional probabilities and the incompletion split."""
    charted = _receiver.load_charted_plays()
    targets = charted.filter(pl.col("complete_pass").eq(1) | pl.col("incomplete_pass").eq(1))
    completions = int(targets.filter(pl.col("complete_pass").eq(1)).height)
    if completions != DOC57_COMPLETIONS:
        raise SystemExit(
            f"the target frame holds {completions:,} completions against document 57 §4's "
            f"{DOC57_COMPLETIONS:,} — it is not the frame the probe was measured on. "
            "Stop and report."
        )

    def share_catchable(frame: pl.DataFrame) -> tuple[float, int]:
        return float(frame["is_catchable_ball"].mean()), int(frame.height)

    contested = targets.filter(pl.col("is_contested_ball"))
    uncontested = targets.filter(~pl.col("is_contested_ball"))
    p_contested, n_contested = share_catchable(contested)
    p_uncontested, n_uncontested = share_catchable(uncontested)

    incomplete = targets.filter(pl.col("incomplete_pass").eq(1))
    inc_contested = incomplete.filter(pl.col("is_contested_ball"))
    inc_uncontested = incomplete.filter(~pl.col("is_contested_ball"))
    unc_contested = float((~inc_contested["is_catchable_ball"]).mean())
    unc_uncontested = float((~inc_uncontested["is_catchable_ball"]).mean())

    ratio = p_contested / p_uncontested
    gap_pp = (unc_contested - unc_uncontested) * 100
    marginal_pp = (p_contested - p_uncontested) * 100

    # Document 58 §3's pre-registered trigger is the *incompletion* split: it
    # fires only if contested incompletions are charted uncatchable far more
    # often. The marginal probabilities are reported beside it because they can
    # disagree with it — a contested throw is far likelier to be incomplete at
    # all, so conditioning on an incompletion is not the same comparison — and a
    # sentence that read only one of the two would be reporting half the answer.
    fires = bool(unc_contested > unc_uncontested)
    marginal_supports = bool(p_contested < p_uncontested)
    lead = (
        f"A contested target is charted catchable {p_contested:.1%} of the time against "
        f"{p_uncontested:.1%} uncontested ({marginal_pp:+.1f} pp, ratio {ratio:.2f}x), and "
        f"among incompletions contested balls are charted uncatchable {unc_contested:.1%} "
        f"against {unc_uncontested:.1%} ({gap_pp:+.1f} pp)."
    )
    if fires:
        verdict = (
            " The pre-registered trigger fires: a contested ball knocked away leaves the "
            "drop frame by the charter's judgement rather than by the receiver's hands, so "
            "`is_catchable_ball` partly encodes the outcome and every drop-skill spread in "
            "document 57 is a floor rather than an estimate."
        )
    elif marginal_supports:
        verdict = (
            " The pre-registered trigger does not fire — it asked for the incompletion "
            "split and that split runs the other way — but the marginal probability does "
            "point where document 57 §4's mechanism said it would: contest makes a ball "
            "less likely to be judged catchable at all. The two are not in contradiction, "
            "because a contested throw is much likelier to be incomplete in the first "
            "place, so conditioning on an incompletion changes the comparison. The honest "
            "reading is that the charter's judgement is associated with contest but the "
            "pre-registered evidence that it encodes the *outcome* is absent, so document "
            "57's spreads carry this as an open caveat rather than a measured floor."
        )
    else:
        verdict = (
            " Neither half fires: the charter's judgement does not remove contested "
            "knock-aways from the drop frame, and document 57's spreads need no floor "
            "caveat on this account."
        )
    direction = lead + verdict

    report = {
        "statistic": (
            "p(catchable | contested) against p(catchable | uncontested) on all "
            "2022-2025 targets, and among incompletions the share charted uncatchable "
            "by contest status"
        ),
        "gated": False,
        "targets": int(targets.height),
        "p_catchable_given_contested": p_contested,
        "n_contested": n_contested,
        "p_catchable_given_uncontested": p_uncontested,
        "n_uncontested": n_uncontested,
        "ratio_contested_to_uncontested": ratio,
        "incompletions": int(incomplete.height),
        "share_uncatchable_incomplete_contested": unc_contested,
        "n_incomplete_contested": int(inc_contested.height),
        "share_uncatchable_incomplete_uncontested": unc_uncontested,
        "n_incomplete_uncontested": int(inc_uncontested.height),
        "gap_pp": gap_pp,
        "marginal_gap_pp": marginal_pp,
        "fires": fires,
        "marginal_supports_mechanism": marginal_supports,
        "direction": direction,
    }

    print(f"\n{'=' * 72}\nS-2 — the contested/uncatchable charting link\n{'=' * 72}")
    print(f"  {report['targets']:,} charted targets over 2022-2025")
    print(
        f"  p(catchable | contested)   {p_contested:.1%}  [n = {n_contested:,}]\n"
        f"  p(catchable | uncontested) {p_uncontested:.1%}  [n = {n_uncontested:,}]"
        f"   ratio {ratio:.2f}x"
    )
    print(
        f"  among {report['incompletions']:,} incompletions, charted uncatchable: "
        f"contested {unc_contested:.1%} [n = {report['n_incomplete_contested']:,}], "
        f"uncontested {unc_uncontested:.1%} [n = {report['n_incomplete_uncontested']:,}]"
    )
    print(f"\n  DIRECTION: {direction}")
    return report


# --------------------------------------------------------------------------


def main() -> None:
    ctx = _audit.load_context()

    # ---- V-1, first and unconditional -------------------------------------
    v13_table, _v13_ledger, v1_first = _audit.v13_pass(ctx)
    if not v1_first["pass"]:
        raise SystemExit("V-1 failed before anything was read. Stop and report.")

    dropped_pick_model = DroppedPickModel.from_posterior(
        paths.RESEARCH_OUTPUT_DIR / _audit.TRACE_NAME,
        paths.RESEARCH_OUTPUT_DIR / _audit.SUMMARY_NAME,
    )
    receiver_model = ReceiverDropModel.from_posterior(
        paths.RESEARCH_OUTPUT_DIR / _mirror.TRACE_NAME,
        paths.RESEARCH_OUTPUT_DIR / _mirror.SUMMARY_NAME,
    )

    # ---- S-1 ---------------------------------------------------------------
    distribution = swing_distribution(per_play_swings(receiver_model))
    caps = {
        f"p{S1_PERCENTILE:.0f} cap": distribution["p95_epa"],
        f"{S1_DP_LARGEST_CELL:.2f} cap": S1_DP_LARGEST_CELL,
    }
    models = {"uncapped": receiver_model} | {
        name: CappedSwingModel(receiver_model, cap) for name, cap in caps.items()
    }

    arms, charted = {}, None
    for direction, dp_model in (("+rd", None), ("full", dropped_pick_model)):
        for name, rd_model in models.items():
            label = f"{direction} {name}"
            table, _ledger = _audit.variant_pass(
                ctx, dp_model, receiver_drop_model=rd_model, label=label
            )
            if charted is None:
                charted = v13_table.filter(
                    pl.col("game_id").is_in(table["game_id"].to_list())
                ).sort("game_id")
            arms[label] = arm_summary(charted, table, label)

    guard = reproduction_guard({"+rd": arms["+rd uncapped"], "full": arms["full uncapped"]})
    s1 = s1_report(arms, distribution)

    # ---- S-2 ---------------------------------------------------------------
    s2 = s2_report()

    # ---- V-1 again, last ---------------------------------------------------
    _table, _ledger, v1_last = _audit.v13_pass(ctx)
    print(
        f"\nV-1 (again, last): max |Δ deserved margin| "
        f"{v1_last['max_abs_gaps']['deserved_margin']:.2e} over "
        f"{v1_last['games_matched']:,} games -> {'PASS' if v1_last['pass'] else 'FAIL'}"
    )
    if not v1_last["pass"]:
        raise SystemExit("V-1 failed after the sensitivity arms. Stop and report.")

    results = {
        "reported_as": (
            "document 58 §3's two sensitivities. Reported, never gated: amendment A-3 "
            "is enacted by ruling R-4 and nothing in the component changes on these "
            "numbers. What changes is what the write-up must say beside them."
        ),
        "settings": {
            "random_seed": _audit.RANDOM_SEED,
            "posterior_draws": _audit.POSTERIOR_DRAWS,
            "coin_draws": _audit.COIN_DRAWS,
            "points_per_epa": ctx.slope,
        },
        "gate_v1_default_off_first": v1_first,
        "gate_v1_default_off_last": v1_last,
        "doc57_reproduction_guard": guard,
        "s1_capped_swing": s1,
        "s2_contested_uncatchable": s2,
    }
    out = paths.RESEARCH_OUTPUT_DIR / OUTPUT_NAME
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
