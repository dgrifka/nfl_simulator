"""Phase 3, step 2 — the sequencing-luck round, run against pre-registered gates.

Design, statistics and thresholds are fixed by `docs/research/08-sequencing-luck.md`,
committed at `2f27c1a` before this script produced a result. Nothing here chooses
anything; it executes what that document committed to.

    Gate S-1  positive control — overall offensive EPA per play must persist
              far above the null, or the harness is broken and nothing else is
              readable.
    Gate S-2  per measure — split-half r must exceed the permutation null's
              95th percentile for that measure.
    Gate S-3  honesty — a failure is only interpretable where power at the
              r = 0.12 reference is at least 0.80.

The power calculation's own machinery is imported rather than re-implemented, so
the executed statistic is literally the simulated one.

    uv run python research/10_sequencing.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_power = import_module("10_sequencing_power")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.ingest import PBP_SEASONS, load_pbp  # noqa: E402

RANDOM_SEED = _power.RANDOM_SEED
N_SPLITS = _power.N_SPLITS
MEASURES = _power.MEASURES

# Pre-registered in document 08 section 6, from the permutation null in
# `research/outputs/10_sequencing_power.json`. Hard-coded here rather than read
# back, so a re-run of the power script cannot silently move a committed gate.
GATE_S2_THRESHOLDS = {
    "S1_redzone_gap": 0.0703,
    "S2_latedown_gap": 0.0689,
    "S3_wpa_epa_gap": 0.0648,
}
GATE_S1_THRESHOLD = 0.0917  # null 99th percentile, S0
REFERENCE_R = _power.REFERENCE_R
MIN_POWER = 0.80

SEQUENCING_MEASURES = tuple(GATE_S2_THRESHOLDS)


def split_half_distribution(
    matrix: np.ndarray,
    mask: np.ndarray,
    starts: np.ndarray,
    slope: float,
) -> np.ndarray:
    """Per-split correlations, so the 5th-95th percentile band can be reported.

    `_power.split_half_r` returns only the mean over splits, which is the gated
    statistic. Document 02 also published the band across splits, and this round
    is directly compared to those numbers, so the band is recomputed here.
    """
    totals = np.add.reduceat(matrix, starts, axis=0)
    sums_a = _power.half_sums(matrix, mask, starts)
    sums_b = totals[None, :, :] - sums_a
    a = _power.half_statistics(sums_a, slope)
    b = _power.half_statistics(sums_b, slope)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("a half statistic was undefined; a denominator hit zero")
    return np.column_stack([_power.correlate(a[:, :, m], b[:, :, m]) for m in range(len(MEASURES))])


def per_game_variant(
    frame: pl.DataFrame, slope: float, rng: np.random.Generator
) -> dict[str, float]:
    """Document 02's own machinery: average a per-game rate, then correlate halves.

    Reported as a secondary because it is the estimator document 02 published, so
    keeping it on the page makes this round's numbers comparable to the ones
    already on record. It is *not* the gated statistic — section 2 of document 08
    records why pooling is the honest estimator when a team-game holds a median
    of nine red-zone plays.
    """
    per_game = frame.with_columns(
        (pl.col("epa_all") / pl.col("n_all")).alias("S0_overall_epa"),
        (pl.col("epa_rz") / pl.col("n_rz") - pl.col("epa_all") / pl.col("n_all")).alias(
            "S1_redzone_gap"
        ),
        (pl.col("succ_ld") / pl.col("n_ld") - pl.col("succ_all") / pl.col("n_all")).alias(
            "S2_latedown_gap"
        ),
        (pl.col("wpa_valued") - slope * pl.col("epa_valued")).alias("S3_wpa_epa_gap"),
    )

    keys = per_game["team_season"].to_numpy()
    boundaries = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1], True])
    spans = [
        (boundaries[i], boundaries[i + 1])
        for i in range(len(boundaries) - 1)
        if boundaries[i + 1] - boundaries[i] >= _power.MIN_GAMES
    ]
    values = {m: per_game[m].to_numpy() for m in MEASURES}

    results = {}
    for measure in MEASURES:
        draws = np.empty(N_SPLITS)
        column = values[measure]
        for split in range(N_SPLITS):
            first, second = [], []
            for lo, hi in spans:
                permuted = rng.permutation(hi - lo) + lo
                cut = (hi - lo) // 2
                block_a = column[permuted[:cut]]
                block_b = column[permuted[cut:]]
                # A team-game with no red-zone plays makes the per-game gap
                # undefined. Dropping those games inside the half is what an
                # analyst averaging per game would do, and it is exactly the
                # fragility section 2 gives as the reason not to gate on this.
                first.append(np.nanmean(block_a))
                second.append(np.nanmean(block_b))
            first, second = np.array(first), np.array(second)
            usable = np.isfinite(first) & np.isfinite(second)
            draws[split] = np.corrcoef(first[usable], second[usable])[0, 1]
        results[measure] = float(draws.mean())
    return results


def s3_confound_checks(
    pbp: pl.DataFrame,
    frame: pl.DataFrame,
    matrix: np.ndarray,
    starts: np.ndarray,
    mask: np.ndarray,
    slope: float,
) -> dict:
    """EXPLORATORY, **not pre-registered.** Added after seeing that S3 persists.

    S3 is ``wpa - slope * epa`` summed over a game. A positive value means the
    team's production moved win probability more than its point value implies,
    which is the definition of good timing. But there is a second, duller way to
    get a positive S3: **play close games all season.** Win probability moves
    fast in a one-score game and barely at all once a game is decided, so a team
    whose games stay competitive converts EPA into WPA at a higher rate than the
    league slope regardless of when inside the game its production landed.

    "Plays close games" is itself a stable team property — it follows from being
    mediocre, or from a hard schedule — so it would produce exactly the
    persistence Gate S-2 detected, without any timing skill existing.

    Three checks, in increasing order of how much they would settle it:

    1. Does team-season S3 correlate with offensive quality (S0)?
    2. Does it correlate with how close that team's games were?
    3. **The decisive one** — does S3 still persist when it is computed only on
       plays in a competitive game state, where the leverage-versus-blowout
       distinction is removed by construction?

    Labelled exploratory throughout so it can never be read as confirmatory,
    following the precedent `research/08_rematch.py` set for its DTW% arm.
    """
    print(f"\n{'=' * 72}\nEXPLORATORY (not pre-registered) — what is S3 measuring?\n{'=' * 72}")

    # ---- 1 & 2: team-season correlates -----------------------------------
    totals = np.add.reduceat(matrix, starts, axis=0)
    season_stats = _power.half_statistics(totals, slope)
    s0_season = season_stats[:, 0]
    s3_season = season_stats[:, 3]

    closeness = (
        pbp.filter(pl.col("posteam").is_not_null() & pl.col("score_differential").is_not_null())
        .group_by(
            pl.concat_str(
                [pl.col("season").cast(pl.String), pl.col("posteam")], separator="_"
            ).alias("team_season")
        )
        .agg(pl.col("score_differential").abs().mean().alias("mean_abs_score_diff"))
        .sort("team_season")
    )
    kept = sorted(frame["team_season"].unique().to_list())
    kept = [
        team_season
        for team_season, size in zip(kept, np.diff(np.r_[starts, len(matrix)]), strict=False)
        if size >= _power.MIN_GAMES
    ]
    closeness_lookup = dict(
        zip(closeness["team_season"], closeness["mean_abs_score_diff"], strict=True)
    )
    close = np.array([closeness_lookup[team_season] for team_season in kept])

    r_s3_s0 = float(np.corrcoef(s3_season, s0_season)[0, 1])
    r_s3_close = float(np.corrcoef(s3_season, close)[0, 1])
    print(f"  corr(season S3, season S0 offensive quality) = {r_s3_s0:+.3f}")
    print(f"  corr(season S3, mean |score differential|)   = {r_s3_close:+.3f}")

    # ---- 3: S3 on competitive plays only ---------------------------------
    # One score is 8 points: within that, either team can still take the lead
    # with a single possession, so no play is in garbage time by any reading.
    competitive = pbp.filter(
        pl.col("posteam").is_not_null()
        & pl.col("epa").is_not_null()
        & pl.col("wpa").is_not_null()
        & (pl.col("score_differential").abs() <= 8)
    )
    epa_c = competitive["epa"].to_numpy()
    wpa_c = competitive["wpa"].to_numpy()
    # The slope is refit on this subset: leverage is uniformly higher here, so
    # reusing the full-sample slope would leave a constant offset in every team's
    # residual and manufacture a correlation from nothing.
    slope_c = float(np.cov(epa_c, wpa_c)[0, 1] / np.var(epa_c, ddof=1))

    competitive_games = (
        competitive.group_by(["season", "posteam", "game_id"])
        .agg(
            pl.len().alias("n_valued"),
            pl.col("epa").sum().alias("epa_valued"),
            pl.col("wpa").sum().alias("wpa_valued"),
        )
        .with_columns(
            pl.concat_str(
                [pl.col("season").cast(pl.String), pl.col("posteam")], separator="_"
            ).alias("team_season")
        )
        .sort(["team_season", "game_id"])
    )
    # Reuse the primary frame's team-games so the grouping is identical; only the
    # S3 columns are swapped for their competitive-state versions.
    swapped = (
        frame.drop(["n_valued", "epa_valued", "wpa_valued"])
        .join(
            competitive_games.drop(["season", "posteam"]),
            on=["game_id", "team_season"],
            how="left",
        )
        .with_columns(
            pl.col("n_valued").fill_null(0.0),
            pl.col("epa_valued").fill_null(0.0),
            pl.col("wpa_valued").fill_null(0.0),
        )
        .select(frame.columns)
        .sort(["team_season", "game_id"])
    )
    swapped_matrix, swapped_starts, _ = _power.to_dense(swapped)
    per_split = split_half_distribution(swapped_matrix, mask, swapped_starts, slope_c)
    s3_competitive = float(per_split[:, 3].mean())

    print(
        f"\n  S3 recomputed on plays with |score differential| <= 8 "
        f"({competitive.height:,} plays, {competitive.height / 451190:.0%} of the sample)"
    )
    print(f"    refit wpa_per_epa slope on this subset: {slope_c:.6f}")
    print(f"    split-half r = {s3_competitive:+.4f}  (full-sample S3 was +0.1800)")
    print(
        "    If this collapses toward zero, S3's persistence was game-state, not timing.\n"
        "    If it survives, timing skill is real inside competitive football."
    )

    return {
        "note": "EXPLORATORY, not pre-registered. Added after seeing Gate S-2's S3 result.",
        "corr_season_s3_vs_s0": r_s3_s0,
        "corr_season_s3_vs_mean_abs_score_diff": r_s3_close,
        "competitive_plays": int(competitive.height),
        "competitive_slope": slope_c,
        "s3_split_half_r_competitive_only": s3_competitive,
    }


def main() -> None:
    paths.ensure_data_dirs()
    pbp = load_pbp(PBP_SEASONS, columns=[*_power.COLUMNS, "score_differential"])
    frame, slope, design = _power.team_game_statistics(pbp)
    matrix, starts, sizes = _power.to_dense(frame)

    print(
        f"{len(starts)} team-seasons, {len(matrix):,} team-games, "
        f"{design['n_scrimmage_plays']:,} scrimmage plays, "
        f"wpa_per_epa slope {slope:.6f}"
    )

    rng = np.random.default_rng(RANDOM_SEED)
    mask = _power.split_masks(starts, sizes, len(matrix), rng, N_SPLITS)
    per_split = split_half_distribution(matrix, mask, starts, slope)

    rows = []
    for m, name in enumerate(MEASURES):
        draws = per_split[:, m]
        rows.append(
            {
                "measure": name,
                "split_half_r": float(draws.mean()),
                "r_p05": float(np.percentile(draws, 5)),
                "r_p95": float(np.percentile(draws, 95)),
            }
        )
    table = pl.DataFrame(rows)
    print("\n=== Split-half persistence, pooled within half (the gated statistic) ===")
    with pl.Config(tbl_cols=-1, fmt_str_lengths=30):
        print(table)

    # ---- Gate S-1 ---------------------------------------------------------
    s0 = float(per_split[:, MEASURES.index("S0_overall_epa")].mean())
    gate_s1 = bool(s0 > GATE_S1_THRESHOLD)
    print(f"\n{'=' * 72}\nGATE S-1 — positive control\n{'=' * 72}")
    print(
        f"  overall offensive EPA/play split-half r = {s0:+.4f} "
        f"vs null 99th pct {GATE_S1_THRESHOLD:.4f}: {'PASS' if gate_s1 else 'FAIL'}"
    )
    print("  (document 02 measured r = +0.519 on the core EPA differential)")
    if not gate_s1:
        print("\n  GATE S-1 FAILED — the harness is broken; nothing below is readable.")

    # ---- Gate S-2 and S-3 -------------------------------------------------
    power_lookup = {
        (row["measure"], row["target_true_r"]): row["power"]
        for row in json.load((paths.RESEARCH_OUTPUT_DIR / "10_sequencing_power.json").open())[
            "power"
        ]
    }

    print(f"\n{'=' * 72}\nGATE S-2 — does each sequencing measure persist?\n{'=' * 72}")
    verdicts = []
    for name in SEQUENCING_MEASURES:
        m = MEASURES.index(name)
        observed = float(per_split[:, m].mean())
        threshold = GATE_S2_THRESHOLDS[name]
        persists = bool(observed > threshold)
        power_at_reference = float(power_lookup[(name, REFERENCE_R)])
        interpretable = bool(power_at_reference >= MIN_POWER)

        if persists:
            verdict = "SKILL — already counted in core; nothing changes"
        elif interpretable:
            verdict = "LUCK — separate reported measure, no ledger rows"
        else:
            verdict = "UNRESOLVABLE — underpowered, no verdict"

        print(
            f"  {name:18s} r = {observed:+.4f}  threshold {threshold:.4f}  "
            f"{'PASS' if persists else 'FAIL'}   power at r={REFERENCE_R} "
            f"{power_at_reference:.2f} ({'S-3 pass' if interpretable else 'S-3 FAIL'})"
        )
        print(f"      -> {verdict}")
        verdicts.append(
            {
                "measure": name,
                "split_half_r": observed,
                "r_p05": float(np.percentile(per_split[:, m], 5)),
                "r_p95": float(np.percentile(per_split[:, m], 95)),
                "gate_s2_threshold": threshold,
                "gate_s2_pass": persists,
                "power_at_reference_r": power_at_reference,
                "gate_s3_pass": interpretable,
                "verdict": verdict,
            }
        )

    # ---- secondary: document 02's own estimator ---------------------------
    print(f"\n{'=' * 72}\nSECONDARY — document 02's per-game-averaged estimator\n{'=' * 72}")
    secondary = per_game_variant(frame, slope, np.random.default_rng(RANDOM_SEED + 1))
    for name, value in secondary.items():
        print(f"  {name:18s} r = {value:+.4f}")
    print(
        "  Reported for comparability with document 02's published table. Not gated —\n"
        "  document 08 section 2 records why pooling is the honest estimator here."
    )

    # ---- exploratory: is S3 timing skill, or just playing close games? ----
    exploratory = s3_confound_checks(pbp, frame, matrix, starts, mask, slope)

    results = {
        "n_team_seasons": int(len(starts)),
        "n_team_games": int(len(matrix)),
        "design": design,
        "primary_pooled": rows,
        "secondary_per_game_averaged": secondary,
        "exploratory_s3_confounds": exploratory,
        "gate_s1_positive_control": {
            "split_half_r": s0,
            "threshold": GATE_S1_THRESHOLD,
            "pass": gate_s1,
        },
        "gate_s2_verdicts": verdicts,
        "n_splits": N_SPLITS,
        "random_seed": RANDOM_SEED,
    }
    out = paths.RESEARCH_OUTPUT_DIR / "10_sequencing.json"
    with out.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
