"""Product layer, task 2 redesign — the redesigned placement meter's six gates.

Document 36 pre-registers this script's construction, its thresholds and its
order of operations, and was committed before this file existed. the maintainer chose
§8's fork option **(i)** on 2026-08-20: the full meter ships, gated on M-1, M-2
(carried forward), M-3 and M-6, with **M-4 descriptive** and its measured ceiling
printed in the same breath. Rung 3's disclosed inconsistency stays recorded, not
fixed — it is not the adopted null and correcting it would re-run M-2 under
amendment C-1 for a rung nothing reads.

    M-1  ->  M-2 (carry-forward assertion)  ->  M-3  ->  M-4  ->  M-5  ->  M-6

Each gate runs and reports on its own, which is why this script takes a gate
name rather than running the ladder end to end:

    uv run python research/52_placement_redesign.py --gate m1

Results accumulate in ``research/outputs/52_placement_redesign.json`` and the
scored team-games are cached beside it. Neither is committed —
``research/outputs/`` is gitignored, this script is the artifact and document
36's §11 is the record of the numbers.

The score, the profile fit, the cells and the four ladder rungs all live in
``nfl_simulator.placement``, under test. Nothing here chooses anything: every
constant below is quoted from document 36 §10.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_power = import_module("49_placement_power")
_redesign_power = import_module("51_placement_redesign_power")
_rematch = import_module("08_rematch")
_rematch_power = import_module("08_rematch_power")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.placement import (  # noqa: E402
    LADDER,
    N_BAND_DRAWS,
    band_from_draws,
    expected_profile,
    leave_one_out_rate,
    permutation_draws,
    pit_of,
    profile_shift,
    redesigned_cell_points,
    score_team_game,
)

# Document 36 §10.
RANDOM_SEED = 20260820

# Document 35's seed, and the reason it appears in a document-36 script: M-2's
# assertion is that each rung's coverage *reproduces* document 35 §11's to within
# 0.1 pp. An independent draw stream moves coverage by roughly 0.15 pp of pure
# resampling noise, which is larger than the tolerance, so a fresh seed would
# test the resampler rather than the carry-forward. The primary M-2 arm therefore
# re-uses document 35's stream, and a document-36-seeded arm is reported beside
# it as the sensitivity read. Disclosed in the report and in §11's register.
DOC35_SEED = 20260819

LEDGER_ARTIFACT = "dtw_ledger_v13.parquet"
GAMES_ARTIFACT = "dtw_games_v13.parquet"
INCUMBENT_SCORES = "placement_scores_v13.parquet"
SCORES_CACHE = "placement_redesign_scores_v13.parquet"
BAND_CACHE = "placement_redesign_band_{rung}.parquet"
RESULTS = "52_placement_redesign.json"

# Document 36 §2's reproduction targets, carried forward from document 35 §2.
EXPECTED_PLAYS = 343_543
EXPECTED_TEAM_GAMES = 5_522
EXPECTED_LEDGER_PLAYS = 5_541
EXPECTED_GAMES = 2_761

# Document 36 §10, every one committed before this script existed.
IDENTITY_TOLERANCE = 1e-9
SELF_BASELINING_TOLERANCE = 0.05
SELF_BASELINING_SAMPLE = 40
COVERAGE_REPRODUCTION_TOLERANCE_PP = 0.1
M3_THRESHOLD = 0.0636
M6_MARGIN = 0.010
N_SPLITS = 200
MIN_GAMES = 8

# Document 35 §11's coverage, the carry-forward assertion's target.
DOC35_COVERAGE = {
    "raw": 82.27,
    "raw_var_matched": 87.45,
    "down_stratified": 85.37,
    "down_stratified_var_matched": 89.10,
}
ADOPTED_RUNG = "raw_var_matched"

# Document 35 §10's ladder stretch factors, unchanged.
CELL_SCALES = (1.0892, 1.4116, 0.8237)
RUNG_SCALES = {
    "raw": (1.0, 1.0, 1.0),
    "raw_var_matched": CELL_SCALES,
    "down_stratified": (1.0, 1.0, 1.0),
    "down_stratified_var_matched": (1.0892, 1.0, 1.0),
}

# Document 36 §7's M-4 references, all simulated, all committed before any real
# correlation was computed. M-4 is descriptive: these bound what it can say, not
# whether it passes.
M4_PIPELINE_SAME_SEASON_MEAN = 0.1681
M4_PIPELINE_SAME_SEASON_SD = 0.0112
M4_BOUND_SAME_SEASON = 0.1857
M4_BOUND_OTHER_SEASONS = 0.1073
M4_DOC35_BOUND_SAME_SEASON = 0.1077
M4_DOC35_BOUND_OTHER_SEASONS = 0.1104
M4_INCUMBENT_VS_OTHER_SEASONS = 0.2098

# Document 33 §2a's three buckets, and document 33 §6's example games.
COIN_FLIP_LOW, COIN_FLIP_HIGH = 0.40, 0.60
EXAMPLE_GAMES = ("2018_05_GB_DET", "2021_14_LV_KC", "2025_17_DET_MIN")


# --------------------------------------------------------------------------
# the stream, the profile fit and the score — document 36 §2
# --------------------------------------------------------------------------


def team_game_table(plays: pl.DataFrame) -> pl.DataFrame:
    """One row per team-game: cell counts, cell EPA sums, and ``s0_loo``.

    ``s0_loo`` is the team's luck-priced offensive EPA per play over the *rest of
    its season*, so the game being scored never enters its own baseline.
    """
    meta = _power.team_game_arrays(plays)["meta"]
    s0 = leave_one_out_rate(
        meta["epa_all"].to_numpy().astype(float),
        meta["n_all"].to_numpy().astype(float),
        meta["team_season"].to_numpy(),
    )
    return meta.with_columns(pl.Series("s0_loo", s0))


def cell_matrices(table: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(n_all, counts, sums, s0)`` — counts and EPA sums as ``(rows, 3)`` matrices."""
    n_all = table["n_all"].to_numpy().astype(float)
    n_rz = table["n_rz"].to_numpy().astype(float)
    n_ld = table["n_ld"].to_numpy().astype(float)
    epa_all = table["epa_all"].to_numpy().astype(float)
    epa_rz = table["epa_rz"].to_numpy().astype(float)
    epa_ld = table["epa_ld"].to_numpy().astype(float)
    counts = np.column_stack([n_rz, n_ld, n_all - n_rz - n_ld])
    sums = np.column_stack([epa_rz, epa_ld, epa_all - epa_rz - epa_ld])
    return n_all, counts, sums, table["s0_loo"].to_numpy().astype(float)


def score_frame(table: pl.DataFrame) -> pl.DataFrame:
    """The redesigned score, its cells, the profile it was centred on, and the shift.

    The fold is the **franchise** — ``posteam`` across every season — so the
    coefficients that score a team were fitted without a single play that team
    ran. The incumbent is carried alongside because M-1's reduction check and
    M-2's carry-forward both read the gap between them.
    """
    n_all, counts, sums, s0 = cell_matrices(table)
    group = table["posteam"].to_numpy()
    mu = expected_profile(counts, sums, s0, group)
    points = redesigned_cell_points(n_all, counts, sums, mu)
    incumbent = redesigned_cell_points(n_all, counts, sums, np.zeros_like(mu))
    return table.with_columns(
        pl.Series("mu_rz", mu[:, 0]),
        pl.Series("mu_ld", mu[:, 1]),
        pl.Series("mu_other", mu[:, 2]),
        pl.Series("red_zone", points[:, 0]),
        pl.Series("late_down", points[:, 1]),
        pl.Series("other", points[:, 2]),
        pl.Series("score", points[:, 0] + points[:, 1]),
        pl.Series("identity_residual", points.sum(axis=1)),
        pl.Series("incumbent_score", incumbent[:, 0] + incumbent[:, 1]),
        pl.Series("profile_shift", profile_shift(counts, mu, n_all)),
    )


def build_scores() -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """``(plays, scored team-games, stream reconciliation)``, freshly computed."""
    plays, reconciliation = _power.load_luck_priced_plays()
    scored = score_frame(team_game_table(plays))
    scored.write_parquet(paths.RESEARCH_OUTPUT_DIR / SCORES_CACHE)
    return plays, scored, reconciliation


def cached_scores() -> pl.DataFrame:
    path = paths.RESEARCH_OUTPUT_DIR / SCORES_CACHE
    if not path.exists():
        build_scores()
    return pl.read_parquet(path)


def differentials(scores: pl.DataFrame, column: str = "score") -> pl.DataFrame:
    """Home minus away, one row per game — the meter's headline."""
    frame = scores.select("game_id", "season", "is_home", pl.col(column).alias("value"))
    home = frame.filter("is_home").select("game_id", "season", pl.col("value").alias("home_score"))
    away = frame.filter(~pl.col("is_home")).select("game_id", pl.col("value").alias("away_score"))
    return home.join(away, on="game_id", how="inner").with_columns(
        (pl.col("home_score") - pl.col("away_score")).alias("differential")
    )


def save(section: str, payload: dict) -> None:
    path = paths.RESEARCH_OUTPUT_DIR / RESULTS
    existing = json.loads(path.read_text()) if path.exists() else {"random_seed": RANDOM_SEED}
    existing[section] = payload
    path.write_text(json.dumps(existing, indent=2, default=float))
    print(f"\nwrote {section} to {path}")


def _report(checks: dict) -> bool:
    for name, check in checks.items():
        print(f"  {'PASS' if check['pass'] else 'FAIL'}  {name}")
        for key, value in check.items():
            if key != "pass":
                print(f"          {key}: {value}")
    return all(check["pass"] for check in checks.values())


# --------------------------------------------------------------------------
# M-1 — the identities, document 36 §4
# --------------------------------------------------------------------------


def _fitted_without_game(
    mean_cell: np.ndarray,
    weight: np.ndarray,
    s0: np.ndarray,
    group: np.ndarray,
    keep: np.ndarray,
    row_s0: float,
    row_group,
) -> float:
    """One row's fitted value from a sample with a whole game removed.

    Written out longhand rather than routed through the production fit, because
    what it checks is the production fit: a second implementation of the same
    weighted least squares, on a deliberately different sample.
    """
    rows = keep & (group != row_group)
    w, x, y = weight[rows], s0[rows], mean_cell[rows]
    s_w, s_wx, s_wxx = w.sum(), (w * x).sum(), (w * x * x).sum()
    s_y, s_yx = (w * y).sum(), (w * y * x).sum()
    slope = (s_w * s_yx - s_wx * s_y) / (s_w * s_wxx - s_wx * s_wx)
    return float((s_y - slope * s_wx) / s_w + slope * row_s0)


def gate_m1() -> dict:
    """Document 36 §4. Arithmetic, not statistics; a failure stops the round."""
    print(f"{'=' * 72}\nM-1 — the identities\n{'=' * 72}")
    plays, scored, reconciliation = build_scores()
    for key, value in reconciliation.items():
        print(f"  {key:<34} {value}")

    stream_matches = {
        "plays": reconciliation["n_plays_in_scope"] == EXPECTED_PLAYS,
        "games": int(plays["game_id"].n_unique()) == EXPECTED_GAMES,
        "ledger_plays": reconciliation["n_plays_carrying_a_ledger_row"] == EXPECTED_LEDGER_PLAYS,
        "team_games": scored.height == EXPECTED_TEAM_GAMES,
    }

    n_all, counts, sums, s0 = cell_matrices(scored)
    group = scored["posteam"].to_numpy()
    mean_cell = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)

    residual = float(np.abs(scored["identity_residual"].to_numpy()).max())
    empty_rz = scored.filter(pl.col("n_rz") == 0)
    empty_ld = scored.filter(pl.col("n_ld") == 0)

    # --- ledger reconciliation, document 35 §4's check unchanged -----------
    ledger = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / LEDGER_ARTIFACT)
    repriced = plays.filter(pl.col("luck_epa_pos") != 0.0).select("game_id", "play_id")
    ledger_in_filter = (
        ledger.join(plays.select("game_id", "play_id"), on=["game_id", "play_id"], how="inner")
        .group_by(["game_id", "play_id"])
        .agg(pl.col("luck_epa").sum().alias("luck_epa_home"))
        .filter(pl.col("luck_epa_home") != 0.0)
    )
    same_rows = repriced.sort(["game_id", "play_id"]).equals(
        ledger_in_filter.select("game_id", "play_id").sort(["game_id", "play_id"])
    )
    joined = plays.join(ledger_in_filter, on=["game_id", "play_id"], how="inner")
    sum_gap = float(joined["luck_epa_pos"].abs().sum() - joined["luck_epa_home"].abs().sum())

    # --- the differential recomputes ---------------------------------------
    diffs = differentials(scored)
    differential_gap = float(
        np.abs(
            (diffs["home_score"] - diffs["away_score"]).to_numpy()
            - diffs["differential"].to_numpy()
        ).max()
    )

    # --- new: the reduction to document 35's score -------------------------
    zero_profile = redesigned_cell_points(n_all, counts, sums, np.zeros_like(counts))
    doc35 = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / INCUMBENT_SCORES).select(
        "game_id", "posteam", pl.col("score").alias("doc35_score")
    )
    reduction = (
        scored.select("game_id", "posteam")
        .with_columns(pl.Series("reduced", zero_profile[:, 0] + zero_profile[:, 1]))
        .join(doc35, on=["game_id", "posteam"], how="inner")
    )
    reduction_gap = float(
        np.abs(reduction["reduced"].to_numpy() - reduction["doc35_score"].to_numpy()).max()
    )

    # --- new: no self-baselining -------------------------------------------
    rng = np.random.default_rng(RANDOM_SEED)
    sample = rng.choice(scored.height, size=SELF_BASELINING_SAMPLE, replace=False)
    game_ids = scored["game_id"].to_numpy()
    score = scored["score"].to_numpy()
    worst_move = 0.0
    for row in sample:
        keep = game_ids != game_ids[row]
        mu_row = np.array(
            [
                _fitted_without_game(
                    mean_cell[:, c], counts[:, c], s0, group, keep, float(s0[row]), group[row]
                )
                for c in range(3)
            ]
        )
        alt = redesigned_cell_points(
            n_all[row : row + 1], counts[row : row + 1], sums[row : row + 1], mu_row[None, :]
        )
        worst_move = max(worst_move, abs(float(alt[0, 0] + alt[0, 1]) - float(score[row])))

    checks = {
        "three_cells_sum_to_zero": {
            "worst_residual_points": residual,
            "tolerance": IDENTITY_TOLERANCE,
            "pass": bool(residual <= IDENTITY_TOLERANCE),
        },
        "empty_red_zone_cell_is_exactly_zero": {
            "n_team_games": int(empty_rz.height),
            "share": float(empty_rz.height / scored.height),
            "pass": bool((empty_rz["red_zone"] == 0.0).all()),
        },
        "empty_late_down_cell_is_exactly_zero": {
            "n_team_games": int(empty_ld.height),
            "pass": bool((empty_ld["late_down"] == 0.0).all()),
        },
        "ledger_reconciliation": {
            "n_repriced_plays": int(repriced.height),
            "rows_are_the_ledger_rows_surviving_the_filter": bool(same_rows),
            "abs_sum_gap_epa": sum_gap,
            "pass": bool(same_rows and abs(sum_gap) <= IDENTITY_TOLERANCE),
        },
        "differential_recomputes": {
            "worst_gap_points": differential_gap,
            "n_games": int(diffs.height),
            "pass": bool(differential_gap <= IDENTITY_TOLERANCE),
        },
        "reduction_to_document_35": {
            "n_team_games_compared": int(reduction.height),
            "worst_gap_points": reduction_gap,
            "tolerance": IDENTITY_TOLERANCE,
            "pass": bool(
                reduction.height == EXPECTED_TEAM_GAMES and reduction_gap <= IDENTITY_TOLERANCE
            ),
        },
        "no_self_baselining": {
            "sampled_team_games": int(len(sample)),
            "worst_abs_score_move_points": worst_move,
            "tolerance": SELF_BASELINING_TOLERANCE,
            "pass": bool(worst_move <= SELF_BASELINING_TOLERANCE),
        },
        "stream_reproduces_the_pre_registration": {
            "detail": stream_matches,
            "pass": all(stream_matches.values()),
        },
    }

    print()
    verdict = _report(checks)
    print(f"\n  M-1 VERDICT: {'PASS' if verdict else 'FAIL'}")
    return {
        "reconciliation": reconciliation,
        "n_team_games": int(scored.height),
        "checks": checks,
        "verdict": "pass" if verdict else "fail",
    }


# --------------------------------------------------------------------------
# M-2 — the band, carried forward by proof and checked here
# --------------------------------------------------------------------------

KS_REFERENCE_P95 = {
    "raw": 0.0198,
    "down_stratified": 0.0305,
    "down_stratified_var_matched": 0.0399,
    "raw_var_matched": 0.0414,
}


def ks_statistic(pit: np.ndarray) -> float:
    u = np.sort(pit)
    n = len(u)
    grid = np.arange(1, n + 1) / n
    return float(max(np.max(grid - u), np.max(u - (grid - 1.0 / n))))


def rung_pit(plays: pl.DataFrame, shifts: dict, rung: str, seed: int) -> pl.DataFrame:
    """Each team-game's redesigned band under one rung, and its PIT inside it.

    Document 36 §6: with the three cell sizes held fixed — which every rung does —
    the profile a draw subtracts is the same whichever plays land where, so the
    redesigned score and every null draw are the incumbent's minus the same
    per-team-game constant ``C``. That identity was checked numerically to
    1 x 10^-14 points on all four rungs by ``research/51_placement_redesign_power.py``,
    recomputing both sides the slow way from a shared assignment. It is applied
    here rather than re-derived: the band moves with the score, the PIT is a rank
    inside the draws, and a rank does not move at all.
    """
    rng = np.random.default_rng(seed)
    scales = RUNG_SCALES[rung]
    rows = []
    for (game_id, posteam), block in plays.group_by(["game_id", "posteam"], maintain_order=True):
        epa = block["epa_priced"].to_numpy().astype(float)
        cell = block["cell"].to_numpy().astype(int)
        down = block["down"].to_numpy().astype(int)
        shift = shifts[(game_id, posteam)]
        realized = score_team_game(epa, cell).score - shift
        draws = permutation_draws(rung, epa, cell, down, N_BAND_DRAWS, rng, scales) - shift
        low, high = band_from_draws(draws)
        rows.append(
            {
                "game_id": game_id,
                "posteam": posteam,
                "score": realized,
                "band_low": low,
                "band_high": high,
                "pit": pit_of(realized, draws),
                "inside": bool(low <= realized <= high),
            }
        )
    return pl.DataFrame(rows)


def gate_m2() -> dict:
    """Document 36 §6. A carry-forward that is not checked in the ship is an assumption."""
    print(f"{'=' * 72}\nM-2 — the band, carried forward\n{'=' * 72}")
    plays, _ = _power.load_luck_priced_plays()
    scored = cached_scores()
    shifts = {
        (row["game_id"], row["posteam"]): row["profile_shift"]
        for row in scored.select("game_id", "posteam", "profile_shift").iter_rows(named=True)
    }
    print(
        f"  {N_BAND_DRAWS} draws per team-game, 89% equal-tailed; each rung's coverage must "
        f"reproduce document 35 §11's within {COVERAGE_REPRODUCTION_TOLERANCE_PP} pp"
    )

    results: dict[str, dict] = {}
    for index, rung in enumerate(LADDER):
        table = rung_pit(plays, shifts, rung, DOC35_SEED + index)
        table.write_parquet(paths.RESEARCH_OUTPUT_DIR / BAND_CACHE.format(rung=rung))
        pit = table["pit"].to_numpy().astype(float)
        coverage = float(table["inside"].mean()) * 100.0
        gap = abs(coverage - DOC35_COVERAGE[rung])
        ks = ks_statistic(pit)

        sensitivity = rung_pit(plays, shifts, rung, RANDOM_SEED + index)
        fresh_coverage = float(sensitivity["inside"].mean()) * 100.0

        results[rung] = {
            "coverage_pct": coverage,
            "document_35_coverage_pct": DOC35_COVERAGE[rung],
            "gap_pp": gap,
            "reproduces": bool(gap <= COVERAGE_REPRODUCTION_TOLERANCE_PP),
            "fresh_seed_coverage_pct": fresh_coverage,
            "fresh_seed_gap_pp": abs(fresh_coverage - DOC35_COVERAGE[rung]),
            "ks": ks,
            "ks_reference_p95": KS_REFERENCE_P95[rung],
            "ks_within_own_null": bool(ks <= KS_REFERENCE_P95[rung]),
            "pit_mean": float(pit.mean()),
            "median_band_width_points": float((table["band_high"] - table["band_low"]).median()),
            "n_team_games": int(table.height),
        }
        print(
            f"\n  {rung:<30} coverage {coverage:6.2f}%  vs document 35's "
            f"{DOC35_COVERAGE[rung]:.2f}%  gap {gap:.3f} pp  "
            f"[{'reproduces' if results[rung]['reproduces'] else 'FAILS TO REPRODUCE'}]"
        )
        print(
            f"  {'':<30} fresh-seed arm {fresh_coverage:6.2f}%   KS {ks:.4f}   "
            f"median band width {results[rung]['median_band_width_points']:.2f} pts"
        )

    verdict = all(block["reproduces"] for block in results.values())
    print(f"\n  adopted rung: {ADOPTED_RUNG} (the maintainer, 2026-08-20), coverage ")
    print(f"  {results[ADOPTED_RUNG]['coverage_pct']:.2f}% of an 89% band, tolerance [87.0, 91.0]")
    print(f"\n  M-2 VERDICT: {'PASS' if verdict else 'FAIL'}")
    print(
        "  Read as 'not detectably miscalibrated at this tolerance', never as "
        "'exactly calibrated' — rung 4 passes its own design truth by one binomial SD."
    )
    return {
        "primary_seed": DOC35_SEED,
        "sensitivity_seed": RANDOM_SEED,
        "tolerance_pp": COVERAGE_REPRODUCTION_TOLERANCE_PP,
        "adopted": ADOPTED_RUNG,
        "rungs": results,
        "verdict": "pass" if verdict else "fail",
    }


# --------------------------------------------------------------------------
# M-3 — is the shipped score still luck?
# --------------------------------------------------------------------------


def gate_m3() -> dict:
    """Document 36 §7. Document 08's split-half machinery, threshold r > 0.0636.

    The machinery is imported from the power script rather than re-implemented,
    because a gate whose statistic differs from the one its threshold was
    simulated on is not the gate that was pre-registered.
    """
    print(f"{'=' * 72}\nM-3 — the luck licence\n{'=' * 72}")
    scored = cached_scores()
    rows, blocks = _power.season_blocks(scored.select("team_season", "game_id"))
    values = scored["score"].to_numpy().astype(float)[rows]
    rng = np.random.default_rng(RANDOM_SEED)
    mask = _power.split_masks(blocks, len(values), rng)

    r = _power.split_half_r(values, mask, blocks)
    per_split = np.array(
        [_power.split_half_r(values, mask[s : s + 1], blocks) for s in range(mask.shape[0])]
    )

    passes = r <= M3_THRESHOLD
    reading = "luck" if passes else "a team property"
    print(f"  {len(blocks)} team-seasons, {len(values)} team-games at the {MIN_GAMES}-game floor")
    print(f"  {N_SPLITS} within-season splits, half statistic = mean placement points per game")
    print(
        f"\n  split-half r = {r:+.4f}   (5th-95th across splits "
        f"{np.quantile(per_split, 0.05):+.4f} to {np.quantile(per_split, 0.95):+.4f})"
    )
    print(f"  pre-registered threshold: r > {M3_THRESHOLD:.4f} means NOT luck")
    print(f"\n  M-3 VERDICT: {'PASS' if passes else 'FAIL'} — placement is {reading}")
    print("  Power 0.916 at r = 0.12, so this is pass/fail rather than descriptive (§7).")
    return {
        "n_team_seasons": int(len(blocks)),
        "n_team_games": int(len(values)),
        "split_half_r": r,
        "per_split_p05": float(np.quantile(per_split, 0.05)),
        "per_split_p95": float(np.quantile(per_split, 0.95)),
        "threshold": M3_THRESHOLD,
        "power_at_reference_r": 0.916,
        "verdict": "pass" if passes else "fail",
        "reading": reading,
    }


# --------------------------------------------------------------------------
# M-4 — does the baseline leak skill?  Descriptive, with its ceiling
# --------------------------------------------------------------------------


def gate_m4() -> dict:
    """Document 36 §7. **Descriptive, not pass/fail** — the Gate S-3 pattern.

    Two facts committed before any real correlation was computed set what this
    gate can say. The score and the same-season covariate are built from the same
    plays, which is worth +0.168 on its own; and the construction removes any
    leak that is a function of quality, so the gate's power against every leak
    shape tried sits at the false-alarm rate. It verifies that the correction was
    implemented. It licenses nothing about leaks.
    """
    print(f"{'=' * 72}\nM-4 — skill preservation (DESCRIPTIVE)\n{'=' * 72}")
    scored = cached_scores()
    season = _redesign_power.season_table(scored).filter(pl.col("games") >= MIN_GAMES)

    same = season["s0_quality"].to_numpy().astype(float)
    other = season["s0_other_seasons"].to_numpy().astype(float)
    mean_score = season["mean_score"].to_numpy().astype(float)
    sd_score = season["sd_score"].to_numpy().astype(float)
    incumbent_mean = (
        scored.group_by("team_season", maintain_order=True)
        .agg(pl.col("incumbent_score").mean())["incumbent_score"]
        .to_numpy()
    )

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.corrcoef(a, b)[0, 1])

    r_same = corr(mean_score, same)
    r_other = corr(mean_score, other)
    z_same = (r_same - M4_PIPELINE_SAME_SEASON_MEAN) / M4_PIPELINE_SAME_SEASON_SD

    # The expected-mix arm — the literal reading of redesign avenue (a) — reported
    # at every gate, never shipped. Document 36 §5's arithmetic disqualifies it:
    # the reweight is unbounded as a cell empties.
    n_all, counts, sums, s0 = cell_matrices(scored)
    group = scored["posteam"].to_numpy()
    mu = np.column_stack(
        [scored["mu_rz"].to_numpy(), scored["mu_ld"].to_numpy(), scored["mu_other"].to_numpy()]
    )
    q = _redesign_power.expected_shares(counts, n_all, s0, group)
    mix = _redesign_power.expected_mix_cell_points(n_all, counts, sums, mu, q)[:, :2].sum(axis=1)
    mix_season = (
        scored.with_columns(pl.Series("mix_score", mix))
        .group_by("team_season", maintain_order=True)
        .agg(pl.col("mix_score").mean())["mix_score"]
        .to_numpy()
    )

    print(f"  {season.height} team-seasons; quality SD {same.std(ddof=1):.4f} EPA per play")
    print("\n  corr(team-season mean placement points, offensive quality):")
    print(
        f"    same-season quality   {r_same:+.4f}   vs the pipeline reference "
        f"{M4_PIPELINE_SAME_SEASON_MEAN:+.4f} +- {M4_PIPELINE_SAME_SEASON_SD:.4f}  "
        f"({z_same:+.2f} SD)"
    )
    print(
        f"    other-seasons quality {r_other:+.4f}   vs its bound "
        f"{M4_BOUND_OTHER_SEASONS:.4f}  "
        f"[{'inside' if abs(r_other) <= M4_BOUND_OTHER_SEASONS else 'outside'}]"
    )
    print("\n  document 35 §7's independent nulls, reported so the reference change is visible:")
    print(
        f"    same season bound {M4_DOC35_BOUND_SAME_SEASON:.4f}   "
        f"other seasons bound {M4_DOC35_BOUND_OTHER_SEASONS:.4f}"
    )
    print("\n  the design this round replaces, read against the same clean covariate:")
    print(
        f"    incumbent vs other-seasons quality {corr(incumbent_mean, other):+.4f}   "
        f"(document 36 §7 committed {M4_INCUMBENT_VS_OTHER_SEASONS:+.4f})"
    )
    print(f"    incumbent vs same-season quality   {corr(incumbent_mean, same):+.4f}")
    print(f"\n  the reported expected-mix arm: same {corr(mix_season, same):+.4f}, ")
    print(f"    other {corr(mix_season, other):+.4f}, max |score| {np.abs(mix).max():.2f} points")
    print(
        f"\n  secondary, descriptive, no rule: corr(team-season spread of scores, quality) "
        f"= {corr(sd_score, same):+.4f}"
    )
    print(
        "\n  M-4 IS DESCRIPTIVE. Its power against a quality-aligned leak of any magnitude\n"
        "  tried is at the false-alarm rate (0.000-0.107, document 36 §7). It verifies that\n"
        "  the correction was implemented; it licenses nothing about leaks."
    )

    # `season_table` carries the score but not its cells, so the per-cell read —
    # the channel decomposition document 35 §11 used to locate the incumbent's
    # leak — is aggregated here on the same team-season key.
    cells = (
        scored.group_by("team_season", maintain_order=True)
        .agg(pl.col("red_zone").mean(), pl.col("late_down").mean())
        .join(season.select("team_season"), on="team_season", how="semi")
    )
    channels = {
        "red_zone_cell": corr(cells["red_zone"].to_numpy().astype(float), same),
        "late_down_cell": corr(cells["late_down"].to_numpy().astype(float), same),
    }
    print("\n  where the incumbent's leak lived, re-read on the redesigned cells:")
    for name, value in channels.items():
        print(f"    {name:<24} {value:+.4f}")

    return {
        "n_team_seasons": int(season.height),
        "gate_type": "descriptive",
        "corr_same_season": r_same,
        "pipeline_reference_same_season": [
            M4_PIPELINE_SAME_SEASON_MEAN,
            M4_PIPELINE_SAME_SEASON_SD,
        ],
        "z_against_pipeline_reference": float(z_same),
        "bound_same_season": M4_BOUND_SAME_SEASON,
        "corr_other_seasons": r_other,
        "bound_other_seasons": M4_BOUND_OTHER_SEASONS,
        "inside_other_seasons_bound": bool(abs(r_other) <= M4_BOUND_OTHER_SEASONS),
        "doc35_independent_bounds": [
            M4_DOC35_BOUND_SAME_SEASON,
            M4_DOC35_BOUND_OTHER_SEASONS,
        ],
        "incumbent_vs_other_seasons": corr(incumbent_mean, other),
        "incumbent_vs_same_season": corr(incumbent_mean, same),
        "expected_mix_arm": {
            "corr_same_season": corr(mix_season, same),
            "corr_other_seasons": corr(mix_season, other),
            "max_abs_score": float(np.abs(mix).max()),
            "score_sd": float(mix.std(ddof=1)),
        },
        "secondary_corr_dispersion_quality": corr(sd_score, same),
        "cell_channels_vs_quality": channels,
        "ceiling_sentence": (
            "descriptive; power against a quality-aligned leak of any magnitude tried "
            "is at the false-alarm rate"
        ),
    }


# --------------------------------------------------------------------------
# M-5 — magnitude.  Report, no threshold (Gate C convention)
# --------------------------------------------------------------------------


def gate_m5() -> dict:
    """Document 36 §7. How often does the meter matter, and how much of it is fitted?"""
    print(f"{'=' * 72}\nM-5 — magnitude (REPORT, no threshold)\n{'=' * 72}")
    scored = cached_scores()
    score = scored["score"].to_numpy().astype(float)
    shift = scored["profile_shift"].to_numpy().astype(float)

    diffs = differentials(scored)
    games = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / GAMES_ARTIFACT)
    joined = diffs.join(
        games.select("game_id", "dtw_home", "actual_margin", "deserved_margin"),
        on="game_id",
        how="inner",
    )
    dtw = joined["dtw_home"].to_numpy().astype(float)
    actual = joined["actual_margin"].to_numpy().astype(float)
    differential = joined["differential"].to_numpy().astype(float)

    # Document 33 §2a's three buckets, definitions copied from `48_magnitude_audit.py`.
    coin_flip = (dtw >= COIN_FLIP_LOW) & (dtw <= COIN_FLIP_HIGH)
    clear_flip = ((actual > 0) != (dtw > 0.5)) & (actual != 0) & ~coin_flip

    band = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / BAND_CACHE.format(rung=ADOPTED_RUNG))
    outside = ~band["inside"].to_numpy()

    per_season = (
        scored.group_by("season", maintain_order=True)
        .agg(
            pl.len().alias("team_games"),
            pl.col("score").mean().alias("mean_score"),
            pl.col("score").std(ddof=1).alias("sd_score"),
            pl.col("score").abs().median().alias("median_abs_score"),
        )
        .sort("season")
    )

    magnitude = {
        "n_team_games": int(scored.height),
        "score_mean": float(score.mean()),
        "score_sd": float(score.std(ddof=1)),
        "median_abs_score": float(np.median(np.abs(score))),
        "q95_abs_score": float(np.quantile(np.abs(score), 0.95)),
        "max_abs_score": float(np.abs(score).max()),
        "differential_sd": float(differential.std(ddof=1)),
        "median_abs_differential": float(np.median(np.abs(differential))),
        "share_team_games_outside_their_own_band": float(outside.mean()),
        "cells": {
            "red_zone_sd": float(scored["red_zone"].to_numpy().std(ddof=1)),
            "late_down_sd": float(scored["late_down"].to_numpy().std(ddof=1)),
            "corr_red_zone_late_down": float(
                np.corrcoef(scored["red_zone"].to_numpy(), scored["late_down"].to_numpy())[0, 1]
            ),
        },
        "profile_shift_points": {
            "mean": float(shift.mean()),
            "sd": float(shift.std(ddof=1)),
            "min": float(shift.min()),
            "max": float(shift.max()),
            "share_of_score_sd": float(shift.std(ddof=1) / score.std(ddof=1)),
        },
        "corr_differential_dtw": float(np.corrcoef(differential, dtw)[0, 1]),
        "corr_abs_differential_abs_deserved_margin": float(
            np.corrcoef(np.abs(differential), np.abs(joined["deserved_margin"].to_numpy()))[0, 1]
        ),
        "flip_overlap": {
            "n_clear_flips": int(clear_flip.sum()),
            "mean_abs_differential_in_clear_flips": float(np.abs(differential[clear_flip]).mean()),
            "mean_abs_differential_elsewhere": float(np.abs(differential[~clear_flip]).mean()),
            "share_of_clear_flips_with_differential_over_3_points": float(
                np.mean(np.abs(differential[clear_flip]) > 3.0)
            ),
        },
        "per_season": per_season.to_dicts(),
    }

    print(
        f"  score: mean {magnitude['score_mean']:+.4f}  SD {magnitude['score_sd']:.3f}  "
        f"median|.| {magnitude['median_abs_score']:.3f}  q95|.| {magnitude['q95_abs_score']:.3f}  "
        f"max|.| {magnitude['max_abs_score']:.3f} points"
    )
    print(
        f"  differential: SD {magnitude['differential_sd']:.3f}  median|.| "
        f"{magnitude['median_abs_differential']:.3f} points over {joined.height} games"
    )
    print(
        f"  team-games outside their own {ADOPTED_RUNG} band: "
        f"{magnitude['share_team_games_outside_their_own_band'] * 100:.2f}%"
    )
    print(
        f"  fitted correction: profile shift mean {shift.mean():+.3f}, SD {shift.std(ddof=1):.3f} "
        f"points, {magnitude['profile_shift_points']['share_of_score_sd'] * 100:.1f}% of the "
        f"score's own SD"
    )
    print(f"  corr(placement differential, DTW%) = {magnitude['corr_differential_dtw']:+.4f}")
    print(
        f"  {int(clear_flip.sum())} clear-flip games (document 33 §2a): mean |placement "
        f"differential| {magnitude['flip_overlap']['mean_abs_differential_in_clear_flips']:.2f} "
        f"vs {magnitude['flip_overlap']['mean_abs_differential_elsewhere']:.2f} points elsewhere"
    )

    print("\n  example games (document 33 §6):")
    examples = {}
    for game_id in EXAMPLE_GAMES:
        row = joined.filter(pl.col("game_id") == game_id)
        teams = scored.filter(pl.col("game_id") == game_id).sort("is_home", descending=True)
        if row.height == 0:
            continue
        examples[game_id] = {
            "differential": float(row["differential"][0]),
            "dtw_home": float(row["dtw_home"][0]),
            "actual_margin": float(row["actual_margin"][0]),
            "deserved_margin": float(row["deserved_margin"][0]),
            "teams": teams.select(
                "posteam", "is_home", "red_zone", "late_down", "score", "profile_shift"
            ).to_dicts(),
        }
        print(
            f"    {game_id}  differential {row['differential'][0]:+.2f} pts   "
            f"DTW% {row['dtw_home'][0]:.3f}   actual {row['actual_margin'][0]:+.0f}   "
            f"deserved {row['deserved_margin'][0]:+.2f}"
        )
        for team in examples[game_id]["teams"]:
            print(
                f"      {team['posteam']:<4} {'home' if team['is_home'] else 'away'}  "
                f"red zone {team['red_zone']:+6.2f}  late down {team['late_down']:+6.2f}  "
                f"score {team['score']:+6.2f}  (correction {team['profile_shift']:+.2f})"
            )
    magnitude["examples"] = examples

    print("\n  per-season stability:")
    for row in magnitude["per_season"]:
        print(
            f"    {row['season']}  mean {row['mean_score']:+.3f}  SD {row['sd_score']:.3f}  "
            f"median|.| {row['median_abs_score']:.3f}"
        )
    print("\n  M-5 is a report. No threshold, nothing passes or fails (Gate C convention).")
    return magnitude


# --------------------------------------------------------------------------
# M-6 — does subtracting the meter lose information?
# --------------------------------------------------------------------------


def gate_m6() -> dict:
    """Document 36 §7. Document 06's harness, non-inferiority margin +0.010.

    Incumbent predictor is game 1's actual margin; challenger is game 1's deserved
    margin **minus the placement differential**. Team A is game 1's host by
    ``rematch_pairs``'s construction, so the home-minus-away differential is
    already oriented to A and needs no re-signing.
    """
    print(f"{'=' * 72}\nM-6 — the rematch harness\n{'=' * 72}")
    scored = cached_scores()
    diffs = differentials(scored).select("game_id", "differential")

    pairs = _rematch.build_pairs(GAMES_ARTIFACT).join(diffs, on="game_id", how="inner")
    actual = pairs["margin_g1_a"].to_numpy().astype(float)
    deserved = pairs["deserved_margin"].to_numpy().astype(float)
    placement = pairs["differential"].to_numpy().astype(float)
    y = (pairs["margin_g2_a"].to_numpy() > 0).astype(float)
    a_home = pairs["a_home_g2"].to_numpy().astype(float)

    fold_rng = np.random.default_rng(_rematch.RANDOM_SEED)
    folds = fold_rng.permutation(pairs.height) % _rematch.N_FOLDS

    per_pair = _rematch_power.paired_log_loss_diff(actual, deserved - placement, y, a_home, folds)
    mean, se, _ = _rematch_power.decision(per_pair)
    upper = mean + 1.96 * se
    passes = bool(_rematch_power.passes_noninferiority(mean, se))

    # Reported beside it: the deserved margin on its own, so the placement
    # subtraction's contribution is separable from the adjudication's.
    base_pair = _rematch_power.paired_log_loss_diff(actual, deserved, y, a_home, folds)
    base_mean, base_se, _ = _rematch_power.decision(base_pair)

    print(f"  {pairs.height} rematch pairs, {_rematch.N_FOLDS} folds, seed {_rematch.RANDOM_SEED}")
    print(
        f"  challenger = deserved margin - placement differential (SD {placement.std(ddof=1):.3f})"
    )
    print(f"\n  mean delta log loss {mean:+.5f}   SE {se:.5f}   95% upper {upper:+.5f}")
    print(f"  non-inferiority margin {M6_MARGIN:+.3f}")
    print(f"\n  M-6 VERDICT: {'PASS' if passes else 'FAIL'}")
    print(
        f"  reported beside it: deserved margin alone gives {base_mean:+.5f} "
        f"(95% upper {base_mean + 1.96 * base_se:+.5f})"
    )
    print(
        "  Power to pass 0.965; when the subtraction is pure harm at this magnitude the\n"
        "  gate still passes 26% of the time. A pass is evidence, not proof; a failure is\n"
        "  the strong signal (document 36 §7)."
    )
    return {
        "n_pairs": int(pairs.height),
        "placement_differential_sd": float(placement.std(ddof=1)),
        "mean_delta_log_loss": float(mean),
        "se": float(se),
        "ci95_upper": float(upper),
        "margin": M6_MARGIN,
        "verdict": "pass" if passes else "fail",
        "power_to_pass": 0.965,
        "false_pass_rate": 0.260,
        "deserved_margin_alone": {
            "mean_delta_log_loss": float(base_mean),
            "ci95_upper": float(base_mean + 1.96 * base_se),
        },
    }


GATES = {"m1": gate_m1, "m2": gate_m2, "m3": gate_m3, "m4": gate_m4, "m5": gate_m5, "m6": gate_m6}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True, choices=sorted(GATES))
    args = parser.parse_args()
    paths.ensure_data_dirs()
    save(args.gate, GATES[args.gate]())


if __name__ == "__main__":
    main()
