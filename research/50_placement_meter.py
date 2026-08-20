"""Product layer, task 2, steps 2-3 — the placement meter's six gates.

Runs the gates document 35 pre-registered, in the order document 35 §8 commits,
against the thresholds it committed **before** any of this was computed:

    M-1  ->  M-2  ->  M-3  ->  M-4  ->  M-5  ->  M-6

Each gate is run and reported on its own, which is why this script takes a gate
name rather than running the ladder end to end:

    uv run python research/50_placement_meter.py --gate m1

Results accumulate in ``research/outputs/50_placement_meter.json``; the scored
team-games are cached beside it so the later gates do not re-derive the stream.
Neither is committed — ``research/outputs/`` is gitignored, the script is the
artifact and document 35's §11 is the record of the numbers.

The score, the cells, the pricing and the four ladder rungs all live in
``nfl_simulator.placement``, under test. Nothing here chooses anything: every
constant below is quoted from document 35 §10.
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

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.ingest import load_pbp  # noqa: E402
from nfl_simulator.placement import (  # noqa: E402
    LADDER,
    N_BAND_DRAWS,
    assign_cells,
    band_from_draws,
    permutation_draws,
    pit_of,
    price_plays,
    score_team_game,
)

RANDOM_SEED = 20260819

LEDGER_ARTIFACT = "dtw_ledger_v13.parquet"
GAMES_ARTIFACT = "dtw_games_v13.parquet"
SCORES_CACHE = "placement_scores_v13.parquet"
RESULTS = "50_placement_meter.json"

# Document 35 §10, every one of them committed before this script existed.
IDENTITY_TOLERANCE = 1e-9
COVERAGE_LOW, COVERAGE_HIGH = 0.870, 0.910
M3_THRESHOLD = 0.0671
M4_BOUND = 0.1065
M6_MARGIN = 0.010
N_SPLITS = 200
MIN_GAMES = 8

# Document 35 §2's reproduction targets. The pre-registration says this script
# reproduces its input stream "or it is wrong" — so it checks.
EXPECTED_PLAYS = 343_543
EXPECTED_TEAM_GAMES = 5_522
EXPECTED_LEDGER_PLAYS = 5_541
EXPECTED_GAMES = 2_761

# Document 35 §5's ladder, in adoption order, with the stretch factors §10 fixes.
CELL_SCALES = (1.0892, 1.4116, 0.8237)
RUNG_SCALES = {
    "raw": (1.0, 1.0, 1.0),
    "raw_var_matched": CELL_SCALES,
    "down_stratified": (1.0, 1.0, 1.0),
    "down_stratified_var_matched": (1.0892, 1.0, 1.0),
}


# --------------------------------------------------------------------------
# the stream and the score
# --------------------------------------------------------------------------


def load_plays() -> tuple[pl.DataFrame, dict]:
    """Document 35 §2's input stream, built from the production module."""
    pbp = load_pbp(
        columns=[
            "game_id",
            "play_id",
            "season",
            "week",
            "posteam",
            "home_team",
            "away_team",
            "play_type",
            "epa",
            "yardline_100",
            "down",
        ]
    )
    scrimmage = pbp.filter(
        pl.col("posteam").is_not_null()
        & pl.col("play_type").is_in(["pass", "run"])
        & pl.col("epa").is_not_null()
        & pl.col("down").is_not_null()
    )
    adjudicated = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / GAMES_ARTIFACT).select("game_id")
    ledger = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / LEDGER_ARTIFACT)

    plays = assign_cells(
        price_plays(scrimmage.join(adjudicated, on="game_id", how="inner"), ledger)
    ).with_columns(
        (pl.col("posteam") == pl.col("home_team")).alias("is_home"),
        pl.concat_str([pl.col("season").cast(pl.String), pl.col("posteam")], separator="_").alias(
            "team_season"
        ),
    )

    reconciliation = {
        "n_scrimmage_plays": int(scrimmage.height),
        "n_plays_in_scope": int(plays.height),
        "n_plays_carrying_a_ledger_row": int((plays["luck_epa_pos"] != 0.0).sum()),
        "n_games": int(plays["game_id"].n_unique()),
        "mean_abs_repricing_epa": float(plays["luck_epa_pos"].abs().mean()),
    }
    return plays.sort(["team_season", "game_id", "play_id"]), reconciliation


def score_all(plays: pl.DataFrame) -> pl.DataFrame:
    """One row per team-game: the three cell scores and their denominators."""
    rows = []
    for (game_id, posteam), block in plays.group_by(["game_id", "posteam"], maintain_order=True):
        epa = block["epa_priced"].to_numpy().astype(float)
        cell = block["cell"].to_numpy().astype(int)
        scored = score_team_game(epa, cell)
        rows.append(
            {
                "game_id": game_id,
                "posteam": posteam,
                "season": int(block["season"][0]),
                "week": int(block["week"][0]),
                "team_season": block["team_season"][0],
                "is_home": bool(block["is_home"][0]),
                "red_zone": scored.red_zone,
                "late_down": scored.late_down,
                "other": scored.other,
                "score": scored.score,
                "n_all": scored.n_all,
                "n_rz": scored.n_red_zone,
                "n_ld": scored.n_late_down,
                "epa_sum": float(epa.sum()),
            }
        )
    return pl.DataFrame(rows).sort(["team_season", "game_id"])


def cached_scores() -> pl.DataFrame:
    path = paths.RESEARCH_OUTPUT_DIR / SCORES_CACHE
    if not path.exists():
        plays, _ = load_plays()
        score_all(plays).write_parquet(path)
    return pl.read_parquet(path)


def differentials(scores: pl.DataFrame) -> pl.DataFrame:
    home = scores.filter("is_home").select("game_id", "season", pl.col("score").alias("home_score"))
    away = scores.filter(~pl.col("is_home")).select("game_id", pl.col("score").alias("away_score"))
    return home.join(away, on="game_id", how="inner").with_columns(
        (pl.col("home_score") - pl.col("away_score")).alias("differential")
    )


# --------------------------------------------------------------------------
# results file
# --------------------------------------------------------------------------


def save(section: str, payload: dict) -> None:
    path = paths.RESEARCH_OUTPUT_DIR / RESULTS
    existing = json.loads(path.read_text()) if path.exists() else {"random_seed": RANDOM_SEED}
    existing[section] = payload
    path.write_text(json.dumps(existing, indent=2, default=float))
    print(f"\nwrote {section} to {path}")


# --------------------------------------------------------------------------
# M-1 — the identities
# --------------------------------------------------------------------------


def gate_m1() -> dict:
    """Document 35 §4. Arithmetic, not statistics; a failure stops the round."""
    plays, reconciliation = load_plays()
    print(f"{'=' * 72}\nM-1 — the identities\n{'=' * 72}")
    for key, value in reconciliation.items():
        print(f"  {key:<34} {value}")

    stream_matches = {
        "plays": reconciliation["n_plays_in_scope"] == EXPECTED_PLAYS,
        "games": reconciliation["n_games"] == EXPECTED_GAMES,
        "ledger_plays": reconciliation["n_plays_carrying_a_ledger_row"] == EXPECTED_LEDGER_PLAYS,
    }
    print(f"\n  reproduces document 35 §2's stream: {stream_matches}")

    scores = score_all(plays)
    scores.write_parquet(paths.RESEARCH_OUTPUT_DIR / SCORES_CACHE)

    residual = (scores["red_zone"] + scores["late_down"] + scores["other"]).abs().max()
    empty_rz = scores.filter(pl.col("n_rz") == 0)
    empty_rz_zero = bool((empty_rz["red_zone"] == 0.0).all())
    empty_ld = scores.filter(pl.col("n_ld") == 0)

    # Ledger reconciliation: the plays the meter re-prices must be exactly the
    # ledger rows that survive the S0-S2 filter, and the repricing must match the
    # ledger's own sum on those rows.
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
    components = sorted(
        ledger.join(repriced, on=["game_id", "play_id"], how="inner")["component"]
        .unique()
        .to_list()
    )
    joined = plays.join(ledger_in_filter, on=["game_id", "play_id"], how="inner")
    sum_gap = float(joined["luck_epa_pos"].abs().sum() - joined["luck_epa_home"].abs().sum())

    diffs = differentials(scores)
    recomputed = diffs["home_score"] - diffs["away_score"]
    differential_gap = float((recomputed - diffs["differential"]).abs().max())

    checks = {
        "three_cells_sum_to_zero": {
            "worst_residual_points": float(residual),
            "tolerance": IDENTITY_TOLERANCE,
            "pass": bool(residual <= IDENTITY_TOLERANCE),
        },
        "empty_red_zone_cell_is_exactly_zero": {
            "n_team_games": int(empty_rz.height),
            "share": float(empty_rz.height / scores.height),
            "pass": empty_rz_zero,
        },
        "empty_late_down_cell_is_exactly_zero": {
            "n_team_games": int(empty_ld.height),
            "pass": bool((empty_ld["late_down"] == 0.0).all()),
        },
        "ledger_reconciliation": {
            "n_repriced_plays": int(repriced.height),
            "rows_are_the_ledger_rows_surviving_the_filter": bool(same_rows),
            "components_touched": components,
            "abs_sum_gap_epa": sum_gap,
            "pass": bool(same_rows and abs(sum_gap) <= IDENTITY_TOLERANCE),
        },
        "differential_recomputes": {
            "worst_gap_points": differential_gap,
            "n_games": int(diffs.height),
            "pass": bool(differential_gap <= IDENTITY_TOLERANCE),
        },
        "stream_reproduces_the_pre_registration": {
            "detail": stream_matches,
            "pass": all(stream_matches.values()),
        },
    }

    print()
    for name, check in checks.items():
        print(f"  {'PASS' if check['pass'] else 'FAIL'}  {name}")
        for key, value in check.items():
            if key != "pass":
                print(f"          {key}: {value}")

    verdict = all(check["pass"] for check in checks.values())
    print(f"\n  M-1 VERDICT: {'PASS' if verdict else 'FAIL'}")
    return {
        "reconciliation": reconciliation,
        "n_team_games": int(scores.height),
        "checks": checks,
        "verdict": "pass" if verdict else "fail",
    }


# --------------------------------------------------------------------------
# M-2 — is the band honest?
# --------------------------------------------------------------------------

# Document 35 §6, secondary and reported only: each rung's KS distance under its
# *own* exchangeable truth. A common tolerance would fail the narrower rungs for
# their construction and a per-rung one would let a wide rung grade itself, so
# neither is a calibration test. Shape information.
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


def rung_pit(plays: pl.DataFrame, rung: str, seed: int) -> pl.DataFrame:
    """One row per team-game: its band under this rung, and its PIT inside it."""
    rng = np.random.default_rng(seed)
    scales = RUNG_SCALES[rung]
    rows = []
    for (game_id, posteam), block in plays.group_by(["game_id", "posteam"], maintain_order=True):
        epa = block["epa_priced"].to_numpy().astype(float)
        cell = block["cell"].to_numpy().astype(int)
        down = block["down"].to_numpy().astype(int)
        realized = score_team_game(epa, cell).score
        draws = permutation_draws(rung, epa, cell, down, N_BAND_DRAWS, rng, scales)
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
    """Document 35 §6. Coverage is the gate; KS is reported as shape only."""
    plays, _ = load_plays()
    print(f"{'=' * 72}\nM-2 — the calibration ladder\n{'=' * 72}")
    print(
        f"  the 89% band must cover {COVERAGE_LOW * 100:.1f}-{COVERAGE_HIGH * 100:.1f}% of the "
        f"team-games; {N_BAND_DRAWS} draws per team-game, 89% equal-tailed"
    )

    results = {}
    for index, rung in enumerate(LADDER):
        table = rung_pit(plays, rung, RANDOM_SEED + index)
        pit = table["pit"].to_numpy().astype(float)
        coverage = float(table["inside"].mean())
        ks = ks_statistic(pit)
        qualifies = COVERAGE_LOW <= coverage <= COVERAGE_HIGH
        ks_within = ks <= KS_REFERENCE_P95[rung]
        table.write_parquet(paths.RESEARCH_OUTPUT_DIR / f"placement_band_{rung}.parquet")
        results[rung] = {
            "coverage": coverage,
            "qualifies": bool(qualifies),
            "ks": ks,
            "ks_reference_p95": KS_REFERENCE_P95[rung],
            "ks_within_own_null": bool(ks_within),
            "readings_agree": bool(qualifies == ks_within),
            "pit_mean": float(pit.mean()),
            "median_band_width_points": float((table["band_high"] - table["band_low"]).median()),
            "n_team_games": int(table.height),
        }
        print(
            f"\n  {rung:<30} coverage {coverage * 100:6.2f}%  "
            f"[{'qualifies' if qualifies else 'FAILS'}]"
        )
        print(
            f"  {'':<30} KS {ks:.4f} vs its own null's 95th {KS_REFERENCE_P95[rung]:.4f}  "
            f"[{'within' if ks_within else 'outside'}]   "
            f"median band width {results[rung]['median_band_width_points']:.2f} pts"
        )

    adopted = next((rung for rung in LADDER if results[rung]["qualifies"]), None)
    disagreements = [rung for rung, block in results.items() if not block["readings_agree"]]

    print(f"\n  adoption rule — the least-constrained qualifying rung, ladder order {LADDER}")
    print(f"  ADOPTED: {adopted if adopted else 'NONE — the meter ships without a band'}")
    if disagreements:
        print(f"  coverage and KS disagree on: {disagreements} — forks to the maintainer (document 35 §6)")

    return {
        "tolerance": [COVERAGE_LOW, COVERAGE_HIGH],
        "n_band_draws": N_BAND_DRAWS,
        "rungs": results,
        "adopted": adopted,
        "readings_disagree_on": disagreements,
        "verdict": "pass" if adopted else "no rung qualifies",
    }


# --------------------------------------------------------------------------
# M-3 — is the shipped score still luck?
# --------------------------------------------------------------------------


def gate_m3() -> dict:
    """Document 35 §7. Document 08's split-half machinery, threshold r > 0.0671.

    The machinery is imported from the power script rather than re-implemented,
    because a gate whose statistic differs from the one its threshold was
    simulated on is not the gate that was pre-registered.
    """
    scores = cached_scores()
    print(f"{'=' * 72}\nM-3 — the luck licence\n{'=' * 72}")

    meta = scores.select("team_season", "game_id")
    rows, blocks = _power.season_blocks(meta)
    values = scores["score"].to_numpy().astype(float)[rows]
    rng = np.random.default_rng(RANDOM_SEED)
    mask = _power.split_masks(blocks, len(values), rng)
    r = _power.split_half_r(values, mask, blocks)

    per_split = []
    for split in range(mask.shape[0]):
        per_split.append(_power.split_half_r(values, mask[split : split + 1], blocks))
    per_split = np.array(per_split)

    verdict = "luck" if r <= M3_THRESHOLD else "a team property"
    print(f"  {len(blocks)} team-seasons, {len(values)} team-games at the {MIN_GAMES}-game floor")
    print(f"  {N_SPLITS} within-season splits, half statistic = mean placement points per game")
    print(
        f"\n  split-half r = {r:+.4f}   "
        f"(5th-95th across splits {np.quantile(per_split, 0.05):+.4f} to "
        f"{np.quantile(per_split, 0.95):+.4f})"
    )
    print(f"  pre-registered threshold: r > {M3_THRESHOLD:.4f} means NOT luck")
    print(f"\n  M-3 VERDICT: {'PASS' if r <= M3_THRESHOLD else 'FAIL'} — placement is {verdict}")
    print(
        "  Power 0.900 at r = 0.12, so this is pass/fail rather than descriptive (document 35 §7)."
    )
    return {
        "n_team_seasons": int(len(blocks)),
        "n_team_games": int(len(values)),
        "split_half_r": r,
        "per_split_p05": float(np.quantile(per_split, 0.05)),
        "per_split_p95": float(np.quantile(per_split, 0.95)),
        "threshold": M3_THRESHOLD,
        "verdict": "pass" if r <= M3_THRESHOLD else "fail",
        "reading": verdict,
    }


# --------------------------------------------------------------------------
# M-4 — does the baseline leak skill?
# --------------------------------------------------------------------------


def gate_m4() -> dict:
    """Document 35 §7. |corr(team-season mean score, S0 quality)| <= 0.1065.

    A pass licenses "no leak larger than roughly 0.16", never "no leak": at
    document 05 §7's +-0.11 detectability floor the design sees a leak only 48%
    of the time. That sentence travels with every report of this gate.
    """
    scores = cached_scores()
    print(f"{'=' * 72}\nM-4 — skill preservation\n{'=' * 72}")

    season = (
        scores.group_by("team_season", maintain_order=True)
        .agg(
            pl.len().alias("games"),
            pl.col("score").mean().alias("mean_score"),
            pl.col("score").std(ddof=1).alias("sd_score"),
            (pl.col("epa_sum").sum() / pl.col("n_all").sum()).alias("s0_quality"),
            pl.col("red_zone").mean().alias("mean_red_zone"),
            pl.col("late_down").mean().alias("mean_late_down"),
            (pl.col("n_rz").sum() / pl.col("n_all").sum()).alias("share_red_zone_plays"),
            (pl.col("n_ld").sum() / pl.col("n_all").sum()).alias("share_late_down_plays"),
        )
        .filter(pl.col("games") >= MIN_GAMES)
    )
    quality = season["s0_quality"].to_numpy().astype(float)
    mean_score = season["mean_score"].to_numpy().astype(float)
    sd_score = season["sd_score"].to_numpy().astype(float)

    def against_quality(column: str) -> float:
        return float(np.corrcoef(season[column].to_numpy().astype(float), quality)[0, 1])

    r = float(np.corrcoef(mean_score, quality)[0, 1])
    dispersion_r = float(np.corrcoef(sd_score, quality)[0, 1])

    print(
        f"  {season.height} team-seasons; offensive quality sd {quality.std(ddof=1):.4f} EPA/play"
    )
    print(f"\n  corr(team-season mean placement points, S0 quality) = {r:+.4f}")
    print(f"  pre-registered bound: |r| <= {M4_BOUND:.4f}")
    print(f"\n  M-4 VERDICT: {'PASS' if abs(r) <= M4_BOUND else 'FAIL'}")
    print(
        "  A pass licenses 'no leak larger than roughly 0.16', not 'no leak': power is "
        "0.48 at a leak of 0.11 (document 35 §7)."
    )
    print(
        f"\n  secondary, descriptive, no rule: corr(team-season spread of scores, quality) "
        f"= {dispersion_r:+.4f}"
    )
    # Where the leak lives, and why. Not a second gate — the verdict is already
    # decided above. This is the material a "back to design" routing needs, and
    # it is computed here rather than in an ad-hoc query so it is reproducible.
    channels = {
        "red_zone_cell": against_quality("mean_red_zone"),
        "late_down_cell": against_quality("mean_late_down"),
        "share_of_plays_in_the_red_zone": against_quality("share_red_zone_plays"),
        "share_of_plays_on_late_downs": against_quality("share_late_down_plays"),
    }
    between_sd = float(season["mean_score"].std(ddof=1))
    within_sd = float(season["sd_score"].mean())
    games = float(season["games"].mean())
    signal = max(between_sd**2 - within_sd**2 / games, 0.0)
    implied_split_half = signal / (signal + within_sd**2 / (games / 2.0))

    print("\n  where the leak lives — correlation with S0 quality:")
    for name, value in channels.items():
        print(f"    {name:<32} {value:+.4f}")
    print(
        f"\n  variance components: between-team-season SD {between_sd:.3f} pts/game against a "
        f"within-season per-game SD of {within_sd:.3f} over {games:.1f} games"
    )
    print(
        f"  -> implied split-half r {implied_split_half:.4f}, which is why M-3 reads near zero on "
        "the same data: correlating against a known covariate is far more powerful than "
        "correlating a statistic with itself"
    )

    return {
        "n_team_seasons": int(season.height),
        "corr_mean_score_quality": r,
        "bound": M4_BOUND,
        "verdict": "pass" if abs(r) <= M4_BOUND else "fail",
        "ceiling_sentence": "a pass licenses no leak larger than roughly 0.16, not no leak",
        "secondary_corr_dispersion_quality": dispersion_r,
        "leak_channels_vs_quality": channels,
        "variance_components": {
            "between_team_season_sd": between_sd,
            "within_season_per_game_sd": within_sd,
            "mean_games": games,
            "implied_split_half_r": implied_split_half,
        },
    }


GATES = {"m1": gate_m1, "m2": gate_m2, "m3": gate_m3, "m4": gate_m4}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", required=True, choices=sorted(GATES))
    args = parser.parse_args()
    save(args.gate, GATES[args.gate]())


if __name__ == "__main__":
    main()
