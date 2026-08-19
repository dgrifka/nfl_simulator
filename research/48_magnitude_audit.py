"""Product layer, task 1 — the magnitude audit on simulator v1.3.

Document 06 priced luck at 6.4% of margin variance realistically and 25.1% at
the ceiling, and document 10 found roughly half of all games nearly luck-free.
The obvious reader question is therefore "if luck is only 6.4%, does the
adjudication ever say anything?" This script answers it on the shipped artifact
rather than by recollection: how often the deserved winner differs from the
realized one, how the DTW% distribution is shaped, how much of it is
degenerate, and how far the deserved margin moves from the realized one.

**Descriptive, not a gate.** Nothing is fitted, nothing is tested, no threshold
is pre-registered — this is a read of `dtw_games_v13.parquet`, produced by
`research/46_simulator_v13.py`. The numbers exist to be quoted in product copy,
which is exactly why they are computed and committed instead of remembered.

Two conventions are inherited, not invented here:

- **Degenerate** means a DTW% outside the open interval (0.001, 0.999) —
  document 10's Gate V-3 boundary, mirrored in `research/16_coverage_power.py`.
  The stricter "exactly 0 or 1" count is reported beside it.
- **Home wins** means a margin strictly greater than zero. The simulator's DTW
  is `P(margin > 0)`, so a tie is not a home win, in the artifact and here.

    uv run python research/48_magnitude_audit.py
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from nfl_simulator import paths

ARTIFACT = "dtw_games_v13.parquet"

# Deterministic facts about the artifact this audit reads. A mismatch means the
# artifact is not the one document 31 published, and the audit stops.
EXPECTED_GAMES = 2761
EXPECTED_SEASONS = list(range(2016, 2026))

DEGENERATE_LOW = 0.001  # document 10, Gate V-3
DEGENERATE_HIGH = 0.999

QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
MARGIN_THRESHOLDS = [0.5, 1.0, 3.0, 7.0]
DTW_BINS = [0.0, 0.001, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.999, 1.0]


def _quantile_table(values: np.ndarray) -> dict[str, float]:
    return {f"q{int(q * 100):02d}": float(np.quantile(values, q)) for q in QUANTILES}


def main() -> None:
    paths.ensure_data_dirs()
    games = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / ARTIFACT)

    seasons = sorted(games["season"].unique().to_list())
    if games.height != EXPECTED_GAMES or seasons != EXPECTED_SEASONS:
        raise SystemExit(
            f"{ARTIFACT} is not the artifact document 31 published: "
            f"{games.height} games over {seasons[0]}-{seasons[-1]}, expected "
            f"{EXPECTED_GAMES} over 2016-2025."
        )
    print(f"{games.height} games, {seasons[0]}-{seasons[-1]}, from {ARTIFACT}")

    actual = games["actual_margin"].to_numpy().astype(float)
    deserved = games["deserved_margin"].to_numpy().astype(float)
    dtw = games["dtw_home"].to_numpy().astype(float)
    low = games["dtw_low"].to_numpy().astype(float)
    high = games["dtw_high"].to_numpy().astype(float)
    n_events = games["n_luck_events"].to_numpy().astype(int)

    # ---------------------------------------------------------------- flips
    # A flip is a sign disagreement between the two margins. Ties are their own
    # category in both directions: a realized tie has no realized winner, and a
    # deserved margin of exactly zero has no deserved winner.
    home_won = actual > 0
    realized_tie = actual == 0
    home_deserved = deserved > 0
    deserved_tie = deserved == 0

    flip = (home_won != home_deserved) & ~realized_tie & ~deserved_tie
    tie_broken = realized_tie & ~deserved_tie  # a drawn game the ledger decides

    print(f"\n{'=' * 72}\nWINNER FLIPS — deserved margin disagrees with the scoreboard\n{'=' * 72}")
    print(f"  realized ties                {int(realized_tie.sum()):5d}")
    print(f"  deserved margin exactly 0    {int(deserved_tie.sum()):5d}")
    print(
        f"  sign flips                   {int(flip.sum()):5d}  "
        f"({flip.mean() * 100:.2f}% of all games)"
    )
    print(f"  realized ties given a winner {int(tie_broken.sum()):5d}")

    # The same question asked of the coin-flip distribution rather than the
    # point estimate: how often does the bootstrap put the realized winner
    # below even money?
    dtw_favours_home = dtw > 0.5
    dtw_flip = (home_won != dtw_favours_home) & ~realized_tie
    coin_flip_games = (dtw >= 0.4) & (dtw <= 0.6)
    print(
        f"  DTW% on the losing side       {int(dtw_flip.sum()):5d}  "
        f"({dtw_flip.mean() * 100:.2f}%)  [DTW% < 0.5 for the realized winner]"
    )
    print(
        f"  DTW% within 0.40-0.60         {int(coin_flip_games.sum()):5d}  "
        f"({coin_flip_games.mean() * 100:.2f}%)  [the genuinely undecided games]"
    )

    # ------------------------------------------- flip-definition reconciliation
    # The two flip counts differ by 24, but that is a NET difference and not the
    # number of games they disagree about: each definition catches games the
    # other misses. The disagreement set is what a product has to care about,
    # so it is counted directly.
    disagree = flip != dtw_flip
    sign_only = flip & ~dtw_flip
    dtw_only = dtw_flip & ~flip
    both = flip & dtw_flip

    print(f"\n{'=' * 72}\nFLIP-DEFINITION RECONCILIATION — where the two disagree\n{'=' * 72}")
    print(f"  both definitions agree it flipped   {int(both.sum()):5d}")
    print(f"  sign flip only                      {int(sign_only.sum()):5d}")
    print(f"  DTW% flip only                      {int(dtw_only.sum()):5d}")
    print(
        f"  disagreements                       {int(disagree.sum()):5d}  "
        f"(NOT the 24-game net difference)"
    )
    print(
        f"  their DTW% range                    "
        f"[{dtw[disagree].min():.3f}, {dtw[disagree].max():.3f}]"
    )
    print(
        f"  their |deserved margin|: median {np.median(np.abs(deserved[disagree])):.3f} pt, "
        f"max {np.abs(deserved[disagree]).max():.3f} pt"
    )
    print(
        f"  of them, inside DTW% 0.40-0.60      "
        f"{int((coin_flip_games & disagree).sum()):5d} of {int(disagree.sum())}"
    )

    # A third bucket collapses the argument: call 0.40-0.60 "too close to call"
    # and the two definitions stop mattering almost everywhere.
    clear_dtw_flip = dtw_flip & ~coin_flip_games
    clear_sign_flip = flip & ~coin_flip_games
    residual = (clear_dtw_flip != clear_sign_flip) & ~realized_tie
    print(
        f"\n  With a 'too close to call' band at DTW% 0.40-0.60:"
        f"\n    too close to call                 {int(coin_flip_games.sum()):5d}  "
        f"({coin_flip_games.mean() * 100:.2f}%)"
        f"\n    clear flips (DTW% definition)     {int(clear_dtw_flip.sum()):5d}  "
        f"({clear_dtw_flip.mean() * 100:.2f}%)"
        f"\n    clear flips (sign definition)     {int(clear_sign_flip.sum()):5d}  "
        f"({clear_sign_flip.mean() * 100:.2f}%)"
        f"\n    residual disagreements            {int(residual.sum()):5d}  "
        f"(down from {int(disagree.sum())})"
    )

    # ----------------------------------------------------------- degeneracy
    degenerate = (dtw <= DEGENERATE_LOW) | (dtw >= DEGENERATE_HIGH)
    exact = (dtw == 0.0) | (dtw == 1.0)
    no_events = n_events == 0
    zero_width = (high - low) == 0.0

    print(f"\n{'=' * 72}\nDEGENERACY — how often there is nothing left to adjudicate\n{'=' * 72}")
    print(
        f"  degenerate (DTW% outside {DEGENERATE_LOW}-{DEGENERATE_HIGH})  "
        f"{int(degenerate.sum()):5d}  ({degenerate.mean() * 100:.2f}%)"
    )
    print(
        f"  DTW% exactly 0 or 1                       {int(exact.sum()):5d}  "
        f"({exact.mean() * 100:.2f}%)"
    )
    print(
        f"  no luck events at all                     {int(no_events.sum()):5d}  "
        f"({no_events.mean() * 100:.2f}%)"
    )
    print(
        f"  zero-width interval                       {int(zero_width.sum()):5d}  "
        f"({zero_width.mean() * 100:.2f}%)"
    )
    print(
        f"  luck events per game: mean {n_events.mean():.2f}, "
        f"median {np.median(n_events):.0f}, max {n_events.max()}"
    )

    # ------------------------------------------------------ DTW% distribution
    hist, _ = np.histogram(dtw, bins=DTW_BINS)
    print(f"\n{'=' * 72}\nDTW% DISTRIBUTION (home team)\n{'=' * 72}")
    for lo, hi, count in zip(DTW_BINS[:-1], DTW_BINS[1:], hist, strict=True):
        bar = "#" * int(round(60 * count / hist.max()))
        print(f"  [{lo:.3f}, {hi:.3f})  {count:5d}  {bar}")
    quant_dtw = _quantile_table(dtw)
    print("  quantiles: " + ", ".join(f"{k} {v:.4f}" for k, v in quant_dtw.items()))
    print(f"  mean interval width {np.mean(high - low):.4f}")
    informative = ~degenerate
    print(
        f"  on the {int(informative.sum())} non-degenerate games: mean width "
        f"{np.mean((high - low)[informative]):.4f}, mean DTW% {dtw[informative].mean():.4f}"
    )

    # -------------------------------------------------------- margin movement
    shift = deserved - actual
    absolute = np.abs(shift)
    print(f"\n{'=' * 72}\nMARGIN MOVEMENT — |realized - deserved|, in points\n{'=' * 72}")
    print(f"  mean   {absolute.mean():.3f}")
    print(f"  median {np.median(absolute):.3f}")
    print("  quantiles: " + ", ".join(f"{k} {v:.3f}" for k, v in _quantile_table(absolute).items()))
    for threshold in MARGIN_THRESHOLDS:
        over = absolute > threshold
        print(
            f"  |shift| > {threshold:>4.1f} pt   {int(over.sum()):5d}  ({over.mean() * 100:.2f}%)"
        )
    print(f"  signed mean {shift.mean():+.4f} pt  (a home-field asymmetry check, not a gate)")
    print(f"  max shift {absolute.max():.2f} pt")

    # ------------------------------------------------- example-game candidates
    # Avenues 1 and 2 need one luck-heavy game and one degenerate game. They are
    # chosen here, from the audit, so the choice is reproducible rather than
    # picked by eye later.
    with_index = games.with_columns(
        pl.Series("abs_shift", absolute),
        pl.Series("is_degenerate", degenerate),
        pl.Series("is_flip", flip),
    )
    luck_heavy = (
        with_index.filter(~pl.col("is_degenerate"))
        .sort("abs_shift", descending=True)
        .head(5)
        .select(
            "game_id",
            "actual_margin",
            "deserved_margin",
            "dtw_home",
            "abs_shift",
            "n_luck_events",
            "is_flip",
        )
    )
    flips_by_size = (
        with_index.filter(pl.col("is_flip"))
        .sort("abs_shift", descending=True)
        .head(5)
        .select(
            "game_id", "actual_margin", "deserved_margin", "dtw_home", "abs_shift", "n_luck_events"
        )
    )
    degenerate_examples = (
        with_index.filter(pl.col("is_degenerate") & (pl.col("n_luck_events") > 0))
        .sort("abs_shift", descending=True)
        .head(5)
        .select(
            "game_id", "actual_margin", "deserved_margin", "dtw_home", "abs_shift", "n_luck_events"
        )
    )
    print(f"\n{'=' * 72}\nEXAMPLE-GAME CANDIDATES for avenues 1 and 2\n{'=' * 72}")
    print("largest margin movement among non-degenerate games:")
    print(luck_heavy)
    print("largest movement among winner flips:")
    print(flips_by_size)
    print("largest movement among degenerate games (luck happened, verdict unmoved):")
    print(degenerate_examples)

    # ----------------------------------------------------------- season table
    by_season = (
        with_index.group_by("season")
        .agg(
            pl.len().alias("games"),
            pl.col("is_flip").sum().alias("flips"),
            pl.col("is_degenerate").mean().alias("degenerate_share"),
            pl.col("abs_shift").mean().alias("mean_abs_shift"),
        )
        .sort("season")
    )
    print(f"\n{'=' * 72}\nBY SEASON\n{'=' * 72}")
    print(by_season)

    results = {
        "artifact": ARTIFACT,
        "reported_as": "descriptive audit, not a gate — nothing fitted, nothing tested",
        "n_games": int(games.height),
        "seasons": seasons,
        "flips": {
            "definition": "sign(deserved_margin) != sign(actual_margin), ties excluded",
            "n": int(flip.sum()),
            "share": float(flip.mean()),
            "n_realized_ties": int(realized_tie.sum()),
            "n_deserved_exact_zero": int(deserved_tie.sum()),
            "n_realized_ties_given_a_winner": int(tie_broken.sum()),
            "n_dtw_below_even_for_realized_winner": int(dtw_flip.sum()),
            "share_dtw_below_even_for_realized_winner": float(dtw_flip.mean()),
            "n_dtw_between_040_and_060": int(coin_flip_games.sum()),
            "share_dtw_between_040_and_060": float(coin_flip_games.mean()),
        },
        "flip_definition_reconciliation": {
            "n_both": int(both.sum()),
            "n_sign_only": int(sign_only.sum()),
            "n_dtw_only": int(dtw_only.sum()),
            "n_disagree": int(disagree.sum()),
            "net_difference": int(dtw_flip.sum() - flip.sum()),
            "disagreement_dtw_range": [float(dtw[disagree].min()), float(dtw[disagree].max())],
            "disagreement_abs_deserved_median": float(np.median(np.abs(deserved[disagree]))),
            "disagreement_abs_deserved_max": float(np.abs(deserved[disagree]).max()),
            "n_disagreements_inside_040_060": int((coin_flip_games & disagree).sum()),
            "with_too_close_band_040_060": {
                "n_too_close": int(coin_flip_games.sum()),
                "share_too_close": float(coin_flip_games.mean()),
                "n_clear_flips_dtw": int(clear_dtw_flip.sum()),
                "share_clear_flips_dtw": float(clear_dtw_flip.mean()),
                "n_clear_flips_sign": int(clear_sign_flip.sum()),
                "n_residual_disagreements": int(residual.sum()),
            },
        },
        "degeneracy": {
            "definition": f"DTW% <= {DEGENERATE_LOW} or >= {DEGENERATE_HIGH} (document 10, Gate V-3)",
            "n_degenerate": int(degenerate.sum()),
            "share_degenerate": float(degenerate.mean()),
            "n_exactly_zero_or_one": int(exact.sum()),
            "share_exactly_zero_or_one": float(exact.mean()),
            "n_no_luck_events": int(no_events.sum()),
            "share_no_luck_events": float(no_events.mean()),
            "n_zero_width_interval": int(zero_width.sum()),
            "luck_events_per_game": {
                "mean": float(n_events.mean()),
                "median": float(np.median(n_events)),
                "max": int(n_events.max()),
            },
        },
        "dtw_distribution": {
            "quantiles": quant_dtw,
            "histogram_bins": DTW_BINS,
            "histogram_counts": [int(c) for c in hist],
            "mean_interval_width_all": float(np.mean(high - low)),
            "mean_interval_width_non_degenerate": float(np.mean((high - low)[informative])),
            "n_non_degenerate": int(informative.sum()),
        },
        "margin_movement": {
            "mean_abs": float(absolute.mean()),
            "median_abs": float(np.median(absolute)),
            "quantiles_abs": _quantile_table(absolute),
            "signed_mean": float(shift.mean()),
            "max_abs": float(absolute.max()),
            "share_over_threshold": {
                str(t): float((absolute > t).mean()) for t in MARGIN_THRESHOLDS
            },
        },
        "example_candidates": {
            "luck_heavy_non_degenerate": luck_heavy.to_dicts(),
            "largest_flips": flips_by_size.to_dicts(),
            "degenerate_with_luck_events": degenerate_examples.to_dicts(),
        },
        "by_season": by_season.to_dicts(),
    }
    out = paths.RESEARCH_OUTPUT_DIR / "48_magnitude_audit.json"
    with out.open("w") as handle:
        json.dump(results, handle, indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
