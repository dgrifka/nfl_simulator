"""Step 4 — does each component persist for a team, and does stripping it help?

Two independent tests.

**Split-half persistence.** Within a team-season, randomly split the games into
two halves, average each component per game in each half, and correlate half A
against half B across all team-seasons. Repeat over many random splits. A
component a team controls shows a positive correlation; a coin flip shows zero.
Fumble recovery is the calibration case — it must come out near zero, or the
machinery is broken and none of the other answers can be trusted.

**Out-of-sample prediction.** Build a pre-game strength rating for each team from
its *previous* games, and use the home-minus-away strength difference to predict
who wins. Do it once with raw EPA differential and once with luck-stripped EPA
differential, and compare both to the Vegas closing spread. If stripping a
component makes the rating a better predictor of future games, that component
was noise polluting the historical record.

Note this is deliberately *not* a logistic regression on within-game EPA
differential. That quantity is measured from the same plays that produced the
score, so it would score ~96% by construction and could not be compared with a
pre-game number like the spread. The question that actually matters for the
simulator is whether luck-stripping improves a *forward-looking* estimate of
team strength, which is what this measures.

    uv run python research/02_skill_vs_luck.py
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from nfl_simulator import paths
from nfl_simulator.components import (
    COMPONENTS,
    build_game_table,
    decompose_games,
    decompose_plays,
    fit_fg_baseline,
    fit_fumble_baseline,
    live_fumble_mask,
    to_team_games,
)
from nfl_simulator.ingest import ANALYSIS_COLUMNS, PBP_SEASONS, load_pbp

TRAIN_SEASONS = tuple(range(2016, 2024))
TEST_SEASONS = (2024, 2025)

N_SPLITS = 200
RANDOM_SEED = 20260817

# Trailing games used to rate a team going into a game, and the minimum history
# required before we are willing to rate them at all.
FORM_WINDOW = 17
MIN_HISTORY = 8


# --------------------------------------------------------------------------
# split-half persistence
# --------------------------------------------------------------------------


def split_half_correlations(
    team_games: pl.DataFrame,
    metrics: list[str],
    n_splits: int = N_SPLITS,
    seed: int = RANDOM_SEED,
) -> pl.DataFrame:
    """Mean split-half correlation per metric, across random within-season splits.

    Splitting *within* a team-season holds constant everything that makes a team
    a team that year — roster, coaching, scheme — so the only thing left to drive
    a correlation is whether the metric is a stable property of the team.
    """
    rng = np.random.default_rng(seed)
    frame = team_games.with_columns(
        pl.concat_str([pl.col("season").cast(pl.String), pl.col("team")], separator="_").alias(
            "team_season"
        )
    )

    # Dense arrays keyed by team-season are far faster than 200 polars group-bys.
    keys = frame["team_season"].to_numpy()
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    values = {metric: frame[metric].to_numpy()[order] for metric in metrics}
    boundaries = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1], True])
    groups = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    # A team-season needs at least 8 games to give 4 per half.
    groups = [(lo, hi) for lo, hi in groups if hi - lo >= 8]

    results = {metric: np.empty(n_splits) for metric in metrics}
    for split in range(n_splits):
        halves = {metric: (np.empty(len(groups)), np.empty(len(groups))) for metric in metrics}
        for index, (lo, hi) in enumerate(groups):
            size = hi - lo
            permuted = rng.permutation(size)
            first, second = permuted[: size // 2], permuted[size // 2 :]
            for metric in metrics:
                block = values[metric][lo:hi]
                halves[metric][0][index] = block[first].mean()
                halves[metric][1][index] = block[second].mean()
        for metric in metrics:
            a, b = halves[metric]
            results[metric][split] = np.corrcoef(a, b)[0, 1]

    rows = []
    for metric in metrics:
        draws = results[metric]
        mean_r = draws.mean()
        # Spearman-Brown: what the reliability would be for a full season rather
        # than a half. Useful for reading the magnitude, not for the zero test.
        full_season = 2 * mean_r / (1 + mean_r) if mean_r > -1 else np.nan
        rows.append(
            {
                "metric": metric,
                "split_half_r": mean_r,
                "r_p05": np.percentile(draws, 5),
                "r_p95": np.percentile(draws, 95),
                "spearman_brown_r": full_season,
                "n_team_seasons": len(groups),
            }
        )
    return pl.DataFrame(rows).sort("split_half_r", descending=True)


def team_fumble_recovery_rates(pbp: pl.DataFrame) -> pl.DataFrame:
    """Per team-season-game: fumbles by that team and how many it got back.

    This is the raw rate the literature quotes, tested directly rather than
    through its EPA translation.
    """
    fumbles = pbp.filter(live_fumble_mask()).select(
        pl.col("game_id"),
        pl.col("season"),
        pl.col("fumbled_1_team").alias("team"),
        (pl.col("fumble_recovery_1_team") == pl.col("fumbled_1_team")).cast(pl.Int8).alias("kept"),
    )
    return fumbles.group_by(["season", "team", "game_id"]).agg(
        pl.len().alias("fumbles"), pl.col("kept").sum().alias("kept")
    )


def fumble_rate_split_half(
    fumble_games: pl.DataFrame, n_splits: int = N_SPLITS, seed: int = RANDOM_SEED
) -> dict:
    """Split-half on the recovery *rate*, pooling fumbles rather than averaging games.

    A team that fumbled once in half A and kept it has a 100% rate off one
    event. Pooling numerator and denominator across the half is the honest
    estimator.
    """
    rng = np.random.default_rng(seed)
    frame = fumble_games.with_columns(
        pl.concat_str([pl.col("season").cast(pl.String), pl.col("team")], separator="_").alias(
            "team_season"
        )
    )
    grouped = frame.group_by("team_season").agg(
        pl.col("fumbles").alias("fumbles"), pl.col("kept").alias("kept")
    )
    # Need enough games that both halves can hold real fumbles.
    grouped = grouped.filter(pl.col("fumbles").list.len() >= 8)

    fumble_lists = grouped["fumbles"].to_list()
    kept_lists = grouped["kept"].to_list()

    draws = np.empty(n_splits)
    for split in range(n_splits):
        first_rates, second_rates = [], []
        for fumbles, kept in zip(fumble_lists, kept_lists, strict=True):
            fumbles_arr = np.asarray(fumbles)
            kept_arr = np.asarray(kept)
            permuted = rng.permutation(len(fumbles_arr))
            first, second = permuted[: len(permuted) // 2], permuted[len(permuted) // 2 :]
            if fumbles_arr[first].sum() == 0 or fumbles_arr[second].sum() == 0:
                continue
            first_rates.append(kept_arr[first].sum() / fumbles_arr[first].sum())
            second_rates.append(kept_arr[second].sum() / fumbles_arr[second].sum())
        draws[split] = np.corrcoef(first_rates, second_rates)[0, 1]

    return {
        "split_half_r": float(draws.mean()),
        "r_p05": float(np.percentile(draws, 5)),
        "r_p95": float(np.percentile(draws, 95)),
        "n_team_seasons": grouped.height,
    }


# --------------------------------------------------------------------------
# out-of-sample prediction
# --------------------------------------------------------------------------


def rolling_strength(team_games: pl.DataFrame, metric: str) -> pl.DataFrame:
    """Each team's mean `metric` over its previous `FORM_WINDOW` games.

    Strictly prior games — the current game is excluded, so nothing leaks from
    the outcome being predicted into the feature predicting it.
    """
    return (
        team_games.sort(["team", "season", "week"])
        .with_columns(
            pl.col(metric)
            .shift(1)
            .rolling_mean(window_size=FORM_WINDOW, min_samples=MIN_HISTORY)
            .over("team")
            .alias("strength"),
            pl.int_range(pl.len()).over("team").alias("games_played"),
        )
        .select("game_id", "team", "strength", "games_played")
    )


def build_prediction_frame(games: pl.DataFrame, metric: str, spreads: pl.DataFrame) -> pl.DataFrame:
    """One row per game: home-minus-away strength, the spread, and who won."""
    team_games = to_team_games(games)
    strength = rolling_strength(team_games, metric)

    home = strength.rename({"team": "home_team", "strength": "home_strength"}).drop("games_played")
    away = strength.rename({"team": "away_team", "strength": "away_strength"}).drop("games_played")

    return (
        games.join(home, on=["game_id", "home_team"], how="left")
        .join(away, on=["game_id", "away_team"], how="left")
        .join(spreads, on="game_id", how="left")
        .filter(
            pl.col("home_strength").is_not_null()
            & pl.col("away_strength").is_not_null()
            & pl.col("spread_line").is_not_null()
            & (pl.col("margin") != 0)
        )
        .with_columns(
            (pl.col("home_strength") - pl.col("away_strength")).alias("strength_diff"),
            (pl.col("margin") > 0).cast(pl.Int8).alias("home_won"),
        )
    )


def evaluate(frame: pl.DataFrame, feature: str, name: str) -> dict:
    """Fit on the training seasons, score on the test seasons."""
    train = frame.filter(pl.col("season").is_in(TRAIN_SEASONS))
    test = frame.filter(pl.col("season").is_in(TEST_SEASONS))

    model = LogisticRegression()
    model.fit(train[[feature]].to_numpy(), train["home_won"].to_numpy())
    probability = model.predict_proba(test[[feature]].to_numpy())[:, 1]
    truth = test["home_won"].to_numpy()

    return {
        "model": name,
        "feature": feature,
        "n_train": train.height,
        "n_test": test.height,
        "log_loss": float(log_loss(truth, probability)),
        "brier": float(brier_score_loss(truth, probability)),
        "auc": float(roc_auc_score(truth, probability)),
        "accuracy": float(((probability > 0.5) == truth).mean()),
        "coef": float(model.coef_[0][0]),
        "_probability": probability,
        "_truth": truth,
    }


def _per_game_loss(probability: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-game negative log likelihood, so losses can be paired across models."""
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    return -(truth * np.log(clipped) + (1 - truth) * np.log(1 - clipped))


def calibration_table(probability: np.ndarray, truth: np.ndarray, n_bins: int = 5) -> list[dict]:
    """Predicted vs actual home win rate, in equal-count bins."""
    order = np.argsort(probability)
    rows = []
    for chunk in np.array_split(order, n_bins):
        rows.append(
            {
                "n": int(len(chunk)),
                "mean_predicted": float(probability[chunk].mean()),
                "actual": float(truth[chunk].mean()),
            }
        )
    return rows


# --------------------------------------------------------------------------


def main() -> None:
    pbp = load_pbp(PBP_SEASONS, columns=ANALYSIS_COLUMNS)
    results: dict = {}

    # ---- persistence -----------------------------------------------------
    games = build_game_table(pbp)
    team_games = to_team_games(games)

    metrics = ["epa_diff", *COMPONENTS]
    persistence = split_half_correlations(team_games, metrics)
    print("\n=== Split-half persistence, within team-season ===")
    print("(correlation between a team's two random halves of a season)")
    print(persistence)
    results["persistence"] = persistence.to_dicts()

    rate = fumble_rate_split_half(team_fumble_recovery_rates(pbp))
    print("\n=== Calibration case: team fumble RECOVERY RATE ===")
    print(
        f"split-half r = {rate['split_half_r']:+.4f} "
        f"[{rate['r_p05']:+.3f}, {rate['r_p95']:+.3f}] "
        f"over {rate['n_team_seasons']} team-seasons"
    )
    results["fumble_rate_persistence"] = rate

    # ---- prediction ------------------------------------------------------
    # Baselines must not see the test seasons, or the luck definition itself
    # would carry information from the games being predicted.
    train_pbp = pbp.filter(pl.col("season").is_in(TRAIN_SEASONS))
    fumble_baseline = fit_fumble_baseline(train_pbp)
    fg_baseline = fit_fg_baseline(train_pbp)
    honest_games = decompose_games(decompose_plays(pbp, fumble_baseline, fg_baseline))

    spreads = (
        pbp.group_by("game_id").agg(pl.col("spread_line").first()).select("game_id", "spread_line")
    )

    variants = {
        "raw EPA differential": "epa_diff",
        "minus fumble luck": ["fumble_luck"],
        "minus FG luck": ["fg_luck"],
        "minus fumble + FG luck": ["fumble_luck", "fg_luck"],
        "minus fumble + FG + INT": ["fumble_luck", "fg_luck", "interception"],
        "minus fumble + FG + penalty": ["fumble_luck", "fg_luck", "penalty"],
    }

    evaluations = []
    for name, spec in variants.items():
        # Overwrite epa_diff itself so `to_team_games` flips the stripped value
        # for the away side along with everything else.
        stripped = pl.col("epa_diff")
        if spec != "epa_diff":
            for component in spec:
                stripped = stripped - pl.col(component)
        frame_games = honest_games.with_columns(stripped.alias("epa_diff"))
        frame = build_prediction_frame(frame_games, "epa_diff", spreads)
        evaluations.append(evaluate(frame, "strength_diff", name))

    # Vegas benchmark on exactly the same games, so log losses are comparable.
    reference = build_prediction_frame(honest_games, "epa_diff", spreads)
    evaluations.append(evaluate(reference, "spread_line", "Vegas spread_line"))

    table = pl.DataFrame(
        [{k: v for k, v in row.items() if not k.startswith("_")} for row in evaluations]
    ).sort("log_loss")
    print(
        f"\n=== Out-of-sample prediction (train {TRAIN_SEASONS[0]}-{TRAIN_SEASONS[-1]}, "
        f"test {TEST_SEASONS[0]}-{TEST_SEASONS[-1]}) ==="
    )
    with pl.Config(tbl_cols=-1, fmt_str_lengths=40):
        print(table)
    results["prediction"] = table.to_dicts()

    # A 569-game test set does not resolve small log-loss gaps on its own. Pair
    # the per-game losses and bootstrap the difference to see whether any gap
    # survives resampling.
    print("\n=== Paired bootstrap vs raw EPA differential (test seasons) ===")
    baseline_row = next(row for row in evaluations if row["model"] == "raw EPA differential")
    baseline_losses = _per_game_loss(baseline_row["_probability"], baseline_row["_truth"])
    rng = np.random.default_rng(RANDOM_SEED)
    comparisons = []
    for row in evaluations:
        if row["model"] == "raw EPA differential":
            continue
        losses = _per_game_loss(row["_probability"], row["_truth"])
        difference = losses - baseline_losses
        indices = rng.integers(0, len(difference), size=(2000, len(difference)))
        draws = difference[indices].mean(axis=1)
        comparison = {
            "model": row["model"],
            "delta_log_loss": float(difference.mean()),
            "ci_low": float(np.percentile(draws, 2.5)),
            "ci_high": float(np.percentile(draws, 97.5)),
            "p_better": float((draws < 0).mean()),
        }
        comparisons.append(comparison)
        print(
            f"{comparison['model']:<32} delta = {comparison['delta_log_loss']:+.4f} "
            f"[{comparison['ci_low']:+.4f}, {comparison['ci_high']:+.4f}]  "
            f"P(better than raw) = {comparison['p_better']:.2f}"
        )
    results["bootstrap_vs_raw"] = comparisons

    print("\n=== Calibration (equal-count bins, test seasons) ===")
    calibrations = {}
    for row in evaluations:
        bins = calibration_table(row["_probability"], row["_truth"])
        calibrations[row["model"]] = bins
        rendered = "  ".join(f"{b['mean_predicted']:.2f}->{b['actual']:.2f}" for b in bins)
        print(f"{row['model']:<32} {rendered}")
    results["calibration"] = calibrations

    out = paths.RESEARCH_OUTPUT_DIR / "02_skill_vs_luck.json"
    with out.open("w") as handle:
        json.dump(results, handle, indent=2, default=str)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
