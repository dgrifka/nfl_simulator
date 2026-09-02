"""Step 6 — the rematch validation, run against the pre-registered gate.

Design, statistic and thresholds are fixed by
`docs/research/06-rematch-validation.md`, committed at `defee5c` before the
simulator produced a single deserve-to-win number. Nothing here chooses
anything; it executes what that document committed to.

    Gate 1  non-inferiority — the 95% CI upper bound on the paired delta log
            loss must sit below +0.010. This is the gate with power.
    Gate 2  direction — reported with its interval, no pass rule, because the
            power calculation put superiority at 7.2%.
    Gate 3  the coefficient sanity check on both arms.

    uv run python research/08_rematch.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

# The power calculation's own machinery, reused so the executed statistic is
# literally the simulated one rather than a re-implementation that might drift.
from importlib import import_module  # noqa: E402

_power = import_module("08_rematch_power")

from nfl_simulator import paths  # noqa: E402

RANDOM_SEED = 20260817
N_FOLDS = _power.N_FOLDS
NONINFERIORITY_MARGIN = _power.NONINFERIORITY_MARGIN


def build_pairs(dtw_artifact: str = "dtw_games.parquet") -> pl.DataFrame:
    """Rematch pairs carrying both game-1 predictors.

    ``dtw_artifact`` defaults to v1's, so this script reproduces document 07
    unchanged. Later versions pass their own artifact — v1.3 does, in
    `research/47_rematch_v13.py`, because a change to `p_make` on every kick is
    the kind of change document 05b §11's weather round re-earned Gate 1 for.
    """
    games = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / "game_components.parquet")
    dtw = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / dtw_artifact).select(
        "game_id", "deserved_margin", "dtw_home"
    )

    pairs = _power.rematch_pairs(games)

    # Recover game 1's id so the deserved margin can be joined on. `rematch_pairs`
    # keys on (season, sorted team pair), so rebuilding the key here is the same
    # join it used.
    keyed = (
        games.drop_nulls("margin")
        .with_columns(
            pl.min_horizontal("home_team", "away_team").alias("t1"),
            pl.max_horizontal("home_team", "away_team").alias("t2"),
        )
        .with_columns(
            pl.concat_str([pl.col("season").cast(pl.String), "t1", "t2"], separator="_").alias(
                "pair"
            )
        )
        .sort(["pair", "week"])
        .with_columns(pl.int_range(pl.len()).over("pair").alias("meeting"))
        .filter(pl.col("meeting") == 0)
        .select("pair", "game_id")
    )

    return pairs.join(keyed, on="pair", how="inner").join(dtw, on="game_id", how="inner")


def evaluate(
    actual: np.ndarray,
    deserved: np.ndarray,
    y: np.ndarray,
    a_home: np.ndarray,
    folds: np.ndarray,
    label: str,
) -> dict:
    """Gate 1 and Gate 2 on one pair of predictors."""
    per_pair = _power.paired_log_loss_diff(actual, deserved, y, a_home, folds)
    mean, se, superior = _power.decision(per_pair)
    ci = (mean - 1.96 * se, mean + 1.96 * se)
    non_inferior = _power.passes_noninferiority(mean, se)

    print(f"\n--- {label} ---")
    print(f"  mean delta log loss  {mean:+.5f}   SE {se:.5f}")
    print(f"  95% CI               [{ci[0]:+.5f}, {ci[1]:+.5f}]")
    print(
        f"  GATE 1 non-inferiority: {'PASS' if non_inferior else 'FAIL'} — "
        f"upper bound {ci[1]:+.5f} vs margin {NONINFERIORITY_MARGIN:+.3f}"
    )
    print(
        f"  GATE 2 direction: point estimate favours "
        f"{'deserve-to-win' if mean < 0 else 'the actual result'}; "
        f"superiority {'would reject' if superior else 'does not reject'} "
        f"(design power 7.2%, so this is not evidence either way)"
    )
    return {
        "label": label,
        "mean_delta_log_loss": mean,
        "se": se,
        "ci95": list(ci),
        "gate1_noninferiority_pass": bool(non_inferior),
        "gate2_direction_favours_deserved": bool(mean < 0),
        "superiority_would_reject": bool(superior),
    }


def coefficient_check(predictor: np.ndarray, y: np.ndarray, a_home: np.ndarray, label: str) -> dict:
    """Gate 3 — a bigger game-1 margin must predict a game-2 win.

    This tests the harness, not the hypothesis. If it fails, the pairing, the
    orientation or a sign convention is broken and nothing else here is readable.
    """
    x = _power.design_matrix(predictor, a_home)
    beta = _power.fit_logistic(x, y)

    p = 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -30, 30)))
    w = np.clip(p * (1.0 - p), 1e-9, None)
    covariance = np.linalg.inv(x.T @ (x * w[:, None]))
    se = float(np.sqrt(covariance[1, 1]))
    ci = (beta[1] - 1.96 * se, beta[1] + 1.96 * se)
    passed = bool(beta[1] > 0 and ci[0] > 0)

    print(
        f"  {label:22s} b1 = {beta[1]:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return {"label": label, "b1": float(beta[1]), "se": se, "ci95": list(ci), "pass": passed}


def main() -> None:
    paths.ensure_data_dirs()
    pairs = build_pairs()
    print(f"{pairs.height} rematch pairs with both predictors")

    actual = pairs["margin_g1_a"].to_numpy().astype(float)
    deserved = pairs["deserved_margin"].to_numpy().astype(float)
    dtw = pairs["dtw_home"].to_numpy().astype(float)
    y = (pairs["margin_g2_a"].to_numpy() > 0).astype(float)
    a_home = pairs["a_home_g2"].to_numpy().astype(float)
    margin_g2 = pairs["margin_g2_a"].to_numpy().astype(float)

    rng = np.random.default_rng(RANDOM_SEED)
    folds = rng.permutation(pairs.height) % N_FOLDS

    print(f"\n{'=' * 72}\nGATE 3 — coefficient sanity check\n{'=' * 72}")
    gate3 = [
        coefficient_check(actual, y, a_home, "game-1 actual margin"),
        coefficient_check(deserved, y, a_home, "game-1 deserved margin"),
    ]
    if not all(check["pass"] for check in gate3):
        print("\nGATE 3 FAILED — the harness is broken; no other number below is readable.")

    print(f"\n{'=' * 72}\nPRIMARY — predict the game-2 winner\n{'=' * 72}")
    primary = evaluate(actual, deserved, y, a_home, folds, "deserved margin vs actual margin")

    print(f"\n{'=' * 72}\nSECONDARY — predict the game-2 margin\n{'=' * 72}")
    per_pair = _power.cv_squared_error(
        _power.design_matrix(deserved, a_home), margin_g2, folds
    ) - _power.cv_squared_error(_power.design_matrix(actual, a_home), margin_g2, folds)
    mean, se, _ = _power.decision(per_pair)
    print(f"  mean delta MSE {mean:+.4f}  SE {se:.4f}  ")
    print(f"  95% CI [{mean - 1.96 * se:+.4f}, {mean + 1.96 * se:+.4f}]")
    secondary = {
        "mean_delta_mse": mean,
        "se": se,
        "ci95": [mean - 1.96 * se, mean + 1.96 * se],
        "favours_deserved": bool(mean < 0),
    }

    # Exploratory, NOT pre-registered. Labelled so it cannot be read as a
    # confirmatory result later.
    print(f"\n{'=' * 72}\nEXPLORATORY (not pre-registered) — DTW% as the predictor\n{'=' * 72}")
    exploratory = evaluate(actual, dtw, y, a_home, folds, "DTW% vs actual margin")

    results = {
        "n_pairs": pairs.height,
        "gate3_coefficient_check": gate3,
        "primary_winner": primary,
        "secondary_margin": secondary,
        "exploratory_dtw": exploratory,
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "random_seed": RANDOM_SEED,
    }
    out = paths.RESEARCH_OUTPUT_DIR / "08_rematch.json"
    with out.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
