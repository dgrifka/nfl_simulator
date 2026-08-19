"""Phase 8, task 4 — the rematch validation, re-run on simulator v1.3.

Document 27 §10 committed to re-running document 06's validation if the refit was
adopted, on document 05b §11's precedent: the weather round moved 47 games'
verdicts and re-earned Gate 1 rather than inheriting it. v1.3 moves fewer, and
it moves every kick's price.

**This is reported as a check, not a gate.** Document 12 measured the rematch
test as nearly blind below roughly 20% damage, so a pass proves little and is
never the sole evidence for anything. The statistic, the folds, the
non-inferiority margin and the seed are document 06's, untouched — only the
deserve-to-win artifact changes.

    uv run python research/47_rematch_v13.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

_rematch = import_module("08_rematch")
_power = import_module("08_rematch_power")

from nfl_simulator import paths  # noqa: E402

ARTIFACT = "dtw_games_v13.parquet"

# Document 07's published result, on v1.0's artifact. Printed beside v1.3's so
# the movement is visible rather than asserted.
DOCUMENT_07 = {
    "n_pairs": 531,
    "mean_delta_log_loss": -0.0015884,
    "ci95_upper": +0.0037710,
}


def main() -> None:
    paths.ensure_data_dirs()
    pairs = _rematch.build_pairs(ARTIFACT)
    print(f"{pairs.height} rematch pairs with both predictors, on {ARTIFACT}")

    actual = pairs["margin_g1_a"].to_numpy().astype(float)
    deserved = pairs["deserved_margin"].to_numpy().astype(float)
    dtw = pairs["dtw_home"].to_numpy().astype(float)
    y = (pairs["margin_g2_a"].to_numpy() > 0).astype(float)
    a_home = pairs["a_home_g2"].to_numpy().astype(float)
    margin_g2 = pairs["margin_g2_a"].to_numpy().astype(float)

    rng = np.random.default_rng(_rematch.RANDOM_SEED)
    folds = rng.permutation(pairs.height) % _rematch.N_FOLDS

    print(
        f"\n{'=' * 72}\nHARNESS CHECK — a bigger game-1 margin must predict a game-2 win\n{'=' * 72}"
    )
    harness = [
        _rematch.coefficient_check(actual, y, a_home, "game-1 actual margin"),
        _rematch.coefficient_check(deserved, y, a_home, "game-1 deserved margin"),
    ]
    if not all(check["pass"] for check in harness):
        print("\nHARNESS FAILED — nothing below is readable.")

    print(f"\n{'=' * 72}\nPRIMARY — predict the game-2 winner (CHECK, not a gate)\n{'=' * 72}")
    primary = _rematch.evaluate(
        actual, deserved, y, a_home, folds, "v1.3 deserved margin vs actual margin"
    )

    print(f"\n{'=' * 72}\nSECONDARY — predict the game-2 margin\n{'=' * 72}")
    per_pair = _power.cv_squared_error(
        _power.design_matrix(deserved, a_home), margin_g2, folds
    ) - _power.cv_squared_error(_power.design_matrix(actual, a_home), margin_g2, folds)
    mean, se, _ = _power.decision(per_pair)
    print(f"  mean delta MSE {mean:+.4f}  SE {se:.4f}")
    print(f"  95% CI [{mean - 1.96 * se:+.4f}, {mean + 1.96 * se:+.4f}]")

    print(f"\n{'=' * 72}\nEXPLORATORY (not pre-registered) — DTW% as the predictor\n{'=' * 72}")
    exploratory = _rematch.evaluate(actual, dtw, y, a_home, folds, "v1.3 DTW% vs actual margin")

    print(
        f"\n  Document 07, on v1.0: {DOCUMENT_07['n_pairs']} pairs, mean delta log loss "
        f"{DOCUMENT_07['mean_delta_log_loss']:+.5f}, 95% upper "
        f"{DOCUMENT_07['ci95_upper']:+.5f}."
        f"\n  v1.3: {pairs.height} pairs, {primary['mean_delta_log_loss']:+.5f}, upper "
        f"{primary['ci95'][1]:+.5f}."
        f"\n  Document 12 measured this test as nearly blind below ~20% damage, so agreement\n"
        f"  here is weak evidence and is reported as such. It is a check, not a gate."
    )

    results = {
        "artifact": ARTIFACT,
        "reported_as": "check, not a gate (document 12's blindness caveat)",
        "n_pairs": pairs.height,
        "harness_check": harness,
        "primary_winner": primary,
        "secondary_margin": {
            "mean_delta_mse": mean,
            "se": se,
            "ci95": [mean - 1.96 * se, mean + 1.96 * se],
            "favours_deserved": bool(mean < 0),
        },
        "exploratory_dtw": exploratory,
        "document_07_on_v1": DOCUMENT_07,
        "noninferiority_margin": _rematch.NONINFERIORITY_MARGIN,
        "random_seed": _rematch.RANDOM_SEED,
    }
    out = paths.RESEARCH_OUTPUT_DIR / "47_rematch_v13.json"
    with out.open("w") as handle:
        json.dump(results, handle, indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
