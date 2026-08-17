"""Phase 3, step 5 — remediation for the Gate V-1 coverage failure.

`docs/research/10-interval-coverage.md` §4 pre-registered what to do when the
check failed in this specific way:

> V-1 fails, **over**-coverage, attributable to coin draws → *"Raise
> `n_coin_draws`, re-measure, and report the corrected coverage. Existing
> published numbers are not wrong, only wide."*

This is that re-measurement. It sweeps `n_coin_draws` at the full 4,000-game
sample and finds the smallest value whose **informative-game** coverage lands
inside the pre-registered [0.86, 0.92] band.

Informative games are the target because Gate V-3 established that a game whose
true DTW% is exactly 0 or 1 is covered trivially by a degenerate interval, and
those are about half the sample. Calibrating on the all-games figure would be
calibrating on an average that is half tautology.

    uv run python research/18_coverage_remediation.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_instrument = import_module("16_coverage_power")

from nfl_simulator import paths  # noqa: E402

RANDOM_SEED = 20260817
N_GAMES = 4000

NOMINAL = 0.89
BAND = (0.86, 0.92)

CANDIDATE_COIN_DRAWS = (100, 400, 800, 1600)


def main() -> None:
    paths.ensure_data_dirs()
    print(f"=== Coin-draw remediation sweep, {N_GAMES:,} games per setting ===")
    print(f"    target: informative-game coverage inside [{BAND[0]:.2f}, {BAND[1]:.2f}]\n")

    arms = []
    for coins in CANDIDATE_COIN_DRAWS:
        arm = _instrument.run_arm(
            f"{coins} coin draws",
            N_GAMES,
            RANDOM_SEED + coins,
            disable_layer_1=False,
            n_coin_draws=coins,
        )
        arm["informative_in_band"] = bool(BAND[0] <= arm["coverage_informative_only"] <= BAND[1])
        arms.append(arm)

    table = pl.DataFrame(
        [
            {
                "coin_draws": arm["n_coin_draws"],
                "coverage_all": round(arm["coverage"], 4),
                "coverage_informative": round(arm["coverage_informative_only"], 4),
                "mean_width": round(arm["mean_interval_width"], 4),
                "in_band": arm["informative_in_band"],
            }
            for arm in arms
        ]
    )
    print()
    with pl.Config(tbl_cols=-1):
        print(table)

    passing = [arm for arm in arms if arm["informative_in_band"]]
    recommended = min(passing, key=lambda a: a["n_coin_draws"])["n_coin_draws"] if passing else None

    print(f"\n{'=' * 72}")
    if recommended is None:
        print("No swept setting lands inside the band. The coin-draw count is not the fix.")
    else:
        chosen = next(a for a in arms if a["n_coin_draws"] == recommended)
        shipped = next(a for a in arms if a["n_coin_draws"] == 100)
        print(
            f"RECOMMENDED n_coin_draws = {recommended}\n"
            f"  informative coverage {chosen['coverage_informative_only']:.4f} "
            f"(was {shipped['coverage_informative_only']:.4f} at 100)\n"
            f"  mean interval width  {chosen['mean_interval_width']:.4f} "
            f"(was {shipped['mean_interval_width']:.4f} at 100, "
            f"{1 - chosen['mean_interval_width'] / shipped['mean_interval_width']:.1%} narrower)"
        )
    print(f"{'=' * 72}")

    out = paths.RESEARCH_OUTPUT_DIR / "18_coverage_remediation.json"
    with out.open("w") as handle:
        json.dump(
            {
                "n_games": N_GAMES,
                "nominal": NOMINAL,
                "band": list(BAND),
                "arms": arms,
                "recommended_coin_draws": recommended,
                "random_seed": RANDOM_SEED,
            },
            handle,
            indent=2,
        )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
