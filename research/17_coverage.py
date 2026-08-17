"""Phase 3, step 5 — the DTW interval-coverage check, against pre-registered gates.

Design, statistic and thresholds are fixed by `docs/research/10-interval-coverage.md`,
committed at `bf95345` before this script existed. Nothing here chooses anything.

    Gate V-1  coverage within 3 pp of the nominal 89%, at the shipped settings
    Gate V-2  which direction any miss runs — reported, no pass rule
    Gate V-3  coverage on informative games only — reported, no pass rule
    Gate V-4  is a miss attributable to Monte Carlo noise? — reported, diagnostic

    uv run python research/17_coverage.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

# The instrument's machinery, reused so the executed check is literally the
# characterized one rather than a re-implementation that could drift.
_instrument = import_module("16_coverage_power")

from nfl_simulator import paths  # noqa: E402

RANDOM_SEED = 20260817
N_GAMES = 4000

NOMINAL = 0.89
GATE_V1_TOLERANCE = 0.03  # pre-registered band [0.86, 0.92]

SHIPPED_COIN_DRAWS = 100
CONVERGED_COIN_DRAWS = 1600


def main() -> None:
    paths.ensure_data_dirs()
    print(f"=== DTW interval coverage, {N_GAMES:,} synthetic games ===")
    print(
        f"    nominal {NOMINAL:.0%}, Gate V-1 band "
        f"[{NOMINAL - GATE_V1_TOLERANCE:.2f}, {NOMINAL + GATE_V1_TOLERANCE:.2f}]\n"
    )

    shipped = _instrument.run_arm(
        f"shipped ({SHIPPED_COIN_DRAWS} coin draws)",
        N_GAMES,
        RANDOM_SEED,
        disable_layer_1=False,
        n_coin_draws=SHIPPED_COIN_DRAWS,
    )
    converged = _instrument.run_arm(
        f"converged ({CONVERGED_COIN_DRAWS} coin draws)",
        N_GAMES,
        RANDOM_SEED + 5,
        disable_layer_1=False,
        n_coin_draws=CONVERGED_COIN_DRAWS,
    )

    coverage = shipped["coverage"]
    low, high = NOMINAL - GATE_V1_TOLERANCE, NOMINAL + GATE_V1_TOLERANCE
    v1_pass = bool(low <= coverage <= high)
    print(f"\n{'=' * 72}\nGATE V-1 — is the interval calibrated?\n{'=' * 72}")
    print(
        f"  coverage {coverage:.4f} (Monte Carlo SE {shipped['monte_carlo_se']:.4f}) "
        f"vs band [{low:.2f}, {high:.2f}]: {'PASS' if v1_pass else 'FAIL'}"
    )

    direction = "over-coverage" if coverage > NOMINAL else "under-coverage"
    print(f"\n{'=' * 72}\nGATE V-2 — which direction? (reported)\n{'=' * 72}")
    print(f"  coverage - nominal = {coverage - NOMINAL:+.4f}  ->  {direction}")
    if direction == "over-coverage":
        print(
            "  Per the pre-registered asymmetry: the simulator UNDERSTATES its own\n"
            "  confidence. The interval is mislabelled, not misleading — no reader is\n"
            "  invited to over-trust the number."
        )
    else:
        print(
            "  Per the pre-registered asymmetry: the simulator OVERCLAIMS. This is the\n"
            "  serious direction, and intervals must not be reported until it is fixed."
        )

    print(f"\n{'=' * 72}\nGATE V-3 — informative games only (reported)\n{'=' * 72}")
    print(
        f"  all games       {coverage:.4f} on {shipped['n_games']:,}\n"
        f"  informative     {shipped['coverage_informative_only']:.4f} on "
        f"{shipped['n_informative']:,} (true DTW% strictly between 0 and 1)"
    )

    print(f"\n{'=' * 72}\nGATE V-4 — is any miss attributable? (diagnostic)\n{'=' * 72}")
    width_shipped = shipped["mean_interval_width"]
    width_converged = converged["mean_interval_width"]
    monte_carlo_share = 1.0 - width_converged / width_shipped
    print(
        f"  mean interval width  {width_shipped:.4f} at {SHIPPED_COIN_DRAWS} coin draws\n"
        f"                       {width_converged:.4f} at {CONVERGED_COIN_DRAWS} coin draws"
    )
    print(f"  share of shipped width that is Monte Carlo noise: {monte_carlo_share:.1%}")
    print(
        f"  coverage at {CONVERGED_COIN_DRAWS} coin draws: {converged['coverage']:.4f} "
        f"(informative {converged['coverage_informative_only']:.4f})"
    )

    verdict = (
        "calibrated"
        if v1_pass
        else (
            "over-covering, attributable to Monte Carlo noise in the coin draws"
            if direction == "over-coverage" and monte_carlo_share > 0.10
            else f"{direction}, not attributable to the coin-draw count"
        )
    )
    print(f"\n{'=' * 72}\nVERDICT: {verdict}\n{'=' * 72}")

    results = {
        "n_games": N_GAMES,
        "nominal": NOMINAL,
        "gate_v1_band": [low, high],
        "gate_v1_pass": v1_pass,
        "gate_v2_direction": direction,
        "shipped_arm": shipped,
        "converged_arm": converged,
        "monte_carlo_share_of_width": monte_carlo_share,
        "verdict": verdict,
        "random_seed": RANDOM_SEED,
    }
    with pl.Config(tbl_cols=-1):
        print(
            pl.DataFrame(
                [
                    {k: v for k, v in arm.items() if k != "coverage_ci95"}
                    for arm in (shipped, converged)
                ]
            )
        )

    out = paths.RESEARCH_OUTPUT_DIR / "17_coverage.json"
    with out.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
