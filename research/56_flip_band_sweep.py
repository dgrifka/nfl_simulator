"""Product layer, round 2, item 3 — the flip-band robustness sweep.

Nothing is fitted, nothing is simulated. This reads the committed v1.3 summary
(`dtw_games_v13.parquet`) and re-labels all 2,761 games at sixteen candidate
"too close to call" bands, from an empty band up to 0.35-0.65, so the shipped
0.40-0.60 can be seen against its neighbours rather than asserted.

Document 33 §2a's counts at the shipped band are checked before anything is
drawn. They are computed a different way here — one label per game from
`plots.bucket_label`, rather than the audit's own arithmetic — so agreement is a
reproduction, not a copy.

    uv run python research/56_flip_band_sweep.py

Writes ``research/outputs/56_flip_band_sweep.png`` and
``research/outputs/56_flip_band_sweep.json``. Neither is committed —
``research/outputs/`` is gitignored, the script is the artifact and document 39
is the record of the numbers.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import polars as pl

from nfl_simulator import paths
from nfl_simulator.plots import BAND_HIGH, BAND_LOW, band_sweep, plot_band_sweep

GAMES_ARTIFACT = "dtw_games_v13.parquet"
RESULTS = "56_flip_band_sweep.json"
FIGURE = "56_flip_band_sweep.png"

# Document 33 §2a and §2, at the shipped band. The audit excluded the realized
# ties from its flip counts; this module labels them, so the tie-excluded count
# is the one that has to reproduce.
DOC_33 = {
    "too_close": 186,
    "clear_flip_ties_excluded": 195,
    "dtw_flips_no_band": 279,
    "n_games": 2761,
    "residual_disagreement_no_band": 56,
    "residual_disagreement_shipped": 7,
}


def residual_disagreement(deserved, actual, dtw, low, high) -> int:
    """Games the two flip definitions still disagree on once the band is applied.

    Document 33 §2a's argument for the band is that it dissolves the definition
    problem — 56 disagreements down to 7 — and the sweep is what shows that 7 is
    not a minimum the band was steered toward. Ties are excluded in both
    directions, as the audit excluded them.
    """
    inside = (dtw >= low) & (dtw <= high)
    live = (~inside) & (actual != 0)
    sign_flip = (deserved > 0) != (actual > 0)
    dtw_flip = (dtw > 0.5) != (actual > 0)
    return int((live & (sign_flip != dtw_flip)).sum())


def main() -> None:
    games = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / GAMES_ARTIFACT)
    dtw = games["dtw_home"].to_numpy()
    margin = games["actual_margin"].to_numpy()
    n_ties = int((margin == 0).sum())

    deserved = games["deserved_margin"].to_numpy()
    rows = band_sweep(dtw, margin)
    residuals = {
        row.half_width: residual_disagreement(deserved, margin, dtw, row.low, row.high)
        for row in rows
    }
    shipped = next(row for row in rows if (row.low, row.high) == (BAND_LOW, BAND_HIGH))
    no_band = next(row for row in rows if row.half_width == 0.0)

    print(f"{'=' * 72}\nREPRODUCTION — document 33 §2a at the shipped band\n{'=' * 72}")
    checks = {
        "n_games": (len(games), DOC_33["n_games"]),
        "too close to call": (shipped.too_close, DOC_33["too_close"]),
        "clear flips, ties excluded": (
            shipped.clear_flip - shipped.ties_outside_band,
            DOC_33["clear_flip_ties_excluded"],
        ),
        "DTW% flips with no band, ties excluded": (
            no_band.clear_flip - no_band.ties_outside_band,
            DOC_33["dtw_flips_no_band"],
        ),
        "definition disagreements, no band": (
            residuals[no_band.half_width],
            DOC_33["residual_disagreement_no_band"],
        ),
        "definition disagreements, shipped band": (
            residuals[shipped.half_width],
            DOC_33["residual_disagreement_shipped"],
        ),
    }
    for name, (here, published) in checks.items():
        status = "ok" if here == published else "MISMATCH"
        print(f"  {name:<40} {here:>6,}  vs doc 33 {published:>6,}  {status}")
    if any(here != published for here, published in checks.values()):
        raise SystemExit(
            "the sweep does not reproduce document 33's published counts. "
            "Stop and report rather than draw a robustness display over a disagreement."
        )
    print(f"\n  {n_ties} realized ties; {shipped.ties_outside_band} sit outside the shipped band")

    print(f"\n{'=' * 72}\nTHE SWEEP\n{'=' * 72}")
    print(
        f"  {'band':<14}{'clear flip':>12}{'too close':>12}{'holds':>10}"
        f"{'ties out':>10}{'disagree':>10}"
    )
    for row in rows:
        band = "0.50 only" if row.half_width == 0.0 else f"{row.low:.2f}-{row.high:.2f}"
        mark = "  <- shipped" if row is shipped else ""
        print(
            f"  {band:<14}{row.clear_flip:>12,}{row.too_close:>12,}"
            f"{row.scoreboard_holds:>10,}{row.ties_outside_band:>10}"
            f"{residuals[row.half_width]:>10}{mark}"
        )

    fig, _axes = plot_band_sweep(rows)
    figure_path = paths.RESEARCH_OUTPUT_DIR / FIGURE
    fig.savefig(figure_path, bbox_inches="tight")
    print(f"\n  wrote {figure_path.name}")

    results_path = paths.RESEARCH_OUTPUT_DIR / RESULTS
    results_path.write_text(
        json.dumps(
            {
                "artifact": GAMES_ARTIFACT,
                "n_games": len(games),
                "n_realized_ties": n_ties,
                "shipped_band": [BAND_LOW, BAND_HIGH],
                "doc_33_checks": {
                    name: {"here": here, "published": published}
                    for name, (here, published) in checks.items()
                },
                "sweep": [
                    asdict(row) | {"definition_disagreements": residuals[row.half_width]}
                    for row in rows
                ],
                "figure": FIGURE,
            },
            indent=2,
        )
    )
    print(f"  wrote {results_path.name}")


if __name__ == "__main__":
    main()
