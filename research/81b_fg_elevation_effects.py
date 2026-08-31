"""Round 24, step 3 — the elevation effect in readable units, by distance.

Reads the posterior `research/81_fg_elevation.py` wrote; **fits nothing**. It
exists so the numbers document 67 quotes — "+4.09 pp at Denver at 45 yards" —
trace to a committed computation rather than to a transcript, which is the
`research-cadence` rule for any number that may end up in public content.

Gate E-3's own report already carries the 45-yard Denver figure. This adds the
distance ladder and Mexico City, both of which the round's deliverable asks for
and neither of which changes a gate.

    uv run python research/81b_fg_elevation_effects.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.data.stadium_elevation import elevation_kft  # noqa: E402

DISTANCE_CENTRE = 40.0
DISTANCES = (33.0, 45.0, 50.0, 55.0)
SITES = (
    ("sea level", 0.0),
    ("Denver", elevation_kft("DEN00")),
    ("Mexico City", elevation_kft("MEX00")),
)


def eti89(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 5.5)), float(np.percentile(values, 94.5))]


def main() -> None:
    with (paths.RESEARCH_OUTPUT_DIR / "81_fg_elevation.json").open() as handle:
        report = json.load(handle)
    elev_centre = report["elevation_centre_kft"]
    posterior = az.from_netcdf(paths.RESEARCH_OUTPUT_DIR / "trace_fg_elevation.nc")["posterior"]
    beta_elev = posterior["beta_elev"].values.ravel()

    print(
        f"beta_elev {beta_elev.mean():+.5f} "
        f"[{eti89(beta_elev)[0]:+.5f}, {eti89(beta_elev)[1]:+.5f}] log-odds per 1,000 ft"
    )
    print(f"elevation centre {elev_centre:.4f} kft — the elevation `alpha` already describes\n")

    rows = []
    for distance in DISTANCES:
        centred = distance - DISTANCE_CENTRE
        logit = (
            posterior["alpha"].values.ravel()
            + posterior["beta"].values.ravel() * centred
            + posterior["gamma"].values.ravel() * centred**2 / 100.0
            + posterior["delta_cubic"].values.ravel() * centred**3 / 1000.0
        )
        for label, kft in SITES:
            p_site = 1.0 / (1.0 + np.exp(-(logit + beta_elev * (kft - elev_centre))))
            p_centre = 1.0 / (1.0 + np.exp(-logit))
            shift = (p_site - p_centre) * 100.0
            rows.append(
                {
                    "distance_yd": distance,
                    "site": label,
                    "elevation_kft": kft,
                    "make_pct_at_centre": float(p_centre.mean() * 100),
                    "make_pct_at_site": float(p_site.mean() * 100),
                    "shift_pp": float(shift.mean()),
                    "shift_pp_eti89": eti89(shift),
                }
            )
        # The contrast a reader actually pictures: Denver against sea level.
        p_sea = 1.0 / (1.0 + np.exp(-(logit + beta_elev * (0.0 - elev_centre))))
        p_den = 1.0 / (1.0 + np.exp(-(logit + beta_elev * (elevation_kft("DEN00") - elev_centre))))
        contrast = (p_den - p_sea) * 100.0
        rows.append(
            {
                "distance_yd": distance,
                "site": "Denver minus sea level",
                "elevation_kft": elevation_kft("DEN00"),
                "make_pct_at_centre": float(p_sea.mean() * 100),
                "make_pct_at_site": float(p_den.mean() * 100),
                "shift_pp": float(contrast.mean()),
                "shift_pp_eti89": eti89(contrast),
            }
        )

    with pl.Config(tbl_rows=40, fmt_str_lengths=30):
        print(pl.DataFrame(rows))

    out = paths.RESEARCH_OUTPUT_DIR / "81b_fg_elevation_effects.json"
    with out.open("w") as handle:
        json.dump({"elevation_centre_kft": elev_centre, "effects": rows}, handle, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
