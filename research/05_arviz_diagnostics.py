"""ArviZ diagnostic figures for the hierarchical rate models.

Two things the numeric gate report in `docs/research/04-bayesian-results.md`
states but does not draw:

* a **forest plot of team posteriors**, which makes the shrinkage visible as a
  stack of near-identical intervals rather than a table of numbers;
* a **posterior predictive check**, so the Gate 4 pass has a picture behind it.

The fitted models are marginalized, so `p_team` was never sampled. It is
reconstructed here by conjugacy — given (mu, kappa), a team's exact posterior is
``Beta(mu*kappa + k, (1-mu)*kappa + n - k)`` — which is the same distribution the
centered model would have sampled, without the funnel.

    uv run python research/05_arviz_diagnostics.py
"""

from __future__ import annotations

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import xarray as xr

from nfl_simulator import paths
from nfl_simulator.ingest import ANALYSIS_COLUMNS, PBP_SEASONS, load_pbp
from nfl_simulator.rates import fumble_recovery_counts

RANDOM_SEED = 20260817
N_SHOWN = 14  # 320 intervals is a smear; the extremes carry the point


def team_posterior_draws(counts: pl.DataFrame, idata) -> np.ndarray:
    """Draws of each team's true rate, by conjugacy. Shape (draws, teams)."""
    rng = np.random.default_rng(RANDOM_SEED)
    mu = idata["posterior"]["mu"].values.ravel()[:, None]
    kappa = idata["posterior"]["kappa"].values.ravel()[:, None]
    n = counts["n"].to_numpy()[None, :]
    k = counts["k"].to_numpy()[None, :]
    return rng.beta(mu * kappa + k, (1.0 - mu) * kappa + (n - k))


def figure_forest(counts: pl.DataFrame, idata) -> None:
    draws = team_posterior_draws(counts, idata)

    # Show the most extreme observed seasons — if even those are pulled to the
    # league rate, nothing in the middle can be doing anything interesting.
    observed = counts["k"].to_numpy() / counts["n"].to_numpy()
    order = np.argsort(observed)
    picked = np.concatenate([order[: N_SHOWN // 2], order[-N_SHOWN // 2 :]])

    labels = [
        f"{counts['team_season'][int(i)]}  ({counts['k'][int(i)]}/{counts['n'][int(i)]}"
        f" = {observed[i]:.0%})"
        for i in picked
    ]
    subset = xr.Dataset(
        {
            "true_recovery_rate": (
                ("chain", "draw", "team_season"),
                draws[:, picked].reshape(1, draws.shape[0], len(picked)),
            )
        },
        coords={"chain": [0], "draw": np.arange(draws.shape[0]), "team_season": labels},
    )
    tree = xr.DataTree.from_dict({"posterior": subset})

    plot = az.plot_forest(tree, var_names=["true_recovery_rate"], combined=True)
    fig = plot.viz["figure"].item()

    # Only one variable is plotted, so ArviZ's variable label is redundant — and it
    # lands mid-column where it overlaps a team name. `visuals` has no key for it,
    # so drop the artist directly.
    axes = plot.viz["plot"].values.ravel()
    for axis in axes:
        for text in list(axis.texts):
            if text.get_text() in {"true_recovery_rate", "variable", "team_season"}:
                text.remove()
    # The label column also carries dimension-name ticks along its bottom edge,
    # which read as stray words under the team names.
    for axis in axes[:-1]:
        axis.set_xticks([])
    # ArviZ 1.x forest plots are two panels — a labels column and the intervals.
    # The intervals are the last one; `.item()` would raise on the pair.
    ax = plot.viz["plot"].values.ravel()[-1]
    league = float(idata["posterior"]["mu"].values.mean())
    ax.axvline(league, color="#e87ba4", linewidth=1.6, zorder=1)
    ax.text(
        league + 0.002,
        ax.get_ylim()[1],
        f"league rate {league:.1%}",
        fontsize=8,
        color="#e87ba4",
        va="top",
    )
    ax.set_xlabel("Posterior for the team's true fumble-recovery rate")
    ax.set_title(
        "The 14 most extreme fumble-recovery seasons, after shrinkage",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.set_size_inches(7.5, 5.0)
    path = paths.RESEARCH_OUTPUT_DIR / "fig7_forest_fumble.png"
    fig.savefig(path, bbox_inches="tight", dpi=160, facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def figure_ppc(idata) -> None:
    plot = az.plot_ppc_dist(idata, var_names=["successes"], kind="ecdf", num_samples=80)
    fig = plot.viz["figure"].item()
    # plot_ppc_dist's viz["plot"] is a DataTree keyed by variable, not the flat
    # DataArray that plot_forest returns — take the axis off the figure instead.
    axis = fig.axes[-1]

    # ArviZ ships the panel bare. Without these, dark-vs-light is unexplained and
    # the x axis has no units — the reader cannot tell what passed.
    axis.set_xlabel("Fumbles recovered by a team in a season")
    axis.set_ylabel("Cumulative share of team-seasons")
    axis.text(
        0.97,
        0.30,
        "black = what happened\nblue = 80 draws from the model",
        transform=axis.transAxes,
        fontsize=8.5,
        color="#52514e",
        ha="right",
    )
    fig.suptitle(
        "Posterior predictive check: fumble recovery counts per team-season",
        fontsize=11,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    fig.set_size_inches(7.0, 4.2)
    path = paths.RESEARCH_OUTPUT_DIR / "fig8_ppc_fumble.png"
    fig.savefig(path, bbox_inches="tight", dpi=160, facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    paths.ensure_data_dirs()
    pbp = load_pbp(PBP_SEASONS, columns=ANALYSIS_COLUMNS)
    counts = fumble_recovery_counts(pbp, exclude_aborted=True)
    idata = az.from_netcdf(paths.RESEARCH_OUTPUT_DIR / "trace_fumble_recovery.nc")

    print(az.summary(idata, var_names=["mu", "kappa", "population_sd"]))
    figure_forest(counts, idata)
    figure_ppc(idata)


if __name__ == "__main__":
    main()
