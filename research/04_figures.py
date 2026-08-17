"""Figures for the Phase 1 findings.

Palette is the validated 5-slot categorical set (light mode, surface #fcfcfb).
It passes the lightness, chroma, CVD-separation and normal-vision checks and
carries a contrast WARN, which obligates visible labels — so every mark in every
figure here is directly labelled and no chart relies on colour alone.

Output is PNG for print, so there is no hover layer and no dark mode.

    uv run python research/04_figures.py
"""

from __future__ import annotations

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from nfl_simulator import paths

# Validated categorical slots, assigned in fixed order and never cycled.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"

mpl.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_MUTED,
        "axes.edgecolor": GRID,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
    }
)


def _finish(ax, title: str, subtitle: str | None = None) -> None:
    """Recessive grid, title above, subtitle in muted ink."""
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=18 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK_MUTED, va="bottom")


def save(fig, name: str) -> None:
    path = paths.RESEARCH_OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# --------------------------------------------------------------------------


def figure_variance_shares(eda: dict) -> None:
    """Where a game's outcome variance lives. Magnitude job -> horizontal bars."""
    shares = pl.DataFrame(eda["variance"]["epa_diff_shares"]).with_columns(
        pl.col("share").cast(pl.Float64)
    )
    labels = {
        "core": "Core offense / defense",
        "interception": "Interceptions",
        "penalty": "Penalties",
        "fumble_luck": "Fumble recovery luck",
        "fg_luck": "Field goal luck",
    }
    shares = shares.filter(pl.col("component").is_in(list(labels))).sort("share")
    colors = {
        "core": BLUE,
        "interception": ORANGE,
        "penalty": AQUA,
        "fumble_luck": MAGENTA,
        "fg_luck": YELLOW,
    }

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    y = np.arange(shares.height)
    values = shares["share"].to_numpy() * 100
    ax.barh(
        y,
        values,
        color=[colors[c] for c in shares["component"]],
        height=0.62,
        edgecolor=SURFACE,
        linewidth=2,  # 2px surface gap between adjacent bars
    )
    for index, value in enumerate(values):
        ax.text(value + 1.0, index, f"{value:.1f}%", va="center", fontsize=10, color=INK)
    ax.set_yticks(y, [labels[c] for c in shares["component"]])
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_xlabel("Share of variance in game EPA differential")
    _finish(
        ax,
        "Only 7% of a game's outcome is coin-flip luck",
        "2,761 games, 2016-2025. Shares sum to 100% by covariance decomposition.",
    )
    save(fig, "fig1_variance_shares")


def figure_fumble_classes(eda: dict) -> None:
    """Recovery rate by fumble class. The 50% reference line is the point."""
    classes = pl.DataFrame(eda["baselines"]["fumble_classes"]).with_columns(
        pl.col("n").cast(pl.Int64), pl.col("p_own").cast(pl.Float64)
    )
    classes = classes.filter(pl.col("n") >= 100).sort("p_own")
    pretty = {
        "pass/live": "Pass play",
        "run/live": "Run play",
        "run/aborted": "Botched snap (run)",
        "punt/live": "Muffed punt",
        "kickoff/live": "Kickoff",
    }

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    y = np.arange(classes.height)
    values = classes["p_own"].to_numpy() * 100
    # Aborted snaps are the finding, so they get the accent slot.
    colors = [ORANGE if "aborted" in c else BLUE for c in classes["fumble_class"]]
    ax.barh(y, values, color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)
    # The 50% rule crosses several value labels. Keep the rule behind the text and
    # give each label a surface-coloured backing so both stay readable.
    ax.axvline(50, color=INK_MUTED, linewidth=1.5, linestyle=(0, (4, 3)), zorder=2)
    for index, (value, count) in enumerate(zip(values, classes["n"].to_numpy(), strict=True)):
        ax.text(
            value + 1.2,
            index,
            f"{value:.1f}%  (n={count:,})",
            va="center",
            fontsize=9.5,
            color=INK,
            zorder=5,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5},
        )
    ax.text(50.8, -0.72, "a true coin flip", fontsize=9, color=INK_MUTED)
    ax.set_yticks(y, [pretty.get(c, c) for c in classes["fumble_class"]])
    ax.set_ylim(-1.0, classes.height - 0.4)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share recovered by the fumbling team")
    _finish(
        ax,
        "The fumble coin is not 50/50 — and it is a different coin per play type",
        "5,914 live fumbles, 2016-2025. Botched snaps come back to the offense three times in four.",
    )
    save(fig, "fig2_fumble_classes")


def figure_persistence(skill: dict) -> None:
    """Split-half r per component. Identity job with uncertainty -> dot + interval."""
    rows = pl.DataFrame(skill["persistence"]).with_columns(
        pl.col("split_half_r").cast(pl.Float64),
        pl.col("r_p05").cast(pl.Float64),
        pl.col("r_p95").cast(pl.Float64),
    )
    labels = {
        "epa_diff": "EPA differential (total)",
        "core": "Core offense / defense",
        "interception": "Interceptions",
        "fg_luck": "Field goal results",
        "penalty": "Penalties",
        "fumble_luck": "Fumble recovery",
    }
    rows = rows.filter(pl.col("metric").is_in(list(labels))).sort("split_half_r")

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    y = np.arange(rows.height)
    centers = rows["split_half_r"].to_numpy()
    low = centers - rows["r_p05"].to_numpy()
    high = rows["r_p95"].to_numpy() - centers
    # Fumble recovery is the calibration case, so it carries the accent.
    colors = [MAGENTA if m == "fumble_luck" else BLUE for m in rows["metric"]]

    ax.errorbar(
        centers, y, xerr=[low, high], fmt="none", ecolor=INK_MUTED, elinewidth=1.6, capsize=4
    )
    ax.scatter(centers, y, s=95, color=colors, zorder=4, edgecolor=SURFACE, linewidth=2)
    for index, value in enumerate(centers):
        ax.text(value, index + 0.30, f"{value:+.3f}", ha="center", fontsize=9.5, color=INK)
    ax.axvline(0, color=INK_MUTED, linewidth=1.5, linestyle=(0, (4, 3)))
    ax.text(0.008, -0.62, "no team skill", fontsize=9, color=INK_MUTED)
    ax.set_yticks(y, [labels[m] for m in rows["metric"]])
    ax.set_ylim(-0.9, rows.height - 0.3)
    ax.set_xlabel("Split-half correlation within a team-season (200 random splits)")
    _finish(
        ax,
        "Fumble recovery is the only component with no team skill in it",
        "320 team-seasons. Bars are the 5th-95th percentile across random splits.",
    )
    save(fig, "fig3_persistence")


def figure_population_sd(rates: dict) -> None:
    """Team spread in each rate, as a share of that rate's league average.

    Plotting raw percentage points here would mislead badly: penalty rates have a
    base of 1-2% while fumble recovery has a base of 47%, so a 0.25pp penalty
    spread and a 2.4pp fumble spread are not remotely the same claim. Dividing by
    the league rate puts all four on one comparable axis — the only honest way to
    stack them in a single chart.
    """
    labels = {
        "fumble_recovery": "Fumble recovery",
        "interception_conversion": "INT-worthy throw becomes an INT",
        "penalty_pre_snap": "Penalties, pre-snap",
        "penalty_judgment": "Penalties, judgment calls",
    }
    rows = []
    for key, label in labels.items():
        result = rates[key]
        base = float(result["observed_rate"])
        rows.append(
            {
                "label": label,
                "mean": float(result["population_sd_mean"]) / base * 100,
                "low": float(result["population_sd_eti89"][0]) / base * 100,
                "high": float(result["population_sd_eti89"][1]) / base * 100,
                "absolute_pp": float(result["population_sd_mean"]) * 100,
                "base_pct": base * 100,
            }
        )
    table = pl.DataFrame(rows).sort("mean")

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    y = np.arange(table.height)
    centers = table["mean"].to_numpy()
    colors = [MAGENTA if "Fumble" in label else BLUE for label in table["label"]]
    ax.errorbar(
        centers,
        y,
        xerr=[centers - table["low"].to_numpy(), table["high"].to_numpy() - centers],
        fmt="none",
        ecolor=INK_MUTED,
        elinewidth=1.6,
        capsize=4,
    )
    ax.scatter(centers, y, s=95, color=colors, zorder=4, edgecolor=SURFACE, linewidth=2)
    for index, row in enumerate(table.iter_rows(named=True)):
        ax.text(
            row["high"] + 0.5,
            index,
            f"{row['mean']:.1f}%   ({row['absolute_pp']:.2f} pp on a "
            f"{row['base_pct']:.1f}% base rate)",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.set_yticks(y, table["label"].to_list())
    ax.set_xlim(0, table["high"].max() * 2.6)
    ax.set_xlabel("Spread of true team rates, as a share of the league average (89% interval)")
    _finish(
        ax,
        "Fumble recovery has three times less team variation than anything else",
        "Hierarchical beta-binomial posteriors. Larger = teams genuinely differ more.",
    )
    save(fig, "fig4_population_sd")


def figure_shrinkage() -> None:
    """Observed vs shrunk fumble-recovery rate. Identity line makes shrinkage visible."""
    counts = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / "fumble_shrinkage.parquet")

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    observed = counts["observed_rate"].to_numpy() * 100
    shrunk = counts["posterior_mean"].to_numpy() * 100
    ax.plot([0, 100], [0, 100], color=INK_MUTED, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
    # Label must sit inside the y-limits — placing it on the identity line beyond
    # them silently expands the axes and collides the title with its subtitle.
    ax.annotate(
        "if the season rate were the true rate",
        xy=(60, 60),
        xytext=(64, 58),
        fontsize=9,
        color=INK_MUTED,
        arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 1},
    )
    ax.scatter(
        observed,
        shrunk,
        s=np.clip(counts["n"].to_numpy() * 4, 12, 130),
        color=BLUE,
        alpha=0.55,
        edgecolor=SURFACE,
        linewidth=1.2,
        zorder=3,
    )
    ax.axhline(46.8, color=MAGENTA, linewidth=1.6, zorder=4)
    ax.text(3, 47.8, "league rate, 46.8%", fontsize=9, color=MAGENTA)
    ax.set_xlim(0, 100)
    ax.set_ylim(30, 65)
    ax.set_xlabel("Observed recovery rate that season (%)")
    ax.set_ylabel("Posterior estimate of the team's true rate (%)")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(
        "A team that recovered 10 of 12 is not an 83% recovery team",
        loc="left",
        fontsize=12,
        fontweight="bold",
        pad=30,  # two-line subtitle needs the extra room
    )
    ax.text(
        0,
        1.015,
        "320 team-seasons. Marker size is fumbles that season; the model pulls all of them\n"
        "back to the league rate because none has enough evidence to move.",
        transform=ax.transAxes,
        fontsize=9,
        color=INK_MUTED,
        va="bottom",
    )
    save(fig, "fig5_shrinkage")


def figure_prediction(skill: dict) -> None:
    """Out-of-sample log loss. Lower is better; Vegas is the only real gap."""
    rows = pl.DataFrame(skill["prediction"]).with_columns(pl.col("log_loss").cast(pl.Float64))
    rows = rows.sort("log_loss", descending=True)

    fig, ax = plt.subplots(figsize=(7.5, 3.3))
    y = np.arange(rows.height)
    values = rows["log_loss"].to_numpy()
    colors = [AQUA if "Vegas" in m else (BLUE if "raw" in m else ORANGE) for m in rows["model"]]

    # Dots, not bars. Log loss has no meaningful zero, so a bar would need a
    # truncated baseline — and a truncated bar exaggerates gaps that the paired
    # bootstrap says are noise. Dots carry no baseline claim.
    baseline = float(rows.filter(pl.col("model") == "raw EPA differential")["log_loss"].item())
    ax.axvline(baseline, color=BLUE, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.hlines(y, values.min() - 0.01, values, color=GRID, linewidth=1.4, zorder=1)
    ax.scatter(values, y, s=110, color=colors, zorder=4, edgecolor=SURFACE, linewidth=2)
    for index, value in enumerate(values):
        ax.text(value + 0.0025, index, f"{value:.4f}", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y, rows["model"].to_list())
    ax.set_xlim(values.min() - 0.012, values.max() + 0.012)
    ax.set_xlabel("Out-of-sample log loss, 2024-2025 (lower is better)")
    _finish(
        ax,
        "Stripping luck did not improve prediction",
        "569 test games. Every stripped variant is within noise of raw EPA; only Vegas separates.",
    )
    save(fig, "fig6_prediction")


def main() -> None:
    paths.ensure_data_dirs()
    with (paths.RESEARCH_OUTPUT_DIR / "01_descriptive_eda.json").open() as handle:
        eda = json.load(handle)
    with (paths.RESEARCH_OUTPUT_DIR / "02_skill_vs_luck.json").open() as handle:
        skill = json.load(handle)
    with (paths.RESEARCH_OUTPUT_DIR / "03_bayesian_rates.json").open() as handle:
        rates = json.load(handle)

    figure_variance_shares(eda)
    figure_fumble_classes(eda)
    figure_persistence(skill)
    figure_population_sd(rates)
    figure_shrinkage()
    figure_prediction(skill)


if __name__ == "__main__":
    main()
