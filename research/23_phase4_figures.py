"""Figures for the Phase 4 findings.

Palette, surface and rcParams are inherited unchanged from `research/04_figures.py`
— the validated 5-slot categorical set on surface #fcfcfb, which passes the
lightness, chroma, CVD-separation and normal-vision checks and carries a contrast
WARN. That WARN obligates visible labels, so **every mark in every figure here is
directly labelled and no chart relies on colour alone.**

Output is PNG for print, so there is no hover layer and no dark mode. One
measure per axis throughout; where two quantities of different scale belong
together they are drawn as two panels rather than two y-scales.

    uv run python research/23_phase4_figures.py
"""

from __future__ import annotations

import json
import re
import sys
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

_style = import_module("04_figures")

from nfl_simulator import paths  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = (
    _style.BLUE,
    _style.ORANGE,
    _style.AQUA,
    _style.YELLOW,
    _style.MAGENTA,
)
INK, INK_MUTED, GRID, SURFACE = _style.INK, _style.INK_MUTED, _style.GRID, _style.SURFACE
save, _finish = _style.save, _style._finish


def _load(name: str) -> dict:
    with (paths.RESEARCH_OUTPUT_DIR / name).open() as handle:
        return json.load(handle)


# --------------------------------------------------------------------------


def figure_drive_summaries(anatomy: dict) -> None:
    """How much of a drive a summary can see. Magnitude job -> paired horizontal bars.

    Two measures of different meaning — variance explained and between-team spread
    retained — so they get **two panels sharing a category axis**, never two
    y-scales on one plot.
    """
    labels = {
        "F0_cellmean": "Depth bins\n(document 08's instrument)",
        "F1_depth": "Depth",
        "F2_advance": "+ start, plays",
        "F3_production": "+ explosive, first downs,\nmax gain, penalty aid",
        "F4_yardage": "+ net yards\n(outcome-entangled)",
    }
    rows = [row for row in anatomy["nested_fits"] if row["feature_set"] in labels]
    names = [labels[row["feature_set"]] for row in rows]
    position = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, key, colour, title, note in (
        (axes[0], "oof_r2", BLUE, "Variance of drive points explained", "out-of-fold R²"),
        (
            axes[1],
            "share_of_spread_retained",
            ORANGE,
            "Between-team scoring spread retained",
            "1.00 = the summary sees every real difference between offenses",
        ),
    ):
        values = [row[key] for row in rows]
        ax.barh(position, values, height=0.62, color=colour)
        for y, value in zip(position, values, strict=True):
            ax.text(
                value + 0.012,
                y,
                f"{value:.0%}" if key != "oof_r2" else f"{value:.2f}",
                va="center",
                fontsize=9,
                color=INK,
            )
        ax.set_xlim(0, 1.12)
        _finish(ax, title, note)
    axes[0].set_yticks(position, names, fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle(
        "A richer drive summary was necessary — and not sufficient",
        x=0.005,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig9_drive_summaries")


def figure_channel_split(anatomy: dict) -> None:
    """Which scoring channel the finishing residual persists in. Magnitude + threshold."""
    rows = anatomy["exploratory_scoring_channels"]["rows"]
    labels = ["Points\n(all channels)", "Touchdown points\nonly", "Field-goal points\nonly"]
    colours = [YELLOW, AQUA, ORANGE]
    values = [row["split_half_r"] for row in rows]
    thresholds = [row["exploratory_null_p95"] for row in rows]

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    position = np.arange(len(rows))
    ax.bar(position, values, width=0.5, color=colours)
    for x, (value, threshold) in enumerate(zip(values, thresholds, strict=True)):
        ax.hlines(threshold, x - 0.32, x + 0.32, color=INK_MUTED, linewidth=2, zorder=3)
        ax.text(
            x, max(value, threshold) + 0.010, f"{value:+.3f}", ha="center", fontsize=10, color=INK
        )
        # The rightmost rule labels leftward so the text stays inside the axes.
        outward = x < len(values) - 1
        ax.text(
            x + (0.35 if outward else -0.35),
            threshold,
            f"noise floor {threshold:.3f}",
            va="center",
            ha="left" if outward else "right",
            fontsize=8,
            color=INK_MUTED,
        )
    ax.set_xticks(position, labels, fontsize=9)
    ax.set_ylabel("split-half correlation across a team's own season")
    ax.axhline(0, color=GRID, linewidth=1)
    ax.set_ylim(-0.02, 0.28)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.set_title(
        "Reaching the end zone is luck. Kicking a field goal is not.",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )
    ax.text(
        0,
        1.02,
        "Same drives, three valuations. A bar above its noise floor is a repeatable team property.",
        transform=ax.transAxes,
        fontsize=9,
        color=INK_MUTED,
        va="bottom",
    )
    fig.tight_layout()
    save(fig, "fig10_channel_split")


def figure_gate_blindness(power: dict) -> None:
    """What the rematch gate can see. Change-over-a-scale job -> a line with markers."""
    rows = power["skill_erasure_catch_rate"]
    erased = [row["spread_erased"] for row in rows]
    caught = [row["power_to_catch"] for row in rows]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(
        erased,
        caught,
        color=BLUE,
        linewidth=2,
        marker="o",
        markersize=8,
        markerfacecolor=BLUE,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        zorder=3,
    )
    ax.axhline(0.80, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(0.415, 0.815, "80% power", fontsize=9, color=INK_MUTED)

    for x, y in zip(erased, caught, strict=True):
        if y > 0.02 or x <= 0.10:
            ax.text(x, y + 0.045, f"{y:.0%}", ha="center", fontsize=9, color=INK)
    ax.annotate(
        "document 08's DQW%\nerased 29.4% — caught",
        xy=(0.294, 0.98),
        xytext=(0.255, 0.50),
        fontsize=9,
        color=INK_MUTED,
        arrowprops={"arrowstyle": "->", "color": INK_MUTED, "linewidth": 1},
    )
    ax.annotate(
        "a measure erasing a tenth of the\ndifference between NFL offenses\npasses every time",
        xy=(0.10, 0.0),
        xytext=(0.02, 0.30),
        fontsize=9,
        color=ORANGE,
        arrowprops={"arrowstyle": "->", "color": ORANGE, "linewidth": 1},
    )
    ax.set_xlabel("share of true between-team strength the measure erases")
    ax.set_ylabel("probability the non-inferiority gate catches it")
    ax.set_xlim(-0.01, 0.43)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4], ["0%", "10%", "20%", "30%", "40%"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    _finish(
        ax,
        "The validation gate is blind to exactly the failure it exists to catch",
        "It caught the Phase 3 measure because that measure was catastrophically bad.",
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    fig.tight_layout()
    save(fig, "fig11_gate_blindness")


def figure_sufficiency(successor: dict) -> None:
    """The successor's scorecard. Status job -> a labelled pass/fail row per criterion."""
    criteria = [
        (
            "SC-1  residual must not persist",
            successor["sc1_residual_persistence"]["split_half_r"],
            successor["sc1_residual_persistence"]["threshold"],
            "≤",
            "{:+.3f}",
            successor["sc1_residual_persistence"]["pass"],
        ),
        (
            "SC-2  spread must survive",
            successor["sc2_spread_retention"]["share_of_spread_retained"],
            successor["sc2_spread_retention"]["threshold"],
            "≥",
            "{:.1%}",
            successor["sc2_spread_retention"]["pass"],
        ),
        (
            "SC-3  excess quality correlation",
            successor["sc3_excess_correlation"]["excess"],
            successor["sc3_excess_correlation"]["threshold"],
            "≤",
            "{:+.3f}",
            successor["sc3_excess_correlation"]["pass"],
        ),
    ]
    sc4 = successor["arms"]["all drives"]["sc4_not_vacuous"]
    criteria.append(
        (
            "SC-4  must change some winners",
            sc4["flip_share"],
            sc4["min_flip_share"],
            "≥",
            "{:.1%}",
            sc4["pass"],
        )
    )

    fig, ax = plt.subplots(figsize=(9.4, 3.6))
    for i, (label, value, threshold, sense, fmt, passed) in enumerate(criteria):
        y = len(criteria) - 1 - i
        colour = AQUA if passed else ORANGE
        ax.scatter([0.30], [y], s=210, color=colour, zorder=3)
        ax.text(0.0, y, label, va="center", fontsize=10, color=INK)
        ax.text(
            0.345, y, f"{fmt.format(value)}", va="center", fontsize=10, fontweight="bold", color=INK
        )
        ax.text(
            0.46,
            y,
            f"needs {sense} {fmt.format(threshold)}",
            va="center",
            fontsize=9,
            color=INK_MUTED,
        )
        ax.text(
            0.70,
            y,
            "PASS" if passed else "FAIL",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=colour,
        )
    ax.set_xlim(-0.02, 0.85)
    ax.set_ylim(-0.6, len(criteria) - 0.4)
    ax.axis("off")
    ax.set_title(
        "The successor was stopped before it reached the gate it would have passed",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=22,
    )
    ax.text(
        0,
        1.03,
        "Sufficiency criteria, committed before the validation gate ran. "
        "Two of four fail, so the measure does not ship.",
        transform=ax.transAxes,
        fontsize=9,
        color=INK_MUTED,
        va="bottom",
    )
    fig.tight_layout()
    save(fig, "fig12_sufficiency_scorecard")


def figure_attribution(attribution: dict) -> None:
    """Whose skill is leverage timing? Identity job -> dot plot with 89% intervals."""
    rows = attribution["factors"]
    labels = [row["label"] for row in rows]
    # Percentage points of win probability, which is the readable unit; the raw
    # scale is thousandths and nobody can hold four leading zeros in their head.
    means = [row["mean"] * 100 for row in rows]
    lows = [row["eti89"][0] * 100 for row in rows]
    highs = [row["eti89"][1] * 100 for row in rows]
    bounds = [row["null_bound"] * 100 for row in rows]
    colours = [BLUE, ORANGE][: len(rows)]

    fig, ax = plt.subplots(figsize=(8.8, 3.4))
    position = np.arange(len(rows))
    for y, (low, high, mean, bound, colour) in enumerate(
        zip(lows, highs, means, bounds, colours, strict=True)
    ):
        ax.hlines(y, low, high, color=colour, linewidth=3)
        ax.scatter([mean], [y], s=150, color=colour, zorder=3, edgecolor=SURFACE, linewidth=2)
        ax.vlines(bound, y - 0.26, y + 0.26, color=INK_MUTED, linewidth=2)
        ax.text(mean, y - 0.34, f"{mean:.2f} pp", ha="center", fontsize=10, color=INK)
    # One noise-floor label, under the lower row, rather than one per row.
    ax.text(
        bounds[-1],
        len(rows) - 0.62,
        "noise floor",
        ha="center",
        va="top",
        fontsize=9,
        color=INK_MUTED,
    )
    ax.set_yticks(position, labels, fontsize=11)
    ax.set_ylim(len(rows) - 0.35, -0.6)
    ax.set_xlabel("win probability added per game, beyond what the points implied (pp)")
    _finish(
        ax,
        attribution["headline"],
        "89% intervals. A bar whose LEFT end clears the noise floor is an effect the design can confirm.",
    )
    fig.tight_layout()
    save(fig, "fig13_s3_attribution")


def figure_punting(special: dict) -> None:
    """Punting: what moves a punt. Magnitude job -> horizontal bars in net yards."""
    adopted = special["punting"]["arms"][special["punting"]["adopted_arm"]]
    roof = adopted["roof_effects"]
    entries = [
        ("A 15 mph wind", adopted["pu4_wind"]["beta_wind_mean"] * 15.0, ORANGE),
        ("A 40 °F colder day", -adopted["pu6_temperature"]["beta_temp_mean"] * 40.0, YELLOW),
        ("Indoors (dome)", roof["dome"]["mean"], AQUA),
        ("A one-SD punter", adopted["pu3_punter_skill"]["sigma_punter_mean"], BLUE),
    ]
    labels = [entry[0] for entry in entries]
    values = [entry[1] for entry in entries]
    colours = [entry[2] for entry in entries]

    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    position = np.arange(len(entries))
    ax.barh(position, values, height=0.6, color=colours)
    for y, value in zip(position, values, strict=True):
        offset = 0.08 if value >= 0 else -0.08
        ax.text(
            value + offset,
            y,
            f"{value:+.2f} yd",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=10,
            color=INK,
        )
    ax.set_yticks(position, labels, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color=GRID, linewidth=1)
    # Room for the outside-the-bar labels on both sides, so a negative value's
    # label never lands on the category axis.
    ax.set_xlim(min(values) - 0.95, max(values) + 0.55)
    ax.set_xlabel("change in net punt yards")
    _finish(
        ax,
        "Punter skill is bigger than the weather",
        "Net punt yards, from a hierarchical model conditioned on the kick situation.",
    )
    fig.tight_layout()
    save(fig, "fig14_punting")


def figure_returns(special: dict) -> None:
    """Return persistence by cell. Magnitude job with per-cell thresholds."""
    rows = [row for row in special["returns"]["cells"] if not row.get("skipped")]
    labels = [re.sub(r"\s*\(.*?\)", "", row["cell"]).replace(" / ", "\n") for row in rows]
    values = [row["split_half_r"] for row in rows]
    thresholds = [row["threshold"] for row in rows]
    colours = [AQUA if row["persists"] else ORANGE for row in rows]
    # Texture, not colour, carries the second dimension: a hatched bar sits in a
    # cell the design was pre-registered as underpowered for.
    hatches = ["" if row["power_at_reference_r"] >= 0.80 else "///" for row in rows]

    fig, ax = plt.subplots(figsize=(max(8.0, 1.7 * len(rows)), 4.4))
    position = np.arange(len(rows))
    ax.bar(position, values, width=0.5, color=colours, hatch=hatches, edgecolor=SURFACE)
    for x, (value, threshold) in enumerate(zip(values, thresholds, strict=True)):
        ax.hlines(threshold, x - 0.32, x + 0.32, color=INK_MUTED, linewidth=2, zorder=3)
        ax.text(
            x,
            value + (0.006 if value >= 0 else -0.018),
            f"{value:+.3f}",
            ha="center",
            fontsize=9,
            color=INK,
        )
    ax.set_xticks(position, labels, fontsize=7.5)
    ax.axhline(0, color=GRID, linewidth=1)
    ax.set_ylabel("split-half correlation, within season")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(
        "Return yardage: skill, or where the season's blocks happened to fall?",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=34,
    )
    ax.text(
        0,
        1.02,
        "Grey rules are each cell's own noise floor; hatched bars sit in cells the design "
        "was pre-registered as underpowered for.\nKickoff eras are never pooled — the 2024 "
        "and 2025 rule changes are structural breaks.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=INK_MUTED,
        va="bottom",
    )
    fig.tight_layout()
    save(fig, "fig15_returns")


def figure_bounce(special: dict) -> None:
    """Why the punt-roll bound is unusable. Polarity job -> a diverging bar chart.

    Two hues around a neutral zero, which is the diverging case: the sign is the
    finding. If the gap were the roll it would be positive everywhere.
    """
    rows = special["punt_bounce"]["matched_bins"]
    gaps = [row["gap"] for row in rows]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.bar(
        range(len(rows)),
        gaps,
        width=0.6,
        color=[AQUA if gap > 0 else ORANGE for gap in gaps],
    )
    ax.axhline(0, color=INK_MUTED, linewidth=1)
    ax.set_xticks(range(len(rows)), [str(row["spot_bin"]) for row in rows], fontsize=9)
    ax.set_xlabel("yards from the opponent's goal line when the punt was struck")
    ax.set_ylabel("extra yards a bouncing punt travels")
    ax.text(0.3, 3.3, "near midfield: bouncing punts go FURTHER", fontsize=9, color=AQUA)
    ax.text(
        len(rows) - 0.6,
        2.3,
        "backed up in your own end:\nthey go SHORTER",
        fontsize=9,
        color=ORANGE,
        ha="right",
        va="top",
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _finish(
        ax,
        "The aggregate bound on the punt roll is measuring intent, not physics",
        "If this were the roll it would be positive everywhere. It flips sign at midfield.",
    )
    fig.tight_layout()
    save(fig, "fig16_punt_bounce")


def main() -> None:
    paths.ensure_data_dirs()
    anatomy = _load("19_drive_anatomy.json")
    figure_drive_summaries(anatomy)
    figure_channel_split(anatomy)

    figure_gate_blindness(_load("20_dq_successor_power.json"))
    figure_sufficiency(_load("20_dq_successor.json"))

    attribution_path = paths.RESEARCH_OUTPUT_DIR / "21_s3_attribution.json"
    if attribution_path.exists():
        figure_attribution(_load("21_s3_attribution.json")["figure"])

    special_path = paths.RESEARCH_OUTPUT_DIR / "22_special_teams.json"
    if special_path.exists():
        special = _load("22_special_teams.json")
        figure_punting(special)
        figure_returns(special)
        figure_bounce(special)


if __name__ == "__main__":
    main()
