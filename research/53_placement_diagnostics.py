"""Placement meter, pre-ship diagnostics — anti-persistence and seed robustness.

**Post-hoc, and nothing here moves a threshold.** Document 36's six gates ran on
2026-08-20 and their verdicts are fixed in §11. This script answers the two
questions the maintainer queued after reading §11f and §11g, both of which change *what
the product says* rather than *whether the meter ships*:

1. **Is the −0.0986 split-half correlation mechanical?** §11f's candidate cause
   is that ``s0_loo`` — the quality baseline each team-game is centred against —
   is computed leave-one-*game*-out across the whole team-season, so a game in
   split-half A enters the baseline of every game in split-half B. That is a
   cross-half dependence with a negative sign, and the full-pipeline simulation
   in ``51_placement_redesign_power.py`` could not carry it: its per-game truth
   is constant within a season, so its games differ only by sampling noise.

   Five arms, all scored by the same production code, differing only in which
   plays the quality baseline is allowed to see:

   | arm | ``s0`` for a game in half A | cross-half channel |
   |---|---|---|
   | ``shipped`` | season minus this game | present (the shipped score) |
   | ``within_half`` | half A minus this game | **cut** |
   | ``cross_half`` | all of half B | **amplified, sign flipped** |
   | ``shared_half`` | a fixed half-sized subsample, minus this game | present, at ``within_half``'s noise |
   | ``other_seasons`` | this team's other seasons | absent entirely |

   ``shared_half`` is the control that matters: ``within_half`` halves the
   baseline's sample size as a side effect, so a move from ``shipped`` to
   ``within_half`` could be the noise rather than the channel. ``shared_half``
   carries the same noise with the channel left in.

2. **Does the pipeline null move if its truth varies within a season?** The
   ``none`` arm of document 36 §7 put the split-half null at +0.0038 ± 0.0390.
   Rerun with per-game truth drawn around the team-season mean at the real
   within-season spread, which is measured here rather than assumed.

3. **Seed robustness.** M-3's r on four further split seeds, and the adopted
   rung's M-2 coverage on three further band seeds — the fresh-seed arm document
   36 §11g flagged as reported-beside, promoted to a read of its own.

Run one part at a time; each prints and saves on its own:

    uv run python research/53_placement_diagnostics.py --part persistence
    uv run python research/53_placement_diagnostics.py --part pipeline
    uv run python research/53_placement_diagnostics.py --part seeds

Results land in ``research/outputs/53_placement_diagnostics.json``, which is
gitignored; this script is the artifact and document 36 §12 is the record.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_power = import_module("49_placement_power")
_redesign_power = import_module("51_placement_redesign_power")
_ship = import_module("52_placement_redesign")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.placement import (  # noqa: E402
    expected_profile,
    leave_one_out_rate,
    redesigned_cell_points,
)

RESULTS = "53_placement_diagnostics.json"

# Document 36 §10's seed is the one the shipped M-3 used; the diagnostic re-uses
# it so the ``shipped`` arm below reproduces §11's −0.0986 exactly, and the four
# arms are compared on *the same 200 splits* rather than on their own draws.
RANDOM_SEED = _ship.RANDOM_SEED
N_SPLITS = _power.N_SPLITS

# Document 35 §3's per-play cell variances, measured on the real stream. Used
# here only to subtract sampling noise out of the observed within-season spread.
CELL_VAR = np.array([2.1845, 3.6690, 1.2494])

# §11f's numbers, quoted so this script's reproduction of them is visible.
DOC36_SHIPPED_R = -0.0986
DOC36_PIPELINE_NULL_MEAN = 0.0038
DOC36_PIPELINE_NULL_SD = 0.0390
DOC36_ADOPTED_COVERAGE = 87.45

M3_EXTRA_SEEDS = (20260821, 20260822, 20260823, 20260824)
M2_EXTRA_SEEDS = (20260821, 20260822, 20260823)


def save(section: str, payload: dict) -> None:
    path = paths.RESEARCH_OUTPUT_DIR / RESULTS
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[section] = payload
    path.write_text(json.dumps(existing, indent=2, default=float))
    print(f"\nwrote {section} to {path}")


# --------------------------------------------------------------------------
# part 1 — where the anti-persistence comes from
# --------------------------------------------------------------------------


def _rescore(table: pl.DataFrame, s0: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """The redesigned score with ``s0`` swapped for a different quality baseline.

    Everything downstream is the production path: the same three weighted
    leave-one-franchise-out fits, the same centring, the same points scale. Only
    the covariate the profile is fitted against changes, which is the whole point
    — an arm that also changed the estimator would not isolate anything.

    Rows whose arm leaves ``s0`` undefined (a team-season too short to have a
    half, a team with no other season) fall back to the shipped baseline. They
    are outside the split-half sample and enter only the league-wide fit.
    """
    n_all, counts, sums, _ = _ship.cell_matrices(table)
    s0 = np.where(np.isfinite(s0), s0, fallback)
    mu = expected_profile(counts, sums, s0, table["posteam"].to_numpy())
    points = redesigned_cell_points(n_all, counts, sums, mu)
    return points[:, 0] + points[:, 1]


def _other_seasons_rate(table: pl.DataFrame) -> np.ndarray:
    """Each team-game's quality from every season of that franchise except this one."""
    season = table.group_by("team_season").agg(
        pl.col("posteam").first(),
        pl.col("epa_all").sum().alias("epa"),
        pl.col("n_all").sum().alias("n"),
    )
    rate = leave_one_out_rate(
        season["epa"].to_numpy().astype(float),
        season["n"].to_numpy().astype(float),
        season["posteam"].to_numpy(),
    )
    lookup = dict(zip(season["team_season"].to_numpy(), rate, strict=True))
    return np.array([lookup[key] for key in table["team_season"].to_numpy()])


def _within_season_spread(table: pl.DataFrame) -> dict:
    """The real game-to-game spread of team quality inside a season, net of sampling.

    A team-game's observed EPA per play scatters around its team-season mean for
    two reasons — the team really was better or worse that day, and 60-odd plays
    is a small sample. The second is computable from document 35 §3's cell
    variances and the game's own cell counts, so the first is what is left.
    """
    _, counts, _, _ = _ship.cell_matrices(table)
    n_all = table["n_all"].to_numpy().astype(float)
    quality = table["epa_all"].to_numpy().astype(float) / n_all
    sampling_var = (counts * CELL_VAR[None, :]).sum(axis=1) / n_all**2

    key, inverse = np.unique(table["team_season"].to_numpy(), return_inverse=True)
    plays = np.bincount(inverse, weights=n_all)
    season_mean = np.bincount(inverse, weights=quality * n_all) / plays
    games = np.bincount(inverse)

    deviation = quality - season_mean[inverse]
    # Unbiased within-season variance, pooled over team-seasons with the g/(g-1)
    # correction each one needs, then the sampling share subtracted.
    scale = (games / np.maximum(games - 1.0, 1.0))[inverse]
    observed_var = float(np.mean(deviation**2 * scale))
    sampling = float(np.mean(sampling_var))
    return {
        "n_team_seasons": int(len(key)),
        "observed_within_season_var": observed_var,
        "sampling_var": sampling,
        "true_within_season_var": observed_var - sampling,
        "true_within_season_sd": float(np.sqrt(max(observed_var - sampling, 0.0))),
        "observed_within_season_sd": float(np.sqrt(observed_var)),
        "sampling_share": sampling / observed_var,
    }


def part_persistence() -> dict:
    print(f"{'=' * 72}\nDiagnostic 1 — where the −0.0986 comes from\n{'=' * 72}")
    table = _ship.cached_scores()
    rows, blocks = _power.season_blocks(table.select("team_season", "game_id"))
    rng = np.random.default_rng(RANDOM_SEED)
    masks = _power.split_masks(blocks, len(rows), rng)

    shipped_s0 = table["s0_loo"].to_numpy().astype(float)
    epa = table["epa_all"].to_numpy().astype(float)
    n_all = table["n_all"].to_numpy().astype(float)
    team_season = table["team_season"].to_numpy()

    shipped_score = table["score"].to_numpy().astype(float)
    r_shipped = _power.split_half_r(shipped_score[rows], masks, blocks)
    print(
        f"  shipped arm reproduces M-3: r = {r_shipped:+.4f} (document 36 §11: {DOC36_SHIPPED_R})"
    )

    spread = _within_season_spread(table)
    print(
        f"\n  within-season quality spread: observed SD {spread['observed_within_season_sd']:.4f}, "
        f"sampling SD {np.sqrt(spread['sampling_var']):.4f}, "
        f"**true SD {spread['true_within_season_sd']:.4f}** EPA/play"
    )
    print(f"  sampling is {spread['sampling_share']:.1%} of the observed within-season variance")

    _, season_index = np.unique(team_season, return_inverse=True)
    n_seasons = season_index.max() + 1

    def subset_rate(member: np.ndarray, exclude_self: bool) -> np.ndarray:
        """Each row's quality over ``member``, optionally minus the row itself."""
        total = np.bincount(season_index, weights=epa * member, minlength=n_seasons)[season_index]
        count = np.bincount(season_index, weights=n_all * member, minlength=n_seasons)[season_index]
        if exclude_self:
            total = total - epa * member
            count = count - n_all * member
        return np.divide(total, count, out=np.full(len(total), np.nan), where=count > 0)

    # The two arms that do not depend on the split are scored once.
    other_seasons = _other_seasons_rate(table)
    static = {
        "other_seasons": _rescore(table, other_seasons, shipped_s0),
    }

    # A fixed half-sized subsample per team-season, drawn once and shared by both
    # split halves — the noise control for ``within_half``. Rows inside the
    # subsample get it minus themselves; rows outside get it outright, which is
    # what "shared" means: every game in the season reads the same plays.
    control_rng = np.random.default_rng(RANDOM_SEED + 7)
    in_control = np.zeros(table.height)
    for start, size in blocks:
        in_control[rows[control_rng.permutation(size)[: size // 2] + start]] = 1.0
    static["shared_half"] = _rescore(table, subset_rate(in_control, True), shipped_s0)

    arms = ["shipped", "within_half", "cross_half", "shared_half", "other_seasons"]
    per_split = {name: np.empty(N_SPLITS) for name in arms}
    last_split_score: dict[str, np.ndarray] = {}
    for split in range(N_SPLITS):
        in_a = np.zeros(table.height)
        in_b = np.zeros(table.height)
        in_a[rows[masks[split]]] = 1.0
        in_b[rows[~masks[split]]] = 1.0

        own_half = np.where(in_a > 0, in_a, in_b)
        rate_a_self, rate_b_self = subset_rate(in_a, True), subset_rate(in_b, True)
        rate_a_all, rate_b_all = subset_rate(in_a, False), subset_rate(in_b, False)

        # Own half minus this game, and the *other* half whole — the game being
        # scored is not in it, so nothing is excluded.
        within_s0 = np.where(in_a > 0, rate_a_self, np.where(in_b > 0, rate_b_self, np.nan))
        cross_s0 = np.where(in_a > 0, rate_b_all, np.where(in_b > 0, rate_a_all, np.nan))
        within_s0 = np.where(own_half > 0, within_s0, np.nan)
        cross_s0 = np.where(own_half > 0, cross_s0, np.nan)

        one = masks[split : split + 1]
        per_split["shipped"][split] = _power.split_half_r(shipped_score[rows], one, blocks)
        for name, s0 in (("within_half", within_s0), ("cross_half", cross_s0)):
            score = _rescore(table, s0, shipped_s0)
            last_split_score[name] = score
            per_split[name][split] = _power.split_half_r(score[rows], one, blocks)
        for name, score in static.items():
            per_split[name][split] = _power.split_half_r(score[rows], one, blocks)
        if (split + 1) % 25 == 0:
            print(f"    split {split + 1}/{N_SPLITS}", flush=True)

    # Each arm gets its own permutation null rather than borrowing the shipped
    # score's. Document 35's null deals the real team-game scores into synthetic
    # team-seasons, which destroys team identity while keeping the score
    # distribution, the team-season sizes and the split pattern — so an arm whose
    # score distribution differs gets the yardstick its own distribution implies.
    score_of_arm = {
        "shipped": shipped_score,
        "within_half": None,
        "cross_half": None,
        **static,
    }
    nulls = {}
    for name in arms:
        values = score_of_arm.get(name)
        if values is None:
            # The split-dependent arms have no single score; their null is taken
            # on the last split's scoring, which is one exchangeable draw of the
            # same construction.
            values = last_split_score[name]
        draws = _power.m3_permutation_null(
            values[rows], masks, blocks, np.random.default_rng(RANDOM_SEED + 11)
        )
        nulls[name] = {"mean": float(draws.mean()), "sd": float(draws.std(ddof=1))}
        print(f"    null for {name}: {draws.mean():+.4f} ± {draws.std(ddof=1):.4f}")

    print(
        f"\n{'arm':<16}{'mean r':>10}{'sd':>9}{'p05':>9}{'p95':>9}   vs shipped     z vs own null"
    )
    summary = {}
    for name in arms:
        values = per_split[name]
        delta = values - per_split["shipped"]
        z = (values.mean() - nulls[name]["mean"]) / nulls[name]["sd"]
        summary[name] = {
            "null_mean": nulls[name]["mean"],
            "null_sd": nulls[name]["sd"],
            "z_vs_own_null": float(z),
            "mean_r": float(values.mean()),
            "sd_across_splits": float(values.std(ddof=1)),
            "p05": float(np.quantile(values, 0.05)),
            "p95": float(np.quantile(values, 0.95)),
            "paired_delta_vs_shipped": float(delta.mean()),
            "paired_delta_sd": float(delta.std(ddof=1)),
        }
        print(
            f"{name:<16}{values.mean():>+10.4f}{values.std(ddof=1):>9.4f}"
            f"{np.quantile(values, 0.05):>+9.4f}{np.quantile(values, 0.95):>+9.4f}"
            f"   {delta.mean():+.4f} ± {delta.std(ddof=1):.4f}   {z:+.2f}"
        )

    payload = {
        "n_splits": N_SPLITS,
        "seed": RANDOM_SEED,
        "n_team_seasons": int(len(blocks)),
        "n_team_games": int(len(rows)),
        "shipped_r_all_splits": float(r_shipped),
        "within_season_spread": spread,
        "null_replicates": _power.N_NULL_REPLICATES,
        "arms": summary,
    }
    save("persistence", payload)
    return payload


# --------------------------------------------------------------------------
# part 2 — the pipeline null, rebuilt so it can carry the mechanism
# --------------------------------------------------------------------------
#
# Document 36 §11f named the wrong limitation. It said the full-pipeline
# simulation could not carry the shared-baseline channel because its per-game
# truth is constant within a season. The binding limitation is one level down:
# in that simulation **every cell responds to team quality with slope 1**
# (``cell_mean = per_game_truth + structural``). The profile shift the redesign
# subtracts is
#
#     C = [n_rz·mu_rz + n_ld·mu_ld − (n_rz + n_ld)·Σ n_c·mu_c / n_all] · points
#
# and with ``mu_c = a_c + b_c·s0`` its whole loading on ``s0`` is
#
#     B = [n_rz·b_rz + n_ld·b_ld − (n_rz + n_ld)·Σ n_c·b_c / n_all] · points
#
# which is **identically zero when every ``b_c`` is equal**. A simulation with
# equal cell slopes therefore subtracts nothing that depends on the baseline, so
# no amount of within-season truth variation can make the channel appear in it.
# On the real stream the fitted slopes are far from equal — red zone 0.764, late
# down 1.201, elsewhere 0.606 — and B has mean +4.66 points per EPA per play.
#
# The 2×2 below turns both knobs, so which one is binding is measured rather
# than asserted. Placement itself is pure noise in all four arms.


def _pipeline_null(
    table: pl.DataFrame,
    within_sd: float,
    slopes: np.ndarray | None,
    replicates: int,
    seed: int,
) -> dict:
    """Document 36 §7's ``none`` arm with two knobs added, and nothing else changed.

    Same cell variances, same structural profile, same real cell denominators,
    same production fit, same split-half statistic. ``within_sd`` lets a team's
    quality move game to game; ``slopes`` lets the three cells respond to that
    quality differently. At ``within_sd = 0`` and ``slopes = None`` this is
    document 36 §7's reference arm and reproduces its +0.0038.

    The differential slopes are applied **around the count-weighted mean slope**,
    so a game's overall EPA per play still tracks its quality one-for-one and the
    simulated league keeps the real spread of ``s0``. Only the *relative* cell
    responses change, which is the thing ``B`` above is a function of.
    """
    n_all, counts, _, _ = _ship.cell_matrices(table)
    group = table["posteam"].to_numpy()
    team_season = table["team_season"].to_numpy()
    _, inverse = np.unique(team_season, return_inverse=True)
    plays_per_season = np.bincount(inverse, weights=n_all)
    quality = table["epa_all"].to_numpy().astype(float) / n_all
    truth = np.bincount(inverse, weights=quality * n_all) / plays_per_season

    structural = np.array([0.01469, -0.04814, 0.01131])
    cell_sd = np.sqrt(CELL_VAR)
    if slopes is None:
        relative = np.zeros(3)
    else:
        share = counts.sum(axis=0) / counts.sum()
        relative = np.asarray(slopes, dtype=float) - float(share @ np.asarray(slopes, dtype=float))

    rows, blocks = _power.season_blocks(table.select("team_season", "game_id"))
    masks = _power.split_masks(blocks, len(rows), np.random.default_rng(RANDOM_SEED))
    grand = float(np.average(truth, weights=plays_per_season))

    rng = np.random.default_rng(seed)
    out = np.empty(replicates)
    for replicate in range(replicates):
        per_game_truth = truth[inverse] + rng.normal(0.0, within_sd, size=table.height)
        cell_mean = (
            per_game_truth[:, None]
            + structural[None, :]
            + (per_game_truth - grand)[:, None] * relative[None, :]
        )
        drawn = np.where(
            counts > 0,
            counts * cell_mean
            + rng.normal(0.0, 1.0, size=counts.shape) * cell_sd[None, :] * np.sqrt(counts),
            0.0,
        )
        s0 = leave_one_out_rate(drawn.sum(axis=1), n_all, team_season)
        mu = expected_profile(counts, drawn, s0, group)
        score = redesigned_cell_points(n_all, counts, drawn, mu)[:, :2].sum(axis=1)
        out[replicate] = _power.split_half_r(score[rows], masks, blocks)
        if (replicate + 1) % 100 == 0:
            print(f"    {replicate + 1}/{replicates}", flush=True)

    return {
        "within_season_truth_sd": within_sd,
        "differential_slopes": None if slopes is None else [float(v) for v in slopes],
        "replicates": replicates,
        "mean": float(out.mean()),
        "sd": float(out.std(ddof=1)),
        "p05": float(np.quantile(out, 0.05)),
        "p95": float(np.quantile(out, 0.95)),
    }


def _fitted_cell_slopes(table: pl.DataFrame) -> np.ndarray:
    """The three cells' pooled weighted slopes of cell EPA per play on ``s0``."""
    _, counts, sums, s0 = _ship.cell_matrices(table)
    mean_cell = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return np.array(
        [float(np.polyfit(s0, mean_cell[:, c], 1, w=np.sqrt(counts[:, c]))[0]) for c in range(3)]
    )


def part_pipeline(replicates: int = 300) -> dict:
    print(f"{'=' * 72}\nDiagnostic 1b — the pipeline null, rebuilt\n{'=' * 72}")
    table = _ship.cached_scores()
    spread = _within_season_spread(table)
    measured = spread["true_within_season_sd"]
    slopes = _fitted_cell_slopes(table)
    print(f"  measured within-season truth SD: {measured:.4f} EPA/play")
    print(
        "  fitted cell slopes on quality: red zone "
        f"{slopes[0]:.3f}, late down {slopes[1]:.3f}, elsewhere {slopes[2]:.3f}"
    )
    print(
        f"  document 36 §7's `none` arm: {DOC36_PIPELINE_NULL_MEAN:+.4f} "
        f"± {DOC36_PIPELINE_NULL_SD:.4f}; the shipped score read {DOC36_SHIPPED_R:+.4f}\n"
    )

    grid = (
        ("reference", 0.0, None),
        ("within_only", measured, None),
        ("slopes_only", 0.0, slopes),
        ("slopes_and_within", measured, slopes),
    )
    arms = {}
    for index, (label, sd, slope) in enumerate(grid):
        arms[label] = _pipeline_null(table, sd, slope, replicates, RANDOM_SEED + 100 * (index + 1))
        block = arms[label]
        print(
            f"  {label:<20} within SD {sd:.4f}  slopes {'real' if slope is not None else 'equal'}"
            f"   split-half r {block['mean']:+.4f} ± {block['sd']:.4f}"
            f"   [{block['p05']:+.4f}, {block['p95']:+.4f}]"
        )

    full = arms["slopes_and_within"]
    z = (DOC36_SHIPPED_R - full["mean"]) / full["sd"]
    print(
        f"\n  the shipped {DOC36_SHIPPED_R:+.4f} sits {z:+.2f} SD from a null that carries both "
        f"knobs, against {(DOC36_SHIPPED_R - DOC36_PIPELINE_NULL_MEAN) / DOC36_PIPELINE_NULL_SD:+.2f} "
        f"SD from document 36 §7's equal-slope null"
    )
    payload = {
        "measured_within_season_sd": measured,
        "fitted_cell_slopes": [float(v) for v in slopes],
        "arms": arms,
        "z_vs_full_null": float(z),
    }
    save("pipeline", payload)
    return payload


# --------------------------------------------------------------------------
# part 3 — seed robustness
# --------------------------------------------------------------------------


def part_seeds() -> dict:
    print(f"{'=' * 72}\nDiagnostic 2 — seed robustness\n{'=' * 72}")
    table = _ship.cached_scores()
    rows, blocks = _power.season_blocks(table.select("team_season", "game_id"))
    values = table["score"].to_numpy().astype(float)[rows]

    print("\n  M-3 — split-half r across split seeds (threshold: r > 0.0636 means NOT luck)")
    m3 = {}
    for seed in (RANDOM_SEED, *M3_EXTRA_SEEDS):
        masks = _power.split_masks(blocks, len(values), np.random.default_rng(seed))
        r = _power.split_half_r(values, masks, blocks)
        m3[str(seed)] = {"split_half_r": r, "passes": bool(r <= _ship.M3_THRESHOLD)}
        tag = "  (document 36 §11's primary)" if seed == RANDOM_SEED else ""
        print(
            f"    seed {seed}   r = {r:+.4f}   {'PASS' if r <= _ship.M3_THRESHOLD else 'FAIL'}{tag}"
        )
    r_values = np.array([block["split_half_r"] for block in m3.values()])
    print(
        f"    across {len(r_values)} seeds: mean {r_values.mean():+.4f}, "
        f"range {r_values.min():+.4f} to {r_values.max():+.4f}, spread {np.ptp(r_values):.4f}"
    )

    print(f"\n  M-2 — {_ship.ADOPTED_RUNG} coverage across band seeds (tolerance [87.0, 91.0])")
    plays, _ = _power.load_luck_priced_plays()
    shifts = {
        (row["game_id"], row["posteam"]): row["profile_shift"]
        for row in table.select("game_id", "posteam", "profile_shift").iter_rows(named=True)
    }
    adopted_index = _ship.LADDER.index(_ship.ADOPTED_RUNG)
    m2 = {}
    for seed, label in (
        (_ship.DOC35_SEED + adopted_index, "document 35's stream (carry-forward arm)"),
        (_ship.RANDOM_SEED + adopted_index, "document 36's stream (fresh-seed arm)"),
        *[(seed + adopted_index, "additional") for seed in M2_EXTRA_SEEDS],
    ):
        band = _ship.rung_pit(plays, shifts, _ship.ADOPTED_RUNG, seed)
        coverage = float(band["inside"].mean()) * 100.0
        m2[str(seed)] = {
            "coverage_pct": coverage,
            "inside_tolerance": bool(87.0 <= coverage <= 91.0),
            "label": label,
        }
        print(f"    seed {seed}   coverage {coverage:6.2f}%   {label}")
    coverages = np.array([block["coverage_pct"] for block in m2.values()])
    print(
        f"    across {len(coverages)} seeds: mean {coverages.mean():.2f}%, "
        f"SD {coverages.std(ddof=1):.3f} pp, "
        f"range {coverages.min():.2f}–{coverages.max():.2f}%"
    )
    print(f"    all inside [87.0, 91.0]: {all(block['inside_tolerance'] for block in m2.values())}")

    payload = {
        "m3": {
            "threshold": _ship.M3_THRESHOLD,
            "by_seed": m3,
            "mean": float(r_values.mean()),
            "min": float(r_values.min()),
            "max": float(r_values.max()),
            "all_pass": bool(all(block["passes"] for block in m3.values())),
        },
        "m2": {
            "rung": _ship.ADOPTED_RUNG,
            "by_seed": m2,
            "mean_coverage_pct": float(coverages.mean()),
            "sd_coverage_pp": float(coverages.std(ddof=1)),
            "all_inside_tolerance": all(block["inside_tolerance"] for block in m2.values()),
        },
    }
    save("seeds", payload)
    return payload


PARTS = {"persistence": part_persistence, "pipeline": part_pipeline, "seeds": part_seeds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", required=True, choices=sorted(PARTS))
    PARTS[parser.parse_args().part]()


if __name__ == "__main__":
    main()
