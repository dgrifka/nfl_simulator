"""Round 24, step 2 — does stadium elevation belong in the make-probability model?

Runs the gates `docs/research/66-fg-elevation-prereg.md` §7 committed at
`e97a918`, before this file existed:

* **E-1** sampler health, **E-2** elevation-by-distance calibration, **E-3**
  elevation resolvable, **E-4** distance calibration preserved, **E-5** held-out
  log-loss over five game-grouped folds, **E-6** the Denver-excluded refit,
  **E-7** materiality against the shipped posterior.

The model, the priors, the sampler settings and the gate machinery are document
05b's, reused from `research/14_fg_weather_model.py` rather than copied wherever
reuse is possible, so the incumbent and the candidate cannot drift apart. The
covariate construction is `research/81a_fg_elevation_power.py`'s, imported for
the same reason: the column fitted here is the column the power was computed for.

    uv run python research/81_fg_elevation.py

Writes:
    research/outputs/trace_fg_elevation.nc         the candidate posterior
    research/outputs/trace_fg_elevation_noden.nc   Gate E-6's arm
    research/outputs/81_fg_elevation.json          the gate report
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import arviz as az
import numpy as np
import polars as pl
import pymc as pm

sys.path.insert(0, str(Path(__file__).parent))

_weather = import_module("14_fg_weather_model")
_power = import_module("81a_fg_elevation_power")

from nfl_simulator import paths  # noqa: E402

RANDOM_SEED = 20260831  # document 66 §11
CHAINS = _weather.CHAINS
TUNE = _weather.TUNE
DRAWS = _weather.DRAWS
TARGET_ACCEPT = _weather.TARGET_ACCEPT
DISTANCE_CENTRE = _weather.DISTANCE_CENTRE
ROOF_LEVELS = _weather.ROOF_LEVELS
MIN_CELL_ATTEMPTS = _weather.MIN_CELL_ATTEMPTS

GATE_E3_NULL_BOUND = -0.00790  # document 66 §7, from research/outputs/81a_...json
BETA_ELEV_PRIOR_SD = 0.10  # document 66 §6
CV_FOLDS = 5  # document 66 §7
BOOTSTRAP_RESAMPLES = 2000  # document 66 §7
MATERIALITY_FLOOR_PP = 1.0  # document 66 §7
ELEVATION_BANDS = (
    (-1.0, 0.5, "<500ft"),
    (0.5, 1.5, "500-1500ft"),
    (1.5, 3.0, "1500-3000ft"),
    (3.0, 99.0, ">=3000ft"),
)
DENVER = "DEN00"


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------


def build_model(
    kicks: pl.DataFrame,
    kicker_levels: list[str],
    kicker_idx: np.ndarray,
    centres: dict,
    elev_centre: float,
    *,
    elevation: bool,
):
    """Document 05b §10's model, cubic arm, with document 66's elevation term.

    ``elevation=False`` is the control: the same code path, the same seed, the
    same design, one column short. It exists so that Gate E-5's comparison is
    between two models that differ in exactly one term rather than between a
    fresh fit and a posterior sampled a year ago under a different seed.
    """
    centred = kicks["distance"].to_numpy() - DISTANCE_CENTRE
    made = kicks["made"].to_numpy()
    is_xp = kicks["is_xp"].to_numpy().astype(float)
    has_weather = kicks["has_weather"].to_numpy().astype(float)
    wind = np.nan_to_num(kicks["wind"].to_numpy().astype(float)) - centres["wind"]
    temp = np.nan_to_num(kicks["temp"].to_numpy().astype(float)) - centres["temp"]
    roof = kicks["roof"].to_numpy()
    roof_design = np.column_stack([(roof == level).astype(float) for level in ROOF_LEVELS])
    elev = kicks["elev_kft"].to_numpy() - elev_centre

    coords = {"kicker_season": kicker_levels, "roof_level": list(ROOF_LEVELS)}
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", mu=2.0, sigma=1.5)
        beta = pm.Normal("beta", mu=0.0, sigma=0.2)
        gamma = pm.Normal("gamma", mu=0.0, sigma=0.2)
        delta_cubic = pm.Normal("delta_cubic", mu=0.0, sigma=0.2)
        sigma_kicker = pm.HalfNormal("sigma_kicker", sigma=1.0)

        roof_effect = pm.Normal("roof", mu=0.0, sigma=0.5, dims="roof_level")
        beta_wind = pm.Normal("beta_wind", mu=0.0, sigma=0.05)
        beta_temp = pm.Normal("beta_temp", mu=0.0, sigma=0.02)
        delta_xp = pm.Normal("delta_xp", mu=0.0, sigma=1.0)
        lambda_xp = pm.Normal("lambda_xp", mu=1.0, sigma=0.5)

        z = pm.Normal("z", mu=0.0, sigma=1.0, dims="kicker_season")
        kicker = pm.Deterministic("kicker", sigma_kicker * z, dims="kicker_season")
        kicker_term = kicker[kicker_idx] * (1.0 + (lambda_xp - 1.0) * is_xp)

        eta = (
            alpha
            + beta * centred
            + gamma * centred**2 / 100.0
            + delta_cubic * centred**3 / 1000.0
            + pm.math.dot(roof_design, roof_effect)
            + beta_wind * wind * has_weather
            + beta_temp * temp * has_weather
            + delta_xp * is_xp
            + kicker_term
        )
        if elevation:
            beta_elev = pm.Normal("beta_elev", mu=0.0, sigma=BETA_ELEV_PRIOR_SD)
            eta = eta + beta_elev * elev
        pm.Bernoulli("made", logit_p=eta, observed=made)
    return model


def make_probabilities(
    idata,
    kicks: pl.DataFrame,
    kicker_idx: np.ndarray,
    centres: dict,
    elev_centre: float,
) -> np.ndarray:
    """Per-kick make probability on every posterior draw.

    ``kicker_idx`` of -1 means a kicker-season the fit never saw, which happens
    only when scoring a held-out fold. Its effect is zero — the same `w = 0`
    endpoint document 05 §1 gives an unknown kicker — rather than a neighbour's
    value or the league mean of a quantity that is already centred on zero.
    """
    posterior = idata["posterior"]

    def flat(name):
        return posterior[name].values.ravel()[:, None]

    n_kickers = posterior["kicker"].shape[-1]
    kicker = posterior["kicker"].values.reshape(-1, n_kickers)
    kicker = np.concatenate([kicker, np.zeros((kicker.shape[0], 1))], axis=1)
    safe_idx = np.where(kicker_idx < 0, n_kickers, kicker_idx)
    roof_draws = posterior["roof"].values.reshape(-1, len(ROOF_LEVELS))

    centred = (kicks["distance"].to_numpy() - DISTANCE_CENTRE)[None, :]
    is_xp = kicks["is_xp"].to_numpy().astype(float)[None, :]
    has_weather = kicks["has_weather"].to_numpy().astype(float)[None, :]
    wind = (np.nan_to_num(kicks["wind"].to_numpy().astype(float)) - centres["wind"])[None, :]
    temp = (np.nan_to_num(kicks["temp"].to_numpy().astype(float)) - centres["temp"])[None, :]
    roof = kicks["roof"].to_numpy()
    roof_design = np.column_stack([(roof == level).astype(float) for level in ROOF_LEVELS])

    eta = (
        flat("alpha")
        + flat("beta") * centred
        + flat("gamma") * centred**2 / 100.0
        + flat("delta_cubic") * centred**3 / 1000.0
        + roof_draws @ roof_design.T
        + flat("beta_wind") * wind * has_weather
        + flat("beta_temp") * temp * has_weather
        + flat("delta_xp") * is_xp
        + kicker[:, safe_idx] * (1.0 + (flat("lambda_xp") - 1.0) * is_xp)
    )
    if "beta_elev" in posterior:
        elev = (kicks["elev_kft"].to_numpy() - elev_centre)[None, :]
        eta = eta + flat("beta_elev") * elev
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))


def sample(model, seed: int):
    with model:
        return pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=seed,
            progressbar=False,
        )


def kicker_index(kicks: pl.DataFrame, levels: list[str] | None = None):
    """Index kicks into kicker-season levels, -1 for a level the fit never saw."""
    if levels is None:
        levels = sorted(kicks["kicker_season"].unique().to_list())
    lookup = {level: i for i, level in enumerate(levels)}
    idx = np.array([lookup.get(v, -1) for v in kicks["kicker_season"].to_list()])
    return levels, idx


def elevation_band(elev_kft: np.ndarray) -> np.ndarray:
    band = np.full(len(elev_kft), "?", dtype=object)
    for low, high, label in ELEVATION_BANDS:
        band[(elev_kft > low) & (elev_kft <= high)] = label
    return band


def denver_gain_pp(beta_elev: np.ndarray, logit_45: np.ndarray, elev_centre: float) -> np.ndarray:
    """`beta_elev`, translated into make-rate points at 45 yards, mean elevation -> Denver."""
    base = 1.0 / (1.0 + np.exp(-logit_45))
    high = 1.0 / (1.0 + np.exp(-(logit_45 + beta_elev * (_power.DENVER_KFT - elev_centre))))
    return (high - base) * 100.0


def league_logit_45(posterior) -> np.ndarray:
    centred = 45.0 - DISTANCE_CENTRE
    return (
        posterior["alpha"].values.ravel()
        + posterior["beta"].values.ravel() * centred
        + posterior["gamma"].values.ravel() * centred**2 / 100.0
        + posterior["delta_cubic"].values.ravel() * centred**3 / 1000.0
    )


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------


def gate_e1(idata, label: str) -> dict:
    summary = az.summary(idata)
    report = {
        "divergences": int(idata["sample_stats"]["diverging"].sum().item()),
        "max_r_hat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "min_ess_tail": float(summary["ess_tail"].min()),
    }
    report["pass"] = bool(
        report["divergences"] == 0
        and report["max_r_hat"] < 1.01
        and report["min_ess_bulk"] > 400
        and report["min_ess_tail"] > 400
    )
    print(
        f"\nGate E-1 ({label}): {'PASS' if report['pass'] else 'FAIL'} — "
        f"divergences {report['divergences']}, max r_hat {report['max_r_hat']:.4f}, "
        f"min ess_bulk {report['min_ess_bulk']:.0f}, min ess_tail {report['min_ess_tail']:.0f}"
    )
    return report


def gate_e3(posterior, elev_centre: float) -> dict:
    beta_elev = posterior["beta_elev"].values.ravel()
    bounds = _weather.eti89(beta_elev)
    logit_45 = league_logit_45(posterior)
    gain = denver_gain_pp(beta_elev, logit_45, elev_centre)
    report = {
        "beta_elev_mean": float(beta_elev.mean()),
        "beta_elev_eti89": bounds,
        "null_bound": GATE_E3_NULL_BOUND,
        "clears_null_bound": bool(bounds[0] > GATE_E3_NULL_BOUND),
        "excludes_zero": bool(bounds[0] > 0.0),
        "denver_gain_pp_at_45yd": float(gain.mean()),
        "denver_gain_pp_eti89": _weather.eti89(gain),
    }
    report["pass"] = bool(report["clears_null_bound"] and report["excludes_zero"])
    print(
        f"\nGate E-3 (elevation resolvable): {'PASS' if report['pass'] else 'FAIL'} — "
        f"beta_elev {report['beta_elev_mean']:+.5f} "
        f"[{bounds[0]:+.5f}, {bounds[1]:+.5f}] per 1,000 ft"
    )
    print(
        f"  clears null bound {GATE_E3_NULL_BOUND:+.5f}: "
        f"{report['clears_null_bound']}; excludes zero (binding): {report['excludes_zero']}"
    )
    print(
        f"  at 45 yd, mean elevation -> Denver is "
        f"{report['denver_gain_pp_at_45yd']:+.2f} pp "
        f"[{report['denver_gain_pp_eti89'][0]:+.2f}, {report['denver_gain_pp_eti89'][1]:+.2f}]"
    )
    return report


def cell_masks(kicks: pl.DataFrame, elev_centre: float) -> tuple[list, list[str]]:
    """Gate E-2's cells: elevation band x 10-yard distance bin, >= 100 attempts."""
    del elev_centre
    band = elevation_band(kicks["elev_kft"].to_numpy())
    dbin = (kicks["distance"].to_numpy() // 10 * 10).astype(int)
    cell = np.array([f"{b}|{d}" for b, d in zip(band, dbin, strict=True)])
    labels = sorted({c for c in np.unique(cell) if (cell == c).sum() >= MIN_CELL_ATTEMPTS})
    return [cell == label for label in labels], labels


def gate_e5_cross_validation(
    kicks: pl.DataFrame, centres: dict, elev_centre: float, seed: int
) -> dict:
    """Five game-grouped folds, both arms refitted inside each, mean held-out log-loss."""
    games = np.array(sorted(kicks["game_id"].unique().to_list()))
    rng = np.random.default_rng(seed)
    fold_of_game = dict(zip(games, rng.permutation(len(games)) % CV_FOLDS, strict=True))
    fold = np.array([fold_of_game[g] for g in kicks["game_id"].to_list()])

    made = kicks["made"].to_numpy().astype(float)
    held_out = {
        "elevation": np.full(kicks.height, np.nan),
        "control": np.full(kicks.height, np.nan),
    }

    for k in range(CV_FOLDS):
        train_mask, test_mask = fold != k, fold == k
        train, test = kicks.filter(train_mask), kicks.filter(test_mask)
        levels, train_idx = kicker_index(train)
        _levels, test_idx = kicker_index(test, levels)
        unseen = int((test_idx < 0).sum())
        print(
            f"\n--- Gate E-5 fold {k + 1}/{CV_FOLDS}: "
            f"train {train.height:,}, test {test.height:,}, "
            f"{unseen} held-out kicks by an unseen kicker-season"
        )
        for arm, use_elevation in (("elevation", True), ("control", False)):
            model = build_model(
                train, levels, train_idx, centres, elev_centre, elevation=use_elevation
            )
            idata = sample(model, seed + 1000 * (k + 1) + (1 if use_elevation else 2))
            div = int(idata["sample_stats"]["diverging"].sum().item())
            p = make_probabilities(idata, test, test_idx, centres, elev_centre).mean(axis=0)
            held_out[arm][np.flatnonzero(test_mask)] = p
            print(f"    {arm:9s} fitted, {div} divergences")

    def log_loss(p: np.ndarray, mask: np.ndarray | None = None) -> float:
        p_clipped = np.clip(p if mask is None else p[mask], 1e-12, 1 - 1e-12)
        y = made if mask is None else made[mask]
        return float(-(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped)).mean())

    assert not np.isnan(held_out["elevation"]).any()
    assert not np.isnan(held_out["control"]).any()

    ll_elev, ll_control = log_loss(held_out["elevation"]), log_loss(held_out["control"])

    # Paired bootstrap over games, the unit the folds were grouped on.
    per_kick = -(
        made * np.log(np.clip(held_out["elevation"], 1e-12, 1 - 1e-12))
        + (1 - made) * np.log(np.clip(1 - held_out["elevation"], 1e-12, 1 - 1e-12))
    ) + (
        made * np.log(np.clip(held_out["control"], 1e-12, 1 - 1e-12))
        + (1 - made) * np.log(np.clip(1 - held_out["control"], 1e-12, 1 - 1e-12))
    )
    game_ids = kicks["game_id"].to_numpy()
    order = np.argsort(game_ids)
    sorted_games, starts = np.unique(game_ids[order], return_index=True)
    groups = np.split(per_kick[order], starts[1:])
    sums = np.array([g.sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    boot_rng = np.random.default_rng(seed)
    picks = boot_rng.integers(0, len(sorted_games), size=(BOOTSTRAP_RESAMPLES, len(sorted_games)))
    boot = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)

    denver_mask = kicks["stadium_id"].to_numpy() == DENVER
    report = {
        "folds": CV_FOLDS,
        "grouping": "game_id",
        "log_loss_elevation": ll_elev,
        "log_loss_control": ll_control,
        "difference": ll_elev - ll_control,
        "difference_eti89": [float(np.percentile(boot, 5.5)), float(np.percentile(boot, 94.5))],
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "log_loss_elevation_denver_only": log_loss(held_out["elevation"], denver_mask),
        "log_loss_control_denver_only": log_loss(held_out["control"], denver_mask),
        "n_denver_kicks": int(denver_mask.sum()),
        "pass": bool(ll_elev <= ll_control),
    }
    print(
        f"\nGate E-5 (held-out log-loss): {'PASS' if report['pass'] else 'FAIL'} — "
        f"elevation {ll_elev:.6f} vs control {ll_control:.6f}, "
        f"difference {report['difference']:+.6f} "
        f"[{report['difference_eti89'][0]:+.6f}, {report['difference_eti89'][1]:+.6f}] "
        "(negative favours elevation)"
    )
    print(
        f"  on the {report['n_denver_kicks']:,} Denver kicks only: "
        f"elevation {report['log_loss_elevation_denver_only']:.6f} vs "
        f"control {report['log_loss_control_denver_only']:.6f}"
    )
    return report


def gate_e7_materiality(
    candidate_p: np.ndarray, incumbent_p: np.ndarray, kicks: pl.DataFrame
) -> dict:
    shift_pp = (candidate_p - incumbent_p) * 100.0
    band = elevation_band(kicks["elev_kft"].to_numpy())
    moved = np.abs(shift_pp) >= MATERIALITY_FLOOR_PP
    games_moved = len(set(kicks["game_id"].to_numpy()[moved]))
    by_band = []
    for _low, _high, label in ELEVATION_BANDS:
        mask = band == label
        if not mask.any():
            continue
        by_band.append(
            {
                "band": label,
                "kicks": int(mask.sum()),
                "mean_shift_pp": float(shift_pp[mask].mean()),
                "median_shift_pp": float(np.median(shift_pp[mask])),
                "max_abs_shift_pp": float(np.abs(shift_pp[mask]).max()),
                "moved_1pp": int((np.abs(shift_pp[mask]) >= 1.0).sum()),
            }
        )
    report = {
        "floor_pp": MATERIALITY_FLOOR_PP,
        "n_kicks": int(kicks.height),
        "moved_1pp": int(moved.sum()),
        "moved_2pp": int((np.abs(shift_pp) >= 2.0).sum()),
        "games_with_a_moved_kick": games_moved,
        "n_games": int(kicks["game_id"].n_unique()),
        "mean_shift_pp": float(shift_pp.mean()),
        "max_abs_shift_pp": float(np.abs(shift_pp).max()),
        "by_band": by_band,
    }
    print(
        f"\nGate E-7 (materiality, reported): {report['moved_1pp']:,} of "
        f"{report['n_kicks']:,} kicks move by >= 1 pp of make probability "
        f"({report['moved_1pp'] / report['n_kicks'] * 100:.2f}%), "
        f"{report['moved_2pp']:,} by >= 2 pp"
    )
    print(
        f"  {report['games_with_a_moved_kick']:,} of {report['n_games']:,} games hold at "
        f"least one kick that moved >= 1 pp; largest single move "
        f"{report['max_abs_shift_pp']:.2f} pp"
    )
    with pl.Config(tbl_rows=10):
        print(pl.DataFrame(by_band))
    return report


# --------------------------------------------------------------------------


def main() -> None:
    paths.ensure_data_dirs()
    kicks = _power.load_elevation_kicks()
    elev_centre = _power.elevation_centre(kicks)
    centres = {"wind": _power.WIND_CENTRE, "temp": _power.TEMP_CENTRE}
    levels, kicker_idx = kicker_index(kicks)

    print(
        f"{kicks.height:,} kicks ({int(kicks.height - kicks['is_xp'].sum()):,} field goals, "
        f"{int(kicks['is_xp'].sum()):,} extra points), {len(levels)} kicker-seasons, "
        f"{kicks['game_id'].n_unique():,} games"
    )
    print(f"elevation centre {elev_centre:.4f} kft; Gate E-3 null bound {GATE_E3_NULL_BOUND:+.5f}")

    results = {
        "random_seed": RANDOM_SEED,
        "n_kicks": int(kicks.height),
        "n_field_goals": int(kicks.height - kicks["is_xp"].sum()),
        "n_extra_points": int(kicks["is_xp"].sum()),
        "n_kicker_seasons": len(levels),
        "n_games": int(kicks["game_id"].n_unique()),
        "elevation_centre_kft": elev_centre,
        "centres": centres,
        "gate_e3_null_bound": GATE_E3_NULL_BOUND,
    }
    out_path = paths.RESEARCH_OUTPUT_DIR / "81_fg_elevation.json"

    def save() -> None:
        with out_path.open("w") as handle:
            json.dump(results, handle, indent=2, default=str)

    # ---- the candidate ----------------------------------------------------
    print(f"\n{'#' * 72}\n### Candidate — document 05b cubic arm plus beta_elev\n{'#' * 72}")
    model = build_model(kicks, levels, kicker_idx, centres, elev_centre, elevation=True)
    idata = sample(model, RANDOM_SEED)
    idata.to_netcdf(paths.RESEARCH_OUTPUT_DIR / "trace_fg_elevation.nc")
    posterior = idata["posterior"]

    results["gate_e1_sampler_health"] = gate_e1(idata, "candidate")
    names = [
        "alpha", "beta", "gamma", "delta_cubic", "sigma_kicker",
        "roof", "beta_wind", "beta_temp", "delta_xp", "lambda_xp", "beta_elev",
    ]  # fmt: skip
    print(
        az.summary(idata, var_names=names)[
            ["mean", "sd", "eti89_lb", "eti89_ub", "ess_bulk", "r_hat"]
        ]
    )
    save()

    p_draws = make_probabilities(idata, kicks, kicker_idx, centres, elev_centre)

    results["gate_e3_elevation_resolvable"] = gate_e3(posterior, elev_centre)
    save()

    # ---- Gate E-2 and Gate E-4 -------------------------------------------
    masks, labels = cell_masks(kicks, elev_centre)
    results["gate_e2_elevation_calibration"] = _weather.calibration_gate(
        "Gate E-2 (elevation x distance calibration)", masks, kicks, p_draws, RANDOM_SEED
    )
    results["gate_e2_cells"] = labels
    made = kicks["made"].to_numpy().astype(float)
    p_hat = p_draws.mean(axis=0)
    per_cell = []
    for label, mask in zip(labels, masks, strict=True):
        expected = p_hat[mask]
        sd = np.sqrt((expected * (1.0 - expected)).sum()) / mask.sum()
        per_cell.append(
            {
                "cell": label,
                "attempts": int(mask.sum()),
                "observed": float(made[mask].mean()),
                "predicted": float(expected.mean()),
                "miss_pp": float((made[mask].mean() - expected.mean()) * 100),
                "standardized": float(abs(made[mask].mean() - expected.mean()) / sd),
            }
        )
    results["gate_e2_per_cell"] = per_cell
    with pl.Config(tbl_rows=20):
        print(pl.DataFrame(per_cell))

    bins = (kicks["distance"].to_numpy() // 5 * 5).astype(int)
    distance_masks = [
        bins == edge for edge in np.unique(bins) if (bins == edge).sum() >= MIN_CELL_ATTEMPTS
    ]
    results["gate_e4_distance_calibration"] = _weather.calibration_gate(
        "Gate E-4 (distance calibration, preserved)", distance_masks, kicks, p_draws, RANDOM_SEED
    )
    save()

    # ---- Gate E-6: without Denver ----------------------------------------
    print(f"\n{'#' * 72}\n### Gate E-6 — the same fit with every Denver kick removed\n{'#' * 72}")
    no_denver = kicks.filter(pl.col("stadium_id") != DENVER)
    nd_levels, nd_idx = kicker_index(no_denver)
    print(
        f"{no_denver.height:,} kicks ({kicks.height - no_denver.height:,} removed), "
        f"{len(nd_levels)} kicker-seasons "
        f"({len(levels) - len(nd_levels)} lost entirely with Denver)"
    )
    nd_model = build_model(no_denver, nd_levels, nd_idx, centres, elev_centre, elevation=True)
    nd_idata = sample(nd_model, RANDOM_SEED + 7)
    nd_idata.to_netcdf(paths.RESEARCH_OUTPUT_DIR / "trace_fg_elevation_noden.nc")
    nd_health = gate_e1(nd_idata, "Denver-excluded arm")
    nd_posterior = nd_idata["posterior"]
    nd_beta = nd_posterior["beta_elev"].values.ravel()
    nd_bounds = _weather.eti89(nd_beta)
    nd_gain = denver_gain_pp(nd_beta, league_logit_45(nd_posterior), elev_centre)
    results["gate_e6_without_denver"] = {
        "n_kicks": int(no_denver.height),
        "kicks_removed": int(kicks.height - no_denver.height),
        "kicker_seasons_lost": int(len(levels) - len(nd_levels)),
        "sampler_health": nd_health,
        "beta_elev_mean": float(nd_beta.mean()),
        "beta_elev_eti89": nd_bounds,
        "excludes_zero": bool(nd_bounds[0] > 0.0),
        "denver_gain_pp_at_45yd": float(nd_gain.mean()),
        "denver_gain_pp_eti89": _weather.eti89(nd_gain),
        "interval_width_ratio_vs_full": float(
            (nd_bounds[1] - nd_bounds[0])
            / (
                results["gate_e3_elevation_resolvable"]["beta_elev_eti89"][1]
                - results["gate_e3_elevation_resolvable"]["beta_elev_eti89"][0]
            )
        ),
    }
    e6 = results["gate_e6_without_denver"]
    print(
        f"\nGate E-6 (without Denver, reported): beta_elev {e6['beta_elev_mean']:+.5f} "
        f"[{nd_bounds[0]:+.5f}, {nd_bounds[1]:+.5f}] — interval "
        f"{'EXCLUDES' if e6['excludes_zero'] else 'CONTAINS'} zero, "
        f"{e6['interval_width_ratio_vs_full']:.2f}x the full arm's width"
    )
    print(
        f"  extrapolated to Denver at 45 yd: {e6['denver_gain_pp_at_45yd']:+.2f} pp "
        f"[{e6['denver_gain_pp_eti89'][0]:+.2f}, {e6['denver_gain_pp_eti89'][1]:+.2f}]"
    )
    save()

    # ---- Gate E-7: materiality against the shipped posterior --------------
    print(f"\n{'#' * 72}\n### Gate E-7 — against the shipped posterior\n{'#' * 72}")
    incumbent = az.from_netcdf(paths.RESEARCH_OUTPUT_DIR / "trace_fg_refit.nc")
    inc_levels = [str(v) for v in incumbent["posterior"].coords["kicker_season"].values]
    _inc_levels, inc_idx = kicker_index(kicks, inc_levels)
    assert (inc_idx >= 0).all(), "the shipped posterior does not cover this population"
    incumbent_p = make_probabilities(incumbent, kicks, inc_idx, centres, elev_centre).mean(axis=0)
    results["gate_e7_materiality"] = gate_e7_materiality(p_hat, incumbent_p, kicks)
    save()

    # ---- Gate E-5: held-out log-loss -------------------------------------
    print(f"\n{'#' * 72}\n### Gate E-5 — five game-grouped folds, both arms\n{'#' * 72}")
    results["gate_e5_held_out_log_loss"] = gate_e5_cross_validation(
        kicks, centres, elev_centre, RANDOM_SEED
    )
    save()

    # ---- verdict ----------------------------------------------------------
    gate_summary = {
        "E-1": results["gate_e1_sampler_health"]["pass"],
        "E-2": results["gate_e2_elevation_calibration"]["pass"],
        "E-3": results["gate_e3_elevation_resolvable"]["pass"],
        "E-4": results["gate_e4_distance_calibration"]["pass"],
        "E-5": results["gate_e5_held_out_log_loss"]["pass"],
    }
    results["gate_summary"] = gate_summary
    results["all_pass_rule_gates_pass"] = bool(all(gate_summary.values()))
    print(f"\n{'=' * 72}")
    for gate, passed in gate_summary.items():
        print(f"  Gate {gate}: {'PASS' if passed else 'FAIL'}")
    print(f"  E-6 and E-7 are reported, no pass rule.\n{'=' * 72}")
    save()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
