"""Step 2 — power calculation for the pre-registered rematch validation gate.

Phase 1's closing lesson was that a gate threshold set from a football-effect-size
argument, with nobody asking whether the data *could* achieve it, is a gate that
fails for reasons unrelated to the hypothesis (document 04, "the gate itself was
the mistake"). So this script runs before `docs/research/06-rematch-validation.md`
commits any threshold.

It answers one question: **with the rematch pairs that actually exist, how large
would deserve-to-win's predictive edge over the raw result have to be before this
design could detect it?**

Nothing here touches a deserve-to-win number — none exists yet. The design
parameters are estimated from realized margins only, and the effect is simulated.

    uv run python research/08_rematch_power.py
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from nfl_simulator import paths

RANDOM_SEED = 20260817

# Phase 1, document 01: each component's share of *points-margin* variance.
# fumble_luck 3.7%, fg_luck 2.7%, interception 18.7%.
LUCK_SHARE_FUMBLE_ONLY = 0.037
LUCK_SHARE_FUMBLE_FG = 0.064
LUCK_SHARE_WITH_INT = 0.251

N_SIMULATIONS = 2000
N_FOLDS = 10


# --------------------------------------------------------------------------
# design parameters, measured from realized margins only
# --------------------------------------------------------------------------


def rematch_pairs(games: pl.DataFrame) -> pl.DataFrame:
    """First two same-season meetings of every pair of teams.

    Orientation is fixed by game 1's host, so "team A" is whoever hosted the
    first meeting. Pairs that met three times (division rivals who then met in
    the playoffs) contribute only their first two meetings — a third pair would
    share a game with its neighbour and would not be independent.
    """
    games = games.drop_nulls("margin").with_columns(
        pl.min_horizontal("home_team", "away_team").alias("t1"),
        pl.max_horizontal("home_team", "away_team").alias("t2"),
    )
    games = games.with_columns(
        pl.concat_str([pl.col("season").cast(pl.String), "t1", "t2"], separator="_").alias("pair")
    )

    ordered = games.sort(["pair", "week"]).with_columns(
        pl.int_range(pl.len()).over("pair").alias("meeting")
    )
    first = ordered.filter(pl.col("meeting") == 0)
    second = ordered.filter(pl.col("meeting") == 1)
    joined = first.join(second, on="pair", how="inner", suffix="_g2")

    # Margins are home-perspective, so game 2's margin flips whenever team A —
    # game 1's host — is the away side in the rematch.
    return joined.select(
        pl.col("pair"),
        pl.col("season"),
        pl.col("home_team").alias("team_a"),
        pl.col("away_team").alias("team_b"),
        pl.col("margin").alias("margin_g1_a"),
        pl.when(pl.col("home_team_g2") == pl.col("home_team"))
        .then(pl.col("margin_g2"))
        .otherwise(-pl.col("margin_g2"))
        .alias("margin_g2_a"),
        (pl.col("home_team_g2") == pl.col("home_team")).cast(pl.Int8).alias("a_home_g2"),
    ).filter(pl.col("margin_g2_a") != 0)  # a tied game 2 has no winner to predict


def design_parameters(pairs: pl.DataFrame) -> dict:
    """Signal-to-noise of a single game as a predictor of its own rematch.

    Both margins decompose as ``delta + noise`` around a shared true strength
    difference, so their correlation *is* the reliable fraction of one game's
    margin. That is everything the simulation needs, and it uses only realized
    margins — no deserve-to-win quantity exists yet to leak.
    """
    a = pairs["margin_g1_a"].to_numpy().astype(float)
    b = pairs["margin_g2_a"].to_numpy().astype(float)
    # Game 1 is always at A's place, game 2 usually is not. Centring each side on
    # its own mean keeps home-field out of the correlation.
    reliability = float(np.corrcoef(a - a.mean(), b - b.mean())[0, 1])
    return {
        "n_pairs": int(pairs.height),
        "sd_margin": float(np.std(a, ddof=1)),
        "reliability": reliability,
        # Half the gap between A's home margin and A's rematch margin.
        "hfa_points": float(a.mean() - b.mean()) / 2.0,
        "a_wins_g2_rate": float((b > 0).mean()),
        "p_a_home_g2": float(pairs["a_home_g2"].mean()),
    }


# --------------------------------------------------------------------------
# the statistic
# --------------------------------------------------------------------------


def fit_logistic(x: np.ndarray, y: np.ndarray, iterations: int = 25) -> np.ndarray:
    """Newton-Raphson logistic fit. Three columns, a few hundred rows."""
    beta = np.zeros(x.shape[1])
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -30, 30)))
        w = np.clip(p * (1.0 - p), 1e-9, None)
        hessian = x.T @ (x * w[:, None]) + 1e-6 * np.eye(x.shape[1])
        step = np.linalg.solve(hessian, x.T @ (y - p))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


def cv_log_loss(x: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Per-observation out-of-fold log loss.

    Cross-validated rather than in-sample because the two models being compared
    are only *structurally* identical: whichever predictor carries more signal
    also overfits differently, so in-sample optimism does not cleanly cancel in
    the paired difference. Ten folds rather than leave-one-out because the power
    simulation needs thousands of replicates and the two agree closely at this n.
    """
    losses = np.empty(len(y))
    for fold in np.unique(folds):
        test = folds == fold
        beta = fit_logistic(x[~test], y[~test])
        p = np.clip(1.0 / (1.0 + np.exp(-np.clip(x[test] @ beta, -30, 30))), 1e-6, 1 - 1e-6)
        losses[test] = -(y[test] * np.log(p) + (1 - y[test]) * np.log(1 - p))
    return losses


def design_matrix(predictor: np.ndarray, a_home_g2: np.ndarray) -> np.ndarray:
    """Intercept, standardized predictor, home-field indicator for game 2.

    Standardizing means the two arms' coefficients are on one scale and the
    comparison cannot be won by a units difference.
    """
    z = (predictor - predictor.mean()) / predictor.std()
    return np.column_stack([np.ones(len(z)), z, a_home_g2.astype(float)])


def paired_log_loss_diff(
    actual: np.ndarray,
    deserved: np.ndarray,
    y: np.ndarray,
    a_home_g2: np.ndarray,
    folds: np.ndarray,
) -> np.ndarray:
    """Per-pair out-of-fold log loss, deserved minus actual.

    Paired at the rematch-pair level, because both models score the very same
    games — the pairing is what removes the large game-to-game variance that
    would otherwise swamp a difference this small.
    """
    loss_actual = cv_log_loss(design_matrix(actual, a_home_g2), y, folds)
    loss_deserved = cv_log_loss(design_matrix(deserved, a_home_g2), y, folds)
    return loss_deserved - loss_actual


def cv_squared_error(x: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Per-observation out-of-fold squared error from an OLS fit.

    The secondary estimand. Predicting game 2's *margin* rather than its winner
    keeps the information that dichotomising throws away — a 3-point win and a
    30-point win are the same event to a logistic model — so if the primary
    design is short of power, this is where the extra power would come from.
    """
    errors = np.empty(len(y))
    for fold in np.unique(folds):
        test = folds == fold
        beta, *_ = np.linalg.lstsq(x[~test], y[~test], rcond=None)
        errors[test] = (y[test] - x[test] @ beta) ** 2
    return errors


# Non-inferiority margin, in log-loss units. Phase 1 measured the Vegas-vs-raw-EPA
# gap at 0.0398, so this is roughly a quarter of the largest predictive gap the
# project has ever resolved — big enough to matter, small enough to be checkable.
NONINFERIORITY_MARGIN = 0.010


def decision(per_pair: np.ndarray) -> tuple[float, float, bool]:
    """Mean difference, its standard error, and whether the 95% CI excludes zero
    on the favourable side. This is the superiority decision rule."""
    mean = float(per_pair.mean())
    se = float(per_pair.std(ddof=1) / np.sqrt(len(per_pair))) if len(per_pair) > 1 else 0.0
    return mean, se, bool(se > 0 and mean + 1.96 * se < 0)


def passes_noninferiority(mean: float, se: float, margin: float = NONINFERIORITY_MARGIN) -> bool:
    """True when the 95% CI's *upper* bound sits below the allowed margin.

    The standard non-inferiority form, and the gate this design can actually
    carry. Superiority asks "is deserve-to-win *better*", which needs power the
    rematch sample does not have; non-inferiority asks "is it *not meaningfully
    worse*", which is both answerable here and the claim Phase 1 actually makes.

    Note the burden of proof runs the opposite way to a superiority test: a
    wide, uninformative interval FAILS this gate rather than passing it, because
    it has not demonstrated the absence of harm.
    """
    return bool(se > 0 and mean + 1.96 * se < margin)


# --------------------------------------------------------------------------
# the simulation
# --------------------------------------------------------------------------


def simulate_once(
    rng: np.random.Generator,
    params: dict,
    luck_share: float,
    *,
    null: bool = False,
    correlated_null: bool = False,
    estimand: str = "winner",
) -> tuple[float, float, bool]:
    """One synthetic rematch dataset. Returns (mean diff, SE, CI excludes zero).

    Generative story, matching the design pre-registered in document 06:

        delta         true strength difference, A minus B, in points
        luck          the neutralizable part of a game's margin
        residual      everything that is neither delta nor luck
        margin_g1   = delta + luck + residual      (what the record says)
        game 2      : A wins if delta + hfa + fresh noise > 0

    **Alternative** (``null=False``): ``deserved = delta + residual``. Luck is
    removed perfectly, with no estimation error anywhere in the neutralization.
    That is deliberately the best case a simulator could ever achieve, so the
    power reported here is an upper bound on the real design's power.

    **Null** (``null=True``): ``deserved = delta + fresh noise of the same total
    variance``. The deserved predictor is genuinely *different* from the actual
    margin but carries exactly the same signal-to-noise, which is the honest
    statement of "deserve-to-win is no better than the result". Setting the null
    to ``luck_share = 0`` instead would make the two predictors numerically
    identical and drive the type-I rate to a meaningless zero.

    **Correlated null** (``correlated_null=True``): the objection to the null
    above is that it makes the two predictors nearly independent, so the paired
    SE comes out roughly six times larger than the real test will see. This
    variant answers it. The neutralization removes a *proportional* slice of the
    margin — signal and noise in the same ratio, so no reliability is gained —
    and adds estimation noise on top, leaving `deserved` highly correlated with
    `actual` and slightly worse. It is a conservative null (the rule can only
    under-reject against it), which is exactly what makes it a useful bound.
    """
    n = params["n_pairs"]
    var_total = params["sd_margin"] ** 2
    var_delta = params["reliability"] * var_total
    var_luck = luck_share * var_total
    var_residual = max(var_total - var_delta - var_luck, 1e-6)

    delta = rng.normal(0.0, np.sqrt(var_delta), n)
    luck = rng.normal(0.0, np.sqrt(var_luck), n)
    residual = rng.normal(0.0, np.sqrt(var_residual), n)

    actual = delta + luck + residual
    if correlated_null:
        # A proportional slice carries signal and noise in the margin's own
        # ratio, so removing it changes nothing; the estimation noise is what
        # makes `deserved` a distinct — and slightly worse — predictor.
        estimation_noise = rng.normal(0.0, np.sqrt(var_luck), n)
        deserved = (1.0 - np.sqrt(luck_share)) * actual - estimation_noise
    elif null:
        deserved = delta + rng.normal(0.0, np.sqrt(var_total - var_delta), n)
    else:
        deserved = delta + residual

    a_home_g2 = (rng.random(n) < params["p_a_home_g2"]).astype(float)
    hfa = params["hfa_points"] * (2.0 * a_home_g2 - 1.0)
    margin_g2 = delta + hfa + rng.normal(0.0, np.sqrt(var_total - var_delta), n)
    y = (margin_g2 > 0).astype(float)

    folds = rng.permutation(n) % N_FOLDS
    if estimand == "margin":
        diff = cv_squared_error(design_matrix(deserved, a_home_g2), margin_g2, folds) - (
            cv_squared_error(design_matrix(actual, a_home_g2), margin_g2, folds)
        )
        return decision(diff)
    return decision(paired_log_loss_diff(actual, deserved, y, a_home_g2, folds))


def draws_for(
    params: dict,
    luck_share: float,
    simulations: int,
    seed: int,
    *,
    null: bool = False,
    correlated_null: bool = False,
    estimand: str = "winner",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (mean diffs, SEs, reject flags) across simulated datasets."""
    rng = np.random.default_rng(seed)
    results = [
        simulate_once(
            rng,
            params,
            luck_share,
            null=null,
            correlated_null=correlated_null,
            estimand=estimand,
        )
        for _ in range(simulations)
    ]
    means, ses, rejects = zip(*results, strict=True)
    return np.array(means), np.array(ses), np.array(rejects)


def main() -> None:
    paths.ensure_data_dirs()
    games = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / "game_components.parquet")
    pairs = rematch_pairs(games)
    params = design_parameters(pairs)

    print("=== Design parameters, measured from realized margins only ===")
    for key, value in params.items():
        print(f"  {key:18s} {value:.4f}" if isinstance(value, float) else f"  {key:18s} {value}")

    # Type-I check. Under the null the deserved predictor differs from the
    # actual margin but carries the same signal, so the decision rule should
    # reject about 2.5% of the time (one tail of a two-sided 95% CI).
    print(f"\n=== Null — deserved differs but is no better ({N_SIMULATIONS} datasets) ===")
    null_means, null_ses, null_rejects = draws_for(
        params, LUCK_SHARE_FUMBLE_FG, N_SIMULATIONS, RANDOM_SEED, null=True
    )
    type_i = float(null_rejects.mean())
    print(f"  mean {null_means.mean():+.5f}   sd {null_means.std(ddof=1):.5f}")
    print(f"  mean SE per dataset {null_ses.mean():.5f}")
    print(f"  type-I rate of the decision rule: {type_i:.3f}  (nominal 0.025)")

    corr_means, corr_ses, corr_rejects = draws_for(
        params, LUCK_SHARE_FUMBLE_FG, N_SIMULATIONS, RANDOM_SEED + 50, correlated_null=True
    )
    type_i_correlated = float(corr_rejects.mean())
    print("\n=== Correlated null — proportional slice removed plus estimation noise ===")
    print(f"  mean {corr_means.mean():+.5f}   sd {corr_means.std(ddof=1):.5f}")
    print(f"  mean SE per dataset {corr_ses.mean():.5f}")
    print(f"  type-I rate: {type_i_correlated:.3f}  (conservative null, expected below 0.025)")

    scenarios = {
        "fumble only (3.7%)": LUCK_SHARE_FUMBLE_ONLY,
        "fumble + FG (6.4%)": LUCK_SHARE_FUMBLE_FG,
        "fumble + FG + INT (25.1%)": LUCK_SHARE_WITH_INT,
        "implausibly large (50%)": 0.50,
    }

    print(f"\n=== Power at each scenario ({N_SIMULATIONS} datasets each) ===")
    rows = [
        {
            "scenario": "null — deserved differs, no better",
            "luck_share": 0.0,
            "mean_delta_log_loss": float(null_means.mean()),
            "sd_across_datasets": float(null_means.std(ddof=1)),
            "mean_se_within_dataset": float(null_ses.mean()),
            "p_improves_at_all": float((null_means < 0).mean()),
            "power": type_i,
        },
        {
            "scenario": "correlated null — proportional + est. noise",
            "luck_share": 0.0,
            "mean_delta_log_loss": float(corr_means.mean()),
            "sd_across_datasets": float(corr_means.std(ddof=1)),
            "mean_se_within_dataset": float(corr_ses.mean()),
            "p_improves_at_all": float((corr_means < 0).mean()),
            "power": type_i_correlated,
        },
    ]
    for offset, (label, share) in enumerate(scenarios.items(), start=1):
        means, ses, rejects = draws_for(params, share, N_SIMULATIONS, RANDOM_SEED + offset)
        rows.append(
            {
                "scenario": label,
                "luck_share": share,
                "mean_delta_log_loss": float(means.mean()),
                "sd_across_datasets": float(means.std(ddof=1)),
                "mean_se_within_dataset": float(ses.mean()),
                "p_improves_at_all": float((means < 0).mean()),
                "power": float(rejects.mean()),
            }
        )
    table = pl.DataFrame(rows)
    with pl.Config(tbl_cols=-1, fmt_str_lengths=40, tbl_width_chars=220):
        print(table)

    # Secondary estimand: predict game 2's margin instead of its winner.
    print(f"\n=== Secondary estimand — game-2 margin, OLS ({N_SIMULATIONS} datasets) ===")
    margin_rows = []
    for offset, (label, share) in enumerate(scenarios.items(), start=20):
        means, ses, rejects = draws_for(
            params, share, N_SIMULATIONS, RANDOM_SEED + offset, estimand="margin"
        )
        margin_rows.append(
            {
                "scenario": label,
                "luck_share": share,
                "mean_delta_mse": float(means.mean()),
                "mean_se_within_dataset": float(ses.mean()),
                "p_improves_at_all": float((means < 0).mean()),
                "power": float(rejects.mean()),
            }
        )
    _, _, margin_null_rejects = draws_for(
        params, LUCK_SHARE_FUMBLE_FG, N_SIMULATIONS, RANDOM_SEED + 60, null=True, estimand="margin"
    )
    margin_type_i = float(margin_null_rejects.mean())
    with pl.Config(tbl_cols=-1, fmt_str_lengths=40, tbl_width_chars=220):
        print(pl.DataFrame(margin_rows))
    print(f"type-I rate (independent null): {margin_type_i:.3f}  (nominal 0.025)")

    # Harm detection: how broken would the neutralization have to be before the
    # non-inferiority gate catches it? The correlated-null generator is the harm
    # generator — it removes a slice carrying no signal gain and adds estimation
    # noise, so raising its share is exactly "the simulator's luck estimate is
    # increasingly wrong".
    print("\n=== Harm detection — power of the non-inferiority gate ===")
    print(f"    (margin = {NONINFERIORITY_MARGIN:+.3f} log loss)")
    harm_rows = []
    for offset, share in enumerate([0.064, 0.15, 0.25, 0.40, 0.60], start=80):
        means, ses, _ = draws_for(
            params, share, N_SIMULATIONS, RANDOM_SEED + offset, correlated_null=True
        )
        caught = np.array(
            [not passes_noninferiority(m, s) for m, s in zip(means, ses, strict=True)]
        )
        harm_rows.append(
            {
                "estimation_noise_share": share,
                "mean_delta_log_loss": float(means.mean()),
                "mean_se": float(ses.mean()),
                "power_to_catch": float(caught.mean()),
            }
        )
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(pl.DataFrame(harm_rows))

    # False-alarm rate: a healthy simulator must not trip the gate.
    healthy_means, healthy_ses, _ = draws_for(
        params, LUCK_SHARE_FUMBLE_FG, N_SIMULATIONS, RANDOM_SEED + 90
    )
    false_alarm = float(
        np.mean(
            [
                not passes_noninferiority(m, s)
                for m, s in zip(healthy_means, healthy_ses, strict=True)
            ]
        )
    )
    print(f"false-alarm rate on a healthy simulator: {false_alarm:.4f}")

    mde = minimum_detectable_share(params)
    print(
        f"\nMinimum detectable luck share at 80% power: {mde:.1%}"
        if mde is not None
        else "\nMinimum detectable luck share at 80% power: >90% — the design cannot reach it"
    )
    print(f"The simulator can remove at most {LUCK_SHARE_WITH_INT:.1%} (and realistically 6.4%).")

    out = paths.RESEARCH_OUTPUT_DIR / "08_rematch_power.json"
    with out.open("w") as handle:
        json.dump(
            {
                "design": params,
                "n_simulations": N_SIMULATIONS,
                "n_folds": N_FOLDS,
                "type_i_rate": type_i,
                "type_i_rate_correlated_null": type_i_correlated,
                "scenarios": rows,
                "secondary_estimand_margin": margin_rows,
                "secondary_type_i_rate": margin_type_i,
                "noninferiority_margin": NONINFERIORITY_MARGIN,
                "harm_detection": harm_rows,
                "noninferiority_false_alarm_rate": false_alarm,
                "mde_luck_share_80pct_power": mde,
            },
            handle,
            indent=2,
        )
    print(f"\nWrote {out}")


def minimum_detectable_share(params: dict) -> float | None:
    """Smallest luck share reaching 80% power under the decision rule."""

    def power_at(share: float) -> float:
        _, _, rejects = draws_for(params, share, 400, RANDOM_SEED + 99)
        return float(rejects.mean())

    low, high = 0.0, 0.90
    if power_at(high) < 0.80:
        return None
    for _ in range(6):
        mid = (low + high) / 2
        if power_at(mid) >= 0.80:
            high = mid
        else:
            low = mid
    return high


if __name__ == "__main__":
    main()
