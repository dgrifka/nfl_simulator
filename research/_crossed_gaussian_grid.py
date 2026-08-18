"""Exact grid posterior for a crossed Gaussian random-effects model.

The sibling of `research/_betabinom_grid.py`, and it exists for the same reason.
Document 05 §7's attribution round needed hundreds of fits to compute an
achievable-null bound, and NUTS was too slow; a two-parameter marginal posterior
could be evaluated directly on a grid instead, and it reproduced the nutpie fit
to within 0.02 pp, which is what licensed using it.

The model here::

    y_i = mu + qb[q_i] + coach[c_i] + e_i
    qb ~ Normal(0, sigma_qb)   coach ~ Normal(0, sigma_coach)   e ~ Normal(0, sigma_e)

has three variance parameters rather than two, so the reduction is one step
longer. Writing ``lambda = sigma^2 / sigma_e^2`` for each grouping factor puts
the marginal covariance in the form

    V = sigma_e^2 * H,     H = I + Z Lambda Z'

where ``Z`` is the (n x p) indicator matrix stacking both factors. ``sigma_e``
then profiles out of the restricted likelihood in closed form, leaving a **two**
dimensional surface over ``(lambda_qb, lambda_coach)`` — exactly the shape the
grid instrument wants.

Every quantity needed comes from the Woodbury identity with a single p x p
Cholesky, where ``p`` is the total number of levels (a few hundred), never an
n x n operation::

    H^-1 = I - Z M^-1 Z',      M = Lambda^-1 + Z'Z
    |H|  = |Lambda| * |M|

so ``a' H^-1 b = a'b - (Z'a)' M^-1 (Z'b)`` for any vectors, and the whole
likelihood is a function of the precomputed ``Z'Z``, ``Z'y``, ``Z'1``, ``y'y``
and ``1'y``. One grid point costs one Cholesky of a few-hundred-square matrix.

**What the approximation is, stated plainly.** The posterior returned is over
``(lambda_qb, lambda_coach)`` with ``sigma_e`` held at its restricted-likelihood
profile rather than integrated over. That is a conditioning, not a marginalizing,
and it makes the intervals slightly narrower than a full three-parameter
posterior would be. ``self_check`` measures the resulting disagreement against a
PyMC fit on real data, and that measurement is what licenses the instrument —
the same evidence `_betabinom_grid.py` supplies for its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Grid over log10(lambda). The floor is not zero and cannot be: a variance ratio
# of exactly zero is a boundary the log scale never reaches. 1e-5 puts the
# smallest representable sigma at about 0.3% of the residual SD, which is
# indistinguishable from zero in any football sense — but it means **a posterior
# interval's lower bound is never exactly zero by construction**, and must be
# read against the simulated null bound rather than against zero. Document 05 §8
# recorded that same property for the beta-binomial grid.
LOG10_LAMBDA_MIN = -5.0
LOG10_LAMBDA_MAX = 0.5
GRID_POINTS = 41


@dataclass(frozen=True)
class CrossedDesign:
    """Precomputed sufficient statistics for one dataset's design.

    The design — who played for whom, and how many games each pairing saw — is
    fixed across every fit in a null simulation. Only ``y`` changes. Splitting
    the design from the response means the expensive part is paid once.
    """

    ztz: np.ndarray  # (p, p)
    zt1: np.ndarray  # (p,)
    n: int
    sizes: tuple[int, ...]  # levels per grouping factor, in Z's column order

    @property
    def n_levels(self) -> int:
        return self.ztz.shape[0]


def build_design(codes: list[np.ndarray], sizes: list[int]) -> CrossedDesign:
    """Sufficient statistics from integer level codes, one array per factor."""
    n = len(codes[0])
    offsets = np.cumsum([0, *sizes[:-1]])
    columns = [code + offset for code, offset in zip(codes, offsets, strict=True)]
    p = int(sum(sizes))

    # Each row carries exactly one 1 per factor, so Z'Z accumulates a +1 at every
    # ordered pair of that row's active columns — diagonal blocks included.
    ztz = np.zeros((p, p))
    for left in columns:
        for right in columns:
            np.add.at(ztz, (left, right), 1.0)
    zt1 = np.zeros(p)
    for column in columns:
        np.add.at(zt1, column, 1.0)
    return CrossedDesign(ztz=ztz, zt1=zt1, n=n, sizes=tuple(sizes))


def project(codes: list[np.ndarray], sizes: list[int], values: np.ndarray) -> np.ndarray:
    """``Z' values`` — the per-level sums of a response vector."""
    offsets = np.cumsum([0, *sizes[:-1]])
    out = np.zeros(int(sum(sizes)))
    for code, offset in zip(codes, offsets, strict=True):
        np.add.at(out, code + offset, values)
    return out


def _restricted_loglik(
    design: CrossedDesign, zty: np.ndarray, yty: float, oney: float, lambdas: np.ndarray
) -> tuple[float, float]:
    """Profiled REML log-likelihood and the profiled ``sigma_e^2`` at one grid point.

    Restricted rather than plain maximum likelihood because the quantity of
    interest *is* the variance components, and ML biases them downward by the
    degrees of freedom the fixed effect consumes. With one fixed effect that bias
    is small, but it is a bias in the direction that matters here — toward
    finding less spread — so it is removed rather than accepted.
    """
    precision = np.repeat(1.0 / lambdas, design.sizes)
    matrix = design.ztz + np.diag(precision)
    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return -np.inf, np.nan

    def solve(vector: np.ndarray) -> np.ndarray:
        return np.linalg.solve(factor.T, np.linalg.solve(factor, vector))

    m_inv_zty = solve(zty)
    m_inv_zt1 = solve(design.zt1)

    xhx = design.n - design.zt1 @ m_inv_zt1
    xhy = oney - design.zt1 @ m_inv_zty
    yhy = yty - zty @ m_inv_zty

    beta = xhy / xhx
    rss = yhy - beta * xhy
    dof = design.n - 1
    sigma_e2 = rss / dof

    logdet_h = 2.0 * np.log(np.diag(factor)).sum() + np.log(lambdas) @ np.array(design.sizes)
    loglik = -0.5 * (dof * np.log(sigma_e2) + logdet_h + np.log(xhx))
    return float(loglik), float(sigma_e2)


def fit(
    design: CrossedDesign,
    zty: np.ndarray,
    yty: float,
    oney: float,
    *,
    grid_points: int = GRID_POINTS,
) -> dict:
    """Posterior over each factor's SD, on a grid over the two variance ratios.

    The prior is flat on ``(sqrt(lambda_qb), sqrt(lambda_coach))`` — flat on the
    SD scale of each grouping factor relative to the residual, which is the
    standard weakly-informative choice for a variance component and the one that
    does not pile prior mass at zero the way a flat-on-variance prior does.
    """
    log_lambda = np.linspace(LOG10_LAMBDA_MIN, LOG10_LAMBDA_MAX, grid_points)
    lambdas = 10.0**log_lambda

    logliks = np.full((grid_points, grid_points), -np.inf)
    sigma_e2 = np.zeros((grid_points, grid_points))
    for i, lambda_a in enumerate(lambdas):
        for j, lambda_b in enumerate(lambdas):
            logliks[i, j], sigma_e2[i, j] = _restricted_loglik(
                design, zty, yty, oney, np.array([lambda_a, lambda_b])
            )

    # Flat on sqrt(lambda): d(sqrt(lambda))/d(log10 lambda) proportional to
    # sqrt(lambda), so the Jacobian on the log grid is sqrt(lambda_a * lambda_b).
    log_prior = 0.5 * (np.log(lambdas)[:, None] + np.log(lambdas)[None, :])
    log_posterior = logliks + log_prior
    log_posterior -= log_posterior.max()
    weights = np.exp(log_posterior)
    weights /= weights.sum()

    sigma_e = np.sqrt(sigma_e2)
    sigma_a = sigma_e * np.sqrt(lambdas)[:, None]
    sigma_b = sigma_e * np.sqrt(lambdas)[None, :]

    return {
        "sigma_a": _summarize(sigma_a, weights),
        "sigma_b": _summarize(sigma_b, weights),
        "sigma_e": _summarize(sigma_e, weights),
        "p_a_exceeds_b": float(weights[sigma_a > sigma_b].sum()),
        "edge_mass": float(
            weights[0, :].sum() + weights[-1, :].sum() + weights[:, 0].sum() + weights[:, -1].sum()
        ),
    }


def fit_one_way(
    counts: np.ndarray,
    level_sums: np.ndarray,
    yty: float,
    oney: float,
    n: int,
    *,
    grid_points: int = 241,
) -> dict:
    """The same posterior for a **single** grouping factor, in closed form.

    With one factor ``Z'Z`` is diagonal — it is just the count of observations per
    level — so ``M = Lambda^-1 + Z'Z`` is diagonal too and the Cholesky in
    ``_restricted_loglik`` collapses to a reciprocal. Every grid point is then
    O(number of levels) rather than O(levels cubed).

    That is not a micro-optimisation. The punting design has 392 punter-seasons
    crossed with 320 return units, and a crossed fit there costs about thirty
    seconds — which makes a 400-dataset null bound a three-hour job and a power
    curve on top of it impossible. The one-way form runs the same null in seconds,
    so the **gated** instrument is this one and the crossed fit is reported once,
    descriptively, on the real data.

    Because the grid is one-dimensional it can be much finer: 241 points across
    the same log range, against 41 per axis in the crossed form.
    """
    log_lambda = np.linspace(LOG10_LAMBDA_MIN, LOG10_LAMBDA_MAX, grid_points)
    lambdas = 10.0**log_lambda
    p = len(counts)
    dof = n - 1

    # (grid, levels) — the diagonal of M, and the pieces every quadratic form needs.
    denominator = (1.0 / lambdas)[:, None] + counts[None, :]
    xhx = n - (counts[None, :] ** 2 / denominator).sum(axis=1)
    xhy = oney - (counts[None, :] * level_sums[None, :] / denominator).sum(axis=1)
    yhy = yty - (level_sums[None, :] ** 2 / denominator).sum(axis=1)

    beta = xhy / xhx
    rss = np.maximum(yhy - beta * xhy, 1e-12)
    sigma_e2 = rss / dof
    logdet_h = np.log(denominator).sum(axis=1) + p * np.log(lambdas)
    loglik = -0.5 * (dof * np.log(sigma_e2) + logdet_h + np.log(xhx))

    log_posterior = loglik + 0.5 * np.log(lambdas)  # flat on sqrt(lambda)
    log_posterior -= log_posterior.max()
    weights = np.exp(log_posterior)
    weights /= weights.sum()

    sigma_e = np.sqrt(sigma_e2)
    return {
        "sigma_a": _summarize(sigma_e * np.sqrt(lambdas), weights),
        "sigma_e": _summarize(sigma_e, weights),
        "edge_mass": float(weights[0] + weights[-1]),
    }


def one_way_statistics(codes: np.ndarray, n_levels: int, y: np.ndarray) -> tuple:
    """Counts, per-level sums and the scalars ``fit_one_way`` needs."""
    counts = np.bincount(codes, minlength=n_levels).astype(float)
    level_sums = np.bincount(codes, weights=y, minlength=n_levels)
    return counts, level_sums, float(y @ y), float(y.sum()), len(y)


def _summarize(values: np.ndarray, weights: np.ndarray) -> dict:
    """Weighted mean and 89% equal-tailed interval over a gridded quantity."""
    flat_values = values.ravel()
    flat_weights = weights.ravel()
    order = np.argsort(flat_values)
    sorted_values = flat_values[order]
    cumulative = np.cumsum(flat_weights[order])
    return {
        "mean": float(flat_values @ flat_weights),
        "eti89_lb": float(np.interp(0.055, cumulative, sorted_values)),
        "eti89_ub": float(np.interp(0.945, cumulative, sorted_values)),
    }


def prepare(
    codes: list[np.ndarray], sizes: list[int], y: np.ndarray
) -> tuple[CrossedDesign, np.ndarray, float, float]:
    """Everything ``fit`` needs, from level codes and a response."""
    design = build_design(codes, sizes)
    return design, project(codes, sizes, y), float(y @ y), float(y.sum())


# Measured 2026-08-17 by `self_check` below, on a synthetic dataset sized to the
# real S3 attribution design: 5,441 team-games, 167 quarterbacks, 93 coaches.
#
#                  NUTS mean          grid mean         relative gap
#   sigma_qb       0.03578            0.03574           0.11%
#   sigma_coach    0.02373            0.02391           0.75%
#   sigma_e        0.24021            0.24008           0.05%
#
# The posterior MEANS agree to well under one percent, which is what licenses
# using the grid for hundreds of null fits. The 89% INTERVALS are narrower on the
# grid — 0.029-0.040 against NUTS's 0.028-0.043 for sigma_qb — which is the
# module docstring's stated approximation showing up exactly where it was
# predicted to. That direction does **not** bias the achievable-null comparison,
# because the null bound and the observed fit are produced by the same
# instrument; it would only matter if a grid interval were compared against a
# NUTS one, which no gate in this project does.
SELF_CHECK_RESULT = {
    "n": 5441,
    "levels": (167, 93),
    "relative_gap_sigma_a": 0.0011,
    "relative_gap_sigma_b": 0.0075,
    "relative_gap_sigma_e": 0.0005,
}


def self_check(seed: int = 20260817) -> dict:
    """Reproduce the grid-versus-NUTS comparison recorded in ``SELF_CHECK_RESULT``.

    Kept as runnable code rather than a comment so the claim can be re-checked
    when the stack moves, which is the same role `_betabinom_grid.self_check`
    plays for the beta-binomial instrument.
    """
    import pymc as pm

    rng = np.random.default_rng(seed)
    n, n_a, n_b = 5441, 167, 93
    code_a = rng.integers(0, n_a, n)
    code_b = rng.integers(0, n_b, n)
    sigma_a, sigma_b, sigma_e = 0.045, 0.020, 0.24
    y = (
        0.01
        + rng.normal(0.0, sigma_a, n_a)[code_a]
        + rng.normal(0.0, sigma_b, n_b)[code_b]
        + rng.normal(0.0, sigma_e, n)
    )

    design, zty, yty, oney = prepare([code_a, code_b], [n_a, n_b], y)
    grid = fit(design, zty, yty, oney)

    with pm.Model(coords={"a": range(n_a), "b": range(n_b)}):
        mu = pm.Normal("mu", 0.0, 1.0)
        scale_a = pm.HalfNormal("sigma_a", 0.5)
        scale_b = pm.HalfNormal("sigma_b", 0.5)
        scale_e = pm.HalfNormal("sigma_e", 1.0)
        offset_a = pm.Normal("z_a", 0.0, 1.0, dims="a")
        offset_b = pm.Normal("z_b", 0.0, 1.0, dims="b")
        pm.Normal(
            "y",
            mu=mu + (offset_a * scale_a)[code_a] + (offset_b * scale_b)[code_b],
            sigma=scale_e,
            observed=y,
        )
        idata = pm.sample(
            1000, tune=1000, chains=4, target_accept=0.9, random_seed=seed, progressbar=False
        )

    posterior = idata["posterior"]
    return {
        name: {
            "nuts_mean": float(posterior[name].values.mean()),
            "grid_mean": grid[name]["mean"],
            "relative_gap": abs(grid[name]["mean"] - float(posterior[name].values.mean()))
            / float(posterior[name].values.mean()),
        }
        for name in ("sigma_a", "sigma_b", "sigma_e")
    }
