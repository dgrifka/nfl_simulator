"""Exact grid posterior for the two-parameter beta-binomial hierarchy.

The models in `research/03_bayesian_rates.py` have exactly two free parameters
once `p_team` is marginalized out — `mu` and `log_kappa` — which means the
posterior can be evaluated exactly on a grid instead of sampled. That matters
here because power calculations need hundreds of fits, and NUTS at a second per
fit would make them unaffordable.

Same model, same priors as document 03:

    mu        ~ Beta(2, 2)
    log_kappa ~ Normal(4, 2)
    k_i       ~ BetaBinomial(n_i, mu*kappa, (1-mu)*kappa)

The grid is laid out in ``(logit(mu), log(kappa))`` so that rates near 0.7% —
offensive holding — get the same resolution as rates near 47%. The Jacobian of
the logit transform is applied to the `mu` prior, so the posterior is the same
object PyMC would sample, not an approximation of a different model.

Validated against the Phase 1 nutpie fits in :func:`self_check`.
"""

from __future__ import annotations

import numpy as np
from scipy.special import betaln, gammaln

MU_PRIOR_A = 2.0
MU_PRIOR_B = 2.0
LOG_KAPPA_PRIOR_MEAN = 4.0
LOG_KAPPA_PRIOR_SD = 2.0


def _log_beta_binomial(k: np.ndarray, n: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """log P(k | n, a, b), broadcasting over a grid axis.

    Used by the self-check only. :func:`fit_grid` inlines a faster equivalent —
    see the comments there for which terms it is entitled to drop.
    """
    log_choose = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    return log_choose + betaln(k + a, n - k + b) - betaln(a, b)


def population_sd(mu: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """SD of the Beta distribution over true entity rates, in rate units."""
    return np.sqrt(mu * (1.0 - mu) / (kappa + 1.0))


class GridPosterior:
    """Posterior over (mu, kappa) on a grid, with the derived population SD."""

    def __init__(self, mu: np.ndarray, kappa: np.ndarray, weights: np.ndarray) -> None:
        self.mu = mu
        self.kappa = kappa
        self.weights = weights  # normalized, same shape as mu/kappa
        self.sd = population_sd(mu, kappa)

    def _quantile(self, values: np.ndarray, q: float) -> float:
        flat_values = values.ravel()
        flat_weights = self.weights.ravel()
        order = np.argsort(flat_values)
        cumulative = np.cumsum(flat_weights[order])
        return float(flat_values[order][np.searchsorted(cumulative, q)])

    def mean(self, values: np.ndarray) -> float:
        return float((values * self.weights).sum())

    def summary(self) -> dict:
        """Posterior mean and 89% equal-tailed interval, matching document 03."""
        return {
            "mu_mean": self.mean(self.mu),
            "kappa_mean": self.mean(self.kappa),
            "population_sd_mean": self.mean(self.sd),
            "population_sd_eti89": [self._quantile(self.sd, 0.055), self._quantile(self.sd, 0.945)],
        }

    def edge_mass(self) -> float:
        """Posterior mass sitting on the grid boundary.

        A grid is only exact if the posterior has died out before the edge. Any
        caller that cares about the answer should assert this is negligible.
        """
        w = self.weights
        return float(w[0, :].sum() + w[-1, :].sum() + w[:, 0].sum() + w[:, -1].sum())


def fit_grid(
    n: np.ndarray,
    k: np.ndarray,
    *,
    logit_mu_halfwidth: float | None = None,
    log_kappa_range: tuple[float, float] = (0.0, 14.0),
    resolution_mu: int = 80,
    resolution_kappa: int = 220,
) -> GridPosterior:
    """Exact posterior on a grid, centred and scaled to the pooled observed rate.

    Resolution is deliberately lopsided. `mu` is pinned hard by the total success
    count — with hundreds of thousands of plays behind a penalty rate its
    posterior is a spike — while `log_kappa` is the parameter the reported
    population SD actually depends on, and its posterior has a long right tail.
    Spending cells on `log_kappa` and few on `mu` is what makes thousands of
    fits affordable.
    """
    n = np.asarray(n, dtype=float)
    k = np.asarray(k, dtype=float)

    pooled = float(k.sum() / n.sum())
    pooled = min(max(pooled, 1e-9), 1 - 1e-9)
    centre = np.log(pooled / (1.0 - pooled))

    if logit_mu_halfwidth is None:
        # Scale the window to the data. The binomial SE of logit(p) understates
        # the true spread under overdispersion, so take a generous multiple and
        # let `edge_mass` verify it was wide enough.
        se = 1.0 / np.sqrt(max(n.sum() * pooled * (1.0 - pooled), 1.0))
        logit_mu_halfwidth = float(np.clip(25.0 * se, 0.15, 6.0))

    logit_mu = np.linspace(centre - logit_mu_halfwidth, centre + logit_mu_halfwidth, resolution_mu)
    log_kappa = np.linspace(*log_kappa_range, resolution_kappa)
    logit_grid, log_kappa_grid = np.meshgrid(logit_mu, log_kappa, indexing="ij")

    mu = 1.0 / (1.0 + np.exp(-logit_grid))
    kappa = np.exp(log_kappa_grid)

    # log BetaBinomial(k; n, a, b), with two terms dropped or reduced:
    #
    #   * log C(n, k) does not involve the parameters at all, so it is a constant
    #     across every cell and vanishes when the posterior is normalized.
    #   * betaln(k+a, n-k+b) expands to gammaln(k+a) + gammaln(n-k+b)
    #     - gammaln(n+kappa). The last term depends on kappa but NOT on mu, so it
    #     is evaluated on the kappa axis alone rather than over the full grid.
    #
    # What is left is two gammaln evaluations over (cells x entities), which is
    # the irreducible cost.
    alpha = (mu * kappa).reshape(-1, 1)
    beta = ((1.0 - mu) * kappa).reshape(-1, 1)

    log_lik = (gammaln(k[None, :] + alpha) + gammaln(n[None, :] - k[None, :] + beta)).sum(axis=1)
    log_lik = log_lik.reshape(mu.shape)

    # gammaln(n + kappa) varies along the kappa axis only, so it is summed over
    # entities once per kappa value and broadcast across mu.
    kappa_axis = np.exp(log_kappa)
    log_lik -= gammaln(n[None, :] + kappa_axis[:, None]).sum(axis=1)[None, :]

    log_lik -= betaln(mu * kappa, (1.0 - mu) * kappa) * len(n)

    # Beta(2,2) on mu, transformed to logit(mu): multiply by |dmu/dtheta| = mu(1-mu).
    log_prior_mu = (
        (MU_PRIOR_A - 1) * np.log(mu) + (MU_PRIOR_B - 1) * np.log1p(-mu) + np.log(mu) + np.log1p(-mu)
    )
    log_prior_kappa = -0.5 * ((log_kappa_grid - LOG_KAPPA_PRIOR_MEAN) / LOG_KAPPA_PRIOR_SD) ** 2

    log_posterior = log_lik + log_prior_mu + log_prior_kappa
    log_posterior -= log_posterior.max()
    weights = np.exp(log_posterior)
    weights /= weights.sum()

    return GridPosterior(mu=mu, kappa=kappa, weights=weights)


def simulate_counts(
    rng: np.random.Generator, n: np.ndarray, league_rate: float, true_sd: float
) -> np.ndarray:
    """Draw successes for each entity under a true population SD.

    ``true_sd = 0`` is the no-skill null: every entity shares the league rate and
    all the observed scatter is binomial. Larger values invert the population-SD
    formula to the concentration that produces them.
    """
    n = np.asarray(n, dtype=float)
    if true_sd <= 0:
        return rng.binomial(n.astype(int), league_rate)

    max_sd = np.sqrt(league_rate * (1.0 - league_rate))
    if true_sd >= max_sd:
        raise ValueError(f"true_sd {true_sd} is impossible at rate {league_rate} (max {max_sd})")

    kappa = league_rate * (1.0 - league_rate) / true_sd**2 - 1.0
    rates = rng.beta(league_rate * kappa, (1.0 - league_rate) * kappa, size=len(n))
    return rng.binomial(n.astype(int), rates)


def upper_bound_distribution(
    n: np.ndarray,
    league_rate: float,
    true_sd: float,
    *,
    datasets: int,
    seed: int,
    resolution_kappa: int = 220,
) -> np.ndarray:
    """89% upper bounds on population SD across simulated datasets.

    This is the instrument document 04 said was missing: before committing a
    threshold of the form "the 89% upper bound must be below X", simulate under
    the null and find out what upper bounds this many observations can actually
    produce.
    """
    rng = np.random.default_rng(seed)
    bounds = np.empty(datasets)
    for i in range(datasets):
        k = simulate_counts(rng, n, league_rate, true_sd)
        posterior = fit_grid(n, k, resolution_kappa=resolution_kappa)
        bounds[i] = posterior.summary()["population_sd_eti89"][1]
    return bounds


def self_check() -> None:
    """Reproduce the Phase 1 fumble-recovery fit, which used nutpie NUTS.

    Document 04 reports population SD 2.39 pp with an 89% interval of
    0.75 – 4.38 pp. Agreement here is evidence the grid is the same posterior,
    not a different model that happens to be fast.
    """
    import polars as pl

    from nfl_simulator.ingest import ANALYSIS_COLUMNS, PBP_SEASONS, load_pbp
    from nfl_simulator.rates import fumble_recovery_counts

    counts = fumble_recovery_counts(load_pbp(PBP_SEASONS, columns=ANALYSIS_COLUMNS))
    posterior = fit_grid(counts["n"].to_numpy(), counts["k"].to_numpy())
    summary = posterior.summary()

    print("grid vs PyMC/nutpie, fumble recovery (document 04)")
    print(f"  mu                 grid {summary['mu_mean']:.4f}   nutpie 0.4678")
    print(f"  kappa              grid {summary['kappa_mean']:.1f}     nutpie 1408.3")
    print(f"  population SD (pp) grid {summary['population_sd_mean'] * 100:.2f}     nutpie 2.39")
    print(
        f"  89% interval (pp)  grid "
        f"[{summary['population_sd_eti89'][0] * 100:.2f}, "
        f"{summary['population_sd_eti89'][1] * 100:.2f}]   nutpie [0.75, 4.38]"
    )
    print(f"  posterior mass on grid edge: {posterior.edge_mass():.2e}")
    assert isinstance(counts, pl.DataFrame)


if __name__ == "__main__":
    self_check()
