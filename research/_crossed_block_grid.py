"""The crossed Gaussian grid of `research/_crossed_gaussian_grid.py`, computed
the fast way — same posterior, same grid, a Schur complement instead of a
``p x p`` Cholesky.

**This is not a new instrument.** It evaluates document 43 §6's instrument at
exactly the grid points that module evaluates, with the same profiled restricted
likelihood and the same prior, and :func:`self_check` is the licence: it fits
both implementations on the same data and reports the largest disagreement in
every summary the caller reads. Round 7 uses it because round 6's implementation
does not scale, and the reason is arithmetic rather than opinion.

**Why it is needed.** The receiver-drop study's gate arm crosses receiver-season
(1,931 levels) with defence-season (128), so ``p = 2,059`` and one grid point
costs a 2,059-square Cholesky plus two general solves. Measured on this machine:
**183.8 s for one fit**, against 1.6 s for the 256-level team-season design.
Part A needs 400 datasets in each of five cells at that grain, which is about
**100 hours**. The dropped-pick study never met this because its largest crossed
design was 408 levels.

**The arithmetic.** For two crossed grouping factors every row carries exactly
one 1 per factor, so each diagonal block of ``Z'Z`` is the *diagonal* matrix of
that factor's level counts, and only the off-diagonal co-occurrence block is
dense::

    M = Z'Z + diag(precision) = [[A, N ],     A = diag(count_a + 1 / lambda_a)
                                 [N', B]]     B = diag(count_b + 1 / lambda_b)

``A`` is diagonal, so it inverts for free and the Schur complement

    S = B - N' A^-1 N          (128 x 128, not 2,059 x 2,059)

carries everything::

    M^-1 v = [ A^-1 (v_a - N x_b) ; x_b ],   x_b = S^-1 (v_b - N' A^-1 v_a)
    log|M| = log|A| + log|S|

and ``A`` depends only on ``lambda_a``, so ``N' A^-1 N`` is formed 41 times per
fit rather than 1,681 times. What was a 2,059-cubed Cholesky at every grid point
becomes a 128-cubed one.

The two factors are **not** interchangeable here: the smaller one must be second,
because the Schur complement is taken on its block. :func:`fit_blocked` sorts
that out itself rather than trusting the caller.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

sys.path.insert(0, str(Path(__file__).parent))

_crossed = import_module("_crossed_gaussian_grid")

# The grid is the other module's, imported rather than restated: two grids that
# could drift apart would make the reproduction check below meaningless.
LOG10_LAMBDA_MIN = _crossed.LOG10_LAMBDA_MIN
LOG10_LAMBDA_MAX = _crossed.LOG10_LAMBDA_MAX
GRID_POINTS = _crossed.GRID_POINTS


@dataclass(frozen=True)
class BlockDesign:
    """``Z'Z`` for two crossed factors, stored as the three blocks it really is.

    ``counts_a`` and ``counts_b`` are the diagonal blocks; ``cross`` is the
    ``(size_a, size_b)`` co-occurrence count matrix. Nothing here is ever
    materialised as a ``p x p`` array, which is the whole point.
    """

    counts_a: np.ndarray
    counts_b: np.ndarray
    cross: np.ndarray
    n: int

    @property
    def size_a(self) -> int:
        return len(self.counts_a)

    @property
    def size_b(self) -> int:
        return len(self.counts_b)


def build_blocks(code_a: np.ndarray, code_b: np.ndarray, size_a: int, size_b: int) -> BlockDesign:
    """The three blocks, from integer level codes."""
    code_a = np.asarray(code_a, dtype=np.int64)
    code_b = np.asarray(code_b, dtype=np.int64)
    counts_a = np.bincount(code_a, minlength=size_a).astype(float)
    counts_b = np.bincount(code_b, minlength=size_b).astype(float)
    flat = code_a * size_b + code_b
    cross = np.bincount(flat, minlength=size_a * size_b).astype(float)
    return BlockDesign(
        counts_a=counts_a,
        counts_b=counts_b,
        cross=cross.reshape(size_a, size_b),
        n=int(len(code_a)),
    )


def project_blocks(
    code_a: np.ndarray, code_b: np.ndarray, size_a: int, size_b: int, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``Z' values``, kept as one vector per factor."""
    return (
        np.bincount(code_a, weights=values, minlength=size_a),
        np.bincount(code_b, weights=values, minlength=size_b),
    )


def fit_blocked(
    design: BlockDesign,
    zty_a: np.ndarray,
    zty_b: np.ndarray,
    yty: float,
    oney: float,
    *,
    grid_points: int = GRID_POINTS,
) -> dict:
    """`_crossed_gaussian_grid.fit`'s return value, computed block-wise.

    The outer loop is over ``lambda_a`` so that everything depending on ``A``
    alone — its inverse, the cross product ``N' A^-1 N``, and the two projected
    right-hand sides — is formed once per row of the grid instead of once per
    grid point. The inner loop then costs one 128-square Cholesky.
    """
    log_lambda = np.linspace(LOG10_LAMBDA_MIN, LOG10_LAMBDA_MAX, grid_points)
    lambdas = 10.0**log_lambda

    counts_a, counts_b, cross = design.counts_a, design.counts_b, design.cross
    size_a, size_b, n = design.size_a, design.size_b, design.n
    dof = n - 1

    logliks = np.full((grid_points, grid_points), -np.inf)
    sigma_e2 = np.zeros((grid_points, grid_points))

    for i, lambda_a in enumerate(lambdas):
        diag_a = counts_a + 1.0 / lambda_a
        inv_a = 1.0 / diag_a
        logdet_a = float(np.log(diag_a).sum())
        # (size_b, size_b) — the only dense product in the fit, 41 of them.
        weighted = cross * inv_a[:, None]
        w = weighted.T @ cross
        # A^-1 v_a for the two right-hand sides, and their projections through N'.
        u_one = counts_a * inv_a
        u_y = zty_a * inv_a
        rhs_one = counts_b - cross.T @ u_one
        rhs_y = zty_b - cross.T @ u_y

        for j, lambda_b in enumerate(lambdas):
            schur = -w.copy()
            schur[np.diag_indices(size_b)] += counts_b + 1.0 / lambda_b
            try:
                factor = cho_factor(schur, lower=True, check_finite=False)
            except np.linalg.LinAlgError:
                continue
            triangular = factor[0]
            if not np.all(np.diag(triangular) > 0):
                continue

            xb_one = cho_solve(factor, rhs_one, check_finite=False)
            xb_y = cho_solve(factor, rhs_y, check_finite=False)
            xa_one = u_one - inv_a * (cross @ xb_one)
            xa_y = u_y - inv_a * (cross @ xb_y)

            one_m_one = float(counts_a @ xa_one + counts_b @ xb_one)
            one_m_y = float(counts_a @ xa_y + counts_b @ xb_y)
            y_m_y = float(zty_a @ xa_y + zty_b @ xb_y)

            xhx = n - one_m_one
            xhy = oney - one_m_y
            yhy = yty - y_m_y
            beta = xhy / xhx
            rss = yhy - beta * xhy
            if not (rss > 0.0 and xhx > 0.0):
                continue
            variance = rss / dof

            logdet_m = logdet_a + 2.0 * float(np.log(np.diag(triangular)).sum())
            logdet_h = logdet_m + np.log(lambda_a) * size_a + np.log(lambda_b) * size_b
            logliks[i, j] = -0.5 * (dof * np.log(variance) + logdet_h + np.log(xhx))
            sigma_e2[i, j] = variance

    # From here the assembly is `_crossed_gaussian_grid.fit`'s, line for line.
    log_prior = 0.5 * (np.log(lambdas)[:, None] + np.log(lambdas)[None, :])
    log_posterior = logliks + log_prior
    log_posterior -= log_posterior.max()
    weights = np.exp(log_posterior)
    weights /= weights.sum()

    sigma_e = np.sqrt(sigma_e2)
    sigma_a = sigma_e * np.sqrt(lambdas)[:, None]
    sigma_b = sigma_e * np.sqrt(lambdas)[None, :]

    return {
        "sigma_a": _crossed._summarize(sigma_a, weights),
        "sigma_b": _crossed._summarize(sigma_b, weights),
        "sigma_e": _crossed._summarize(sigma_e, weights),
        "p_a_exceeds_b": float(weights[sigma_a > sigma_b].sum()),
        "edge_mass": float(
            weights[0, :].sum() + weights[-1, :].sum() + weights[:, 0].sum() + weights[:, -1].sum()
        ),
    }


def fit_from_codes(
    code_a: np.ndarray,
    code_b: np.ndarray,
    size_a: int,
    size_b: int,
    values: np.ndarray,
) -> dict:
    """`fit_blocked` from codes and a response, with the blocks built for you.

    The factor with fewer levels is put second, because the Schur complement is
    taken on the second block and taking it on the larger one would throw the
    speed-up away. ``sigma_a`` and ``sigma_b`` are swapped back afterwards, so a
    caller reads the SDs in the order it asked for them.
    """
    swap = size_b > size_a
    if swap:
        code_a, code_b = code_b, code_a
        size_a, size_b = size_b, size_a

    design = build_blocks(code_a, code_b, size_a, size_b)
    zty_a, zty_b = project_blocks(code_a, code_b, size_a, size_b, values)
    fitted = fit_blocked(design, zty_a, zty_b, float(values @ values), float(values.sum()))
    if swap:
        fitted = {
            **fitted,
            "sigma_a": fitted["sigma_b"],
            "sigma_b": fitted["sigma_a"],
            "p_a_exceeds_b": 1.0 - fitted["p_a_exceeds_b"],
        }
    return fitted


def self_check(seed: int = 20260827, *, verbose: bool = True) -> dict:
    """Reproduce `_crossed_gaussian_grid.fit` on a problem small enough to run it.

    The licence for using this module in place of the other one, in the shape
    `_crossed_gaussian_grid.self_check` used to license the grid against a PyMC
    fit. Two designs are checked: a balanced-ish one, and one with the receiver
    study's shape in miniature (many thin levels on one factor, few fat ones on
    the other), because a Schur complement on a nearly-singular block is where
    an equivalence claim would break if it were going to.
    """
    rng = np.random.default_rng(seed)
    cases = []
    for label, size_a, size_b, n in (
        ("balanced 40 x 12", 40, 12, 900),
        ("thin-levels 300 x 16", 300, 16, 2400),
    ):
        code_a = rng.integers(0, size_a, n)
        code_b = rng.integers(0, size_b, n)
        y = (
            rng.normal(0.0, 0.30, size_a)[code_a]
            + rng.normal(0.0, 0.12, size_b)[code_b]
            + rng.normal(0.0, 1.0, n)
        )
        reference = _crossed.fit(
            _crossed.build_design([code_a, code_b], [size_a, size_b]),
            _crossed.project([code_a, code_b], [size_a, size_b], y),
            float(y @ y),
            float(y.sum()),
        )
        blocked = fit_from_codes(code_a, code_b, size_a, size_b, y)

        gaps = {}
        for parameter in ("sigma_a", "sigma_b", "sigma_e"):
            for statistic in ("mean", "eti89_lb", "eti89_ub"):
                gaps[f"{parameter}.{statistic}"] = abs(
                    reference[parameter][statistic] - blocked[parameter][statistic]
                )
        gaps["edge_mass"] = abs(reference["edge_mass"] - blocked["edge_mass"])
        gaps["p_a_exceeds_b"] = abs(reference["p_a_exceeds_b"] - blocked["p_a_exceeds_b"])
        cases.append(
            {
                "case": label,
                "rows": n,
                "levels": [size_a, size_b],
                "max_abs_gap": float(max(gaps.values())),
                "gaps": {name: float(value) for name, value in gaps.items()},
                "reference_sigma_a_eti89_ub": reference["sigma_a"]["eti89_ub"],
                "blocked_sigma_a_eti89_ub": blocked["sigma_a"]["eti89_ub"],
            }
        )

    worst = max(case["max_abs_gap"] for case in cases)
    report = {
        "statistic": (
            "largest absolute disagreement between `_crossed_gaussian_grid.fit` and "
            "`fit_blocked` over every summary a caller reads"
        ),
        "tolerance": 1e-10,
        "max_abs_gap": worst,
        "cases": cases,
        "pass": bool(worst <= 1e-10),
    }
    if verbose:
        print("\n=== the blocked grid against the p x p one (the licence) ===")
        for case in cases:
            print(
                f"  {case['case']:22s} {case['rows']:5,d} rows, "
                f"{case['levels'][0]} x {case['levels'][1]} levels: "
                f"max |Δ| {case['max_abs_gap']:.2e}   "
                f"(σ_a 89% ub {case['reference_sigma_a_eti89_ub']:.6f} vs "
                f"{case['blocked_sigma_a_eti89_ub']:.6f})"
            )
        print(
            f"  worst disagreement {worst:.2e} against a {report['tolerance']:.0e} "
            f"tolerance -> {'PASS' if report['pass'] else 'FAIL'}"
        )
    return report


if __name__ == "__main__":
    self_check()
