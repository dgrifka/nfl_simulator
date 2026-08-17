"""Field-goal make probability, as the simulator consumes it.

`research/07_fg_model.py` fits the hierarchical model pre-registered in
`docs/research/05b-fg-model-foundations.md`; this is the read side. It answers
one question — *what was the probability this kicker made this kick from this
distance?* — and answers it **once per posterior draw** rather than as a point
estimate.

That vector return is not a convenience. Document 05 §4 requires the
deserve-to-win interval to carry the uncertainty in `p` itself, not just the
coin flip, and a model that collapsed to a mean here would quietly make the
simulator's intervals too tight.

The adopted form is the quadratic fallback, because the pre-registered linear
distance term failed Gate FG-2:

    logit p = alpha + beta * (distance - 40) + gamma * (distance - 40)^2 / 100
              + kicker_effect
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_DISTANCE_CENTRE = 40.0


@dataclass(frozen=True)
class FieldGoalModel:
    """Posterior draws of the make-probability curve plus per-kicker effects."""

    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    kicker_effects: dict[str, np.ndarray] = field(default_factory=dict)
    distance_centre: float = DEFAULT_DISTANCE_CENTRE

    def __post_init__(self) -> None:
        lengths = {len(self.alpha), len(self.beta), len(self.gamma)}
        if len(lengths) != 1:
            raise ValueError(
                "alpha, beta and gamma must have the same number of posterior draws, "
                f"got {sorted(lengths)}"
            )
        for name, effect in self.kicker_effects.items():
            if len(effect) != len(self.alpha):
                raise ValueError(
                    f"kicker effect for {name} has {len(effect)} draws, expected {len(self.alpha)}"
                )

    @property
    def n_draws(self) -> int:
        return len(self.alpha)

    def _logit(self, distance: float) -> np.ndarray:
        centred = distance - self.distance_centre
        # Divided by 100 to match the fitted parameterization, which scales the
        # quadratic so its coefficient sits on the same order as the slope.
        return self.alpha + self.beta * centred + self.gamma * centred**2 / 100.0

    def league_make_probability(self, distance: float) -> np.ndarray:
        """Make probability for an average kicker, one value per posterior draw."""
        return _sigmoid(self._logit(distance))

    def make_probability(self, kicker_season: str | None, distance: float) -> np.ndarray:
        """Make probability for this kicker, one value per posterior draw.

        A kicker with no fitted effect — a rookie, or anyone outside the fitted
        seasons — falls back to the league curve rather than borrowing a
        neighbour's effect. Under document 05 §1's rule that is the `w = 0`
        endpoint: no evidence about this entity, so no entity term.
        """
        effect = self.kicker_effects.get(kicker_season) if kicker_season else None
        if effect is None:
            return self.league_make_probability(distance)
        return _sigmoid(self._logit(distance) + effect)

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------

    @classmethod
    def from_posterior(
        cls, trace_path: str | Path, *, distance_centre: float = DEFAULT_DISTANCE_CENTRE
    ) -> FieldGoalModel:
        """Load the fitted posterior written by `research/07_fg_model.py`."""
        import arviz as az

        trace_path = Path(trace_path)
        if not trace_path.exists():
            raise FileNotFoundError(
                f"no fitted field-goal posterior at {trace_path} — "
                "run `uv run python research/07_fg_model.py`"
            )

        posterior = az.from_netcdf(trace_path)["posterior"]
        alpha = posterior["alpha"].values.ravel()
        beta = posterior["beta"].values.ravel()
        # The linear arm has no gamma. Zeros keep the same code path rather than
        # branching on which arm was adopted.
        gamma = posterior["gamma"].values.ravel() if "gamma" in posterior else np.zeros_like(alpha)

        kicker = posterior["kicker"]
        levels = [str(level) for level in kicker.coords["kicker_season"].values]
        draws = kicker.values.reshape(-1, len(levels))
        effects = {level: draws[:, i] for i, level in enumerate(levels)}

        return cls(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            kicker_effects=effects,
            distance_centre=distance_centre,
        )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Clipped so an absurd distance returns a probability rather than a 0, a 1,
    # or an overflow warning. The simulator must never book infinite luck.
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))
