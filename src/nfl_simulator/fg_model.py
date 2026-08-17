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
distance term failed Gate FG-2, plus the weather terms added in Phase 3:

    logit p = alpha + beta * (distance - 40) + gamma * (distance - 40)^2 / 100
              + roof_effect
              + beta_wind * (wind - wind_centre) * has_weather
              + beta_temp * (temp - temp_centre) * has_weather
              + kicker_effect

Weather is optional throughout. A posterior fitted before Phase 3 carries no
weather parameters, and passing weather to such a model is a no-op rather than
an error — the Phase 2 ledger has to stay reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_DISTANCE_CENTRE = 40.0

# Roofs under which no weather reaches the ball. nflverse leaves temp and wind
# null on 3,196 of the 3,200 dome and closed-roof attempts; the four exceptions
# carry a stadium-ambient reading that leaked through.
INDOOR_ROOFS: frozenset[str] = frozenset({"dome", "closed"})

# The raw `wind` column's maximum is 71 mph, on three attempts in one 2016 game.
# That is not a credible sustained wind for a game that was played, and the bins
# above 25 mph hold 31 attempts across ten seasons — far too few to carry a
# slope. Uncapped, three kicks would lever the league's wind coefficient.
WIND_CAP_MPH = 30.0


@dataclass(frozen=True)
class Weather:
    """Conditions for one kick, after sanitizing.

    ``has_weather`` is deliberately all-or-nothing. Both terms are centred on
    their outdoor league means, so a kick with a wind reading but no temperature
    would be centred on one and guessed on the other — a half-measurement priced
    as a whole one. Treating it as no measurement is the honest reading, and it
    costs nothing: the two columns are null together in every row of the source.
    """

    roof: str | None
    wind: float | None
    temp: float | None

    @property
    def has_weather(self) -> bool:
        return self.wind is not None and self.temp is not None


def sanitize_weather(roof: str | None, wind: float | None, temp: float | None) -> Weather:
    """Apply the data-quality rules from `docs/research/05b` §10 before use.

    Two rules, and both exist because a wrong reading is more dangerous than a
    missing one — a missing value is visibly missing, while a wrong one is
    silently used:

    * **Indoors, there is no weather.** A reading recorded under a closed roof
      describes the air outside the stadium, not the air the ball flew through.
    * **Wind is capped.** See ``WIND_CAP_MPH``.

    This is the single implementation. The fit and the simulator both call it,
    so the model cannot be trained on one definition of a windy day and applied
    to another.
    """
    if roof in INDOOR_ROOFS:
        return Weather(roof=roof, wind=None, temp=None)
    return Weather(
        roof=roof,
        wind=None if wind is None else min(float(wind), WIND_CAP_MPH),
        temp=None if temp is None else float(temp),
    )


@dataclass(frozen=True)
class FieldGoalModel:
    """Posterior draws of the make-probability curve plus per-kicker effects."""

    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    kicker_effects: dict[str, np.ndarray] = field(default_factory=dict)
    distance_centre: float = DEFAULT_DISTANCE_CENTRE
    roof_effects: dict[str, np.ndarray] = field(default_factory=dict)
    beta_wind: np.ndarray | None = None
    beta_temp: np.ndarray | None = None
    wind_centre: float = 0.0
    temp_centre: float = 0.0

    def __post_init__(self) -> None:
        lengths = {len(self.alpha), len(self.beta), len(self.gamma)}
        if len(lengths) != 1:
            raise ValueError(
                "alpha, beta and gamma must have the same number of posterior draws, "
                f"got {sorted(lengths)}"
            )
        for label, effects in (("kicker", self.kicker_effects), ("roof", self.roof_effects)):
            for name, effect in effects.items():
                if len(effect) != len(self.alpha):
                    raise ValueError(
                        f"{label} effect for {name} has {len(effect)} draws, "
                        f"expected {len(self.alpha)}"
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

    def _weather_logit(self, weather: Weather | None) -> np.ndarray | float:
        """Log-odds adjustment for conditions. Zero whenever there is nothing to say.

        Every fallback here is to *no adjustment*, never to a neighbour's value:
        an unknown roof level, a model fitted without weather, and an outdoor
        kick with no reading all return zero. That is the same `w = 0` endpoint
        document 05 §1 gives an unknown kicker — no evidence about this kick's
        conditions, so no conditions term.
        """
        if weather is None:
            return 0.0

        adjustment: np.ndarray | float = 0.0
        roof_effect = self.roof_effects.get(weather.roof) if weather.roof else None
        if roof_effect is not None:
            adjustment = adjustment + roof_effect

        if weather.has_weather:
            if self.beta_wind is not None:
                adjustment = adjustment + self.beta_wind * (weather.wind - self.wind_centre)
            if self.beta_temp is not None:
                adjustment = adjustment + self.beta_temp * (weather.temp - self.temp_centre)
        return adjustment

    def make_probability(
        self,
        kicker_season: str | None,
        distance: float,
        *,
        weather: Weather | None = None,
    ) -> np.ndarray:
        """Make probability for this kicker in these conditions, one value per draw.

        A kicker with no fitted effect — a rookie, or anyone outside the fitted
        seasons — falls back to the league curve rather than borrowing a
        neighbour's effect. Under document 05 §1's rule that is the `w = 0`
        endpoint: no evidence about this entity, so no entity term.

        `weather` is optional so a Phase 2 posterior, which has no weather
        parameters, stays callable and reproduces its original ledger exactly.
        """
        effect = self.kicker_effects.get(kicker_season) if kicker_season else None
        logit = self._logit(distance) + self._weather_logit(weather)
        if effect is not None:
            logit = logit + effect
        return _sigmoid(logit)

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------

    @classmethod
    def from_posterior(
        cls,
        trace_path: str | Path,
        *,
        distance_centre: float = DEFAULT_DISTANCE_CENTRE,
        wind_centre: float = 0.0,
        temp_centre: float = 0.0,
    ) -> FieldGoalModel:
        """Load a fitted posterior written by `research/07_` or `research/14_`.

        Weather parameters are read when present and skipped when absent, so the
        Phase 2 trace and the Phase 3 trace both load through one path. The
        centring constants are passed in rather than stored in the trace because
        they are properties of the *sample* the model was fitted on, and a caller
        scoring new seasons must use the same ones the fit did.
        """
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

        return cls(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            kicker_effects=_indexed_effects(posterior, "kicker", "kicker_season"),
            distance_centre=distance_centre,
            roof_effects=_indexed_effects(posterior, "roof", "roof_level"),
            beta_wind=(posterior["beta_wind"].values.ravel() if "beta_wind" in posterior else None),
            beta_temp=(posterior["beta_temp"].values.ravel() if "beta_temp" in posterior else None),
            wind_centre=wind_centre,
            temp_centre=temp_centre,
        )


def _indexed_effects(posterior, variable: str, dimension: str) -> dict[str, np.ndarray]:
    """Per-level posterior draws for a grouped effect, keyed by level name."""
    if variable not in posterior:
        return {}
    values = posterior[variable]
    levels = [str(level) for level in values.coords[dimension].values]
    draws = values.values.reshape(-1, len(levels))
    return {level: draws[:, i] for i, level in enumerate(levels)}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Clipped so an absurd distance returns a probability rather than a 0, a 1,
    # or an overflow warning. The simulator must never book infinite luck.
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))
