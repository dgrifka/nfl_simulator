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

The adopted form is Phase 3's cubic curve — the pre-registered linear term
failed Gate FG-2 and the quadratic failed Gate W-4 — plus the weather terms and
the extra-point arm fitted with it:

    logit p = alpha + beta * d + gamma * d^2 / 100 + delta_cubic * d^3 / 1000
              + roof_effect
              + beta_wind * (wind - wind_centre) * has_weather
              + beta_temp * (temp - temp_centre) * has_weather
              + delta_xp * is_extra_point
              + kicker_effect * (1 + (lambda_xp - 1) * is_extra_point)
              + beta_elev * (elev_kft - elevation_centre)

    where d = distance - distance_centre, and elev_kft is the kick's stadium
    elevation in thousands of feet, looked up from `stadium_id`

**Every fitted term is optional and absent means absent**, never zero-by-
assumption: a posterior fitted before Phase 3 carries no weather parameters and
no cubic term, and a quadratic-arm trace carries no `delta_cubic`. Passing
weather — or asking for an extra point — of such a model is a no-op rather than
an error, because the v1.1 and v1.2 ledgers have to stay reproducible.

The cubic and extra-point terms were fitted in Phase 3 and **were not read here
until v1.3** (document 27 §14f). Until then the simulator priced every kick on a
quadratic curve whose `gamma` had been fitted jointly with a cubic term it then
discarded, and priced extra points as plain field goals from 33 yards. Document
30 §5a makes agreement with the fit a gate rather than a hope: the read side
must reproduce `research/14_fg_weather_model.make_probabilities` on every kick.

**Elevation arrived in v1.4** (documents 66, 67 and 68). It is the first term
resolved from something that is not on the play row: the covariate is a lookup
from `stadium_id` through `nfl_simulator.data.stadium_elevation`, so a caller
prices a kick at altitude by naming the stadium rather than by computing
anything. The lookup **raises** on a stadium nobody has entered — sea level is
a real elevation in that table, so a silent default would be indistinguishable
from a correct row — while *no* `stadium_id` at all stays the same `w = 0`
endpoint an unknown kicker gets, and prices the kick at the fitted centre.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nfl_simulator.data.stadium_elevation import elevation_kft

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
    delta_cubic: np.ndarray | None = None
    delta_xp: np.ndarray | None = None
    lambda_xp: np.ndarray | None = None
    beta_elev: np.ndarray | None = None
    elevation_centre: float = 0.0

    def __post_init__(self) -> None:
        lengths = {len(self.alpha), len(self.beta), len(self.gamma)}
        for optional in (self.delta_cubic, self.delta_xp, self.lambda_xp, self.beta_elev):
            if optional is not None:
                lengths.add(len(optional))
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

    def _logit(self, distance: float, *, extra_point: bool = False) -> np.ndarray:
        centred = distance - self.distance_centre
        # Divided by 100 and 1000 to match the fitted parameterization, which
        # scales each polynomial term so its coefficient sits on the same order
        # as the slope. Getting either divisor wrong is silent, so they are
        # pinned against the fit's own arithmetic by a test.
        logit = self.alpha + self.beta * centred + self.gamma * centred**2 / 100.0
        if self.delta_cubic is not None:
            logit = logit + self.delta_cubic * centred**3 / 1000.0
        if extra_point and self.delta_xp is not None:
            logit = logit + self.delta_xp
        return logit

    def league_make_probability(
        self,
        distance: float,
        *,
        extra_point: bool = False,
        stadium_id: str | None = None,
    ) -> np.ndarray:
        """Make probability for an average kicker, one value per posterior draw."""
        return _sigmoid(
            self._logit(distance, extra_point=extra_point) + self._elevation_logit(stadium_id)
        )

    def _kicker_logit(self, effect: np.ndarray, *, extra_point: bool) -> np.ndarray:
        """A kicker's effect, transferred to extra points at the fitted rate.

        `lambda_xp` is how much of a kicker's field-goal ability shows up on an
        extra point; the fit centres it on full transfer and lets the data argue.
        A posterior without it — anything before Phase 3 — transfers in full,
        which is what the code did before the parameter existed.
        """
        if not extra_point or self.lambda_xp is None:
            return effect
        return effect * self.lambda_xp

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

    def _elevation_logit(self, stadium_id: str | None) -> np.ndarray | float:
        """Log-odds adjustment for the air this stadium sits in. Zero when silent.

        Two different silences, and they mean different things:

        * **This posterior has no `beta_elev`** — anything before v1.4. The term
          is not in the model, so there is nothing to add, and a caller handing
          a `stadium_id` to a v1.3 posterior gets the v1.3 number back
          unchanged. That is what keeps every shipped ledger reproducible.
        * **No `stadium_id` was passed** — a Phase 2 replay frame, which has no
          such column. The kick is priced at the fitted centre, which is
          document 05 §1's `w = 0` endpoint applied to the air: no evidence
          about where this kick was taken, so no elevation term.

        An *unknown* stadium is neither of those and raises, from
        `stadium_elevation.elevation_kft`. Sea level is a real value in that
        table, so falling back to it would price a Mexico City kick as a New
        Jersey one and say nothing about it.
        """
        if self.beta_elev is None or stadium_id is None:
            return 0.0
        return self.beta_elev * (elevation_kft(stadium_id) - self.elevation_centre)

    def make_probability(
        self,
        kicker_season: str | None,
        distance: float,
        *,
        weather: Weather | None = None,
        extra_point: bool = False,
        stadium_id: str | None = None,
    ) -> np.ndarray:
        """Make probability for this kicker in these conditions, one value per draw.

        A kicker with no fitted effect — a rookie, or anyone outside the fitted
        seasons — falls back to the league curve rather than borrowing a
        neighbour's effect. Under document 05 §1's rule that is the `w = 0`
        endpoint: no evidence about this entity, so no entity term.

        `weather` is optional so a Phase 2 posterior, which has no weather
        parameters, stays callable and reproduces its original ledger exactly.

        `extra_point` selects the fitted extra-point arm — the `delta_xp` offset
        and the `lambda_xp` transfer of kicker ability. An extra point carries
        its own `kick_distance`, so it goes through the same distance curve
        rather than being pinned at a constant, and `delta_xp` then means what
        the fit says it means: the difference between an extra point and a field
        goal *from the same distance*.

        `stadium_id` selects the elevation term fitted in v1.4 (document 68).
        The stadium is named rather than the elevation passed, so the read side
        and the fit consult one table and cannot disagree about how high Denver
        is. Omitting it prices the kick at the fitted centre; naming a stadium
        the table does not hold raises.
        """
        effect = self.kicker_effects.get(kicker_season) if kicker_season else None
        logit = (
            self._logit(distance, extra_point=extra_point)
            + self._weather_logit(weather)
            + self._elevation_logit(stadium_id)
        )
        if effect is not None:
            logit = logit + self._kicker_logit(effect, extra_point=extra_point)
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
        elevation_centre: float | None = None,
    ) -> FieldGoalModel:
        """Load a fitted posterior written by `research/07_`, `research/14_` or
        `research/42_`.

        Every optional parameter is read when present and skipped when absent, so
        the Phase 2 trace, the quadratic arm, the adopted cubic arm, the v1.3
        refit and the v1.4 elevation posterior all load through one path. **Skipped means the term is not in the
        model**, never that it is zero by assumption — the distinction is what
        keeps an older ledger reproducible. The
        centring constants are passed in rather than stored in the trace because
        they are properties of the *sample* the model was fitted on, and a caller
        scoring new seasons must use the same ones the fit did.

        `elevation_centre` is the one centring constant that is **required when
        its term is present**, rather than defaulting to zero like the other
        two. The reason is that its default would not be silent-but-harmless:
        wind and temperature only reach a kick that has a reading, so a missing
        centre affects a subset, while every kick has an elevation. Loading a
        v1.4 posterior at a centre of zero would price the whole league as if
        it sat 569 feet lower than it does — one uniform, invisible shift of
        exactly the kind document 30 was written to stop.
        """
        import arviz as az

        trace_path = Path(trace_path)
        if not trace_path.exists():
            raise FileNotFoundError(
                f"no fitted field-goal posterior at {trace_path} — "
                "run `uv run python research/82_fg_v14_refit.py`"
            )

        posterior = az.from_netcdf(trace_path)["posterior"]
        alpha = posterior["alpha"].values.ravel()
        beta = posterior["beta"].values.ravel()
        # The linear arm has no gamma. Zeros keep the same code path rather than
        # branching on which arm was adopted.
        gamma = posterior["gamma"].values.ravel() if "gamma" in posterior else np.zeros_like(alpha)

        def optional(name: str) -> np.ndarray | None:
            return posterior[name].values.ravel() if name in posterior else None

        beta_elev = optional("beta_elev")
        if beta_elev is not None and elevation_centre is None:
            raise ValueError(
                f"{trace_path.name} carries an elevation term, so it needs the "
                "elevation_centre it was fitted at — document 66 §11 puts it at "
                "0.5687 kft, and `fg_v14_summary.json` carries it under "
                "centres['elevation']"
            )

        return cls(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            kicker_effects=_indexed_effects(posterior, "kicker", "kicker_season"),
            distance_centre=distance_centre,
            roof_effects=_indexed_effects(posterior, "roof", "roof_level"),
            beta_wind=optional("beta_wind"),
            beta_temp=optional("beta_temp"),
            wind_centre=wind_centre,
            temp_centre=temp_centre,
            delta_cubic=optional("delta_cubic"),
            delta_xp=optional("delta_xp"),
            lambda_xp=optional("lambda_xp"),
            beta_elev=beta_elev,
            elevation_centre=0.0 if elevation_centre is None else elevation_centre,
        )


def load_fitted_model(trace: str, summary: str) -> tuple[FieldGoalModel, dict]:
    """A fitted posterior plus the centring constants it was fitted at.

    Both files are read from `paths.artifact_dir()` — the repo's
    `research/outputs/` in a checkout, and whatever `NFL_SIM_ARTIFACT_DIR` names
    in an installed one. Round E moved this out of
    `research/44_read_side_fix.py`, which imports it back under its old name:
    every production read of a posterior goes through this one function, so
    document 30's correction has exactly one home.

    `centres["elevation"]` is v1.4's and is passed only when the summary has
    it, so a v1.1/v1.2/v1.3 summary loads through this same call unchanged. The
    read side refuses a trace that carries `beta_elev` without it, so the pair
    cannot come apart silently: a v1.4 trace loaded against a v1.3 summary
    raises here rather than pricing the league 569 feet too low.
    """
    from nfl_simulator import paths

    directory = paths.artifact_dir()
    with (directory / summary).open() as handle:
        centres = json.load(handle)["centres"]
    return FieldGoalModel.from_posterior(
        directory / trace,
        wind_centre=centres["wind"],
        temp_centre=centres["temp"],
        elevation_centre=centres.get("elevation"),
    ), centres


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
