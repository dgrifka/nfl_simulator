"""The one fit the meter is allowed to score with, and the pin that proves it.

The library never refits at score time. It loads :data:`WEIGHTS_PATH` — a 1 KB
JSON committed beside this module — and draws from it, so every number the meter
prints comes from the fit that passed the gate rather than from whatever the
cache happened to hold that morning.

:data:`WEIGHTS_PIN` is what makes that a guarantee rather than a hope. The load
refuses an artifact whose weights or constants miss the pin, so a wrong file
fails loudly at the start of a run instead of quietly scoring a game. The three
weights are the arm of record's, fit on 2016-2023; the constants are
:mod:`.record`'s own, so moving one of them there without moving it here is a
test failure. ``margin_sd`` is pinned because it is the residual scale
``sigma_once`` was measured against.

Re-fit the artifact with :mod:`.fit`, which re-measures every pinned number and
refuses to write if one has moved.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nfl_simulator.process_meter.features import FOLDN_ARM
from nfl_simulator.process_meter.posterior import COLUMNS, CoefficientPosterior
from nfl_simulator.process_meter.record import (
    N_DRAWS_RECORD,
    SEED_OF_RECORD,
    SIGMA_ONCE_RECORD,
)

#: The committed artifact. It is package data, so an installed wheel carries it.
WEIGHTS_PATH = Path(__file__).with_name("weights.json")

#: What a usable weights artifact must say.
WEIGHTS_PIN = {
    "columns": list(COLUMNS),
    "mean": [-16.829, 35.126, 4.178],
    "decimals": 3,
    "margin_sd": 8.875,
    "sigma_once": SIGMA_ONCE_RECORD,
    "n_draws": N_DRAWS_RECORD,
    "seed": SEED_OF_RECORD,
    "arm": FOLDN_ARM,
}


def check_weights(payload: dict) -> dict:
    """Raise unless ``payload`` is the model of record; return it unchanged if it is."""
    columns = list(payload.get("columns", []))
    if columns != WEIGHTS_PIN["columns"]:
        raise ValueError(
            f"weights columns {columns} are not the model of record's {WEIGHTS_PIN['columns']}"
        )
    decimals = WEIGHTS_PIN["decimals"]
    mean = [round(float(v), decimals) for v in payload.get("mean", [])]
    if mean != WEIGHTS_PIN["mean"]:
        off = [
            f"{name} {got} (pinned {want})"
            for name, got, want in zip(columns, mean, WEIGHTS_PIN["mean"], strict=False)
            if got != want
        ]
        raise ValueError(
            f"weights are not the model of record at {decimals} decimals: {', '.join(off)}"
        )
    margin_sd = payload.get("margin_sd")
    if margin_sd is None or round(float(margin_sd), decimals) != WEIGHTS_PIN["margin_sd"]:
        raise ValueError(
            f"margin_sd is {margin_sd!r}, not the model of record's "
            f"{WEIGHTS_PIN['margin_sd']!r} at {decimals} decimals"
        )
    for name in ("sigma_once", "n_draws", "seed", "arm"):
        if payload.get(name) != WEIGHTS_PIN[name]:
            raise ValueError(
                f"{name} is {payload.get(name)!r}, not the model of record's {WEIGHTS_PIN[name]!r}"
            )
    return payload


def load_weights(path=None) -> CoefficientPosterior:
    """The coefficient posterior the meter draws from, checked against the pin.

    Reads the committed artifact unless ``path`` names another one. What comes
    back is a
    :class:`~nfl_simulator.process_meter.posterior.CoefficientPosterior`, the
    same type the fit returns, so
    :func:`~nfl_simulator.process_meter.record.margin_draws_per_game` cannot tell
    the two apart.
    """
    payload = json.loads(Path(path or WEIGHTS_PATH).read_text())
    check_weights(payload)
    return CoefficientPosterior(
        columns=list(payload["columns"]),
        mean=np.asarray(payload["mean"], dtype=float),
        cov=np.asarray(payload["cov"], dtype=float),
        sigma2=float(payload["sigma2"]),
        n_rows=int(payload["n_train_team_games"]),
    )


__all__ = ["WEIGHTS_PATH", "WEIGHTS_PIN", "check_weights", "load_weights"]
