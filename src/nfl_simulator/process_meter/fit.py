"""Rebuild the weights artifact from the play-by-play cache, or stop.

This is the only place the coefficient posterior of the model of record is
computed. Run it by hand when the model moves, never on a schedule::

    python -m nfl_simulator.process_meter.fit --data-dir data --out weights.json

The fit is :mod:`.score`'s, function for function — the same loaders, the same
frame, the same posterior — so a refit cannot quietly become a second model.

**Four checks before anything is written, all fatal.** The weights must round to
the three of record; the plain least-squares fit must agree with the posterior
mean, because the two build the same regression on the same rows and a
disagreement means one of the designs moved; ``margin_sd`` must round to the
pinned residual scale; and ``sigma_once`` must round to the pinned scale at four
decimals. That last one is **measured here, never imported**: ``sqrt(margin_sd^2
- the median per-game margin-draw variance)`` over the gate window, at the draw
count and seed of record. A refit that moves the scale therefore stops here
instead of shipping the old constant on new weights.

**Which ``margin_sd`` goes into that square root is a convention, not a detail.**
The published three-decimal residual SD is the one of record; the full one gives
a scale about 6e-4 larger. Both are printed on every run, and
``docs/research/75-process-meter-foundations.md`` states the convention.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from nfl_simulator.process_meter.features import OWN_TERMS, team_game_features_foldn
from nfl_simulator.process_meter.plays import TRAIN_SEASONS, load_plays_fold, load_scores
from nfl_simulator.process_meter.posterior import (
    NOISE_ONCE_FLOOR,
    coefficient_posterior,
    fit_arm,
    noise_once_sd,
    production_posterior_foldn,
)
from nfl_simulator.process_meter.record import (
    N_DRAWS_RECORD,
    SEED_OF_RECORD,
    margin_draws_per_game,
)
from nfl_simulator.process_meter.weights import WEIGHTS_PIN, check_weights

#: Every season the frame is built over. The fit itself reads `TRAIN_SEASONS`.
ALL_SEASONS = tuple(range(2016, 2026))

#: The public document the artifact's `source` field names.
SOURCE = "docs/research/75-process-meter-foundations.md"

#: The window `sigma_once` is measured over: decided regular-season games with
#: both team rows present, outside the training seasons.
GATE_SEASONS = (2024, 2025)
EXPECTED_GATE_GAMES = 543

#: 2,095 training games, two team rows each.
EXPECTED_TRAIN_TEAM_GAMES = 4_190
COEFFICIENT_LINE = 5e-4  # points; the plain fit against the posterior mean
SIGMA_ONCE_DECIMALS = 4


class RefitStopped(RuntimeError):
    """A pinned number moved, so the artifact was not written."""


def gate_games(scores, frame):
    """The gate window: decided games with both team rows in the frame."""
    gate = scores[scores.season.isin(GATE_SEASONS)].copy()
    gate["result"] = gate.home_score - gate.away_score
    both_rows = frame.groupby("game_id").size()
    gate = gate[gate.game_id.isin(both_rows[both_rows == 2].index) & (gate.result != 0)]
    return gate.sort_values(["season", "week", "game_id"], ignore_index=True)


def measure_sigma_once(frame, scores, coef, prod_post):
    """The per-game margin-draw variances over the gate window, and how many games."""
    gate = gate_games(scores, frame)
    if len(gate) != EXPECTED_GATE_GAMES:
        raise RefitStopped(f"{len(gate)} gate games, expected {EXPECTED_GATE_GAMES}")
    draws = margin_draws_per_game(gate, coef, prod_post, N_DRAWS_RECORD, SEED_OF_RECORD)
    return draws.var(axis=1), len(gate)


def build(seasons=ALL_SEASONS, data_dir=None, verbose=True) -> dict:
    """The payload, checked against the pin, against the plain fit and against its own draws."""
    plays = load_plays_fold(seasons, data_dir)
    scores = load_scores(seasons, data_dir)
    frame = team_game_features_foldn(plays, scores, "all")

    coef = coefficient_posterior(frame, OWN_TERMS, TRAIN_SEASONS)
    arm = fit_arm(frame, OWN_TERMS, TRAIN_SEASONS)

    gap = max(
        abs(arm.points_coef[name] - value)
        for name, value in zip(coef.columns, coef.mean, strict=True)
    )
    if gap > COEFFICIENT_LINE:
        raise RefitStopped(
            f"the plain fit and the coefficient posterior disagree by {gap:.2e} points "
            f"(line {COEFFICIENT_LINE:.0e}) — the two designs have drifted apart"
        )
    if verbose:
        print(f"the plain fit agrees with the posterior mean to {gap:.2e} points")

    if coef.n_rows != EXPECTED_TRAIN_TEAM_GAMES:
        raise RefitStopped(
            f"{coef.n_rows} training team-games, expected {EXPECTED_TRAIN_TEAM_GAMES}"
        )

    in_frame = plays.merge(frame[["game_id", "posteam"]], on=["game_id", "posteam"], how="inner")
    prod_post = production_posterior_foldn(in_frame, frame)
    margin_var, n_gate = measure_sigma_once(frame, scores, coef, prod_post)

    # The published, rounded residual SD is the one of record; see the docstring.
    margin_sd_published = round(float(arm.margin_sd), WEIGHTS_PIN["decimals"])
    sigma_once, _ = noise_once_sd(margin_var, margin_sd_published, mode="pooled")
    sigma_once_full, _ = noise_once_sd(margin_var, arm.margin_sd, mode="pooled")
    if verbose:
        print(
            f"sigma_once measured on the arm's own draws over {n_gate} gate games: "
            f"{sigma_once:.10f} from the published margin_sd {margin_sd_published} "
            f"(the full {arm.margin_sd:.10f} would give {sigma_once_full:.10f}); "
            f"median margin variance {float(np.median(margin_var)):.10f}, "
            f"floor {NOISE_ONCE_FLOOR}"
        )
    if sigma_once <= NOISE_ONCE_FLOOR:
        raise RefitStopped(
            f"the measured sigma_once {sigma_once:.4f} is at or below the floor {NOISE_ONCE_FLOOR}"
        )
    pinned = WEIGHTS_PIN["sigma_once"]
    if round(sigma_once, SIGMA_ONCE_DECIMALS) != pinned:
        raise RefitStopped(
            f"the measured sigma_once {sigma_once:.10f} is not the pinned {pinned} at "
            f"{SIGMA_ONCE_DECIMALS} decimals — re-pin record.SIGMA_ONCE_RECORD, do not "
            "ship the old constant on new weights"
        )

    payload = {
        "columns": list(coef.columns),
        "mean": [float(v) for v in coef.mean],
        "cov": [[float(v) for v in row] for row in np.asarray(coef.cov)],
        "sigma2": float(coef.sigma2),
        "n_train_team_games": int(coef.n_rows),
        "train_seasons": list(TRAIN_SEASONS),
        "margin_sd": float(arm.margin_sd),
        "sigma_once": round(float(sigma_once), SIGMA_ONCE_DECIMALS),
        "sigma_once_measured": float(sigma_once),
        "sigma_once_from_full_margin_sd": float(sigma_once_full),
        "median_margin_draw_variance": float(np.median(margin_var)),
        "n_draws": N_DRAWS_RECORD,
        "seed": SEED_OF_RECORD,
        "arm": WEIGHTS_PIN["arm"],
        "source": SOURCE,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    check_weights(payload)
    return payload


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="where to write the JSON")
    parser.add_argument("--data-dir", default=None, help="cache root; NFL_SIM_DATA_DIR by default")
    args = parser.parse_args(argv)

    payload = build(data_dir=args.data_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\nthe weights of record ({args.out}):")
    for name, value, pinned in zip(
        payload["columns"], payload["mean"], WEIGHTS_PIN["mean"], strict=True
    ):
        print(f"  {name:<22} {value:+.6f}   (pinned {pinned:+.3f})")
    print(
        f"  margin_sd              {payload['margin_sd']:.10f}   (pinned {WEIGHTS_PIN['margin_sd']})"
    )
    print(
        f"  sigma_once             {payload['sigma_once_measured']:.10f}   "
        f"(pinned {WEIGHTS_PIN['sigma_once']}, measured here)"
    )
    print(f"  arm                    {payload['arm']}")
    print(f"  n_draws                {payload['n_draws']:,}   seed {payload['seed']}")
    print(
        f"  train team-games       {payload['n_train_team_games']:,} "
        f"over {TRAIN_SEASONS[0]}-{TRAIN_SEASONS[-1]}"
    )


if __name__ == "__main__":
    main()
