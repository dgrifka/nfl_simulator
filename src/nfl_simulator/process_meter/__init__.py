"""The process meter: one deserve-to-win number per game.

Given the plays each team ran in one game, how many points does play like that
usually score, and how often does the home side's play win? Two inputs per team —
its success rate and its yards per play, with a takeaway folded into each — go
through weights fit on 2016-2023, and 40,000 draws per game turn the two teams'
uncertainty into a single home share.

    >>> from nfl_simulator.process_meter import load_weights, score_games
    >>> weights = load_weights()                       # the committed artifact
    >>> table = score_games(2025, ["2025_01_CIN_CLE"], weights)   # doctest: +SKIP
    >>> round(float(table.p_home.iloc[0]), 4)          # doctest: +SKIP
    0.7488

Nothing here draws. The subpackage imports no plotting library, so scoring a
season never pulls a rendering stack in behind it.

=====================  ====================================================
:mod:`.plays`          the loaders, the one filter, the takeaway credit
:mod:`.features`       the team-game frame and the fold of record
:mod:`.posterior`      the weights, a team's rate, its yards, the scale
:mod:`.record`         the per-game seed, the draws, the percent of record
:mod:`.weights`        the committed artifact and the pin that guards it
:mod:`.score`          cache plus game ids in, one row per game out
:mod:`.fit`            rebuild the artifact, or stop
:mod:`.readiness`      which finished games have enough play-by-play
=====================  ====================================================
"""

from nfl_simulator.process_meter.features import (
    FOLDN_ARM,
    OWN_TERMS,
    team_game_features_foldn,
)
from nfl_simulator.process_meter.plays import (
    TRAIN_SEASONS,
    load_plays_fold,
    load_scores,
)
from nfl_simulator.process_meter.posterior import (
    CoefficientPosterior,
    coefficient_posterior,
    production_posterior_foldn,
)
from nfl_simulator.process_meter.readiness import MIN_SCRIMMAGE_PLAYS, ready_finals
from nfl_simulator.process_meter.record import (
    N_DRAWS_RECORD,
    SEED_OF_RECORD,
    SIGMA_ONCE_RECORD,
    game_seed,
    margin_draws_per_game,
    percent_record,
    points_draws_per_game,
)
from nfl_simulator.process_meter.score import (
    SCORE_COLUMNS,
    build_frames,
    margin_draws,
    score_games,
)
from nfl_simulator.process_meter.weights import WEIGHTS_PIN, check_weights, load_weights

__all__ = [
    "FOLDN_ARM",
    "MIN_SCRIMMAGE_PLAYS",
    "N_DRAWS_RECORD",
    "OWN_TERMS",
    "SCORE_COLUMNS",
    "SEED_OF_RECORD",
    "SIGMA_ONCE_RECORD",
    "TRAIN_SEASONS",
    "WEIGHTS_PIN",
    "CoefficientPosterior",
    "build_frames",
    "check_weights",
    "coefficient_posterior",
    "game_seed",
    "load_plays_fold",
    "load_scores",
    "load_weights",
    "margin_draws",
    "margin_draws_per_game",
    "percent_record",
    "points_draws_per_game",
    "production_posterior_foldn",
    "ready_finals",
    "score_games",
    "team_game_features_foldn",
]
