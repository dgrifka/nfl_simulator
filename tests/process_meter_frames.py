"""Synthetic plays, scores and frames the process-meter tests are built from.

One game, two teams, hand-countable numbers: HOM runs ten plays and AWY nine,
and a takeaway is added by name. Every assertion in the suite can be checked
against these by hand, which is the point of building them rather than reading
the cache.
"""

import numpy as np
import pandas as pd

G1 = "2016_01_AWY_HOM"

BASE = {
    "season": 2016,
    "week": 1,
    "play_type": "pass",
    "epa": 0.1,
    "success": 1,
    "down": 1,
    "yards_gained": 5,
    "interception": 0,
    "fumble_lost": 0,
    "wp": 0.5,
    "third_down_converted": 0,
    "third_down_failed": 0,
    "return_yards": 0.0,
    "return_team": None,
    "fumble_recovery_1_team": None,
    "fumble_recovery_1_yards": np.nan,
    "fumble_recovery_2_team": None,
    "fumble_recovery_2_yards": np.nan,
    "touchdown": 0.0,
    "td_team": None,
}


def _play(posteam, **changes):
    defteam = "AWY" if posteam == "HOM" else "HOM"
    return (
        BASE
        | {
            "game_id": G1,
            "home_team": "HOM",
            "away_team": "AWY",
            "posteam": posteam,
            "defteam": defteam,
        }
        | changes
    )


def _scores():
    return pd.DataFrame(
        [
            dict(
                game_id=G1,
                season=2016,
                week=1,
                home="HOM",
                away="AWY",
                home_score=17,
                away_score=21,
            )
        ]
    )


def _pick(epa=-4.0, return_yards=30):
    """AWY throws a pick HOM returns `return_yards`; `epa` is the offense's (AWY's)."""
    return _play(
        "AWY",
        interception=1,
        success=int(epa > 0),
        yards_gained=0,
        epa=epa,
        return_yards=float(return_yards),
        return_team="HOM",
    )


def _fumble(epa=-2.5, yards=0.0):
    """AWY runs and loses a fumble HOM recovers and returns `yards`."""
    return _play(
        "AWY",
        play_type="run",
        fumble_lost=1,
        success=int(epa > 0),
        yards_gained=3,
        epa=epa,
        fumble_recovery_1_team="HOM",
        fumble_recovery_1_yards=float(yards),
    )


def _game(*takeaways, home_successes=6, ladder=False):
    """HOM runs 10 plays, `home_successes` successful; AWY 9 for 36 plus the takeaways.

    `ladder` gives HOM's plays a spread of yards, so its yards SD is not 0.
    """
    plays = [
        _play(
            "HOM",
            yards_gained=(5 + i) if ladder else 5,
            success=int(i < home_successes),
            epa=0.1 if i < home_successes else -0.1,
        )
        for i in range(10)
    ]
    plays += list(takeaways)
    plays += [_play("AWY", yards_gained=4) for _ in range(9)]
    return pd.DataFrame(plays)


def _row(frame, team):
    return frame[frame.posteam == team].iloc[0]
