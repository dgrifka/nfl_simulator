"""NFL deserve-to-win meter.

One number per game, from the plays each team ran:

    >>> from nfl_simulator.process_meter import load_weights, score_games

:mod:`nfl_simulator.process_meter` is the model. Nothing in it draws, so
scoring a season never pulls a plotting stack in behind it; :mod:`.style` and
:mod:`.teams` are the figure layer and are imported only by something that
renders.
"""

__version__ = "2.1.0"

__all__ = ["__version__"]
