"""NFL deserve-to-win meter.

One number per game, from the plays each team ran:

    >>> from nfl_simulator.process_meter import load_weights, score_games

:mod:`nfl_simulator.process_meter` is the model. Nothing in it draws, so
scoring a season never pulls a plotting stack in behind it; :mod:`.style` and
:mod:`.teams` are the figure layer and are imported only by something that
renders.
"""

from importlib.metadata import version

#: Read from the installed distribution's metadata rather than repeated here,
#: so the package version has exactly one source: `pyproject.toml`.
__version__ = version("nfl-simulator")

__all__ = ["__version__"]
