"""NFL deserve-to-win simulator.

The product has two front doors:

* :func:`nfl_simulator.render.render_game` — a game the research record covers,
  drawn from the committed artifacts.
* :func:`nfl_simulator.live.adjudicate_live_game` — a game that has just gone
  final, adjudicated from its own play-by-play. It is re-exported here because
  it is the entry point a caller outside this repo uses.

The second is exposed lazily: importing it pulls in matplotlib and the whole
figure stack, and `from nfl_simulator import paths` should not pay for that.
"""

__version__ = "0.1.0"

__all__ = ["adjudicate_live_game"]


def __getattr__(name: str):
    if name == "adjudicate_live_game":
        from nfl_simulator.live import adjudicate_live_game

        return adjudicate_live_game
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
