"""A1 — market-implied probabilities as the BoC reference forecast.

Self-contained package for step A1 of the enhancement plan: replace the
climatology baseline with the strongest available reference (the CORRA futures
curve) and report skill against both.

Modules
-------
``fetch_market_data``
    Download CORRA futures settlements (TMX Montréal Exchange) plus CORRA and
    the target rate (Bank of Canada Valet) into ``data/``.
``implied``
    Convert settlement prices into cut/hold/hike probabilities.
``build_table``
    Price every origin in a spec and commit the resulting table.
``market_implied``
    :class:`MarketImpliedPredictor`, which replays that table.
``leaderboard``
    Skill columns against both references, from stored per-meeting scores.

Nothing here modifies existing use-case modules; the folder is additive so the
A1 diff is easy to review in isolation.
"""

from .implied import ImpliedQuote, implied_quote, move_to_probabilities
from .market_implied import MarketImpliedPredictor, load_market_table


__all__ = [
    "ImpliedQuote",
    "MarketImpliedPredictor",
    "implied_quote",
    "load_market_table",
    "move_to_probabilities",
]
