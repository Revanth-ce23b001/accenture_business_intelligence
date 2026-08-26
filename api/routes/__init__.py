"""The endpoint modules, grouped by what a caller is doing.

    catalog    what exists: KPIs, the watchlist, contracts, evidence
    cases      running an investigation, reading it, acting on it
    ask        turning a question into one of the above
    learning   feedback in, and what it moved back out
    platform   what the system says about itself
"""

from api.routes import ask, cases, catalog, learning, platform

__all__ = ["ask", "cases", "catalog", "learning", "platform"]
