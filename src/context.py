"""Catalog context and knowledge base for search guidance.

Designed around the ``ContextProvider`` interface so the static knowledge
can later be replaced with a RAG-based implementation without changing
the agent or prompt construction code.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ContextProvider(Protocol):
    """Interface for providing search context to the LLM agent.

    Implement this protocol to swap in a RAG-based provider,
    a database-backed provider, or any other dynamic source.
    """

    def get_context(self) -> str:
        """Return a context string for injection into the agent system prompt."""
        ...


# ---------------------------------------------------------------------------
# Static knowledge base
# ---------------------------------------------------------------------------

CATALOG_CONTEXT = """\

=== PRODUCT CATALOG KNOWLEDGE BASE ===

CATEGORIES AND SUBCATEGORIES:
- vibrators: clitoral, rabbit, wand, bullet, g-spot, mini, wearable, kits
- couples: rings, kits, wearable, games
- wellness: massage (oils, candles)
- anal: prostate, plugs
- bondage: sets, rope, accessories
- dildos: glass, realistic
- lingerie: robes, sets, bodysuits
- male: strokers, rings
- essentials: lubricants, condoms, cleaning

AVAILABLE FILTER TAGS:
  beginner-friendly, solo, couples, intermediate, advanced,
  rechargeable, waterproof, app-connected, remote-control,
  discreet, travel, quiet, luxury, gift, gift-set, romantic,
  valentines, sensory, bondage, soft-bondage, body-safe,
  temperature-play, harness-compatible, essential, hygiene,
  safe-sex, variety, non-intimidating, mains-powered, graduated,
  dual-stimulation, powerful, bold, fun, cotton

MARKETS (region filter): UK, US, DE (Germany), FR (France), AU (Australia), JP (Japan)

PRICE RANGE: £6.99 – £79.99 GBP

SEARCH QUERY GUIDANCE:
- Gift queries           → tags: ["gift", "gift-set", "romantic"], category: "wellness" or "couples"
- Beginner customers     → tag: "beginner-friendly"
- Couples play           → category: "couples" or tag: "couples"
- Discreet / travel use  → tags: ["discreet", "travel"]
- Premium / luxury items → tag: "luxury" or include "luxury premium" in query
- Wellness / massage     → category: "wellness"
- Safety-focused         → tag: "body-safe"
- App / remote control   → tag: "app-connected" or "remote-control"
- Waterproof items       → tag: "waterproof"
- Bondage / BDSM         → category: "bondage"
- Men's products         → category: "male"
- Anal play              → category: "anal"
"""


class StaticContextProvider:
    """Provides static, hardcoded catalog knowledge to the LLM agent."""

    def get_context(self) -> str:
        """Return the static catalog context string."""
        return CATALOG_CONTEXT


def get_catalog_context() -> str:
    """Return the default static catalog context. Convenience wrapper."""
    return CATALOG_CONTEXT
