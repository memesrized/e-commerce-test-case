"""Catalog context and knowledge base for search guidance.

Designed around the ``ContextProvider`` interface so the static knowledge
can later be replaced with a RAG-based implementation without changing
the agent or prompt construction code.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
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


# ---------------------------------------------------------------------------
# Shared search guidance (used by both static and dynamic providers)
# ---------------------------------------------------------------------------

_SEARCH_GUIDANCE = """\

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


# ---------------------------------------------------------------------------
# Dataset-derived context provider
# ---------------------------------------------------------------------------


class DatasetContextProvider:
    """Generates catalog context dynamically from actual product data.

    Extracts category distribution, price ranges, popular tags, and
    market coverage to give the LLM realistic catalog awareness.
    """

    def __init__(self, products: list[dict]) -> None:
        self._products = products

    @classmethod
    def from_file(cls, path: str | Path) -> "DatasetContextProvider":
        """Load products from a JSON file and build the context provider.

        Args:
            path: Path to the products JSON file.
        """
        products = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(products)

    def get_context(self) -> str:
        """Return a context string derived from the product catalog."""
        categories: dict[str, int] = defaultdict(int)
        subcategories: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        all_tags: dict[str, int] = defaultdict(int)
        markets: dict[str, int] = defaultdict(int)
        languages: dict[str, int] = defaultdict(int)
        prices: list[float] = []
        ratings: list[float] = []

        for p in self._products:
            categories[p["category"]] += 1
            if p.get("subcategory"):
                subcategories[p["category"]][p["subcategory"]] += 1
            for t in p.get("tags", []):
                all_tags[t] += 1
            for m in p.get("markets", []):
                markets[m] += 1
            languages[p.get("language", "en")] += 1
            prices.append(p["price_gbp"])
            if p.get("rating"):
                ratings.append(p["rating"])

        lines: list[str] = ["\n=== PRODUCT CATALOG KNOWLEDGE BASE ===\n"]

        # Overview
        lines.append("CATALOG OVERVIEW:")
        lines.append(f"  Total products: {len(self._products)}")
        lines.append(f"  Price range: £{min(prices):.2f} – £{max(prices):.2f}")
        if ratings:
            lines.append(
                f"  Rating range: {min(ratings):.1f} – {max(ratings):.1f}"
            )
        lines.append(f"  Languages: {', '.join(sorted(languages))}")
        lines.append(f"  Markets: {', '.join(sorted(markets))}")

        # Categories
        lines.append("\nCATEGORIES AND SUBCATEGORIES:")
        for cat in sorted(categories):
            subs = subcategories.get(cat, {})
            sub_str = ", ".join(sorted(subs))
            count = categories[cat]
            lines.append(f"  - {cat} ({count} products): {sub_str}")

        # Tags
        lines.append("\nAVAILABLE FILTER TAGS (by popularity):")
        sorted_tags = sorted(all_tags.items(), key=lambda x: -x[1])
        lines.append("  " + ", ".join(t for t, _ in sorted_tags))

        # Market details
        lines.append("\nMARKET COVERAGE:")
        for m in sorted(markets):
            lines.append(f"  - {m}: {markets[m]} products")

        # Search guidance
        lines.append(_SEARCH_GUIDANCE)

        return "\n".join(lines)
