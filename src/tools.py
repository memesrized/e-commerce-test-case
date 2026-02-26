"""LangChain tools that wrap the ProductIndex search engine.

Each tool's docstring doubles as the schema description the LLM reads,
so keep them precise and field-complete.
"""
from __future__ import annotations

import json
from typing import List, Optional

from langchain_core.tools import tool

from .search import ProductIndex


def create_search_tools(index: ProductIndex) -> list:
    """Create and return all search tools bound to the given product index.

    Args:
        index: A built ``ProductIndex`` instance.

    Returns:
        List of LangChain tool objects ready to pass to an agent.
    """

    @tool
    def search_products(
        query: str,
        category: Optional[str] = None,
        subcategory: Optional[str] = None,
        market: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_rating: Optional[float] = None,
        top_k: int = 5,
    ) -> str:
        """Search for products using hybrid semantic + keyword search with optional facet filters.

        Args:
            query: Feature-based search query rephrased from the user's intent.
                   Focus on product attributes and benefits — NOT raw user words.
                   Good: "rechargeable quiet clitoral vibrator waterproof"
                   Good: "romantic massage oil gift set for couples"
            category: Filter by category. One of:
                      vibrators, couples, wellness, anal, bondage, dildos, lingerie, male, essentials
            subcategory: Filter by subcategory (e.g. "clitoral", "rabbit", "plugs", "prostate", "wand").
            market: Filter by market region. One of: UK, US, DE, FR, AU, JP
            tags: Keep only products that match at least one tag.
                  Examples: ["beginner-friendly", "waterproof"], ["gift", "romantic"]
            min_price: Minimum price in GBP.
            max_price: Maximum price in GBP.
            min_rating: Minimum product rating out of 5 (e.g. 4.5).
            top_k: Number of results to return (default 5, max 10 recommended).

        Returns:
            JSON array of matching products with id, name, description,
            category, tags, price_gbp, rating, review_count, and markets.
        """
        results = index.search(
            query=query,
            category=category,
            subcategory=subcategory,
            market=market,
            tags=tags,
            min_price=min_price,
            max_price=max_price,
            min_rating=min_rating,
            top_k=top_k,
        )

        formatted = [
            {
                "id": p["id"],
                "name": p["name"],
                "description": p["description"],
                "category": p["category"],
                "subcategory": p.get("subcategory"),
                "tags": p.get("tags", []),
                "price_gbp": p["price_gbp"],
                "rating": p.get("rating"),
                "review_count": p.get("review_count"),
                "markets": p.get("markets", []),
            }
            for p in results
        ]

        if not formatted:
            return json.dumps({"message": "No products found matching the query and filters."})

        return json.dumps(formatted, ensure_ascii=False, indent=2)

    return [search_products]
