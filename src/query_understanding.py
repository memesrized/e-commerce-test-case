"""LLM-powered query understanding for smart product search.

Provides two key capabilities:

- **Facet selection** — determines which catalog facets to apply based on the query.
- **Keyword extraction** — generates BM25-optimised keywords with multilingual support.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Market → language mapping
# ---------------------------------------------------------------------------

MARKET_LANGUAGES: dict[str, str] = {
    "UK": "en",
    "US": "en",
    "AU": "en",
    "DE": "de",
    "FR": "fr",
    "JP": "ja",
}

ENGLISH_MARKETS: set[str] = {"UK", "US", "AU"}


# ---------------------------------------------------------------------------
# Structured output models
# ---------------------------------------------------------------------------


class FacetSelection(BaseModel):
    """LLM-selected facets for filtering search results."""

    category: Optional[str] = Field(
        None, description="Product category to filter by"
    )
    subcategory: Optional[str] = Field(
        None, description="Product subcategory to filter by"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags to filter by (OR logic — at least one must match)",
    )
    min_price: Optional[float] = Field(None, description="Minimum price in GBP")
    max_price: Optional[float] = Field(None, description="Maximum price in GBP")
    min_rating: Optional[float] = Field(
        None, description="Minimum product rating (0–5)"
    )


class KeywordResult(BaseModel):
    """LLM-extracted keywords for BM25 search."""

    primary_keywords: str = Field(
        description="Space-separated search keywords in English"
    )
    localized_keywords: str = Field(
        "",
        description=(
            "Space-separated search keywords in the local market language "
            "(empty string if market is English-speaking)"
        ),
    )
    detected_market: Optional[str] = Field(
        None,
        description="Target market inferred from query context, if any",
    )

    @field_validator("detected_market", mode="before")
    @classmethod
    def _sanitise_market(cls, v: str | None) -> str | None:
        """Strip LLM artefacts like the literal string 'null'."""
        if v is None:
            return None
        v = v.strip()
        if v.lower() in ("null", "none", ""):
            return None
        valid = {"UK", "US", "DE", "FR", "AU", "JP"}
        return v.upper() if v.upper() in valid else None


# ---------------------------------------------------------------------------
# Catalog facets extractor
# ---------------------------------------------------------------------------


class CatalogFacets:
    """Available facets extracted from the product catalog at startup."""

    def __init__(self, products: list[dict]) -> None:
        self.categories: list[str] = sorted(
            {p["category"] for p in products}
        )

        _sub_map: dict[str, set[str]] = defaultdict(set)
        for p in products:
            if p.get("subcategory"):
                _sub_map[p["category"]].add(p["subcategory"])
        self.subcategories_by_category: dict[str, list[str]] = {
            cat: sorted(subs) for cat, subs in _sub_map.items()
        }

        self.tags: list[str] = sorted(
            {t for p in products for t in p.get("tags", [])}
        )
        self.markets: list[str] = sorted(
            {m for p in products for m in p.get("markets", [])}
        )

        prices = [p["price_gbp"] for p in products]
        self.price_min: float = min(prices)
        self.price_max: float = max(prices)

        ratings = [p["rating"] for p in products if p.get("rating")]
        self.rating_min: float = min(ratings) if ratings else 0.0
        self.rating_max: float = max(ratings) if ratings else 5.0

    def to_prompt(self) -> str:
        """Format all facets as a text block for LLM prompts."""
        lines: list[str] = ["AVAILABLE CATALOG FACETS:"]

        lines.append("\nCATEGORIES AND SUBCATEGORIES:")
        for cat in self.categories:
            subs = self.subcategories_by_category.get(cat, [])
            if subs:
                lines.append(f"  - {cat}: {', '.join(subs)}")
            else:
                lines.append(f"  - {cat}")

        lines.append(f"\nTAGS: {', '.join(self.tags)}")
        lines.append(f"\nMARKETS: {', '.join(self.markets)}")
        lines.append(
            f"\nPRICE RANGE: £{self.price_min:.2f} – £{self.price_max:.2f}"
        )
        lines.append(
            f"RATING RANGE: {self.rating_min:.1f} – {self.rating_max:.1f}"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Query Understanding engine
# ---------------------------------------------------------------------------


class QueryUnderstanding:
    """LLM-powered query analysis for intelligent product search.

    Analyzes natural-language queries to:

    1. Select appropriate facet filters from the catalog.
    2. Extract optimised BM25 keywords, with multilingual support
       when the target market speaks a non-English language.
    """

    def __init__(
        self,
        catalog_facets: CatalogFacets,
        context: str = "",
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ) -> None:
        """Initialise with catalog facets and optional context.

        Args:
            catalog_facets: Pre-extracted facets from the product catalog.
            context: Additional catalog context (e.g. from ContextProvider).
            model: OpenAI model name.
            temperature: LLM sampling temperature.
        """
        self._facets = catalog_facets
        self._context = context
        self._llm = ChatOpenAI(model=model, temperature=temperature)

    def select_facets(self, query: str) -> FacetSelection:
        """Determine which catalog facets to apply for the given query.

        Args:
            query: Natural-language search query.

        Returns:
            A ``FacetSelection`` with only the relevant filters populated.
        """
        prompt = (
            "Based on the user's search query, determine which product catalog "
            "facets should be used to filter results.\n\n"
            "RULES:\n"
            "- Only set facets that are CLEARLY and UNAMBIGUOUSLY relevant.\n"
            "- Leave others as null/empty — when in doubt, DO NOT filter.\n"
            "- For 'category': only set when the user explicitly names or "
            "strongly implies one specific category. If the query is vague or "
            "could span multiple categories, leave it null.\n"
            "- For 'tags': ONLY use tags from the EXACT list below. Do NOT "
            "invent tags. If no tag matches exactly, leave tags empty.\n"
            "- Prefer fewer filters over more — let the search engine rank "
            "by relevance rather than over-constraining results.\n\n"
            f"{self._facets.to_prompt()}\n\n"
            f'User query: "{query}"'
        )
        structured_llm = self._llm.with_structured_output(FacetSelection)
        return structured_llm.invoke(prompt)

    def extract_keywords(
        self, query: str, market: str | None = None
    ) -> KeywordResult:
        """Extract BM25-optimised keywords, optionally in multiple languages.

        When the target market is non-English (DE, FR, JP), the model
        generates keywords in both English **and** the local language.

        Args:
            query: Natural-language search query.
            market: Optional ISO market code (UK, US, DE, FR, AU, JP).

        Returns:
            A ``KeywordResult`` with primary and optional localised keywords.
        """
        market_instruction = ""
        if market and market not in ENGLISH_MARKETS:
            lang = MARKET_LANGUAGES.get(market, "en")
            market_instruction = (
                f"\nThe target market is {market} (language: {lang}).\n"
                f"Generate BOTH English keywords AND keywords in {lang}.\n"
                "The localised keywords should be natural product-search terms "
                f"in {lang}, not mechanical translations."
            )
        elif market:
            market_instruction = (
                f"\nThe target market is {market} (English-speaking). "
                "Only generate English keywords."
            )

        prompt = (
            "You are a search-keyword optimiser for an e-commerce product catalog.\n\n"
            f"{self._context}\n\n"
            "Given the user's query, extract the best BM25 search keywords.\n"
            "Focus on:\n"
            "- Product attributes, features, and benefits\n"
            "- Specific product types and categories\n"
            "- Material, functionality, and use-case terms\n"
            "- Do NOT include filler words or conversational tone\n"
            f"{market_instruction}\n\n"
            f'User query: "{query}"\n\n'
            "Generate concise, attribute-focused search keywords optimised "
            "for keyword matching."
        )
        structured_llm = self._llm.with_structured_output(KeywordResult)
        return structured_llm.invoke(prompt)
