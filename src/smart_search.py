"""Smart product search with LLM-powered query understanding.

Composes ``ProductIndex`` (hybrid search engine) with ``QueryUnderstanding``
(LLM-based facet selection and keyword extraction) for an end-to-end
natural-language search pipeline.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

from .context import DatasetContextProvider
from .query_understanding import CatalogFacets, QueryUnderstanding
from .search import RRF_K, ProductIndex


class SmartProductSearch:
    """Intelligent search combining LLM query understanding with hybrid search.

    Pipeline:

    1. LLM selects appropriate facet filters from catalog metadata.
    2. LLM extracts BM25-optimised keywords (multilingual when needed).
    3. Semantic search runs on the original query.
    4. BM25 runs on LLM-extracted keywords (merged if multilingual).
    5. Results are fused with RRF and filtered by LLM-selected facets.
    """

    def __init__(
        self,
        index: ProductIndex,
        query_understanding: QueryUnderstanding,
    ) -> None:
        """Initialise with a pre-built index and query understanding engine.

        Args:
            index: A built ``ProductIndex`` instance.
            query_understanding: Configured ``QueryUnderstanding`` instance.
        """
        self._index = index
        self._qu = query_understanding

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        context: str = "",
        model: str = "gpt-4o-mini",
    ) -> "SmartProductSearch":
        """Build a SmartProductSearch from a product catalog JSON file.

        Args:
            path: Path to the products JSON file.
            context: Additional catalog context for the query understanding LLM.
            model: OpenAI model name for query understanding.
        """
        index = ProductIndex.from_file(path)
        facets = CatalogFacets(index.products)
        qu = QueryUnderstanding(facets, context=context, model=model)
        return cls(index, qu)

    def search(
        self,
        query: str,
        market: Optional[str] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Run the full smart search pipeline.

        Args:
            query: Natural-language search query.
            market: Optional target market code (UK, US, DE, FR, AU, JP).
            top_k: Maximum number of results.

        Returns:
            List of product dicts sorted by relevance, each with ``_score``.
        """
        # 1. LLM: determine facet filters
        facets = self._qu.select_facets(query)

        # Sanitise LLM-selected tags: drop any that don't exist in the catalog
        valid_tags = {t for p in self._index.products for t in p.get("tags", [])}
        facets.tags = [t for t in facets.tags if t in valid_tags]

        # 2. LLM: extract BM25 keywords (potentially multilingual)
        keywords = self._qu.extract_keywords(query, market=market)
        effective_market = market or keywords.detected_market

        # 3. Semantic search — uses original query (model handles multilingual)
        semantic_ranks = self._index._semantic_ranks(query)

        # 4. BM25 — uses LLM-extracted keywords
        bm25_primary = self._index._bm25_ranks(keywords.primary_keywords)
        if keywords.localized_keywords:
            bm25_local = self._index._bm25_ranks(keywords.localized_keywords)
            bm25_ranks = self._merge_ranked_lists(bm25_primary, bm25_local)
        else:
            bm25_ranks = bm25_primary

        # 5. RRF fusion (semantic + BM25)
        rrf_scores = self._index._compute_rrf(semantic_ranks, bm25_ranks)

        ranked_indices = sorted(
            rrf_scores, key=lambda i: rrf_scores[i], reverse=True
        )

        # 6. Apply LLM-determined facet filters with progressive relaxation.
        #    Each level is tried in order. Results accumulate (deduped)
        #    until we have top_k items:
        #      a) full filters  →  b) drop category/subcategory  →  c) no filters
        filter_tags = facets.tags if facets.tags else None

        filter_levels: list[dict] = [
            # Level 0 — all LLM-selected filters
            dict(
                category=facets.category,
                subcategory=facets.subcategory,
                market=effective_market,
                tags=filter_tags,
                min_price=facets.min_price,
                max_price=facets.max_price,
                min_rating=facets.min_rating,
            ),
            # Level 1 — drop category + subcategory (most common LLM mistake)
            dict(
                category=None,
                subcategory=None,
                market=effective_market,
                tags=filter_tags,
                min_price=facets.min_price,
                max_price=facets.max_price,
                min_rating=facets.min_rating,
            ),
            # Level 2 — only market filter (pure relevance ranking)
            dict(
                category=None,
                subcategory=None,
                market=effective_market,
                tags=None,
                min_price=None,
                max_price=None,
                min_rating=None,
            ),
        ]

        results: list[dict] = []
        seen_ids: set[str] = set()

        for filters in filter_levels:
            candidates = self._apply_filters(
                ranked_indices, rrf_scores, filters, top_k
            )
            for product in candidates:
                pid = product["id"]
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    results.append(product)
            if len(results) >= top_k:
                break

        return results[:top_k]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_filters(
        self,
        ranked_indices: list[int],
        rrf_scores: dict[int, float],
        filters: dict,
        top_k: int,
    ) -> list[dict]:
        """Apply facet filters to ranked results.

        Args:
            ranked_indices: Product indices sorted by RRF score (desc).
            rrf_scores: RRF scores keyed by product index.
            filters: Dict of filter kwargs for ``_passes_filters``.
            top_k: Maximum number of results.

        Returns:
            List of matching product dicts with ``_score``.
        """
        results: list[dict] = []
        for idx in ranked_indices:
            product = self._index._products[idx]
            if not self._index._passes_filters(
                product,
                category=filters["category"],
                subcategory=filters["subcategory"],
                market=filters["market"],
                tags=filters["tags"],
                min_price=filters["min_price"],
                max_price=filters["max_price"],
                min_rating=filters["min_rating"],
            ):
                continue
            results.append({**product, "_score": rrf_scores[idx]})
            if len(results) >= top_k:
                break
        return results

    @staticmethod
    def _merge_ranked_lists(*ranked_lists: list[int]) -> list[int]:
        """Merge multiple ranked lists using Reciprocal Rank Fusion."""
        scores: dict[int, float] = defaultdict(float)
        for ranked_list in ranked_lists:
            for rank, idx in enumerate(ranked_list):
                scores[idx] += 1.0 / (RRF_K + rank + 1)
        return sorted(scores, key=lambda i: scores[i], reverse=True)
