"""Hybrid product search engine.

Combines semantic embeddings (cosine similarity via numpy) and BM25 keyword
search, fused with Reciprocal Rank Fusion (RRF). Supports facet filtering
by category, subcategory, market, tags, price range, and rating.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import bm25s
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/distiluse-base-multilingual-cased-v2"

# Standard RRF constant — higher value = smoother rank blending
RRF_K = 60

# How many top BM25 candidates to pull before fusion
BM25_CANDIDATES = 50


def _build_doc_text(product: dict) -> str:
    """Combine product fields into a single searchable text string."""
    parts = [
        product.get("name", ""),
        product.get("description", ""),
        product.get("category", ""),
        product.get("subcategory", ""),
        " ".join(product.get("tags", [])),
    ]
    return " ".join(filter(None, parts))


class ProductIndex:
    """Hybrid product search index.

    Builds two internal indices on construction:
    - A dense embedding matrix (sentence-transformers) for semantic search.
    - A BM25 index (bm25s) for keyword search.

    At query time the two ranked lists are merged with Reciprocal Rank Fusion
    before any facet filters are applied.
    """

    def __init__(self, products: list[dict], model_name: str = DEFAULT_MODEL) -> None:
        """Build the search index from a list of product dicts.

        Args:
            products: List of product dictionaries from the catalog JSON.
            model_name: Sentence-transformers model for multilingual embeddings.
        """
        self._products = products
        self._doc_texts = [_build_doc_text(p) for p in products]

        print(f"Loading embedding model: {model_name}")
        self._model = SentenceTransformer(model_name)

        print("Building embedding index…")
        self._embeddings: np.ndarray = self._model.encode(
            self._doc_texts, normalize_embeddings=True, show_progress_bar=True
        )

        print("Building BM25 index…")
        corpus_tokens = bm25s.tokenize(self._doc_texts)
        self._bm25 = bm25s.BM25()
        self._bm25.index(corpus_tokens)

    @property
    def products(self) -> list[dict]:
        """The underlying product catalog."""
        return self._products

    @classmethod
    def from_file(cls, path: str | Path, model_name: str = DEFAULT_MODEL) -> "ProductIndex":
        """Load products from a JSON file and build the index.

        Args:
            path: Path to the products JSON file.
            model_name: Sentence-transformers model name.
        """
        products = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(products, model_name)

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        subcategory: Optional[str] = None,
        market: Optional[str] = None,
        tags: Optional[list[str]] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_rating: Optional[float] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Hybrid search with optional facet filters.

        Merges semantic and BM25 rankings via RRF, then applies filters.

        Args:
            query: Descriptive, feature-focused search query.
            category: Filter by product category.
            subcategory: Filter by product subcategory.
            market: Filter by market region (UK, US, DE, FR, AU, JP).
            tags: Keep only products that have at least one of these tags.
            min_price: Minimum price in GBP.
            max_price: Maximum price in GBP.
            min_rating: Minimum product rating (0–5).
            top_k: Maximum number of results to return.

        Returns:
            List of matching product dicts sorted by relevance,
            each with an added ``_score`` field.
        """
        semantic_ranks = self._semantic_ranks(query)
        bm25_ranks = self._bm25_ranks(query)
        rrf_scores = self._compute_rrf(semantic_ranks, bm25_ranks)

        ranked_indices = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)

        results = []
        for idx in ranked_indices:
            product = self._products[idx]
            if not self._passes_filters(
                product, category, subcategory, market, tags,
                min_price, max_price, min_rating,
            ):
                continue
            results.append({**product, "_score": rrf_scores[idx]})
            if len(results) >= top_k:
                break

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _semantic_ranks(self, query: str) -> list[int]:
        """Return all product indices ranked by cosine similarity."""
        query_emb = self._model.encode([query], normalize_embeddings=True)[0]
        scores = np.dot(self._embeddings, query_emb)
        return np.argsort(scores)[::-1].tolist()

    def _bm25_ranks(self, query: str) -> list[int]:
        """Return top product indices ranked by BM25 score."""
        k = min(BM25_CANDIDATES, len(self._products))
        query_tokens = bm25s.tokenize([query])
        results, _ = self._bm25.retrieve(query_tokens, k=k)
        return results[0].tolist()

    def _compute_rrf(
        self,
        semantic_ranks: list[int],
        bm25_ranks: list[int],
    ) -> dict[int, float]:
        """Compute Reciprocal Rank Fusion scores from two ranked lists."""
        scores: dict[int, float] = defaultdict(float)
        for rank, idx in enumerate(semantic_ranks):
            scores[idx] += 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(bm25_ranks):
            scores[idx] += 1.0 / (RRF_K + rank + 1)
        return scores

    def _passes_filters(
        self,
        product: dict,
        category: Optional[str],
        subcategory: Optional[str],
        market: Optional[str],
        tags: Optional[list[str]],
        min_price: Optional[float],
        max_price: Optional[float],
        min_rating: Optional[float],
    ) -> bool:
        """Return True if the product satisfies all active facet filters."""
        if category and product.get("category") != category:
            return False
        if subcategory and product.get("subcategory") != subcategory:
            return False
        if market and market not in product.get("markets", []):
            return False
        if tags and not any(t in product.get("tags", []) for t in tags):
            return False
        price = product.get("price_gbp", 0.0)
        if min_price is not None and price < min_price:
            return False
        if max_price is not None and price > max_price:
            return False
        if min_rating is not None and product.get("rating", 0.0) < min_rating:
            return False
        return True
