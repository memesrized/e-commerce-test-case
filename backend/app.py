"""FastAPI backend for e-commerce smart search.

Exposes two endpoints:
- ``POST /api/search`` — raw NLQ search returning product results.
- ``POST /api/chat``   — conversational search with LLM-formatted response.
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Ensure project root is on the path so ``src`` is importable regardless
# of how the app is started (Docker, uvicorn, pytest, …).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.context import DatasetContextProvider  # noqa: E402
from src.guardrails import GUARDRAIL_PROMPT, check_response  # noqa: E402
from src.smart_search import SmartProductSearch  # noqa: E402

from .config import DATA_FILE, LLM_MODEL  # noqa: E402


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """Payload for the ``/api/search`` endpoint."""

    query: str = Field(..., description="Natural language search query")
    market: Optional[str] = Field(
        None, description="Target market (UK, US, DE, FR, AU, JP)"
    )
    top_k: int = Field(5, ge=1, le=20, description="Max results to return")


class ProductResult(BaseModel):
    """Serialised product returned to clients."""

    id: str
    name: str
    description: str
    category: str
    subcategory: Optional[str] = None
    tags: list[str] = []
    price_gbp: float
    rating: Optional[float] = None
    review_count: Optional[int] = None
    markets: list[str] = []


class SearchResponse(BaseModel):
    """Response for ``/api/search``."""

    products: list[ProductResult]
    query: str


class ChatMessage(BaseModel):
    """A single message in a conversation."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Payload for the ``/api/chat`` endpoint."""

    message: str = Field(..., description="User message")
    history: list[ChatMessage] = Field(
        default_factory=list, description="Previous conversation messages"
    )


class ChatResponse(BaseModel):
    """Response for ``/api/chat``."""

    message: str
    products: list[ProductResult]


# ---------------------------------------------------------------------------
# Application state (populated during lifespan)
# ---------------------------------------------------------------------------

_smart_search: SmartProductSearch | None = None
_chat_llm: ChatOpenAI | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Load the search index and LLM on startup; tear down on shutdown."""
    global _smart_search, _chat_llm  # noqa: PLW0603

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is required"
        )

    print("Loading product index and building search engine…")
    context_provider = DatasetContextProvider.from_file(DATA_FILE)
    _smart_search = SmartProductSearch.from_file(
        DATA_FILE,
        context=context_provider.get_context(),
        model=LLM_MODEL,
    )
    _chat_llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3)

    print("Backend ready!")
    yield
    print("Shutting down…")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="E-Commerce Smart Search API",
    description="AI-powered product search with LLM query understanding",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_product(p: dict) -> ProductResult:
    return ProductResult(
        id=p["id"],
        name=p["name"],
        description=p["description"],
        category=p["category"],
        subcategory=p.get("subcategory"),
        tags=p.get("tags", []),
        price_gbp=p["price_gbp"],
        rating=p.get("rating"),
        review_count=p.get("review_count"),
        markets=p.get("markets", []),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict[str, str]:
    """Simple liveness probe."""
    return {"status": "ok"}


@app.post("/api/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Run a smart NLQ search and return matching products."""
    if _smart_search is None:
        raise HTTPException(503, "Search engine not initialised")

    results = _smart_search.search(
        query=request.query,
        market=request.market,
        top_k=request.top_k,
    )
    return SearchResponse(
        products=[_format_product(p) for p in results],
        query=request.query,
    )


_CHAT_SYSTEM = f"""\
You are a helpful and tasteful shopping assistant for a luxury e-commerce platform.
Given search results, present the most relevant products naturally.
Highlight key features, price, and why each product fits the customer's needs.
Keep responses concise and friendly.

{GUARDRAIL_PROMPT}"""


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Conversational search: find products and respond naturally."""
    if _smart_search is None or _chat_llm is None:
        raise HTTPException(503, "Search engine not initialised")

    # 1. Smart search
    results = _smart_search.search(request.message, top_k=5)
    formatted = [_format_product(p) for p in results]

    products_text = json.dumps(
        [p.model_dump() for p in formatted],
        indent=2,
        ensure_ascii=False,
    )

    # 2. Build conversation for LLM
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _CHAT_SYSTEM}
    ]
    for msg in request.history[-10:]:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append(
        {
            "role": "user",
            "content": f"{request.message}\n\n[Search results]\n{products_text}",
        }
    )

    # 3. Generate and guard response
    response = _chat_llm.invoke(messages)
    _, safe_text = check_response(response.content)

    return ChatResponse(message=safe_text, products=formatted)
