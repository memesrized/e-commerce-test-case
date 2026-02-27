# E-Commerce Smart Search — POC

A multilingual, AI-powered product search system for a large e-commerce platform.  
Users interact with a conversational chatbot that translates natural-language requests into precise, hybrid product searches using semantic embeddings, BM25 keyword matching, LLM-powered query understanding, and facet filtering.

The system is available in two modes:
- **CLI chatbot** — interactive terminal agent (`main.py`)
- **Web application** — FastAPI backend + Streamlit chat UI (`docker compose up`)

---

## Architecture overview

```
User input (any language)
        │
        ├──── CLI ──────────────────────────────────────────┐
        │                                                    │
        ├──── Streamlit UI ──► FastAPI backend               │
        │          (ui/)            (backend/)               │
        ▼                              │                     │
  ┌──────────────┐                     ▼                     │
  │  LLM Agent   │    ┌────────────────────────────┐         │
  │  (GPT-4o)    │    │    SmartProductSearch       │  src/smart_search.py
  │  + guardrails│    │    (backend pipeline)       │         │
  └──────┬───────┘    └──────────┬─────────────────┘         │
         │ tool calls            │                           │
         ▼                       ▼                           │
  ┌─────────────────────────────────────────────────────┐    │
  │             QueryUnderstanding (LLM)                │  src/query_understanding.py
  │                                                     │
  │  ┌──────────────────┐   ┌─────────────────────────┐ │
  │  │  Facet selection  │   │  Keyword extraction     │ │
  │  │  (LLM picks      │   │  (multilingual BM25     │ │
  │  │   filters from   │   │   keywords: EN + local) │ │
  │  │   CatalogFacets) │   │                         │ │
  │  └──────────────────┘   └─────────────────────────┘ │
  └──────────────────────┬──────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────┐
  │                   ProductIndex                       │  src/search.py
  │                                                      │
  │  ┌───────────────────┐   ┌────────────────────────┐  │
  │  │  Semantic search  │   │   BM25 keyword search  │  │
  │  │  (sentence-trans- │   │   (bm25s)              │  │
  │  │   formers + numpy │   │                        │  │
  │  │   cosine sim)     │   │                        │  │
  │  └────────┬──────────┘   └───────────┬────────────┘  │
  │           │    Reciprocal Rank Fusion │               │
  │           └──────────────┬───────────┘               │
  │                          ▼                            │
  │               Facet filters (LLM-selected)            │
  └──────────────────────────────────────────────────────┘
                         │
                         ▼
                 ┌───────────────┐
                 │  Guardrails   │  src/guardrails.py
                 │  (input +     │
                 │   output)     │
                 └───────────────┘
```

### Module breakdown

| File | Responsibility |
|---|---|
| `src/context.py` | Catalog knowledge base — static and dataset-derived providers |
| `src/search.py` | Hybrid search engine — embeddings + BM25 fused with RRF |
| `src/query_understanding.py` | LLM-powered facet selection and multilingual keyword extraction |
| `src/smart_search.py` | End-to-end smart search pipeline (query understanding → hybrid search) |
| `src/guardrails.py` | Content safety — blocked-term filters and prompt-level guardrails |
| `src/tools.py` | LangChain `@tool` wrapping `ProductIndex` for agent use |
| `src/agent.py` | LangGraph-backed ReAct agent factory with conversation memory |
| `main.py` | Interactive CLI chatbot entry point |
| `backend/app.py` | FastAPI backend with NLQ search and conversational endpoints |
| `ui/app.py` | Streamlit chat UI |

---

## Modules in detail

### `src/context.py` — Catalog knowledge base

Defines a `ContextProvider` **Protocol** (interface) that any knowledge source must satisfy:

```python
class ContextProvider(Protocol):
    def get_context(self) -> str: ...
```

Two implementations are provided:

- **`StaticContextProvider`** — hardcoded text covering all categories, subcategories, available filter tags, market codes, price range, and per-intent query guidance (e.g. *gift queries → tags: ["gift", "romantic"]*).

- **`DatasetContextProvider`** — dynamically extracts catalog context from actual product data: category distribution and counts, price ranges, popular tags by frequency, market coverage per region, and language breakdown. This gives the LLM a realistic picture of what's in the catalog.

The context is injected into the LLM system prompt so the agent understands the catalog schema before it writes any tool call.  
**To extend**: swap in a `RAGContextProvider` that retrieves relevant passages from a vector DB at query time — the agent and tools need no changes.

---

### `src/query_understanding.py` — LLM-powered query analysis

Introduces an LLM-in-the-loop step **before** search execution. Two structured-output calls:

**Facet selection** (`QueryUnderstanding.select_facets`)  
`CatalogFacets` extracts all available categories, subcategories, tags, markets, and price/rating ranges from the product catalog at startup. The LLM receives these facets along with the user query and returns a `FacetSelection` — only populating filters that are clearly relevant. The prompt enforces strict rules: tags must be exact matches from the catalog list (no invented tags), categories are only set when unambiguously implied, and fewer filters are preferred over more.

**Keyword extraction** (`QueryUnderstanding.extract_keywords`)  
Generates BM25-optimised keywords focused on product attributes, features, and use-case terms — stripping filler and conversational tone. Supports **multilingual search**: when the target market is non-English (DE, FR, JP), the LLM generates keywords in **both English and the local language** (natural product-search terms, not mechanical translations). English-only markets get English-only keywords. The `detected_market` field is validated against known market codes — LLM artefacts like the literal string `"null"` are handled gracefully.

Market-language mapping:
| Market | Language | Keyword strategy |
|---|---|---|
| UK, US, AU | English | English only |
| DE | German | English + German |
| FR | French | English + French |
| JP | Japanese | English + Japanese |

**Extensibility**: the facet selection can later be fed only the most relevant facets (pre-filtered by category embeddings, for example) instead of the full catalog — the interface is ready for it.

---

### `src/smart_search.py` — End-to-end search pipeline

`SmartProductSearch` composes `ProductIndex` + `QueryUnderstanding` into a single pipeline:

1. **LLM selects facets** from catalog metadata based on the query
2. **LLM-selected tags are sanitised** against the real catalog — invented tags are dropped
3. **LLM extracts keywords** (multilingual when target market is non-English)
4. **Semantic search** runs on the original query (multilingual model handles this natively)
5. **BM25** runs on LLM-extracted keywords (multiple ranked lists merged via RRF when multilingual)
6. **RRF fusion** merges semantic and BM25 rankings
7. **Progressive filter relaxation** fills results up to `top_k`:

| Level | Filters applied | Purpose |
|---|---|---|
| 0 | All LLM-selected (category, subcategory, tags, price, market) | Best precision |
| 1 | Drop category + subcategory, keep tags + price + market | Recover from wrong category guesses |
| 2 | Market only (pure relevance ranking) | Ensure results for vague queries |

Results accumulate across levels (deduped by product ID). Strict-match products appear first, then broader matches fill remaining slots. This ensures a vague query like *"something fun for date night"* returns the best exact match **and** additional relevant options.

This pipeline is used by the FastAPI backend. The CLI agent uses the original `ProductIndex` via LangChain tools (the agent itself handles query rephrasing).

---

### `src/search.py` — Hybrid search engine

`ProductIndex` builds two indices from the product catalog JSON at startup:

**Semantic index**  
Products are embedded with [`distiluse-base-multilingual-cased-v2`](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2) — a multilingual sentence-transformers model supporting 50+ languages. Embeddings are L2-normalised; cosine similarity is a fast numpy dot product at query time.

**BM25 index**  
`bm25s` builds a sparse keyword index over the same concatenated product text (name + description + category + tags). Good at exact term matching and compensates for semantic search weaknesses on rare or very specific terms.

**Fusion**  
Both methods produce independent ranked lists, merged with **Reciprocal Rank Fusion (RRF)**:

$$\text{score}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)}, \quad k = 60$$

This boosts documents that rank highly in *both* methods, requiring no score normalisation.

**Facet filters** applied post-fusion (selected by LLM or passed explicitly):

| Parameter | Type | Description |
|---|---|---|
| `category` | `str` | e.g. `"vibrators"`, `"wellness"`, `"bondage"` |
| `subcategory` | `str` | e.g. `"clitoral"`, `"plugs"`, `"wand"` |
| `market` | `str` | `UK`, `US`, `DE`, `FR`, `AU`, `JP` |
| `tags` | `list[str]` | At least one must match (OR logic) |
| `min_price` / `max_price` | `float` | GBP |
| `min_rating` | `float` | 0–5 scale |

---

### `src/guardrails.py` — Content safety

Ensures the chatbot stays on-brand and appropriate:

- **Blocked-term filter** — regex-based scan of both user input and LLM output for non-consensual, minor-related, offensive, or illegal terms. Flagged responses are replaced with a safe fallback message.
- **Prompt-level guardrails** — a `GUARDRAIL_PROMPT` string is injected into the agent system prompt, instructing the LLM to keep responses tasteful, professional, and on-brand. Inappropriate user requests are redirected to product search without judgement.
- **`check_input(text)`** — returns `False` if user input triggers a blocked term.
- **`check_response(text)`** — returns `(is_safe, output_text)`, replacing unsafe content with a fallback.

This is a lightweight demo implementation. For production, extend with LLM-based moderation (e.g. OpenAI Moderation API).

---

### `src/tools.py` — LangChain tool

`create_search_tools(index)` returns a single `search_products` `@tool` whose docstring acts as the schema description the LLM reads to understand how to call it.

The key design choice is the `query` field guidance:

> *Focus on product attributes and benefits — NOT raw user words.*  
> Good: `"rechargeable quiet clitoral vibrator waterproof"`  
> Good: `"romantic massage oil gift set for couples"`

This separation of user intent from search query is enforced both in the tool docstring and the agent system prompt, making the agent robust to vague or colloquial phrasing.

**To extend**: add more tools (e.g. `get_product_details`, `get_recommendations_by_id`) by adding functions to `tools.py` and passing them to `create_agent`.

---

### `src/agent.py` — ReAct agent

Uses **LangChain 1.x `create_agent`** backed by LangGraph's tool-calling loop.  
Conversation memory is handled automatically by `MemorySaver` (LangGraph checkpointer). Each session receives a unique `thread_id`; all message history is stored in-process without any external database.

The system prompt explicitly instructs the LLM:
- Never use raw user input as the search `query`
- Rephrase into feature-based, attribute-focused queries
- Run multiple tool calls for multi-part requests
- Broaden the query and retry if no results are found
- Follow content guardrails at all times

**To extend**: pass a custom `ContextProvider`, swap `MemorySaver` for `PostgresSaver`/`RedisSaver` for persistent cross-session memory, or change the `model` parameter to use a different LLM.

---

### `backend/app.py` — FastAPI backend

A standalone REST API that exposes the smart search pipeline without the LangChain agent overhead:

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Liveness probe |
| `/api/search` | POST | NLQ search → structured product results |
| `/api/chat` | POST | Conversational search → LLM-formatted response + products |

**`/api/search`** runs the full `SmartProductSearch` pipeline (LLM facet selection → LLM keyword extraction → hybrid search → RRF → facet filters) and returns raw product results.

**`/api/chat`** runs the same search, then passes results through a chat LLM to generate a natural-language response with guardrail checking. Accepts conversation history for context continuity.

---

### `ui/app.py` — Streamlit chat UI

A full chat interface that communicates with the FastAPI backend:

- Message history with user/assistant bubbles
- Expandable product cards showing name, price, rating, review count, description, and tags
- Sidebar with "how it works" explanation and clear-chat button
- Error handling for backend connectivity issues

---

## Setup

### Option 1: CLI chatbot (local)

**Requirements:** Python 3.12+, an OpenAI API key.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...

# 4. Run the chatbot
python main.py
```

### Option 2: Web application (Docker)

**Requirements:** Docker, Docker Compose, an OpenAI API key.

```bash
# 1. Configure your API key
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...

# 2. Build and start both services
docker compose up --build

# 3. Open the UI
# Streamlit UI: http://localhost:8501
# API docs:     http://localhost:8000/docs
```

### Option 3: Run backend and UI separately (local)

```bash
# Terminal 1 — Backend
pip install -r backend/requirements.txt
uvicorn backend.app:app --reload

# Terminal 2 — UI
pip install -r ui/requirements.txt
streamlit run ui/app.py
```

---

## Example session

```
You: I'm looking for a quiet gift for my partner, budget around £30

Assistant: Here are some great quiet options under £30 for your partner:

1. Luxury Couples Vibrating Ring — £29.99
   Whisper-quiet, 10 vibration modes, rechargeable and waterproof. A lovely couples gift.

2. Rechargeable Bullet Vibrator — £24.99
   Compact and discreet, 10 modes, travel-lock. Great for solo or shared use.

3. Satin Blindfold & Restraint Set — £24.99
   Beginner-friendly soft-bondage kit, thoughtful for couples exploring together.

Shall I filter by your region or narrow down further?

You: show me only waterproof ones

Assistant: Of course! From that selection, the waterproof options are:

1. Luxury Couples Vibrating Ring — £29.99 (waterproof, rechargeable, 10 modes)

Want me to check for other waterproof products across the full catalogue?
```

---

## Extending the system

| Component | How to replace |
|---|---|
| LLM | Change `model=` in `create_agent()` or pass any `BaseChatModel` |
| Context / knowledge | Implement `ContextProvider` protocol — use `DatasetContextProvider` for data-driven context, or a RAG provider for production |
| Search engine | Swap `ProductIndex` for any class with a `.search()` method |
| Query understanding | Replace `QueryUnderstanding` with a custom implementation; adjust `SmartProductSearch` accordingly |
| Tools | Add more `@tool` functions in `tools.py` and pass them to `create_agent` |
| Memory | Replace `MemorySaver` in `agent.py` with `PostgresSaver` / `RedisSaver` |
| Guardrails | Extend `guardrails.py` with LLM-based moderation (OpenAI Moderation API, custom classifier) |
| UI | Replace Streamlit with any frontend that calls the FastAPI endpoints |

---

## Data format

Products are loaded from a JSON array. Each item requires:

```json
{
  "id": "LH-001",
  "name": "...",
  "description": "...",
  "category": "vibrators",
  "subcategory": "clitoral",
  "tags": ["beginner-friendly", "waterproof"],
  "price_gbp": 49.99,
  "markets": ["UK", "US", "DE"],
  "language": "en",
  "rating": 4.6,
  "review_count": 1823
}
```
