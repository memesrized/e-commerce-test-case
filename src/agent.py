"""LangChain ReAct agent for the e-commerce search chatbot.

Uses LangChain 1.x ``create_agent`` (LangGraph-backed tool-calling loop)
with ``MemorySaver`` for per-session conversation history.
"""
from __future__ import annotations

from langchain.agents import create_agent as _lc_create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from .context import ContextProvider, StaticContextProvider
from .guardrails import GUARDRAIL_PROMPT

# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful and discreet shopping assistant for a luxury adult e-commerce platform.
{catalog_context}
{guardrails}
IMPORTANT — HOW TO SEARCH:
- NEVER pass the customer's exact words as the "query" argument to a tool
- Rephrase into a product-feature query describing attributes, benefits, or use cases
  Example: "something quiet for myself"   → query: "quiet discreet vibrator solo use"
  Example: "gift for my girlfriend"       → query: "romantic gift set couples wellness"
- Choose filters (category, market, tags, price) that meaningfully narrow results
- Run multiple tool calls when the request covers several types or interpretations
- If no results appear, broaden the query or drop a filter and try again
- Present results clearly: product name, price in £, key features, and why it fits
"""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_agent(
    tools: list,
    context_provider: ContextProvider | None = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
) -> CompiledStateGraph:
    """Create a ReAct-style agent graph for e-commerce product search.

    Conversation memory is handled automatically by LangGraph's
    ``MemorySaver`` checkpointer. Pass a ``thread_id`` in the config
    dict to each ``invoke`` call to maintain per-session history.

    Args:
        tools: LangChain tools the agent may use.
        context_provider: Provides catalog knowledge for the system prompt.
                          Defaults to ``StaticContextProvider``.
        model: OpenAI model name (e.g. "gpt-4o-mini", "gpt-4o").
        temperature: LLM sampling temperature.

    Returns:
        Compiled ``StateGraph`` ready to call with
        ``graph.invoke({"messages": [...]}, config={"configurable": {"thread_id": ...}})``.
    """
    if context_provider is None:
        context_provider = StaticContextProvider()

    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        catalog_context=context_provider.get_context(),
        guardrails=GUARDRAIL_PROMPT,
    )

    llm = ChatOpenAI(model=model, temperature=temperature)

    return _lc_create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=MemorySaver(),
    )

