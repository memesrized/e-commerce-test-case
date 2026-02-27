"""Entry point for the e-commerce smart search chatbot.

Run:
    python main.py
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

DATA_FILE = Path(__file__).parent / "sample-products.json"


def run_chatbot() -> None:
    """Run the interactive e-commerce search chatbot in the terminal."""
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set. Add it to a .env file (see .env.example).")
        sys.exit(1)

    # Deferred imports so the user sees feedback before heavy loading begins
    from src.search import ProductIndex
    from src.tools import create_search_tools
    from src.agent import create_agent
    from src.context import DatasetContextProvider
    from src.guardrails import check_response, check_input

    print("Building product search index (may take a moment on first run)…")
    index = ProductIndex.from_file(DATA_FILE)
    context_provider = DatasetContextProvider(index.products)

    graph = create_agent(
        create_search_tools(index), context_provider=context_provider
    )

    # Each session gets a unique thread_id so LangGraph's MemorySaver
    # keeps conversation history isolated.
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("\n" + "=" * 55)
    print("  Shopping assistant ready!  Type 'exit' to quit.")
    print("=" * 55 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        if not check_input(user_input):
            print("\nAssistant: I'd be happy to help you find products. "
                  "Could you rephrase your request?\n")
            continue

        result = graph.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        response: str = result["messages"][-1].content
        _, response = check_response(response)
        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    run_chatbot()

