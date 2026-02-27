"""Streamlit chat interface for the e-commerce smart search system.

Communicates with the FastAPI backend via ``/api/chat``.
"""
from __future__ import annotations

import os

import requests
import streamlit as st

BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Smart Shopping Assistant",
    page_icon="🛍️",
    layout="centered",
)

st.title("🛍️ Smart Shopping Assistant")
st.caption("AI-powered product search with multilingual support")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages: list[dict[str, str]] = []

# ---------------------------------------------------------------------------
# Display chat history
# ---------------------------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("What are you looking for today?"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call backend
    with st.chat_message("assistant"):
        with st.spinner("Searching for the perfect products…"):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/api/chat",
                    json={
                        "message": prompt,
                        "history": [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages[:-1]
                        ],
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()

                # Display the conversational response
                st.markdown(data["message"])

                # Show product cards in an expandable section
                products = data.get("products", [])
                if products:
                    with st.expander(
                        f"📦 {len(products)} product(s) found",
                        expanded=False,
                    ):
                        for p in products:
                            rating = f"⭐ {p['rating']}" if p.get("rating") else ""
                            reviews = (
                                f"({p['review_count']} reviews)"
                                if p.get("review_count")
                                else ""
                            )
                            desc = p.get("description", "")
                            tags = ", ".join(p.get("tags", []))

                            st.markdown(
                                f"**{p['name']}** — "
                                f"£{p['price_gbp']:.2f}  {rating} {reviews}\n\n"
                                f"_{desc}_\n\n"
                                f"Tags: {tags}"
                            )
                            st.divider()

                st.session_state.messages.append(
                    {"role": "assistant", "content": data["message"]}
                )

            except requests.exceptions.ConnectionError:
                err = (
                    "⚠️ Cannot connect to the backend. "
                    "Make sure it's running at "
                    f"`{BACKEND_URL}`."
                )
                st.error(err)
                st.session_state.messages.append(
                    {"role": "assistant", "content": err}
                )
            except requests.exceptions.HTTPError as exc:
                err = f"⚠️ Backend error: {exc}"
                st.error(err)
                st.session_state.messages.append(
                    {"role": "assistant", "content": err}
                )

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "AI-powered product search with semantic understanding, "
        "multilingual support, and smart filtering."
    )
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        "1. Your message is analysed by an LLM to extract intent\n"
        "2. Smart facets and keywords are generated automatically\n"
        "3. Hybrid search (semantic + BM25) finds the best matches\n"
        "4. Results are presented in a conversational format"
    )
    st.markdown("---")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()
