"""Content guardrails for chatbot responses.

Provides simple stopword-based filtering to ensure the chatbot does not
generate off-brand, offensive, or inappropriate content.

Suitable for a demo — extend with LLM-based moderation for production.
"""
from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Blocked terms — words/phrases that must not appear in bot output
# ---------------------------------------------------------------------------

BLOCKED_TERMS: list[str] = [
    # Non-consensual / violence
    "non-consensual",
    "without consent",
    "forced",
    # Minors
    "underage",
    "minor",
    "child",
    "children",
    "teen",
    "teenager",
    # Offensive / derogatory
    "slut",
    "whore",
    # Illegal
    "illegal",
    "narcotic",
]

_BLOCKED_RE: re.Pattern[str] = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in BLOCKED_TERMS) + r")\b",
    re.IGNORECASE,
)

FALLBACK_RESPONSE: str = (
    "I'm sorry, I wasn't able to generate an appropriate response. "
    "Could you please rephrase your question? I'm here to help you "
    "find the perfect product."
)


# ---------------------------------------------------------------------------
# System-prompt guardrail snippet (injected into the agent prompt)
# ---------------------------------------------------------------------------

GUARDRAIL_PROMPT: str = """\
CONTENT GUIDELINES — you MUST follow these at all times:
- Keep all responses tasteful, professional, and on-brand for a luxury retailer.
- Never discuss, encourage, or reference illegal activities or non-consensual scenarios.
- Do not use crude, derogatory, or overly clinical language.
- If a user request is inappropriate or off-topic, politely redirect them \
to product search without judgement.
"""


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def check_response(text: str) -> tuple[bool, str]:
    """Check whether a chatbot response passes content guardrails.

    Args:
        text: The LLM-generated response.

    Returns:
        ``(is_safe, output_text)`` — if unsafe the text is replaced with
        a generic fallback message.
    """
    if _BLOCKED_RE.search(text):
        return False, FALLBACK_RESPONSE
    return True, text


def check_input(text: str) -> bool:
    """Return ``True`` if the user input is appropriate for processing."""
    return not _BLOCKED_RE.search(text)
