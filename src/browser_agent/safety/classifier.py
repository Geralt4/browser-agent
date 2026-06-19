from __future__ import annotations

import re
from typing import TYPE_CHECKING

from browser_agent.safety.types import PendingAction

if TYPE_CHECKING:
    from browser_use.llm.base import BaseChatModel

# Irreversible / sensitive intents the brief calls out: send, publish, purchase,
# delete, submit. Matched as substrings against an action's textual params
# (element label, target url, typed text).
SENSITIVE_KEYWORDS = (
    "send",
    "submit",
    "publish",
    "post",
    "purchase",
    "buy",
    "checkout",
    "place order",
    "pay",
    "payment",
    "delete",
    "remove",
    "destroy",
    "transfer",
    "wire",
    "subscribe",
    "deactivate",
    "close account",
)

# Personal / payment data typed into a field -> always confirm.
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,16}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")

_SENSITIVITY_PROMPT = """Classify a browser-agent action as sensitive or harmless.

Action: {action}

A sensitive action is one that is irreversible, destructive (e.g. delete, destroy),
financial (purchase, transfer, pay), publishes data, submits forms, modifies account
state (subscribe, deactivate, close account), or types personal data (credit cards, SSNs,
passwords, emails).

Reply with a single word: YES (sensitive) or NO (harmless)."""


def is_sensitive(action: PendingAction) -> bool:
    """Heuristic stub. Phase 4 replaces this with the full classifier."""
    blob = " ".join(str(v) for v in action.params.values()).lower()
    if any(kw in blob for kw in SENSITIVE_KEYWORDS):
        return True

    if action.name in {"type_text", "input"}:
        text = str(action.params.get("text", ""))
        if _CARD_RE.search(text) or _SSN_RE.search(text) or _EMAIL_RE.search(text):
            return True

    return False


async def classify_sensitive_llm(
    action: PendingAction, chat_model: BaseChatModel
) -> bool | None:
    """LLM-based sensitivity classification fallback.

    Called only when the keyword heuristic returns False and the user has
    opted into LLM-based classification via SENSITIVITY_LLM. Returns True if
    the model flags the action as sensitive, False if harmless, or None on
    model error (caller should treat as "not sensitive" — default-allow).
    """
    from browser_use.llm.messages import UserMessage

    prompt = _SENSITIVITY_PROMPT.format(action=action.summary())
    messages = [UserMessage(content=prompt)]

    try:
        result = await chat_model.ainvoke(messages)
    except Exception:
        return None

    content = _extract_text(result)
    if content is None:
        return None
    return content.strip().upper().startswith("YES")


def _extract_text(result: object) -> str | None:
    """Extract text from a ChatOpenAI ainvoke result, which may be wrapped."""
    if isinstance(result, str):
        return result
    if hasattr(result, "content"):
        return str(result.content)
    if hasattr(result, "completion"):
        return str(result.completion)
    return str(result)
