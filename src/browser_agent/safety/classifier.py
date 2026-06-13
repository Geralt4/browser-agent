from __future__ import annotations

import re

from browser_agent.safety.types import PendingAction

# Irreversible / sensitive intents the brief calls out: send, publish, purchase,
# delete, submit. Matched as substrings against an action's textual params
# (element label, target url, typed text).
SENSITIVE_KEYWORDS = (
    "send",
    "submit",
    "publish",
    "post ",
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
