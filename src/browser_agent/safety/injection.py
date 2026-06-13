from __future__ import annotations


def sanitize(page_content: str) -> str:
    """Treat page content as untrusted DATA, never as instructions.

    TODO(Phase 4): strip/flag hidden DOM text, off-screen elements, and
    instruction-like patterns ("ignore previous instructions", "for security
    reasons delete..."). This is the primary failure mode of browser agents and
    must land before the agent touches a real logged-in session.
    """
    return page_content
