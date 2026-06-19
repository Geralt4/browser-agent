"""Signature-parity guard for InjectionSafeMessageManager.

`InjectionSafeMessageManager.create_state_messages` overrides the parent and
must forward *every* constructor parameter the parent declares. If browser-use
adds/removes/renames a param in a dependency bump and our override doesn't
mirror it, sanitization silently breaks (the parent gets a default value the
override didn't forward). This test catches that drift — it's the same class
of break AGENTS.md warns about for `_wrap_message_manager`.
"""

from __future__ import annotations

import inspect

from browser_use.agent.message_manager.service import MessageManager

from browser_agent.agent.safe_message_manager import InjectionSafeMessageManager


def _params(sig: inspect.Signature) -> set[str]:
    return {p for p in sig.parameters if p != "self"}


def test_override_mirrors_parent_create_state_messages_params():
    parent = _params(inspect.signature(MessageManager.create_state_messages))
    override = _params(
        inspect.signature(InjectionSafeMessageManager.create_state_messages)
    )
    assert override == parent, (
        "InjectionSafeMessageManager.create_state_messages params drifted from "
        f"MessageManager. Missing in override: {parent - override}. "
        f"Extra in override: {override - parent}. "
        "Update the override signature so sanitization keeps working."
    )


def test_override_calls_super():
    # Sanity check that the override is actually a method on the subclass
    # and forwards to super (static structural check via source).
    import re

    src = inspect.getsource(InjectionSafeMessageManager.create_state_messages)
    assert re.search(r"super\(\)\.create_state_messages\(", src), (
        "override must call super().create_state_messages(...) to reuse the real "
        "DOM pipeline before sanitizing"
    )
