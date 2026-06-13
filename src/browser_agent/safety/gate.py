from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from browser_agent.safety.types import PendingAction, SafetyDecision

# A confirmation strategy: given a sensitive action, return an allow/deny.
# The CLI default is swappable so the Phase 5 chat UI can drop in an inline
# approval prompt without touching the safety layer.
ConfirmCallback = Callable[[PendingAction], Awaitable[SafetyDecision]]


async def cli_confirm(action: PendingAction) -> SafetyDecision:
    prompt = (
        f"\n[safety] Sensitive action requires approval:\n"
        f"  {action.summary()}\n"
        f"  approve? [y/N] "
    )
    answer = (await asyncio.to_thread(input, prompt)).strip().lower()
    if answer in {"y", "yes"}:
        return SafetyDecision(allow=True, reason="approved by human")
    return SafetyDecision(allow=False, reason="denied by human")


class ConfirmationGate:
    """Pauses for human approval before an irreversible/sensitive action."""

    def __init__(self, confirm: ConfirmCallback = cli_confirm) -> None:
        self._confirm = confirm

    async def confirm(self, action: PendingAction) -> SafetyDecision:
        return await self._confirm(action)
