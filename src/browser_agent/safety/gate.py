from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from browser_agent.safety.types import PendingAction, SafetyDecision

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


class StreamingConfirmationGate(ConfirmationGate):
    """Pauses for human approval via an asyncio.Queue for UI-driven workflows.

    The queue is polled by the SSE endpoint; the gate waits on an internal
    event that fires when the user clicks approve/deny in the UI.
    """

    def __init__(self, timeout: float = 300.0) -> None:
        super().__init__(confirm=self._streaming_confirm)
        self._events: dict[str, tuple[asyncio.Event, bool]] = {}
        self._pending_queue: asyncio.Queue[dict[str, Any]] | None = None
        self._timeout = timeout

    def set_queue(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._pending_queue = queue

    async def _streaming_confirm(self, action: PendingAction) -> SafetyDecision:
        gate_id = uuid.uuid4().hex[:8]
        payload = {
            "type": "gate",
            "gate_id": gate_id,
            "name": action.name,
            "params": action.params,
            "summary": action.summary(),
        }

        if self._pending_queue is not None:
            await self._pending_queue.put(payload)

        event = asyncio.Event()
        self._events[gate_id] = (event, False)

        try:
            await asyncio.wait_for(event.wait(), timeout=self._timeout)
        except TimeoutError:
            self._events.pop(gate_id, None)
            return SafetyDecision(allow=False, reason="gate timed out — denied by default")
        except asyncio.CancelledError:
            # The streaming task was cancelled (client disconnect, orphan
            # cleanup) while waiting for approval. Drop the entry so it can't
            # leak; a late resolve() would otherwise no-op on a dead task.
            self._events.pop(gate_id, None)
            raise

        _, allowed = self._events.pop(gate_id, (None, False))
        if allowed:
            return SafetyDecision(allow=True, reason="approved by human")
        return SafetyDecision(allow=False, reason="denied by human")

    def resolve(self, gate_id: str, allow: bool) -> bool:
        entry = self._events.get(gate_id)
        if entry is None:
            return False
        event, _ = entry
        self._events[gate_id] = (event, allow)
        event.set()
        return True
