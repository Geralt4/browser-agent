from __future__ import annotations

from browser_agent.config import Config
from browser_agent.safety.classifier import is_sensitive
from browser_agent.safety.gate import ConfirmationGate
from browser_agent.safety.policy import check_navigation
from browser_agent.safety.types import PendingAction, SafetyDecision


class SafetyLayer:
    """Single choke point every tool routes through before acting.

    Order: kill switch -> site policy -> sensitivity -> confirmation gate.
    Phase 4 fills in the injection filter and category block lists; the seam is
    here so nothing downstream changes when it does.
    """

    def __init__(self, cfg: Config, gate: ConfirmationGate | None = None) -> None:
        self._cfg = cfg
        self._gate = gate or ConfirmationGate()

    async def guard(self, action: PendingAction) -> SafetyDecision:
        if self._cfg.kill_switch:
            return SafetyDecision(allow=False, reason="kill switch engaged")

        if action.name == "navigate":
            verdict = check_navigation(
                str(action.params.get("url", "")),
                self._cfg.allow_hosts,
                self._cfg.block_hosts,
            )
            if verdict is not None:
                return verdict

        if is_sensitive(action):
            return await self._gate.confirm(action)

        return SafetyDecision(allow=True, reason="not sensitive")
