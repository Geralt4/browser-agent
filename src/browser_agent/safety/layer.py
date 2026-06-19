from __future__ import annotations

from typing import TYPE_CHECKING

from browser_agent.config import Config
from browser_agent.safety.classifier import classify_sensitive_llm, is_sensitive
from browser_agent.safety.gate import ConfirmationGate
from browser_agent.safety.policy import check_navigation
from browser_agent.safety.types import PendingAction, SafetyDecision

if TYPE_CHECKING:
    from browser_use.llm.base import BaseChatModel


class SafetyLayer:
    """Single choke point every tool routes through before acting.

    Order: kill switch -> site policy -> sensitivity -> confirmation gate.
    When an optional LLM is provided and SENSITIVITY_LLM is enabled, the
    keyword heuristic runs first; only if it returns False does the LLM
    fallback fire (cost-conscious).
    """

    def __init__(
        self,
        cfg: Config,
        gate: ConfirmationGate | None = None,
        chat_model: BaseChatModel | None = None,
    ) -> None:
        self._cfg = cfg
        self._gate = gate or ConfirmationGate()
        self._chat_model = chat_model

    @property
    def gate(self) -> ConfirmationGate:
        """The confirmation gate this layer consults for sensitive actions.

        Read-only access for callers that need to wire the gate to a transport
        (e.g. the streaming loop sets the gate's queue). Mutating the gate's
        own state is the caller's responsibility.
        """
        return self._gate

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

        if (
            self._cfg.sensitivity_llm
            and self._chat_model is not None
            and await classify_sensitive_llm(action, self._chat_model)
        ):
            return await self._gate.confirm(action)

        return SafetyDecision(allow=True, reason="not sensitive")
