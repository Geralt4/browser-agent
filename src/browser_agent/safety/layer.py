from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from browser_agent.config import Config
from browser_agent.safety.classifier import classify_sensitive_llm, is_sensitive
from browser_agent.safety.gate import ConfirmationGate
from browser_agent.safety.policy import check_navigation
from browser_agent.safety.types import PendingAction, SafetyDecision

if TYPE_CHECKING:
    from browser_use.llm.base import BaseChatModel

_log = logging.getLogger(__name__)


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

    @property
    def max_extract_chars(self) -> int:
        """Maximum characters of DOM content returned by the extract tool."""
        return self._cfg.max_extract_chars

    async def guard(self, action: PendingAction) -> SafetyDecision:
        if self._cfg.kill_switch:
            decision = SafetyDecision(allow=False, reason="kill switch engaged")
            _log.info("guard action=%s allow=%s reason=%r", action.name, decision.allow, decision.reason)
            return decision

        if action.name == "navigate":
            verdict = check_navigation(
                str(action.params.get("url", "")),
                self._cfg.allow_hosts,
                self._cfg.block_hosts,
            )
            if verdict is not None:
                _log.info("guard action=%s allow=%s reason=%r", action.name, verdict.allow, verdict.reason)
                return verdict

        needs_gate = is_sensitive(action) or (
            self._cfg.sensitivity_llm
            and self._chat_model is not None
            and await classify_sensitive_llm(action, self._chat_model)
        )
        if needs_gate:
            decision = await self._gate.confirm(action)
            # Re-check the kill switch after the await: the operator may have
            # engaged it during the (potentially long) confirmation wait, and
            # the kill switch is the one control that must be unconditional.
            # NOTE: this re-check reads self._cfg.kill_switch on the same
            # Config instance — it works for in-memory toggles (e.g. a
            # future UI button that sets cfg.kill_switch = True). It does NOT
            # see .env rewrites; Config is loaded once at construction.
            # Reactive .env reloading is addressed separately (Phase D.6, M9).
            if self._cfg.kill_switch and decision.allow:
                decision = SafetyDecision(
                    allow=False, reason="kill switch engaged during confirmation"
                )
            _log.info("guard action=%s allow=%s reason=%r", action.name, decision.allow, decision.reason)
            return decision

        decision = SafetyDecision(allow=True, reason="not sensitive")
        _log.info("guard action=%s allow=%s reason=%r", action.name, decision.allow, decision.reason)
        return decision
