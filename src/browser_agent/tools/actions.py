from browser_use import ActionResult, BrowserSession, Tools
from browser_use.browser.events import (
    ClickElementEvent,
    NavigateToUrlEvent,
    ScrollEvent,
    TypeTextEvent,
)
from pydantic import BaseModel, ConfigDict

from browser_agent.safety import PendingAction, SafetyLayer

# NOTE: do not add `from __future__ import annotations` here — browser-use's
# action registry compares the *runtime class* of `browser_session`, and PEP 563
# string annotations would break that check.

# Built-ins with no gated replacement that we drop outright. `input` is replaced
# by the gated `type_text`; `evaluate` (arbitrary JS) is an easy attack-surface
# win to remove. navigate/click/scroll are overridden in place by re-registering
# the same name below. Tightening the full toolset to exactly the Phase 3 six is
# a Phase 4 hardening step.
EXCLUDED_BUILTINS = ["input", "evaluate"]


class DoneParams(BaseModel):
    """Param model for the done action that tolerates extra fields.

    DeepSeek (and some other providers) emit ``done: {result: "...", success: true}``
    — the ``success`` field is not part of our schema.  With the default
    ``extra='forbid'`` that browser-use infers from the function signature, that
    extra field triggers a discriminated-union explosion (all 24 action variants
    fail) and the agent hits the consecutive-failure stop after 6 retries.

    ``extra='ignore'`` silently drops ``success`` (and any other extras) so the
    union resolves cleanly to the done variant.
    """

    model_config = ConfigDict(extra="ignore")
    result: str


def _node_label(node) -> str:
    try:
        text = node.get_meaningful_text_for_llm() or node.node_name or ""
    except Exception:
        import logging
        _log = logging.getLogger(__name__)
        _log.debug("Failed to extract node label", exc_info=True)
        text = ""
    return text.strip()[:120]


def _blocked(reason: str) -> ActionResult:
    return ActionResult(error=f"blocked by safety layer: {reason}")


def build_tools(safety: SafetyLayer) -> Tools:
    """Phase 3 toolset: index-based, every action routed through the gate.

    Each tool asks `safety.guard()` first; only on approval does it dispatch the
    same browser-use event the built-in action uses, so we reuse the real DOM
    pipeline instead of reimplementing it. Registering a built-in name overwrites
    it, so navigate/click/scroll become the gated versions.
    """
    tools = Tools(exclude_actions=EXCLUDED_BUILTINS)

    @tools.action("Navigate to a URL")
    async def navigate(
        url: str, browser_session: BrowserSession, new_tab: bool = False
    ) -> ActionResult:
        # `new_tab` mirrors the built-in so browser-use's initial-action/directly_open_url
        # path (which passes new_tab) validates against this tool's param model.
        decision = await safety.guard(
            PendingAction(name="navigate", params={"url": url, "new_tab": new_tab})
        )
        if not decision.allow:
            return _blocked(decision.reason)
        event = browser_session.event_bus.dispatch(NavigateToUrlEvent(url=url, new_tab=new_tab))
        await event
        await event.event_result(raise_if_any=True, raise_if_none=False)
        target = "new tab" if new_tab else "current tab"
        return ActionResult(
            extracted_content=f"navigated to {url} ({target})", include_in_memory=True
        )

    @tools.action("Click the interactive element with the given index")
    async def click(index: int, browser_session: BrowserSession) -> ActionResult:
        node = await browser_session.get_element_by_index(index)
        if node is None:
            return ActionResult(error=f"element index {index} not found")
        label = _node_label(node)
        decision = await safety.guard(
            PendingAction(name="click", params={"index": index, "element_text": label})
        )
        if not decision.allow:
            return _blocked(decision.reason)
        event = browser_session.event_bus.dispatch(ClickElementEvent(node=node))
        await event
        await event.event_result(raise_if_any=True, raise_if_none=False)
        return ActionResult(
            extracted_content=f"clicked element {index} ({label!r})", include_in_memory=True
        )

    @tools.action("Type text into the element with the given index")
    async def type_text(index: int, text: str, browser_session: BrowserSession) -> ActionResult:
        node = await browser_session.get_element_by_index(index)
        if node is None:
            return ActionResult(error=f"element index {index} not found")
        decision = await safety.guard(
            PendingAction(
                name="type_text",
                params={"index": index, "text": text, "element_text": _node_label(node)},
            )
        )
        if not decision.allow:
            return _blocked(decision.reason)
        event = browser_session.event_bus.dispatch(TypeTextEvent(node=node, text=text, clear=True))
        await event
        await event.event_result(raise_if_any=True, raise_if_none=False)
        return ActionResult(extracted_content=f"typed into element {index}", include_in_memory=True)

    @tools.action("Scroll the page up or down by roughly one viewport")
    async def scroll(
        browser_session: BrowserSession, down: bool = True, amount: int = 800
    ) -> ActionResult:
        decision = await safety.guard(
            PendingAction(name="scroll", params={"down": down, "amount": amount})
        )
        if not decision.allow:
            return _blocked(decision.reason)
        direction = "down" if down else "up"
        event = browser_session.event_bus.dispatch(
            ScrollEvent(node=None, direction=direction, amount=amount)
        )
        await event
        await event.event_result(raise_if_any=True, raise_if_none=False)
        return ActionResult(extracted_content=f"scrolled {direction} {amount}px")

    @tools.action("Extract information from the current page based on a query")
    async def extract(query: str, browser_session: BrowserSession) -> ActionResult:
        decision = await safety.guard(
            PendingAction(name="extract", params={"query": query})
        )
        if not decision.allow:
            return _blocked(decision.reason)
        state = await browser_session.get_browser_state_summary(include_screenshot=False)
        dom_text = state.dom_state.llm_representation() if state.dom_state else ""
        max_chars = safety.max_extract_chars
        return ActionResult(
            extracted_content=f"DOM content for extraction query '{query}':\n{dom_text[:max_chars]}",
            include_in_memory=True,
        )

    @tools.action(
        "Signal that the task is complete and return the final result to the user",
        param_model=DoneParams,
    )
    async def done(params: DoneParams, browser_session: BrowserSession) -> ActionResult:
        decision = await safety.guard(
            PendingAction(name="done", params={"result": params.result})
        )
        if not decision.allow:
            return _blocked(decision.reason)
        return ActionResult(
            extracted_content=params.result,
            is_done=True,
        )

    return tools
