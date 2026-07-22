from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from browser_use import Agent, BrowserProfile
from browser_use.llm.base import BaseChatModel

from browser_agent.agent.safe_message_manager import InjectionSafeMessageManager
from browser_agent.agent.step import AgentStep
from browser_agent.config import Config
from browser_agent.models.base import ModelAdapter
from browser_agent.perception.vision_router import resolve_use_vision, should_use_vision
from browser_agent.safety import SafetyLayer, StreamingConfirmationGate
from browser_agent.tools.actions import build_tools

_PARAM_TO_ATTR: dict[str, str] = {
    "system_message": "system_prompt",  # __init__ param ↔ instance attr
}


def _extract_step(model_output: object, n_steps: int) -> AgentStep:
    """Map a browser-use model_output onto our AgentStep schema.

    The browser-use AgentState attribute names (evaluation_previous_goal,
    next_goal, memory) are an unstable coupling point. Centralizing the
    mapping here means a browser-use rename only has to be fixed in one
    place — and tests in test_agent_step.py pin the mapping.
    """
    state = getattr(model_output, "current_state", None) if model_output else None
    actions = getattr(model_output, "action", None) if model_output else None
    if not actions:
        action_repr = ""
    else:
        action_repr = repr(actions[0])
    return AgentStep(
        step_n=n_steps,
        assessment=getattr(state, "evaluation_previous_goal", "") or "",
        memory=getattr(state, "memory", "") or "",
        next_subgoal=getattr(state, "next_goal", "") or "",
        action=action_repr,
    )


def _wrap_message_manager(agent: Agent) -> None:
    """Replace agent._message_manager with the injection-safe variant.

    Introspects MessageManager.__init__ by name so the patch survives
    browser-use upgrades that add, remove, or reorder params. The only
    manual piece is _PARAM_TO_ATTR for the one known name mismatch.

    If the patch fails (e.g. a browser-use upgrade introduces an
    incompatible parameter), a warning is logged and the agent continues
    with the original MessageManager — injection sanitization is degraded
    but the agent still works.
    """
    import inspect
    import logging

    from browser_use.agent.message_manager.service import MessageManager

    log = logging.getLogger(__name__)
    original = agent._message_manager
    sig = inspect.signature(MessageManager.__init__)

    kwargs: dict[str, Any] = {}
    for param_name in sig.parameters:
        if param_name == "self":
            continue
        attr = _PARAM_TO_ATTR.get(param_name, param_name)
        if hasattr(original, attr):
            kwargs[param_name] = getattr(original, attr)

    try:
        agent._message_manager = InjectionSafeMessageManager(**kwargs)
    except Exception:
        log.warning(
            "Failed to install InjectionSafeMessageManager — "
            "injection sanitization is disabled for this run. "
            "This usually means a browser-use upgrade changed "
            "MessageManager.__init__ parameters. "
            "Check _PARAM_TO_ATTR in agent/loop.py.",
            exc_info=True,
        )


async def run_task(
    task: str, *, cfg: Config, adapter: ModelAdapter, safety: SafetyLayer
):
    """Wire adapter + gated tools + safety into a browser-use Agent and run it.

    For UI-driven workflows, use `run_task_streaming()` instead — it accepts
    an event queue so the caller can stream step results and gate prompts.

    For benchmark use, call `run_task_with_model()` instead so the caller can
    pass a pre-wrapped (e.g. token-counting) chat model.
    """
    return await run_task_with_model(
        task,
        cfg=cfg,
        llm=adapter.chat_model(),
        supports_vision=adapter.supports_vision,
        safety=safety,
    )


class _VisionOnlyAdapter(ModelAdapter):
    """Minimal ModelAdapter shim used only to feed `resolve_use_vision`.

    `run_task_with_model` already has the chat model in hand; building a
    full ModelAdapter just to pass it through `resolve_use_vision` is
    wasteful. This shim satisfies the protocol with the one attribute the
    vision router reads (supports_vision).
    """

    name = "_benchmark_shim"
    supports_vision: bool

    def __init__(self, supports_vision: bool) -> None:
        self.supports_vision = supports_vision

    def chat_model(self) -> BaseChatModel:  # pragma: no cover - never called
        raise NotImplementedError


async def run_task_with_model(
    task: str,
    *,
    cfg: Config,
    llm: BaseChatModel,
    supports_vision: bool,
    safety: SafetyLayer,
    category: str | None = None,
    on_step_state: Callable[[object, object, int], Any] | None = None,
):
    """Run a task with a caller-supplied chat model.

    Lets the benchmark pass a TokenCountingChatModel-wrapped LLM without
    needing to subclass ModelAdapter. `supports_vision` is taken from the
    adapter rather than introspected because the wrapper is transparent.

    `category`, when provided, is forwarded to `resolve_use_vision` so
    `vision_mode="category"` can apply its data-driven routing rule.

    `on_step_state`, when provided, is registered as a
    `register_new_step_callback` on the Agent. It receives
    `(browser_state_summary, model_output, n_steps)` from browser-use on
    every step. The benchmark uses this to capture the live DOM and final
    URL *while the browser session is alive* — by the time `agent.run()`
    returns, the session is torn down and the DOM is gone.
    """
    tools = build_tools(safety)
    shim = _VisionOnlyAdapter(supports_vision=supports_vision)
    agent = Agent(
        task=task,
        llm=llm,
        tools=tools,
        browser_profile=BrowserProfile(**_browser_profile_kwargs(cfg)),
        use_vision=resolve_use_vision(cfg, shim, task, category=category),
        max_actions_per_step=1,
        use_judge=False,
        register_new_step_callback=on_step_state,
    )

    _wrap_message_manager(agent)

    return await agent.run(max_steps=cfg.max_steps)


def _browser_profile_kwargs(cfg: Config) -> dict[str, Any]:
    """Build BrowserProfile kwargs from Config.

    When `cdp_url` is set, connect to an existing browser instance via
    Chrome DevTools Protocol. browser-use's `cdp_url` parameter takes
    precedence over `headless` — setting both is harmless but cdp_url
    wins. This lets the extension user point the agent at their own
    Chrome session so the agent sees the tabs they're actually viewing.
    """
    if cfg.cdp_url:
        return {"cdp_url": cfg.cdp_url}
    return {"headless": cfg.headless}


async def run_task_streaming(
    task: str,
    *,
    cfg: Config,
    adapter: ModelAdapter,
    safety: SafetyLayer,
    queue: asyncio.Queue[dict[str, Any]] | None = None,
):
    """Run a task with streaming step events and gate prompts via an asyncio.Queue.

    The queue receives:
      {"type": "start", "message": str}
      {"type": "step", "step_n": int, "assessment": str, "memory": str, "next_subgoal": str, "action": str}
      {"type": "done", "result": str}
      {"type": "error", "message": str}

    The caller is responsible for wiring a StreamingConfirmationGate into the safety
    layer before invoking. Set gate.set_queue(queue) to enable gate prompts.
    """
    gate = safety.gate
    if queue is not None and isinstance(gate, StreamingConfirmationGate):
        gate.set_queue(queue)

    if queue is not None:
        await queue.put({"type": "start", "message": f"Starting task: {task}"})

        # Emit a vision nudge only when the task is visual but the model
        # can't deliver. Non-visual tasks don't need vision, so the nudge
        # would be noise.
        if should_use_vision(task, model_supports_vision=True) and not adapter.supports_vision:
            await queue.put({
                "type": "system",
                "message": (
                    "Vision is not enabled for this model. For tasks involving "
                    "charts, screenshots, or visual layouts, consider switching "
                    "to a vision-capable model and adding it to your vision "
                    "models list."
                ),
            })

    async def on_step(browser_state_summary, model_output, n_steps):
        if queue is None:
            return
        step = _extract_step(model_output, n_steps)
        await queue.put({"type": "step", **step.model_dump()})

    tools = build_tools(safety)
    agent = Agent(
        task=task,
        llm=adapter.chat_model(),
        tools=tools,
        browser_profile=BrowserProfile(**_browser_profile_kwargs(cfg)),
        use_vision=resolve_use_vision(cfg, adapter, task),
        max_actions_per_step=1,
        use_judge=False,
        register_new_step_callback=on_step,
    )

    _wrap_message_manager(agent)

    try:
        history = await agent.run(max_steps=cfg.max_steps)
        result = history.final_result() if hasattr(history, "final_result") else str(history)
    except asyncio.CancelledError:
        if queue is not None:
            await queue.put({"type": "error", "message": "Task cancelled"})
        raise
    except Exception as exc:
        if queue is not None:
            await queue.put({"type": "error", "message": str(exc)})
        raise
    else:
        if queue is not None:
            await queue.put({"type": "done", "result": str(result)})

    return history
