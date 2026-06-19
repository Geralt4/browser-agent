from __future__ import annotations

import asyncio
from typing import Any

from browser_use import Agent, BrowserProfile

from browser_agent.agent.safe_message_manager import InjectionSafeMessageManager
from browser_agent.config import Config
from browser_agent.models.base import ModelAdapter
from browser_agent.perception.vision_router import resolve_use_vision, should_use_vision
from browser_agent.safety import SafetyLayer, StreamingConfirmationGate
from browser_agent.tools.actions import build_tools

_PARAM_TO_ATTR: dict[str, str] = {
    "system_message": "system_prompt",  # __init__ param ↔ instance attr
}


def _wrap_message_manager(agent: Agent) -> None:
    """Replace agent._message_manager with the injection-safe variant.

    Introspects MessageManager.__init__ by name so the patch survives
    browser-use upgrades that add, remove, or reorder params. The only
    manual piece is _PARAM_TO_ATTR for the one known name mismatch.
    """
    import inspect

    from browser_use.agent.message_manager.service import MessageManager

    original = agent._message_manager
    sig = inspect.signature(MessageManager.__init__)

    kwargs: dict[str, Any] = {}
    for param_name in sig.parameters:
        if param_name == "self":
            continue
        attr = _PARAM_TO_ATTR.get(param_name, param_name)
        if hasattr(original, attr):
            kwargs[param_name] = getattr(original, attr)

    agent._message_manager = InjectionSafeMessageManager(**kwargs)


async def run_task(
    task: str, *, cfg: Config, adapter: ModelAdapter, safety: SafetyLayer
):
    """Wire adapter + gated tools + safety into a browser-use Agent and run it.

    For UI-driven workflows, use `run_task_streaming()` instead — it accepts
    an event queue so the caller can stream step results and gate prompts.
    """
    tools = build_tools(safety)
    agent = Agent(
        task=task,
        llm=adapter.chat_model(),
        tools=tools,
        browser_profile=BrowserProfile(headless=cfg.headless),
        use_vision=resolve_use_vision(cfg, adapter, task),
        max_actions_per_step=1,
        use_judge=False,
    )

    _wrap_message_manager(agent)

    return await agent.run(max_steps=cfg.max_steps)


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
    gate = safety._gate
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
        state = model_output.current_state if model_output else None
        actions = model_output.action if model_output else []
        action_repr = repr(actions[0]) if actions else ""
        queue.put_nowait({
            "type": "step",
            "step_n": n_steps,
            "assessment": getattr(state, "evaluation_previous_goal", "") if state else "",
            "memory": getattr(state, "memory", "") if state else "",
            "next_subgoal": getattr(state, "next_goal", "") if state else "",
            "action": action_repr,
        })

    tools = build_tools(safety)
    agent = Agent(
        task=task,
        llm=adapter.chat_model(),
        tools=tools,
        browser_profile=BrowserProfile(headless=cfg.headless),
        use_vision=resolve_use_vision(cfg, adapter, task),
        max_actions_per_step=1,
        use_judge=False,
        register_new_step_callback=on_step,
    )

    _wrap_message_manager(agent)

    try:
        history = await agent.run(max_steps=cfg.max_steps)
        result = history.final_result() if hasattr(history, "final_result") else str(history)
    except Exception as exc:
        if queue is not None:
            await queue.put({"type": "error", "message": str(exc)})
        raise
    else:
        if queue is not None:
            await queue.put({"type": "done", "result": str(result)})

    return history
