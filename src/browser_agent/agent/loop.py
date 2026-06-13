from __future__ import annotations

from browser_use import Agent, BrowserProfile

from browser_agent.config import Config
from browser_agent.models.base import ModelAdapter
from browser_agent.safety import SafetyLayer
from browser_agent.tools.actions import build_tools


async def run_task(
    task: str, *, cfg: Config, adapter: ModelAdapter, safety: SafetyLayer
):
    """Wire adapter + gated tools + safety into a browser-use Agent and run it."""
    tools = build_tools(safety)
    agent = Agent(
        task=task,
        llm=adapter.chat_model(),
        tools=tools,
        browser_profile=BrowserProfile(headless=cfg.headless),
        use_vision=False,  # DOM-first per the brief; vision routing is Phase 2.
        max_actions_per_step=1,  # one gated action per step keeps the gate auditable.
        use_judge=False,  # optional LLM-as-judge post-trace; off (extra call + K2.7 fences its JSON).
    )
    return await agent.run(max_steps=cfg.max_steps)
