"""Tests for the vision nudge system event in run_task_streaming.

We don't spin up a real browser here — we test that the queue receives a
`{"type": "system", "message": "..."}` event early on when the model does
NOT support vision. The actual agent.run() would require a real LLM, so we
patch it to put a "done" event on the queue immediately and then exit.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

from browser_agent.agent.loop import run_task_streaming
from browser_agent.config import Config
from browser_agent.safety import SafetyLayer, StreamingConfirmationGate


class _FakeAdapter:
    name = "gpt-3.5-turbo"
    supports_vision = False

    def chat_model(self):
        return MagicMock()


class _FakeHistory:
    def final_result(self):
        return "ok"


def _make_agent_factory():
    """Return a function that patches Agent to a stub that runs to completion
    immediately, putting a done event on the queue."""
    def factory(task, llm, tools, browser_profile, use_vision, max_actions_per_step, use_judge, register_new_step_callback=None):
        agent = MagicMock()
        agent.register_new_step_callback = register_new_step_callback

        async def fake_run(max_steps):
            # Mirror the real loop: emit "done" when finished
            return _FakeHistory()

        agent.run = fake_run

        # Stash the real message manager so the loop's monkey-patch doesn't fail
        mm = MagicMock()
        mm.task = task
        mm.system_prompt = "you are an agent"
        mm.file_system = None
        mm.state = MagicMock()
        mm.use_thinking = False
        mm.include_attributes = []
        mm.sensitive_data = None
        mm.max_history_items = None
        mm.vision_detail_level = None
        mm.include_tool_call_examples = False
        mm.include_recent_events = True
        mm.sample_images = None
        mm.llm_screenshot_size = None
        mm.max_clickable_elements_length = None
        agent._message_manager = mm
        return agent

    return factory


class TestVisionNudge:
    @patch("browser_agent.agent.loop.InjectionSafeMessageManager")
    @patch("browser_agent.agent.loop.Agent", new=_make_agent_factory())
    @patch("browser_agent.agent.loop.build_tools", return_value=[])
    def test_nudge_emitted_when_visual_task_and_no_vision(self, _bt, _ism):
        # Task is visual, model can't deliver → nudge
        cfg = Config(llm_model="gpt-3.5-turbo", llm_api_key="sk", vision_mode="auto", vision_models="")
        adapter = _FakeAdapter()  # supports_vision = False
        safety = SafetyLayer(cfg, gate=StreamingConfirmationGate())
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        asyncio.run(run_task_streaming("describe the chart on this page", cfg=cfg, adapter=adapter, safety=safety, queue=queue))

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        types = [e["type"] for e in events]
        assert "system" in types
        nudge = next(e for e in events if e["type"] == "system")
        assert "Vision is not enabled" in nudge["message"]

    @patch("browser_agent.agent.loop.InjectionSafeMessageManager")
    @patch("browser_agent.agent.loop.Agent", new=_make_agent_factory())
    @patch("browser_agent.agent.loop.build_tools", return_value=[])
    def test_no_nudge_when_non_visual_task(self, _bt, _ism):
        # Non-visual task: no nudge even though model can't do vision
        cfg = Config(llm_model="gpt-3.5-turbo", llm_api_key="sk", vision_mode="auto", vision_models="")
        adapter = _FakeAdapter()  # supports_vision = False
        safety = SafetyLayer(cfg, gate=StreamingConfirmationGate())
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        asyncio.run(run_task_streaming("go to example.com", cfg=cfg, adapter=adapter, safety=safety, queue=queue))

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        types = [e["type"] for e in events]
        assert "system" not in types

    @patch("browser_agent.agent.loop.InjectionSafeMessageManager")
    @patch("browser_agent.agent.loop.Agent", new=_make_agent_factory())
    @patch("browser_agent.agent.loop.build_tools", return_value=[])
    def test_no_nudge_when_model_supports_vision(self, _bt, _ism):
        # Visual task + model supports vision → no nudge
        class VAdapter:
            name = "gpt-4o"
            supports_vision = True
            def chat_model(self): return MagicMock()

        cfg = Config(llm_model="gpt-4o", llm_api_key="sk", vision_mode="auto", vision_models="gpt-4o")
        adapter = VAdapter()
        safety = SafetyLayer(cfg, gate=StreamingConfirmationGate())
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        asyncio.run(run_task_streaming("describe the chart", cfg=cfg, adapter=adapter, safety=safety, queue=queue))

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        types = [e["type"] for e in events]
        assert "system" not in types
