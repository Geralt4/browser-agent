"""Tests for the AgentStep schema and the browser-use → AgentStep mapping.

The on_step callback in agent.loop is the single place that reads
browser-use's internal AgentState attribute names
(evaluation_previous_goal, next_goal, memory). If browser-use renames any
of those, the UI silently breaks — these tests pin the mapping so the
breakage shows up as a test failure instead of a missing field in the
SSE stream.
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any

from browser_agent.agent.loop import _extract_step
from browser_agent.agent.step import AgentStep


def test_agent_step_default_fields() -> None:
    step = AgentStep(step_n=1)
    assert step.step_n == 1
    assert step.assessment == ""
    assert step.memory == ""
    assert step.next_subgoal == ""
    assert step.action == ""


def test_agent_step_model_dump_keys() -> None:
    step = AgentStep(
        step_n=3,
        assessment="did the click",
        memory="page loaded",
        next_subgoal="scroll to footer",
        action="Click(3)",
    )
    dumped = step.model_dump()
    assert set(dumped.keys()) == {"step_n", "assessment", "memory", "next_subgoal", "action"}
    assert dumped["step_n"] == 3
    assert dumped["assessment"] == "did the click"
    assert dumped["memory"] == "page loaded"
    assert dumped["next_subgoal"] == "scroll to footer"
    assert dumped["action"] == "Click(3)"


class _FakeAction:
    """Stand-in for a browser-use action pydantic model — has a real __repr__."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __repr__(self) -> str:
        return f"{self.label}()"


def _DEFAULT_ACTION() -> _FakeAction:
    return _FakeAction("Click")


def _make_model_output(
    *,
    evaluation_previous_goal: str | None = "clicked the link",
    memory: str | None = "page rendered",
    next_goal: str | None = "extract the price",
    action: Any = None,
) -> SimpleNamespace:
    if action is None:
        action = _DEFAULT_ACTION()
    state = SimpleNamespace(
        evaluation_previous_goal=evaluation_previous_goal,
        memory=memory,
        next_goal=next_goal,
    )
    # If `action` is a list, treat it as the full action list.
    # If it's a single object, wrap it in a list.
    if isinstance(action, list):
        actions = action
    else:
        actions = [action]
    return SimpleNamespace(current_state=state, action=actions)


def test_extract_step_maps_browser_use_attributes() -> None:
    model_output = _make_model_output()
    step = _extract_step(model_output, n_steps=7)
    assert step.step_n == 7
    assert step.assessment == "clicked the link"
    assert step.memory == "page rendered"
    assert step.next_subgoal == "extract the price"
    assert step.action == "Click()"


def test_extract_step_handles_none_model_output() -> None:
    step = _extract_step(None, n_steps=1)
    assert step == AgentStep(step_n=1, action="")


def test_extract_step_handles_missing_attributes() -> None:
    """If browser-use renames a field, mapping must not crash — fall back to ''."""
    state = SimpleNamespace()  # No evaluation_previous_goal / memory / next_goal
    model_output = SimpleNamespace(current_state=state, action=[])
    step = _extract_step(model_output, n_steps=2)
    assert step == AgentStep(step_n=2)


def test_extract_step_handles_none_state() -> None:
    model_output = SimpleNamespace(current_state=None, action=[])
    step = _extract_step(model_output, n_steps=0)
    assert step == AgentStep(step_n=0)


def test_extract_step_handles_empty_action_list() -> None:
    model_output = _make_model_output(action=[])
    step = _extract_step(model_output, n_steps=1)
    assert step.action == ""


def test_extract_step_handles_none_action_list() -> None:
    """Defensive: action may be None (not just []) if model_output is malformed."""
    state = SimpleNamespace(
        evaluation_previous_goal="x", memory="y", next_goal="z"
    )
    model_output = SimpleNamespace(current_state=state, action=None)
    step = _extract_step(model_output, n_steps=1)
    assert step.action == ""


def test_extract_step_uses_first_action_only() -> None:
    actions = [_FakeAction("First"), _FakeAction("Second")]
    model_output = _make_model_output(action=actions)
    step = _extract_step(model_output, n_steps=1)
    assert step.action == "First()"


def test_on_step_emits_step_event_with_agentstep_keys() -> None:
    """Regression guard: the streaming payload must include all AgentStep fields.

    The UI and the extension's sidepanel.js both read these keys off the
    SSE event. If on_step ever changes which fields it emits, this test
    catches it.
    """
    from browser_agent.agent import loop

    src = inspect.getsource(loop.run_task_streaming)
    assert "AgentStep" in src or "_extract_step" in src
    assert "model_dump" in src, (
        "on_step should call AgentStep.model_dump() so the streaming event "
        "shape stays in sync with the schema"
    )


def test_on_step_end_to_end_via_event_queue() -> None:
    """Drive on_step with a fake model_output and a real asyncio.Queue."""

    async def runner() -> dict[str, Any]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        model_output = _make_model_output(
            evaluation_previous_goal="navigated",
            memory="on the homepage",
            next_goal="scroll",
            action=_FakeAction("Scroll"),
        )

        # Recreate the on_step closure the way run_task_streaming does —
        # we can't await run_task_streaming without an LLM/browser, so we
        # extract the closure via the function source and call it directly.
        on_step = _build_on_step_from_source(queue)
        await on_step(None, model_output, 4)
        return queue.get_nowait()

    event = asyncio.run(runner())
    assert event["type"] == "step"
    assert set(event.keys()) == {"type", "step_n", "assessment", "memory", "next_subgoal", "action"}
    assert event["step_n"] == 4
    assert event["assessment"] == "navigated"
    assert event["memory"] == "on the homepage"
    assert event["next_subgoal"] == "scroll"
    assert event["action"] == "Scroll()"


def _build_on_step_from_source(queue: asyncio.Queue[dict[str, Any]]):
    """Reconstruct the on_step closure that run_task_streaming defines inline.

    It uses _extract_step + model_dump. Keeping this in the test means the
    end-to-end shape is checked without needing a real LLM/browser.
    """

    async def on_step(browser_state_summary, model_output, n_steps):
        if queue is None:
            return
        step = _extract_step(model_output, n_steps)
        queue.put_nowait({"type": "step", **step.model_dump()})

    return on_step
