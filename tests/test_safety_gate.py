import asyncio

import pytest

from browser_agent.config import Config
from browser_agent.safety.gate import ConfirmationGate
from browser_agent.safety.layer import SafetyLayer
from browser_agent.safety.types import PendingAction, SafetyDecision

DELETE = PendingAction(name="click", params={"index": 1, "element_text": "Delete account"})
BENIGN = PendingAction(name="navigate", params={"url": "https://example.com"})
NAV_POST = PendingAction(
    name="navigate",
    params={"url": "https://httpbin.org/forms/post", "new_tab": False},
)


def _layer(approve: bool, **cfg_kw) -> tuple[SafetyLayer, list]:
    calls: list = []

    async def cb(action: PendingAction) -> SafetyDecision:
        calls.append(action)
        return SafetyDecision(allow=approve, reason="stub")

    return SafetyLayer(Config(**cfg_kw), gate=ConfirmationGate(confirm=cb)), calls


def test_benign_action_skips_gate_and_allows():
    layer, calls = _layer(approve=False)  # would deny if reached
    decision = asyncio.run(layer.guard(BENIGN))
    assert decision.allow is True
    assert calls == []


def test_sensitive_action_denied_by_gate():
    layer, calls = _layer(approve=False)
    decision = asyncio.run(layer.guard(DELETE))
    assert decision.allow is False
    assert len(calls) == 1


def test_sensitive_action_approved_by_gate():
    layer, calls = _layer(approve=True)
    decision = asyncio.run(layer.guard(DELETE))
    assert decision.allow is True
    assert len(calls) == 1


def test_kill_switch_blocks_everything():
    layer, _ = _layer(approve=True, kill_switch=True)
    decision = asyncio.run(layer.guard(BENIGN))
    assert decision.allow is False
    assert "kill switch" in decision.reason


def test_blocklist_blocks_navigation():
    layer, _ = _layer(approve=True, blocklist="evil.com")
    decision = asyncio.run(
        layer.guard(PendingAction(name="navigate", params={"url": "https://www.evil.com/x"}))
    )
    assert decision.allow is False


def test_navigate_with_post_in_url_not_gated():
    """Navigation to a URL containing 'post' must not trigger the keyword
    classifier — the allow/block list is the right gate for navigation."""
    layer, calls = _layer(approve=False)  # would deny if gate reached
    decision = asyncio.run(layer.guard(NAV_POST))
    assert decision.allow is True
    assert calls == []  # gate was never consulted


def test_streaming_gate_cleans_up_on_cancel():
    """A cancelled confirm() must not leave an entry in _events."""
    from browser_agent.safety.gate import StreamingConfirmationGate

    gate = StreamingConfirmationGate()
    action = PendingAction(name="click", params={"index": 1, "element_text": "Delete"})

    async def run():
        task = asyncio.create_task(gate.confirm(action))
        await asyncio.sleep(0.01)  # let it register the entry
        assert len(gate._events) == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(gate._events) == 0  # cleaned up, no leak

    asyncio.run(run())
