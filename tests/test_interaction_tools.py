import asyncio

from browser_use import BrowserProfile, BrowserSession

from browser_agent.config import Config
from browser_agent.safety import SafetyLayer
from browser_agent.safety.gate import ConfirmationGate
from browser_agent.safety.types import SafetyDecision
from browser_agent.tools.actions import build_tools


async def _js(session: BrowserSession, expr: str):
    cdp = await session.get_or_create_cdp_session()
    result = await cdp.cdp_client.send.Runtime.evaluate(
        params={"expression": expr, "returnByValue": True, "awaitPromise": True},
        session_id=cdp.session_id,
    )
    return result["result"].get("value")


async def _run_interaction(url: str) -> dict:
    """Drive the gated tools directly (no LLM), as the agent's registry does."""
    registry = build_tools(SafetyLayer(Config())).registry
    session = BrowserSession(browser_profile=BrowserProfile(headless=True))
    await session.start()
    try:
        await registry.execute_action("navigate", {"url": url}, browser_session=session)

        # Populate the index -> element selector map, then resolve by tag.
        await session.get_browser_state_summary(include_screenshot=False)
        selector_map = await session.get_selector_map()
        indices: dict[str, int] = {}
        for index, node in selector_map.items():
            tag = (getattr(node, "tag_name", "") or "").lower()
            if tag == "input":
                indices.setdefault("input", index)
            elif tag == "button":
                indices.setdefault("button", index)

        type_res = await registry.execute_action(
            "type_text", {"index": indices["input"], "text": "hello"}, browser_session=session
        )
        input_value = await _js(session, "document.getElementById('q').value")

        click_res = await registry.execute_action(
            "click", {"index": indices["button"]}, browser_session=session
        )
        out_text = await _js(session, "document.getElementById('out').textContent")

        scroll_res = await registry.execute_action(
            "scroll", {"down": True, "amount": 200}, browser_session=session
        )

        return {
            "indices": indices,
            "input_value": input_value,
            "out_text": out_text,
            "type_error": type_res.error,
            "click_error": click_res.error,
            "scroll_error": scroll_res.error,
        }
    finally:
        await session.kill()


def test_gated_type_and_click_drive_the_page(fixture_url):
    """Phase 1 interaction smoke: proves click/type/scroll + element indexing."""
    result = asyncio.run(_run_interaction(fixture_url("search.html")))

    # Element-index behavior: distinct indices resolved for the input and button.
    assert "input" in result["indices"] and "button" in result["indices"]
    assert result["indices"]["input"] != result["indices"]["button"]

    # No wrapped tool reported an error.
    assert result["type_error"] is None
    assert result["click_error"] is None
    assert result["scroll_error"] is None

    # Type landed in the field; click fired the page's handler.
    assert result["input_value"] == "hello"
    assert result["out_text"] == "searched:hello"


async def _run_blocked_destructive(url: str) -> dict:
    async def deny(action):
        return SafetyDecision(allow=False, reason="user denied (test)")

    safety = SafetyLayer(Config(), gate=ConfirmationGate(confirm=deny))
    registry = build_tools(safety).registry
    session = BrowserSession(browser_profile=BrowserProfile(headless=True))
    await session.start()
    try:
        await registry.execute_action("navigate", {"url": url}, browser_session=session)
        await session.get_browser_state_summary(include_screenshot=False)
        selector_map = await session.get_selector_map()
        button = next(
            i
            for i, node in selector_map.items()
            if (getattr(node, "tag_name", "") or "").lower() == "button"
        )
        result = await registry.execute_action("click", {"index": button}, browser_session=session)
        out_text = await _js(session, "document.getElementById('out').textContent")
        return {"error": result.error, "out_text": out_text}
    finally:
        await session.kill()


def test_gate_blocks_destructive_click(fixture_url):
    """A denied sensitive click must not reach the DOM."""
    result = asyncio.run(_run_blocked_destructive(fixture_url("destructive.html")))

    assert result["error"] is not None
    assert "blocked by safety layer" in result["error"]
    # The page's onclick never fired -> the destructive effect was prevented.
    assert result["out_text"] == "safe"
