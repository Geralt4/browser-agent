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


def test_done_params_tolerates_deepseek_extra_success():
    """DeepSeek emits done: {result: ..., success: true} — the extra success
    field must not trigger a union explosion."""
    from browser_agent.tools.actions import DoneParams

    d = DoneParams.model_validate({"result": "The page title is httpbin.org.", "success": True})
    assert d.result == "The page title is httpbin.org."


def test_done_action_registry_accepts_deepseek_payload():
    """The full action registry must validate and execute a done action with
    the DeepSeek-shaped payload."""
    from browser_agent.config import Config
    from browser_agent.safety import SafetyLayer
    from browser_agent.tools.actions import build_tools

    registry = build_tools(SafetyLayer(Config())).registry
    am = registry.create_action_model()
    import typing

    args = typing.get_args(am.model_fields["root"].annotation)
    done_variant = next(a for a in args if "Done" in a.__name__)
    out = done_variant.model_validate({"done": {"result": "ok", "success": True}})
    assert out.done.result == "ok"


def test_build_tools_exposes_only_gated_actions():
    """Only the six gated actions (click, done, extract, navigate, scroll,
    type_text) are exposed. Every other browser-use built-in is excluded —
    otherwise un-gated actions like `search`, `send_keys`, `write_file` would
    bypass SafetyLayer.guard() entirely (e.g. `search` would skip the
    navigation blocklist, `send_keys` would skip the password-field check)."""
    from browser_agent.config import Config
    from browser_agent.safety import SafetyLayer
    from browser_agent.tools.actions import _GATED, build_tools

    tools = build_tools(SafetyLayer(Config()))
    registered = set(tools.registry.registry.actions.keys())
    assert registered == _GATED, (
        f"Registered actions mismatch: extra={registered - _GATED} "
        f"missing={_GATED - registered}"
    )


def test_search_action_is_excluded():
    """`search` would dispatch NavigateToUrlEvent un-gated and skip the
    navigation blocklist. It must not be in the registered set."""
    from browser_agent.config import Config
    from browser_agent.safety import SafetyLayer
    from browser_agent.tools.actions import build_tools

    tools = build_tools(SafetyLayer(Config()))
    assert "search" not in tools.registry.registry.actions


def test_write_file_action_is_excluded():
    """`write_file` would mutate the filesystem with no sensitivity gate. It
    must not be in the registered set."""
    from browser_agent.config import Config
    from browser_agent.safety import SafetyLayer
    from browser_agent.tools.actions import build_tools

    tools = build_tools(SafetyLayer(Config()))
    assert "write_file" not in tools.registry.registry.actions
    assert "replace_file" not in tools.registry.registry.actions
    assert "read_file" not in tools.registry.registry.actions


def test_send_keys_action_is_excluded():
    """`send_keys` would dispatch keyboard input un-gated and skip the
    password-field check. It must not be in the registered set."""
    from browser_agent.config import Config
    from browser_agent.safety import SafetyLayer
    from browser_agent.tools.actions import build_tools

    tools = build_tools(SafetyLayer(Config()))
    assert "send_keys" not in tools.registry.registry.actions
