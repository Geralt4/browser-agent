from browser_agent.safety.classifier import classify_sensitive_llm, is_sensitive
from browser_agent.safety.types import PendingAction


def test_benign_navigation_not_sensitive():
    action = PendingAction(name="navigate", params={"url": "https://example.com"})
    assert is_sensitive(action) is False


def test_read_more_link_not_sensitive():
    action = PendingAction(name="click", params={"index": 3, "element_text": "Read more"})
    assert is_sensitive(action) is False


def test_delete_button_sensitive():
    action = PendingAction(name="click", params={"index": 3, "element_text": "Delete account"})
    assert is_sensitive(action) is True


def test_buy_button_sensitive():
    action = PendingAction(name="click", params={"index": 2, "element_text": "Buy now"})
    assert is_sensitive(action) is True


def test_typing_credit_card_sensitive():
    action = PendingAction(name="type_text", params={"index": 1, "text": "4111 1111 1111 1111"})
    assert is_sensitive(action) is True


def test_typing_plain_text_not_sensitive():
    action = PendingAction(name="type_text", params={"index": 1, "text": "hello world"})
    assert is_sensitive(action) is False


def test_post_button_sensitive():
    action = PendingAction(name="click", params={"index": 1, "element_text": "Post"})
    assert is_sensitive(action) is True


def test_post_at_end_of_string_sensitive():
    action = PendingAction(name="click", params={"index": 1, "element_text": "Post"})
    assert is_sensitive(action) is True


# --- LLM classifier tests (mocked) ---


def test_llm_flags_sensitive_yes():
    import asyncio

    async def run():
        class FakeModel:
            async def ainvoke(self, messages, output_format=None, **kwargs):
                return "YES"

        action = PendingAction(name="click", params={"element_text": "Revoke all access keys"})
        return await classify_sensitive_llm(action, FakeModel())

    assert asyncio.run(run()) is True


def test_llm_returns_no_for_benign():
    import asyncio

    async def run():
        class FakeModel:
            async def ainvoke(self, messages, output_format=None, **kwargs):
                return "NO"

        action = PendingAction(name="navigate", params={"url": "https://example.com"})
        return await classify_sensitive_llm(action, FakeModel())

    assert asyncio.run(run()) is False


def test_llm_errors_return_none():
    import asyncio

    async def run():
        class BrokenModel:
            async def ainvoke(self, messages, output_format=None, **kwargs):
                raise RuntimeError("api down")

        action = PendingAction(name="click", params={"element_text": "Read more"})
        return await classify_sensitive_llm(action, BrokenModel())

    assert asyncio.run(run()) is None
