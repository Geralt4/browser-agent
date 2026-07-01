from browser_agent.safety.classifier import classify_sensitive_llm, is_sensitive
from browser_agent.safety.types import PendingAction


def test_benign_navigation_not_sensitive():
    action = PendingAction(name="navigate", params={"url": "https://example.com"})
    assert is_sensitive(action) is False


def test_navigate_url_with_post_keyword_not_sensitive():
    """URLs containing 'post' in the path must not trigger the keyword
    classifier — navigation is not a sensitive intent."""
    action = PendingAction(
        name="navigate",
        params={"url": "https://httpbin.org/forms/post", "new_tab": False},
    )
    assert is_sensitive(action) is False


def test_navigate_url_with_delete_keyword_not_sensitive():
    """Same for 'delete' in a URL path."""
    action = PendingAction(
        name="navigate",
        params={"url": "https://example.com/delete/account", "new_tab": False},
    )
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


# --- Word-boundary regression tests (false positives the old substring
#     matcher would have produced). ---


def test_poster_link_not_sensitive():
    """"poster" contains "post" as a substring, but the word-boundary matcher
    must not flag it. This is the canonical false-positive the regex refactor
    exists to fix."""
    action = PendingAction(name="click", params={"index": 4, "element_text": "View poster"})
    assert is_sensitive(action) is False


def test_buyer_label_not_sensitive():
    """"buyer" contains "buy" as a substring but is a noun, not the buy intent."""
    action = PendingAction(name="click", params={"index": 1, "element_text": "Show buyer profile"})
    assert is_sensitive(action) is False


def test_sending_word_not_sensitive():
    """"sending" as part of a label like "Currently sending..." should NOT match
    the "send" intent — the agent isn't initiating a send, it's observing one.
    The word-boundary regex matches "send" only as a whole word, so the
    gerund form "sending" is excluded. (If the element text were exactly
    "Send" or "Send email", it would still match — see test_send_button.)"""
    action = PendingAction(name="extract", params={"query": "Currently sending 3 items"})
    assert is_sensitive(action) is False


def test_send_button_sensitive():
    action = PendingAction(name="click", params={"index": 2, "element_text": "Send"})
    assert is_sensitive(action) is True


def test_reset_button_sensitive():
    action = PendingAction(name="click", params={"index": 5, "element_text": "Reset password"})
    assert is_sensitive(action) is True


def test_revoke_button_sensitive():
    action = PendingAction(name="click", params={"index": 1, "element_text": "Revoke access"})
    assert is_sensitive(action) is True


def test_unsubscribe_button_sensitive():
    action = PendingAction(name="click", params={"index": 3, "element_text": "Unsubscribe"})
    assert is_sensitive(action) is True


def test_donate_button_sensitive():
    action = PendingAction(name="click", params={"index": 1, "element_text": "Donate now"})
    assert is_sensitive(action) is True


def test_export_button_sensitive():
    action = PendingAction(name="click", params={"index": 2, "element_text": "Export data"})
    assert is_sensitive(action) is True


def test_disconnect_button_sensitive():
    action = PendingAction(name="click", params={"index": 4, "element_text": "Disconnect wallet"})
    assert is_sensitive(action) is True


def test_withdraw_button_sensitive():
    action = PendingAction(name="click", params={"index": 1, "element_text": "Withdraw funds"})
    assert is_sensitive(action) is True


def test_close_account_button_sensitive():
    action = PendingAction(name="click", params={"index": 1, "element_text": "Close account"})
    assert is_sensitive(action) is True


def test_close_account_with_extra_whitespace_sensitive():
    """Multi-word keywords tolerate arbitrary whitespace between words."""
    action = PendingAction(name="click", params={"index": 1, "element_text": "Close    account"})
    assert is_sensitive(action) is True


def test_place_order_button_sensitive():
    action = PendingAction(name="click", params={"index": 1, "element_text": "Place order"})
    assert is_sensitive(action) is True


def test_benign_postage_label_not_sensitive():
    """"postage" contains "post" but is a different word."""
    action = PendingAction(name="click", params={"index": 7, "element_text": "Calculate postage"})
    assert is_sensitive(action) is False


# --- Password-field detection ---


def test_typing_into_password_field_sensitive():
    """Typing *anything* into a field labeled 'password' is sensitive —
    the content is a secret regardless of what the agent thinks it is."""
    action = PendingAction(
        name="type_text",
        params={"index": 1, "text": "hello123", "element_text": "Password"},
    )
    assert is_sensitive(action) is True


def test_typing_into_pwd_field_sensitive():
    action = PendingAction(
        name="type_text",
        params={"index": 1, "text": "abc", "element_text": "PWD"},
    )
    assert is_sensitive(action) is True


def test_typing_into_api_key_field_sensitive():
    action = PendingAction(
        name="type_text",
        params={"index": 1, "text": "sk-...", "element_text": "API key"},
    )
    assert is_sensitive(action) is True


def test_typing_into_token_field_sensitive():
    action = PendingAction(
        name="type_text",
        params={"index": 1, "text": "xyz", "element_text": "2FA token"},
    )
    assert is_sensitive(action) is True


def test_typing_into_unrelated_field_with_passwordish_text_not_sensitive():
    """The password check is on the *field label*, not the text content —
    typing the literal string "password" into a comment box is not sensitive
    on its own (no keyword match, no PII regex match)."""
    action = PendingAction(
        name="type_text",
        params={"index": 1, "text": "my password is secret", "element_text": "Comment"},
    )
    assert is_sensitive(action) is False


def test_typing_email_into_unlabeled_field_sensitive():
    """Email regex still fires on type_text without a label."""
    action = PendingAction(
        name="type_text",
        params={"index": 1, "text": "user@example.com", "element_text": ""},
    )
    assert is_sensitive(action) is True
