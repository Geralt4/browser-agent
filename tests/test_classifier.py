from browser_agent.safety.classifier import is_sensitive
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
