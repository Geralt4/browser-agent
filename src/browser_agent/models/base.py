from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from browser_use.llm.base import BaseChatModel

if TYPE_CHECKING:
    from browser_agent.models.token_counter import TokenTotals


class ModelAdapter(ABC):
    """Provider-agnostic wrapper around a browser-use chat model.

    Centralizes per-model quirks (compat flags, vision capability, model id,
    base_url) so swapping providers is a config change, not a code change.
    """

    name: str
    supports_vision: bool

    @abstractmethod
    def chat_model(self) -> BaseChatModel:
        """Return a browser-use chat model ready to pass to Agent(llm=...)."""
        raise NotImplementedError

    def with_token_counter(self) -> tuple[BaseChatModel, "TokenTotals"]:  # noqa: UP037
        """Wrap the chat model in a token-counting proxy.

        Default implementation: returns a TokenCountingChatModel wrapping
        `chat_model()` and a reference to the wrapper's internal `totals`
        bag. Subclasses may override if they need to share state across
        multiple chat models.

        `TokenTotals` is imported lazily to avoid forcing benchmark-only
        imports on the hot path of normal agent runs. The quoted
        annotation keeps ruff from requiring a top-level import.
        """
        from browser_agent.models.token_counter import TokenCountingChatModel

        wrapper = TokenCountingChatModel(self.chat_model())
        return wrapper, wrapper.totals
