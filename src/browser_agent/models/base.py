from __future__ import annotations

from abc import ABC, abstractmethod

from browser_use.llm.base import BaseChatModel


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
