from __future__ import annotations

from browser_agent.config import Config
from browser_agent.models.base import ModelAdapter
from browser_agent.models.kimi import KimiAdapter
from browser_agent.models.openai_compat import GenericOpenAIAdapter

_ADAPTERS = {
    "kimi": KimiAdapter,
    "openai": GenericOpenAIAdapter,
}


def get_adapter(cfg: Config) -> ModelAdapter:
    provider = cfg.provider.lower()
    try:
        adapter_cls = _ADAPTERS[provider]
    except KeyError:
        raise ValueError(
            f"Unknown provider {provider!r}. Known: {sorted(_ADAPTERS)}"
        ) from None
    return adapter_cls(cfg)
