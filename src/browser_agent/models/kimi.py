from __future__ import annotations

from browser_use import ChatOpenAI
from browser_use.llm.base import BaseChatModel

from browser_agent.config import Config
from browser_agent.models.base import ModelAdapter

MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"
KIMI_MODEL = "kimi-k2.6"


class KimiAdapter(ModelAdapter):
    """Kimi K2.6 via Moonshot's OpenAI-compatible endpoint.

    First-class target model: strong agentic tool-calling plus integrated
    vision, so a single model covers DOM reasoning and the occasional
    screenshot.
    """

    name = KIMI_MODEL
    supports_vision = True

    def __init__(self, cfg: Config) -> None:
        if not cfg.moonshot_api_key:
            raise ValueError(
                "MOONSHOT_API_KEY is not set. Add it to .env or switch "
                "PROVIDER to a different provider."
            )
        self._cfg = cfg

    def chat_model(self) -> BaseChatModel:
        return ChatOpenAI(
            model=KIMI_MODEL,
            base_url=MOONSHOT_BASE_URL,
            api_key=self._cfg.moonshot_api_key,
            # Compatibility flags for non-OpenAI structured output. Kimi's
            # OpenAI-compatible endpoint rejects the strict response_format
            # schema browser-use sends by default, so we keep the schema in
            # the system prompt and skip forcing structured output.
            add_schema_to_system_prompt=True,
            remove_min_items_from_schema=True,
            dont_force_structured_output=True,
        )
