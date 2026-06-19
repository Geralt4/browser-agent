from __future__ import annotations

from browser_use import ChatOpenAI
from browser_use.llm.base import BaseChatModel

from browser_agent.config import Config
from browser_agent.models.base import ModelAdapter


class GenericOpenAIAdapter(ModelAdapter):
    """Any OpenAI-compatible endpoint, configured entirely from env.

    Used to prove the agent loop before the Kimi swap; identical wiring, so
    moving to Kimi is just a provider/config change.
    """

    def __init__(self, cfg: Config) -> None:
        if not cfg.llm_model:
            raise ValueError("LLM_MODEL is not set for the 'openai' provider.")
        if not cfg.llm_api_key:
            raise ValueError("LLM_API_KEY is not set for the 'openai' provider.")
        self._cfg = cfg
        self.name = cfg.llm_model

    @property
    def supports_vision(self) -> bool:
        if self._cfg.vision_models:
            known = [m.strip().lower() for m in self._cfg.vision_models.split(",") if m.strip()]
            return self.name.lower() in known
        return False

    def chat_model(self) -> BaseChatModel:
        return ChatOpenAI(
            model=self._cfg.llm_model,
            base_url=self._cfg.llm_base_url,  # None -> api.openai.com
            api_key=self._cfg.llm_api_key,
            add_schema_to_system_prompt=True,
            remove_min_items_from_schema=True,
        )
