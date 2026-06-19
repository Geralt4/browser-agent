from __future__ import annotations

from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip().lower() for item in value.split(",") if item.strip()]


class Config(BaseSettings):
    """Runtime configuration, loaded from environment / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Which adapter to build. "kimi" is the target; "openai" is any
    # OpenAI-compatible endpoint used to prove the loop before the Kimi swap.
    provider: str = Field(default="kimi")

    # Kimi / Moonshot
    moonshot_api_key: str | None = None

    # Generic OpenAI-compatible provider (smoke test / swappable)
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: str | None = None

    # Browser + loop
    headless: bool = True
    max_steps: int = 25

    # Vision routing: "dom" = always DOM-only, "auto" = per-task heuristic,
    # "vision" = always use vision (if model supports it).
    vision_mode: str = "auto"

    # User-configured comma-separated list of vision-capable model names.
    # GenericOpenAIAdapter treats the model as vision-capable iff its name
    # appears (case-insensitive) in this list. We don't auto-detect — the
    # user knows their provider's model capabilities best.
    vision_models: str | None = None

    # Safety: LLM-based sensitivity classifier as fallback when the keyword
    # heuristic returns False. Off by default (cost); enable for production.
    sensitivity_llm: bool = False

    # Safety policy (comma-separated host substrings)
    allowlist: str | None = None
    blocklist: str | None = None
    kill_switch: bool = False

    @property
    def allow_hosts(self) -> list[str]:
        return _csv(self.allowlist)

    @property
    def block_hosts(self) -> list[str]:
        return _csv(self.blocklist)

    def with_overrides(self, **kwargs: Any) -> Config:
        """Return a new Config with the given fields overridden.

        Use this for per-request overrides from the UI / Chrome extension
        without mutating the process-wide .env-loaded config. Only fields
        that are actually present in the override and non-None are applied
        (None values are treated as "don't override").
        """
        updates = {k: v for k, v in kwargs.items() if v is not None and k in type(self).model_fields}
        return self.model_copy(update=updates)


def load_config() -> Config:
    return Config()
