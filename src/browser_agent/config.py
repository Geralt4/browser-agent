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
    # "vision" = always use vision (if model supports it). Default is "vision"
    # because the 135-task A/B (kimi-k2.6) showed +9.6pp pass rate, -46%
    # latency, and -27% input tokens vs DOM-only. "auto" remains available
    # for users who want per-task routing; "dom" stays for token-cost-critical
    # workloads.
    vision_mode: str = Field(default="vision", pattern=r"^(dom|vision|auto|category)$")

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

    # Maximum characters of DOM content returned by the extract tool.
    # Default 8000 — enough for context without blowing out the token window.
    max_extract_chars: int = 8000

    # Maximum number of concurrent browser-agent tasks. When the limit is
    # reached, POST /api/task returns 429. Default 3 — browser sessions are
    # resource-heavy; raise only if you have the RAM/CPU for it.
    max_concurrent_tasks: int = 3

    # UI API auth: shared secret required for mutating/discovery endpoints
    # (POST /api/config, GET /api/models). When unset, those endpoints are
    # disabled (403). Set via .env — NOT writable through POST /api/config.
    browser_agent_api_token: str | None = None

    # Data-driven per-category routing (vision_mode="category"). Comma-
    # separated category names that should route to DOM-only instead of
    # vision. Populated from the A/B benchmark; safe to leave empty for
    # users who don't have repeat-based data yet.
    #
    # Default rule from 45-task x 3 repeats A/B (kimi-k2.6): the only
    # category where vision's CI excludes zero is `multi-step` (+33pp).
    # `safety` is excluded because the DOM-wins signal there was an
    # httpbin 503 artifact, not a real routing signal.
    dom_categories: str | None = None

    @property
    def allow_hosts(self) -> list[str]:
        return _csv(self.allowlist)

    @property
    def block_hosts(self) -> list[str]:
        return _csv(self.blocklist)

    @property
    def dom_category_list(self) -> list[str]:
        """Categories to route to DOM-only when vision_mode == 'category'."""
        return _csv(self.dom_categories)

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
