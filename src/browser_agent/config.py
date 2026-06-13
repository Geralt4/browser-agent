from __future__ import annotations

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


def load_config() -> Config:
    return Config()
