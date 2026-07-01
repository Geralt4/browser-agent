"""Tests for browser_agent.config.with_overrides()."""

from __future__ import annotations

import os

import pytest

from browser_agent.config import Config


@pytest.fixture
def clean_env(monkeypatch):
    """Strip all browser_agent env vars so tests don't depend on .env."""
    for key in list(os.environ):
        if key.startswith(("PROVIDER", "MOONSHOT", "LLM_", "VISION_", "HEADLESS",
                          "MAX_STEPS", "ALLOWLIST", "BLOCKLIST", "KILL_SWITCH")):
            monkeypatch.delenv(key, raising=False)


class TestWithOverrides:
    def test_overrides_llm_model(self, clean_env):
        cfg = Config(llm_model="gpt-4o-mini", llm_api_key="sk-test")
        new = cfg.with_overrides(llm_model="gpt-4o")
        assert new.llm_model == "gpt-4o"
        assert cfg.llm_model == "gpt-4o-mini"  # original unchanged

    def test_overrides_vision_settings(self, clean_env):
        cfg = Config(vision_mode="auto", vision_models="gpt-4o")
        new = cfg.with_overrides(vision_mode="vision", vision_models="gpt-4o,llava")
        assert new.vision_mode == "vision"
        assert new.vision_models == "gpt-4o,llava"

    def test_none_override_ignored(self, clean_env):
        cfg = Config(llm_model="original", llm_api_key="sk-test")
        new = cfg.with_overrides(llm_model=None)
        assert new.llm_model == "original"

    def test_unknown_field_ignored(self, clean_env):
        cfg = Config(llm_api_key="sk-test")
        new = cfg.with_overrides(unknown_field="x", totally_made_up=42)
        assert not hasattr(new, "unknown_field")
        assert not hasattr(new, "totally_made_up")
        assert new.llm_api_key == "sk-test"

    def test_multiple_overrides(self, clean_env):
        cfg = Config(llm_api_key="sk-test")
        new = cfg.with_overrides(
            llm_model="gpt-4o",
            llm_base_url="https://api.openai.com",
            vision_mode="vision",
        )
        assert new.llm_model == "gpt-4o"
        assert new.llm_base_url == "https://api.openai.com"
        assert new.vision_mode == "vision"

    def test_provider_override(self, clean_env):
        cfg = Config()
        new = cfg.with_overrides(provider="openai")
        assert new.provider == "openai"

    def test_default_vision_mode_is_vision(self, clean_env):
        # 135-task A/B (kimi-k2.6) showed vision is +9.6pp pass rate, -46%
        # latency, -27% input tokens vs DOM. Default flipped from "auto" to
        # "vision" on that evidence. Pin it here so a refactor can't quietly
        # undo the decision.
        cfg = Config()
        assert cfg.vision_mode == "vision"

    def test_returns_new_instance(self, clean_env):
        cfg = Config(llm_api_key="sk-test")
        new = cfg.with_overrides(llm_model="x")
        assert new is not cfg
        assert isinstance(new, Config)

    def test_empty_kwargs_returns_copy(self, clean_env):
        cfg = Config(llm_api_key="sk-test", llm_model="gpt-4o")
        new = cfg.with_overrides()
        assert new.llm_model == "gpt-4o"
        assert new is not cfg
