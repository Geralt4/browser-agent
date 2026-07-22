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


class TestCdpUrlValidation:
    """S13: cdp_url must point to a loopback address only.

    CDP gives the holder full control of the target browser. A non-
    loopback URL would expose that to anyone on the network, so we
    refuse everything except localhost / 127.0.0.1 / ::1.
    """

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:9222",
            "http://127.0.0.1:9222",
            "http://[::1]:9222",
            "https://localhost:9223",  # https is technically fine
        ],
    )
    def test_loopback_urls_accepted(self, clean_env, url):
        cfg = Config(cdp_url=url, llm_api_key="sk-test")
        assert cfg.cdp_url == url

    @pytest.mark.parametrize(
        "url",
        [
            "http://192.168.1.5:9222",     # LAN address
            "http://10.0.0.1:9222",        # LAN address
            "http://169.254.169.254:80",   # AWS metadata
            "http://example.com:9222",     # public hostname
            "http://evil.local:9222",      # any non-loopback host
            "ftp://localhost:9222",        # wrong scheme
            "ws://localhost:9222",         # wrong scheme
        ],
    )
    def test_non_loopback_urls_rejected(self, clean_env, url):
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as excinfo:
            Config(cdp_url=url, llm_api_key="sk-test")
        # The error message varies: "loopback" for valid scheme + bad host,
        # "must use http or https" for invalid scheme. Accept either.
        msg = str(excinfo.value).lower()
        assert "loopback" in msg or "must use http" in msg

    def test_none_cdp_url_allowed(self, clean_env):
        """Default (no CDP) must remain valid — most users use the
        auto-spawned headless browser."""
        cfg = Config(llm_api_key="sk-test")
        assert cfg.cdp_url is None

    def test_with_overrides_validates_cdp_url(self, clean_env):
        """The S13 fix changes with_overrides to use the constructor
        (not model_copy) so the validator runs on per-request overrides.
        Regression test: a malicious cdp_url from a request body must
        be rejected."""
        from pydantic import ValidationError

        cfg = Config(llm_api_key="sk-test")
        with pytest.raises(ValidationError) as excinfo:
            cfg.with_overrides(cdp_url="http://192.168.1.5:9222")
        assert "loopback" in str(excinfo.value).lower()

    def test_with_overrides_accepts_loopback(self, clean_env):
        cfg = Config(llm_api_key="sk-test")
        new = cfg.with_overrides(cdp_url="http://127.0.0.1:9222")
        assert new.cdp_url == "http://127.0.0.1:9222"
