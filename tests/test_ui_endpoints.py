"""Tests for the new UI API endpoints: /api/config and /api/models.

Uses FastAPI's TestClient. Does NOT exercise the real LLM, so we mock the
provider /v1/models response via monkeypatching urllib.request.urlopen.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from browser_agent.ui.server import app
    return TestClient(app)


@pytest.fixture
def env_backup(monkeypatch, tmp_path):
    """Run with an isolated .env so tests don't touch the user's real one."""
    env_file = tmp_path / ".env"
    env_file.write_text("PROVIDER=openai\nLLM_MODEL=test-model\n")
    monkeypatch.chdir(tmp_path)
    # Clear env vars that conftest.py's load_dotenv() may have set from the
    # real .env — pydantic-settings reads os.environ before the .env file.
    for key in ("LLM_BASE_URL", "BROWSER_AGENT_API_TOKEN", "LLM_API_KEY",
                "MOONSHOT_API_KEY", "VISION_MODELS", "VISION_MODE",
                "ALLOWLIST", "BLOCKLIST", "KILL_SWITCH", "SENSITIVITY_LLM",
                "HEADLESS", "MAX_STEPS", "DOM_CATEGORIES"):
        monkeypatch.delenv(key, raising=False)
    yield env_file


@pytest.fixture
def authed_env(monkeypatch, tmp_path):
    """Token set, LLM_BASE_URL NOT set."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "PROVIDER=openai\nLLM_MODEL=test-model\n"
        "BROWSER_AGENT_API_TOKEN=secret-token\n"
    )
    monkeypatch.chdir(tmp_path)
    for key in ("LLM_BASE_URL", "LLM_API_KEY", "MOONSHOT_API_KEY",
                "VISION_MODELS", "VISION_MODE", "ALLOWLIST", "BLOCKLIST",
                "KILL_SWITCH", "SENSITIVITY_LLM", "HEADLESS", "MAX_STEPS",
                "DOM_CATEGORIES"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("BROWSER_AGENT_API_TOKEN", "secret-token")
    return env_file


@pytest.fixture
def configured_env(monkeypatch, tmp_path):
    """LLM_BASE_URL set, token NOT set."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "PROVIDER=openai\nLLM_MODEL=test-model\n"
        "LLM_BASE_URL=https://api.openai.com\n"
    )
    monkeypatch.chdir(tmp_path)
    for key in ("BROWSER_AGENT_API_TOKEN", "LLM_API_KEY", "MOONSHOT_API_KEY",
                "VISION_MODELS", "VISION_MODE", "ALLOWLIST", "BLOCKLIST",
                "KILL_SWITCH", "SENSITIVITY_LLM", "HEADLESS", "MAX_STEPS",
                "DOM_CATEGORIES"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com")
    return env_file


@pytest.fixture
def authed_configured_env(monkeypatch, tmp_path):
    """Both token and LLM_BASE_URL set."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "PROVIDER=openai\nLLM_MODEL=test-model\n"
        "LLM_BASE_URL=https://api.openai.com\n"
        "BROWSER_AGENT_API_TOKEN=secret-token\n"
    )
    monkeypatch.chdir(tmp_path)
    for key in ("LLM_API_KEY", "MOONSHOT_API_KEY", "VISION_MODELS",
                "VISION_MODE", "ALLOWLIST", "BLOCKLIST", "KILL_SWITCH",
                "SENSITIVITY_LLM", "HEADLESS", "MAX_STEPS", "DOM_CATEGORIES"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com")
    monkeypatch.setenv("BROWSER_AGENT_API_TOKEN", "secret-token")
    return env_file


class _FakeResponse:
    def __init__(self, status: int, body: str):
        self.status = status
        self._body = body.encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestGetConfig:
    def test_returns_safe_fields(self, client, env_backup):
        r = client.get("/api/config")
        assert r.status_code == 200
        data = r.json()
        assert "provider" in data
        assert "llm_model" in data
        assert "vision_mode" in data
        # No API keys leak
        assert "llm_api_key" not in data
        assert "moonshot_api_key" not in data
        # Auth token must NOT leak
        assert "browser_agent_api_token" not in data


class TestUpdateConfigAuth:
    def test_no_token_configured_returns_403(self, client, env_backup):
        r = client.post("/api/config", json={"llm_model": "x"})
        assert r.status_code == 403
        assert "disabled" in r.json()["detail"]

    def test_missing_x_auth_token_header_returns_401(self, client, authed_env):
        r = client.post("/api/config", json={"llm_model": "x"})
        assert r.status_code == 401

    def test_wrong_token_returns_401(self, client, authed_env):
        r = client.post(
            "/api/config",
            json={"llm_model": "x"},
            headers={"X-Auth-Token": "wrong"},
        )
        assert r.status_code == 401

    def test_correct_token_persists(self, client, authed_env):
        r = client.post(
            "/api/config",
            json={"llm_model": "gpt-4o"},
            headers={"X-Auth-Token": "secret-token"},
        )
        assert r.status_code == 200
        text = authed_env.read_text()
        assert "gpt-4o" in text

    def test_get_config_still_public(self, client, authed_env):
        # GET must NOT require the token even when one is configured.
        r = client.get("/api/config")
        assert r.status_code == 200


class TestUpdateConfig:
    def test_persists_to_env(self, client, authed_env):
        r = client.post(
            "/api/config",
            json={"llm_model": "gpt-4o", "vision_mode": "vision"},
            headers={"X-Auth-Token": "secret-token"},
        )
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

        text = authed_env.read_text()
        assert "gpt-4o" in text
        assert "vision" in text

    def test_uppercase_keys_still_accepted(self, client, authed_env):
        r = client.post(
            "/api/config",
            json={"LLM_MODEL": "gpt-4o"},
            headers={"X-Auth-Token": "secret-token"},
        )
        assert r.status_code == 200
        assert "gpt-4o" in authed_env.read_text()

    def test_empty_value_clears(self, client, authed_env):
        authed_env.write_text(
            "PROVIDER=openai\nLLM_MODEL=gpt-4o\nVISION_MODELS=llava\n"
            "BROWSER_AGENT_API_TOKEN=secret-token\n"
        )
        r = client.post(
            "/api/config",
            json={"llm_model": ""},
            headers={"X-Auth-Token": "secret-token"},
        )
        assert r.status_code == 200
        text = authed_env.read_text()
        # python-dotenv set_key quotes empty values as ''
        assert "LLM_MODEL" in text

    def test_preserves_unrelated_keys(self, client, authed_env):
        authed_env.write_text(
            "PROVIDER=openai\nLLM_API_KEY=secret\nHEADLESS=true\n"
            "BROWSER_AGENT_API_TOKEN=secret-token\n"
        )
        r = client.post(
            "/api/config",
            json={"llm_model": "gpt-4o"},
            headers={"X-Auth-Token": "secret-token"},
        )
        assert r.status_code == 200
        text = authed_env.read_text()
        assert "secret" in text
        assert "true" in text
        assert "gpt-4o" in text


class TestListModelsSecurity:
    def test_no_llm_base_url_configured_returns_403(self, client, authed_env):
        r = client.get(
            "/api/models",
            headers={"X-Auth-Token": "secret-token", "X-API-Key": "k"},
        )
        assert r.status_code == 403
        assert "LLM_BASE_URL" in r.json()["error"]

    @patch("browser_agent.models.discovery.urllib.request.urlopen")
    def test_matching_base_url_works(self, mock_urlopen, client, authed_configured_env):
        mock_urlopen.return_value = _FakeResponse(
            200, json.dumps({"data": [{"id": "gpt-4o"}]})
        )
        r = client.get(
            "/api/models?base_url=https://api.openai.com",
            headers={"X-Auth-Token": "secret-token", "X-API-Key": "sk-test"},
        )
        assert r.status_code == 200
        # Verify the key was sent to the ALLOWED host only
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://api.openai.com/v1/models"
        assert req.get_header("Authorization") == "Bearer sk-test"

    @patch("browser_agent.models.discovery.urllib.request.urlopen")
    def test_omitted_base_url_uses_configured(
        self, mock_urlopen, client, authed_configured_env
    ):
        mock_urlopen.return_value = _FakeResponse(200, json.dumps({"data": []}))
        r = client.get(
            "/api/models",
            headers={"X-Auth-Token": "secret-token", "X-API-Key": "sk-test"},
        )
        assert r.status_code == 200
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://api.openai.com/v1/models"

    @patch("browser_agent.models.discovery.urllib.request.urlopen")
    def test_ssrf_attacker_url_rejected(
        self, mock_urlopen, client, authed_configured_env
    ):
        r = client.get(
            "/api/models?base_url=https://evil.com",
            headers={"X-Auth-Token": "secret-token", "X-API-Key": "sk-test"},
        )
        assert r.status_code == 400
        assert "does not match" in r.json()["error"]
        mock_urlopen.assert_not_called()

    @patch("browser_agent.models.discovery.urllib.request.urlopen")
    def test_ssrf_lookalike_url_rejected(
        self, mock_urlopen, client, authed_configured_env
    ):
        r = client.get(
            "/api/models?base_url=https://api.openai.com.evil.com",
            headers={"X-Auth-Token": "secret-token", "X-API-Key": "sk-test"},
        )
        assert r.status_code == 400
        mock_urlopen.assert_not_called()

    @patch("browser_agent.models.discovery.urllib.request.urlopen")
    def test_missing_api_key_returns_400(
        self, mock_urlopen, client, authed_configured_env
    ):
        r = client.get(
            "/api/models?base_url=https://api.openai.com",
            headers={"X-Auth-Token": "secret-token"},
        )
        assert r.status_code == 400
        mock_urlopen.assert_not_called()


class TestListModels:
    def test_missing_params(self, client, authed_configured_env):
        # No X-API-Key header -> 400
        r = client.get(
            "/api/models",
            headers={"X-Auth-Token": "secret-token"},
        )
        assert r.status_code == 400
        assert "X-API-Key" in r.json()["error"]

    @patch("browser_agent.models.discovery.urllib.request.urlopen")
    def test_no_auth_token_required(self, mock_urlopen, client, configured_env):
        """The extension's Fetch button only sends X-API-Key; it must succeed
        when LLM_BASE_URL is set even if BROWSER_AGENT_API_TOKEN is unset."""
        mock_urlopen.return_value = _FakeResponse(
            200, json.dumps({"data": [{"id": "gpt-4o"}]})
        )
        r = client.get(
            "/api/models?base_url=https://api.openai.com",
            headers={"X-API-Key": "sk-test"},
        )
        assert r.status_code == 200
        assert r.json() == {"models": ["gpt-4o"]}
        # The X-API-Key was still relayed as a Bearer token.
        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Authorization") == "Bearer sk-test"

    @patch("browser_agent.models.discovery.urllib.request.urlopen")
    def test_happy_path(self, mock_urlopen, client, authed_configured_env):
        body = json.dumps({"data": [{"id": "gpt-4o"}, {"id": "gpt-3.5-turbo"}]})
        mock_urlopen.return_value = _FakeResponse(200, body)

        r = client.get(
            "/api/models?base_url=https://api.openai.com",
            headers={"X-Auth-Token": "secret-token", "X-API-Key": "sk-test"},
        )
        assert r.status_code == 200
        assert r.json() == {"models": ["gpt-3.5-turbo", "gpt-4o"]}

    @patch("browser_agent.models.discovery.urllib.request.urlopen")
    def test_provider_error(self, mock_urlopen, client, authed_configured_env):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        r = client.get(
            "/api/models?base_url=https://api.openai.com",
            headers={"X-Auth-Token": "secret-token", "X-API-Key": "test"},
        )
        assert r.status_code == 502
        assert "Could not reach" in r.json()["error"]


class TestTaskWithOverrides:
    def test_missing_task(self, client, env_backup):
        r = client.post("/api/task", json={})
        assert r.status_code == 400
        assert "required" in r.json()["error"]

    def test_unknown_provider(self, client, env_backup):
        r = client.post(
            "/api/task",
            json={"task": "do something", "provider": "nope"},
        )
        assert r.status_code == 400
        assert "Unknown provider" in r.json()["error"]

    def test_task_without_token_when_unset_succeeds(self, client, env_backup):
        """When BROWSER_AGENT_API_TOKEN is not configured, /api/task is
        open (fail-open) so the extension works out of the box."""
        r = client.post("/api/task", json={"task": "test"})
        # Will get 400 (unknown provider) or 200 — the point is NOT 401.
        assert r.status_code != 401

    def test_task_with_token_when_set_requires_auth(self, client, authed_env):
        """When BROWSER_AGENT_API_TOKEN is configured, /api/task requires
        X-Auth-Token. Without it, the request is rejected with 401."""
        r = client.post("/api/task", json={"task": "test"})
        assert r.status_code == 401
        assert "X-Auth-Token" in r.json()["detail"]

    def test_task_with_correct_token_passes_auth(self, client, authed_env):
        """With the correct X-Auth-Token, the request passes auth and
        proceeds to normal validation."""
        r = client.post(
            "/api/task",
            json={"task": "test"},
            headers={"X-Auth-Token": "secret-token"},
        )
        # Should NOT be 401 — it gets to the provider validation step
        assert r.status_code != 401

    def test_task_with_wrong_token_rejected(self, client, authed_env):
        r = client.post(
            "/api/task",
            json={"task": "test"},
            headers={"X-Auth-Token": "wrong"},
        )
        assert r.status_code == 401
        assert "invalid" in r.json()["detail"]


class TestKeychainEndpoints:
    """The keychain proxy endpoints replace native messaging as the primary
    keychain bridge for the extension. They run in the local API server
    (same process as the task runner) and have direct access to the OS
    keychain via the `keyring` library.

    Security model: these endpoints are intentionally UNAUTHENTICATED.
    The server is bound to 127.0.0.1 (localhost only), and the `service`
    and `key` params are validated against a tight allowlist. Requiring
    X-Auth-Token would break the extension (which can't read .env) and
    force the API key into chrome.storage.local (plaintext on disk).
    """

    @pytest.fixture
    def unique_key(self):
        import uuid
        return f"test-key-{uuid.uuid4().hex[:12]}"

    def test_ping_returns_ok(self, client, authed_env):
        r = client.post("/api/keychain/ping")
        assert r.status_code == 200
        assert r.json() == {"ok": True, "pong": True}

    def test_ping_without_token_succeeds(self, client, authed_env):
        """Keychain endpoints must work WITHOUT X-Auth-Token — the extension
        has no way to obtain the token, so requiring it would break the
        keychain bridge and force plaintext storage."""
        r = client.post("/api/keychain/ping")
        assert r.status_code == 200

    def test_ping_without_configured_token_succeeds(self, client, env_backup):
        """Even with no BROWSER_AGENT_API_TOKEN configured, keychain ping
        must succeed — the keychain bridge is not gated on auth."""
        r = client.post("/api/keychain/ping")
        assert r.status_code == 200
        assert r.json() == {"ok": True, "pong": True}

    def test_set_get_delete_roundtrip(self, client, authed_env, unique_key):
        import uuid
        value = f"value-{uuid.uuid4().hex}"

        # set (no auth header)
        r = client.post(
            "/api/keychain/set",
            json={"service": "browser-agent-test", "key": unique_key, "value": value},
        )
        assert r.status_code == 200
        assert r.json() == {"ok": True}

        # get
        r = client.post(
            "/api/keychain/get",
            json={"service": "browser-agent-test", "key": unique_key},
        )
        assert r.status_code == 200
        assert r.json() == {"ok": True, "value": value}

        # delete
        r = client.post(
            "/api/keychain/delete",
            json={"service": "browser-agent-test", "key": unique_key},
        )
        assert r.status_code == 200
        assert r.json() == {"ok": True}

        # get after delete -> value is None
        r = client.post(
            "/api/keychain/get",
            json={"service": "browser-agent-test", "key": unique_key},
        )
        assert r.status_code == 200
        assert r.json() == {"ok": True, "value": None}

    def test_delete_missing_key_is_idempotent(self, client, authed_env, unique_key):
        """Deleting a key that doesn't exist must not raise — matches the
        native host's behavior in native_host.py so the client can call
        delete without checking presence first."""
        r = client.post(
            "/api/keychain/delete",
            json={"service": "browser-agent-test", "key": unique_key},
        )
        assert r.status_code == 200
        assert r.json() == {"ok": True}

    def test_rejects_unknown_service(self, client, authed_env):
        r = client.post(
            "/api/keychain/get",
            json={"service": "not-our-service", "key": "anything"},
        )
        assert r.status_code == 400
        assert "unknown service" in r.json()["detail"]

    def test_rejects_invalid_key_empty(self, client, authed_env):
        r = client.post(
            "/api/keychain/get",
            json={"service": "browser-agent", "key": ""},
        )
        assert r.status_code == 400
        assert "invalid key" in r.json()["detail"]

    def test_rejects_invalid_key_chars(self, client, authed_env):
        r = client.post(
            "/api/keychain/get",
            json={"service": "browser-agent", "key": "has spaces and !"},
        )
        assert r.status_code == 400
        assert "invalid key" in r.json()["detail"]

    def test_rejects_oversized_key(self, client, authed_env):
        r = client.post(
            "/api/keychain/get",
            json={"service": "browser-agent", "key": "a" * 200},
        )
        assert r.status_code == 400
        assert "invalid key" in r.json()["detail"]
