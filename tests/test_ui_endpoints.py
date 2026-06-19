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
    yield env_file
    # server will rewrite the cwd .env, but tmp_path is cleaned up


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


class TestUpdateConfig:
    def test_persists_to_env(self, client, env_backup):
        r = client.post("/api/config", json={"LLM_MODEL": "gpt-4o", "VISION_MODE": "vision"})
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

        # Read back from disk
        text = env_backup.read_text()
        assert "LLM_MODEL=gpt-4o" in text
        assert "VISION_MODE=vision" in text

    def test_empty_value_clears(self, client, env_backup):
        env_backup.write_text("PROVIDER=openai\nLLM_MODEL=gpt-4o\nVISION_MODELS=llava\n")
        r = client.post("/api/config", json={"LLM_MODEL": ""})
        assert r.status_code == 200
        text = env_backup.read_text()
        assert "LLM_MODEL=" in text
        # the empty value is set
        assert "LLM_MODEL=\n" in text

    def test_preserves_unrelated_keys(self, client, env_backup):
        env_backup.write_text("PROVIDER=openai\nLLM_API_KEY=secret\nHEADLESS=true\n")
        r = client.post("/api/config", json={"LLM_MODEL": "gpt-4o"})
        assert r.status_code == 200
        text = env_backup.read_text()
        assert "LLM_API_KEY=secret" in text
        assert "HEADLESS=true" in text
        assert "LLM_MODEL=gpt-4o" in text


class TestListModels:
    def test_missing_params(self, client, env_backup):
        r = client.get("/api/models")
        assert r.status_code == 400
        assert "required" in r.json()["error"]

    @patch("browser_agent.models.discovery.urllib.request.urlopen")
    def test_happy_path(self, mock_urlopen, client, env_backup):
        body = json.dumps({"data": [{"id": "gpt-4o"}, {"id": "gpt-3.5-turbo"}]})
        mock_urlopen.return_value = _FakeResponse(200, body)

        r = client.get(
            "/api/models?base_url=https://api.openai.com",
            headers={"X-API-Key": "sk-test"},
        )
        assert r.status_code == 200
        assert r.json() == {"models": ["gpt-3.5-turbo", "gpt-4o"]}

    @patch("browser_agent.models.discovery.urllib.request.urlopen")
    def test_provider_error(self, mock_urlopen, client, env_backup):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        r = client.get(
            "/api/models?base_url=http://127.0.0.1:1",
            headers={"X-API-Key": "test"},
        )
        assert r.status_code == 502
        assert "Could not reach" in r.json()["error"]


class TestTaskWithOverrides:
    def test_missing_task(self, client, env_backup):
        r = client.post("/api/task", json={})
        assert r.status_code == 400
        assert "required" in r.json()["error"]

    def test_unknown_provider(self, client, env_backup):
        r = client.post("/api/task", json={"task": "do something", "provider": "nope"})
        assert r.status_code == 400
        assert "Unknown provider" in r.json()["error"]
