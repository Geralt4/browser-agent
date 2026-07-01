"""Tests for browser_agent.models.discovery."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import patch

import pytest

from browser_agent.models.discovery import (
    ModelDiscoveryError,
    _extract_model_ids,
    _normalize_url,
    fetch_models,
    is_allowed_base_url,
)


class TestNormalizeUrl:
    def test_strips_trailing_slash(self):
        assert _normalize_url("https://api.openai.com/") == "https://api.openai.com/v1"

    def test_appends_v1(self):
        assert _normalize_url("https://api.openai.com") == "https://api.openai.com/v1"

    def test_preserves_existing_v1(self):
        assert _normalize_url("https://api.openai.com/v1") == "https://api.openai.com/v1"

    def test_strips_trailing_slash_with_v1(self):
        assert _normalize_url("https://api.openai.com/v1/") == "https://api.openai.com/v1"

    def test_empty_raises(self):
        with pytest.raises(ModelDiscoveryError):
            _normalize_url("")

    def test_strips_whitespace(self):
        assert _normalize_url("  https://api.openai.com  ") == "https://api.openai.com/v1"


class TestIsAllowedBaseUrl:
    def test_exact_match(self):
        assert is_allowed_base_url(
            "https://api.openai.com", "https://api.openai.com"
        )

    def test_trailing_slash_match(self):
        assert is_allowed_base_url(
            "https://api.openai.com/", "https://api.openai.com"
        )

    def test_v1_suffix_match(self):
        assert is_allowed_base_url(
            "https://api.openai.com/v1", "https://api.openai.com"
        )

    def test_mismatch_rejected(self):
        assert not is_allowed_base_url(
            "https://evil.com", "https://api.openai.com"
        )

    def test_malicious_subdomain_rejected(self):
        assert not is_allowed_base_url(
            "https://api.openai.com.evil.com", "https://api.openai.com"
        )

    def test_no_configured_url_rejects_all(self):
        assert not is_allowed_base_url("https://api.openai.com", None)
        assert not is_allowed_base_url("https://api.openai.com", "")

    def test_no_requested_url_rejected(self):
        assert not is_allowed_base_url("", "https://api.openai.com")
        assert not is_allowed_base_url(None, "https://api.openai.com")

    def test_malformed_requested_rejected(self):
        assert not is_allowed_base_url("not a url", "https://api.openai.com")


class TestExtractModelIds:
    def test_openai_format(self):
        data = {
            "object": "list",
            "data": [
                {"id": "gpt-4o", "object": "model"},
                {"id": "gpt-4o-mini", "object": "model"},
                {"id": "gpt-3.5-turbo", "object": "model"},
            ],
        }
        assert _extract_model_ids(data) == ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]

    def test_bare_list(self):
        data = ["a", "b", "c"]
        assert _extract_model_ids(data) == ["a", "b", "c"]

    def test_alternative_models_key(self):
        data = {"models": [{"id": "x"}, {"id": "y"}]}
        assert _extract_model_ids(data) == ["x", "y"]

    def test_handles_string_items(self):
        data = [{"id": "a"}, "b", {"id": "c"}]
        assert _extract_model_ids(data) == ["a", "b", "c"]

    def test_deduplicates(self):
        data = {"data": [{"id": "a"}, {"id": "a"}, {"id": "b"}]}
        assert _extract_model_ids(data) == ["a", "b"]

    def test_unexpected_shape(self):
        with pytest.raises(ModelDiscoveryError):
            _extract_model_ids("not a dict or list")

    def test_data_not_a_list(self):
        with pytest.raises(ModelDiscoveryError):
            _extract_model_ids({"data": "not a list"})


class TestFetchModels:
    def _make_response(self, status, body):
        class FakeResponse:
            def __init__(self, status, body):
                self.status = status
                self._body = body

            def read(self):
                return self._body.encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return FakeResponse(status, body)

    @patch("browser_agent.models.discovery.urllib.request.urlopen")
    def test_happy_path(self, mock_urlopen):
        body = json.dumps({"data": [{"id": "gpt-4o"}, {"id": "gpt-3.5-turbo"}]})
        mock_urlopen.return_value = self._make_response(200, body)

        result = fetch_models("https://api.openai.com", "sk-test")
        assert result == ["gpt-3.5-turbo", "gpt-4o"]

    @patch("browser_agent.models.discovery.urllib.request.urlopen")
    def test_url_appends_v1(self, mock_urlopen):
        body = json.dumps({"data": []})
        mock_urlopen.return_value = self._make_response(200, body)

        fetch_models("https://api.example.com", "key")
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.full_url == "https://api.example.com/v1/models"

    @patch("browser_agent.models.discovery.urllib.request.urlopen")
    def test_preserves_existing_v1(self, mock_urlopen):
        body = json.dumps({"data": []})
        mock_urlopen.return_value = self._make_response(200, body)

        fetch_models("https://api.example.com/v1", "key")
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.full_url == "https://api.example.com/v1/models"

    @patch("browser_agent.models.discovery.urllib.request.urlopen")
    def test_sends_bearer_auth(self, mock_urlopen):
        body = json.dumps({"data": []})
        mock_urlopen.return_value = self._make_response(200, body)

        fetch_models("https://api.example.com", "sk-secret-123")
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.get_header("Authorization") == "Bearer sk-secret-123"

    @patch("browser_agent.models.discovery.urllib.request.urlopen")
    def test_non_2xx_raises(self, mock_urlopen):
        mock_urlopen.return_value = self._make_response(401, "Unauthorized")

        with pytest.raises(ModelDiscoveryError) as exc:
            fetch_models("https://api.example.com", "key")
        assert "401" in str(exc.value)

    @patch("browser_agent.models.discovery.urllib.request.urlopen")
    def test_http_error_raises(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://api.example.com/v1/models", 403, "Forbidden", {}, None
        )
        with pytest.raises(ModelDiscoveryError) as exc:
            fetch_models("https://api.example.com", "key")
        assert "403" in str(exc.value)

    @patch("browser_agent.models.discovery.urllib.request.urlopen")
    def test_url_error_raises(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        with pytest.raises(ModelDiscoveryError) as exc:
            fetch_models("https://api.example.com", "key")
        assert "Could not reach" in str(exc.value)

    @patch("browser_agent.models.discovery.urllib.request.urlopen")
    def test_timeout_raises(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()
        with pytest.raises(ModelDiscoveryError) as exc:
            fetch_models("https://api.example.com", "key")
        assert "Timeout" in str(exc.value)

    @patch("browser_agent.models.discovery.urllib.request.urlopen")
    def test_invalid_json_raises(self, mock_urlopen):
        mock_urlopen.return_value = self._make_response(200, "not json")
        with pytest.raises(ModelDiscoveryError) as exc:
            fetch_models("https://api.example.com", "key")
        assert "non-JSON" in str(exc.value)

    def test_empty_base_url_raises(self):
        with pytest.raises(ModelDiscoveryError):
            fetch_models("", "key")
