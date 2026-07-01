from __future__ import annotations

import json
import urllib.error
import urllib.request


class ModelDiscoveryError(Exception):
    """Raised when model discovery from the provider fails."""


def fetch_models(base_url: str, api_key: str, timeout: float = 10.0) -> list[str]:
    """Fetch the list of available model IDs from an OpenAI-compatible endpoint.

    Calls GET {base_url}/v1/models with Bearer auth. Returns a list of model ID
    strings. The base_url is normalized: trailing slashes are stripped, and
    /v1 is appended only if not already present.

    Raises ModelDiscoveryError on connection failure, non-2xx response, or
    malformed JSON. The caller (typically /api/models) should surface a
    user-friendly error in that case.
    """
    url = _normalize_url(base_url) + "/models"
    req = urllib.request.Request(url, method="GET")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                raise ModelDiscoveryError(
                    f"Provider returned HTTP {resp.status} for {url}"
                )
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise ModelDiscoveryError(
            f"Provider returned HTTP {exc.code} for {url}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ModelDiscoveryError(
            f"Could not reach {url}: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise ModelDiscoveryError(
            f"Timeout while fetching models from {url}"
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModelDiscoveryError(
            f"Provider returned non-JSON response from {url}"
        ) from exc

    return _extract_model_ids(data)


def _normalize_url(base_url: str) -> str:
    """Normalize base_url: strip trailing slashes, ensure /v1 suffix.

    Examples:
        "https://api.openai.com"     -> "https://api.openai.com/v1"
        "https://api.openai.com/"    -> "https://api.openai.com/v1"
        "https://api.openai.com/v1"  -> "https://api.openai.com/v1"
        "https://api.openai.com/v1/" -> "https://api.openai.com/v1"
    """
    url = (base_url or "").strip().rstrip("/")
    if not url:
        raise ModelDiscoveryError("base_url is required")
    if not url.endswith("/v1"):
        url = url + "/v1"
    return url


def is_allowed_base_url(requested: str | None, configured: str | None) -> bool:
    """Return True iff `requested` is safe to fetch models from (SSRF guard).

    The caller-supplied base_url must normalize-match the configured
    LLM_BASE_URL. If `configured` is unset, no URL is allowed. If
    `requested` is empty/None, returns False (caller should fall back to
    `configured` before calling this).
    """
    if not configured:
        return False
    if not requested:
        return False
    try:
        return _normalize_url(requested) == _normalize_url(configured)
    except ModelDiscoveryError:
        return False


def _extract_model_ids(data: object) -> list[str]:
    """Extract a list of model ID strings from the OpenAI /v1/models response.

    OpenAI format: {"object": "list", "data": [{"id": "...", ...}, ...]}
    Some providers return a bare list. We accept both.
    """
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("data")
        if items is None and "models" in data:
            items = data["models"]
    else:
        raise ModelDiscoveryError("Unexpected response shape from provider")

    if not isinstance(items, list):
        raise ModelDiscoveryError("Provider response did not contain a model list")

    ids: list[str] = []
    for item in items:
        if isinstance(item, str):
            ids.append(item)
        elif isinstance(item, dict) and "id" in item:
            ids.append(str(item["id"]))
    return sorted(set(ids))
